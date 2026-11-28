//! Transport and continuity tests for native KV event publishers.

#![cfg(all(test, feature = "cpu", feature = "ready"))]

use super::*;
use crate::kv_index::source::binding::tests::{qwen35_payload, qwen_index};
use crate::kv_index::source::test_support::{event_payload, full_binding, full_store, gpu_remove};
use crate::kv_index::{KVEvent, KvBlockIndexer, RoutableTarget, SourceBinding};

const BLOCK_SIZE: u32 = 4;
const PROMPT: [u32; 5] = [1, 2, 3, 4, 5];

fn batch_payload(events: &[KVEvent]) -> Vec<u8> {
    event_payload(1.0, events, Some(0))
}

fn store_prompt() -> crate::kv_index::BlockStored {
    full_store(&PROMPT[..BLOCK_SIZE as usize], BLOCK_SIZE, None)
}

fn credit(index: &SourceBinding) -> u64 {
    index.reusable_tokens(&PROMPT, usize::MAX)
}

fn next_seq(index: &SourceBinding) -> Option<u64> {
    index.with_stream_batch(|state, _| state.next_seq)
}

async fn wait_until(ready: impl Fn() -> bool) {
    tokio::time::timeout(Duration::from_secs(4), async {
        while !ready() {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .expect("subscriber state did not converge");
}

async fn receive(socket: &zmq::Socket) -> Vec<Vec<u8>> {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match socket.recv_multipart(zmq::DONTWAIT) {
                Ok(frames) => return frames,
                Err(zmq::Error::EAGAIN) => tokio::time::sleep(Duration::from_millis(2)).await,
                Err(error) => panic!("subscriber socket: {error}"),
            }
        }
    })
    .await
    .expect("no subscriber socket message")
}

async fn signal(rx: &mut mpsc::Receiver<IngestionSignal>) -> IngestionSignal {
    tokio::time::timeout(Duration::from_secs(4), rx.recv())
        .await
        .expect("subscriber did not report invalidation")
        .expect("subscriber signal channel closed")
}

/// TCP publisher plus replay server. XPUB confirms the subscription before
/// the test sends live messages.
struct TcpReplay {
    ctx: Arc<zmq::Context>,
    replay: zmq::Socket,
    publisher: Option<zmq::Socket>,
    endpoint: String,
    hwm: i32,
}

impl TcpReplay {
    fn new() -> Self {
        Self::with_hwm(64)
    }

    fn with_hwm(hwm: i32) -> Self {
        let ctx = Arc::new(zmq::Context::new());
        let publisher = ctx.socket(zmq::XPUB).unwrap();
        publisher.set_linger(0).unwrap();
        publisher.set_sndhwm(hwm).unwrap();
        // Report replacement subscriptions even when the old pipe is closing.
        publisher.set_xpub_verbose(true).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
        let replay = ctx.socket(zmq::ROUTER).unwrap();
        replay.set_linger(0).unwrap();
        replay.bind("inproc://tcp-recovery-replay").unwrap();
        Self {
            ctx,
            replay,
            publisher: Some(publisher),
            endpoint,
            hwm,
        }
    }

    fn start(
        &self,
        index: Arc<SourceBinding>,
        signals: Option<mpsc::Sender<IngestionSignal>>,
    ) -> SubscriberHandle {
        self.start_with_recovery(index, signals, false)
    }

    fn start_with_recovery(
        &self,
        index: Arc<SourceBinding>,
        signals: Option<mpsc::Sender<IngestionSignal>>,
        strict: bool,
    ) -> SubscriberHandle {
        self.start_with_idle_timeout(index, signals, strict, REPLAY_IDLE_TIMEOUT)
    }

    fn start_with_idle_timeout(
        &self,
        index: Arc<SourceBinding>,
        signals: Option<mpsc::Sender<IngestionSignal>>,
        strict: bool,
        replay_idle_timeout: Duration,
    ) -> SubscriberHandle {
        let source = EventSource {
            source: index.target.instance_id.clone(),
            dp_rank: index.target.dp_rank,
            pub_endpoint: self.endpoint.clone(),
            replay_endpoint: Some("inproc://tcp-recovery-replay".into()),
            topic: "kv".into(),
            hwm: Some(self.hwm),
        };
        let stream = Arc::new(EventStream::new(index).with_strict_recovery(strict));
        let lease = stream.begin_replay().unwrap();
        spawn_with_idle_timeout(
            source,
            stream,
            lease,
            self.ctx.clone(),
            signals,
            replay_idle_timeout,
        )
    }

    async fn subscribed(&self) {
        loop {
            let frames = receive(self.publisher.as_ref().unwrap()).await;
            if frames[0] == b"\x00kv" {
                continue;
            }
            assert_eq!(frames[0], b"\x01kv");
            break;
        }
    }

    async fn request(&self, start: u64) -> Vec<Vec<u8>> {
        let frames = receive(&self.replay).await;
        assert_eq!(frames[2], start.to_be_bytes());
        frames
    }

    fn live(&self, seq: i64, payload: &[u8]) {
        self.live_topic(b"kv", seq, payload);
    }

    fn live_topic(&self, topic: &[u8], seq: i64, payload: &[u8]) {
        self.publisher
            .as_ref()
            .unwrap()
            .send_multipart([topic, &seq.to_be_bytes(), payload], 0)
            .unwrap();
    }

    fn history(&self, request: &[Vec<u8>], frames: &[(i64, &[u8])]) {
        for (seq, payload) in frames
            .iter()
            .copied()
            .chain(std::iter::once((END_SEQ, &[][..])))
        {
            self.reply(request, seq, payload);
        }
    }

    fn reply(&self, request: &[Vec<u8>], seq: i64, payload: &[u8]) {
        let topic = if seq == END_SEQ { &[][..] } else { b"kv" };
        self.reply_topic(request, topic, seq, payload);
    }

    fn reply_topic(&self, request: &[Vec<u8>], topic: &[u8], seq: i64, payload: &[u8]) {
        self.replay
            .send_multipart(
                [
                    request[0].as_slice(),
                    &[],
                    topic,
                    &seq.to_be_bytes(),
                    payload,
                ],
                0,
            )
            .unwrap();
    }

    fn disconnect(&mut self) {
        drop(self.publisher.take());
    }

    async fn reopen(&mut self) {
        let publisher = self.ctx.socket(zmq::XPUB).unwrap();
        publisher.set_linger(0).unwrap();
        publisher.bind(&self.endpoint).unwrap();
        self.publisher = Some(publisher);
        self.subscribed().await;
    }
}

fn target(name: &str) -> RoutableTarget {
    RoutableTarget {
        instance_id: Arc::from(name),
        dp_rank: 0,
    }
}

#[tokio::test]
async fn emit_drops_when_full_and_records_loss() {
    let (tx, mut rx) = mpsc::channel(1);
    let signals = Some(tx);
    let before = crate::kv_index::signal_loss_epoch();
    for from_seq in [1, 2] {
        emit(
            &signals,
            IngestionSignal::Gap {
                source: Arc::from("worker"),
                from_seq,
                to_seq: from_seq,
            },
        );
    }
    assert!(matches!(
        rx.recv().await,
        Some(IngestionSignal::Gap { from_seq: 1, .. })
    ));
    assert!(rx.try_recv().is_err());
    assert!(crate::kv_index::signal_loss_epoch() > before);
}

#[tokio::test]
async fn full_attention_live_store_remove_and_clear_update_credit() {
    let tcp = TcpReplay::new();
    let index = Arc::new(full_binding(target("full"), BLOCK_SIZE));
    let (tx, mut signals) = mpsc::channel(8);
    let handle = tcp.start(index.clone(), Some(tx));
    let store = store_prompt();
    let remove = batch_payload(&[gpu_remove(Some(0), store.block_hashes.clone())]);
    let store = batch_payload(&[KVEvent::BlockStored(store)]);

    tcp.subscribed().await;
    tcp.history(&tcp.request(0).await, &[]);
    tcp.live(0, &store);
    wait_until(|| credit(&index) == 4).await;

    tcp.live_topic(b"kv-other", 1, &remove);
    tcp.live(1, &store);
    wait_until(|| next_seq(&index) == Some(2)).await;
    tcp.live(2, &remove);
    wait_until(|| next_seq(&index) == Some(3)).await;
    assert_eq!(credit(&index), 4, "the other device copy is still resident");
    tcp.live(3, &remove);
    wait_until(|| credit(&index) == 0).await;

    tcp.live(4, &store);
    wait_until(|| credit(&index) == 4).await;
    tcp.live(
        5,
        &batch_payload(&[KVEvent::AllBlocksCleared(Default::default())]),
    );
    assert!(matches!(
        signal(&mut signals).await,
        IngestionSignal::Invalidated {
            dp_rank: 0,
            reason: "engine cleared KV cache",
            ..
        }
    ));
    assert_eq!(credit(&index), 0);

    tcp.live(6, &store);
    wait_until(|| credit(&index) == 4).await;
    tcp.live(0, &batch_payload(&[]));
    assert!(matches!(
        signal(&mut signals).await,
        IngestionSignal::IncarnationReset { .. }
    ));
    let new_epoch = tcp.request(0).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&new_epoch, &[(0, &batch_payload(&[]))]);
    tcp.live(1, &store);
    wait_until(|| credit(&index) == 4).await;
    handle.shutdown();
    wait_until(|| credit(&index) == 0).await;
}

#[tokio::test]
async fn exhausted_replay_recovers_on_the_configured_live_boundary() {
    for strict in [false, true] {
        let tcp = TcpReplay::new();
        let index = Arc::new(full_binding(target("recovery-mode"), BLOCK_SIZE));
        let _handle = tcp.start_with_recovery(index.clone(), None, strict);
        let store = store_prompt();
        let live_store = batch_payload(&[KVEvent::BlockStored(store.clone())]);
        tcp.subscribed().await;
        for _ in 0..MAX_REPLAY_FAILURES {
            let request = tcp.request(0).await;
            tcp.history(&request, &[(10, &live_store)]);
        }
        tcp.subscribed().await;
        assert_eq!(credit(&index), 0);
        tcp.live(11, &live_store);

        if strict {
            tokio::time::sleep(Duration::from_millis(30)).await;
            assert_eq!(
                credit(&index),
                0,
                "a store alone cannot restore strict credit"
            );
            let clear = batch_payload(&[
                KVEvent::AllBlocksCleared(Default::default()),
                KVEvent::BlockStored(store.clone()),
            ]);
            tcp.live(12, &clear);
            wait_until(|| next_seq(&index) == Some(13)).await;
            assert_eq!(credit(&index), 4);
        } else {
            wait_until(|| next_seq(&index) == Some(12)).await;
            assert_eq!(credit(&index), 4);
        }
        let seq = if strict { 13 } else { 12 };
        tcp.live(
            seq,
            &batch_payload(&[gpu_remove(Some(0), store.block_hashes)]),
        );
        wait_until(|| next_seq(&index) == Some(seq as u64 + 1)).await;
        assert_eq!(credit(&index), 0);
    }
}

#[tokio::test]
async fn mooncake_cpu_store_preserves_worker_credit_live_and_on_replay() {
    use crate::kv_index::StorageTier;
    use crate::policies::TieredMatchProvider;

    let mut tcp = TcpReplay::new();
    let index = Arc::new(full_binding(target("mixed-offload"), BLOCK_SIZE).require_root_proof());
    let _handle = tcp.start(index.clone(), None);
    let prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let first = store_prompt();
    let mut host = first.clone();
    host.medium = Some("CPU".into());
    let mut second = full_store(&prompt[..8], BLOCK_SIZE, None);
    second.parent_block_hash = Some(second.block_hashes.remove(0));
    second.token_ids = prompt[4..8].to_vec();
    let mut mooncake = second.clone();
    mooncake.medium = Some("cpu".into());
    mooncake.kv_cache_spec_kind = None;

    let mut gpu_hashes = first.block_hashes.clone();
    gpu_hashes.extend(second.block_hashes.iter().cloned());
    let remove_gpu = batch_payload(&[gpu_remove(Some(0), gpu_hashes)]);
    let first_batch = batch_payload(&[KVEvent::BlockStored(first), KVEvent::BlockStored(host)]);
    let mixed = batch_payload(&[KVEvent::BlockStored(mooncake), KVEvent::BlockStored(second)]);
    let reuse = || index.reusable_tokens(&prompt, usize::MAX);

    tcp.subscribed().await;
    tcp.history(&tcp.request(0).await, &[]);
    tcp.live(0, &first_batch);
    wait_until(|| reuse() == 4).await;
    tcp.live(1, &mixed);
    wait_until(|| next_seq(&index) == Some(2)).await;
    assert_eq!(reuse(), 8);

    tcp.disconnect();
    wait_until(|| reuse() == 0).await;
    tcp.reopen().await;
    tcp.history(&tcp.request(0).await, &[(0, &first_batch), (1, &mixed)]);
    wait_until(|| reuse() == 8).await;

    tcp.live(2, &remove_gpu);
    wait_until(|| next_seq(&index) == Some(3)).await;
    assert_eq!(
        reuse(),
        4,
        "only the local CPU copy should survive GPU eviction"
    );
    let matches = index.matches(&prompt, BLOCK_SIZE, None, 0);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].tier, StorageTier::HostPinned);
}

#[tokio::test]
async fn delayed_replay_applies_gap_batch_and_drains_the_live_queue() {
    let mut tcp = TcpReplay::new();
    tcp.hwm = 2;
    let index = Arc::new(full_binding(target("buffered-recovery"), BLOCK_SIZE));
    let _handle = tcp.start(index.clone(), None);
    let store = store_prompt();
    let remove = batch_payload(&[gpu_remove(Some(0), store.block_hashes.clone())]);
    let warm = batch_payload(&[KVEvent::BlockStored(store)]);
    let empty = batch_payload(&[]);

    tcp.subscribed().await;
    tcp.history(&tcp.request(0).await, &[]);
    tcp.live(0, &warm);
    wait_until(|| credit(&index) == 4).await;
    tcp.live(2, &remove);
    let request = tcp.request(1).await;
    for seq in 3..24 {
        tcp.live(seq, &empty);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    tcp.live(24, &empty);
    assert_eq!(credit(&index), 0);
    tcp.history(&request, &[(1, &empty)]);
    wait_until(|| next_seq(&index) == Some(25)).await;
    assert_eq!(credit(&index), 0);
}

#[tokio::test]
async fn replay_retries_from_the_first_unapplied_sequence() {
    for timed_out in [false, true] {
        let tcp = TcpReplay::new();
        let index = Arc::new(full_binding(target("retained-gap"), BLOCK_SIZE));
        let _handle =
            tcp.start_with_idle_timeout(index.clone(), None, true, Duration::from_secs(1));
        let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);
        let empty = batch_payload(&[]);

        tcp.subscribed().await;
        tcp.history(&tcp.request(0).await, &[]);
        tcp.live(0, &store);
        tcp.live(1, &empty);
        wait_until(|| next_seq(&index) == Some(2) && credit(&index) == 4).await;
        tcp.live(4, &empty);

        let request = tcp.request(2).await;
        assert_eq!(credit(&index), 0);
        let missing = if timed_out {
            tcp.reply(&request, 2, &empty);
            wait_until(|| next_seq(&index) == Some(3)).await;
            3
        } else {
            tcp.history(&request, &[]);
            2
        };

        let retry = tcp.request(missing as u64).await;
        assert_eq!(credit(&index), 0);
        let suffix: Vec<_> = (missing..4).map(|seq| (seq, empty.as_slice())).collect();
        tcp.history(&retry, &suffix);
        wait_until(|| next_seq(&index) == Some(5) && credit(&index) == 4).await;
    }
}

#[tokio::test]
async fn buffered_live_batch_can_fill_a_gap_before_a_higher_replay_sequence() {
    let tcp = TcpReplay::new();
    let index = Arc::new(full_binding(target("observed-watermark"), BLOCK_SIZE));
    let _handle = tcp.start_with_idle_timeout(index.clone(), None, true, Duration::from_secs(1));
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);
    let empty = batch_payload(&[]);
    tcp.subscribed().await;
    tcp.history(&tcp.request(0).await, &[]);
    tcp.live(0, &store);
    wait_until(|| credit(&index) == 4).await;
    tcp.live(2, &empty);

    let request = tcp.request(1).await;
    tcp.reply(&request, 3, &empty);
    let retry = tcp.request(1).await;
    tcp.history(&retry, &[(1, &empty)]);
    let retry = tcp.request(3).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&retry, &[(3, &empty)]);
    wait_until(|| next_seq(&index) == Some(4) && credit(&index) == 4).await;
}

#[tokio::test]
async fn truncated_idle_replay_cannot_publish_on_an_empty_retry() {
    let tcp = TcpReplay::new();
    let index = Arc::new(full_binding(target("truncated-idle"), BLOCK_SIZE));
    let (tx, mut signals) = mpsc::channel(8);
    let _handle =
        tcp.start_with_idle_timeout(index.clone(), Some(tx), true, Duration::from_secs(1));
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);
    let empty = batch_payload(&[]);
    tcp.subscribed().await;
    tcp.history(&tcp.request(0).await, &[]);
    tcp.live(0, &store);
    wait_until(|| credit(&index) == 4).await;

    let request = tcp.request(1).await;
    tcp.reply(&request, 3, &empty);
    assert!(matches!(
        signal(&mut signals).await,
        IngestionSignal::Invalidated { .. }
    ));
    tcp.history(&tcp.request(1).await, &[]);
    let retry = tcp.request(1).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&retry, &[(1, &empty), (2, &empty), (3, &empty)]);
    wait_until(|| credit(&index) == 4).await;
}

#[tokio::test]
async fn replay_can_restore_from_a_retained_clear() {
    let tcp = TcpReplay::new();
    let index = Arc::new(full_binding(target("retained-clear"), BLOCK_SIZE));
    let _handle = tcp.start_with_recovery(index.clone(), None, true);
    let empty = batch_payload(&[]);
    let clear = batch_payload(&[KVEvent::AllBlocksCleared(Default::default())]);
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);

    tcp.subscribed().await;
    let request = tcp.request(0).await;
    tcp.live(8, &empty);
    tokio::time::sleep(Duration::from_millis(20)).await;
    tcp.reply(&request, 9, &empty);
    tcp.reply(&request, 10, &clear);
    tcp.reply(&request, 11, &store);
    wait_until(|| next_seq(&index) == Some(12)).await;
    assert_eq!(credit(&index), 0, "replay is still open");
    tcp.reply(&request, END_SEQ, &[]);
    wait_until(|| credit(&index) == 4).await;
    tcp.live(12, &empty);
    wait_until(|| next_seq(&index) == Some(13) && credit(&index) == 4).await;
}

#[tokio::test]
async fn retained_clear_does_not_hide_a_buffered_payload_conflict() {
    let tcp = TcpReplay::new();
    let index = Arc::new(full_binding(target("clear-conflict"), BLOCK_SIZE));
    let _handle = tcp.start_with_recovery(index.clone(), None, true);
    let empty = batch_payload(&[]);
    let clear = batch_payload(&[KVEvent::AllBlocksCleared(Default::default())]);
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);
    tcp.subscribed().await;
    let request = tcp.request(0).await;
    tcp.live(9, &store);
    tokio::time::sleep(Duration::from_millis(20)).await;
    tcp.history(&request, &[(9, &empty), (10, &clear), (11, &store)]);

    let without_conflicting_seq = tcp.request(0).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&without_conflicting_seq, &[(10, &clear), (11, &store)]);

    let matching = tcp.request(0).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&matching, &[(9, &store), (10, &clear), (11, &store)]);
    wait_until(|| credit(&index) == 4).await;
}

#[tokio::test]
async fn live_buffer_overflow_preserves_a_known_payload_conflict() {
    use super::super::live_event_buffer::MAX_PENDING_LIVE_FRAMES;

    let tcp = TcpReplay::with_hwm((MAX_PENDING_LIVE_FRAMES * 2) as i32);
    let index = Arc::new(full_binding(target("overflow-conflict"), BLOCK_SIZE));
    let _handle = tcp.start_with_recovery(index.clone(), None, true);
    let empty = batch_payload(&[]);
    let clear = batch_payload(&[KVEvent::AllBlocksCleared(Default::default())]);
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);

    tcp.subscribed().await;
    let request = tcp.request(0).await;
    tcp.live(9, &store);
    tokio::time::sleep(Duration::from_millis(20)).await;
    for seq in 0..=9 {
        tcp.reply(&request, seq, &empty);
    }
    wait_until(|| next_seq(&index) == Some(10)).await;

    // Overflow while replay is still open, before the overlap check at EOF.
    for _ in 1..MAX_PENDING_LIVE_FRAMES {
        tcp.live(11, &store);
    }
    let without_conflicting_seq = tcp.request(0).await;
    tcp.history(&without_conflicting_seq, &[(10, &clear), (11, &store)]);

    let matching = tcp.request(0).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&matching, &[(9, &store), (10, &clear), (11, &store)]);
    wait_until(|| credit(&index) == 4).await;
}

#[tokio::test]
async fn live_buffer_overflow_preserves_a_conflict_known_from_the_handoff() {
    use super::super::live_event_buffer::MAX_PENDING_LIVE_FRAMES;

    let tcp = TcpReplay::with_hwm((MAX_PENDING_LIVE_FRAMES * 2) as i32);
    let index = Arc::new(full_binding(target("overflow-after-handoff"), BLOCK_SIZE));
    let _handle = tcp.start_with_recovery(index.clone(), None, true);
    let empty = batch_payload(&[]);
    let clear = batch_payload(&[KVEvent::AllBlocksCleared(Default::default())]);
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);

    tcp.subscribed().await;
    let request = tcp.request(0).await;
    tcp.live(9, &store);
    tokio::time::sleep(Duration::from_millis(20)).await;
    tcp.history(&request, &[(9, &empty), (10, &clear), (11, &store)]);

    // The rejected handoff clears the replay fingerprint. Overflow the retry
    // before it records any new replay payloads.
    tcp.request(0).await;
    assert_eq!(credit(&index), 0);
    for _ in 1..MAX_PENDING_LIVE_FRAMES {
        tcp.live(11, &store);
    }
    let without_conflicting_seq = tcp.request(0).await;
    tcp.history(&without_conflicting_seq, &[(10, &clear), (11, &store)]);

    let matching = tcp.request(0).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&matching, &[(9, &store), (10, &clear), (11, &store)]);
    wait_until(|| credit(&index) == 4).await;
}

#[tokio::test]
async fn replay_rejection_preserves_a_conflict_observed_before_handoff() {
    let tcp = TcpReplay::new();
    let index = Arc::new(full_binding(target("rejected-conflict"), BLOCK_SIZE));
    let _handle = tcp.start_with_recovery(index.clone(), None, true);
    let empty = batch_payload(&[]);
    let clear = batch_payload(&[KVEvent::AllBlocksCleared(Default::default())]);
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);

    tcp.subscribed().await;
    let request = tcp.request(0).await;
    tcp.live(9, &store);
    tokio::time::sleep(Duration::from_millis(20)).await;
    tcp.reply(&request, 9, &empty);
    tcp.reply(&request, 10, &[0xc1]);

    let retry = tcp.request(0).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&retry, &[(10, &clear), (11, &store)]);

    let matching = tcp.request(0).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&matching, &[(9, &store), (10, &clear), (11, &store)]);
    wait_until(|| credit(&index) == 4).await;
}

#[tokio::test]
async fn replay_eof_cannot_substitute_a_disputed_gap_trigger() {
    let tcp = TcpReplay::new();
    let index = Arc::new(full_binding(target("disputed-trigger-end"), BLOCK_SIZE));
    let _handle = tcp.start_with_recovery(index.clone(), None, true);
    let empty = batch_payload(&[]);
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);

    tcp.subscribed().await;
    tcp.history(&tcp.request(0).await, &[]);
    tcp.live(0, &store);
    wait_until(|| credit(&index) == 4).await;
    tcp.live(2, &empty);
    let request = tcp.request(1).await;
    tcp.reply(&request, 1, &empty);
    tcp.reply(&request, 2, &store);

    let retry = tcp.request(0).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&retry, &[(0, &store), (1, &empty)]);

    let matching = tcp.request(2).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&matching, &[(2, &empty), (3, &empty)]);
    wait_until(|| next_seq(&index) == Some(4) && credit(&index) == 4).await;
}

#[tokio::test]
async fn replay_after_clear_must_fill_later_gaps_before_publishing() {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let metrics = recorder.handle();
    let _recorder = metrics::set_default_local_recorder(&recorder);
    let tcp = TcpReplay::new();
    let index = Arc::new(full_binding(target("clear-gap"), BLOCK_SIZE));
    let _handle = tcp.start_with_recovery(index.clone(), None, true);
    let clear = batch_payload(&[KVEvent::AllBlocksCleared(Default::default())]);
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);
    let empty = batch_payload(&[]);

    tcp.subscribed().await;
    let request = tcp.request(0).await;
    tcp.history(&request, &[(10, &clear), (11, &store), (13, &clear)]);
    let short_retry = tcp.request(0).await;
    tcp.history(&short_retry, &[(10, &clear), (11, &store)]);
    let retry = tcp.request(12).await;
    assert_eq!(credit(&index), 0);
    assert!(!metrics.render().contains("reason=\"clear_barrier\""));
    tcp.history(&retry, &[(12, &empty), (13, &clear), (14, &store)]);
    wait_until(|| next_seq(&index) == Some(15) && credit(&index) == 4).await;
    assert!(metrics.render().lines().any(|line| {
        line.starts_with("vllm_router_kv_source_recovery_events_total{")
            && line.contains("reason=\"clear_barrier\"")
            && line.ends_with(" 1")
    }));
}

#[tokio::test]
async fn invalid_queued_live_sequence_is_rejected_only_once() {
    let tcp = TcpReplay::new();
    let index = Arc::new(full_binding(target("invalid-live-overlap"), BLOCK_SIZE));
    let _handle = tcp.start_with_recovery(index.clone(), None, true);
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);
    let clear = batch_payload(&[KVEvent::AllBlocksCleared(Default::default())]);
    let empty = batch_payload(&[]);
    tcp.subscribed().await;
    let request = tcp.request(0).await;
    tcp.live(i64::MAX, &empty);
    tokio::time::sleep(Duration::from_millis(20)).await;
    tcp.history(&request, &[(0, &store), (1, &clear)]);

    let retry = tcp.request(0).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&retry, &[(0, &store)]);
    let retry = tcp.request(1).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&retry, &[(1, &clear), (2, &store)]);
    wait_until(|| next_seq(&index) == Some(3) && credit(&index) == 4).await;
}

#[tokio::test]
async fn failed_handoff_preserves_live_frames_for_the_next_replay() {
    let tcp = TcpReplay::new();
    let index = Arc::new(full_binding(target("failed-handoff"), BLOCK_SIZE));
    let _handle = tcp.start_with_recovery(index.clone(), None, true);
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);
    let clear = batch_payload(&[KVEvent::AllBlocksCleared(Default::default())]);
    let empty = batch_payload(&[]);
    tcp.subscribed().await;
    let request = tcp.request(0).await;
    tcp.live(9, &store);
    tcp.live(i64::MAX, &empty);
    tokio::time::sleep(Duration::from_millis(20)).await;
    tcp.history(&request, &[(10, &clear), (11, &store)]);

    let conflicting = tcp.request(0).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&conflicting, &[(9, &empty), (10, &clear), (11, &store)]);

    let matching = tcp.request(0).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&matching, &[(9, &store), (10, &clear), (11, &store)]);
    wait_until(|| credit(&index) == 4).await;
}

#[tokio::test]
async fn long_replay_verifies_the_oldest_buffered_live_frame() {
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);
    let empty = batch_payload(&[]);
    let last = 4_096;
    for matches_replay in [true, false] {
        let tcp = TcpReplay::new();
        let index = Arc::new(full_binding(target("long-overlap"), BLOCK_SIZE));
        let (tx, mut signals) = mpsc::channel(16);
        let _handle = tcp.start_with_recovery(index.clone(), Some(tx), true);

        tcp.subscribed().await;
        let request = tcp.request(0).await;
        tcp.live(0, if matches_replay { &store } else { &empty });
        if matches_replay {
            tcp.live(0, &store);
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        tcp.reply(&request, 0, &store);
        for seq in 1..=last {
            tcp.reply(&request, seq, &empty);
            if seq % 128 == 0 {
                // Keep the inproc replay socket below its send high-water mark.
                wait_until(|| next_seq(&index) == Some(seq as u64 + 1)).await;
            }
        }
        tcp.reply(&request, END_SEQ, &[]);
        if matches_replay {
            tcp.live(last + 1, &empty);
            tokio::select! {
                _ = wait_until(|| next_seq(&index) == Some(last as u64 + 2) && credit(&index) == 4) => {}
                signal = signals.recv() => panic!("unexpected recovery signal: {signal:?}"),
            }
        } else {
            tcp.request(0).await;
            assert_eq!(credit(&index), 0);
            assert!(signals.try_recv().is_err());
        }
    }
}

#[tokio::test]
async fn repeated_live_buffer_overflow_does_not_exhaust_replay() {
    use super::super::live_event_buffer::MAX_PENDING_LIVE_FRAMES;

    let tcp = TcpReplay::with_hwm((MAX_PENDING_LIVE_FRAMES * 2) as i32);
    let index = Arc::new(full_binding(target("live-backpressure"), BLOCK_SIZE));
    let _handle = tcp.start_with_recovery(index.clone(), None, true);
    let empty = batch_payload(&[]);
    tcp.subscribed().await;

    let mut seq = 0;
    for _ in 0..MAX_REPLAY_FAILURES {
        tcp.request(0).await;
        for _ in 0..MAX_PENDING_LIVE_FRAMES {
            tcp.live(seq, &empty);
            seq += 1;
        }
        assert_eq!(credit(&index), 0);
    }
    // Each aborted replay still has a retry available after the next buffer
    // overflow. Backpressure alone never enters the live-only fallback.
    tcp.request(0).await;
}

#[tokio::test]
async fn replay_requires_continuous_history_and_valid_topic_rank_and_payload() {
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);
    let empty = batch_payload(&[]);
    let clear = batch_payload(&[KVEvent::AllBlocksCleared(Default::default())]);
    let bad_rank = event_payload(1.0, &[], Some(1));

    for failure in [
        "missing_start",
        "middle_gap",
        "corrupt",
        "wrong_rank",
        "wrong_topic",
        "conflicting_live_batch",
    ] {
        let tcp = TcpReplay::new();
        let index = Arc::new(full_binding(target("replay"), BLOCK_SIZE));
        let (tx, mut signals) = mpsc::channel(8);
        let _handle = tcp.start(index.clone(), Some(tx));
        tcp.subscribed().await;
        tcp.history(&tcp.request(0).await, &[]);
        tcp.live(0, &store);
        wait_until(|| credit(&index) == 4).await;
        tcp.live(2, &empty);
        assert!(matches!(
            signal(&mut signals).await,
            IngestionSignal::Gap {
                from_seq: 1,
                to_seq: 1,
                ..
            }
        ));

        let request = tcp.request(1).await;
        assert_eq!(credit(&index), 0);
        match failure {
            "missing_start" => tcp.reply(&request, 2, &empty),
            "middle_gap" => {
                tcp.reply(&request, 1, &empty);
                tcp.reply(&request, 3, &empty);
            }
            "corrupt" => tcp.reply(&request, 1, &[0xc1]),
            "wrong_rank" => tcp.reply(&request, 1, &bad_rank),
            "wrong_topic" => tcp.reply_topic(&request, b"kv-other", 1, &empty),
            "conflicting_live_batch" => {
                tcp.reply(&request, 1, &empty);
                tcp.reply(&request, 2, &store);
            }
            _ => unreachable!(),
        }
        tcp.reply(&request, END_SEQ, &[]);
        let mut retry = tcp.request(0).await;
        assert_eq!(credit(&index), 0, "bad replay exposed credit: {failure}");
        if failure == "conflicting_live_batch" {
            tcp.history(&retry, &[(3, &clear), (4, &store)]);
            retry = tcp.request(0).await;
            assert_eq!(
                credit(&index),
                0,
                "replay skipped the conflicting gap trigger"
            );
        }
        tcp.reply(&retry, 0, &store);
        wait_until(|| next_seq(&index) == Some(1)).await;
        assert_eq!(credit(&index), 0, "open replay exposed credit: {failure}");
        let tail = [(1, empty.as_slice()), (2, &empty), (3, &empty), (4, &empty)];
        let tail_len = match failure {
            "conflicting_live_batch" => 4,
            "middle_gap" => 3,
            _ => 2,
        };
        tcp.history(&retry, &tail[..tail_len]);
        wait_until(|| credit(&index) == 4).await;
    }
}

async fn tcp_recovery(monitor_failure: bool) {
    let source = target("tcp-worker");
    for (index, payload, prompt, expected) in [
        (
            full_binding(source.clone(), BLOCK_SIZE),
            batch_payload(&[KVEvent::BlockStored(store_prompt())]),
            PROMPT.to_vec(),
            4,
        ),
        (
            qwen_index(source, 8),
            qwen35_payload(),
            (0..1024).map(|i| 100 + i % 100).collect(),
            528,
        ),
    ] {
        let mut tcp = TcpReplay::new();
        let index = Arc::new(index);
        let handle = tcp.start(index.clone(), None);
        let credit = || index.reusable_tokens(&prompt, usize::MAX);
        let empty = batch_payload(&[]);
        tcp.subscribed().await;
        let request = tcp.request(0).await;
        tcp.history(&request, &[(0, &payload)]);
        wait_until(|| credit() == expected).await;
        tcp.live(1, &empty);
        wait_until(|| next_seq(&index) == Some(2)).await;

        if monitor_failure {
            super::super::transport::fail_next_monitor_drain(&tcp.endpoint);
        } else {
            tcp.disconnect();
        }
        wait_until(|| credit() == 0).await;
        if !monitor_failure {
            assert_eq!(
                tcp.replay.recv_multipart(zmq::DONTWAIT),
                Err(zmq::Error::EAGAIN)
            );
            tcp.reopen().await;
        } else {
            tcp.subscribed().await;
        }
        let request = tcp.request(0).await;
        tcp.history(&request, &[(2, &payload)]);
        let retry = tcp.request(0).await;

        tcp.reply(&retry, 0, &payload);
        wait_until(|| next_seq(&index) == Some(1)).await;
        assert_eq!(credit(), 0, "open replay exposed credit");
        tcp.history(&retry, &[(1, &empty), (2, &empty)]);
        wait_until(|| credit() == expected).await;
        tcp.live(3, &empty);
        wait_until(|| next_seq(&index) == Some(4)).await;

        handle.shutdown();
        wait_until(|| credit() == 0).await;
    }
}

#[tokio::test]
async fn first_live_batch_overlaps_only_the_same_replayed_payload() {
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);
    let empty = batch_payload(&[]);
    for matches_replay in [true, false] {
        let tcp = TcpReplay::new();
        let index = Arc::new(full_binding(target("zero-overlap"), BLOCK_SIZE));
        let (tx, mut signals) = mpsc::channel(16);
        let _handle = tcp.start(index.clone(), Some(tx));
        tcp.subscribed().await;
        tcp.history(&tcp.request(0).await, &[(0, &store)]);
        wait_until(|| credit(&index) == 4).await;

        if matches_replay {
            tcp.live(0, &store);
            tcp.live(1, &empty);
            wait_until(|| next_seq(&index) == Some(2)).await;
            assert_eq!(credit(&index), 4);
            assert!(signals.try_recv().is_err());
        } else {
            tcp.live(0, &empty);
            assert!(matches!(
                signal(&mut signals).await,
                IngestionSignal::IncarnationReset { .. }
            ));
            assert_eq!(credit(&index), 0);
        }
    }
}

#[tokio::test]
async fn malformed_live_batch_can_recover_through_replay() {
    let tcp = TcpReplay::new();
    let index = Arc::new(full_binding(target("malformed-live"), BLOCK_SIZE));
    let (tx, mut signals) = mpsc::channel(8);
    let _handle = tcp.start(index.clone(), Some(tx));
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);
    let empty = batch_payload(&[]);

    tcp.subscribed().await;
    tcp.history(&tcp.request(0).await, &[]);
    tcp.live(0, &store);
    wait_until(|| credit(&index) == 4).await;

    tcp.live(1, &[0xc1]);
    assert!(matches!(
        signal(&mut signals).await,
        IngestionSignal::Invalidated {
            reason: "invalid live KV event batch",
            ..
        }
    ));
    let insufficient = tcp.request(0).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&insufficient, &[(0, &store)]);

    let replay = tcp.request(1).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&replay, &[(1, &empty)]);
    wait_until(|| credit(&index) == 4 && next_seq(&index) == Some(2)).await;
}

#[tokio::test]
async fn idle_subscriber_shutdown_retires_its_source() {
    let tcp = TcpReplay::new();
    let index = Arc::new(full_binding(target("idle-shutdown"), BLOCK_SIZE));
    let (tx, mut signals) = mpsc::channel(8);
    let handle = tcp.start(index.clone(), Some(tx));
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);

    tcp.subscribed().await;
    tcp.history(&tcp.request(0).await, &[]);
    tcp.live(0, &store);
    wait_until(|| credit(&index) == 4 && next_seq(&index) == Some(1)).await;

    handle.shutdown();
    let signal = tokio::time::timeout(Duration::from_secs(4), signals.recv())
        .await
        .expect("idle subscriber did not stop");
    assert!(signal.is_none());
    assert_eq!(next_seq(&index), None);
    assert_eq!(credit(&index), 0);
}

#[tokio::test]
async fn live_sequence_without_room_for_a_successor_fences_cache() {
    let tcp = TcpReplay::new();
    let index = Arc::new(full_binding(target("overflow"), BLOCK_SIZE));
    let (tx, mut signals) = mpsc::channel(8);
    let _handle = tcp.start(index.clone(), Some(tx));
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);
    let empty = batch_payload(&[]);
    tcp.subscribed().await;
    tcp.history(&tcp.request(0).await, &[(0, &store)]);
    wait_until(|| credit(&index) == 4).await;

    tcp.live(i64::MAX, &empty);
    assert!(matches!(
        signal(&mut signals).await,
        IngestionSignal::Invalidated {
            reason: "invalid live sequence",
            ..
        }
    ));
    assert_eq!(credit(&index), 0);
    tcp.history(&tcp.request(0).await, &[(0, &store)]);
    wait_until(|| credit(&index) == 4).await;
}

#[tokio::test]
async fn tcp_disconnect_fences_until_replay_and_live_handoff() {
    tcp_recovery(false).await;
}

#[tokio::test]
async fn tcp_monitor_failure_rebuilds_and_replays() {
    tcp_recovery(true).await;
}

#[tokio::test]
async fn tcp_disconnect_during_gap_replay_never_republishes_old_source() {
    let mut tcp = TcpReplay::new();
    let index = Arc::new(full_binding(target("gap-worker"), BLOCK_SIZE));
    let (tx, mut signals) = mpsc::channel(16);
    let _handle = tcp.start(index.clone(), Some(tx));
    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);
    let empty = batch_payload(&[]);
    tcp.subscribed().await;
    tcp.history(&tcp.request(0).await, &[]);
    tcp.live(0, &store);
    wait_until(|| credit(&index) == 4).await;
    tcp.live(2, &empty);
    assert!(matches!(
        signal(&mut signals).await,
        IngestionSignal::Gap { .. }
    ));
    let in_flight = tcp.request(1).await;
    tcp.reply(&in_flight, 1, &empty);
    wait_until(|| next_seq(&index) == Some(2)).await;

    tcp.disconnect();
    // Let ZMQ report the TCP disconnect while this replay is still open.
    tokio::time::sleep(Duration::from_millis(200)).await;
    tcp.history(&in_flight, &[(2, &empty)]);
    assert!(matches!(
        signal(&mut signals).await,
        IngestionSignal::Invalidated {
            reason: "KV event publisher disconnected",
            ..
        }
    ));
    assert_eq!(credit(&index), 0);

    tcp.reopen().await;
    let retry = tcp.request(0).await;
    assert_eq!(credit(&index), 0);
    tcp.history(&retry, &[(0, &store), (1, &empty), (2, &empty)]);
    wait_until(|| credit(&index) == 4).await;
}

#[test]
fn monitor_failure_fences_only_active_source_and_respects_stale_lease() {
    let shared = Arc::new(KvBlockIndexer::new());
    let active =
        Arc::new(full_binding(target("active"), BLOCK_SIZE).with_shared_index(shared.clone()));
    let peer = Arc::new(full_binding(target("peer"), BLOCK_SIZE).with_shared_index(shared));
    let active_stream = Arc::new(EventStream::new(active.clone()));
    let peer_stream = EventStream::new(peer.clone());
    let stale = active_stream.begin_replay().unwrap();
    let active_lease = active_stream.begin_replay().unwrap();

    let store = batch_payload(&[KVEvent::BlockStored(store_prompt())]);
    let peer_lease = peer_stream.begin_replay().unwrap();
    for (stream, lease) in [
        (active_stream.as_ref(), active_lease),
        (&peer_stream, peer_lease),
    ] {
        stream.apply_payload(lease, 0, &store).unwrap();
        stream.finish_replay(lease, 1).unwrap();
    }
    let subscriber = |lease| Subscriber {
        replay_idle_timeout: REPLAY_IDLE_TIMEOUT,
        source: EventSource {
            source: active.target.instance_id.clone(),
            dp_rank: 0,
            pub_endpoint: String::new(),
            replay_endpoint: None,
            topic: String::new(),
            hwm: None,
        },
        binding: Some((active_stream.clone(), lease)),
        signals: None,
    };
    subscriber(stale).monitor_failed("fault injection");
    assert_eq!(credit(&active), 4);
    subscriber(active_lease).monitor_failed("fault injection");
    assert_eq!(credit(&active), 0);
    assert_eq!(credit(&peer), 4);
}
