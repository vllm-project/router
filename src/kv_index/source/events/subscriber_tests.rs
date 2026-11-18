#![cfg(all(test, feature = "cpu", feature = "ready"))]

use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use super::*;
use crate::kv_index::index::MatchQuery;
use crate::kv_index::source::events::validate::FullAttentionRejectionReason;
use crate::kv_index::source::test_support::{event_batch, event_payload, gpu_remove, stored_block};
use crate::kv_index::types::Locality;
use crate::kv_index::wire::{AllBlocksCleared, BlockRemoved, BlockStored};

fn stored(
    block_hashes: Vec<ExternalBlockHash>,
    token_ids: Vec<u32>,
    medium: Option<&str>,
    ownership: Option<&str>,
) -> BlockStored {
    let mut event = stored_block(block_hashes, token_ids, 4);
    event.medium = medium.map(String::from);
    event.ownership = ownership.map(String::from);
    event.group_idx = Some(0);
    event
}

#[test]
fn full_attention_incremental_last_device_copy_controls_residency() {
    let index = KvBlockIndexer::new();
    let source = SourceId::from("w");
    let hash = ExternalBlockHash::Int(10);
    let store = KVEvent::BlockStored(stored(
        vec![hash.clone()],
        vec![1, 2, 3, 4],
        Some("GPU"),
        None,
    ));
    let batch = |events| event_batch(1.0, events);
    let remove = gpu_remove(Some(0), vec![hash]);
    let query = MatchQuery {
        group_idx: 0,
        local_hashes: local_hashes(&[1, 2, 3, 4], 4, None)
            .iter()
            .map(|value| Arc::from(hex(value)))
            .collect(),
        tiers_of_interest: vec![StorageTier::Device],
    };
    apply_batch(
        &index,
        &source,
        0,
        0,
        &batch(vec![store.clone(), store]),
        Some(4),
    )
    .unwrap();
    apply_batch(&index, &source, 0, 0, &batch(vec![remove.clone()]), Some(4)).unwrap();
    assert_eq!(index.find_matches(&query).len(), 1);
    apply_batch(&index, &source, 0, 0, &batch(vec![remove]), Some(4)).unwrap();
    assert!(index.find_matches(&query).is_empty());
}

#[test]
fn unmodeled_extra_keys_never_alias_plain_or_named_lora_credit() {
    for (name, lora_id, extra, expected) in [
        (None, None, None, 1),
        (None, None, Some(vec![rmpv::Value::Nil]), 1),
        (None, None, Some(vec![rmpv::Value::Nil; 2]), 0),
        (
            None,
            None,
            Some(vec![rmpv::Value::Array(vec![rmpv::Value::from("salt")])]),
            0,
        ),
        (
            Some("adapter"),
            Some(7),
            Some(vec![rmpv::Value::Array(vec![rmpv::Value::from("adapter")])]),
            1,
        ),
        (
            Some("adapter"),
            Some(7),
            Some(vec![rmpv::Value::Array(vec![
                rmpv::Value::from("adapter"),
                rmpv::Value::from("salt"),
            ])]),
            0,
        ),
        (None, None, Some(vec![]), 0),
        (None, None, Some(vec![rmpv::Value::from("salt")]), 0),
        (None, Some(7), None, 0),
        (Some(""), Some(7), None, 0),
    ] {
        for full_attention_only in [false, true] {
            let index = KvBlockIndexer::new();
            let mut event = stored(
                vec![ExternalBlockHash::Int(10)],
                vec![1, 2, 3, 4],
                Some("GPU"),
                None,
            );
            event.lora_name = name.map(str::to_owned);
            event.lora_id = lora_id;
            event.extra_keys = extra.clone();
            event.session_id = Some("session-annotation".to_owned());
            assert_eq!(
                apply_batch(
                    &index,
                    &SourceId::from("p"),
                    0,
                    0,
                    &event_batch(1.0, vec![KVEvent::BlockStored(event)]),
                    full_attention_only.then_some(4),
                )
                .is_ok(),
                !full_attention_only || expected > 0
            );
            let query = MatchQuery {
                group_idx: 0,
                local_hashes: local_hashes(&[1, 2, 3, 4], 4, name)
                    .iter()
                    .map(|hash| Arc::from(hex(hash)))
                    .collect(),
                tiers_of_interest: vec![StorageTier::Device],
            };
            assert_eq!(
                index.find_matches(&query).len(),
                if full_attention_only { expected } else { 1 }
            );
        }
    }
}

#[test]
fn apply_batch_routes_ownership_to_tier_and_rank_from_batch() {
    let query = |tokens: &[u32], tier| MatchQuery {
        group_idx: 0,
        local_hashes: crate::kv_index::local_block_hash_strings(tokens, 4, None),
        tiers_of_interest: vec![tier],
    };
    for wire_rank in [None, Some(0), Some(7)] {
        let idx = KvBlockIndexer::new();
        let source = SourceId::from("w0");
        let rank = wire_rank.unwrap_or(7);
        let mut batch = event_batch(
            1.0,
            vec![
                KVEvent::BlockStored(stored(
                    vec![ExternalBlockHash::Int(10)],
                    vec![1, 2, 3, 4],
                    Some("GPU"),
                    Some("mooncake"),
                )),
                KVEvent::BlockStored(stored(
                    vec![ExternalBlockHash::Int(20)],
                    vec![5, 6, 7, 8],
                    Some("CPU"),
                    None,
                )),
            ],
        );
        batch.data_parallel_rank = wire_rank;
        apply_batch(&idx, &source, 7, 0, &batch, Some(4)).unwrap();
        for (tokens, tier, locality) in [
            ([1, 2, 3, 4], StorageTier::External, Locality::Remote),
            ([5, 6, 7, 8], StorageTier::HostPinned, Locality::Local),
        ] {
            let matches = idx.find_matches(&query(&tokens, tier));
            assert_eq!(matches.len(), 1);
            assert_eq!(matches[0].tier, tier);
            assert_eq!(matches[0].locality, locality);
            assert_eq!(&*matches[0].target.instance_id, "w0");
            assert_eq!(matches[0].target.dp_rank, rank);
        }

        // The same wire hashes at a neighboring rank survive Remove and Clear.
        let neighbor = if rank == 0 { 7 } else { 0 };
        let mut other = batch.clone();
        other.data_parallel_rank = Some(neighbor);
        apply_batch(&idx, &source, 7, 0, &other, Some(4)).unwrap();
        let host = query(&[5, 6, 7, 8], StorageTier::HostPinned);
        assert_eq!(idx.find_matches(&host).len(), 2);
        let store = batch.events[1].clone();
        batch.events = vec![KVEvent::BlockRemoved(BlockRemoved {
            block_hashes: vec![ExternalBlockHash::Int(20)],
            medium: Some("CPU".into()),
            group_idx: Some(0),
            locality: None,
            ownership: None,
        })];
        apply_batch(&idx, &source, 7, 0, &batch, Some(4)).unwrap();
        let remaining = idx.find_matches(&host);
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].target.dp_rank, neighbor);
        batch.events = vec![store, KVEvent::AllBlocksCleared(Default::default())];
        apply_batch(&idx, &source, 7, 0, &batch, Some(4)).unwrap();
        let remaining = idx.find_matches(&host);
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].target.dp_rank, neighbor);
    }
}

#[test]
fn full_attention_and_generic_subscribers_preserve_residency_boundaries() {
    let source = SourceId::from("dense");
    let valid = stored(
        vec![ExternalBlockHash::Int(1)],
        vec![1, 2, 3, 4],
        Some("GPU"),
        None,
    );
    let generic = KvBlockIndexer::new();
    for expected in [None, Some(4)] {
        let index = KvBlockIndexer::new();
        let mut unknown = valid.clone();
        unknown.medium = Some("unrecognized".into());
        assert_eq!(
            apply_batch(
                &index,
                &source,
                0,
                0,
                &event_batch(0.0, vec![KVEvent::BlockStored(unknown)]),
                expected
            )
            .unwrap_err()
            .reason,
            FullAttentionRejectionReason::Residency
        );
        assert!(hits(&index, &[1, 2, 3, 4]).is_empty());
    }
    let mut remote = valid.clone();
    remote.locality = Some("REMOTE".into());
    let batch = event_batch(0.0, vec![KVEvent::BlockStored(remote)]);
    assert!(apply_batch(&generic, &source, 0, 0, &batch, None).is_ok());
    assert_eq!(hits(&generic, &[1, 2, 3, 4]).len(), 1);
    for (medium, ownership, locality, tier) in [
        ("GPU", None, "LOCAL", StorageTier::Device),
        ("CPU", None, "LOCAL", StorageTier::HostPinned),
        ("GPU", Some("pool"), "REMOTE", StorageTier::External),
    ] {
        let index = KvBlockIndexer::new();
        let mut event = valid.clone();
        event.medium = Some(medium.into());
        event.ownership = ownership.map(str::to_owned);
        event.locality = Some(locality.into());
        let batch = KVEventBatch {
            ts: 0.0,
            events: vec![KVEvent::BlockStored(event)],
            data_parallel_rank: Some(3),
        };
        assert!(apply_batch(&index, &source, 0, 0, &batch, Some(4)).is_ok());
        let matches = index.find_matches(&MatchQuery {
            group_idx: 0,
            local_hashes: crate::kv_index::local_block_hash_strings(&[1, 2, 3, 4], 4, None),
            tiers_of_interest: vec![tier],
        });
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].tier, tier);
        assert_eq!(matches[0].target.dp_rank, 3);
    }
}

#[test]
fn wire_sequence_reads_big_endian_and_rejects_truncated_frames() {
    assert_eq!(read_seq(&42i64.to_be_bytes()), 42);
    assert_eq!(read_seq(&(-1i64).to_be_bytes()), -1);
    assert_eq!(read_seq(&[0, 1]), -1);
}

#[test]
fn wire_hash_keys_keep_integer_and_bytes_separate() {
    assert_eq!(&*hash_str(&ExternalBlockHash::Int(7)), "i:7");
    assert_eq!(
        &*hash_str(&ExternalBlockHash::Bytes(vec![0, 255])),
        "b:00ff"
    );
}

#[test]
fn local_hashes_keep_partial_blocks_and_adapter_identity() {
    let ids = [1, 2, 3, 4, 5];
    let base = local_hashes(&ids, 2, None);
    assert_eq!(base.len(), 3);
    assert!(base.iter().all(|hash| hash.len() == 32));
    assert_ne!(base, local_hashes(&ids, 2, Some("adapter-a")));
    assert!(local_hashes(&ids, 0, None).is_empty());
}

#[tokio::test]
async fn emit_drops_when_full() {
    let (tx, mut rx) = mpsc::channel::<IngestionSignal>(1);
    let signal_tx = Some(tx);
    // Buffer holds one; the second emit hits Full and drops without panic.
    emit(
        &signal_tx,
        IngestionSignal::Advance {
            source: SourceId::from("s"),
            last_seq: 1,
        },
    );
    emit(
        &signal_tx,
        IngestionSignal::Advance {
            source: SourceId::from("s"),
            last_seq: 2,
        },
    );
    // Only the first survived; the second was dropped (Full), not buffered.
    assert!(matches!(
        rx.recv().await,
        Some(IngestionSignal::Advance { last_seq: 1, .. })
    ));
    assert!(rx.try_recv().is_err());
}

/// Drive a store / remove / clear sequence through `spawn`. The `Advance`
/// signal (emitted after `apply_batch`) synchronizes the assertions; PUB
/// drops frames with no subscriber, so republish until it arrives.
#[tokio::test]
async fn spawn_applies_store_remove_clear_end_to_end() {
    use std::sync::atomic::{AtomicU32, Ordering};
    static TAG: AtomicU32 = AtomicU32::new(0);
    let tag = TAG.fetch_add(1, Ordering::Relaxed);
    let pub_path = format!("ipc:///tmp/kvtest-pub-{tag}-{}", std::process::id());

    let ctx = Arc::new(zmq::Context::new());
    let pub_sock = ctx.socket(zmq::PUB).unwrap();
    pub_sock.bind(&pub_path).unwrap();

    let indexer = Arc::new(KvBlockIndexer::new());
    let (sig_tx, mut sig_rx) = mpsc::channel::<IngestionSignal>(64);
    let handle = spawn(
        EventSource {
            source: SourceId::from("w0"),
            dp_rank: 0,
            pub_endpoint: pub_path,
            replay_endpoint: None,
            topic: "kv".into(),
            hwm: None,
        },
        indexer.clone(),
        Some(sig_tx),
        ctx.clone(),
    );

    let local: Arc<str> = Arc::from(hex(&local_hashes(&[1, 2, 3, 4], 4, None)[0]));
    let dev = |hashes: &[Arc<str>]| MatchQuery {
        group_idx: 0,
        local_hashes: hashes.to_vec(),
        tiers_of_interest: vec![StorageTier::Device],
    };

    // 1) STORE — one block (seq hash 100) on w0/rank0, GPU/Device.
    publish_until_advanced(
        &pub_sock,
        &mut sig_rx,
        b"kv",
        1,
        1.0,
        &[store_event(100, &[1, 2, 3, 4])],
        Some(0),
    )
    .await;
    let res = indexer.find_matches(&dev(std::slice::from_ref(&local)));
    assert_eq!(res.len(), 1);
    assert_eq!(&*res[0].target.instance_id, "w0");
    assert_eq!(res[0].tier, StorageTier::Device);
    assert_eq!(res[0].segments, vec![(0, 1)]);

    // 2) REMOVE — drop block 100's claim; index empties.
    publish_until_advanced(
        &pub_sock,
        &mut sig_rx,
        b"kv",
        2,
        2.0,
        &[gpu_remove(Some(0), vec![ExternalBlockHash::Int(100)])],
        Some(0),
    )
    .await;
    assert!(indexer
        .find_matches(&dev(std::slice::from_ref(&local)))
        .is_empty());

    // 3) CLEAR — AllBlocksCleared exercises the clear path on an empty index.
    publish_until_advanced(
        &pub_sock,
        &mut sig_rx,
        b"kv",
        3,
        3.0,
        &[KVEvent::AllBlocksCleared(AllBlocksCleared {})],
        Some(0),
    )
    .await;
    assert!(indexer.find_matches(&dev(&[local])).is_empty());

    handle.shutdown();
}

#[tokio::test]
async fn idle_subscriber_stops_after_shutdown() {
    let ctx = Arc::new(zmq::Context::new());
    let publisher = ctx.socket(zmq::XPUB).unwrap();
    publisher.set_linger(0).unwrap();
    publisher.bind("inproc://idle-subscriber-shutdown").unwrap();
    let (signals, mut received) = mpsc::channel(1);
    let handle = spawn(
        EventSource {
            source: SourceId::from("idle"),
            dp_rank: 0,
            pub_endpoint: "inproc://idle-subscriber-shutdown".into(),
            replay_endpoint: None,
            topic: "idle".into(),
            hwm: None,
        },
        Arc::new(KvBlockIndexer::new()),
        Some(signals),
        ctx,
    );
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            match publisher.recv_bytes(zmq::DONTWAIT) {
                Ok(frame) if frame == b"\x01idle" => break,
                Err(zmq::Error::EAGAIN) => tokio::time::sleep(Duration::from_millis(1)).await,
                other => panic!("unexpected subscription: {other:?}"),
            }
        }
    })
    .await
    .expect("subscriber did not connect");
    handle.shutdown();
    assert!(
        tokio::time::timeout(Duration::from_secs(2), received.recv())
            .await
            .expect("subscriber did not stop")
            .is_none()
    );
}

#[tokio::test]
async fn generic_subscriber_preserves_nonzero_rank_and_nondevice_tiers() {
    let ctx = Arc::new(zmq::Context::new());
    let publisher = ctx.socket(zmq::PUB).unwrap();
    publisher.bind("inproc://generic-rank-tier").unwrap();
    let indexer = Arc::new(KvBlockIndexer::new());
    let (tx, mut signals) = mpsc::channel(64);
    let handle = spawn(
        EventSource {
            source: SourceId::from("generic-worker"),
            dp_rank: 3,
            pub_endpoint: "inproc://generic-rank-tier".into(),
            replay_endpoint: None,
            topic: "kv/generic".into(),
            hwm: Some(64),
        },
        indexer.clone(),
        Some(tx),
        ctx,
    );
    let host_tokens = [1, 2, 3, 4];
    let external_tokens = [5, 6, 7, 8];
    publish_until_advanced(
        &publisher,
        &mut signals,
        b"kv/generic",
        0,
        1.0,
        &[
            KVEvent::BlockStored(stored(
                vec![ExternalBlockHash::Int(100)],
                host_tokens.to_vec(),
                Some("CPU"),
                None,
            )),
            KVEvent::BlockStored(stored(
                vec![ExternalBlockHash::Int(101)],
                external_tokens.to_vec(),
                Some("GPU"),
                Some("external-pool"),
            )),
        ],
        Some(3),
    )
    .await;
    for (tokens, tier) in [
        (host_tokens, StorageTier::HostPinned),
        (external_tokens, StorageTier::External),
    ] {
        let hits = indexer.find_matches(&MatchQuery {
            group_idx: 0,
            local_hashes: crate::kv_index::local_block_hash_strings(&tokens, 4, None),
            tiers_of_interest: vec![tier],
        });
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].target.instance_id.as_ref(), "generic-worker");
        assert_eq!(hits[0].target.dp_rank, 3);
        assert_eq!(hits[0].tier, tier);
        assert_eq!(hits[0].segments, [(0, 1)]);
    }
    handle.shutdown();
}

async fn publish_until_advanced(
    pub_sock: &zmq::Socket,
    sig_rx: &mut mpsc::Receiver<IngestionSignal>,
    topic: &[u8],
    seq: i64,
    ts: f64,
    events: &[KVEvent],
    rank: Option<u32>,
) -> Vec<i64> {
    // vLLM's EventBatch is array-like, but each tagged event is a map.
    let payload = event_payload(ts, events, rank);
    let decoded = crate::kv_index::decode_batch(&payload);
    assert!(decoded.is_ok(), "invalid publisher fixture: {decoded:?}");
    let seq_bytes = seq.to_be_bytes();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut advances = Vec::new();
    loop {
        assert!(
            tokio::time::Instant::now() < deadline,
            "subscriber never advanced to seq {seq}"
        );
        let frames: Vec<&[u8]> = vec![topic, seq_bytes.as_slice(), payload.as_slice()];
        pub_sock.send_multipart(frames, 0).unwrap();
        match tokio::time::timeout(Duration::from_millis(300), sig_rx.recv()).await {
            Ok(Some(IngestionSignal::Advance { last_seq, .. })) => {
                advances.push(last_seq);
                if last_seq == seq {
                    return advances;
                }
            }
            Ok(_) => continue,
            Err(_) if tokio::time::Instant::now() >= deadline => {
                panic!("subscriber never advanced to seq {seq}")
            }
            Err(_) => continue,
        }
    }
}

struct Fixture {
    _ctx: Arc<zmq::Context>,
    publisher: zmq::Socket,
    replay_server: Option<zmq::Socket>,
    indexer: Arc<KvBlockIndexer>,
    signals: mpsc::Receiver<IngestionSignal>,
    _handle: SubscriberHandle,
}

fn fixture(with_replay: bool) -> Fixture {
    fixture_with_capacity(with_replay, None, None, 128)
}

fn fixture_with_capacity(
    with_replay: bool,
    expected_block_size: Option<u32>,
    binding: Option<SourceLease>,
    capacity: usize,
) -> Fixture {
    use std::sync::atomic::{AtomicU32, Ordering};
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let tag = NEXT.fetch_add(1, Ordering::Relaxed);
    let ctx = Arc::new(zmq::Context::new());
    let publisher = ctx.socket(zmq::PUB).unwrap();
    let pub_endpoint = format!("inproc://kv-loss-pub-{tag}");
    publisher.bind(&pub_endpoint).unwrap();
    let replay_endpoint = format!("inproc://kv-loss-replay-{tag}");
    let replay_server = with_replay.then(|| {
        let socket = ctx.socket(zmq::ROUTER).unwrap();
        socket.bind(&replay_endpoint).unwrap();
        socket
    });
    let indexer = Arc::new(KvBlockIndexer::new());
    let (tx, signals) = mpsc::channel(capacity);
    let source = EventSource {
        source: SourceId::from("w0"),
        dp_rank: 0,
        pub_endpoint,
        replay_endpoint: with_replay.then_some(replay_endpoint),
        topic: "kv".into(),
        hwm: None,
    };
    let handle = spawn_inner(
        source,
        indexer.clone(),
        Some(tx),
        ctx.clone(),
        expected_block_size,
        binding,
    );
    Fixture {
        _ctx: ctx,
        publisher,
        replay_server,
        indexer,
        signals,
        _handle: handle,
    }
}

fn store_event(seq_hash: u64, tokens: &[u32]) -> KVEvent {
    KVEvent::BlockStored(stored(
        vec![ExternalBlockHash::Int(seq_hash)],
        tokens.to_vec(),
        Some("GPU"),
        None,
    ))
}

fn session_store_event(seq_hash: u64, tokens: &[u32], session: &str) -> KVEvent {
    let KVEvent::BlockStored(mut event) = store_event(seq_hash, tokens) else {
        unreachable!()
    };
    event.session_id = Some(session.into());
    KVEvent::BlockStored(event)
}

fn hits(index: &KvBlockIndexer, tokens: &[u32]) -> Vec<crate::kv_index::TierMatch> {
    index.find_matches(&MatchQuery {
        group_idx: 0,
        local_hashes: local_hashes(tokens, 4, None)
            .iter()
            .map(|hash| Arc::from(hex(hash)))
            .collect(),
        tiers_of_interest: vec![StorageTier::Device],
    })
}

#[tokio::test]
async fn bound_active_frames_preserve_facts_and_external_invalidation_recovers() {
    let ctx = Arc::new(zmq::Context::new());
    let publisher = ctx.socket(zmq::PUB).unwrap();
    let endpoint = "inproc://kv-bound-active-resume";
    publisher.bind(endpoint).unwrap();
    let indexer = Arc::new(KvBlockIndexer::new());
    let strict_rank = Arc::new(Mutex::new(true));
    let (tx, mut signals) = mpsc::channel(64);
    let handle = spawn_bound(
        EventSource {
            source: SourceId::from("w0"),
            dp_rank: 0,
            pub_endpoint: endpoint.into(),
            replay_endpoint: None,
            topic: "kv".into(),
            hwm: None,
        },
        indexer.clone(),
        Some(tx),
        ctx.clone(),
        Some(4),
        SourceLease {
            strict_rank: strict_rank.clone(),
            active: Arc::new(AtomicBool::new(true)),
        },
    );
    let old = [1, 2, 3, 4];
    let current = [5, 6, 7, 8];
    let fresh = [9, 10, 11, 12];
    publish_until_advanced(
        &publisher,
        &mut signals,
        b"kv",
        0,
        0.0,
        &[store_event(100, &old)],
        Some(0),
    )
    .await;
    publish_until_advanced(
        &publisher,
        &mut signals,
        b"kv",
        1,
        1.0,
        &[store_event(101, &current)],
        Some(0),
    )
    .await;
    for tokens in [&old, &current] {
        let found = hits(&indexer, tokens);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].segments, [(0, 1)]);
    }

    // External promotion/invalidation is not reflected by the subscriber's
    // local `suspended` flag. Its next owned frame must still restore credit.
    {
        let _guard = strict_rank.lock();
        indexer.invalidate_source(&SourceId::from("w0"), 0);
    }
    assert!(hits(&indexer, &current).is_empty());
    publish_until_advanced(
        &publisher,
        &mut signals,
        b"kv",
        2,
        2.0,
        &[store_event(102, &fresh)],
        Some(0),
    )
    .await;
    assert!(hits(&indexer, &old).is_empty());
    assert!(hits(&indexer, &current).is_empty());
    assert_eq!(hits(&indexer, &fresh)[0].segments, [(0, 1)]);
    publish_until_advanced(&publisher, &mut signals, b"kv", 3, 3.0, &[], Some(0)).await;
    assert_eq!(hits(&indexer, &fresh)[0].segments, [(0, 1)]);
    handle.shutdown();
}

#[tokio::test]
async fn bound_foreign_live_frames_advance_without_mutation_or_resume() {
    let strict_rank = Arc::new(Mutex::new(true));
    let mut f = fixture_with_capacity(
        false,
        Some(4),
        Some(SourceLease {
            strict_rank: strict_rank.clone(),
            active: Arc::new(AtomicBool::new(true)),
        }),
        128,
    );
    let old = [1, 2, 3, 4];
    let fresh = [5, 6, 7, 8];
    let foreign = [9, 10, 11, 12];
    publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        0,
        0.0,
        &[store_event(100, &old)],
        Some(0),
    )
    .await;
    assert_eq!(hits(&f.indexer, &old).len(), 1);
    {
        let _guard = strict_rank.lock();
        f.indexer.suspend_source(&SourceId::from("w0"), 0);
    }
    publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        1,
        1.0,
        &[store_event(200, &foreign)],
        Some(1),
    )
    .await;
    assert!(
        hits(&f.indexer, &old).is_empty(),
        "foreign rank resumed credit"
    );
    assert!(
        hits(&f.indexer, &foreign).is_empty(),
        "foreign rank acquired credit"
    );
    let advances = publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        2,
        2.0,
        &[store_event(101, &fresh)],
        Some(0),
    )
    .await;
    assert_eq!(advances, [2], "foreign sequence created an artificial gap");
    assert_eq!(
        hits(&f.indexer, &old).len(),
        1,
        "foreign frame revoked history"
    );
    assert_eq!(hits(&f.indexer, &fresh).len(), 1);

    // A genuine missing global sequence still revokes history even when
    // the observed frame belongs to another rank.
    publish_until_advanced(&f.publisher, &mut f.signals, b"kv", 4, 4.0, &[], Some(1)).await;
    assert!(hits(&f.indexer, &old).is_empty());
    assert!(hits(&f.indexer, &fresh).is_empty());
    let advances = publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        5,
        5.0,
        &[store_event(102, &fresh)],
        Some(0),
    )
    .await;
    assert_eq!(advances, [5]);
    assert_eq!(hits(&f.indexer, &fresh).len(), 1);
}

#[tokio::test]
async fn bound_replay_tracks_foreign_sequences_and_rejects_missing_rank() {
    for foreign_rank in [Some(1), None] {
        let mut f = fixture_with_capacity(
            true,
            Some(4),
            Some(SourceLease {
                strict_rank: Arc::new(Mutex::new(true)),
                active: Arc::new(AtomicBool::new(true)),
            }),
            128,
        );
        let old = [1, 2, 3, 4];
        let recovered = [5, 6, 7, 8];
        let fresh = [9, 10, 11, 12];
        let foreign = [13, 14, 15, 16];
        let server = f.replay_server.as_ref().unwrap();
        let request = replay_request(server).await;
        assert_eq!(read_seq(request.last().unwrap()), 0);
        send_replay_replies(server, &request, &[(END_SEQ, b"", vec![])]);
        publish_until_advanced(
            &f.publisher,
            &mut f.signals,
            b"kv",
            0,
            0.0,
            &[store_event(100, &old)],
            Some(0),
        )
        .await;
        assert_eq!(hits(&f.indexer, &old).len(), 1);
        let live = [store_event(102, &fresh)];
        let replies = [
            (
                1,
                b"kv".as_slice(),
                batch_payload(&[store_event(101, &recovered)], 0),
            ),
            (
                2,
                b"kv".as_slice(),
                rmp_serde::to_vec_named(&(2.0, [store_event(200, &foreign)], foreign_rank))
                    .unwrap(),
            ),
            (3, b"kv".as_slice(), batch_payload(&live, 0)),
        ];
        let query = MatchQuery {
            group_idx: 0,
            local_hashes: crate::kv_index::local_block_hash_strings(&old, 4, None),
            tiers_of_interest: vec![StorageTier::Device],
        };
        let suspended_queries = [query];
        let (advances, ()) = tokio::join!(
            publish_until_advanced(&f.publisher, &mut f.signals, b"kv", 3, 3.0, &live, Some(0)),
            serve_replay(
                f.replay_server.as_ref().unwrap(),
                &replies,
                &f.indexer,
                &suspended_queries
            ),
        );
        assert!(
            advances.contains(&1),
            "partial owned replay was not applied"
        );
        assert_eq!(advances.contains(&2), foreign_rank.is_some());
        for tokens in [&old, &recovered] {
            assert_eq!(
                hits(&f.indexer, tokens).len(),
                usize::from(foreign_rank.is_some())
            );
        }
        assert_eq!(hits(&f.indexer, &fresh).len(), 1);
        assert!(hits(&f.indexer, &foreign).is_empty());
    }
}

#[tokio::test]
async fn unrecoverable_gap_discards_old_claims_but_preserves_other_source_and_rank() {
    let mut f = fixture(false);
    let tokens = [1, 2, 3, 4];
    publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        0,
        0.0,
        &[store_event(100, &tokens)],
        Some(0),
    )
    .await;
    for (source, rank) in [("w0", 1), ("w1", 0)] {
        apply_batch(
            &f.indexer,
            &SourceId::from(source),
            rank,
            0,
            &KVEventBatch {
                ts: 0.0,
                events: vec![store_event(100, &tokens)],
                data_parallel_rank: Some(rank),
            },
            None,
        )
        .unwrap();
    }
    // Seq 1 was a lost eviction; seq 2 contains no replacement store.
    publish_until_advanced(&f.publisher, &mut f.signals, b"kv", 2, 2.0, &[], Some(0)).await;
    let remaining = hits(&f.indexer, &tokens);
    assert_eq!(remaining.len(), 2);
    assert!(remaining
        .iter()
        .all(|hit| &*hit.target.instance_id != "w0" || hit.target.dp_rank != 0));
    publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        3,
        3.0,
        &[store_event(101, &tokens)],
        Some(0),
    )
    .await;
    assert_eq!(hits(&f.indexer, &tokens).len(), 3);
}

#[tokio::test]
async fn repeated_sequence_zero_replaces_previous_incarnation() {
    let mut f = fixture(false);
    let old = [1, 2, 3, 4];
    let current = [5, 6, 7, 8];
    publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        0,
        0.0,
        &[store_event(100, &old)],
        Some(0),
    )
    .await;
    assert_eq!(hits(&f.indexer, &old).len(), 1);
    apply_batch(
        &f.indexer,
        &SourceId::from("w1"),
        0,
        0,
        &KVEventBatch {
            ts: 0.0,
            events: vec![store_event(100, &old)],
            data_parallel_rank: Some(0),
        },
        None,
    )
    .unwrap();

    // A new publisher starts again at zero after the old publisher sent
    // only one batch. Its first event does not mention the old prefix.
    publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        0,
        1.0,
        &[store_event(101, &current)],
        Some(0),
    )
    .await;
    let surviving = hits(&f.indexer, &old);
    assert_eq!(surviving.len(), 1);
    assert_eq!(surviving[0].target.instance_id.as_ref(), "w1");
    assert_eq!(hits(&f.indexer, &current).len(), 1);
}

#[tokio::test]
async fn restart_fences_warm_state_when_signal_queue_is_full() {
    let mut f = fixture_with_capacity(false, None, None, 1);
    let old = [1, 2, 3, 4];
    let current = [5, 6, 7, 8];
    publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        0,
        0.0,
        &[store_event(100, &old)],
        Some(0),
    )
    .await;
    let payload = batch_payload(&[], 0);
    f.publisher
        .send_multipart(
            [b"kv".as_slice(), &1_i64.to_be_bytes(), payload.as_slice()],
            0,
        )
        .unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        while f.signals.len() != 1 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    let before = crate::kv_index::signal_loss_epoch();
    let payload = batch_payload(&[store_event(101, &current)], 0);
    f.publisher
        .send_multipart(
            [b"kv".as_slice(), &0_i64.to_be_bytes(), payload.as_slice()],
            0,
        )
        .unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        while hits(&f.indexer, &current).is_empty() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    assert!(hits(&f.indexer, &old).is_empty());
    assert!(crate::kv_index::signal_loss_epoch() > before);
    assert!(matches!(
        f.signals.try_recv(),
        Ok(IngestionSignal::Advance { last_seq: 1, .. })
    ));
    assert!(f.signals.try_recv().is_err());
}

#[tokio::test]
async fn gap_fences_warm_state_before_replay_completes() {
    let mut f = fixture(true);
    let old = [1, 2, 3, 4];
    publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        0,
        0.0,
        &[store_event(100, &old)],
        Some(0),
    )
    .await;
    let before = crate::kv_index::signal_loss_epoch();
    let payload = batch_payload(&[], 0);
    f.publisher
        .send_multipart(
            [b"kv".as_slice(), &2_i64.to_be_bytes(), payload.as_slice()],
            0,
        )
        .unwrap();
    let request = replay_request(f.replay_server.as_ref().unwrap()).await;
    assert_eq!(read_seq(request.last().unwrap()), 1);
    assert!(hits(&f.indexer, &old).is_empty());
    assert!(crate::kv_index::signal_loss_epoch() > before);
    // Stop the subscriber without completing the replay.
    f._handle.shutdown();
}

#[tokio::test]
async fn malformed_payload_does_not_advance_and_same_seq_can_be_retried() {
    let mut f = fixture(false);
    let tokens = [1, 2, 3, 4];
    publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        0,
        0.0,
        &[store_event(100, &tokens)],
        Some(0),
    )
    .await;
    let seq = 1i64.to_be_bytes();
    f.publisher
        .send_multipart([b"kv".as_slice(), seq.as_slice(), &[0xc1]], 0)
        .unwrap();
    let signal = tokio::time::timeout(Duration::from_secs(5), f.signals.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        signal,
        IngestionSignal::Gap {
            from_seq: 1,
            to_seq: 1,
            ..
        }
    ));
    assert!(hits(&f.indexer, &tokens).is_empty());
    assert!(
        f.signals.try_recv().is_err(),
        "bad payload must not emit Advance"
    );
    publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        1,
        1.0,
        &[store_event(100, &tokens)],
        Some(0),
    )
    .await;
    assert_eq!(hits(&f.indexer, &tokens).len(), 1);
}

#[tokio::test]
async fn live_topic_prefix_does_not_publish_or_consume_the_sequence() {
    let mut f = fixture_with_capacity(false, Some(4), None, 128);
    publish_until_advanced(&f.publisher, &mut f.signals, b"kv", 0, 0.0, &[], Some(0)).await;
    let unrelated = [1, 2, 3, 4];
    let expected = [5, 6, 7, 8];
    let payload = batch_payload(&[store_event(100, &unrelated)], 0);
    f.publisher
        .send_multipart([b"kv-other".as_slice(), &1i64.to_be_bytes(), &payload], 0)
        .unwrap();
    publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        1,
        1.0,
        &[session_store_event(101, &expected, "session-live")],
        Some(0),
    )
    .await;
    assert!(hits(&f.indexer, &unrelated).is_empty());
    let matches = hits(&f.indexer, &expected);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].target.instance_id.as_ref(), "w0");
    assert_eq!(matches[0].target.dp_rank, 0);
}

#[tokio::test]
async fn live_unknown_field_rejects_the_whole_batch_without_advance() {
    let mut f = fixture(false);
    publish_until_advanced(&f.publisher, &mut f.signals, b"kv", 0, 0.0, &[], Some(0)).await;
    let tokens = [1, 2, 3, 4];
    let mut event = serde_json::to_value(store_event(100, &tokens)).unwrap();
    event["future_namespace"] = serde_json::json!("not-plain");
    let payload = rmp_serde::to_vec(&(
        1.0,
        [
            serde_json::to_value(store_event(101, &[5, 6, 7, 8])).unwrap(),
            event,
        ],
        0u32,
    ))
    .unwrap();
    f.publisher
        .send_multipart([b"kv".as_slice(), &1i64.to_be_bytes(), &payload], 0)
        .unwrap();
    let signal = tokio::time::timeout(Duration::from_secs(5), f.signals.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        signal,
        IngestionSignal::Gap {
            from_seq: 1,
            to_seq: 1,
            ..
        }
    ));
    assert!(hits(&f.indexer, &tokens).is_empty());
    assert!(hits(&f.indexer, &[5, 6, 7, 8]).is_empty());
    assert!(
        f.signals.try_recv().is_err(),
        "bad batch must not emit Advance"
    );
    publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        1,
        1.0,
        &[store_event(100, &tokens)],
        Some(0),
    )
    .await;
    assert_eq!(hits(&f.indexer, &tokens).len(), 1);
}

fn batch_payload(events: &[KVEvent], rank: u32) -> Vec<u8> {
    event_payload(1.0, events, Some(rank))
}

#[tokio::test]
async fn bound_quiet_start_replays_and_catches_up_before_live() {
    let mut f = fixture_with_capacity(
        true,
        Some(4),
        Some(SourceLease {
            strict_rank: Arc::new(Mutex::new(true)),
            active: Arc::new(AtomicBool::new(true)),
        }),
        128,
    );
    let server = f.replay_server.as_ref().unwrap();
    let tokens = [1, 2, 3, 4];
    let request = replay_request(server).await;
    assert_eq!(read_seq(request.last().unwrap()), 0);
    send_replay_replies(
        server,
        &request,
        &[
            (0, b"kv", batch_payload(&[store_event(100, &tokens)], 0)),
            (END_SEQ, b"", vec![]),
        ],
    );
    tokio::time::timeout(Duration::from_secs(3), async {
        while hits(&f.indexer, &tokens).is_empty() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    let request = replay_request(server).await;
    assert_eq!(read_seq(request.last().unwrap()), 1);
    send_replay_replies(
        server,
        &request,
        &[
            (
                1,
                b"kv",
                batch_payload(&[KVEvent::AllBlocksCleared(AllBlocksCleared::default())], 0),
            ),
            (END_SEQ, b"", vec![]),
        ],
    );
    tokio::time::timeout(Duration::from_secs(3), async {
        while !hits(&f.indexer, &tokens).is_empty() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        2,
        2.0,
        &[store_event(100, &tokens)],
        Some(0),
    )
    .await;
    assert_eq!(hits(&f.indexer, &tokens).len(), 1);

    // A final eviction can be missed even after the live handoff succeeded.
    let request = replay_request(server).await;
    assert_eq!(read_seq(request.last().unwrap()), 3);
    // A normal vLLM replay response can wait for the publisher's queue poll.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(hits(&f.indexer, &tokens).len(), 1);
    send_replay_replies(server, &request, &[(END_SEQ, b"", vec![])]);
    let request = replay_request(server).await;
    assert_eq!(read_seq(request.last().unwrap()), 3);
    assert_eq!(hits(&f.indexer, &tokens).len(), 1);
    let removal = gpu_remove(Some(0), vec![ExternalBlockHash::Int(100)]);
    let payload = batch_payload(&[removal], 0);
    send_replay_replies(
        server,
        &request,
        &[(3, b"kv", payload.clone()), (END_SEQ, b"", vec![])],
    );
    f.publisher
        .send_multipart([b"kv".as_slice(), &3_i64.to_be_bytes(), &payload], 0)
        .unwrap();
    publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        4,
        4.0,
        &[store_event(100, &tokens)],
        Some(0),
    )
    .await;
    assert_eq!(hits(&f.indexer, &tokens).len(), 1);

    // A failed idle check must leave the source hidden until full replay works.
    let request = replay_request(server).await;
    assert_eq!(read_seq(request.last().unwrap()), 5);
    assert_eq!(hits(&f.indexer, &tokens).len(), 1);
    let request = tokio::time::timeout(Duration::from_secs(8), async {
        loop {
            match server.recv_multipart(zmq::DONTWAIT) {
                Ok(frames) => break frames,
                Err(zmq::Error::EAGAIN) => tokio::time::sleep(Duration::from_millis(1)).await,
                Err(error) => panic!("replay request failed: {error}"),
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(read_seq(request.last().unwrap()), 0);
    assert!(hits(&f.indexer, &tokens).is_empty());
    let store = batch_payload(&[store_event(100, &tokens)], 0);
    let clear = batch_payload(&[KVEvent::AllBlocksCleared(AllBlocksCleared::default())], 0);
    send_replay_replies(
        server,
        &request,
        &[
            (0, b"kv", store.clone()),
            (1, b"kv", clear),
            (2, b"kv", store.clone()),
            (3, b"kv", payload),
            (4, b"kv", store),
            (END_SEQ, b"", vec![]),
        ],
    );
    tokio::time::timeout(Duration::from_secs(3), async {
        while hits(&f.indexer, &tokens).is_empty() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn bound_quiet_catchup_gap_invalidates_visible_credit() {
    let mut f = fixture_with_capacity(
        true,
        Some(4),
        Some(SourceLease {
            strict_rank: Arc::new(Mutex::new(true)),
            active: Arc::new(AtomicBool::new(true)),
        }),
        128,
    );
    let server = f.replay_server.as_ref().unwrap();
    let tokens = [1, 2, 3, 4];
    let request = replay_request(server).await;
    send_replay_replies(
        server,
        &request,
        &[
            (0, b"kv", batch_payload(&[store_event(100, &tokens)], 0)),
            (END_SEQ, b"", vec![]),
        ],
    );
    tokio::time::timeout(Duration::from_secs(3), async {
        while hits(&f.indexer, &tokens).is_empty() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    let request = replay_request(server).await;
    assert_eq!(read_seq(request.last().unwrap()), 1);
    send_replay_replies(
        server,
        &request,
        &[(
            2,
            b"kv",
            batch_payload(&[store_event(101, &[5, 6, 7, 8])], 0),
        )],
    );
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if matches!(
                f.signals.recv().await,
                Some(IngestionSignal::Invalidated { .. })
            ) {
                break;
            }
        }
    })
    .await
    .unwrap();
    assert!(hits(&f.indexer, &tokens).is_empty());
}

#[test]
fn full_attention_session_annotations_preserve_shared_and_unrelated_blocks() {
    let index = KvBlockIndexer::new();
    let source = SourceId::from("w0");
    let shared = [1, 2, 3, 4];
    let retained = [5, 6, 7, 8];
    for (seq, session, blocks) in [
        (0, "session-a", vec![(100, shared), (101, retained)]),
        (1, "session-b", vec![(100, shared)]),
    ] {
        let events: Vec<_> = blocks
            .into_iter()
            .map(|(hash, tokens)| session_store_event(hash, &tokens, session))
            .collect();
        let batch = event_batch(seq as f64, events);
        assert!(apply_batch(&index, &source, 0, 0, &batch, Some(4)).is_ok());
        for tokens in [&shared, &retained] {
            let matches = hits(&index, tokens);
            assert_eq!(matches.len(), 1, "session annotation erased placement");
            assert_eq!(matches[0].target.instance_id.as_ref(), "w0");
            assert_eq!(matches[0].target.dp_rank, 0);
        }
    }
}

#[test]
fn full_attention_block_size_mismatch_preflights_stores_and_removes_atomically() {
    for (actual, reason) in [
        (0, FullAttentionRejectionReason::ZeroBlockSize),
        (8, FullAttentionRejectionReason::BlockSizeMismatch),
    ] {
        let index = KvBlockIndexer::new();
        let source = SourceId::from("dense");
        let initial = event_batch(0.0, vec![store_event(100, &[1, 2, 3, 4])]);
        assert!(apply_batch(&index, &source, 0, 0, &initial, Some(4)).is_ok());
        let mut bad = stored(
            vec![ExternalBlockHash::Int(102)],
            (20..28).collect(),
            Some("GPU"),
            None,
        );
        bad.block_size = actual;
        let batch = KVEventBatch {
            ts: 1.0,
            events: vec![
                store_event(101, &[5, 6, 7, 8]),
                gpu_remove(Some(0), vec![ExternalBlockHash::Int(100)]),
                KVEvent::AllBlocksCleared(AllBlocksCleared {}),
                KVEvent::BlockStored(bad),
            ],
            data_parallel_rank: Some(0),
        };
        assert_eq!(
            apply_batch(&index, &source, 0, 0, &batch, Some(4)),
            Err(FullAttentionBatchRejection {
                reason,
                event_index: 3
            })
        );
        assert_eq!(
            hits(&index, &[1, 2, 3, 4]).len(),
            1,
            "rejected batch applied an eviction"
        );
        assert!(
            hits(&index, &[5, 6, 7, 8]).is_empty(),
            "rejected batch published a valid store"
        );
        assert!(hits(&index, &(20..28).collect::<Vec<_>>()).is_empty());
    }
}

#[tokio::test]
async fn full_attention_replay_block_size_mismatch_clears_partial_recovery() {
    for (expected, actual) in [(16, 8), (8, 16)] {
        let mut f = fixture_with_capacity(true, Some(expected), None, 128);
        let old: Vec<u32> = (0..expected).collect();
        let partial: Vec<u32> = (100..100 + expected).collect();
        let bad: Vec<u32> = (200..200 + actual).collect();
        let fresh: Vec<u32> = (300..300 + expected).collect();
        let event = |hash, tokens: &[u32], size| {
            let mut value = stored(
                vec![ExternalBlockHash::Int(hash)],
                tokens.to_vec(),
                Some("GPU"),
                None,
            );
            value.block_size = size;
            KVEvent::BlockStored(value)
        };
        publish_until_advanced(
            &f.publisher,
            &mut f.signals,
            b"kv",
            0,
            0.0,
            &[event(100, &old, expected)],
            Some(0),
        )
        .await;
        let query = |tokens: &[u32]| MatchQuery {
            group_idx: 0,
            local_hashes: crate::kv_index::local_block_hash_strings(tokens, expected, None),
            tiers_of_interest: vec![StorageTier::Device],
        };
        let old_hits = f.indexer.find_matches(&query(&old));
        assert_eq!(old_hits.len(), 1);
        assert_eq!(old_hits[0].target.instance_id.as_ref(), "w0");
        assert_eq!(old_hits[0].target.dp_rank, 0);
        let live = [event(400, &fresh, expected)];
        let replies = [
            (
                1,
                b"kv".as_slice(),
                batch_payload(&[event(200, &partial, expected)], 0),
            ),
            (
                2,
                b"kv".as_slice(),
                batch_payload(&[event(300, &bad, actual)], 0),
            ),
            (3, b"kv".as_slice(), batch_payload(&live, 0)),
        ];
        let suspended_queries = [query(&old)];
        let (advances, ()) = tokio::join!(
            publish_until_advanced(&f.publisher, &mut f.signals, b"kv", 3, 3.0, &live, Some(0)),
            serve_replay(
                f.replay_server.as_ref().unwrap(),
                &replies,
                &f.indexer,
                &suspended_queries
            ),
        );
        assert!(
            advances.contains(&1),
            "partial replay Store was not applied"
        );
        assert!(f.indexer.find_matches(&query(&old)).is_empty());
        assert!(f.indexer.find_matches(&query(&partial)).is_empty());
        assert!(f.indexer.find_matches(&query(&bad)).is_empty());
        assert_eq!(f.indexer.find_matches(&query(&fresh)).len(), 1);
    }
}

async fn serve_replay(
    socket: &zmq::Socket,
    replies: &[(i64, &[u8], Vec<u8>)],
    index: &KvBlockIndexer,
    suspended_queries: &[MatchQuery],
) {
    let request = replay_request(socket).await;
    assert_eq!(read_seq(request.last().unwrap()), 1);
    // No previously indexed credit may escape while backfill is incomplete.
    for query in suspended_queries {
        assert!(
            index.find_matches(query).is_empty(),
            "credit escaped during replay"
        );
    }
    send_replay_replies(socket, &request, replies);
}

async fn replay_request(socket: &zmq::Socket) -> Vec<Vec<u8>> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        match socket.recv_multipart(zmq::DONTWAIT) {
            Ok(request) => return request,
            Err(zmq::Error::EAGAIN) => {
                assert!(tokio::time::Instant::now() < deadline, "no replay request");
                tokio::task::yield_now().await;
            }
            Err(e) => panic!("replay request failed: {e}"),
        }
    }
}

fn send_replay_replies(
    socket: &zmq::Socket,
    request: &[Vec<u8>],
    replies: &[(i64, &[u8], Vec<u8>)],
) {
    for (seq, topic, payload) in replies {
        socket
            .send_multipart(
                [
                    request[0].clone(),
                    vec![],
                    if *seq == END_SEQ {
                        vec![]
                    } else {
                        topic.to_vec()
                    },
                    seq.to_be_bytes().to_vec(),
                    payload.clone(),
                ],
                0,
            )
            .unwrap();
    }
}

async fn check_replay_case(case: &str) {
    let mut f = fixture_with_capacity(true, (case == "complete").then_some(4), None, 128);
    let old = [1, 2, 3, 4];
    let keep = [5, 6, 7, 8];
    let partial = [9, 10, 11, 12];
    let fresh = [13, 14, 15, 16];
    publish_until_advanced(
        &f.publisher,
        &mut f.signals,
        b"kv",
        0,
        0.0,
        &[store_event(100, &old), store_event(101, &keep)],
        Some(0),
    )
    .await;
    let query = |tokens: &[u32]| MatchQuery {
        group_idx: 0,
        local_hashes: crate::kv_index::local_block_hash_strings(tokens, 4, None),
        tiers_of_interest: vec![StorageTier::Device],
    };
    for tokens in [&old, &keep] {
        let matches = f.indexer.find_matches(&query(tokens));
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].target.instance_id.as_ref(), "w0");
        assert_eq!(matches[0].target.dp_rank, 0);
    }
    let first = batch_payload(&[session_store_event(200, &partial, "session-replay")], 0);
    let eviction = batch_payload(&[gpu_remove(Some(0), vec![ExternalBlockHash::Int(100)])], 0);
    let third = batch_payload(&[store_event(400, &fresh)], 0);
    let eof = (-1, b"".as_slice(), vec![]);
    let topic = b"kv".as_slice();
    let replies = match case {
        "complete" => vec![
            // An old replayed Clear must not erase the valid historical prefix.
            (
                0,
                topic,
                batch_payload(&[KVEvent::AllBlocksCleared(Default::default())], 0),
            ),
            (1, topic, first),
            (2, topic, eviction),
            (3, topic, third),
        ],
        "missing_start" => vec![(2, topic, eviction), eof],
        "middle_gap" => vec![(1, topic, first), (3, topic, third), eof],
        "corrupt" => vec![(1, topic, first), (2, topic, vec![0xc1]), eof],
        "early_eof" => vec![(1, topic, first), eof],
        "wrong_rank" => vec![(1, topic, first), (2, topic, batch_payload(&[], 1)), eof],
        "wrong_topic" => vec![
            (1, topic, first),
            (2, b"kv-other".as_slice(), eviction),
            (3, topic, third),
        ],
        _ => panic!("unknown replay case"),
    };
    let live_events = [store_event(400, &fresh)];
    let suspended_queries = [query(&old), query(&keep)];
    let (advances, ()) = tokio::join!(
        publish_until_advanced(
            &f.publisher,
            &mut f.signals,
            b"kv",
            3,
            3.0,
            &live_events,
            Some(0)
        ),
        serve_replay(
            f.replay_server.as_ref().unwrap(),
            &replies,
            &f.indexer,
            &suspended_queries
        ),
    );
    if case != "missing_start" {
        assert!(
            advances.contains(&1),
            "partial replay Store was not applied: {case}"
        );
    }
    assert!(
        hits(&f.indexer, &old).is_empty(),
        "lost eviction must not leave old credit: {case}"
    );
    assert_eq!(
        hits(&f.indexer, &keep).len(),
        usize::from(case == "complete"),
        "retain valid historical credit only on complete replay: {case}"
    );
    assert_eq!(
        hits(&f.indexer, &partial).len(),
        usize::from(case == "complete"),
        "partial replay must not escape: {case}"
    );
    if case == "complete" {
        let matches = hits(&f.indexer, &partial);
        assert_eq!(matches[0].target.instance_id.as_ref(), "w0");
        assert_eq!(matches[0].target.dp_rank, 0);
    }
    assert_eq!(
        hits(&f.indexer, &fresh).len(),
        1,
        "fresh observations restore credit: {case}"
    );
}

#[tokio::test]
async fn replay_requires_continuous_history_and_valid_topic_rank_and_payload() {
    for case in [
        "complete",
        "missing_start",
        "middle_gap",
        "corrupt",
        "early_eof",
        "wrong_rank",
        "wrong_topic",
    ] {
        check_replay_case(case).await;
    }
}

// Isolated reproduction: a live TCP publisher disconnects after Store and
// remains silent while the simulated engine evicts the page.
#[tokio::test]
async fn tcp_disconnect_fences_until_replay_and_live_handoff() {
    tcp_full_recovery(false).await;
}

#[tokio::test]
async fn tcp_monitor_failure_rebuilds_and_replays() {
    tcp_full_recovery(true).await;
}

pub(crate) async fn wait_until(ready: impl Fn() -> bool) {
    tokio::time::timeout(Duration::from_secs(4), async {
        while !ready() {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .expect("subscriber state did not converge");
}

/// Wire fixture shared by full and layout recovery; their cache assertions stay separate.
pub(crate) struct TcpReplay {
    pub(crate) ctx: Arc<zmq::Context>,
    replay: Option<zmq::Socket>,
    publisher: Option<zmq::Socket>,
    endpoint: String,
}

impl TcpReplay {
    pub(crate) fn new() -> Self {
        Self::with_replay(true)
    }

    fn with_replay(with_replay: bool) -> Self {
        let ctx = Arc::new(zmq::Context::new());
        let publisher = ctx.socket(zmq::XPUB).unwrap();
        publisher.set_linger(0).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
        let replay = with_replay.then(|| {
            let replay = ctx.socket(zmq::ROUTER).unwrap();
            replay.set_linger(0).unwrap();
            replay.bind("inproc://tcp-recovery-replay").unwrap();
            replay
        });
        Self {
            ctx,
            replay,
            publisher: Some(publisher),
            endpoint,
        }
    }

    pub(crate) fn source(&self, name: &str) -> EventSource {
        EventSource {
            source: Arc::from(name),
            dp_rank: 0,
            pub_endpoint: self.endpoint.clone(),
            replay_endpoint: self
                .replay
                .as_ref()
                .map(|_| "inproc://tcp-recovery-replay".into()),
            topic: "kv".into(),
            hwm: Some(64),
        }
    }

    pub(crate) async fn subscribed(&self) {
        let frames = replay_request(self.publisher.as_ref().unwrap()).await;
        assert_eq!(frames[0], b"\x01kv");
    }

    pub(crate) async fn request(&self, start: u64) -> Vec<Vec<u8>> {
        let frames = replay_request(self.replay.as_ref().unwrap()).await;
        assert_eq!(frames[2], start.to_be_bytes());
        frames
    }

    pub(crate) fn live(&self, seq: i64, payload: &[u8]) {
        self.publisher
            .as_ref()
            .unwrap()
            .send_multipart([b"kv".as_slice(), &seq.to_be_bytes(), payload], 0)
            .unwrap();
    }

    pub(crate) fn history(&self, request: &[Vec<u8>], frames: &[(i64, &[u8])]) {
        for (seq, payload) in frames.iter().copied().chain(std::iter::once((-1, &[][..]))) {
            self.reply(request, seq, payload);
        }
    }

    fn reply(&self, request: &[Vec<u8>], seq: i64, payload: &[u8]) {
        let topic = if seq == -1 { &[][..] } else { b"kv" };
        self.replay
            .as_ref()
            .unwrap()
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

    pub(crate) async fn reconnect(&mut self, monitor_failure: bool, cold: impl Fn() -> bool) {
        if monitor_failure {
            crate::kv_index::source::events::transport::fail_next_monitor_drain(&self.endpoint);
        } else {
            self.disconnect();
        }
        wait_until(cold).await;
        if !monitor_failure {
            if let Some(replay) = &self.replay {
                assert_eq!(
                    replay.recv_multipart(zmq::DONTWAIT),
                    Err(zmq::Error::EAGAIN),
                    "replay started before SUB reconnect"
                );
            }
            let publisher = self.ctx.socket(zmq::XPUB).unwrap();
            publisher.set_linger(0).unwrap();
            publisher.bind(&self.endpoint).unwrap();
            self.publisher = Some(publisher);
            self.subscribed().await;
        }
    }
}

async fn tcp_full_recovery(monitor_failure: bool) {
    let mut tcp = TcpReplay::new();
    let indexer = Arc::new(KvBlockIndexer::new());
    let (tx, mut signals) = mpsc::channel(64);
    let handle = spawn_bound(
        tcp.source("tcp-worker"),
        indexer.clone(),
        Some(tx),
        tcp.ctx.clone(),
        Some(4),
        SourceLease {
            strict_rank: Arc::new(Mutex::new(true)),
            active: Arc::new(AtomicBool::new(true)),
        },
    );
    tcp.subscribed().await;
    let request = tcp.request(0).await;
    tcp.history(&request, &[]);
    let prompt = [1, 2, 3, 4];
    let payload = batch_payload(&[store_event(111, &prompt)], 0);
    tcp.live(0, &payload);
    let signal = tokio::time::timeout(Duration::from_secs(3), signals.recv())
        .await
        .unwrap();
    assert!(
        matches!(signal, Some(IngestionSignal::Advance { last_seq: 0, .. })),
        "unexpected {signal:?}"
    );
    assert_eq!(hits(&indexer, &prompt).len(), 1);
    tokio::time::sleep(Duration::from_millis(250)).await;
    assert_eq!(hits(&indexer, &prompt).len(), 1);

    tcp.reconnect(monitor_failure, || hits(&indexer, &prompt).is_empty())
        .await;
    let request = tcp.request(0).await;
    tcp.history(&request, &[(2, &payload)]);
    let request = tcp.request(0).await;
    assert!(
        hits(&indexer, &prompt).is_empty(),
        "truncated history restored credit"
    );
    tcp.history(&request, &[(0, &payload)]);
    wait_until(|| !hits(&indexer, &prompt).is_empty()).await;

    // Full attention first catches up from its cursor; after an incomplete
    // answer it must rebuild and cover the highest sequence observed live.
    tcp.live(2, &payload);
    let request = tcp.request(1).await;
    tcp.history(&request, &[]);
    let request = tcp.request(0).await;
    assert!(
        hits(&indexer, &prompt).is_empty(),
        "incomplete replay restored credit"
    );
    tcp.history(&request, &[(0, &payload)]);
    let request = tcp.request(0).await;
    assert!(
        hits(&indexer, &prompt).is_empty(),
        "replay did not cover observed seq 2"
    );
    let empty = batch_payload(&[], 0);
    tcp.history(&request, &[(0, &payload), (1, &empty), (2, &payload)]);
    wait_until(|| !hits(&indexer, &prompt).is_empty()).await;
    tcp.live(3, &payload);
    handle.shutdown();
}

#[tokio::test]
async fn tcp_disconnect_without_replay_recovers_only_new_live_facts() {
    let mut tcp = TcpReplay::with_replay(false);
    let indexer = Arc::new(KvBlockIndexer::new());
    let (tx, mut signals) = mpsc::channel(64);
    let handle = spawn_bound(
        tcp.source("tcp-no-replay"),
        indexer.clone(),
        Some(tx),
        tcp.ctx.clone(),
        Some(4),
        SourceLease {
            strict_rank: Arc::new(Mutex::new(true)),
            active: Arc::new(AtomicBool::new(true)),
        },
    );
    tcp.subscribed().await;
    let old = [1, 2, 3, 4];
    let fresh = [5, 6, 7, 8];
    let payload = batch_payload(&[store_event(100, &old)], 0);
    tcp.live(0, &payload);
    tokio::time::timeout(Duration::from_secs(3), signals.recv())
        .await
        .unwrap();
    assert_eq!(hits(&indexer, &old).len(), 1);
    tcp.reconnect(false, || hits(&indexer, &old).is_empty())
        .await;
    assert!(hits(&indexer, &old).is_empty());
    let payload = event_payload(2.0, &[store_event(200, &fresh)], Some(0));
    tcp.live(2, &payload);
    wait_until(|| !hits(&indexer, &fresh).is_empty()).await;
    assert!(hits(&indexer, &old).is_empty());
    handle.shutdown();
}

#[tokio::test]
async fn tcp_disconnect_only_fences_affected_source() {
    let index = Arc::new(KvBlockIndexer::new());
    let mut publishers = Vec::new();
    let mut handles = Vec::new();
    let mut receivers = Vec::new();
    for name in ["worker-one", "worker-two"] {
        let tcp = TcpReplay::with_replay(false);
        let (tx, rx) = mpsc::channel(64);
        let handle = spawn_bound(
            tcp.source(name),
            index.clone(),
            Some(tx),
            tcp.ctx.clone(),
            Some(4),
            SourceLease {
                strict_rank: Arc::new(Mutex::new(true)),
                active: Arc::new(AtomicBool::new(true)),
            },
        );
        tcp.subscribed().await;
        publishers.push(tcp);
        receivers.push(rx);
        handles.push(handle);
    }
    for (i, prompt) in [[1, 2, 3, 4], [5, 6, 7, 8]].iter().enumerate() {
        let payload = batch_payload(&[store_event(111 + i as u64, prompt)], 0);
        publishers[i].live(0, &payload);
        tokio::time::timeout(Duration::from_secs(3), receivers[i].recv())
            .await
            .unwrap();
        assert_eq!(hits(&index, prompt).len(), 1);
    }
    publishers[0].disconnect();
    wait_until(|| hits(&index, &[1, 2, 3, 4]).is_empty()).await;
    assert_eq!(
        hits(&index, &[5, 6, 7, 8]).len(),
        1,
        "other source incorrectly fenced"
    );
    for h in handles {
        h.shutdown();
    }
}

#[tokio::test]
async fn tcp_old_lease_disconnect_cannot_clear_replacement_source() {
    let index = Arc::new(KvBlockIndexer::new());
    let old_active = Arc::new(AtomicBool::new(true));
    let mut pubs = Vec::new();
    let mut handles = Vec::new();
    let mut receivers = Vec::new();
    for (i, active) in [old_active.clone(), Arc::new(AtomicBool::new(true))]
        .into_iter()
        .enumerate()
    {
        if i == 1 {
            old_active.store(false, Ordering::Release);
            index.invalidate_source(&SourceId::from("replacement"), 0);
        }
        let tcp = TcpReplay::with_replay(false);
        let (tx, rx) = mpsc::channel(64);
        let handle = spawn_bound(
            tcp.source("replacement"),
            index.clone(),
            Some(tx),
            tcp.ctx.clone(),
            Some(4),
            SourceLease {
                strict_rank: Arc::new(Mutex::new(true)),
                active,
            },
        );
        tcp.subscribed().await;
        pubs.push(tcp);
        receivers.push(rx);
        handles.push(handle);
        let prompt = if i == 0 { [1, 2, 3, 4] } else { [5, 6, 7, 8] };
        let payload = batch_payload(&[store_event(111 + i as u64, &prompt)], 0);
        pubs[i].live(0, &payload);
        tokio::time::timeout(Duration::from_secs(3), receivers[i].recv())
            .await
            .unwrap();
        assert_eq!(hits(&index, &prompt).len(), 1);
    }
    pubs[0].disconnect();
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        hits(&index, &[5, 6, 7, 8]).len(),
        1,
        "retired source cleared replacement"
    );
    for h in handles {
        h.shutdown();
    }
}

#[tokio::test]
async fn tcp_disconnect_during_gap_replay_never_republishes_old_source() {
    let mut tcp = TcpReplay::new();
    let index = Arc::new(KvBlockIndexer::new());
    let (tx, mut signals) = mpsc::channel(64);
    let handle = spawn_bound(
        tcp.source("gap-worker"),
        index.clone(),
        Some(tx),
        tcp.ctx.clone(),
        Some(4),
        SourceLease {
            strict_rank: Arc::new(Mutex::new(true)),
            active: Arc::new(AtomicBool::new(true)),
        },
    );
    tcp.subscribed().await;
    let request = tcp.request(0).await;
    tcp.history(&request, &[]);
    let prompt = [1, 2, 3, 4];
    let payload = batch_payload(&[store_event(111, &prompt)], 0);
    tcp.live(0, &payload);
    tokio::time::timeout(Duration::from_secs(3), signals.recv())
        .await
        .unwrap();
    assert_eq!(hits(&index, &prompt).len(), 1);
    tcp.live(2, &payload);
    let request = tcp.request(1).await;
    assert!(hits(&index, &prompt).is_empty());
    tcp.disconnect();
    tokio::time::sleep(Duration::from_millis(200)).await;
    let empty = event_payload(2.0, &[], Some(0));
    tcp.reply(&request, 1, &empty);
    tcp.reply(&request, 2, &payload);
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(
        hits(&index, &prompt).is_empty(),
        "replay exposed cache while live channel was disconnected"
    );
    handle.shutdown();
}

#[test]
fn monitor_failure_fences_only_active_source_and_respects_stale_lease() {
    let index = Arc::new(KvBlockIndexer::new());
    let source_a = SourceId::from("monitor-a");
    let source_b = SourceId::from("monitor-b");
    let source_replaced = SourceId::from("monitor-replaced");
    for (source, hash, prompt) in [
        (&source_a, 1, [1, 2, 3, 4]),
        (&source_b, 2, [5, 6, 7, 8]),
        (&source_replaced, 3, [9, 10, 11, 12]),
    ] {
        let batch = event_batch(1.0, vec![store_event(hash, &prompt)]);
        apply_batch(&index, source, 0, 0, &batch, Some(4)).unwrap();
        assert_eq!(hits(&index, &prompt).len(), 1);
    }
    for (active, source) in [(false, &source_replaced), (true, &source_a)] {
        let subscriber = Subscriber {
            source: EventSource {
                source: source.clone(),
                dp_rank: 0,
                pub_endpoint: String::new(),
                replay_endpoint: None,
                topic: String::new(),
                hwm: None,
            },
            sink: EventSink::Full {
                stream: FullStream {
                    index: index.clone(),
                    owner: ResidencyOwner::Worker {
                        source: source.clone(),
                        dp_rank: 0,
                        incarnation: crate::kv_index::next_owner_incarnation(),
                    },
                    control: Arc::new(StreamControl::default()),
                },
                signals: None,
                expected_block_size: Some(4),
                binding: Some(SourceLease {
                    strict_rank: Arc::new(Mutex::new(true)),
                    active: Arc::new(AtomicBool::new(active)),
                }),
            },
            incarnation: 0,
        };
        subscriber.monitor_failed("fault injection");
        assert_eq!(hits(&index, &[9, 10, 11, 12]).len(), 1);
    }
    assert!(hits(&index, &[1, 2, 3, 4]).is_empty());
    assert_eq!(hits(&index, &[5, 6, 7, 8]).len(), 1);
    assert_eq!(hits(&index, &[9, 10, 11, 12]).len(), 1);
}
