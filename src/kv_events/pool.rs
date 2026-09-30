use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Arc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::kv_index::KVBlockIndex;

use super::{
    decoder::{decode_batch, MAX_PAYLOAD_BYTES},
    resolve_endpoints,
};

/// Owns the subscriber threads for the router lifetime. Dropping this value
/// cancels subscriptions, purges ownership and joins every owned thread.
pub struct KVEventPool {
    index: Arc<KVBlockIndex>,
    workers: Vec<String>,
    stopping: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl std::fmt::Debug for KVEventPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KVEventPool")
            .field("workers", &self.workers)
            .finish_non_exhaustive()
    }
}

impl KVEventPool {
    pub fn start(
        workers: Vec<(String, String)>,
        topic: String,
        block_size: usize,
        index: Arc<KVBlockIndex>,
    ) -> Result<Self, String> {
        if block_size == 0 {
            return Err("KV event block size must be greater than zero".into());
        }
        let urls: Vec<_> = workers.iter().map(|(url, _)| url.clone()).collect();
        // Validate direct callers as well as CLI callers. All mappings are
        // explicit here, so the fallback port is unused.
        let workers = resolve_endpoints(&urls, &workers, 5557)?;
        let mut pool = Self {
            index,
            workers: Vec::with_capacity(workers.len()),
            stopping: Arc::new(AtomicBool::new(false)),
            threads: Vec::with_capacity(workers.len()),
        };
        for (ordinal, (worker, endpoint)) in workers.into_iter().enumerate() {
            let generation = pool.index.begin_worker(&worker);
            pool.workers.push(worker.clone());
            let index = Arc::clone(&pool.index);
            let stopping = Arc::clone(&pool.stopping);
            let topic = topic.clone();
            let (ready_tx, ready_rx) = mpsc::sync_channel(1);
            let handle = thread::Builder::new()
                .name(format!("kv-events-{ordinal}"))
                .spawn(move || {
                    subscriber_loop(
                        worker, endpoint, topic, block_size, index, stopping, generation, ready_tx,
                    );
                })
                .map_err(|error| format!("cannot start KV subscriber: {error}"))?;
            pool.threads.push(handle);
            ready_rx
                .recv_timeout(Duration::from_secs(5))
                .map_err(|error| format!("KV subscriber initialization failed: {error}"))??;
        }
        Ok(pool)
    }

    /// Health failure/removal must invalidate synchronously before routing can
    /// observe recovery. A successful recovery calls index.begin_worker().
    pub fn invalidate_worker(&self, worker: &str) {
        self.index.retire_worker(worker);
    }
}

impl Drop for KVEventPool {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        for worker in &self.workers {
            self.index.retire_worker(worker);
        }
        for thread in self.threads.drain(..) {
            thread.thread().unpark();
            if thread.join().is_err() {
                warn!("KV event subscriber panicked during shutdown");
            }
        }
    }
}

struct Subscription {
    socket: zmq::Socket,
    monitor: zmq::Socket,
}

impl Subscription {
    fn open(context: &zmq::Context, endpoint: &str, topic: &str) -> Result<Self, zmq::Error> {
        static NEXT_MONITOR: AtomicU64 = AtomicU64::new(0);
        // ZMQ closes endpoints asynchronously; a fresh address prevents a
        // reconnect racing the prior monitor's inproc endpoint teardown.
        let monitor_endpoint = format!(
            "inproc://kv-events-monitor-{}",
            NEXT_MONITOR.fetch_add(1, Ordering::Relaxed)
        );
        let socket = context.socket(zmq::SUB)?;
        socket.set_linger(0)?;
        // Endpoint validation accepts IPv6; libzmq otherwise defaults to IPv4 only.
        socket.set_ipv6(true)?;
        socket.set_rcvhwm(1024)?;
        socket.set_maxmsgsize(MAX_PAYLOAD_BYTES as i64)?;
        socket.set_subscribe(topic.as_bytes())?;
        socket.monitor(&monitor_endpoint, zmq::SocketEvent::DISCONNECTED as i32)?;
        let monitor = context.socket(zmq::PAIR)?;
        monitor.set_linger(0)?;
        monitor.connect(&monitor_endpoint)?;
        socket.connect(endpoint)?;
        Ok(Self { socket, monitor })
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SequenceAction {
    Accept,
    Duplicate,
    Reconnect,
}

fn sequence_action(previous: Option<(u64, [u8; 32])>, current: (u64, [u8; 32])) -> SequenceAction {
    match previous {
        None => SequenceAction::Accept,
        Some(previous) if current == previous => SequenceAction::Duplicate,
        Some(previous) if previous.0.checked_add(1) == Some(current.0) => SequenceAction::Accept,
        Some(_) => SequenceAction::Reconnect,
    }
}

#[allow(clippy::too_many_arguments)]
fn subscriber_loop(
    worker: String,
    endpoint: String,
    topic: String,
    block_size: usize,
    index: Arc<KVBlockIndex>,
    stopping: Arc<AtomicBool>,
    mut generation: u64,
    ready: mpsc::SyncSender<Result<(), String>>,
) {
    // Unexpected thread exits/panics must also invalidate the last ownership.
    struct RetireOnExit(Arc<KVBlockIndex>, String);
    impl Drop for RetireOnExit {
        fn drop(&mut self) {
            self.0.retire_worker(&self.1);
        }
    }
    let _retire_on_exit = RetireOnExit(Arc::clone(&index), worker.clone());
    let context = zmq::Context::new();
    let mut subscription = match Subscription::open(&context, &endpoint, &topic) {
        Ok(subscription) => {
            let _ = ready.send(Ok(()));
            Some(subscription)
        }
        Err(error) => {
            index.retire_worker(&worker);
            let _ = ready.send(Err(format!("KV event endpoint {endpoint}: {error}")));
            return;
        }
    };
    let mut last_sequence = None;
    while !stopping.load(Ordering::Acquire) {
        let Some(current_generation) = index.current_generation(&worker) else {
            subscription = None;
            last_sequence = None;
            thread::park_timeout(Duration::from_millis(100));
            continue;
        };
        if current_generation != generation {
            // Health recovery or explicit lifecycle fencing must recreate the
            // socket; queued messages belong to the retired generation.
            subscription = None;
            last_sequence = None;
            generation = current_generation;
        }
        if subscription.is_none() {
            match Subscription::open(&context, &endpoint, &topic) {
                Ok(fresh) => subscription = Some(fresh),
                Err(error) => {
                    warn!(worker, %error, "KV subscriber reconnect failed");
                    thread::park_timeout(Duration::from_millis(100));
                    continue;
                }
            }
        }
        let active = subscription.as_ref().unwrap();
        let mut poll_items = [
            active.monitor.as_poll_item(zmq::POLLIN),
            active.socket.as_poll_item(zmq::POLLIN),
        ];
        if let Err(error) = zmq::poll(&mut poll_items, 100) {
            if error == zmq::Error::EINTR {
                continue;
            }
            warn!(worker, %error, "KV subscriber poll failed; discarding ownership");
            reconnect(
                &index,
                &worker,
                &mut generation,
                &mut subscription,
                &mut last_sequence,
            );
            continue;
        }
        // Process disconnect before queued data from the disconnected socket.
        if poll_items[0].is_readable() {
            let _ = active.monitor.recv_multipart(zmq::DONTWAIT);
            warn!(worker, "KV publisher disconnected; discarding ownership");
            reconnect(
                &index,
                &worker,
                &mut generation,
                &mut subscription,
                &mut last_sequence,
            );
            continue;
        }
        if !poll_items[1].is_readable() {
            continue;
        }
        let frames = match active.socket.recv_multipart(zmq::DONTWAIT) {
            Ok(frames) => frames,
            Err(zmq::Error::EAGAIN) => continue,
            Err(error) => {
                warn!(worker, %error, "KV subscriber receive failed; discarding ownership");
                reconnect(
                    &index,
                    &worker,
                    &mut generation,
                    &mut subscription,
                    &mut last_sequence,
                );
                continue;
            }
        };
        if frames.len() != 3 || frames[0] != topic.as_bytes() || frames[1].len() != 8 {
            // Exact topics avoid interleaving sequence spaces from prefix
            // subscriptions. The configured topic is a full publisher topic.
            warn!(
                worker,
                "invalid KV multipart envelope; subscriber quarantined"
            );
            index.retire_worker(&worker);
            return;
        }
        let sequence = u64::from_be_bytes(frames[1].as_slice().try_into().unwrap());
        let payload_hash: [u8; 32] = Sha256::digest(&frames[2]).into();
        let current_sequence = (sequence, payload_hash);
        match sequence_action(last_sequence, current_sequence) {
            SequenceAction::Duplicate => continue,
            SequenceAction::Reconnect => {
                warn!(
                    worker,
                    previous = ?last_sequence.map(|previous| previous.0),
                    sequence,
                    "KV sequence discontinuity; discarding ownership and event"
                );
                reconnect(
                    &index,
                    &worker,
                    &mut generation,
                    &mut subscription,
                    &mut last_sequence,
                );
                continue;
            }
            SequenceAction::Accept => {}
        }
        match decode_batch(&frames[2], block_size) {
            Ok(events) => {
                // This single atomic operation also rejects a health callback
                // retiring the generation while receive/decode was running.
                if index.apply_batch(&worker, generation, &events) {
                    last_sequence = Some(current_sequence);
                }
            }
            Err(error) => {
                warn!(worker, %error, "unsupported KV event batch; subscriber quarantined until router restart");
                index.retire_worker(&worker);
                return;
            }
        }
    }
    info!(worker, "KV event subscriber stopped");
}

fn reconnect(
    index: &KVBlockIndex,
    worker: &str,
    generation: &mut u64,
    subscription: &mut Option<Subscription>,
    last_sequence: &mut Option<(u64, [u8; 32])>,
) {
    if let Some(next) = index.roll_worker(worker, *generation) {
        *generation = next;
    }
    *subscription = None;
    *last_sequence = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subscription_enables_dual_stack_addresses() {
        let context = zmq::Context::new();
        let publisher = context.socket(zmq::PUB).unwrap();
        publisher.set_linger(0).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
        let subscription = Subscription::open(&context, &endpoint, "kv").unwrap();
        assert!(subscription.socket.is_ipv6().unwrap());
    }

    #[test]
    fn sequence_gaps_rollbacks_and_overflow_require_new_subscription() {
        let payload = [1; 32];
        assert_eq!(
            sequence_action(None, (400, payload)),
            SequenceAction::Accept
        );
        assert_eq!(
            sequence_action(Some((400, payload)), (401, payload)),
            SequenceAction::Accept
        );
        assert_eq!(
            sequence_action(Some((400, payload)), (400, payload)),
            SequenceAction::Duplicate
        );
        // A restarted publisher can reuse a sequence number with a different
        // event; this is not a duplicate and cannot preserve old ownership.
        assert_eq!(
            sequence_action(Some((400, payload)), (400, [2; 32])),
            SequenceAction::Reconnect
        );
        assert_eq!(
            sequence_action(Some((400, payload)), (402, payload)),
            SequenceAction::Reconnect
        );
        assert_eq!(
            sequence_action(Some((400, payload)), (0, payload)),
            SequenceAction::Reconnect
        );
        assert_eq!(
            sequence_action(Some((u64::MAX, payload)), (0, payload)),
            SequenceAction::Reconnect
        );
    }

    #[test]
    fn reconnect_cannot_override_health_retirement() {
        let index = KVBlockIndex::new(8);
        let mut generation = index.begin_worker("w0");
        index.store("w0", generation, &[[1; 32]]);
        index.retire_worker("w0");
        let mut subscription = None;
        let mut last_sequence = Some((10, [1; 32]));
        reconnect(
            &index,
            "w0",
            &mut generation,
            &mut subscription,
            &mut last_sequence,
        );
        assert_eq!(index.current_generation("w0"), None);
        assert_eq!(index.ownership_count(), 0);
        assert!(!index.store("w0", generation, &[[1; 32]]));
        assert_eq!(last_sequence, None);
    }

    #[test]
    fn pool_drop_joins_and_retires_owned_worker() {
        let context = zmq::Context::new();
        let publisher = context.socket(zmq::PUB).unwrap();
        publisher.set_linger(0).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
        let index = Arc::new(KVBlockIndex::new(8));
        let pool = KVEventPool::start(
            vec![("http://127.0.0.1:8000".into(), endpoint)],
            "kv".into(),
            16,
            Arc::clone(&index),
        )
        .unwrap();
        let generation = index.current_generation("http://127.0.0.1:8000").unwrap();
        index.store("http://127.0.0.1:8000", generation, &[[1; 32]]);
        drop(pool);
        assert_eq!(index.ownership_count(), 0);
        assert_eq!(index.current_generation("http://127.0.0.1:8000"), None);
    }

    /// Exercise the production subscriber over an owned localhost TCP socket.
    /// XPUB reports subscription readiness, including after reconnect, so no
    /// slow-joiner sleeps or assumptions about scheduler timing are necessary.
    #[test]
    fn real_zmq_transport_applies_events_and_fences_sequence_gaps() {
        use rmpv::Value;
        use std::time::Instant;

        const WORKER: &str = "http://127.0.0.1:8000";
        const TOPIC: &[u8] = b"kv-transport-regression";

        fn event(fields: Vec<(&str, Value)>) -> Value {
            Value::Map(
                fields
                    .into_iter()
                    .map(|(key, value)| (key.into(), value))
                    .collect(),
            )
        }

        fn stored(blocks: &[u8]) -> Value {
            event(vec![
                ("type", "BlockStored".into()),
                (
                    "block_hashes",
                    Value::Array(
                        blocks
                            .iter()
                            .map(|&block| Value::Binary(vec![block; 32]))
                            .collect(),
                    ),
                ),
                ("parent_block_hash", Value::Nil),
                (
                    "token_ids",
                    Value::Array(
                        (0..blocks.len() * 2)
                            .map(|id| Value::from(id as u64))
                            .collect(),
                    ),
                ),
                ("block_size", 2.into()),
                ("lora_id", Value::Nil),
                ("medium", "GPU".into()),
                ("lora_name", Value::Nil),
                ("group_idx", 0.into()),
                ("kv_cache_spec_kind", "full_attention".into()),
            ])
        }

        fn publish(publisher: &zmq::Socket, sequence: u64, events: Vec<Value>) {
            let mut payload = Vec::new();
            rmpv::encode::write_value(
                &mut payload,
                &Value::Array(vec![1.0.into(), Value::Array(events), 0.into()]),
            )
            .unwrap();
            publisher
                .send_multipart([TOPIC, &sequence.to_be_bytes(), payload.as_slice()], 0)
                .unwrap();
        }

        fn wait_for(
            publisher: &zmq::Socket,
            subscriptions: &mut usize,
            condition: impl Fn(usize) -> bool,
            label: &str,
        ) {
            let deadline = Instant::now() + Duration::from_secs(5);
            while !condition(*subscriptions) {
                let remaining = deadline.saturating_duration_since(Instant::now());
                assert!(!remaining.is_zero(), "timed out waiting for {label}");
                // Poll also yields to the subscriber while awaiting an index
                // change. Drain control frames so reconnect handshakes remain
                // counted even if they arrive before the state predicate.
                let timeout_ms = remaining.as_millis().clamp(1, 10) as i64;
                if publisher.poll(zmq::POLLIN, timeout_ms).unwrap() > 0 {
                    let frame = publisher.recv_bytes(zmq::DONTWAIT).unwrap();
                    assert_eq!(frame.get(1..), Some(TOPIC), "unexpected XPUB topic");
                    match frame.first() {
                        Some(1) => *subscriptions += 1,
                        Some(0) => {}
                        _ => panic!("invalid XPUB subscription control frame"),
                    }
                }
            }
        }

        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        publisher.set_linger(0).unwrap();
        publisher.set_xpub_verbose(true).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
        let index = Arc::new(KVBlockIndex::new(16));
        let pool = KVEventPool::start(
            vec![(WORKER.into(), endpoint)],
            String::from_utf8(TOPIC.to_vec()).unwrap(),
            2,
            Arc::clone(&index),
        )
        .unwrap();
        let old_generation = index.current_generation(WORKER).unwrap();
        let mut subscriptions = 0;
        wait_for(
            &publisher,
            &mut subscriptions,
            |count| count == 1,
            "initial subscription",
        );

        publish(&publisher, 10, vec![stored(&[1, 2])]);
        wait_for(
            &publisher,
            &mut subscriptions,
            |_| index.prefix_score(WORKER, &[[1; 32], [2; 32]]) == 2,
            "stored blocks",
        );
        publish(
            &publisher,
            11,
            vec![event(vec![
                ("type", "BlockRemoved".into()),
                (
                    "block_hashes",
                    Value::Array(vec![Value::Binary(vec![2; 32])]),
                ),
                ("medium", "GPU".into()),
            ])],
        );
        wait_for(
            &publisher,
            &mut subscriptions,
            |_| index.ownership_count() == 1,
            "removed block",
        );
        assert_eq!(index.prefix_score(WORKER, &[[1; 32], [2; 32]]), 1);

        publish(&publisher, 12, vec![stored(&[2])]);
        wait_for(
            &publisher,
            &mut subscriptions,
            |_| index.ownership_count() == 2,
            "restored block",
        );
        publish(
            &publisher,
            13,
            vec![event(vec![("type", "AllBlocksCleared".into())])],
        );
        wait_for(
            &publisher,
            &mut subscriptions,
            |_| index.ownership_count() == 0,
            "cache clear",
        );
        publish(&publisher, 14, vec![stored(&[1])]);
        wait_for(
            &publisher,
            &mut subscriptions,
            |_| index.ownership_count() == 1,
            "pre-gap ownership",
        );

        let prior_subscriptions = subscriptions;
        publish(&publisher, 16, vec![stored(&[3])]);
        wait_for(
            &publisher,
            &mut subscriptions,
            |count| {
                count > prior_subscriptions
                    && index
                        .current_generation(WORKER)
                        .is_some_and(|generation| generation != old_generation)
                    && index.ownership_count() == 0
            },
            "gap purge and fresh subscription handshake",
        );
        assert!(!index.store(WORKER, old_generation, &[[4; 32]]));
        assert_eq!(
            index.prefix_score(WORKER, &[[3; 32]]),
            0,
            "the gap event must be discarded"
        );

        publish(&publisher, 17, vec![stored(&[3])]);
        wait_for(
            &publisher,
            &mut subscriptions,
            |_| index.prefix_score(WORKER, &[[3; 32]]) == 1,
            "post-gap ownership",
        );

        // No partial application: a malformed removal following a store
        // quarantines the subscriber and purges all earlier ownership.
        publish(
            &publisher,
            18,
            vec![stored(&[5]), event(vec![("type", "BlockRemoved".into())])],
        );
        wait_for(
            &publisher,
            &mut subscriptions,
            |_| index.current_generation(WORKER).is_none() && index.ownership_count() == 0,
            "malformed-batch quarantine",
        );
        drop(pool);
        assert_eq!(index.current_generation(WORKER), None);
        assert_eq!(index.ownership_count(), 0);
    }
}
