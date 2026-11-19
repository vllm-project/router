//! ZMQ SUB + DEALER replay subscriber, one task per (worker, rank).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{broadcast, mpsc};
use tracing::{debug, info, warn};

use super::event_stream::{EventStream, StreamLease};
use super::transport::{live_sequence, retry_backoff, ConnectionMonitor, IDLE_REPLAY_INTERVAL};
use crate::kv_index::IngestionSignal;

/// The signed sentinel ending a replay stream: `seq = -1`.
const END_SEQ: i64 = -1;

pub use crate::kv_index::source::endpoints::EventSource;

/// A handle to a running subscriber task; drop it (or call `shutdown`) to stop.
pub(in crate::kv_index::source) struct SubscriberHandle {
    shutdown_tx: broadcast::Sender<()>,
}

impl SubscriberHandle {
    pub(super) fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }
}

pub(super) fn spawn(
    source: EventSource,
    stream: Arc<EventStream>,
    lease: StreamLease,
    ctx: Arc<zmq::Context>,
    signals: Option<mpsc::Sender<IngestionSignal>>,
) -> SubscriberHandle {
    spawn_subscriber(
        Subscriber {
            source,
            binding: Some((stream, lease)),
            signals: signals.filter(|tx| !tx.is_closed()),
        },
        ctx,
    )
}

/// Observe continuity for advisory listeners without creating cache state.
pub(in crate::kv_index::source) fn spawn_observer(
    source: EventSource,
    ctx: Arc<zmq::Context>,
    signals: mpsc::Sender<IngestionSignal>,
) -> SubscriberHandle {
    spawn_subscriber(
        Subscriber {
            source,
            binding: None,
            signals: Some(signals),
        },
        ctx,
    )
}

struct Subscriber {
    source: EventSource,
    binding: Option<(Arc<EventStream>, StreamLease)>,
    signals: Option<mpsc::Sender<IngestionSignal>>,
}

impl Subscriber {
    fn active(&self) -> bool {
        match &self.binding {
            Some((stream, lease)) => stream.is_active(*lease),
            None => self.signals.as_ref().is_some_and(|tx| !tx.is_closed()),
        }
    }

    fn invalidate(&self, loss: bool) -> Result<(), ()> {
        if let Some((stream, lease)) = &self.binding {
            return stream.restart_replay(*lease, loss).map_err(|_| ());
        }
        if loss {
            crate::kv_index::record_signal_loss();
        }
        Ok(())
    }

    fn suspend(&self, loss: bool) -> Result<(), ()> {
        self.binding.as_ref().map_or(Ok(()), |(stream, lease)| {
            stream.suspend_for_replay(*lease, loss).map_err(|_| ())
        })
    }

    fn resume(&self, last: i64) -> Result<(), ()> {
        self.binding.as_ref().map_or(Ok(()), |(stream, lease)| {
            stream
                .finish_replay(*lease, (last + 1) as u64)
                .map_err(|_| ())
        })
    }

    fn apply(&self, seq: u64, payload: &[u8]) -> Result<(), ()> {
        let cleared = if let Some((stream, lease)) = &self.binding {
            stream
                .apply_publisher_payload(*lease, seq, payload)
                .map_err(|_| ())?
        } else {
            let batch = crate::kv_index::decode_batch(payload).map_err(|_| ())?;
            batch
                .data_parallel_rank
                .is_none_or(|rank| rank == self.source.dp_rank)
                && batch
                    .events
                    .iter()
                    .any(|event| matches!(event, crate::kv_index::KVEvent::AllBlocksCleared(_)))
        };
        if cleared {
            self.invalidated("engine cleared KV cache");
        }
        Ok(())
    }

    fn invalidated(&self, reason: &'static str) {
        emit(
            &self.signals,
            IngestionSignal::Invalidated {
                source: self.source.source.clone(),
                dp_rank: self.source.dp_rank,
                reason,
            },
        );
    }

    fn monitor_failed(&self, error: impl std::fmt::Display) {
        warn!(%error, source = %self.source.source, "KV event connection monitor failed");
        if self.invalidate(true).is_ok() {
            self.invalidated("KV event connection monitor failed");
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn replay(
        &self,
        ctx: &zmq::Context,
        mut last: i64,
        through: Option<i64>,
        mut replayed: Option<&mut HashMap<u64, u64>>,
        socket: &mut Option<zmq::Socket>,
        reconcile: bool,
        shutdown: &broadcast::Receiver<()>,
    ) -> Option<i64> {
        let dealer = match socket.take() {
            Some(socket) => socket,
            None => {
                let socket = ctx.socket(zmq::DEALER).ok()?;
                socket.set_linger(0).ok()?;
                socket
                    .connect(self.source.replay_endpoint.as_deref()?)
                    .ok()?;
                socket
            }
        };
        let start = u64::try_from(last.checked_add(1)?).ok()?;
        let through = through.map(u64::try_from).transpose().ok()?;
        let topic = self.source.topic.clone();
        if let Err(error) = dealer.send_multipart([&[][..], &start.to_be_bytes()], zmq::DONTWAIT) {
            warn!(%topic, %error, "KV replay request failed");
            return None;
        }
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        let mut next = start;
        while tokio::time::Instant::now() < deadline
            && shutdown.is_empty()
            && !shutdown.is_closed()
            && self.active()
        {
            match dealer.recv_multipart(zmq::DONTWAIT) {
                Ok(frames) => {
                    let (seq, frame_topic, payload) = super::transport::replay_frame(&frames)?;
                    if seq == END_SEQ && frame_topic.is_empty() && payload.is_empty() {
                        // Publish only a complete replay that covers the observed gap.
                        if through.is_some_and(|required| next <= required) {
                            return None;
                        }
                        *socket = Some(dealer);
                        return Some(last);
                    }
                    let seq = u64::try_from(seq).ok()?;
                    if frame_topic != topic.as_bytes() {
                        return None;
                    }
                    if seq != next {
                        warn!(%topic, seq, expected = next, "KV replay sequence gap");
                        return None;
                    }
                    if reconcile && seq == start {
                        self.suspend(false).ok()?;
                        self.invalidated("idle replay found unapplied events");
                    }
                    self.apply(seq, payload).ok()?;
                    if let Some(replayed) = replayed.as_deref_mut() {
                        super::transport::remember_replay(replayed, seq, payload);
                    }
                    last = i64::try_from(seq).ok()?;
                    next = seq.checked_add(1)?;
                }
                Err(zmq::Error::EAGAIN) => tokio::time::sleep(Duration::from_millis(1)).await,
                Err(error) => {
                    warn!(%topic, %error, "KV replay receive failed");
                    return None;
                }
            }
        }
        if tokio::time::Instant::now() >= deadline {
            warn!(%topic, "KV replay timed out");
        }
        None
    }
}

impl Drop for Subscriber {
    fn drop(&mut self) {
        if let Some((stream, lease)) = &self.binding {
            let _ = stream.retire(*lease);
        }
    }
}

fn spawn_subscriber(subscriber: Subscriber, ctx: Arc<zmq::Context>) -> SubscriberHandle {
    let (shutdown_tx, mut shutdown_rx) = broadcast::channel::<()>(1);
    let source = &subscriber.source;
    let pub_endpoint = source.pub_endpoint.clone();
    let topic = source.topic.clone();
    let source_id = source.source.clone();
    let dp_rank = source.dp_rank;
    let hwm = source.hwm;
    let can_replay = subscriber.binding.is_some();

    tokio::spawn(async move {
        let mut failures = 0u32;
        loop {
            if is_shutdown(&mut shutdown_rx) || !subscriber.active() {
                return;
            }
            let started = tokio::time::Instant::now();
            let (mut sub, mut connection) =
                match ConnectionMonitor::fresh_sub(&ctx, &pub_endpoint, &topic, hwm) {
                    Ok(sockets) => sockets,
                    Err(error) => {
                        subscriber.monitor_failed(error);
                        let delay = retry_backoff(failures);
                        failures = failures.saturating_add(1);
                        tokio::select! {
                            _ = tokio::time::sleep(delay) => {},
                            _ = shutdown_rx.recv() => return,
                        }
                        continue;
                    }
                };
            info!(
                "kv_index subscriber for {} rank{} on {}",
                source_id, dp_rank, pub_endpoint
            );

            let mut last_seq: i64 = -1;
            if subscriber.invalidate(false).is_err() {
                return;
            }
            let mut suspended = false;
            // A native publisher may already have cached blocks when Router starts.
            // Catch up by replay until a live frame confirms the SUB handoff.
            let mut bootstrap = can_replay;
            let mut allow_overlap = bootstrap;
            let mut replayed = HashMap::new();
            let mut replay_socket = None;
            let mut next_catchup = tokio::time::Instant::now();
            let mut next_idle_replay = tokio::time::Instant::now() + IDLE_REPLAY_INTERVAL;
            let mut idle_reconcile = false;
            let mut disconnected = false;
            let mut reconnect = false;
            let mut recovery_through = None;

            'live: loop {
                if is_shutdown(&mut shutdown_rx) || !subscriber.active() {
                    info!("kv_index subscriber {} shutting down", source_id);
                    return;
                }
                let (lost, latest) = match connection.drain() {
                    Ok(events) => events,
                    Err(error) => {
                        subscriber.monitor_failed(error);
                        break;
                    }
                };
                let lost = std::mem::take(&mut reconnect) || lost;
                if lost {
                    if subscriber.invalidate(true).is_err() {
                        break;
                    }
                    subscriber.invalidated("KV event publisher disconnected");
                    suspended = can_replay;
                    match ConnectionMonitor::fresh_sub(&ctx, &pub_endpoint, &topic, hwm) {
                        Ok((new_sub, new_monitor)) => {
                            sub = new_sub;
                            connection = new_monitor;
                        }
                        Err(_) => break,
                    }
                    disconnected = true;
                    recovery_through = None;
                    last_seq = -1;
                    bootstrap = false;
                    idle_reconcile = false;
                    allow_overlap = false;
                    replayed.clear();
                    replay_socket = None;
                }
                if !lost && disconnected && latest == Some(true) {
                    disconnected = false;
                    bootstrap = can_replay;
                    allow_overlap = can_replay;
                    next_catchup = tokio::time::Instant::now();
                }
                if disconnected {
                    tokio::select! {
                        result = connection.wait() => {
                            if let Err(error) = result {
                                subscriber.monitor_failed(error);
                                break;
                            }
                        },
                        _ = shutdown_rx.recv() => return,
                        _ = tokio::time::sleep(Duration::from_millis(10)) => {},
                    }
                    continue;
                }
                if can_replay
                    && !bootstrap
                    && last_seq >= 0
                    && tokio::time::Instant::now() >= next_idle_replay
                {
                    // A timer is not evidence of loss. Keep the verified view
                    // until replay actually returns an unapplied event.
                    idle_reconcile = true;
                    bootstrap = true;
                    allow_overlap = true;
                    next_catchup = tokio::time::Instant::now();
                }
                if bootstrap && tokio::time::Instant::now() >= next_catchup {
                    if last_seq == -1 && !suspended {
                        if subscriber.suspend(false).is_err() {
                            break;
                        }
                        suspended = true;
                    }
                    let result = {
                        let replay = subscriber.replay(
                            &ctx,
                            last_seq,
                            recovery_through,
                            Some(&mut replayed),
                            &mut replay_socket,
                            idle_reconcile,
                            &shutdown_rx,
                        );
                        match super::transport::while_connected(&mut connection, replay, !suspended)
                            .await
                        {
                            Ok(Some(result)) => result,
                            Ok(None) => {
                                reconnect = true;
                                continue;
                            }
                            Err(error) => {
                                subscriber.monitor_failed(error);
                                reconnect = true;
                                continue;
                            }
                        }
                    };
                    let (lost_during_replay, _) = match connection.drain() {
                        Ok(events) => events,
                        Err(error) => {
                            subscriber.monitor_failed(error);
                            break;
                        }
                    };
                    if lost_during_replay {
                        reconnect = true;
                        continue;
                    }
                    if let Some(high) = result {
                        let advanced = high != last_seq;
                        last_seq = high;
                        recovery_through = None;
                        if suspended || (idle_reconcile && advanced) {
                            if subscriber.resume(last_seq).is_err() {
                                break;
                            }
                            suspended = false;
                        }
                        if advanced {
                            debug!(source = %source_id, rank = dp_rank, seq = last_seq, "kv_index replay sequence advanced");
                        }
                        if idle_reconcile {
                            idle_reconcile = false;
                            bootstrap = false;
                            next_idle_replay = tokio::time::Instant::now() + IDLE_REPLAY_INTERVAL;
                        } else {
                            next_catchup = tokio::time::Instant::now() + Duration::from_millis(100);
                        }
                    } else {
                        let had_credit = !suspended || idle_reconcile;
                        idle_reconcile = false;
                        if subscriber.invalidate(had_credit).is_err() {
                            break;
                        }
                        last_seq = -1;
                        suspended = true;
                        replayed.clear();
                        if had_credit {
                            subscriber.invalidated("replay lost continuity");
                        }
                        next_catchup = tokio::time::Instant::now() + Duration::from_millis(500);
                    }
                }
                // Queued live stores cannot prove missed evictions.
                if bootstrap && suspended {
                    tokio::select! {
                        _ = shutdown_rx.recv() => return,
                        _ = tokio::time::sleep(Duration::from_millis(10)) => {},
                    }
                    continue;
                }
                let (through, overlap) = 'live_batch: {
                    match sub.socket().recv_multipart(zmq::DONTWAIT) {
                        Ok(frames) => {
                            let seq = match live_sequence(&frames, topic.as_bytes()) {
                                Ok(Some(seq)) if seq >= 0 => seq,
                                Ok(None) => {
                                    tokio::task::yield_now().await;
                                    continue 'live;
                                }
                                Ok(Some(_)) | Err(()) => {
                                    if subscriber.invalidate(true).is_err() {
                                        break 'live;
                                    }
                                    subscriber.invalidated("invalid live sequence");
                                    suspended = true;
                                    tokio::task::yield_now().await;
                                    break 'live_batch (None, true);
                                }
                            };
                            next_idle_replay = tokio::time::Instant::now() + IDLE_REPLAY_INTERVAL;
                            if allow_overlap
                                && seq <= last_seq
                                && super::transport::replay_overlap(
                                    &replayed, seq as u64, &frames[2],
                                )
                            {
                                bootstrap = false;
                                replay_socket = None;
                                continue 'live;
                            }
                            // Sequence zero outside the verified replay overlap
                            // can belong to a new publisher lifetime.
                            if last_seq != -1 && (seq < last_seq || seq == 0) {
                                let incarnation = crate::kv_index::next_owner_incarnation();
                                warn!(
                                    "kv_index {}: restart seq {} after {} → incarnation {}",
                                    source_id, seq, last_seq, incarnation
                                );
                                if subscriber.invalidate(true).is_err() {
                                    break 'live;
                                }
                                emit(
                                    &subscriber.signals,
                                    IngestionSignal::IncarnationReset {
                                        source: source_id.clone(),
                                        incarnation,
                                    },
                                );
                                suspended = true;
                                break 'live_batch (Some(seq), true);
                            }
                            if seq <= last_seq {
                                if subscriber.invalidate(true).is_err() {
                                    break 'live;
                                }
                                subscriber.invalidated("duplicate live sequence");
                                suspended = true;
                                break 'live_batch (Some(seq), allow_overlap);
                            }
                            if (can_replay || last_seq >= 0) && seq > last_seq + 1 {
                                if subscriber.invalidate(true).is_err() {
                                    break 'live;
                                }
                                emit(
                                    &subscriber.signals,
                                    IngestionSignal::Gap {
                                        source: source_id.clone(),
                                        from_seq: last_seq + 1,
                                        to_seq: seq - 1,
                                    },
                                );
                                suspended = true;
                                break 'live_batch (Some(seq), true);
                            }
                            if subscriber.apply(seq as u64, &frames[2]).is_err() {
                                if subscriber.invalidate(true).is_err() {
                                    break 'live;
                                }
                                subscriber.invalidated("invalid live KV event batch");
                                suspended = true;
                                tokio::task::yield_now().await;
                                break 'live_batch (Some(seq), true);
                            }
                            if suspended {
                                if subscriber.resume(seq).is_err() {
                                    break 'live;
                                }
                                suspended = false;
                            }
                            last_seq = seq;
                            bootstrap = false;
                            replay_socket = None;
                            allow_overlap = false;
                            replayed.clear();
                            debug!(source = %source_id, rank = dp_rank, seq, "kv_index live sequence advanced");
                            // A sustained publisher starves EAGAIN; yield so this task
                            // can't monopolize the runtime worker.
                            tokio::task::yield_now().await;
                        }
                        Err(zmq::Error::EAGAIN) => {
                            tokio::select! {
                                result = connection.wait() => {
                                    if let Err(error) = result {
                                        subscriber.monitor_failed(error);
                                        break 'live;
                                    }
                                },
                                result = sub.wait() => {
                                    if let Err(error) = result {
                                        warn!("kv_index sub {} readiness: {}", source_id, error);
                                        tokio::time::sleep(Duration::from_millis(100)).await;
                                    }
                                }
                                _ = shutdown_rx.recv() => return,
                                // Preserve timely lease checks even without incoming messages.
                                _ = tokio::time::sleep(Duration::from_millis(10)) => {}
                            }
                        }
                        Err(e) => {
                            warn!("kv_index sub {} recv: {}", source_id, e);
                            tokio::time::sleep(Duration::from_millis(100)).await;
                        }
                    }
                    continue 'live;
                };
                if !can_replay {
                    // Loss fenced the advisory state. Start a new observation at
                    // this watermark without reconstructing past cache residency.
                    last_seq = through.unwrap_or(-1);
                    suspended = false;
                    continue;
                }
                last_seq = -1;
                recovery_through = through;
                bootstrap = true;
                allow_overlap = overlap;
                replayed.clear();
                replay_socket = None;
            }
            if is_shutdown(&mut shutdown_rx) || !subscriber.active() {
                return;
            }
            subscriber.monitor_failed("subscriber stopped");
            if started.elapsed() >= Duration::from_secs(5) {
                failures = 0;
            }
            let delay = retry_backoff(failures);
            failures = failures.saturating_add(1);
            tokio::select! {
                _ = tokio::time::sleep(delay) => {},
                _ = shutdown_rx.recv() => return,
            }
        }
    });

    SubscriberHandle { shutdown_tx }
}

/// Shut down on an explicit signal OR the handle (sender) being dropped.
fn is_shutdown(rx: &mut broadcast::Receiver<()>) -> bool {
    matches!(
        rx.try_recv(),
        Ok(()) | Err(broadcast::error::TryRecvError::Closed)
    )
}

/// Advisory signal send: `try_send` so a slow arbiter never stalls the hot loop.
/// Source invalidation is synchronous in the indexer, independent of signals.
pub(in crate::kv_index) fn emit(
    tx: &Option<mpsc::Sender<IngestionSignal>>,
    signal: IngestionSignal,
) {
    if let Some(tx) = tx {
        if let Err(mpsc::error::TrySendError::Full(_)) = tx.try_send(signal) {
            crate::kv_index::record_signal_loss();
            debug!("kv_index signal queue full");
        }
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[path = "subscriber_tests.rs"]
pub(super) mod tests;
