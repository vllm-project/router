//! ZMQ SUB + DEALER replay subscriber, one task per (worker, rank).

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{broadcast, mpsc};
use tracing::{debug, info, warn};

use super::event_stream::{batch_contains_clear, EventStream, StreamLease};
use super::live_event_buffer::LiveEventBuffer;
use super::transport::{
    live_sequence, retry_backoff, ConnectionMonitor, ReplayPoll, IDLE_REPLAY_INTERVAL,
};
use crate::kv_index::types::SourceRecoveryReason as Reason;
use crate::kv_index::{decode_batch, IngestionSignal, KVEventBatch};
use crate::metrics::RouterMetrics;

/// The signed sentinel ending a replay stream: `seq = -1`.
const END_SEQ: i64 = -1;
const MAX_REPLAY_FAILURES: u32 = 3;
const REPLAY_IDLE_TIMEOUT: Duration = Duration::from_secs(5);

struct ReplayProgress {
    applied_seq: i64,
}

enum ReplayError {
    Incomplete(ReplayProgress),
    Rejected,
}

#[derive(Default)]
struct ReplayWatermarks {
    required_live_seq: Option<i64>,
    highest_seen_replay_seq: Option<i64>,
    clear_seq: Option<i64>,
}

impl ReplayWatermarks {
    fn covers_required_sequences(&self, next_expected_seq: i64) -> bool {
        self.required_live_seq
            .max(self.highest_seen_replay_seq)
            .is_none_or(|required_seq| next_expected_seq > required_seq)
    }
}

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
    spawn_with_idle_timeout(source, stream, lease, ctx, signals, REPLAY_IDLE_TIMEOUT)
}

fn spawn_with_idle_timeout(
    source: EventSource,
    stream: Arc<EventStream>,
    lease: StreamLease,
    ctx: Arc<zmq::Context>,
    signals: Option<mpsc::Sender<IngestionSignal>>,
    replay_idle_timeout: Duration,
) -> SubscriberHandle {
    spawn_subscriber(
        Subscriber {
            source,
            replay_idle_timeout,
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
            replay_idle_timeout: REPLAY_IDLE_TIMEOUT,
            binding: None,
            signals: Some(signals),
        },
        ctx,
    )
}

struct Subscriber {
    source: EventSource,
    // Production uses REPLAY_IDLE_TIMEOUT. Tests can inject a shorter timeout.
    replay_idle_timeout: Duration,
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

    fn invalidate(&self, loss: bool, reason: Reason) -> Result<(), ()> {
        if let Some((stream, lease)) = &self.binding {
            return stream.restart_replay(*lease, loss, reason).map_err(|_| ());
        }
        if loss {
            crate::kv_index::record_signal_loss();
        }
        Ok(())
    }

    fn suspend(&self, reason: Reason) -> Result<(), ()> {
        self.binding.as_ref().map_or(Ok(()), |(stream, lease)| {
            stream.suspend_for_replay(*lease, reason).map_err(|_| ())
        })
    }

    fn resume(&self, applied_seq: i64, reason: Reason) -> Result<(), ()> {
        self.binding.as_ref().map_or(Ok(()), |(stream, lease)| {
            let next = applied_seq.checked_add(1).ok_or(())?;
            stream.resume(*lease, next as u64, reason).map_err(|_| ())
        })
    }

    fn apply_next_batch(&self, seq: i64, payload: &[u8]) -> Result<(), ()> {
        let batch = decode_batch(payload).ok();
        let cleared = if let Some((stream, lease)) = &self.binding {
            stream
                .apply_next_batch(*lease, seq as u64, batch.as_ref())
                .map_err(|_| ())?
        } else {
            self.observer_detected_clear(batch.as_ref().ok_or(())?)
        };
        self.notify_if_cleared(cleared);
        Ok(())
    }

    fn apply_fresh_live_batch(&self, seq: i64, payload: &[u8]) -> Result<(), ()> {
        let batch = decode_batch(payload).map_err(|_| ())?;
        let cleared = if let Some((stream, lease)) = &self.binding {
            stream
                .reset_and_try_publish_from_live_batch(*lease, seq as u64, &batch)
                .map_err(|_| ())?
        } else {
            self.observer_detected_clear(&batch)
        };
        self.notify_if_cleared(cleared);
        Ok(())
    }

    fn observer_detected_clear(&self, batch: &KVEventBatch) -> bool {
        batch
            .data_parallel_rank
            .is_none_or(|rank| rank == self.source.dp_rank)
            && batch_contains_clear(batch)
    }

    fn notify_if_cleared(&self, cleared: bool) {
        if cleared {
            self.invalidated("engine cleared KV cache");
        }
    }

    /// A retained suffix can start after the requested cursor. Discard earlier
    /// claims only if the suffix supplies a valid clear for this worker rank.
    fn apply_replay_clear_if_present(&self, seq: i64, payload: &[u8]) -> Result<bool, ReplayError> {
        let batch = decode_batch(payload).map_err(|_| ReplayError::Rejected)?;
        // Check even skipped batches while looking for a clear.
        if batch.data_parallel_rank != Some(self.source.dp_rank) {
            return Err(ReplayError::Rejected);
        }
        if !batch_contains_clear(&batch) {
            return Ok(false);
        }
        let (stream, lease) = self.binding.as_ref().ok_or(ReplayError::Rejected)?;
        stream
            .reset_hidden_index_from_replay_clear(*lease, seq as u64, &batch)
            .map_err(|_| ReplayError::Rejected)?;
        self.invalidated("engine cleared KV cache");
        Ok(true)
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
        if self.invalidate(true, Reason::ValidationFailed).is_ok() {
            self.invalidated("KV event connection monitor failed");
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn replay(
        &self,
        ctx: &zmq::Context,
        mut applied_seq: i64,
        watermarks: &mut ReplayWatermarks,
        live_events: &LiveEventBuffer,
        socket: &mut Option<zmq::Socket>,
        idle_reconcile: bool,
        shutdown: &broadcast::Receiver<()>,
    ) -> Result<ReplayProgress, ReplayError> {
        let dealer = match socket.take() {
            Some(socket) => socket,
            None => {
                let socket = ctx
                    .socket(zmq::DEALER)
                    .map_err(|_| ReplayError::Incomplete(ReplayProgress { applied_seq }))?;
                socket
                    .set_maxmsgsize(super::MAX_KV_EVENT_BYTES as i64)
                    .map_err(|_| ReplayError::Incomplete(ReplayProgress { applied_seq }))?;
                socket
                    .set_linger(0)
                    .map_err(|_| ReplayError::Incomplete(ReplayProgress { applied_seq }))?;
                socket
                    .connect(
                        self.source
                            .replay_endpoint
                            .as_deref()
                            .ok_or(ReplayError::Rejected)?,
                    )
                    .map_err(|_| ReplayError::Incomplete(ReplayProgress { applied_seq }))?;
                socket
            }
        };
        let requested_start_seq = applied_seq
            .checked_add(1)
            .filter(|&seq| seq >= 0)
            .ok_or(ReplayError::Rejected)?;
        let topic = self.source.topic.clone();
        if let Err(error) =
            dealer.send_multipart([&[][..], &requested_start_seq.to_be_bytes()], zmq::DONTWAIT)
        {
            warn!(%topic, %error, "KV replay request failed");
            return Err(ReplayError::Incomplete(ReplayProgress { applied_seq }));
        }
        let mut deadline = tokio::time::Instant::now() + self.replay_idle_timeout;
        let mut next_expected_seq = requested_start_seq;
        let mut waiting_for_replay_clear = false;
        while tokio::time::Instant::now() < deadline
            && shutdown.is_empty()
            && !shutdown.is_closed()
            && self.active()
        {
            match dealer.recv_multipart(zmq::DONTWAIT) {
                Ok(frames) => {
                    let (seq, frame_topic, payload) = super::transport::replay_frame(&frames)
                        .ok_or_else(|| {
                            warn!(%topic, "Malformed KV replay frame");
                            ReplayError::Rejected
                        })?;
                    if seq == END_SEQ && frame_topic.is_empty() && payload.is_empty() {
                        if waiting_for_replay_clear {
                            warn!(%topic, "Truncated KV replay ended without a clear");
                            return Err(ReplayError::Rejected);
                        }
                        // The first live batch after a gap can finish a contiguous
                        // replay when the publisher has not buffered that batch yet.
                        if watermarks.required_live_seq == Some(next_expected_seq)
                            && live_events
                                .apply_gap_trigger(next_expected_seq, |payload| {
                                    self.apply_next_batch(next_expected_seq, payload)
                                })
                                .map_err(|_| ReplayError::Rejected)?
                        {
                            applied_seq = next_expected_seq;
                            next_expected_seq += 1;
                        }
                        if !watermarks.covers_required_sequences(next_expected_seq) {
                            return Err(ReplayError::Incomplete(ReplayProgress { applied_seq }));
                        }
                        *socket = Some(dealer);
                        return Ok(ReplayProgress { applied_seq });
                    }
                    if !(0..i64::MAX).contains(&seq) || frame_topic != topic.as_bytes() {
                        warn!(%topic, seq, "KV replay frame has an invalid sequence or topic");
                        return Err(ReplayError::Rejected);
                    }
                    // A rejected reply can still reveal a sequence the next replay must cover.
                    watermarks.highest_seen_replay_seq =
                        watermarks.highest_seen_replay_seq.max(Some(seq));
                    if seq != next_expected_seq {
                        if next_expected_seq != requested_start_seq || seq < requested_start_seq {
                            warn!(%topic, seq, expected = next_expected_seq, "KV replay sequence gap");
                            return Err(ReplayError::Rejected);
                        }
                        waiting_for_replay_clear = true;
                    }
                    if idle_reconcile && next_expected_seq == requested_start_seq {
                        self.suspend(Reason::IdleCatchup)
                            .map_err(|_| ReplayError::Rejected)?;
                        self.invalidated("idle replay found unapplied events");
                    }
                    record_replay_payload(live_events, &topic, seq, payload)?;
                    if waiting_for_replay_clear {
                        if self.apply_replay_clear_if_present(seq, payload)? {
                            watermarks.clear_seq = Some(seq);
                            waiting_for_replay_clear = false;
                        }
                    } else {
                        self.apply_next_batch(seq, payload)
                            .map_err(|_| ReplayError::Rejected)?;
                    }
                    if !waiting_for_replay_clear {
                        applied_seq = seq;
                    }
                    next_expected_seq = seq + 1;
                    deadline = tokio::time::Instant::now() + self.replay_idle_timeout;
                    tokio::task::yield_now().await;
                }
                Err(zmq::Error::EAGAIN) => tokio::time::sleep(Duration::from_millis(1)).await,
                Err(error) => {
                    warn!(%topic, %error, "KV replay receive failed");
                    return Err(ReplayError::Incomplete(ReplayProgress { applied_seq }));
                }
            }
        }
        if tokio::time::Instant::now() >= deadline {
            warn!(%topic, "KV replay made no progress before the idle timeout");
        }
        Err(ReplayError::Incomplete(ReplayProgress { applied_seq }))
    }
}

fn record_replay_payload(
    live_events: &LiveEventBuffer,
    topic: &str,
    seq: i64,
    payload: &[u8],
) -> Result<(), ReplayError> {
    live_events.record_replay_payload(seq, payload).map_err(|_| {
        warn!(%topic, seq, "KV replay payload differs from an earlier payload for this sequence");
        ReplayError::Rejected
    })
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

            let mut applied_seq: i64 = -1;
            if subscriber.invalidate(false, Reason::Registered).is_err() {
                return;
            }
            let mut suspended = false;
            // A native publisher may already have cached blocks when Router starts.
            // Catch up by replay until a live frame confirms the SUB handoff.
            let mut bootstrap = can_replay;
            let mut check_live_duplicates = bootstrap;
            let mut live_events = LiveEventBuffer::default();
            let mut replay_socket = None;
            let mut next_catchup = tokio::time::Instant::now();
            let mut next_idle_replay = tokio::time::Instant::now() + IDLE_REPLAY_INTERVAL;
            let mut idle_reconcile = false;
            let mut disconnected = false;
            let mut reconnect = false;
            let mut watermarks = ReplayWatermarks::default();
            let mut replay_failures = 0;
            let mut monitor_error = None;

            'live: loop {
                if is_shutdown(&mut shutdown_rx) || !subscriber.active() {
                    info!("kv_index subscriber {} shutting down", source_id);
                    return;
                }
                let (lost, latest) = match connection.drain() {
                    Ok(events) => events,
                    Err(error) => {
                        monitor_error = Some(error.to_string());
                        break;
                    }
                };
                let lost = std::mem::take(&mut reconnect) || lost;
                if lost {
                    if subscriber
                        .invalidate(true, Reason::IncarnationReset)
                        .is_err()
                    {
                        break;
                    }
                    subscriber.invalidated("KV event publisher disconnected");
                    suspended = can_replay;
                    match ConnectionMonitor::fresh_sub(&ctx, &pub_endpoint, &topic, hwm) {
                        Ok((new_sub, new_monitor)) => {
                            sub = new_sub;
                            connection = new_monitor;
                        }
                        Err(error) => {
                            monitor_error = Some(error.to_string());
                            break;
                        }
                    }
                    disconnected = true;
                    watermarks = ReplayWatermarks::default();
                    replay_failures = 0;
                    applied_seq = -1;
                    bootstrap = false;
                    idle_reconcile = false;
                    check_live_duplicates = false;
                    live_events.reset();
                    replay_socket = None;
                }
                if !lost && disconnected && latest == Some(true) {
                    disconnected = false;
                    bootstrap = can_replay;
                    check_live_duplicates = can_replay;
                    next_catchup = tokio::time::Instant::now();
                }
                if disconnected {
                    tokio::select! {
                        result = connection.wait() => {
                            if let Err(error) = result {
                                monitor_error = Some(error.to_string());
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
                    && applied_seq >= 0
                    && tokio::time::Instant::now() >= next_idle_replay
                {
                    // A timer is not evidence of loss. Keep the verified view
                    // until replay actually returns an unapplied event.
                    idle_reconcile = true;
                    bootstrap = true;
                    check_live_duplicates = true;
                    next_catchup = tokio::time::Instant::now();
                }
                if bootstrap && tokio::time::Instant::now() >= next_catchup {
                    if applied_seq == -1 && !suspended {
                        if subscriber.suspend(Reason::Registered).is_err() {
                            break;
                        }
                        suspended = true;
                    }
                    let result = {
                        let replay = subscriber.replay(
                            &ctx,
                            applied_seq,
                            &mut watermarks,
                            &live_events,
                            &mut replay_socket,
                            idle_reconcile,
                            &shutdown_rx,
                        );
                        match super::transport::while_connected(
                            &mut connection,
                            replay,
                            !suspended,
                            &mut sub,
                            topic.as_bytes(),
                            |frames, topic| live_events.push(frames, topic),
                        )
                        .await
                        {
                            Ok(ReplayPoll::Disconnected) => {
                                reconnect = true;
                                continue;
                            }
                            Ok(result) => result,
                            Err(error) => {
                                monitor_error = Some(error.to_string());
                                break;
                            }
                        }
                    };
                    let (lost_during_replay, _) = match connection.drain() {
                        Ok(events) => events,
                        Err(error) => {
                            monitor_error = Some(error.to_string());
                            break;
                        }
                    };
                    if lost_during_replay {
                        reconnect = true;
                        continue;
                    }
                    let result = match result {
                        ReplayPoll::Finished(Ok(progress)) => {
                            if let Err(error) = live_events.validate_handoff(
                                topic.as_bytes(),
                                progress.applied_seq,
                                watermarks.clear_seq,
                            ) {
                                warn!(%topic, applied_seq = progress.applied_seq, %error, "KV replay handoff rejected");
                                replay_socket = None;
                                ReplayPoll::Finished(Err(ReplayError::Rejected))
                            } else {
                                ReplayPoll::Finished(Ok(progress))
                            }
                        }
                        result => result,
                    };
                    if let ReplayPoll::Finished(Ok(progress)) = result {
                        RouterMetrics::record_kv_replay_outcome("applied");
                        replay_failures = 0;
                        let advanced = progress.applied_seq != applied_seq;
                        applied_seq = progress.applied_seq;
                        let reason = if std::mem::take(&mut watermarks).clear_seq.is_some() {
                            Reason::ClearBarrier
                        } else {
                            Reason::ReplayApplied
                        };
                        if suspended || (idle_reconcile && advanced) {
                            if subscriber.resume(applied_seq, reason).is_err() {
                                break;
                            }
                            suspended = false;
                        }
                        if advanced {
                            debug!(source = %source_id, rank = dp_rank, seq = applied_seq, "kv_index replay sequence advanced");
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
                        let overflow = matches!(result, ReplayPoll::BufferFull);
                        if overflow {
                            // The replay must cover any discarded live data.
                            watermarks.required_live_seq = watermarks
                                .required_live_seq
                                .max(live_events.discard_buffered_frames(topic.as_bytes()));
                        } else {
                            replay_failures += 1;
                        }
                        let exhausted = replay_failures == MAX_REPLAY_FAILURES;
                        let (reason, outcome) = if overflow {
                            (Reason::LiveBufferFull, "buffer_full")
                        } else if exhausted {
                            (Reason::RetryExhausted, "retry_exhausted")
                        } else {
                            (Reason::ReplayFailed, "failed")
                        };
                        RouterMetrics::record_kv_replay_outcome(outcome);
                        if let ReplayPoll::Finished(Err(ReplayError::Incomplete(progress))) = result
                        {
                            if !exhausted {
                                // Leave the verified prefix hidden and retry only unapplied
                                // batches. Keep the live sequence requirement and its queued frame.
                                if subscriber.suspend(reason).is_err() {
                                    break;
                                }
                                applied_seq = progress.applied_seq;
                                suspended = true;
                                next_catchup =
                                    tokio::time::Instant::now() + Duration::from_millis(500);
                                continue;
                            }
                        }
                        if subscriber.invalidate(had_credit, reason).is_err() {
                            break;
                        }
                        applied_seq = -1;
                        watermarks.clear_seq = None;
                        suspended = true;
                        if exhausted {
                            live_events.reset();
                        } else if !overflow {
                            live_events.clear_fingerprints(topic.as_bytes());
                        }
                        if had_credit {
                            subscriber.invalidated("replay lost continuity");
                        }
                        if exhausted {
                            // Discard frames queued during the failed replay. In
                            // compatibility mode only fresh observations can
                            // start a new view. Strict mode waits for a clear.
                            match ConnectionMonitor::fresh_sub(&ctx, &pub_endpoint, &topic, hwm) {
                                Ok((new_sub, new_monitor)) => {
                                    sub = new_sub;
                                    connection = new_monitor;
                                }
                                Err(error) => {
                                    monitor_error = Some(error.to_string());
                                    break;
                                }
                            }
                            bootstrap = false;
                            check_live_duplicates = false;
                            watermarks = ReplayWatermarks::default();
                            continue;
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
                let (required_live_seq, check_replayed_duplicates, resume_after_seq) = 'live_batch: {
                    let applied_seq_for_duplicates = check_live_duplicates.then_some(applied_seq);
                    let received = live_events
                        .pop(topic.as_bytes(), applied_seq_for_duplicates)
                        .map_or_else(
                            || {
                                sub.socket().recv_multipart(zmq::DONTWAIT).map(|frames| {
                                    let is_duplicate = live_events.is_replayed_duplicate(
                                        &frames,
                                        topic.as_bytes(),
                                        applied_seq_for_duplicates,
                                    );
                                    (frames, is_duplicate)
                                })
                            },
                            Ok,
                        );
                    match received {
                        Ok((frames, is_replayed_duplicate)) => {
                            let seq = match live_sequence(&frames, topic.as_bytes()) {
                                Ok(Some(seq)) if (0..i64::MAX).contains(&seq) => seq,
                                Ok(None) => {
                                    tokio::task::yield_now().await;
                                    continue 'live;
                                }
                                Ok(Some(_)) | Err(()) => {
                                    if replay_failures == MAX_REPLAY_FAILURES {
                                        tokio::task::yield_now().await;
                                        continue 'live;
                                    }
                                    if subscriber
                                        .invalidate(true, Reason::ValidationFailed)
                                        .is_err()
                                    {
                                        break 'live;
                                    }
                                    subscriber.invalidated("invalid live sequence");
                                    suspended = true;
                                    tokio::task::yield_now().await;
                                    break 'live_batch (None, true, -1);
                                }
                            };
                            next_idle_replay = tokio::time::Instant::now() + IDLE_REPLAY_INTERVAL;
                            if replay_failures == MAX_REPLAY_FAILURES {
                                if subscriber.apply_fresh_live_batch(seq, &frames[2]).is_ok() {
                                    applied_seq = seq;
                                    suspended = false;
                                    replay_failures = 0;
                                }
                                tokio::task::yield_now().await;
                                continue 'live;
                            }
                            if is_replayed_duplicate {
                                bootstrap = false;
                                replay_socket = None;
                                continue 'live;
                            }
                            // Sequence zero outside the verified replay overlap
                            // can belong to a new publisher lifetime.
                            if applied_seq != -1 && (seq < applied_seq || seq == 0) {
                                let incarnation = crate::kv_index::next_owner_incarnation();
                                warn!(
                                    "kv_index {}: restart seq {} after {} → incarnation {}",
                                    source_id, seq, applied_seq, incarnation
                                );
                                if subscriber
                                    .invalidate(true, Reason::IncarnationReset)
                                    .is_err()
                                {
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
                                break 'live_batch (Some(seq), true, -1);
                            }
                            if seq <= applied_seq {
                                if subscriber
                                    .invalidate(true, Reason::ValidationFailed)
                                    .is_err()
                                {
                                    break 'live;
                                }
                                subscriber.invalidated("duplicate live sequence");
                                suspended = true;
                                break 'live_batch (Some(seq), check_live_duplicates, -1);
                            }
                            let expected = applied_seq + 1; // Every accepted sequence leaves room.
                            if (can_replay || applied_seq >= 0) && seq > expected {
                                if can_replay && applied_seq >= 0 {
                                    if subscriber.suspend(Reason::SequenceGap).is_err() {
                                        break 'live;
                                    }
                                    crate::kv_index::record_signal_loss();
                                } else if subscriber.invalidate(true, Reason::SequenceGap).is_err()
                                {
                                    break 'live;
                                }
                                emit(
                                    &subscriber.signals,
                                    IngestionSignal::Gap {
                                        source: source_id.clone(),
                                        from_seq: expected,
                                        to_seq: seq - 1,
                                    },
                                );
                                suspended = true;
                                if can_replay {
                                    live_events.push_gap_trigger(seq, frames);
                                }
                                break 'live_batch (Some(seq), true, applied_seq);
                            }
                            if subscriber.apply_next_batch(seq, &frames[2]).is_err() {
                                if subscriber
                                    .invalidate(true, Reason::ValidationFailed)
                                    .is_err()
                                {
                                    break 'live;
                                }
                                subscriber.invalidated("invalid live KV event batch");
                                suspended = true;
                                tokio::task::yield_now().await;
                                break 'live_batch (Some(seq), true, -1);
                            }
                            if suspended {
                                if subscriber.resume(seq, Reason::InitialSequence).is_err() {
                                    break 'live;
                                }
                                suspended = false;
                            }
                            applied_seq = seq;
                            bootstrap = false;
                            replay_socket = None;
                            check_live_duplicates = false;
                            live_events.clear_fingerprints(topic.as_bytes());
                            debug!(source = %source_id, rank = dp_rank, seq, "kv_index live sequence advanced");
                            // A sustained publisher starves EAGAIN; yield so this task
                            // can't monopolize the runtime worker.
                            tokio::task::yield_now().await;
                        }
                        Err(zmq::Error::EAGAIN) => {
                            tokio::select! {
                                result = connection.wait() => {
                                    if let Err(error) = result {
                                        monitor_error = Some(error.to_string());
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
                    applied_seq = required_live_seq.unwrap_or(-1);
                    suspended = false;
                    continue;
                }
                applied_seq = resume_after_seq;
                watermarks.required_live_seq = required_live_seq;
                watermarks.highest_seen_replay_seq = None;
                bootstrap = true;
                next_catchup = tokio::time::Instant::now();
                check_live_duplicates = check_replayed_duplicates;
                live_events.clear_fingerprints(topic.as_bytes());
                replay_socket = None;
            }
            if is_shutdown(&mut shutdown_rx) || !subscriber.active() {
                return;
            }
            subscriber.monitor_failed(monitor_error.as_deref().unwrap_or("subscriber stopped"));
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
