//! ZMQ SUB + DEALER replay subscriber, one task per (worker, rank).

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{broadcast, mpsc};
use tracing::{debug, info, warn};

use super::event_stream::{EventStream, StreamLease};
use super::transport::{
    live_sequence, retry_backoff, ConnectionMonitor, PendingLive, ReplayError, ReplayOverlap,
    ReplayPoll, IDLE_REPLAY_INTERVAL,
};
use crate::kv_index::types::SourceRecoveryReason as Reason;
use crate::kv_index::IngestionSignal;
use crate::metrics::RouterMetrics;

/// The signed sentinel ending a replay stream: `seq = -1`.
const END_SEQ: i64 = -1;
const MAX_REPLAY_FAILURES: u32 = 3;
const REPLAY_IDLE_TIMEOUT: Duration = Duration::from_secs(5);

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

    fn resume(&self, last: i64, reason: Reason) -> Result<(), ()> {
        self.binding.as_ref().map_or(Ok(()), |(stream, lease)| {
            let next = last.checked_add(1).ok_or(())?;
            stream.resume(*lease, next as u64, reason).map_err(|_| ())
        })
    }

    fn apply(&self, seq: u64, payload: &[u8], fresh: bool) -> Result<(), ()> {
        let cleared = if let Some((stream, lease)) = &self.binding {
            if fresh {
                stream.observe_live(*lease, seq, payload)
            } else {
                stream.apply_publisher_payload(*lease, seq, payload)
            }
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

    /// A retained suffix can start after the requested cursor. Discard earlier
    /// claims only if the suffix supplies a valid clear for this worker rank.
    fn try_replay_clear(&self, seq: u64, payload: &[u8]) -> Result<bool, ReplayError> {
        let batch = crate::kv_index::decode_batch(payload).map_err(|_| ReplayError::Invalid)?;
        // The publisher stamps the batch rank, and the bound index requires it.
        if batch.data_parallel_rank != Some(self.source.dp_rank) {
            return Err(ReplayError::Invalid);
        }
        if !batch
            .events
            .iter()
            .any(|event| matches!(event, crate::kv_index::KVEvent::AllBlocksCleared(_)))
        {
            return Ok(false);
        }
        let (stream, lease) = self.binding.as_ref().ok_or(ReplayError::Invalid)?;
        stream
            .rebase_replay(*lease, seq, batch)
            .map_err(|_| ReplayError::Invalid)?;
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
        mut last: i64,
        through: Option<i64>,
        gap_batch: Option<&(u64, Vec<u8>)>,
        replayed: &ReplayOverlap,
        socket: &mut Option<zmq::Socket>,
        reconcile: bool,
        shutdown: &broadcast::Receiver<()>,
    ) -> Result<i64, ReplayError> {
        let dealer = match socket.take() {
            Some(socket) => socket,
            None => {
                let socket = ctx
                    .socket(zmq::DEALER)
                    .map_err(|_| ReplayError::Retry(last))?;
                socket
                    .set_maxmsgsize(super::MAX_KV_EVENT_BYTES as i64)
                    .map_err(|_| ReplayError::Retry(last))?;
                socket.set_linger(0).map_err(|_| ReplayError::Retry(last))?;
                socket
                    .connect(
                        self.source
                            .replay_endpoint
                            .as_deref()
                            .ok_or(ReplayError::Invalid)?,
                    )
                    .map_err(|_| ReplayError::Retry(last))?;
                socket
            }
        };
        let start = u64::try_from(last.checked_add(1).ok_or(ReplayError::Invalid)?)
            .map_err(|_| ReplayError::Invalid)?;
        let through = through
            .map(u64::try_from)
            .transpose()
            .map_err(|_| ReplayError::Invalid)?;
        let topic = self.source.topic.clone();
        if let Err(error) = dealer.send_multipart([&[][..], &start.to_be_bytes()], zmq::DONTWAIT) {
            warn!(%topic, %error, "KV replay request failed");
            return Err(ReplayError::Retry(last));
        }
        let mut deadline = tokio::time::Instant::now() + self.replay_idle_timeout;
        let mut next = start;
        let mut awaiting_clear = false;
        while tokio::time::Instant::now() < deadline
            && shutdown.is_empty()
            && !shutdown.is_closed()
            && self.active()
        {
            match dealer.recv_multipart(zmq::DONTWAIT) {
                Ok(frames) => {
                    let (wire_seq, frame_topic, payload) =
                        super::transport::replay_frame(&frames).ok_or(ReplayError::Invalid)?;
                    if wire_seq == END_SEQ && frame_topic.is_empty() && payload.is_empty() {
                        if awaiting_clear {
                            return Err(ReplayError::Invalid);
                        }
                        // The first live batch after a gap can finish a contiguous
                        // replay when the publisher has not buffered that batch yet.
                        if let Some((seq, payload)) = gap_batch.filter(|(seq, _)| {
                            *seq == next && through.is_some_and(|required| *seq == required)
                        }) {
                            self.apply(*seq, payload, false)
                                .map_err(|_| ReplayError::Invalid)?;
                            replayed
                                .record(*seq, payload)
                                .map_err(|_| ReplayError::Invalid)?;
                            last = *seq as i64;
                            next += 1;
                        }
                        if through.is_some_and(|required| next <= required) {
                            return Err(ReplayError::Retry(last));
                        }
                        *socket = Some(dealer);
                        return Ok(last);
                    }
                    if !(0..i64::MAX).contains(&wire_seq) || frame_topic != topic.as_bytes() {
                        return Err(ReplayError::Invalid);
                    }
                    let seq = wire_seq as u64;
                    if seq != next {
                        if next != start || seq < start {
                            warn!(%topic, seq, expected = next, "KV replay sequence gap");
                            return Err(ReplayError::Invalid);
                        }
                        awaiting_clear = true;
                    }
                    if gap_batch.is_some_and(|(live_seq, live_payload)| {
                        *live_seq == seq && live_payload.as_slice() != payload
                    }) {
                        warn!(%topic, seq, "KV replay conflicts with the buffered live batch");
                        return Err(ReplayError::Invalid);
                    }
                    if reconcile && next == start {
                        self.suspend(Reason::IdleCatchup)
                            .map_err(|_| ReplayError::Invalid)?;
                        self.invalidated("idle replay found unapplied events");
                    }
                    replayed
                        .record(seq, payload)
                        .map_err(|_| ReplayError::Invalid)?;
                    if awaiting_clear {
                        awaiting_clear = !self.try_replay_clear(seq, payload)?;
                    } else {
                        self.apply(seq, payload, false)
                            .map_err(|_| ReplayError::Invalid)?;
                    }
                    if !awaiting_clear {
                        last = wire_seq;
                    }
                    next = seq + 1;
                    deadline = tokio::time::Instant::now() + self.replay_idle_timeout;
                    tokio::task::yield_now().await;
                }
                Err(zmq::Error::EAGAIN) => tokio::time::sleep(Duration::from_millis(1)).await,
                Err(error) => {
                    warn!(%topic, %error, "KV replay receive failed");
                    return Err(ReplayError::Retry(last));
                }
            }
        }
        if tokio::time::Instant::now() >= deadline {
            warn!(%topic, "KV replay made no progress before the idle timeout");
        }
        Err(ReplayError::Retry(last))
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
            if subscriber.invalidate(false, Reason::Registered).is_err() {
                return;
            }
            let mut suspended = false;
            // A native publisher may already have cached blocks when Router starts.
            // Catch up by replay until a live frame confirms the SUB handoff.
            let mut bootstrap = can_replay;
            let mut allow_overlap = bootstrap;
            let mut replayed = ReplayOverlap::default();
            let mut replay_socket = None;
            let mut pending_live = PendingLive::default();
            let mut next_catchup = tokio::time::Instant::now();
            let mut next_idle_replay = tokio::time::Instant::now() + IDLE_REPLAY_INTERVAL;
            let mut idle_reconcile = false;
            let mut disconnected = false;
            let mut reconnect = false;
            let mut recovery_through = None;
            let mut gap_batch = None;
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
                    recovery_through = None;
                    gap_batch = None;
                    replay_failures = 0;
                    last_seq = -1;
                    bootstrap = false;
                    idle_reconcile = false;
                    allow_overlap = false;
                    replayed = ReplayOverlap::default();
                    replay_socket = None;
                    pending_live = PendingLive::default();
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
                        if subscriber.suspend(Reason::Registered).is_err() {
                            break;
                        }
                        suspended = true;
                    }
                    let result = {
                        let replay = subscriber.replay(
                            &ctx,
                            last_seq,
                            recovery_through,
                            gap_batch.as_ref(),
                            &replayed,
                            &mut replay_socket,
                            idle_reconcile,
                            &shutdown_rx,
                        );
                        match super::transport::while_connected(
                            &mut connection,
                            replay,
                            !suspended,
                            &mut sub,
                            &mut pending_live,
                            topic.as_bytes(),
                            &replayed,
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
                        ReplayPoll::Finished(Ok(high))
                            if !pending_live.verifies_overlap(
                                topic.as_bytes(),
                                high,
                                &replayed,
                            ) =>
                        {
                            warn!(%topic, high, "KV replay could not verify buffered live overlap");
                            replay_socket = None;
                            ReplayPoll::Finished(Err(ReplayError::Invalid))
                        }
                        result => result,
                    };
                    if let ReplayPoll::Finished(Ok(high)) = result {
                        RouterMetrics::record_kv_replay_outcome("applied");
                        replay_failures = 0;
                        let advanced = high != last_seq;
                        last_seq = high;
                        recovery_through = None;
                        gap_batch = None;
                        if suspended || (idle_reconcile && advanced) {
                            if subscriber.resume(last_seq, Reason::ReplayApplied).is_err() {
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
                        let overflow = matches!(result, ReplayPoll::BufferFull);
                        if overflow {
                            // The replay must cover any discarded live data.
                            recovery_through =
                                recovery_through.max(pending_live.discard(topic.as_bytes()));
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
                        if let ReplayPoll::Finished(Err(ReplayError::Retry(high))) = result {
                            if !exhausted {
                                // Leave the verified prefix hidden and retry only unapplied
                                // batches. The gap watermark and its buffered frame still apply.
                                if subscriber.suspend(reason).is_err() {
                                    break;
                                }
                                last_seq = high;
                                suspended = true;
                                next_catchup =
                                    tokio::time::Instant::now() + Duration::from_millis(500);
                                continue;
                            }
                        }
                        if subscriber.invalidate(had_credit, reason).is_err() {
                            break;
                        }
                        last_seq = -1;
                        suspended = true;
                        if overflow || exhausted {
                            replayed = ReplayOverlap::default();
                        } else {
                            replayed.clear();
                        }
                        if !overflow {
                            gap_batch = None;
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
                            allow_overlap = false;
                            recovery_through = None;
                            pending_live = PendingLive::default();
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
                let (through, overlap, restart_from) = 'live_batch: {
                    let pending_frame = pending_live.pop();
                    let was_buffered = pending_frame.is_some();
                    let received = pending_frame
                        .map_or_else(|| sub.socket().recv_multipart(zmq::DONTWAIT), Ok);
                    match received {
                        Ok(mut frames) => {
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
                            let matches_replay = allow_overlap
                                && seq <= last_seq
                                && replayed.matches(seq as u64, &frames[2]);
                            if was_buffered {
                                replayed.release_live(seq as u64);
                            }
                            next_idle_replay = tokio::time::Instant::now() + IDLE_REPLAY_INTERVAL;
                            if replay_failures == MAX_REPLAY_FAILURES {
                                if subscriber.apply(seq as u64, &frames[2], true).is_ok() {
                                    last_seq = seq;
                                    suspended = false;
                                    replay_failures = 0;
                                }
                                tokio::task::yield_now().await;
                                continue 'live;
                            }
                            if matches_replay {
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
                            if seq <= last_seq {
                                if subscriber
                                    .invalidate(true, Reason::ValidationFailed)
                                    .is_err()
                                {
                                    break 'live;
                                }
                                subscriber.invalidated("duplicate live sequence");
                                suspended = true;
                                break 'live_batch (Some(seq), allow_overlap, -1);
                            }
                            let expected = last_seq + 1; // Every accepted sequence leaves room.
                            if (can_replay || last_seq >= 0) && seq > expected {
                                if can_replay && last_seq >= 0 {
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
                                    gap_batch = Some((
                                        seq as u64,
                                        frames.pop().expect("validated live payload"),
                                    ));
                                }
                                break 'live_batch (Some(seq), true, last_seq);
                            }
                            if subscriber.apply(seq as u64, &frames[2], false).is_err() {
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
                    last_seq = through.unwrap_or(-1);
                    suspended = false;
                    continue;
                }
                last_seq = restart_from;
                recovery_through = through;
                bootstrap = true;
                next_catchup = tokio::time::Instant::now();
                allow_overlap = overlap;
                replayed.clear();
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
