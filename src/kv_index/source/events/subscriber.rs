//! ZMQ SUB + DEALER replay subscriber, one task per (worker, rank).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{broadcast, mpsc};
use tracing::{debug, info, warn};

use crate::kv_index::index::source::{StreamControl, StreamPhase, StreamState};
use crate::kv_index::index::{IndexBatch, KvBlockIndexer};
use crate::kv_index::source::binding::SourceLease;
use crate::kv_index::source::events::apply::prepare_full_batch;
#[cfg(all(test, feature = "cpu", feature = "ready"))]
use crate::kv_index::source::events::apply::{apply_batch, hash_str, hex};
use crate::kv_index::source::events::transport::live_sequence;
#[cfg(test)]
use crate::kv_index::source::events::transport::read_seq;
use crate::kv_index::source::events::transport::{
    retry_backoff, ConnectionMonitor, IDLE_REPLAY_INTERVAL,
};
use crate::kv_index::source::events::validate::FullAttentionBatchRejection;
#[cfg(all(test, feature = "cpu", feature = "ready"))]
use crate::kv_index::types::StorageTier;
use crate::kv_index::types::{ResidencyOwner, SourceId};
#[cfg(all(test, feature = "cpu", feature = "ready"))]
use crate::kv_index::wire::ExternalBlockHash;
use crate::kv_index::wire::{KVEvent, KVEventBatch};
use crate::kv_index::{decode_batch, IngestionSignal};

/// sha256 of each block's token-id bytes, seeded by `lora_name`.
/// The seed is re-applied per block (no parent chaining here) so identical
/// tokens under different adapters land on different nodes.
pub fn local_hashes(token_ids: &[u32], block_size: u32, lora_name: Option<&str>) -> Vec<[u8; 32]> {
    crate::kv_index::local_block_hashes(token_ids, block_size, lora_name)
}

/// The signed sentinel ending a replay stream: `seq = -1`.
const END_SEQ: i64 = -1;

pub use crate::kv_index::source::endpoints::EventSource;

/// A handle to a running subscriber task; drop it (or call `shutdown`) to stop.
pub struct SubscriberHandle {
    shutdown_tx: broadcast::Sender<()>,
}

impl SubscriberHandle {
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }
}

/// Spawn one subscriber task for an event source. Applies events to the shared
/// indexer in `seq` order; on a gap, opens a DEALER replay to `replay_endpoint`.
/// `signal_tx` receives ingestion signals for the trust arbiter; None disables.
pub fn spawn(
    source: EventSource,
    indexer: Arc<KvBlockIndexer>,
    signal_tx: Option<mpsc::Sender<IngestionSignal>>,
    ctx: Arc<zmq::Context>,
) -> SubscriberHandle {
    spawn_inner(source, indexer, signal_tx, ctx, None, None)
}

/// The supervisor shares this guard across owners of one publisher/exact topic.
/// Its active lease only fences this local task, not the remote publisher.
pub(in crate::kv_index) fn spawn_bound(
    source: EventSource,
    indexer: Arc<KvBlockIndexer>,
    signal_tx: Option<mpsc::Sender<IngestionSignal>>,
    ctx: Arc<zmq::Context>,
    expected_block_size: Option<u32>,
    lease: SourceLease,
) -> SubscriberHandle {
    spawn_inner(
        source,
        indexer,
        signal_tx,
        ctx,
        expected_block_size,
        Some(lease),
    )
}

/// Keep the final rank decision and every index mutation in the same critical
/// section as supervisor promotion/retirement. Never retain it across an await.
fn with_binding<T>(binding: Option<&SourceLease>, operation: impl FnOnce(bool) -> T) -> Option<T> {
    if let Some(binding) = binding {
        binding.with_active(operation)
    } else {
        Some(operation(false))
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum BatchRank {
    Owned,
    Foreign,
    Rejected,
}

fn batch_rank(rank: Option<u32>, dp_rank: u32, strict: bool) -> BatchRank {
    match rank {
        Some(rank) if rank == dp_rank => BatchRank::Owned,
        Some(_) if strict => BatchRank::Foreign,
        None if !strict => BatchRank::Owned,
        _ => BatchRank::Rejected,
    }
}

fn spawn_inner(
    source: EventSource,
    indexer: Arc<KvBlockIndexer>,
    signal_tx: Option<mpsc::Sender<IngestionSignal>>,
    ctx: Arc<zmq::Context>,
    expected_block_size: Option<u32>,
    binding: Option<SourceLease>,
) -> SubscriberHandle {
    let stream = FullStream {
        index: indexer,
        owner: ResidencyOwner::Worker {
            source: source.source.clone(),
            dp_rank: source.dp_rank,
            incarnation: crate::kv_index::next_owner_incarnation(),
        },
        control: Arc::new(StreamControl::default()),
    };
    spawn_subscriber(
        Subscriber {
            source,
            sink: EventSink::Full {
                stream,
                signals: signal_tx.filter(|signals| !signals.is_closed()),
                expected_block_size,
                binding,
            },
            incarnation: 0,
        },
        ctx,
    )
}

pub(super) fn spawn_layout(
    source: EventSource,
    stream: Arc<super::event_stream::EventStream>,
    lease: super::event_stream::StreamLease,
    ctx: Arc<zmq::Context>,
) -> SubscriberHandle {
    spawn_subscriber(
        Subscriber {
            source,
            sink: EventSink::Layout { stream, lease },
            incarnation: 0,
        },
        ctx,
    )
}

enum EventSink {
    Full {
        stream: FullStream,
        signals: Option<mpsc::Sender<IngestionSignal>>,
        expected_block_size: Option<u32>,
        binding: Option<SourceLease>,
    },
    Layout {
        stream: Arc<super::event_stream::EventStream>,
        lease: super::event_stream::StreamLease,
    },
}

/// The ordinary worker path uses the same source state as a declared layout.
struct FullStream {
    index: Arc<KvBlockIndexer>,
    owner: ResidencyOwner,
    control: Arc<StreamControl>,
}

impl FullStream {
    fn with_state<R>(
        &self,
        binding: Option<&SourceLease>,
        operation: impl FnOnce(bool, &mut StreamState, &mut IndexBatch<'_>) -> R,
    ) -> Option<R> {
        with_binding(binding, |strict| {
            self.index.with_stream_batch(
                &self.owner.target(),
                &self.owner,
                &self.control,
                |state, tx| {
                    if state.lease_id == 0 {
                        state.begin_lease();
                        state.reset(Some(0), true);
                    }
                    operation(strict, state, tx)
                },
            )
        })
    }

    fn retire(&self) {
        self.index.with_existing_stream_batch(
            &self.owner.target(),
            &self.owner,
            &self.control,
            |state, _| {
                state.reset(None, false);
                state.lease_id = 0;
            },
        );
    }
}

struct Subscriber {
    source: EventSource,
    sink: EventSink,
    incarnation: u64,
}

impl Subscriber {
    fn is_layout(&self) -> bool {
        matches!(self.sink, EventSink::Layout { .. })
    }

    fn active(&self) -> bool {
        match &self.sink {
            EventSink::Full { binding, .. } => binding.as_ref().is_none_or(SourceLease::is_active),
            EventSink::Layout { stream, lease } => stream.is_active(*lease),
        }
    }

    fn bootstrap(&self) -> bool {
        match &self.sink {
            EventSink::Full {
                binding,
                expected_block_size,
                ..
            } => binding.is_some() && expected_block_size.is_some(),
            EventSink::Layout { .. } => true,
        }
    }

    fn invalidate(&self, loss: bool) -> Result<(), ()> {
        match &self.sink {
            EventSink::Full {
                binding, stream, ..
            } => stream
                .with_state(binding.as_ref(), |_, state, tx| {
                    if loss {
                        crate::kv_index::record_signal_loss();
                    }
                    tx.invalidate_source(&self.source.source, self.source.dp_rank);
                    state.reset(Some(0), false);
                })
                .ok_or(()),
            EventSink::Layout { stream, lease } => {
                stream.restart_replay(*lease, loss).map_err(|_| ())
            }
        }
    }

    fn suspend(&self, loss: bool) -> Result<(), ()> {
        match &self.sink {
            EventSink::Full {
                binding, stream, ..
            } => stream
                .with_state(binding.as_ref(), |_, state, tx| {
                    if loss {
                        crate::kv_index::record_signal_loss();
                    }
                    tx.suspend_source(&self.source.source, self.source.dp_rank);
                    state.hide();
                    state.phase = StreamPhase::Replaying;
                })
                .ok_or(()),
            EventSink::Layout { stream, lease } => {
                stream.suspend_for_replay(*lease, loss).map_err(|_| ())
            }
        }
    }

    fn resume(&self, last: i64) -> Result<(), ()> {
        match &self.sink {
            EventSink::Full {
                binding, stream, ..
            } => stream
                .with_state(binding.as_ref(), |_, state, tx| {
                    tx.resume_source(&self.source.source, self.source.dp_rank);
                    state.next_seq = Some((last + 1) as u64);
                    state.publish();
                    state.phase = StreamPhase::Live;
                })
                .ok_or(()),
            EventSink::Layout { stream, lease } => stream
                .finish_replay(*lease, (last + 1) as u64)
                .map_err(|_| ()),
        }
    }

    fn valid_batch(&self, batch: Option<&KVEventBatch>) -> bool {
        match &self.sink {
            EventSink::Full { binding, .. } => with_binding(binding.as_ref(), |strict| {
                batch.is_some_and(|batch| {
                    batch_rank(batch.data_parallel_rank, self.source.dp_rank, strict)
                        != BatchRank::Rejected
                })
            })
            .unwrap_or(false),
            // Layout decoding and group validation are atomic with its source transaction.
            EventSink::Layout { .. } => true,
        }
    }

    /// Foreign-rank batches advance the publisher cursor without resuming this rank.
    fn apply(
        &mut self,
        seq: u64,
        payload: &[u8],
        batch: Option<KVEventBatch>,
        resume: bool,
    ) -> Result<bool, ()> {
        match &mut self.sink {
            EventSink::Full {
                binding,
                stream,
                expected_block_size,
                ..
            } => {
                let batch = batch.or_else(|| decode(payload)).ok_or(())?;
                match with_binding(binding.as_ref(), |strict| {
                    batch_rank(batch.data_parallel_rank, self.source.dp_rank, strict)
                })
                .ok_or(())?
                {
                    BatchRank::Foreign => return Ok(false),
                    BatchRank::Rejected => return Err(()),
                    BatchRank::Owned => {}
                }
                let prepared = prepare_full_batch(
                    &self.source.source,
                    self.source.dp_rank,
                    self.incarnation,
                    &batch,
                    *expected_block_size,
                );
                stream
                    .with_state(binding.as_ref(), |strict, state, tx| {
                        match batch_rank(batch.data_parallel_rank, self.source.dp_rank, strict) {
                            BatchRank::Owned => prepared
                                .and_then(|prepared| prepared.apply(tx))
                                .map(|()| {
                                    if batch
                                        .events
                                        .iter()
                                        .any(|event| matches!(event, KVEvent::AllBlocksCleared(_)))
                                    {
                                        state.bump_generation();
                                    }
                                    state.next_seq = Some(seq + 1);
                                    if resume {
                                        tx.resume_source(&self.source.source, self.source.dp_rank);
                                        state.publish();
                                        state.phase = StreamPhase::Live;
                                    }
                                    true
                                })
                                .map_err(|rejection| {
                                    tx.invalidate_source(&self.source.source, self.source.dp_rank);
                                    state.reset(Some(0), false);
                                    report_full_attention_rejection(
                                        rejection,
                                        &self.source.source,
                                        self.source.dp_rank,
                                        &mut state.rejection_reported,
                                    )
                                }),
                            BatchRank::Foreign => Ok(false),
                            BatchRank::Rejected => Err(()),
                        }
                    })
                    .ok_or(())?
            }
            EventSink::Layout { stream, lease } => stream
                .apply_payload(*lease, seq, payload)
                .map(|()| true)
                .map_err(|_| ()),
        }
    }

    fn emit(&self, signal: IngestionSignal) {
        if let EventSink::Full { signals, .. } = &self.sink {
            emit(signals, signal);
        }
    }

    fn monitor_failed(&self, error: impl std::fmt::Display) {
        warn!(%error, source = %self.source.source, "KV event connection monitor failed");
        let _ = self.invalidate(true);
    }

    #[allow(clippy::too_many_arguments)]
    async fn replay(
        &mut self,
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
        let layout = self.is_layout();
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
                        // Layout recovery must reach EOF and cover the observed gap;
                        // full-attention gap recovery stops as soon as its target is applied.
                        if through.is_some_and(|required| !layout || next <= required) {
                            return None;
                        }
                        *socket = Some(dealer);
                        return Some(last);
                    }
                    let seq = u64::try_from(seq).ok()?;
                    if frame_topic != topic.as_bytes() {
                        return None;
                    }
                    if !layout && seq < next {
                        continue;
                    }
                    if seq != next {
                        warn!(%topic, seq, expected = next, "KV replay sequence gap");
                        return None;
                    }
                    if reconcile && seq == start {
                        self.suspend(false).ok()?;
                    }
                    self.apply(seq, payload, None, false).ok()?;
                    if let Some(replayed) = replayed.as_deref_mut() {
                        super::transport::remember_replay(replayed, seq, payload);
                    }
                    last = i64::try_from(seq).ok()?;
                    self.emit(IngestionSignal::Advance {
                        source: self.source.source.clone(),
                        last_seq: last,
                    });
                    next = seq.checked_add(1)?;
                    if !layout && through.is_some_and(|required| seq >= required) {
                        return Some(last);
                    }
                    if !layout {
                        tokio::task::yield_now().await;
                    }
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
        match &self.sink {
            EventSink::Full { stream, .. } => stream.retire(),
            EventSink::Layout { stream, lease } => {
                let _ = stream.retire(*lease);
            }
        }
    }
}

fn spawn_subscriber(mut subscriber: Subscriber, ctx: Arc<zmq::Context>) -> SubscriberHandle {
    let (shutdown_tx, mut shutdown_rx) = broadcast::channel::<()>(1);
    let source = &subscriber.source;
    let pub_endpoint = source.pub_endpoint.clone();
    let replay_endpoint = source.replay_endpoint.clone();
    let topic = source.topic.clone();
    let source_id = source.source.clone();
    let dp_rank = source.dp_rank;
    let hwm = source.hwm;
    let layout = subscriber.is_layout();
    let bound_full = !layout && subscriber.bootstrap();
    let can_replay = subscriber.bootstrap() && replay_endpoint.is_some();

    tokio::spawn(async move {
        let mut failures = 0u32;
        let mut restarting = false;
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
                        restarting = true;
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
            subscriber.incarnation = crate::kv_index::next_owner_incarnation();
            if layout && subscriber.invalidate(false).is_err() {
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
            let mut recovery_pending = restarting && bootstrap;
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
                    suspended = true;
                    match ConnectionMonitor::fresh_sub(&ctx, &pub_endpoint, &topic, hwm) {
                        Ok((new_sub, new_monitor)) => {
                            sub = new_sub;
                            connection = new_monitor;
                        }
                        Err(_) => break,
                    }
                    disconnected = true;
                    recovery_pending = true;
                    recovery_through = None;
                    last_seq = -1;
                    subscriber.incarnation = crate::kv_index::next_owner_incarnation();
                    bootstrap = false;
                    idle_reconcile = false;
                    allow_overlap = false;
                    replayed.clear();
                    replay_socket = None;
                }
                if !lost && disconnected && latest == Some(true) {
                    disconnected = false;
                    bootstrap = can_replay;
                    if !bootstrap {
                        recovery_pending = false;
                    }
                    allow_overlap = bootstrap;
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
                if !bootstrap
                    && last_seq >= 0
                    && can_replay
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
                        if suspended || (idle_reconcile && (!layout || advanced)) {
                            if subscriber.resume(last_seq).is_err() {
                                break;
                            }
                            suspended = false;
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
                            subscriber.emit(IngestionSignal::Invalidated {
                                source: source_id.clone(),
                                dp_rank,
                                reason: "startup replay lost continuity",
                            });
                        }
                        next_catchup = tokio::time::Instant::now() + Duration::from_millis(500);
                    }
                }
                // A layout needs a completed replay before any live batch can
                // restore credit; queued live stores cannot prove missed evictions.
                if layout && bootstrap && suspended {
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
                                Ok(Some(seq)) if seq >= 0 || (seq == END_SEQ && !layout) => seq,
                                Ok(None) => {
                                    tokio::task::yield_now().await;
                                    continue 'live;
                                }
                                Ok(Some(_)) | Err(()) => {
                                    if subscriber.invalidate(true).is_err() {
                                        break 'live;
                                    }
                                    suspended = true;
                                    tokio::task::yield_now().await;
                                    if layout {
                                        break 'live_batch (None, true);
                                    }
                                    continue 'live;
                                }
                            };
                            if seq == END_SEQ {
                                continue 'live;
                            }
                            next_idle_replay = tokio::time::Instant::now() + IDLE_REPLAY_INTERVAL;
                            if allow_overlap
                                && seq > 0
                                && seq <= last_seq
                                && super::transport::replay_overlap(
                                    &replayed, seq as u64, &frames[2],
                                )
                            {
                                bootstrap = false;
                                replay_socket = None;
                                if recovery_pending {
                                    if suspended && subscriber.resume(last_seq).is_err() {
                                        break 'live;
                                    }
                                    suspended = false;
                                    recovery_pending = false;
                                }
                                continue 'live;
                            }
                            // A restarted publisher can emit sequence zero while our
                            // last observed event was also zero. It is indistinguishable
                            // from a duplicate, so discard the old incarnation before
                            // accepting it; retaining the old claims can report blocks
                            // from a dead engine as resident.
                            if last_seq != -1 && (seq < last_seq || seq == 0) {
                                subscriber.incarnation = crate::kv_index::next_owner_incarnation();
                                warn!(
                                    "kv_index {}: restart seq {} after {} → incarnation {}",
                                    source_id, seq, last_seq, subscriber.incarnation
                                );
                                if subscriber.invalidate(true).is_err() {
                                    break 'live;
                                }
                                suspended = true;
                                last_seq = -1;
                                if layout {
                                    break 'live_batch (Some(seq), true);
                                }
                                subscriber.emit(IngestionSignal::IncarnationReset {
                                    source: source_id.clone(),
                                    incarnation: subscriber.incarnation,
                                });
                            }
                            if layout && seq <= last_seq {
                                if subscriber.invalidate(true).is_err() {
                                    break 'live;
                                }
                                suspended = true;
                                break 'live_batch (Some(seq), allow_overlap);
                            }
                            if seq <= last_seq {
                                tokio::task::yield_now().await;
                                continue 'live;
                            }
                            let batch = (!layout).then(|| decode(&frames[2])).flatten();
                            let valid_rank = subscriber.valid_batch(batch.as_ref());
                            // Include initial attachment to an already-running publisher.
                            // A corrupt event is a one-event hole, not an applied seq.
                            if seq > last_seq + 1 || !valid_rank {
                                let from = last_seq + 1;
                                let to = if !valid_rank { seq } else { seq - 1 };
                                if subscriber.suspend(true).is_err() {
                                    break 'live;
                                }
                                suspended = true;
                                subscriber.emit(IngestionSignal::Gap {
                                    source: source_id.clone(),
                                    from_seq: from,
                                    to_seq: to,
                                });
                                if layout {
                                    if subscriber.invalidate(true).is_err() {
                                        break 'live;
                                    }
                                    break 'live_batch (Some(seq), true);
                                }
                                if replay_endpoint.is_some() {
                                    let replay_result = subscriber
                                        .replay(
                                            &ctx,
                                            last_seq,
                                            Some(seq),
                                            None,
                                            &mut replay_socket,
                                            false,
                                            &shutdown_rx,
                                        )
                                        .await;
                                    let (lost_during_gap, _) = match connection.drain() {
                                        Ok(events) => events,
                                        Err(error) => {
                                            subscriber.monitor_failed(error);
                                            break 'live;
                                        }
                                    };
                                    if lost_during_gap {
                                        reconnect = true;
                                        continue 'live;
                                    }
                                    if let Some(last_replay) = replay_result {
                                        last_seq = last_replay;
                                        if subscriber.resume(last_seq).is_err() {
                                            break 'live;
                                        }
                                        suspended = false;
                                        recovery_pending = false;
                                        subscriber.emit(IngestionSignal::ReplayApplied {
                                            source: source_id.clone(),
                                            replay_seq: last_replay,
                                        });
                                        tokio::task::yield_now().await;
                                        continue 'live;
                                    }
                                }
                                // Partial replay can include stores whose later
                                // evictions were lost. Never expose that partial state.
                                //
                                // The gap already fenced Program warm candidates.
                                if subscriber.invalidate(false).is_err() {
                                    break 'live;
                                }
                                if recovery_pending {
                                    // A new live gap before handoff disproves this replay.
                                    // Keep the source hidden and retry the complete history.
                                    next_catchup =
                                        tokio::time::Instant::now() + Duration::from_millis(100);
                                    break 'live_batch (Some(seq), true);
                                }
                            }
                            if valid_rank {
                                let resume = suspended || bound_full;
                                let owned =
                                    match subscriber.apply(seq as u64, &frames[2], batch, resume) {
                                        Ok(owned) => owned,
                                        Err(()) => {
                                            if subscriber.invalidate(true).is_err() {
                                                break 'live;
                                            }
                                            suspended = true;
                                            tokio::task::yield_now().await;
                                            if layout {
                                                break 'live_batch (Some(seq), true);
                                            }
                                            continue 'live;
                                        }
                                    };
                                if owned && (suspended || bound_full) {
                                    if layout && subscriber.resume(seq).is_err() {
                                        break 'live;
                                    }
                                    suspended = false;
                                    recovery_pending = false;
                                }
                                last_seq = seq;
                                bootstrap = false;
                                replay_socket = None;
                                allow_overlap = false;
                                replayed.clear();
                                subscriber.emit(IngestionSignal::Advance {
                                    source: source_id.clone(),
                                    last_seq: seq,
                                });
                                debug!(source = %source_id, rank = dp_rank, seq, "kv_index live sequence advanced");
                            }
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
            restarting = true;
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
fn emit(tx: &Option<mpsc::Sender<IngestionSignal>>, signal: IngestionSignal) {
    if let Some(tx) = tx {
        if let Err(mpsc::error::TrySendError::Full(_)) = tx.try_send(signal) {
            debug!("kv_index signal queue full");
        }
    }
}

fn decode(payload: &[u8]) -> Option<KVEventBatch> {
    match decode_batch(payload) {
        Ok(b) => Some(b),
        Err(e) => {
            warn!("kv_index decode batch: {}", e);
            None
        }
    }
}

fn report_full_attention_rejection(
    rejection: FullAttentionBatchRejection,
    source_id: &SourceId,
    rank: u32,
    reported: &mut bool,
) {
    // A noisy publisher gets at most one diagnostic per subscription, including replay.
    if !*reported {
        debug!(source = %source_id, rank, reason = ?rejection.reason,
            event_index = rejection.event_index, "kv_index dense batch rejected");
        *reported = true;
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[path = "subscriber_tests.rs"]
pub(super) mod tests;
