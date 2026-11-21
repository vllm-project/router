//! ZMQ SUB + DEALER replay subscriber, one task per (worker, rank).

use std::collections::VecDeque;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{broadcast, mpsc};
use tracing::{debug, info, warn};

use crate::kv_hash::{BlockKeyGenerator, SequenceHash};
use crate::kv_index::admit::{
    validate_full_attention_batch, FullAttentionBatchRejection, FullAttentionRejectionReason,
};
use crate::kv_index::binding::SourceLease;
use crate::kv_index::indexer::KvBlockIndexer;
use crate::kv_index::storage::PageChanges;
use crate::kv_index::types::{
    RecoveryBoundary, ResidencyOwner, SourceId, SourceRecoveryReason, StorageTier,
};
use crate::kv_index::wire::{ExternalBlockHash, KVEvent, KVEventBatch};
use crate::kv_index::{decode_batch, ordinary_block_keys, IngestionSignal};
use crate::metrics::RouterMetrics;

/// The signed sentinel ending a replay stream: `seq = -1`.
const END_SEQ: i64 = -1;

/// One event source for one (worker, rank).
#[derive(Clone, Debug)]
pub struct EventSource {
    pub source: SourceId,
    pub dp_rank: u32,
    pub pub_endpoint: String,
    pub replay_endpoint: Option<String>,
    pub topic: String,
    /// SUB receive high-water mark; None leaves the ZMQ default.
    pub hwm: Option<i32>,
    /// Opt-in authoritative-history recovery; otherwise retain fresh-observation credit.
    pub strict_recovery: bool,
}

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
    spawn_inner(
        source,
        indexer,
        signal_tx,
        ctx,
        None,
        None,
        ReplayLimits::default(),
    )
}

/// The supervisor shares this guard across owners of one publisher/exact topic.
/// Its active lease only fences this local task, not the remote publisher.
struct SubscriberBinding {
    lease: SourceLease,
    #[cfg(test)]
    replay_cleanup_barrier: Option<Arc<tokio::sync::Barrier>>,
    #[cfg(test)]
    replay_commit_barrier: Option<Arc<tokio::sync::Barrier>>,
}

pub(super) fn spawn_bound(
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
        Some(SubscriberBinding {
            lease,
            #[cfg(test)]
            replay_cleanup_barrier: None,
            #[cfg(test)]
            replay_commit_barrier: None,
        }),
        ReplayLimits::default(),
    )
}

/// Keep the final rank decision and every index mutation in the same critical
/// section as supervisor promotion/retirement. Never retain it across an await.
fn with_binding<T>(
    binding: Option<&SubscriberBinding>,
    operation: impl FnOnce(bool) -> T,
) -> Option<T> {
    if let Some(binding) = binding {
        binding.lease.with_active(operation)
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

#[derive(Clone, Copy)]
struct ReplayLimits {
    timeout: Duration,
    retry_base: Duration,
}

impl Default for ReplayLimits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            retry_base: Duration::from_secs(1),
        }
    }
}

fn spawn_inner(
    source: EventSource,
    indexer: Arc<KvBlockIndexer>,
    signal_tx: Option<mpsc::Sender<IngestionSignal>>,
    ctx: Arc<zmq::Context>,
    expected_block_size: Option<u32>,
    binding: Option<SubscriberBinding>,
    limits: ReplayLimits,
) -> SubscriberHandle {
    let (shutdown_tx, mut shutdown_rx) = broadcast::channel::<()>(1);
    let pub_endpoint = source.pub_endpoint.clone();
    let replay_endpoint = source.replay_endpoint.clone();
    let topic = source.topic.clone();
    let source_id = source.source.clone();
    let dp_rank = source.dp_rank;
    let hwm = source.hwm;
    let signals = signal_tx;
    let strict_recovery = source.strict_recovery;

    tokio::spawn(async move {
        let mut sub = match ctx.socket(zmq::SUB) {
            Ok(s) => s,
            Err(e) => {
                warn!("kv_index sub socket for {}: {}", source_id, e);
                return;
            }
        };
        if let Some(h) = hwm {
            sub.set_rcvhwm(h).ok();
        }
        if let Err(e) = sub.connect(&pub_endpoint) {
            warn!("kv_index connect {} to {}: {}", source_id, pub_endpoint, e);
            return;
        }
        if let Err(error) = sub.set_subscribe(topic.as_bytes()) {
            warn!(%source_id, %error, "kv_index subscribe failed");
            return;
        }
        if with_binding(binding.as_ref(), |_| {
            indexer.register_source(&source_id, dp_rank)
        })
        .is_none()
        {
            return;
        }
        // zmq recv is sync; DONTWAIT + sleep keeps the tokio worker free. No
        // rcvtimeo — it's a no-op under DONTWAIT.
        info!(
            "kv_index subscriber for {} rank{} on {}",
            source_id, dp_rank, pub_endpoint
        );

        let mut last_seq: i64 = -1;
        let mut incarnation: u64 = 0;
        let mut full_attention_rejection_reported = false;
        let mut retries = ReplayRetryBudget::new(limits.retry_base);
        let mut pending_live = VecDeque::new();

        loop {
            if is_shutdown(&mut shutdown_rx)
                || binding
                    .as_ref()
                    .is_some_and(|binding| !binding.lease.is_active())
            {
                info!("kv_index subscriber {} shutting down", source_id);
                break;
            }
            let received = pending_live
                .pop_front()
                .map_or_else(|| sub.recv_multipart(zmq::DONTWAIT), Ok);
            match received {
                Ok(frames) => {
                    // ZMQ SUB matches topic prefixes, not complete topics.
                    if frames
                        .first()
                        .is_some_and(|frame| frame.as_slice() != topic.as_bytes())
                    {
                        tokio::task::yield_now().await;
                        continue;
                    }
                    if frames.len() != 3 || frames[1].len() != 8 {
                        if with_binding(binding.as_ref(), |_| {
                            retries.invalidate_once(
                                &indexer,
                                &source_id,
                                dp_rank,
                                SourceRecoveryReason::ValidationFailed,
                            );
                        })
                        .is_none()
                        {
                            break;
                        }
                        tokio::task::yield_now().await;
                        continue;
                    }
                    let seq = read_seq(&frames[1]);
                    if seq == END_SEQ {
                        continue;
                    }
                    if last_seq != -1 && seq < last_seq {
                        incarnation += 1;
                        warn!(
                            "kv_index {}: restart seq {} < {} → incarnation {}",
                            source_id, seq, last_seq, incarnation
                        );
                        if with_binding(binding.as_ref(), |_| {
                            indexer.invalidate_source_for(
                                &source_id,
                                dp_rank,
                                SourceRecoveryReason::IncarnationReset,
                            );
                        })
                        .is_none()
                        {
                            break;
                        }
                        last_seq = -1;
                        retries.reset();
                        emit(
                            &signals,
                            IngestionSignal::IncarnationReset {
                                source: source_id.clone(),
                                incarnation,
                            },
                        );
                    }
                    if seq <= last_seq {
                        tokio::task::yield_now().await;
                        continue;
                    }
                    let batch = decode(&frames[2]);
                    let Some(valid_rank) = with_binding(binding.as_ref(), |strict| {
                        batch.as_ref().is_some_and(|batch| {
                            batch_rank(batch.data_parallel_rank, dp_rank, strict)
                                != BatchRank::Rejected
                        })
                    }) else {
                        break;
                    };
                    let needs_replay = seq > last_seq + 1 || !valid_rank;
                    let can_retry = retries.allowed(Instant::now());
                    if needs_replay
                        && !can_retry
                        && with_binding(binding.as_ref(), |_| {
                            retries.invalidate_once(
                                &indexer,
                                &source_id,
                                dp_rank,
                                SourceRecoveryReason::SequenceGap,
                            );
                        })
                        .is_none()
                    {
                        break;
                    }
                    // Include initial attachment to an already-running publisher.
                    // A corrupt event is a one-event hole, not an applied seq.
                    if needs_replay && can_retry {
                        let from = last_seq + 1;
                        let to = if !valid_rank { seq } else { seq - 1 };
                        let Some((replay_start, attempt)) = with_binding(binding.as_ref(), |_| {
                            indexer.begin_replay(&source_id, dp_rank, last_seq, strict_recovery)
                        }) else {
                            break;
                        };
                        emit(
                            &signals,
                            IngestionSignal::Gap {
                                source: source_id.clone(),
                                from_seq: from,
                                to_seq: to,
                            },
                        );
                        let result = if let Some(ep) = replay_endpoint.as_deref() {
                            let recovery = replay(
                                ep,
                                replay_start,
                                seq,
                                attempt,
                                &ctx,
                                &indexer,
                                &source_id,
                                dp_rank,
                                incarnation,
                                signals.clone(),
                                expected_block_size,
                                &topic,
                                &mut full_attention_rejection_reported,
                                binding.as_ref(),
                                limits.timeout,
                            );
                            buffer_live_during_replay(
                                recovery,
                                &mut sub,
                                &topic,
                                &mut pending_live,
                                MAX_PENDING_LIVE_BYTES,
                                &mut shutdown_rx,
                                binding.as_ref(),
                            )
                            .await
                        } else {
                            debug!(source = %source_id, rank = dp_rank, endpoint_configured = false,
                                "kv_index replay unavailable: no endpoint configured");
                            ReplayOutcome::Failed
                        };
                        #[cfg(test)]
                        if let Some(barrier) = binding.as_ref().and_then(|binding| {
                            if matches!(result, ReplayOutcome::Complete(_)) {
                                binding.replay_commit_barrier.as_ref()
                            } else {
                                binding.replay_cleanup_barrier.as_ref()
                            }
                        }) {
                            barrier.wait().await;
                            barrier.wait().await;
                        }
                        let failure = if !matches!(result, ReplayOutcome::Complete(_)) {
                            Some(SourceRecoveryReason::ReplayFailed)
                        } else {
                            None
                        };
                        let current = with_binding(binding.as_ref(), |_| {
                            indexer.finish_replay_if_current(
                                &source_id,
                                dp_rank,
                                attempt,
                                failure.map(|reason| {
                                    if result == ReplayOutcome::Failed {
                                        retries.failure_reason(reason)
                                    } else {
                                        reason
                                    }
                                }),
                            )
                        });
                        let outcome = if current != Some(true) || result == ReplayOutcome::Stopped {
                            "superseded"
                        } else if replay_endpoint.is_none() {
                            "unavailable"
                        } else if !matches!(result, ReplayOutcome::Complete(_)) {
                            "failed"
                        } else {
                            "success"
                        };
                        RouterMetrics::record_kv_replay_outcome(outcome);
                        if current.is_none() || result == ReplayOutcome::Stopped {
                            break;
                        }
                        if let (Some(true), ReplayOutcome::Complete(high)) = (current, result) {
                            last_seq = high;
                            retries.reset();
                            debug!(
                                source = %source_id,
                                rank = dp_rank,
                                seq = high,
                                "kv_index replay sequence advanced"
                            );
                            emit(
                                &signals,
                                IngestionSignal::ReplayApplied {
                                    source: source_id.clone(),
                                    replay_seq: high,
                                },
                            );
                            tokio::task::yield_now().await;
                            continue;
                        }
                        if current == Some(true) {
                            // A local queue limit still clears partial recovery, but
                            // must not exhaust retries against the publisher.
                            if result == ReplayOutcome::Failed {
                                retries.failed(Instant::now());
                            }
                            retries.note_loss();
                        }
                    }
                    if let Some(batch) = batch {
                        // Strict mode requires an authoritative boundary;
                        // compatibility mode exposes only fresh observations.
                        let applied = with_binding(binding.as_ref(), |strict| {
                            match batch_rank(batch.data_parallel_rank, dp_rank, strict) {
                                BatchRank::Owned => {
                                    let clear_barrier = batch
                                        .events
                                        .iter()
                                        .any(|event| matches!(event, KVEvent::AllBlocksCleared(_)));
                                    if let Err(rejection) = apply_batch(
                                        &indexer,
                                        &source_id,
                                        dp_rank,
                                        incarnation,
                                        &batch,
                                        expected_block_size,
                                    ) {
                                        report_full_attention_rejection(
                                            rejection,
                                            &source_id,
                                            dp_rank,
                                            &mut full_attention_rejection_reported,
                                        );
                                        retries.invalidate_once(
                                            &indexer,
                                            &source_id,
                                            dp_rank,
                                            SourceRecoveryReason::ValidationFailed,
                                        );
                                        return false;
                                    }
                                    retries.loss_recorded = false;
                                    let initial_boundary = last_seq == -1 && seq == 0;
                                    let boundary = if initial_boundary {
                                        Some(RecoveryBoundary::Initial)
                                    } else if clear_barrier {
                                        Some(RecoveryBoundary::Clear)
                                    } else {
                                        (!strict_recovery).then_some(RecoveryBoundary::Observed)
                                    };
                                    if let Some(boundary) = boundary {
                                        if boundary != RecoveryBoundary::Observed {
                                            retries.reset();
                                        }
                                        indexer.resume_boundary(
                                            &source_id,
                                            dp_rank,
                                            boundary,
                                            strict_recovery,
                                        );
                                    }
                                    true
                                }
                                BatchRank::Foreign => {
                                    let initial_boundary = last_seq == -1 && seq == 0;
                                    if initial_boundary {
                                        retries.reset();
                                        indexer.resume_boundary(
                                            &source_id,
                                            dp_rank,
                                            RecoveryBoundary::Initial,
                                            strict_recovery,
                                        );
                                    }
                                    true
                                }
                                BatchRank::Rejected => {
                                    retries.invalidate_once(
                                        &indexer,
                                        &source_id,
                                        dp_rank,
                                        SourceRecoveryReason::ValidationFailed,
                                    );
                                    false
                                }
                            }
                        });
                        match applied {
                            None => break,
                            Some(false) => {
                                tokio::task::yield_now().await;
                                continue;
                            }
                            Some(true) => {}
                        }
                        last_seq = seq;
                        emit(
                            &signals,
                            IngestionSignal::Advance {
                                source: source_id.clone(),
                                last_seq: seq,
                            },
                        );
                        debug!(source = %source_id, rank = dp_rank, seq, "kv_index live sequence advanced");
                    }
                    // A sustained publisher starves EAGAIN; yield so this task
                    // can't monopolize the runtime worker.
                    tokio::task::yield_now().await;
                }
                Err(zmq::Error::EAGAIN) => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Err(e) => {
                    warn!("kv_index sub {} recv: {}", source_id, e);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    });

    SubscriberHandle { shutdown_tx }
}

// Recovery keeps draining SUB into a bounded queue, without applying live
// mutations over an in-progress rebuild. The original ingestion loop later
// consumes this queue in order, deduplicating through the replay watermark.
const MAX_PENDING_LIVE_BYTES: usize = 64 * 1024 * 1024;
const MAX_PENDING_LIVE_FRAMES: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplayOutcome {
    Complete(i64),
    Failed,
    BufferFull,
    Stopped,
}

async fn buffer_live_during_replay(
    recovery: impl Future<Output = Option<i64>>,
    sub: &mut zmq::Socket,
    topic: &str,
    pending: &mut VecDeque<Vec<Vec<u8>>>,
    byte_limit: usize,
    shutdown: &mut broadcast::Receiver<()>,
    binding: Option<&SubscriberBinding>,
) -> ReplayOutcome {
    tokio::pin!(recovery);
    let mut bytes: usize = pending.iter().flatten().map(Vec::len).sum();
    loop {
        if binding.is_some_and(|binding| !binding.lease.is_active()) {
            return ReplayOutcome::Stopped;
        }
        tokio::select! {
            biased;
            _ = shutdown.recv() => return ReplayOutcome::Stopped,
            result = &mut recovery => return match result {
                Some(high) => ReplayOutcome::Complete(high),
                None => ReplayOutcome::Failed,
            },
            _ = tokio::time::sleep(Duration::from_millis(1)) => {
                // Bound work per poll so replay can progress under a busy publisher.
                for _ in 0..64 {
                    if pending.len() >= MAX_PENDING_LIVE_FRAMES || bytes >= byte_limit {
                        warn!("kv_index live recovery buffer full; aborting replay before resuming ingestion");
                        return ReplayOutcome::BufferFull;
                    }
                    match sub.recv_multipart(zmq::DONTWAIT) {
                        Ok(frames) => {
                            if frames.first().is_some_and(|frame| frame.as_slice() != topic.as_bytes()) {
                                continue;
                            }
                            bytes = bytes.saturating_add(frames.iter().map(Vec::len).sum::<usize>());
                            // Keep even the frame crossing the budget: it has already
                            // left ZMQ and must not be silently lost. Memory can exceed
                            // the queue budget by this one transport frame.
                            pending.push_back(frames);
                            if bytes >= byte_limit {
                                warn!("kv_index live recovery byte budget reached; aborting replay");
                                return ReplayOutcome::BufferFull;
                            }
                        }
                        Err(zmq::Error::EAGAIN) => break,
                        Err(error) => {
                            warn!(%error, "kv_index live receive failed during replay");
                            return ReplayOutcome::Failed;
                        }
                    }
                }
            }
        }
    }
}

struct ReplayRetryBudget {
    failures: u32,
    loss_recorded: bool,
    next_attempt: Option<Instant>,
    retry_base: Duration,
}

impl Default for ReplayRetryBudget {
    fn default() -> Self {
        Self::new(ReplayLimits::default().retry_base)
    }
}

impl ReplayRetryBudget {
    const MAX_FAILURES: u32 = 3;

    fn new(retry_base: Duration) -> Self {
        Self {
            failures: 0,
            loss_recorded: false,
            next_attempt: None,
            retry_base,
        }
    }

    fn note_loss(&mut self) {
        if !self.loss_recorded {
            self.loss_recorded = true;
            crate::kv_index::record_signal_loss();
        }
    }

    fn invalidate_once(
        &mut self,
        indexer: &KvBlockIndexer,
        source: &SourceId,
        rank: u32,
        reason: SourceRecoveryReason,
    ) {
        if !self.loss_recorded {
            let reason = if self.failures >= Self::MAX_FAILURES {
                SourceRecoveryReason::RetryExhausted
            } else {
                reason
            };
            indexer.invalidate_source_for(source, rank, reason);
            self.note_loss();
        }
    }

    fn failure_reason(&self, reason: SourceRecoveryReason) -> SourceRecoveryReason {
        if self.failures + 1 >= Self::MAX_FAILURES {
            SourceRecoveryReason::RetryExhausted
        } else {
            reason
        }
    }

    fn allowed(&self, now: Instant) -> bool {
        self.failures < Self::MAX_FAILURES
            && self.next_attempt.is_none_or(|deadline| now >= deadline)
    }

    fn failed(&mut self, now: Instant) {
        self.failures = self.failures.saturating_add(1);
        self.next_attempt = (self.failures < Self::MAX_FAILURES)
            .then(|| now + self.retry_base * (1 << self.failures));
    }

    fn reset(&mut self) {
        *self = Self::new(self.retry_base);
    }
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
        if let Err(e) = tx.try_send(signal) {
            debug!("kv_index signal dropped: {}", e);
        }
    }
}

fn read_seq(bytes: &[u8]) -> i64 {
    if bytes.len() >= 8 {
        i64::from_be_bytes(bytes[..8].try_into().unwrap_or([0u8; 8]))
    } else {
        -1
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

/// Build the owner an event's blocks belong to. A non-empty `ownership` field
/// tags the block as a shared-pool copy (`CacheOwner`); otherwise the worker's
/// own residency (`Worker`).
fn event_owner(
    source_id: &SourceId,
    dp_rank: u32,
    incarnation: u64,
    ownership: Option<&str>,
) -> ResidencyOwner {
    match ownership {
        Some(pool) if !pool.is_empty() => ResidencyOwner::CacheOwner {
            pool_id: Arc::from(pool),
            source: source_id.clone(),
            dp_rank,
        },
        _ => ResidencyOwner::Worker {
            source: source_id.clone(),
            dp_rank,
            incarnation,
        },
    }
}

/// Tier from the owner domain: CacheOwner → External, Worker → from `medium`.
fn event_tier(owner: &ResidencyOwner, medium: Option<&str>) -> StorageTier {
    match owner {
        ResidencyOwner::CacheOwner { .. } => StorageTier::External,
        ResidencyOwner::Worker { .. } => StorageTier::from_medium(medium),
    }
}

fn wire_hash(hash: &ExternalBlockHash) -> Option<SequenceHash> {
    match hash {
        ExternalBlockHash::Bytes(bytes) => bytes.as_slice().try_into().ok(),
        ExternalBlockHash::Int(_) => None,
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

fn apply_batch(
    indexer: &KvBlockIndexer,
    source_id: &SourceId,
    dp_rank: u32,
    incarnation: u64,
    batch: &KVEventBatch,
    expected_block_size: Option<u32>,
) -> Result<(), FullAttentionBatchRejection> {
    use FullAttentionRejectionReason::{Capacity, HashContract, Layout};
    let expected = expected_block_size.or_else(|| {
        batch.events.iter().find_map(|event| match event {
            KVEvent::BlockStored(store) => Some(store.block_size),
            _ => None,
        })
    });
    if let Some(expected) = expected {
        validate_full_attention_batch(batch, expected)?;
    }

    enum Change {
        Store(ResidencyOwner, StorageTier, Vec<SequenceHash>),
        Remove(ResidencyOwner, StorageTier, Vec<SequenceHash>),
        ClearWorker,
    }

    let rank = batch.data_parallel_rank.unwrap_or(dp_rank);
    let changes = batch
        .events
        .iter()
        .enumerate()
        .map(|(event_index, event)| {
            let reject = |reason| FullAttentionBatchRejection {
                event_index,
                reason,
            };
            let (group, medium, ownership, hashes) = match event {
                KVEvent::BlockStored(store) => {
                    let generator = if let Some(parent) = &store.parent_block_hash {
                        let parent = wire_hash(parent).ok_or_else(|| reject(HashContract))?;
                        BlockKeyGenerator::from_root_hash(store.block_size as usize, parent)
                    } else {
                        ordinary_block_keys(store.block_size).ok_or_else(|| reject(HashContract))?
                    };
                    let first_hash = store.block_hashes.first().and_then(wire_hash);
                    if first_hash.is_none()
                        || first_hash
                            != generator.first_block(&store.token_ids, store.lora_name.as_deref())
                    {
                        return Err(reject(HashContract));
                    }
                    (
                        store.group_idx,
                        &store.medium,
                        &store.ownership,
                        &store.block_hashes,
                    )
                }
                KVEvent::BlockRemoved(remove) => (
                    remove.group_idx,
                    &remove.medium,
                    &remove.ownership,
                    &remove.block_hashes,
                ),
                KVEvent::AllBlocksCleared(_) => return Ok(Change::ClearWorker),
            };
            if group.is_some_and(|group| group != 0) {
                return Err(reject(Layout));
            }
            let owner = event_owner(source_id, rank, incarnation, ownership.as_deref());
            let tier = event_tier(&owner, medium.as_deref());
            let hashes = hashes
                .iter()
                .map(wire_hash)
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| reject(HashContract))?;
            Ok(match event {
                KVEvent::BlockStored(_) => Change::Store(owner, tier, hashes),
                _ => Change::Remove(owner, tier, hashes),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let touched = PageChanges::from_batches(
        changes.iter().map(|event| match event {
            Change::Store(_, _, hashes) | Change::Remove(_, _, hashes) => Some(hashes.as_slice()),
            Change::ClearWorker => None,
        }),
        usize::MAX,
    );
    indexer.write_batch(source_id, rank, touched, |write| {
        for (event_index, change) in changes.into_iter().enumerate() {
            match change {
                Change::Store(owner, tier, hashes) => write
                    .store_hash_pages(
                        &owner,
                        0,
                        tier,
                        None,
                        &hashes,
                        tier == StorageTier::Device,
                        usize::MAX,
                    )
                    .map_err(|_| FullAttentionBatchRejection {
                        event_index,
                        reason: Capacity,
                    })?,
                Change::Remove(owner, tier, hashes) => {
                    write.remove_hash_pages(&owner, 0, tier, &hashes, tier == StorageTier::Device)
                }
                Change::ClearWorker => {
                    write.clear_pages(|owner| matches!(owner, ResidencyOwner::Worker { .. }))
                }
            }
        }
        Ok(())
    })
}

/// Backfill `(last_seq, ∞)` from the replay ROUTER via a DEALER socket.
/// Frames with `seq <= last_seq` are skipped — deduped so an out-of-order
/// re-send can't re-apply a stale `remove` over a freshly re-`store`d block.
/// Returns success only after every sequence through `through_seq` was decoded
/// and applied. The caller suspends credit during replay and clears partial
/// observations on failure.
#[allow(clippy::too_many_arguments)]
async fn replay(
    endpoint: &str,
    last_seq: i64,
    through_seq: i64,
    attempt: u64,
    ctx: &zmq::Context,
    indexer: &Arc<KvBlockIndexer>,
    source_id: &SourceId,
    dp_rank: u32,
    incarnation: u64,
    signals: Option<mpsc::Sender<IngestionSignal>>,
    expected_block_size: Option<u32>,
    topic: &str,
    full_attention_rejection_reported: &mut bool,
    binding: Option<&SubscriberBinding>,
    timeout: Duration,
) -> Option<i64> {
    let dealer = match ctx.socket(zmq::DEALER) {
        Ok(s) => s,
        Err(e) => {
            warn!("kv_index replay socket for {}: {}", source_id, e);
            return None;
        }
    };
    if let Err(e) = dealer.connect(endpoint) {
        warn!("kv_index replay connect {}: {}", endpoint, e);
        return None;
    }
    // Request everything strictly past the live high-water.
    let start_bytes = ((last_seq + 1) as u64).to_be_bytes();
    // DEALER sends [empty delim, start_seq].
    if dealer
        .send_multipart([vec![], start_bytes.to_vec()], zmq::DONTWAIT)
        .is_err()
    {
        warn!("kv_index replay {} send failed", source_id);
        return None;
    }
    // ponytail: bounded poll loop. Replay is brief and rare; a hard deadline
    // guards against a silent router. Swap for backoff if replay grows large.
    let deadline = tokio::time::Instant::now() + timeout;
    let mut high = last_seq;
    loop {
        if binding.is_some_and(|binding| !binding.lease.is_active()) {
            return None;
        }
        if tokio::time::Instant::now() >= deadline {
            warn!("kv_index replay {} timed out", source_id);
            break;
        }
        match dealer.recv_multipart(zmq::DONTWAIT) {
            Ok(frames) => {
                // DEALER reply: [empty, topic, seq, payload]; seq is 2nd-from-last.
                if frames.len() != 4 || !frames[0].is_empty() || frames[2].len() != 8 {
                    break;
                }
                let seq = read_seq(&frames[frames.len() - 2]);
                if seq == END_SEQ {
                    // vLLM sends an empty topic and payload for this control frame.
                    if !frames[1].is_empty() || !frames[3].is_empty() {
                        return None;
                    }
                    break;
                }
                if frames[1].as_slice() != topic.as_bytes() || seq < 0 {
                    return None;
                }
                if seq <= high {
                    continue;
                }
                if seq != high + 1 {
                    warn!(
                        "kv_index replay {}: noncontiguous seq {}, expected {}",
                        source_id,
                        seq,
                        high + 1
                    );
                    return None;
                }
                let batch = decode(&frames[frames.len() - 1])?;
                let accepted = with_binding(binding, |strict| {
                    indexer.with_current_replay(source_id, dp_rank, attempt, || {
                        match batch_rank(batch.data_parallel_rank, dp_rank, strict) {
                            BatchRank::Owned => {
                                if let Err(rejection) = apply_batch(
                                    indexer,
                                    source_id,
                                    dp_rank,
                                    incarnation,
                                    &batch,
                                    expected_block_size,
                                ) {
                                    report_full_attention_rejection(
                                        rejection,
                                        source_id,
                                        dp_rank,
                                        full_attention_rejection_reported,
                                    );
                                    false
                                } else {
                                    true
                                }
                            }
                            BatchRank::Foreign => true,
                            BatchRank::Rejected => false,
                        }
                    })
                })
                .flatten()?;
                if !accepted {
                    return None;
                }
                high = seq;
                emit(
                    &signals,
                    IngestionSignal::Advance {
                        source: source_id.clone(),
                        last_seq: seq,
                    },
                );
                debug!(source = %source_id, rank = dp_rank, seq, "kv_index replay sequence advanced");
                if high >= through_seq {
                    return Some(high);
                }
                tokio::task::yield_now().await;
            }
            Err(zmq::Error::EAGAIN) => {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            Err(e) => {
                warn!("kv_index replay {} recv: {}", source_id, e);
                break;
            }
        }
    }
    None
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use parking_lot::Mutex;

    use super::*;
    use crate::kv_index::admit::FullAttentionRejectionReason;
    use crate::kv_index::indexer::MatchQuery;
    use crate::kv_index::types::Locality;
    use crate::kv_index::wire::{AllBlocksCleared, BlockRemoved, BlockStored};

    fn wire_hashes_for(tokens: &[u32], block_size: u32) -> Vec<ExternalBlockHash> {
        crate::kv_index::ordinary_block_hashes(tokens, block_size)
            .into_iter()
            .map(|hash| ExternalBlockHash::Bytes(hash.to_vec()))
            .collect()
    }

    fn stored(
        block_count: usize,
        token_ids: Vec<u32>,
        medium: Option<&str>,
        ownership: Option<&str>,
    ) -> BlockStored {
        BlockStored {
            extra_keys: None,
            block_hashes: wire_hashes_for(&token_ids, 4)
                .into_iter()
                .take(block_count)
                .collect(),
            parent_block_hash: None,
            token_ids,
            block_size: 4,
            lora_id: None,
            medium: medium.map(String::from),
            lora_name: None,
            group_idx: Some(0),
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            locality: None,
            ownership: ownership.map(String::from),
            session_id: None,
        }
    }

    #[test]
    fn unmodeled_extra_keys_never_alias_plain_or_named_lora_credit() {
        for (name, lora_id, extra, expected) in [
            (None, None, None, 1),
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
                let mut event = stored(1, vec![1, 2, 3, 4], Some("GPU"), None);
                let keys = ordinary_block_keys(4).unwrap();
                let hashes = match name {
                    Some(name) => keys.generate_block_keys_for_adapter_name(&event.token_ids, name),
                    None => keys.generate_block_keys(&event.token_ids),
                };
                event.block_hashes = hashes
                    .iter()
                    .map(|hash| ExternalBlockHash::Bytes(hash.to_vec()))
                    .collect();
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
                        &KVEventBatch {
                            ts: 1.0,
                            events: vec![KVEvent::BlockStored(event)],
                            data_parallel_rank: Some(0),
                        },
                        full_attention_only.then_some(4),
                    )
                    .is_ok(),
                    expected > 0
                );
                let query = MatchQuery {
                    group_idx: 0,
                    hashes,
                    tiers_of_interest: vec![StorageTier::Device],
                };
                assert_eq!(index.find_matches(&query).len(), expected);
            }
        }
    }

    #[test]
    fn apply_batch_routes_ownership_to_tier_and_rank_from_batch() {
        let idx = Arc::new(KvBlockIndexer::new());
        let batch = KVEventBatch {
            ts: 1.0,
            events: vec![
                // ownership set → CacheOwner → External (not GPU/Device).
                KVEvent::BlockStored(stored(1, vec![1, 2, 3, 4], Some("GPU"), Some("mooncake"))),
                // ownership None, medium CPU → Worker → HostPinned.
                KVEvent::BlockStored(stored(1, vec![5, 6, 7, 8], Some("CPU"), None)),
            ],
            data_parallel_rank: Some(1),
        };
        apply_batch(&idx, &SourceId::from("w0"), 0, 0, &batch, None).unwrap();

        let pool_locals = crate::kv_index::ordinary_block_hashes(&[1, 2, 3, 4], 4);
        let pool_res = idx.find_matches(&MatchQuery {
            group_idx: 0,
            hashes: pool_locals,
            tiers_of_interest: vec![StorageTier::External],
        });
        assert_eq!(pool_res.len(), 1);
        assert_eq!(pool_res[0].tier, StorageTier::External);
        assert_eq!(pool_res[0].locality, Locality::Remote);
        assert_eq!(&*pool_res[0].target.instance_id, "w0");
        assert_eq!(pool_res[0].target.dp_rank, 1);

        let wkr_locals = crate::kv_index::ordinary_block_hashes(&[5, 6, 7, 8], 4);
        let wkr_res = idx.find_matches(&MatchQuery {
            group_idx: 0,
            hashes: wkr_locals,
            tiers_of_interest: vec![StorageTier::HostPinned],
        });
        assert_eq!(wkr_res.len(), 1);
        assert_eq!(wkr_res[0].tier, StorageTier::HostPinned);
        assert_eq!(wkr_res[0].locality, Locality::Local);
        assert_eq!(&*wkr_res[0].target.instance_id, "w0");
        assert_eq!(wkr_res[0].target.dp_rank, 1);
    }

    #[test]
    fn full_attention_subscribers_preserve_residency_boundaries() {
        let source = SourceId::from("dense");
        let valid = stored(1, vec![1, 2, 3, 4], Some("GPU"), None);
        let generic = KvBlockIndexer::new();
        let mut remote = valid.clone();
        remote.locality = Some("REMOTE".into());
        let batch = KVEventBatch {
            ts: 0.0,
            events: vec![KVEvent::BlockStored(remote)],
            data_parallel_rank: Some(0),
        };
        assert!(apply_batch(&generic, &source, 0, 0, &batch, None).is_err());
        assert!(hits(&generic, &[1, 2, 3, 4]).is_empty());
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
                hashes: crate::kv_index::ordinary_block_hashes(&[1, 2, 3, 4], 4),
                tiers_of_interest: vec![tier],
            });
            assert_eq!(matches.len(), 1);
            assert_eq!(matches[0].tier, tier);
            assert_eq!(matches[0].target.dp_rank, 3);
        }
    }

    #[test]
    fn read_seq_big_endian_signed() {
        assert_eq!(read_seq(&(-1i64).to_be_bytes()), -1);
        assert_eq!(read_seq(&42i64.to_be_bytes()), 42);
    }

    #[test]
    fn read_seq_short_input_is_sentinel() {
        assert_eq!(read_seq(&[0u8, 1]), -1);
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
                strict_recovery: true,
            },
            indexer.clone(),
            Some(sig_tx),
            ctx.clone(),
        );

        let local = crate::kv_index::ordinary_block_hashes(&[1, 2, 3, 4], 4)[0];
        let dev = |hashes: &[SequenceHash]| MatchQuery {
            group_idx: 0,
            hashes: hashes.to_vec(),
            tiers_of_interest: vec![StorageTier::Device],
        };

        // 1) STORE — one block (seq hash 100) on w0/rank0, GPU/Device.
        publish_until_advanced(
            &pub_sock,
            &mut sig_rx,
            b"kv",
            0,
            1.0,
            &[KVEvent::BlockStored(BlockStored {
                extra_keys: None,
                block_hashes: wire_hashes_for(&[1, 2, 3, 4], 4),
                parent_block_hash: None,
                token_ids: vec![1, 2, 3, 4],
                block_size: 4,
                lora_id: None,
                medium: Some("GPU".into()),
                lora_name: None,
                group_idx: Some(0),
                kv_cache_spec_kind: None,
                kv_cache_spec_sliding_window: None,
                locality: None,
                ownership: None,
                session_id: None,
            })],
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
            1,
            2.0,
            &[KVEvent::BlockRemoved(BlockRemoved {
                block_hashes: wire_hashes_for(&[1, 2, 3, 4], 4),
                medium: Some("GPU".into()),
                group_idx: Some(0),
                locality: None,
                ownership: None,
            })],
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
            2,
            3.0,
            &[KVEvent::AllBlocksCleared(AllBlocksCleared {})],
            Some(0),
        )
        .await;
        assert!(indexer.find_matches(&dev(&[local])).is_empty());

        handle.shutdown();
    }

    #[tokio::test]
    async fn compatibility_subscriber_preserves_nonzero_rank_and_nondevice_tiers() {
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
                strict_recovery: false,
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
                KVEvent::BlockStored(stored(1, host_tokens.to_vec(), Some("CPU"), None)),
                KVEvent::BlockStored(stored(
                    1,
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
                hashes: crate::kv_index::ordinary_block_hashes(&tokens, 4),
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
        let payload = rmp_serde::to_vec_named(&(ts, events, &rank)).unwrap();
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
        fixture_with_block_size(with_replay, None)
    }

    fn fixture_with_block_size(with_replay: bool, expected_block_size: Option<u32>) -> Fixture {
        fixture_with_binding(with_replay, expected_block_size, None)
    }

    fn fixture_with_binding(
        with_replay: bool,
        expected_block_size: Option<u32>,
        binding: Option<SubscriberBinding>,
    ) -> Fixture {
        fixture_with_recovery(with_replay, expected_block_size, binding, true)
    }

    fn fixture_with_recovery(
        with_replay: bool,
        expected_block_size: Option<u32>,
        binding: Option<SubscriberBinding>,
        strict_recovery: bool,
    ) -> Fixture {
        fixture_with_receive_limit(
            with_replay,
            expected_block_size,
            binding,
            strict_recovery,
            None,
            ReplayLimits::default(),
        )
    }

    fn fixture_with_receive_limit(
        with_replay: bool,
        expected_block_size: Option<u32>,
        binding: Option<SubscriberBinding>,
        strict_recovery: bool,
        hwm: Option<i32>,
        limits: ReplayLimits,
    ) -> Fixture {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let tag = NEXT.fetch_add(1, Ordering::Relaxed);
        let ctx = Arc::new(zmq::Context::new());
        let publisher = ctx.socket(zmq::PUB).unwrap();
        if let Some(hwm) = hwm {
            publisher.set_sndhwm(hwm).unwrap();
        }
        let pub_endpoint = format!("inproc://kv-loss-pub-{tag}");
        publisher.bind(&pub_endpoint).unwrap();
        let replay_endpoint = format!("inproc://kv-loss-replay-{tag}");
        let replay_server = with_replay.then(|| {
            let socket = ctx.socket(zmq::ROUTER).unwrap();
            socket.bind(&replay_endpoint).unwrap();
            socket
        });
        let indexer = Arc::new(KvBlockIndexer::new());
        let (tx, signals) = mpsc::channel(128);
        let source = EventSource {
            source: SourceId::from("w0"),
            dp_rank: 0,
            pub_endpoint,
            replay_endpoint: with_replay.then_some(replay_endpoint),
            topic: "kv".into(),
            hwm,
            strict_recovery,
        };
        let handle = spawn_inner(
            source,
            indexer.clone(),
            Some(tx),
            ctx.clone(),
            expected_block_size,
            binding,
            limits,
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

    fn store_event(tokens: &[u32]) -> KVEvent {
        KVEvent::BlockStored(stored(1, tokens.to_vec(), Some("GPU"), None))
    }

    fn session_store_event(tokens: &[u32], session: &str) -> KVEvent {
        let KVEvent::BlockStored(mut event) = store_event(tokens) else {
            unreachable!()
        };
        event.session_id = Some(session.into());
        KVEvent::BlockStored(event)
    }

    fn hits(index: &KvBlockIndexer, tokens: &[u32]) -> Vec<crate::kv_index::TierMatch> {
        index.find_matches(&MatchQuery {
            group_idx: 0,
            hashes: crate::kv_index::ordinary_block_hashes(tokens, 4),
            tiers_of_interest: vec![StorageTier::Device],
        })
    }

    #[tokio::test]
    async fn bound_external_invalidation_requires_clear_barrier() {
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
                strict_recovery: true,
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
            &[store_event(&old)],
            Some(0),
        )
        .await;
        let generation = indexer.source_generation_for_test();
        publish_until_advanced(
            &publisher,
            &mut signals,
            b"kv",
            1,
            1.0,
            &[store_event(&current)],
            Some(0),
        )
        .await;
        assert_eq!(indexer.source_generation_for_test(), generation);
        for tokens in [&old, &current] {
            let found = hits(&indexer, tokens);
            assert_eq!(found.len(), 1);
            assert_eq!(found[0].segments, [(0, 1)]);
        }

        // External promotion/invalidation removes the source's authority.
        // A later ordinary Store cannot re-establish the missing history.
        {
            let _guard = strict_rank.lock();
            indexer.invalidate_source(&SourceId::from("w0"), 0);
        }
        assert_eq!(indexer.source_generation_for_test(), generation + 1);
        assert!(hits(&indexer, &current).is_empty());
        publish_until_advanced(
            &publisher,
            &mut signals,
            b"kv",
            2,
            2.0,
            &[store_event(&fresh)],
            Some(0),
        )
        .await;
        assert_eq!(indexer.source_generation_for_test(), generation + 1);
        assert!(hits(&indexer, &old).is_empty());
        assert!(hits(&indexer, &current).is_empty());
        assert!(hits(&indexer, &fresh).is_empty());
        publish_until_advanced(
            &publisher,
            &mut signals,
            b"kv",
            3,
            3.0,
            &[
                KVEvent::AllBlocksCleared(AllBlocksCleared {}),
                store_event(&fresh),
            ],
            Some(0),
        )
        .await;
        assert_eq!(indexer.source_generation_for_test(), generation + 2);
        assert_eq!(hits(&indexer, &fresh)[0].segments, [(0, 1)]);
        publish_until_advanced(&publisher, &mut signals, b"kv", 4, 4.0, &[], Some(0)).await;
        assert_eq!(indexer.source_generation_for_test(), generation + 2);
        assert_eq!(hits(&indexer, &fresh)[0].segments, [(0, 1)]);
        handle.shutdown();
    }

    #[tokio::test]
    async fn bound_foreign_live_frames_advance_without_mutation_or_resume() {
        let strict_rank = Arc::new(Mutex::new(true));
        let mut f = fixture_with_binding(
            false,
            Some(4),
            Some(SubscriberBinding {
                lease: SourceLease {
                    strict_rank: strict_rank.clone(),
                    active: Arc::new(AtomicBool::new(true)),
                },
                replay_cleanup_barrier: None,
                replay_commit_barrier: None,
            }),
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
            &[store_event(&old)],
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
            &[store_event(&foreign)],
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
            &[store_event(&fresh)],
            Some(0),
        )
        .await;
        assert_eq!(advances, [2], "foreign sequence created an artificial gap");
        assert!(hits(&f.indexer, &old).is_empty());
        assert!(hits(&f.indexer, &fresh).is_empty());

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
            &[store_event(&fresh)],
            Some(0),
        )
        .await;
        assert_eq!(advances, [5]);
        assert!(hits(&f.indexer, &fresh).is_empty());
        publish_until_advanced(
            &f.publisher,
            &mut f.signals,
            b"kv",
            6,
            6.0,
            &[
                KVEvent::AllBlocksCleared(AllBlocksCleared {}),
                store_event(&fresh),
            ],
            Some(0),
        )
        .await;
        assert_eq!(hits(&f.indexer, &fresh).len(), 1);
    }

    #[tokio::test]
    async fn bound_replay_tracks_foreign_sequences_and_rejects_missing_rank() {
        for foreign_rank in [Some(1), None] {
            let mut f = fixture_with_binding(
                true,
                Some(4),
                Some(SubscriberBinding {
                    lease: SourceLease {
                        strict_rank: Arc::new(Mutex::new(true)),
                        active: Arc::new(AtomicBool::new(true)),
                    },
                    replay_cleanup_barrier: None,
                    replay_commit_barrier: None,
                }),
            );
            let old = [1, 2, 3, 4];
            let recovered = [5, 6, 7, 8];
            let fresh = [9, 10, 11, 12];
            let foreign = [13, 14, 15, 16];
            publish_until_advanced(
                &f.publisher,
                &mut f.signals,
                b"kv",
                0,
                0.0,
                &[store_event(&old)],
                Some(0),
            )
            .await;
            assert_eq!(hits(&f.indexer, &old).len(), 1);
            let live = [store_event(&fresh)];
            let replies = [
                (
                    1,
                    b"kv".as_slice(),
                    batch_payload(&[store_event(&recovered)], 0),
                ),
                (
                    2,
                    b"kv".as_slice(),
                    rmp_serde::to_vec_named(&(2.0, [store_event(&foreign)], foreign_rank)).unwrap(),
                ),
                (3, b"kv".as_slice(), batch_payload(&live, 0)),
            ];
            let query = MatchQuery {
                group_idx: 0,
                hashes: crate::kv_index::ordinary_block_hashes(&old, 4),
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
            assert_eq!(
                hits(&f.indexer, &fresh).len(),
                usize::from(foreign_rank.is_some())
            );
            assert!(hits(&f.indexer, &foreign).is_empty());
        }
    }

    #[tokio::test]
    async fn retired_replay_cannot_clear_or_mutate_a_replacement_subscription() {
        let strict_rank = Arc::new(Mutex::new(false));
        let active = Arc::new(AtomicBool::new(true));
        let cleanup_barrier = Arc::new(tokio::sync::Barrier::new(2));
        let mut f = fixture_with_binding(
            true,
            Some(4),
            Some(SubscriberBinding {
                lease: SourceLease {
                    strict_rank: strict_rank.clone(),
                    active: active.clone(),
                },
                replay_cleanup_barrier: Some(cleanup_barrier.clone()),
                replay_commit_barrier: None,
            }),
        );
        let old = [1, 2, 3, 4];
        let partial = [5, 6, 7, 8];
        let replacement = [9, 10, 11, 12];
        publish_until_advanced(
            &f.publisher,
            &mut f.signals,
            b"kv",
            0,
            0.0,
            &[store_event(&old)],
            Some(0),
        )
        .await;
        assert_eq!(hits(&f.indexer, &old).len(), 1);
        f.publisher
            .send_multipart(
                [
                    b"kv".as_slice(),
                    &3i64.to_be_bytes(),
                    &batch_payload(&[], 0),
                ],
                0,
            )
            .unwrap();
        let server = f.replay_server.as_ref().unwrap();
        let request = replay_request(server).await;
        assert!(hits(&f.indexer, &old).is_empty());
        send_replay_replies(
            server,
            &request,
            &[(
                1,
                b"kv".as_slice(),
                batch_payload(&[store_event(&partial)], 0),
            )],
        );
        loop {
            let signal = tokio::time::timeout(Duration::from_secs(5), f.signals.recv())
                .await
                .unwrap()
                .unwrap();
            if matches!(signal, IngestionSignal::Advance { last_seq: 1, .. }) {
                break;
            }
        }
        // Fail replay while its lease is still active, then hold the old task
        // immediately before cleanup until replacement credit is established.
        send_replay_replies(server, &request, &[(2, b"kv".as_slice(), vec![0xc1])]);
        tokio::time::timeout(Duration::from_secs(5), cleanup_barrier.wait())
            .await
            .unwrap();
        assert!(active.load(Ordering::Acquire));
        {
            let _guard = strict_rank.lock();
            active.store(false, Ordering::Release);
            f.indexer.forget_source(&SourceId::from("w0"), 0);
        }

        let publisher = f._ctx.socket(zmq::PUB).unwrap();
        let endpoint = format!(
            "{}-replacement",
            f.publisher.get_last_endpoint().unwrap().unwrap()
        );
        publisher.bind(&endpoint).unwrap();
        let (tx, mut signals) = mpsc::channel(64);
        let handle = spawn_bound(
            EventSource {
                source: SourceId::from("w0"),
                dp_rank: 0,
                pub_endpoint: endpoint,
                replay_endpoint: None,
                topic: "kv".into(),
                hwm: None,
                strict_recovery: true,
            },
            f.indexer.clone(),
            Some(tx),
            f._ctx.clone(),
            Some(4),
            SourceLease {
                strict_rank,
                active: Arc::new(AtomicBool::new(true)),
            },
        );
        publish_until_advanced(
            &publisher,
            &mut signals,
            b"kv",
            0,
            0.0,
            &[store_event(&replacement)],
            Some(0),
        )
        .await;
        assert_eq!(hits(&f.indexer, &replacement).len(), 1);
        // The retired lease must fence the already-pending cleanup, not merely
        // exit early before any replacement exists.
        tokio::time::timeout(Duration::from_secs(5), cleanup_barrier.wait())
            .await
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(5), f.signals.recv())
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(hits(&f.indexer, &replacement).len(), 1);
        assert!(hits(&f.indexer, &old).is_empty());
        assert!(hits(&f.indexer, &partial).is_empty());
        handle.shutdown();
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
            &[store_event(&tokens)],
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
                    events: vec![store_event(&tokens)],
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
            &[store_event(&tokens)],
            Some(0),
        )
        .await;
        assert_eq!(hits(&f.indexer, &tokens).len(), 2);
        publish_until_advanced(
            &f.publisher,
            &mut f.signals,
            b"kv",
            4,
            4.0,
            &[
                KVEvent::AllBlocksCleared(AllBlocksCleared {}),
                store_event(&tokens),
            ],
            Some(0),
        )
        .await;
        assert_eq!(hits(&f.indexer, &tokens).len(), 3);
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
            &[
                KVEvent::AllBlocksCleared(AllBlocksCleared {}),
                store_event(&tokens),
            ],
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
            &[
                KVEvent::AllBlocksCleared(AllBlocksCleared {}),
                store_event(&tokens),
            ],
            Some(0),
        )
        .await;
        assert_eq!(hits(&f.indexer, &tokens).len(), 1);
    }

    #[tokio::test]
    async fn live_topic_prefix_does_not_publish_or_consume_the_sequence() {
        let mut f = fixture_with_block_size(false, Some(4));
        publish_until_advanced(&f.publisher, &mut f.signals, b"kv", 0, 0.0, &[], Some(0)).await;
        let unrelated = [1, 2, 3, 4];
        let expected = [5, 6, 7, 8];
        let payload = batch_payload(&[store_event(&unrelated)], 0);
        f.publisher
            .send_multipart([b"kv-other".as_slice(), &1i64.to_be_bytes(), &payload], 0)
            .unwrap();
        publish_until_advanced(
            &f.publisher,
            &mut f.signals,
            b"kv",
            1,
            1.0,
            &[session_store_event(&expected, "session-live")],
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
        let mut event = serde_json::to_value(store_event(&tokens)).unwrap();
        event["future_namespace"] = serde_json::json!("not-plain");
        let payload = rmp_serde::to_vec(&(
            1.0,
            [
                serde_json::to_value(store_event(&[5, 6, 7, 8])).unwrap(),
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
            &[
                KVEvent::AllBlocksCleared(AllBlocksCleared {}),
                store_event(&tokens),
            ],
            Some(0),
        )
        .await;
        assert_eq!(hits(&f.indexer, &tokens).len(), 1);
    }

    fn batch_payload(events: &[KVEvent], rank: u32) -> Vec<u8> {
        rmp_serde::to_vec_named(&(1.0, events, Some(rank))).unwrap()
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
                .map(|(_, tokens)| session_store_event(&tokens, session))
                .collect();
            let batch = KVEventBatch {
                ts: seq as f64,
                events,
                data_parallel_rank: Some(0),
            };
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
            let initial = KVEventBatch {
                ts: 0.0,
                events: vec![store_event(&[1, 2, 3, 4])],
                data_parallel_rank: Some(0),
            };
            assert!(apply_batch(&index, &source, 0, 0, &initial, Some(4)).is_ok());
            let mut bad = stored(1, (20..28).collect(), Some("GPU"), None);
            bad.block_size = actual;
            let batch = KVEventBatch {
                ts: 1.0,
                events: vec![
                    store_event(&[5, 6, 7, 8]),
                    KVEvent::BlockRemoved(BlockRemoved {
                        block_hashes: wire_hashes_for(&[1, 2, 3, 4], 4),
                        medium: Some("GPU".into()),
                        group_idx: Some(0),
                        locality: None,
                        ownership: None,
                    }),
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
        let event = |_id, tokens: &[u32], size| {
            let mut value = stored(1, tokens.to_vec(), Some("GPU"), None);
            value.block_size = size;
            value.block_hashes = wire_hashes_for(tokens, size);
            KVEvent::BlockStored(value)
        };
        let query = |tokens: &[u32], block_size| MatchQuery {
            group_idx: 0,
            hashes: crate::kv_index::ordinary_block_hashes(tokens, block_size),
            tiers_of_interest: vec![StorageTier::Device],
        };
        let cases = [(16, 8), (8, 16)].map(|(expected, actual)| {
            let tokens: [Vec<u32>; 4] = [
                (0..expected).collect(),
                (100..100 + expected).collect(),
                (200..200 + actual).collect(),
                (300..300 + expected).collect(),
            ];
            let [_, partial, bad, fresh] = &tokens;
            let live = [event(400, fresh, expected)];
            let replies = [
                (
                    1,
                    b"kv".as_slice(),
                    batch_payload(&[event(200, partial, expected)], 0),
                ),
                (
                    2,
                    b"kv".as_slice(),
                    batch_payload(&[event(300, bad, actual)], 0),
                ),
                (3, b"kv".as_slice(), batch_payload(&live, 0)),
            ];
            (expected, tokens, live, replies)
        });
        for (expected, [old, partial, bad, fresh], live, replies) in cases {
            let mut f = fixture_with_block_size(true, Some(expected));
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
            let old_hits = f.indexer.find_matches(&query(&old, expected));
            assert_eq!(old_hits.len(), 1);
            assert_eq!(old_hits[0].target.instance_id.as_ref(), "w0");
            assert_eq!(old_hits[0].target.dp_rank, 0);
            let suspended_queries = [query(&old, expected)];
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
            assert!(f.indexer.find_matches(&query(&old, expected)).is_empty());
            assert!(f
                .indexer
                .find_matches(&query(&partial, expected))
                .is_empty());
            assert!(f.indexer.find_matches(&query(&bad, expected)).is_empty());
            assert!(f.indexer.find_matches(&query(&fresh, expected)).is_empty());
            publish_until_advanced(
                &f.publisher,
                &mut f.signals,
                b"kv",
                4,
                4.0,
                &[
                    KVEvent::AllBlocksCleared(AllBlocksCleared {}),
                    event(400, &fresh, expected),
                ],
                Some(0),
            )
            .await;
            assert_eq!(f.indexer.find_matches(&query(&fresh, expected)).len(), 1);
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

    // Each case owns its ZMQ fixture so failed recovery cannot contaminate the next case.
    async fn check_replay_case(case: &str) {
        let mut f = fixture_with_block_size(true, (case == "complete").then_some(4));
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
            &[store_event(&old), store_event(&keep)],
            Some(0),
        )
        .await;
        let query = |tokens: &[u32]| MatchQuery {
            group_idx: 0,
            hashes: crate::kv_index::ordinary_block_hashes(tokens, 4),
            tiers_of_interest: vec![StorageTier::Device],
        };
        for tokens in [&old, &keep] {
            let matches = f.indexer.find_matches(&query(tokens));
            assert_eq!(matches.len(), 1);
            assert_eq!(matches[0].target.instance_id.as_ref(), "w0");
            assert_eq!(matches[0].target.dp_rank, 0);
        }
        let first = batch_payload(&[session_store_event(&partial, "session-replay")], 0);
        let eviction = batch_payload(
            &[KVEvent::BlockRemoved(BlockRemoved {
                block_hashes: wire_hashes_for(&[1, 2, 3, 4], 4),
                medium: Some("GPU".into()),
                group_idx: Some(0),
                locality: None,
                ownership: None,
            })],
            0,
        );
        let third = batch_payload(&[store_event(&fresh)], 0);
        let eof = (-1, b"".as_slice(), vec![]);
        let topic = b"kv".as_slice();
        let replies = match case {
            "complete" => vec![(1, topic, first), (2, topic, eviction), (3, topic, third)],
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
        let live_events = [store_event(&fresh)];
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
            usize::from(case == "complete"),
            "only an authoritative recovery boundary restores credit: {case}"
        );
        assert_eq!(
            f.indexer.source_recovery_status(&SourceId::from("w0"), 0),
            Some(if case == "complete" {
                crate::kv_index::SourceRecoveryStatus {
                    state: crate::kv_index::SourceRecoveryState::WorkerExact,
                    reason: crate::kv_index::SourceRecoveryReason::ReplayApplied,
                }
            } else {
                crate::kv_index::SourceRecoveryStatus {
                    state: crate::kv_index::SourceRecoveryState::Fallback,
                    reason: crate::kv_index::SourceRecoveryReason::ReplayFailed,
                }
            })
        );
    }

    #[tokio::test]
    async fn continuous_replay_preserves_valid_prefix_and_applies_lost_eviction() {
        check_replay_case("complete").await;
    }

    #[tokio::test]
    async fn failed_replay_requires_full_rebuild_before_worker_credit_returns() {
        let retry_base = Duration::from_millis(25);
        let mut fixture = fixture_with_receive_limit(
            true,
            Some(4),
            None,
            true,
            None,
            ReplayLimits {
                retry_base,
                ..ReplayLimits::default()
            },
        );
        let source = SourceId::from("w0");
        let tokens = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
        ];
        let payloads: Vec<_> = tokens
            .iter()
            .map(|token_ids| batch_payload(&[store_event(token_ids)], 0))
            .collect();

        publish_until_advanced(
            &fixture.publisher,
            &mut fixture.signals,
            b"kv",
            0,
            0.0,
            &[store_event(&tokens[0])],
            Some(0),
        )
        .await;

        let first_replay = async {
            let request = replay_request(fixture.replay_server.as_ref().unwrap()).await;
            assert_eq!(read_seq(request.last().unwrap()), 1);
            send_replay_replies(
                fixture.replay_server.as_ref().unwrap(),
                &request,
                &[
                    (1, b"kv".as_slice(), payloads[1].clone()),
                    (END_SEQ, b"".as_slice(), vec![]),
                ],
            );
        };
        let seq_two_events = [store_event(&tokens[2])];
        tokio::join!(
            publish_until_advanced(
                &fixture.publisher,
                &mut fixture.signals,
                b"kv",
                2,
                2.0,
                &seq_two_events,
                Some(0),
            ),
            first_replay,
        );
        assert_eq!(
            fixture.indexer.source_recovery_status(&source, 0),
            Some(crate::kv_index::SourceRecoveryStatus {
                state: crate::kv_index::SourceRecoveryState::Fallback,
                reason: crate::kv_index::SourceRecoveryReason::ReplayFailed,
            })
        );

        tokio::time::sleep(retry_base * 2).await;
        let second_replay = async {
            let request = replay_request(fixture.replay_server.as_ref().unwrap()).await;
            assert_eq!(read_seq(request.last().unwrap()), 0);
            let replies: Vec<_> = payloads
                .iter()
                .enumerate()
                .map(|(seq, payload)| (seq as i64, b"kv".as_slice(), payload.clone()))
                .collect();
            send_replay_replies(fixture.replay_server.as_ref().unwrap(), &request, &replies);
        };
        let seq_four_events = [store_event(&tokens[4])];
        tokio::join!(
            publish_until_advanced(
                &fixture.publisher,
                &mut fixture.signals,
                b"kv",
                4,
                4.0,
                &seq_four_events,
                Some(0),
            ),
            second_replay,
        );

        assert_eq!(
            fixture.indexer.source_recovery_status(&source, 0),
            Some(crate::kv_index::SourceRecoveryStatus {
                state: crate::kv_index::SourceRecoveryState::WorkerExact,
                reason: crate::kv_index::SourceRecoveryReason::ReplayApplied,
            })
        );
        for token_ids in &tokens {
            assert_eq!(hits(&fixture.indexer, token_ids).len(), 1);
        }
    }

    #[tokio::test]
    async fn external_invalidation_requires_full_replay() {
        let mut fixture = fixture_with_block_size(true, Some(4));
        let source = SourceId::from("w0");
        let tokens = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ];
        let payloads: Vec<_> = tokens
            .iter()
            .map(|token_ids| batch_payload(&[store_event(token_ids)], 0))
            .collect();

        for (seq, token_ids) in tokens.iter().take(2).enumerate() {
            let events = [store_event(token_ids)];
            publish_until_advanced(
                &fixture.publisher,
                &mut fixture.signals,
                b"kv",
                seq as i64,
                seq as f64,
                &events,
                Some(0),
            )
            .await;
        }
        fixture
            .indexer
            .invalidate_source_for(&source, 0, SourceRecoveryReason::RankScopeChanged);

        let replay = async {
            let request = replay_request(fixture.replay_server.as_ref().unwrap()).await;
            assert_eq!(read_seq(request.last().unwrap()), 0);
            let replies: Vec<_> = payloads
                .iter()
                .enumerate()
                .map(|(seq, payload)| (seq as i64, b"kv".as_slice(), payload.clone()))
                .collect();
            send_replay_replies(fixture.replay_server.as_ref().unwrap(), &request, &replies);
        };
        let seq_three_events = [store_event(&tokens[3])];
        tokio::join!(
            publish_until_advanced(
                &fixture.publisher,
                &mut fixture.signals,
                b"kv",
                3,
                3.0,
                &seq_three_events,
                Some(0),
            ),
            replay,
        );

        assert_eq!(
            fixture.indexer.source_recovery_status(&source, 0),
            Some(crate::kv_index::SourceRecoveryStatus {
                state: crate::kv_index::SourceRecoveryState::WorkerExact,
                reason: crate::kv_index::SourceRecoveryReason::ReplayApplied,
            })
        );
        for token_ids in &tokens {
            assert_eq!(hits(&fixture.indexer, token_ids).len(), 1);
        }
    }

    #[tokio::test]
    async fn superseded_replay_cannot_write_between_frames_before_observed_resume() {
        let strict_rank = Arc::new(Mutex::new(false));
        let mut f = fixture_with_recovery(
            true,
            Some(4),
            Some(SubscriberBinding {
                lease: SourceLease {
                    strict_rank: strict_rank.clone(),
                    active: Arc::new(AtomicBool::new(true)),
                },
                replay_cleanup_barrier: None,
                replay_commit_barrier: None,
            }),
            false,
        );
        let source = SourceId::from("w0");
        let stale = [5, 6, 7, 8];
        let live = [9, 10, 11, 12];
        publish_until_advanced(
            &f.publisher,
            &mut f.signals,
            b"kv",
            0,
            0.0,
            &[store_event(&[1, 2, 3, 4])],
            Some(0),
        )
        .await;
        f.publisher
            .send_multipart(
                [
                    b"kv".as_slice(),
                    &4i64.to_be_bytes(),
                    &batch_payload(&[store_event(&live)], 0),
                ],
                0,
            )
            .unwrap();
        let server = f.replay_server.as_ref().unwrap();
        let request = replay_request(server).await;
        send_replay_replies(
            server,
            &request,
            &[(1, b"kv".as_slice(), batch_payload(&[], 0))],
        );
        async fn next_advance(rx: &mut mpsc::Receiver<IngestionSignal>) -> i64 {
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    if let IngestionSignal::Advance { last_seq, .. } = rx.recv().await.unwrap() {
                        return last_seq;
                    }
                }
            })
            .await
            .unwrap()
        }
        assert_eq!(next_advance(&mut f.signals).await, 1);
        {
            let mut rank = strict_rank.lock();
            *rank = true;
            f.indexer
                .invalidate_source_for(&source, 0, SourceRecoveryReason::RankScopeChanged);
        }
        send_replay_replies(
            server,
            &request,
            &[
                (
                    2,
                    b"kv".as_slice(),
                    batch_payload(&[store_event(&stale)], 0),
                ),
                (3, b"kv".as_slice(), vec![0xc1]),
            ],
        );
        assert_eq!(next_advance(&mut f.signals).await, 4);
        assert!(hits(&f.indexer, &stale).is_empty());
        assert_eq!(hits(&f.indexer, &live).len(), 1);
    }

    #[tokio::test]
    async fn external_invalidation_during_replay_prevents_exact_commit() {
        let strict_rank = Arc::new(Mutex::new(false));
        let commit_barrier = Arc::new(tokio::sync::Barrier::new(2));
        let mut fixture = fixture_with_binding(
            true,
            Some(4),
            Some(SubscriberBinding {
                lease: SourceLease {
                    strict_rank: strict_rank.clone(),
                    active: Arc::new(AtomicBool::new(true)),
                },
                replay_cleanup_barrier: None,
                replay_commit_barrier: Some(commit_barrier.clone()),
            }),
        );
        let source = SourceId::from("w0");
        let tokens = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];

        publish_until_advanced(
            &fixture.publisher,
            &mut fixture.signals,
            b"kv",
            0,
            0.0,
            &[store_event(&tokens[0])],
            Some(0),
        )
        .await;

        let replay = async {
            let request = replay_request(fixture.replay_server.as_ref().unwrap()).await;
            assert_eq!(read_seq(request.last().unwrap()), 1);
            send_replay_replies(
                fixture.replay_server.as_ref().unwrap(),
                &request,
                &[
                    (
                        1,
                        b"kv".as_slice(),
                        batch_payload(&[store_event(&tokens[1])], 0),
                    ),
                    (
                        2,
                        b"kv".as_slice(),
                        batch_payload(&[store_event(&tokens[2])], 0),
                    ),
                ],
            );
        };
        let invalidate = async {
            commit_barrier.wait().await;
            {
                let _guard = strict_rank.lock();
                fixture.indexer.invalidate_source_for(
                    &source,
                    0,
                    SourceRecoveryReason::RankScopeChanged,
                );
            }
            commit_barrier.wait().await;
        };
        let seq_two_events = [store_event(&tokens[2])];
        tokio::join!(
            publish_until_advanced(
                &fixture.publisher,
                &mut fixture.signals,
                b"kv",
                2,
                2.0,
                &seq_two_events,
                Some(0),
            ),
            replay,
            invalidate,
        );

        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if fixture.indexer.source_recovery_status(&source, 0)
                    == Some(crate::kv_index::SourceRecoveryStatus {
                        state: crate::kv_index::SourceRecoveryState::Fallback,
                        reason: crate::kv_index::SourceRecoveryReason::RankScopeChanged,
                    })
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        for token_ids in &tokens {
            assert!(hits(&fixture.indexer, token_ids).is_empty());
        }
        // A superseded replay must not put the next genuine gap on cooldown.
        fixture
            .publisher
            .send_multipart(
                [
                    b"kv".as_slice(),
                    &4i64.to_be_bytes(),
                    &batch_payload(&[], 0),
                ],
                0,
            )
            .unwrap();
        let next = replay_request(fixture.replay_server.as_ref().unwrap()).await;
        assert_eq!(read_seq(next.last().unwrap()), 0);
    }

    #[tokio::test]
    async fn slow_replay_drains_small_hwm_and_then_catches_up_live_events() {
        let mut f =
            fixture_with_receive_limit(true, None, None, true, Some(2), ReplayLimits::default());
        publish_until_advanced(
            &f.publisher,
            &mut f.signals,
            b"kv",
            0,
            0.0,
            &[store_event(&[1, 2, 3, 4])],
            Some(0),
        )
        .await;
        let server = f.replay_server.as_ref().unwrap();
        let final_tokens = [90, 91, 92, 93];
        let producer = async {
            // A missing seq 1 starts replay through seq 2.
            f.publisher
                .send_multipart(
                    [
                        b"kv".as_slice(),
                        &2i64.to_be_bytes(),
                        &batch_payload(&[], 0),
                    ],
                    0,
                )
                .unwrap();
            let request = replay_request(server).await;
            assert_eq!(read_seq(request.last().unwrap()), 1);
            // More than the combined PUB/SUB HWM arrives before replay replies.
            for seq in 2i64..=42 {
                let events = if seq == 42 {
                    vec![store_event(&final_tokens)]
                } else {
                    vec![]
                };
                f.publisher
                    .send_multipart(
                        [
                            b"kv".as_slice(),
                            &seq.to_be_bytes(),
                            &batch_payload(&events, 0),
                        ],
                        0,
                    )
                    .unwrap();
                tokio::time::sleep(Duration::from_millis(4)).await;
            }
            assert!(
                hits(&f.indexer, &[1, 2, 3, 4]).is_empty(),
                "pending live frames must not restore credit during replay"
            );
            send_replay_replies(
                server,
                &request,
                &[
                    (1, b"kv".as_slice(), batch_payload(&[], 0)),
                    (2, b"kv".as_slice(), batch_payload(&[], 0)),
                ],
            );
        };
        let consumer = async {
            let mut gaps = 0;
            let mut replays = 0;
            let mut second_advances = 0;
            tokio::time::timeout(Duration::from_secs(3), async {
                while let Some(signal) = f.signals.recv().await {
                    match signal {
                        IngestionSignal::Gap { .. } => gaps += 1,
                        IngestionSignal::ReplayApplied { .. } => replays += 1,
                        IngestionSignal::Advance { last_seq: 42, .. } => break,
                        IngestionSignal::Advance { last_seq: 2, .. } => second_advances += 1,
                        _ => {}
                    }
                }
            })
            .await
            .expect("live queue did not catch up after replay");
            assert_eq!(gaps, 1, "replay itself must not induce a new live gap");
            assert_eq!(replays, 1);
            assert_eq!(
                second_advances, 1,
                "replayed live duplicate must be skipped"
            );
        };
        tokio::join!(producer, consumer);
        assert_eq!(hits(&f.indexer, &final_tokens).len(), 1);
        assert_eq!(
            f.indexer
                .source_recovery_status(&SourceId::from("w0"), 0)
                .unwrap()
                .state,
            crate::kv_index::SourceRecoveryState::WorkerExact
        );
    }

    #[tokio::test]
    async fn live_buffer_exhaustion_aborts_replay_without_losing_the_received_frame() {
        let ctx = zmq::Context::new();
        let sender = ctx.socket(zmq::PAIR).unwrap();
        sender.bind("inproc://replay-live-budget").unwrap();
        let mut receiver = ctx.socket(zmq::PAIR).unwrap();
        receiver.connect("inproc://replay-live-budget").unwrap();
        let frames = vec![
            b"kv".to_vec(),
            10i64.to_be_bytes().to_vec(),
            vec![b'x'; 128],
        ];
        sender.send_multipart(&frames, 0).unwrap();
        let mut pending = VecDeque::new();
        let (_shutdown, mut shutdown) = broadcast::channel(1);
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            buffer_live_during_replay(
                std::future::pending(),
                &mut receiver,
                "kv",
                &mut pending,
                64,
                &mut shutdown,
                None,
            ),
        )
        .await
        .unwrap();
        assert_eq!(result, ReplayOutcome::BufferFull);
        assert_eq!(pending.pop_front(), Some(frames));
        assert!(pending.is_empty());
    }

    async fn wait_for_advance(signals: &mut mpsc::Receiver<IngestionSignal>, expected: i64) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while let Some(signal) = signals.recv().await {
                if matches!(signal, IngestionSignal::Advance { last_seq, .. } if last_seq == expected) {
                    return;
                }
            }
            panic!("subscriber stopped before sequence {expected}");
        }).await.expect("subscriber did not advance");
    }

    #[tokio::test]
    async fn unbound_shutdown_interrupts_replay_and_clears_partial_history() {
        for drop_handle in [false, true] {
            let mut f = fixture_with_block_size(true, Some(4));
            let old = [1, 2, 3, 4];
            let partial = [5, 6, 7, 8];
            publish_until_advanced(
                &f.publisher,
                &mut f.signals,
                b"kv",
                0,
                0.0,
                &[store_event(&old)],
                Some(0),
            )
            .await;
            f.publisher
                .send_multipart(
                    [
                        b"kv".as_slice(),
                        &3i64.to_be_bytes(),
                        &batch_payload(&[], 0),
                    ],
                    0,
                )
                .unwrap();
            let server = f.replay_server.as_ref().unwrap();
            let request = replay_request(server).await;
            send_replay_replies(
                server,
                &request,
                &[(
                    1,
                    b"kv".as_slice(),
                    batch_payload(&[store_event(&partial)], 0),
                )],
            );
            wait_for_advance(&mut f.signals, 1).await;
            if drop_handle {
                drop(f._handle);
            } else {
                f._handle.shutdown();
            }
            tokio::time::timeout(Duration::from_millis(500), async {
                while f.signals.recv().await.is_some() {}
            })
            .await
            .expect("shutdown waited for the five-second replay deadline");
            assert!(hits(&f.indexer, &old).is_empty());
            assert!(hits(&f.indexer, &partial).is_empty());
            assert_eq!(
                f.indexer
                    .source_recovery_status(&SourceId::from("w0"), 0)
                    .unwrap()
                    .state,
                crate::kv_index::SourceRecoveryState::Fallback
            );
        }
    }

    #[tokio::test]
    async fn retired_lease_interrupts_the_live_replay_wait() {
        let ctx = zmq::Context::new();
        let sender = ctx.socket(zmq::PAIR).unwrap();
        sender.bind("inproc://replay-retired-lease").unwrap();
        let mut receiver = ctx.socket(zmq::PAIR).unwrap();
        receiver.connect("inproc://replay-retired-lease").unwrap();
        let active = Arc::new(AtomicBool::new(true));
        let binding = SubscriberBinding {
            lease: SourceLease {
                strict_rank: Arc::new(Mutex::new(false)),
                active: active.clone(),
            },
            replay_cleanup_barrier: None,
            replay_commit_barrier: None,
        };
        let (_shutdown, mut shutdown) = broadcast::channel(1);
        let (entered, started) = tokio::sync::oneshot::channel();
        let recovery = async {
            entered.send(()).unwrap();
            std::future::pending().await
        };
        let retire = async {
            started.await.unwrap();
            active.store(false, Ordering::Release);
        };
        let mut pending = VecDeque::new();
        let (result, ()) = tokio::time::timeout(Duration::from_millis(500), async {
            tokio::join!(
                buffer_live_during_replay(
                    recovery,
                    &mut receiver,
                    "kv",
                    &mut pending,
                    MAX_PENDING_LIVE_BYTES,
                    &mut shutdown,
                    Some(&binding)
                ),
                retire,
            )
        })
        .await
        .expect("retired lease did not cancel the replay wait");
        assert_eq!(result, ReplayOutcome::Stopped);
        assert!(pending.is_empty());
    }

    #[tokio::test]
    async fn queue_full_abort_preserves_retry_budget_and_clears_partial_recovery() {
        let cleanup = Arc::new(tokio::sync::Barrier::new(2));
        let mut f = fixture_with_receive_limit(
            true,
            Some(4),
            Some(SubscriberBinding {
                lease: SourceLease {
                    strict_rank: Arc::new(Mutex::new(false)),
                    active: Arc::new(AtomicBool::new(true)),
                },
                replay_cleanup_barrier: Some(cleanup.clone()),
                replay_commit_barrier: None,
            }),
            false,
            Some(4096),
            ReplayLimits::default(),
        );
        let old = [1, 2, 3, 4];
        let partial = [5, 6, 7, 8];
        let fresh = [9, 10, 11, 12];
        publish_until_advanced(
            &f.publisher,
            &mut f.signals,
            b"kv",
            0,
            0.0,
            &[store_event(&old)],
            Some(0),
        )
        .await;
        f.publisher
            .send_multipart(
                [
                    b"kv".as_slice(),
                    &2i64.to_be_bytes(),
                    &batch_payload(&[], 0),
                ],
                0,
            )
            .unwrap();
        let server = f.replay_server.as_ref().unwrap();
        let first = replay_request(server).await;
        send_replay_replies(
            server,
            &first,
            &[(
                1,
                b"kv".as_slice(),
                batch_payload(&[store_event(&partial)], 0),
            )],
        );
        wait_for_advance(&mut f.signals, 1).await;
        // Fill the frame limit using small valid events, without a 64 MiB allocation.
        let last = 2 + MAX_PENDING_LIVE_FRAMES as i64;
        for seq in 3..=last {
            f.publisher
                .send_multipart(
                    [b"kv".as_slice(), &seq.to_be_bytes(), &batch_payload(&[], 0)],
                    0,
                )
                .unwrap();
        }
        tokio::time::timeout(Duration::from_secs(1), cleanup.wait())
            .await
            .expect("the live frame queue did not cancel replay");
        cleanup.wait().await;
        wait_for_advance(&mut f.signals, last).await;
        assert!(hits(&f.indexer, &old).is_empty());
        assert!(hits(&f.indexer, &partial).is_empty());
        // A genuine gap must be retryable immediately, rather than on a 2 s cooldown.
        let next = last + 2;
        f.publisher
            .send_multipart(
                [
                    b"kv".as_slice(),
                    &next.to_be_bytes(),
                    &batch_payload(&[], 0),
                ],
                0,
            )
            .unwrap();
        let request = tokio::time::timeout(Duration::from_secs(1), replay_request(server))
            .await
            .expect("a local queue abort consumed the publisher retry budget");
        assert_eq!(read_seq(request.last().unwrap()), last + 1);
        send_replay_replies(
            server,
            &request,
            &[
                (last + 1, b"kv".as_slice(), batch_payload(&[], 0)),
                (
                    next,
                    b"kv".as_slice(),
                    batch_payload(&[store_event(&fresh)], 0),
                ),
            ],
        );
        tokio::time::timeout(Duration::from_secs(2), async {
            while let Some(signal) = f.signals.recv().await {
                if matches!(signal, IngestionSignal::ReplayApplied { replay_seq, .. } if replay_seq == next) {
                    return;
                }
            }
            panic!("subscriber stopped before replay completed");
        }).await.expect("retry did not complete");
        assert_eq!(hits(&f.indexer, &fresh).len(), 1);
    }

    #[tokio::test]
    async fn replay_rejections_clear_history_and_partial_stores() {
        for case in [
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

    #[tokio::test]
    async fn worker_clear_only_restores_credit_in_its_domain() {
        use crate::kv_index::SourceRecoveryState::{Exact, Fallback, Observed, WorkerExact};
        let old = [1, 2, 3, 4];
        let fresh = [5, 6, 7, 8];
        let local = [9, 10, 11, 12];
        let initial = [13, 14, 15, 16];
        let hidden = [17, 18, 19, 20];
        let pool_store = |_id, tokens: &[u32]| {
            KVEvent::BlockStored(stored(1, tokens.to_vec(), Some("GPU"), Some("pool")))
        };
        let source = SourceId::from("w0");
        for (strict, gap, expected) in [
            (true, false, WorkerExact),
            (false, false, Exact),
            (true, true, WorkerExact),
            (false, true, Observed),
        ] {
            let mut f = fixture_with_recovery(strict && gap, Some(4), None, strict);
            tokio::time::timeout(Duration::from_secs(2), async {
                while f.indexer.source_recovery_status(&source, 0).is_none() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert_eq!(
                f.indexer.source_recovery_status(&source, 0),
                Some(crate::kv_index::SourceRecoveryStatus {
                    state: Fallback,
                    reason: SourceRecoveryReason::Registered,
                })
            );
            publish_until_advanced(
                &f.publisher,
                &mut f.signals,
                b"kv",
                0,
                0.0,
                &[pool_store(100, &old), store_event(&initial)],
                Some(0),
            )
            .await;
            assert_eq!(
                f.indexer.source_recovery_status(&source, 0).unwrap().state,
                if strict { WorkerExact } else { Exact }
            );
            assert_eq!(hits(&f.indexer, &initial).len(), 1);
            if gap {
                let gap_events = [store_event(&hidden)];
                tokio::join!(
                    publish_until_advanced(
                        &f.publisher,
                        &mut f.signals,
                        b"kv",
                        2,
                        0.0,
                        &gap_events,
                        Some(0)
                    ),
                    async {
                        if let Some(server) = &f.replay_server {
                            let request = replay_request(server).await;
                            assert_eq!(read_seq(request.last().unwrap()), 1);
                            send_replay_replies(
                                server,
                                &request,
                                &[(END_SEQ, b"".as_slice(), vec![])],
                            );
                        }
                    },
                );
                assert_eq!(
                    f.indexer.source_recovery_status(&source, 0).unwrap().state,
                    if strict { Fallback } else { Observed }
                );
                assert!(hits(&f.indexer, &initial).is_empty());
                assert_eq!(hits(&f.indexer, &hidden).len(), usize::from(!strict));
            } else {
                publish_until_advanced(&f.publisher, &mut f.signals, b"kv", 1, 0.0, &[], Some(0))
                    .await;
            }
            let clear_seq = if gap { 3 } else { 2 };
            publish_until_advanced(
                &f.publisher,
                &mut f.signals,
                b"kv",
                clear_seq,
                0.0,
                &[
                    KVEvent::AllBlocksCleared(AllBlocksCleared {}),
                    store_event(&local),
                ],
                Some(0),
            )
            .await;
            assert_eq!(
                f.indexer.source_recovery_status(&source, 0).unwrap().reason,
                SourceRecoveryReason::ClearBarrier
            );
            let fresh_seq = clear_seq + 1;
            publish_until_advanced(
                &f.publisher,
                &mut f.signals,
                b"kv",
                fresh_seq,
                0.0,
                &[pool_store(102, &fresh)],
                Some(0),
            )
            .await;
            assert_eq!(
                f.indexer.source_recovery_status(&source, 0).unwrap().state,
                expected
            );
            assert_eq!(hits(&f.indexer, &local).len(), 1);
            let pool_hits = |tokens: &[u32]| {
                f.indexer
                    .find_matches(&MatchQuery {
                        group_idx: 0,
                        hashes: crate::kv_index::ordinary_block_hashes(tokens, 4),
                        tiers_of_interest: vec![StorageTier::External],
                    })
                    .len()
            };
            assert_eq!(pool_hits(&old), usize::from(!strict && !gap));
            assert_eq!(pool_hits(&fresh), usize::from(!strict));
            if strict && gap {
                let server = f.replay_server.as_ref().unwrap();
                let next = fresh_seq + 2;
                tokio::join!(
                    publish_until_advanced(
                        &f.publisher,
                        &mut f.signals,
                        b"kv",
                        next,
                        0.0,
                        &[],
                        Some(0)
                    ),
                    async {
                        let request = replay_request(server).await;
                        assert_eq!(read_seq(request.last().unwrap()), fresh_seq + 1);
                        send_replay_replies(
                            server,
                            &request,
                            &[
                                (fresh_seq + 1, b"kv".as_slice(), batch_payload(&[], 0)),
                                (next, b"kv".as_slice(), batch_payload(&[], 0)),
                            ],
                        );
                    },
                );
                assert_eq!(
                    f.indexer.source_recovery_status(&source, 0).unwrap().state,
                    WorkerExact
                );
                assert_eq!(pool_hits(&old), 0);
                assert_eq!(pool_hits(&fresh), 0);
                assert_eq!(hits(&f.indexer, &local).len(), 1);
            }
        }
    }

    #[test]
    fn replay_backoff_has_a_budget_and_boundary_reset() {
        let mut retries = ReplayRetryBudget::default();
        let mut now = Instant::now();
        for _ in 0..3 {
            assert!(retries.allowed(now));
            retries.failed(now);
            assert!(!retries.allowed(now));
            now += Duration::from_secs(60);
        }
        assert!(!retries.allowed(now));
        retries.reset();
        assert!(retries.allowed(now));
    }

    #[test]
    fn exhausted_bad_frames_invalidate_once_until_fresh_observations() {
        let indexer = KvBlockIndexer::new();
        let source = SourceId::from("w0");
        let mut retries = ReplayRetryBudget::default();
        let start = Instant::now();
        retries.failed(start);
        assert!(!retries.allowed(start + Duration::from_secs(1)));
        assert!(retries.allowed(start + Duration::from_secs(2)));
        retries.failed(start + Duration::from_secs(2));
        assert!(!retries.allowed(start + Duration::from_secs(5)));
        assert!(retries.allowed(start + Duration::from_secs(6)));
        assert_eq!(
            retries.failure_reason(SourceRecoveryReason::ReplayFailed),
            SourceRecoveryReason::RetryExhausted
        );
        retries.failed(start + Duration::from_secs(6));
        retries.invalidate_once(&indexer, &source, 0, SourceRecoveryReason::SequenceGap);
        let generation = indexer.source_generation_for_test();
        for reason in [
            SourceRecoveryReason::ValidationFailed,
            SourceRecoveryReason::SequenceGap,
        ] {
            retries.invalidate_once(&indexer, &source, 0, reason);
        }
        assert_eq!(indexer.source_generation_for_test(), generation);
        assert_eq!(
            indexer.source_recovery_status(&source, 0).unwrap().reason,
            SourceRecoveryReason::RetryExhausted
        );
        retries.loss_recorded = false;
        retries.invalidate_once(&indexer, &source, 0, SourceRecoveryReason::SequenceGap);
        assert_eq!(indexer.source_generation_for_test(), generation + 1);
    }

    #[tokio::test]
    async fn publisher_restart_restores_only_worker_credit_live_or_replayed() {
        let source = SourceId::from("w0");
        let pool_store = |_id, tokens: &[u32]| {
            KVEvent::BlockStored(stored(1, tokens.to_vec(), Some("GPU"), Some("pool")))
        };
        for use_replay in [false, true] {
            let mut f = fixture_with_block_size(use_replay, Some(4));
            let old_local = [13, 14, 15, 16];
            let old_pool = [1, 2, 3, 4];
            let new_pool = [5, 6, 7, 8];
            let new_local = [9, 10, 11, 12];
            let pool_query = |tokens: &[u32]| MatchQuery {
                group_idx: 0,
                hashes: crate::kv_index::ordinary_block_hashes(tokens, 4),
                tiers_of_interest: vec![StorageTier::External],
            };
            for seq in 0..=2 {
                let events = if seq == 0 {
                    vec![pool_store(100, &old_pool), store_event(&old_local)]
                } else {
                    vec![]
                };
                publish_until_advanced(
                    &f.publisher,
                    &mut f.signals,
                    b"kv",
                    seq,
                    0.0,
                    &events,
                    Some(0),
                )
                .await;
            }
            // Give the old generation a known full baseline. The new generation
            // does not re-announce the surviving pool block.
            f.indexer
                .resume_boundary(&source, 0, RecoveryBoundary::Initial, false);
            assert_eq!(f.indexer.find_matches(&pool_query(&old_pool)).len(), 1);
            let initial = [pool_store(101, &new_pool), store_event(&new_local)];
            let reason = if let Some(server) = &f.replay_server {
                tokio::join!(
                    publish_until_advanced(
                        &f.publisher,
                        &mut f.signals,
                        b"kv",
                        1,
                        0.0,
                        &[],
                        Some(0)
                    ),
                    async {
                        let request = replay_request(server).await;
                        assert_eq!(read_seq(request.last().unwrap()), 0);
                        send_replay_replies(
                            server,
                            &request,
                            &[
                                (0, b"kv".as_slice(), batch_payload(&initial, 0)),
                                (1, b"kv".as_slice(), batch_payload(&[], 0)),
                            ],
                        );
                    },
                );
                tokio::time::timeout(Duration::from_secs(5), async {
                    while let Some(signal) = f.signals.recv().await {
                        if matches!(signal, IngestionSignal::ReplayApplied { replay_seq: 1, .. }) {
                            return;
                        }
                    }
                    panic!("subscriber closed before replay committed");
                })
                .await
                .unwrap();
                SourceRecoveryReason::ReplayApplied
            } else {
                publish_until_advanced(
                    &f.publisher,
                    &mut f.signals,
                    b"kv",
                    0,
                    0.0,
                    &initial,
                    Some(0),
                )
                .await;
                SourceRecoveryReason::InitialSequence
            };
            assert_eq!(
                f.indexer.source_recovery_status(&source, 0),
                Some(crate::kv_index::SourceRecoveryStatus {
                    state: crate::kv_index::SourceRecoveryState::WorkerExact,
                    reason,
                })
            );
            assert!(hits(&f.indexer, &old_local).is_empty());
            assert_eq!(hits(&f.indexer, &new_local).len(), 1);
            assert!(f.indexer.find_matches(&pool_query(&old_pool)).is_empty());
            assert!(f.indexer.find_matches(&pool_query(&new_pool)).is_empty());
        }
    }

    #[tokio::test]
    async fn first_nonzero_sequence_respects_recovery_mode() {
        for strict in [false, true] {
            for replay_available in [false, true] {
                let mut f = fixture_with_recovery(replay_available, Some(4), None, strict);
                let events = [store_event(&[1, 2, 3, 4])];
                tokio::join!(
                    publish_until_advanced(
                        &f.publisher,
                        &mut f.signals,
                        b"kv",
                        20,
                        0.0,
                        &events,
                        Some(0)
                    ),
                    async {
                        if let Some(server) = &f.replay_server {
                            let request = replay_request(server).await;
                            assert_eq!(read_seq(request.last().unwrap()), 0);
                            send_replay_replies(
                                server,
                                &request,
                                &[(END_SEQ, b"".as_slice(), vec![])],
                            );
                        }
                    },
                );
                let expected = if strict {
                    crate::kv_index::SourceRecoveryState::Fallback
                } else {
                    crate::kv_index::SourceRecoveryState::Observed
                };
                assert_eq!(
                    f.indexer
                        .source_recovery_status(&SourceId::from("w0"), 0)
                        .unwrap()
                        .state,
                    expected,
                    "after seq 20: strict={strict}, replay_available={replay_available}"
                );
                assert_eq!(
                    hits(&f.indexer, &[1, 2, 3, 4]).len(),
                    usize::from(!strict),
                    "after seq 20: strict={strict}, replay_available={replay_available}"
                );
                publish_until_advanced(
                    &f.publisher,
                    &mut f.signals,
                    b"kv",
                    21,
                    0.0,
                    &events,
                    Some(0),
                )
                .await;
                let status = f
                    .indexer
                    .source_recovery_status(&SourceId::from("w0"), 0)
                    .unwrap();
                assert_eq!(
                    status.state, expected,
                    "strict={strict}, replay_available={replay_available}"
                );
                if strict {
                    assert_eq!(status.reason, SourceRecoveryReason::ReplayFailed);
                }
                assert_eq!(hits(&f.indexer, &[1, 2, 3, 4]).len(), usize::from(!strict));
                if !replay_available {
                    publish_until_advanced(
                        &f.publisher,
                        &mut f.signals,
                        b"kv",
                        23,
                        0.0,
                        &[store_event(&[5, 6, 7, 8])],
                        Some(0),
                    )
                    .await;
                    assert!(
                        hits(&f.indexer, &[1, 2, 3, 4]).is_empty(),
                        "backoff must still discard claims across a new gap"
                    );
                    assert_eq!(hits(&f.indexer, &[5, 6, 7, 8]).len(), usize::from(!strict));
                }
            }
        }
    }

    #[tokio::test]
    async fn persistent_rejected_frames_do_not_replay_per_frame() {
        let mut f = fixture_with_block_size(true, Some(4));
        publish_until_advanced(
            &f.publisher,
            &mut f.signals,
            b"kv",
            0,
            0.0,
            &[store_event(&[1, 2, 3, 4])],
            Some(0),
        )
        .await;
        let mut invalid = stored(1, vec![1, 2, 3, 4, 5, 6, 7, 8], Some("GPU"), None);
        invalid.block_size = 8;
        let payload = batch_payload(&[KVEvent::BlockStored(invalid)], 0);
        let server = f.replay_server.as_ref().unwrap();
        let started = Instant::now();
        let request = loop {
            f.publisher
                .send_multipart([b"kv".as_slice(), &2i64.to_be_bytes(), &payload], 0)
                .unwrap();
            match server.recv_multipart(zmq::DONTWAIT) {
                Ok(request) => break request,
                Err(zmq::Error::EAGAIN) => tokio::time::sleep(Duration::from_millis(10)).await,
                Err(error) => panic!("{error}"),
            }
            assert!(started.elapsed() < Duration::from_secs(3));
        };
        send_replay_replies(server, &request, &[(END_SEQ, b"".as_slice(), vec![])]);
        tokio::time::sleep(Duration::from_millis(30)).await;
        for seq in 3i64..20 {
            f.publisher
                .send_multipart([b"kv".as_slice(), &seq.to_be_bytes(), &payload], 0)
                .unwrap();
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(matches!(
            server.recv_multipart(zmq::DONTWAIT),
            Err(zmq::Error::EAGAIN)
        ));
        assert_eq!(
            f.indexer
                .source_recovery_status(&SourceId::from("w0"), 0)
                .unwrap()
                .state,
            crate::kv_index::SourceRecoveryState::Fallback
        );
    }

    #[tokio::test]
    async fn full_replay_deadline_does_not_publish_partial_history() {
        let f = fixture(true);
        let source = SourceId::from("w0");
        // This test drives replay directly; wait for the fixture subscriber's registration.
        tokio::time::timeout(Duration::from_secs(2), async {
            while f.indexer.source_recovery_status(&source, 0).is_none() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let attempt = f
            .indexer
            .suspend_source_for(&source, 0, SourceRecoveryReason::SequenceGap);
        let mut reported = false;
        let endpoint = f
            .replay_server
            .as_ref()
            .unwrap()
            .get_last_endpoint()
            .unwrap()
            .unwrap();
        let (progress, mut advances) = mpsc::channel(1);
        let replay_future = replay(
            &endpoint,
            -1,
            1,
            attempt,
            &f._ctx,
            &f.indexer,
            &source,
            0,
            0,
            Some(progress),
            Some(4),
            "kv",
            &mut reported,
            None,
            Duration::from_millis(50),
        );
        let server = async {
            let request = replay_request(f.replay_server.as_ref().unwrap()).await;
            send_replay_replies(
                f.replay_server.as_ref().unwrap(),
                &request,
                &[(
                    0,
                    b"kv".as_slice(),
                    batch_payload(&[store_event(&[1, 2, 3, 4])], 0),
                )],
            );
        };
        let (result, ()) = tokio::time::timeout(Duration::from_secs(1), async {
            tokio::join!(replay_future, server)
        })
        .await
        .expect("configured replay deadline did not terminate recovery");
        assert!(result.is_none());
        assert!(
            matches!(
                advances.try_recv(),
                Ok(IngestionSignal::Advance { last_seq: 0, .. })
            ),
            "deadline test must apply a partial batch before cleanup"
        );
        assert!(f.indexer.finish_replay_if_current(
            &source,
            0,
            attempt,
            Some(SourceRecoveryReason::ReplayFailed)
        ));
        assert!(hits(&f.indexer, &[1, 2, 3, 4]).is_empty());
    }

    #[tokio::test]
    async fn connect_failure_does_not_register_a_cold_source() {
        let indexer = Arc::new(KvBlockIndexer::new());
        let source = SourceId::from("bad-connect");
        let (tx, mut rx) = mpsc::channel(1);
        let _handle = spawn(
            EventSource {
                source: source.clone(),
                dp_rank: 0,
                pub_endpoint: "unsupported://endpoint".into(),
                replay_endpoint: None,
                topic: "kv".into(),
                hwm: None,
                strict_recovery: true,
            },
            indexer.clone(),
            Some(tx),
            Arc::new(zmq::Context::new()),
        );
        assert!(tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .is_none());
        assert_eq!(indexer.source_recovery_status(&source, 0), None);
    }
}
