//! ZMQ SUB + DEALER replay subscriber, one task per (worker, rank).

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{broadcast, mpsc};
use tracing::{debug, info, warn};

use crate::kv_index::admit::{validate_dense_batch, DenseBatchRejection};
use crate::kv_index::binding::SourceLease;
use crate::kv_index::indexer::KvBlockIndexer;
use crate::kv_index::types::{ClearScope, ResidencyOwner, SourceId, StorageTier};
use crate::kv_index::wire::{ExternalBlockHash, KVEvent, KVEventBatch};
use crate::kv_index::{decode_batch, IngestionSignal};

/// sha256 of each block's token-id bytes, seeded by `lora_name`.
/// The seed is re-applied per block (no parent chaining here) so identical
/// tokens under different adapters land on different nodes.
// ponytail: cache_salt/MM isolation deferred — needs raw msgpack capture of
// per-block extra_keys tuples (custom newtype or rmpv); add when cost model
// scores cross-salt collisions. Until then lora_name alone distinguishes adapters.
pub fn local_hashes(token_ids: &[u32], block_size: u32, lora_name: Option<&str>) -> Vec<[u8; 32]> {
    crate::kv_index::local_block_hashes(token_ids, block_size, lora_name)
}

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
    spawn_inner(source, indexer, signal_tx, ctx, None, None)
}

/// Dense HTTP inputs can only query identities represented by local token hashes.
/// Event block boundaries must agree with the source's cache/query identity.
#[cfg(test)]
pub(crate) fn spawn_dense(
    source: EventSource,
    indexer: Arc<KvBlockIndexer>,
    signal_tx: Option<mpsc::Sender<IngestionSignal>>,
    ctx: Arc<zmq::Context>,
    expected_block_size: u32,
) -> SubscriberHandle {
    spawn_inner(
        source,
        indexer,
        signal_tx,
        ctx,
        Some(expected_block_size),
        None,
    )
}

/// The supervisor shares this guard across owners of one publisher/exact topic.
/// Its active lease only fences this local task, not the remote publisher.
struct SubscriberBinding {
    lease: SourceLease,
    #[cfg(test)]
    replay_cleanup_barrier: Option<Arc<tokio::sync::Barrier>>,
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
        }),
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

fn spawn_inner(
    source: EventSource,
    indexer: Arc<KvBlockIndexer>,
    signal_tx: Option<mpsc::Sender<IngestionSignal>>,
    ctx: Arc<zmq::Context>,
    expected_block_size: Option<u32>,
    binding: Option<SubscriberBinding>,
) -> SubscriberHandle {
    let (shutdown_tx, mut shutdown_rx) = broadcast::channel::<()>(1);
    let pub_endpoint = source.pub_endpoint.clone();
    let replay_endpoint = source.replay_endpoint.clone();
    let topic = source.topic.clone();
    let source_id = source.source.clone();
    let dp_rank = source.dp_rank;
    let hwm = source.hwm;
    let signals = signal_tx;

    tokio::spawn(async move {
        let sub = match ctx.socket(zmq::SUB) {
            Ok(s) => s,
            Err(e) => {
                warn!("kv_index sub socket for {}: {}", source_id, e);
                return;
            }
        };
        if let Err(e) = sub.connect(&pub_endpoint) {
            warn!("kv_index connect {} to {}: {}", source_id, pub_endpoint, e);
            return;
        }
        sub.set_subscribe(topic.as_bytes()).ok();
        if let Some(h) = hwm {
            sub.set_rcvhwm(h).ok();
        }
        // zmq recv is sync; DONTWAIT + sleep keeps the tokio worker free. No
        // rcvtimeo — it's a no-op under DONTWAIT.
        info!(
            "kv_index subscriber for {} rank{} on {}",
            source_id, dp_rank, pub_endpoint
        );

        let mut last_seq: i64 = -1;
        let mut incarnation: u64 = 0;
        let mut suspended = false;
        let mut dense_rejection_reported = false;

        loop {
            if is_shutdown(&mut shutdown_rx)
                || binding
                    .as_ref()
                    .is_some_and(|binding| !binding.lease.is_active())
            {
                info!("kv_index subscriber {} shutting down", source_id);
                break;
            }
            match sub.recv_multipart(zmq::DONTWAIT) {
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
                            indexer.invalidate_source(&source_id, dp_rank);
                        })
                        .is_none()
                        {
                            break;
                        }
                        suspended = true;
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
                            indexer.invalidate_source(&source_id, dp_rank);
                        })
                        .is_none()
                        {
                            break;
                        }
                        suspended = true;
                        last_seq = -1;
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
                    // Include initial attachment to an already-running publisher.
                    // A corrupt event is a one-event hole, not an applied seq.
                    if seq > last_seq + 1 || !valid_rank {
                        let from = last_seq + 1;
                        let to = if !valid_rank { seq } else { seq - 1 };
                        if with_binding(binding.as_ref(), |_| {
                            indexer.suspend_source(&source_id, dp_rank);
                        })
                        .is_none()
                        {
                            break;
                        }
                        suspended = true;
                        emit(
                            &signals,
                            IngestionSignal::Gap {
                                source: source_id.clone(),
                                from_seq: from,
                                to_seq: to,
                            },
                        );
                        if let Some(ep) = replay_endpoint.as_deref() {
                            if let Some(last_replay) = replay(
                                ep,
                                last_seq,
                                seq,
                                &ctx,
                                &indexer,
                                &source_id,
                                dp_rank,
                                incarnation,
                                signals.clone(),
                                expected_block_size,
                                &topic,
                                &mut dense_rejection_reported,
                                binding.as_ref(),
                            )
                            .await
                            {
                                last_seq = last_replay;
                                if with_binding(binding.as_ref(), |_| {
                                    indexer.resume_source(&source_id, dp_rank);
                                })
                                .is_none()
                                {
                                    break;
                                }
                                suspended = false;
                                emit(
                                    &signals,
                                    IngestionSignal::ReplayApplied {
                                        source: source_id.clone(),
                                        replay_seq: last_replay,
                                    },
                                );
                                tokio::task::yield_now().await;
                                continue;
                            }
                            #[cfg(test)]
                            if let Some(barrier) = binding
                                .as_ref()
                                .and_then(|binding| binding.replay_cleanup_barrier.as_ref())
                            {
                                barrier.wait().await;
                                barrier.wait().await;
                            }
                        }
                        // Partial replay can include stores whose later
                        // evictions were lost. Never expose that partial state.
                        if with_binding(binding.as_ref(), |_| {
                            indexer.invalidate_source(&source_id, dp_rank);
                        })
                        .is_none()
                        {
                            break;
                        }
                    }
                    if let Some(batch) = batch {
                        // After unrecoverable loss, credit only fresh observations
                        // on the now-clean source; absence is a conservative miss.
                        let applied = with_binding(binding.as_ref(), |strict| {
                            match batch_rank(batch.data_parallel_rank, dp_rank, strict) {
                                BatchRank::Owned => {
                                    if let Err(rejection) = apply_batch(
                                        &indexer,
                                        &source_id,
                                        dp_rank,
                                        incarnation,
                                        &batch,
                                        expected_block_size,
                                    ) {
                                        report_dense_rejection(
                                            rejection,
                                            &source_id,
                                            dp_rank,
                                            &mut dense_rejection_reported,
                                        );
                                        indexer.invalidate_source(&source_id, dp_rank);
                                        suspended = true;
                                        return false;
                                    }
                                    // Promotion/retirement may have invalidated the
                                    // source outside this task's local state.
                                    if suspended || binding.is_some() {
                                        indexer.resume_source(&source_id, dp_rank);
                                        suspended = false;
                                    }
                                    true
                                }
                                BatchRank::Foreign => true,
                                BatchRank::Rejected => {
                                    indexer.invalidate_source(&source_id, dp_rank);
                                    suspended = true;
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

fn owner_for(source_id: &SourceId, dp_rank: u32, incarnation: u64) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: source_id.clone(),
        dp_rank,
        incarnation,
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

fn hash_str(h: &ExternalBlockHash) -> Arc<str> {
    // Discriminator: Int(16) and Bytes([0x16]) both render "16" without it.
    match h {
        ExternalBlockHash::Int(i) => Arc::from(format!("i:{i}")),
        ExternalBlockHash::Bytes(b) => Arc::from(format!("b:{}", hex(b))),
    }
}

fn report_dense_rejection(
    rejection: DenseBatchRejection,
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
) -> Result<(), DenseBatchRejection> {
    // Preflight the whole batch before any Store, Remove, or Clear mutation.
    if let Some(expected) = expected_block_size {
        validate_dense_batch(batch, expected)?;
    }
    // Batch rank if the 3-element array carries it; else the discovery rank.
    let rank = batch.data_parallel_rank.unwrap_or(dp_rank);
    for ev in &batch.events {
        match ev {
            KVEvent::BlockStored(s) => {
                let group = s.group_idx.unwrap_or(0);
                let owner = event_owner(source_id, rank, incarnation, s.ownership.as_deref());
                let tier = event_tier(&owner, s.medium.as_deref());
                let locals: Vec<Arc<str>> =
                    local_hashes(&s.token_ids, s.block_size, s.lora_name.as_deref())
                        .iter()
                        .map(|b| Arc::from(hex(b)))
                        .collect();
                // vLLM emits one BlockStored per contiguous chain (block[i]'s parent
                // is block[i-1]); zipping seqs with local hashes yields one edge.
                let seqs: Vec<Arc<str>> = s.block_hashes.iter().map(hash_str).collect();
                let blocks: Vec<(Arc<str>, Arc<str>)> = seqs.into_iter().zip(locals).collect();
                let parent_seq = s.parent_block_hash.as_ref().map(hash_str);
                indexer.store(group, owner, tier, parent_seq, &blocks);
            }
            KVEvent::BlockRemoved(r) => {
                let group = r.group_idx.unwrap_or(0);
                let owner = event_owner(source_id, rank, incarnation, r.ownership.as_deref());
                let tier = event_tier(&owner, r.medium.as_deref());
                let hashes: Vec<Arc<str>> = r.block_hashes.iter().map(hash_str).collect();
                indexer.remove(group, &owner, tier, &hashes);
            }
            KVEvent::AllBlocksCleared(_) => {
                indexer.clear(&owner_for(source_id, rank, incarnation), ClearScope::Worker);
            }
        }
    }
    Ok(())
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
    ctx: &zmq::Context,
    indexer: &Arc<KvBlockIndexer>,
    source_id: &SourceId,
    dp_rank: u32,
    incarnation: u64,
    signals: Option<mpsc::Sender<IngestionSignal>>,
    expected_block_size: Option<u32>,
    topic: &str,
    dense_rejection_reported: &mut bool,
    binding: Option<&SubscriberBinding>,
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
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
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
                                report_dense_rejection(
                                    rejection,
                                    source_id,
                                    dp_rank,
                                    dense_rejection_reported,
                                );
                                false
                            } else {
                                true
                            }
                        }
                        BatchRank::Foreign => true,
                        BatchRank::Rejected => false,
                    }
                })?;
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

fn hex(b: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        write!(s, "{:02x}", byte).unwrap();
    }
    s
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use parking_lot::Mutex;

    use super::*;
    use crate::kv_index::admit::DenseRejectionReason;
    use crate::kv_index::indexer::MatchQuery;
    use crate::kv_index::types::Locality;
    use crate::kv_index::wire::{AllBlocksCleared, BlockRemoved, BlockStored};

    fn stored(
        block_hashes: Vec<ExternalBlockHash>,
        token_ids: Vec<u32>,
        medium: Option<&str>,
        ownership: Option<&str>,
    ) -> BlockStored {
        BlockStored {
            extra_keys: None,
            block_hashes,
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
            for dense_only in [false, true] {
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
                        &KVEventBatch {
                            ts: 1.0,
                            events: vec![KVEvent::BlockStored(event)],
                            data_parallel_rank: Some(0),
                        },
                        dense_only.then_some(4),
                    )
                    .is_ok(),
                    !dense_only || expected > 0
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
                    if dense_only { expected } else { 1 }
                );
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
                KVEvent::BlockStored(stored(
                    vec![ExternalBlockHash::Int(10)],
                    vec![1, 2, 3, 4],
                    Some("GPU"),
                    Some("mooncake"),
                )),
                // ownership None, medium CPU → Worker → HostPinned.
                KVEvent::BlockStored(stored(
                    vec![ExternalBlockHash::Int(20)],
                    vec![5, 6, 7, 8],
                    Some("CPU"),
                    None,
                )),
            ],
            data_parallel_rank: Some(1),
        };
        apply_batch(&idx, &SourceId::from("w0"), 0, 0, &batch, None).unwrap();

        let pool_locals: Vec<Arc<str>> = local_hashes(&[1, 2, 3, 4], 4, None)
            .iter()
            .map(|b| Arc::from(hex(b)))
            .collect();
        let pool_res = idx.find_matches(&MatchQuery {
            group_idx: 0,
            local_hashes: pool_locals,
            tiers_of_interest: vec![StorageTier::External],
        });
        assert_eq!(pool_res.len(), 1);
        assert_eq!(pool_res[0].tier, StorageTier::External);
        assert_eq!(pool_res[0].locality, Locality::Remote);
        assert_eq!(&*pool_res[0].target.instance_id, "w0");
        assert_eq!(pool_res[0].target.dp_rank, 1);

        let wkr_locals: Vec<Arc<str>> = local_hashes(&[5, 6, 7, 8], 4, None)
            .iter()
            .map(|b| Arc::from(hex(b)))
            .collect();
        let wkr_res = idx.find_matches(&MatchQuery {
            group_idx: 0,
            local_hashes: wkr_locals,
            tiers_of_interest: vec![StorageTier::HostPinned],
        });
        assert_eq!(wkr_res.len(), 1);
        assert_eq!(wkr_res[0].tier, StorageTier::HostPinned);
        assert_eq!(wkr_res[0].locality, Locality::Local);
        assert_eq!(&*wkr_res[0].target.instance_id, "w0");
        assert_eq!(wkr_res[0].target.dp_rank, 1);
    }

    #[test]
    fn dense_batch_rejects_unmodeled_residency_and_layout_atomically() {
        let source = SourceId::from("dense");
        let valid = stored(
            vec![ExternalBlockHash::Int(1)],
            vec![1, 2, 3, 4],
            Some("GPU"),
            None,
        );
        for case in [
            "remote",
            "local-pool",
            "unknown-medium",
            "layout",
            "sliding-window",
            "token-count",
        ] {
            let index = KvBlockIndexer::new();
            let mut invalid = valid.clone();
            invalid.block_hashes = vec![ExternalBlockHash::Int(2)];
            invalid.token_ids = vec![5, 6, 7, 8];
            let reason = match case {
                "remote" => {
                    invalid.locality = Some("REMOTE".into());
                    DenseRejectionReason::Residency
                }
                "local-pool" => {
                    invalid.locality = Some("LOCAL".into());
                    invalid.ownership = Some("pool".into());
                    DenseRejectionReason::Residency
                }
                "unknown-medium" => {
                    invalid.medium = Some("unknown".into());
                    DenseRejectionReason::Residency
                }
                "layout" => {
                    invalid.kv_cache_spec_kind = Some("sliding_attention".into());
                    DenseRejectionReason::Layout
                }
                "sliding-window" => {
                    invalid.kv_cache_spec_sliding_window = Some(8);
                    DenseRejectionReason::Layout
                }
                "token-count" => {
                    invalid.block_hashes.push(ExternalBlockHash::Int(3));
                    DenseRejectionReason::TokenCountMismatch
                }
                _ => unreachable!(),
            };
            let batch = KVEventBatch {
                ts: 0.0,
                events: vec![
                    KVEvent::BlockStored(valid.clone()),
                    KVEvent::BlockStored(invalid),
                ],
                data_parallel_rank: Some(0),
            };
            assert_eq!(
                apply_batch(&index, &source, 0, 0, &batch, Some(4)),
                Err(DenseBatchRejection {
                    reason,
                    event_index: 1
                }),
                "{case}"
            );
            let query = |tokens: &[u32]| MatchQuery {
                group_idx: 0,
                local_hashes: crate::kv_index::local_block_hash_strings(tokens, 4, None),
                tiers_of_interest: vec![
                    StorageTier::Device,
                    StorageTier::HostPinned,
                    StorageTier::Disk,
                    StorageTier::External,
                ],
            };
            for tokens in [[1, 2, 3, 4], [5, 6, 7, 8]] {
                assert!(
                    index.find_matches(&query(&tokens)).is_empty(),
                    "rejected batch leaked credit: {case}"
                );
            }
            // The default generic subscriber keeps its wider structural domain.
            assert!(
                apply_batch(&index, &source, 0, 0, &batch, None).is_ok(),
                "{case}"
            );
            assert_eq!(
                index.find_matches(&query(&[5, 6, 7, 8])).len(),
                1,
                "generic: {case}"
            );
        }
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
    fn read_seq_big_endian_signed() {
        assert_eq!(read_seq(&(-1i64).to_be_bytes()), -1);
        assert_eq!(read_seq(&42i64.to_be_bytes()), 42);
    }

    #[test]
    fn read_seq_short_input_is_sentinel() {
        assert_eq!(read_seq(&[0u8, 1]), -1);
    }

    #[test]
    fn hex_encodes_bytes() {
        assert_eq!(hex(&[0u8, 255]), "00ff");
    }

    #[test]
    fn hash_str_int_and_bytes() {
        assert_eq!(&*hash_str(&ExternalBlockHash::Int(7)), "i:7");
        assert_eq!(
            &*hash_str(&ExternalBlockHash::Bytes(vec![0xab, 0xcd])),
            "b:abcd"
        );
    }

    #[test]
    fn local_hashes_chunks_by_block_size() {
        let ids = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let hashes = local_hashes(&ids, 2, None);
        assert_eq!(hashes.len(), 4);
        assert_eq!(hashes[0].len(), 32);
    }

    #[test]
    fn local_hashes_trailing_partial_block_included() {
        let ids = vec![1u32, 2, 3, 4, 5];
        let hashes = local_hashes(&ids, 2, None);
        assert_eq!(hashes.len(), 3);
    }

    #[test]
    fn local_hashes_block_size_zero_yields_empty() {
        assert!(local_hashes(&[1u32, 2, 3], 0, None).is_empty());
    }

    #[test]
    fn local_hashes_determinism() {
        let ids = vec![10u32, 20, 30, 40];
        assert_eq!(local_hashes(&ids, 2, None), local_hashes(&ids, 2, None));
    }

    #[test]
    fn local_hashes_seed_distinguishes_lora() {
        let ids = vec![1u32, 2, 3, 4];
        let base = local_hashes(&ids, 2, None);
        let lora = local_hashes(&ids, 2, Some("adapter-a"));
        assert_ne!(base, lora);
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
            &[KVEvent::BlockStored(BlockStored {
                extra_keys: None,
                block_hashes: vec![ExternalBlockHash::Int(100)],
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
            2,
            2.0,
            &[KVEvent::BlockRemoved(BlockRemoved {
                block_hashes: vec![ExternalBlockHash::Int(100)],
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
        let (tx, signals) = mpsc::channel(128);
        let source = EventSource {
            source: SourceId::from("w0"),
            dp_rank: 0,
            pub_endpoint,
            replay_endpoint: with_replay.then_some(replay_endpoint),
            topic: "kv".into(),
            hwm: None,
        };
        let handle = if let Some(binding) = binding {
            spawn_inner(
                source,
                indexer.clone(),
                Some(tx),
                ctx.clone(),
                expected_block_size,
                Some(binding),
            )
        } else if let Some(block_size) = expected_block_size {
            spawn_dense(source, indexer.clone(), Some(tx), ctx.clone(), block_size)
        } else {
            spawn(source, indexer.clone(), Some(tx), ctx.clone())
        };
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
    async fn bound_active_frames_preserve_generation_and_external_invalidation_recovers() {
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
        let generation = indexer.source_generation_for_test();
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
        assert_eq!(indexer.source_generation_for_test(), generation);
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
        assert_eq!(indexer.source_generation_for_test(), generation + 1);
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
        assert_eq!(indexer.source_generation_for_test(), generation + 2);
        assert!(hits(&indexer, &old).is_empty());
        assert!(hits(&indexer, &current).is_empty());
        assert_eq!(hits(&indexer, &fresh)[0].segments, [(0, 1)]);
        publish_until_advanced(&publisher, &mut signals, b"kv", 3, 3.0, &[], Some(0)).await;
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
            let mut f = fixture_with_binding(
                true,
                Some(4),
                Some(SubscriberBinding {
                    lease: SourceLease {
                        strict_rank: Arc::new(Mutex::new(true)),
                        active: Arc::new(AtomicBool::new(true)),
                    },
                    replay_cleanup_barrier: None,
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
            &[store_event(100, &old)],
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
                batch_payload(&[store_event(101, &partial)], 0),
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
            f.indexer.invalidate_source(&SourceId::from("w0"), 0);
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
            &[store_event(200, &replacement)],
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
        let mut f = fixture(false);
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
            &[store_event(101, &expected)],
            Some(0),
        )
        .await;
        assert!(hits(&f.indexer, &unrelated).is_empty());
        assert_eq!(hits(&f.indexer, &expected).len(), 1);
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
        rmp_serde::to_vec_named(&(1.0, events, Some(rank))).unwrap()
    }

    #[tokio::test]
    async fn dense_session_annotation_live_preserves_shared_blocks() {
        let mut f = fixture_with_block_size(false, Some(4));
        let shared = [1, 2, 3, 4];
        let retained = [5, 6, 7, 8];
        for (seq, session, blocks) in [
            (0, "session-a", vec![(100, shared), (101, retained)]),
            (1, "session-b", vec![(100, shared)]),
        ] {
            let events: Vec<_> = blocks
                .into_iter()
                .map(|(hash, tokens)| {
                    let mut event = stored(
                        vec![ExternalBlockHash::Int(hash)],
                        tokens.to_vec(),
                        Some("GPU"),
                        None,
                    );
                    event.session_id = Some(session.into());
                    KVEvent::BlockStored(event)
                })
                .collect();
            publish_until_advanced(
                &f.publisher,
                &mut f.signals,
                b"kv",
                seq,
                seq as f64,
                &events,
                Some(0),
            )
            .await;
            for tokens in [&shared, &retained] {
                let matches = hits(&f.indexer, tokens);
                assert_eq!(matches.len(), 1, "session annotation erased placement");
                assert_eq!(matches[0].target.instance_id.as_ref(), "w0");
                assert_eq!(matches[0].target.dp_rank, 0);
            }
        }
    }

    #[tokio::test]
    async fn dense_session_annotation_replay_applies_shared_blocks() {
        let f = fixture_with_block_size(true, Some(4));
        let tokens = [1, 2, 3, 4];
        let events: Vec<_> = ["session-a", "session-b"]
            .into_iter()
            .map(|session| {
                let mut event = stored(
                    vec![ExternalBlockHash::Int(100)],
                    tokens.to_vec(),
                    Some("GPU"),
                    None,
                );
                event.session_id = Some(session.into());
                KVEvent::BlockStored(event)
            })
            .collect();
        let replies = [(1, b"kv".as_slice(), batch_payload(&events, 0))];
        let endpoint = f
            .replay_server
            .as_ref()
            .unwrap()
            .get_last_endpoint()
            .unwrap()
            .unwrap();
        let source = SourceId::from("w0");
        let mut reported = false;
        let (result, ()) = tokio::join!(
            replay(
                &endpoint,
                0,
                1,
                &f._ctx,
                &f.indexer,
                &source,
                0,
                0,
                None,
                Some(4),
                "kv",
                &mut reported,
                None,
            ),
            serve_replay(f.replay_server.as_ref().unwrap(), &replies, &f.indexer, &[]),
        );
        assert_eq!(result, Some(1));
        assert_eq!(hits(&f.indexer, &tokens).len(), 1);
    }

    #[test]
    fn dense_block_size_mismatch_preflights_stores_and_removes_atomically() {
        let index = KvBlockIndexer::new();
        let source = SourceId::from("dense");
        let initial = KVEventBatch {
            ts: 0.0,
            events: vec![store_event(100, &[1, 2, 3, 4])],
            data_parallel_rank: Some(0),
        };
        assert!(apply_batch(&index, &source, 0, 0, &initial, Some(4)).is_ok());
        let mut bad = stored(
            vec![ExternalBlockHash::Int(102)],
            (20..28).collect(),
            Some("GPU"),
            None,
        );
        bad.block_size = 8;
        let batch = KVEventBatch {
            ts: 1.0,
            events: vec![
                store_event(101, &[5, 6, 7, 8]),
                KVEvent::BlockRemoved(BlockRemoved {
                    block_hashes: vec![ExternalBlockHash::Int(100)],
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
            Err(DenseBatchRejection {
                reason: DenseRejectionReason::BlockSizeMismatch,
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

    #[tokio::test]
    async fn dense_replay_block_size_mismatch_clears_partial_recovery() {
        for (expected, actual) in [(16, 8), (8, 16)] {
            let mut f = fixture_with_block_size(true, Some(expected));
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
        let mut f = fixture(true);
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
        let first = batch_payload(&[store_event(200, &partial)], 0);
        let eviction = batch_payload(
            &[KVEvent::BlockRemoved(BlockRemoved {
                block_hashes: vec![ExternalBlockHash::Int(100)],
                medium: Some("GPU".into()),
                group_idx: Some(0),
                locality: None,
                ownership: None,
            })],
            0,
        );
        let third = batch_payload(&[store_event(400, &fresh)], 0);
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
        assert_eq!(
            hits(&f.indexer, &fresh).len(),
            1,
            "fresh observations restore credit: {case}"
        );
    }

    #[tokio::test]
    async fn continuous_replay_preserves_valid_prefix_and_applies_lost_eviction() {
        check_replay_case("complete").await;
    }

    #[tokio::test]
    async fn replay_missing_start_clears_history() {
        check_replay_case("missing_start").await;
    }

    #[tokio::test]
    async fn replay_middle_gap_clears_partial_stores() {
        check_replay_case("middle_gap").await;
    }

    #[tokio::test]
    async fn replay_corrupt_payload_clears_partial_stores() {
        check_replay_case("corrupt").await;
    }

    #[tokio::test]
    async fn replay_early_eof_clears_partial_stores() {
        check_replay_case("early_eof").await;
    }

    #[tokio::test]
    async fn replay_wrong_rank_clears_partial_stores() {
        check_replay_case("wrong_rank").await;
    }

    #[tokio::test]
    async fn replay_wrong_topic_clears_partial_stores() {
        check_replay_case("wrong_topic").await;
    }

    #[test]
    fn dense_zero_block_size_is_rejected_before_mutation() {
        let index = KvBlockIndexer::new();
        let tokens = vec![1, 2, 3, 4];
        let mut event = stored(
            vec![ExternalBlockHash::Int(10)],
            tokens.clone(),
            Some("GPU"),
            None,
        );
        event.block_size = 0;
        let batch = KVEventBatch {
            ts: 1.0,
            events: vec![KVEvent::BlockStored(event)],
            data_parallel_rank: Some(0),
        };
        let rejection = Err(DenseBatchRejection {
            reason: DenseRejectionReason::ZeroBlockSize,
            event_index: 0,
        });
        assert_eq!(validate_dense_batch(&batch, 4), rejection);
        assert_eq!(
            apply_batch(&index, &SourceId::from("p"), 0, 0, &batch, Some(4)),
            rejection
        );
        // Explicit rejection is required even though zero-size hashing is empty.
        let query = MatchQuery {
            group_idx: 0,
            local_hashes: local_hashes(&tokens, 4, None)
                .iter()
                .map(|hash| Arc::from(hex(hash)))
                .collect(),
            tiers_of_interest: vec![StorageTier::Device],
        };
        assert_eq!(index.find_matches(&query).len(), 0);
    }
}
