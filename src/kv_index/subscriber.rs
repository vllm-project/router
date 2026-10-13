//! ZMQ SUB + DEALER replay subscriber, one task per (worker, rank).

use std::sync::Arc;
use std::time::Duration;

use rmp_serde::from_slice;
use sha2::{Digest, Sha256};
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, info, warn};

use crate::kv_index::indexer::KvBlockIndexer;
use crate::kv_index::types::{ClearScope, ResidencyOwner, SourceId, StorageTier};
use crate::kv_index::wire::{ExternalBlockHash, KVEvent, KVEventBatch};
use crate::kv_index::IngestionSignal;

/// sha256 of each block's token-id bytes, seeded by `lora_name`.
/// The seed is re-applied per block (no parent chaining here) so identical
/// tokens under different adapters land on different nodes.
// ponytail: cache_salt/MM isolation deferred — needs raw msgpack capture of
// per-block extra_keys tuples (custom newtype or rmpv); add when cost model
// scores cross-salt collisions. Until then lora_name alone distinguishes adapters.
fn local_hashes(token_ids: &[u32], block_size: u32, lora_name: Option<&str>) -> Vec<[u8; 32]> {
    let bs = block_size as usize;
    if bs == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(token_ids.len().div_ceil(bs));
    for chunk in token_ids.chunks(bs) {
        let mut hasher = Sha256::new();
        if let Some(l) = lora_name {
            hasher.update(l.as_bytes());
        }
        for &tid in chunk {
            hasher.update(tid.to_le_bytes());
        }
        out.push(hasher.finalize().into());
    }
    out
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

        loop {
            if is_shutdown(&mut shutdown_rx) {
                info!("kv_index subscriber {} shutting down", source_id);
                break;
            }
            match sub.recv_multipart(zmq::DONTWAIT) {
                Ok(frames) => {
                    if frames.len() < 3 {
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
                        indexer.clear(
                            &owner_for(&source_id, dp_rank, incarnation),
                            ClearScope::Worker,
                        );
                        last_seq = -1;
                        emit(
                            &signals,
                            IngestionSignal::IncarnationReset {
                                source: source_id.clone(),
                                incarnation,
                            },
                        );
                    }
                    if last_seq != -1 && seq > last_seq + 1 {
                        let from = last_seq + 1;
                        let to = seq - 1;
                        debug!("kv_index {}: gap {}..{}", source_id, from, to);
                        emit(
                            &signals,
                            IngestionSignal::Gap {
                                source: source_id.clone(),
                                from_seq: from,
                                to_seq: to,
                            },
                        );
                        if let Some(ep) = replay_endpoint.as_deref() {
                            let applied = replay(
                                ep,
                                last_seq,
                                &ctx,
                                &indexer,
                                &source_id,
                                dp_rank,
                                incarnation,
                                signals.clone(),
                            )
                            .await;
                            if let Some(last_replay) = applied {
                                last_seq = last_replay;
                                emit(
                                    &signals,
                                    IngestionSignal::ReplayApplied {
                                        source: source_id.clone(),
                                        replay_seq: last_replay,
                                    },
                                );
                            }
                        }
                    }
                    if seq > last_seq {
                        if let Some(batch) = decode(&frames[2]) {
                            apply_batch(&indexer, &source_id, dp_rank, incarnation, &batch);
                        }
                        last_seq = seq;
                        emit(
                            &signals,
                            IngestionSignal::Advance {
                                source: source_id.clone(),
                                last_seq: seq,
                            },
                        );
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
/// A dropped signal under load is covered by the DEGRADED window.
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
    match from_slice::<KVEventBatch>(payload) {
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

fn apply_batch(
    indexer: &KvBlockIndexer,
    source_id: &SourceId,
    dp_rank: u32,
    incarnation: u64,
    batch: &KVEventBatch,
) {
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
}

/// Backfill `(last_seq, ∞)` from the replay ROUTER via a DEALER socket.
/// Frames with `seq <= last_seq` are skipped — deduped so an out-of-order
/// re-send can't re-apply a stale `remove` over a freshly re-`store`d block.
/// Returns the highest seq applied, for the `ReplayApplied` signal.
#[allow(clippy::too_many_arguments)]
async fn replay(
    endpoint: &str,
    last_seq: i64,
    ctx: &zmq::Context,
    indexer: &Arc<KvBlockIndexer>,
    source_id: &SourceId,
    dp_rank: u32,
    incarnation: u64,
    signals: Option<mpsc::Sender<IngestionSignal>>,
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
    let mut last_applied: Option<i64> = None;
    loop {
        if tokio::time::Instant::now() >= deadline {
            warn!("kv_index replay {} timed out", source_id);
            break;
        }
        match dealer.recv_multipart(zmq::DONTWAIT) {
            Ok(frames) => {
                // DEALER reply: [empty, topic, seq, payload]; seq is 2nd-from-last.
                if frames.len() < 4 {
                    break;
                }
                let seq = read_seq(&frames[frames.len() - 2]);
                if seq == END_SEQ {
                    break;
                }
                if seq <= high {
                    continue;
                }
                if let Some(batch) = decode(&frames[frames.len() - 1]) {
                    apply_batch(indexer, source_id, dp_rank, incarnation, &batch);
                }
                high = seq;
                last_applied = Some(seq);
                emit(
                    &signals,
                    IngestionSignal::Advance {
                        source: source_id.clone(),
                        last_seq: seq,
                    },
                );
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
    last_applied
}

fn hex(b: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        write!(s, "{:02x}", byte).unwrap();
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
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
        apply_batch(&idx, &SourceId::from("w0"), 0, 0, &batch);

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
        assert_eq!(res[0].matched_depth, 1);

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

    async fn publish_until_advanced(
        pub_sock: &zmq::Socket,
        sig_rx: &mut mpsc::Receiver<IngestionSignal>,
        topic: &[u8],
        seq: i64,
        ts: f64,
        events: &[KVEvent],
        rank: Option<u32>,
    ) {
        let payload = rmp_serde::to_vec(&(ts, events, &rank)).unwrap();
        let seq_bytes = seq.to_be_bytes();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let frames: Vec<&[u8]> = vec![topic, seq_bytes.as_slice(), payload.as_slice()];
            pub_sock.send_multipart(frames, 0).unwrap();
            match tokio::time::timeout(Duration::from_millis(300), sig_rx.recv()).await {
                Ok(Some(IngestionSignal::Advance { last_seq, .. })) if last_seq == seq => return,
                Ok(_) => continue,
                Err(_) if tokio::time::Instant::now() >= deadline => {
                    panic!("subscriber never advanced to seq {seq}")
                }
                Err(_) => continue,
            }
        }
    }
}
