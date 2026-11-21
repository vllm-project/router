//! Multi-tier KV cache placement index, fed by engine events.

mod admit;
mod binding;
mod decode;

pub mod discovery;
mod endpoints;
pub mod indexer;
pub mod request_hash;
mod storage;
pub mod subscriber;
pub mod types;
pub mod wire;

pub use admit::validate_device_batch;
pub use decode::decode_batch;
pub use discovery::{
    CacheKey, KvEventSourceEntry, KvEventSourcesResponse, KvIndexSupervisor, WorkerKvInfo,
};
pub(crate) use endpoints::canonical_endpoint;
pub use endpoints::{parse_endpoint_mapping, resolve_endpoints};
pub use indexer::{KvBlockIndexer, MatchQuery, TierMatch};
pub(crate) use types::SourceRecoveryReason;
pub use types::{
    CacheOwnerId, ClearScope, HashMode, Locality, ResidencyOwner, RoutableTarget, SourceId,
    StorageTier,
};
#[cfg(all(test, feature = "cpu", feature = "ready"))]
pub(crate) use types::{SourceRecoveryState, SourceRecoveryStatus};
pub use wire::{AllBlocksCleared, BlockRemoved, BlockStored, KVEvent, KVEventBatch};

// Compatibility export; the consumer owns the port and the index implements it.
pub use crate::policies::TieredMatchProvider;

pub(crate) const DEFAULT_HASH_SEED: &str = "vllm-none-hash";

pub(crate) fn ordinary_block_keys(block_size: u32) -> Option<request_hash::BlockKeyGenerator> {
    use std::sync::OnceLock;
    static ROOT: OnceLock<request_hash::SequenceHash> = OnceLock::new();
    if block_size == 0 {
        return None;
    }
    let root = ROOT.get_or_init(|| {
        request_hash::BlockKeyGenerator::from_seed(1, DEFAULT_HASH_SEED).root_hash()
    });
    Some(request_hash::BlockKeyGenerator::from_root_hash(
        block_size as usize,
        *root,
    ))
}

pub(crate) fn ordinary_block_hashes(
    tokens: &[u32],
    block_size: u32,
) -> Vec<crate::kv_hash::SequenceHash> {
    ordinary_block_keys(block_size).map_or_else(Vec::new, |keys| keys.generate_block_keys(tokens))
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) fn test_store_event(
    tokens: &[u32],
    block_size: u32,
    medium: &str,
    owner: Option<&str>,
) -> rmpv::Value {
    use rmpv::Value;
    let hashes = ordinary_block_hashes(tokens, block_size)
        .into_iter()
        .map(|hash| Value::Binary(hash.to_vec()))
        .collect();
    let tokens = tokens.iter().copied().map(Value::from).collect();
    Value::Map(vec![
        ("type".into(), "BlockStored".into()),
        ("block_hashes".into(), Value::Array(hashes)),
        ("parent_block_hash".into(), Value::Nil),
        ("token_ids".into(), Value::Array(tokens)),
        ("block_size".into(), block_size.into()),
        ("medium".into(), medium.into()),
        ("lora_name".into(), Value::Nil),
        ("lora_id".into(), Value::Nil),
        ("ownership".into(), owner.map_or(Value::Nil, Value::from)),
    ])
}

/// Out-of-band loss fence: advisory queue overflow can never leave warm state trusted.
static SIGNAL_LOSS_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn signal_loss_epoch() -> u64 {
    SIGNAL_LOSS_EPOCH.load(std::sync::atomic::Ordering::Acquire)
}
pub(crate) fn record_signal_loss() {
    SIGNAL_LOSS_EPOCH.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
}

/// Raw signals ingestion emits; the trust arbiter consumes these. The
/// `EXACT`/`DEGRADED`/`FALLBACK` state machine is the trust arbiter's contract.
#[derive(Debug, Clone)]
pub enum IngestionSignal {
    Advance {
        source: SourceId,
        last_seq: i64,
    },
    Gap {
        source: SourceId,
        from_seq: i64,
        to_seq: i64,
    },
    ReplayApplied {
        source: SourceId,
        replay_seq: i64,
    },
    IncarnationReset {
        source: SourceId,
        incarnation: u64,
    },
}
