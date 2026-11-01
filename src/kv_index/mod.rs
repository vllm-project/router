//! Multi-tier KV cache placement index, fed by engine events.

mod admit;
mod binding;
mod decode;

use std::sync::Arc;

use sha2::{Digest, Sha256};
pub mod discovery;
mod endpoints;
pub mod indexer;
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
pub use subscriber::local_hashes;
pub use types::{
    CacheOwnerId, ClearScope, HashMode, Locality, ResidencyOwner, RoutableTarget, SourceId,
    StorageTier,
};
pub use wire::{AllBlocksCleared, BlockRemoved, BlockStored, KVEvent, KVEventBatch};

// Compatibility export; the consumer owns the port and the index implements it.
pub use crate::policies::TieredMatchProvider;

/// SHA256 of each token block, seeded independently with the LoRA identity.
pub(crate) fn local_block_hashes(
    token_ids: &[u32],
    block_size: u32,
    lora_name: Option<&str>,
) -> Vec<[u8; 32]> {
    let block_size = block_size as usize;
    if block_size == 0 {
        return Vec::new();
    }
    token_ids
        .chunks(block_size)
        .map(|chunk| {
            let mut hasher = Sha256::new();
            if let Some(name) = lora_name.filter(|name| !name.is_empty()) {
                hasher.update(name.as_bytes());
            }
            for &token_id in chunk {
                hasher.update(token_id.to_le_bytes());
            }
            hasher.finalize().into()
        })
        .collect()
}

pub(crate) fn local_block_hash_strings(
    token_ids: &[u32],
    block_size: u32,
    lora_name: Option<&str>,
) -> Vec<Arc<str>> {
    local_block_hashes(token_ids, block_size, lora_name)
        .into_iter()
        .map(|digest| {
            let mut value = String::with_capacity(digest.len() * 2);
            for byte in digest {
                use std::fmt::Write;
                let _ = write!(value, "{byte:02x}");
            }
            Arc::from(value)
        })
        .collect()
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
