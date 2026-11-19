//! Multi-tier KV cache placement index, fed by engine events.

pub mod request_hash;
mod routing_provider;
pub(crate) use routing_provider::PromptKey;
#[doc(hidden)]
pub use routing_provider::RequestHashCache;
pub(crate) mod source;

pub use source::discovery;
pub use source::events::{subscriber, wire};

/// Local observation ownership only; this is not an authenticated engine epoch.
/// Both ingestion paths use one allocator so delayed cleanup cannot cross modes.
/// Stream generations instead fence snapshots across resets of the same owner.
pub(crate) fn next_owner_incarnation() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let value = NEXT.fetch_add(1, Ordering::Relaxed);
    assert_ne!(value, u64::MAX, "KV owner generation exhausted");
    value
}

use sha2::{Digest, Sha256};
pub(crate) mod layout;
pub mod storage;
pub mod types;

pub use discovery::{
    CacheKey, KvEventSourceEntry, KvEventSourcesResponse, KvIndexSupervisor, WorkerKvInfo,
};
pub use layout::{
    common_reusable_tokens, AdapterCapability, CacheCapabilities, CacheGroup, GroupCapability,
    GroupRule, GroupView, RankCapability,
};
pub use source::binding::SourceBinding;
pub(crate) use source::endpoints::canonical_endpoint;
pub use source::events::decode::decode_batch;
pub use source::events::event_stream::{
    spawn_zmq, EventStream, EventSubscriberHandle, StreamError, StreamLease,
};
pub use source::SourceError;
pub use storage::{KvBlockIndexer, MatchQuery, TierMatch};
pub use types::{
    CacheOwnerId, ClearScope, Locality, ResidencyOwner, RoutableTarget, SourceId, StorageTier,
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

// Out-of-band loss fence: advisory queue overflow can never leave warm state trusted.
static SIGNAL_LOSS_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn signal_loss_epoch() -> u64 {
    SIGNAL_LOSS_EPOCH.load(std::sync::atomic::Ordering::Acquire)
}
pub(crate) fn record_signal_loss() {
    SIGNAL_LOSS_EPOCH.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
}

/// Ingestion progress and continuity signals. Loss also fences the index
/// synchronously, independently of this advisory channel.
#[derive(Debug, Clone)]
pub enum IngestionSignal {
    Invalidated {
        source: SourceId,
        dp_rank: u32,
        reason: &'static str,
    },
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    Advance {
        source: SourceId,
        last_seq: i64,
    },
    Gap {
        source: SourceId,
        from_seq: i64,
        to_seq: i64,
    },
    IncarnationReset {
        source: SourceId,
        incarnation: u64,
    },
}
