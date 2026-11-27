//! Multi-tier KV cache placement index, fed by engine events.

pub(crate) mod config;
pub use config::apply_kv_layout_file;
pub mod request_hash;
mod routing_provider;
pub use routing_provider::TieredMatchProvider;
pub(crate) use routing_provider::{PlacementEvidence, PlacementPinError, PromptKey};
#[doc(hidden)]
pub use routing_provider::{RequestHashCache, SharedPrefixLookup};
pub(crate) mod source;

pub use source::discovery;
pub use source::events::{subscriber, wire};

/// Local observation ownership only; this is not an authenticated engine epoch.
/// Stream generations instead fence snapshots across resets of the same owner.
pub(crate) fn next_owner_incarnation() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let value = NEXT.fetch_add(1, Ordering::Relaxed);
    assert_ne!(value, u64::MAX, "KV owner generation exhausted");
    value
}

pub(crate) mod layout;
pub mod storage;
pub(crate) use storage::SourceReadGuard;
pub mod types;

pub use discovery::{
    CacheKey, KvEventSourceEntry, KvEventSourcesResponse, KvIndexSupervisor, WorkerKvInfo,
};
pub use layout::{
    AdapterCapability, CacheCapabilities, CacheGroup, GroupCapability, GroupRule, RankCapability,
};
pub use source::binding::SourceBinding;
pub(crate) use source::endpoints::canonical_endpoint;
pub use source::events::decode::decode_batch;
pub use source::events::event_stream::{
    EventStream, EventSubscriberHandle, StreamError, StreamLease,
};
pub use source::SourceError;
pub use storage::KvBlockIndexer;
pub use types::{ResidencyOwner, RoutableTarget, SourceId, StorageTier, TierMatch};
pub use wire::{AllBlocksCleared, BlockRemoved, BlockStored, KVEvent, KVEventBatch};

// Out-of-band loss fence: advisory queue overflow can never leave warm state trusted.
static SIGNAL_LOSS_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn signal_loss_epoch() -> u64 {
    SIGNAL_LOSS_EPOCH.load(std::sync::atomic::Ordering::Acquire)
}
pub(crate) fn record_signal_loss() {
    SIGNAL_LOSS_EPOCH.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
}

/// Ingestion continuity signals. Loss also fences the index
/// synchronously, independently of this advisory channel.
#[derive(Debug, Clone)]
pub enum IngestionSignal {
    Invalidated {
        source: SourceId,
        dp_rank: u32,
        reason: &'static str,
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
