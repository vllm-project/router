//! Multi-tier KV cache placement index, fed by engine events.

pub mod discovery;
pub mod indexer;
pub mod subscriber;
pub mod types;
pub mod wire;

pub use discovery::{
    CacheKey, KvEventSourceEntry, KvEventSourcesResponse, KvIndexSupervisor, WorkerKvInfo,
};
pub use indexer::{KvBlockIndexer, MatchQuery, TierMatch};
pub use types::{
    CacheOwnerId, ClearScope, HashMode, Locality, ResidencyOwner, RoutableTarget, SourceId,
    StorageTier,
};
pub use wire::{AllBlocksCleared, BlockRemoved, BlockStored, KVEvent, KVEventBatch};

/// Read-only tiered-match contract the policy and cost model consume.
/// `KvBlockIndexer` implements this.
pub trait TieredMatchProvider: Send + Sync {
    fn find_tiered_matches(&self, query: &MatchQuery) -> Vec<TierMatch>;
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
