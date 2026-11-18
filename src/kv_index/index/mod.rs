//! Concurrent radix-tree index of resident KV blocks, keyed by local hash.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::{mapref::entry::Entry, DashMap};
use parking_lot::{Mutex, RwLock};

use crate::kv_hash::SequenceHash;
mod anchors;
mod mutations;
mod prefix_tree;
mod records;
pub(crate) mod source;
mod transaction;

use self::anchors::AnchorSet;
pub(crate) use self::mutations::{PageBatch, PageWriteError, RemovedPages};
use self::prefix_tree::{edge_common_prefix, unique_alias_position, EdgeBlock, TreeNode};
pub(crate) use self::records::qualified_seq;
pub(crate) use self::records::TierBits;
use self::records::{namespaced_seq, sparse_key, unqualified_seq, ResidencyRecords};
pub(crate) use self::records::{prepare_sparse_keys, FactKey};
use self::source::{StreamControl, StreamState};
pub(crate) use self::transaction::{IndexBatch, IndexSnapshot};
use crate::kv_index::types::{
    ClearScope, Locality, ResidencyOwner, RoutableTarget, SourceId, StorageTier,
};

pub use crate::kv_index::types::{MatchQuery, TierMatch};

type OwnerTier = (ResidencyOwner, StorageTier);
type NodeRef = Arc<RwLock<TreeNode>>;
type IncomingBlock = (Arc<str>, Arc<str>); // (seq_hash, local_hash)
type Fast = foldhash::fast::RandomState;
type SequenceSet = HashSet<Arc<str>, Fast>;
type WaitingChildren = HashMap<Arc<str>, Arc<str>, Fast>;

fn same_clear_owner(owner: &ResidencyOwner, other: &ResidencyOwner) -> bool {
    match (owner, other) {
        (
            ResidencyOwner::Worker {
                source: s1,
                dp_rank: r1,
                ..
            },
            ResidencyOwner::Worker {
                source: s2,
                dp_rank: r2,
                ..
            },
        ) => s1 == s2 && r1 == r2,
        (
            ResidencyOwner::CacheOwner { pool_id: p1, .. },
            ResidencyOwner::CacheOwner { pool_id: p2, .. },
        ) => p1 == p2,
        _ => false,
    }
}
/// Derived position of a source-scoped alias in the compressed tree.
#[derive(Clone)]
struct Anchor {
    node: NodeRef,
    offset: usize,
}

/// One `group_idx`'s compressed radix tree + owner/tier claims for a single
/// kv-cache-spec group. Blocks from different groups never share a node because
/// they live in separate shards — the engine's sequence hash doesn't fold
/// `group_idx`, so the same tokens under two groups yield the same seq hash and
/// would clobber a shared anchor map; sharding keeps them apart.
struct GroupShard {
    /// Serializes topology-changing operations across different sources.
    topology: RwLock<()>,
    /// Tree root (sentinel: empty edge, no claims, only children).
    root: NodeRef,
    /// Engine seq hash → tree node and edge offset, so a store jumps to its parent and a
    /// `BlockRemoved` (carrying only seq hashes) finds the node to drop.
    // The same opaque integer may refer to different paths at two publishers.
    // Keep parent anchors scoped by the publishing target, not by hash alone.
    scoped_anchors: DashMap<Arc<str>, AnchorSet, Fast>,
    /// Children waiting for a source-scoped parent to acquire a tree position.
    waiting: DashMap<(RoutableTarget, Arc<str>), WaitingChildren, Fast>,
    /// Empty nodes are unlinked by the next topology writer.
    prunable: Mutex<Vec<NodeRef>>,
    /// One fact per owner and sequence hash, with the observed tiers as bits.
    reverse: ResidencyRecords,
}

impl GroupShard {
    fn new() -> Self {
        GroupShard {
            topology: RwLock::new(()),
            root: Arc::new(RwLock::new(TreeNode::root())),
            scoped_anchors: DashMap::with_hasher(Fast::default()),
            waiting: DashMap::with_hasher(Fast::default()),
            prunable: Mutex::new(Vec::new()),
            reverse: DashMap::new(),
        }
    }

    fn tier_sequences(&self, owner: &ResidencyOwner, tier: StorageTier) -> Vec<Arc<str>> {
        self.reverse
            .get(owner)
            .map(|facts| {
                facts
                    .iter()
                    .filter(|(_, tiers)| tiers.contains(tier))
                    .map(|(hash, _)| hash.to_seq())
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Serializes one worker/rank across its cache groups.
struct SourceLock {
    target: Arc<RoutableTarget>,
    access: RwLock<SourceData>,
    generation: AtomicU64,
}

#[derive(Default)]
struct SourceData {
    records: SourceRecords,
    streams: HashMap<ResidencyOwner, StreamState>,
}

#[derive(Default)]
struct SourceRecords {
    owners: HashMap<ResidencyOwner, OwnerMetadata>,
    reclaim: HashMap<u32, SequenceSet, Fast>,
}

#[derive(Default)]
struct OwnerMetadata {
    // Unbounded path writes invalidate this count; bounded batches maintain it.
    base_pages: Option<usize>,
    namespaces: HashMap<String, usize>,
}

/// The same record transition applies to a navigable page and a hash-only page.
struct StoreClaim {
    key: OwnerTier,
    incremental: bool,
}

/// Identity and trust generation of one source at snapshot time.
pub(crate) struct SourceVersion {
    target: RoutableTarget,
    source: Arc<SourceLock>,
    generation: u64,
}

impl SourceLock {
    fn new(target: RoutableTarget) -> Self {
        Self {
            target: Arc::new(target),
            access: RwLock::new(SourceData::default()),
            generation: AtomicU64::new(0),
        }
    }
}

/// One shared fact index per cache identity, sharded by cache group.
/// Implements `TieredMatchProvider` directly for full attention.
pub struct KvBlockIndexer {
    /// One shard per `group_idx` (kv-cache-spec group).
    // ponytail: group_idx is None on the wire → 0; single-spec models have one
    // group and multi-spec always emits group_idx, so 0 is a safe default.
    shards: DashMap<u32, Arc<GroupShard>>,
    suspended_sources: DashMap<(SourceId, u32), ()>,
    /// Gates may be removed only by `retire_source` under the global barrier.
    source_locks: DashMap<RoutableTarget, Arc<SourceLock>>,
    /// Ordinary access shares the barrier; pool clear and retirement exclude it.
    domain: RwLock<()>,
}

impl KvBlockIndexer {
    pub fn new() -> Self {
        KvBlockIndexer {
            shards: DashMap::new(),
            suspended_sources: DashMap::new(),
            source_locks: DashMap::new(),
            domain: RwLock::new(()),
        }
    }

    /// Get-or-create the shard for `group_idx`. Fast path takes a read lock so
    /// the hot query path doesn't serialize on shard creation.
    fn shard(&self, group_idx: u32) -> Arc<GroupShard> {
        if let Some(s) = self.shards.get(&group_idx) {
            return Arc::clone(&s);
        }
        self.shards
            .entry(group_idx)
            .or_insert_with(|| Arc::new(GroupShard::new()))
            .clone()
    }

    fn source_lock(&self, target: &RoutableTarget) -> Arc<SourceLock> {
        if let Some(source) = self.source_locks.get(target) {
            return source.clone();
        }
        self.source_locks
            .entry(target.clone())
            .or_insert_with(|| Arc::new(SourceLock::new(target.clone())))
            .clone()
    }

    /// Check a snapshot after dropping its locks. Ordinary store/remove is a
    /// valid snapshot transition; suspension, resume, or retirement invalidates
    /// the result. This does not reserve engine blocks.
    pub(crate) fn source_version_is_current(&self, version: &SourceVersion) -> bool {
        // Resume and retirement increment the generation before revealing
        // the source. Read it last to fence snapshots across that transition.
        !self
            .suspended_sources
            .contains_key(&(version.target.instance_id.clone(), version.target.dp_rank))
            && version.source.generation.load(Ordering::Acquire) == version.generation
    }

    pub fn store(
        &self,
        group_idx: u32,
        owner: ResidencyOwner,
        tier: StorageTier,
        parent_seq: Option<Arc<str>>,
        blocks: &[(Arc<str>, Arc<str>)],
    ) {
        let target = owner.target();
        self.with_source_batch(&target, |batch| {
            batch.store(group_idx, owner, tier, parent_seq, blocks)
        });
    }

    pub fn remove(
        &self,
        group_idx: u32,
        owner: &ResidencyOwner,
        tier: StorageTier,
        seq_hashes: &[Arc<str>],
    ) {
        let target = owner.target();
        self.with_source_batch(&target, |batch| {
            batch.remove(group_idx, owner, tier, seq_hashes)
        });
    }

    /// Clear one owner's residency across every group. A worker reset /
    /// `AllBlocksCleared` spans all kv-cache-spec groups the worker holds.
    pub fn clear(&self, owner: &ResidencyOwner, scope: ClearScope) {
        if matches!(owner, ResidencyOwner::Worker { .. }) && scope == ClearScope::Worker {
            self.with_source_batch(&owner.target(), |batch| batch.clear(owner, scope));
        } else {
            self.with_batch(|batch| batch.clear(owner, scope));
        }
    }

    /// Hide a source's observations while contiguous replay repairs its state.
    /// Correctness does not depend on delivery of advisory ingestion signals.
    pub fn suspend_source(&self, source: &SourceId, dp_rank: u32) {
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        let _domain = self.domain.read();
        let source_lock = self.source_lock(&target);
        let _guard = source_lock.access.write();
        self.suspend_source_unlocked(source, dp_rank);
    }

    fn suspend_source_unlocked(&self, source: &SourceId, dp_rank: u32) {
        self.suspended_sources.insert((source.clone(), dp_rank), ());
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        self.source_lock(&target)
            .generation
            .fetch_add(1, Ordering::AcqRel);
    }

    /// Stop crediting a lossy source and discard all its old observations.
    pub fn invalidate_source(&self, source: &SourceId, dp_rank: u32) {
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        let _domain = self.domain.read();
        let source_lock = self.source_lock(&target);
        let mut data = source_lock.access.write();
        self.invalidate_source_unlocked(source, dp_rank, &mut data.records);
    }

    /// Permanently retire a fenced source. The global barrier drains all
    /// transactions before removing its gate; in-flight owned snapshots retain
    /// the old gate and observe its incremented generation.
    pub(crate) fn retire_source(&self, source: &SourceId, dp_rank: u32) {
        let _domain = self.domain.write();
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        let gate = self.source_lock(&target);
        let mut data = gate.access.write();
        self.invalidate_source_unlocked(source, dp_rank, &mut data.records);
        drop(data);
        self.source_locks.remove(&target);
        self.suspended_sources.remove(&(source.clone(), dp_rank));
    }

    fn invalidate_source_unlocked(
        &self,
        source: &SourceId,
        dp_rank: u32,
        records: &mut SourceRecords,
    ) {
        self.suspend_source_unlocked(source, dp_rank);
        let shards: Vec<Arc<GroupShard>> = self.shards.iter().map(|e| e.value().clone()).collect();
        for shard in shards {
            let _topology = shard.topology.write();
            shard.clear_source(source, dp_rank);
        }
        records.owners.clear();
        records.reclaim.clear();
    }

    /// Enable fresh observations after complete replay, or after discarding
    /// incomplete replay and applying a new valid live batch to a clean source.
    /// Already-active sources must not invalidate concurrent query snapshots.
    pub fn resume_source(&self, source: &SourceId, dp_rank: u32) {
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        let _domain = self.domain.read();
        let source_lock = self.source_lock(&target);
        let _guard = source_lock.access.write();
        self.resume_source_unlocked(source, dp_rank);
    }

    fn resume_source_unlocked(&self, source: &SourceId, dp_rank: u32) {
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        // Never hold a suspended-sources map entry while acquiring a
        // source-lock map entry: final query checks access both maps.
        let gate = self.source_lock(&target);
        if let Entry::Occupied(entry) = self.suspended_sources.entry((source.clone(), dp_rank)) {
            // Fence the visibility transition before removing its hidden state.
            gate.generation.fetch_add(1, Ordering::AcqRel);
            entry.remove();
        }
    }

    pub fn find_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
        let _domain = self.domain.read();
        let mut sources: Vec<_> = self
            .source_locks
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        sources.sort_unstable_by(|a, b| {
            a.0.instance_id
                .cmp(&b.0.instance_id)
                .then(a.0.dp_rank.cmp(&b.0.dp_rank))
        });
        let _guards: Vec<_> = sources
            .iter()
            .map(|(_, source)| source.access.read())
            .collect();
        let snapshots: HashMap<_, _> = sources
            .iter()
            .map(|(target, source)| {
                (
                    target.clone(),
                    SourceVersion {
                        target: target.clone(),
                        source: source.clone(),
                        generation: source.generation.load(Ordering::Acquire),
                    },
                )
            })
            .collect();
        let shard = self.shard(query.group_idx);
        let topology = shard.topology.read();
        // A target absent from this source snapshot, suspended, or invalidated
        // before the result is returned is filtered below. Doing the same
        // lookup for every page adds O(pages * owners) duplicate checks.
        let node_claims = shard.collect_claims(query);
        drop(topology);
        drop(_guards);
        drop(_domain);
        GroupShard::aggregate_claims(query, &node_claims)
            .into_iter()
            .filter(|hit| {
                snapshots
                    .get(&hit.target)
                    .is_some_and(|version| self.source_version_is_current(version))
            })
            .collect()
    }

    /// Run a whole publisher batch under one domain write lock.
    pub(crate) fn with_batch<R>(&self, f: impl FnOnce(&mut IndexBatch<'_>) -> R) -> R {
        let _guard = self.domain.write();
        f(&mut IndexBatch {
            index: self,
            scope: None,
            source_data: None,
        })
    }

    /// A source batch is atomic for this target without blocking other targets.
    pub(crate) fn with_source_batch<R>(
        &self,
        target: &RoutableTarget,
        f: impl FnOnce(&mut IndexBatch<'_>) -> R,
    ) -> R {
        let _domain = self.domain.read();
        let source = self.source_lock(target);
        let mut data = source.access.write();
        f(&mut IndexBatch {
            index: self,
            scope: Some(target.clone()),
            source_data: Some(&mut data.records),
        })
    }

    /// Apply a stream batch and its cache mutations under the same source lock.
    pub(crate) fn with_stream_batch<R>(
        &self,
        target: &RoutableTarget,
        owner: &ResidencyOwner,
        control: &Arc<StreamControl>,
        f: impl FnOnce(&mut StreamState, &mut IndexBatch<'_>) -> R,
    ) -> R {
        self.stream_batch(target, owner, control, true, f).unwrap()
    }

    /// Cleanup must not recreate a gate or owner that discovery already retired.
    pub(crate) fn with_existing_stream_batch<R>(
        &self,
        target: &RoutableTarget,
        owner: &ResidencyOwner,
        control: &Arc<StreamControl>,
        f: impl FnOnce(&mut StreamState, &mut IndexBatch<'_>) -> R,
    ) -> Option<R> {
        self.stream_batch(target, owner, control, false, f)
    }

    fn stream_batch<R>(
        &self,
        target: &RoutableTarget,
        owner: &ResidencyOwner,
        control: &Arc<StreamControl>,
        create: bool,
        f: impl FnOnce(&mut StreamState, &mut IndexBatch<'_>) -> R,
    ) -> Option<R> {
        assert_eq!(owner.target(), *target, "stream crossed targets");
        let _domain = self.domain.read();
        let source = if create {
            self.source_lock(target)
        } else {
            self.source_locks.get(target)?.clone()
        };
        let mut data = source.access.write();
        let SourceData { records, streams } = &mut *data;
        let state = if create {
            streams
                .entry(owner.clone())
                .or_insert_with(|| StreamState::new(control.clone()))
        } else {
            streams.get_mut(owner)?
        };
        assert!(
            state.has_control(control),
            "stream owner reused with another control"
        );
        let mut tx = IndexBatch {
            index: self,
            scope: Some(target.clone()),
            source_data: Some(records),
        };
        let result = f(state, &mut tx);
        if state.lease_id == 0 && state.next_seq.is_none() && !state.visible() {
            streams.remove(owner);
        }
        Some(result)
    }

    pub(crate) fn stream_lease_is_active(&self, owner: &ResidencyOwner, lease: u64) -> bool {
        let _domain = self.domain.read();
        let Some(source) = self
            .source_locks
            .get(&owner.target())
            .map(|entry| entry.clone())
        else {
            return false;
        };
        let data = source.access.read();
        data.streams.get(owner).is_some_and(|state| {
            state.lease_id == lease && state.phase != source::StreamPhase::Retired
        })
    }

    /// Read all groups for one target from the same source transaction.
    pub(crate) fn with_source_snapshot<R>(
        &self,
        target: &RoutableTarget,
        f: impl FnOnce(&IndexSnapshot<'_>) -> R,
    ) -> R {
        let _domain = self.domain.read();
        let source = self.source_lock(target);
        let data = source.access.read();
        f(&IndexSnapshot {
            index: self,
            scope: Some(target.clone()),
            source_data: Some(&data),
            source: Some(&source),
        })
    }
}

const TIERS: [StorageTier; 4] = [
    StorageTier::Device,
    StorageTier::HostPinned,
    StorageTier::Disk,
    StorageTier::External,
];

impl Default for KvBlockIndexer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests;
