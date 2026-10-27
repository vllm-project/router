//! Concurrent radix-tree index of resident KV blocks, keyed by local hash.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::{mapref::entry::Entry, DashMap};
use parking_lot::RwLock;

use crate::kv_index::types::{
    ClearScope, Locality, ResidencyOwner, RoutableTarget, SourceId, StorageTier,
};

/// One request's query against the index.
#[derive(Debug, Clone)]
pub struct MatchQuery {
    /// kv-cache-spec group to query — selects the shard within the indexer.
    pub group_idx: u32,
    /// Block-local hashes, walked from the shard root in prefix order.
    pub local_hashes: Vec<Arc<str>>,
    /// Tiers to report; the cost model requests the tiers it scores.
    pub tiers_of_interest: Vec<StorageTier>,
}

/// One hit: the block intervals `[start, end)` a target holds at this tier,
/// each maximal. Flat — one row per `(target, tier)` — so the cost model
/// applies two-segment reuse over a target's tiers in one pass.
#[derive(Debug, Clone)]
pub struct TierMatch {
    pub target: RoutableTarget,
    pub tier: StorageTier,
    /// Held block intervals `[start, end)`, each maximal; a gap closes a
    /// segment but later segments survive.
    pub segments: Vec<(u32, u32)>,
    pub locality: Locality,
}

type OwnerTier = (ResidencyOwner, StorageTier);
type NodeRef = Arc<RwLock<TreeNode>>;
type IncomingBlock = (Arc<str>, Arc<str>); // (seq_hash, local_hash)
type EdgeBlock = (HashSet<Arc<str>>, Arc<str>); // (historical seq hashes, local_hash)

#[cfg(test)]
std::thread_local! {
    static AFTER_ANCHOR_CLONE: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    static AFTER_QUERY_EDGE: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn after_anchor_clone() {
    let hook = AFTER_ANCHOR_CLONE.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

#[cfg(test)]
fn after_query_edge() {
    let hook = AFTER_QUERY_EDGE.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Length of the common prefix between an edge and the incoming blocks,
/// compared by local hash only.
fn edge_common_prefix(edge: &[EdgeBlock], incoming: &[IncomingBlock]) -> usize {
    edge.iter()
        .zip(incoming.iter())
        .take_while(|(e, b)| e.1 == b.1)
        .count()
}

/// One node in the compressed radix tree. An edge is a run of one or more
/// blocks; the path from root to a node encodes its chain position, so
/// identical tokens at different positions land on different nodes. Each edge
/// block carries its own claim set — residency is per block, so a `remove` of
/// one block never disturbs the claims on its siblings.
struct TreeNode {
    /// Edge blocks with historical identities; empty on the root sentinel.
    /// Topology and parent anchors survive removal of residency claims.
    /// Historical aliases are retained without a memory bound.
    edge: Vec<EdgeBlock>,
    /// Per-block residency claims, parallel to `edge`; `claims[i]` owns
    /// `edge[i]`. Empty on the root sentinel.
    claims: Vec<Vec<OwnerTier>>,
    /// Children keyed by the first local hash of each child's edge.
    children: HashMap<Arc<str>, NodeRef>,
}

impl TreeNode {
    fn root() -> Self {
        TreeNode {
            edge: Vec::new(),
            claims: Vec::new(),
            children: HashMap::new(),
        }
    }
    fn with_edge(blocks: &[IncomingBlock]) -> Self {
        let claims = (0..blocks.len()).map(|_| Vec::new()).collect();
        TreeNode {
            edge: blocks
                .iter()
                .map(|(seq, local)| (HashSet::from([seq.clone()]), local.clone()))
                .collect(),
            claims,
            children: HashMap::new(),
        }
    }
}

/// One `group_idx`'s compressed radix tree + owner/tier claims for a single
/// kv-cache-spec group. Blocks from different groups never share a node because
/// they live in separate shards — the engine's sequence hash doesn't fold
/// `group_idx`, so the same tokens under two groups yield the same seq hash and
/// would clobber a shared anchor map; sharding keeps them apart.
struct GroupShard {
    /// Tree root (sentinel: empty edge, no claims, only children).
    root: NodeRef,
    /// Engine seq hash → tree node, so a store jumps to its parent and a
    /// `BlockRemoved` (carrying only seq hashes) finds the node to drop.
    // ponytail: never trimmed — topology outlives residency; unique-prefix churn
    // grows topology and historical alias sets without limit. Revisit with an
    // epoch leaf sweep if growth shows.
    anchors: DashMap<Arc<str>, NodeRef>,
    /// Reverse lookup: (owner, tier) → seq hashes claimed.
    reverse: DashMap<OwnerTier, HashSet<Arc<str>>>,
    /// Per-(owner, tier) count of distinct active engine sequence observations.
    /// Multiple aliases of one logical block remain distinct observations.
    counts: DashMap<OwnerTier, usize>,
}

impl GroupShard {
    fn new() -> Self {
        GroupShard {
            root: Arc::new(RwLock::new(TreeNode::root())),
            anchors: DashMap::new(),
            reverse: DashMap::new(),
            counts: DashMap::new(),
        }
    }

    /// Idempotent link of a batch to (owner, tier). Each block is a
    /// `(seq_hash, local_hash)` pair. Starting at the parent anchor (or root),
    /// split after an interior parent block before following its children.
    /// Incoming blocks are matched against each child's edge: on a full
    /// edge match we record the edge and descend; on a partial match we split
    /// the edge at the divergence, record the shared prefix, then append a new
    /// child for the diverging remainder; on no child we append a new child
    /// edge. The seq hash records into `anchors` for `remove`/parent
    /// resolution. The same block on a second tier is a separate claim — no
    /// cross-tier dedup.
    pub fn store(
        &self,
        owner: ResidencyOwner,
        tier: StorageTier,
        parent_seq: Option<Arc<str>>,
        blocks: &[IncomingBlock],
    ) {
        let key = (owner, tier);
        // An unknown ancestor cannot be replaced by root: block-local hashes
        // do not prove that this suffix is a stand-alone request prefix.
        if let Some(seq) = &parent_seq {
            if !self.anchors.contains_key(seq) {
                tracing::debug!(parent_seq = %seq, "kv_index: unknown parent; skipping unanchored store");
                return;
            }
        }
        let mut cursor = parent_seq;
        let mut bi = 0;
        while bi < blocks.len() {
            let node = match &cursor {
                Some(seq) => self
                    .anchors
                    .get(seq)
                    .map(|node| node.clone())
                    .expect("known cursor must remain anchored; anchors are insert-only"),
                None => self.root.clone(),
            };
            #[cfg(test)]
            after_anchor_clone();
            let mut parent = node.write();
            if let Some(seq) = &cursor {
                let Some(pos) = parent
                    .edge
                    .iter()
                    .position(|(aliases, _)| aliases.contains(seq))
                else {
                    // The same split-only progress argument as remove_claim
                    // applies to this fixed cursor; resolve its moved anchor.
                    drop(parent);
                    std::thread::yield_now();
                    continue;
                };
                if pos + 1 < parent.edge.len() {
                    self.split_locked(&mut parent, pos + 1);
                }
            }
            let child = parent.children.get(&blocks[bi].1).cloned();
            let Some(child) = child else {
                // Keep the parent locked from lookup through publication so
                // another writer cannot replace a newly installed child.
                let new_node = Arc::new(RwLock::new(TreeNode::with_edge(&blocks[bi..])));
                let mut new = new_node.write();
                self.record_locked(&mut new, &new_node, &key, &blocks[bi..]);
                parent
                    .children
                    .insert(blocks[bi].1.clone(), new_node.clone());
                return;
            };
            drop(parent);
            // Match, split, and claim under one write lock: a concurrent split
            // can't shorten this edge between the match and the split.
            let mut c = child.write();
            let matched = edge_common_prefix(&c.edge, &blocks[bi..]);
            debug_assert!(matched >= 1, "child key matches its first edge block");
            if matched < c.edge.len() {
                self.split_locked(&mut c, matched);
            }
            self.record_locked(&mut c, &child, &key, &blocks[bi..bi + matched]);
            bi += matched;
            // record_locked registered this incoming key, so its current
            // anchor remains a valid cursor even if a later split moves it.
            cursor = Some(blocks[bi - 1].0.clone());
        }
    }

    /// Record incoming identities into anchors, reverse, counts, and
    /// attach `(owner, tier)` as a per-block claim. Idempotent per
    /// (owner, tier, block). Caller holds `node`'s write lock.
    fn record_locked(
        &self,
        n: &mut TreeNode,
        node: &NodeRef,
        key: &OwnerTier,
        incoming: &[IncomingBlock],
    ) {
        for (i, (seq, _)) in incoming.iter().enumerate() {
            n.edge[i].0.insert(seq.clone());
            self.anchors.insert(seq.clone(), node.clone());
            if self
                .reverse
                .entry(key.clone())
                .or_default()
                .insert(seq.clone())
            {
                *self.counts.entry(key.clone()).or_insert(0) += 1;
            }
            if !n.claims[i].contains(key) {
                n.claims[i].push(key.clone());
            }
        }
    }

    /// Split `child`'s edge at `pos`: the first `pos` blocks stay on `child`
    /// with their claims, the remainder becomes a single new child carrying
    /// the remainder's claims and the old children. The remainder's seq hashes
    /// are re-anchored to the new node so `remove`/`clear` still find them.
    /// Called under the caller's write lock on `child`.
    fn split_locked(&self, c: &mut TreeNode, pos: usize) {
        let remainder: Vec<EdgeBlock> = c.edge.split_off(pos);
        let rem_claims: Vec<Vec<OwnerTier>> = c.claims.split_off(pos);
        let old_children = std::mem::take(&mut c.children);
        let rem_first = remainder[0].1.clone();
        let rem_node = Arc::new(RwLock::new(TreeNode {
            edge: remainder,
            claims: rem_claims,
            children: old_children,
        }));
        // An anchor exposes the remainder to other writers. Keep it locked
        // through all re-anchoring and publication so a split cannot race us.
        let r = rem_node.write();
        for (aliases, _) in &r.edge {
            for seq in aliases {
                self.anchors.insert(seq.clone(), rem_node.clone());
            }
        }
        c.children.insert(rem_first, rem_node.clone());
    }

    /// Drop `owner`'s claim on `seq_hashes` at `tier`. Each seq hash resolves
    /// to its node via `anchors` and its index in that node's edge; a miss
    /// (block not stored here, or its node already dropped) is skipped. Other
    /// owners/tiers survive.
    pub fn remove(&self, owner: &ResidencyOwner, tier: StorageTier, seq_hashes: &[Arc<str>]) {
        let key = (owner.clone(), tier);
        for seq in seq_hashes {
            let mut removed = false;
            if let Some(mut claims) = self.reverse.get_mut(&key) {
                if claims.remove(seq) {
                    removed = true;
                }
            }
            self.remove_claim(seq, &key);
            if removed {
                if let Some(mut c) = self.counts.get_mut(&key) {
                    if *c > 0 {
                        *c -= 1;
                    }
                }
                // Atomic: a concurrent store bumping back above zero wins.
                self.counts.remove_if(&key, |_, c| *c == 0);
            }
        }
    }

    /// Clear `owner`'s residency in `scope`. `Worker` never crosses into
    /// `CacheOwner`: the scope filters by domain, then the owner is matched
    /// within its domain (Worker by source+dp_rank, CacheOwner by pool_id).
    pub fn clear(&self, owner: &ResidencyOwner, scope: ClearScope) {
        let keys: Vec<OwnerTier> = self.reverse.iter().map(|e| e.key().clone()).collect();
        for key in keys {
            if !matches!(scope, ClearScope::All) && scope != key.0.domain() {
                continue;
            }
            let same = match (owner, &key.0) {
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
            };
            if !same {
                continue;
            }
            let seqs: Vec<Arc<str>> = self
                .reverse
                .get(&key)
                .map(|seqs| seqs.iter().cloned().collect())
                .unwrap_or_default();
            self.remove(&key.0, key.1, &seqs);
            self.reverse.remove_if(&key, |_, seqs| seqs.is_empty());
            self.counts.remove_if(&key, |_, count| *count == 0);
        }
    }

    /// Forget only this publisher/rank's observations, including pool copies
    /// attributed to it. A pool-wide clear would erase other publishers too.
    fn clear_source(&self, source: &SourceId, dp_rank: u32) {
        let keys: Vec<OwnerTier> = self
            .reverse
            .iter()
            .filter(|entry| {
                let target = entry.key().0.target();
                target.instance_id == *source && target.dp_rank == dp_rank
            })
            .map(|entry| entry.key().clone())
            .collect();
        for key in keys {
            let seqs: Vec<Arc<str>> = self
                .reverse
                .get(&key)
                .map(|seqs| seqs.iter().cloned().collect())
                .unwrap_or_default();
            self.remove(&key.0, key.1, &seqs);
            self.reverse.remove_if(&key, |_, seqs| seqs.is_empty());
        }
    }
    /// Drop the logical block claim only after its last active alias is gone.
    fn remove_claim(&self, seq: &Arc<str>, key: &OwnerTier) {
        // Contract: within a shard an engine key always identifies the same
        // full logical path. Topology only splits, never merges or prepends.
        // Moving a fixed alias at position p cuts a nonempty prefix k, leaving
        // p-k; thus only finitely many split-induced re-resolutions are possible.
        // This does not bound elapsed time or lock starvation. Do not cap retries:
        // reverse already lost this identity, so abandoned cleanup leaves a claim
        // that Clear cannot enumerate.
        loop {
            // Never hold an anchors map guard while taking the node lock.
            let Some(node) = self.anchors.get(seq).map(|n| n.clone()) else {
                return;
            };
            #[cfg(test)]
            after_anchor_clone();
            let mut n = node.write();
            let Some(idx) = n.edge.iter().position(|(aliases, _)| aliases.contains(seq)) else {
                // A concurrent split moved this alias; resolve it again.
                drop(n);
                std::thread::yield_now();
                continue;
            };
            let active = self
                .reverse
                .get(key)
                .is_some_and(|seqs| n.edge[idx].0.iter().any(|alias| seqs.contains(alias)));
            if !active {
                n.claims[idx].retain(|claim| claim != key);
            }
            return;
        }
    }

    /// Walk the request's `local_hashes` from the root, collecting the
    /// per-position `(owner, tier)` claims along the matched prefix; the
    /// collapse below builds maximal `[start, end)` segments from them. Each
    /// edge element's claims are the claims of the node that edge belongs to.
    pub fn find_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
        let mut node_claims: Vec<Vec<OwnerTier>> = Vec::with_capacity(query.local_hashes.len());
        let mut qi = 0; // index into query.local_hashes
        let mut next = query
            .local_hashes
            .first()
            .and_then(|local| self.root.read().children.get(local).cloned());
        while let Some(child) = next {
            // Match and resolve the next child under the same read guard. A
            // split moves children behind the remainder edge; re-reading this
            // node's children after advancing by its old length misaligns depth.
            let c = child.read();
            let mut advanced = 0;
            while advanced < c.edge.len()
                && qi + advanced < query.local_hashes.len()
                && c.edge[advanced].1 == query.local_hashes[qi + advanced]
            {
                // Copy only matched claims, never historical alias sets.
                node_claims.push(c.claims[advanced].clone());
                advanced += 1;
            }
            // Partial edge match: children sit after the full edge in the chain,
            // so descending on a partial match misaligns chain positions.
            if advanced == 0 || advanced < c.edge.len() {
                break;
            }
            qi += advanced;
            next = query
                .local_hashes
                .get(qi)
                .and_then(|local| c.children.get(local).cloned());
            drop(c);
            #[cfg(test)]
            after_query_edge();
        }

        // Per (target, tier), run-length collect the maximal contiguous
        // [start, end) block intervals the target holds at that tier. The index
        // stores by owner but answers by target; locality is derived from the
        // owner domain. A `(target, tier)` may have several owners at one
        // depth (e.g. two pools pushed the same block); each depth is counted
        // once, so the segment extends by one, not per owner.
        let mut best: HashMap<(RoutableTarget, StorageTier), TierMatch> = HashMap::new();
        // Reused across depths; avoids a per-block allocation on the hot path.
        let mut seen: HashSet<(RoutableTarget, StorageTier)> = HashSet::new();
        for (depth, owners) in node_claims.iter().enumerate() {
            seen.clear();
            for (owner, tier) in owners {
                if !query.tiers_of_interest.contains(tier) {
                    continue;
                }
                let target = owner.target();
                let first_at_depth = seen.insert((target.clone(), *tier));
                let owner_locality = owner.locality();
                let entry = best.entry((target.clone(), *tier)).or_insert(TierMatch {
                    target,
                    tier: *tier,
                    segments: Vec::new(),
                    locality: owner_locality,
                });
                // Segment extension is per depth, but every owner contributes
                // to locality. Prefer the closest copy regardless of claim order.
                if owner_locality == Locality::Local {
                    entry.locality = Locality::Local;
                }
                if !first_at_depth {
                    continue;
                }
                match entry.segments.last_mut() {
                    Some((_, e)) if *e == depth as u32 => *e = (depth + 1) as u32,
                    _ => entry.segments.push((depth as u32, (depth + 1) as u32)),
                }
            }
        }
        best.into_values()
            .filter(|m| !m.segments.is_empty())
            .collect()
    }
}

/// One radix-tree index per cache identity, sharded by `group_idx`. Implements
/// `TieredMatchProvider` directly.
pub struct KvBlockIndexer {
    /// One shard per `group_idx` (kv-cache-spec group).
    // ponytail: group_idx is None on the wire → 0; single-spec models have one
    // group and multi-spec always emits group_idx, so 0 is a safe default.
    shards: DashMap<u32, Arc<GroupShard>>,
    suspended_sources: DashMap<(SourceId, u32), ()>,
    /// Reject query snapshots crossing an invalidation/resume boundary.
    source_generation: AtomicU64,
}

impl KvBlockIndexer {
    pub fn new() -> Self {
        KvBlockIndexer {
            shards: DashMap::new(),
            suspended_sources: DashMap::new(),
            source_generation: AtomicU64::new(0),
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

    pub fn store(
        &self,
        group_idx: u32,
        owner: ResidencyOwner,
        tier: StorageTier,
        parent_seq: Option<Arc<str>>,
        blocks: &[(Arc<str>, Arc<str>)],
    ) {
        self.shard(group_idx).store(owner, tier, parent_seq, blocks);
    }

    pub fn remove(
        &self,
        group_idx: u32,
        owner: &ResidencyOwner,
        tier: StorageTier,
        seq_hashes: &[Arc<str>],
    ) {
        self.shard(group_idx).remove(owner, tier, seq_hashes);
    }

    /// Clear one owner's residency across every group. A worker reset /
    /// `AllBlocksCleared` spans all kv-cache-spec groups the worker holds.
    pub fn clear(&self, owner: &ResidencyOwner, scope: ClearScope) {
        let shards: Vec<Arc<GroupShard>> = self.shards.iter().map(|e| e.value().clone()).collect();
        for s in shards {
            s.clear(owner, scope);
        }
    }

    /// Hide a source's observations while contiguous replay repairs its state.
    /// Correctness does not depend on delivery of advisory ingestion signals.
    pub fn suspend_source(&self, source: &SourceId, dp_rank: u32) {
        self.suspended_sources.insert((source.clone(), dp_rank), ());
        self.source_generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Stop crediting a lossy source and discard all its old observations.
    pub fn invalidate_source(&self, source: &SourceId, dp_rank: u32) {
        self.suspend_source(source, dp_rank);
        let shards: Vec<Arc<GroupShard>> = self.shards.iter().map(|e| e.value().clone()).collect();
        for shard in shards {
            shard.clear_source(source, dp_rank);
        }
    }

    /// Enable fresh observations after complete replay, or after discarding
    /// incomplete replay and applying a new valid live batch to a clean source.
    /// Already-active sources must not invalidate concurrent query snapshots.
    pub fn resume_source(&self, source: &SourceId, dp_rank: u32) {
        if let Entry::Occupied(entry) = self.suspended_sources.entry((source.clone(), dp_rank)) {
            // Fence the visibility transition before removing its hidden state.
            self.source_generation.fetch_add(1, Ordering::AcqRel);
            entry.remove();
        }
    }

    #[cfg(test)]
    pub(super) fn source_generation_for_test(&self) -> u64 {
        self.source_generation.load(Ordering::Acquire)
    }

    pub fn find_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
        let generation = self.source_generation.load(Ordering::Acquire);
        let matches = self.shard(query.group_idx).find_matches(query);
        let matches = matches
            .into_iter()
            .filter(|hit| {
                !self
                    .suspended_sources
                    .contains_key(&(hit.target.instance_id.clone(), hit.target.dp_rank))
            })
            .collect();
        if self.source_generation.load(Ordering::Acquire) != generation {
            return Vec::new();
        }
        matches
    }
}

impl Default for KvBlockIndexer {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::policies::TieredMatchProvider for KvBlockIndexer {
    fn find_tiered_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
        self.find_matches(query)
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc};

    use super::*;
    use crate::kv_index::types::SourceId;

    fn worker(name: &str) -> ResidencyOwner {
        ResidencyOwner::Worker {
            source: SourceId::from(name),
            dp_rank: 0,
            incarnation: 0,
        }
    }

    fn worker_rank(name: &str, rank: u32) -> ResidencyOwner {
        ResidencyOwner::Worker {
            source: SourceId::from(name),
            dp_rank: rank,
            incarnation: 0,
        }
    }

    fn pool(pool_id: &str, pushed_by: &str) -> ResidencyOwner {
        ResidencyOwner::CacheOwner {
            pool_id: Arc::from(pool_id),
            source: SourceId::from(pushed_by),
            dp_rank: 0,
        }
    }

    fn query(hashes: &[&str], tiers: &[StorageTier]) -> MatchQuery {
        MatchQuery {
            group_idx: 0,
            local_hashes: hashes.iter().map(|h| Arc::<str>::from(*h)).collect(),
            tiers_of_interest: tiers.to_vec(),
        }
    }

    fn query_group(group: u32, hashes: &[&str], tiers: &[StorageTier]) -> MatchQuery {
        MatchQuery {
            group_idx: group,
            local_hashes: hashes.iter().map(|h| Arc::<str>::from(*h)).collect(),
            tiers_of_interest: tiers.to_vec(),
        }
    }

    fn h(s: &str) -> Arc<str> {
        Arc::<str>::from(s)
    }

    #[test]
    fn source_suspension_and_invalidation_cover_tiers_groups_and_only_that_rank() {
        let idx = KvBlockIndexer::new();
        let tiers = [
            StorageTier::Device,
            StorageTier::HostPinned,
            StorageTier::External,
        ];
        for group in [0, 1] {
            for (owner, tier, seq) in [
                (worker("a"), StorageTier::Device, "a-gpu"),
                (worker("a"), StorageTier::HostPinned, "a-cpu"),
                (pool("shared", "a"), StorageTier::External, "a-pool"),
                (worker_rank("a", 1), StorageTier::Device, "a-rank1"),
                (pool("shared", "b"), StorageTier::External, "b-pool"),
            ] {
                idx.store(group, owner, tier, None, &[(h(seq), h("x"))]);
            }
            assert_eq!(
                idx.find_matches(&query_group(group, &["x"], &tiers)).len(),
                5
            );
        }
        let source = SourceId::from("a");
        let generation = idx.source_generation_for_test();
        for _ in 0..3 {
            idx.resume_source(&source, 0);
        }
        assert_eq!(idx.source_generation_for_test(), generation);
        idx.suspend_source(&source, 0);
        assert_eq!(idx.source_generation_for_test(), generation + 1);
        for group in [0, 1] {
            let hits = idx.find_matches(&query_group(group, &["x"], &tiers));
            assert_eq!(hits.len(), 2);
            assert!(hits
                .iter()
                .all(|hit| hit.target.instance_id != source || hit.target.dp_rank != 0));
        }
        // Complete replay may resume credit without forgetting still-valid claims.
        idx.resume_source(&source, 0);
        assert_eq!(idx.source_generation_for_test(), generation + 2);
        assert_eq!(idx.find_matches(&query(&["x"], &tiers)).len(), 5);
        idx.resume_source(&source, 0);
        assert_eq!(idx.source_generation_for_test(), generation + 2);
        idx.invalidate_source(&source, 0);
        assert_eq!(idx.source_generation_for_test(), generation + 3);
        idx.resume_source(&source, 0);
        assert_eq!(idx.source_generation_for_test(), generation + 4);
        for group in [0, 1] {
            assert_eq!(
                idx.find_matches(&query_group(group, &["x"], &tiers)).len(),
                2
            );
        }
        // Only a fresh store can restore a cleared observation.
        idx.store(
            0,
            worker("a"),
            StorageTier::Device,
            None,
            &[(h("fresh"), h("y"))],
        );
        assert_eq!(idx.find_matches(&query(&["y"], &tiers)).len(), 1);
        assert_eq!(idx.find_matches(&query(&["x"], &tiers)).len(), 2);
    }

    #[test]
    fn query_keeps_snapshot_when_active_source_is_resumed() {
        let idx = Arc::new(KvBlockIndexer::new());
        idx.store(
            0,
            worker("a"),
            StorageTier::Device,
            None,
            &[(h("s"), h("x"))],
        );
        let resuming = idx.clone();
        AFTER_QUERY_EDGE.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                resuming.resume_source(&SourceId::from("a"), 0);
            }));
        });
        let hits = idx.find_matches(&query(&["x"], &[StorageTier::Device]));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].segments, [(0, 1)]);
    }

    #[test]
    fn query_discards_snapshot_crossing_invalidation_and_resume() {
        let idx = Arc::new(KvBlockIndexer::new());
        idx.store(
            0,
            worker("a"),
            StorageTier::Device,
            None,
            &[(h("s"), h("x"))],
        );
        let invalidating = idx.clone();
        AFTER_QUERY_EDGE.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                let source = SourceId::from("a");
                invalidating.invalidate_source(&source, 0);
                invalidating.resume_source(&source, 0);
            }));
        });
        assert!(idx
            .find_matches(&query(&["x"], &[StorageTier::Device]))
            .is_empty());
    }

    fn segments_of(res: &[TierMatch], target_id: &str, tier: StorageTier) -> Vec<(u32, u32)> {
        res.iter()
            .find(|m| &*m.target.instance_id == target_id && m.tier == tier)
            .map(|m| m.segments.clone())
            .unwrap_or_default()
    }

    fn depth_of(res: &[TierMatch], target_id: &str, tier: StorageTier) -> u32 {
        segments_of(res, target_id, tier)
            .into_iter()
            .find(|(start, _)| *start == 0)
            .map_or(0, |(_, end)| end)
    }

    fn assert_observations(shard: &GroupShard, key: &OwnerTier, expected: &[&str]) {
        let actual = shard
            .reverse
            .get(key)
            .map(|seqs| seqs.clone())
            .unwrap_or_default();
        assert_eq!(actual, expected.iter().map(|seq| h(seq)).collect());
        assert_eq!(
            shard.counts.get(key).map_or(0, |count| *count),
            expected.len()
        );
        for seq in &actual {
            let node = shard
                .anchors
                .get(seq)
                .expect("active alias has an anchor")
                .clone();
            let node = node.read();
            let pos = node
                .edge
                .iter()
                .position(|(aliases, _)| aliases.contains(seq))
                .expect("anchor contains alias");
            assert!(node.claims[pos].contains(key));
        }
    }

    #[test]
    fn alias_observation_counts_and_last_claim_are_idempotent() {
        let idx = KvBlockIndexer::new();
        let owner = worker("w0");
        let key = (owner.clone(), StorageTier::Device);
        for seq in ["i:900", "i:42", "i:42"] {
            idx.store(
                0,
                owner.clone(),
                StorageTier::Device,
                None,
                &[(h(seq), h("local"))],
            );
        }
        let shard = idx.shard(0);
        assert_observations(&shard, &key, &["i:900", "i:42"]);
        let node = shard.anchors.get("i:900").unwrap().clone();
        assert!(node.read().edge[0].0.contains("i:900"));
        assert!(node.read().edge[0].0.contains("i:42"));

        idx.remove(
            0,
            &owner,
            StorageTier::Device,
            &[h("i:42"), h("i:42"), h("i:404")],
        );
        assert_observations(&shard, &key, &["i:900"]);
        assert_eq!(node.read().claims[0], vec![key.clone()]);
        idx.remove(0, &owner, StorageTier::Device, &[h("i:900"), h("i:900")]);
        assert_observations(&shard, &key, &[]);
        assert!(node.read().claims[0].is_empty());
        assert_eq!(node.read().edge[0].0.len(), 2);
        assert!(shard.counts.get(&key).is_none());
    }

    #[test]
    fn split_reanchors_all_aliases_and_clear_keeps_other_counts() {
        let idx = KvBlockIndexer::new();
        let a = worker("a");
        let b = worker("b");
        let ka = (a.clone(), StorageTier::Device);
        let kb = (b.clone(), StorageTier::Device);
        let path_a = [
            (h("i:100"), h("x")),
            (h("i:101"), h("y")),
            (h("i:102"), h("z")),
        ];
        let path_b = [
            (h("i:200"), h("x")),
            (h("i:201"), h("y")),
            (h("i:202"), h("z")),
        ];
        idx.store(0, a.clone(), StorageTier::Device, None, &path_a);
        idx.store(0, b.clone(), StorageTier::Device, None, &path_b);
        idx.store(
            0,
            b.clone(),
            StorageTier::Device,
            None,
            &[(h("b:0200"), h("x"))],
        );
        let shard = idx.shard(0);
        assert_observations(&shard, &ka, &["i:100", "i:101", "i:102"]);
        assert_observations(&shard, &kb, &["i:200", "i:201", "i:202", "b:0200"]);
        idx.store(
            0,
            b.clone(),
            StorageTier::Device,
            Some(h("i:201")),
            &[(h("i:203"), h("branch"))],
        );
        assert_observations(&shard, &ka, &["i:100", "i:101", "i:102"]);
        assert_observations(&shard, &kb, &["i:200", "i:201", "i:202", "b:0200", "i:203"]);
        idx.clear(&b, ClearScope::Worker);
        idx.clear(&b, ClearScope::Worker);
        assert_observations(&shard, &ka, &["i:100", "i:101", "i:102"]);
        assert_observations(&shard, &kb, &[]);
        assert!(shard.reverse.get(&kb).is_none());
        assert!(shard.counts.get(&kb).is_none());
        for seq in ["i:200", "i:201", "i:202", "b:0200", "i:203"] {
            let node = shard.anchors.get(seq).unwrap().clone();
            let node = node.read();
            let pos = node
                .edge
                .iter()
                .position(|(aliases, _)| aliases.contains(seq))
                .unwrap();
            assert!(!node.claims[pos].contains(&kb));
        }
        assert_eq!(
            depth_of(
                &idx.find_matches(&query(&["x", "y", "z"], &[StorageTier::Device])),
                "a",
                StorageTier::Device
            ),
            3
        );
    }

    #[test]
    fn remove_retries_stale_anchor_after_split_without_leaving_claim() {
        let idx = Arc::new(KvBlockIndexer::new());
        let a = worker("a");
        let b = worker("b");
        for (owner, seqs) in [
            (a.clone(), ["i:100", "i:101", "i:102"]),
            (b.clone(), ["i:200", "i:201", "i:202"]),
        ] {
            idx.store(
                0,
                owner,
                StorageTier::Device,
                None,
                &[
                    (h(seqs[0]), h("x")),
                    (h(seqs[1]), h("y")),
                    (h(seqs[2]), h("z")),
                ],
            );
        }
        let shard = idx.shard(0);
        let old_anchor = shard.anchors.get("i:202").unwrap().clone();
        let split_index = idx.clone();
        let retried = Rc::new(Cell::new(false));
        let retry_observed = retried.clone();
        AFTER_ANCHOR_CLONE.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                // Legal interleaving: another owner splits after remove has
                // cloned its anchor, but before it takes that node's lock.
                split_index.store(
                    0,
                    worker("a"),
                    StorageTier::Device,
                    None,
                    &[(h("i:100"), h("x"))],
                );
                let moved = split_index.shard(0).anchors.get("i:202").unwrap().clone();
                assert!(!Arc::ptr_eq(&old_anchor, &moved));
                assert!(!old_anchor
                    .read()
                    .edge
                    .iter()
                    .any(|(seqs, _)| seqs.contains("i:202")));
                AFTER_ANCHOR_CLONE.with(|slot| {
                    *slot.borrow_mut() = Some(Box::new(move || retry_observed.set(true)));
                });
            }));
        });
        idx.remove(0, &b, StorageTier::Device, &[h("i:202")]);
        assert!(retried.get());
        let ka = (a, StorageTier::Device);
        let kb = (b, StorageTier::Device);
        assert_observations(&shard, &ka, &["i:100", "i:101", "i:102"]);
        assert_observations(&shard, &kb, &["i:200", "i:201"]);
        let moved = shard.anchors.get("i:202").unwrap().clone();
        let node = moved.read();
        let pos = node
            .edge
            .iter()
            .position(|(seqs, _)| seqs.contains("i:202"))
            .unwrap();
        assert!(!node.claims[pos].contains(&kb));
        assert!(node.claims[pos].contains(&ka));
        drop(node);
        let hits = idx.find_matches(&query(&["x", "y", "z"], &[StorageTier::Device]));
        assert_eq!(depth_of(&hits, "b", StorageTier::Device), 2);
        assert_eq!(depth_of(&hits, "a", StorageTier::Device), 3);
    }

    #[test]
    fn store_cursor_retries_stale_anchor_after_split() {
        let idx = Arc::new(KvBlockIndexer::new());
        let a = worker("a");
        let b = worker("b");
        for (owner, seqs) in [
            (a.clone(), ["i:100", "i:101", "i:102"]),
            (b.clone(), ["i:200", "i:201", "i:202"]),
        ] {
            idx.store(
                0,
                owner,
                StorageTier::Device,
                None,
                &[
                    (h(seqs[0]), h("x")),
                    (h(seqs[1]), h("y")),
                    (h(seqs[2]), h("z")),
                ],
            );
        }
        let shard = idx.shard(0);
        let old_anchor = shard.anchors.get("i:201").unwrap().clone();
        let split_index = idx.clone();
        let retried = Rc::new(Cell::new(false));
        let retry_observed = retried.clone();
        AFTER_ANCHOR_CLONE.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                split_index.store(
                    0,
                    worker("a"),
                    StorageTier::Device,
                    None,
                    &[(h("i:100"), h("x"))],
                );
                let moved = split_index.shard(0).anchors.get("i:201").unwrap().clone();
                assert!(!Arc::ptr_eq(&old_anchor, &moved));
                assert!(!old_anchor
                    .read()
                    .edge
                    .iter()
                    .any(|(seqs, _)| seqs.contains("i:201")));
                AFTER_ANCHOR_CLONE.with(|slot| {
                    *slot.borrow_mut() = Some(Box::new(move || retry_observed.set(true)));
                });
            }));
        });
        idx.store(
            0,
            b.clone(),
            StorageTier::Device,
            Some(h("i:201")),
            &[(h("i:203"), h("branch"))],
        );
        assert!(retried.get());
        let ka = (a, StorageTier::Device);
        let kb = (b, StorageTier::Device);
        assert_observations(&shard, &ka, &["i:100", "i:101", "i:102"]);
        assert_observations(&shard, &kb, &["i:200", "i:201", "i:202", "i:203"]);
        let branch = shard.anchors.get("i:203").unwrap().clone();
        assert_eq!(branch.read().claims[0], vec![kb]);
        let original = idx.find_matches(&query(&["x", "y", "z"], &[StorageTier::Device]));
        assert_eq!(depth_of(&original, "a", StorageTier::Device), 3);
        assert_eq!(depth_of(&original, "b", StorageTier::Device), 3);
        let extended = idx.find_matches(&query(&["x", "y", "branch"], &[StorageTier::Device]));
        assert_eq!(depth_of(&extended, "a", StorageTier::Device), 2);
        assert_eq!(depth_of(&extended, "b", StorageTier::Device), 3);
    }

    #[test]
    fn concurrent_distinct_owner_alias_writes_splits_and_removes() {
        use std::sync::{mpsc, Barrier};
        use std::time::Duration;

        let idx = Arc::new(KvBlockIndexer::new());
        idx.store(
            0,
            worker("seed"),
            StorageTier::Device,
            None,
            &[(h("i:1"), h("x")), (h("i:2"), h("y")), (h("i:3"), h("z"))],
        );
        let barrier = Arc::new(Barrier::new(4));
        let (done, completions) = mpsc::channel();
        let mut threads = Vec::new();
        for writer in 0..4 {
            let idx = idx.clone();
            let barrier = barrier.clone();
            let done = done.clone();
            threads.push(std::thread::spawn(move || {
                let owner = worker(&format!("writer-{writer}"));
                let seqs: Vec<Arc<str>> = (0..4)
                    .map(|block| h(&format!("i:{}", 100 + writer * 10 + block)))
                    .collect();
                let path = [
                    (seqs[0].clone(), h("x")),
                    (seqs[1].clone(), h("y")),
                    (seqs[2].clone(), h("z")),
                ];
                let branch = [(seqs[3].clone(), h(&format!("branch-{writer}")))];
                barrier.wait();
                for _ in 0..32 {
                    idx.store(0, owner.clone(), StorageTier::Device, None, &path);
                    idx.store(
                        0,
                        owner.clone(),
                        StorageTier::Device,
                        Some(seqs[1].clone()),
                        &branch,
                    );
                    idx.remove(0, &owner, StorageTier::Device, &seqs[1..2]);
                    idx.store(
                        0,
                        owner.clone(),
                        StorageTier::Device,
                        Some(seqs[0].clone()),
                        &path[1..2],
                    );
                    assert_observations(
                        &idx.shard(0),
                        &(owner.clone(), StorageTier::Device),
                        &seqs.iter().map(|seq| seq.as_ref()).collect::<Vec<_>>(),
                    );
                    idx.remove(0, &owner, StorageTier::Device, &seqs);
                    assert_observations(&idx.shard(0), &(owner.clone(), StorageTier::Device), &[]);
                }
                done.send(()).unwrap();
            }));
        }
        drop(done);
        // The receiver is outside writer threads: a lock deadlock cannot starve
        // its wall-clock deadline, and each owner has exactly one writer.
        for _ in 0..4 {
            completions
                .recv_timeout(Duration::from_secs(30))
                .expect("alias writer must complete without deadlock");
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert_observations(
            &idx.shard(0),
            &(worker("seed"), StorageTier::Device),
            &["i:1", "i:2", "i:3"],
        );
        assert_eq!(
            depth_of(
                &idx.find_matches(&query(&["x", "y", "z"], &[StorageTier::Device])),
                "seed",
                StorageTier::Device
            ),
            3
        );
    }

    #[test]
    fn store_and_match_single_owner() {
        let idx = KvBlockIndexer::new();
        let blocks = vec![(h("h0"), h("h0")), (h("h1"), h("h1")), (h("h2"), h("h2"))];
        idx.store(0, worker("w0"), StorageTier::Device, None, &blocks);

        let res = idx.find_matches(&query(&["h0", "h1", "h2", "h3"], &[StorageTier::Device]));
        assert_eq!(res.len(), 1);
        // h3 absent → match depth 3.
        assert_eq!(segments_of(&res, "w0", StorageTier::Device), vec![(0, 3)]);
    }

    #[test]
    fn store_is_idempotent_per_owner_tier_block() {
        let idx = KvBlockIndexer::new();
        let blocks = vec![(h("h0"), h("h0"))];
        idx.store(0, worker("w0"), StorageTier::Device, None, &blocks);
        idx.store(0, worker("w0"), StorageTier::Device, None, &blocks);

        let key = (worker("w0"), StorageTier::Device);
        let count = idx
            .shards
            .get(&0)
            .and_then(|s| s.counts.get(&key).map(|c| *c))
            .unwrap_or(0);
        assert_eq!(count, 1);
    }

    #[test]
    fn cross_owner_refcount_survives_until_both_remove() {
        let idx = KvBlockIndexer::new();
        let blocks = vec![(h("h0"), h("h0"))];
        idx.store(0, worker("w0"), StorageTier::Device, None, &blocks);
        idx.store(0, worker("w1"), StorageTier::Device, None, &blocks);

        idx.remove(0, &worker("w0"), StorageTier::Device, &[h("h0")]);
        let res = idx.find_matches(&query(&["h0"], &[StorageTier::Device]));
        assert_eq!(res.len(), 1);

        idx.remove(0, &worker("w1"), StorageTier::Device, &[h("h0")]);
        let res = idx.find_matches(&query(&["h0"], &[StorageTier::Device]));
        assert!(res.is_empty());
    }

    #[test]
    fn same_block_second_tier_is_separate_claim() {
        let idx = KvBlockIndexer::new();
        let blocks = vec![(h("h0"), h("h0"))];
        idx.store(0, worker("w0"), StorageTier::Device, None, &blocks);
        idx.store(0, worker("w0"), StorageTier::HostPinned, None, &blocks);

        // Remove Device only; HostPinned survives.
        idx.remove(0, &worker("w0"), StorageTier::Device, &[h("h0")]);
        let res = idx.find_matches(&query(&["h0"], &[StorageTier::HostPinned]));
        assert_eq!(res.len(), 1);
        assert_eq!(
            segments_of(&res, "w0", StorageTier::HostPinned),
            vec![(0, 1)]
        );
    }

    #[test]
    fn partial_prefix_match_depth() {
        let idx = KvBlockIndexer::new();
        let blocks = vec![(h("h0"), h("h0")), (h("h1"), h("h1"))];
        idx.store(0, worker("w0"), StorageTier::Device, None, &blocks);

        // Request has h0,h1,h2 but only h0,h1 are resident → depth 2.
        let res = idx.find_matches(&query(&["h0", "h1", "h2"], &[StorageTier::Device]));
        assert_eq!(segments_of(&res, "w0", StorageTier::Device), vec![(0, 2)]);
    }

    #[test]
    fn remove_mid_chain_stops_prefix_walk() {
        // Removing a middle block's claim breaks the prefix walk there: the
        // trailing block must not be credited past the gap.
        let idx = KvBlockIndexer::new();
        let blocks = vec![(h("h0"), h("h0")), (h("h1"), h("h1"))];
        idx.store(0, worker("w0"), StorageTier::Device, None, &blocks);

        idx.remove(0, &worker("w0"), StorageTier::Device, &[h("h1")]);
        // h0 still matched (depth 1); the query's h1 now finds no owner.
        let res = idx.find_matches(&query(&["h0", "h1"], &[StorageTier::Device]));
        assert_eq!(segments_of(&res, "w0", StorageTier::Device), vec![(0, 1)]);
    }

    #[test]
    fn interior_remove_breaks_prefix_descendants_survive() {
        // [a, b, c] as separate parent-chained nodes. Removing b (interior)
        // stops the prefix walk at a; c's claim survives but isn't credited
        // past the gap. Re-storing b re-links and credits the full prefix.
        let idx = KvBlockIndexer::new();
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            None,
            &[(h("sa"), h("a"))],
        );
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            Some(h("sa")),
            &[(h("sb"), h("b"))],
        );
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            Some(h("sb")),
            &[(h("sc"), h("c"))],
        );

        idx.remove(0, &worker("w0"), StorageTier::Device, &[h("sb")]);
        let res = idx.find_matches(&query(&["a", "b", "c"], &[StorageTier::Device]));
        // b removed mid-prefix: c survives as a separate segment. The cost
        // model credits only [0, d_gpu); the index reports both.
        assert_eq!(
            segments_of(&res, "w0", StorageTier::Device),
            vec![(0, 1), (2, 3)]
        );

        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            Some(h("sa")),
            &[(h("sb"), h("b"))],
        );
        let res = idx.find_matches(&query(&["a", "b", "c"], &[StorageTier::Device]));
        assert_eq!(segments_of(&res, "w0", StorageTier::Device), vec![(0, 3)]);
    }

    #[test]
    fn two_owners_overlapping_prefix_per_target_depth() {
        let idx = KvBlockIndexer::new();
        // w0 has h0,h1,h2; w1 has h0,h1.
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            None,
            &[(h("h0"), h("h0")), (h("h1"), h("h1")), (h("h2"), h("h2"))],
        );
        idx.store(
            0,
            worker("w1"),
            StorageTier::Device,
            None,
            &[(h("h0"), h("h0")), (h("h1"), h("h1"))],
        );

        let res = idx.find_matches(&query(&["h0", "h1", "h2", "h3"], &[StorageTier::Device]));
        assert_eq!(res.len(), 2);
        assert_eq!(segments_of(&res, "w0", StorageTier::Device), vec![(0, 3)]);
        assert_eq!(segments_of(&res, "w1", StorageTier::Device), vec![(0, 2)]);
    }

    #[test]
    fn clear_worker_domain_does_not_cross_into_cache_owner() {
        let idx = KvBlockIndexer::new();
        idx.store(
            0,
            pool("mooncake", "w0"),
            StorageTier::External,
            None,
            &[(h("h0"), h("h0"))],
        );
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            None,
            &[(h("h0"), h("h0"))],
        );

        idx.clear(&worker("w0"), ClearScope::Worker);

        // Pool residency survives.
        let res = idx.find_matches(&query(&["h0"], &[StorageTier::External]));
        assert_eq!(res.len(), 1);
        // Worker residency gone.
        let res = idx.find_matches(&query(&["h0"], &[StorageTier::Device]));
        assert!(res.is_empty());
    }

    #[test]
    fn clear_wipes_all_media_of_one_rank() {
        // One rank offloads the same block to Host; a Worker-domain clear must
        // drop both Device and Host claims for that (source, rank).
        let idx = KvBlockIndexer::new();
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            None,
            &[(h("h0"), h("h0"))],
        );
        idx.store(
            0,
            worker("w0"),
            StorageTier::HostPinned,
            None,
            &[(h("h0"), h("h0"))],
        );

        idx.clear(&worker("w0"), ClearScope::Worker);
        assert!(idx
            .find_matches(&query(&["h0"], &[StorageTier::Device]))
            .is_empty());
        assert!(idx
            .find_matches(&query(&["h0"], &[StorageTier::HostPinned]))
            .is_empty());
    }

    #[test]
    fn clear_is_per_rank() {
        // Same source, two ranks; clearing rank 0 leaves rank 1 intact.
        let idx = KvBlockIndexer::new();
        idx.store(
            0,
            worker_rank("w0", 0),
            StorageTier::Device,
            None,
            &[(h("h0"), h("h0"))],
        );
        idx.store(
            0,
            worker_rank("w0", 1),
            StorageTier::Device,
            None,
            &[(h("h0"), h("h0"))],
        );

        idx.clear(&worker_rank("w0", 0), ClearScope::Worker);
        let res = idx.find_matches(&query(&["h0"], &[StorageTier::Device]));
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].target.dp_rank, 1);
    }

    #[test]
    fn clear_cache_owner_by_pool_id() {
        // Two pools pushed by the same worker share one (target, tier) entry,
        // so find_matches can't tell them apart — verify via counts instead.
        let idx = KvBlockIndexer::new();
        idx.store(
            0,
            pool("poolA", "w0"),
            StorageTier::External,
            None,
            &[(h("h0"), h("h0"))],
        );
        idx.store(
            0,
            pool("poolB", "w0"),
            StorageTier::External,
            None,
            &[(h("h0"), h("h0"))],
        );
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            None,
            &[(h("h0"), h("h0"))],
        );

        idx.clear(&pool("poolA", "w0"), ClearScope::CacheOwner);

        let shard = idx.shards.get(&0).expect("group 0 shard exists");
        // poolA cleared; poolB intact; Worker untouched.
        assert!(shard
            .counts
            .get(&(pool("poolA", "w0"), StorageTier::External))
            .is_none());
        assert_eq!(
            *shard
                .counts
                .get(&(pool("poolB", "w0"), StorageTier::External))
                .unwrap(),
            1
        );
        assert_eq!(
            *shard
                .counts
                .get(&(worker("w0"), StorageTier::Device))
                .unwrap(),
            1
        );
    }

    #[test]
    fn cache_owner_hit_attributed_to_pushing_worker() {
        // A pool block pushed by w0: the hit's target is w0 (it can read what
        // it pushed), at the External tier, Remote locality.
        let idx = KvBlockIndexer::new();
        idx.store(
            0,
            pool("mooncake", "w0"),
            StorageTier::External,
            None,
            &[(h("h0"), h("h0"))],
        );

        let res = idx.find_matches(&query(&["h0"], &[StorageTier::External]));
        assert_eq!(res.len(), 1);
        let m = &res[0];
        assert_eq!(&*m.target.instance_id, "w0");
        assert_eq!(m.tier, StorageTier::External);
        assert_eq!(m.locality, Locality::Remote);
    }

    #[test]
    fn group_idx_shards_isolate_same_local_hash() {
        // The engine's seq hash doesn't fold group_idx, so the same (seq, local)
        // can appear under two groups; separate shards keep them apart.
        let idx = KvBlockIndexer::new();
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            None,
            &[(h("seq"), h("loc"))],
        );
        idx.store(
            1,
            worker("w0"),
            StorageTier::Device,
            None,
            &[(h("seq"), h("loc"))],
        );

        // Each query sees only its own group's residency.
        assert_eq!(
            idx.find_matches(&query_group(0, &["loc"], &[StorageTier::Device]))
                .len(),
            1
        );
        assert_eq!(
            idx.find_matches(&query_group(1, &["loc"], &[StorageTier::Device]))
                .len(),
            1
        );

        // Remove in group 0 leaves group 1 intact: separate anchor maps,
        // so group 0's remove can't clobber group 1's node.
        idx.remove(0, &worker("w0"), StorageTier::Device, &[h("seq")]);
        assert!(idx
            .find_matches(&query_group(0, &["loc"], &[StorageTier::Device]))
            .is_empty());
        assert_eq!(
            idx.find_matches(&query_group(1, &["loc"], &[StorageTier::Device]))
                .len(),
            1
        );
    }

    #[test]
    fn repeated_tokens_do_not_collapse_across_chain_positions() {
        // Same token block ("x") at two chain positions reached via different
        // parents: each stores as a distinct child under a distinct node, so
        // the prefix walk matches through the second occurrence too.
        let idx = KvBlockIndexer::new();
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            None,
            &[(h("sa"), h("a"))],
        );
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            Some(h("sa")),
            &[(h("sx1"), h("x"))],
        );
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            Some(h("sx1")),
            &[(h("sb"), h("b"))],
        );
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            Some(h("sb")),
            &[(h("sx2"), h("x"))],
        );
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            Some(h("sx2")),
            &[(h("sc"), h("c"))],
        );

        let res = idx.find_matches(&query(&["a", "x", "b", "x", "c"], &[StorageTier::Device]));
        assert_eq!(segments_of(&res, "w0", StorageTier::Device), vec![(0, 5)]);
    }

    #[test]
    fn store_partial_match_splits_edge_and_reanchors_remainder() {
        // A 4-block edge, then a store diverging at depth 2: the edge splits,
        // the remainder re-anchors to a new child, the diverging block appends
        // as a sibling. Both branches stay reachable; the relocated seqs stay
        // findable by remove.
        let idx = KvBlockIndexer::new();
        let long = vec![
            (h("s0"), h("h0")),
            (h("s1"), h("h1")),
            (h("s2"), h("h2")),
            (h("s3"), h("h3")),
        ];
        idx.store(0, worker("w0"), StorageTier::Device, None, &long);
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            None,
            &[(h("s0"), h("h0")), (h("s1"), h("h1")), (h("s4"), h("h4"))],
        );

        assert_eq!(
            segments_of(
                &idx.find_matches(&query(&["h0", "h1", "h2", "h3"], &[StorageTier::Device])),
                "w0",
                StorageTier::Device
            ),
            vec![(0, 4)]
        );
        assert_eq!(
            segments_of(
                &idx.find_matches(&query(&["h0", "h1", "h4"], &[StorageTier::Device])),
                "w0",
                StorageTier::Device
            ),
            vec![(0, 3)]
        );

        idx.remove(0, &worker("w0"), StorageTier::Device, &[h("s2")]);
        // h2 removed mid-prefix: h3 survives as a separate segment.
        assert_eq!(
            segments_of(
                &idx.find_matches(&query(&["h0", "h1", "h2", "h3"], &[StorageTier::Device])),
                "w0",
                StorageTier::Device
            ),
            vec![(0, 2), (3, 4)]
        );
    }

    #[test]
    fn store_resumes_from_incoming_alias_after_full_and_partial_matches() {
        for original in [
            vec![(h("sx"), h("x")), (h("sy"), h("y"))],
            vec![(h("sx"), h("x")), (h("sy"), h("y")), (h("sz"), h("z"))],
        ] {
            let shard = GroupShard::new();
            let key = (worker("w0"), StorageTier::Device);
            shard.store(worker("w0"), StorageTier::Device, None, &original);
            shard.store(
                worker("w0"),
                StorageTier::Device,
                None,
                &[
                    (h("incoming-x"), h("x")),
                    (h("incoming-y"), h("y")),
                    (h("sw"), h("w")),
                ],
            );

            assert_eq!(
                segments_of(
                    &shard.find_matches(&query(&["x", "y", "w"], &[StorageTier::Device])),
                    "w0",
                    StorageTier::Device,
                ),
                vec![(0, 3)]
            );
            assert_eq!(
                segments_of(
                    &shard.find_matches(&query(&["x", "y", "z"], &[StorageTier::Device])),
                    "w0",
                    StorageTier::Device,
                ),
                vec![(0, original.len() as u32)]
            );
            assert!(shard.anchors.contains_key("incoming-x"));
            assert!(shard.anchors.contains_key("incoming-y"));
            assert!(!shard.root.read().children.contains_key("w"));
            assert_eq!(
                *shard
                    .counts
                    .get(&(worker("w0"), StorageTier::Device))
                    .unwrap(),
                original.len() + 3
            );
            // Matching local hashes registers incoming observations alongside
            // the original identities, rather than only claiming stored keys.
            let mut expected: Vec<&str> = original
                .iter()
                .map(|(seq, _)| seq.as_ref())
                .chain(["incoming-x", "incoming-y", "sw"])
                .collect();
            assert_observations(&shard, &key, &expected);

            shard.remove(&key.0, key.1, &[h("incoming-y")]);
            expected.retain(|seq| *seq != "incoming-y");
            assert_observations(&shard, &key, &expected);
            let y = shard.anchors.get("sy").unwrap().clone();
            let y_pos = y
                .read()
                .edge
                .iter()
                .position(|(seqs, _)| seqs.contains("sy"))
                .unwrap();
            assert!(y.read().claims[y_pos].contains(&key));
            for (hashes, depth) in [
                (vec!["x", "y", "w"], 3),
                (vec!["x", "y", "z"], original.len() as u32),
            ] {
                assert_eq!(
                    depth_of(
                        &shard.find_matches(&query(&hashes, &[StorageTier::Device])),
                        "w0",
                        StorageTier::Device,
                    ),
                    depth
                );
            }

            shard.remove(&key.0, key.1, &[h("sy")]);
            expected.retain(|seq| *seq != "sy");
            assert_observations(&shard, &key, &expected);
            assert!(!y.read().claims[y_pos].contains(&key));
            let expected_z_segments = if original.len() == 3 {
                vec![(0, 1), (2, 3)]
            } else {
                vec![(0, 1)]
            };
            for (hashes, expected_segments) in [
                (["x", "y", "w"], vec![(0, 1), (2, 3)]),
                (["x", "y", "z"], expected_z_segments),
            ] {
                assert_eq!(
                    segments_of(
                        &shard.find_matches(&query(&hashes, &[StorageTier::Device])),
                        "w0",
                        StorageTier::Device,
                    ),
                    expected_segments
                );
            }
        }
    }

    #[test]
    fn store_retries_new_incoming_cursor_moved_before_lock() {
        let shard = Arc::new(GroupShard::new());
        let key = (worker("w0"), StorageTier::Device);
        let splitter_key = (worker("splitter"), StorageTier::Device);
        shard.store(
            key.0.clone(),
            key.1,
            None,
            &[(h("sx"), h("x")), (h("sy"), h("y")), (h("sz"), h("z"))],
        );
        let split_shard = shard.clone();
        let retried = Rc::new(Cell::new(false));
        let retry_observed = retried.clone();
        AFTER_ANCHOR_CLONE.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                // The root lookup happens before the incoming keys are
                // registered. Interleave on the next, incoming-y lookup.
                AFTER_ANCHOR_CLONE.with(|slot| {
                    *slot.borrow_mut() = Some(Box::new(move || {
                        let key = (worker("w0"), StorageTier::Device);
                        assert_observations(
                            &split_shard,
                            &key,
                            &["sx", "sy", "sz", "incoming-x", "incoming-y"],
                        );
                        let old_anchor = split_shard.anchors.get("incoming-y").unwrap().clone();
                        split_shard.store(
                            worker("splitter"),
                            StorageTier::Device,
                            None,
                            &[(h("split-x"), h("x"))],
                        );
                        let moved = split_shard.anchors.get("incoming-y").unwrap().clone();
                        assert!(!Arc::ptr_eq(&old_anchor, &moved));
                        assert!(!old_anchor
                            .read()
                            .edge
                            .iter()
                            .any(|(seqs, _)| seqs.contains("incoming-y")));
                        assert!(moved.read().edge[0].0.contains("incoming-y"));
                        assert!(moved.read().edge[0].0.contains("sy"));
                        AFTER_ANCHOR_CLONE.with(|slot| {
                            *slot.borrow_mut() = Some(Box::new(move || retry_observed.set(true)));
                        });
                    }));
                });
            }));
        });
        shard.store(
            key.0.clone(),
            key.1,
            None,
            &[
                (h("incoming-x"), h("x")),
                (h("incoming-y"), h("y")),
                (h("sw"), h("w")),
            ],
        );
        assert!(retried.get());
        assert_observations(
            &shard,
            &key,
            &["sx", "sy", "sz", "incoming-x", "incoming-y", "sw"],
        );
        assert_observations(&shard, &splitter_key, &["split-x"]);
        for (hashes, expected) in [
            (vec!["x", "y", "w"], 3),
            (vec!["x", "y", "z"], 3),
            (vec!["x", "y", "z", "w"], 3),
            (vec!["w"], 0),
        ] {
            assert_eq!(
                depth_of(
                    &shard.find_matches(&query(&hashes, &[StorageTier::Device])),
                    "w0",
                    StorageTier::Device,
                ),
                expected,
                "query {hashes:?}"
            );
        }
        assert_eq!(
            depth_of(
                &shard.find_matches(&query(&["x", "y", "w"], &[StorageTier::Device])),
                "splitter",
                StorageTier::Device,
            ),
            1
        );
        assert!(!shard.root.read().children.contains_key("w"));
    }

    #[test]
    fn store_re_resolves_parent_moved_by_concurrent_split() {
        use std::sync::mpsc;
        use std::time::Duration;

        let shard = Arc::new(GroupShard::new());
        shard.store(
            worker("w0"),
            StorageTier::Device,
            None,
            &[(h("sx"), h("x")), (h("sy"), h("y")), (h("sz"), h("z"))],
        );
        let old_anchor = shard.anchors.get("sy").unwrap().clone();
        let timeout = Duration::from_secs(10);
        let (resolved_tx, resolved_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let (stored_tx, stored_rx) = mpsc::channel();
        let writer_shard = shard.clone();
        let writer = std::thread::spawn(move || {
            AFTER_ANCHOR_CLONE.with(|slot| {
                *slot.borrow_mut() = Some(Box::new(move || {
                    resolved_tx.send(()).unwrap();
                    resume_rx.recv_timeout(timeout).unwrap();
                }));
            });
            writer_shard.store(
                worker("w0"),
                StorageTier::Device,
                Some(h("sy")),
                &[(h("sw"), h("w"))],
            );
            stored_tx.send(()).unwrap();
        });
        resolved_rx.recv_timeout(timeout).unwrap();

        // Split after x while the other store has cloned sy's old anchor but
        // has not locked it. A separate completion channel bounds deadlocks.
        let (split_tx, split_rx) = mpsc::channel();
        let splitter_shard = shard.clone();
        let splitter = std::thread::spawn(move || {
            splitter_shard.store(
                worker("w0"),
                StorageTier::Device,
                None,
                &[(h("sx"), h("x"))],
            );
            split_tx.send(()).unwrap();
        });
        split_rx.recv_timeout(timeout).unwrap();
        let moved_anchor = shard.anchors.get("sy").unwrap().clone();
        assert!(!Arc::ptr_eq(&old_anchor, &moved_anchor));
        assert!(!old_anchor
            .read()
            .edge
            .iter()
            .any(|(seqs, _)| seqs.contains("sy")));
        resume_tx.send(()).unwrap();
        stored_rx.recv_timeout(timeout).unwrap();
        writer.join().unwrap();
        splitter.join().unwrap();

        for (hashes, expected) in [
            (vec!["x", "y", "w"], 3),
            (vec!["x", "y", "z"], 3),
            (vec!["x", "y", "z", "w"], 3),
            (vec!["w"], 0),
        ] {
            assert_eq!(
                segments_of(
                    &shard.find_matches(&query(&hashes, &[StorageTier::Device])),
                    "w0",
                    StorageTier::Device,
                ),
                if expected == 0 {
                    vec![]
                } else {
                    vec![(0, expected)]
                },
                "query {hashes:?}"
            );
        }
        assert!(!shard.root.read().children.contains_key("w"));
    }

    #[test]
    fn clear_all_scope_wipes_owner_domain_without_crossing_domains() {
        let idx = KvBlockIndexer::new();
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            None,
            &[(h("h0"), h("h0"))],
        );
        idx.store(
            0,
            pool("mooncake", "w0"),
            StorageTier::External,
            None,
            &[(h("h0"), h("h0"))],
        );
        idx.store(
            0,
            worker("w1"),
            StorageTier::Device,
            None,
            &[(h("h0"), h("h0"))],
        );

        idx.clear(&worker("w0"), ClearScope::All);

        let dev = idx.find_matches(&query(&["h0"], &[StorageTier::Device]));
        assert_eq!(dev.len(), 1);
        assert_eq!(&*dev[0].target.instance_id, "w1");
        let ext = idx.find_matches(&query(&["h0"], &[StorageTier::External]));
        assert_eq!(ext.len(), 1);
        assert_eq!(&*ext[0].target.instance_id, "w0");
    }

    #[test]
    fn device_plus_host_continuation_reports_both_tiers() {
        // Device holds [0,3) and host holds [3,10) as a parent-chained tail:
        // host lacks block 0 yet the engine reuses the 7-block tail, so the
        // index reports each tier's held intervals.
        let idx = KvBlockIndexer::new();
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            None,
            &[(h("s0"), h("h0")), (h("s1"), h("h1")), (h("s2"), h("h2"))],
        );
        idx.store(
            0,
            worker("w0"),
            StorageTier::HostPinned,
            Some(h("s2")),
            &[
                (h("s3"), h("h3")),
                (h("s4"), h("h4")),
                (h("s5"), h("h5")),
                (h("s6"), h("h6")),
                (h("s7"), h("h7")),
                (h("s8"), h("h8")),
                (h("s9"), h("h9")),
            ],
        );

        let res = idx.find_matches(&query(
            &["h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9"],
            &[StorageTier::Device, StorageTier::HostPinned],
        ));
        assert_eq!(segments_of(&res, "w0", StorageTier::Device), vec![(0, 3)]);
        assert_eq!(
            segments_of(&res, "w0", StorageTier::HostPinned),
            vec![(3, 10)]
        );
    }

    #[test]
    fn same_depth_multiple_owners_no_duplicate_segment() {
        // Two pools push the same block to the same worker at the same tier:
        // one (target, tier) at one depth must extend the segment by one, not
        // once per owner — otherwise the run-length loop double-counts the
        // depth into two identical segments.
        let idx = KvBlockIndexer::new();
        idx.store(
            0,
            pool("poolA", "w0"),
            StorageTier::External,
            None,
            &[(h("s0"), h("h0"))],
        );
        idx.store(
            0,
            pool("poolB", "w0"),
            StorageTier::External,
            None,
            &[(h("s0"), h("h0"))],
        );

        let res = idx.find_matches(&query(&["h0"], &[StorageTier::External]));
        assert_eq!(segments_of(&res, "w0", StorageTier::External), vec![(0, 1)]);
    }

    #[test]
    fn same_depth_multiple_owners_prefer_local_copy() {
        let idx = KvBlockIndexer::new();
        let block = [(h("s0"), h("h0"))];
        // Insert the remote claim first to ensure locality does not depend on
        // the first owner accepted by the per-depth de-duplication guard.
        idx.store(0, pool("poolA", "w0"), StorageTier::Device, None, &block);
        idx.store(0, worker("w0"), StorageTier::Device, None, &block);

        let res = idx.find_matches(&query(&["h0"], &[StorageTier::Device]));
        let m = res
            .iter()
            .find(|m| &*m.target.instance_id == "w0" && m.tier == StorageTier::Device)
            .expect("matching target and tier");
        assert_eq!(m.segments, vec![(0, 1)]);
        assert_eq!(m.locality, Locality::Local);
    }

    #[test]
    fn partial_edge_match_does_not_descend_into_misaligned_child() {
        // [a,b,c] edge with a Host child x; query [a,x] diverges at b, so the
        // walk must stop at a — x is at tree position 3, not query position 1.
        let idx = KvBlockIndexer::new();
        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            None,
            &[(h("sa"), h("a")), (h("sb"), h("b")), (h("sc"), h("c"))],
        );
        idx.store(
            0,
            worker("w0"),
            StorageTier::HostPinned,
            Some(h("sc")),
            &[(h("sx"), h("x"))],
        );

        let res = idx.find_matches(&query(
            &["a", "x"],
            &[StorageTier::Device, StorageTier::HostPinned],
        ));
        assert_eq!(segments_of(&res, "w0", StorageTier::Device), vec![(0, 1)]);
        assert!(segments_of(&res, "w0", StorageTier::HostPinned).is_empty());
    }

    #[test]
    fn query_split_between_edges_preserves_position_and_tier() {
        let idx = Arc::new(KvBlockIndexer::new());
        idx.store(
            0,
            worker("a"),
            StorageTier::Device,
            None,
            &[(h("s0"), h("x")), (h("s1"), h("y"))],
        );
        // Repeating token-local hashes at different chain positions is legal.
        idx.store(
            0,
            worker("a"),
            StorageTier::HostPinned,
            Some(h("s1")),
            &[(h("s2"), h("y"))],
        );
        let splitting = idx.clone();
        AFTER_QUERY_EDGE.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                // Another owner's store splits the edge after query matched it.
                // Neither the local path nor A's residency changes.
                splitting.store(
                    0,
                    worker("b"),
                    StorageTier::Device,
                    None,
                    &[(h("b0"), h("x"))],
                );
            }));
        });
        let hits = idx.find_matches(&query(
            &["x", "y", "y"],
            &[StorageTier::Device, StorageTier::HostPinned],
        ));
        assert_eq!(segments_of(&hits, "a", StorageTier::Device), vec![(0, 2)]);
        assert_eq!(
            segments_of(&hits, "a", StorageTier::HostPinned),
            vec![(2, 3)]
        );
    }
}
