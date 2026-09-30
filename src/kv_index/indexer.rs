//! Concurrent radix-tree index of resident KV blocks, keyed by local hash.

use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock;

use crate::kv_index::types::{
    ClearScope, Locality, ResidencyOwner, RoutableTarget, StorageTier,
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

/// One hit: a target holds a contiguous prefix `[0, depth)` at this tier.
/// Flat so the policy sorts in one pass — one entry per `(target, tier)`.
#[derive(Debug, Clone)]
pub struct TierMatch {
    pub target: RoutableTarget,
    pub tier: StorageTier,
    /// Contiguous prefix depth (blocks `[0, depth)`).
    pub matched_depth: u32,
    pub locality: Locality,
}

type OwnerTier = (ResidencyOwner, StorageTier);
type NodeRef = Arc<RwLock<TreeNode>>;
type EdgeBlock = (Arc<str>, Arc<str>); // (seq_hash, local_hash)

/// Length of the common prefix between an edge and the incoming blocks,
/// compared by local hash only.
fn edge_common_prefix(edge: &[EdgeBlock], incoming: &[EdgeBlock]) -> usize {
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
    /// Edge blocks leading into this node; empty on the root sentinel.
    edge: Vec<EdgeBlock>,
    /// Per-block residency claims, parallel to `edge`; `claims[i]` owns
    /// `edge[i]`. Empty on the root sentinel.
    claims: Vec<Vec<OwnerTier>>,
    /// Children keyed by the first local hash of each child's edge.
    children: HashMap<Arc<str>, NodeRef>,
}

impl TreeNode {
    fn root() -> Self {
        TreeNode { edge: Vec::new(), claims: Vec::new(), children: HashMap::new() }
    }
    fn with_edge(edge: Vec<EdgeBlock>) -> Self {
        let claims = (0..edge.len()).map(|_| Vec::new()).collect();
        TreeNode { edge, claims, children: HashMap::new() }
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
    // grows it without limit. Revisit with an epoch leaf sweep if growth shows.
    anchors: DashMap<Arc<str>, NodeRef>,
    /// Reverse lookup: (owner, tier) → seq hashes claimed.
    reverse: DashMap<OwnerTier, Vec<Arc<str>>>,
    /// Per-(owner, tier) resident block count, for capacity accounting.
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
    /// the incoming blocks are matched against each child's edge: on a full
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
        blocks: &[EdgeBlock],
    ) {
        let key = (owner.clone(), tier);
        let mut node: NodeRef = parent_seq
            .as_ref()
            .and_then(|s| self.anchors.get(s).map(|n| n.clone()))
            .unwrap_or_else(|| self.root.clone());
        let mut bi = 0;
        while bi < blocks.len() {
            let child = {
                let n = node.read();
                n.children.get(&blocks[bi].1).cloned()
            };
            let Some(child) = child else {
                // No child: append a new edge with all remaining blocks.
                let edge = blocks[bi..].to_vec();
                let new_node = Arc::new(RwLock::new(TreeNode::with_edge(edge)));
                self.install_new_child(&node, &blocks[bi].1, &new_node);
                let mut g = new_node.write();
                self.record_locked(&mut g, &new_node, &key);
                return;
            };
            // Match, split, and claim under one write lock: a concurrent split
            // can't shorten this edge between the match and the split.
            let descend = {
                let mut c = child.write();
                let matched = edge_common_prefix(&c.edge, &blocks[bi..]);
                if matched >= c.edge.len() {
                    self.record_locked(&mut c, &child, &key);
                    bi += c.edge.len();
                    true
                } else {
                    self.split_locked(&mut c, matched);
                    self.record_locked(&mut c, &child, &key);
                    bi += matched;
                    false
                }
            };
            if descend {
                node = child;
                continue;
            }
            // Partial match ended the traversal; append remaining blocks as a sibling.
            if bi < blocks.len() {
                let edge = blocks[bi..].to_vec();
                let new_node = Arc::new(RwLock::new(TreeNode::with_edge(edge)));
                self.install_new_child(&child, &blocks[bi].1, &new_node);
                let mut g = new_node.write();
                self.record_locked(&mut g, &new_node, &key);
            }
            return;
        }
    }

    /// Link a new child under its first local hash.
    fn install_new_child(&self, parent: &NodeRef, first_local: &Arc<str>, child: &NodeRef) {
        parent.write().children.insert(first_local.clone(), child.clone());
    }

    /// Record every block on `node`'s edge into anchors, reverse, counts, and
    /// attach `(owner, tier)` as a per-block claim. Idempotent per
    /// (owner, tier, block). Caller holds `node`'s write lock.
    fn record_locked(&self, n: &mut TreeNode, node: &NodeRef, key: &OwnerTier) {
        for (seq, _) in &n.edge {
            self.anchors.insert(seq.clone(), node.clone());
            let already = self
                .reverse
                .get(key)
                .is_some_and(|c| c.iter().any(|h| h == seq));
            if !already {
                self.reverse.entry(key.clone()).or_default().push(seq.clone());
                *self.counts.entry(key.clone()).or_insert(0) += 1;
            }
        }
        for claims in n.claims.iter_mut() {
            if !claims.contains(key) {
                claims.push(key.clone());
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
        let rem_seqs: Vec<Arc<str>> = remainder.iter().map(|(s, _)| s.clone()).collect();
        let rem_node = Arc::new(RwLock::new(TreeNode::with_edge(remainder)));
        {
            let mut r = rem_node.write();
            r.claims = rem_claims;
            r.children = old_children;
        }
        for seq in rem_seqs {
            self.anchors.insert(seq, rem_node.clone());
        }
        c.children.insert(rem_first, rem_node);
    }

    /// Drop `owner`'s claim on `seq_hashes` at `tier`. Each seq hash resolves
    /// to its node via `anchors` and its index in that node's edge; a miss
    /// (block not stored here, or its node already dropped) is skipped. Other
    /// owners/tiers survive.
    pub fn remove(&self, owner: &ResidencyOwner, tier: StorageTier, seq_hashes: &[Arc<str>]) {
        let key = (owner.clone(), tier);
        for seq in seq_hashes {
            let node = match self.anchors.get(seq) {
                Some(n) => n.clone(),
                None => continue,
            };
            let mut removed = false;
            if let Some(mut claims) = self.reverse.get_mut(&key) {
                if let Some(pos) = claims.iter().position(|h| h == seq) {
                    claims.swap_remove(pos);
                    removed = true;
                }
            }
            self.remove_claim(&node, seq, &key);
            if removed {
                if let Some(mut c) = self.counts.get_mut(&key) {
                    if *c > 0 {
                        *c -= 1;
                    }
                }
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
                (ResidencyOwner::Worker { source: s1, dp_rank: r1, .. },
                 ResidencyOwner::Worker { source: s2, dp_rank: r2, .. }) => s1 == s2 && r1 == r2,
                (ResidencyOwner::CacheOwner { pool_id: p1, .. },
                 ResidencyOwner::CacheOwner { pool_id: p2, .. }) => p1 == p2,
                _ => false,
            };
            if !same {
                continue;
            }
            if let Some((_, seqs)) = self.reverse.remove(&key) {
                for seq in &seqs {
                    // Clone out of the anchors shard before the node write:
                    // holding it across remove_claim's node.write() deadlocks store.
                    let node = match self.anchors.get(seq) {
                        Some(n) => n.clone(),
                        None => continue,
                    };
                    self.remove_claim(&node, seq, &key);
                }
            }
            self.counts.remove(&key);
        }
    }

    /// Drop `key` from the per-block claim set of the edge block whose seq is
    /// `seq` on `node`.
    fn remove_claim(&self, node: &NodeRef, seq: &Arc<str>, key: &OwnerTier) {
        let mut n = node.write();
        if let Some(idx) = n.edge.iter().position(|(s, _)| s == seq) {
            if let Some(claims) = n.claims.get_mut(idx) {
                claims.retain(|c| c != key);
            }
        }
    }

    /// Walk the request's `local_hashes` from the root; per `(target, tier)`,
    /// the matched depth is the deepest `d` where the target owns a contiguous
    /// `(owner, tier)` claim on every node in `[0, d)`. Each edge element's
    /// claims are the claims of the node that edge belongs to.
    pub fn find_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
        let mut node_claims: Vec<Vec<OwnerTier>> = Vec::with_capacity(query.local_hashes.len());
        let mut cur = self.root.clone();
        let mut qi = 0; // index into query.local_hashes
        while qi < query.local_hashes.len() {
            let child_local = &query.local_hashes[qi];
            let child = {
                let n = cur.read();
                n.children.get(child_local).cloned()
            };
            let child = match child {
                Some(c) => c,
                None => break,
            };
            // Snapshot read; a concurrent split may shrink this edge after. Safe —
            // claims move with their block, so a stale snapshot is at worst suboptimal.
            let (edge, claims) = {
                let c = child.read();
                (c.edge.clone(), c.claims.clone())
            };
            // Advance through as many edge elements as the query matches.
            let mut advanced = 0;
            while advanced < edge.len()
                && qi + advanced < query.local_hashes.len()
                && edge[advanced].1 == query.local_hashes[qi + advanced]
            {
                node_claims.push(claims[advanced].clone());
                advanced += 1;
            }
            if advanced == 0 {
                break;
            }
            qi += advanced;
            cur = child;
        }

        // Per (target, tier), advance depth only while the owner claims every
        // node [0, depth] contiguously. The index stores by owner but answers
        // by target; locality is derived from the owner domain.
        let mut best: HashMap<(RoutableTarget, StorageTier), TierMatch> = HashMap::new();
        for (depth, owners) in node_claims.iter().enumerate() {
            for (owner, tier) in owners {
                if !query.tiers_of_interest.contains(tier) {
                    continue;
                }
                let target = owner.target();
                let key = (target.clone(), *tier);
                let entry = best.entry(key).or_insert(TierMatch {
                    target: target.clone(),
                    tier: *tier,
                    matched_depth: 0,
                    locality: owner.locality(),
                });
                if (entry.matched_depth as usize) == depth {
                    entry.matched_depth = (depth + 1) as u32;
                }
                entry.locality = owner.locality();
            }
        }
        best.into_values().collect()
    }
}

/// One radix-tree index per cache identity, sharded by `group_idx`. Implements
/// `TieredMatchProvider` directly.
pub struct KvBlockIndexer {
    /// One shard per `group_idx` (kv-cache-spec group).
    // ponytail: group_idx is None on the wire → 0; single-spec models have one
    // group and multi-spec always emits group_idx, so 0 is a safe default.
    shards: DashMap<u32, Arc<GroupShard>>,
}

impl KvBlockIndexer {
    pub fn new() -> Self {
        KvBlockIndexer { shards: DashMap::new() }
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

    pub fn find_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
        self.shard(query.group_idx).find_matches(query)
    }
}

impl Default for KvBlockIndexer {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::kv_index::TieredMatchProvider for KvBlockIndexer {
    fn find_tiered_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
        self.find_matches(query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_index::types::SourceId;

    fn worker(name: &str) -> ResidencyOwner {
        ResidencyOwner::Worker { source: SourceId::from(name), dp_rank: 0, incarnation: 0 }
    }

    fn worker_rank(name: &str, rank: u32) -> ResidencyOwner {
        ResidencyOwner::Worker { source: SourceId::from(name), dp_rank: rank, incarnation: 0 }
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

    fn depth_of(res: &[TierMatch], target_id: &str, tier: StorageTier) -> u32 {
        res.iter()
            .find(|m| &*m.target.instance_id == target_id && m.tier == tier)
            .map(|m| m.matched_depth)
            .unwrap_or(0)
    }

    #[test]
    fn store_and_match_single_owner() {
        let idx = KvBlockIndexer::new();
        let blocks = vec![(h("h0"), h("h0")), (h("h1"), h("h1")), (h("h2"), h("h2"))];
        idx.store(0, worker("w0"), StorageTier::Device, None, &blocks);

        let res = idx.find_matches(&query(&["h0", "h1", "h2", "h3"], &[StorageTier::Device]));
        assert_eq!(res.len(), 1);
        // h3 absent → match depth 3.
        assert_eq!(depth_of(&res, "w0", StorageTier::Device), 3);
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
        assert_eq!(depth_of(&res, "w0", StorageTier::HostPinned), 1);
    }

    #[test]
    fn partial_prefix_match_depth() {
        let idx = KvBlockIndexer::new();
        let blocks = vec![(h("h0"), h("h0")), (h("h1"), h("h1"))];
        idx.store(0, worker("w0"), StorageTier::Device, None, &blocks);

        // Request has h0,h1,h2 but only h0,h1 are resident → depth 2.
        let res = idx.find_matches(&query(&["h0", "h1", "h2"], &[StorageTier::Device]));
        assert_eq!(depth_of(&res, "w0", StorageTier::Device), 2);
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
        assert_eq!(depth_of(&res, "w0", StorageTier::Device), 1);
    }

    #[test]
    fn interior_remove_breaks_prefix_descendants_survive() {
        // [a, b, c] as separate parent-chained nodes. Removing b (interior)
        // stops the prefix walk at a; c's claim survives but isn't credited
        // past the gap. Re-storing b re-links and credits the full prefix.
        let idx = KvBlockIndexer::new();
        idx.store(0, worker("w0"), StorageTier::Device, None, &[(h("sa"), h("a"))]);
        idx.store(0, worker("w0"), StorageTier::Device, Some(h("sa")), &[(h("sb"), h("b"))]);
        idx.store(0, worker("w0"), StorageTier::Device, Some(h("sb")), &[(h("sc"), h("c"))]);

        idx.remove(0, &worker("w0"), StorageTier::Device, &[h("sb")]);
        let res = idx.find_matches(&query(&["a", "b", "c"], &[StorageTier::Device]));
        assert_eq!(depth_of(&res, "w0", StorageTier::Device), 1);

        idx.store(0, worker("w0"), StorageTier::Device, Some(h("sa")), &[(h("sb"), h("b"))]);
        let res = idx.find_matches(&query(&["a", "b", "c"], &[StorageTier::Device]));
        assert_eq!(depth_of(&res, "w0", StorageTier::Device), 3);
    }

    #[test]
    fn two_owners_overlapping_prefix_per_target_depth() {
        let idx = KvBlockIndexer::new();
        // w0 has h0,h1,h2; w1 has h0,h1.
        idx.store(0, worker("w0"), StorageTier::Device, None, &[(h("h0"), h("h0")), (h("h1"), h("h1")), (h("h2"), h("h2"))]);
        idx.store(0, worker("w1"), StorageTier::Device, None, &[(h("h0"), h("h0")), (h("h1"), h("h1"))]);

        let res = idx.find_matches(&query(&["h0", "h1", "h2", "h3"], &[StorageTier::Device]));
        assert_eq!(res.len(), 2);
        assert_eq!(depth_of(&res, "w0", StorageTier::Device), 3);
        assert_eq!(depth_of(&res, "w1", StorageTier::Device), 2);
    }

    #[test]
    fn clear_worker_domain_does_not_cross_into_cache_owner() {
        let idx = KvBlockIndexer::new();
        idx.store(0, pool("mooncake", "w0"), StorageTier::External, None, &[(h("h0"), h("h0"))]);
        idx.store(0, worker("w0"), StorageTier::Device, None, &[(h("h0"), h("h0"))]);

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
        idx.store(0, worker("w0"), StorageTier::Device, None, &[(h("h0"), h("h0"))]);
        idx.store(0, worker("w0"), StorageTier::HostPinned, None, &[(h("h0"), h("h0"))]);

        idx.clear(&worker("w0"), ClearScope::Worker);
        assert!(idx.find_matches(&query(&["h0"], &[StorageTier::Device])).is_empty());
        assert!(idx.find_matches(&query(&["h0"], &[StorageTier::HostPinned])).is_empty());
    }

    #[test]
    fn clear_is_per_rank() {
        // Same source, two ranks; clearing rank 0 leaves rank 1 intact.
        let idx = KvBlockIndexer::new();
        idx.store(0, worker_rank("w0", 0), StorageTier::Device, None, &[(h("h0"), h("h0"))]);
        idx.store(0, worker_rank("w0", 1), StorageTier::Device, None, &[(h("h0"), h("h0"))]);

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
        idx.store(0, pool("poolA", "w0"), StorageTier::External, None, &[(h("h0"), h("h0"))]);
        idx.store(0, pool("poolB", "w0"), StorageTier::External, None, &[(h("h0"), h("h0"))]);
        idx.store(0, worker("w0"), StorageTier::Device, None, &[(h("h0"), h("h0"))]);

        idx.clear(&pool("poolA", "w0"), ClearScope::CacheOwner);

        let shard = idx.shards.get(&0).expect("group 0 shard exists");
        // poolA cleared; poolB intact; Worker untouched.
        assert!(shard.counts.get(&(pool("poolA", "w0"), StorageTier::External)).is_none());
        assert_eq!(*shard.counts.get(&(pool("poolB", "w0"), StorageTier::External)).unwrap(), 1);
        assert_eq!(*shard.counts.get(&(worker("w0"), StorageTier::Device)).unwrap(), 1);
    }

    #[test]
    fn cache_owner_hit_attributed_to_pushing_worker() {
        // A pool block pushed by w0: the hit's target is w0 (it can read what
        // it pushed), at the External tier, Remote locality.
        let idx = KvBlockIndexer::new();
        idx.store(0, pool("mooncake", "w0"), StorageTier::External, None, &[(h("h0"), h("h0"))]);

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
        idx.store(0, worker("w0"), StorageTier::Device, None, &[(h("seq"), h("loc"))]);
        idx.store(1, worker("w0"), StorageTier::Device, None, &[(h("seq"), h("loc"))]);

        // Each query sees only its own group's residency.
        assert_eq!(idx.find_matches(&query_group(0, &["loc"], &[StorageTier::Device])).len(), 1);
        assert_eq!(idx.find_matches(&query_group(1, &["loc"], &[StorageTier::Device])).len(), 1);

        // Remove in group 0 leaves group 1 intact: separate anchor maps,
        // so group 0's remove can't clobber group 1's node.
        idx.remove(0, &worker("w0"), StorageTier::Device, &[h("seq")]);
        assert!(idx.find_matches(&query_group(0, &["loc"], &[StorageTier::Device])).is_empty());
        assert_eq!(idx.find_matches(&query_group(1, &["loc"], &[StorageTier::Device])).len(), 1);
    }

    #[test]
    fn repeated_tokens_do_not_collapse_across_chain_positions() {
        // Same token block ("x") at two chain positions reached via different
        // parents: each stores as a distinct child under a distinct node, so
        // the prefix walk matches through the second occurrence too.
        let idx = KvBlockIndexer::new();
        idx.store(0, worker("w0"), StorageTier::Device, None, &[(h("sa"), h("a"))]);
        idx.store(0, worker("w0"), StorageTier::Device, Some(h("sa")), &[(h("sx1"), h("x"))]);
        idx.store(0, worker("w0"), StorageTier::Device, Some(h("sx1")), &[(h("sb"), h("b"))]);
        idx.store(0, worker("w0"), StorageTier::Device, Some(h("sb")), &[(h("sx2"), h("x"))]);
        idx.store(0, worker("w0"), StorageTier::Device, Some(h("sx2")), &[(h("sc"), h("c"))]);

        let res = idx.find_matches(&query(&["a", "x", "b", "x", "c"], &[StorageTier::Device]));
        assert_eq!(depth_of(&res, "w0", StorageTier::Device), 5);
    }

    #[test]
    fn store_partial_match_splits_edge_and_reanchors_remainder() {
        // A 4-block edge, then a store diverging at depth 2: the edge splits,
        // the remainder re-anchors to a new child, the diverging block appends
        // as a sibling. Both branches stay reachable; the relocated seqs stay
        // findable by remove.
        let idx = KvBlockIndexer::new();
        let long = vec![
            (h("s0"), h("h0")), (h("s1"), h("h1")), (h("s2"), h("h2")), (h("s3"), h("h3")),
        ];
        idx.store(0, worker("w0"), StorageTier::Device, None, &long);
        idx.store(0, worker("w0"), StorageTier::Device, None, &[
            (h("s0"), h("h0")), (h("s1"), h("h1")), (h("s4"), h("h4")),
        ]);

        assert_eq!(depth_of(&idx.find_matches(&query(&["h0", "h1", "h2", "h3"], &[StorageTier::Device])), "w0", StorageTier::Device), 4);
        assert_eq!(depth_of(&idx.find_matches(&query(&["h0", "h1", "h4"], &[StorageTier::Device])), "w0", StorageTier::Device), 3);

        idx.remove(0, &worker("w0"), StorageTier::Device, &[h("s2")]);
        assert_eq!(depth_of(&idx.find_matches(&query(&["h0", "h1", "h2", "h3"], &[StorageTier::Device])), "w0", StorageTier::Device), 2);
    }

    #[test]
    fn clear_all_scope_wipes_owner_domain_without_crossing_domains() {
        let idx = KvBlockIndexer::new();
        idx.store(0, worker("w0"), StorageTier::Device, None, &[(h("h0"), h("h0"))]);
        idx.store(0, pool("mooncake", "w0"), StorageTier::External, None, &[(h("h0"), h("h0"))]);
        idx.store(0, worker("w1"), StorageTier::Device, None, &[(h("h0"), h("h0"))]);

        idx.clear(&worker("w0"), ClearScope::All);

        let dev = idx.find_matches(&query(&["h0"], &[StorageTier::Device]));
        assert_eq!(dev.len(), 1);
        assert_eq!(&*dev[0].target.instance_id, "w1");
        let ext = idx.find_matches(&query(&["h0"], &[StorageTier::External]));
        assert_eq!(ext.len(), 1);
        assert_eq!(&*ext[0].target.instance_id, "w0");
    }
}
