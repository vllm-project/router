//! Concurrent radix-tree index of resident KV blocks, keyed by local hash.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock;

use crate::kv_index::types::{ClearScope, Locality, ResidencyOwner, RoutableTarget, StorageTier};

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
    /// Historical engine identities for each logical block, parallel to `edge`.
    /// Topology and parent anchors survive removal of residency claims.
    aliases: Vec<HashSet<Arc<str>>>,
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
            aliases: Vec::new(),
            claims: Vec::new(),
            children: HashMap::new(),
        }
    }
    fn with_edge(edge: Vec<EdgeBlock>) -> Self {
        let claims = (0..edge.len()).map(|_| Vec::new()).collect();
        let aliases = (0..edge.len()).map(|_| HashSet::new()).collect();
        TreeNode {
            edge,
            aliases,
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
    // grows it without limit. Revisit with an epoch leaf sweep if growth shows.
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
        let key = (owner, tier);
        // Only the first unknown parent falls back to the root. Later cursors
        // were recorded by this store and must follow their current anchor.
        let mut cursor = parent_seq.filter(|seq| self.anchors.contains_key(seq));
        let mut bi = 0;
        while bi < blocks.len() {
            let node = cursor
                .as_ref()
                .and_then(|seq| self.anchors.get(seq).map(|n| n.clone()))
                .unwrap_or_else(|| self.root.clone());
            let mut parent = node.write();
            if let Some(seq) = &cursor {
                let Some(pos) = parent
                    .aliases
                    .iter()
                    .position(|aliases| aliases.contains(seq))
                else {
                    // A split moved the cursor after its anchor was cloned.
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
                let new_node = Arc::new(RwLock::new(TreeNode::with_edge(blocks[bi..].to_vec())));
                let mut new = new_node.write();
                self.record_locked(&mut new, &new_node, &key, &blocks[bi..]);
                parent
                    .children
                    .insert(blocks[bi].1.clone(), new_node.clone());
                return;
            };
            drop(parent);
            let mut c = child.write();
            let matched = edge_common_prefix(&c.edge, &blocks[bi..]);
            if matched < c.edge.len() {
                self.split_locked(&mut c, matched);
            }
            self.record_locked(&mut c, &child, &key, &blocks[bi..bi + matched]);
            bi += matched;
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
        incoming: &[EdgeBlock],
    ) {
        for (i, (seq, _)) in incoming.iter().enumerate() {
            n.aliases[i].insert(seq.clone());
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
        let rem_aliases = c.aliases.split_off(pos);
        let rem_claims: Vec<Vec<OwnerTier>> = c.claims.split_off(pos);
        let old_children = std::mem::take(&mut c.children);
        let rem_first = remainder[0].1.clone();
        let rem_node = Arc::new(RwLock::new(TreeNode {
            edge: remainder,
            aliases: rem_aliases,
            claims: rem_claims,
            children: old_children,
        }));
        for aliases in &rem_node.read().aliases {
            for seq in aliases {
                self.anchors.insert(seq.clone(), rem_node.clone());
            }
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

    /// Drop the logical block claim only after its last active alias is gone.
    fn remove_claim(&self, seq: &Arc<str>, key: &OwnerTier) {
        loop {
            // Never hold an anchors map guard while taking the node lock.
            let Some(node) = self.anchors.get(seq).map(|n| n.clone()) else {
                return;
            };
            let mut n = node.write();
            let Some(idx) = n.aliases.iter().position(|aliases| aliases.contains(seq)) else {
                // A concurrent split moved this alias; resolve it again.
                continue;
            };
            let active = self
                .reverse
                .get(key)
                .is_some_and(|seqs| n.aliases[idx].iter().any(|alias| seqs.contains(alias)));
            if !active {
                n.claims[idx].retain(|claim| claim != key);
            }
            return;
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
        // A 0-depth entry = no prefix match; absence, not a value to filter.
        best.into_values().filter(|m| m.matched_depth > 0).collect()
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
        KvBlockIndexer {
            shards: DashMap::new(),
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

    fn depth_of(res: &[TierMatch], target_id: &str, tier: StorageTier) -> u32 {
        res.iter()
            .find(|m| &*m.target.instance_id == target_id && m.tier == tier)
            .map(|m| m.matched_depth)
            .unwrap_or(0)
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
                .aliases
                .iter()
                .position(|aliases| aliases.contains(seq))
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
        assert_eq!(node.read().edge[0].0, h("i:900"));

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
        assert_eq!(node.read().aliases[0].len(), 2);
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
                .aliases
                .iter()
                .position(|aliases| aliases.contains(seq))
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
        assert_eq!(depth_of(&res, "w0", StorageTier::Device), 1);

        idx.store(
            0,
            worker("w0"),
            StorageTier::Device,
            Some(h("sa")),
            &[(h("sb"), h("b"))],
        );
        let res = idx.find_matches(&query(&["a", "b", "c"], &[StorageTier::Device]));
        assert_eq!(depth_of(&res, "w0", StorageTier::Device), 3);
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
        assert_eq!(depth_of(&res, "w0", StorageTier::Device), 3);
        assert_eq!(depth_of(&res, "w1", StorageTier::Device), 2);
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
            depth_of(
                &idx.find_matches(&query(&["h0", "h1", "h2", "h3"], &[StorageTier::Device])),
                "w0",
                StorageTier::Device
            ),
            4
        );
        assert_eq!(
            depth_of(
                &idx.find_matches(&query(&["h0", "h1", "h4"], &[StorageTier::Device])),
                "w0",
                StorageTier::Device
            ),
            3
        );

        idx.remove(0, &worker("w0"), StorageTier::Device, &[h("s2")]);
        assert_eq!(
            depth_of(
                &idx.find_matches(&query(&["h0", "h1", "h2", "h3"], &[StorageTier::Device])),
                "w0",
                StorageTier::Device
            ),
            2
        );
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
}
