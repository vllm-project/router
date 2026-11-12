//! Concurrent radix-tree index of resident KV blocks, keyed by local hash.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::{mapref::entry::Entry, DashMap};
use parking_lot::RwLock;

use crate::kv_hash::SequenceHash;
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
/// Historical sequence aliases are publisher-scoped: identical opaque hashes
/// at different sources need not describe the same path.
#[derive(Default)]
struct Aliases {
    scoped: HashMap<RoutableTarget, HashSet<Arc<str>>>,
}

impl Aliases {
    fn has(&self, target: &RoutableTarget, seq: &Arc<str>) -> bool {
        self.scoped
            .get(target)
            .is_some_and(|hashes| hashes.contains(seq))
    }

    #[cfg(test)]
    fn contains(&self, seq: &str) -> bool {
        self.scoped.values().any(|hashes| hashes.contains(seq))
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.scoped.values().map(HashSet::len).sum()
    }
}

type EdgeBlock = (Aliases, Arc<str>); // (historical scoped aliases, local_hash)

#[cfg(all(test, feature = "cpu", feature = "ready"))]
std::thread_local! {
    static AFTER_ANCHOR_CLONE: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    static AFTER_QUERY_EDGE: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    static AFTER_QUERY_UNLOCK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    static AFTER_QUERY_SOURCE_LOOKUP: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
fn after_anchor_clone() {
    let hook = AFTER_ANCHOR_CLONE.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
fn after_query_edge() {
    let hook = AFTER_QUERY_EDGE.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
fn after_query_unlock() {
    let hook = AFTER_QUERY_UNLOCK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
fn after_query_source_lookup() {
    let hook = AFTER_QUERY_SOURCE_LOOKUP.with(|slot| slot.borrow_mut().take());
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
    fn with_edge(blocks: &[IncomingBlock], target: &RoutableTarget) -> Self {
        let claims = (0..blocks.len()).map(|_| Vec::new()).collect();
        TreeNode {
            edge: blocks
                .iter()
                .map(|(seq, local)| {
                    (
                        Aliases {
                            scoped: HashMap::from([(target.clone(), HashSet::from([seq.clone()]))]),
                        },
                        local.clone(),
                    )
                })
                .collect(),
            claims,
            children: HashMap::new(),
        }
    }
}

fn unique_alias_position(node: &NodeRef, target: &RoutableTarget, seq: &Arc<str>) -> Option<usize> {
    let node = node.read();
    let mut positions = node
        .edge
        .iter()
        .enumerate()
        .filter_map(|(i, (aliases, _))| aliases.has(target, seq).then_some(i));
    let position = positions.next()?;
    positions.next().is_none().then_some(position)
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
    /// Engine seq hash → tree node, so a store jumps to its parent and a
    /// `BlockRemoved` (carrying only seq hashes) finds the node to drop.
    // ponytail: never trimmed — topology outlives residency; unique-prefix churn
    // grows topology and historical alias sets without limit. Revisit with an
    // epoch leaf sweep if growth shows.
    // The same opaque integer may refer to different paths at two publishers.
    // Keep parent anchors scoped by the publishing target, not by hash alone.
    scoped_anchors: DashMap<Arc<str>, HashMap<RoutableTarget, NodeRef>>,
    /// Reverse lookup: (owner, tier) → seq hashes claimed.
    reverse: DashMap<OwnerTier, HashSet<Arc<str>>>,
}

impl GroupShard {
    fn new() -> Self {
        GroupShard {
            topology: RwLock::new(()),
            root: Arc::new(RwLock::new(TreeNode::root())),
            scoped_anchors: DashMap::new(),
            reverse: DashMap::new(),
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
    ) -> bool {
        let key = (owner, tier);
        let target = key.0.target();
        if blocks.is_empty() {
            return true;
        }
        // A parent may borrow only a uniquely established foreign content
        // position; this is topology evidence, never a residency claim.
        let parent = if let Some(seq) = &parent_seq {
            match self.resolve_parent(&target, seq) {
                Ok(Some(parent)) => Some(parent),
                Ok(None) => {
                    tracing::debug!(parent_seq = %seq, "kv_index: unknown or ambiguous parent; skipping unanchored store");
                    return true;
                }
                Err(()) => return false,
            }
        } else {
            None
        };
        if !self.store_path_is_consistent(&target, parent.as_ref(), parent_seq.as_ref(), blocks) {
            return false;
        }
        if let (Some(seq), Some((node, offset, true))) = (&parent_seq, &parent) {
            let mut node_guard = node.write();
            node_guard.edge[*offset]
                .0
                .scoped
                .entry(target.clone())
                .or_default()
                .insert(seq.clone());
            self.scoped_anchors
                .entry(seq.clone())
                .or_default()
                .insert(target.clone(), node.clone());
        }
        let mut cursor = parent_seq;
        let mut bi = 0;
        while bi < blocks.len() {
            let node = match &cursor {
                Some(seq) => self
                    .anchor(&target, seq)
                    .expect("known cursor must remain anchored; anchors are insert-only"),
                None => self.root.clone(),
            };
            #[cfg(all(test, feature = "cpu", feature = "ready"))]
            after_anchor_clone();
            let mut parent = node.write();
            if let Some(seq) = &cursor {
                let Some(pos) = parent
                    .edge
                    .iter()
                    .position(|(aliases, _)| aliases.has(&target, seq))
                else {
                    // The same split-only progress argument as remove_claim
                    // applies to this fixed cursor; resolve its moved anchor.
                    drop(parent);
                    std::thread::yield_now();
                    continue;
                };
                if pos + 1 < parent.edge.len() {
                    self.split_locked(&mut parent, &node, pos + 1);
                }
            }
            let child = parent.children.get(&blocks[bi].1).cloned();
            let Some(child) = child else {
                // Keep the parent locked from lookup through publication so
                // another writer cannot replace a newly installed child.
                let new_node = Arc::new(RwLock::new(TreeNode::with_edge(&blocks[bi..], &target)));
                let mut new = new_node.write();
                self.record_locked(&mut new, &new_node, &key, &blocks[bi..]);
                parent
                    .children
                    .insert(blocks[bi].1.clone(), new_node.clone());
                return true;
            };
            drop(parent);
            // Match, split, and claim under one write lock: a concurrent split
            // can't shorten this edge between the match and the split.
            let mut c = child.write();
            let matched = edge_common_prefix(&c.edge, &blocks[bi..]);
            debug_assert!(matched >= 1, "child key matches its first edge block");
            if matched < c.edge.len() {
                self.split_locked(&mut c, &child, matched);
            }
            self.record_locked(&mut c, &child, &key, &blocks[bi..bi + matched]);
            bi += matched;
            // record_locked registered this incoming key, so its current
            // anchor remains a valid cursor even if a later split moves it.
            cursor = Some(blocks[bi - 1].0.clone());
        }
        true
    }

    /// Verify every previously seen source hash resolves to the exact same
    /// logical path and edge offset before mutating any part of this store.
    /// Production callers hold the domain write lock throughout the check and
    /// store; independent source hashes may legitimately have the same bytes.
    fn store_path_is_consistent(
        &self,
        target: &RoutableTarget,
        parent: Option<&(NodeRef, usize, bool)>,
        parent_seq: Option<&Arc<str>>,
        blocks: &[IncomingBlock],
    ) -> bool {
        let mut path = Some(match parent {
            None => (self.root.clone(), 0),
            Some((node, position, _)) => (node.clone(), position + 1),
        });
        let mut incoming = HashSet::with_capacity(blocks.len());
        if let Some(parent) = parent_seq {
            incoming.insert(parent.clone());
        }
        for (seq, local) in blocks {
            // Each entry is a different prefix position, even if local tokens
            // repeat; an identical sequence hash cannot name both safely.
            if !incoming.insert(seq.clone()) {
                return false;
            }
            let expected = path.take().and_then(|(node, offset)| {
                let current = node.read();
                if offset < current.edge.len() {
                    (current.edge[offset].1 == *local).then(|| (node.clone(), offset))
                } else {
                    current.children.get(local).cloned().map(|child| (child, 0))
                }
            });
            if let Some(anchor) = self.anchor(target, seq) {
                let Some((node, offset)) = &expected else {
                    return false;
                };
                if !Arc::ptr_eq(&anchor, node)
                    || !node
                        .read()
                        .edge
                        .get(*offset)
                        .is_some_and(|(aliases, _)| aliases.has(target, seq))
                {
                    return false;
                }
            }
            path = expected.map(|(node, offset)| (node, offset + 1));
        }
        true
    }

    /// Return the exact source-scoped parent, or borrow one unambiguous global
    /// topology anchor for a full byte digest / legacy opaque identifier.
    /// A foreign integer or short byte hash is never adequate identity proof.
    fn resolve_parent(
        &self,
        target: &RoutableTarget,
        seq: &Arc<str>,
    ) -> Result<Option<(NodeRef, usize, bool)>, ()> {
        if let Some(node) = self.anchor(target, seq) {
            let position = unique_alias_position(&node, target, seq).ok_or(())?;
            return Ok(Some((node, position, false)));
        }
        if seq.starts_with("i:") {
            return Ok(None);
        }
        if let Some(digest) = seq.strip_prefix("b:") {
            if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Ok(None);
            }
        }
        let Some(anchors) = self.scoped_anchors.get(seq) else {
            return Ok(None);
        };
        let candidates: Vec<_> = anchors
            .iter()
            .map(|(scope, node)| (scope.clone(), node.clone()))
            .collect();
        drop(anchors);
        let mut found: Option<(NodeRef, usize)> = None;
        for (scope, node) in candidates {
            let Some(position) = unique_alias_position(&node, &scope, seq) else {
                return Ok(None);
            };
            if let Some((known, known_position)) = &found {
                if !Arc::ptr_eq(known, &node) || *known_position != position {
                    return Ok(None);
                }
            } else {
                found = Some((node, position));
            }
        }
        Ok(found.map(|(node, position)| (node, position, true)))
    }

    /// Attach a sparse fact only when an existing source-scoped anchor proves
    /// one exact tree position; an unanchored sparse fact remains detached.
    fn attach_existing(&self, owner: &ResidencyOwner, tier: StorageTier, seq: &Arc<str>) {
        let target = owner.target();
        let Some(node) = self.anchor(&target, seq) else {
            return;
        };
        let mut node = node.write();
        let mut positions = node
            .edge
            .iter()
            .enumerate()
            .filter_map(|(i, (aliases, _))| aliases.has(&target, seq).then_some(i));
        let Some(position) = positions.next() else {
            return;
        };
        if positions.next().is_some() {
            return;
        }
        drop(positions);
        let claim = (owner.clone(), tier);
        if !node.claims[position].contains(&claim) {
            node.claims[position].push(claim);
        }
    }

    /// Record incoming identities into anchors and reverse, and
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
            n.edge[i]
                .0
                .scoped
                .entry(key.0.target())
                .or_default()
                .insert(seq.clone());
            self.scoped_anchors
                .entry(seq.clone())
                .or_default()
                .insert(key.0.target(), node.clone());
            self.reverse
                .entry(key.clone())
                .or_default()
                .insert(seq.clone());
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
    fn split_locked(&self, c: &mut TreeNode, original: &NodeRef, pos: usize) {
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
            for (target, seqs) in &aliases.scoped {
                for seq in seqs {
                    if let Some(mut anchors) = self.scoped_anchors.get_mut(seq) {
                        if let Some(anchor) = anchors.get_mut(target) {
                            if Arc::ptr_eq(anchor, original) {
                                *anchor = rem_node.clone();
                            }
                        }
                    }
                }
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
            if let Some(mut claims) = self.reverse.get_mut(&key) {
                claims.remove(seq);
            }
            self.remove_claim(seq, &key);
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
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        self.scoped_anchors.retain(|_, anchors| {
            anchors.remove(&target);
            !anchors.is_empty()
        });
        self.prune_source_paths(&target);
    }

    /// After all facts and anchors for a source are gone, remove its historical
    /// aliases and empty leaves. Retain every node with a live alias, claim, or
    /// descendant: another source may still use it as a parent anchor.
    fn prune_source_paths(&self, target: &RoutableTarget) {
        let mut stack = vec![(self.root.clone(), false)];
        while let Some((node, visited)) = stack.pop() {
            if !visited {
                stack.push((node.clone(), true));
                let children: Vec<_> = node.read().children.values().cloned().collect();
                stack.extend(children.into_iter().map(|child| (child, false)));
                continue;
            }
            let mut current = node.write();
            for (aliases, _) in &mut current.edge {
                aliases.scoped.remove(target);
            }
            current.children.retain(|_, child| {
                let child = child.read();
                !child.children.is_empty()
                    || child
                        .edge
                        .iter()
                        .any(|(aliases, _)| !aliases.scoped.is_empty())
                    || child.claims.iter().any(|claims| !claims.is_empty())
            });
        }
    }

    fn anchor(&self, target: &RoutableTarget, seq: &Arc<str>) -> Option<NodeRef> {
        self.scoped_anchors
            .get(seq)
            .and_then(|entries| entries.get(target).cloned())
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
            let Some(node) = self.anchor(&key.0.target(), seq) else {
                return;
            };
            #[cfg(all(test, feature = "cpu", feature = "ready"))]
            after_anchor_clone();
            let mut n = node.write();
            let target = key.0.target();
            let Some(idx) = n
                .edge
                .iter()
                .position(|(aliases, _)| aliases.has(&target, seq))
            else {
                // A concurrent split moved this alias; resolve it again.
                drop(n);
                std::thread::yield_now();
                continue;
            };
            let active = self.reverse.get(key).is_some_and(|seqs| {
                n.edge[idx]
                    .0
                    .scoped
                    .get(&target)
                    .is_some_and(|aliases| aliases.iter().any(|alias| seqs.contains(alias)))
            });
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
    fn collect_claims(&self, query: &MatchQuery) -> Vec<Vec<OwnerTier>> {
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
                // Derived references are maintained in the same source or
                // global transaction as `reverse`, the authoritative facts.
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
            #[cfg(all(test, feature = "cpu", feature = "ready"))]
            after_query_edge();
        }

        node_claims
    }

    fn aggregate_claims(query: &MatchQuery, node_claims: &[Vec<OwnerTier>]) -> Vec<TierMatch> {
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

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    fn find_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
        Self::aggregate_claims(query, &self.collect_claims(query))
    }
}

/// Serializes one worker/rank across its cache groups.
struct SourceLock {
    access: RwLock<()>,
    generation: AtomicU64,
}

/// Identity and trust generation of one source at snapshot time.
pub(crate) struct SourceVersion {
    target: RoutableTarget,
    source: Arc<SourceLock>,
    generation: u64,
}

impl SourceLock {
    fn new() -> Self {
        Self {
            access: RwLock::new(()),
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
    /// Retained for lifecycle tests; production uses per-source generations.
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    source_generation: AtomicU64,
    /// Gates may be removed only by `retire_source` under the global barrier.
    source_locks: DashMap<RoutableTarget, Arc<SourceLock>>,
    /// Namespace directory for remove events, which carry no identity. This
    /// contains names only; `GroupShard::reverse` remains the sole fact table.
    sparse_namespaces: DashMap<ResidencyOwner, HashSet<String>>,
    /// Recomputable per-owner/identity cardinalities; never residency facts.
    /// Domain writes maintain the counters or invalidate them for dense edits.
    sparse_counts: DashMap<(ResidencyOwner, Option<String>), usize>,
    /// Ordinary access shares the barrier; pool clear and retirement exclude it.
    domain: RwLock<()>,
}

impl KvBlockIndexer {
    pub fn new() -> Self {
        KvBlockIndexer {
            shards: DashMap::new(),
            suspended_sources: DashMap::new(),
            #[cfg(all(test, feature = "cpu", feature = "ready"))]
            source_generation: AtomicU64::new(0),
            source_locks: DashMap::new(),
            sparse_namespaces: DashMap::new(),
            sparse_counts: DashMap::new(),
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
            .or_insert_with(|| Arc::new(SourceLock::new()))
            .clone()
    }

    /// Check a snapshot after dropping its locks. Ordinary store/remove is a
    /// valid snapshot transition; suspension, resume, or retirement invalidates
    /// the result. This does not reserve engine blocks.
    pub(crate) fn source_version_is_current(&self, version: &SourceVersion) -> bool {
        let current = self
            .source_locks
            .get(&version.target)
            .is_some_and(|entry| Arc::ptr_eq(entry.value(), &version.source));
        #[cfg(all(test, feature = "cpu", feature = "ready"))]
        after_query_source_lookup();
        current
            && version.source.generation.load(Ordering::Acquire) == version.generation
            && !self
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
        #[cfg(all(test, feature = "cpu", feature = "ready"))]
        self.source_generation.fetch_add(1, Ordering::AcqRel);
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
        let _guard = source_lock.access.write();
        self.invalidate_source_unlocked(source, dp_rank);
    }

    /// Permanently retire a fenced source. The global barrier drains all
    /// transactions before removing its gate; in-flight owned snapshots retain
    /// the old gate and observe its incremented generation.
    pub(crate) fn retire_source(&self, source: &SourceId, dp_rank: u32) {
        let _domain = self.domain.write();
        self.invalidate_source_unlocked(source, dp_rank);
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        self.source_locks.remove(&target);
        self.suspended_sources.remove(&(source.clone(), dp_rank));
    }

    fn invalidate_source_unlocked(&self, source: &SourceId, dp_rank: u32) {
        self.suspend_source_unlocked(source, dp_rank);
        let shards: Vec<Arc<GroupShard>> = self.shards.iter().map(|e| e.value().clone()).collect();
        for shard in shards {
            let _topology = shard.topology.write();
            shard.clear_source(source, dp_rank);
        }
        self.sparse_namespaces.retain(|owner, _| {
            let target = owner.target();
            target.instance_id != *source || target.dp_rank != dp_rank
        });
        self.sparse_counts.retain(|(owner, _), _| {
            let target = owner.target();
            target.instance_id != *source || target.dp_rank != dp_rank
        });
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
            #[cfg(all(test, feature = "cpu", feature = "ready"))]
            self.source_generation.fetch_add(1, Ordering::AcqRel);
            gate.generation.fetch_add(1, Ordering::AcqRel);
            entry.remove();
        }
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    pub(super) fn source_generation_for_test(&self) -> u64 {
        self.source_generation.load(Ordering::Acquire)
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
        #[cfg(all(test, feature = "cpu", feature = "ready"))]
        after_query_unlock();
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
    pub(crate) fn with_batch<R>(&self, f: impl FnOnce(&IndexBatch<'_>) -> R) -> R {
        let _guard = self.domain.write();
        f(&IndexBatch {
            index: self,
            scope: None,
        })
    }

    /// Test-only global snapshot for cases covering multiple owners.
    #[cfg(test)]
    pub(crate) fn with_snapshot<R>(&self, f: impl FnOnce(&IndexSnapshot<'_>) -> R) -> R {
        let _guard = self.domain.write();
        f(&IndexSnapshot {
            index: self,
            scope: None,
        })
    }

    /// A source batch is atomic for this target without blocking other targets.
    pub(crate) fn with_source_batch<R>(
        &self,
        target: &RoutableTarget,
        f: impl FnOnce(&IndexBatch<'_>) -> R,
    ) -> R {
        let _domain = self.domain.read();
        let source = self.source_lock(target);
        let _guard = source.access.write();
        f(&IndexBatch {
            index: self,
            scope: Some(target.clone()),
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
        let _guard = source.access.read();
        f(&IndexSnapshot {
            index: self,
            scope: Some(target.clone()),
        })
    }
}

/// A batch holds either the global barrier exclusively, or the barrier for
/// reading together with its scoped source write lock. Its closure must not
/// re-enter any locking `KvBlockIndexer` method.
pub(crate) struct IndexBatch<'a> {
    index: &'a KvBlockIndexer,
    scope: Option<RoutableTarget>,
}

/// A consistent view held under either the global barrier or a source lock.
pub(crate) struct IndexSnapshot<'a> {
    index: &'a KvBlockIndexer,
    scope: Option<RoutableTarget>,
}

fn sparse_digest(hash: &SequenceHash) -> [u8; 66] {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut key = [0; 66];
    key[0] = b'b';
    key[1] = b':';
    for (i, byte) in hash.iter().copied().enumerate() {
        key[2 + 2 * i] = HEX[usize::from(byte >> 4)];
        key[3 + 2 * i] = HEX[usize::from(byte & 0x0f)];
    }
    key
}

fn sparse_key(namespace: Option<&str>, hash: &SequenceHash) -> Arc<str> {
    let key = sparse_digest(hash);
    let digest = std::str::from_utf8(&key).expect("digest uses ASCII hex");
    match namespace {
        None => Arc::from(digest),
        Some(name) => Arc::from(format!("n:{}:{}:{}", name.len(), name, digest)),
    }
}

/// Borrowed lookups do not need an allocation per base-model hash.
pub(crate) enum PreparedKey {
    Inline([u8; 66]),
    Owned(Arc<str>),
}

impl PreparedKey {
    fn as_str(&self) -> &str {
        match self {
            Self::Inline(key) => std::str::from_utf8(key).expect("digest uses ASCII hex"),
            Self::Owned(key) => key.as_ref(),
        }
    }
}

/// Encode a request's hashes once outside the domain lock and reuse the result
/// for every group. Namespace is part of the key, never a query fallback.
pub(crate) fn prepare_sparse_keys(
    namespace: Option<&str>,
    hashes: &[SequenceHash],
) -> Vec<PreparedKey> {
    match namespace {
        None => hashes
            .iter()
            .map(|hash| PreparedKey::Inline(sparse_digest(hash)))
            .collect(),
        Some(name) => hashes
            .iter()
            .map(|hash| PreparedKey::Owned(sparse_key(Some(name), hash)))
            .collect(),
    }
}

fn in_namespace(key: &str, namespace: Option<&str>) -> bool {
    match namespace {
        None => !key.starts_with("n:"),
        Some(name) => key.starts_with(&format!("n:{}:{}:", name.len(), name)),
    }
}

const TIERS: [StorageTier; 4] = [
    StorageTier::Device,
    StorageTier::HostPinned,
    StorageTier::Disk,
    StorageTier::External,
];

impl IndexBatch<'_> {
    fn check_target(&self, target: &RoutableTarget) {
        if let Some(scope) = &self.scope {
            assert_eq!(scope, target, "source transaction crossed targets");
        } else {
            self.index.source_lock(target);
        }
    }

    #[cfg(test)]
    pub(crate) fn suspend_source(&self, source: &SourceId, rank: u32) {
        self.check_target(&RoutableTarget {
            instance_id: source.clone(),
            dp_rank: rank,
        });
        self.index.suspend_source_unlocked(source, rank);
    }

    pub(crate) fn resume_source(&self, source: &SourceId, rank: u32) {
        self.check_target(&RoutableTarget {
            instance_id: source.clone(),
            dp_rank: rank,
        });
        self.index.resume_source_unlocked(source, rank);
    }

    pub(crate) fn store(
        &self,
        group: u32,
        owner: ResidencyOwner,
        tier: StorageTier,
        parent: Option<Arc<str>>,
        blocks: &[(Arc<str>, Arc<str>)],
    ) -> bool {
        let target = owner.target();
        self.check_target(&target);
        self.index
            .sparse_counts
            .retain(|(candidate, _), _| candidate != &owner);
        let shard = self.index.shard(group);
        let topology = shard.topology.write();
        let accepted = shard.store(owner, tier, parent, blocks);
        drop(topology);
        if !accepted {
            // Hide and discard every observation from the ambiguous source
            // while the domain write lock still protects all readers.
            self.index
                .invalidate_source_unlocked(&target.instance_id, target.dp_rank);
        }
        accepted
    }

    pub(crate) fn remove(
        &self,
        group: u32,
        owner: &ResidencyOwner,
        tier: StorageTier,
        hashes: &[Arc<str>],
    ) {
        self.check_target(&owner.target());
        let shard = self.index.shard(group);
        let _topology = shard.topology.read();
        shard.remove(owner, tier, hashes);
        self.index
            .sparse_counts
            .retain(|(candidate, _), _| candidate != owner);
    }

    pub(crate) fn clear(&self, owner: &ResidencyOwner, scope: ClearScope) {
        if self.scope.is_some() {
            assert!(
                matches!(owner, ResidencyOwner::Worker { .. }) && scope == ClearScope::Worker,
                "cross-source clear requires a global transaction"
            );
        }
        self.check_target(&owner.target());
        let shards: Vec<_> = self
            .index
            .shards
            .iter()
            .map(|e| e.value().clone())
            .collect();
        for shard in shards {
            let _topology = shard.topology.read();
            shard.clear(owner, scope);
        }
        self.index.sparse_namespaces.retain(|candidate, _| {
            if !matches!(scope, ClearScope::All) && scope != candidate.domain() {
                return true;
            }
            !same_clear_owner(owner, candidate)
        });
        self.index.sparse_counts.retain(|(candidate, _), _| {
            (!matches!(scope, ClearScope::All) && scope != candidate.domain())
                || !same_clear_owner(owner, candidate)
        });
    }

    /// Remove only this precise owner, including its incarnation.
    pub(crate) fn clear_owner_exact(&self, owner: &ResidencyOwner) {
        self.check_target(&owner.target());
        let shards: Vec<_> = self
            .index
            .shards
            .iter()
            .map(|e| e.value().clone())
            .collect();
        for shard in shards {
            let _topology = shard.topology.read();
            for tier in TIERS {
                let key = (owner.clone(), tier);
                let hashes: Vec<_> = shard
                    .reverse
                    .get(&key)
                    .map(|v| v.iter().cloned().collect())
                    .unwrap_or_default();
                shard.remove(owner, tier, &hashes);
                shard.reverse.remove_if(&key, |_, hashes| hashes.is_empty());
            }
        }
        self.index.sparse_namespaces.remove(owner);
        self.index
            .sparse_counts
            .retain(|(candidate, _), _| candidate != owner);
    }

    fn count(&self, owner: &ResidencyOwner, namespace: Option<&str>) -> usize {
        let key = (owner.clone(), namespace.map(str::to_owned));
        if let Some(count) = self.index.sparse_counts.get(&key) {
            return *count;
        }
        let mut entries = HashSet::new();
        for entry in self.index.shards.iter() {
            let group = *entry.key();
            for tier in TIERS {
                if let Some(hashes) = entry.value().reverse.get(&(owner.clone(), tier)) {
                    for hash in hashes.iter().filter(|h| in_namespace(h, namespace)) {
                        entries.insert((group, hash.clone()));
                    }
                }
            }
        }
        let count = entries.len();
        self.index.sparse_counts.insert(key, count);
        count
    }

    /// Sparse events and radix-backed events update the same fact table.
    pub(crate) fn store_sparse(
        &self,
        owner: &ResidencyOwner,
        namespace: Option<&str>,
        group: u32,
        hash: &SequenceHash,
        tier: StorageTier,
        max_entries: usize,
    ) -> bool {
        self.check_target(&owner.target());
        let shard = self.index.shard(group);
        let hash = sparse_key(namespace, hash);
        let existed = TIERS.into_iter().any(|t| {
            shard
                .reverse
                .get(&(owner.clone(), t))
                .is_some_and(|set| set.contains(&hash))
        });
        if !existed && self.count(owner, namespace) >= max_entries {
            return false;
        }
        let key = (owner.clone(), tier);
        shard.reverse.entry(key).or_default().insert(hash.clone());
        if namespace.is_none() && !shard.scoped_anchors.is_empty() {
            let _topology = shard.topology.read();
            shard.attach_existing(owner, tier, &hash);
        }
        if let Some(name) = namespace {
            self.index
                .sparse_namespaces
                .entry(owner.clone())
                .or_default()
                .insert(name.to_owned());
        }
        if !existed {
            let key = (owner.clone(), namespace.map(str::to_owned));
            if let Some(mut count) = self.index.sparse_counts.get_mut(&key) {
                *count += 1;
            }
        }
        true
    }

    /// Removal events do not identify the namespace, so invalidate every
    /// matching namespace of this owner and group, but no other owner.
    pub(crate) fn remove_sparse(
        &self,
        owner: &ResidencyOwner,
        group: u32,
        hash: &SequenceHash,
        tier: Option<StorageTier>,
    ) {
        self.check_target(&owner.target());
        let shard = self.index.shard(group);
        let _topology = shard.topology.read();
        let mut candidates = vec![(None, sparse_key(None, hash))];
        if let Some(names) = self.index.sparse_namespaces.get(owner) {
            candidates.extend(
                names
                    .iter()
                    .map(|name| (Some(name.clone()), sparse_key(Some(name), hash))),
            );
        }
        let mut removed = HashSet::new();
        for t in TIERS
            .into_iter()
            .filter(|t| tier.is_none_or(|wanted| wanted == *t))
        {
            let key = (owner.clone(), t);
            let hits: Vec<_> = shard
                .reverse
                .get(&key)
                .map(|hashes| {
                    candidates
                        .iter()
                        .filter(|(_, candidate)| hashes.contains(candidate))
                        .cloned()
                        .collect()
                })
                .unwrap_or_default();
            let hashes: Vec<_> = hits.iter().map(|(_, hash)| hash.clone()).collect();
            removed.extend(hits);
            shard.remove(owner, t, &hashes);
        }
        // Capacity counts are an optimization, not residency facts.
        // Decrement only when this page has disappeared from every tier.
        for (namespace, hash) in removed {
            if TIERS.into_iter().any(|t| {
                shard
                    .reverse
                    .get(&(owner.clone(), t))
                    .is_some_and(|set| set.contains(&hash))
            }) {
                continue;
            }
            if let Some(mut count) = self
                .index
                .sparse_counts
                .get_mut(&(owner.clone(), namespace))
            {
                *count = count.saturating_sub(1);
            }
        }
    }
}

impl IndexSnapshot<'_> {
    fn check_target(&self, target: &RoutableTarget) {
        if let Some(scope) = &self.scope {
            assert_eq!(scope, target, "source snapshot crossed targets");
        }
    }

    pub(crate) fn source_version(&self, target: &RoutableTarget) -> SourceVersion {
        self.check_target(target);
        let source = self
            .index
            .source_locks
            .get(target)
            .map(|entry| entry.value().clone())
            .expect("source snapshot must have a registered gate");
        let generation = source.generation.load(Ordering::Acquire);
        SourceVersion {
            target: target.clone(),
            source,
            generation,
        }
    }
    #[cfg(test)]
    pub(crate) fn best_sparse_tier(
        &self,
        owner: &ResidencyOwner,
        namespace: Option<&str>,
        group: u32,
        hash: &SequenceHash,
    ) -> Option<StorageTier> {
        let hash = sparse_key(namespace, hash);
        self.best_sparse_prepared(owner, group, &hash)
    }

    #[cfg(test)]
    pub(crate) fn best_sparse_prepared(
        &self,
        owner: &ResidencyOwner,
        group: u32,
        hash: &Arc<str>,
    ) -> Option<StorageTier> {
        let target = owner.target();
        self.check_target(&target);
        if self
            .index
            .suspended_sources
            .contains_key(&(target.instance_id, target.dp_rank))
        {
            return None;
        }
        let shard = self.index.shards.get(&group)?;
        TIERS.into_iter().find(|tier| {
            shard
                .reverse
                .get(&(owner.clone(), *tier))
                .is_some_and(|set| set.contains(hash))
        })
    }

    /// Resolve a group in tier order, acquiring each fact-table guard once.
    pub(crate) fn sparse_tiers(
        &self,
        owner: &ResidencyOwner,
        group: u32,
        keys: &[PreparedKey],
    ) -> Vec<Option<StorageTier>> {
        let mut result = vec![None; keys.len()];
        let target = owner.target();
        self.check_target(&target);
        if self
            .index
            .suspended_sources
            .contains_key(&(target.instance_id, target.dp_rank))
        {
            return result;
        }
        let Some(shard) = self.index.shards.get(&group) else {
            return result;
        };
        for tier in TIERS {
            if let Some(hashes) = shard.reverse.get(&(owner.clone(), tier)) {
                for (key, hit) in keys.iter().zip(&mut result) {
                    if hit.is_none() && hashes.contains(key.as_str()) {
                        *hit = Some(tier);
                    }
                }
            }
        }
        result
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

#[cfg(all(test, feature = "cpu", feature = "ready"))]
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

    fn anchor_for(shard: &GroupShard, owner: &ResidencyOwner, seq: &str) -> NodeRef {
        let target = owner.target();
        let hash = h(seq);
        let node = shard.anchor(&target, &hash).expect("source-scoped anchor");
        assert!(unique_alias_position(&node, &target, &hash).is_some());
        node
    }

    #[test]
    fn prepared_sparse_keys_match_canonical_base_and_adapter_keys() {
        let hash: SequenceHash = std::array::from_fn(|i| (i * 7) as u8);
        let base = prepare_sparse_keys(None, &[hash]);
        assert!(matches!(base[0], PreparedKey::Inline(_)));
        assert_eq!(base[0].as_str(), sparse_key(None, &hash).as_ref());
        let expected = format!(
            "b:{}",
            hash.iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<Vec<_>>()
                .join("")
        );
        assert_eq!(base[0].as_str(), expected);
        assert_eq!(base[0].as_str().len(), 66);
        assert!(base[0].as_str().starts_with("b:00070e"));
        let adapter = prepare_sparse_keys(Some("适配:a"), &[hash]);
        assert!(matches!(adapter[0], PreparedKey::Owned(_)));
        assert_eq!(
            adapter[0].as_str(),
            sparse_key(Some("适配:a"), &hash).as_ref()
        );
        assert_eq!(
            adapter[0].as_str(),
            format!("n:{}:适配:a:{}", "适配:a".len(), expected)
        );
        assert_ne!(base[0].as_str(), adapter[0].as_str());
    }

    fn assert_tree_references_have_facts(index: &KvBlockIndexer) {
        for shard in index.shards.iter() {
            let shard = shard.value();
            let mut stack = vec![shard.root.clone()];
            while let Some(node) = stack.pop() {
                let node = node.read();
                for (edge, claims) in node.edge.iter().zip(&node.claims) {
                    for key in claims {
                        let aliases = edge
                            .0
                            .scoped
                            .get(&key.0.target())
                            .expect("claim source alias");
                        let facts = shard.reverse.get(key).expect("claim owner facts");
                        assert!(
                            aliases.iter().any(|hash| facts.contains(hash)),
                            "stale tree reference"
                        );
                    }
                }
                stack.extend(node.children.values().cloned());
            }
        }
    }

    #[test]
    fn derived_tree_references_track_facts_across_mutations() {
        let index = KvBlockIndexer::new();
        let a = worker("a");
        let b = worker("b");
        let sparse_hash = [11; 32];
        let sparse_seq = sparse_key(None, &sparse_hash);
        index.store(
            0,
            a.clone(),
            StorageTier::Device,
            None,
            &[(sparse_seq.clone(), h("x")), (h("a2"), h("y"))],
        );
        index.store(
            0,
            b.clone(),
            StorageTier::Device,
            None,
            &[(h("b1"), h("x")), (h("b2"), h("z"))],
        );
        index.store(
            0,
            b.clone(),
            StorageTier::HostPinned,
            None,
            &[(h("b1"), h("x"))],
        );
        assert_tree_references_have_facts(&index);
        index.remove(0, &a, StorageTier::Device, &[sparse_seq]);
        assert_tree_references_have_facts(&index);
        index.with_batch(|batch| {
            assert!(batch.store_sparse(&a, None, 0, &sparse_hash, StorageTier::Device, 8));
        });
        assert_tree_references_have_facts(&index);
        index.with_batch(|batch| batch.clear_owner_exact(&b));
        assert_tree_references_have_facts(&index);
        index.invalidate_source(&SourceId::from("a"), 0);
        assert_tree_references_have_facts(&index);
    }

    #[test]
    fn sparse_records_are_scoped_by_group_identity_and_incarnation() {
        let index = KvBlockIndexer::new();
        let first = worker("a");
        let second = ResidencyOwner::Worker {
            source: SourceId::from("a"),
            dp_rank: 0,
            incarnation: 1,
        };
        let hash = [7; 32];
        index.with_batch(|batch| {
            assert!(batch.store_sparse(&first, None, 0, &hash, StorageTier::Device, 4));
            assert!(batch.store_sparse(
                &first,
                Some("adapter"),
                0,
                &hash,
                StorageTier::HostPinned,
                4
            ));
            assert!(batch.store_sparse(&first, None, 1, &hash, StorageTier::Disk, 4));
            assert!(batch.store_sparse(&second, None, 0, &hash, StorageTier::Device, 4));
        });
        index.with_snapshot(|view| {
            assert_eq!(
                view.best_sparse_tier(&first, None, 0, &hash),
                Some(StorageTier::Device)
            );
            let prepared = prepare_sparse_keys(None, &[hash, [8; 32]]);
            assert_eq!(
                view.sparse_tiers(&first, 0, &prepared),
                vec![Some(StorageTier::Device), None]
            );
            assert_eq!(
                view.best_sparse_tier(&first, Some("adapter"), 0, &hash),
                Some(StorageTier::HostPinned)
            );
            let adapter_keys = prepare_sparse_keys(Some("adapter"), &[hash]);
            assert_eq!(
                view.sparse_tiers(&first, 0, &adapter_keys),
                vec![Some(StorageTier::HostPinned)]
            );
            assert_eq!(
                view.best_sparse_tier(&first, None, 1, &hash),
                Some(StorageTier::Disk)
            );
        });
        index.with_batch(|batch| batch.clear_owner_exact(&first));
        index.with_snapshot(|view| {
            assert_eq!(view.best_sparse_tier(&first, None, 0, &hash), None);
            assert_eq!(
                view.best_sparse_tier(&second, None, 0, &hash),
                Some(StorageTier::Device)
            );
        });
    }

    #[test]
    fn sparse_and_dense_use_same_residency_facts() {
        let index = KvBlockIndexer::new();
        let owner = worker("a");
        let hash = [9; 32];
        let seq = sparse_key(None, &hash);
        index.store(
            0,
            owner.clone(),
            StorageTier::Device,
            None,
            &[(seq, h("x"))],
        );
        assert_eq!(
            index.with_snapshot(|v| v.best_sparse_tier(&owner, None, 0, &hash)),
            Some(StorageTier::Device)
        );
        index.with_batch(|batch| batch.remove_sparse(&owner, 0, &hash, Some(StorageTier::Device)));
        assert!(index
            .find_matches(&query(&["x"], &[StorageTier::Device]))
            .is_empty());
        // Reinsert through the sparse API: the source-scoped anchor proves
        // the existing dense path, so its lookup reference must be restored.
        index.with_batch(|batch| {
            assert!(batch.store_sparse(&owner, None, 0, &hash, StorageTier::Device, 4));
        });
        assert_eq!(
            depth_of(
                &index.find_matches(&query(&["x"], &[StorageTier::Device])),
                "a",
                StorageTier::Device
            ),
            1
        );
        index.with_batch(|batch| batch.remove_sparse(&owner, 0, &hash, Some(StorageTier::Device)));
        assert!(index
            .find_matches(&query(&["x"], &[StorageTier::Device]))
            .is_empty());
    }

    #[test]
    fn sparse_fact_without_anchor_does_not_create_a_dense_path() {
        let index = KvBlockIndexer::new();
        let owner = worker("a");
        let hash = [12; 32];
        index.with_batch(|batch| {
            assert!(batch.store_sparse(&owner, None, 0, &hash, StorageTier::Device, 4));
        });
        assert_eq!(
            index.with_snapshot(|view| view.best_sparse_tier(&owner, None, 0, &hash)),
            Some(StorageTier::Device)
        );
        assert!(index
            .find_matches(&query(&["x"], &[StorageTier::Device]))
            .is_empty());
    }

    #[test]
    fn conflicting_source_hash_is_rejected_before_any_false_hit() {
        let index = KvBlockIndexer::new();
        let a = worker("a");
        let b = worker("b");
        index.store(
            0,
            a.clone(),
            StorageTier::Device,
            None,
            &[(h("i:7"), h("x"))],
        );
        index.store(0, b, StorageTier::Device, None, &[(h("i:9"), h("x"))]);
        index.store(0, a, StorageTier::Device, None, &[(h("i:7"), h("y"))]);
        let x = index.find_matches(&query(&["x"], &[StorageTier::Device]));
        assert_eq!(depth_of(&x, "a", StorageTier::Device), 0);
        assert_eq!(depth_of(&x, "b", StorageTier::Device), 1);
        assert!(index
            .find_matches(&query(&["y"], &[StorageTier::Device]))
            .is_empty());
    }

    #[test]
    fn foreign_parent_shares_only_topology_for_opaque_or_full_digest() {
        for parent in ["s2".to_owned(), format!("b:{}", "ab".repeat(32))] {
            let index = KvBlockIndexer::new();
            let w1 = worker("w1");
            let w2 = worker("w2");
            let parent = h(&parent);
            index.store(
                0,
                w1,
                StorageTier::Device,
                None,
                &[(h("s1"), h("x")), (parent.clone(), h("y"))],
            );
            index.store(
                0,
                w2.clone(),
                StorageTier::Device,
                None,
                &[(h("w2x"), h("x"))],
            );
            index.store(
                0,
                w2.clone(),
                StorageTier::Device,
                Some(parent.clone()),
                &[(h("w2z"), h("z"))],
            );
            let hits = index.find_matches(&query(&["x", "y", "z"], &[StorageTier::Device]));
            assert_eq!(
                segments_of(&hits, "w2", StorageTier::Device),
                [(0, 1), (2, 3)]
            );
            let shard = index.shard(0);
            assert!(shard.anchor(&w2.target(), &parent).is_some());
            assert!(!shard
                .reverse
                .get(&(w2, StorageTier::Device))
                .is_some_and(|facts| facts.contains(&parent)));
        }
    }

    #[test]
    fn foreign_integer_and_short_or_invalid_byte_parents_are_not_borrowed() {
        for parent in [
            "i:42".to_owned(),
            "b:aabb".to_owned(),
            format!("b:{}gg", "ab".repeat(31)),
            format!("b:{}", "ab".repeat(31)),
        ] {
            let index = KvBlockIndexer::new();
            let w1 = worker("w1");
            let w2 = worker("w2");
            let parent = h(&parent);
            index.store(
                0,
                w1,
                StorageTier::Device,
                None,
                &[(parent.clone(), h("x"))],
            );
            index.store(
                0,
                w2.clone(),
                StorageTier::Device,
                Some(parent.clone()),
                &[(h("w2y"), h("y"))],
            );
            assert!(index.shard(0).anchor(&w2.target(), &parent).is_none());
            assert_eq!(
                segments_of(
                    &index.find_matches(&query(&["x", "y"], &[StorageTier::Device])),
                    "w2",
                    StorageTier::Device
                ),
                Vec::<(u32, u32)>::new()
            );
        }
    }

    #[test]
    fn foreign_parent_rejects_conflicting_nodes_and_offsets() {
        for same_node in [false, true] {
            let index = KvBlockIndexer::new();
            let parent = h("s-parent");
            index.store(
                0,
                worker("w1"),
                StorageTier::Device,
                None,
                &[(parent.clone(), h("x")), (h("w1y"), h("y"))],
            );
            let second = if same_node {
                vec![(h("w2x"), h("x")), (parent.clone(), h("y"))]
            } else {
                vec![(parent.clone(), h("other"))]
            };
            index.store(0, worker("w2"), StorageTier::Device, None, &second);
            let w3 = worker("w3");
            index.store(
                0,
                w3.clone(),
                StorageTier::Device,
                Some(parent.clone()),
                &[(h("w3z"), h("z"))],
            );
            assert!(index.shard(0).anchor(&w3.target(), &parent).is_none());
            assert!(index
                .find_matches(&query(&["x", "y", "z"], &[StorageTier::Device]))
                .iter()
                .all(|hit| hit.target != w3.target()));
        }
    }

    #[test]
    fn rejected_store_does_not_bind_a_borrowed_parent() {
        let index = KvBlockIndexer::new();
        let parent = h("parent");
        let w2 = worker("w2");
        index.store(
            0,
            worker("w1"),
            StorageTier::Device,
            None,
            &[(parent.clone(), h("x"))],
        );
        index.store(
            0,
            w2.clone(),
            StorageTier::Device,
            None,
            &[(h("used"), h("other"))],
        );
        let shard = index.shard(0);
        assert!(!shard.store(
            w2.clone(),
            StorageTier::Device,
            Some(parent.clone()),
            &[(h("used"), h("z"))]
        ));
        assert!(shard.anchor(&w2.target(), &parent).is_none());
        assert_eq!(
            depth_of(
                &index.find_matches(&query(&["other"], &[StorageTier::Device])),
                "w2",
                StorageTier::Device
            ),
            1
        );
    }

    #[test]
    fn conflicting_hash_at_another_offset_or_in_one_store_is_rejected() {
        let index = KvBlockIndexer::new();
        let a = worker("a");
        index.store(
            0,
            a.clone(),
            StorageTier::Device,
            None,
            &[(h("i:1"), h("x")), (h("i:2"), h("y"))],
        );
        index.store(
            0,
            a.clone(),
            StorageTier::Device,
            Some(h("i:1")),
            &[(h("i:1"), h("y"))],
        );
        assert!(index
            .find_matches(&query(&["x", "y"], &[StorageTier::Device]))
            .is_empty());
        let b = worker("b");
        index.store(
            0,
            b,
            StorageTier::Device,
            None,
            &[(h("i:3"), h("x")), (h("i:3"), h("y"))],
        );
        assert!(index
            .find_matches(&query(&["x", "y"], &[StorageTier::Device]))
            .is_empty());
    }

    #[test]
    fn sparse_capacity_counts_pages_not_tiers_or_repeated_stores() {
        let index = KvBlockIndexer::new();
        let owner = worker("a");
        index.with_batch(|batch| {
            assert!(batch.store_sparse(&owner, None, 0, &[1; 32], StorageTier::Device, 1));
            assert!(batch.store_sparse(&owner, None, 0, &[1; 32], StorageTier::HostPinned, 1));
            assert!(!batch.store_sparse(&owner, None, 1, &[2; 32], StorageTier::Device, 1));
            batch.remove_sparse(&owner, 0, &[1; 32], None);
            assert!(batch.store_sparse(&owner, None, 1, &[2; 32], StorageTier::Device, 1));
        });
    }

    #[test]
    fn persistent_sparse_capacity_tracks_tier_removal_across_batches() {
        let index = KvBlockIndexer::new();
        let owner = worker("a");
        index.with_batch(|tx| {
            assert!(tx.store_sparse(&owner, None, 0, &[1; 32], StorageTier::Device, 1))
        });
        index.with_batch(|tx| {
            assert!(tx.store_sparse(&owner, None, 0, &[1; 32], StorageTier::HostPinned, 1))
        });
        index.with_batch(|tx| tx.remove_sparse(&owner, 0, &[1; 32], Some(StorageTier::Device)));
        index.with_batch(|tx| {
            assert!(!tx.store_sparse(&owner, None, 1, &[2; 32], StorageTier::Device, 1))
        });
        index.with_batch(|tx| tx.remove_sparse(&owner, 0, &[1; 32], Some(StorageTier::HostPinned)));
        index.with_batch(|tx| {
            assert!(tx.store_sparse(&owner, None, 1, &[2; 32], StorageTier::Device, 1))
        });
        index.with_batch(|tx| tx.clear_owner_exact(&owner));
        index.with_batch(|tx| {
            assert!(tx.store_sparse(&owner, None, 0, &[1; 32], StorageTier::Device, 1))
        });
    }

    #[test]
    fn dense_edits_invalidate_the_owner_sparse_capacity_cache() {
        let index = KvBlockIndexer::new();
        let owner = worker("a");
        index.with_batch(|tx| {
            assert!(tx.store_sparse(&owner, None, 0, &[1; 32], StorageTier::Device, 2))
        });
        let dense = sparse_key(None, &[2; 32]);
        index.store(
            0,
            owner.clone(),
            StorageTier::Device,
            None,
            &[(dense.clone(), h("x"))],
        );
        index.with_batch(|tx| {
            assert!(!tx.store_sparse(&owner, None, 0, &[3; 32], StorageTier::Device, 2))
        });
        index.remove(0, &owner, StorageTier::Device, &[dense]);
        index.with_batch(|tx| {
            assert!(tx.store_sparse(&owner, None, 0, &[3; 32], StorageTier::Device, 2))
        });
    }

    #[test]
    fn opaque_hash_anchors_are_scoped_to_each_source() {
        let index = KvBlockIndexer::new();
        let a = worker("a");
        let b = worker("b");
        index.store(
            0,
            a.clone(),
            StorageTier::Device,
            None,
            &[(h("i:1"), h("x"))],
        );
        index.store(
            0,
            b.clone(),
            StorageTier::Device,
            None,
            &[(h("i:1"), h("y"))],
        );
        index.store(
            0,
            a.clone(),
            StorageTier::Device,
            Some(h("i:1")),
            &[(h("i:2"), h("z"))],
        );
        index.store(
            0,
            b.clone(),
            StorageTier::Device,
            Some(h("i:1")),
            &[(h("i:3"), h("q"))],
        );
        assert_eq!(
            depth_of(
                &index.find_matches(&query(&["x", "z"], &[StorageTier::Device])),
                "a",
                StorageTier::Device
            ),
            2
        );
        assert_eq!(
            depth_of(
                &index.find_matches(&query(&["y", "q"], &[StorageTier::Device])),
                "b",
                StorageTier::Device
            ),
            2
        );
        index.remove(0, &a, StorageTier::Device, &[h("i:1")]);
        assert_eq!(
            depth_of(
                &index.find_matches(&query(&["y", "q"], &[StorageTier::Device])),
                "b",
                StorageTier::Device
            ),
            2
        );
    }

    #[test]
    fn foreign_alias_cannot_keep_a_removed_source_claim_alive() {
        let index = KvBlockIndexer::new();
        let w1 = worker("w1");
        let w2 = worker("w2");
        index.store(
            0,
            w1.clone(),
            StorageTier::Device,
            None,
            &[(h("i:1"), h("a"))],
        );
        index.store(
            0,
            w2.clone(),
            StorageTier::Device,
            None,
            &[(h("i:7"), h("a"))],
        );
        index.store(
            0,
            w1.clone(),
            StorageTier::Device,
            None,
            &[(h("i:7"), h("x"))],
        );
        index.remove(0, &w1, StorageTier::Device, &[h("i:1")]);
        let a = index.find_matches(&query(&["a"], &[StorageTier::Device]));
        assert_eq!(depth_of(&a, "w1", StorageTier::Device), 0);
        assert_eq!(depth_of(&a, "w2", StorageTier::Device), 1);
        let x = index.find_matches(&query(&["x"], &[StorageTier::Device]));
        assert_eq!(depth_of(&x, "w1", StorageTier::Device), 1);
    }

    #[test]
    fn source_suspension_hides_sparse_records_until_resumed() {
        let index = KvBlockIndexer::new();
        let owner = worker("a");
        let hash = [3; 32];
        index.with_batch(|tx| {
            assert!(tx.store_sparse(&owner, None, 0, &hash, StorageTier::Device, 10));
            tx.suspend_source(&SourceId::from("a"), 0);
        });
        assert_eq!(
            index.with_snapshot(|v| v.best_sparse_tier(&owner, None, 0, &hash)),
            None
        );
        index.with_batch(|tx| tx.resume_source(&SourceId::from("a"), 0));
        assert_eq!(
            index.with_snapshot(|v| v.best_sparse_tier(&owner, None, 0, &hash)),
            Some(StorageTier::Device)
        );
    }

    #[test]
    fn source_invalidation_prunes_only_unreferenced_paths_and_aliases() {
        let index = KvBlockIndexer::new();
        let a = worker("a");
        let b = worker("b");
        index.store(
            0,
            a.clone(),
            StorageTier::Device,
            None,
            &[(h("a1"), h("x")), (h("a2"), h("y"))],
        );
        index.store(0, a, StorageTier::Device, None, &[(h("aq"), h("q"))]);
        index.store(
            0,
            b.clone(),
            StorageTier::Device,
            None,
            &[(h("b1"), h("x")), (h("b2"), h("z"))],
        );
        index.invalidate_source(&SourceId::from("a"), 0);
        let shard = index.shard(0);
        assert!(!shard.root.read().children.contains_key("q"));
        assert!(shard.anchor(&worker("a").target(), &h("a1")).is_none());
        assert!(shard.anchor(&b.target(), &h("b2")).is_some());
        index.store(
            0,
            b,
            StorageTier::Device,
            Some(h("b2")),
            &[(h("b3"), h("w"))],
        );
        assert_eq!(
            depth_of(
                &index.find_matches(&query(&["x", "z", "w"], &[StorageTier::Device])),
                "b",
                StorageTier::Device
            ),
            3
        );
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
    fn four_source_readers_do_not_block_an_unrelated_source_writer() {
        use std::sync::{mpsc, Barrier};
        use std::time::Duration;

        let index = Arc::new(KvBlockIndexer::new());
        let entered = Arc::new(Barrier::new(5));
        let release = Arc::new(Barrier::new(5));
        let mut readers = Vec::new();
        for id in 0..4 {
            let index = index.clone();
            let entered = entered.clone();
            let release = release.clone();
            readers.push(std::thread::spawn(move || {
                let target = worker(&format!("reader-{id}")).target();
                index.with_source_snapshot(&target, |_| {
                    entered.wait();
                    release.wait();
                });
            }));
        }
        entered.wait();
        let writer_index = index.clone();
        let (done, received) = mpsc::channel();
        let writer = std::thread::spawn(move || {
            let owner = worker("writer");
            writer_index.with_source_batch(&owner.target(), |batch| {
                assert!(batch.store_sparse(&owner, None, 0, &[1; 32], StorageTier::Device, 8));
            });
            done.send(()).unwrap();
        });
        let completed = received.recv_timeout(Duration::from_secs(5)).is_ok();
        release.wait();
        for reader in readers {
            reader.join().unwrap();
        }
        writer.join().unwrap();
        assert!(
            completed,
            "unrelated writer waited for other source snapshots"
        );
    }

    #[test]
    fn same_source_snapshot_waits_for_complete_multi_group_batch() {
        use std::sync::mpsc;
        use std::time::Duration;

        let index = Arc::new(KvBlockIndexer::new());
        let owner = worker("a");
        let target = owner.target();
        let (entered, ready) = mpsc::channel();
        let (release, resume) = mpsc::channel();
        let writer_index = index.clone();
        let writer_owner = owner.clone();
        let writer_target = target.clone();
        let writer = std::thread::spawn(move || {
            writer_index.with_source_batch(&writer_target, |batch| {
                assert!(batch.store_sparse(
                    &writer_owner,
                    None,
                    0,
                    &[1; 32],
                    StorageTier::Device,
                    8
                ));
                entered.send(()).unwrap();
                resume.recv_timeout(Duration::from_secs(5)).unwrap();
                assert!(batch.store_sparse(
                    &writer_owner,
                    None,
                    1,
                    &[1; 32],
                    StorageTier::Device,
                    8
                ));
            });
        });
        ready.recv_timeout(Duration::from_secs(5)).unwrap();
        {
            let _domain = index.domain.read();
            let gate = index.source_lock(&target);
            assert!(gate.access.try_read().is_none());
        }
        release.send(()).unwrap();
        writer.join().unwrap();
        let keys = prepare_sparse_keys(None, &[[1; 32]]);
        index.with_source_snapshot(&target, |view| {
            assert_eq!(
                view.sparse_tiers(&owner, 0, &keys),
                [Some(StorageTier::Device)]
            );
            assert_eq!(
                view.sparse_tiers(&owner, 1, &keys),
                [Some(StorageTier::Device)]
            );
        });
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
                resuming.resume_source_unlocked(&SourceId::from("a"), 0);
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
                // Simulate a visibility transition at this hook without
                // recursively taking the query's topology read lock.
                invalidating.suspend_source_unlocked(&source, 0);
                invalidating.resume_source_unlocked(&source, 0);
            }));
        });
        assert!(idx
            .find_matches(&query(&["x"], &[StorageTier::Device]))
            .is_empty());
    }

    #[test]
    fn source_transition_does_not_discard_other_targets() {
        let index = Arc::new(KvBlockIndexer::new());
        index.store(
            0,
            worker("a"),
            StorageTier::Device,
            None,
            &[(h("a-hash"), h("x"))],
        );
        index.store(
            0,
            worker("b"),
            StorageTier::Device,
            None,
            &[(h("b-hash"), h("x"))],
        );
        let changing = index.clone();
        AFTER_QUERY_EDGE.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                changing.suspend_source_unlocked(&SourceId::from("b"), 0);
            }));
        });
        let matches = index.find_matches(&query(&["x"], &[StorageTier::Device]));
        assert_eq!(depth_of(&matches, "a", StorageTier::Device), 1);
        assert_eq!(depth_of(&matches, "b", StorageTier::Device), 0);
    }

    #[test]
    fn query_final_check_drops_source_map_guard_before_suspended_lookup() {
        use std::sync::mpsc;
        use std::time::Duration;

        let index = Arc::new(KvBlockIndexer::new());
        index.store(
            0,
            worker("a"),
            StorageTier::Device,
            None,
            &[(h("hash"), h("x"))],
        );
        let (unlocked, unlocked_rx) = mpsc::channel();
        let (held, held_rx) = mpsc::channel();
        let (looked, looked_rx) = mpsc::channel();
        let (finished, finished_rx) = mpsc::channel();
        let query_index = index.clone();
        let query_thread = std::thread::spawn(move || {
            AFTER_QUERY_UNLOCK.with(|slot| {
                *slot.borrow_mut() = Some(Box::new(move || {
                    unlocked.send(()).unwrap();
                    held_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                }));
            });
            AFTER_QUERY_SOURCE_LOOKUP.with(|slot| {
                *slot.borrow_mut() = Some(Box::new(move || {
                    looked.send(()).unwrap();
                }));
            });
            let matches = query_index.find_matches(&query(&["x"], &[StorageTier::Device]));
            finished.send(matches.len()).unwrap();
        });
        let contender_index = index.clone();
        let contender = std::thread::spawn(move || {
            unlocked_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let _domain = contender_index.domain.read();
            let target = worker("a").target();
            let entry = contender_index
                .suspended_sources
                .entry((target.instance_id.clone(), target.dp_rank));
            held.send(()).unwrap();
            looked_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            // This deliberately takes the opposite map order. It can finish
            // only if the query already released its source-map guard.
            let _gate = contender_index.source_lock(&target);
            drop(entry);
        });
        let result = finished_rx.recv_timeout(Duration::from_secs(10));
        if result.is_ok() {
            query_thread.join().unwrap();
            contender.join().unwrap();
        }
        assert_eq!(result.expect("source/suspended map lock inversion"), 1);
    }

    #[test]
    fn source_invalidated_after_snapshot_is_filtered_without_losing_other_targets() {
        let index = Arc::new(KvBlockIndexer::new());
        index.store(
            0,
            worker("a"),
            StorageTier::Device,
            None,
            &[(h("a-hash"), h("x"))],
        );
        index.store(
            0,
            worker("b"),
            StorageTier::Device,
            None,
            &[(h("b-hash"), h("x"))],
        );
        let changing = index.clone();
        AFTER_QUERY_UNLOCK.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                let source = SourceId::from("a");
                changing.invalidate_source(&source, 0);
                changing.resume_source(&source, 0);
            }));
        });
        let matches = index.find_matches(&query(&["x"], &[StorageTier::Device]));
        assert_eq!(depth_of(&matches, "a", StorageTier::Device), 0);
        assert_eq!(depth_of(&matches, "b", StorageTier::Device), 1);
    }

    #[test]
    fn retired_and_reregistered_source_does_not_pass_old_snapshot_generation() {
        let index = Arc::new(KvBlockIndexer::new());
        index.store(
            0,
            worker("a"),
            StorageTier::Device,
            None,
            &[(h("old"), h("x"))],
        );
        index.store(
            0,
            worker("b"),
            StorageTier::Device,
            None,
            &[(h("b-hash"), h("x"))],
        );
        let changing = index.clone();
        AFTER_QUERY_UNLOCK.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                changing.retire_source(&SourceId::from("a"), 0);
                changing.store(
                    0,
                    worker("a"),
                    StorageTier::Device,
                    None,
                    &[(h("new"), h("x"))],
                );
            }));
        });
        let old_snapshot = index.find_matches(&query(&["x"], &[StorageTier::Device]));
        assert_eq!(depth_of(&old_snapshot, "a", StorageTier::Device), 0);
        assert_eq!(depth_of(&old_snapshot, "b", StorageTier::Device), 1);
        let fresh = index.find_matches(&query(&["x"], &[StorageTier::Device]));
        assert_eq!(depth_of(&fresh, "a", StorageTier::Device), 1);
        assert_eq!(depth_of(&fresh, "b", StorageTier::Device), 1);
    }

    #[test]
    fn retiring_many_sources_reclaims_gates_without_erasing_other_owners() {
        let index = KvBlockIndexer::new();
        index.store(
            0,
            worker("keep"),
            StorageTier::Device,
            None,
            &[(h("keep-hash"), h("x"))],
        );
        for id in 0..1000 {
            let name = format!("retired-{id}");
            let owner = worker(&name);
            index.store(
                0,
                owner,
                StorageTier::Device,
                None,
                &[(h("source-hash"), h("x"))],
            );
            index.retire_source(&SourceId::from(name), 0);
        }
        assert_eq!(index.source_locks.len(), 1);
        assert!(index.suspended_sources.is_empty());
        let matches = index.find_matches(&query(&["x"], &[StorageTier::Device]));
        assert_eq!(depth_of(&matches, "keep", StorageTier::Device), 1);
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
        assert_eq!(actual.len(), expected.len());
        for seq in &actual {
            let node = anchor_for(shard, &key.0, seq);
            let pos = unique_alias_position(&node, &key.0.target(), seq)
                .expect("source-scoped anchor contains alias");
            let node = node.read();
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
        let node = anchor_for(&shard, &owner, "i:900");
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
        assert_eq!(shard.reverse.get(&key).map_or(0, |seqs| seqs.len()), 0);
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
        for seq in ["i:200", "i:201", "i:202", "b:0200", "i:203"] {
            let node = anchor_for(&shard, &b, seq);
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
        let old_anchor = anchor_for(&shard, &b, "i:202");
        let split_index = idx.clone();
        let retried = Rc::new(Cell::new(false));
        let retry_observed = retried.clone();
        AFTER_ANCHOR_CLONE.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                // Legal interleaving: another owner splits after remove has
                // cloned its anchor, but before it takes that node's lock.
                split_index.shard(0).store(
                    worker("a"),
                    StorageTier::Device,
                    None,
                    &[(h("i:100"), h("x"))],
                );
                let moved = anchor_for(&split_index.shard(0), &worker("b"), "i:202");
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
        let moved = anchor_for(&shard, &kb.0, "i:202");
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
        let old_anchor = anchor_for(&shard, &b, "i:201");
        let split_index = idx.clone();
        let retried = Rc::new(Cell::new(false));
        let retry_observed = retried.clone();
        AFTER_ANCHOR_CLONE.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                split_index.shard(0).store(
                    worker("a"),
                    StorageTier::Device,
                    None,
                    &[(h("i:100"), h("x"))],
                );
                let moved = anchor_for(&split_index.shard(0), &worker("b"), "i:201");
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
        let branch = anchor_for(&shard, &kb.0, "i:203");
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
            .and_then(|s| s.reverse.get(&key).map(|seqs| seqs.len()))
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
        // so find_matches can't tell them apart — verify their records instead.
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
            .reverse
            .get(&(pool("poolA", "w0"), StorageTier::External))
            .is_none());
        assert_eq!(
            shard
                .reverse
                .get(&(pool("poolB", "w0"), StorageTier::External))
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            shard
                .reverse
                .get(&(worker("w0"), StorageTier::Device))
                .unwrap()
                .len(),
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
            anchor_for(&shard, &key.0, "incoming-x");
            anchor_for(&shard, &key.0, "incoming-y");
            assert!(!shard.root.read().children.contains_key("w"));
            assert_eq!(
                shard
                    .reverse
                    .get(&(worker("w0"), StorageTier::Device))
                    .unwrap()
                    .len(),
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
            let y = anchor_for(&shard, &key.0, "sy");
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
                        let old_anchor = anchor_for(&split_shard, &key.0, "incoming-y");
                        split_shard.store(
                            worker("splitter"),
                            StorageTier::Device,
                            None,
                            &[(h("split-x"), h("x"))],
                        );
                        let moved = anchor_for(&split_shard, &key.0, "incoming-y");
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
        let old_anchor = anchor_for(&shard, &worker("w0"), "sy");
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
        let moved_anchor = anchor_for(&shard, &worker("w0"), "sy");
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
                splitting.shard(0).store(
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
