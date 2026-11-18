//! Insert prefixes, attach claims, and split compressed edges.

use super::super::*;
use super::AliasWriter;

type ClaimList = Arc<Vec<OwnerTier>>;

/// Adjacent pages often have the same immutable claim list. Transform that
/// list once per event; individual removals still use copy-on-write.
struct ClaimWriter<'a> {
    key: &'a OwnerTier,
    previous: Option<(ClaimList, ClaimList)>,
}

impl<'a> ClaimWriter<'a> {
    fn new(key: &'a OwnerTier) -> Self {
        Self {
            key,
            previous: None,
        }
    }

    fn add(&mut self, claims: &mut ClaimList) {
        if let Some((before, after)) = &self.previous {
            if Arc::ptr_eq(before, claims) {
                *claims = after.clone();
                return;
            }
        }
        if !claims.contains(self.key) {
            let shared = (Arc::strong_count(claims) > 1).then(|| claims.clone());
            Arc::make_mut(claims).push(self.key.clone());
            self.previous = shared.map(|before| (before, claims.clone()));
        }
    }
}

impl GroupShard {
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
    pub(in crate::kv_index::index) fn store_for_source(
        &self,
        target: &Arc<RoutableTarget>,
        claim: &StoreClaim,
        parent_seq: Option<Arc<str>>,
        blocks: &[IncomingBlock],
    ) -> bool {
        let accepted = self.store_topology(target, Some(claim), parent_seq.clone(), blocks);
        if accepted && !self.waiting.is_empty() {
            let ready = parent_seq
                .into_iter()
                .chain(blocks.iter().map(|b| b.0.clone()))
                .collect();
            self.connect_waiting(target, ready);
        }
        accepted
    }

    pub(in crate::kv_index::index) fn store_topology(
        &self,
        target: &Arc<RoutableTarget>,
        key: Option<&StoreClaim>,
        parent_seq: Option<Arc<str>>,
        blocks: &[IncomingBlock],
    ) -> bool {
        if blocks.is_empty() {
            return true;
        }
        // A parent may borrow only a uniquely established foreign content
        // position; this is topology evidence, never a residency claim.
        let parent = if let Some(seq) = &parent_seq {
            match self.resolve_parent(target, seq) {
                Ok(Some(parent)) => Some(parent),
                Ok(None) => {
                    if let Some(key) = key {
                        if !self.register_paths(target, parent_seq.as_ref(), blocks) {
                            return false;
                        }
                        let mut facts = self.reverse.entry(key.key.0.clone()).or_default();
                        let mut parent = seq.clone();
                        for (seq, local) in blocks {
                            if !facts
                                .entry(FactKey::from_seq(seq))
                                .or_default()
                                .observe(key.key.1, key.incremental)
                            {
                                return false;
                            }
                            self.waiting
                                .entry((target.as_ref().clone(), parent))
                                .or_default()
                                .insert(seq.clone(), local.clone());
                            parent = seq.clone();
                        }
                    }
                    return true;
                }
                Err(()) => return false,
            }
        } else {
            None
        };
        if !self.store_path_is_consistent(target, parent.as_ref(), parent_seq.as_ref(), blocks) {
            return false;
        }
        if let (Some(seq), Some((node, offset, true))) = (&parent_seq, &parent) {
            let mut node_guard = node.write();
            if !node_guard.edge[*offset].0.has(target, seq) {
                node_guard.edge[*offset]
                    .0
                    .insert(target.clone(), seq.clone());
                node_guard.alias_count += 1;
            }
            self.set_position(target, seq, node, *offset, (None, 0));
        }
        let mut cursor = parent_seq;
        let mut bi = 0;
        while bi < blocks.len() {
            let (node, offset) = match &cursor {
                Some(seq) => {
                    let anchor = self
                        .positioned_anchor(target, seq)
                        .expect("known cursor remains anchored during a topology write");
                    (anchor.node, Some(anchor.offset))
                }
                None => (self.root.clone(), None),
            };

            let mut parent = node.write();
            if let Some(seq) = &cursor {
                let pos = offset.expect("a cursor has an anchor offset");
                assert!(
                    parent
                        .edge
                        .get(pos)
                        .is_some_and(|(aliases, _)| aliases.has(target, seq)),
                    "prefix cursor moved under the topology write guard"
                );
                if pos + 1 < parent.edge.len() {
                    let start = pos + 1;
                    let matched = edge_common_prefix(&parent.edge[start..], &blocks[bi..]);
                    if matched > 0 {
                        if !self.record_locked_at(
                            &mut parent,
                            &node,
                            target,
                            key,
                            (cursor.as_ref(), &blocks[bi..bi + matched]),
                            start,
                        ) {
                            return false;
                        }
                        bi += matched;
                        cursor = Some(blocks[bi - 1].0.clone());
                        if bi == blocks.len() {
                            return true;
                        }
                        if start + matched < parent.edge.len() {
                            self.split_locked(&mut parent, &node, start + matched);
                        }
                        drop(parent);
                        continue;
                    }
                    self.split_locked(&mut parent, &node, start);
                }
            }
            // A childless leaf can absorb an appended suffix without adding
            // another node. Claims and aliases remain per block.
            if cursor.is_some() && parent.children.is_empty() {
                let offset = parent.edge.len();
                let extension = TreeNode::with_edge(&blocks[bi..], target, key.map(|c| &c.key));
                parent.alias_count += extension.alias_count;
                parent.edge.extend(extension.edge);
                parent.claims.extend(extension.claims);
                return self.record_locked_at(
                    &mut parent,
                    &node,
                    target,
                    key,
                    (cursor.as_ref(), &blocks[bi..]),
                    offset,
                );
            }
            let child = parent.children.get(&blocks[bi].1).cloned();
            let Some(child) = child else {
                // Keep the parent locked from lookup through publication so
                // another writer cannot replace a newly installed child.
                let new_node = Arc::new(RwLock::new(TreeNode::with_edge(
                    &blocks[bi..],
                    target,
                    key.map(|c| &c.key),
                )));
                let mut new = new_node.write();
                new.parent = Arc::downgrade(&node);
                if !self.record_locked_at(
                    &mut new,
                    &new_node,
                    target,
                    key,
                    (cursor.as_ref(), &blocks[bi..]),
                    0,
                ) {
                    return false;
                }
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
            if matched < c.edge.len() && bi + matched < blocks.len() {
                self.split_locked(&mut c, &child, matched);
            }
            if !self.record_locked_at(
                &mut c,
                &child,
                target,
                key,
                (cursor.as_ref(), &blocks[bi..bi + matched]),
                0,
            ) {
                return false;
            }
            bi += matched;
            // record_locked registered this incoming key, so its current
            // anchor remains a valid cursor even if a later split moves it.
            cursor = Some(blocks[bi - 1].0.clone());
        }
        true
    }

    /// Restore established adjacent pages; false leaves the index unchanged.
    /// The caller holds the topology read lock, so an edge cannot split.
    pub(in crate::kv_index::index) fn try_store_anchored(
        &self,
        claim: &StoreClaim,
        parent: Option<&Arc<str>>,
        blocks: &[IncomingBlock],
    ) -> bool {
        if blocks.is_empty() {
            return false;
        }
        let Some(parent) = parent else {
            return false;
        };
        let target = claim.key.0.target();
        let Some(anchor) = self.positioned_anchor(&target, parent) else {
            return false;
        };
        let mut node = anchor.node.write();
        if !node
            .edge
            .get(anchor.offset)
            .is_some_and(|(aliases, _)| aliases.has(&target, parent))
        {
            return false;
        }
        let offset = anchor.offset + 1;
        if blocks.len() > node.edge.len().saturating_sub(offset)
            || blocks.iter().enumerate().any(|(i, (seq, local))| {
                let (aliases, stored) = &node.edge[offset + i];
                seq == parent || stored != local || !aliases.has(&target, seq)
            })
        {
            return false;
        }
        let mut claims = ClaimWriter::new(&claim.key);
        let mut facts = self.reverse.entry(claim.key.0.clone()).or_default();
        if claim.incremental
            && blocks.iter().any(|(seq, _)| {
                facts
                    .get(&FactKey::from_seq(seq))
                    .is_some_and(|r| !r.can_observe(claim.incremental))
            })
        {
            return false;
        }
        for (i, (seq, _)) in blocks.iter().enumerate() {
            if !facts
                .entry(FactKey::from_seq(seq))
                .or_default()
                .observe(claim.key.1, claim.incremental)
            {
                return false;
            }
            claims.add(&mut node.claims[offset + i]);
        }
        true
    }

    /// Verify every previously seen source hash resolves to the exact same
    /// logical path and edge offset before mutating any part of this store.
    /// Callers hold a source transaction and the topology write guard throughout;
    /// independent source hashes may legitimately have the same bytes.
    pub(in crate::kv_index::index) fn store_path_is_consistent(
        &self,
        target: &RoutableTarget,
        parent: Option<&(NodeRef, usize, bool)>,
        parent_seq: Option<&Arc<str>>,
        blocks: &[IncomingBlock],
    ) -> bool {
        let mut incoming = HashSet::with_capacity_and_hasher(blocks.len() + 1, Fast::default());
        if let Some(parent) = parent_seq {
            incoming.insert(parent.clone());
        }
        let mut known = false;
        for (seq, _) in blocks {
            if !incoming.insert(seq.clone()) {
                return false;
            }
            if !known {
                known = self
                    .scoped_anchors
                    .get(seq)
                    .is_some_and(|paths| paths.get(target).is_some());
            }
        }
        // New identities are placed by the insertion walk; only an already
        // observed source identity can conflict with its previous position.
        if !known {
            return true;
        }
        let mut path = Some(match parent {
            None => (self.root.clone(), 0),
            Some((node, position, _)) => (node.clone(), position + 1),
        });
        for (idx, (seq, local)) in blocks.iter().enumerate() {
            let expected = path.take().and_then(|(node, offset)| {
                let current = node.read();
                if offset < current.edge.len() {
                    (current.edge[offset].1 == *local).then(|| (node.clone(), offset))
                } else {
                    current.children.get(local).cloned().map(|child| (child, 0))
                }
            });
            let paths = self.scoped_anchors.get(seq);
            let known = paths.as_ref().and_then(|paths| paths.get(target));
            if let Some(path) = known.filter(|path| path.position.is_none()) {
                if let Some(parent) = &path.parent {
                    let preceding = if idx == 0 {
                        parent_seq
                    } else {
                        Some(&blocks[idx - 1].0)
                    };
                    if Some(parent) != preceding
                        || self
                            .waiting
                            .get(&(target.clone(), parent.clone()))
                            .is_some_and(|children| {
                                children.get(seq).is_some_and(|known| known != local)
                            })
                    {
                        return false;
                    }
                }
            }
            let anchor =
                known.and_then(|path| path.position.as_ref().map(|anchor| anchor.node.clone()));
            drop(paths);
            if let Some(anchor) = anchor {
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
    pub(in crate::kv_index::index) fn resolve_parent(
        &self,
        target: &RoutableTarget,
        seq: &Arc<str>,
    ) -> Result<Option<(NodeRef, usize, bool)>, ()> {
        if let Some(anchor) = self.positioned_anchor(target, seq) {
            let valid = anchor
                .node
                .read()
                .edge
                .get(anchor.offset)
                .is_some_and(|(aliases, _)| aliases.has(target, seq));
            if !valid {
                return Err(());
            }
            return Ok(Some((anchor.node, anchor.offset, false)));
        }
        let wire_hash = unqualified_seq(seq);
        if wire_hash.starts_with("i:") {
            return Ok(None);
        }
        if let Some(digest) = wire_hash.strip_prefix("b:") {
            if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Ok(None);
            }
        }
        let Some(anchors) = self.scoped_anchors.get(seq) else {
            return Ok(None);
        };
        let candidates: Vec<_> = anchors
            .iter()
            .filter_map(|(scope, path)| {
                path.position
                    .as_ref()
                    .map(|anchor| (scope.clone(), anchor.node.clone()))
            })
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
    pub(in crate::kv_index::index) fn attach_existing(
        &self,
        owner: &ResidencyOwner,
        tier: StorageTier,
        seq: &Arc<str>,
    ) {
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
            Arc::make_mut(&mut node.claims[position]).push(claim);
        }
    }

    /// Install navigation positions; the source transaction owns the facts.
    fn record_locked_at(
        &self,
        n: &mut TreeNode,
        node: &NodeRef,
        target: &Arc<RoutableTarget>,
        key: Option<&StoreClaim>,
        (parent, incoming): (Option<&Arc<str>>, &[IncomingBlock]),
        offset: usize,
    ) -> bool {
        let mut facts = key.map(|key| self.reverse.entry(key.key.0.clone()).or_default());
        if let Some(facts) = facts.as_mut().filter(|facts| facts.is_empty()) {
            facts.reserve(incoming.len());
        }
        let mut claims = key.map(|key| ClaimWriter::new(&key.key));
        let mut aliases = AliasWriter::new(target);
        // Register from the leaf back: each new parent link increments the
        // preceding record in the same visit, rather than a second map write.
        let mut children = 0;
        for (position, (seq, _)) in incoming.iter().enumerate().rev() {
            let parent = if position == 0 {
                parent
            } else {
                Some(&incoming[position - 1].0)
            };
            let i = position + offset;
            if aliases.add(&mut n.edge[i].0, seq) {
                n.alias_count += 1;
            }
            children = usize::from(self.set_position(target, seq, node, i, (parent, children)));
            if let Some(key) = key {
                if !facts
                    .as_mut()
                    .unwrap()
                    .entry(FactKey::from_seq(seq))
                    .or_default()
                    .observe(key.key.1, key.incremental)
                {
                    return false;
                }
                claims.as_mut().unwrap().add(&mut n.claims[i]);
            }
        }
        if children != 0 {
            let mut paths = self
                .scoped_anchors
                .entry(parent.expect("new link has a parent").clone())
                .or_default();
            paths.get_or_insert(target).children += children;
        }
        true
    }

    /// Split `child`'s edge at `pos`: the first `pos` blocks stay on `child`
    /// with their claims, the remainder becomes a single new child carrying
    /// the remainder's claims and the old children. The remainder's seq hashes
    /// are re-anchored to the new node so `remove`/`clear` still find them.
    /// Called under the caller's write lock on `child`.
    pub(in crate::kv_index::index) fn split_locked(
        &self,
        c: &mut TreeNode,
        original: &NodeRef,
        pos: usize,
    ) {
        let remainder: Vec<EdgeBlock> = c.edge.split_off(pos);
        let rem_claims = c.claims.split_off(pos);
        let old_children = std::mem::take(&mut c.children);
        let rem_first = remainder[0].1.clone();
        let remainder_alias_count = remainder.iter().map(|e| e.0.count()).sum();
        c.alias_count -= remainder_alias_count;
        let rem_node = Arc::new(RwLock::new(TreeNode {
            edge: remainder,
            claims: rem_claims,
            children: old_children,
            parent: Arc::downgrade(original),
            alias_count: remainder_alias_count,
            prune_queued: false,
        }));
        // An anchor exposes the remainder to other writers. Keep it locked
        // through all re-anchoring and publication so a split cannot race us.
        let r = rem_node.write();
        for child in r.children.values() {
            child.write().parent = Arc::downgrade(&rem_node);
        }
        for (offset, (aliases, _)) in r.edge.iter().enumerate() {
            aliases.for_each(|target, seq| {
                if let Some(mut anchors) = self.scoped_anchors.get_mut(seq) {
                    if let Some(anchor) = anchors.get_mut(target).and_then(|p| p.position.as_mut())
                    {
                        if Arc::ptr_eq(&anchor.node, original) {
                            *anchor = Anchor {
                                node: rem_node.clone(),
                                offset,
                            };
                        }
                    }
                }
            });
        }
        c.children.insert(rem_first, rem_node.clone());
        if r.alias_count == 0 {
            drop(r);
            rem_node.write().prune_queued = true;
            self.prunable.lock().push(rem_node);
        }
    }
}
