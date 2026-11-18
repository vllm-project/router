//! Keep navigation only for current facts and the ancestors they reference.

use super::super::*;
use super::AliasRemover;

impl GroupShard {
    pub(in crate::kv_index::index) fn register_paths(
        &self,
        target: &Arc<RoutableTarget>,
        parent: Option<&Arc<str>>,
        blocks: &[IncomingBlock],
    ) -> bool {
        let mut seen = HashSet::with_hasher(Fast::default());
        for (index, (seq, local)) in blocks.iter().enumerate() {
            if !seen.insert(seq.clone()) {
                return false;
            }
            if let Some(paths) = self.scoped_anchors.get(seq) {
                if let Some(path) = paths.get(target) {
                    // An established identity cannot acquire an unproven
                    // different parent. The normal rooted path validates it.
                    if path.position.is_some() {
                        return false;
                    }
                    let preceding = if index == 0 {
                        parent
                    } else {
                        Some(&blocks[index - 1].0)
                    };
                    if path
                        .parent
                        .as_ref()
                        .is_some_and(|known| Some(known) != preceding)
                    {
                        return false;
                    }
                    if let Some(parent) = &path.parent {
                        if self
                            .waiting
                            .get(&(target.as_ref().clone(), parent.clone()))
                            .is_some_and(|children| {
                                children.get(seq).is_some_and(|known| known != local)
                            })
                        {
                            return false;
                        }
                    }
                }
            }
        }
        // An unrooted chain must not close a cycle through an older pending chain.
        let mut cursor = parent.cloned();
        while let Some(seq) = cursor {
            if !seen.insert(seq.clone()) {
                return false;
            }
            cursor = self.scoped_anchors.get(&seq).and_then(|paths| {
                paths.get(target).and_then(|path| {
                    if path.position.is_some() {
                        None
                    } else {
                        path.parent.clone()
                    }
                })
            });
        }
        let mut parent = parent.cloned();
        for (seq, _) in blocks {
            let mut paths = self.scoped_anchors.entry(seq.clone()).or_default();
            let needs_link = paths
                .get(target)
                .is_none_or(|path| path.parent.is_none() && path.position.is_none());
            if needs_link {
                paths.get_or_insert(target).parent = parent.clone();
            }
            drop(paths);
            if needs_link {
                if let Some(parent_hash) = &parent {
                    let mut paths = self.scoped_anchors.entry(parent_hash.clone()).or_default();
                    paths.get_or_insert(target).children += 1;
                }
            }
            parent = Some(seq.clone());
        }
        true
    }

    pub(in crate::kv_index::index) fn set_position(
        &self,
        target: &Arc<RoutableTarget>,
        seq: &Arc<str>,
        node: &NodeRef,
        offset: usize,
        (parent, children): (Option<&Arc<str>>, usize),
    ) -> bool {
        let mut paths = self.scoped_anchors.entry(seq.clone()).or_default();
        let path = paths.get_or_insert(target);
        let link = path.parent.is_none() && parent.is_some();
        if link {
            path.parent = parent.cloned();
        }
        path.children += children;
        path.position = Some(Anchor {
            node: node.clone(),
            offset,
        });
        link
    }

    pub(in crate::kv_index::index) fn connect_waiting(
        &self,
        target: &Arc<RoutableTarget>,
        mut ready: Vec<Arc<str>>,
    ) {
        while let Some(parent) = ready.pop() {
            if self.positioned_anchor(target, &parent).is_none() {
                continue;
            }
            let children = self
                .waiting
                .remove(&(target.as_ref().clone(), parent.clone()))
                .map(|(_, children)| children)
                .unwrap_or_default();
            for (seq, local) in children {
                if self.positioned_anchor(target, &seq).is_none()
                    && !self.store_topology(
                        target,
                        None,
                        Some(parent.clone()),
                        &[(seq.clone(), local)],
                    )
                {
                    continue;
                }
                let fact = FactKey::from_seq(&seq);
                let claims: Vec<_> = self
                    .reverse
                    .iter()
                    .filter(|entry| entry.key().target() == **target)
                    .filter_map(|entry| {
                        entry
                            .get(&fact)
                            .copied()
                            .map(|bits| (entry.key().clone(), bits))
                    })
                    .collect();
                for (owner, bits) in claims {
                    for tier in TIERS.into_iter().filter(|tier| bits.contains(*tier)) {
                        self.attach_existing(&owner, tier, &seq);
                    }
                }
                ready.push(seq);
            }
        }
    }

    /// The caller holds this source's transaction and the topology write lock.
    pub(in crate::kv_index::index) fn release_paths(
        &self,
        target: &RoutableTarget,
        owners: &[ResidencyOwner],
        hashes: &[Arc<str>],
    ) {
        let mut work = hashes.to_vec();
        let mut aliases = AliasRemover::new(target);
        while let Some(seq) = work.pop() {
            let fact = FactKey::from_seq(&seq);
            if owners.iter().any(|owner| {
                self.reverse
                    .get(owner)
                    .is_some_and(|facts| facts.contains_key(&fact))
            }) {
                continue;
            }
            let Some(mut paths) = self.scoped_anchors.get_mut(&seq) else {
                continue;
            };
            let Some(path) = paths.get_mut(target) else {
                continue;
            };
            if path.children != 0 {
                continue;
            }
            let path = std::mem::take(path);
            paths.remove(target);
            drop(paths);
            self.scoped_anchors
                .remove_if(&seq, |_, paths| paths.is_empty());
            if let Some(anchor) = path.position {
                let mut node = anchor.node.write();
                if node
                    .edge
                    .get(anchor.offset)
                    .is_some_and(|e| e.0.has(target, &seq))
                {
                    aliases.remove(&mut node.edge[anchor.offset].0, Some(&seq));
                    node.alias_count -= 1;
                }
                if !node.prune_queued {
                    node.prune_queued = true;
                    self.prunable.lock().push(anchor.node.clone());
                }
            }
            if let Some(parent) = path.parent {
                let key = (target.clone(), parent.clone());
                if let Some(mut waiting) = self.waiting.get_mut(&key) {
                    waiting.remove(&seq);
                }
                self.waiting
                    .remove_if(&key, |_, children| children.is_empty());
                if let Some(mut paths) = self.scoped_anchors.get_mut(&parent) {
                    if let Some(parent_path) = paths.get_mut(target) {
                        // Only a recorded child link can make an ancestor eligible.
                        if let Some(children) = parent_path.children.checked_sub(1) {
                            parent_path.children = children;
                            work.push(parent.clone());
                        } else {
                            debug_assert!(false, "retired child has no matching parent link");
                        }
                    }
                }
            }
        }
    }

    /// Only a topology writer may unlink nodes. Readers with old Arc snapshots finish normally.
    pub(in crate::kv_index::index) fn prune_dead_nodes(&self) {
        let mut work = std::mem::take(&mut *self.prunable.lock());
        while let Some(node) = work.pop() {
            let mut current = node.write();
            current.prune_queued = false;
            if !current.children.is_empty() {
                continue;
            }
            if current.alias_count != 0 {
                let len = current
                    .edge
                    .iter()
                    .rposition(|edge| !edge.0.is_empty())
                    .map_or(0, |i| i + 1);
                current.edge.truncate(len);
                current.claims.truncate(len);
                // Geometric shrinking bounds retained allocation without
                // reallocating for every individual page eviction.
                if len <= current.edge.capacity() / 2 {
                    current.edge.shrink_to_fit();
                }
                if len <= current.claims.capacity() / 2 {
                    current.claims.shrink_to_fit();
                }
                continue;
            }
            debug_assert!(
                current.claims.iter().all(|claims| claims.is_empty()),
                "detached prefix nodes cannot retain cache claims"
            );
            let Some(parent) = current.parent.upgrade() else {
                continue;
            };
            let Some(first) = current.edge.first().map(|e| e.1.clone()) else {
                continue;
            };
            drop(current);
            let mut p = parent.write();
            if p.children
                .get(&first)
                .is_some_and(|child| Arc::ptr_eq(child, &node))
            {
                p.children.remove(&first);
            }
            let empty = p.children.is_empty();
            drop(p);
            if empty {
                work.push(parent);
            }
        }
    }
}
