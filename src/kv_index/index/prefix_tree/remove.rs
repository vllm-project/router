//! Remove claims and prune source-scoped prefix paths.

use super::super::*;
use super::AliasRemover;

impl GroupShard {
    /// Drop `owner`'s claim on `seq_hashes` at `tier`. Each seq hash resolves
    /// to its node via `anchors` and its index in that node's edge; a miss
    /// (block not stored here, or its node already dropped) is skipped. Other
    /// owners/tiers survive.
    pub fn remove(
        &self,
        owner: &ResidencyOwner,
        tier: StorageTier,
        seq_hashes: &[Arc<str>],
    ) -> Vec<Arc<str>> {
        self.remove_pages(owner, Some(tier), seq_hashes, false)
    }

    pub(in crate::kv_index::index) fn remove_pages(
        &self,
        owner: &ResidencyOwner,
        tier: Option<StorageTier>,
        seq_hashes: &[Arc<str>],
        incremental: bool,
    ) -> Vec<Arc<str>> {
        if seq_hashes.len() > 1 {
            if let Some(tier) = tier {
                if let Some(removed) =
                    self.try_remove_same_node(owner, tier, seq_hashes, incremental)
                {
                    return removed;
                }
            }
        }
        let mut removed = Vec::new();
        for seq in seq_hashes {
            let mut last = false;
            if let Some(mut claims) = self.reverse.get_mut(owner) {
                let fact = FactKey::from_seq(seq);
                if let Some(bits) = claims.get_mut(&fact) {
                    bits.withdraw(tier, incremental);
                    if bits.tiers.0 == 0 {
                        claims.remove(&fact);
                        last = true;
                    }
                }
            }
            for t in TIERS
                .into_iter()
                .filter(|t| tier.is_none_or(|tier| tier == *t))
            {
                self.remove_claim(seq, &(owner.clone(), t));
            }
            if last {
                removed.push(seq.clone());
            }
        }
        removed
    }

    // All anchors are checked before mutation. Callers hold the source
    // transaction and topology read guard, so the batch is one observation.
    fn try_remove_same_node(
        &self,
        owner: &ResidencyOwner,
        tier: StorageTier,
        seqs: &[Arc<str>],
        incremental: bool,
    ) -> Option<Vec<Arc<str>>> {
        let target = owner.target();
        let first = self.positioned_anchor(&target, &seqs[0])?;
        let last = self.positioned_anchor(&target, &seqs[seqs.len() - 1])?;
        if !Arc::ptr_eq(&first.node, &last.node) {
            return None;
        }
        let mut offsets = Vec::with_capacity(seqs.len());
        offsets.push(first.offset);
        for seq in &seqs[1..seqs.len() - 1] {
            let anchor = self.positioned_anchor(&target, seq)?;
            if !Arc::ptr_eq(&first.node, &anchor.node) {
                return None;
            }
            offsets.push(anchor.offset);
        }
        offsets.push(last.offset);
        let mut node = first.node.write();
        if seqs.iter().zip(&offsets).any(|(seq, &idx)| {
            !node
                .edge
                .get(idx)
                .is_some_and(|(aliases, _)| aliases.has(&target, seq))
        }) {
            return None;
        }
        let mut facts = self.reverse.get_mut(owner);
        let mut removed = Vec::new();
        if let Some(facts) = facts.as_mut() {
            for seq in seqs {
                let fact = FactKey::from_seq(seq);
                if let Some(bits) = facts.get_mut(&fact) {
                    bits.withdraw(Some(tier), incremental);
                    if bits.tiers.0 == 0 {
                        facts.remove(&fact);
                        removed.push(seq.clone());
                    }
                }
            }
        }
        let key = (owner.clone(), tier);
        for &idx in &offsets {
            let active = facts.as_ref().is_some_and(|facts| {
                node.edge[idx].0.any_for(&target, |alias| {
                    facts
                        .get(&FactKey::from_seq(alias))
                        .is_some_and(|bits| bits.contains(tier))
                })
            });
            if !active {
                Arc::make_mut(&mut node.claims[idx]).retain(|claim| claim != &key);
            }
        }
        Some(removed)
    }

    /// Clear `owner`'s residency in `scope`. `Worker` never crosses into
    /// `CacheOwner`: the scope filters by domain, then the owner is matched
    /// within its domain (Worker by source+dp_rank, CacheOwner by pool_id).
    pub fn clear(&self, owner: &ResidencyOwner, scope: ClearScope) {
        let owners: Vec<ResidencyOwner> = self.reverse.iter().map(|e| e.key().clone()).collect();
        for candidate in owners {
            if !matches!(scope, ClearScope::All) && scope != candidate.domain() {
                continue;
            }
            if !same_clear_owner(owner, &candidate) {
                continue;
            }
            for tier in TIERS {
                let seqs = self.tier_sequences(&candidate, tier);
                self.remove(&candidate, tier, &seqs);
            }
            self.reverse
                .remove_if(&candidate, |_, facts| facts.is_empty());
        }
    }

    /// Forget only this publisher/rank's observations, including pool copies
    /// attributed to it. A pool-wide clear would erase other publishers too.
    pub(in crate::kv_index::index) fn clear_source(&self, source: &SourceId, dp_rank: u32) {
        let owners: Vec<ResidencyOwner> = self
            .reverse
            .iter()
            .filter(|entry| {
                let target = entry.key().target();
                target.instance_id == *source && target.dp_rank == dp_rank
            })
            .map(|entry| entry.key().clone())
            .collect();
        for owner in owners {
            for tier in TIERS {
                let seqs = self.tier_sequences(&owner, tier);
                self.remove(&owner, tier, &seqs);
            }
            self.reverse.remove_if(&owner, |_, facts| facts.is_empty());
        }
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        self.scoped_anchors.retain(|_, anchors| {
            anchors.remove(&target);
            !anchors.is_empty()
        });
        self.waiting.retain(|(scope, _), _| scope != &target);
        self.prune_source_paths(&target);
        self.prune_dead_nodes();
    }

    /// After all facts and anchors for a source are gone, remove its historical
    /// aliases and empty leaves. Retain every node with a live alias, claim, or
    /// descendant: another source may still use it as a parent anchor.
    pub(in crate::kv_index::index) fn prune_source_paths(&self, target: &RoutableTarget) {
        let mut aliases_to_remove = AliasRemover::new(target);
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
                aliases_to_remove.remove(aliases, None);
            }
            current.alias_count = current.edge.iter().map(|edge| edge.0.count()).sum();
            current.children.retain(|_, child| {
                let child = child.read();
                !child.children.is_empty()
                    || child.edge.iter().any(|(aliases, _)| !aliases.is_empty())
                    || child.claims.iter().any(|claims| !claims.is_empty())
            });
        }
    }

    pub(in crate::kv_index::index) fn anchor(
        &self,
        target: &RoutableTarget,
        seq: &Arc<str>,
    ) -> Option<NodeRef> {
        self.positioned_anchor(target, seq)
            .map(|anchor| anchor.node)
    }

    pub(in crate::kv_index::index) fn positioned_anchor(
        &self,
        target: &RoutableTarget,
        seq: &Arc<str>,
    ) -> Option<Anchor> {
        self.scoped_anchors
            .get(seq)
            .and_then(|entries| entries.get(target).and_then(|path| path.position.clone()))
    }
    /// Drop the logical block claim only after its last active alias is gone.
    pub(in crate::kv_index::index) fn remove_claim(&self, seq: &Arc<str>, key: &OwnerTier) {
        let target = key.0.target();
        let Some(anchor) = self.positioned_anchor(&target, seq) else {
            return;
        };
        let mut n = anchor.node.write();
        let idx = anchor.offset;
        // The caller holds the topology guard; a split cannot move this alias.
        assert!(
            n.edge
                .get(idx)
                .is_some_and(|(aliases, _)| aliases.has(&target, seq)),
            "prefix anchor moved under the topology guard"
        );
        let active = self.reverse.get(&key.0).is_some_and(|facts| {
            n.edge[idx].0.any_for(&target, |alias| {
                facts
                    .get(&FactKey::from_seq(alias))
                    .is_some_and(|bits| bits.contains(key.1))
            })
        });
        if !active {
            Arc::make_mut(&mut n.claims[idx]).retain(|claim| claim != key);
        }
    }
}
