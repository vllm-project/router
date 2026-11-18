//! Source-scoped cache updates and consistent snapshots.

use super::*;

/// A batch holds either the global barrier exclusively, or the barrier for
/// reading together with its scoped source write lock. Its closure must not
/// re-enter any locking `KvBlockIndexer` method.
pub(crate) struct IndexBatch<'a> {
    pub(super) index: &'a KvBlockIndexer,
    pub(super) scope: Option<RoutableTarget>,
    pub(super) source_data: Option<&'a mut SourceRecords>,
}

/// A consistent view held under either the global barrier or a source lock.
pub(crate) struct IndexSnapshot<'a> {
    pub(super) index: &'a KvBlockIndexer,
    pub(super) scope: Option<RoutableTarget>,
    pub(super) source_data: Option<&'a SourceData>,
    pub(super) source: Option<&'a Arc<SourceLock>>,
}

impl IndexBatch<'_> {
    pub(super) fn check_target(&self, target: &RoutableTarget) {
        if let Some(scope) = &self.scope {
            assert_eq!(scope, target, "source transaction crossed targets");
        } else {
            self.index.source_lock(target);
        }
    }

    pub(super) fn with_data<R>(
        &mut self,
        target: &RoutableTarget,
        operation: impl FnOnce(&mut HashMap<ResidencyOwner, OwnerMetadata>) -> R,
    ) -> R {
        self.with_records(target, |records| operation(&mut records.owners))
    }

    pub(super) fn with_records<R>(
        &mut self,
        target: &RoutableTarget,
        operation: impl FnOnce(&mut SourceRecords) -> R,
    ) -> R {
        self.check_target(target);
        if let Some(data) = self.source_data.as_deref_mut() {
            operation(data)
        } else {
            // The global domain write guard excludes every source transaction.
            let source = self.index.source_lock(target);
            let mut data = source.access.write();
            operation(&mut data.records)
        }
    }

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

    pub(crate) fn invalidate_source(&mut self, source: &SourceId, rank: u32) {
        let index = self.index;
        self.with_records(
            &RoutableTarget {
                instance_id: source.clone(),
                dp_rank: rank,
            },
            |records| index.invalidate_source_unlocked(source, rank, records),
        );
    }

    pub(crate) fn clear(&mut self, owner: &ResidencyOwner, scope: ClearScope) {
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
            .map(|e| (*e.key(), e.value().clone()))
            .collect();
        for (group, shard) in shards {
            let _topology = shard.topology.read();
            let mut removed: HashMap<RoutableTarget, Vec<Arc<str>>> = HashMap::new();
            for facts in &shard.reverse {
                let candidate = facts.key();
                if (scope == ClearScope::All || scope == candidate.domain())
                    && same_clear_owner(owner, candidate)
                {
                    removed.entry(candidate.target()).or_default().extend(
                        facts.keys().map(FactKey::to_seq).filter(|hash| {
                            shard
                                .scoped_anchors
                                .get(hash)
                                .is_some_and(|paths| paths.get(&candidate.target()).is_some())
                        }),
                    );
                }
            }
            shard.clear(owner, scope);
            drop(_topology);
            for (target, hashes) in removed {
                self.with_records(&target, |records| {
                    records.defer(group, hashes);
                    records.try_collect(&shard, &target, group, false);
                });
            }
        }
        let keep = |candidate: &ResidencyOwner, _: &mut OwnerMetadata| {
            (!matches!(scope, ClearScope::All) && scope != candidate.domain())
                || !same_clear_owner(owner, candidate)
        };
        if let Some(data) = self.source_data.as_deref_mut() {
            data.owners.retain(keep);
        } else {
            let sources: Vec<_> = self
                .index
                .source_locks
                .iter()
                .map(|entry| entry.value().clone())
                .collect();
            for source in sources {
                source.access.write().records.owners.retain(keep);
            }
        }
    }

    /// Remove only this precise owner, including its incarnation.
    pub(crate) fn clear_owner_exact(&mut self, owner: &ResidencyOwner) {
        self.check_target(&owner.target());
        let shards: Vec<_> = self
            .index
            .shards
            .iter()
            .map(|e| (*e.key(), e.value().clone()))
            .collect();
        for (group, shard) in shards {
            let _topology = shard.topology.read();
            if shard.scoped_anchors.is_empty() {
                shard.reverse.remove(owner);
                continue;
            }
            let mut removed = Vec::new();
            for tier in TIERS {
                let hashes = shard.tier_sequences(owner, tier);
                removed.extend(shard.remove(owner, tier, &hashes));
            }
            shard.reverse.remove_if(owner, |_, facts| facts.is_empty());
            let target = owner.target();
            drop(_topology);
            self.with_records(&target, |records| {
                records.defer(group, removed);
                records.try_collect(&shard, &target, group, false);
            });
        }
        self.with_data(&owner.target(), |data| {
            data.remove(owner);
        });
    }
}

impl SourceRecords {
    pub(super) fn defer(&mut self, group: u32, hashes: Vec<Arc<str>>) {
        if !hashes.is_empty() {
            self.reclaim.entry(group).or_default().extend(hashes);
        }
    }

    /// The source transaction and topology writer protect all reclamation.
    pub(super) fn collect(&mut self, shard: &GroupShard, target: &RoutableTarget, group: u32) {
        if let Some(hashes) = self.reclaim.remove(&group) {
            let owners: Vec<_> = self.owners.keys().cloned().collect();
            shard.release_paths(target, &owners, &hashes.into_iter().collect::<Vec<_>>());
        }
        shard.prune_dead_nodes();
    }

    pub(super) fn try_collect(
        &mut self,
        shard: &GroupShard,
        target: &RoutableTarget,
        group: u32,
        empty_only: bool,
    ) {
        if !self.reclaim.contains_key(&group)
            || (empty_only
                && self.owners.keys().any(|owner| {
                    shard
                        .reverse
                        .get(owner)
                        .is_some_and(|facts| !facts.is_empty())
                }))
        {
            return;
        }
        if let Some(_topology) = shard.topology.try_write() {
            self.collect(shard, target, group);
        }
    }
}

impl IndexSnapshot<'_> {
    pub(super) fn check_target(&self, target: &RoutableTarget) {
        if let Some(scope) = &self.scope {
            assert_eq!(scope, target, "source snapshot crossed targets");
        }
    }

    pub(crate) fn stream(&self, owner: &ResidencyOwner) -> Option<&StreamState> {
        self.check_target(&owner.target());
        self.source_data?.streams.get(owner)
    }

    pub(crate) fn source_version(&self, target: &RoutableTarget) -> SourceVersion {
        self.check_target(target);
        let source = self
            .source
            .expect("source version requires a scoped snapshot")
            .clone();
        let generation = source.generation.load(Ordering::Acquire);
        SourceVersion {
            target: target.clone(),
            source,
            generation,
        }
    }
    /// Resolve all requested hashes under one owner fact-table guard.
    pub(crate) fn page_tiers(
        &self,
        owner: &ResidencyOwner,
        group: u32,
        keys: &[FactKey],
    ) -> Vec<TierBits> {
        let mut result = vec![TierBits::default(); keys.len()];
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
        if let Some(facts) = shard.reverse.get(owner) {
            for (key, hit) in keys.iter().zip(&mut result) {
                *hit = facts
                    .get(key)
                    .map_or(TierBits::default(), |record| record.tiers);
            }
        }
        result
    }
}
