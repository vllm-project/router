//! Full-attention path updates under one source transaction.

use super::*;

impl IndexBatch<'_> {
    pub(crate) fn store(
        &mut self,
        group: u32,
        owner: ResidencyOwner,
        tier: StorageTier,
        parent: Option<Arc<str>>,
        blocks: &[IncomingBlock],
    ) -> bool {
        let target = owner.target();
        self.check_target(&target);
        self.with_data(&target, |owners| {
            owners.entry(owner.clone()).or_default();
        });
        self.store_path(
            group,
            &StoreClaim {
                key: (owner, tier),
                incremental: false,
            },
            parent,
            blocks,
        )
    }

    fn store_path(
        &mut self,
        group: u32,
        claim: &StoreClaim,
        parent: Option<Arc<str>>,
        blocks: &[(Arc<str>, Arc<str>)],
    ) -> bool {
        let target = claim.key.0.target();
        self.check_target(&target);
        let shard = self.index.shard(group);
        if parent.is_some() {
            let topology = shard.topology.read();
            if shard.try_store_anchored(claim, parent.as_ref(), blocks) {
                return true;
            }
            drop(topology);
        }
        let topology = shard.topology.write();
        let source = self.index.source_lock(&target);
        let accepted = shard.store_for_source(&source.target, claim, parent, blocks);
        self.with_records(&target, |records| {
            records.collect(&shard, &target, group);
        });
        drop(topology);
        if !accepted {
            // Hide and discard every observation from the ambiguous source
            // while the domain write lock still protects all readers.
            let index = self.index;
            self.with_records(&target, |records| {
                index.invalidate_source_unlocked(&target.instance_id, target.dp_rank, records)
            });
        }
        accepted
    }

    pub(crate) fn remove(
        &mut self,
        group: u32,
        owner: &ResidencyOwner,
        tier: StorageTier,
        hashes: &[Arc<str>],
    ) {
        let target = owner.target();
        self.check_target(&target);
        let shard = self.index.shard(group);
        let topology = shard.topology.read();
        let removed = shard.remove(owner, tier, hashes);
        self.with_records(&target, |records| {
            records.defer(
                group,
                removed
                    .into_iter()
                    .filter(|hash| {
                        shard
                            .scoped_anchors
                            .get(hash)
                            .is_some_and(|paths| paths.get(&target).is_some())
                    })
                    .collect(),
            );
        });
        drop(topology);
        self.with_records(&target, |records| {
            records.try_collect(&shard, &target, group, true)
        });
    }
}
