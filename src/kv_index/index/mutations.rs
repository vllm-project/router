//! Page updates shared by path-addressed and hash-addressed cache groups.

use super::*;

/// Both forms address the same group records. A path additionally supplies navigation.
pub(crate) enum PageBatch<'a> {
    Path {
        namespace: Option<&'a str>,
        parent: Option<Arc<str>>,
        blocks: &'a [IncomingBlock],
    },
    Hashes {
        namespace: Option<&'a str>,
        hashes: &'a [SequenceHash],
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PageWriteError {
    Capacity,
    ConflictingPath,
}

pub(crate) enum RemovedPages<'a> {
    Path(&'a [Arc<str>]),
    Hashes(&'a [SequenceHash]),
}

impl IndexBatch<'_> {
    pub(crate) fn store(
        &mut self,
        group: u32,
        owner: ResidencyOwner,
        tier: StorageTier,
        parent: Option<Arc<str>>,
        blocks: &[IncomingBlock],
    ) -> bool {
        self.store_pages(
            group,
            &owner,
            tier,
            PageBatch::Path {
                namespace: None,
                parent,
                blocks,
            },
            false,
            usize::MAX,
        )
        .is_ok()
    }

    /// One production entry for complete paths, partial pages and checkpoints.
    /// Addressing changes navigation only; admission and residency are shared.
    pub(crate) fn store_pages(
        &mut self,
        group: u32,
        owner: &ResidencyOwner,
        tier: StorageTier,
        pages: PageBatch<'_>,
        incremental: bool,
        max_entries: usize,
    ) -> Result<(), PageWriteError> {
        let target = owner.target();
        self.check_target(&target);
        if incremental && tier != StorageTier::Device {
            return Err(PageWriteError::ConflictingPath);
        }
        // Callers supply wire identities in both forms. Namespace qualification
        // belongs here, so an event adapter cannot create a differently keyed fact.
        let qualified = match &pages {
            PageBatch::Path {
                namespace: Some(name),
                blocks,
                ..
            } => Some(
                blocks
                    .iter()
                    .map(|(seq, local)| (qualified_seq(name, seq), local.clone()))
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        };
        let pages = match pages {
            PageBatch::Path {
                namespace: Some(name),
                parent,
                ..
            } => PageBatch::Path {
                namespace: Some(name),
                parent: parent.map(|seq| qualified_seq(name, &seq)),
                blocks: qualified.as_deref().unwrap(),
            },
            other => other,
        };
        let namespace = match &pages {
            PageBatch::Hashes { namespace, .. } | PageBatch::Path { namespace, .. } => *namespace,
        };
        let admitted = if max_entries != usize::MAX || namespace.is_some() {
            let keys: Vec<_> = match &pages {
                PageBatch::Path { blocks, .. } => {
                    blocks.iter().map(|b| FactKey::from_seq(&b.0)).collect()
                }
                PageBatch::Hashes { hashes, .. } => prepare_sparse_keys(namespace, hashes),
            };
            Some(
                self.admit(owner, namespace, group, &keys, max_entries)
                    .ok_or(PageWriteError::Capacity)?,
            )
        } else {
            // Reclamation also uses this roster to protect live ancestor pages.
            self.with_data(&target, |data| {
                data.entry(owner.clone()).or_default().base_pages = None;
            });
            None
        };
        let claim = StoreClaim {
            key: (owner.clone(), tier),
            incremental,
        };
        let result = match pages {
            PageBatch::Path { parent, blocks, .. } => {
                if incremental {
                    let shard = self.index.shard(group);
                    if shard.reverse.get(owner).is_some_and(|facts| {
                        blocks.iter().any(|(seq, _)| {
                            facts
                                .get(&FactKey::from_seq(seq))
                                .is_some_and(|record| !record.can_observe(true))
                        })
                    }) {
                        return Err(PageWriteError::Capacity);
                    }
                }
                self.store_path(group, &claim, parent, blocks)
                    .then_some(())
                    .ok_or(PageWriteError::ConflictingPath)
            }
            PageBatch::Hashes { namespace, hashes } => {
                let shard = self.index.shard(group);
                let mut facts = shard.reverse.entry(owner.clone()).or_default();
                if facts.is_empty() {
                    facts.reserve(hashes.len());
                }
                for hash in hashes {
                    let key = match namespace {
                        None => FactKey::Digest(*hash),
                        Some(name) => FactKey::Text(sparse_key(Some(name), hash)),
                    };
                    if !facts.entry(key).or_default().observe(tier, incremental) {
                        return Err(PageWriteError::Capacity);
                    }
                }
                drop(facts);
                if namespace.is_none() && !shard.scoped_anchors.is_empty() {
                    let _topology = shard.topology.read();
                    for hash in hashes {
                        shard.attach_existing(owner, tier, &FactKey::Digest(*hash).to_seq());
                    }
                }
                Ok(())
            }
        };
        if let Some(next) = admitted.filter(|_| result.is_ok()) {
            self.with_data(&target, |data| {
                let metadata = data.entry(owner.clone()).or_default();
                match namespace {
                    Some(name) if next != 0 => {
                        metadata.namespaces.insert(name.to_owned(), next);
                    }
                    None => metadata.base_pages = Some(next),
                    _ => {}
                }
            });
        }
        result
    }

    fn admit(
        &mut self,
        owner: &ResidencyOwner,
        namespace: Option<&str>,
        group: u32,
        keys: &[FactKey],
        max_entries: usize,
    ) -> Option<usize> {
        let index = self.index;
        let shard = index.shard(group);
        self.with_data(&owner.target(), |data| {
            let metadata = data.entry(owner.clone()).or_default();
            let count = match namespace {
                Some(name) => metadata.namespaces.get(name).copied().unwrap_or(0),
                // Rebuild once after an unbounded write, then maintain the
                // number of rows, independent of their tiers and replicas.
                None => *metadata.base_pages.get_or_insert_with(|| {
                    index
                        .shards
                        .iter()
                        .filter_map(|entry| entry.reverse.get(owner).map(|facts| facts.len()))
                        .sum::<usize>()
                        .saturating_sub(metadata.namespaces.values().sum())
                }),
            };
            let facts = shard.reverse.get(owner);
            let new: HashSet<_, Fast> = keys
                .iter()
                .filter(|key| facts.as_ref().is_none_or(|facts| !facts.contains_key(*key)))
                .collect();
            count
                .checked_add(new.len())
                .filter(|next| *next <= max_entries)
        })
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
        self.remove_pages(group, owner, Some(tier), RemovedPages::Path(hashes), false);
    }

    pub(crate) fn remove_pages(
        &mut self,
        group: u32,
        owner: &ResidencyOwner,
        tier: Option<StorageTier>,
        pages: RemovedPages<'_>,
        incremental: bool,
    ) {
        if incremental && tier != Some(StorageTier::Device) {
            return;
        }
        match pages {
            RemovedPages::Path(hashes) => self.remove_path(group, owner, tier, hashes, incremental),
            RemovedPages::Hashes(hashes) => {
                self.remove_hashes(owner, group, hashes, tier, incremental);
            }
        }
    }

    fn remove_path(
        &mut self,
        group: u32,
        owner: &ResidencyOwner,
        tier: Option<StorageTier>,
        hashes: &[Arc<str>],
        incremental: bool,
    ) {
        let target = owner.target();
        self.check_target(&target);
        let shard = self.index.shard(group);
        let namespaces = self.with_data(&target, |data| {
            data.get(owner)
                .map(|meta| meta.namespaces.clone())
                .unwrap_or_default()
        });
        let expanded = (!namespaces.is_empty()).then(|| {
            hashes
                .iter()
                .flat_map(|hash| {
                    std::iter::once(hash.clone())
                        .chain(namespaces.keys().map(|name| qualified_seq(name, hash)))
                })
                .collect::<Vec<_>>()
        });
        let hashes = expanded.as_deref().unwrap_or(hashes);
        let _topology = shard.topology.read();
        let removed = shard.remove_pages(owner, tier, hashes, incremental);
        self.with_records(&target, |records| {
            if let Some(metadata) = records.owners.get_mut(owner) {
                for hash in &removed {
                    release_page_count(metadata, namespaced_seq(hash).map(|(name, _)| name));
                }
            }
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
        drop(_topology);
        self.with_records(&target, |records| {
            records.try_collect(&shard, &target, group, true);
        });
    }

    /// Removal events do not identify the namespace, so invalidate every
    /// matching namespace of this owner and group, but no other owner.
    fn remove_hashes(
        &mut self,
        owner: &ResidencyOwner,
        group: u32,
        hashes: &[SequenceHash],
        tier: Option<StorageTier>,
        incremental: bool,
    ) {
        if hashes.is_empty() {
            return;
        }
        let target = owner.target();
        let shard = self.index.shard(group);
        let _topology = shard.topology.read();
        // Topology writers are excluded until every fact and claim is updated.
        let has_anchors = !shard.scoped_anchors.is_empty();
        self.with_records(&target, |records| {
            let data = &mut records.owners;
            let namespaces: Vec<_> = data
                .get(owner)
                .map(|metadata| metadata.namespaces.keys().cloned().collect())
                .unwrap_or_default();
            let mut removed = Vec::new();
            let mut released = Vec::new();
            for hash in hashes {
                if let Some(mut facts) = shard.reverse.get_mut(owner) {
                    let candidates = std::iter::once((None, FactKey::Digest(*hash))).chain(
                        namespaces.iter().map(|name| {
                            (
                                Some(name.as_str()),
                                FactKey::Text(sparse_key(Some(name), hash)),
                            )
                        }),
                    );
                    for (namespace, hash) in candidates {
                        if let Some(bits) = facts.get_mut(&hash) {
                            let removed_tiers = bits.withdraw(tier, incremental);
                            if removed_tiers.0 == 0 {
                                continue;
                            }
                            let removed_last = bits.tiers.0 == 0;
                            if removed_last {
                                facts.remove(&hash);
                            }
                            removed.push((namespace, hash, removed_tiers, removed_last));
                        }
                    }
                }
                // Keep the shared fact-map guard short and reuse per-page scratch.
                for (namespace, hash, removed_tiers, removed_last) in removed.drain(..) {
                    if has_anchors {
                        let seq = hash.to_seq();
                        if removed_last
                            && shard
                                .scoped_anchors
                                .get(&seq)
                                .is_some_and(|paths| paths.get(&target).is_some())
                        {
                            released.push(seq.clone());
                        }
                        for t in TIERS.into_iter().filter(|t| removed_tiers.contains(*t)) {
                            shard.remove_claim(&seq, &(owner.clone(), t));
                        }
                    }
                    if removed_last {
                        if let Some(metadata) = data.get_mut(owner) {
                            release_page_count(metadata, namespace);
                        }
                    }
                }
            }
            if has_anchors {
                records.defer(group, released);
            }
        });
        drop(_topology);
        self.with_records(&target, |records| {
            records.try_collect(&shard, &target, group, true);
        });
    }
}

/// Capacity counts change only when the final tier/replica releases a row.
fn release_page_count(metadata: &mut OwnerMetadata, namespace: Option<&str>) {
    let count = match namespace {
        Some(name) => metadata.namespaces.get_mut(name),
        None => metadata.base_pages.as_mut(),
    };
    if let Some(count) = count {
        *count -= 1;
        if *count == 0 {
            if let Some(name) = namespace {
                metadata.namespaces.remove(name);
            }
        }
    }
}
