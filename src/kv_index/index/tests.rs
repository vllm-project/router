#![cfg(all(test, feature = "cpu", feature = "ready"))]

use super::*;
use crate::kv_index::types::SourceId;

impl GroupShard {
    fn store(
        &self,
        owner: ResidencyOwner,
        tier: StorageTier,
        parent_seq: Option<Arc<str>>,
        blocks: &[IncomingBlock],
    ) -> bool {
        self.store_for_source(
            &Arc::new(owner.target()),
            &StoreClaim {
                key: (owner, tier),
                incremental: false,
            },
            parent_seq,
            blocks,
        )
    }

    fn find_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
        Self::aggregate_claims(query, &self.collect_claims(query))
    }
}

impl KvBlockIndexer {
    fn with_snapshot<R>(&self, f: impl FnOnce(&IndexSnapshot<'_>) -> R) -> R {
        let _guard = self.domain.write();
        f(&IndexSnapshot {
            index: self,
            scope: None,
            source_data: None,
            source: None,
        })
    }
}

impl IndexBatch<'_> {
    fn store_sparse(
        &mut self,
        owner: &ResidencyOwner,
        namespace: Option<&str>,
        group: u32,
        hash: &SequenceHash,
        tier: StorageTier,
        max_entries: usize,
    ) -> bool {
        self.store_pages(
            group,
            owner,
            tier,
            PageBatch::Hashes {
                namespace,
                hashes: &[*hash],
            },
            false,
            max_entries,
        )
        .is_ok()
    }
    fn remove_sparse(
        &mut self,
        owner: &ResidencyOwner,
        group: u32,
        hash: &SequenceHash,
        tier: Option<StorageTier>,
    ) {
        self.remove_pages(group, owner, tier, RemovedPages::Hashes(&[*hash]), false);
    }
}

impl IndexSnapshot<'_> {
    fn best_sparse_tier(
        &self,
        owner: &ResidencyOwner,
        namespace: Option<&str>,
        group: u32,
        hash: &SequenceHash,
    ) -> Option<StorageTier> {
        let hash = match namespace {
            None => FactKey::Digest(*hash),
            Some(name) => FactKey::Text(sparse_key(Some(name), hash)),
        };
        self.page_tiers(owner, group, &[hash])[0].best()
    }

    fn sparse_tiers(
        &self,
        owner: &ResidencyOwner,
        group: u32,
        keys: &[FactKey],
    ) -> Vec<Option<StorageTier>> {
        self.page_tiers(owner, group, keys)
            .into_iter()
            .map(TierBits::best)
            .collect()
    }
}

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
fn path_and_hash_pages_have_identical_replica_tier_and_clear_transitions() {
    use super::mutations::{PageBatch, RemovedPages};
    let index = KvBlockIndexer::new();
    let a = worker_rank("w1", 0);
    let b = worker_rank("w2", 1);
    let hashes = [[1; 32], [2; 32], [3; 32]];
    let blocks: Vec<_> = hashes
        .iter()
        .enumerate()
        .map(|(i, hash)| (sparse_key(None, hash), h(&format!("local-{i}"))))
        .collect();
    let keys = prepare_sparse_keys(None, &hashes);
    let check = || {
        let left = index.with_source_snapshot(&a.target(), |view| view.page_tiers(&a, 0, &keys));
        let right = index.with_source_snapshot(&b.target(), |view| view.page_tiers(&b, 0, &keys));
        assert_eq!(left, right);
        let hits = index.find_matches(&MatchQuery {
            group_idx: 0,
            local_hashes: blocks.iter().map(|b| b.1.clone()).collect(),
            tiers_of_interest: TIERS.to_vec(),
        });
        for tier in TIERS {
            for (i, bits) in left.iter().enumerate() {
                let actual = segments_of(&hits, "w1", tier)
                    .iter()
                    .any(|&(s, e)| s <= i as u32 && (i as u32) < e);
                assert_eq!(actual, bits.contains(tier));
            }
        }
    };
    for (tier, incremental) in [
        (StorageTier::Device, true),
        (StorageTier::Device, true),
        (StorageTier::HostPinned, false),
    ] {
        index.with_source_batch(&a.target(), |tx| {
            assert!(tx
                .store_pages(
                    0,
                    &a,
                    tier,
                    PageBatch::Path {
                        namespace: None,
                        parent: None,
                        blocks: &blocks
                    },
                    incremental,
                    3
                )
                .is_ok())
        });
        index.with_source_batch(&b.target(), |tx| {
            assert!(tx
                .store_pages(
                    0,
                    &b,
                    tier,
                    PageBatch::Hashes {
                        namespace: None,
                        hashes: &hashes
                    },
                    incremental,
                    3
                )
                .is_ok())
        });
        check();
    }
    for tier in [
        StorageTier::Device,
        StorageTier::Device,
        StorageTier::HostPinned,
    ] {
        let incremental = tier == StorageTier::Device;
        let seqs: Vec<_> = blocks.iter().map(|b| b.0.clone()).collect();
        index.with_source_batch(&a.target(), |tx| {
            tx.remove_pages(0, &a, Some(tier), RemovedPages::Path(&seqs), incremental)
        });
        index.with_source_batch(&b.target(), |tx| {
            tx.remove_pages(
                0,
                &b,
                Some(tier),
                RemovedPages::Hashes(&hashes),
                incremental,
            )
        });
        check();
    }
    assert!(index.shard(0).scoped_anchors.is_empty());
    for owner in [&a, &b] {
        index.with_source_batch(&owner.target(), |tx| {
            assert!(tx
                .store_pages(
                    1,
                    owner,
                    StorageTier::Device,
                    PageBatch::Hashes {
                        namespace: None,
                        hashes: &hashes[..1]
                    },
                    true,
                    1
                )
                .is_ok());
            assert!(tx
                .store_pages(
                    1,
                    owner,
                    StorageTier::Device,
                    PageBatch::Hashes {
                        namespace: None,
                        hashes: &hashes[1..]
                    },
                    true,
                    1
                )
                .is_err());
            tx.clear_owner_exact(owner);
        });
    }
    assert!(index.shard(1).reverse.is_empty());
}

#[test]
fn named_adapter_path_and_hash_pages_share_one_key_and_copy_count() {
    let index = KvBlockIndexer::new();
    let owner = worker_rank("w1", 1);
    let name = "适配:a";
    let hash = [7; 32];
    let blocks = [(sparse_key(None, &hash), h("adapter-local"))];
    let keys = prepare_sparse_keys(Some(name), &[hash]);
    index.with_source_batch(&owner.target(), |tx| {
        assert!(tx
            .store_pages(
                0,
                &owner,
                StorageTier::Device,
                PageBatch::Path {
                    namespace: Some(name),
                    parent: None,
                    blocks: &blocks
                },
                true,
                1
            )
            .is_ok());
        assert!(tx
            .store_pages(
                0,
                &owner,
                StorageTier::Device,
                PageBatch::Hashes {
                    namespace: Some(name),
                    hashes: &[hash]
                },
                true,
                1
            )
            .is_ok());
        tx.remove_pages(
            0,
            &owner,
            Some(StorageTier::Device),
            RemovedPages::Hashes(&[hash]),
            true,
        );
    });
    assert_eq!(
        index.with_source_snapshot(&owner.target(), |v| v.page_tiers(&owner, 0, &keys)),
        vec![TierBits(TierBits::bit(StorageTier::Device))]
    );
    assert_eq!(
        depth_of(
            &index.find_matches(&query(&["adapter-local"], &[StorageTier::Device])),
            "w1",
            StorageTier::Device
        ),
        1
    );
    index.with_source_batch(&owner.target(), |tx| {
        tx.remove_pages(
            0,
            &owner,
            Some(StorageTier::Device),
            RemovedPages::Path(&[sparse_key(None, &hash)]),
            true,
        )
    });
    assert_eq!(
        index.with_source_snapshot(&owner.target(), |v| v.page_tiers(&owner, 0, &keys)),
        vec![TierBits::default()]
    );
    assert!(index
        .find_matches(&query(&["adapter-local"], &[StorageTier::Device]))
        .is_empty());
}

#[test]
fn namespaced_integer_parent_cannot_borrow_another_source_path() {
    let index = KvBlockIndexer::new();
    let first = worker("first");
    let second = worker("second");
    let parent = h("i:1");
    let child = h("i:2");
    index.with_source_batch(&first.target(), |tx| {
        assert!(tx
            .store_pages(
                0,
                &first,
                StorageTier::Device,
                PageBatch::Path {
                    namespace: Some("lora"),
                    parent: None,
                    blocks: &[(parent.clone(), h("x"))]
                },
                false,
                8
            )
            .is_ok())
    });
    index.with_source_batch(&second.target(), |tx| {
        assert!(tx
            .store_pages(
                0,
                &second,
                StorageTier::Device,
                PageBatch::Path {
                    namespace: Some("lora"),
                    parent: Some(parent),
                    blocks: &[(child.clone(), h("y"))]
                },
                false,
                8
            )
            .is_ok())
    });
    assert!(index
        .shard(0)
        .positioned_anchor(&second.target(), &qualified_seq("lora", &child))
        .is_none());
    assert_eq!(
        depth_of(
            &index.find_matches(&query(&["x", "y"], &[StorageTier::Device])),
            "second",
            StorageTier::Device
        ),
        0
    );
}

#[test]
fn retired_adapter_names_do_not_accumulate_while_base_pages_stay_live() {
    let index = KvBlockIndexer::new();
    let owner = worker("w");
    let base = [(h("base"), h("base-local"))];
    index.store(0, owner.clone(), StorageTier::Device, None, &base);
    for i in 0..8 {
        let name = format!("adapter-{i}");
        let hash = [i as u8; 32];
        let blocks = [(sparse_key(None, &hash), h(&format!("adapter-local-{i}")))];
        index.with_source_batch(&owner.target(), |tx| {
            let pages = if i % 2 == 0 {
                PageBatch::Path {
                    namespace: Some(&name),
                    parent: None,
                    blocks: &blocks,
                }
            } else {
                PageBatch::Hashes {
                    namespace: Some(&name),
                    hashes: std::slice::from_ref(&hash),
                }
            };
            assert!(tx
                .store_pages(0, &owner, StorageTier::Device, pages, true, 1)
                .is_ok());
        });
        index.with_source_batch(&owner.target(), |tx| {
            if i % 2 == 0 {
                tx.remove_pages(
                    0,
                    &owner,
                    Some(StorageTier::Device),
                    RemovedPages::Hashes(&[hash]),
                    true,
                );
            } else {
                tx.remove_pages(
                    0,
                    &owner,
                    Some(StorageTier::Device),
                    RemovedPages::Path(&[sparse_key(None, &hash)]),
                    true,
                );
            }
        });
        assert_eq!(
            index.with_source_snapshot(&owner.target(), |view| {
                view.page_tiers(&owner, 0, &prepare_sparse_keys(Some(&name), &[hash]))
            }),
            vec![TierBits::default()]
        );
        let gate = index.source_lock(&owner.target());
        let state = gate.access.read();
        let metadata = state.records.owners.get(&owner).unwrap();
        assert!(metadata.namespaces.is_empty(), "iteration {i}");
    }
    index.store(0, owner.clone(), StorageTier::Device, None, &base);
    assert_eq!(index.shard(0).scoped_anchors.len(), 1);
    assert_eq!(
        depth_of(
            &index.find_matches(&query(&["base-local"], &[StorageTier::Device])),
            "w",
            StorageTier::Device
        ),
        1
    );
}

#[test]
fn fact_keys_preserve_wire_identity_and_canonical_digest_equivalence() {
    let hash = [0xab; 32];
    let canonical = sparse_key(None, &hash);
    assert_eq!(FactKey::from_seq(&canonical), FactKey::Digest(hash));
    assert_eq!(FactKey::Digest(hash).to_seq(), canonical);
    let uppercase = h(&format!("b:{}", "AB".repeat(32)));
    let short = h("b:ab");
    let integer = h("i:42");
    let adapter = sparse_key(Some("adapter"), &hash);
    for seq in [&uppercase, &short, &integer, &adapter] {
        let key = FactKey::from_seq(seq);
        assert!(matches!(key, FactKey::Text(_)));
        assert_eq!(&key.to_seq(), seq);
        assert_ne!(key, FactKey::Digest(hash));
    }
}

#[test]
fn prepared_sparse_keys_match_canonical_base_and_adapter_keys() {
    let hash: SequenceHash = std::array::from_fn(|i| (i * 7) as u8);
    let base = prepare_sparse_keys(None, &[hash]);
    assert!(matches!(base[0], FactKey::Digest(_)));
    assert_eq!(base[0].to_seq(), sparse_key(None, &hash));
    let expected = format!(
        "b:{}",
        hash.iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<Vec<_>>()
            .join("")
    );
    assert_eq!(base[0].to_seq().as_ref(), expected);
    assert_eq!(base[0].to_seq().len(), 66);
    assert!(base[0].to_seq().starts_with("b:00070e"));
    let adapter = prepare_sparse_keys(Some("适配:a"), &[hash]);
    assert!(matches!(adapter[0], FactKey::Text(_)));
    assert_eq!(
        adapter[0].to_seq().as_ref(),
        sparse_key(Some("适配:a"), &hash).as_ref()
    );
    assert_eq!(
        adapter[0].to_seq().as_ref(),
        format!("n:{}:适配:a:{}", "适配:a".len(), expected)
    );
    assert_ne!(base[0], adapter[0]);
}

#[test]
fn inline_aliases_preserve_multiple_sources_and_hashes() {
    use super::prefix_tree::Aliases;

    let a = Arc::new(worker("a").target());
    let b = Arc::new(worker("b").target());
    let mut aliases = Aliases::One(a.clone(), h("first"));
    aliases.insert(a.clone(), h("first"));
    assert_eq!(aliases.count(), 1);
    aliases.insert(a.clone(), h("second"));
    aliases.insert(b.clone(), h("first"));
    assert_eq!(aliases.count(), 3);
    assert!(aliases.has(&a, &h("first")));
    assert!(aliases.has(&a, &h("second")));
    assert!(aliases.has(&b, &h("first")));
    aliases.remove_target(&a);
    assert!(!aliases.has(&a, &h("first")));
    assert!(aliases.has(&b, &h("first")));
    aliases.remove_target(&b);
    assert!(aliases.is_empty());
    aliases.insert(a.clone(), h("first"));
    aliases.insert(a.clone(), h("second"));
    aliases.insert(b.clone(), h("third"));
    aliases.remove_target(&b);
    assert_eq!(aliases.count(), 2);
    aliases.remove_alias(&a, &h("first"));
    assert_eq!(aliases.count(), 1);
    assert!(aliases.has(&a, &h("second")));
    assert!(!aliases.has(&a, &h("first")));
    aliases.remove_alias(&a, &h("second"));
    assert!(aliases.is_empty());
}

#[test]
fn shared_alias_rosters_detach_for_page_removal_and_different_hashes() {
    use super::prefix_tree::{AliasRemover, AliasWriter, Aliases};

    let a = Arc::new(worker("a").target());
    let b = Arc::new(worker_rank("b", 1).target());
    let mut first = Aliases::One(a.clone(), h("first"));
    let mut second = Aliases::One(a.clone(), h("second"));
    let mut writer = AliasWriter::new(&b);
    assert!(writer.add(&mut first, &h("first")));
    assert!(writer.add(&mut second, &h("second")));
    assert_eq!(first.count(), 2);
    assert_eq!(second.count(), 2);

    first.remove_alias(&b, &h("first"));
    assert!(!first.has(&b, &h("first")));
    assert!(second.has(&b, &h("second")));
    assert!(first.has(&a, &h("first")));

    second.insert(b.clone(), h("different"));
    assert!(second.has(&a, &h("second")));
    assert!(second.has(&b, &h("second")));
    assert!(second.has(&b, &h("different")));
    second.remove_alias(&b, &h("second"));
    assert!(second.has(&a, &h("second")));
    assert!(second.has(&b, &h("different")));
    assert!(!second.has(&b, &h("second")));
    second.remove_target(&b);
    assert_eq!(second.count(), 1);

    let c = Arc::new(worker("c").target());
    let mut pages = [
        Aliases::One(a.clone(), h("x")),
        Aliases::One(a.clone(), h("y")),
    ];
    for target in [&b, &c] {
        let mut writer = AliasWriter::new(target);
        writer.add(&mut pages[0], &h("x"));
        writer.add(&mut pages[1], &h("y"));
    }
    let mut remover = AliasRemover::new(&b);
    remover.remove(&mut pages[0], Some(&h("x")));
    remover.remove(&mut pages[1], Some(&h("y")));
    assert_eq!(pages[0].count(), 2);
    assert_eq!(pages[1].count(), 2);
    assert!(!pages[1].has(&b, &h("y")));
    assert!(pages[1].has(&c, &h("y")));
    assert!(!pages[0].has(&b, &h("x")));
    assert!(pages[0].has(&c, &h("x")));
    AliasRemover::new(&a).remove(&mut pages[1], Some(&h("wrong")));
    assert!(pages[1].has(&a, &h("y")));
}

#[test]
fn shared_claim_lists_keep_page_tier_and_snapshot_isolation() {
    let index = KvBlockIndexer::new();
    let a = worker("a");
    let b = worker("b");
    let blocks = [(h("s0"), h("x")), (h("s1"), h("y")), (h("s2"), h("z"))];
    let q = query(
        &["x", "y", "z"],
        &[StorageTier::Device, StorageTier::HostPinned],
    );
    index.store(0, a.clone(), StorageTier::Device, None, &blocks);
    let snapshot = index.shard(0).collect_claims(&q);
    index.store(0, b.clone(), StorageTier::HostPinned, None, &blocks);
    index.remove(0, &a, StorageTier::Device, &[h("s1")]);
    let matches = index.find_matches(&q);
    assert_eq!(
        segments_of(&matches, "a", StorageTier::Device),
        vec![(0, 1), (2, 3)]
    );
    assert_eq!(
        segments_of(&matches, "b", StorageTier::HostPinned),
        vec![(0, 3)]
    );
    for claims in snapshot {
        assert_eq!(claims.owners.as_slice(), [(a.clone(), StorageTier::Device)]);
    }
    // Restore across both modified and unchanged lists, then remove one tier
    // from another page. Neither transformation may spread to its neighbours.
    index.store(0, a, StorageTier::Device, Some(h("s0")), &blocks[1..]);
    index.remove(0, &b, StorageTier::HostPinned, &[h("s2")]);
    let matches = index.find_matches(&q);
    assert_eq!(
        segments_of(&matches, "a", StorageTier::Device),
        vec![(0, 3)]
    );
    assert_eq!(
        segments_of(&matches, "b", StorageTier::HostPinned),
        vec![(0, 2)]
    );
}

#[test]
fn stale_query_version_cannot_credit_replayed_or_replaced_sources() {
    let index = KvBlockIndexer::new();
    let owner = worker("w");
    let target = owner.target();
    let blocks = [(h("s0"), h("x"))];
    let q = query(&["x"], &[StorageTier::Device]);
    index.store(0, owner.clone(), StorageTier::Device, None, &blocks);
    let version = || {
        let source = index.source_lock(&target);
        SourceVersion {
            target: target.clone(),
            generation: source.generation.load(Ordering::Acquire),
            source,
        }
    };
    let before_replay = version();
    let claims = index.shard(0).collect_claims(&q);
    assert!(index.source_version_is_current(&before_replay));
    index.invalidate_source(&target.instance_id, target.dp_rank);
    index.store(0, owner.clone(), StorageTier::Device, None, &blocks);
    index.resume_source(&target.instance_id, target.dp_rank);
    assert!(!GroupShard::aggregate_claims(&q, &claims).is_empty());
    assert!(!index.source_version_is_current(&before_replay));
    let before_retirement = version();
    index.retire_source(&target.instance_id, target.dp_rank);
    index.store(0, owner, StorageTier::Device, None, &blocks);
    assert!(!index.source_version_is_current(&before_retirement));
    assert!(index.source_version_is_current(&version()));
    assert_eq!(index.find_matches(&q).len(), 1);
}

#[test]
fn successive_stores_share_prefix_until_divergence() {
    let index = KvBlockIndexer::new();
    let w1 = worker("w1");
    let w2 = worker("w2");
    let blocks = [(h("s0"), h("a")), (h("s1"), h("b")), (h("s2"), h("c"))];
    for owner in [w1.clone(), w2.clone()] {
        for i in 0..blocks.len() {
            let parent = i.checked_sub(1).map(|p| blocks[p].0.clone());
            index.store(
                0,
                owner.clone(),
                StorageTier::Device,
                parent,
                &blocks[i..=i],
            );
        }
    }
    let shard = index.shard(0);
    let first = shard.root.read().children.get(&h("a")).unwrap().clone();
    assert_eq!(first.read().edge.len(), 3);
    assert!(first.read().children.is_empty());

    index.store(
        0,
        w1.clone(),
        StorageTier::Device,
        Some(h("s1")),
        &[(h("sx"), h("x"))],
    );
    assert_eq!(first.read().edge.len(), 2);
    assert_eq!(first.read().children.len(), 2);
    let original = index.find_matches(&query(&["a", "b", "c"], &[StorageTier::Device]));
    assert_eq!(depth_of(&original, "w1", StorageTier::Device), 3);
    assert_eq!(depth_of(&original, "w2", StorageTier::Device), 3);
    let branch = index.find_matches(&query(&["a", "b", "x"], &[StorageTier::Device]));
    assert_eq!(depth_of(&branch, "w1", StorageTier::Device), 3);
    assert_eq!(depth_of(&branch, "w2", StorageTier::Device), 2);

    index.remove(0, &w1, StorageTier::Device, &[h("s1")]);
    let after = index.find_matches(&query(&["a", "b", "c"], &[StorageTier::Device]));
    assert_eq!(
        segments_of(&after, "w1", StorageTier::Device),
        [(0, 1), (2, 3)]
    );
    assert_eq!(depth_of(&after, "w2", StorageTier::Device), 3);
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
fn sparse_remove_preserves_other_dense_alias_and_tier() {
    let index = KvBlockIndexer::new();
    let owner = worker("a");
    let hash = [9; 32];
    let seq = sparse_key(None, &hash);
    let alias = h("legacy-alias");
    for tier in [StorageTier::Device, StorageTier::HostPinned] {
        index.store(0, owner.clone(), tier, None, &[(seq.clone(), h("x"))]);
        index.store(0, owner.clone(), tier, None, &[(alias.clone(), h("x"))]);
    }
    index.with_batch(|batch| batch.remove_sparse(&owner, 0, &hash, Some(StorageTier::Device)));
    assert_eq!(
        depth_of(
            &index.find_matches(&query(&["x"], &[StorageTier::Device])),
            "a",
            StorageTier::Device
        ),
        1
    );
    assert_eq!(
        index.with_snapshot(|view| view.best_sparse_tier(&owner, None, 0, &hash)),
        Some(StorageTier::HostPinned)
    );
    index.remove(0, &owner, StorageTier::Device, &[alias]);
    assert!(index
        .find_matches(&query(&["x"], &[StorageTier::Device]))
        .is_empty());
    assert_eq!(
        depth_of(
            &index.find_matches(&query(&["x"], &[StorageTier::HostPinned])),
            "a",
            StorageTier::HostPinned
        ),
        1
    );
}

#[test]
fn batched_remove_keeps_other_aliases_sources_and_tiers() {
    let index = KvBlockIndexer::new();
    let a = worker("a");
    let b = worker("b");
    for seq in ["a0", "a1", "a2"] {
        for tier in [StorageTier::Device, StorageTier::HostPinned] {
            index.store(0, a.clone(), tier, None, &[(h(seq), h("x"))]);
        }
    }
    index.store(0, b, StorageTier::Device, None, &[(h("b0"), h("x"))]);

    index.remove(0, &a, StorageTier::Device, &[h("a0"), h("a1"), h("a0")]);
    let hits = index.find_matches(&query(
        &["x"],
        &[StorageTier::Device, StorageTier::HostPinned],
    ));
    assert_eq!(depth_of(&hits, "a", StorageTier::Device), 1);
    assert_eq!(depth_of(&hits, "a", StorageTier::HostPinned), 1);
    assert_eq!(depth_of(&hits, "b", StorageTier::Device), 1);

    index.remove(0, &a, StorageTier::Device, &[h("a2"), h("unknown")]);
    let hits = index.find_matches(&query(
        &["x"],
        &[StorageTier::Device, StorageTier::HostPinned],
    ));
    assert_eq!(depth_of(&hits, "a", StorageTier::Device), 0);
    assert_eq!(depth_of(&hits, "a", StorageTier::HostPinned), 1);
    assert_eq!(depth_of(&hits, "b", StorageTier::Device), 1);
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
        &[(seq, h("x")), (h("tail"), h("y"))],
    );
    assert_eq!(
        index.with_snapshot(|v| v.best_sparse_tier(&owner, None, 0, &hash)),
        Some(StorageTier::Device)
    );
    index.with_batch(|batch| {
        assert!(batch.store_sparse(&owner, None, 0, &hash, StorageTier::HostPinned, 4));
    });
    index.with_batch(|batch| batch.remove_sparse(&owner, 0, &hash, Some(StorageTier::Device)));
    assert!(index
        .find_matches(&query(&["x"], &[StorageTier::Device]))
        .is_empty());
    // The host fact retains the navigation needed by a device reinsert.
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
    index.with_batch(|batch| batch.remove_sparse(&owner, 0, &hash, Some(StorageTier::HostPinned)));
    assert_eq!(
        segments_of(
            &index.find_matches(&query(&["x", "y"], &[StorageTier::Device])),
            "a",
            StorageTier::Device
        ),
        [(1, 2)]
    );
    index.remove(0, &owner, StorageTier::Device, &[h("tail")]);
    assert!(index.shard(0).scoped_anchors.is_empty());
    index.with_batch(|batch| {
        assert!(batch.store_sparse(&owner, None, 0, &hash, StorageTier::Device, 4));
    });
    // A hash-only fact cannot reconstruct a reclaimed local-token path.
    assert_eq!(
        index.with_snapshot(|v| v.best_sparse_tier(&owner, None, 0, &hash)),
        Some(StorageTier::Device)
    );
    assert!(index
        .find_matches(&query(&["x"], &[StorageTier::Device]))
        .is_empty());
    index.store(
        0,
        owner,
        StorageTier::Device,
        None,
        &[(sparse_key(None, &hash), h("x"))],
    );
    assert_eq!(
        depth_of(
            &index.find_matches(&query(&["x"], &[StorageTier::Device])),
            "a",
            StorageTier::Device
        ),
        1
    );
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
            .tier_sequences(&w2, StorageTier::Device)
            .contains(&parent));
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
        b.clone(),
        StorageTier::Device,
        None,
        &[
            (qualified_seq("adapter", "i:3"), h("x")),
            (qualified_seq("adapter", "i:3"), h("y")),
        ],
    );
    assert!(index
        .find_matches(&query(&["x", "y"], &[StorageTier::Device]))
        .is_empty());
    assert!(!index
        .source_lock(&b.target())
        .access
        .read()
        .records
        .owners
        .contains_key(&b));
}

#[test]
fn clear_sparse_owner_preserves_other_owner_and_resets_capacity() {
    let index = KvBlockIndexer::new();
    let a = worker("clear-a");
    let b = worker("clear-b");
    let hash = [5; 32];
    index.with_batch(|batch| {
        for tier in [StorageTier::Device, StorageTier::HostPinned] {
            assert!(batch.store_sparse(&a, None, 0, &hash, tier, 1));
            assert!(batch.store_sparse(&b, None, 0, &hash, tier, 1));
        }
        assert!(batch.store_sparse(&a, Some("adapter"), 1, &hash, StorageTier::Disk, 1));
        batch.clear_owner_exact(&a);
        assert!(batch.store_sparse(&a, None, 1, &[6; 32], StorageTier::Device, 1));
    });
    index.with_snapshot(|view| {
        assert_eq!(view.best_sparse_tier(&a, None, 0, &hash), None);
        assert_eq!(view.best_sparse_tier(&a, Some("adapter"), 1, &hash), None);
        assert_eq!(
            view.best_sparse_tier(&b, None, 0, &hash),
            Some(StorageTier::Device)
        );
        assert_eq!(
            view.best_sparse_tier(&a, None, 1, &[6; 32]),
            Some(StorageTier::Device)
        );
    });
}

#[test]
fn sparse_tier_lookup_falls_back_as_resident_tiers_are_removed() {
    let index = KvBlockIndexer::new();
    let owner = worker("tier-fallback");
    let hash = [23; 32];
    let best = || index.with_snapshot(|view| view.best_sparse_tier(&owner, None, 0, &hash));

    assert_eq!(best(), None);
    index.with_batch(|batch| {
        for tier in TIERS.into_iter().rev() {
            assert!(batch.store_sparse(&owner, None, 0, &hash, tier, 1));
        }
    });
    // Removing a tier on a missing page must leave the observed page alone.
    index.with_batch(|batch| batch.remove_sparse(&owner, 0, &[24; 32], Some(TIERS[0])));
    for tier in TIERS {
        assert_eq!(best(), Some(tier));
        index.with_batch(|batch| {
            batch.remove_sparse(&owner, 0, &hash, Some(tier));
            batch.remove_sparse(&owner, 0, &hash, Some(tier));
        });
    }
    assert_eq!(best(), None);
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
    for namespace in [None, Some("adapter")] {
        let index = KvBlockIndexer::new();
        let owner = worker("a");
        index.with_batch(|tx| {
            assert!(tx.store_sparse(&owner, namespace, 0, &[1; 32], StorageTier::Device, 1))
        });
        index.with_batch(|tx| {
            assert!(tx.store_sparse(&owner, namespace, 0, &[1; 32], StorageTier::HostPinned, 1))
        });
        index.with_batch(|tx| {
            tx.store_pages(
                0,
                &owner,
                StorageTier::Device,
                PageBatch::Hashes {
                    namespace,
                    hashes: &[[1; 32]; 2],
                },
                true,
                1,
            )
            .unwrap();
            tx.remove_pages(
                0,
                &owner,
                Some(StorageTier::Device),
                RemovedPages::Hashes(&[[1; 32], [255; 32], [1; 32]]),
                true,
            );
            assert!(!tx.store_sparse(&owner, namespace, 1, &[2; 32], StorageTier::Device, 1));
        });
        assert_eq!(
            index.with_snapshot(|view| view.best_sparse_tier(&owner, namespace, 0, &[1; 32])),
            Some(StorageTier::HostPinned)
        );
        index.with_batch(|tx| tx.remove_sparse(&owner, 0, &[1; 32], Some(StorageTier::Device)));
        index.with_batch(|tx| {
            assert!(!tx.store_sparse(&owner, namespace, 1, &[2; 32], StorageTier::Device, 1))
        });
        index.with_batch(|tx| {
            tx.remove_pages(
                0,
                &owner,
                Some(StorageTier::HostPinned),
                RemovedPages::Hashes(&[[1; 32], [255; 32], [1; 32]]),
                false,
            )
        });
        index.with_batch(|tx| {
            assert!(tx.store_sparse(&owner, namespace, 1, &[2; 32], StorageTier::Device, 1))
        });
        index.with_batch(|tx| tx.clear_owner_exact(&owner));
        index.with_batch(|tx| {
            assert!(tx.store_sparse(&owner, namespace, 0, &[1; 32], StorageTier::Device, 1))
        });
    }
}

#[test]
fn retired_streams_are_removed_without_reusing_snapshot_generations() {
    let index = KvBlockIndexer::new();
    let owner = worker("a");
    let target = owner.target();
    let control = Arc::new(StreamControl::default());
    let mut previous = 0;
    for _ in 0..8 {
        let generation = index.with_stream_batch(&target, &owner, &control, |state, _| {
            state.reset(Some(0), true);
            state.reset_generation
        });
        assert_ne!(generation, previous);
        previous = generation;
        index.with_stream_batch(&target, &owner, &control, |state, _| {
            state.reset(None, false)
        });
        assert!(index.source_lock(&target).access.read().streams.is_empty());
    }
}

#[test]
fn independent_sources_update_sparse_capacity_during_dense_edits() {
    use std::sync::{mpsc, Barrier};
    use std::time::Duration;

    let index = Arc::new(KvBlockIndexer::new());
    let barrier = Arc::new(Barrier::new(7));
    let (done, finished) = mpsc::channel();
    let mut threads = Vec::new();
    for id in 0..6 {
        let index = index.clone();
        let barrier = barrier.clone();
        let done = done.clone();
        threads.push(std::thread::spawn(move || {
            let owner = worker(&format!("source-{id}"));
            let target = owner.target();
            let hash = [id as u8; 32];
            barrier.wait();
            for _ in 0..250 {
                index.with_source_batch(&target, |tx| {
                    assert!(tx.store_sparse(&owner, None, 0, &hash, StorageTier::Device, 1));
                    assert!(tx.store_sparse(&owner, None, 0, &hash, StorageTier::HostPinned, 1));
                    assert!(!tx.store_sparse(&owner, None, 1, &[255; 32], StorageTier::Device, 1));
                    tx.remove_sparse(&owner, 0, &hash, Some(StorageTier::Device));
                    tx.remove_sparse(&owner, 0, &hash, Some(StorageTier::HostPinned));
                });
            }
            done.send(()).unwrap();
        }));
    }
    {
        let index = index.clone();
        let barrier = barrier.clone();
        let done = done.clone();
        threads.push(std::thread::spawn(move || {
            let owner = worker("dense");
            let sequence = h("dense-sequence");
            barrier.wait();
            for _ in 0..250 {
                index.store(
                    0,
                    owner.clone(),
                    StorageTier::Device,
                    None,
                    &[(sequence.clone(), h("local"))],
                );
                index.remove(
                    0,
                    &owner,
                    StorageTier::Device,
                    std::slice::from_ref(&sequence),
                );
            }
            done.send(()).unwrap();
        }));
    }
    for _ in 0..7 {
        finished
            .recv_timeout(Duration::from_secs(15))
            .expect("concurrent writers stalled");
    }
    for thread in threads {
        thread.join().unwrap();
    }
    for id in 0..6 {
        let owner = worker(&format!("source-{id}"));
        index.with_snapshot(|tx| {
            assert_eq!(tx.best_sparse_tier(&owner, None, 0, &[id as u8; 32]), None)
        });
    }
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
    for _ in 0..3 {
        idx.resume_source(&source, 0);
    }
    idx.suspend_source(&source, 0);
    for group in [0, 1] {
        let hits = idx.find_matches(&query_group(group, &["x"], &tiers));
        assert_eq!(hits.len(), 2);
        assert!(hits
            .iter()
            .all(|hit| hit.target.instance_id != source || hit.target.dp_rank != 0));
    }
    // Complete replay may resume credit without forgetting still-valid claims.
    idx.resume_source(&source, 0);
    assert_eq!(idx.find_matches(&query(&["x"], &tiers)).len(), 5);
    idx.resume_source(&source, 0);
    idx.invalidate_source(&source, 0);
    idx.resume_source(&source, 0);
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
fn source_batches_publish_atomically_to_dense_queries_and_group_snapshots() {
    use std::sync::mpsc;
    use std::time::Duration;

    for dense in [false, true] {
        let index = Arc::new(KvBlockIndexer::new());
        let owner = worker("a");
        if dense {
            index.store(
                0,
                owner.clone(),
                StorageTier::Device,
                None,
                &[(h("s1"), h("x")), (h("s2"), h("y"))],
            );
        }
        let (entered, ready) = mpsc::channel();
        let (release, resume) = mpsc::channel();
        let writer_index = index.clone();
        let writer_owner = owner.clone();
        let writer = std::thread::spawn(move || {
            writer_index.with_source_batch(&writer_owner.target(), |batch| {
                if dense {
                    batch.remove(0, &writer_owner, StorageTier::Device, &[h("s1")]);
                } else {
                    assert!(batch.store_sparse(
                        &writer_owner,
                        None,
                        0,
                        &[1; 32],
                        StorageTier::Device,
                        8
                    ));
                }
                entered.send(()).unwrap();
                resume.recv_timeout(Duration::from_secs(5)).unwrap();
                if dense {
                    batch.remove(0, &writer_owner, StorageTier::Device, &[h("s2")]);
                } else {
                    assert!(batch.store_sparse(
                        &writer_owner,
                        None,
                        1,
                        &[1; 32],
                        StorageTier::Device,
                        8
                    ));
                }
            });
        });
        ready.recv_timeout(Duration::from_secs(5)).unwrap();
        let (started, scheduled) = mpsc::channel();
        let (done, result) = mpsc::channel();
        let reader_index = index.clone();
        let reader = std::thread::spawn(move || {
            started.send(()).unwrap();
            let complete = if dense {
                reader_index
                    .find_matches(&query(&["x", "y"], &[StorageTier::Device]))
                    .is_empty()
            } else {
                reader_index.with_source_snapshot(&owner.target(), |view| {
                    [0, 1].into_iter().all(|group| {
                        view.best_sparse_tier(&owner, None, group, &[1; 32])
                            == Some(StorageTier::Device)
                    })
                })
            };
            done.send(complete).unwrap();
        });
        scheduled.recv_timeout(Duration::from_secs(5)).unwrap();
        let premature = result.recv_timeout(Duration::from_millis(100));
        release.send(()).unwrap();
        writer.join().unwrap();
        reader.join().unwrap();
        assert!(
            matches!(premature, Err(mpsc::RecvTimeoutError::Timeout)),
            "a query observed an incomplete source transaction (dense={dense})"
        );
        assert!(result.recv_timeout(Duration::from_secs(5)).unwrap());
    }
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
    let actual: HashSet<_> = shard.tier_sequences(&key.0, key.1).into_iter().collect();
    assert_eq!(actual, expected.iter().map(|seq| h(seq)).collect());
    assert_eq!(actual.len(), expected.len());
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
    let request = query(&["local"], &[StorageTier::Device]);
    assert_eq!(
        depth_of(&idx.find_matches(&request), "w0", StorageTier::Device),
        1
    );

    idx.remove(
        0,
        &owner,
        StorageTier::Device,
        &[h("i:42"), h("i:42"), h("i:404")],
    );
    assert_observations(&shard, &key, &["i:900"]);
    assert_eq!(
        depth_of(&idx.find_matches(&request), "w0", StorageTier::Device),
        1
    );
    idx.remove(0, &owner, StorageTier::Device, &[h("i:900"), h("i:900")]);
    assert_observations(&shard, &key, &[]);
    assert!(idx.find_matches(&request).is_empty());
    assert!(shard.scoped_anchors.is_empty());
    assert!(shard.tier_sequences(&key.0, key.1).is_empty());
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
    assert_eq!(
        depth_of(
            &idx.find_matches(&query(&["x", "y", "branch"], &[StorageTier::Device])),
            "b",
            StorageTier::Device
        ),
        3
    );
    idx.clear(&b, ClearScope::Worker);
    idx.clear(&b, ClearScope::Worker);
    assert_observations(&shard, &ka, &["i:100", "i:101", "i:102"]);
    assert_observations(&shard, &kb, &[]);
    for (locals, depth) in [(["x", "y", "z"], 3), (["x", "y", "branch"], 2)] {
        let hits = idx.find_matches(&query(&locals, &[StorageTier::Device]));
        assert_eq!(depth_of(&hits, "a", StorageTier::Device), depth);
        assert!(segments_of(&hits, "b", StorageTier::Device).is_empty());
    }
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
            let name = format!("writer-{writer}");
            let owner = worker(&name);
            let branch_local = format!("branch-{writer}");
            let seqs: Vec<Arc<str>> = (0..4)
                .map(|block| h(&format!("i:{}", 100 + writer * 10 + block)))
                .collect();
            let path = [
                (seqs[0].clone(), h("x")),
                (seqs[1].clone(), h("y")),
                (seqs[2].clone(), h("z")),
            ];
            let branch = [(seqs[3].clone(), h(&branch_local))];
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
                for tail in ["z", branch_local.as_str()] {
                    assert_eq!(
                        depth_of(
                            &idx.find_matches(&query(&["x", "y", tail], &[StorageTier::Device])),
                            &name,
                            StorageTier::Device
                        ),
                        3
                    );
                }
                idx.remove(0, &owner, StorageTier::Device, &seqs);
                assert_eq!(
                    depth_of(
                        &idx.find_matches(&query(&["x", "y", "z"], &[StorageTier::Device])),
                        &name,
                        StorageTier::Device
                    ),
                    0
                );
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
fn late_child_waits_for_reclaimed_parent_and_reconnects_without_new_store() {
    let idx = KvBlockIndexer::new();
    let owner = worker("w0");
    idx.store(
        0,
        owner.clone(),
        StorageTier::Device,
        None,
        &[(h("sa"), h("a"))],
    );
    idx.remove(0, &owner, StorageTier::Device, &[h("sa")]);
    idx.store(
        0,
        owner.clone(),
        StorageTier::Device,
        Some(h("sa")),
        &[(h("sb"), h("b"))],
    );
    let hits = idx.find_matches(&query(&["a", "b"], &[StorageTier::Device]));
    assert!(
        hits.is_empty(),
        "a missing parent cannot prove a reusable prefix"
    );
    idx.store(0, owner, StorageTier::Device, None, &[(h("sa"), h("a"))]);
    let hits = idx.find_matches(&query(&["a", "b"], &[StorageTier::Device]));
    assert_eq!(segments_of(&hits, "w0", StorageTier::Device), vec![(0, 2)]);
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
        .tier_sequences(&pool("poolA", "w0"), StorageTier::External)
        .is_empty());
    assert_eq!(
        shard
            .tier_sequences(&pool("poolB", "w0"), StorageTier::External)
            .len(),
        1
    );
    assert_eq!(
        shard
            .tier_sequences(&worker("w0"), StorageTier::Device)
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
        assert!(!shard.root.read().children.contains_key("w"));
        assert_eq!(
            shard
                .tier_sequences(&worker("w0"), StorageTier::Device)
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
fn aggregation_deduplicates_overlapping_owners_across_gaps() {
    let remote = (pool("poolA", "w0"), StorageTier::Device);
    let local = (worker("w0"), StorageTier::Device);
    let other_tier = (worker("w0"), StorageTier::HostPinned);
    let claims: Vec<_> = vec![
        vec![remote.clone(), local.clone(), other_tier.clone()],
        vec![],
        vec![remote.clone(), local.clone()],
        vec![local, remote, other_tier],
    ]
    .into_iter()
    .enumerate()
    .map(|(i, owners)| prefix_tree::ClaimSpan {
        end: i + 1,
        owners: Arc::new(owners),
    })
    .collect();
    let res = GroupShard::aggregate_claims(
        &query(
            &["h0", "h1", "h2", "h3"],
            &[StorageTier::Device, StorageTier::HostPinned],
        ),
        &claims,
    );
    assert_eq!(
        segments_of(&res, "w0", StorageTier::Device),
        [(0, 1), (2, 4)]
    );
    assert_eq!(
        segments_of(&res, "w0", StorageTier::HostPinned),
        [(0, 1), (3, 4)]
    );
    let device = res
        .iter()
        .find(|hit| hit.tier == StorageTier::Device)
        .unwrap();
    assert_eq!(device.locality, Locality::Local);
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
fn anchored_store_restores_existing_pages_without_topology_write_lock() {
    use std::sync::mpsc;
    use std::time::Duration;

    let index = Arc::new(KvBlockIndexer::new());
    let owner = worker("a");
    let blocks = [(h("s1"), h("x")), (h("s2"), h("y")), (h("s3"), h("z"))];
    index.store(0, owner.clone(), StorageTier::Device, None, &blocks);
    index.remove(0, &owner, StorageTier::Device, &[h("s2")]);
    let shard = index.shard(0);
    let topology = shard.topology.read();
    let (tx, rx) = mpsc::channel();
    let writer_index = index.clone();
    let writer_owner = owner.clone();
    let writer = std::thread::spawn(move || {
        writer_index.store(
            0,
            writer_owner,
            StorageTier::Device,
            Some(h("s1")),
            &[(h("s2"), h("y"))],
        );
        tx.send(()).unwrap();
    });
    let completed = rx.recv_timeout(Duration::from_secs(5)).is_ok();
    drop(topology);
    writer.join().unwrap();
    assert!(
        completed,
        "existing anchored Store waited for exclusive topology"
    );
    let hits = index.find_matches(&query(&["x", "y", "z"], &[StorageTier::Device]));
    assert_eq!(depth_of(&hits, "a", StorageTier::Device), 3);
}

#[test]
fn anchored_store_rejects_mismatch_and_duplicate_without_mutation() {
    let index = KvBlockIndexer::new();
    let owner = worker("a");
    index.store(
        0,
        owner.clone(),
        StorageTier::Device,
        None,
        &[(h("s1"), h("x")), (h("s2"), h("y"))],
    );
    index.remove(0, &owner, StorageTier::Device, &[h("s2")]);
    let shard = index.shard(0);
    let _topology = shard.topology.read();
    assert!(!shard.try_store_anchored(
        &StoreClaim {
            key: (owner.clone(), StorageTier::Device),
            incremental: false
        },
        Some(&h("s1")),
        &[(h("s2"), h("wrong"))]
    ));
    assert!(!shard.try_store_anchored(
        &StoreClaim {
            key: (owner.clone(), StorageTier::Device),
            incremental: false
        },
        Some(&h("s1")),
        &[(h("s1"), h("y"))]
    ));
    assert!(!shard.try_store_anchored(
        &StoreClaim {
            key: (owner.clone(), StorageTier::Device),
            incremental: false
        },
        Some(&h("s1")),
        &[(h("new"), h("y"))]
    ));
    assert!(!shard
        .tier_sequences(&owner, StorageTier::Device)
        .contains(&h("s2")));
}

#[test]
fn concurrent_anchored_stores_and_queries_keep_per_source_prefixes() {
    use std::sync::{mpsc, Barrier};
    use std::time::Duration;

    let index = Arc::new(KvBlockIndexer::new());
    for name in ["a", "b"] {
        index.store(
            0,
            worker(name),
            StorageTier::Device,
            None,
            &[
                (h(&format!("{name}-1")), h("x")),
                (h(&format!("{name}-2")), h("y")),
            ],
        );
    }
    // Force the existing-position read-lock path for both publishers.
    let shard = index.shard(0);
    let topology = shard.topology.read();
    let barrier = Arc::new(Barrier::new(4));
    let (done, completed) = mpsc::channel();
    let mut threads = Vec::new();
    for name in ["a", "b"] {
        let (index, barrier, done) = (index.clone(), barrier.clone(), done.clone());
        threads.push(std::thread::spawn(move || {
            let owner = worker(name);
            let parent = h(&format!("{name}-1"));
            let last = h(&format!("{name}-2"));
            barrier.wait();
            for _ in 0..400 {
                index.remove(0, &owner, StorageTier::Device, std::slice::from_ref(&last));
                index.store(
                    0,
                    owner.clone(),
                    StorageTier::Device,
                    Some(parent.clone()),
                    &[(last.clone(), h("y"))],
                );
            }
            done.send(()).unwrap();
        }));
    }
    for _ in 0..2 {
        let (index, barrier, done) = (index.clone(), barrier.clone(), done.clone());
        threads.push(std::thread::spawn(move || {
            barrier.wait();
            for _ in 0..800 {
                let hits = index.find_matches(&query(&["x", "y"], &[StorageTier::Device]));
                for name in ["a", "b"] {
                    assert!([1, 2].contains(&depth_of(&hits, name, StorageTier::Device)));
                }
            }
            done.send(()).unwrap();
        }));
    }
    drop(done);
    let mut finished = true;
    for _ in 0..4 {
        if completed.recv_timeout(Duration::from_secs(20)).is_err() {
            finished = false;
            break;
        }
    }
    drop(topology);
    assert!(finished, "anchored writer or query failed to finish");
    for thread in threads {
        thread.join().unwrap();
    }
    let hits = index.find_matches(&query(&["x", "y"], &[StorageTier::Device]));
    assert_eq!(depth_of(&hits, "a", StorageTier::Device), 2);
    assert_eq!(depth_of(&hits, "b", StorageTier::Device), 2);
}

#[test]
fn shared_alias_rosters_preserve_source_isolation_during_concurrent_reinsertion() {
    use std::sync::{mpsc, Barrier};
    use std::time::Duration;

    const SOURCES: usize = 4;
    const PAGES: usize = 16;
    let index = Arc::new(KvBlockIndexer::new());
    let blocks: Vec<_> = (0..PAGES)
        .map(|i| (h(&format!("seq-{i}")), h(&format!("local-{i}"))))
        .collect();
    let request = MatchQuery {
        group_idx: 0,
        local_hashes: blocks.iter().map(|b| b.1.clone()).collect(),
        tiers_of_interest: vec![StorageTier::Device, StorageTier::HostPinned],
    };
    // Keep navigation live while every GPU publisher detaches and rejoins
    // the shared roster. This source must never gain or lose GPU residency.
    index.store(0, worker("host"), StorageTier::HostPinned, None, &blocks);
    for id in 0..SOURCES {
        index.store(
            0,
            worker(&format!("source-{id}")),
            StorageTier::Device,
            None,
            &blocks,
        );
    }
    let barrier = Arc::new(Barrier::new(SOURCES + 1));
    let (done, finished) = mpsc::channel();
    let mut threads = Vec::new();
    for id in 0..SOURCES {
        let (index, barrier, done, blocks, request) = (
            index.clone(),
            barrier.clone(),
            done.clone(),
            blocks.clone(),
            request.clone(),
        );
        threads.push(std::thread::spawn(move || {
            let name = format!("source-{id}");
            let owner = worker(&name);
            let seqs: Vec<_> = blocks.iter().map(|b| b.0.clone()).collect();
            barrier.wait();
            for _ in 0..100 {
                index.remove(0, &owner, StorageTier::Device, &seqs);
                assert_eq!(
                    depth_of(&index.find_matches(&request), &name, StorageTier::Device),
                    0
                );
                index.store(0, owner.clone(), StorageTier::Device, None, &blocks);
                assert_eq!(
                    depth_of(&index.find_matches(&request), &name, StorageTier::Device),
                    PAGES as u32
                );
            }
            done.send(()).unwrap();
        }));
    }
    {
        let (index, barrier, done, request) = (
            index.clone(),
            barrier.clone(),
            done.clone(),
            request.clone(),
        );
        threads.push(std::thread::spawn(move || {
            barrier.wait();
            for _ in 0..400 {
                let hits = index.find_matches(&request);
                assert_eq!(
                    depth_of(&hits, "host", StorageTier::HostPinned),
                    PAGES as u32
                );
                assert_eq!(depth_of(&hits, "host", StorageTier::Device), 0);
                for id in 0..SOURCES {
                    let name = format!("source-{id}");
                    let depth = depth_of(&hits, &name, StorageTier::Device);
                    assert!(
                        depth == 0 || depth == PAGES as u32,
                        "partial source snapshot: {depth}"
                    );
                    assert_eq!(depth_of(&hits, &name, StorageTier::HostPinned), 0);
                }
            }
            done.send(()).unwrap();
        }));
    }
    for _ in 0..=SOURCES {
        finished
            .recv_timeout(Duration::from_secs(20))
            .expect("shared roster batch stalled");
    }
    for thread in threads {
        thread.join().unwrap();
    }
    let hits = index.find_matches(&request);
    for id in 0..SOURCES {
        assert_eq!(
            depth_of(&hits, &format!("source-{id}"), StorageTier::Device),
            PAGES as u32
        );
    }
}

#[test]
fn same_edge_batched_removals_are_atomic_across_sources_and_tiers() {
    use std::sync::{mpsc, Barrier};
    use std::time::Duration;

    const SOURCES: usize = 4;
    const PAGES: usize = 16;
    let index = Arc::new(KvBlockIndexer::new());
    let locals: Vec<_> = (0..PAGES).map(|i| h(&format!("local-{i}"))).collect();
    let blocks: Vec<Vec<_>> = (0..SOURCES)
        .map(|source| {
            (0..PAGES)
                .map(|i| (h(&format!("seq-{source}-{i}")), locals[i].clone()))
                .collect()
        })
        .collect();
    for (id, blocks) in blocks.iter().enumerate() {
        for tier in [StorageTier::Device, StorageTier::HostPinned] {
            index.store(0, worker(&format!("source-{id}")), tier, None, blocks);
        }
    }
    let barrier = Arc::new(Barrier::new(SOURCES + 1));
    let (done, finished) = mpsc::channel();
    let mut threads = Vec::new();
    for (id, pages) in blocks.into_iter().enumerate() {
        let (index, barrier, done) = (index.clone(), barrier.clone(), done.clone());
        threads.push(std::thread::spawn(move || {
            let owner = worker(&format!("source-{id}"));
            let seqs: Vec<_> = pages.iter().map(|(seq, _)| seq.clone()).collect();
            barrier.wait();
            for _ in 0..200 {
                index.remove(0, &owner, StorageTier::Device, &seqs);
                index.store(0, owner.clone(), StorageTier::Device, None, &pages);
            }
            done.send(()).unwrap();
        }));
    }
    {
        let (index, barrier, done) = (index.clone(), barrier.clone(), done.clone());
        threads.push(std::thread::spawn(move || {
            let request = MatchQuery {
                group_idx: 0,
                local_hashes: locals,
                tiers_of_interest: vec![StorageTier::Device, StorageTier::HostPinned],
            };
            barrier.wait();
            for _ in 0..500 {
                let hits = index.find_matches(&request);
                for id in 0..SOURCES {
                    let source = format!("source-{id}");
                    assert_eq!(
                        depth_of(&hits, &source, StorageTier::HostPinned),
                        PAGES as u32
                    );
                    let depth = depth_of(&hits, &source, StorageTier::Device);
                    assert!(
                        depth == 0 || depth == PAGES as u32,
                        "partial source snapshot: {depth}"
                    );
                }
            }
            done.send(()).unwrap();
        }));
    }
    for _ in 0..=SOURCES {
        finished
            .recv_timeout(Duration::from_secs(20))
            .expect("concurrent batch stalled");
    }
    for thread in threads {
        thread.join().unwrap();
    }
}

#[test]
fn batched_remove_crosses_split_edge_without_harming_other_sources_or_tiers() {
    let index = KvBlockIndexer::new();
    let a = worker("a");
    let b = worker("b");
    let blocks: Vec<_> = (0..256)
        .map(|i| (h(&format!("seq-{i}")), h(&format!("local-{i}"))))
        .collect();
    index.store(0, a.clone(), StorageTier::Device, None, &blocks);
    index.store(0, a.clone(), StorageTier::HostPinned, None, &blocks);
    index.store(0, b.clone(), StorageTier::Device, None, &blocks);
    let branch: Vec<_> = (0..32)
        .map(|i| {
            (
                h(&format!("branch-seq-{i}")),
                h(&format!("branch-local-{i}")),
            )
        })
        .collect();
    index.store(
        0,
        a.clone(),
        StorageTier::Device,
        Some(blocks[223].0.clone()),
        &branch,
    );
    let hashes: Vec<_> = blocks[192..].iter().map(|(seq, _)| seq.clone()).collect();
    index.remove(0, &a, StorageTier::Device, &hashes);
    let request = MatchQuery {
        group_idx: 0,
        local_hashes: blocks.iter().map(|(_, local)| local.clone()).collect(),
        tiers_of_interest: vec![StorageTier::Device, StorageTier::HostPinned],
    };
    let hits = index.find_matches(&request);
    assert_eq!(depth_of(&hits, "a", StorageTier::Device), 192);
    assert_eq!(depth_of(&hits, "a", StorageTier::HostPinned), 256);
    assert_eq!(depth_of(&hits, "b", StorageTier::Device), 256);
}

#[test]
fn duplicate_filter_handles_both_sides_of_small_batch_threshold() {
    for count in [1, 2, 7, 8, 9, 16] {
        let index = KvBlockIndexer::new();
        let owner = worker("threshold");
        let target = owner.target();
        let parent = h("i:parent");
        index.store(
            0,
            owner,
            StorageTier::Device,
            None,
            &[(parent.clone(), h("root"))],
        );
        let shard = index.shard(0);
        let anchor = shard.resolve_parent(&target, &parent).unwrap();
        assert!(anchor.is_some());
        let blocks: Vec<_> = (0..count)
            .map(|i| (h(&format!("i:{i}")), h(&format!("local-{i}"))))
            .collect();
        assert!(
            shard.store_path_is_consistent(&target, anchor.as_ref(), Some(&parent), &blocks),
            "unique batch {count}"
        );
        for pos in 0..count {
            let mut duplicate_parent = blocks.clone();
            duplicate_parent[pos].0 = parent.clone();
            assert!(
                !shard.store_path_is_consistent(
                    &target,
                    anchor.as_ref(),
                    Some(&parent),
                    &duplicate_parent
                ),
                "parent duplicate count={count} pos={pos}"
            );
        }
        for first in 0..count {
            for last in first + 1..count {
                let mut duplicate = blocks.clone();
                duplicate[last].0 = duplicate[first].0.clone();
                assert!(
                    !shard.store_path_is_consistent(
                        &target,
                        anchor.as_ref(),
                        Some(&parent),
                        &duplicate
                    ),
                    "batch duplicate count={count} first={first} last={last}"
                );
            }
        }
    }
}

#[test]
fn prefix_churn_reclaims_retired_aliases_and_keeps_shared_live_paths() {
    for keep_shared_prefix in [false, true] {
        let index = KvBlockIndexer::new();
        let owner = worker("churn");
        if keep_shared_prefix {
            index.store(
                0,
                worker("seed"),
                StorageTier::Device,
                None,
                &[(h("seed"), h("local"))],
            );
        }
        let shard = index.shard(0);
        for i in 0..200 {
            let seq = h(&format!("seq-{i}"));
            index.store(
                0,
                owner.clone(),
                StorageTier::Device,
                None,
                &[(seq.clone(), h("local"))],
            );
            index.remove(0, &owner, StorageTier::Device, &[seq]);
            assert_eq!(shard.scoped_anchors.len(), usize::from(keep_shared_prefix));
            assert!(shard.waiting.is_empty());
            assert_eq!(
                shard.root.read().children.len(),
                usize::from(keep_shared_prefix)
            );
        }
        let hits = index.find_matches(&query(&["local"], &[StorageTier::Device]));
        assert_eq!(
            depth_of(&hits, "seed", StorageTier::Device),
            u32::from(keep_shared_prefix)
        );
        assert_eq!(depth_of(&hits, "churn", StorageTier::Device), 0);
    }
}

#[test]
fn pending_descendants_keep_only_required_ancestors_and_all_tiers() {
    let index = KvBlockIndexer::new();
    let owner = worker("pending");
    index.store(
        0,
        owner.clone(),
        StorageTier::Device,
        Some(h("a")),
        &[(h("b"), h("B")), (h("c"), h("C"))],
    );
    index.store(
        0,
        owner.clone(),
        StorageTier::HostPinned,
        Some(h("b")),
        &[(h("c"), h("C"))],
    );
    index.remove(0, &owner, StorageTier::Device, &[h("b")]);
    index.store(
        0,
        owner.clone(),
        StorageTier::Device,
        None,
        &[(h("a"), h("A"))],
    );
    let hits = index.find_matches(&query(
        &["A", "B", "C"],
        &[StorageTier::Device, StorageTier::HostPinned],
    ));
    assert_eq!(
        segments_of(&hits, "pending", StorageTier::Device),
        vec![(0, 1), (2, 3)]
    );
    assert_eq!(
        segments_of(&hits, "pending", StorageTier::HostPinned),
        vec![(2, 3)]
    );
    index.remove(0, &owner, StorageTier::Device, &[h("a"), h("c")]);
    assert!(!index.shard(0).scoped_anchors.is_empty());
    index.remove(0, &owner, StorageTier::HostPinned, &[h("c")]);
    assert!(index.shard(0).scoped_anchors.is_empty());
    assert!(index.shard(0).waiting.is_empty());
    assert!(index.shard(0).root.read().children.is_empty());
}

#[test]
fn removed_pending_block_cannot_reappear_when_parent_is_restored() {
    let index = KvBlockIndexer::new();
    let owner = worker("pending-remove");
    index.store(
        0,
        owner.clone(),
        StorageTier::Device,
        Some(h("a")),
        &[(h("b"), h("B"))],
    );
    index.remove(0, &owner, StorageTier::Device, &[h("b")]);
    assert!(index.shard(0).scoped_anchors.is_empty());
    index.store(0, owner, StorageTier::Device, None, &[(h("a"), h("A"))]);
    let hits = index.find_matches(&query(&["A", "B"], &[StorageTier::Device]));
    assert_eq!(
        segments_of(&hits, "pending-remove", StorageTier::Device),
        vec![(0, 1)]
    );
}

#[test]
fn cyclic_pending_chains_are_rejected_without_credit() {
    let index = KvBlockIndexer::new();
    let owner = worker("cycle");
    index.store(
        0,
        owner.clone(),
        StorageTier::Device,
        Some(h("a")),
        &[(h("b"), h("B"))],
    );
    index.store(
        0,
        owner,
        StorageTier::Device,
        Some(h("b")),
        &[(h("a"), h("A"))],
    );
    assert!(index.shard(0).scoped_anchors.is_empty());
    assert!(index.shard(0).waiting.is_empty());
    assert!(index
        .find_matches(&query(&["A", "B"], &[StorageTier::Device]))
        .is_empty());
}

#[test]
fn reclaim_trims_dead_leaf_suffix_and_preserves_live_fork() {
    let index = KvBlockIndexer::new();
    let owner = worker("trim");
    let blocks: Vec<_> = (0..512)
        .map(|i| (h(&format!("s{i}")), h(&format!("l{i}"))))
        .collect();
    index.store(0, owner.clone(), StorageTier::Device, None, &blocks);
    let tail: Vec<_> = blocks[1..].iter().map(|block| block.0.clone()).collect();
    index.remove(0, &owner, StorageTier::Device, &tail);
    // A topology write drains the bounded deletion debt of this live source.
    index.store(0, owner.clone(), StorageTier::Device, None, &blocks[..1]);
    let shard = index.shard(0);
    let anchor = shard
        .positioned_anchor(&owner.target(), &blocks[0].0)
        .unwrap();
    assert_eq!(anchor.node.read().edge.len(), 1);
    assert_eq!(shard.scoped_anchors.len(), 1);
    index.store(
        0,
        owner.clone(),
        StorageTier::Device,
        Some(blocks[0].0.clone()),
        &[(h("fork"), h("branch"))],
    );
    index.remove(0, &owner, StorageTier::Device, &[blocks[0].0.clone()]);
    assert_eq!(
        shard.scoped_anchors.len(),
        2,
        "the live branch retains its ancestor"
    );
    index.remove(0, &owner, StorageTier::Device, &[h("fork")]);
    assert!(shard.scoped_anchors.is_empty());
    assert!(shard.root.read().children.is_empty());
}

#[test]
fn pool_clear_reclaims_all_publishing_targets_but_keeps_worker_facts() {
    let index = KvBlockIndexer::new();
    for source in ["a", "b"] {
        index.store(
            0,
            pool("shared", source),
            StorageTier::HostPinned,
            None,
            &[(h(source), h(source))],
        );
    }
    index.store(
        0,
        worker("b"),
        StorageTier::Device,
        None,
        &[(h("b"), h("b"))],
    );
    index.clear(&pool("shared", "a"), ClearScope::CacheOwner);
    let shard = index.shard(0);
    assert!(shard
        .positioned_anchor(&worker("a").target(), &h("a"))
        .is_none());
    assert!(shard
        .positioned_anchor(&worker("b").target(), &h("b"))
        .is_some());
    assert_eq!(
        depth_of(
            &index.find_matches(&query(&["b"], &[StorageTier::Device])),
            "b",
            StorageTier::Device
        ),
        1
    );
    index.clear(&worker("b"), ClearScope::Worker);
    assert!(shard.scoped_anchors.is_empty());
    assert!(shard.root.read().children.is_empty());
}

#[test]
fn source_retirement_drops_pending_and_deferred_reclamation() {
    let index = KvBlockIndexer::new();
    let owner = worker("retire");
    index.store(
        0,
        owner.clone(),
        StorageTier::Device,
        None,
        &[(h("a"), h("A"))],
    );
    let shard = index.shard(0);
    let topology = shard.topology.read();
    index.remove(0, &owner, StorageTier::Device, &[h("a")]);
    assert_eq!(
        index
            .source_lock(&owner.target())
            .access
            .read()
            .records
            .reclaim
            .len(),
        1
    );
    drop(topology);
    index.store(
        0,
        owner.clone(),
        StorageTier::Device,
        Some(h("missing")),
        &[(h("child"), h("C"))],
    );
    assert!(!shard.waiting.is_empty());
    index.retire_source(&owner.target().instance_id, 0);
    assert!(shard.waiting.is_empty());
    assert!(!index.source_locks.contains_key(&owner.target()));
    assert!(shard.scoped_anchors.is_empty());
    assert!(shard.root.read().children.is_empty());
}

#[test]
fn rooted_batch_attaches_preexisting_pending_facts_from_every_owner_and_tier() {
    let index = KvBlockIndexer::new();
    let owner = worker("same-source");
    let cache = pool("host", "same-source");
    index.store(
        0,
        cache,
        StorageTier::HostPinned,
        Some(h("a")),
        &[(h("b"), h("B"))],
    );
    index.store(
        0,
        owner.clone(),
        StorageTier::Device,
        None,
        &[(h("a"), h("A")), (h("b"), h("B"))],
    );
    let hits = index.find_matches(&query(
        &["A", "B"],
        &[StorageTier::Device, StorageTier::HostPinned],
    ));
    assert_eq!(
        segments_of(&hits, "same-source", StorageTier::Device),
        vec![(0, 2)]
    );
    assert_eq!(
        segments_of(&hits, "same-source", StorageTier::HostPinned),
        vec![(1, 2)]
    );
    assert!(index.shard(0).waiting.is_empty());
    index.remove(0, &owner, StorageTier::Device, &[h("a"), h("b")]);
    assert_eq!(
        segments_of(
            &index.find_matches(&query(&["A", "B"], &[StorageTier::HostPinned])),
            "same-source",
            StorageTier::HostPinned
        ),
        vec![(1, 2)]
    );
}

#[test]
fn sparse_churn_and_unknown_removes_do_not_accumulate_navigation_debt() {
    let index = KvBlockIndexer::new();
    let owner = worker("sparse");
    index.store(
        0,
        worker("dense"),
        StorageTier::Device,
        None,
        &[(h("keep"), h("local"))],
    );
    index.with_batch(|tx| {
        assert!(tx.store_sparse(&owner, None, 0, &[255; 32], StorageTier::Device, 8));
    });
    for n in 0u32..200 {
        let mut hash = [0; 32];
        hash[..4].copy_from_slice(&n.to_le_bytes());
        index.with_batch(|tx| {
            assert!(tx.store_sparse(&owner, None, 0, &hash, StorageTier::Device, 8));
            tx.remove_sparse(&owner, 0, &hash, Some(StorageTier::Device));
        });
        index.remove(
            0,
            &owner,
            StorageTier::Device,
            &[h(&format!("unknown-{n}"))],
        );
    }
    assert!(index
        .source_lock(&owner.target())
        .access
        .read()
        .records
        .reclaim
        .is_empty());
    assert_eq!(index.shard(0).scoped_anchors.len(), 1);

    index.with_batch(|tx| tx.clear_owner_exact(&owner));
    let hash = [254; 32];
    index.store(
        0,
        owner.clone(),
        StorageTier::Device,
        None,
        &[(sparse_key(None, &hash), h("discarded"))],
    );
    let shard = index.shard(0);
    let source = index.source_lock(&owner.target());
    let topology = shard.topology.read();
    index.with_batch(|tx| {
        tx.remove_sparse(&owner, 0, &hash, Some(StorageTier::Device));
        tx.clear_owner_exact(&owner);
    });
    assert!(!shard.reverse.contains_key(&owner));
    assert!(source.access.read().records.reclaim.contains_key(&0));
    drop(topology);

    index.with_batch(|tx| tx.remove_sparse(&owner, 0, &[253; 32], Some(StorageTier::Device)));
    assert!(source.access.read().records.reclaim.is_empty());
    assert_eq!(shard.scoped_anchors.len(), 1);
}
