//! Source binding, residency and lifecycle regression tests.

#![cfg(all(test, feature = "cpu", feature = "ready"))]

use std::sync::Arc;

use super::SourceBinding;
use crate::kv_index::layout::{CacheGroup, GroupRule};
use crate::kv_index::request_hash::{BlockKeyGenerator, SequenceHash};
use crate::kv_index::source::events::validate::event_kind_matches;
use crate::kv_index::source::test_support::{event_batch, event_payload, gpu_remove, stored_block};
use crate::kv_index::wire::ExternalBlockHash;
use crate::kv_index::wire::{BlockRemoved, BlockStored};
use crate::kv_index::SourceError;
use crate::kv_index::{KVEvent, KVEventBatch, RequestHashCache, RoutableTarget};

impl SourceBinding {
    pub(crate) fn reusable_tokens(&self, tokens: &[u32], max_reusable_tokens: usize) -> u64 {
        self.reusable_tokens_in(None, tokens, max_reusable_tokens)
    }

    fn reusable_tokens_in(
        &self,
        namespace: Option<&str>,
        tokens: &[u32],
        max_reusable_tokens: usize,
    ) -> u64 {
        self.credit_parts(
            namespace,
            tokens,
            max_reusable_tokens,
            &mut RequestHashCache::for_single_target(),
        )
        .map_or(0, |(credited, _, _)| credited)
    }
}

fn target() -> RoutableTarget {
    RoutableTarget {
        instance_id: Arc::from("w"),
        dp_rank: 0,
    }
}

#[test]
fn inferred_full_waits_for_root_evidence_after_a_replay_restart() {
    use crate::kv_index::source::test_support::{full_binding, full_store};

    let index = full_binding(target(), 16).require_root_proof();
    let tokens: Vec<_> = (0..33).collect();
    let root = full_store(&tokens[..32], 16, None);
    let mut suffix = root.clone();
    suffix.parent_block_hash = Some(ExternalBlockHash::Bytes(vec![7; 32]));
    let suffix = event_batch(vec![KVEvent::BlockStored(suffix)]);
    let root = event_batch(vec![KVEvent::BlockStored(root)]);

    index.begin_replay(0);
    index.finish_replay(0).unwrap();
    assert_eq!(index.control.generation(), 0);
    index.apply_batch(0, &suffix).unwrap();
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 0);
    index.apply_batch(1, &root).unwrap();
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 32);

    index.begin_replay(0);
    index.apply_batch(0, &suffix).unwrap();
    index.finish_replay(1).unwrap();
    assert_eq!(index.control.generation(), 0);
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 0);
    index.apply_batch(1, &root).unwrap();
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 32);
}

#[test]
fn unknown_medium_does_not_revoke_known_residency() {
    use crate::kv_index::source::test_support::{full_binding, full_store};

    let index = full_binding(target(), 16).require_root_proof();
    index.begin_empty(0);
    let tokens: Vec<_> = (0..33).collect();
    let known = full_store(&tokens[..32], 16, None);
    let mut unknown = known.clone();
    unknown.medium = Some("UNSUPPORTED".into());
    let remove = BlockRemoved {
        group_idx: Some(0),
        block_hashes: unknown.block_hashes.clone(),
        medium: unknown.medium.clone(),
        locality: None,
        ownership: None,
    };
    let batch = event_batch(vec![
        KVEvent::BlockStored(known),
        KVEvent::BlockStored(unknown),
        KVEvent::BlockRemoved(remove),
    ]);
    assert_eq!(index.apply_batch(0, &batch), Ok(()));
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 32);
}

#[test]
fn inferred_full_skips_foreign_and_unconfigured_adapter_shapes() {
    use crate::kv_index::source::test_support::{full_binding, full_store};

    let index = full_binding(target(), 16).require_root_proof();
    index.begin_empty(0);
    let tokens: Vec<_> = (0..33).collect();
    let local = full_store(&tokens[..32], 16, None);
    index
        .apply_batch(0, &event_batch(vec![KVEvent::BlockStored(local.clone())]))
        .unwrap();
    let mut foreign = local.clone();
    foreign.group_idx = Some(1);
    foreign.ownership = Some("offload".into());
    let mut adapter = local;
    adapter.block_size = 8;
    adapter.lora_name = Some("unconfigured".into());
    index
        .apply_batch(
            1,
            &event_batch(vec![
                KVEvent::BlockStored(foreign),
                KVEvent::BlockStored(adapter),
            ]),
        )
        .unwrap();
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 32);
}

#[test]
fn inferred_full_revokes_credit_on_an_incompatible_root_or_event_shape() {
    use crate::kv_index::source::test_support::{full_binding, full_store};

    let tokens: Vec<_> = (0..33).collect();
    let first = full_store(&tokens[..32], 16, None);
    for (case, expected) in [
        ("root", SourceError::RootHashMismatch),
        ("group", SourceError::InvalidBatch),
        ("block size", SourceError::InvalidBatch),
        ("cache kind", SourceError::InvalidBatch),
        ("missing cache kind", SourceError::InvalidBatch),
        ("lookahead", SourceError::InvalidBatch),
    ] {
        let index = full_binding(target(), 16).require_root_proof();
        index.begin_empty(0);
        index
            .apply_batch(0, &event_batch(vec![KVEvent::BlockStored(first.clone())]))
            .unwrap();
        assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 32);

        let mut wrong = first.clone();
        match case {
            "root" => {
                let different_root = BlockKeyGenerator::new(16, 42);
                wrong.block_hashes[0] = ExternalBlockHash::Bytes(
                    different_root.generate_block_keys(&tokens[..16])[0].to_vec(),
                );
            }
            "group" => wrong.group_idx = Some(1),
            "block size" => wrong.block_size = 8,
            "cache kind" => wrong.kv_cache_spec_kind = Some("mla_attention".into()),
            "missing cache kind" => wrong.kv_cache_spec_kind = None,
            "lookahead" => wrong.lookahead_token_ids = Some(vec![vec![1]; 2]),
            _ => unreachable!(),
        }
        assert_eq!(
            index.apply_batch(1, &event_batch(vec![KVEvent::BlockStored(wrong)])),
            Err(expected),
            "{case}"
        );
        assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 0, "{case}");
    }
}

#[test]
fn full_and_checkpoint_layouts_share_hash_pages_but_preserve_tier_scoring() {
    use crate::kv_index::{KvBlockIndexer, StorageTier};
    use crate::policies::TieredMatchProvider;
    let tokens: Vec<_> = (0..65).collect();
    let hashes = BlockKeyGenerator::new(16, 0).generate_block_keys(&tokens);
    let shared = Arc::new(KvBlockIndexer::new());
    let full_group = CacheGroup {
        group_idx: 0,
        block_size: 16,
        kind: GroupRule::FullAttention,
    };
    let full = SourceBinding::new(target(), vec![full_group], 16, 0, 16)
        .unwrap()
        .with_shared_index(shared.clone());
    let mut other = target();
    other.instance_id = Arc::from("hybrid");
    let hybrid = SourceBinding::new(
        other,
        vec![
            full_group,
            CacheGroup {
                group_idx: 1,
                block_size: 16,
                kind: GroupRule::Checkpoint,
            },
        ],
        16,
        0,
        16,
    )
    .unwrap()
    .with_shared_index(shared);
    full.begin_empty(0);
    hybrid.begin_empty(0);
    let mut initially_empty = crate::kv_index::RequestHashCache::default();
    assert!(hybrid
        .matches_with_cache(&tokens, 16, None, 0, &mut initially_empty)
        .is_empty());
    let gpu = gpu_store(0, "full_attention", 16, &tokens[..32], &hashes[..2]);
    let mut host = gpu_store(0, "full_attention", 16, &tokens[32..64], &hashes[2..]);
    host.medium = Some("CPU".into());
    full.apply_batch(
        0,
        &event_batch(vec![
            KVEvent::BlockStored(gpu.clone()),
            KVEvent::BlockStored(host.clone()),
        ]),
    )
    .unwrap();
    let checkpoint = gpu_store(1, "mamba", 16, &tokens[48..64], &hashes[3..]);
    hybrid
        .apply_batch(
            0,
            &event_batch(vec![
                KVEvent::BlockStored(gpu),
                KVEvent::BlockStored(host),
                KVEvent::BlockStored(checkpoint),
            ]),
        )
        .unwrap();
    for (score_unit, middle, end) in [(16, 2, 4), (32, 1, 2)] {
        let full_hits = full.matches(&tokens, score_unit, None, 0);
        assert_eq!(full_hits.len(), 2);
        assert!(full_hits
            .iter()
            .any(|h| h.tier == StorageTier::Device && h.segments == [(0, middle)]));
        assert!(full_hits
            .iter()
            .any(|h| h.tier == StorageTier::HostPinned && h.segments == [(middle, end)]));
    }
    assert!(full.matches(&tokens, 64, None, 0).is_empty());
    let hybrid_hits = hybrid.matches_with_cache(&tokens, 16, None, 0, &mut initially_empty);
    assert_eq!(hybrid_hits.len(), 1);
    assert_eq!(hybrid_hits[0].tier, StorageTier::HostPinned);
    assert_eq!(hybrid_hits[0].segments, [(0, 4)]);

    // P/D stages can share prompt hashes while scoring different numbers of targets.
    let mut cache = crate::kv_index::RequestHashCache::for_single_target();
    let one = hybrid.matches_with_cache(&tokens, 16, None, 0, &mut cache);
    assert_eq!(one[0].tier, StorageTier::HostPinned);
    assert_eq!(one[0].segments, [(0, 4)]);
    cache.single_target = false;
    // Keep the request's original shared evidence across a committed deletion.
    // The next lookup must refresh the edited source while preserving the
    // other source's evidence, even when both use the same hash chain.
    assert_eq!(
        full.matches_with_cache(&tokens, 16, None, 0, &mut cache)
            .len(),
        2
    );
    let evict_first = event_batch(vec![gpu_remove(
        Some(0),
        vec![ExternalBlockHash::Bytes(hashes[0].to_vec())],
    )]);
    full.apply_batch(1, &evict_first).unwrap();
    cache.single_target = true;
    let hits = full.matches_with_cache(&tokens, 16, None, 0, &mut cache);
    assert!(hits
        .iter()
        .any(|h| h.tier == StorageTier::Device && h.segments == [(1, 2)]));
    let coarse_hits = full.matches_with_cache(&tokens, 32, None, 0, &mut cache);
    assert_eq!(coarse_hits.len(), 1);
    assert_eq!(coarse_hits[0].tier, StorageTier::HostPinned);
    assert_eq!(coarse_hits[0].segments, [(1, 2)]);
    assert_eq!(
        hybrid.matches_with_cache(&tokens, 16, None, 0, &mut cache)[0].segments,
        [(0, 4)]
    );
    hybrid.apply_batch(1, &evict_first).unwrap();
    for single_target in [true, false] {
        let mut shared = crate::kv_index::RequestHashCache::default();
        shared.single_target = single_target;
        assert!(hybrid
            .matches_with_cache(&tokens, 16, None, 0, &mut shared)
            .is_empty());
        let hits = full.matches_with_cache(&tokens, 16, None, 0, &mut shared);
        assert!(hits
            .iter()
            .any(|hit| hit.tier == StorageTier::Device && hit.segments == [(1, 2)]));
        assert!(hits
            .iter()
            .any(|hit| hit.tier == StorageTier::HostPinned && hit.segments == [(2, 4)]));
    }

    let mut scored = crate::kv_index::RequestHashCache::default();
    scored.stop_at_gap = true;
    assert!(full
        .matches_with_cache(&tokens, 16, None, 0, &mut scored)
        .is_empty());
    scored.stop_at_gap = false;
    assert!(full
        .matches_with_cache(&tokens, 16, None, 0, &mut scored)
        .iter()
        .any(|hit| hit.tier == StorageTier::HostPinned && hit.segments == [(2, 4)]));
    full.begin_replay(0);
    assert!(full
        .matches_with_cache(&tokens, 16, None, 0, &mut cache)
        .is_empty());
}

#[test]
fn partial_stores_restore_the_joint_boundary_and_survive_tier_removal() {
    use crate::kv_index::StorageTier;
    use crate::policies::TieredMatchProvider;

    let tokens: Vec<u32> = (0..1201).collect();
    let hashes = BlockKeyGenerator::new(16, 0).generate_block_keys(&tokens);
    let groups = vec![
        CacheGroup {
            group_idx: 0,
            block_size: 576,
            kind: GroupRule::FullAttention,
        },
        CacheGroup {
            group_idx: 1,
            block_size: 576,
            kind: GroupRule::Checkpoint,
        },
    ];
    let mut previous = gpu_store(1, "mamba", 576, &tokens[..576], &hashes[35..36]);
    previous.medium = Some("STORAGE".into());
    let mut partial = gpu_store(1, "mamba", 16, &tokens[1120..1136], &hashes[70..71]);
    partial.medium = Some("CPU".into());
    let mut device_copy = partial.clone();
    device_copy.medium = Some("GPU".into());
    let batch = event_batch(vec![
        KVEvent::BlockStored(gpu_store(
            0,
            "full_attention",
            576,
            &tokens[..576],
            &hashes[35..36],
        )),
        KVEvent::BlockStored(gpu_store(
            0,
            "full_attention",
            16,
            &tokens[1120..1136],
            &hashes[70..71],
        )),
        KVEvent::BlockStored(previous),
        KVEvent::BlockStored(partial),
        KVEvent::BlockStored(device_copy.clone()),
        KVEvent::BlockStored(device_copy),
    ]);
    let default_index = SourceBinding::new(target(), groups.clone(), 16, 0, 8).unwrap();
    default_index.begin_empty(0);
    assert_eq!(default_index.apply_batch(0, &batch), Ok(()));
    assert_eq!(default_index.reusable_tokens(&tokens, usize::MAX), 576);

    let index = SourceBinding::new(target(), groups, 16, 0, 8)
        .unwrap()
        .with_partial_hash_hits(true);
    index.begin_empty(0);
    assert_eq!(index.apply_batch(0, &batch), Ok(()));
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 1136);
    assert_eq!(index.reusable_tokens(&tokens, 1120), 576);
    let matches = index.matches(&tokens, 16, None, 0);
    assert_eq!(matches[0].tier, StorageTier::Device);
    let removal = event_batch(vec![gpu_remove(
        Some(1),
        vec![ExternalBlockHash::Bytes(hashes[70].to_vec())],
    )]);
    assert_eq!(index.apply_batch(1, &removal), Ok(()));
    let matches = index.matches(&tokens, 16, None, 0);
    assert_eq!(matches[0].tier, StorageTier::Device);
    assert_eq!(index.apply_batch(2, &removal), Ok(()));
    let matches = index.matches(&tokens, 16, None, 0);
    assert_eq!(matches[0].tier, StorageTier::HostPinned);
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 1136);
}

#[test]
fn single_full_attention_group_credits_only_published_partial_pages() {
    use crate::policies::TieredMatchProvider;

    let tokens: Vec<u32> = (0..49).collect();
    let hashes = BlockKeyGenerator::new(16, 0).generate_block_keys(&tokens);
    let group = CacheGroup {
        group_idx: 0,
        block_size: 32,
        kind: GroupRule::FullAttention,
    };
    let batch = event_batch(vec![
        KVEvent::BlockStored(gpu_store(
            0,
            "full_attention",
            32,
            &tokens[..32],
            &hashes[1..2],
        )),
        KVEvent::BlockStored(gpu_store(
            0,
            "full_attention",
            16,
            &tokens[32..48],
            &hashes[2..3],
        )),
    ]);
    for (partial, end) in [(false, 2), (true, 3)] {
        let index = SourceBinding::new(target(), vec![group], 16, 0, 8)
            .unwrap()
            .with_partial_hash_hits(partial);
        index.begin_empty(0);
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        let matches = index.matches(&tokens, 16, None, 0);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].segments, [(0, end)]);
        assert_eq!(index.reusable_tokens(&tokens, 32), 32);
        if partial {
            index
                .apply_batch(
                    1,
                    &event_batch(vec![gpu_remove(
                        Some(0),
                        vec![ExternalBlockHash::Bytes(hashes[1].to_vec())],
                    )]),
                )
                .unwrap();
            assert!(index.matches(&tokens, 16, None, 0).is_empty());
        }
    }
}

fn checkpoint_groups(block_size: u32, attention: GroupRule) -> Vec<CacheGroup> {
    (0..4)
        .map(|group_idx| CacheGroup {
            group_idx,
            block_size,
            kind: if group_idx == 3 {
                attention
            } else {
                GroupRule::Checkpoint
            },
        })
        .collect()
}

fn groups() -> Vec<CacheGroup> {
    checkpoint_groups(528, GroupRule::FullAttention)
}

fn prompt() -> Vec<u32> {
    (0..1024).map(|i| 100 + i % 100).collect()
}

/// A GPU store of `hashes` for one group, as the engine publishes it.
fn gpu_store(
    group_idx: u32,
    spec: &str,
    block_size: u32,
    token_ids: &[u32],
    hashes: &[SequenceHash],
) -> BlockStored {
    let mut event = stored_block(
        hashes
            .iter()
            .map(|hash| ExternalBlockHash::Bytes(hash.to_vec()))
            .collect(),
        token_ids.to_vec(),
        block_size,
    );
    event.extra_keys = Some(vec![rmpv::Value::Nil; hashes.len()]);
    event.group_idx = Some(group_idx);
    event.kv_cache_spec_kind = Some(spec.into());
    event
}

/// A Qwen3.5 batch: one 528-token block in three Mamba groups and one
/// full-attention group, hashed with seed 0.
fn qwen_batch() -> KVEventBatch {
    let tokens = prompt();
    let hash = BlockKeyGenerator::new(528, 0).generate_block_keys(&tokens)[0];
    let events = groups()
        .iter()
        .map(|group| {
            let spec = if group.group_idx == 3 {
                "full_attention"
            } else {
                "mamba"
            };
            KVEvent::BlockStored(gpu_store(
                group.group_idx,
                spec,
                528,
                &tokens[..528],
                &[hash],
            ))
        })
        .collect();
    event_batch(events)
}

#[test]
fn incremental_second_copy_survives_first_remove() {
    // This test assumes native incremental events, where each Store is a
    // new physical copy. Full-report announcements do not have that meaning.
    let tokens = prompt();
    let hash = BlockKeyGenerator::new(528, 0).generate_block_keys(&tokens)[0];
    let index = SourceBinding::new(target(), groups(), 528, 0, 8).unwrap();
    index.begin_empty(0);
    let mut batch = qwen_batch();
    let mut copy = gpu_store(3, "full_attention", 528, &tokens[..528], &[hash]);
    copy.medium = Some("NPU".into());
    batch.events.push(KVEvent::BlockStored(copy));
    assert_eq!(index.apply_batch(0, &batch), Ok(()));
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 528);

    let mut removal = gpu_remove(Some(3), vec![ExternalBlockHash::Bytes(hash.to_vec())]);
    if let KVEvent::BlockRemoved(event) = &mut removal {
        event.medium = Some("ASCEND".into());
    }
    let removal = event_batch(vec![removal]);
    assert_eq!(index.apply_batch(1, &removal), Ok(()));
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 528);
    assert_eq!(index.apply_batch(2, &removal), Ok(()));
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 0);

    // A cache clear resets both residency and copy counts before new Stores.
    assert_eq!(index.apply_batch(3, &batch), Ok(()));
    let clear = event_batch(vec![KVEvent::AllBlocksCleared(Default::default())]);
    assert_eq!(index.apply_batch(4, &clear), Ok(()));
    assert_eq!(index.apply_batch(5, &qwen_batch()), Ok(()));
    assert_eq!(index.apply_batch(6, &removal), Ok(()));
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 0);

    // A complete replay rebuilds counts before the source is published.
    index.begin_replay(0);
    assert_eq!(index.apply_batch(0, &batch), Ok(()));
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 0);
    assert_eq!(index.finish_replay(1), Ok(()));
    assert_eq!(index.apply_batch(1, &removal), Ok(()));
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 528);
    assert_eq!(index.apply_batch(2, &removal), Ok(()));
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 0);
}

/// `qwen_batch()` encoded as the engine publishes it.
pub(crate) fn qwen35_payload() -> Vec<u8> {
    let batch = qwen_batch();
    event_payload(1.0, &batch.events, batch.data_parallel_rank)
}

pub(crate) fn qwen_index(target: RoutableTarget, limit: usize) -> SourceBinding {
    SourceBinding::new(target, groups(), 528, 0, limit).unwrap()
}

fn index(limit: usize) -> SourceBinding {
    qwen_index(target(), limit)
}

fn fresh_index(limit: usize) -> SourceBinding {
    let index = index(limit);
    index.begin_empty(0);
    index
}

#[test]
fn incremental_copy_counts_are_isolated_by_worker() {
    let shared = Arc::new(crate::kv_index::KvBlockIndexer::new());
    let first = index(8).with_shared_index(shared.clone());
    let second = SourceBinding::new(
        RoutableTarget {
            instance_id: Arc::from("w2"),
            dp_rank: 0,
        },
        groups(),
        528,
        0,
        8,
    )
    .unwrap()
    .with_shared_index(shared);
    let tokens = prompt();
    let hash = BlockKeyGenerator::new(528, 0).generate_block_keys(&tokens)[0];
    first.begin_empty(0);
    second.begin_empty(0);
    let mut two_copies = qwen_batch();
    two_copies.events.push(KVEvent::BlockStored(gpu_store(
        3,
        "full_attention",
        528,
        &tokens[..528],
        &[hash],
    )));
    first.apply_batch(0, &two_copies).unwrap();
    second.apply_batch(0, &qwen_batch()).unwrap();
    let removal = event_batch(vec![gpu_remove(
        Some(3),
        vec![ExternalBlockHash::Bytes(hash.to_vec())],
    )]);
    first.apply_batch(1, &removal).unwrap();
    assert_eq!(first.reusable_tokens(&tokens, usize::MAX), 528);
    second.apply_batch(1, &removal).unwrap();
    assert_eq!(second.reusable_tokens(&tokens, usize::MAX), 0);
    assert_eq!(first.reusable_tokens(&tokens, usize::MAX), 528);
    first.apply_batch(2, &removal).unwrap();
    assert_eq!(first.reusable_tokens(&tokens, usize::MAX), 0);
}

#[test]
fn shared_domain_keeps_workers_and_handle_generations_independent() {
    use crate::kv_index::KvBlockIndexer;

    let shared = Arc::new(KvBlockIndexer::new());
    let old = index(8).with_shared_index(shared.clone());
    let replacement = index(8).with_shared_index(shared.clone());
    let other = SourceBinding::new(
        RoutableTarget {
            instance_id: Arc::from("w2"),
            dp_rank: 0,
        },
        groups(),
        528,
        0,
        8,
    )
    .unwrap()
    .with_shared_index(shared);
    for index in [&old, &replacement, &other] {
        index.begin_empty(0);
        index.apply_batch(0, &qwen_batch()).unwrap();
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
    }
    old.invalidate();
    assert_eq!(old.reusable_tokens(&prompt(), usize::MAX), 0);
    assert_eq!(replacement.reusable_tokens(&prompt(), usize::MAX), 528);
    assert_eq!(other.reusable_tokens(&prompt(), usize::MAX), 528);

    // Rejecting a batch only revokes that exact handle's observations.
    assert_eq!(
        replacement.apply_batch(9, &qwen_batch()),
        Err(SourceError::Sequence)
    );
    assert_eq!(replacement.reusable_tokens(&prompt(), usize::MAX), 0);
    assert_eq!(other.reusable_tokens(&prompt(), usize::MAX), 528);
}

#[test]
fn shared_source_invalidation_removes_every_group_without_hiding_other_workers() {
    use crate::kv_index::KvBlockIndexer;

    let shared = Arc::new(KvBlockIndexer::new());
    let first = index(8).with_shared_index(shared.clone());
    let other = SourceBinding::new(
        RoutableTarget {
            instance_id: Arc::from("w2"),
            dp_rank: 0,
        },
        groups(),
        528,
        0,
        8,
    )
    .unwrap()
    .with_shared_index(shared.clone());
    for index in [&first, &other] {
        index.begin_empty(0);
        index.apply_batch(0, &qwen_batch()).unwrap();
    }
    shared.invalidate_source(&Arc::from("w"), 0);
    assert_eq!(first.reusable_tokens(&prompt(), usize::MAX), 0);
    assert_eq!(other.reusable_tokens(&prompt(), usize::MAX), 528);
    first.begin_replay(0);
    first.apply_batch(0, &qwen_batch()).unwrap();
    assert_eq!(first.reusable_tokens(&prompt(), usize::MAX), 0);
    first.finish_replay(1).unwrap();
    assert_eq!(first.reusable_tokens(&prompt(), usize::MAX), 528);
}

#[test]
fn snapshot_calculation_does_not_block_same_source_writes_or_survive_reset() {
    use std::sync::mpsc;
    use std::time::Duration;

    // Rebuild to the same sequence and visibility: checking those fields
    // alone cannot distinguish the old snapshot from the new cache view.
    for reset in 0..4 {
        let first = Arc::new(index(8));
        first.begin_empty(0);
        first.apply_batch(0, &qwen_batch()).unwrap();
        let tokens = prompt();
        let mut cache = crate::kv_index::RequestHashCache::default();
        let (_claims, snapshot) = first
            .page_snapshot(None, &tokens, false, &mut cache)
            .unwrap();
        assert!(first.snapshot_is_current(&snapshot));
        let writer = first.clone();
        let (sent, received) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            match reset {
                0 => writer.invalidate(),
                1 => writer.begin_replay(0),
                2 => assert_eq!(
                    writer.apply_batch(2, &qwen_batch()),
                    Err(SourceError::Sequence)
                ),
                _ => writer.begin_empty(0),
            }
            writer.begin_empty(0);
            writer.apply_batch(0, &qwen_batch()).unwrap();
            sent.send(()).unwrap();
        });
        received
            .recv_timeout(Duration::from_secs(2))
            .expect("page evidence must not hold the stream read lock");
        handle.join().unwrap();
        assert!(!first.snapshot_is_current(&snapshot));
        assert_eq!(first.reusable_tokens(&prompt(), usize::MAX), 528);
    }
}

#[test]
fn retired_source_cannot_reactivate_a_shared_domain() {
    use crate::kv_index::{source::binding::SourceLease, KvBlockIndexer};
    use std::sync::atomic::{AtomicBool, Ordering};

    let shared = Arc::new(KvBlockIndexer::new());
    let active = Arc::new(AtomicBool::new(true));
    let old = index(8)
        .with_shared_index(shared.clone())
        .with_source_binding(SourceLease::new(active.clone()));
    old.begin_replay(0);
    old.apply_batch(0, &qwen_batch()).unwrap();
    active.store(false, Ordering::Release);
    let replacement = index(8).with_shared_index(shared.clone());
    replacement.begin_empty(0);
    replacement.apply_batch(0, &qwen_batch()).unwrap();
    shared.suspend_source(&Arc::from("w"), 0);
    assert_eq!(old.finish_replay(1), Err(SourceError::Untrusted));
    old.begin_empty(1);
    assert_eq!(replacement.reusable_tokens(&prompt(), usize::MAX), 0);
    shared.resume_source(&Arc::from("w"), 0);
    assert_eq!(replacement.reusable_tokens(&prompt(), usize::MAX), 528);
    old.invalidate();
    assert_eq!(replacement.reusable_tokens(&prompt(), usize::MAX), 528);
}

#[test]
fn old_hybrid_cleanup_preserves_a_new_full_attention_owner() {
    use crate::kv_index::source::test_support::full_store;
    use crate::kv_index::KvBlockIndexer;

    let shared = Arc::new(KvBlockIndexer::new());
    let old = index(8).with_shared_index(shared.clone());
    old.begin_empty(0);
    old.apply_batch(0, &qwen_batch()).unwrap();

    let next = SourceBinding::new(target(), vec![groups()[3]], 528, 0, 8)
        .unwrap()
        .with_shared_index(shared);
    assert_ne!(old.owner, next.owner);
    next.begin_empty(0);
    let tokens = prompt();
    let mut store = full_store(&tokens[..528], 528, None);
    store.group_idx = Some(3);
    next.apply_batch(0, &event_batch(vec![KVEvent::BlockStored(store)]))
        .unwrap();
    assert_eq!(old.reusable_tokens(&tokens, usize::MAX), 528);
    assert_eq!(next.reusable_tokens(&tokens, usize::MAX), 528);

    old.invalidate();
    assert_eq!(old.reusable_tokens(&tokens, usize::MAX), 0);
    assert_eq!(next.reusable_tokens(&tokens, usize::MAX), 528);
}

#[test]
fn decoded_qwen_layout_requires_all_groups_and_tracks_removal() {
    let payload = qwen35_payload();
    let batch = crate::kv_index::decode_batch(&payload).unwrap();
    // A payload that does not decode revokes the view.
    let corrupt = fresh_index(8);
    assert_eq!(corrupt.apply_payload(0, &payload), Ok(()));
    assert_eq!(
        corrupt.apply_payload(1, &[0xc1]),
        Err(SourceError::InvalidBatch)
    );
    assert_eq!(corrupt.reusable_tokens(&prompt(), usize::MAX), 0);
    let index = index(8);
    assert_eq!(index.apply_batch(0, &batch), Err(SourceError::Untrusted));
    index.begin_empty(0);
    assert_eq!(index.apply_batch(0, &batch), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
    assert_eq!(index.reusable_tokens(&prompt(), 527), 0);
    let mut different_prompt = prompt();
    different_prompt[0] += 1;
    assert_eq!(index.reusable_tokens(&different_prompt, usize::MAX), 0);
    let KVEvent::BlockStored(store) = &batch.events[2] else {
        panic!("expected store")
    };
    let removed = event_batch(vec![gpu_remove(Some(2), store.block_hashes.clone())]);
    assert_eq!(index.apply_batch(1, &removed), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
}

fn host_store(
    batch: &crate::kv_index::KVEventBatch,
    medium: &str,
) -> crate::kv_index::KVEventBatch {
    let mut batch = batch.clone();
    for event in &mut batch.events {
        if let KVEvent::BlockStored(store) = event {
            store.medium = Some(medium.to_string());
        }
    }
    batch
}

#[test]
fn evicting_the_device_copy_keeps_the_host_credit() {
    use crate::policies::TieredMatchProvider;

    let index = fresh_index(16);
    assert_eq!(index.apply_batch(0, &qwen_batch()), Ok(()));
    assert_eq!(
        index.apply_batch(1, &host_store(&qwen_batch(), "CPU")),
        Ok(())
    );

    let mut removed = qwen_batch();
    for event in &mut removed.events {
        if let KVEvent::BlockStored(store) = event {
            *event = gpu_remove(store.group_idx, store.block_hashes.clone());
        }
    }
    assert_eq!(index.apply_batch(2, &removed), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
    let matches = index.matches(&prompt(), 528, None, 0);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].tier, crate::kv_index::StorageTier::HostPinned);
}

#[test]
fn a_foreign_owner_is_skipped_without_revoking_the_view() {
    let index = fresh_index(16);
    assert_eq!(index.apply_batch(0, &qwen_batch()), Ok(()));
    let mut foreign = qwen_batch();
    for event in &mut foreign.events {
        if let KVEvent::BlockStored(store) = event {
            store.ownership = Some("offload".into());
            store.block_hashes = vec![ExternalBlockHash::Int(42)];
            store.parent_block_hash = Some(ExternalBlockHash::Int(41));
        }
    }
    let local = qwen_batch();
    let KVEvent::BlockStored(store) = &local.events[2] else {
        panic!("expected store")
    };
    let mut removal = BlockRemoved {
        block_hashes: vec![ExternalBlockHash::Int(42)],
        medium: Some("GPU".into()),
        group_idx: store.group_idx,
        locality: Some("REMOTE".into()),
        ownership: Some("offload".into()),
    };
    foreign.events.push(KVEvent::BlockRemoved(removal.clone()));
    assert_eq!(index.apply_batch(1, &foreign), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);

    let foreign_removal = removal.clone();
    removal.locality = Some("LOCAL".into());
    removal.ownership = None;
    removal.block_hashes = store.block_hashes.clone();
    let local_removal = event_batch(vec![
        KVEvent::BlockRemoved(foreign_removal),
        KVEvent::BlockRemoved(removal),
    ]);
    assert_eq!(index.apply_batch(2, &local_removal), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
}

/// A required group that never published a page caps reuse at zero: its
/// silence is not evidence that the page is unnecessary.
#[test]
fn a_required_group_without_page_evidence_caps_reuse_at_zero() {
    let index = fresh_index(16);
    let mut batch = qwen_batch();
    batch.events.retain(|event| match event {
        KVEvent::BlockStored(store) => store.group_idx == Some(3),
        _ => false,
    });
    assert_eq!(index.apply_batch(0, &batch), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
}

/// An adapter publishes under its own hash chain. Such a batch must leave
/// the base model's claims intact instead of revoking the whole view.
#[test]
fn adapter_stores_do_not_revoke_the_base_view() {
    let index = fresh_index(16);
    assert_eq!(index.apply_batch(0, &qwen_batch()), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);

    let mut adapter_batch = qwen_batch();
    for event in &mut adapter_batch.events {
        if let KVEvent::BlockStored(store) = event {
            store.lora_name = Some("lora1".to_string());
        }
    }
    assert_eq!(index.apply_batch(1, &adapter_batch), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
    // The base chain keeps applying afterwards.
    assert_eq!(index.apply_batch(2, &qwen_batch()), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
}

/// The same pages stored under one adapter: the engine hashes them with
/// the adapter's `(name, path)` extra keys, so this is a different chain.
fn adapter_batch(name: &str, path: &str) -> KVEventBatch {
    let tokens = prompt();
    let hash =
        BlockKeyGenerator::new(528, 0).generate_block_keys_for_adapter(&tokens, name, path)[0];
    let events = groups()
        .iter()
        .map(|group| {
            let spec = if group.group_idx == 3 {
                "full_attention"
            } else {
                "mamba"
            };
            let mut store = gpu_store(group.group_idx, spec, 528, &tokens[..528], &[hash]);
            store.lora_name = Some(name.to_string());
            store.extra_keys = None;
            KVEvent::BlockStored(store)
        })
        .collect();
    event_batch(events)
}

/// An adapter's pages live under their own chain: they are indexed under
/// the declared adapter, never credited to the base model, and never to
/// another adapter.
#[test]
fn adapter_pages_are_indexed_under_their_own_chain() {
    let index = fresh_index(16).with_adapters(vec![
        ("lora1".to_string(), "/adapters/lora1".to_string()),
        ("lora2".to_string(), "/adapters/lora2".to_string()),
    ]);
    assert_eq!(
        index.apply_batch(0, &adapter_batch("lora1", "/adapters/lora1")),
        Ok(())
    );
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    assert_eq!(
        index.reusable_tokens_in(Some("lora1"), &prompt(), usize::MAX),
        528
    );
    assert_eq!(
        index.reusable_tokens_in(Some("lora2"), &prompt(), usize::MAX),
        0
    );

    // The base model stores the same tokens under its own chain.
    assert_eq!(index.apply_batch(1, &qwen_batch()), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
    assert_eq!(
        index.reusable_tokens_in(Some("lora1"), &prompt(), usize::MAX),
        528
    );

    // An eviction names hashes, not a chain: the engine removes the page
    // from every group, and both views lose it.
    let adapter_hash = BlockKeyGenerator::new(528, 0).generate_block_keys_for_adapter(
        &prompt(),
        "lora1",
        "/adapters/lora1",
    )[0]
    .to_vec();
    let base_hash = BlockKeyGenerator::new(528, 0).generate_block_keys(&prompt())[0].to_vec();
    let removed = event_batch(
        groups()
            .iter()
            .map(|group| {
                KVEvent::BlockRemoved(BlockRemoved {
                    block_hashes: vec![
                        ExternalBlockHash::Bytes(adapter_hash.clone()),
                        ExternalBlockHash::Bytes(base_hash.clone()),
                    ],
                    medium: Some("GPU".to_string()),
                    group_idx: Some(group.group_idx),
                    locality: Some("LOCAL".to_string()),
                    ownership: None,
                })
            })
            .collect(),
    );
    assert_eq!(index.apply_batch(2, &removed), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    assert_eq!(
        index.reusable_tokens_in(Some("lora1"), &prompt(), usize::MAX),
        0
    );
}

#[test]
fn hybrid_first_page_accepts_either_adapter_hash_endpoint() {
    use crate::kv_index::StorageTier;
    use crate::policies::TieredMatchProvider;

    let groups = vec![
        CacheGroup {
            group_idx: 0,
            block_size: 32,
            kind: GroupRule::FullAttention,
        },
        CacheGroup {
            group_idx: 1,
            block_size: 32,
            kind: GroupRule::Checkpoint,
        },
    ];
    let index = SourceBinding::new(target(), groups, 16, 0, 16)
        .unwrap()
        .with_partial_hash_hits(true)
        .with_adapters(vec![("a".into(), "/a".into()), ("b".into(), "/b".into())]);
    index.begin_empty(0);
    let tokens: Vec<_> = (0..65).collect();
    let hashes = BlockKeyGenerator::new(16, 0).generate_block_keys_for_adapter(&tokens, "a", "/a");
    let stores = |start: usize, end: usize, medium: &str| {
        [(0, "full_attention"), (1, "mamba")].map(|(group, kind)| {
            let mut store = gpu_store(
                group,
                kind,
                (end - start) as u32,
                &tokens[start..end],
                &hashes[end / 16 - 1..end / 16],
            );
            store.lora_name = Some("a".into());
            store.medium = Some(medium.into());
            KVEvent::BlockStored(store)
        })
    };
    index
        .apply_batch(
            0,
            &event_batch(
                stores(0, 32, "GPU")
                    .into_iter()
                    .chain(stores(32, 48, "GPU"))
                    .collect(),
            ),
        )
        .unwrap();
    assert_eq!(
        index.matches(&tokens, 16, Some("a"), 0)[0].segments,
        [(0, 3)]
    );
    for other in [None, Some("b")] {
        assert!(index.matches(&tokens, 16, other, 0).is_empty());
    }
    let evict = gpu_remove(Some(0), vec![ExternalBlockHash::Bytes(hashes[1].to_vec())]);
    index.apply_batch(1, &event_batch(vec![evict])).unwrap();
    assert!(index.matches(&tokens, 16, Some("a"), 0).is_empty());
    index
        .apply_batch(2, &event_batch(stores(0, 16, "CPU").to_vec()))
        .unwrap();
    let hits = index.matches(&tokens, 16, Some("a"), 0);
    assert_eq!(hits[0].segments, [(0, 1)]);
    assert_eq!(hits[0].tier, StorageTier::HostPinned);
}

/// A page for an adapter this layout does not declare is skipped: it can
/// neither credit the base model nor revoke the claims the rank holds.
/// A real adapter store carries its identity as extra keys; that is not a
/// foreign identity, so the pages still count for the adapter's chain.
#[test]
fn adapter_stores_carrying_lora_extra_keys_are_credited() {
    let index =
        fresh_index(16).with_adapters(vec![("lora1".to_string(), "/adapters/lora1".to_string())]);
    let mut batch = adapter_batch("lora1", "/adapters/lora1");
    for event in &mut batch.events {
        if let KVEvent::BlockStored(store) = event {
            store.extra_keys = Some(vec![rmpv::Value::String("lora1".into())]);
        }
    }
    assert_eq!(index.apply_batch(0, &batch), Ok(()));
    assert_eq!(
        index.reusable_tokens_in(Some("lora1"), &prompt(), usize::MAX),
        528
    );
}

#[test]
fn undeclared_adapter_stores_are_skipped() {
    use crate::policies::TieredMatchProvider;

    let index = fresh_index(16);
    assert_eq!(index.apply_batch(0, &qwen_batch()), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);

    assert_eq!(
        index.apply_batch(1, &adapter_batch("ghost", "/adapters/ghost")),
        Ok(())
    );
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
    assert_eq!(
        index.reusable_tokens_in(Some("ghost"), &prompt(), usize::MAX),
        0
    );
    assert!(index.matches(&prompt(), 528, Some("ghost"), 0).is_empty());
    assert!(index.matches(&prompt(), 528, None, 0).len() == 1);
}

/// A V4 MLA layout without a declared state ratio restores from the root,
/// so a full-attention group accepts its pages.
#[test]
fn mla_pages_are_full_attention_stores() {
    let store = gpu_store(0, "mla_attention", 528, &prompt()[..528], &[[7u8; 32]]);
    assert!(event_kind_matches(GroupRule::FullAttention, &store));
    let mut sliding = store;
    sliding.kv_cache_spec_sliding_window = Some(128);
    assert!(!event_kind_matches(GroupRule::FullAttention, &sliding));
}

#[test]
fn gap_and_old_sequence_revoke_the_whole_view() {
    let index = index(8);
    let batch = qwen_batch();
    index.begin_empty(0);
    assert_eq!(index.apply_batch(0, &batch), Ok(()));
    assert_eq!(index.apply_batch(2, &batch), Err(SourceError::Sequence));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);

    index.begin_empty(0);
    assert_eq!(index.apply_batch(0, &batch), Ok(()));
    assert_eq!(index.apply_batch(0, &batch), Err(SourceError::Sequence));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    assert_eq!(index.apply_batch(1, &batch), Err(SourceError::Untrusted));
    index.begin_empty(5);
    assert_eq!(index.apply_batch(5, &batch), Ok(()));
    assert_eq!(index.apply_batch(2, &batch), Err(SourceError::Sequence));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
}

#[test]
fn replay_stays_hidden_until_the_verified_watermark() {
    let index = index(8);
    let batch = qwen_batch();
    index.begin_replay(0);
    assert_eq!(index.apply_batch(0, &batch), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    assert_eq!(index.finish_replay(1), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);

    let KVEvent::BlockStored(store) = &batch.events[2] else {
        panic!("expected store")
    };
    let eviction = event_batch(vec![gpu_remove(Some(2), store.block_hashes.clone())]);
    index.begin_replay(0);
    assert_eq!(index.apply_batch(0, &batch), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    assert_eq!(index.apply_batch(1, &eviction), Ok(()));
    assert_eq!(index.finish_replay(2), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);

    index.begin_replay(0);
    assert_eq!(index.apply_batch(0, &batch), Ok(()));
    assert_eq!(index.finish_replay(2), Err(SourceError::Sequence));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    assert_eq!(index.apply_batch(1, &batch), Err(SourceError::Untrusted));
}

#[test]
fn invalid_event_or_rank_cannot_expose_a_partial_batch() {
    let index = fresh_index(8);
    let mut batch = qwen_batch();
    let KVEvent::BlockStored(last) = &mut batch.events[3] else {
        panic!("expected store")
    };
    // An integer hash is the engine's default event encoding, so it gets its
    // own error rather than the generic shape mismatch.
    last.block_hashes = vec![ExternalBlockHash::Int(42)];
    assert_eq!(
        index.apply_batch(0, &batch),
        Err(SourceError::IntBlockHashes)
    );
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    assert_eq!(
        index.apply_batch(0, &qwen_batch()),
        Err(SourceError::Untrusted)
    );

    let mut wrong_rank = qwen_batch();
    wrong_rank.data_parallel_rank = Some(1);
    index.begin_empty(0);
    assert_eq!(
        index.apply_batch(0, &wrong_rank),
        Err(SourceError::InvalidBatch)
    );
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
}

#[test]
fn an_int_parent_hash_is_reported_as_int_block_hashes() {
    let index = fresh_index(8);
    let mut batch = qwen_batch();
    let KVEvent::BlockStored(last) = &mut batch.events[3] else {
        panic!("expected store")
    };
    last.parent_block_hash = Some(ExternalBlockHash::Int(7));
    assert_eq!(
        index.apply_batch(0, &batch),
        Err(SourceError::IntBlockHashes)
    );
}

#[test]
fn unmodeled_cache_identity_and_group_metadata_are_rejected() {
    let index = fresh_index(8);
    let mut salted = qwen_batch();
    let KVEvent::BlockStored(last) = &mut salted.events[3] else {
        panic!("expected store")
    };
    last.extra_keys = Some(vec![rmpv::Value::from("unknown-salt")]);
    assert_eq!(
        index.apply_batch(0, &salted),
        Err(SourceError::InvalidBatch)
    );
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);

    let mut wrong_kind = qwen_batch();
    let KVEvent::BlockStored(last) = &mut wrong_kind.events[3] else {
        panic!("expected store")
    };
    last.kv_cache_spec_kind = Some("mamba".into());
    index.begin_empty(0);
    assert_eq!(
        index.apply_batch(0, &wrong_kind),
        Err(SourceError::InvalidBatch)
    );
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
}

#[test]
fn salted_events_leave_replay_open_for_the_unsalted_chain() {
    let index = fresh_index(16);
    let original = qwen_batch();
    let mut salted = original.clone();
    for event in &mut salted.events {
        let KVEvent::BlockStored(store) = event else {
            panic!("expected store");
        };
        let mut extra_keys = vec![rmpv::Value::Nil; store.block_hashes.len()];
        extra_keys[0] = rmpv::Value::Array(vec![rmpv::Value::from("private-request")]);
        store.extra_keys = Some(extra_keys);
    }
    assert_eq!(index.apply_batch(0, &salted), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    assert_eq!(index.apply_batch(1, &original), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
}

#[test]
fn salted_image_events_preserve_unrelated_prefix_credit() {
    use crate::kv_index::source::test_support::{
        full_binding, full_store, warm_full, TEST_IMAGE_HASH,
    };
    use rmpv::Value;

    let index = full_binding(target(), 16);
    let warm: Vec<_> = (100..133).collect();
    index.begin_empty(0);
    warm_full(&index, 0, &warm[..32]);
    assert_eq!(index.reusable_tokens(&warm, usize::MAX), 32);

    let image = Value::Array(vec![TEST_IMAGE_HASH.into(), 0.into()]);
    let salt = Value::from("private-request");
    let tokens: Vec<_> = (0..33).collect();
    for (seq, keys) in [
        vec![Value::Array(vec![image.clone(), salt.clone()]), Value::Nil],
        vec![Value::Array(vec![salt]), Value::Array(vec![image])],
    ]
    .into_iter()
    .enumerate()
    {
        let mut store = full_store(&tokens[..32], 16, None);
        store.extra_keys = Some(keys);
        let batch = event_batch(vec![KVEvent::BlockStored(store)]);
        assert_eq!(index.apply_batch(seq as u64 + 1, &batch), Ok(()));
        assert_eq!(index.reusable_tokens(&warm, usize::MAX), 32);
        assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 0);
    }
}

#[test]
fn declared_adapter_store_does_not_hide_layout_errors() {
    for error in ["group", "kind", "locality", "block size", "token packing"] {
        let index = fresh_index(16).with_adapters(vec![("lora1".into(), "/adapters/lora1".into())]);
        index.apply_batch(0, &qwen_batch()).unwrap();
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
        let mut batch = adapter_batch("lora1", "/adapters/lora1");
        let KVEvent::BlockStored(store) = &mut batch.events[0] else {
            panic!("expected store");
        };
        match error {
            "group" => store.group_idx = Some(u32::MAX),
            "kind" => store.kv_cache_spec_kind = Some("sliding_window".into()),
            "locality" => store.locality = Some("REMOTE".into()),
            "block size" => store.block_size += 1,
            "token packing" => {
                store.token_ids.pop();
            }
            _ => unreachable!(),
        }
        assert_eq!(
            index.apply_batch(1, &batch),
            Err(SourceError::InvalidBatch),
            "{error}"
        );
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
        index.begin_replay(0);
        index.apply_batch(0, &qwen_batch()).unwrap();
        index
            .apply_batch(1, &adapter_batch("lora1", "/adapters/lora1"))
            .unwrap();
        index.finish_replay(2).unwrap();
        assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
        assert_eq!(
            index.reusable_tokens_in(Some("lora1"), &prompt(), usize::MAX),
            528
        );
    }
}

#[test]
fn capacity_failure_invalidates_all_groups_and_requires_a_new_barrier() {
    let index = fresh_index(3);
    let batch = qwen_batch();
    assert_eq!(index.apply_batch(0, &batch), Err(SourceError::Capacity));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    assert_eq!(index.apply_batch(1, &batch), Err(SourceError::Untrusted));
}

#[test]
fn adapter_capacity_failure_invalidates_source_before_copy_counts_grow() {
    let index = fresh_index(3).with_adapters(vec![("lora1".into(), "/adapters/lora1".into())]);
    let batch = adapter_batch("lora1", "/adapters/lora1");
    assert_eq!(index.apply_batch(0, &batch), Err(SourceError::Capacity));
    assert_eq!(
        index.reusable_tokens_in(Some("lora1"), &prompt(), usize::MAX),
        0
    );
    assert_eq!(index.apply_batch(1, &batch), Err(SourceError::Untrusted));
}

#[test]
fn group_index_removal_reduces_the_common_restore_point() {
    let tokens: Vec<_> = (0..64).collect();
    let full = crate::kv_index::source::test_support::full_store(&tokens, 16, None);
    let mut checkpoint = full.clone();
    checkpoint.group_idx = Some(1);
    checkpoint.kv_cache_spec_kind = Some("mamba".into());
    let removed = checkpoint.block_hashes[2].clone();
    let index = SourceBinding::new(
        target(),
        vec![
            CacheGroup {
                group_idx: 0,
                block_size: 16,
                kind: GroupRule::FullAttention,
            },
            CacheGroup {
                group_idx: 1,
                block_size: 16,
                kind: GroupRule::Checkpoint,
            },
        ],
        16,
        0,
        8,
    )
    .unwrap();
    index.begin_empty(0);
    let stored = event_batch(vec![
        KVEvent::BlockStored(full),
        KVEvent::BlockStored(checkpoint),
    ]);
    index.apply_batch(0, &stored).unwrap();
    assert_eq!(index.reusable_tokens(&tokens, 63), 48);

    index
        .apply_batch(1, &event_batch(vec![gpu_remove(Some(1), vec![removed])]))
        .unwrap();
    assert_eq!(index.reusable_tokens(&tokens, 63), 32);
}

#[test]
fn speculative_wire_events_require_complete_blocks_and_matching_lookahead() {
    use crate::kv_index::source::test_support::{full_binding, full_store};

    let prompt: Vec<u32> = (0..34).collect();
    let mut binding = full_binding(target(), 16);
    binding.keys = binding.keys.with_lookahead(1);

    let mut store = full_store(&prompt[..32], 16, None);
    store.block_hashes = binding
        .keys
        .generate_block_keys(&prompt)
        .into_iter()
        .map(|hash| ExternalBlockHash::Bytes(hash.to_vec()))
        .collect();
    store.lookahead_token_ids = Some(vec![vec![16], vec![32]]);
    let payload = |store| event_payload(1.0, &[KVEvent::BlockStored(store)], Some(0));
    for remove_lookahead in [false, true] {
        binding.begin_empty(0);
        binding.apply_payload(0, &payload(store.clone())).unwrap();
        assert_eq!(binding.reusable_tokens(&prompt, 34), 32);
        let mut invalid = store.clone();
        if remove_lookahead {
            invalid.lookahead_token_ids = None;
        } else {
            invalid.token_ids.truncate(16);
        }
        assert_eq!(
            binding.apply_payload(1, &payload(invalid)),
            Err(SourceError::InvalidBatch)
        );
        assert_eq!(binding.reusable_tokens(&prompt, 34), 0);
    }
}

#[test]
fn clear_drops_every_group_before_new_events_are_applied() {
    let index = fresh_index(8);
    let batch = qwen_batch();
    assert_eq!(index.apply_batch(0, &batch), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 528);
    let cleared = event_batch(vec![KVEvent::AllBlocksCleared(Default::default())]);
    assert_eq!(index.apply_batch(1, &cleared), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
    let partial = event_batch(vec![batch.events[0].clone()]);
    assert_eq!(index.apply_batch(2, &partial), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 0);
}

#[test]
fn unsupported_or_ambiguous_layouts_are_rejected() {
    assert!(matches!(
        SourceBinding::new(target(), groups(), 0, 0, 8),
        Err(SourceError::InvalidLayout)
    ));
    let mut wrong = groups();
    wrong[0].kind = GroupRule::SlidingWindow { window_tokens: 0 };
    assert!(matches!(
        SourceBinding::new(target(), wrong, 16, 0, 8),
        Err(SourceError::InvalidLayout)
    ));
    let mut duplicate = groups();
    duplicate[0].group_idx = 1;
    assert!(matches!(
        SourceBinding::new(target(), duplicate, 528, 0, 8),
        Err(SourceError::InvalidLayout)
    ));
}

#[test]
fn sparse_checkpoint_does_not_require_its_parent_to_be_resident() {
    let tokens: Vec<u32> = (0..17).collect();
    let hashes = BlockKeyGenerator::new(4, 0).generate_block_keys(&tokens);
    let layout = vec![
        CacheGroup {
            group_idx: 0,
            block_size: 4,
            kind: GroupRule::FullAttention,
        },
        CacheGroup {
            group_idx: 1,
            block_size: 8,
            kind: GroupRule::Checkpoint,
        },
    ];
    let mut checkpoint = gpu_store(1, "mamba", 8, &tokens[8..16], &hashes[3..4]);
    checkpoint.parent_block_hash = Some(ExternalBlockHash::Bytes(hashes[1].to_vec()));
    let batch = event_batch(vec![
        KVEvent::BlockStored(gpu_store(0, "full_attention", 4, &tokens[..16], &hashes)),
        KVEvent::BlockStored(checkpoint),
    ]);
    let index = SourceBinding::new(target(), layout, 4, 0, 8).unwrap();
    index.begin_empty(0);
    assert_eq!(index.apply_batch(0, &batch), Ok(()));
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 16);
}

#[test]
fn concurrent_queries_never_see_a_partially_applied_batch() {
    let index = fresh_index(8);
    let initial = qwen_batch();
    let KVEvent::BlockStored(store) = &initial.events[2] else {
        panic!("expected store")
    };
    let replacement = event_batch(vec![
        gpu_remove(Some(2), store.block_hashes.clone()),
        KVEvent::BlockStored(store.clone()),
    ]);
    assert_eq!(index.apply_batch(0, &initial), Ok(()));
    let start = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let tokens = prompt();
            start.wait();
            for _ in 0..32 {
                assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 528);
            }
        });
        start.wait();
        for seq in 1..=32 {
            assert_eq!(index.apply_batch(seq, &replacement), Ok(()));
        }
    });
}

fn qwen4_groups() -> Vec<CacheGroup> {
    checkpoint_groups(
        320,
        GroupRule::CompressedFullAttention {
            tokens_per_state: 4,
        },
    )
}

fn qwen4_batch(kind_override: Option<&str>) -> KVEventBatch {
    let tokens = prompt();
    let hash = BlockKeyGenerator::new(320, 0).generate_block_keys(&tokens)[0];
    let events = qwen4_groups()
        .iter()
        .map(|group| {
            let compressed = matches!(group.kind, GroupRule::CompressedFullAttention { .. });
            let default = if compressed { "mla_attention" } else { "mamba" };
            let spec = kind_override.unwrap_or(default);
            KVEvent::BlockStored(gpu_store(
                group.group_idx,
                spec,
                320,
                &tokens[..320],
                &[hash],
            ))
        })
        .collect();
    event_batch(events)
}

#[test]
fn qwen4_compressed_group_accepts_mla_events_and_rejects_other_specs() {
    let index = SourceBinding::new(target(), qwen4_groups(), 320, 0, 8).unwrap();
    index.begin_empty(0);
    assert_eq!(index.apply_batch(0, &qwen4_batch(None)), Ok(()));
    assert_eq!(index.reusable_tokens(&prompt(), usize::MAX), 320);

    let wrong = SourceBinding::new(target(), qwen4_groups(), 320, 0, 8).unwrap();
    wrong.begin_empty(0);
    assert_eq!(
        wrong.apply_batch(0, &qwen4_batch(Some("full_attention"))),
        Err(SourceError::InvalidBatch)
    );
}

#[test]
fn qwen4_compressed_partial_stores_extend_prefix_reuse() {
    let tokens = prompt();
    let hashes = BlockKeyGenerator::new(16, 0).generate_block_keys(&tokens);
    let groups = qwen4_groups();
    let batch = event_batch(
        groups
            .iter()
            .flat_map(|group| {
                let spec = if group.kind == GroupRule::Checkpoint {
                    "mamba"
                } else {
                    "mla_attention"
                };
                [
                    KVEvent::BlockStored(gpu_store(
                        group.group_idx,
                        spec,
                        320,
                        &tokens[..320],
                        &hashes[19..20],
                    )),
                    KVEvent::BlockStored(gpu_store(
                        group.group_idx,
                        spec,
                        16,
                        &tokens[464..480],
                        &hashes[29..30],
                    )),
                ]
            })
            .collect(),
    );
    let index = SourceBinding::new(target(), groups, 16, 0, 8)
        .unwrap()
        .with_partial_hash_hits(true);
    index.begin_empty(0);
    assert_eq!(index.apply_batch(0, &batch), Ok(()));
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 480);
}

#[test]
fn compressed_group_must_align_with_its_block_size() {
    let mut groups = qwen4_groups();
    groups[3].kind = GroupRule::CompressedFullAttention {
        tokens_per_state: 7,
    };
    assert!(matches!(
        SourceBinding::new(target(), groups, 320, 0, 8),
        Err(SourceError::InvalidLayout)
    ));
}

#[test]
fn checkpoint_event_may_cover_the_whole_cached_chunk() {
    let tokens: Vec<u32> = (0..2048).map(|i| 100 + i % 100).collect();
    let keys = BlockKeyGenerator::new(320, 0).generate_block_keys(&tokens);
    // Captured from vLLM 0.33: a Mamba group publishes only the retained
    // checkpoint hash while token_ids spans the whole cached chunk. The
    // QSA attention group still publishes one hash per block.
    let batch = |attention_hashes: usize, checkpoint_hashes: usize| {
        let events = qwen4_groups()
            .iter()
            .map(|group| {
                let attention = matches!(group.kind, GroupRule::CompressedFullAttention { .. });
                let count = if attention {
                    attention_hashes
                } else {
                    checkpoint_hashes
                };
                let (spec, hashes) = if attention {
                    ("mla_attention", &keys[..count])
                } else {
                    ("mamba", &keys[6 - count..6])
                };
                KVEvent::BlockStored(gpu_store(
                    group.group_idx,
                    spec,
                    320,
                    &tokens[..1920],
                    hashes,
                ))
            })
            .collect();
        event_batch(events)
    };
    let index = SourceBinding::new(target(), qwen4_groups(), 320, 0, 16).unwrap();
    index.begin_empty(0);
    assert_eq!(index.apply_batch(0, &batch(6, 1)), Ok(()));
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 1920);

    let sparse_attention = SourceBinding::new(target(), qwen4_groups(), 320, 0, 16).unwrap();
    sparse_attention.begin_empty(0);
    assert_eq!(
        sparse_attention.apply_batch(0, &batch(1, 1)),
        Err(SourceError::InvalidBatch)
    );
    assert_eq!(sparse_attention.reusable_tokens(&tokens, usize::MAX), 0);

    // vLLM can emit a checkpoint group with an empty hash list when all
    // blocks in the chunk are null. The stream stays valid, but that group
    // has no residency and cannot complete a hit.
    let empty_checkpoints = SourceBinding::new(target(), qwen4_groups(), 320, 0, 16).unwrap();
    empty_checkpoints.begin_empty(0);
    assert_eq!(empty_checkpoints.apply_batch(0, &batch(6, 0)), Ok(()));
    assert_eq!(empty_checkpoints.reusable_tokens(&tokens, usize::MAX), 0);
    assert_eq!(empty_checkpoints.apply_batch(1, &batch(6, 1)), Ok(()));
    assert_eq!(empty_checkpoints.reusable_tokens(&tokens, usize::MAX), 1920);
}

#[test]
fn qwen35_checkpoint_groups_may_omit_hashes_on_a_reused_chunk() {
    // Captured from Qwen3.5-4B on vLLM 0.33 (block size 576): re-running a
    // cached prompt emits a stored event whose Mamba groups carry zero
    // hashes and a parent, while full attention carries one hash.
    let tokens: Vec<u32> = (0..2048).map(|i| 100 + i % 100).collect();
    let keys = BlockKeyGenerator::new(576, 0).generate_block_keys(&tokens);
    let groups = checkpoint_groups(576, GroupRule::FullAttention);
    let store = |group_idx: u32, hashes: &[SequenceHash], tokens_len: usize| {
        let spec = if group_idx == 3 {
            "full_attention"
        } else {
            "mamba"
        };
        KVEvent::BlockStored(gpu_store(
            group_idx,
            spec,
            576,
            &tokens[..tokens_len],
            hashes,
        ))
    };
    let index = SourceBinding::new(target(), groups, 576, 0, 64).unwrap();
    index.begin_empty(0);
    let first = event_batch(vec![
        store(0, &keys[2..3], 1728),
        store(1, &keys[2..3], 1728),
        store(2, &keys[2..3], 1728),
        store(3, &keys[..3], 1728),
    ]);
    assert_eq!(index.apply_batch(0, &first), Ok(()));
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 1728);
    // Re-run of the same prompt: checkpoints publish nothing new.
    let rerun = event_batch(vec![
        store(0, &[], 576),
        store(1, &[], 576),
        store(2, &[], 576),
        store(3, &keys[..1], 576),
    ]);
    assert_eq!(index.apply_batch(1, &rerun), Ok(()));
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 1728);
}

#[test]
fn sliding_group_restores_only_verified_windows() {
    let tokens = prompt();
    let keys = BlockKeyGenerator::new(64, 0).generate_block_keys(&tokens);
    let groups = vec![
        CacheGroup {
            group_idx: 0,
            block_size: 64,
            kind: GroupRule::FullAttention,
        },
        CacheGroup {
            group_idx: 1,
            block_size: 64,
            kind: GroupRule::SlidingWindow { window_tokens: 64 },
        },
    ];
    let dense = gpu_store(0, "full_attention", 64, &tokens[..512], &keys[..8]);
    for (window, pages, result, hit) in [
        // The last window suffices, even without preceding sliding pages.
        (64, 7..8, Ok(()), 512),
        // Missing the last window falls back to the last complete one.
        (64, 0..6, Ok(()), 384),
        (32, 7..8, Err(SourceError::InvalidBatch), 0),
    ] {
        let index = SourceBinding::new(target(), groups.clone(), 64, 0, 64).unwrap();
        index.begin_empty(0);
        let mut sliding = gpu_store(
            1,
            "sliding_window_mla",
            64,
            &tokens[pages.start * 64..pages.end * 64],
            &keys[pages],
        );
        sliding.kv_cache_spec_sliding_window = Some(window);
        let batch = event_batch(vec![
            KVEvent::BlockStored(dense.clone()),
            KVEvent::BlockStored(sliding),
        ]);
        assert_eq!(index.apply_batch(0, &batch), result);
        assert_eq!(index.reusable_tokens(&tokens, usize::MAX), hit);
    }
}

/// The event shape a DeepSeek V4 layout publishes: sliding groups keep only
/// their resident tail, one group publishes 4-token pages next to its own
/// 256-token blocks, and the groups use different block sizes that share
/// one hash chain at the layout's 4-token granularity.
#[test]
fn tail_window_events_credit_the_engine_boundary() {
    let tokens: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
    let keys = BlockKeyGenerator::new(4, 0).generate_block_keys(&tokens);
    let view = |block: usize| -> Vec<SequenceHash> {
        let stride = block / 4;
        keys.iter()
            .skip(stride - 1)
            .step_by(stride)
            .copied()
            .collect()
    };
    let groups = vec![
        CacheGroup {
            group_idx: 0,
            block_size: 64,
            kind: GroupRule::SlidingWindow { window_tokens: 128 },
        },
        CacheGroup {
            group_idx: 1,
            block_size: 64,
            kind: GroupRule::SlidingWindow { window_tokens: 128 },
        },
        CacheGroup {
            group_idx: 2,
            block_size: 256,
            kind: GroupRule::CompressedFullAttention {
                tokens_per_state: 4,
            },
        },
        CacheGroup {
            group_idx: 3,
            block_size: 4,
            kind: GroupRule::SlidingWindow { window_tokens: 8 },
        },
        CacheGroup {
            group_idx: 4,
            block_size: 8,
            kind: GroupRule::SlidingWindow { window_tokens: 128 },
        },
    ];
    let store = |group: CacheGroup,
                 spec: &str,
                 window: Option<u32>,
                 hashes: Vec<SequenceHash>,
                 block: usize,
                 span: usize| {
        let mut store = gpu_store(
            group.group_idx,
            spec,
            block as u32,
            &tokens[..span],
            &hashes,
        );
        store.kv_cache_spec_sliding_window = window;
        KVEvent::BlockStored(store)
    };
    let sliding_groups = [
        (0, 128, 10..12),
        (1, 128, 10..12),
        (3, 8, 190..192),
        (4, 128, 80..96),
    ];
    let sliding_stores = |publish_tail| {
        let groups = &groups;
        sliding_groups.iter().map(move |(id, window, pages)| {
            let group = groups[*id];
            let block = group.block_size as usize;
            let hashes = if publish_tail {
                view(block)[pages.clone()].to_vec()
            } else {
                Vec::new()
            };
            store(
                group,
                "sliding_window_mla",
                Some(*window),
                hashes,
                block,
                1024,
            )
        })
    };
    let index = SourceBinding::new(target(), groups.clone(), 4, 0, 4096).unwrap();
    index.begin_empty(0);

    // Empty stores do not make cacheable sliding groups optional.
    let mut initial: Vec<_> = sliding_stores(false).collect();
    initial.push(store(
        groups[2],
        "mla_attention",
        None,
        view(256),
        256,
        1024,
    ));
    assert_eq!(index.apply_batch(0, &event_batch(initial)), Ok(()));
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 0);

    // Reuse stops at the boundary for which all four sliding tails are present.
    let tail = event_batch(sliding_stores(true).collect());
    assert_eq!(index.apply_batch(1, &tail), Ok(()));
    assert_eq!(index.reusable_tokens(&tokens, usize::MAX), 768);
}

#[test]
fn sliding_tier_prices_only_pages_needed_by_engine() {
    use crate::kv_index::StorageTier;
    use crate::policies::TieredMatchProvider;

    let tokens: Vec<u32> = (0..49).collect();
    let hashes = BlockKeyGenerator::new(16, 0).generate_block_keys(&tokens);
    for (window, expected_tiers) in [
        (
            1,
            [
                StorageTier::HostPinned,
                StorageTier::Device,
                StorageTier::Device,
            ],
        ),
        (
            17,
            [
                StorageTier::HostPinned,
                StorageTier::Device,
                StorageTier::Device,
            ],
        ),
        (
            18,
            [
                StorageTier::HostPinned,
                StorageTier::HostPinned,
                StorageTier::Device,
            ],
        ),
    ] {
        let index = SourceBinding::new(
            target(),
            vec![CacheGroup {
                group_idx: 0,
                block_size: 16,
                kind: GroupRule::SlidingWindow {
                    window_tokens: window,
                },
            }],
            16,
            0,
            16,
        )
        .unwrap();
        index.begin_empty(0);
        let mut stores = Vec::new();
        for i in 0..3 {
            let mut store = gpu_store(
                0,
                "sliding_window",
                16,
                &tokens[i * 16..(i + 1) * 16],
                &hashes[i..i + 1],
            );
            store.kv_cache_spec_sliding_window = Some(window as u32);
            if i == 0 {
                store.medium = Some("CPU".into());
            }
            stores.push(KVEvent::BlockStored(store));
        }
        let batch = event_batch(stores);
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        for (pages, expected_tier) in (1..=3).zip(expected_tiers) {
            let matches = index.matches(&tokens[..pages * 16 + 1], 16, None, 0);
            assert_eq!(matches.len(), 1, "window {window}, pages {pages}");
            assert_eq!(
                matches[0].segments,
                vec![(0, pages as u32)],
                "window {window}"
            );
            assert_eq!(
                matches[0].tier, expected_tier,
                "window {window}, pages {pages}"
            );
        }
    }
}

/// The token span may include pages that vLLM skipped. Only the published
/// hashes consume capacity and prove residency.
#[test]
fn sliding_claims_only_the_published_page_boundaries() {
    let tokens: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
    let keys = BlockKeyGenerator::new(4, 0).generate_block_keys(&tokens);
    let group = CacheGroup {
        group_idx: 0,
        block_size: 64,
        kind: GroupRule::SlidingWindow { window_tokens: 128 },
    };
    let tail: Vec<SequenceHash> = keys.iter().skip(15).step_by(16).copied().collect();
    // The three pages that hold the window ending at the last full block.
    let tail = &tail[tail.len() - 3..];
    let mut store = gpu_store(0, "sliding_window_mla", 64, &tokens[..1024], tail);
    store.kv_cache_spec_sliding_window = Some(128);
    let batch = event_batch(vec![KVEvent::BlockStored(store)]);
    // Three published pages are three claims, not sixteen chain steps.
    for (capacity, result, hit) in [(3, Ok(()), 960), (2, Err(SourceError::Capacity), 0)] {
        let index = SourceBinding::new(target(), vec![group], 4, 0, capacity).unwrap();
        index.begin_empty(0);
        assert_eq!(index.apply_batch(0, &batch), result);
        assert_eq!(index.reusable_tokens(&tokens, usize::MAX), hit);
    }
}

/// The event shape a DeepSeek V4.1 layout publishes: one 128-token
/// compressed group carries the whole prefix and the sliding groups publish
/// nothing, so the layout charges their shared replay window instead.
#[test]
fn bounded_replay_credits_one_window_below_the_engine() {
    let tokens: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
    let keys = BlockKeyGenerator::new(128, 0).generate_block_keys(&tokens);
    let groups = vec![CacheGroup {
        group_idx: 5,
        block_size: 128,
        kind: GroupRule::CompressedFullAttention {
            tokens_per_state: 1,
        },
    }];
    let batch = event_batch(vec![KVEvent::BlockStored(gpu_store(
        5,
        "mla_attention",
        128,
        &tokens,
        &keys,
    ))]);
    // The restorable boundary is 896. A larger replay window must saturate
    // at zero credit without rejecting an otherwise valid declaration.
    for (replay_tokens, credit) in [(128, 768), (u32::MAX, 0)] {
        let index = SourceBinding::new(target(), groups.clone(), 128, 0, 4096)
            .unwrap()
            .with_replay_tokens(replay_tokens);
        index.begin_empty(0);
        assert_eq!(index.apply_batch(0, &batch), Ok(()));
        assert_eq!(index.reusable_tokens(&tokens, usize::MAX), credit);
        let mut cache = crate::kv_index::RequestHashCache::for_single_target();
        cache.count_replay_pages = true;
        assert_eq!(
            index
                .credit_parts(None, &tokens, usize::MAX, &mut cache)
                .map(|(tokens, tier, _)| (tokens, tier)),
            Some((896, crate::kv_index::StorageTier::Device))
        );
    }
}

#[test]
fn another_dp_rank_never_answers_for_this_rank() {
    use crate::policies::TieredMatchProvider;

    let index = fresh_index(8);
    index.apply_payload(0, &qwen35_payload()).unwrap();
    // Each rank serves independently; its events, ownership and history
    // belong to that rank only.
    assert!(!TieredMatchProvider::matches(&index, &prompt(), 528, None, 0).is_empty());
    assert!(TieredMatchProvider::matches(&index, &prompt(), 528, None, 1).is_empty());
}
