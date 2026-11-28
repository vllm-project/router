#![cfg(all(test, feature = "cpu", feature = "ready"))]

use std::sync::Arc;

use super::*;
use crate::kv_index::subscriber::EventSource;
use crate::kv_index::types::TierBits;
use crate::kv_index::{RoutableTarget, StorageTier};

fn target(name: &str, rank: u32) -> RoutableTarget {
    RoutableTarget {
        instance_id: Arc::from(name),
        dp_rank: rank,
    }
}

#[test]
fn aligned_tail_requires_dense_and_window_guard_pages() {
    let mut groups = [
        group(0, 560, GroupRule::Checkpoint, &[(3, 4), (7, 8)]),
        group(5, 560, GroupRule::FullAttention, &[(0, 10)]),
        group(6, 560, GroupRule::FullAttention, &[(0, 10)]),
        group(
            7,
            64,
            GroupRule::SlidingWindow {
                window_tokens: 2048,
            },
            &[(3, 80)],
        ),
    ];
    let tail = AlignedTailDrop {
        groups: vec![5, 6, 7],
        alignment: 2240,
    };
    let hit = |groups: &[GroupEvidence], cap, legacy| {
        super::restore::aligned_tail_boundary(groups, cap, &tail, legacy)
    };
    assert_eq!(hit(&groups, 2799, true), 0);
    assert_eq!(hit(&groups, 2800, true), 2240);
    assert_eq!(hit(&groups, 4480, true), 2240);
    assert_eq!(hit(&groups, 4480, false), 4480);

    groups[2].segments = vec![(0, 4)];
    assert_eq!(hit(&groups, 2800, true), 0);
    groups[2].segments = vec![(0, 10)];
    groups[3].segments = vec![(3, 35)];
    assert_eq!(hit(&groups, 2800, true), 0);
    groups[3].segments = vec![(3, 36)];
    assert_eq!(hit(&groups, 2800, true), 2240);
}

fn page_view(group: &[TierBits]) -> crate::kv_index::storage::PageTiers<'_> {
    let prefix = group.first().copied().unwrap_or_default();
    let prefix_len = group.iter().take_while(|&&bits| bits == prefix).count();
    crate::kv_index::storage::PageTiers {
        prefix_len,
        prefix,
        tail: (&group[prefix_len..]).into(),
    }
}

#[test]
fn fine_grained_tail_drop_follows_dense_lookup_batches() {
    use StorageTier::{Device, HostPinned};

    let groups = [0, 1].map(|group_idx| CacheGroup {
        group_idx,
        block_size: 64,
        kind: GroupRule::FullAttention,
    });
    let mut claims = [[TierBits::default(); 8]; 2];
    for group in &mut claims {
        group[3] = TierBits(TierBits::bit(Device));
    }
    claims[0][6] = TierBits(TierBits::bit(Device));
    claims[1][5] = TierBits(TierBits::bit(HostPinned));
    let views = claims.each_ref().map(|pages| page_view(pages));
    let dense = |positions| DenseLookup {
        positions,
        drop_tail: true,
    };
    let credit = |dense_lookups: &[DenseLookup]| {
        credit_from_pages(
            &groups,
            &views,
            16,
            127,
            0,
            TailDrop {
                fine_grained: true,
                dense_lookups,
                proof_within_cap: true,
                ..Default::default()
            },
        )
    };
    // The installed vLLM 0.30 coordinator matches 48, 80 and 48 tokens
    // for these three plans with the same underlying cache pages.
    assert_eq!(credit(&[dense(vec![0, 1])]), (48, Some(Device)));
    assert_eq!(
        credit(&[dense(vec![0]), dense(vec![1])]),
        (80, Some(HostPinned))
    );
    assert_eq!(
        credit(&[dense(vec![1]), dense(vec![0])]),
        (48, Some(Device))
    );
}

#[test]
fn full_attention_rounding_preserves_gaps_between_same_tier_pages() {
    use StorageTier::{Device, HostPinned};

    let group = CacheGroup {
        group_idx: 0,
        block_size: 32,
        kind: GroupRule::FullAttention,
    };
    // Each fact is at the end of a 32-token page in a 16-token hash chain.
    let claims: Vec<_> = [Device, Device, HostPinned, Device, Device, Device]
        .into_iter()
        .flat_map(|tier| [TierBits::default(), TierBits(TierBits::bit(tier))])
        .collect();
    let hits = full_attention_matches(
        &target("worker", 0),
        &group,
        page_view(&claims),
        16,
        64,
        192,
    );
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].tier, Device);
    assert_eq!(hits[0].segments, [(0, 1), (2, 3)]);
}

// The index tests cover path/hash equivalence; restoration only needs page facts.
#[test]
fn restore_coverage_unites_local_tiers_and_excludes_external_pages() {
    use StorageTier::{Device, Disk, External, HostPinned};

    let full = CacheGroup {
        group_idx: 0,
        block_size: 16,
        kind: GroupRule::FullAttention,
    };
    let claims = vec![vec![
        Some(Device),
        Some(HostPinned),
        Some(Disk),
        Some(External),
    ]];
    assert_eq!(
        credit_from_groups(&[full], &claims, 16, 64, 0),
        (48, Some(Disk))
    );
    for mask in 0..16 {
        let expected = [Device, HostPinned, Disk]
            .into_iter()
            .find(|tier| mask & TierBits::bit(*tier) != 0);
        assert_eq!(
            credit_from_pages(
                &[full],
                &[page_view(&[TierBits(mask)])],
                16,
                16,
                0,
                TailDrop::default(),
            ),
            (if expected.is_some() { 16 } else { 0 }, expected),
            "residency mask {mask}"
        );
    }
    let partial = CacheGroup {
        block_size: 32,
        ..full
    };
    for (pages, expected) in [
        (
            vec![None, Some(Device), Some(HostPinned)],
            (48, Some(HostPinned)),
        ),
        (vec![None, None, Some(HostPinned)], (0, None)),
    ] {
        assert_eq!(
            credit_from_groups(&[partial], &[pages], 16, 48, 0),
            expected
        );
    }
    // Both runs end inside a physical block. The partial checkpoint after
    // the gap determines the restore boundary and tier before replay.
    let checkpoint = CacheGroup {
        group_idx: 1,
        kind: GroupRule::Checkpoint,
        ..partial
    };
    let dense = vec![
        Some(Device),
        Some(Device),
        Some(Device),
        Some(HostPinned),
        Some(Device),
    ];
    let sparse = vec![None, None, None, Some(HostPinned), Some(HostPinned)];
    assert_eq!(
        credit_from_groups(&[partial, checkpoint], &[dense, sparse], 16, 80, 16),
        (64, Some(HostPinned))
    );
}

fn credit_from_groups(
    groups: &[CacheGroup],
    claims: &[Vec<Option<StorageTier>>],
    hash_unit: usize,
    cap: u64,
    replay: u32,
) -> (u64, Option<StorageTier>) {
    let pages: Vec<_> = claims
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|tier| TierBits(tier.map_or(0, TierBits::bit)))
                .collect::<Vec<_>>()
        })
        .collect();
    let views: Vec<_> = pages.iter().map(|pages| page_view(pages)).collect();
    credit_from_pages(groups, &views, hash_unit, cap, replay, TailDrop::default())
}

#[test]
fn group_evidence_combines_partial_checkpoint_and_tier() {
    use StorageTier::{Device, HostPinned};

    let groups = [
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
    ];
    let mut claims = vec![
        vec![Some(Device); 4],
        vec![None, Some(Device), Some(HostPinned), None],
    ];
    assert_eq!(
        credit_from_groups(&groups[..1], &claims[..1], 8, 32, 0),
        (32, Some(Device))
    );
    assert_eq!(
        credit_from_groups(&groups, &claims, 8, 32, 0),
        (24, Some(HostPinned))
    );
    assert_eq!(
        credit_from_groups(&groups, &claims, 8, 23, 0),
        (16, Some(Device))
    );
    claims[1][2] = None;
    assert_eq!(
        credit_from_groups(&groups, &claims, 8, 32, 0),
        (16, Some(Device))
    );
}

#[test]
fn partial_restore_does_not_charge_a_later_block_endpoint() {
    use StorageTier::{Device, HostPinned};

    let dense = CacheGroup {
        group_idx: 0,
        block_size: 16,
        kind: GroupRule::FullAttention,
    };
    assert_eq!(
        credit_from_groups(&[dense], &[vec![Some(Device), Some(HostPinned)]], 8, 8, 0),
        (8, Some(Device))
    );
}

#[test]
fn replay_prices_groups_at_the_restore_boundary() {
    use StorageTier::{Device, HostPinned};

    let groups = [
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
        CacheGroup {
            group_idx: 2,
            block_size: 16,
            kind: GroupRule::SlidingWindow { window_tokens: 32 },
        },
    ];
    let claims = vec![
        vec![Some(Device); 4],
        vec![None, None, None, Some(HostPinned)],
        vec![None, None, Some(Device), Some(HostPinned)],
    ];
    assert_eq!(
        credit_from_groups(&groups, &claims, 16, 64, 16),
        (48, Some(HostPinned))
    );
    assert_eq!(
        credit_from_groups(&groups, &claims, 16, 64, 0),
        (64, Some(HostPinned))
    );
    assert_eq!(credit_from_groups(&groups, &claims, 16, 64, 64), (0, None));
}

#[test]
fn mtp_tail_proof_restores_the_mamba_checkpoint_and_prices_the_dense_page() {
    use StorageTier::{Device, HostPinned};

    let groups = [
        CacheGroup {
            group_idx: 0,
            block_size: 832,
            kind: GroupRule::Checkpoint,
        },
        CacheGroup {
            group_idx: 3,
            block_size: 832,
            kind: GroupRule::FullAttention,
        },
    ];
    let mut claims = [
        vec![TierBits::default(); 112],
        vec![TierBits::default(); 112],
    ];
    for (group, end, tier) in [
        (0, 832, Device),
        (0, 1776, Device),
        (1, 832, Device),
        (1, 1664, Device),
        (1, 1792, HostPinned),
    ] {
        claims[group][end / 16 - 1] = TierBits(TierBits::bit(tier));
    }
    let credit = |claims: &[Vec<TierBits>], cap, proof_within_cap| {
        let views: Vec<_> = claims.iter().map(|pages| page_view(pages)).collect();
        credit_from_pages(
            &groups,
            &views,
            16,
            cap,
            0,
            TailDrop {
                group: Some(3),
                fine_grained: true,
                proof_within_cap,
                ..Default::default()
            },
        )
    };
    assert_eq!(credit(&claims, 1792, false), (1776, Some(HostPinned)));
    assert_eq!(credit(&claims, 1776, false), (1776, Some(HostPinned)));
    assert_eq!(credit(&claims, 1775, false), (832, Some(Device)));
    // In vLLM 0.30 and 0.31 the final partial page cannot prove a hit
    // when its hash boundary exceeds the Prefill lookup limit.
    assert_eq!(credit(&claims, 1791, true), (832, Some(Device)));
    assert_eq!(credit(&claims, 1791, false), (1776, Some(HostPinned)));
    claims[0][1792 / 16 - 1] = TierBits(TierBits::bit(Device));
    // Both groups reach 1792, below the cap, so the dense proof still loses one unit.
    assert_eq!(credit(&claims, 1808, false), (1776, Some(HostPinned)));
    claims[1][1792 / 16 - 1] = TierBits::default();
    assert_eq!(credit(&claims, 1792, false), (832, Some(Device)));
}

fn group(id: u32, block_size: u32, kind: GroupRule, segments: &[(u32, u32)]) -> GroupEvidence {
    GroupEvidence {
        group: CacheGroup {
            group_idx: id,
            block_size,
            kind,
        },
        segments: segments.to_vec(),
        partial_ends: Vec::new(),
    }
}

#[test]
fn invalid_page_units_and_overflow_do_not_produce_credit() {
    use crate::kv_index::storage::PageTiers;

    let target = target("worker", 0);
    let dense = group(0, 16, GroupRule::FullAttention, &[]).group;
    let invalid_group = CacheGroup {
        block_size: 0,
        ..dense
    };
    let pages = |prefix_len, tail| PageTiers {
        prefix_len,
        prefix: TierBits(TierBits::bit(StorageTier::Device)),
        tail,
    };
    for (group, hash_unit, score_unit) in [(dense, 0, 16), (dense, 16, 0), (invalid_group, 16, 16)]
    {
        assert!(full_attention_matches(
            &target,
            &group,
            pages(1, Default::default()),
            hash_unit,
            score_unit,
            16,
        )
        .is_empty());
    }
    for (group, hash_unit, len) in [
        (dense, 0, 1),
        (invalid_group, 16, 1),
        (dense, 16, u32::MAX as usize),
    ] {
        assert_eq!(
            credit_from_pages(
                &[group],
                &[pages(len, vec![TierBits::default()].into())],
                hash_unit,
                16,
                0,
                TailDrop::default(),
            ),
            (0, None)
        );
    }
    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        credit_from_pages(
            &[dense],
            &[pages(u32::MAX as usize + 1, Default::default())],
            16,
            16,
            0,
            TailDrop::default(),
        ),
        (0, None)
    );
}

#[test]
fn restore_rules_respect_the_cap_and_page_gaps() {
    let dense = [(0, 4)];
    let cases = [
        (GroupRule::FullAttention, vec![(0, 2)], 8, 4),
        (GroupRule::FullAttention, vec![(1, 4)], 8, 0),
        (
            GroupRule::CompressedFullAttention {
                tokens_per_state: 1,
            },
            vec![(0, 4)],
            7,
            6,
        ),
        (GroupRule::Checkpoint, vec![(2, 4)], 7, 6),
        (GroupRule::Checkpoint, vec![(1, 2), (3, 4)], 7, 4),
        (
            GroupRule::Checkpoint,
            vec![(3, 4), (0, 1), (1, 3), (1, 2)],
            7,
            6,
        ),
        (
            GroupRule::SlidingWindow { window_tokens: 5 },
            vec![(0, 1)],
            8,
            2,
        ),
        (
            GroupRule::SlidingWindow { window_tokens: 5 },
            vec![(1, 3)],
            8,
            6,
        ),
        (
            GroupRule::SlidingWindow { window_tokens: 5 },
            vec![(1, 3)],
            5,
            0,
        ),
        (
            GroupRule::SlidingWindow { window_tokens: 5 },
            vec![(1, 2)],
            8,
            0,
        ),
    ];
    for (kind, segments, cap, expected) in cases {
        let groups = [
            group(0, 2, GroupRule::FullAttention, &dense),
            group(1, 2, kind, &segments),
        ];
        assert_eq!(
            common_reusable_tokens(&groups, cap),
            expected,
            "kind={kind:?} segments={segments:?} cap={cap}"
        );
    }
}

#[test]
fn missing_sliding_pages_never_inherit_dense_group_credit() {
    let dense = [(0, 4)];
    let sliding = group(1, 16, GroupRule::SlidingWindow { window_tokens: 32 }, &[]);
    let groups = [group(0, 16, GroupRule::FullAttention, &dense), sliding];
    assert_eq!(common_reusable_tokens(&groups, 64), 0);
}

#[test]
fn interleaved_checkpoints_descend_to_zero() {
    let odd: Vec<_> = (0..64).step_by(2).map(|i| (i, i + 1)).collect();
    let even: Vec<_> = (1..64).step_by(2).map(|i| (i, i + 1)).collect();
    let groups = [
        group(0, 1, GroupRule::Checkpoint, &odd),
        group(1, 1, GroupRule::Checkpoint, &even),
    ];
    assert_eq!(common_reusable_tokens(&groups, 64), 0);
}

#[test]
fn sliding_window_is_rechecked_after_checkpoint_rewinds() {
    let checkpoints = [(3, 4)];
    let tail = [(4, 6)];
    let groups = [
        group(0, 16, GroupRule::Checkpoint, &checkpoints),
        group(1, 16, GroupRule::SlidingWindow { window_tokens: 32 }, &tail),
    ];
    assert_eq!(common_reusable_tokens(&groups, 96), 0);
}

/// vLLM also caches a prefix ending inside a page (`cache_partial_block`).
/// The boundary is the partial token end, not the page end.
#[test]
fn a_partial_page_boundary_is_restorable() {
    // Page 576, a partial entry at 560: no page-level hash exists yet.
    let mut full_group = group(0, 576, GroupRule::FullAttention, &[]);
    full_group.partial_ends = vec![560];
    assert_eq!(common_reusable_tokens(&[full_group], 1024), 560);
}

#[test]
fn partial_checkpoint_respects_cap_and_does_not_need_earlier_pages() {
    let mut checkpoint = group(0, 576, GroupRule::Checkpoint, &[]);
    checkpoint.partial_ends = vec![560, 1136];
    assert_eq!(
        common_reusable_tokens(std::slice::from_ref(&checkpoint), 544),
        0
    );
    assert_eq!(
        common_reusable_tokens(std::slice::from_ref(&checkpoint), 1100),
        560
    );
    assert_eq!(
        common_reusable_tokens(std::slice::from_ref(&checkpoint), 1152),
        1136
    );
}

#[test]
fn dense_partial_requires_earlier_full_pages() {
    let mut full = group(0, 576, GroupRule::FullAttention, &[]);
    full.partial_ends = vec![1136];
    assert_eq!(common_reusable_tokens(&[full], 1152), 0);
    let hits = [(0, 1)];
    let mut full = group(0, 576, GroupRule::FullAttention, &hits);
    full.partial_ends = vec![1136];
    assert_eq!(common_reusable_tokens(&[full], 1152), 1136);
}

#[test]
fn sparse_checkpoints_and_different_block_sizes_use_the_same_boundary() {
    let full = [(0, 6)]; // 0..96, blocks of 16
    let state = [(0, 2)]; // checkpoints at 32 and 64
    let groups = [
        group(0, 16, GroupRule::FullAttention, &full),
        group(1, 32, GroupRule::Checkpoint, &state),
    ];
    assert_eq!(common_reusable_tokens(&groups, 127), 64);
    assert_eq!(common_reusable_tokens(&groups, 63), 32);
    assert_eq!(common_reusable_tokens(&groups, 31), 0);
}

#[test]
fn missing_full_blocks_or_checkpoint_falls_back_to_an_earlier_boundary() {
    let full = [(0, 3), (4, 6)]; // gap at 48..64
    let state = [(0, 1), (2, 3)];
    let groups = [
        group(0, 16, GroupRule::FullAttention, &full),
        group(1, 32, GroupRule::Checkpoint, &state),
    ];
    assert_eq!(common_reusable_tokens(&groups, 128), 32);
}

#[test]
fn sliding_window_needs_only_the_required_local_interval() {
    let full = [(0, 8)];
    let window = [(5, 8)];
    let groups = [
        group(0, 16, GroupRule::FullAttention, &full),
        group(
            1,
            16,
            GroupRule::SlidingWindow { window_tokens: 33 },
            &window,
        ),
    ];
    assert_eq!(common_reusable_tokens(&groups, 128), 128);
    // The last token cannot be reused, so both groups report 7 blocks. The
    // sliding group still holds two contiguous blocks below that boundary,
    // which is what its window needs, so the reduced boundary stands.
    assert_eq!(common_reusable_tokens(&groups, 127), 112);
}

/// The sliding group's retained tail can sit entirely above a boundary
/// another group pulled down. The window that ends at that boundary is then
/// absent, so there is no shared restore point and no credit.
#[test]
fn a_reduced_boundary_that_loses_the_sliding_window_credits_nothing() {
    let full = [(0, 4)]; // 0..64
    let window = [(6, 8)]; // 96..128, a 32-token window
    let groups = [
        group(0, 16, GroupRule::FullAttention, &full),
        group(
            1,
            16,
            GroupRule::SlidingWindow { window_tokens: 32 },
            &window,
        ),
    ];
    assert_eq!(common_reusable_tokens(&groups, 143), 0);
}

/// A sliding group whose resident run starts at the root has no earlier
/// state the window could be missing, so a prefix shorter than the window is
/// still reusable: the engine restores the whole run.
#[test]
fn a_root_anchored_prefix_shorter_than_the_window_is_reusable() {
    let full = [(0, 1)];
    let window = [(0, 1)];
    let groups = [
        group(0, 64, GroupRule::FullAttention, &full),
        group(
            1,
            64,
            GroupRule::SlidingWindow { window_tokens: 128 },
            &window,
        ),
    ];
    // A 65-token request keeps one 64-token block, well inside the window.
    assert_eq!(common_reusable_tokens(&groups, 64), 64);
    assert_eq!(common_reusable_tokens(&groups, 128), 64);
}

/// A sliding group whose run sits below the boundary still supports it,
/// because the engine's lookup accepts any run of the window's blocks under
/// the candidate length.
#[test]
fn a_sliding_run_below_the_boundary_keeps_the_credit() {
    let full = [(0, 6)]; // 0..96
    let window = [(4, 8)]; // 64..128, a 32-token window
    let groups = [
        group(0, 16, GroupRule::FullAttention, &full),
        group(
            1,
            16,
            GroupRule::SlidingWindow { window_tokens: 32 },
            &window,
        ),
    ];
    assert_eq!(common_reusable_tokens(&groups, 143), 96);
}

mod declarations {
    use super::*;
    use crate::kv_index::SourceBinding;

    fn bind_layout(value: &serde_json::Value, topic: &str) -> Result<SourceBinding, &'static str> {
        let target = target("worker", 0);
        let source = EventSource {
            source: target.instance_id.clone(),
            dp_rank: target.dp_rank,
            pub_endpoint: "inproc://publisher".into(),
            replay_endpoint: Some("inproc://replay".into()),
            topic: topic.into(),
            hwm: None,
        };
        serde_json::from_value::<CacheCapabilities>(value.clone())
            .expect("layout")
            .index_for_source(&source, target, 16)
    }

    #[test]
    fn a_declared_single_attention_layout_is_admitted() {
        let group = serde_json::json!({
            "group_idx": 5, "kind": "full_attention", "spec_type": "FullAttentionSpec",
            "block_size": 128, "sliding_window": null, "prefix_cacheable": true,
            "prefix_replay_tokens": 0, "mamba_cache_mode": null,
        });
        let mut value = serde_json::json!({"schema_version": 1, "ranks": [{
            "rank": 0, "model": "v41", "weight_version": "default",
            "publisher_topic": "epoch-1",
            "instance_topic": true, "hash_algorithm": "sha256_cbor",
            "full_hashes": true,
            "root_hash": "4e1195df020de59e0d65a33a4279f1183e7ae4e5d980e309f8b55adff2e61c3e",
            "hash_block_size": 128, "hash_lookahead_tokens": 0,
            "speculative_tokens": 0, "prefix_caching": true,
            "declared_groups": 1,
            "groups": [group]
        }]});
        assert!(bind_layout(&value, "epoch-1").is_ok());
        // Without the explicit total the same layout is indistinguishable
        // from a truncated multi-group declaration, so it stays rejected.
        value["ranks"][0]["declared_groups"] = serde_json::Value::Null;
        assert!(bind_layout(&value, "epoch-1").is_err());
        // A declared single group that is not an attention group is useless.
        value["ranks"][0]["groups"][0]["kind"] = serde_json::json!("mamba");
        value["ranks"][0]["groups"][0]["spec_type"] = serde_json::json!("MambaSpec");
        value["ranks"][0]["groups"][0]["mamba_cache_mode"] = serde_json::json!("align");
        value["ranks"][0]["declared_groups"] = serde_json::json!(1);
        assert!(bind_layout(&value, "epoch-1").is_err());
    }

    fn qwen_layout() -> serde_json::Value {
        let group = |idx, kind, spec, mode| {
            serde_json::json!({
                "group_idx": idx, "kind": kind, "spec_type": spec, "block_size": 528,
                "sliding_window": null, "prefix_cacheable": true,
                "prefix_replay_tokens": 0, "mamba_cache_mode": mode,
            })
        };
        serde_json::json!({"schema_version": 1, "ranks": [{
            "rank": 0, "model": "qwen", "weight_version": "default",
            "publisher_topic": "epoch-1", "instance_topic": true,
            "hash_algorithm": "sha256_cbor", "full_hashes": true,
            "root_hash": "4e1195df020de59e0d65a33a4279f1183e7ae4e5d980e309f8b55adff2e61c3e",
            "hash_block_size": 528, "hash_lookahead_tokens": 0,
            "speculative_tokens": 0, "prefix_caching": true,
            "groups": [group(0, "mamba", "MambaSpec", Some("align")),
                group(1, "mamba", "MambaSpec", Some("align")),
                group(2, "mamba", "MambaSpec", Some("align")),
                group(3, "full_attention", "FullAttentionSpec", None)]
        }]})
    }

    #[test]
    fn fine_grained_speculation_requires_a_complete_dense_lookup_order() {
        let mut value = qwen_layout();
        let rank = &mut value["ranks"][0];
        let mut second_dense = rank["groups"][3].clone();
        second_dense["group_idx"] = serde_json::json!(4);
        rank["groups"].as_array_mut().unwrap().push(second_dense);
        rank["declared_groups"] = serde_json::json!(5);
        rank["hash_block_size"] = serde_json::json!(16);
        rank["partial_hash_hits"] = serde_json::json!(true);
        rank["speculative_tokens"] = serde_json::json!(8);
        rank["tail_drop_groups"] = serde_json::json!([3, 4]);
        assert!(bind_layout(&value, "epoch-1").is_err());
        for (lookups, accepted) in [
            (serde_json::json!([[3, 4]]), true),
            (serde_json::json!([[4], [3]]), true),
            (serde_json::json!([[3], [3]]), false),
            (serde_json::json!([[3]]), false),
            (serde_json::json!([[0, 3, 4]]), false),
            (serde_json::json!([[], [3, 4]]), false),
        ] {
            value["ranks"][0]["dense_lookup_groups"] = lookups;
            assert_eq!(bind_layout(&value, "epoch-1").is_ok(), accepted);
        }
    }

    #[test]
    fn qwen_layout_restores_prefix_and_respects_pd_prefill_boundary() {
        use crate::config::KvConnector;

        let value = qwen_layout();
        let prompt: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
        let payload = crate::kv_index::source::binding::tests::qwen35_payload();
        for (connector, at_boundary) in [
            (None, 528),
            (Some(KvConnector::Nixl), 0),
            (Some(KvConnector::Mooncake), 0),
            (Some(KvConnector::MooncakeV2), 0),
        ] {
            let binding = bind_layout(&value, "epoch-1")
                .unwrap()
                .with_pd_prefill(connector, false, None);
            binding.begin_empty(0);
            binding.apply_payload(0, &payload).unwrap();
            assert_eq!(
                binding.reusable_tokens(&prompt[..529], usize::MAX),
                at_boundary
            );
            assert_eq!(binding.reusable_tokens(&prompt[..530], usize::MAX), 528);
        }
    }

    #[test]
    fn nixl_decode_replaces_a_partial_checkpoint_only_for_a_longer_pull() {
        let mut value = qwen_layout();
        let rank = &mut value["ranks"][0];
        rank["hash_block_size"] = 16.into();
        rank["partial_hash_hits"] = true.into();
        for group in rank["groups"].as_array_mut().unwrap() {
            group["block_size"] = 576.into();
        }
        let index = bind_layout(&value, "epoch-1").unwrap();
        assert_eq!(index.nixl_decode_credit(1801, 1792), 1728);
        assert_eq!(index.nixl_decode_credit(1793, 1792), 1792);
        assert_eq!(index.nixl_decode_credit(1801, 1728), 1728);
    }

    #[test]
    fn layout_rejects_inconsistent_groups_publisher_and_hash_contract() {
        let mut value = qwen_layout();
        // Reject malformed group sizes and identities during source admission.
        for (field, invalid) in [("block_size", 0), ("group_idx", 0)] {
            let mut invalid_layout = value.clone();
            invalid_layout["ranks"][0]["groups"][1][field] = serde_json::json!(invalid);
            assert!(bind_layout(&invalid_layout, "epoch-1").is_err());
        }
        for (field, invalid) in [
            ("publisher_topic", serde_json::json!("other")),
            ("declared_groups", serde_json::json!(3)),
            ("full_hashes", serde_json::json!(false)),
            ("speculative_tokens", serde_json::json!(1)),
            ("root_hash", serde_json::json!("bad")),
        ] {
            let saved = value["ranks"][0][field].clone();
            value["ranks"][0][field] = invalid;
            assert!(bind_layout(&value, "epoch-1").is_err(), "{field}");
            value["ranks"][0][field] = saved;
        }
        value["ranks"][0]["groups"][0]["prefix_replay_tokens"] = serde_json::json!(128);
        assert!(bind_layout(&value, "epoch-1").is_err());
    }

    #[test]
    fn layout_validates_speculative_contract_and_resolved_seed() {
        let mut spec = qwen_layout();
        spec["ranks"][0]["speculative_tokens"] = serde_json::json!(3);
        let unsupported_speculation = Some("unsupported speculative cache contract");
        for (lookahead, tail_groups, expected_error) in [
            (0, serde_json::json!(null), unsupported_speculation),
            (0, serde_json::json!([]), None),
            (0, serde_json::json!([3]), None),
            (
                0,
                serde_json::json!([0]),
                Some("unsupported tail-drop layout. Set tail_drop_groups to the effective groups and use [] when the worker skips the drop"),
            ),
            (1, serde_json::json!(null), None),
            (1, serde_json::json!([3]), unsupported_speculation),
        ] {
            spec["ranks"][0]["hash_lookahead_tokens"] = serde_json::json!(lookahead);
            spec["ranks"][0]["tail_drop_groups"] = tail_groups;
            assert_eq!(bind_layout(&spec, "epoch-1").err(), expected_error, "{spec}");
        }
        let mut seeded = qwen_layout();
        let root_hash = seeded["ranks"][0]["root_hash"].clone();
        seeded["ranks"][0]["hash_seed"] = serde_json::json!("0");
        assert!(bind_layout(&seeded, "epoch-1").is_ok());
        seeded["ranks"][0]
            .as_object_mut()
            .unwrap()
            .remove("root_hash");
        assert!(bind_layout(&seeded, "epoch-1").is_ok());
        seeded["ranks"][0]["hash_seed"] = serde_json::json!("");
        assert!(bind_layout(&seeded, "epoch-1").is_err());
        seeded["ranks"][0]["hash_seed"] = serde_json::json!("42");
        seeded["ranks"][0]["root_hash"] = root_hash;
        assert!(bind_layout(&seeded, "epoch-1").is_err());
    }

    /// DeepSeek V4 can declare MLA pages without `tokens_per_state` next to
    /// sliding-window groups on a 4-token hash chain; that layout has to bind.
    #[test]
    fn deepseek_v4_layout_binds_without_a_checkpoint_group() {
        let group = |idx, kind, spec, block, window: Option<u32>| {
            serde_json::json!({
                "group_idx": idx, "kind": kind, "spec_type": spec, "block_size": block,
                "sliding_window": window, "prefix_cacheable": true,
                "prefix_replay_tokens": 0, "mamba_cache_mode": null,
            })
        };
        let mut value = serde_json::json!({"schema_version": 1, "ranks": [{
            "rank": 0, "model": "deepseek-v4", "weight_version": "default",
            "publisher_topic": "ds4", "instance_topic": true,
            "hash_algorithm": "sha256_cbor", "root_hash":
                "9bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef0",
            "full_hashes": true, "hash_block_size": 4, "hash_lookahead_tokens": 0,
            "speculative_tokens": 0, "prefix_caching": true,
            "groups": [
                group(0, "sliding_window_mla", "SlidingWindowMLASpec", 64, Some(128)),
                group(1, "sliding_window_mla", "SlidingWindowMLASpec", 64, Some(128)),
                group(2, "mla_attention", "MLAAttentionSpec", 256, None),
                group(3, "sliding_window_mla", "SlidingWindowMLASpec", 4, Some(8)),
                group(4, "sliding_window_mla", "SlidingWindowMLASpec", 8, Some(128)),
            ]
        }]});
        bind_layout(&value, "ds4").expect("deepseek v4 layout must bind");
        value["ranks"][0]["groups"][2]["tokens_per_state"] = "4".into();
        value["ranks"][0]["groups"][2]["cache_role"] = "sparse".into();
        use crate::config::KvConnector;
        for (connector, truncates_compressed, expected) in [
            (Some(KvConnector::Mooncake), None, 16383),
            (Some(KvConnector::Mooncake), Some(false), 16384),
            (Some(KvConnector::Nixl), Some(true), 16383),
            (Some(KvConnector::Nixl), Some(false), 16384),
            (Some(KvConnector::Nixl), None, 16384),
            (None, Some(true), 16384),
        ] {
            let binding = bind_layout(&value, "ds4").unwrap().with_pd_prefill(
                connector,
                true,
                truncates_compressed,
            );
            assert_eq!(binding.reusable_prompt_cap(16385), expected);
            assert_eq!(binding.reusable_prompt_cap(16386), expected + 1);
        }
        let window = &mut value["ranks"][0]["groups"][0];
        window["prefix_cacheable"] = false.into();
        window["prefix_replay_tokens"] = 128.into();
        for (connector, truncates, expected) in [
            (Some(KvConnector::Nixl), true, 1023),
            (Some(KvConnector::Nixl), false, 1024),
            (None, true, 1024),
        ] {
            let binding = bind_layout(&value, "ds4")
                .unwrap()
                .with_pd_prefill(connector, truncates, None);
            assert_eq!(binding.reusable_prompt_cap(1025), expected);
        }
        value["ranks"][0]["groups"][0]["sliding_window"] = 0.into();
        assert!(bind_layout(&value, "ds4").is_err());
        value["ranks"][0]["groups"] = serde_json::json!([]);
        assert!(bind_layout(&value, "ds4").is_err());
    }

    #[test]
    fn qwen4_layout_admits_compressed_group_and_skips_the_scratch_ring() {
        let group = |idx, kind, spec, extra: serde_json::Value| {
            let mut value = serde_json::json!({
                "group_idx": idx, "kind": kind, "spec_type": spec, "block_size": 320,
                "sliding_window": null, "prefix_cacheable": true,
                "prefix_replay_tokens": 0, "mamba_cache_mode": null,
                "tokens_per_state": "1",
                "cache_role": null,
            });
            value
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            value
        };
        let mamba = || {
            group(
                0,
                "mamba",
                "MambaSpec",
                serde_json::json!({"mamba_cache_mode": "align", "tokens_per_state": "-1", "block_size": 320}),
            )
        };
        // Captured from a Qwen4Exp (QSA) vLLM 0.33.0 probe: three GDN groups,
        // one per-request key ring that emits no block events, and one
        // compressed MLA group whose states cover four tokens each.
        let mut value = serde_json::json!({"schema_version": 1, "ranks": [{
            "rank": 0, "model": "qwen4", "weight_version": "default",
            "publisher_topic": "kv-events", "instance_topic": true,
            "hash_algorithm": "sha256_cbor", "full_hashes": true,
            "root_hash": "4e1195df020de59e0d65a33a4279f1183e7ae4e5d980e309f8b55adff2e61c3e",
            "hash_block_size": 320, "hash_lookahead_tokens": 0,
            "speculative_tokens": 0, "prefix_caching": true,
            "groups": [
                mamba(),
                group(1, "mamba", "MambaSpec", serde_json::json!({"mamba_cache_mode": "align", "tokens_per_state": "-1"})),
                group(2, "mamba", "MambaSpec", serde_json::json!({"mamba_cache_mode": "align", "tokens_per_state": "-1"})),
                group(3, "unknown", "CircularBufferSpec", serde_json::json!({"prefix_cacheable": false, "block_size": 4})),
                group(4, "mla_attention", "MLAAttentionSpec", serde_json::json!({"tokens_per_state": "4", "cache_role": "sparse"}))
            ]
        }]});
        assert!(bind_layout(&value, "kv-events").is_ok());

        let mut speculative = value.clone();
        let rank = &mut speculative["ranks"][0];
        rank["speculative_tokens"] = 3.into();
        rank["tail_drop_groups"] = serde_json::json!([3, 4]);
        let binding = bind_layout(&speculative, "kv-events").unwrap();
        assert_eq!(binding.tail_drop_group, Some(4));
        let evidence: Vec<_> = binding
            .groups
            .into_iter()
            .map(|group| GroupEvidence {
                group,
                segments: vec![(0, 2)],
                partial_ends: Vec::new(),
            })
            .collect();
        assert_eq!(
            super::super::restore::restore_boundary(&evidence, 641, Some(4), 0, false, &[]).0,
            320
        );

        let group = &mut value["ranks"][0]["groups"][4];
        let ratio = group["tokens_per_state"].take();
        for role in ["sparse", "indexer"] {
            value["ranks"][0]["groups"][4]["cache_role"] = role.into();
            assert!(matches!(
                bind_layout(&value, "kv-events"),
                Err("MLA cache role requires tokens_per_state")
            ));
        }
        value["ranks"][0]["groups"][4]["tokens_per_state"] = ratio;
        value["ranks"][0]["groups"][4]["cache_role"] = "sparse".into();

        for (path, invalid) in [(4, serde_json::json!("3")), (4, serde_json::json!("field"))] {
            let saved = value["ranks"][0]["groups"][path]["tokens_per_state"].clone();
            value["ranks"][0]["groups"][path]["tokens_per_state"] = invalid;
            assert!(bind_layout(&value, "kv-events").is_err(), "{path}");
            value["ranks"][0]["groups"][path]["tokens_per_state"] = saved;
        }
        value["ranks"][0]["groups"][4]["cache_role"] = serde_json::json!("unknown");
        assert!(bind_layout(&value, "kv-events").is_err());
        value["ranks"][0]["groups"][4]["cache_role"] = serde_json::json!("sparse");
        value["ranks"][0]["groups"][3]["spec_type"] = serde_json::json!("FullAttentionSpec");
        assert!(bind_layout(&value, "kv-events").is_err());
        value["ranks"][0]["groups"][3]["spec_type"] = serde_json::json!("CircularBufferSpec");
        value["ranks"][0]["groups"][4]["kind"] = serde_json::json!("full_attention");
        assert!(bind_layout(&value, "kv-events").is_err());
    }

    #[test]
    fn documented_hy4_flash_layouts_restore_the_expected_prefix() {
        use crate::config::{KvConnector, RoutingMode};
        use crate::kv_index::source::test_support::{event_batch, stored_block};
        use crate::kv_index::wire::{ExternalBlockHash, KVEvent};

        let guide = include_str!("../../../docs/load_balancing/kv_aware.md").replace("\r\n", "\n");
        let worker = "http://prefill";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("layout.json");
        let mode = RoutingMode::VllmPrefillDecode {
            prefill_urls: vec![(worker.into(), None)],
            decode_urls: vec!["http://decode".into()],
            prefill_policy: None,
            decode_policy: None,
            discovery_address: None,
        };
        let source = EventSource {
            source: Arc::from(worker),
            dp_rank: 0,
            pub_endpoint: "inproc://pub".into(),
            replay_endpoint: Some("inproc://replay".into()),
            topic: "epoch".into(),
            hwm: None,
        };
        for (label, page, ratio, hit, pull_hit) in [
            ("GPU · HY4 Flash", 256, 4, 2048, 0),
            ("Ascend A5 · HY4 Flash", 64, 1, 64, 64),
        ] {
            let (_, section) = guide
                .split_once(&format!("<summary>{label}</summary>"))
                .unwrap();
            let (_, example) = section.split_once("\n```json\n").unwrap();
            let rank: serde_json::Value =
                serde_json::from_str(example.split_once("\n```").unwrap().0).unwrap();
            std::fs::write(
                &path,
                serde_json::to_vec(&serde_json::json!({(worker): {"ranks": [rank]}})).unwrap(),
            )
            .unwrap();
            let mut pd = std::collections::HashMap::new();
            crate::kv_index::apply_kv_layout_file(&path, &mode, &mut Default::default(), &mut pd)
                .unwrap();
            let layout = pd.get_mut(worker).unwrap().hybrid.as_mut().unwrap();
            // Discovery supplies the worker model and its live publisher topic.
            layout.ranks[0].model = "hy4-flash".into();
            layout.ranks[0].publisher_topic = Some(source.topic.clone());
            let binding = layout
                .bind_worker_source(&source, "hy4-flash", 128)
                .unwrap();
            assert_eq!(
                binding.groups[0].kind,
                GroupRule::CompressedFullAttention {
                    tokens_per_state: ratio
                }
            );
            let tokens: Vec<u32> = (0..=hit as u32).collect();
            let hashes = binding
                .keys
                .generate_block_keys(&tokens[..hit])
                .iter()
                .map(|hash| ExternalBlockHash::Bytes(hash.to_vec()))
                .collect();
            let store = stored_block(hashes, tokens[..hit].to_vec(), page);
            let events = binding
                .groups
                .iter()
                .map(|group| {
                    let mut store = store.clone();
                    store.group_idx = Some(group.group_idx);
                    if group.group_idx == 0 {
                        store.kv_cache_spec_kind = Some("mla_attention".into());
                    } else {
                        // A sliding group retains only the final two pages in this case.
                        let first = store.block_hashes.len() - 2;
                        store.parent_block_hash = Some(store.block_hashes[first - 1].clone());
                        store.block_hashes = store.block_hashes.split_off(first);
                        store.token_ids = store.token_ids.split_off(first * page as usize);
                        store.kv_cache_spec_kind = Some("sliding_window_mla".into());
                        store.kv_cache_spec_sliding_window = Some(512);
                    }
                    KVEvent::BlockStored(store)
                })
                .collect();
            binding.begin_empty(0);
            binding.apply_batch(0, &event_batch(events)).unwrap();
            assert_eq!(
                binding.reusable_tokens(&tokens, usize::MAX),
                hit as u64,
                "{label}"
            );
            let binding = binding.with_pd_prefill(Some(KvConnector::MooncakeV2), false, None);
            assert_eq!(
                binding.reusable_tokens(&tokens, usize::MAX),
                pull_hit,
                "{label}"
            );
            if pull_hit != 0 {
                let hf_compressed =
                    binding.with_pd_prefill(Some(KvConnector::MooncakeV2), false, Some(true));
                assert_eq!(hf_compressed.reusable_tokens(&tokens, usize::MAX), 0);
            }
        }
    }
}
