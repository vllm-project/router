//! Event-shape admission, separate from structural decoding and transport.
use crate::kv_hash::SequenceHash;
use crate::kv_index::layout::{CacheGroup, GroupRule};
use crate::kv_index::types::StorageTier;
use crate::kv_index::wire::extra_keys_match;
use crate::kv_index::wire::{BlockStored, ExternalBlockHash, KVEvent, KVEventBatch};

pub(super) fn native_device(ownership: Option<&str>, medium: Option<&str>) -> bool {
    ownership.is_none() && matches!(medium, None | Some("GPU"))
}

pub(super) fn sequence_hash(hash: &ExternalBlockHash) -> Option<SequenceHash> {
    match hash {
        ExternalBlockHash::Bytes(bytes) => bytes.as_slice().try_into().ok(),
        ExternalBlockHash::Int(_) => None,
    }
}

pub(super) fn publishes_int_hash(event: &KVEvent) -> bool {
    match event {
        KVEvent::BlockStored(store) => {
            store.ownership.is_none()
                && store
                    .block_hashes
                    .iter()
                    .chain(store.parent_block_hash.iter())
                    .any(|hash| matches!(hash, ExternalBlockHash::Int(_)))
        }
        KVEvent::BlockRemoved(remove) => {
            remove.ownership.is_none()
                && remove
                    .block_hashes
                    .iter()
                    .any(|hash| matches!(hash, ExternalBlockHash::Int(_)))
        }
        KVEvent::AllBlocksCleared(_) => false,
    }
}

fn complete_blocks(store: &BlockStored) -> bool {
    store
        .block_hashes
        .len()
        .checked_mul(store.block_size as usize)
        == Some(store.token_ids.len())
}

/// Return false for a secondary page size that proves no prefix evidence.
pub(super) fn validate_hybrid_store(
    group: &CacheGroup,
    store: &BlockStored,
    hash_block_size: usize,
    lookahead_tokens: usize,
    partial_hash_hits: bool,
) -> Result<bool, ()> {
    if store.locality.as_deref().is_some_and(|s| s != "LOCAL")
        || match &store.lookahead_token_ids {
            None => lookahead_tokens != 0 && !store.block_hashes.is_empty(),
            Some(ahead) => {
                lookahead_tokens == 0
                    || ahead.len() != store.block_hashes.len()
                    || ahead.iter().any(|tokens| tokens.len() != lookahead_tokens)
            }
        }
        || (store.lora_name.is_none()
            && !extra_keys_match(store.extra_keys.as_ref(), store.block_hashes.len(), None))
        || store
            .parent_block_hash
            .as_ref()
            .is_some_and(|hash| sequence_hash(hash).is_none())
        || !event_kind_matches(group.kind, store)
    {
        return Err(());
    }
    let partial = partial_hash_hits
        && matches!(group.kind, GroupRule::FullAttention | GroupRule::Checkpoint)
        && store.block_size < group.block_size
        && store.block_size as usize == hash_block_size
        && store.block_hashes.len() == 1
        && store.token_ids.len() == hash_block_size;
    if store.block_size != group.block_size && !partial {
        tracing::debug!(
            group_idx = group.group_idx,
            declared = group.block_size,
            published = store.block_size,
            "hybrid group published a secondary page size"
        );
        return Ok(false);
    }
    let block_size = store.block_size as usize;
    if block_size == 0
        || store.token_ids.is_empty()
        || !store.token_ids.len().is_multiple_of(block_size)
        || !match group.kind {
            GroupRule::Checkpoint | GroupRule::SlidingWindow { .. } => {
                store.block_hashes.len() <= store.token_ids.len() / block_size
            }
            _ => complete_blocks(store),
        }
    {
        return Err(());
    }
    Ok(true)
}

pub(in crate::kv_index) fn event_kind_matches(kind: GroupRule, store: &BlockStored) -> bool {
    let spec = store.kv_cache_spec_kind.as_deref();
    match kind {
        GroupRule::FullAttention => {
            matches!(spec, Some("full_attention") | Some("mla_attention"))
                && store.kv_cache_spec_sliding_window.is_none()
        }
        GroupRule::CompressedFullAttention { .. } => {
            spec == Some("mla_attention") && store.kv_cache_spec_sliding_window.is_none()
        }
        GroupRule::Checkpoint => {
            spec == Some("mamba") && store.kv_cache_spec_sliding_window.is_none()
        }
        GroupRule::SlidingWindow { window_tokens } => {
            matches!(spec, Some("sliding_window" | "sliding_window_mla"))
                && store.kv_cache_spec_sliding_window.map(u64::from) == Some(window_tokens)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::kv_index) enum FullAttentionRejectionReason {
    ExtraKeys,
    Layout,
    Residency,
    Adapter,
    ZeroBlockSize,
    BlockSizeMismatch,
    TokenCountMismatch,
    ConflictingHash,
    Capacity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::kv_index) struct FullAttentionBatchRejection {
    pub(in crate::kv_index) reason: FullAttentionRejectionReason,
    pub(in crate::kv_index) event_index: usize,
}

/// Admit full-attention KV layouts, not a dense-versus-MoE model type.
/// Stores must use the actual source block size and contain complete blocks of
/// tokens, with only modeled extra keys, named adapters, and residency metadata.
/// Session annotations, multiple storage tiers, and absent layout metadata remain
/// compatible. Acceptance is not proof of remote identity or recovery.
pub(in crate::kv_index) fn validate_full_attention_batch(
    batch: &KVEventBatch,
    expected_block_size: u32,
) -> Result<(), FullAttentionBatchRejection> {
    for (event_index, event) in batch.events.iter().enumerate() {
        let reason = match event {
            KVEvent::BlockStored(s) => {
                // Keep unmodeled salt/MM/layout identities out of this token-only
                // query space. Named LoRA is already isolated by the hash seed.
                // session_id annotates the request, not the block's cache identity.
                if s.lookahead_token_ids.is_some()
                    || !extra_keys_match(
                        s.extra_keys.as_ref(),
                        s.block_hashes.len(),
                        s.lora_name.as_deref(),
                    )
                {
                    Some(FullAttentionRejectionReason::ExtraKeys)
                } else if s
                    .kv_cache_spec_kind
                    .as_deref()
                    .is_some_and(|kind| !matches!(kind, "full_attention" | "mla_attention"))
                    || s.kv_cache_spec_sliding_window.is_some()
                    // Undeclared full attention answers queries from group 0 only. A
                    // second group means the engine holds state this path
                    // cannot see, so group-0 credit would overstate reuse.
                    || s.group_idx.is_some_and(|group| group != 0)
                {
                    Some(FullAttentionRejectionReason::Layout)
                } else if !is_supported_residency(
                    s.medium.as_deref(),
                    s.locality.as_deref(),
                    s.ownership.as_deref(),
                ) {
                    Some(FullAttentionRejectionReason::Residency)
                } else if (s.lora_id.is_some() && s.lora_name.is_none())
                    || s.lora_name.as_ref().is_some_and(|name| name.is_empty())
                {
                    Some(FullAttentionRejectionReason::Adapter)
                } else if s.block_hashes.is_empty() {
                    // Dense attention stores one hash per block, so an empty
                    // list is meaningless here. Only a sparse checkpoint group
                    // may legitimately publish none.
                    Some(FullAttentionRejectionReason::TokenCountMismatch)
                } else if s.block_size == 0 {
                    Some(FullAttentionRejectionReason::ZeroBlockSize)
                } else if s.block_size != expected_block_size {
                    Some(FullAttentionRejectionReason::BlockSizeMismatch)
                } else if !complete_blocks(s) {
                    Some(FullAttentionRejectionReason::TokenCountMismatch)
                } else {
                    None
                }
            }
            KVEvent::BlockRemoved(r) => {
                if r.group_idx.is_some_and(|group| group != 0) {
                    Some(FullAttentionRejectionReason::Layout)
                } else if r.block_hashes.is_empty() {
                    Some(FullAttentionRejectionReason::TokenCountMismatch)
                } else {
                    (!is_supported_residency(
                        r.medium.as_deref(),
                        r.locality.as_deref(),
                        r.ownership.as_deref(),
                    ))
                    .then_some(FullAttentionRejectionReason::Residency)
                }
            }
            KVEvent::AllBlocksCleared(_) => None,
        };
        if let Some(reason) = reason {
            return Err(FullAttentionBatchRejection {
                reason,
                event_index,
            });
        }
    }
    Ok(())
}

fn is_supported_residency(
    medium: Option<&str>,
    locality: Option<&str>,
    ownership: Option<&str>,
) -> bool {
    let pool = ownership.is_some_and(|owner| !owner.is_empty());
    locality.is_none_or(|locality| locality == if pool { "REMOTE" } else { "LOCAL" })
        && medium.is_none_or(|medium| StorageTier::from_medium(Some(medium)).is_some())
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::kv_index::wire::{BlockStored, ExternalBlockHash};

    #[test]
    fn speculative_store_checks_lookahead_for_each_emitted_hash() {
        let mut store: BlockStored = serde_json::from_value(serde_json::json!({
            "block_hashes": [], "parent_block_hash": null,
            "token_ids": [1, 2, 3, 4, 5, 6, 7, 8], "block_size": 4,
            "medium": "GPU", "lora_id": null, "lora_name": null,
            "group_idx": 0, "kv_cache_spec_kind": "mamba"
        }))
        .unwrap();
        let group = CacheGroup {
            group_idx: 0,
            block_size: 4,
            kind: GroupRule::Checkpoint,
        };
        assert_eq!(validate_hybrid_store(&group, &store, 2, 1, true), Ok(true));
        store
            .block_hashes
            .push(ExternalBlockHash::Bytes(vec![0; 32]));
        assert_eq!(validate_hybrid_store(&group, &store, 2, 1, true), Err(()));
        store.lookahead_token_ids = Some(vec![vec![9, 10]]);
        assert_eq!(validate_hybrid_store(&group, &store, 2, 1, true), Err(()));
        store.lookahead_token_ids = Some(vec![vec![9]]);
        assert_eq!(validate_hybrid_store(&group, &store, 2, 1, true), Ok(true));
        assert_eq!(validate_hybrid_store(&group, &store, 2, 0, true), Err(()));
    }

    #[test]
    fn full_attention_shape_matrix_preserves_rejection_reason_and_event_index() {
        let valid: BlockStored = serde_json::from_value(serde_json::json!({
            "block_hashes": [1], "parent_block_hash": null,
            "token_ids": [1, 2, 3, 4], "block_size": 4, "medium": "GPU",
            "lora_id": null, "lora_name": null
        }))
        .unwrap();
        let mut batch = KVEventBatch {
            ts: 0.0,
            events: vec![
                KVEvent::AllBlocksCleared(Default::default()),
                KVEvent::BlockStored(valid.clone()),
            ],
            data_parallel_rank: Some(0),
        };
        assert_eq!(validate_full_attention_batch(&batch, 4), Ok(()));
        for (case, reason) in [
            ("remote", FullAttentionRejectionReason::Residency),
            ("local-pool", FullAttentionRejectionReason::Residency),
            ("unknown-medium", FullAttentionRejectionReason::Residency),
            ("layout", FullAttentionRejectionReason::Layout),
            ("sliding-window", FullAttentionRejectionReason::Layout),
            (
                "token-count",
                FullAttentionRejectionReason::TokenCountMismatch,
            ),
        ] {
            let mut event = valid.clone();
            match case {
                "remote" => event.locality = Some("REMOTE".into()),
                "local-pool" => {
                    event.locality = Some("LOCAL".into());
                    event.ownership = Some("pool".into());
                }
                "unknown-medium" => event.medium = Some("unknown".into()),
                "layout" => event.kv_cache_spec_kind = Some("sliding_attention".into()),
                "sliding-window" => event.kv_cache_spec_sliding_window = Some(8),
                "token-count" => event.block_hashes.push(ExternalBlockHash::Int(2)),
                _ => unreachable!(),
            }
            batch.events[1] = KVEvent::BlockStored(event);
            assert_eq!(
                validate_full_attention_batch(&batch, 4),
                Err(FullAttentionBatchRejection {
                    reason,
                    event_index: 1
                }),
                "{case}"
            );
        }
    }

    #[test]
    fn full_attention_rejects_events_from_a_second_cache_group() {
        let stored = |group: Option<u32>| {
            let mut event: BlockStored = serde_json::from_value(serde_json::json!({
                "block_hashes": [1], "parent_block_hash": null,
                "token_ids": [1, 2, 3, 4], "block_size": 4, "medium": "GPU",
                "lora_id": null, "lora_name": null
            }))
            .unwrap();
            event.group_idx = group;
            KVEvent::BlockStored(event)
        };
        let removed = |group: Option<u32>| {
            KVEvent::BlockRemoved(crate::kv_index::wire::BlockRemoved {
                block_hashes: vec![ExternalBlockHash::Int(1)],
                medium: Some("GPU".into()),
                group_idx: group,
                locality: None,
                ownership: None,
            })
        };
        let batch = |event| KVEventBatch {
            ts: 0.0,
            events: vec![event],
            data_parallel_rank: Some(0),
        };
        // Single-group engines emit group 0 or omit it; both stay compatible.
        for event in [
            stored(None),
            stored(Some(0)),
            removed(None),
            removed(Some(0)),
        ] {
            assert_eq!(validate_full_attention_batch(&batch(event), 4), Ok(()));
        }
        for event in [stored(Some(1)), removed(Some(3))] {
            assert_eq!(
                validate_full_attention_batch(&batch(event), 4),
                Err(FullAttentionBatchRejection {
                    reason: FullAttentionRejectionReason::Layout,
                    event_index: 0,
                })
            );
        }
    }

    #[test]
    fn full_attention_block_size_matrix_preserves_rejection_reason_and_event_index() {
        for expected in [8, 16, 32] {
            for actual in [0, 8, 16, 32] {
                let stored = serde_json::from_value(serde_json::json!({
                    "type": "BlockStored", "block_hashes": [1, 2],
                    "parent_block_hash": null, "token_ids": vec![1u32; actual * 2],
                    "block_size": actual, "medium": "CPU",
                    "lora_id": null, "lora_name": null
                }))
                .unwrap();
                let mut batch = KVEventBatch {
                    ts: 1.0,
                    events: vec![KVEvent::AllBlocksCleared(Default::default()), stored],
                    data_parallel_rank: Some(3),
                };
                let reason = if actual == 0 {
                    Some(FullAttentionRejectionReason::ZeroBlockSize)
                } else if actual as u32 != expected {
                    Some(FullAttentionRejectionReason::BlockSizeMismatch)
                } else {
                    None
                };
                assert_eq!(
                    validate_full_attention_batch(&batch, expected),
                    reason.map_or(Ok(()), |reason| Err(FullAttentionBatchRejection {
                        reason,
                        event_index: 1,
                    })),
                    "source={expected}, event={actual}"
                );
                if reason.is_none() {
                    let KVEvent::BlockStored(stored) = &mut batch.events[1] else {
                        unreachable!()
                    };
                    stored.token_ids.pop();
                    assert_eq!(
                        validate_full_attention_batch(&batch, expected),
                        Err(FullAttentionBatchRejection {
                            reason: FullAttentionRejectionReason::TokenCountMismatch,
                            event_index: 1,
                        })
                    );
                }
            }
        }
    }

    #[test]
    fn full_attention_rejects_empty_hash_lists() {
        let batch = crate::kv_index::wire::KVEventBatch {
            ts: 0.0,
            events: vec![crate::kv_index::wire::KVEvent::BlockStored(
                crate::kv_index::wire::BlockStored {
                    block_hashes: vec![],
                    parent_block_hash: None,
                    token_ids: vec![1, 2],
                    block_size: 2,
                    medium: Some("GPU".into()),
                    locality: Some("LOCAL".into()),
                    ownership: None,
                    lora_id: None,
                    lora_name: None,
                    extra_keys: None,
                    group_idx: Some(0),
                    kv_cache_spec_kind: Some("full_attention".into()),
                    kv_cache_spec_sliding_window: None,
                    session_id: None,
                    lookahead_token_ids: None,
                },
            )],
            data_parallel_rank: None,
        };
        let r = validate_full_attention_batch(&batch, 2);
        assert!(r.is_err(), "{r:?}");
    }
}
