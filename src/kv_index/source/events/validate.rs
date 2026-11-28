//! Event-shape admission, separate from structural decoding and transport.
use crate::kv_index::layout::{CacheGroup, GroupRule};
use crate::kv_index::request_hash::{is_image_hash, SequenceHash};
use crate::kv_index::wire::{BlockStored, ExternalBlockHash, KVEvent};
use crate::kv_index::SourceError;

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

fn image_block_key(value: &rmpv::Value) -> Option<(&str, i64)> {
    let [hash, offset] = value.as_array()?.as_slice() else {
        return None;
    };
    Some((hash.as_str()?, offset.as_i64()?))
}

pub(super) fn image_block_keys(key: &rmpv::Value) -> Option<Vec<(&str, i64)>> {
    let features = key.as_array().filter(|features| !features.is_empty())?;
    features.iter().map(image_block_key).collect()
}

/// Return false when a store provides no usable prefix evidence.
pub(super) fn validate_store(
    group: &CacheGroup,
    store: &BlockStored,
    hash_block_size: usize,
    lookahead_tokens: usize,
    partial_hash_hits: bool,
) -> Result<bool, SourceError> {
    if store.locality.as_deref().is_some_and(|s| s != "LOCAL")
        || match &store.lookahead_token_ids {
            None => lookahead_tokens != 0 && !store.block_hashes.is_empty(),
            Some(ahead) => {
                lookahead_tokens == 0
                    || ahead.len() != store.block_hashes.len()
                    || ahead.iter().any(|tokens| tokens.len() != lookahead_tokens)
            }
        }
        || store
            .parent_block_hash
            .as_ref()
            .is_some_and(|hash| sequence_hash(hash).is_none())
        || !event_kind_matches(group.kind, store)
    {
        return Err(SourceError::InvalidBatch);
    }
    let mut unsupported_extra_keys = false;
    if let Some(keys) = store
        .extra_keys
        .as_ref()
        .filter(|_| store.lora_name.is_none())
    {
        if keys.len() != store.block_hashes.len() {
            return Err(SourceError::InvalidBatch);
        }
        for (index, key) in keys.iter().enumerate().filter(|(_, key)| !key.is_nil()) {
            let parts = key.as_array().ok_or(SourceError::InvalidBatch)?;
            let Some((last, preceding)) = parts.split_last() else {
                return Err(SourceError::InvalidBatch);
            };
            // vLLM appends the cache salt after any image keys on the first block.
            let media = if index == 0 && last.as_str().is_some_and(|salt| !salt.is_empty()) {
                unsupported_extra_keys = true;
                preceding
            } else {
                parts.as_slice()
            };
            for feature in media {
                let (hash, _) = image_block_key(feature).ok_or(SourceError::InvalidBatch)?;
                unsupported_extra_keys |= !is_image_hash(hash);
            }
        }
    }
    let partial = partial_hash_hits
        && matches!(
            group.kind,
            GroupRule::FullAttention
                | GroupRule::CompressedFullAttention { .. }
                | GroupRule::Checkpoint
        )
        && store.block_size < group.block_size
        && store.block_size as usize == hash_block_size
        && store.block_hashes.len() == 1
        && store.token_ids.len() == hash_block_size;
    if store.block_size == 0 {
        return Err(SourceError::InvalidBatch);
    }
    if store.block_size != group.block_size && !partial {
        if store.lora_name.is_some() {
            return Err(SourceError::InvalidBatch);
        }
        tracing::debug!(
            group_idx = group.group_idx,
            declared = group.block_size,
            published = store.block_size,
            "KV group published a secondary page size"
        );
        return Ok(false);
    }
    let block_size = store.block_size as usize;
    if store.token_ids.is_empty()
        || !store.token_ids.len().is_multiple_of(block_size)
        || !match group.kind {
            GroupRule::Checkpoint | GroupRule::SlidingWindow { .. } => {
                store.block_hashes.len() <= store.token_ids.len() / block_size
            }
            _ => complete_blocks(store),
        }
    {
        return Err(SourceError::InvalidBatch);
    }
    Ok(!unsupported_extra_keys)
}

pub(in crate::kv_index) fn event_kind_matches(kind: GroupRule, store: &BlockStored) -> bool {
    let spec = store.kv_cache_spec_kind.as_deref();
    // Both forms of MLA publish the same wire kind. Compression is
    // determined by the verified cache-group layout.
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
        assert_eq!(validate_store(&group, &store, 2, 1, true), Ok(true));
        store
            .block_hashes
            .push(ExternalBlockHash::Bytes(vec![0; 32]));
        assert_eq!(
            validate_store(&group, &store, 2, 1, true),
            Err(SourceError::InvalidBatch)
        );
        store.lookahead_token_ids = Some(vec![vec![9, 10]]);
        assert_eq!(
            validate_store(&group, &store, 2, 1, true),
            Err(SourceError::InvalidBatch)
        );
        store.lookahead_token_ids = Some(vec![vec![9]]);
        assert_eq!(validate_store(&group, &store, 2, 1, true), Ok(true));
        assert_eq!(
            validate_store(&group, &store, 2, 0, true),
            Err(SourceError::InvalidBatch)
        );
    }
}
