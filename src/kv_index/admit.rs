//! Device event-shape admission, separate from structural decoding.
use crate::kv_index::wire::{BlockStored, ExternalBlockHash, KVEvent, KVEventBatch};

/// Validate the narrow normal-dense GPU event shape, preserving the batch rank.
/// This does not establish publisher identity, deployment layout, independently
/// addressable ranks, cache namespace, hash algorithm or freshness. The source
/// adapter/target layer must bind sources to ranks and establish route eligibility;
/// scoring must consider only eligible targets. Shape acceptance does not enable
/// arbitrary-rank or native DP routing.
pub fn validate_device_batch(
    batch: &KVEventBatch,
    expected_block_size: usize,
) -> Result<(), String> {
    if expected_block_size == 0 {
        return Err("expected KV block size must be nonzero".into());
    }
    for event in &batch.events {
        let (hashes, medium, group, locality, ownership) = match event {
            KVEvent::AllBlocksCleared(_) => continue,
            KVEvent::BlockRemoved(event) => (
                &event.block_hashes,
                &event.medium,
                event.group_idx,
                &event.locality,
                &event.ownership,
            ),
            KVEvent::BlockStored(event) => {
                if !normal_dense_shape(event, expected_block_size)
                    || !no_unsupported_extensions(event)
                {
                    return Err(
                        "stored event is outside the normal-dense complete-block contract".into(),
                    );
                }
                if let Some(parent) = &event.parent_block_hash {
                    validate_hash(parent)?;
                }
                (
                    &event.block_hashes,
                    &event.medium,
                    event.group_idx,
                    &event.locality,
                    &event.ownership,
                )
            }
        };
        if medium.as_deref() != Some("GPU")
            || group.is_some_and(|group| group != 0)
            || locality
                .as_deref()
                .is_some_and(|locality| locality != "LOCAL")
            || ownership.is_some()
        {
            return Err("event is outside the GPU group-0 local ownership contract".into());
        }
        if hashes.is_empty() {
            return Err("KV event hashes cannot be empty".into());
        }
        for hash in hashes {
            validate_hash(hash)?;
        }
    }
    Ok(())
}

fn normal_dense_shape(event: &BlockStored, expected_block_size: usize) -> bool {
    event.block_size as usize == expected_block_size
        && event.block_hashes.len().checked_mul(expected_block_size) == Some(event.token_ids.len())
}

fn no_unsupported_extensions(event: &BlockStored) -> bool {
    event.lora_id.is_none()
        && event.lora_name.is_none()
        // vLLM v0.29.0 block_pool.py builds one extra-key entry per emitted
        // block; a present nonempty all-None list is normal publisher output.
        && event.extra_keys.as_ref().is_none_or(|keys| {
            keys.len() == event.block_hashes.len() && keys.iter().all(|key| key.is_nil())
        })
        && event
            .kv_cache_spec_kind
            .as_deref()
            .is_none_or(|kind| kind == "full_attention")
        && event.kv_cache_spec_sliding_window.is_none()
        && event.session_id.is_none()
}

fn validate_hash(hash: &ExternalBlockHash) -> Result<(), String> {
    match hash {
        ExternalBlockHash::Bytes(bytes) if bytes.len() == 32 => Ok(()),
        _ => Err("KV hash must be 32-byte sha256_cbor bytes".into()),
    }
}
