//! Event-shape admission, separate from structural decoding and transport.
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DenseRejectionReason {
    ExtraKeys,
    Layout,
    Residency,
    Adapter,
    ZeroBlockSize,
    BlockSizeMismatch,
    TokenCountMismatch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DenseBatchRejection {
    pub(super) reason: DenseRejectionReason,
    pub(super) event_index: usize,
}

pub(super) fn validate_dense_batch(
    batch: &KVEventBatch,
    expected_block_size: u32,
) -> Result<(), DenseBatchRejection> {
    for (event_index, event) in batch.events.iter().enumerate() {
        let reason = match event {
            KVEvent::BlockStored(s) => {
                // Keep unmodeled salt/MM/layout identities out of this token-only
                // query space. Named LoRA is already isolated by the hash seed.
                // session_id annotates the request, not the block's cache identity.
                if s.extra_keys.as_ref().is_some_and(|keys| {
                    keys.len() != s.block_hashes.len()
                        || keys.iter().any(|key| {
                            !key.is_nil()
                                && !s.lora_name.as_ref().is_some_and(|name| {
                                    key.as_array().is_some_and(|parts| {
                                        parts.len() == 1 && parts[0].as_str() == Some(name.as_str())
                                    })
                                })
                        })
                }) {
                    Some(DenseRejectionReason::ExtraKeys)
                } else if s
                    .kv_cache_spec_kind
                    .as_deref()
                    .is_some_and(|kind| kind != "full_attention")
                    || s.kv_cache_spec_sliding_window.is_some()
                {
                    Some(DenseRejectionReason::Layout)
                } else if !dense_residency_supported(
                    s.medium.as_deref(),
                    s.locality.as_deref(),
                    s.ownership.as_deref(),
                ) {
                    Some(DenseRejectionReason::Residency)
                } else if (s.lora_id.is_some() && s.lora_name.is_none())
                    || s.lora_name.as_ref().is_some_and(|name| name.is_empty())
                {
                    Some(DenseRejectionReason::Adapter)
                } else if s.block_size == 0 {
                    Some(DenseRejectionReason::ZeroBlockSize)
                } else if s.block_size != expected_block_size {
                    Some(DenseRejectionReason::BlockSizeMismatch)
                } else if s.block_hashes.len().checked_mul(s.block_size as usize)
                    != Some(s.token_ids.len())
                {
                    Some(DenseRejectionReason::TokenCountMismatch)
                } else {
                    None
                }
            }
            KVEvent::BlockRemoved(r) => (!dense_residency_supported(
                r.medium.as_deref(),
                r.locality.as_deref(),
                r.ownership.as_deref(),
            ))
            .then_some(DenseRejectionReason::Residency),
            KVEvent::AllBlocksCleared(_) => None,
        };
        if let Some(reason) = reason {
            return Err(DenseBatchRejection {
                reason,
                event_index,
            });
        }
    }
    Ok(())
}

fn dense_residency_supported(
    medium: Option<&str>,
    locality: Option<&str>,
    ownership: Option<&str>,
) -> bool {
    let pool = ownership.is_some_and(|owner| !owner.is_empty());
    locality.is_none_or(|locality| locality == if pool { "REMOTE" } else { "LOCAL" })
        && medium.is_none_or(|medium| {
            matches!(
                medium.to_ascii_uppercase().as_str(),
                "GPU" | "DEVICE" | "CPU" | "HOST" | "HOST_PINNED" | "STORAGE" | "DISK"
            )
        })
}
