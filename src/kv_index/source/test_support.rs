//! Shared event fixtures; callers set model and residency-specific fields.

use super::events::wire::{BlockRemoved, BlockStored, ExternalBlockHash, KVEvent, KVEventBatch};

pub(crate) fn stored_block(
    block_hashes: Vec<ExternalBlockHash>,
    token_ids: Vec<u32>,
    block_size: u32,
) -> BlockStored {
    BlockStored {
        block_hashes,
        parent_block_hash: None,
        token_ids,
        block_size,
        medium: Some("GPU".into()),
        extra_keys: None,
        lora_id: None,
        lora_name: None,
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
        locality: None,
        ownership: None,
        session_id: None,
        lookahead_token_ids: None,
    }
}

pub(crate) fn event_batch(ts: f64, events: Vec<KVEvent>) -> KVEventBatch {
    KVEventBatch {
        ts,
        events,
        data_parallel_rank: Some(0),
    }
}

pub(crate) fn gpu_remove(group_idx: Option<u32>, block_hashes: Vec<ExternalBlockHash>) -> KVEvent {
    KVEvent::BlockRemoved(BlockRemoved {
        block_hashes,
        medium: Some("GPU".into()),
        group_idx,
        locality: None,
        ownership: None,
    })
}

pub(crate) fn event_payload(ts: f64, events: &[KVEvent], rank: Option<u32>) -> Vec<u8> {
    rmp_serde::to_vec_named(&(ts, events, rank)).unwrap()
}
