//! Shared event fixtures; callers set model and residency-specific fields.

use std::sync::Arc;

use super::events::wire::{BlockRemoved, BlockStored, ExternalBlockHash, KVEvent, KVEventBatch};
use crate::kv_index::{CacheGroup, GroupRule, RoutableTarget, SourceBinding};
use crate::policies::RegistrationCurrent;

/// A live single-rank owner for fixtures with no supervisor lifecycle.
pub(crate) fn current_registration() -> Arc<dyn RegistrationCurrent> {
    struct Current;

    impl RegistrationCurrent for Current {
        fn current(&self, rank: u32) -> bool {
            rank == 0
        }
    }

    Arc::new(Current)
}

pub(crate) fn full_binding(target: RoutableTarget, block_size: u32) -> SourceBinding {
    SourceBinding::new(
        target,
        vec![CacheGroup {
            group_idx: 0,
            block_size,
            kind: GroupRule::FullAttention,
        }],
        block_size,
        0,
        10_000,
    )
    .unwrap()
}

pub(crate) fn full_binding_with_adapter(
    target: RoutableTarget,
    block_size: u32,
    name: &str,
    path: &str,
) -> SourceBinding {
    full_binding(target, block_size).with_adapters(vec![(name.into(), path.into())])
}

pub(crate) fn full_store(
    tokens: &[u32],
    block_size: u32,
    adapter: Option<(&str, &str)>,
) -> BlockStored {
    let keys = crate::kv_index::request_hash::BlockKeyGenerator::new(block_size as usize, 0);
    let hashes = match adapter {
        Some((name, path)) => keys.generate_block_keys_for_adapter(tokens, name, path),
        None => keys.generate_block_keys(tokens),
    };
    let mut store = stored_block(
        hashes
            .into_iter()
            .map(|hash| ExternalBlockHash::Bytes(hash.to_vec()))
            .collect(),
        tokens.to_vec(),
        block_size,
    );
    store.group_idx = Some(0);
    store.kv_cache_spec_kind = Some("full_attention".into());
    store.lora_name = adapter.map(|(name, _)| name.to_owned());
    store
}

pub(crate) fn warm_full(binding: &SourceBinding, seq: u64, tokens: &[u32]) {
    let mut batch = event_batch(vec![KVEvent::BlockStored(full_store(
        tokens,
        binding.keys.block_size() as u32,
        None,
    ))]);
    batch.data_parallel_rank = Some(binding.target.dp_rank);
    binding.apply_batch(seq, &batch).unwrap();
}

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
        lookahead_token_ids: None,
    }
}

pub(crate) fn event_batch(events: Vec<KVEvent>) -> KVEventBatch {
    KVEventBatch {
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
