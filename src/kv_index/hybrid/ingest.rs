//! Admission and application of engine event batches.

use super::super::wire::ExternalBlockHash;
use super::super::{KVEvent, KVEventBatch};
use super::layout::{event_kind_matches, stored_shape_matches, HybridGroupKind};
use super::residency::State;
use super::{HybridEventIndex, IndexBatch};
use crate::kv_hash::SequenceHash;
use crate::kv_index::wire::extra_keys_match;
use crate::kv_index::ResidencyOwner;
use crate::kv_index::StorageTier;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridEventError {
    InvalidLayout,
    Untrusted,
    Sequence,
    InvalidBatch,
    /// The source publishes integer block hashes, which the sha256 index cannot
    /// address; no batch from it can ever be applied.
    IntBlockHashes,
    Capacity,
}

impl HybridEventIndex {
    /// Decode a wire payload before applying it. Corrupt payloads invalidate
    /// all groups, including claims from previously accepted batches.
    ///
    pub fn apply_payload(&self, seq: u64, payload: &[u8]) -> Result<(), HybridEventError> {
        let batch = match crate::kv_index::decode_batch(payload) {
            Ok(batch) => batch,
            Err(error) => {
                tracing::warn!(
                    error,
                    bytes = payload.len(),
                    "hybrid payload did not decode"
                );
                self.invalidate();
                return Err(HybridEventError::InvalidBatch);
            }
        };
        self.apply_batch(seq, &batch)
    }

    /// Apply an ordered batch atomically. A sequence mismatch, including a
    /// duplicate, invalidates the view: without an independently verified
    /// publisher epoch, a repeated sequence can also mean a restart.
    pub fn apply_batch(&self, seq: u64, batch: &KVEventBatch) -> Result<(), HybridEventError> {
        self.with_active_binding(|| {
            let mut state = self.state.write();
            self.index.with_source_batch(&self.target, |tx| {
                let Some(next_seq) = state.next_seq else {
                    return Err(HybridEventError::Untrusted);
                };
                if seq != next_seq || seq == u64::MAX {
                    return Self::fail(&mut state, tx, &self.owner, HybridEventError::Sequence);
                }
                if batch.data_parallel_rank != Some(self.target.dp_rank)
                    || !batch.ts.is_finite()
                    || batch.ts < 0.0
                {
                    return Self::fail(&mut state, tx, &self.owner, HybridEventError::InvalidBatch);
                }
                // vLLM can encode block hashes as integers instead of 32-byte digests
                // (VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES, which is on by default). The
                // hybrid index is built on the sha256 chain, so such a source can never
                // be indexed; naming the cause keeps an unindexable engine from looking
                // like an ordinary layout mismatch.
                if batch.events.iter().any(publishes_int_hash) {
                    return Self::fail(
                        &mut state,
                        tx,
                        &self.owner,
                        HybridEventError::IntBlockHashes,
                    );
                }
                for event in &batch.events {
                    let result = match event {
                        KVEvent::BlockStored(store) => {
                            // An adapter caches its pages under its own hash chain,
                            // seeded by the adapter identity. Index them under that
                            // identity and never as evidence for the base chain; an
                            // adapter store this layout cannot address is skipped
                            // rather than revoking claims it cannot speak for.
                            let namespace = store.lora_name.clone();
                            let adapter = match namespace.as_deref() {
                                None => false,
                                Some(name) => {
                                    if self.adapter_path(name).is_none() {
                                        continue;
                                    }
                                    true
                                }
                            };
                            // Identity and shape guards run before the page-size skip:
                            // a foreign cache identity must fail the whole batch even
                            // when it publishes a page size this layout does not
                            // address.
                            // `session_id` annotates a request without changing the
                            // block hash, and an adapter store carries its own identity
                            // as extra keys: neither makes the page foreign. A base
                            // store must carry no extra keys at all.
                            // The medium names a cache tier, not a different cache
                            // identity: a host or storage copy of the same page is
                            // restorable (at a transfer cost), so it is claimed at that
                            // tier instead of rejecting the batch. External owners are
                            // not this rank's residency and are skipped.
                            // A copy owned by another domain, or one whose medium this
                            // build cannot place, is not this rank's addressable
                            // residency: skip it instead of revoking the view.
                            if store.ownership.is_some() {
                                continue;
                            }
                            let Some(tier) = StorageTier::from_medium(store.medium.as_deref())
                            else {
                                continue;
                            };
                            if store.locality.as_deref().is_some_and(|s| s != "LOCAL")
                                || (namespace.is_none()
                                    && !extra_keys_match(
                                        store.extra_keys.as_ref(),
                                        store.block_hashes.len(),
                                        None,
                                    ))
                                || store
                                    .parent_block_hash
                                    .as_ref()
                                    .is_some_and(|hash| sequence_hash(hash).is_none())
                            {
                                if adapter {
                                    continue;
                                }
                                return Self::fail(
                                    &mut state,
                                    tx,
                                    &self.owner,
                                    HybridEventError::InvalidBatch,
                                );
                            }
                            // An engine may publish several page sizes for one group
                            // index (V4-class compressors store 4-token pages next to
                            // the group's own pages). Only the declared granularity is
                            // addressable by the layout; the other pages cover the same
                            // tokens and add no boundary information.
                            let Some(group) = self
                                .groups
                                .iter()
                                .find(|group| store.group_idx == Some(group.group_idx))
                            else {
                                if adapter {
                                    continue;
                                }
                                return Self::fail(
                                    &mut state,
                                    tx,
                                    &self.owner,
                                    HybridEventError::InvalidBatch,
                                );
                            };
                            if !event_kind_matches(group.kind, store) {
                                if adapter {
                                    continue;
                                }
                                return Self::fail(
                                    &mut state,
                                    tx,
                                    &self.owner,
                                    HybridEventError::InvalidBatch,
                                );
                            }
                            // vLLM publishes partial full-attention and Mamba entries
                            // as one hash-granularity store. Other secondary page sizes
                            // (notably compressed groups) are not partial evidence.
                            let partial = self.partial_hash_hits
                                && matches!(
                                    group.kind,
                                    HybridGroupKind::FullAttention | HybridGroupKind::Checkpoint
                                )
                                && store.block_size < group.block_size
                                && store.block_size as usize == self.keys.block_size()
                                && store.block_hashes.len() == 1
                                && store.token_ids.len() == self.keys.block_size();
                            if store.block_size != group.block_size && !partial {
                                tracing::debug!(
                                    group_idx = group.group_idx,
                                    declared = group.block_size,
                                    published = store.block_size,
                                    "hybrid group published a secondary page size"
                                );
                                continue;
                            }
                            if !stored_shape_matches(group.kind, store) {
                                if adapter {
                                    continue;
                                }
                                return Self::fail(
                                    &mut state,
                                    tx,
                                    &self.owner,
                                    HybridEventError::InvalidBatch,
                                );
                            }
                            let Ok(keys) = self.stored_keys(group, store) else {
                                if adapter {
                                    continue;
                                }
                                return Self::fail(
                                    &mut state,
                                    tx,
                                    &self.owner,
                                    HybridEventError::InvalidBatch,
                                );
                            };
                            let group_idx = group.group_idx;
                            // A checkpoint or compressed group publishes every page it
                            // holds, so an empty list means it holds none and must bound
                            // the prefix at zero. A sliding group in a V4-class layout
                            // publishes no pages at all; its state is tracked by the
                            // compressed group, so a key-only store is not evidence.
                            if !keys.is_empty()
                                || !matches!(group.kind, HybridGroupKind::SlidingWindow { .. })
                            {
                                state
                                    .page_evidence
                                    .entry(namespace.clone())
                                    .or_default()
                                    .insert(group_idx);
                            }
                            for hash in keys {
                                if !tx.store_sparse(
                                    &self.owner,
                                    namespace.as_deref(),
                                    group_idx,
                                    &hash,
                                    tier,
                                    self.max_entries,
                                ) {
                                    // An adapter that fills its own budget must not
                                    // revoke the base model's claims; skip the page.
                                    if adapter {
                                        continue;
                                    }
                                    return Self::fail(
                                        &mut state,
                                        tx,
                                        &self.owner,
                                        HybridEventError::Capacity,
                                    );
                                }
                            }
                            Ok(())
                        }
                        KVEvent::BlockRemoved(remove) => {
                            if StorageTier::from_medium(remove.medium.as_deref()).is_none()
                                || remove.locality.as_deref().is_some_and(|s| s != "LOCAL")
                                || remove.ownership.is_some()
                                || !self
                                    .groups
                                    .iter()
                                    .any(|group| remove.group_idx == Some(group.group_idx))
                                || remove.block_hashes.is_empty()
                            {
                                Err(HybridEventError::InvalidBatch)
                            } else {
                                if remove.ownership.is_some() {
                                    return Ok(());
                                }
                                let group_idx = remove.group_idx.expect("validated group");
                                // An unknown medium names no tier, so the removal
                                // applies to every copy of the page: dropping more
                                // evidence is safe, keeping stale claims is not.
                                let tier = StorageTier::from_medium(remove.medium.as_deref());
                                for hash in &remove.block_hashes {
                                    let Some(hash) = sequence_hash(hash) else {
                                        return Self::fail(
                                            &mut state,
                                            tx,
                                            &self.owner,
                                            HybridEventError::InvalidBatch,
                                        );
                                    };
                                    tx.remove_sparse(&self.owner, group_idx, &hash, tier);
                                }
                                Ok(())
                            }
                        }
                        KVEvent::AllBlocksCleared(_) => {
                            // The clear is about claims: a group that has published
                            // pages still cannot bound a prefix, so its evidence stays.
                            tx.clear_owner_exact(&self.owner);
                            Ok(())
                        }
                    };
                    if let Err(error) = result {
                        return Self::fail(&mut state, tx, &self.owner, error);
                    }
                }
                state.next_seq = Some(seq + 1);
                Ok(())
            })
        })
        .unwrap_or(Err(HybridEventError::Untrusted))
    }

    pub(super) fn fail<T>(
        state: &mut State,
        tx: &IndexBatch<'_>,
        owner: &ResidencyOwner,
        error: HybridEventError,
    ) -> Result<T, HybridEventError> {
        tx.clear_owner_exact(owner);
        state.page_evidence.clear();
        state.next_seq = None;
        state.visible = false;
        Err(error)
    }
}

pub(super) fn sequence_hash(hash: &ExternalBlockHash) -> Option<SequenceHash> {
    match hash {
        ExternalBlockHash::Bytes(bytes) => bytes.as_slice().try_into().ok(),
        ExternalBlockHash::Int(_) => None,
    }
}

fn publishes_int_hash(event: &KVEvent) -> bool {
    fn any_int<'a>(hashes: impl Iterator<Item = &'a ExternalBlockHash>) -> bool {
        hashes
            .into_iter()
            .any(|hash| matches!(hash, ExternalBlockHash::Int(_)))
    }
    match event {
        KVEvent::BlockStored(store) => any_int(
            store
                .block_hashes
                .iter()
                .chain(store.parent_block_hash.iter()),
        ),
        KVEvent::BlockRemoved(remove) => any_int(remove.block_hashes.iter()),
        KVEvent::AllBlocksCleared(_) => false,
    }
}
