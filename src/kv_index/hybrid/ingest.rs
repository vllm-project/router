//! Admission and application of engine event batches.

use super::super::wire::ExternalBlockHash;
use super::super::{KVEvent, KVEventBatch};
use super::layout::{event_kind_matches, stored_shape_matches};
use super::residency::State;
use super::HybridEventIndex;
use crate::kv_hash::SequenceHash;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridEventError {
    InvalidLayout,
    Untrusted,
    Sequence,
    InvalidBatch,
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
        let mut state = self.state.write();
        let Some(next_seq) = state.next_seq else {
            return Err(HybridEventError::Untrusted);
        };
        if seq != next_seq || seq == u64::MAX {
            return Self::fail(&mut state, HybridEventError::Sequence);
        }
        if batch.data_parallel_rank != Some(self.target.dp_rank)
            || !batch.ts.is_finite()
            || batch.ts < 0.0
        {
            return Self::fail(&mut state, HybridEventError::InvalidBatch);
        }
        for event in &batch.events {
            let result = match event {
                KVEvent::BlockStored(store) => {
                    // An engine may publish several page sizes for one group
                    // index. Only the declared granularity is addressable by
                    // the layout; the other pages cover the same tokens and
                    // add no boundary information.
                    let Some(group) = self
                        .groups
                        .iter()
                        .find(|group| store.group_idx == Some(group.group_idx))
                    else {
                        return Self::fail(&mut state, HybridEventError::InvalidBatch);
                    };
                    if store.block_size != group.block_size {
                        tracing::debug!(
                            group_idx = group.group_idx,
                            declared = group.block_size,
                            published = store.block_size,
                            "hybrid group published a secondary page size"
                        );
                        continue;
                    }
                    if store.medium.as_deref() != Some("GPU")
                        || store.locality.as_deref().is_some_and(|s| s != "LOCAL")
                        || store.ownership.is_some()
                        || store.lora_id.is_some()
                        || store.lora_name.is_some()
                        || !event_kind_matches(group.kind, store)
                        || !stored_shape_matches(group.kind, store)
                        || store.extra_keys.as_ref().is_some_and(|keys| {
                            keys.len() != store.block_hashes.len()
                                || keys.iter().any(|key| !key.is_nil())
                        })
                        || store
                            .parent_block_hash
                            .as_ref()
                            .is_some_and(|hash| sequence_hash(hash).is_none())
                    {
                        Err(HybridEventError::InvalidBatch)
                    } else {
                        let Some(keys) = store
                            .block_hashes
                            .iter()
                            .map(sequence_hash)
                            .collect::<Option<Vec<_>>>()
                        else {
                            return Self::fail(&mut state, HybridEventError::InvalidBatch);
                        };
                        let group_idx = group.group_idx;
                        for hash in keys {
                            let key = (group_idx, hash);
                            if !state.claims.contains(&key)
                                && state.claims.len() >= self.max_entries
                            {
                                return Self::fail(&mut state, HybridEventError::Capacity);
                            }
                            state.claims.insert(key);
                        }
                        Ok(())
                    }
                }
                KVEvent::BlockRemoved(remove) => {
                    if remove.medium.as_deref() != Some("GPU")
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
                        let group_idx = remove.group_idx.expect("validated group");
                        for hash in &remove.block_hashes {
                            let Some(hash) = sequence_hash(hash) else {
                                return Self::fail(&mut state, HybridEventError::InvalidBatch);
                            };
                            state.claims.remove(&(group_idx, hash));
                        }
                        Ok(())
                    }
                }
                KVEvent::AllBlocksCleared(_) => {
                    state.claims.clear();
                    Ok(())
                }
            };
            if let Err(error) = result {
                return Self::fail(&mut state, error);
            }
        }
        state.next_seq = Some(seq + 1);
        Ok(())
    }

    pub(super) fn fail<T>(
        state: &mut State,
        error: HybridEventError,
    ) -> Result<T, HybridEventError> {
        state.claims.clear();
        state.next_seq = None;
        state.visible = false;
        Err(error)
    }
}

fn sequence_hash(hash: &ExternalBlockHash) -> Option<SequenceHash> {
    match hash {
        ExternalBlockHash::Bytes(bytes) => bytes.as_slice().try_into().ok(),
        ExternalBlockHash::Int(_) => None,
    }
}
