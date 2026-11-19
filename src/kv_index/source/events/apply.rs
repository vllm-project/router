//! Prepare engine events before committing them through one page-update path.

use std::sync::Arc;

use crate::kv_hash::SequenceHash;
use crate::kv_index::storage::source_transaction::StreamState as State;
use crate::kv_index::storage::{IndexBatch, PageWriteError};
use crate::kv_index::wire::{BlockStored, ExternalBlockHash};
use crate::kv_index::{
    ClearScope, KVEvent, KVEventBatch, ResidencyOwner, SourceBinding, SourceError, SourceId,
    StorageTier,
};

use super::validate::{
    native_device, publishes_int_hash, sequence_hash, validate_full_attention_batch,
    validate_hybrid_store, FullAttentionBatchRejection, FullAttentionRejectionReason,
};

pub(super) enum StoredPages {
    Full {
        parent: Option<Arc<str>>,
        blocks: Vec<(Arc<str>, SequenceHash)>,
    },
    Hashes(Vec<SequenceHash>),
}

pub(super) enum RemovedKeys {
    Full(Vec<Arc<str>>),
    Hashes(Vec<SequenceHash>),
}

/// Layout validation chooses the addresses; both paths commit the same facts.
pub(super) enum PreparedEvent<'a> {
    Store {
        group: u32,
        tier: StorageTier,
        incremental: bool,
        namespace: Option<&'a str>,
        pages: StoredPages,
    },
    Remove {
        group: u32,
        tier: StorageTier,
        incremental: bool,
        keys: RemovedKeys,
    },
    // A plain worker clear includes older local incarnations; a bound stream
    // clears only its exact owner. Neither clears an external pool by accident.
    Clear(Option<ClearScope>),
}

impl PreparedEvent<'_> {
    fn apply(
        self,
        tx: &mut IndexBatch<'_>,
        owner: &ResidencyOwner,
        max_entries: usize,
    ) -> Result<(), PageWriteError> {
        match self {
            Self::Store {
                group,
                tier,
                incremental,
                namespace,
                pages,
            } => match pages {
                StoredPages::Full { parent, blocks } => {
                    tx.store_full_pages(group, owner, tier, parent.as_ref(), &blocks, incremental)
                }
                StoredPages::Hashes(hashes) => tx.store_hash_pages(
                    owner,
                    group,
                    tier,
                    namespace,
                    &hashes,
                    incremental,
                    max_entries,
                ),
            },
            Self::Remove {
                group,
                tier,
                incremental,
                keys,
            } => {
                match keys {
                    RemovedKeys::Full(wires) => {
                        tx.remove_full_pages(group, owner, tier, &wires, incremental)
                    }
                    RemovedKeys::Hashes(hashes) => {
                        tx.remove_hash_pages(owner, group, tier, &hashes, incremental)
                    }
                }
                Ok(())
            }
            Self::Clear(scope) => {
                match scope {
                    Some(scope) => tx.clear(owner, scope),
                    None => tx.clear_owner_exact(owner),
                }
                Ok(())
            }
        }
    }
}

pub(super) struct PreparedFullBatch<'a> {
    mutations: Vec<(ResidencyOwner, PreparedEvent<'a>)>,
}

impl PreparedFullBatch<'_> {
    pub(super) fn apply(self, tx: &mut IndexBatch<'_>) -> Result<(), FullAttentionBatchRejection> {
        for (event_index, (owner, event)) in self.mutations.into_iter().enumerate() {
            if let Err(error) = event.apply(tx, &owner, usize::MAX) {
                if error == PageWriteError::Capacity {
                    tx.clear_owner_exact(&owner);
                    let target = owner.target();
                    tx.suspend_source(&target.instance_id, target.dp_rank);
                }
                return Err(FullAttentionBatchRejection {
                    reason: match error {
                        PageWriteError::Capacity => FullAttentionRejectionReason::Capacity,
                        PageWriteError::ConflictingHash => {
                            FullAttentionRejectionReason::ConflictingHash
                        }
                    },
                    event_index,
                });
            }
        }
        Ok(())
    }
}

pub(super) fn prepare_full_batch<'a>(
    source: &SourceId,
    dp_rank: u32,
    incarnation: u64,
    batch: &'a KVEventBatch,
    expected_block_size: Option<u32>,
) -> Result<PreparedFullBatch<'a>, FullAttentionBatchRejection> {
    if let Some(expected) = expected_block_size {
        validate_full_attention_batch(batch, expected)?;
    }
    let rank = batch.data_parallel_rank.unwrap_or(dp_rank);
    let mutations = batch
        .events
        .iter()
        .enumerate()
        .map(|(event_index, event)| {
            let tier = |owner: &ResidencyOwner, medium: Option<&str>| {
                event_tier(owner, medium).ok_or(FullAttentionBatchRejection {
                    reason: FullAttentionRejectionReason::Residency,
                    event_index,
                })
            };
            Ok(match event {
                KVEvent::BlockStored(store) => {
                    let owner = event_owner(source, rank, incarnation, store.ownership.as_deref());
                    let locals = crate::kv_index::local_block_hashes(
                        &store.token_ids,
                        store.block_size,
                        store.lora_name.as_deref(),
                    );
                    let prepared = PreparedEvent::Store {
                        group: store.group_idx.unwrap_or(0),
                        tier: tier(&owner, store.medium.as_deref())?,
                        incremental: expected_block_size.is_some()
                            && native_device(store.ownership.as_deref(), store.medium.as_deref()),
                        namespace: store.lora_name.as_deref(),
                        pages: StoredPages::Full {
                            parent: store.parent_block_hash.as_ref().map(hash_str),
                            blocks: store
                                .block_hashes
                                .iter()
                                .map(hash_str)
                                .zip(locals)
                                .collect(),
                        },
                    };
                    (owner, prepared)
                }
                KVEvent::BlockRemoved(remove) => {
                    let owner = event_owner(source, rank, incarnation, remove.ownership.as_deref());
                    let prepared = PreparedEvent::Remove {
                        group: remove.group_idx.unwrap_or(0),
                        tier: tier(&owner, remove.medium.as_deref())?,
                        incremental: expected_block_size.is_some()
                            && native_device(remove.ownership.as_deref(), remove.medium.as_deref()),
                        keys: RemovedKeys::Full(remove.block_hashes.iter().map(hash_str).collect()),
                    };
                    (owner, prepared)
                }
                KVEvent::AllBlocksCleared(_) => (
                    event_owner(source, rank, incarnation, None),
                    PreparedEvent::Clear(Some(ClearScope::Worker)),
                ),
            })
        })
        .collect::<Result<_, FullAttentionBatchRejection>>()?;
    Ok(PreparedFullBatch { mutations })
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(super) fn apply_batch(
    index: &crate::kv_index::KvBlockIndexer,
    source: &SourceId,
    dp_rank: u32,
    incarnation: u64,
    batch: &KVEventBatch,
    expected_block_size: Option<u32>,
) -> Result<(), FullAttentionBatchRejection> {
    let prepared = prepare_full_batch(source, dp_rank, incarnation, batch, expected_block_size)?;
    let target = crate::kv_index::RoutableTarget {
        instance_id: source.clone(),
        dp_rank: batch.data_parallel_rank.unwrap_or(dp_rank),
    };
    index.with_source_batch(&target, |tx| prepared.apply(tx))
}

impl SourceBinding {
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(crate) fn apply_payload(&self, seq: u64, payload: &[u8]) -> Result<(), SourceError> {
        match crate::kv_index::decode_batch(payload) {
            Ok(batch) => self.apply_batch(seq, &batch),
            Err(error) => {
                tracing::warn!(
                    error,
                    bytes = payload.len(),
                    "hybrid payload did not decode"
                );
                self.invalidate();
                Err(SourceError::InvalidBatch)
            }
        }
    }

    /// Prepare immutable layout and hash work before the source transaction.
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(crate) fn apply_batch(&self, seq: u64, batch: &KVEventBatch) -> Result<(), SourceError> {
        let prepared = self.prepare_batch(batch);
        self.with_active_binding(|| {
            self.with_stream_batch(|state, tx| self.apply_prepared(state, tx, seq, batch, prepared))
        })
        .unwrap_or(Err(SourceError::Untrusted))
    }

    pub(super) fn prepare_batch<'a>(
        &self,
        batch: &'a KVEventBatch,
    ) -> Result<Vec<PreparedEvent<'a>>, SourceError> {
        if batch.events.iter().any(publishes_int_hash) {
            return Err(SourceError::IntBlockHashes);
        }
        let mut prepared = Vec::with_capacity(batch.events.len());
        for event in &batch.events {
            match event {
                KVEvent::BlockStored(store) => match self.prepare_store(store) {
                    Ok(Some(event)) => prepared.push(event),
                    Ok(None) => {}
                    // An unsupported adapter cannot revoke base-model facts.
                    Err(SourceError::InvalidBatch) if store.lora_name.is_some() => {}
                    Err(error) => return Err(error),
                },
                KVEvent::BlockRemoved(remove) => {
                    if remove.ownership.is_some() {
                        continue;
                    }
                    let tier = StorageTier::from_medium(remove.medium.as_deref())
                        .ok_or(SourceError::InvalidBatch)?;
                    if remove.locality.as_deref().is_some_and(|s| s != "LOCAL")
                        || !self
                            .groups
                            .iter()
                            .any(|group| remove.group_idx == Some(group.group_idx))
                        || remove.block_hashes.is_empty()
                    {
                        return Err(SourceError::InvalidBatch);
                    }
                    let hashes = remove
                        .block_hashes
                        .iter()
                        .map(sequence_hash)
                        .collect::<Option<Vec<_>>>()
                        .ok_or(SourceError::InvalidBatch)?;
                    prepared.push(PreparedEvent::Remove {
                        group: remove.group_idx.expect("validated group"),
                        tier,
                        incremental: native_device(
                            remove.ownership.as_deref(),
                            remove.medium.as_deref(),
                        ),
                        keys: RemovedKeys::Hashes(hashes),
                    });
                }
                KVEvent::AllBlocksCleared(_) => prepared.push(PreparedEvent::Clear(None)),
            }
        }
        Ok(prepared)
    }

    /// The caller validates its lease in this same source transaction.
    pub(super) fn apply_prepared(
        &self,
        state: &mut State,
        tx: &mut IndexBatch<'_>,
        seq: u64,
        batch: &KVEventBatch,
        prepared: Result<Vec<PreparedEvent<'_>>, SourceError>,
    ) -> Result<(), SourceError> {
        let Some(next_seq) = state.next_seq else {
            return Err(SourceError::Untrusted);
        };
        if seq != next_seq || seq == u64::MAX {
            return Self::fail(state, tx, &self.owner, SourceError::Sequence);
        }
        if batch.data_parallel_rank != Some(self.target.dp_rank)
            || !batch.ts.is_finite()
            || batch.ts < 0.0
        {
            return Self::fail(state, tx, &self.owner, SourceError::InvalidBatch);
        }
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return Self::fail(state, tx, &self.owner, error),
        };
        for event in prepared {
            let clear = matches!(event, PreparedEvent::Clear(_));
            if event.apply(tx, &self.owner, self.max_entries).is_err() {
                return Self::fail(state, tx, &self.owner, SourceError::Capacity);
            }
            if clear {
                state.bump_generation();
            }
        }
        state.next_seq = Some(seq + 1);
        Ok(())
    }

    fn prepare_store<'a>(
        &self,
        store: &'a BlockStored,
    ) -> Result<Option<PreparedEvent<'a>>, SourceError> {
        let namespace = store.lora_name.as_deref();
        if namespace.is_some_and(|name| self.adapter_path(name).is_none())
            || store.ownership.is_some()
        {
            return Ok(None);
        }
        let Some(tier) = StorageTier::from_medium(store.medium.as_deref()) else {
            return Ok(None);
        };
        let group = self
            .groups
            .iter()
            .find(|group| store.group_idx == Some(group.group_idx))
            .ok_or(SourceError::InvalidBatch)?;
        if !validate_hybrid_store(
            group,
            store,
            self.keys.block_size(),
            self.keys.lookahead_tokens(),
            self.partial_hash_hits,
        )
        .map_err(|()| SourceError::InvalidBatch)?
        {
            return Ok(None);
        }
        let hashes = store
            .block_hashes
            .iter()
            .map(sequence_hash)
            .collect::<Option<Vec<_>>>()
            .ok_or(SourceError::InvalidBatch)?;
        Ok(Some(PreparedEvent::Store {
            group: group.group_idx,
            tier,
            namespace,
            incremental: native_device(store.ownership.as_deref(), store.medium.as_deref()),
            pages: StoredPages::Hashes(hashes),
        }))
    }

    pub(in crate::kv_index) fn fail<T>(
        state: &mut State,
        tx: &mut IndexBatch<'_>,
        owner: &ResidencyOwner,
        error: SourceError,
    ) -> Result<T, SourceError> {
        tx.clear_owner_exact(owner);
        state.reset(None, false);
        Err(error)
    }
}

fn event_owner(
    source: &SourceId,
    dp_rank: u32,
    incarnation: u64,
    ownership: Option<&str>,
) -> ResidencyOwner {
    match ownership {
        Some(pool) if !pool.is_empty() => ResidencyOwner::CacheOwner {
            pool_id: Arc::from(pool),
            source: source.clone(),
            dp_rank,
        },
        _ => ResidencyOwner::Worker {
            source: source.clone(),
            dp_rank,
            incarnation,
        },
    }
}

fn event_tier(owner: &ResidencyOwner, medium: Option<&str>) -> Option<StorageTier> {
    match owner {
        ResidencyOwner::CacheOwner { .. } => Some(StorageTier::External),
        ResidencyOwner::Worker { .. } => StorageTier::from_medium(medium),
    }
}

pub(super) fn hash_str(hash: &ExternalBlockHash) -> Arc<str> {
    match hash {
        ExternalBlockHash::Int(value) => Arc::from(format!("i:{value}")),
        ExternalBlockHash::Bytes(bytes) => {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            let mut encoded = String::with_capacity(2 + bytes.len() * 2);
            encoded.push_str("b:");
            for &byte in bytes {
                encoded.push(char::from(HEX[usize::from(byte >> 4)]));
                encoded.push(char::from(HEX[usize::from(byte & 0xf)]));
            }
            Arc::from(encoded)
        }
    }
}
