//! Prepare engine events before committing their native hashes.

use crate::kv_hash::SequenceHash;
use crate::kv_index::storage::source_transaction::StreamState as State;
use crate::kv_index::storage::{IndexBatch, PageWriteError};
use crate::kv_index::wire::BlockStored;
use crate::kv_index::{
    KVEvent, KVEventBatch, ResidencyOwner, SourceBinding, SourceError, StorageTier,
};

use super::validate::{native_device, publishes_int_hash, sequence_hash, validate_store};

pub(super) enum PreparedEvent<'a> {
    Store {
        group: u32,
        tier: StorageTier,
        incremental: bool,
        namespace: Option<&'a str>,
        hashes: Vec<SequenceHash>,
    },
    Remove {
        group: u32,
        tier: StorageTier,
        incremental: bool,
        hashes: Vec<SequenceHash>,
    },
    Clear,
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
                hashes,
            } => tx.store_hash_pages(
                owner,
                group,
                tier,
                namespace,
                &hashes,
                incremental,
                max_entries,
            ),
            Self::Remove {
                group,
                tier,
                incremental,
                hashes,
            } => {
                tx.remove_hash_pages(owner, group, tier, &hashes, incremental);
                Ok(())
            }
            Self::Clear => {
                tx.clear_owner_exact(owner);
                Ok(())
            }
        }
    }
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
                    "KV event payload did not decode"
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
                        hashes,
                    });
                }
                KVEvent::AllBlocksCleared(_) => prepared.push(PreparedEvent::Clear),
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
            let clear = matches!(event, PreparedEvent::Clear);
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
        if self.strict_event_block_size && store.block_size != group.block_size {
            return Err(SourceError::InvalidBatch);
        }
        if !validate_store(
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
            hashes,
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
