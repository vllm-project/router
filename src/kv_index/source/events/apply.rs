//! Prepare engine events before committing their native hashes.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::kv_index::request_hash::SequenceHash;
use crate::kv_index::storage::source_transaction::{StreamPhase, StreamState as State};
use crate::kv_index::storage::{IndexBatch, PageChanges, PageWriteError, PageWriteOutcome};
use crate::kv_index::wire::BlockStored;
use crate::kv_index::{
    KVEvent, KVEventBatch, ResidencyOwner, SourceBinding, SourceError, StorageTier,
};

use crate::metrics::RouterMetrics;

use super::validate::{publishes_int_hash, sequence_hash, validate_store};

fn tier_or_warn(medium: Option<&str>) -> Option<StorageTier> {
    static WARNED: AtomicBool = AtomicBool::new(false);
    let tier = StorageTier::from_medium(medium);
    if let (None, Some(medium)) = (tier, medium) {
        if !WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(medium, "unknown KV cache medium; ignoring its events");
        }
    }
    tier
}

pub(super) enum PreparedEvent<'a> {
    Store {
        group: u32,
        tier: StorageTier,
        incremental: bool,
        namespace: Option<&'a str>,
        hashes: Vec<SequenceHash>,
        proves_root: bool,
    },
    Remove {
        group: u32,
        tier: StorageTier,
        incremental: bool,
        hashes: Vec<SequenceHash>,
    },
    Clear,
}

#[derive(Clone, Copy)]
pub(super) enum EventKind {
    Stored,
    Removed,
    Cleared,
}

impl EventKind {
    const ALL: [Self; 3] = [Self::Stored, Self::Removed, Self::Cleared];

    fn label(self) -> &'static str {
        match self {
            Self::Stored => "stored",
            Self::Removed => "removed",
            Self::Cleared => "cleared",
        }
    }
}

pub(super) struct Rejection {
    cause: SourceError,
    status: &'static str,
}

impl Rejection {
    fn invalid(status: &'static str) -> Self {
        Self {
            cause: SourceError::InvalidBatch,
            status,
        }
    }
}

impl From<SourceError> for Rejection {
    fn from(cause: SourceError) -> Self {
        Self {
            cause,
            status: "invalid_block",
        }
    }
}

impl PreparedEvent<'_> {
    pub(super) fn page_changes(events: &[Self], max_entries: usize) -> Option<PageChanges> {
        PageChanges::from_batches(
            events.iter().map(|event| match event {
                Self::Store { hashes, .. } | Self::Remove { hashes, .. } => Some(hashes.as_slice()),
                Self::Clear => None,
            }),
            max_entries,
        )
    }

    fn kind(&self) -> EventKind {
        match self {
            Self::Store { .. } => EventKind::Stored,
            Self::Remove { .. } => EventKind::Removed,
            Self::Clear => EventKind::Cleared,
        }
    }

    fn apply(
        self,
        tx: &mut IndexBatch<'_>,
        owner: &ResidencyOwner,
        max_entries: usize,
    ) -> Result<PageWriteOutcome, PageWriteError> {
        match self {
            Self::Store {
                group,
                tier,
                incremental,
                namespace,
                hashes,
                ..
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
            } => tx.remove_hash_pages(owner, group, tier, &hashes, incremental),
            Self::Clear => {
                tx.clear_owner_exact(owner);
                Ok(PageWriteOutcome::Applied)
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
        let planned = prepared
            .as_deref()
            .ok()
            .and_then(|events| PreparedEvent::page_changes(events, self.max_entries));
        self.with_active_binding(|| {
            self.with_page_mutation_batch(planned, |state, tx| {
                self.apply_prepared(state, tx, seq, batch, prepared)
            })
        })
        .unwrap_or(Err(SourceError::Untrusted))
    }

    pub(super) fn prepare_batch<'a>(
        &self,
        batch: &'a KVEventBatch,
    ) -> Result<Vec<PreparedEvent<'a>>, (EventKind, Rejection)> {
        // MooncakeStoreConnector reports writes as CPU stores without a cache
        // spec kind. That event shape cannot establish worker-local residency.
        let events = batch.events.iter().filter(|event| {
            !matches!(event, KVEvent::BlockStored(store)
                if store.kv_cache_spec_kind.is_none()
                    && store.medium.as_deref().is_some_and(|medium| medium.eq_ignore_ascii_case("CPU")))
        });
        let mut prepared = Vec::with_capacity(batch.events.len());
        for event in events {
            let kind = match event {
                KVEvent::BlockStored(_) => EventKind::Stored,
                KVEvent::BlockRemoved(_) => EventKind::Removed,
                KVEvent::AllBlocksCleared(_) => EventKind::Cleared,
            };
            if publishes_int_hash(event) {
                return Err((kind, SourceError::IntBlockHashes.into()));
            }
            let next = match event {
                KVEvent::BlockStored(store) => self.prepare_store(store),
                KVEvent::BlockRemoved(remove) => self.prepare_remove(remove),
                KVEvent::AllBlocksCleared(_) => Ok(Some(PreparedEvent::Clear)),
            }
            .map_err(|error| (kind, error))?;
            prepared.extend(next);
        }
        Ok(prepared)
    }

    fn prepare_remove(
        &self,
        remove: &super::wire::BlockRemoved,
    ) -> Result<Option<PreparedEvent<'static>>, Rejection> {
        if remove.ownership.is_some() {
            return Ok(None);
        }
        let Some(tier) = tier_or_warn(remove.medium.as_deref()) else {
            return Ok(None);
        };
        if (self.requires_root_proof && remove.group_idx != Some(0))
            || !self
                .groups
                .iter()
                .any(|group| remove.group_idx == Some(group.group_idx))
        {
            return Err(Rejection::invalid("unsupported_group"));
        }
        if remove.locality.as_deref().is_some_and(|s| s != "LOCAL")
            || remove.block_hashes.is_empty()
        {
            return Err(SourceError::InvalidBatch.into());
        }
        let hashes = remove
            .block_hashes
            .iter()
            .map(sequence_hash)
            .collect::<Option<Vec<_>>>()
            .ok_or(SourceError::InvalidBatch)?;
        Ok(Some(PreparedEvent::Remove {
            group: remove.group_idx.expect("validated group"),
            tier,
            incremental: tier == StorageTier::Device,
            hashes,
        }))
    }

    /// The caller validates its lease in this same source transaction.
    pub(super) fn apply_prepared(
        &self,
        state: &mut State,
        tx: &mut IndexBatch<'_>,
        seq: u64,
        batch: &KVEventBatch,
        prepared: Result<Vec<PreparedEvent<'_>>, (EventKind, Rejection)>,
    ) -> Result<(), SourceError> {
        let Some(next_seq) = state.next_seq else {
            return Err(SourceError::Untrusted);
        };
        if seq != next_seq || seq == u64::MAX {
            if seq < next_seq {
                RouterMetrics::record_kv_event_warning("stale_sequence", 1);
            }
            return Self::fail(state, tx, &self.owner, SourceError::Sequence);
        }
        // The vLLM publisher stamps every batch with its configured DP rank.
        if batch.data_parallel_rank != Some(self.target.dp_rank) {
            return Self::fail(state, tx, &self.owner, SourceError::InvalidBatch);
        }
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err((kind, rejection)) => {
                if matches!(
                    rejection.cause,
                    SourceError::IntBlockHashes | SourceError::RootHashMismatch
                ) {
                    self.control
                        .hash_contract_failed
                        .store(true, Ordering::Relaxed);
                    RouterMetrics::record_kv_hash_contract_error();
                }
                RouterMetrics::record_kv_events(kind.label(), rejection.status, 1);
                return Self::fail(state, tx, &self.owner, rejection.cause);
            }
        };
        let mut accepted = [0_u64; 3];
        let mut duplicate_stores = 0;
        let mut unknown_removes = 0;
        let mut proved_root = false;
        for event in prepared {
            // Keep observations before the root proof under the same source
            // barrier. Removes and clears must still apply while it is hidden.
            if !state.root_verified
                && matches!(
                    &event,
                    PreparedEvent::Store {
                        proves_root: true,
                        ..
                    }
                )
            {
                state.root_verified = true;
                proved_root = true;
                tracing::debug!(
                    source = %self.target.instance_id,
                    rank = self.target.dp_rank,
                    seq,
                    "verified default KV hash root"
                );
            }
            let kind = event.kind();
            match event.apply(tx, &self.owner, self.max_entries) {
                Ok(PageWriteOutcome::Applied) => {}
                Ok(PageWriteOutcome::DuplicateStore) => duplicate_stores += 1,
                Ok(PageWriteOutcome::UnknownRemove) => unknown_removes += 1,
                Err(error) => {
                    let (error, status) = match error {
                        PageWriteError::Capacity => {
                            RouterMetrics::record_kv_event_warning("capacity_exhausted", 1);
                            (SourceError::Capacity, "capacity_exhausted")
                        }
                        PageWriteError::InvalidResidencyMode => {
                            (SourceError::InvalidBatch, "invalid_block")
                        }
                    };
                    RouterMetrics::record_kv_events(kind.label(), status, 1);
                    return Self::fail(state, tx, &self.owner, error);
                }
            }
            accepted[kind as usize] += 1;
            if matches!(kind, EventKind::Cleared) {
                state.bump_generation();
            }
        }
        // A later rejection invalidates the whole batch, including earlier writes.
        for (kind, n) in EventKind::ALL.into_iter().zip(accepted) {
            if n > 0 {
                RouterMetrics::record_kv_events(kind.label(), "ok", n);
            }
        }
        for (kind, count) in [
            ("duplicate_store", duplicate_stores),
            ("unknown_remove", unknown_removes),
        ] {
            if count > 0 {
                RouterMetrics::record_kv_event_warning(kind, count);
            }
        }
        if accepted[EventKind::Stored as usize] > 0
            && (!self.requires_root_proof || state.root_verified)
            && self.control.hash_contract_failed.load(Ordering::Relaxed)
        {
            self.control
                .hash_contract_failed
                .store(false, Ordering::Relaxed);
        }
        state.next_seq = Some(seq + 1);
        if proved_root && state.phase == StreamPhase::Live && !state.visible() {
            tx.resume_source(&self.target.instance_id, self.target.dp_rank);
            state.publish();
        }
        Ok(())
    }

    fn prepare_store<'a>(
        &self,
        store: &'a BlockStored,
    ) -> Result<Option<PreparedEvent<'a>>, Rejection> {
        let Some(tier) = tier_or_warn(store.medium.as_deref()) else {
            return Ok(None);
        };
        let namespace = store.lora_name.as_deref();
        // Undeclared adapters have no claims in this source.
        if namespace.is_some_and(|name| self.adapter_path(name).is_none())
            || (self.requires_root_proof && store.lora_id.is_some())
            || store.ownership.is_some()
        {
            return Ok(None);
        }
        let group = self
            .groups
            .iter()
            .find(|group| store.group_idx == Some(group.group_idx))
            .ok_or_else(|| Rejection::invalid("unsupported_group"))?;
        if self.requires_root_proof && store.group_idx != Some(0) {
            return Err(Rejection::invalid("unsupported_group"));
        }
        if self.requires_root_proof && store.block_size as usize != self.keys.block_size()
            || self.strict_event_block_size && store.block_size != group.block_size
        {
            return Err(Rejection::invalid("block_size_mismatch"));
        }
        if self.requires_root_proof
            && (store.kv_cache_spec_kind.as_deref() != Some("full_attention")
                || store.kv_cache_spec_sliding_window.is_some()
                || store.lookahead_token_ids.is_some())
        {
            return Err(SourceError::InvalidBatch.into());
        }
        if !validate_store(
            group,
            store,
            self.keys.block_size(),
            self.keys.lookahead_tokens(),
            self.partial_hash_hits,
        )? {
            return Ok(None);
        }
        let hashes = store
            .block_hashes
            .iter()
            .map(sequence_hash)
            .collect::<Option<Vec<_>>>()
            .ok_or(SourceError::InvalidBatch)?;
        let incremental = tier == StorageTier::Device;
        let proves_root = self.requires_root_proof
            && namespace.is_none()
            && incremental
            && store.parent_block_hash.is_none();
        if proves_root && self.keys.plain_root_block(&store.token_ids).as_ref() != hashes.first() {
            return Err(SourceError::RootHashMismatch.into());
        }
        Ok(Some(PreparedEvent::Store {
            group: group.group_idx,
            tier,
            namespace,
            incremental,
            hashes,
            proves_root,
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
