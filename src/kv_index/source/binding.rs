//! Bind event sources to their layout, shared index and subscription lease.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use smallvec::SmallVec;

/// All ranks of a worker share `active`; owners of one canonical publisher and
/// exact topic share `strict_rank`. Neither identifies a remote engine lifetime.
/// Lock order is supervisor workers -> strict rank -> indexer. Keep the final
/// rank decision and index mutation together, never holding this lock over await.
#[derive(Clone)]
pub(in crate::kv_index) struct SourceLease {
    pub(in crate::kv_index) strict_rank: Arc<Mutex<bool>>,
    pub(in crate::kv_index) active: Arc<AtomicBool>,
}

impl SourceLease {
    pub(in crate::kv_index) fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    pub(in crate::kv_index) fn with_active<T>(
        &self,
        operation: impl FnOnce(bool) -> T,
    ) -> Option<T> {
        let strict = self.strict_rank.lock();
        self.is_active().then(|| operation(*strict))
    }
}

use crate::kv_hash::BlockKeyGenerator;
#[cfg(all(test, feature = "cpu", feature = "ready"))]
use crate::kv_index::layout::valid_groups;
use crate::kv_index::layout::{CacheGroup, GroupRule, TailDrop};
use crate::kv_index::storage::source_transaction::{StreamControl, StreamPhase, StreamState};
use crate::kv_index::storage::{
    IndexBatch, PageEvidence, PageMatch, PageQuery, PageTiers, SourceVersion,
};
use crate::kv_index::SourceError;
use crate::kv_index::{
    KvBlockIndexer, RequestHashCache, ResidencyOwner, RoutableTarget, StorageTier,
};

/// A bounded index for one verified identity, worker, rank and incarnation.
/// Stores and removals across all groups share one lock and one sequence.
pub struct SourceBinding {
    pub(in crate::kv_index) target: RoutableTarget,
    pub(in crate::kv_index) groups: Vec<CacheGroup>,
    pub(in crate::kv_index) keys: BlockKeyGenerator,
    /// Layout-declared restore-time recomputation window (vLLM
    /// `prefix_replay_tokens`). It applies to the whole layout: a group that
    /// publishes no events can still make this window mandatory.
    pub(in crate::kv_index) replay_tokens: u32,
    pub(in crate::kv_index) tail_drop_group: Option<u32>,
    mooncake_truncates_prompt: bool,
    /// LoRA adapters this worker serves, as `(name, path)`. The engine seeds an
    /// adapter's chain with both, so a layout that does not declare an adapter
    /// cannot credit it, and one that declares the wrong path credits nothing.
    pub(in crate::kv_index) adapters: Vec<(String, String)>,
    pub(in crate::kv_index) max_entries: usize,
    pub(in crate::kv_index) partial_hash_hits: bool,
    pub(in crate::kv_index) strict_event_block_size: bool,
    pub(in crate::kv_index) requires_root_proof: bool,
    /// The same domain index used by full-attention workers. Layout and stream
    /// state live here; cache residency lives only in this shared index.
    pub(in crate::kv_index) index: Arc<KvBlockIndexer>,
    pub(in crate::kv_index) owner: ResidencyOwner,
    pub(in crate::kv_index) control: Arc<StreamControl>,
    pub(in crate::kv_index) binding: Option<crate::kv_index::source::binding::SourceLease>,
}

type PageSnapshot<'a> = (SmallVec<[PageTiers<'a>; 1]>, SourceVersion, u64);

impl SourceBinding {
    /// Test constructor keyed by a `PYTHONHASHSEED` instead of a root digest.
    #[cfg_attr(coverage_nightly, coverage(off))]
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    pub fn new(
        target: RoutableTarget,
        groups: Vec<CacheGroup>,
        hash_block_size: u32,
        hash_seed: u32,
        max_entries: usize,
    ) -> Result<Self, SourceError> {
        if max_entries == 0 || !valid_groups(&groups, hash_block_size as usize) {
            return Err(SourceError::InvalidLayout);
        }
        Ok(Self::bind(
            target,
            groups,
            max_entries,
            BlockKeyGenerator::new(hash_block_size as usize, u64::from(hash_seed)),
        ))
    }

    fn bind(
        target: RoutableTarget,
        groups: Vec<CacheGroup>,
        max_entries: usize,
        keys: BlockKeyGenerator,
    ) -> Self {
        let incarnation = crate::kv_index::next_owner_incarnation();
        let owner = ResidencyOwner::Worker {
            source: target.instance_id.clone(),
            dp_rank: target.dp_rank,
            incarnation,
        };
        Self {
            target,
            groups,
            keys,
            replay_tokens: 0,
            tail_drop_group: None,
            mooncake_truncates_prompt: false,
            adapters: Vec::new(),
            max_entries,
            partial_hash_hits: false,
            strict_event_block_size: false,
            requires_root_proof: false,
            index: Arc::new(KvBlockIndexer::new()),
            owner,
            control: Arc::new(StreamControl::default()),
            binding: None,
        }
    }

    /// Bind an admitted layout to its supervisor's cache domain before replay.
    pub(crate) fn with_shared_index(mut self, index: Arc<KvBlockIndexer>) -> Self {
        debug_assert!(self
            .index
            .with_source_snapshot(&self.target, |view| {
                view.stream(&self.owner)
                    .is_none_or(|state| state.next_seq.is_none())
            })
            .unwrap_or(true));
        self.index = index;
        self
    }

    /// An inferred plain layout needs a Store proving the actual hash root.
    pub(in crate::kv_index) fn require_root_proof(mut self) -> Self {
        self.requires_root_proof = true;
        self.strict_event_block_size = true;
        self
    }

    pub(in crate::kv_index) fn with_source_binding(
        mut self,
        binding: crate::kv_index::source::binding::SourceLease,
    ) -> Self {
        self.binding = Some(binding);
        self
    }

    pub(in crate::kv_index) fn with_active_binding<R>(
        &self,
        operation: impl FnOnce() -> R,
    ) -> Option<R> {
        match &self.binding {
            Some(binding) => binding.with_active(|_| operation()),
            None => Some(operation()),
        }
    }

    pub(in crate::kv_index) fn with_stream_batch<R>(
        &self,
        operation: impl FnOnce(&mut StreamState, &mut IndexBatch<'_>) -> R,
    ) -> R {
        self.index
            .with_stream_batch(&self.target, &self.owner, &self.control, operation)
    }

    pub(in crate::kv_index) fn with_page_mutation_batch<R>(
        &self,
        planned: Option<crate::kv_index::storage::PageChanges>,
        operation: impl FnOnce(&mut StreamState, &mut IndexBatch<'_>) -> R,
    ) -> R {
        self.index.with_page_mutation_batch(
            &self.target,
            &self.owner,
            &self.control,
            planned,
            operation,
        )
    }

    /// Declare the layout's restore-time recomputation window
    /// (`prefix_replay_tokens`); it lowers the credit for the whole layout.
    fn with_replay_tokens(mut self, replay_tokens: u32) -> Self {
        self.replay_tokens = replay_tokens;
        self
    }

    fn with_tail_drop_group(mut self, group: Option<u32>) -> Self {
        self.tail_drop_group = group;
        self
    }

    pub(in crate::kv_index) fn with_mooncake_prefill(
        mut self,
        enabled: bool,
        truncates_compressed: bool,
    ) -> Self {
        self.mooncake_truncates_prompt = enabled
            && (self.replay_tokens != 0
                || self.groups.iter().any(|group| {
                    group.kind == GroupRule::Checkpoint
                        || (truncates_compressed
                            && matches!(group.kind, GroupRule::CompressedFullAttention { .. }))
                }));
        self
    }

    pub(in crate::kv_index) fn reusable_prompt_cap(&self, len: usize) -> usize {
        // A Mooncake connector can remove the last prompt token before cache
        // lookup. Keep the full hash chain and lower only the lookup limit.
        len.saturating_sub(1 + usize::from(self.mooncake_truncates_prompt))
    }

    /// Enable only when the source contract identifies hash-sized stores as
    /// fine-grained prefix entries, rather than secondary pages.
    fn with_partial_hash_hits(mut self, enabled: bool) -> Self {
        self.partial_hash_hits = enabled;
        self
    }

    /// Declare the LoRA adapters this worker serves. Adapter pages are a
    /// separate chain keyed by `(name, path)`; without the declaration their
    /// pages are skipped rather than credited to the base model.
    pub(super) fn with_adapters(mut self, adapters: Vec<(String, String)>) -> Self {
        self.adapters = adapters;
        self
    }

    pub(in crate::kv_index) fn adapter_path(&self, name: &str) -> Option<&str> {
        self.adapters
            .iter()
            .find(|(declared, _)| declared == name)
            .map(|(_, path)| path.as_str())
    }

    /// Begin tracking from an externally established empty-cache barrier.
    /// This is not a snapshot or replay implementation. Never call it merely
    /// because a subscription started or a gap occurred.
    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(crate) fn begin_empty(&self, next_seq: u64) {
        self.with_active_binding(|| {
            self.with_stream_batch(|state, tx| {
                self.reset_stream(state, tx, Some(next_seq), true);
            });
        });
    }

    /// Start a full replay from an externally verified empty origin. Queries
    /// remain cold until the caller verifies a continuous replay watermark.
    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(in crate::kv_index::source) fn begin_replay(&self, next_seq: u64) {
        self.with_active_binding(|| {
            self.with_stream_batch(|state, tx| {
                self.reset_stream(state, tx, Some(next_seq), false);
            });
        });
    }

    pub(in crate::kv_index::source) fn reset_stream(
        &self,
        state: &mut StreamState,
        tx: &mut IndexBatch<'_>,
        next_seq: Option<u64>,
        visible: bool,
    ) {
        tx.clear_owner_exact(&self.owner);
        state.reset(next_seq, visible);
        if visible {
            tx.resume_source(&self.target.instance_id, self.target.dp_rank);
        }
    }

    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(in crate::kv_index::source) fn finish_replay(
        &self,
        expected_next_seq: u64,
    ) -> Result<(), SourceError> {
        self.with_active_binding(|| {
            self.with_stream_batch(|state, tx| self.publish_stream(state, tx, expected_next_seq))
        })
        .unwrap_or(Err(SourceError::Untrusted))
    }

    pub(in crate::kv_index::source) fn publish_stream(
        &self,
        state: &mut StreamState,
        tx: &mut IndexBatch<'_>,
        expected_next_seq: u64,
    ) -> Result<(), SourceError> {
        if state.phase != StreamPhase::Replaying || state.next_seq != Some(expected_next_seq) {
            return Self::fail(state, tx, &self.owner, SourceError::Sequence);
        }
        if !self.requires_root_proof || state.root_verified {
            tx.resume_source(&self.target.instance_id, self.target.dp_rank);
            state.publish();
        }
        // Track subsequent sequences even if no root store has been seen yet.
        state.phase = StreamPhase::Live;
        Ok(())
    }

    /// Discard all claims on a gap, restart, lease retirement or lost event.
    pub(in crate::kv_index::source) fn invalidate(&self) {
        self.with_stream_batch(|state, tx| {
            self.reset_stream(state, tx, None, false);
            state.lease_id = 0;
        });
    }

    /// Calculate one consistent local snapshot. Zero means a miss or an
    /// untrusted source.
    pub fn reusable_tokens(&self, tokens: &[u32], max_reusable_tokens: usize) -> u64 {
        self.reusable_tokens_in(None, tokens, max_reusable_tokens)
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[path = "binding_tests.rs"]
pub(crate) mod tests;

use crate::kv_index::source::endpoints::EventSource;
use crate::kv_index::CacheCapabilities;

impl CacheCapabilities {
    /// Construct an index for one rank and its event source. Unknown layouts
    /// get no credit.
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(crate) fn index_for_source(
        &self,
        source: &EventSource,
        target: RoutableTarget,
        max_entries: usize,
    ) -> Result<SourceBinding, &'static str> {
        self.bind_source(source, target, None, max_entries)
    }

    pub(crate) fn bind_worker_source(
        &self,
        source: &EventSource,
        model: &str,
        max_entries: usize,
    ) -> Result<SourceBinding, &'static str> {
        let target = RoutableTarget {
            instance_id: source.source.clone(),
            dp_rank: source.dp_rank,
        };
        self.bind_source(source, target, Some(model), max_entries)
    }

    fn bind_source(
        &self,
        source: &EventSource,
        target: RoutableTarget,
        model: Option<&str>,
        max_entries: usize,
    ) -> Result<SourceBinding, &'static str> {
        if target.dp_rank != source.dp_rank || target.instance_id != source.source {
            return Err("invalid capability or source binding");
        }
        if max_entries == 0 {
            return Err("invalid group layout");
        }
        let layout = self.verify_layout(
            source.dp_rank,
            model,
            Some(&source.topic),
            source.replay_endpoint.as_deref(),
        )?;
        let keys = BlockKeyGenerator::from_root_hash(layout.hash_block_size as usize, layout.root)
            .with_lookahead(layout.hash_lookahead_tokens);
        let mut binding = SourceBinding::bind(target, layout.groups, max_entries, keys)
            .with_replay_tokens(layout.replay_tokens)
            .with_tail_drop_group(layout.tail_drop_group)
            .with_adapters(layout.adapters)
            .with_partial_hash_hits(layout.partial_hash_hits);
        binding.strict_event_block_size = self
            .rank_capability(source.dp_rank)
            .is_some_and(|rank| rank.has_plain_full_attention_shape());
        Ok(binding)
    }

    /// Check startup declarations. Registration also checks the observed topic.
    pub(crate) fn validate_declared_source(
        &self,
        rank: u32,
        model: &str,
        replay_endpoint: &str,
    ) -> Result<(), &'static str> {
        self.verify_layout(rank, Some(model), None, Some(replay_endpoint))
            .map(drop)
    }
}

impl SourceBinding {
    pub(in crate::kv_index) fn page_snapshot<'a, 'b>(
        &self,
        namespace: Option<&str>,
        tokens: &'a [u32],
        stop_at_gap: bool,
        cache: &'b mut RequestHashCache<'a>,
    ) -> Option<PageSnapshot<'b>> {
        if self
            .binding
            .as_ref()
            .is_some_and(|binding| !binding.is_active())
        {
            return None;
        }
        let generation = self.control.generation();
        if generation == 0 {
            return None;
        }
        let adapter = match namespace {
            None => None,
            Some(name) => Some((name, self.adapter_path(name)?)),
        };
        let queries = self.groups.iter().map(|group| PageQuery {
            group: group.group_idx,
            // One hash must cover the entire physical page. A smaller hash may
            // be absent even when a later endpoint proves the same page.
            stop_at_gap: stop_at_gap
                && group.kind == GroupRule::FullAttention
                && group.block_size as usize == self.keys.block_size(),
        });
        let (prepared, evidence) =
            cache.pages(tokens, &self.keys, adapter, &self.index, queries.clone());
        if let Some(matched) = PageEvidence::unchanged_groups(
            &self.index,
            &self.target,
            &self.owner,
            prepared,
            queries.clone().map(|query| evidence.get(&query)),
        ) {
            return match matched {
                PageMatch::Miss => None,
                PageMatch::Hit(claims, version) => Some((claims, version, generation)),
            };
        }
        // Borrow unchanged evidence. Refresh all required groups together
        // when the source changed since collection.
        self.index
            .with_source_snapshot(&self.target, |view| {
                let state = view.stream(&self.owner)?;
                if state.next_seq.is_none() || !state.visible() {
                    return None;
                }
                // A hybrid restore needs a reachable endpoint in the first physical
                // page of every dense group. Later hashes still count for plain full
                // attention, which reports segments beyond gaps.
                if self.groups.len() > 1 {
                    if let Some(dense) = self.groups.iter().find(|group| {
                        matches!(
                            group.kind,
                            GroupRule::FullAttention | GroupRule::CompressedFullAttention { .. }
                        )
                    }) {
                        let width = dense.block_size as usize / self.keys.block_size();
                        if prepared.len() > width
                            && !view.contains_any(
                                &self.owner,
                                dense.group_idx,
                                namespace,
                                &prepared[..width],
                            )
                        {
                            return None;
                        }
                    }
                }
                let claims =
                    view.hash_page_runs(&self.owner, namespace, prepared, evidence, queries);
                Some((claims, view.source_version(), state.reset_generation))
            })
            .flatten()
    }

    /// The credited prefix and the tier it is credited at. A page held in host
    /// memory or on storage is still restorable, but the policy weighs it by the
    /// transfer credit of its tier, so the boundary is reported at the slowest
    /// tier that any group needs to satisfy it.
    pub(in crate::kv_index) fn credit_parts<'a>(
        &self,
        namespace: Option<&str>,
        tokens: &'a [u32],
        max_reusable_tokens: usize,
        cache: &mut RequestHashCache<'a>,
    ) -> (u64, Option<StorageTier>) {
        let snapshot = self.page_snapshot(namespace, tokens, true, cache);
        let Some((claims, version, reset_generation)) = snapshot else {
            return (0, None);
        };
        if claims
            .iter()
            .all(|pages| pages.prefix_len == 0 && pages.tail.is_empty())
        {
            return (0, None);
        }
        let (credited, tier) = crate::kv_index::layout::credit_from_pages(
            &self.groups,
            &claims,
            self.keys.block_size(),
            self.reusable_prompt_cap(tokens.len())
                .min(max_reusable_tokens) as u64,
            self.replay_tokens,
            TailDrop {
                group: self.tail_drop_group,
                fine_grained: self.partial_hash_hits,
            },
        );
        if tier.is_none() || u32::try_from(credited / self.keys.block_size() as u64).is_err() {
            return (0, None);
        }
        if !self.snapshot_is_current(&version, reset_generation) {
            return (0, None);
        }
        (credited, tier)
    }
    pub(in crate::kv_index) fn snapshot_is_current(
        &self,
        version: &SourceVersion,
        generation: u64,
    ) -> bool {
        self.control.generation() == generation
            && self.index.source_version_is_current(version)
            && self.binding.as_ref().is_none_or(SourceLease::is_active)
    }

    pub(in crate::kv_index) fn reusable_tokens_in(
        &self,
        namespace: Option<&str>,
        tokens: &[u32],
        max_reusable_tokens: usize,
    ) -> u64 {
        self.credit_parts(
            namespace,
            tokens,
            max_reusable_tokens,
            &mut RequestHashCache::for_single_target(),
        )
        .0
    }
}
