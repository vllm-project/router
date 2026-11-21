//! Full-attention KV index and source recovery.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::{mapref::entry::Entry, DashMap};
use parking_lot::Mutex;

use super::storage::{
    IndexBatch, KvBlockIndexer as PageIndex, PageChanges, PageMatch, PageQuery, PageTiers,
};
use crate::kv_hash::SequenceHash;

use crate::kv_index::types::{
    ClearScope, Locality, RecoveryBoundary, ResidencyOwner, RoutableTarget, SourceId,
    SourceRecoveryReason, SourceRecoveryState, SourceRecoveryStatus, StorageTier,
};
use crate::metrics::RouterMetrics;

/// One request's query against the index.
#[derive(Debug, Clone)]
pub struct MatchQuery {
    /// KV-cache group. Ordinary full attention uses group zero.
    pub group_idx: u32,
    /// Complete engine-compatible block hashes, in prompt order.
    pub hashes: Vec<SequenceHash>,
    /// Tiers to report; the cost model requests the tiers it scores.
    pub tiers_of_interest: Vec<StorageTier>,
}

/// One hit: the block intervals `[start, end)` a target holds at this tier,
/// each maximal. Flat — one row per `(target, tier)` — so the cost model
/// applies two-segment reuse over a target's tiers in one pass.
#[derive(Debug, Clone)]
pub struct TierMatch {
    pub target: RoutableTarget,
    pub tier: StorageTier,
    /// Held block intervals `[start, end)`, each maximal; a gap closes a
    /// segment but later segments survive.
    pub segments: Vec<(u32, u32)>,
    pub locality: Locality,
}

/// Shared page residency with source-scoped replay and trust.
pub struct KvBlockIndexer {
    pages: PageIndex,
    source_mutations: Mutex<()>,
    recovery_attempt: AtomicU64,
    source_states: DashMap<(SourceId, u32), SourceRecoveryRecord>,
    /// Reject query snapshots crossing an invalidation/resume boundary.
    source_generation: AtomicU64,
}

#[derive(Debug, Clone, Copy)]
struct SourceRecoveryRecord {
    status: SourceRecoveryStatus,
    attempt: u64,
    replay_state: Option<SourceRecoveryState>,
}

impl KvBlockIndexer {
    pub fn new() -> Self {
        KvBlockIndexer {
            pages: PageIndex::new(),
            source_mutations: Mutex::new(()),
            recovery_attempt: AtomicU64::new(0),
            source_states: DashMap::new(),
            source_generation: AtomicU64::new(0),
        }
    }

    /// Register a source before its subscriber can publish observations.
    /// Newly registered sources remain hidden until the subscriber establishes an
    /// authoritative sequence or clear boundary.
    pub(crate) fn register_source(&self, source: &SourceId, dp_rank: u32) {
        let _guard = self.source_mutations.lock();
        debug_assert!(!self.source_states.contains_key(&(source.clone(), dp_rank)));
        self.transition_source(
            source,
            dp_rank,
            SourceRecoveryState::Fallback,
            SourceRecoveryReason::Registered,
        );
    }

    fn transition_source(
        &self,
        source: &SourceId,
        dp_rank: u32,
        state: SourceRecoveryState,
        reason: SourceRecoveryReason,
    ) {
        let key = (source.clone(), dp_rank);
        let status = SourceRecoveryStatus { state, reason };
        let previous = match self.source_states.entry(key) {
            Entry::Occupied(mut entry) => {
                let previous = entry.get().status;
                if previous == status {
                    return;
                }
                entry.insert(SourceRecoveryRecord {
                    status,
                    attempt: self.recovery_attempt.fetch_add(1, Ordering::Relaxed),
                    replay_state: None,
                });
                Some(previous)
            }
            Entry::Vacant(entry) => {
                entry.insert(SourceRecoveryRecord {
                    status,
                    attempt: self.recovery_attempt.fetch_add(1, Ordering::Relaxed),
                    replay_state: None,
                });
                None
            }
        };
        tracing::debug!(source = %source, rank = dp_rank,
            from = previous.map_or("unregistered", |status| status.state.as_str()),
            to = state.as_str(), reason = reason.as_str(),
            "kv_index source recovery transition");
        RouterMetrics::record_kv_source_recovery_transition(
            previous.map(|status| (status.state.as_str(), status.reason.as_str())),
            state.as_str(),
            reason.as_str(),
        );
    }

    pub(super) fn write_batch<R>(
        &self,
        source: &SourceId,
        dp_rank: u32,
        changes: Option<PageChanges>,
        apply: impl FnOnce(&mut IndexBatch<'_>) -> R,
    ) -> R {
        self.pages.with_source_pages(
            &RoutableTarget {
                instance_id: source.clone(),
                dp_rank,
            },
            changes,
            apply,
        )
    }

    pub fn store(
        &self,
        group: u32,
        owner: ResidencyOwner,
        tier: StorageTier,
        hashes: &[SequenceHash],
    ) {
        let (source, rank) = owner.target_parts();
        let changes = PageChanges::from_batches(std::iter::once(Some(hashes)), usize::MAX);
        self.write_batch(source, rank, changes, |batch| {
            batch.store_hash_pages(&owner, group, tier, None, hashes, false, usize::MAX)
        })
        .expect("idempotent store into an uncapped page table");
    }

    pub fn remove(
        &self,
        group: u32,
        owner: &ResidencyOwner,
        tier: StorageTier,
        hashes: &[SequenceHash],
    ) {
        let (source, rank) = owner.target_parts();
        let changes = PageChanges::from_batches(std::iter::once(Some(hashes)), usize::MAX);
        self.write_batch(source, rank, changes, |batch| {
            batch.remove_hash_pages(owner, group, tier, hashes, false);
        });
    }

    pub fn clear(&self, owner: &ResidencyOwner, scope: ClearScope) {
        match owner {
            ResidencyOwner::Worker {
                source, dp_rank, ..
            } => {
                if scope != ClearScope::CacheOwner {
                    self.write_batch(source, *dp_rank, None, |batch| {
                        batch.clear_pages(|candidate| {
                            matches!(candidate, ResidencyOwner::Worker { .. })
                        })
                    });
                }
            }
            ResidencyOwner::CacheOwner { pool_id, .. } => {
                if scope != ClearScope::Worker {
                    for target in self.pages.targets() {
                        self.write_batch(&target.instance_id, target.dp_rank, None, |batch| {
                            batch.clear_pages(|candidate| {
                                matches!(candidate, ResidencyOwner::CacheOwner { pool_id: id, .. } if id == pool_id)
                            })
                        });
                    }
                }
            }
        }
    }

    /// Hide a source's observations while contiguous replay repairs its state.
    /// Correctness does not depend on delivery of advisory ingestion signals.
    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(super) fn suspend_source(&self, source: &SourceId, dp_rank: u32) {
        self.suspend_source_for(source, dp_rank, SourceRecoveryReason::SequenceGap);
    }

    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(super) fn suspend_source_for(
        &self,
        source: &SourceId,
        dp_rank: u32,
        reason: SourceRecoveryReason,
    ) -> u64 {
        let _guard = self.source_mutations.lock();
        self.suspend_source_locked(source, dp_rank, reason, SourceRecoveryState::Exact)
    }

    fn suspend_source_locked(
        &self,
        source: &SourceId,
        dp_rank: u32,
        reason: SourceRecoveryReason,
        replay_state: SourceRecoveryState,
    ) -> u64 {
        self.transition_source(source, dp_rank, SourceRecoveryState::Degraded, reason);
        self.source_generation.fetch_add(1, Ordering::AcqRel);
        let attempt = self.recovery_attempt.fetch_add(1, Ordering::Relaxed);
        let mut record = self
            .source_states
            .get_mut(&(source.clone(), dp_rank))
            .unwrap();
        record.attempt = attempt;
        record.replay_state = Some(replay_state);
        attempt
    }

    pub(crate) fn begin_replay(
        &self,
        source: &SourceId,
        dp_rank: u32,
        last_seq: i64,
        strict: bool,
    ) -> (i64, u64) {
        let _guard = self.source_mutations.lock();
        use SourceRecoveryState::{Exact, Observed, WorkerExact};
        let baseline = self
            .source_states
            .get(&(source.clone(), dp_rank))
            .map(|r| r.status.state);
        // A suffix can preserve a trusted baseline, but cannot expand its scope.
        // Without one, strict recovery rebuilds the publisher's worker history from zero.
        let (start, replay_state) = match (last_seq, baseline) {
            (cursor, Some(state @ (Exact | WorkerExact))) if cursor >= 0 => (cursor, state),
            _ if strict => (-1, WorkerExact),
            (-1, _) => (-1, Exact), // legacy compatibility mode
            _ => (last_seq, Observed),
        };
        let attempt = self.suspend_source_locked(
            source,
            dp_rank,
            SourceRecoveryReason::SequenceGap,
            replay_state,
        );
        (start, attempt)
    }

    /// Stop crediting a lossy source and discard all its old observations.
    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(super) fn invalidate_source(&self, source: &SourceId, dp_rank: u32) {
        self.invalidate_source_for(source, dp_rank, SourceRecoveryReason::ValidationFailed);
    }

    pub(crate) fn invalidate_source_for(
        &self,
        source: &SourceId,
        dp_rank: u32,
        reason: SourceRecoveryReason,
    ) {
        let _guard = self.source_mutations.lock();
        self.invalidate_source_locked(source, dp_rank, reason);
    }

    fn invalidate_source_locked(
        &self,
        source: &SourceId,
        dp_rank: u32,
        reason: SourceRecoveryReason,
    ) {
        self.transition_source(source, dp_rank, SourceRecoveryState::Fallback, reason);
        self.source_generation.fetch_add(1, Ordering::AcqRel);
        self.pages.retire_source(source, dp_rank);
    }

    /// Run one synchronous replay mutation under the same fence as source invalidation.
    /// Drop the map entry before the operation, which may update the source state.
    pub(super) fn with_current_replay<T>(
        &self,
        source: &SourceId,
        dp_rank: u32,
        attempt: u64,
        operation: impl FnOnce() -> T,
    ) -> Option<T> {
        let _guard = self.source_mutations.lock();
        {
            let record = self.source_states.get(&(source.clone(), dp_rank))?;
            if record.attempt != attempt
                || record.status.state != SourceRecoveryState::Degraded
                || record.status.reason != SourceRecoveryReason::SequenceGap
                || record.replay_state.is_none()
            {
                return None;
            }
        }
        Some(operation())
    }

    pub(crate) fn finish_replay_if_current(
        &self,
        source: &SourceId,
        dp_rank: u32,
        attempt: u64,
        failure: Option<SourceRecoveryReason>,
    ) -> bool {
        if let Some(reason) = failure {
            self.with_current_replay(source, dp_rank, attempt, || {
                self.invalidate_source_locked(source, dp_rank, reason);
            })
            .is_some()
        } else {
            self.resume_boundary(source, dp_rank, RecoveryBoundary::Replay { attempt }, false)
        }
    }

    fn visible_state(state: SourceRecoveryState) -> bool {
        matches!(
            state,
            SourceRecoveryState::Exact
                | SourceRecoveryState::WorkerExact
                | SourceRecoveryState::Observed
        )
    }

    /// Map each evidence boundary to its credit scope under the mutation fence.
    /// Worker clears cannot repair missing pool history; replay retains only the
    /// scope captured by its still-current attempt.
    pub(super) fn resume_boundary(
        &self,
        source: &SourceId,
        dp_rank: u32,
        boundary: RecoveryBoundary,
        strict: bool,
    ) -> bool {
        let _guard = self.source_mutations.lock();
        let record = self
            .source_states
            .get(&(source.clone(), dp_rank))
            .map(|record| *record);
        let previous = record.map(|record| record.status.state);
        use SourceRecoveryState::{Exact, Observed, WorkerExact};
        let (state, reason) = match boundary {
            RecoveryBoundary::Initial => (
                if strict { WorkerExact } else { Exact },
                SourceRecoveryReason::InitialSequence,
            ),
            RecoveryBoundary::Clear => (
                if previous == Some(Exact) {
                    Exact
                } else if strict {
                    WorkerExact
                } else {
                    Observed
                },
                SourceRecoveryReason::ClearBarrier,
            ),
            RecoveryBoundary::Observed => {
                if matches!(previous, Some(Exact | WorkerExact)) {
                    return true;
                }
                (Observed, SourceRecoveryReason::FreshObservation)
            }
            RecoveryBoundary::Replay { attempt } => {
                let Some(replay_state) = record
                    .filter(|record| {
                        record.attempt == attempt
                            && record.status.state == SourceRecoveryState::Degraded
                            && record.status.reason == SourceRecoveryReason::SequenceGap
                    })
                    .and_then(|record| record.replay_state)
                else {
                    return false;
                };
                (replay_state, SourceRecoveryReason::ReplayApplied)
            }
        };
        if previous.is_some_and(|s| !Self::visible_state(s))
            || (previous == Some(SourceRecoveryState::WorkerExact))
                != (state == SourceRecoveryState::WorkerExact)
        {
            self.source_generation.fetch_add(1, Ordering::AcqRel);
        }
        self.transition_source(source, dp_rank, state, reason);
        true
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(super) fn source_recovery_status(
        &self,
        source: &SourceId,
        dp_rank: u32,
    ) -> Option<SourceRecoveryStatus> {
        self.source_states
            .get(&(source.clone(), dp_rank))
            .map(|record| record.status)
    }

    /// Remove lifecycle state after the supervisor has invalidated and retired
    /// the only subscriber allowed to mutate this source/rank.
    pub(crate) fn forget_source(&self, source: &SourceId, dp_rank: u32) {
        let _guard = self.source_mutations.lock();
        // Hold the state entry until residency is cleared and the query fence
        // advances, so readers cannot observe an unregistered stale claim.
        let entry = self.source_states.entry((source.clone(), dp_rank));
        self.pages.retire_source(source, dp_rank);
        self.source_generation.fetch_add(1, Ordering::AcqRel);
        if let Entry::Occupied(entry) = entry {
            let record = entry.remove();
            RouterMetrics::clear_kv_source_recovery_state(
                record.status.state.as_str(),
                record.status.reason.as_str(),
            );
        }
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(super) fn source_generation_for_test(&self) -> u64 {
        self.source_generation.load(Ordering::Acquire)
    }

    pub fn find_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
        let generation = self.source_generation.load(Ordering::Acquire);
        let group = PageQuery::from(query.group_idx);
        let pages = self
            .pages
            .collect_hash_groups(std::iter::once(group), None, &query.hashes);
        let Some(evidence) = pages.get(&group) else {
            return Vec::new();
        };
        let mut hits = HashMap::<(RoutableTarget, StorageTier), TierMatch>::new();
        for owner in evidence.owners() {
            let target = owner.target();
            let state = self
                .source_states
                .get(&(target.instance_id.clone(), target.dp_rank))
                .map(|record| record.status.state);
            if state.is_some_and(|state| !Self::visible_state(state)) {
                continue;
            }
            let mut collect = |tiers: &PageTiers<'_>| {
                for &tier in &query.tiers_of_interest {
                    if tier == StorageTier::External
                        && state == Some(SourceRecoveryState::WorkerExact)
                    {
                        continue;
                    }
                    let mut add = |start, finish| {
                        let hit = hits
                            .entry((target.clone(), tier))
                            .or_insert_with(|| TierMatch {
                                target: target.clone(),
                                tier,
                                locality: owner.locality(),
                                segments: Vec::new(),
                            });
                        if let Some((_, end)) =
                            hit.segments.last_mut().filter(|(_, end)| *end == start)
                        {
                            *end = finish;
                        } else {
                            hit.segments.push((start, finish));
                        }
                    };
                    if tiers.prefix_len > 0 && tiers.prefix.contains(tier) {
                        let Ok(end) = u32::try_from(tiers.prefix_len) else {
                            continue;
                        };
                        add(0, end);
                    }
                    for (offset, bits) in tiers.tail.iter().enumerate() {
                        if bits.contains(tier) {
                            let Ok(end) = u32::try_from(tiers.prefix_len + offset + 1) else {
                                break;
                            };
                            add(end - 1, end);
                        }
                    }
                }
            };
            match super::storage::PageEvidence::unchanged_groups(
                &self.pages,
                &target,
                owner,
                &query.hashes,
                std::iter::once(Some(evidence)),
            ) {
                Some(PageMatch::Hit(runs, version))
                    if self.pages.source_version_is_current(&version) =>
                {
                    collect(&runs[0])
                }
                Some(_) => (),
                None => {
                    self.pages.with_source_snapshot(&target, |snapshot| {
                        let runs = snapshot.hash_page_runs(
                            owner,
                            None,
                            &query.hashes,
                            &pages,
                            std::iter::once(group),
                        );
                        collect(&runs[0]);
                    });
                }
            }
        }
        let matches = hits
            .into_values()
            .map(|mut hit| {
                hit.segments.sort_unstable();
                let mut count = 0;
                for read in 0..hit.segments.len() {
                    let (start, end) = hit.segments[read];
                    if count > 0 && start <= hit.segments[count - 1].1 {
                        hit.segments[count - 1].1 = hit.segments[count - 1].1.max(end);
                    } else {
                        hit.segments[count] = (start, end);
                        count += 1;
                    }
                }
                hit.segments.truncate(count);
                hit
            })
            .collect();
        if self.source_generation.load(Ordering::Acquire) != generation {
            return Vec::new();
        }
        matches
    }
}

impl Default for KvBlockIndexer {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::policies::TieredMatchProvider for KvBlockIndexer {
    fn find_tiered_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
        self.find_matches(query)
    }
}
