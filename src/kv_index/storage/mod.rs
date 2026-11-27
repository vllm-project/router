//! Shared hash-page storage and source transactions, for full and mixed attention.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use parking_lot::RwLock;

use crate::kv_index::request_hash::SequenceHash;
mod page_table;
mod records;
pub(crate) mod source_transaction;

use self::page_table::{ActivePages, PageTable};
pub(crate) use self::page_table::{PageEvidence, PageMatch, PageTiers};
use self::source_transaction::SourceWrite;
pub(crate) use self::source_transaction::{IndexBatch, IndexSnapshot};
use self::source_transaction::{StreamControl, StreamPhase, StreamState};
use crate::kv_index::types::{ResidencyOwner, RoutableTarget, SourceId, StorageTier, TierBits};

type Fast = foldhash::fast::RandomState;
pub(crate) type GroupPages = HashMap<PageQuery, PageEvidence, Fast>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PageQuery {
    pub(crate) group: u32,
    pub(crate) stop_at_gap: bool,
}

impl From<u32> for PageQuery {
    fn from(group: u32) -> Self {
        Self {
            group,
            stop_at_gap: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PageWriteError {
    Capacity,
    InvalidResidencyMode,
}

/// Serializes one worker/rank across its cache groups.
struct SourceLock {
    target: RoutableTarget,
    access: RwLock<SourceData>,
    // Even generations are visible. Odd generations are suspended or retired.
    generation: AtomicU64,
    revision: AtomicU64,
    planned: RwLock<Option<(u64, PageChanges)>>,
}

#[derive(Default)]
struct SourceData {
    active_pages: ActivePages,
    streams: HashMap<ResidencyOwner, StreamState>,
}

/// Identity and trust generation of one source at snapshot time.
pub(crate) struct SourceVersion {
    source: Arc<SourceLock>,
    generation: u64,
}

pub(crate) struct SourceReadGuard<'a> {
    _read: parking_lot::RwLockReadGuard<'a, SourceData>,
}

impl SourceVersion {
    pub(crate) fn target(&self) -> &RoutableTarget {
        &self.source.target
    }

    /// Ordinary store/remove keeps the version current. Suspension, resume,
    /// and retirement invalidate it.
    pub(crate) fn is_current(&self) -> bool {
        self.generation.is_multiple_of(2)
            && self.source.generation.load(Ordering::Acquire) == self.generation
    }

    pub(crate) fn try_read(&self) -> Option<SourceReadGuard<'_>> {
        Some(SourceReadGuard {
            // Batched prompts can pin the same source while already holding its read guard.
            _read: self.source.access.try_read_recursive()?,
        })
    }
}

impl SourceLock {
    fn new(target: RoutableTarget, committed: u64) -> Self {
        Self {
            target,
            access: RwLock::new(SourceData::default()),
            generation: AtomicU64::new(0),
            revision: AtomicU64::new(committed),
            planned: RwLock::default(),
        }
    }

    fn committed_at(&self, scan: &PageScan) -> bool {
        let revision = self.revision.load(Ordering::Acquire);
        revision.is_multiple_of(2) && revision <= scan.committed
    }
}

/// One shared fact index per cache identity, sharded by cache group.
pub struct KvBlockIndexer {
    hash_pages: PageTable,
    /// Gates may be removed only by `retire_source` under the global barrier.
    // Membership changes only on source attachment/retirement. A compact map
    // avoids scanning CPU-count-sized sets of empty shards on every query.
    source_locks: RwLock<HashMap<RoutableTarget, Arc<SourceLock>, Fast>>,
    // A scan samples this once before reading any page. Only completed writes advance it.
    committed: AtomicU64,
    /// Source mutations share the barrier; retirement excludes them.
    domain: RwLock<()>,
    /// Batches that can hide a previously committed page hold a shared guard until publish.
    removals: RwLock<()>,
}

impl KvBlockIndexer {
    pub fn new() -> Self {
        KvBlockIndexer {
            hash_pages: PageTable::default(),
            source_locks: RwLock::new(HashMap::default()),
            committed: AtomicU64::new(0),
            domain: RwLock::new(()),
            removals: RwLock::new(()),
        }
    }

    fn existing_source(&self, target: &RoutableTarget) -> Option<Arc<SourceLock>> {
        self.source_locks.read().get(target).cloned()
    }

    fn source_lock(&self, target: &RoutableTarget) -> Arc<SourceLock> {
        if let Some(source) = self.existing_source(target) {
            return source;
        }
        self.source_locks
            .write()
            .entry(target.clone())
            .or_insert_with(|| {
                // A replacement must not validate observations taken before the old source retired.
                Arc::new(SourceLock::new(
                    target.clone(),
                    self.committed.load(Ordering::Acquire),
                ))
            })
            .clone()
    }

    /// Hide a source while contiguous replay repairs its observations.
    fn suspend_source_unlocked(&self, source: &SourceId, dp_rank: u32) {
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        self.source_lock(&target)
            .generation
            .fetch_or(1, Ordering::AcqRel);
    }

    /// Permanently retire a fenced source. The global barrier drains all
    /// transactions before removing its gate; in-flight owned snapshots retain
    /// the old gate and observe its incremented generation.
    pub(crate) fn retire_source(&self, source: &SourceId, dp_rank: u32) {
        // Share an epoch across the page removals. Enter it before locking the
        // domain so reclamation of earlier batches cannot hold up lookups.
        let _epoch = scc::Guard::new();
        let _domain = self.domain.write();
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        let gate = self.source_lock(&target);
        let mut data = gate.access.write();
        let mut write = SourceWrite::new(&gate.revision, self);
        self.invalidate_source_unlocked(source, dp_rank, &mut data.active_pages, &mut write);
        drop(data);
        self.source_locks.write().remove(&target);
    }

    fn invalidate_source_unlocked(
        &self,
        source: &SourceId,
        dp_rank: u32,
        active_pages: &mut ActivePages,
        write: &mut SourceWrite<'_>,
    ) {
        write.mark_removal();
        self.suspend_source_unlocked(source, dp_rank);
        let owners: Vec<_> = active_pages.keys().cloned().collect();
        for owner in owners {
            self.hash_pages.clear_owner(active_pages, &owner);
        }
    }

    /// Enable fresh observations after successful replay.
    fn resume_source_unlocked(&self, source: &SourceId, dp_rank: u32) {
        let target = RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
        };
        let gate = self.source_lock(&target);
        let generation = gate.generation.load(Ordering::Relaxed);
        if !generation.is_multiple_of(2) {
            gate.generation
                .store(generation.wrapping_add(1), Ordering::Release);
        }
    }

    /// Apply a stream batch and its cache mutations under the same source lock.
    pub(crate) fn with_stream_batch<R>(
        &self,
        target: &RoutableTarget,
        owner: &ResidencyOwner,
        control: &Arc<StreamControl>,
        f: impl FnOnce(&mut StreamState, &mut IndexBatch<'_>) -> R,
    ) -> R {
        self.stream_batch(target, owner, control, true, None, f)
            .unwrap()
    }

    /// Cleanup must not recreate a gate or owner that discovery already retired.
    pub(crate) fn with_existing_stream_batch<R>(
        &self,
        target: &RoutableTarget,
        owner: &ResidencyOwner,
        control: &Arc<StreamControl>,
        f: impl FnOnce(&mut StreamState, &mut IndexBatch<'_>) -> R,
    ) -> Option<R> {
        self.stream_batch(target, owner, control, false, None, f)
    }

    pub(crate) fn with_page_mutation_batch<R>(
        &self,
        target: &RoutableTarget,
        owner: &ResidencyOwner,
        control: &Arc<StreamControl>,
        planned: Option<PageChanges>,
        f: impl FnOnce(&mut StreamState, &mut IndexBatch<'_>) -> R,
    ) -> R {
        self.stream_batch(target, owner, control, true, planned, f)
            .unwrap()
    }

    fn stream_batch<R>(
        &self,
        target: &RoutableTarget,
        owner: &ResidencyOwner,
        control: &Arc<StreamControl>,
        create: bool,
        planned: Option<PageChanges>,
        f: impl FnOnce(&mut StreamState, &mut IndexBatch<'_>) -> R,
    ) -> Option<R> {
        assert_eq!(owner.target(), *target, "stream crossed targets");
        let _domain = self.domain.read();
        let source = if create {
            self.source_lock(target)
        } else {
            self.existing_source(target)?
        };
        let mut data = source.access.write();
        let write = SourceWrite::new(&source.revision, self);
        let revision = source.revision.load(Ordering::Relaxed);
        *source.planned.write() = planned.map(|changes| (revision, changes));
        let SourceData {
            active_pages,
            streams,
        } = &mut *data;
        let state = if create {
            streams
                .entry(owner.clone())
                .or_insert_with(|| StreamState::new(control.clone()))
        } else {
            streams.get_mut(owner)?
        };
        assert!(
            state.has_control(control),
            "stream owner reused with another control"
        );
        let mut tx = IndexBatch {
            index: self,
            target,
            active_pages,
            source: &source,
            write,
        };
        let result = f(state, &mut tx);
        if state.lease_id == 0 && state.next_seq.is_none() && !state.visible() {
            streams.remove(owner);
        }
        Some(result)
    }

    pub(crate) fn stream_lease_is_active(&self, owner: &ResidencyOwner, lease: u64) -> bool {
        let _domain = self.domain.read();
        let Some(source) = self.existing_source(&owner.target()) else {
            return false;
        };
        let data = source.access.read();
        data.streams
            .get(owner)
            .is_some_and(|state| state.lease_id == lease && state.phase != StreamPhase::Retired)
    }

    /// Read all groups from one attached source without creating a source gate.
    pub(crate) fn with_source_snapshot<R>(
        &self,
        target: &RoutableTarget,
        f: impl FnOnce(&IndexSnapshot<'_>) -> R,
    ) -> Option<R> {
        // Pin this gate across replacement. Retirement clears its pages and
        // advances its generation while holding its write lock.
        let source = self.existing_source(target)?;
        let data = source.access.read();
        Some(f(&IndexSnapshot {
            index: self,
            source: &source,
            source_data: &data,
        }))
    }

    pub(crate) fn has_multiple_sources(&self) -> bool {
        self.source_locks.read().len() > 1
    }

    fn confirms_miss_at(&self, committed: u64) -> bool {
        // An unpublished removal could hide a page. Completed mutations advance the cutoff.
        self.removals
            .try_write()
            .is_some_and(|_removals| self.committed.load(Ordering::Acquire) == committed)
    }

    pub(crate) fn collect_hash_groups(
        &self,
        groups: impl Iterator<Item = impl Into<PageQuery>>,
        namespace: Option<&str>,
        keys: &[SequenceHash],
    ) -> GroupPages {
        // Each source validates this scan against its transaction revision.
        let scan = Arc::new(PageScan::new(self.committed.load(Ordering::Acquire)));
        groups
            .map(|query| {
                let query = query.into();
                let (prefix_len, head, holders) =
                    self.hash_pages.collect_split(query, namespace, keys, self);
                (
                    query,
                    PageEvidence {
                        scan: Arc::clone(&scan),
                        prefix_len,
                        prefix: head,
                        holders,
                    },
                )
            })
            .collect()
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn collect_hash_pages(&self, group: u32, keys: &[SequenceHash]) -> PageEvidence {
        self.collect_hash_groups(std::iter::once(group), None, keys)
            .remove(&PageQuery::from(group))
            .unwrap()
    }
}

/// A tag collision only requires an exact source snapshot. Page equality still uses the full hash.
fn page_tag(key: &SequenceHash) -> u32 {
    u32::from_le_bytes(key[..4].try_into().unwrap())
}

struct PageScan {
    committed: u64,
    request_tags: OnceLock<HashSet<u32, Fast>>,
    confirmed: OnceLock<bool>,
}

impl PageScan {
    fn new(committed: u64) -> Self {
        Self {
            committed,
            request_tags: OnceLock::new(),
            confirmed: OnceLock::new(),
        }
    }

    fn confirms_miss(&self, index: &KvBlockIndexer) -> bool {
        let confirmed = self
            .confirmed
            .get_or_init(|| index.confirms_miss_at(self.committed));
        *confirmed && index.committed.load(Ordering::Acquire) == self.committed
    }
}

/// At most one event batch is retained per source. Readers copy only a tag or an Arc
/// before releasing the publication lock, regardless of batch and prompt length.
#[derive(Clone)]
pub(crate) enum PageChanges {
    Inline(Option<u32>),
    Batch(Arc<[u32]>),
}

impl PageChanges {
    pub(crate) fn from_batches<'a>(
        batches: impl Iterator<Item = Option<&'a [SequenceHash]>>,
        max_entries: usize,
    ) -> Option<Self> {
        let mut tags = smallvec::SmallVec::<[u32; 1]>::new();
        for batch in batches {
            let Some(hashes) = batch else {
                // A clear can affect any page, so readers must validate under the source lock.
                return None;
            };
            if hashes.len() > max_entries - tags.len() {
                return None;
            }
            tags.extend(hashes.iter().map(page_tag));
        }
        Some(if tags.len() <= 1 {
            Self::Inline(tags.pop())
        } else {
            Self::Batch(tags.into_vec().into())
        })
    }

    fn as_slice(&self) -> &[u32] {
        match self {
            Self::Inline(tag) => tag.as_slice(),
            Self::Batch(tags) => tags,
        }
    }
}

impl Default for KvBlockIndexer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests;
