//! Hash-addressed page facts; reverse memberships live in the source transaction.

use std::borrow::{Borrow, Cow};
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};

use hashbrown::hash_map::EntryRef;
use parking_lot::RwLock;

use super::{
    page_tag, records::Residency, Fast, KvBlockIndexer, PageQuery, PageWriteError, SourceVersion,
};
use crate::kv_index::request_hash::SequenceHash;
use crate::kv_index::types::{ResidencyOwner, RoutableTarget, StorageTier, TierBits};

#[derive(Clone)]
struct PageKey {
    group: u32,
    fingerprint: u32,
    digest: SequenceHash,
}

impl PartialEq for PageKey {
    fn eq(&self, other: &Self) -> bool {
        self.group == other.group && self.digest == other.digest
    }
}

impl Eq for PageKey {}

impl PageKey {
    fn new(group: u32, digest: SequenceHash) -> Self {
        static HASHER: LazyLock<Fast> = LazyLock::new(Fast::default);
        // Hash the fixed-width SHA-256 as two words. Both halves and the group
        // participate in bucket selection; equality still checks the whole key.
        let low = u128::from_le_bytes(digest[..16].try_into().unwrap());
        let high = u128::from_le_bytes(digest[16..].try_into().unwrap());
        let fingerprint = HASHER.hash_one((low, high, group)) as u32;
        Self {
            group,
            fingerprint,
            digest,
        }
    }
}

impl Hash for PageKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(self.fingerprint);
    }
}
#[derive(Clone)]
struct PageRef(Arc<PageRecord>);

struct PageRecord {
    key: PageKey,
    row: RwLock<PageRow>,
}

impl PageRef {
    fn new(key: PageKey) -> Self {
        Self(Arc::new(PageRecord {
            key,
            row: RwLock::default(),
        }))
    }
}

impl Borrow<PageKey> for PageRef {
    fn borrow(&self) -> &PageKey {
        &self.0.key
    }
}

impl PartialEq for PageRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.key == other.0.key
    }
}

impl Eq for PageRef {}

impl Hash for PageRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.key.hash(state);
    }
}

type HolderMap = HashMap<u64, PageHolder, Fast>;
type Claim = (Arc<PageOwner>, TierBits);
type PrefixClaims = HashMap<ResidencyOwner, TierBits, Fast>;
type OwnerClaims = HashMap<ResidencyOwner, Vec<TierBits>, Fast>;
type OwnerRuns = HashMap<u64, (Arc<PageOwner>, Vec<TierBits>), Fast>;
type Pages = hashbrown::HashMap<PageRef, (), Fast>;

// A page holds one pointer to a namespace shared across the source's pages.
#[derive(Clone, Eq, Hash, PartialEq)]
struct Namespace(Arc<String>);

impl Borrow<str> for Namespace {
    fn borrow(&self) -> &str {
        self.0.as_str()
    }
}

impl From<&str> for Namespace {
    fn from(name: &str) -> Self {
        Self(Arc::new(name.to_owned()))
    }
}

// Store borrows the namespace and owner keys before allocating them on a miss.
type Namespaces = hashbrown::HashMap<Namespace, Pages, Fast>;
pub(super) type ActivePages = hashbrown::HashMap<Arc<ResidencyOwner>, OwnerPages, Fast>;

struct PageOwner {
    id: u64,
    owner: ResidencyOwner,
}

struct PageHolder {
    owner: Arc<PageOwner>,
    residency: Residency,
}

#[derive(Default)]
enum Holders {
    #[default]
    Empty,
    One(PageHolder),
    // Keep the map header out of single-owner page rows.
    Shared(Box<HolderMap>),
}

impl Holders {
    fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::One(_) => 1,
            Self::Shared(holders) => holders.len(),
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    fn get(&self, id: u64) -> Option<&PageHolder> {
        match self {
            Self::One(holder) if holder.owner.id == id => Some(holder),
            Self::Shared(holders) => holders.get(&id),
            _ => None,
        }
    }

    fn get_mut(&mut self, id: u64) -> Option<&mut PageHolder> {
        match self {
            Self::One(holder) if holder.owner.id == id => Some(holder),
            Self::Shared(holders) => holders.get_mut(&id),
            _ => None,
        }
    }

    fn insert(&mut self, holder: PageHolder) {
        match self {
            Self::Empty => *self = Self::One(holder),
            Self::Shared(holders) => {
                holders.insert(holder.owner.id, holder);
            }
            Self::One(_) => {
                let Self::One(first) = std::mem::take(self) else {
                    unreachable!()
                };
                let mut holders = HolderMap::default();
                holders.insert(first.owner.id, first);
                holders.insert(holder.owner.id, holder);
                *self = Self::Shared(Box::new(holders));
            }
        }
    }

    fn remove(&mut self, id: u64) {
        match self {
            Self::One(holder) if holder.owner.id == id => *self = Self::Empty,
            Self::Shared(holders) => {
                holders.remove(&id);
                if holders.len() == 1 {
                    let only = std::mem::take(holders.as_mut())
                        .into_values()
                        .next()
                        .unwrap();
                    *self = Self::One(only);
                }
            }
            _ => {}
        }
    }

    fn values(&self) -> impl Iterator<Item = &PageHolder> {
        let (one, shared) = match self {
            Self::Empty => (None, None),
            Self::One(holder) => (Some(holder), None),
            Self::Shared(holders) => (None, Some(holders)),
        };
        one.into_iter()
            .chain(shared.into_iter().flat_map(|map| map.values()))
    }
}

pub(super) struct OwnerPages {
    owner: Arc<PageOwner>,
    base: Pages,
    named: Namespaces,
    page_count: usize,
    // Count non-device tier claims, including named pages and mixed-tier pages.
    non_device_tiers: usize,
}

impl OwnerPages {
    fn pages(&self, namespace: Option<&str>) -> Option<&Pages> {
        match namespace {
            Some(name) => self.named.get(name),
            None => Some(&self.base),
        }
    }

    pub(super) fn contains_any(
        &self,
        group: u32,
        namespace: Option<&str>,
        keys: &[SequenceHash],
    ) -> bool {
        self.pages(namespace).is_some_and(|pages| {
            keys.iter()
                .rev()
                .any(|&hash| pages.contains_key(&PageKey::new(group, hash)))
        })
    }

    pub(super) fn device_only(&self) -> bool {
        self.non_device_tiers == 0
    }
}

#[derive(Default)]
struct ScopedHolders {
    namespace: Option<Namespace>,
    holders: Holders,
}

enum PageRow {
    Single(ScopedHolders),
    Shared(Vec<ScopedHolders>),
}

impl Default for PageRow {
    fn default() -> Self {
        Self::Single(ScopedHolders::default())
    }
}

impl PageRow {
    fn scopes(&self) -> &[ScopedHolders] {
        match self {
            Self::Single(scope) => std::slice::from_ref(scope),
            Self::Shared(scopes) => scopes,
        }
    }

    fn scopes_mut(&mut self) -> &mut [ScopedHolders] {
        match self {
            Self::Single(scope) => std::slice::from_mut(scope),
            Self::Shared(scopes) => scopes,
        }
    }

    fn holders(&self, namespace: Option<&str>) -> Option<&Holders> {
        self.scopes()
            .iter()
            .find(|scope| scope.namespace.as_ref().map(|name| name.0.as_str()) == namespace)
            .map(|scope| &scope.holders)
    }

    fn holders_mut(&mut self, namespace: Option<&Namespace>) -> &mut Holders {
        let index = self
            .scopes()
            .iter()
            .position(|scope| scope.namespace.as_ref() == namespace)
            .unwrap_or_else(|| {
                let scope = ScopedHolders {
                    namespace: namespace.cloned(),
                    holders: Holders::default(),
                };
                match self {
                    Self::Single(first) if first.holders.is_empty() => {
                        *first = scope;
                        0
                    }
                    Self::Single(first) => {
                        *self = Self::Shared(vec![std::mem::take(first), scope]);
                        1
                    }
                    Self::Shared(scopes) => {
                        scopes.push(scope);
                        scopes.len() - 1
                    }
                }
            });
        &mut self.scopes_mut()[index].holders
    }

    fn prune(&mut self) -> bool {
        match self {
            Self::Single(scope) => scope.holders.is_empty(),
            Self::Shared(scopes) => {
                scopes.retain(|scope| !scope.holders.is_empty());
                if scopes.len() == 1 {
                    *self = Self::Single(scopes.pop().unwrap());
                    return false;
                }
                scopes.is_empty()
            }
        }
    }
}

#[derive(Default)]
pub(super) struct PageTable {
    rows: scc::HashIndex<PageRef, (), Fast>,
    next_owner: AtomicU64,
}

/// An optimistic page scan and the shared commit sequence sampled before it started.
pub(crate) struct PageEvidence {
    pub(super) scan: Arc<super::PageScan>,
    pub(super) prefix_len: usize,
    pub(super) prefix: PrefixClaims,
    pub(super) holders: OwnerClaims,
}

pub(crate) struct PageTiers<'a> {
    pub(crate) prefix_len: usize,
    pub(crate) prefix: TierBits,
    pub(crate) tail: Cow<'a, [TierBits]>,
}

pub(crate) enum PageMatch<'a> {
    Miss,
    Hit(smallvec::SmallVec<[PageTiers<'a>; 1]>, SourceVersion),
}

impl<'a> PageTiers<'a> {
    pub(crate) fn get(&self, index: usize) -> Option<TierBits> {
        let tiers = if index < self.prefix_len {
            self.prefix
        } else {
            *self.tail.get(index - self.prefix_len)?
        };
        (tiers.0 != 0).then_some(tiers)
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(crate) fn expanded(self) -> Cow<'a, [TierBits]> {
        if self.prefix_len == 0 {
            return self.tail;
        }
        let mut pages = Vec::with_capacity(self.prefix_len + self.tail.len());
        pages.resize(self.prefix_len, self.prefix);
        pages.extend_from_slice(&self.tail);
        Cow::Owned(pages)
    }
}

impl KvBlockIndexer {
    /// A plain prefix with no first-page holders can finish from one observation.
    pub(crate) fn collect_prefix(
        &self,
        group: PageQuery,
        keys: &[SequenceHash],
    ) -> Option<PageEvidence> {
        debug_assert!(group.stop_at_gap);
        let (&first, rest) = keys.split_first()?;
        let committed = self.committed.load(Ordering::Acquire);
        let row = self
            .hash_pages
            .rows
            .peek_with(&PageKey::new(group.group, first), |page, ()| page.clone());
        if row.is_none() && self.confirms_miss_at(committed) {
            return None;
        }

        // The row handle stays valid through retirement. Each source validates
        // its evidence against the cutoff taken before looking up the head.
        let head = row
            .map(|page| PageTable::row_claims(&page.0.row.read(), None))
            .unwrap_or_default();
        let (prefix_len, prefix, holders) =
            self.hash_pages.collect_rest(group, None, rest, self, head);
        Some(PageEvidence {
            scan: Arc::new(super::PageScan::new(committed)),
            prefix_len,
            prefix,
            holders,
        })
    }
}

impl PageEvidence {
    /// A scan stays coherent through one transaction if it cannot touch a requested page.
    pub(super) fn validate_source(
        &self,
        index: &KvBlockIndexer,
        target: &RoutableTarget,
        keys: &[SequenceHash],
    ) -> Option<SourceVersion> {
        let source = index.existing_source(target)?;
        let generation = source.generation.load(Ordering::Acquire);
        if !generation.is_multiple_of(2) {
            return Some(SourceVersion { source, generation });
        }
        let now = source.revision.load(Ordering::Acquire);
        if now > self.scan.committed || !now.is_multiple_of(2) {
            let changes = {
                let published = source.planned.read();
                let (writing, changes) = published.as_ref()?;
                let same_batch = if now.is_multiple_of(2) {
                    now > *writing
                } else {
                    now == *writing
                };
                // The prior commit must predate the scan. Validate the plan against the
                // same revision because writers mark the source busy before publishing it.
                if self.scan.committed < writing - 1
                    || !same_batch
                    || source.revision.load(Ordering::Acquire) != now
                {
                    return None;
                }
                changes.clone()
            };
            let overlaps = if let [key] = keys {
                changes.as_slice().contains(&page_tag(key))
            } else {
                let tags = self
                    .scan
                    .request_tags
                    .get_or_init(|| keys.iter().map(page_tag).collect());
                changes.as_slice().iter().any(|tag| tags.contains(tag))
            };
            if overlaps {
                return None;
            }
        }
        Some(SourceVersion { source, generation })
    }

    pub(crate) fn confirms_empty(&self, index: &KvBlockIndexer) -> bool {
        self.prefix.is_empty() && self.holders.is_empty() && self.scan.confirms_miss(index)
    }

    pub(crate) fn unchanged_groups<'a>(
        index: &KvBlockIndexer,
        target: &RoutableTarget,
        owner: &ResidencyOwner,
        keys: &[SequenceHash],
        mut groups: impl Iterator<Item = Option<&'a Self>>,
    ) -> Option<PageMatch<'a>> {
        let first = groups.next()??;
        if first.prefix.is_empty() && first.holders.is_empty() {
            if first.scan.confirms_miss(index) {
                return Some(PageMatch::Miss);
            }
            // An empty run of length one ends at the first page. Later pages
            // cannot change a prefix miss, but can affect a full scan.
            let keys = if first.prefix_len == 1 {
                &keys[..1]
            } else {
                keys
            };
            return first
                .validate_source(index, target, keys)
                .map(|_| PageMatch::Miss);
        }
        let head = first.runs_for_owner(owner);
        let missing = head.prefix_len == 0 && head.tail.is_empty();
        // A required group has no pages for this owner. A confirmed domain
        // scan establishes this for every non-holder without another source lookup.
        if missing && first.scan.confirms_miss(index) {
            return Some(PageMatch::Miss);
        }
        let version = first.validate_source(index, target, keys)?;
        // A hidden rank cannot earn credit. Skip the exact fallback while
        // retirement is still removing its entries.
        if missing || !version.generation.is_multiple_of(2) {
            return Some(PageMatch::Miss);
        }
        let mut runs = smallvec::SmallVec::with_capacity(groups.size_hint().0 + 1);
        runs.push(head);
        for group in groups {
            let group = group?;
            if !Arc::ptr_eq(&first.scan, &group.scan) {
                return None;
            }
            runs.push(group.runs_for_owner(owner));
        }
        Some(PageMatch::Hit(runs, version))
    }

    pub(super) fn runs_for_owner(&self, owner: &ResidencyOwner) -> PageTiers<'_> {
        let prefix = self.prefix.get(owner).copied().unwrap_or_default();
        let tail = self.holders.get(owner).map_or(&[][..], Vec::as_slice);
        PageTiers {
            prefix_len: if prefix.0 != 0 || !tail.is_empty() {
                self.prefix_len
            } else {
                0
            },
            prefix,
            tail: Cow::Borrowed(tail),
        }
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(super) fn for_owner(&self, owner: &ResidencyOwner) -> Cow<'_, [TierBits]> {
        self.runs_for_owner(owner).expanded()
    }
}

fn observe(
    holders: &mut Holders,
    owner: &Arc<PageOwner>,
    non_device_tiers: &mut usize,
    tier: StorageTier,
    incremental: bool,
) -> Result<(), PageWriteError> {
    if let Some(holder) = holders.get_mut(owner.id) {
        let fact = &mut holder.residency;
        let new_non_device = tier != StorageTier::Device && !fact.tiers.contains(tier);
        if !fact.observe(tier, incremental) {
            return Err(PageWriteError::Capacity);
        }
        if new_non_device {
            *non_device_tiers += 1;
        }
    } else {
        let mut residency = Residency::default();
        if !residency.observe(tier, incremental) {
            return Err(PageWriteError::Capacity);
        }
        if tier != StorageTier::Device {
            *non_device_tiers += 1;
        }
        holders.insert(PageHolder {
            owner: owner.clone(),
            residency,
        });
    }
    Ok(())
}

fn withdraw(
    holders: &mut Holders,
    owner_id: u64,
    non_device_tiers: &mut usize,
    tier: StorageTier,
    incremental: bool,
) -> bool {
    let Some(holder) = holders.get_mut(owner_id) else {
        return false;
    };
    let fact = &mut holder.residency;
    let removed = fact.withdraw(tier, incremental);
    if tier != StorageTier::Device && removed {
        *non_device_tiers -= 1;
    }
    if fact.tiers.0 != 0 {
        return false;
    }
    holders.remove(owner_id);
    true
}

impl PageTable {
    fn read_row<R>(&self, key: &PageKey, read: impl FnOnce(&PageRow) -> R) -> Option<R> {
        self.rows
            .peek_with(key, |page, ()| read(&page.0.row.read()))
    }

    fn remove_row_if(&self, key: &PageKey, withdraw: impl FnOnce(&mut PageRow) -> bool) {
        if let Some(entry) = self.rows.get_sync(key) {
            let empty = withdraw(&mut entry.key().0.row.write());
            if empty {
                entry.remove_entry();
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn store(
        &self,
        active: &mut ActivePages,
        owner: &ResidencyOwner,
        group: u32,
        tier: StorageTier,
        namespace: Option<&str>,
        hashes: &[SequenceHash],
        incremental: bool,
        max_entries: usize,
    ) -> Result<(), PageWriteError> {
        if incremental != (tier == StorageTier::Device) {
            return Err(PageWriteError::InvalidResidencyMode);
        }
        if hashes.is_empty() {
            return Ok(());
        }
        // The source gate keeps memberships stable until the whole store is applied.
        let check_capacity = |count: usize, keys: Option<&Pages>| {
            let remaining = max_entries.saturating_sub(count);
            if hashes.len() > remaining {
                let mut new = HashSet::<_, Fast>::default();
                for hash in hashes {
                    if keys.is_some_and(|keys| keys.contains_key(&PageKey::new(group, *hash))) {
                        continue;
                    }
                    if remaining == 0 || (new.insert(hash) && new.len() > remaining) {
                        return Err(PageWriteError::Capacity);
                    }
                }
            }
            Ok(())
        };
        let pages = match active.entry_ref(owner) {
            EntryRef::Occupied(entry) => {
                check_capacity(entry.get().page_count, entry.get().pages(namespace))?;
                entry.into_mut()
            }
            EntryRef::Vacant(entry) => {
                check_capacity(0, None)?;
                entry.insert_with_key(
                    Arc::new(owner.clone()),
                    OwnerPages {
                        owner: Arc::new(PageOwner {
                            // Only ID exhaustion rejects the owner allocation.
                            id: self
                                .next_owner
                                .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| {
                                    id.checked_add(1)
                                })
                                .map_err(|_| PageWriteError::Capacity)?,
                            owner: owner.clone(),
                        }),
                        base: Pages::default(),
                        named: Namespaces::default(),
                        page_count: 0,
                        non_device_tiers: 0,
                    },
                )
            }
        };
        let (namespace, keys) = match namespace {
            None => (None, &mut pages.base),
            Some(name) => {
                let entry = match pages.named.entry_ref(name) {
                    EntryRef::Occupied(entry) => entry,
                    EntryRef::Vacant(entry) => entry.insert_entry(Pages::default()),
                };
                (Some(entry.key().clone()), entry.into_mut())
            }
        };
        if hashes.len() > keys.len() {
            keys.reserve(
                hashes
                    .len()
                    .min(max_entries.saturating_sub(pages.page_count)),
            );
        }
        let mut observe_page = |page: &PageRef| {
            let mut row = page.0.row.write();
            observe(
                row.holders_mut(namespace.as_ref()),
                &pages.owner,
                &mut pages.non_device_tiers,
                tier,
                incremental,
            )
        };
        for &hash in hashes {
            let page = PageKey::new(group, hash);
            match keys.entry_ref(&page) {
                EntryRef::Occupied(member) => {
                    // This source keeps the row attached while its write gate is held.
                    observe_page(member.key())?;
                }
                EntryRef::Vacant(member) => {
                    // Hold the directory entry until both the row and its membership exist.
                    let entry = self.rows.get_sync(&page).unwrap_or_else(|| {
                        self.rows
                            .entry_sync(PageRef::new(page.clone()))
                            .or_default()
                    });
                    observe_page(entry.key())?;
                    member.insert_with_key(entry.key().clone(), ());
                    pages.page_count += 1;
                }
            }
        }
        Ok(())
    }

    pub(super) fn remove(
        &self,
        active: &mut ActivePages,
        owner: &ResidencyOwner,
        group: u32,
        tier: StorageTier,
        hashes: &[SequenceHash],
        incremental: bool,
    ) -> Result<(), PageWriteError> {
        if incremental != (tier == StorageTier::Device) {
            return Err(PageWriteError::InvalidResidencyMode);
        }
        let Some(pages) = active.get_mut(owner) else {
            return Ok(());
        };
        let id = pages.owner.id;
        for &hash in hashes {
            // Remove carries the raw hash but no adapter name. Only namespaces
            // sharing this exact page need to be visited.
            let page = PageKey::new(group, hash);
            self.remove_row_if(&page, |row| {
                for scope in row.scopes_mut() {
                    if !withdraw(
                        &mut scope.holders,
                        id,
                        &mut pages.non_device_tiers,
                        tier,
                        incremental,
                    ) {
                        continue;
                    }
                    pages.page_count -= 1;
                    if let Some(name) = &scope.namespace {
                        if let EntryRef::Occupied(mut entry) = pages.named.entry_ref(name) {
                            entry.get_mut().remove(&page);
                            if entry.get().is_empty() {
                                entry.remove();
                            }
                        }
                    } else {
                        pages.base.remove(&page);
                    }
                }
                row.prune()
            });
        }
        if pages.base.is_empty() && pages.named.is_empty() {
            active.remove(owner);
        }
        Ok(())
    }

    pub(super) fn clear_owner(&self, active: &mut ActivePages, owner: &ResidencyOwner) {
        let Some(pages) = active.remove(owner) else {
            return;
        };
        let id = pages.owner.id;
        for page in pages
            .base
            .into_keys()
            .chain(pages.named.into_values().flat_map(Pages::into_keys))
        {
            self.remove_row_if(page.borrow(), |row| {
                for scope in row.scopes_mut() {
                    scope.holders.remove(id);
                }
                row.prune()
            });
        }
    }

    fn read_hint<R>(
        &self,
        hint: Option<&PageRef>,
        group: u32,
        key: SequenceHash,
        read: impl FnOnce(&PageRow) -> R,
    ) -> Option<R> {
        match hint {
            Some(page) => Some(read(&page.0.row.read())),
            None => self.read_row(&PageKey::new(group, key), read),
        }
    }

    fn claims_at(&self, group: u32, namespace: Option<&str>, key: SequenceHash) -> Vec<Claim> {
        self.read_row(&PageKey::new(group, key), |row| {
            Self::row_claims(row, namespace)
        })
        .unwrap_or_default()
    }

    fn row_claims(row: &PageRow, namespace: Option<&str>) -> Vec<Claim> {
        row.holders(namespace)
            .map(|holders| {
                holders
                    .values()
                    .map(|holder| (holder.owner.clone(), holder.residency.tiers))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub(super) fn collect_split(
        &self,
        query: PageQuery,
        namespace: Option<&str>,
        keys: &[SequenceHash],
        index: &KvBlockIndexer,
    ) -> (usize, PrefixClaims, OwnerClaims) {
        let Some((&first, rest)) = keys.split_first() else {
            return Default::default();
        };
        let first = self.claims_at(query.group, namespace, first);
        self.collect_rest(query, namespace, rest, index, first)
    }

    /// Compress the initial shared run and pass the first different row on to the tail.
    fn collect_rest(
        &self,
        query: PageQuery,
        namespace: Option<&str>,
        rest: &[SequenceHash],
        index: &KvBlockIndexer,
        mut run: Vec<Claim>,
    ) -> (usize, PrefixClaims, OwnerClaims) {
        let group = query.group;
        if rest.is_empty() || (query.stop_at_gap && run.is_empty()) {
            return (1, Self::prefix_claims(run), OwnerClaims::default());
        }
        let mut hints = Vec::new();
        if let Some((owner, _)) = run.first() {
            Self::source_hints(index, group, namespace, owner, rest, &mut hints);
        }
        run.sort_unstable_by_key(|(owner, _)| owner.id);
        let compact = run
            .first()
            .zip(run.last())
            .is_some_and(|(first, last)| last.0.id - first.0.id == (run.len() - 1) as u64);
        if !compact {
            let first = run
                .into_iter()
                .map(|(owner, tier)| (owner.id, (owner, vec![tier])))
                .collect();
            let tail = self.collect_claims(query, namespace, rest, &hints, index, first);
            return (0, PrefixClaims::default(), tail);
        }

        let first_id = run[0].0.id;
        let mut first_tail = OwnerRuns::default();
        for (offset, &key) in rest.iter().enumerate() {
            let same = self
                .read_hint(hints.get(offset), group, key, |row| {
                    let Some(owners) = row.holders(namespace) else {
                        return false;
                    };
                    // Same cardinality plus the exact ID and tier of each holder.
                    let same = owners.len() == run.len()
                        && owners.values().all(|holder| {
                            usize::try_from(holder.owner.id.wrapping_sub(first_id))
                                .ok()
                                .and_then(|slot| run.get(slot))
                                .is_some_and(|(_, tiers)| *tiers == holder.residency.tiers)
                        });
                    if !same {
                        first_tail.extend(owners.values().map(|holder| {
                            (
                                holder.owner.id,
                                (holder.owner.clone(), vec![holder.residency.tiers]),
                            )
                        }));
                    }
                    same
                })
                .unwrap_or(false);
            if !same {
                let tail = if query.stop_at_gap && first_tail.is_empty() {
                    OwnerClaims::default()
                } else {
                    self.collect_claims(
                        query,
                        namespace,
                        &rest[offset + 1..],
                        hints.get(offset + 1..).unwrap_or_default(),
                        index,
                        first_tail,
                    )
                };
                return (offset + 1, Self::prefix_claims(run), tail);
            }
        }
        (
            rest.len() + 1,
            Self::prefix_claims(run),
            OwnerClaims::default(),
        )
    }

    fn prefix_claims(claims: Vec<Claim>) -> PrefixClaims {
        claims
            .into_iter()
            .map(|(owner, tier)| (owner.owner.clone(), tier))
            .collect()
    }

    // The source read gate protects the reverse set while handles are copied.
    // Each handle's current namespace, holders and tiers are checked after releasing the gate.
    fn source_hints(
        index: &KvBlockIndexer,
        group: u32,
        namespace: Option<&str>,
        owner: &PageOwner,
        keys: &[SequenceHash],
        hints: &mut Vec<PageRef>,
    ) {
        let Some(source) = index.existing_source(&owner.owner.target()) else {
            return;
        };
        let Some(data) = source.access.try_read() else {
            return;
        };
        let Some(pages) = data
            .active_pages
            .get(&owner.owner)
            .and_then(|stored| stored.pages(namespace))
        else {
            return;
        };
        hints.extend(keys.iter().map_while(|&key| {
            pages
                .get_key_value(&PageKey::new(group, key))
                .map(|(page, ())| page.clone())
        }));
    }

    fn collect_claims(
        &self,
        query: PageQuery,
        namespace: Option<&str>,
        keys: &[SequenceHash],
        hint: &[PageRef],
        index: &KvBlockIndexer,
        mut holders: OwnerRuns,
    ) -> OwnerClaims {
        // Nested directory reads can share the thread's epoch for this scan.
        let _guard = scc::Guard::new();
        let mut record = |position, row: &PageRow, need_hint: bool| {
            let owners = row.holders(namespace).filter(|owners| !owners.is_empty())?;
            for holder in owners.values() {
                let (_, pages) = holders
                    .entry(holder.owner.id)
                    .or_insert_with(|| (holder.owner.clone(), Vec::new()));
                pages.resize(position, TierBits::default());
                pages.push(holder.residency.tiers);
            }
            Some(if need_hint {
                owners.values().next().map(|holder| holder.owner.clone())
            } else {
                None
            })
        };
        let mut position = 0;
        let mut pending = hint.iter();
        let mut hints = Vec::new();
        'tail: while position < keys.len() {
            let hint = pending.next();
            let found = self
                .read_hint(hint, query.group, keys[position], |row| {
                    record(
                        position + 1,
                        row,
                        hint.is_none() && position + 1 < keys.len(),
                    )
                })
                .flatten();
            position += 1;
            if query.stop_at_gap && found.is_none() {
                break;
            }
            if let Some(Some(owner)) = found {
                Self::source_hints(
                    index,
                    query.group,
                    namespace,
                    &owner,
                    &keys[position..],
                    &mut hints,
                );
                for page in hints.drain(..) {
                    if record(position + 1, &page.0.row.read(), false).is_none()
                        && query.stop_at_gap
                    {
                        break 'tail;
                    }
                    position += 1;
                }
            }
        }
        holders
            .into_values()
            .map(|(owner, pages)| (owner.owner.clone(), pages))
            .collect()
    }

    pub(super) fn for_owner(
        &self,
        active: Option<&OwnerPages>,
        group: u32,
        namespace: Option<&str>,
        keys: &[SequenceHash],
    ) -> Vec<TierBits> {
        let Some(active) = active else {
            return Vec::new();
        };
        let Some(pages) = active.pages(namespace) else {
            return vec![TierBits::default(); keys.len()];
        };
        if active.device_only() {
            // The source read gate covers both membership and the tier count.
            let device = TierBits(TierBits::bit(StorageTier::Device));
            return keys
                .iter()
                .map(|&hash| {
                    if pages.contains_key(&PageKey::new(group, hash)) {
                        device
                    } else {
                        TierBits::default()
                    }
                })
                .collect();
        }
        keys.iter()
            .map(|&key| {
                pages
                    .get_key_value(&PageKey::new(group, key))
                    .and_then(|(page, ())| {
                        page.0
                            .row
                            .read()
                            .holders(namespace)?
                            .get(active.owner.id)
                            .map(|holder| holder.residency.tiers)
                    })
                    .unwrap_or_default()
            })
            .collect()
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::kv_index::storage::GroupPages;
    use crate::kv_index::{CacheGroup, GroupRule, KvBlockIndexer};

    fn owner(name: &str) -> ResidencyOwner {
        ResidencyOwner {
            source: Arc::from(name),
            dp_rank: 0,
            incarnation: 1,
        }
    }

    fn put(index: &KvBlockIndexer, owner: &ResidencyOwner, group: u32) {
        index.with_source_batch(&owner.target(), |tx| {
            tx.store_hash_pages(owner, group, StorageTier::Device, None, &[[7; 32]], true, 1)
                .unwrap();
        });
    }

    #[test]
    fn invalid_device_mode_and_holder_overflow_preserve_existing_pages() {
        let index = KvBlockIndexer::new();
        let owner = owner("worker");
        let key = [7; 32];
        index.with_source_batch(&owner.target(), |tx| {
            assert_eq!(
                tx.store_hash_pages(&owner, 0, StorageTier::Device, None, &[key], false, 1),
                Err(PageWriteError::InvalidResidencyMode)
            );
        });
        assert!(index.hash_pages.rows.is_empty());

        put(&index, &owner, 0);
        let entry = index
            .hash_pages
            .rows
            .get_sync(&PageKey::new(0, key))
            .unwrap();
        {
            let mut row = entry.key().0.row.write();
            let holders = row.holders_mut(None);
            let id = holders.values().next().unwrap().owner.id;
            holders
                .get_mut(id)
                .unwrap()
                .residency
                .set_device_copies_for_test(u32::MAX);
        }
        drop(entry);
        index.with_source_batch(&owner.target(), |tx| {
            assert_eq!(
                tx.store_hash_pages(&owner, 0, StorageTier::Device, None, &[key], true, 1),
                Err(PageWriteError::Capacity)
            );
            assert_eq!(
                tx.remove_hash_pages(&owner, 0, StorageTier::Device, &[key], false),
                Err(PageWriteError::InvalidResidencyMode)
            );
            tx.remove_hash_pages(&owner, 0, StorageTier::Device, &[key], true)
                .unwrap();
        });
        assert!(
            index.collect_hash_pages(0, &[key]).for_owner(&owner)[0].contains(StorageTier::Device)
        );
    }

    #[test]
    fn concurrent_owner_registration_keeps_all_shared_page_claims() {
        let index = KvBlockIndexer::new();
        let threads = 8;
        let owners_per_thread = 128;
        let barrier = std::sync::Barrier::new(threads);
        std::thread::scope(|scope| {
            for thread in 0..threads {
                let (index, barrier) = (&index, &barrier);
                scope.spawn(move || {
                    barrier.wait();
                    for slot in 0..owners_per_thread {
                        put(index, &owner(&format!("worker-{thread}-{slot}")), 0);
                    }
                });
            }
        });
        assert_eq!(
            index.hash_pages.read_row(&PageKey::new(0, [7; 32]), |row| {
                row.holders(None).unwrap().values().count()
            }),
            Some(threads * owners_per_thread)
        );
    }

    #[test]
    fn last_holder_removal_leaves_a_pinned_row_empty_after_another_rank_recreates_it() {
        let index = KvBlockIndexer::new();
        let worker = |dp_rank| ResidencyOwner {
            source: Arc::from("w"),
            dp_rank,
            incarnation: 1,
        };
        let old = worker(0);
        let new = worker(1);
        let digest = [7; 32];
        put(&index, &old, 0);
        let hash = PageKey::new(0, digest);
        let page = index
            .hash_pages
            .rows
            .peek_with(&hash, |page, ()| page.clone())
            .unwrap();
        index.with_source_batch(&old.target(), |tx| {
            tx.remove_hash_pages(&old, 0, StorageTier::Device, &[digest], true)
                .unwrap();
        });
        put(&index, &new, 0);
        let owns = |row: &PageRow, owner: &ResidencyOwner| {
            row.holders(None)
                .is_some_and(|holders| holders.values().any(|holder| holder.owner.owner == *owner))
        };
        let retained = page.0.row.read();
        assert!(!owns(&retained, &old));
        assert!(!owns(&retained, &new));
        assert_eq!(
            index
                .hash_pages
                .read_row(&hash, |row| (owns(row, &old), owns(row, &new))),
            Some((false, true))
        );
    }

    #[test]
    fn shared_queries_keep_survivor_hits_while_retirement_clears_pages() {
        let index = KvBlockIndexer::new();
        let retiring = owner("retiring");
        let survivor = owner("survivor");
        let target = retiring.target();
        let gate = index.source_lock(&target);
        let keys = [[7; 32]];
        let cold = [8; 32];
        for owner in [&retiring, &survivor] {
            index.with_source_batch(&owner.target(), |tx| {
                for group in 0..2 {
                    tx.store_hash_pages(owner, group, StorageTier::Device, None, &keys, true, 8)
                        .unwrap();
                }
            });
        }
        index.with_source_batch(&target, |tx| {
            tx.store_hash_pages(&retiring, 0, StorageTier::Device, None, &[cold], true, 8)
                .unwrap();
        });
        let page = index
            .hash_pages
            .rows
            .peek_with(&PageKey::new(0, cold), |page, ()| page.clone())
            .unwrap();

        std::thread::scope(|scope| {
            // Hold an unrelated page open so a real retirement cannot complete.
            let blocked = page.0.row.read();
            let index = &index;
            let target = &target;
            scope.spawn(move || index.retire_source(&target.instance_id, target.dp_rank));
            let timeout = std::time::Duration::from_secs(5);
            let started = std::time::Instant::now();
            while gate.generation.load(Ordering::Acquire).is_multiple_of(2) {
                assert!(
                    started.elapsed() < timeout,
                    "source did not enter retirement"
                );
                std::thread::yield_now();
            }

            let (done, completed) = std::sync::mpsc::channel();
            scope.spawn(move || {
                let prefix = index
                    .collect_prefix(
                        PageQuery {
                            group: 0,
                            stop_at_gap: true,
                        },
                        &keys,
                    )
                    .unwrap();
                let groups = index.collect_hash_groups(0u32..2, None, &keys);
                for evidence in [vec![&prefix], groups.values().collect()] {
                    for (owner, visible) in [(&retiring, false), (&survivor, true)] {
                        let matched = PageEvidence::unchanged_groups(
                            index,
                            &owner.target(),
                            owner,
                            &keys,
                            evidence.iter().copied().map(Some),
                        );
                        if visible {
                            let Some(PageMatch::Hit(pages, version)) = matched else {
                                panic!("survivor lost shared pages")
                            };
                            assert!(pages.iter().all(|group| group.get(0).is_some()));
                            assert!(version.is_current());
                        } else {
                            assert!(matches!(matched, Some(PageMatch::Miss)));
                        }
                    }
                }
                done.send(()).unwrap();
            });
            let result = completed.recv_timeout(timeout);
            drop(blocked);
            result.expect("shared queries must finish while a different rank retires");
        });
    }

    #[test]
    fn page_keys_keep_groups_and_digests_distinct_on_bucket_hash_collision() {
        let key = PageKey::new(0, [7; 32]);
        let fingerprint = key.fingerprint;
        let mut pages = Pages::default();
        for key in [
            key,
            PageKey {
                group: 1,
                fingerprint,
                digest: [7; 32],
            },
            PageKey {
                group: 0,
                fingerprint,
                digest: [8; 32],
            },
        ] {
            assert!(pages.insert(PageRef::new(key), ()).is_none());
        }
        assert!(pages.contains_key(&PageKey::new(0, [7; 32])));
    }

    #[test]
    fn resident_writes_at_quota_preserve_exact_handle_across_other_sources_evictions() {
        let index = KvBlockIndexer::new();
        let a = owner("keeper");
        let b = owner("churn");
        let ns = Some("adapter");
        let hash = [19; 32];
        let extra = [23; 32];
        index.with_source_batch(&a.target(), |tx| {
            tx.store_hash_pages(&a, 0, StorageTier::Device, ns, &[hash], true, 1)
                .unwrap();
            assert!(matches!(
                tx.store_hash_pages(&a, 0, StorageTier::Device, ns, &[extra], true, 1),
                Err(PageWriteError::Capacity)
            ));
        });
        let key = PageKey::new(0, hash);
        let pinned = index
            .hash_pages
            .rows
            .peek_with(&key, |p, _| p.clone())
            .unwrap();
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|s| {
            s.spawn(|| {
                barrier.wait();
                for _ in 0..1000 {
                    index.with_source_batch(&b.target(), |tx| {
                        tx.store_hash_pages(&b, 0, StorageTier::Device, ns, &[hash], true, 1)
                            .unwrap()
                    });
                    index.with_source_batch(&b.target(), |tx| tx.clear_owner_exact(&b));
                }
            });
            barrier.wait();
            for _ in 0..1000 {
                index.with_source_batch(&a.target(), |tx| {
                    tx.store_hash_pages(&a, 0, StorageTier::Device, ns, &[hash], true, 1)
                        .unwrap();
                    tx.store_hash_pages(&a, 0, StorageTier::HostPinned, ns, &[hash], false, 1)
                        .unwrap();
                    tx.remove_hash_pages(&a, 0, StorageTier::Device, &[hash], true)
                        .unwrap();
                });
                let g = index.collect_hash_groups(0u32..1, ns, &[hash, extra]);
                let e = g[&0.into()].for_owner(&a);
                assert_eq!(e[0].0, 3);
                assert!(e.get(1).is_none_or(|p| p.0 == 0));
                assert!(index
                    .hash_pages
                    .rows
                    .peek_with(&key, |p, _| Arc::ptr_eq(&p.0, &pinned.0))
                    .unwrap());
                index.with_source_batch(&a.target(), |tx| {
                    tx.remove_hash_pages(&a, 0, StorageTier::HostPinned, &[hash], false)
                        .unwrap()
                });
            }
        });
        index.with_source_batch(&a.target(), |tx| {
            tx.remove_hash_pages(&a, 0, StorageTier::Device, &[hash], true)
                .unwrap()
        });
        assert!(index.hash_pages.rows.is_empty());
    }

    #[test]
    fn an_over_capacity_store_leaves_memberships_and_replica_counts_unchanged() {
        let index = KvBlockIndexer::new();
        let worker = owner("worker");
        let existing = [7; 32];
        let admitted = [8; 32];
        let rejected = [9; 32];
        put(&index, &worker, 0);
        index.with_source_batch(&worker.target(), |tx| {
            assert_eq!(
                tx.store_hash_pages(
                    &worker,
                    0,
                    StorageTier::Device,
                    None,
                    &[existing, admitted, rejected],
                    true,
                    2,
                ),
                Err(PageWriteError::Capacity)
            );
            tx.remove_hash_pages(&worker, 0, StorageTier::Device, &[existing], true)
                .unwrap();
        });
        assert!(index.hash_pages.rows.is_empty());

        index.with_source_batch(&worker.target(), |tx| {
            tx.store_hash_pages(
                &worker,
                0,
                StorageTier::Device,
                None,
                &[admitted; 2],
                true,
                1,
            )
            .unwrap();
            tx.remove_hash_pages(&worker, 0, StorageTier::Device, &[admitted], true)
                .unwrap();
        });
        assert!(!index.hash_pages.rows.is_empty());
        index.with_source_batch(&worker.target(), |tx| {
            tx.remove_hash_pages(&worker, 0, StorageTier::Device, &[admitted], true)
                .unwrap();
        });
        assert!(index.hash_pages.rows.is_empty());
    }

    #[test]
    fn base_and_adapter_pages_share_one_source_budget() {
        let index = KvBlockIndexer::new();
        let holder = owner("worker");
        let base = [[1; 32]];
        let a = [[2; 32]];
        let b = [[3; 32]];
        index.with_source_batch(&holder.target(), |tx| {
            tx.store_hash_pages(&holder, 0, StorageTier::Device, None, &base, true, 2)
                .unwrap();
            tx.store_hash_pages(&holder, 0, StorageTier::Device, Some("a"), &a, true, 2)
                .unwrap();
            assert_eq!(
                tx.store_hash_pages(&holder, 0, StorageTier::Device, Some("b"), &b, true, 2),
                Err(PageWriteError::Capacity)
            );
            tx.remove_hash_pages(&holder, 0, StorageTier::Device, &a, true)
                .unwrap();
            tx.store_hash_pages(&holder, 0, StorageTier::Device, Some("b"), &b, true, 2)
                .unwrap();
        });
        assert_eq!(index.hash_pages.rows.len(), 2);
    }

    #[test]
    fn replicas_tiers_capacity_and_retirement_keep_one_shared_fact() {
        let index = KvBlockIndexer::new();
        let a = owner("a");
        let b = owner("b");
        put(&index, &a, 0);
        put(&index, &a, 0);
        put(&index, &b, 0);
        index.with_source_batch(&a.target(), |tx| {
            tx.remove_hash_pages(&a, 0, StorageTier::Device, &[[7; 32]], true)
                .unwrap();

            assert_eq!(
                tx.store_hash_pages(&a, 1, StorageTier::Device, None, &[[8; 32]], true, 1),
                Err(PageWriteError::Capacity)
            );
        });
        let key = PageKey::new(0, [7; 32]);
        let holds = |owner: &ResidencyOwner| {
            index
                .hash_pages
                .read_row(&key, |row| {
                    row.holders(None).is_some_and(|holders| {
                        holders.values().any(|holder| holder.owner.owner == *owner)
                    })
                })
                .unwrap_or(false)
        };
        assert!(holds(&a));
        index.with_source_batch(&a.target(), |tx| {
            tx.remove_hash_pages(&a, 0, StorageTier::Device, &[[7; 32]], true)
                .unwrap();
        });
        assert!(!holds(&a));
        assert!(holds(&b));
        index.retire_source(&b.target().instance_id, 0);
        assert!(index.hash_pages.rows.is_empty());
        assert!(index
            .source_lock(&a.target())
            .access
            .read()
            .active_pages
            .is_empty());
    }

    #[test]
    fn raw_remove_updates_only_matching_namespaces_of_its_owner() {
        let index = KvBlockIndexer::new();
        let a = owner("a");
        let b = owner("b");
        let hash = [7; 32];
        for (owner, name, copies) in [
            (&a, "adapter-a", 2),
            (&a, "adapter-b", 1),
            (&b, "adapter-b", 1),
        ] {
            index.with_source_batch(&owner.target(), |tx| {
                for _ in 0..copies {
                    tx.store_hash_pages(
                        owner,
                        0,
                        StorageTier::Device,
                        Some(name),
                        &[hash],
                        true,
                        3,
                    )
                    .unwrap();
                }
            });
        }
        index.with_source_batch(&a.target(), |tx| {
            tx.store_hash_pages(&a, 0, StorageTier::Device, None, &[hash], true, 3)
                .unwrap();
            tx.remove_hash_pages(&a, 0, StorageTier::Device, &[hash], true)
                .unwrap();
        });
        let has = |owner, namespace| {
            index
                .collect_hash_groups(std::iter::once(0u32), namespace, &[hash])
                .remove(&0.into())
                .unwrap()
                .for_owner(owner)
                .first()
                .is_some_and(|tiers| tiers.contains(StorageTier::Device))
        };
        assert!(!has(&a, None));
        assert!(has(&a, Some("adapter-a")));
        assert!(!has(&a, Some("adapter-b")));
        assert!(has(&b, Some("adapter-b")));
        index.with_source_batch(&a.target(), |tx| {
            tx.remove_hash_pages(&a, 0, StorageTier::Device, &[hash], true)
                .unwrap();
            tx.store_hash_pages(
                &a,
                0,
                StorageTier::Device,
                Some("adapter-a"),
                &[[8; 32]],
                true,
                1,
            )
            .unwrap();
        });
        assert!(!has(&a, Some("adapter-a")));
        index.retire_source(&a.target().instance_id, 0);
        assert!(has(&b, Some("adapter-b")));
        index.retire_source(&b.target().instance_id, 0);
        assert!(index.hash_pages.rows.is_empty());
    }

    #[test]
    fn changed_and_initially_missing_sources_get_an_exact_fallback() {
        let index = KvBlockIndexer::new();
        let a = owner("a");
        let keys = [[7; 32]];
        let mut postings = GroupPages::default();
        postings.insert(0.into(), index.collect_hash_pages(0, &keys));
        put(&index, &a, 0);
        let groups = [CacheGroup {
            group_idx: 0,
            block_size: 16,
            kind: GroupRule::FullAttention,
        }];
        let pages = index
            .with_source_snapshot(&a.target(), |view| {
                view.hash_page_tiers(
                    &a,
                    None,
                    &keys,
                    &postings,
                    groups.iter().map(|g| g.group_idx),
                )
            })
            .unwrap();
        assert!(pages[0][0].contains(StorageTier::Device));
        postings.insert(0.into(), index.collect_hash_pages(0, &keys));
        index.with_source_batch(&a.target(), |tx| tx.clear_owner_exact(&a));
        let pages = index
            .with_source_snapshot(&a.target(), |view| {
                view.hash_page_tiers(
                    &a,
                    None,
                    &keys,
                    &postings,
                    groups.iter().map(|g| g.group_idx),
                )
            })
            .unwrap();
        assert!(pages[0].is_empty());
    }

    #[test]
    fn stale_evidence_keeps_exact_tiers_through_mixed_residency_and_last_copy() {
        let index = KvBlockIndexer::new();
        let owner = owner("worker");
        let keys = [[7; 32], [8; 32]];
        let snapshots = GroupPages::default();
        let repair = |group, namespace| {
            index
                .with_source_snapshot(&owner.target(), |view| {
                    view.hash_page_tiers(
                        &owner,
                        namespace,
                        &keys,
                        &snapshots,
                        std::iter::once(group),
                    )[0]
                    .iter()
                    .map(|bits| bits.0)
                    .collect::<Vec<_>>()
                })
                .unwrap()
        };
        let store = |group, namespace, tier| {
            index.with_source_batch(&owner.target(), |tx| {
                tx.store_hash_pages(
                    &owner,
                    group,
                    tier,
                    namespace,
                    &keys[..1],
                    tier == StorageTier::Device,
                    4,
                )
                .unwrap();
            });
        };
        let remove = |group, tier| {
            index.with_source_batch(&owner.target(), |tx| {
                tx.remove_hash_pages(&owner, group, tier, &keys, tier == StorageTier::Device)
                    .unwrap();
            });
        };

        store(0, None, StorageTier::Device);
        store(0, None, StorageTier::Device);
        store(1, Some("adapter"), StorageTier::Device);
        assert_eq!(repair(0, None), [1, 0]);
        assert_eq!(repair(1, Some("adapter")), [1, 0]);
        store(0, None, StorageTier::HostPinned);
        store(0, None, StorageTier::HostPinned);
        store(1, Some("adapter"), StorageTier::Disk);
        assert_eq!(repair(0, None), [3, 0]);
        assert_eq!(repair(1, Some("adapter")), [5, 0]);
        remove(0, StorageTier::HostPinned);
        remove(0, StorageTier::HostPinned);
        assert_eq!(repair(0, None), [1, 0]);
        assert_eq!(repair(1, Some("adapter")), [5, 0]);
        remove(1, StorageTier::Disk);
        remove(0, StorageTier::Device);
        assert_eq!(repair(0, None), [1, 0]);
        remove(0, StorageTier::Device);
        assert_eq!(repair(0, None), [0, 0]);
        assert_eq!(repair(1, Some("adapter")), [1, 0]);
    }

    #[test]
    fn adjacent_pages_require_the_same_owners_and_tiers() {
        let index = KvBlockIndexer::new();
        let a = owner("a");
        let b = owner("b");
        let c = owner("c");
        let keys = [[1; 32], [2; 32], [3; 32], [4; 32], [5; 32], [6; 32]];
        for (owner, tier, positions) in [
            (&a, StorageTier::Device, &[0, 1, 3, 5][..]),
            (&b, StorageTier::Device, &[0, 1, 2][..]),
            (&a, StorageTier::HostPinned, &[2][..]),
            (&c, StorageTier::Device, &[3, 5][..]),
        ] {
            let hashes: Vec<_> = positions.iter().map(|&position| keys[position]).collect();
            index.with_source_batch(&owner.target(), |tx| {
                tx.store_hash_pages(
                    owner,
                    0,
                    tier,
                    None,
                    &hashes,
                    tier == StorageTier::Device,
                    6,
                )
                .unwrap();
            });
        }
        // Pages 0 and 1 are identical. Page 2 changes the tier, page 3
        // replaces one owner, and page 4 breaks the run before page 5.
        // The IDs of a and c also exercise a noncontiguous roster.
        let evidence = index.collect_hash_pages(0, &keys);
        for (owner, expected) in [
            (&a, &[1, 1, 2, 1, 0, 1][..]),
            (&b, &[1, 1, 1][..]),
            (&c, &[0, 0, 0, 1, 0, 1][..]),
        ] {
            let actual: Vec<_> = evidence
                .for_owner(owner)
                .iter()
                .map(|tiers| tiers.0)
                .collect();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn a_later_holder_supplies_handles_without_implying_ownership_or_tiers() {
        let index = KvBlockIndexer::new();
        let (a, b, c) = (owner("a"), owner("b"), owner("c"));
        let keys: Vec<_> = (0..7).map(|byte| [byte; 32]).collect();
        let named = Some("adapter");
        for (owner, namespace, tier, positions) in [
            (&b, named, StorageTier::Device, &[0][..]),
            (&a, named, StorageTier::Device, &[1, 2, 3, 5, 6][..]),
            (&a, named, StorageTier::HostPinned, &[3][..]),
            (&c, named, StorageTier::Device, &[2, 6][..]),
            (&b, None, StorageTier::Device, &[4][..]),
        ] {
            let hashes: Vec<_> = positions.iter().map(|&i| keys[i]).collect();
            index.with_source_batch(&owner.target(), |tx| {
                tx.store_hash_pages(
                    owner,
                    0,
                    tier,
                    namespace,
                    &hashes,
                    tier == StorageTier::Device,
                    7,
                )
                .unwrap();
            });
        }
        let groups = index.collect_hash_groups(0u32..1, named, &keys);
        for (owner, expected) in [
            (&a, &[0, 1, 1, 3, 0, 1, 1][..]),
            (&b, &[1][..]),
            (&c, &[0, 0, 1, 0, 0, 0, 1][..]),
        ] {
            let tiers: Vec<_> = groups[&0.into()]
                .for_owner(owner)
                .iter()
                .map(|value| value.0)
                .collect();
            assert_eq!(tiers, expected);
        }
    }

    #[test]
    fn sparse_evidence_keeps_owner_group_and_tier_boundaries() {
        let index = KvBlockIndexer::new();
        let a = owner("a");
        let b = owner("b");
        for (owner, group, tier, hashes) in [
            (&a, 0, StorageTier::Device, vec![[1; 32], [3; 32]]),
            (&a, 1, StorageTier::Device, vec![[3; 32]]),
            (&b, 0, StorageTier::HostPinned, vec![[2; 32]]),
        ] {
            index.with_source_batch(&owner.target(), |tx| {
                tx.store_hash_pages(
                    owner,
                    group,
                    tier,
                    None,
                    &hashes,
                    tier == StorageTier::Device,
                    4,
                )
                .unwrap();
            });
        }
        let keys = [[1; 32], [2; 32], [3; 32], [4; 32]];
        let evidence = index.collect_hash_groups(0u32..2, None, &keys);
        let snapshot = |owner: &ResidencyOwner| {
            index
                .with_source_snapshot(&owner.target(), |view| {
                    view.hash_page_tiers(owner, None, &keys, &evidence, 0..2)
                        .into_iter()
                        .map(|pages| pages.iter().map(|page| page.0).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                })
                .unwrap()
        };
        assert_eq!(snapshot(&a), vec![vec![1, 0, 1], vec![0, 0, 1]]);
        assert_eq!(snapshot(&b), vec![vec![0, 2], vec![]]);
    }

    #[test]
    fn prefix_scan_keeps_tier_continuations_and_stops_at_an_empty_page() {
        let index = KvBlockIndexer::new();
        let holder = owner("a");
        let keys = [[1; 32], [2; 32], [3; 32], [4; 32]];
        let prefix = PageQuery {
            group: 0,
            stop_at_gap: true,
        };
        let complete = PageQuery::from(0);
        index.with_source_batch(&holder.target(), |tx| {
            for (tier, hashes) in [
                (StorageTier::Device, &keys[..1]),
                (StorageTier::HostPinned, &keys[1..2]),
                (StorageTier::HostPinned, &keys[3..]),
            ] {
                tx.store_hash_pages(
                    &holder,
                    0,
                    tier,
                    None,
                    hashes,
                    tier == StorageTier::Device,
                    8,
                )
                .unwrap();
            }
        });
        let groups = index.collect_hash_groups([prefix, complete].into_iter(), None, &keys);
        let tiers = |query| {
            groups[&query]
                .for_owner(&holder)
                .iter()
                .map(|bits| bits.0)
                .collect::<Vec<_>>()
        };
        assert_eq!(tiers(prefix), [1, 2]);
        assert_eq!(tiers(complete), [1, 2, 0, 2]);
    }

    #[test]
    fn concurrent_sources_keep_their_own_copy_when_the_other_source_removes() {
        let index = Arc::new(KvBlockIndexer::new());
        let start = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            for name in ["a", "b"] {
                let index = &index;
                let start = &start;
                scope.spawn(move || {
                    let owner = owner(name);
                    let keys = [[7; 32]];
                    start.wait();
                    for _ in 0..500 {
                        put(index, &owner, 0);
                        let postings =
                            index.collect_hash_groups(std::iter::once(0u32), None, &keys);
                        let pages = index
                            .with_source_snapshot(&owner.target(), |view| {
                                view.hash_page_tiers(
                                    &owner,
                                    None,
                                    &keys,
                                    &postings,
                                    std::iter::once(0),
                                )
                            })
                            .unwrap();
                        assert!(pages[0][0].contains(StorageTier::Device));
                        index.with_source_batch(&owner.target(), |tx| {
                            tx.remove_hash_pages(&owner, 0, StorageTier::Device, &[[7; 32]], true)
                                .unwrap();
                        });
                    }
                });
            }
        });
        assert!(index.hash_pages.rows.is_empty());
        for source in index.source_locks.read().values() {
            assert!(source.access.read().active_pages.is_empty());
        }
    }

    #[test]
    fn concurrent_postings_never_combine_groups_from_different_commits() {
        let index = Arc::new(KvBlockIndexer::new());
        let a = owner("writer");
        put(&index, &a, 0);
        std::thread::scope(|scope| {
            let writer_index = index.clone();
            let writer_owner = a.clone();
            scope.spawn(move || {
                let index = writer_index;
                let a = writer_owner;
                for i in 0..500 {
                    index.with_source_batch(&a.target(), |tx| {
                        tx.clear_owner_exact(&a);
                        tx.store_hash_pages(
                            &a,
                            i % 2,
                            StorageTier::Device,
                            None,
                            &[[7; 32]],
                            true,
                            1,
                        )
                        .unwrap();
                    });
                }
            });
            for _ in 0..2 {
                let index = index.clone();
                let a = a.clone();
                scope.spawn(move || {
                    let groups: Vec<_> = (0..2)
                        .map(|group_idx| CacheGroup {
                            group_idx,
                            block_size: 16,
                            kind: GroupRule::Checkpoint,
                        })
                        .collect();
                    let keys = [[7; 32]];
                    for _ in 0..500 {
                        let postings = (0..2)
                            .map(|group| (group.into(), index.collect_hash_pages(group, &keys)))
                            .collect();
                        let pages = index
                            .with_source_snapshot(&a.target(), |view| {
                                view.hash_page_tiers(
                                    &a,
                                    None,
                                    &keys,
                                    &postings,
                                    groups.iter().map(|g| g.group_idx),
                                )
                            })
                            .unwrap();
                        assert_eq!(
                            pages
                                .iter()
                                .filter(|p| p
                                    .first()
                                    .is_some_and(|bits| bits.contains(StorageTier::Device)))
                                .count(),
                            1
                        );
                    }
                });
            }
        });
    }
}
