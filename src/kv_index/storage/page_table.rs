//! Hash-addressed page facts; reverse memberships live in the source transaction.

use std::borrow::{Borrow, Cow};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::LazyLock;

use super::records::Residency;
use super::*;
use hashbrown::hash_map::EntryRef;

#[derive(PartialEq, Eq)]
struct PageKey {
    group: u32,
    fingerprint: u32,
    digest: SequenceHash,
}

impl PageKey {
    fn new(group: u32, digest: SequenceHash) -> Self {
        static HASHER: LazyLock<Fast> = LazyLock::new(Fast::default);
        let fingerprint = HASHER.hash_one((group, digest)) as u32;
        Self {
            group,
            fingerprint,
            digest,
        }
    }
}

impl Hash for PageKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Rehashing either index only needs the cached bucket hash. Equality
        // still includes the complete digest and group.
        state.write_u32(self.fingerprint);
    }
}
type HolderMap = HashMap<u64, PageHolder, Fast>;
type Claim = (Arc<PageOwner>, TierBits);
type OwnerClaims = HashMap<ResidencyOwner, Vec<TierBits>, Fast>;
type Pages = HashSet<Arc<PageKey>, Fast>;

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

type Namespaces<T> = hashbrown::HashMap<Namespace, T, Fast>;
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
    named: Namespaces<Pages>,
    // Count non-device tier claims, including named pages and mixed-tier pages.
    non_device_tiers: usize,
}

impl OwnerPages {
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
    rows: DashMap<Arc<PageKey>, PageRow, Fast>,
    next_owner: AtomicU64,
}

/// An optimistic page scan and the source versions sampled before it started.
pub(crate) struct PageEvidence {
    pub(super) versions: Arc<HashMap<RoutableTarget, (Arc<SourceLock>, u64), Fast>>,
    pub(super) prefix_len: usize,
    pub(super) prefix: HashMap<ResidencyOwner, TierBits, Fast>,
    pub(super) holders: OwnerClaims,
}

pub(crate) struct PageTiers<'a> {
    pub(crate) prefix_len: usize,
    pub(crate) prefix: TierBits,
    pub(crate) tail: Cow<'a, [TierBits]>,
}

impl<'a> PageTiers<'a> {
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

impl PageEvidence {
    /// Reuse the scan for a source whose transaction did not change during it.
    pub(crate) fn unchanged(
        &self,
        target: &RoutableTarget,
        owner: &ResidencyOwner,
    ) -> Option<(PageTiers<'_>, SourceVersion)> {
        let (source, revision) = self.versions.get(target)?;
        let generation = source.generation.load(Ordering::Acquire);
        if !revision.is_multiple_of(2) || source.revision.load(Ordering::Acquire) != *revision {
            return None;
        }
        Some((
            self.runs_for_owner(owner),
            SourceVersion {
                source: source.clone(),
                generation,
            },
        ))
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
    keys: &mut Pages,
    page: &Arc<PageKey>,
    non_device_tiers: &mut usize,
    tier: StorageTier,
    incremental: bool,
) -> Result<(), PageWriteError> {
    if let Some(holder) = holders.get_mut(owner.id) {
        let fact = &mut holder.residency;
        let new_non_device = tier != StorageTier::Device && !fact.contains(tier);
        if !fact.observe(tier, incremental) {
            return Err(PageWriteError::Capacity);
        }
        if new_non_device {
            *non_device_tiers += 1;
        }
    } else {
        let mut residency = Residency::default();
        let accepted = residency.observe(tier, incremental);
        debug_assert!(accepted);
        if tier != StorageTier::Device {
            *non_device_tiers += 1;
        }
        holders.insert(PageHolder {
            owner: owner.clone(),
            residency,
        });
        keys.insert(page.clone());
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
        if incremental && tier != StorageTier::Device {
            return Err(PageWriteError::ConflictingHash);
        }
        if hashes.is_empty() {
            return Ok(());
        }
        let pages = match active.entry_ref(owner) {
            EntryRef::Occupied(entry) => entry.into_mut(),
            EntryRef::Vacant(entry) => entry.insert_with_key(
                Arc::new(owner.clone()),
                OwnerPages {
                    owner: Arc::new(PageOwner {
                        id: self
                            .next_owner
                            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| {
                                id.checked_add(1)
                            })
                            .expect("KV page owner IDs exhausted"),
                        owner: owner.clone(),
                    }),
                    base: Pages::default(),
                    named: Namespaces::default(),
                    non_device_tiers: 0,
                },
            ),
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
            keys.reserve(hashes.len().min(max_entries.saturating_sub(keys.len())));
        }
        for &hash in hashes {
            let page = PageKey::new(group, hash);
            if keys.len() >= max_entries && !keys.contains(&page) {
                return Err(PageWriteError::Capacity);
            }
            // The entry path rechecks after a miss racing another source.
            let mut row = self
                .rows
                .get_mut(&page)
                .unwrap_or_else(|| self.rows.entry(Arc::new(page)).or_default());
            let (key, row) = row.pair_mut();
            observe(
                row.holders_mut(namespace.as_ref()),
                &pages.owner,
                keys,
                key,
                &mut pages.non_device_tiers,
                tier,
                incremental,
            )?;
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
    ) {
        if incremental && tier != StorageTier::Device {
            return;
        }
        let Some(pages) = active.get_mut(owner) else {
            return;
        };
        let id = pages.owner.id;
        for &hash in hashes {
            // Remove carries the raw hash but no adapter name. Only namespaces
            // sharing this exact page need to be visited.
            self.rows
                .remove_if_mut(&PageKey::new(group, hash), |key, row| {
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
                        if let Some(name) = &scope.namespace {
                            if let EntryRef::Occupied(mut entry) = pages.named.entry_ref(name) {
                                entry.get_mut().remove(key);
                                if entry.get().is_empty() {
                                    entry.remove();
                                }
                            }
                        } else {
                            pages.base.remove(key);
                        }
                    }
                    row.prune()
                });
        }
        if pages.base.is_empty() && pages.named.is_empty() {
            active.remove(owner);
        }
    }

    pub(super) fn clear_owner(&self, active: &mut ActivePages, owner: &ResidencyOwner) {
        let Some(pages) = active.remove(owner) else {
            return;
        };
        let id = pages.owner.id;
        for page in pages
            .base
            .into_iter()
            .chain(pages.named.into_values().flatten())
        {
            self.rows.remove_if_mut(page.as_ref(), |_, row| {
                for scope in row.scopes_mut() {
                    scope.holders.remove(id);
                }
                row.prune()
            });
        }
        if self.rows.is_empty() {
            self.rows.shrink_to_fit();
        }
    }

    /// Full-attention callers can emit one interval for the common run.
    pub(super) fn collect_split(
        &self,
        group: u32,
        namespace: Option<&str>,
        keys: &[SequenceHash],
    ) -> (usize, Vec<(ResidencyOwner, TierBits)>, OwnerClaims) {
        let (skip, head) = self
            .shared_prefix(group, namespace, keys)
            .unwrap_or_default();
        let head = head
            .into_iter()
            .map(|(owner, tier)| (owner.owner.clone(), tier))
            .collect();
        let tail = self.collect_claims(group, namespace, &keys[skip..]);
        (skip, head, tail)
    }

    fn collect_claims(
        &self,
        group: u32,
        namespace: Option<&str>,
        keys: &[SequenceHash],
    ) -> OwnerClaims {
        let mut holders: HashMap<_, _, Fast> = HashMap::default();
        for (position, &key) in keys.iter().enumerate() {
            let Some(row) = self.rows.get(&PageKey::new(group, key)) else {
                continue;
            };
            if let Some(group) = row.holders(namespace) {
                for holder in group.values() {
                    let (_, pages) = holders
                        .entry(holder.owner.id)
                        .or_insert_with(|| (holder.owner.clone(), Vec::new()));
                    pages.resize(position, TierBits::default());
                    pages.push(holder.residency.tiers);
                }
            }
        }
        holders
            .into_values()
            .map(|(owner, pages)| (owner.owner.clone(), pages))
            .collect()
    }

    /// Collapse the initial shared run. Stop at the first different page so
    /// fragmented prefixes use the ordinary one-pass collection thereafter.
    fn shared_prefix(
        &self,
        group: u32,
        namespace: Option<&str>,
        keys: &[SequenceHash],
    ) -> Option<(usize, Vec<Claim>)> {
        let (head, tail) = keys.split_first()?;
        if tail.is_empty() {
            return None;
        }
        let row = self.rows.get(&PageKey::new(group, *head))?;
        let owners = row.holders(namespace)?;
        if owners.len() < 2 {
            return None;
        }
        let mut run: Vec<_> = owners
            .values()
            .map(|holder| (holder.owner.clone(), holder.residency.tiers))
            .collect();
        drop(row);
        run.sort_unstable_by_key(|(owner, _)| owner.id);
        let first_id = run.first()?.0.id;
        if run.last()?.0.id - first_id != (run.len() - 1) as u64 {
            return None;
        }
        // Same cardinality plus an exact ID and tier check for every holder
        // proves equality. This works across different hash table orders.
        let len = tail
            .iter()
            .take_while(|&&key| {
                let Some(row) = self.rows.get(&PageKey::new(group, key)) else {
                    return false;
                };
                let Some(owners) = row.holders(namespace) else {
                    return false;
                };
                owners.len() == run.len()
                    && owners.values().all(|holder| {
                        usize::try_from(holder.owner.id.wrapping_sub(first_id))
                            .ok()
                            .and_then(|slot| run.get(slot))
                            .is_some_and(|(_, tiers)| *tiers == holder.residency.tiers)
                    })
            })
            .count();
        (len != 0).then_some((len + 1, run))
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
        if active.device_only() {
            // The source read gate covers both this membership set and the
            // counter. Each member then has exactly the device tier.
            let pages = match namespace {
                Some(name) => active.named.get(name),
                None => Some(&active.base),
            };
            let Some(pages) = pages else {
                return vec![TierBits::default(); keys.len()];
            };
            let device = TierBits(TierBits::bit(StorageTier::Device));
            return keys
                .iter()
                .map(|&hash| {
                    if pages.contains(&PageKey::new(group, hash)) {
                        device
                    } else {
                        TierBits::default()
                    }
                })
                .collect();
        }
        keys.iter()
            .map(|&key| {
                self.rows
                    .get(&PageKey::new(group, key))
                    .and_then(|row| {
                        row.holders(namespace)?
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
    use crate::kv_index::{CacheGroup, GroupRule};

    fn owner(name: &str) -> ResidencyOwner {
        ResidencyOwner::Worker {
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
    fn page_keys_keep_groups_and_digests_distinct_on_bucket_hash_collision() {
        let key = PageKey::new(0, [7; 32]);
        let fingerprint = key.fingerprint;
        let mut pages = Pages::default();
        assert!(pages.insert(Arc::new(key)));
        assert!(pages.insert(Arc::new(PageKey {
            group: 1,
            fingerprint,
            digest: [7; 32],
        })));
        assert!(pages.insert(Arc::new(PageKey {
            group: 0,
            fingerprint,
            digest: [8; 32],
        })));
        assert!(pages.contains(&PageKey::new(0, [7; 32])));
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
            tx.remove_hash_pages(&a, 0, StorageTier::Device, &[[7; 32]], true);
            assert_eq!(
                tx.store_hash_pages(&a, 1, StorageTier::Device, None, &[[8; 32]], true, 1),
                Err(PageWriteError::Capacity)
            );
        });
        let key = PageKey::new(0, [7; 32]);
        let holds = |owner: &ResidencyOwner| {
            index
                .hash_pages
                .rows
                .get(&key)
                .and_then(|row| {
                    row.holders(None)
                        .map(|holders| holders.values().any(|holder| holder.owner.owner == *owner))
                })
                .unwrap_or(false)
        };
        assert!(holds(&a));
        index.with_source_batch(&a.target(), |tx| {
            tx.remove_hash_pages(&a, 0, StorageTier::Device, &[[7; 32]], true)
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
                        1,
                    )
                    .unwrap();
                }
            });
        }
        put(&index, &a, 0);
        index.with_source_batch(&a.target(), |tx| {
            tx.remove_hash_pages(&a, 0, StorageTier::Device, &[hash], true);
        });
        let has = |owner, namespace| {
            index
                .hash_pages
                .collect_split(0, namespace, &[hash])
                .2
                .contains_key(owner)
        };
        assert!(!has(&a, None));
        assert!(has(&a, Some("adapter-a")));
        assert!(!has(&a, Some("adapter-b")));
        assert!(has(&b, Some("adapter-b")));
        index.with_source_batch(&a.target(), |tx| {
            tx.remove_hash_pages(&a, 0, StorageTier::Device, &[hash], true);
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
        let mut postings = HashMap::new();
        postings.insert(0, index.collect_hash_pages(0, &keys));
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
        postings.insert(0, index.collect_hash_pages(0, &keys));
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
        let snapshots = HashMap::new();
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
                tx.remove_hash_pages(&owner, group, tier, &keys, tier == StorageTier::Device);
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
                tx.store_hash_pages(owner, 0, tier, None, &hashes, false, 6)
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
                tx.store_hash_pages(owner, group, tier, None, &hashes, false, 4)
                    .unwrap();
            });
        }
        let keys = [[1; 32], [2; 32], [3; 32], [4; 32]];
        let evidence = index.collect_hash_groups(0..2, None, &keys);
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
                        let postings = index.collect_hash_groups(std::iter::once(0), None, &keys);
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
                            tx.remove_hash_pages(&owner, 0, StorageTier::Device, &[[7; 32]], true);
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
                            .map(|group| (group, index.collect_hash_pages(group, &keys)))
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
