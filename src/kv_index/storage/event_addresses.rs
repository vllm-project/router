//! Resolve a full-attention publisher's opaque block IDs to shared page keys.

use super::records::Residency;
use super::*;
use sha2::{compress256, digest::generic_array::GenericArray};
use smallvec::SmallVec;

pub(super) type FullBlock = (Arc<str>, SequenceHash); // wire ID, block-local hash

pub(super) fn prefix_key(parent: &SequenceHash, local: &SequenceHash) -> SequenceHash {
    // This internal key is recalculated on replay. Its fixed-length (parent,
    // local) input uses one SHA-256 compression with a domain-specific IV.
    // IV = SHA-256("vllm-router/full-attention/page-key/compress-v1"), in big-endian words.
    const IV: [u32; 8] = [
        0x22fabb15, 0xed95270a, 0x0af599b0, 0x53b1bf79, 0xbd088b37, 0x93e2873f, 0x7a7167b7,
        0xcf6720a0,
    ];
    let mut state = IV;
    let mut message = GenericArray::default();
    message[..32].copy_from_slice(parent);
    message[32..].copy_from_slice(local);
    compress256(&mut state, &[message]);
    let mut key = [0; 32];
    for (bytes, word) in key.chunks_exact_mut(4).zip(state) {
        bytes.copy_from_slice(&word.to_be_bytes());
    }
    key
}

#[derive(Default)]
pub(super) struct EventAddresses {
    groups: HashMap<u32, GroupAddresses, Fast>,
}

#[derive(Default)]
struct GroupAddresses {
    blocks: HashMap<Arc<str>, Address, Fast>,
    waiting: HashMap<Arc<str>, HashSet<Arc<str>, Fast>, Fast>,
    // Device facts count their contributing wire IDs directly in the page table.
    aliases: HashMap<SequenceHash, Aliases, Fast>,
}

struct Address {
    parent: Option<Arc<str>>,
    key: AddressKey,
    children: usize,
    owners: SmallVec<[(Arc<ResidencyOwner>, Residency); 1]>,
}

#[derive(Clone, Copy)]
enum AddressKey {
    Pending(SequenceHash),
    Resolved(SequenceHash),
}

impl AddressKey {
    fn resolved(self) -> Option<SequenceHash> {
        match self {
            Self::Resolved(key) => Some(key),
            Self::Pending(_) => None,
        }
    }
}

impl Address {
    fn has_non_device(&self) -> bool {
        self.owners
            .iter()
            .any(|(_, observation)| observation.tiers.0 & !TierBits::bit(StorageTier::Device) != 0)
    }

    fn owner_position(&self, owner: &ResidencyOwner) -> Option<usize> {
        self.owners
            .iter()
            .position(|(candidate, _)| candidate.as_ref() == owner)
    }

    fn residency(&self, owner: &ResidencyOwner) -> Option<&Residency> {
        self.owner_position(owner)
            .map(|position| &self.owners[position].1)
    }
}

// Multiple producer IDs for the same normalized page are uncommon.
enum Aliases {
    One(Arc<str>),
    Many(Vec<Arc<str>>),
}

impl Aliases {
    fn as_slice(&self) -> &[Arc<str>] {
        match self {
            Self::One(wire) => std::slice::from_ref(wire),
            Self::Many(wires) => wires,
        }
    }

    fn insert(&mut self, wire: Arc<str>) {
        match self {
            Self::One(first) => *self = Self::Many(vec![first.clone(), wire]),
            Self::Many(wires) => wires.push(wire),
        }
    }

    fn remove(&mut self, wire: &Arc<str>) -> bool {
        match self {
            Self::One(only) => {
                debug_assert_eq!(only, wire);
                true
            }
            Self::Many(wires) => {
                let position = wires.iter().position(|member| member == wire).unwrap();
                wires.swap_remove(position);
                if wires.len() == 1 {
                    *self = Self::One(wires.pop().unwrap());
                }
                false
            }
        }
    }
}

fn record_alias(
    aliases: &mut HashMap<SequenceHash, Aliases, Fast>,
    key: SequenceHash,
    wire: &Arc<str>,
) {
    use std::collections::hash_map::Entry;
    match aliases.entry(key) {
        Entry::Vacant(entry) => {
            entry.insert(Aliases::One(wire.clone()));
        }
        Entry::Occupied(mut entry) => entry.get_mut().insert(wire.clone()),
    }
}

fn forget_alias(
    aliases: &mut HashMap<SequenceHash, Aliases, Fast>,
    key: SequenceHash,
    wire: &Arc<str>,
) {
    if let std::collections::hash_map::Entry::Occupied(mut entry) = aliases.entry(key) {
        if entry.get_mut().remove(wire) {
            entry.remove();
        }
    }
}

impl IndexBatch<'_> {
    pub(crate) fn store_full_pages(
        &mut self,
        group: u32,
        owner: &ResidencyOwner,
        tier: StorageTier,
        parent: Option<&Arc<str>>,
        blocks: &[FullBlock],
        incremental: bool,
    ) -> Result<(), PageWriteError> {
        if blocks.is_empty() {
            return Ok(());
        }
        let index = self.index;
        self.with_records(&owner.target(), |records| {
            let addresses = records.addresses.groups.entry(group).or_default();
            let new_wires = addresses.validate(owner, parent, blocks, incremental)?;
            addresses.blocks.reserve(new_wires);
            let device = tier == StorageTier::Device;
            if !device && addresses.waiting.is_empty() {
                addresses.aliases.reserve(new_wires);
            }
            let mut wire_owner = None;
            let mut parent = parent;
            let mut parent_key = match parent {
                None => Some([0; 32]),
                Some(wire) => addresses.blocks.get(wire).and_then(|a| a.key.resolved()),
            };
            for (wire, local) in blocks {
                use std::collections::hash_map::Entry;
                let (address, new) = match addresses.blocks.entry(wire.clone()) {
                    Entry::Occupied(entry) => (entry.into_mut(), false),
                    Entry::Vacant(entry) => {
                        let key = parent_key.map_or_else(
                            || AddressKey::Pending(*local),
                            |parent| AddressKey::Resolved(prefix_key(&parent, local)),
                        );
                        let address = entry.insert(Address {
                            parent: parent.cloned(),
                            key,
                            children: addresses.waiting.get(wire).map_or(0, HashSet::len),
                            owners: SmallVec::new(),
                        });
                        (address, true)
                    }
                };
                let register_alias = !device && !address.has_non_device();
                let position = address.owner_position(owner).unwrap_or_else(|| {
                    let position = address.owners.len();
                    // Reuse the identity already held by the page table when available.
                    let shared = wire_owner.get_or_insert_with(|| {
                        records
                            .hash_keys
                            .get_key_value(owner)
                            .map_or_else(|| Arc::new(owner.clone()), |(owner, _)| owner.clone())
                    });
                    address.owners.push((shared.clone(), Residency::default()));
                    position
                });
                let residency = &mut address.owners[position].1;
                let already_present = residency.contains(tier);
                if !residency.observe(tier, incremental) {
                    unreachable!("full-attention batch was validated");
                }
                let key = address.key.resolved();
                let has_waiters = new && address.children != 0;
                if new {
                    if let Some(previous) = parent {
                        if let Some(parent) = addresses.blocks.get_mut(previous) {
                            parent.children += 1;
                        }
                    }
                    if key.is_none() {
                        addresses
                            .waiting
                            .entry(parent.expect("root has a known key").clone())
                            .or_default()
                            .insert(wire.clone());
                    }
                }
                if let Some(key) = key {
                    if register_alias {
                        record_alias(&mut addresses.aliases, key, wire);
                    }
                    if !already_present {
                        index.hash_pages.store(
                            &mut records.hash_keys,
                            owner,
                            group,
                            tier,
                            None,
                            &[key],
                            device,
                            usize::MAX,
                        )?;
                    }
                    if has_waiters {
                        addresses.resolve_children(
                            wire,
                            key,
                            group,
                            &index.hash_pages,
                            &mut records.hash_keys,
                        )?;
                    }
                }
                parent = Some(wire);
                parent_key = key;
            }
            Ok(())
        })
    }

    pub(crate) fn remove_full_pages(
        &mut self,
        group: u32,
        owner: &ResidencyOwner,
        tier: StorageTier,
        wires: &[Arc<str>],
        incremental: bool,
    ) {
        let index = self.index;
        self.with_records(&owner.target(), |records| {
            let Some(addresses) = records.addresses.groups.get_mut(&group) else {
                return;
            };
            let device = tier == StorageTier::Device;
            for wire in wires {
                let Some(address) = addresses.blocks.get_mut(wire) else {
                    continue;
                };
                let Some(position) = address.owner_position(owner) else {
                    continue;
                };
                let residency = &mut address.owners[position].1;
                let removed = residency.withdraw(tier, incremental);
                let key = address.key.resolved();
                if residency.tiers.0 == 0 {
                    address.owners.swap_remove(position);
                }
                if removed {
                    if let Some(key) = key {
                        if !device && !address.has_non_device() {
                            forget_alias(&mut addresses.aliases, key, wire);
                        }
                        if device || !addresses.has_tier(&key, owner, tier) {
                            index.hash_pages.remove(
                                &mut records.hash_keys,
                                owner,
                                group,
                                tier,
                                &[key],
                                device,
                            );
                        }
                    }
                }
                addresses.prune(wire.clone());
            }
            if addresses.blocks.is_empty() {
                records.addresses.groups.remove(&group);
            }
        });
    }
}

impl GroupAddresses {
    fn validate(
        &self,
        owner: &ResidencyOwner,
        parent: Option<&Arc<str>>,
        blocks: &[FullBlock],
        incremental: bool,
    ) -> Result<usize, PageWriteError> {
        let mut seen = HashSet::with_capacity_and_hasher(blocks.len(), Fast::default());
        let mut previous = parent;
        let mut new_wires = 0;
        for (wire, local) in blocks {
            if !seen.insert(wire) {
                return Err(PageWriteError::ConflictingHash);
            }
            if let Some(known) = self.blocks.get(wire) {
                let same_hash = match known.key {
                    AddressKey::Pending(pending) => &pending == local,
                    AddressKey::Resolved(expected) => {
                        let parent_key = match previous {
                            None => Some([0; 32]),
                            Some(parent) => self.blocks.get(parent).and_then(|a| a.key.resolved()),
                        };
                        parent_key.is_some_and(|parent| prefix_key(&parent, local) == expected)
                    }
                };
                if !same_hash || known.parent.as_ref() != previous {
                    return Err(PageWriteError::ConflictingHash);
                }
                if known
                    .residency(owner)
                    .is_some_and(|r| !r.can_observe(incremental))
                {
                    return Err(PageWriteError::Capacity);
                }
            } else {
                new_wires += 1;
            }
            previous = Some(wire);
        }
        // An out-of-order batch must not complete a cycle of unresolved aliases.
        let mut cursor = parent;
        while let Some(wire) = cursor {
            if seen.contains(wire) {
                return Err(PageWriteError::ConflictingHash);
            }
            cursor = self
                .blocks
                .get(wire)
                .filter(|address| matches!(address.key, AddressKey::Pending(_)))
                .and_then(|address| address.parent.as_ref());
        }
        Ok(new_wires)
    }

    fn resolve_children(
        &mut self,
        wire: &Arc<str>,
        parent_key: SequenceHash,
        group: u32,
        pages: &PageTable,
        active: &mut ActivePages,
    ) -> Result<(), PageWriteError> {
        let Some(children) = self.waiting.remove(wire) else {
            return Ok(());
        };
        let mut children = children.into_iter();
        let mut next = children.next().map(|child| (child, parent_key));
        let mut work: Vec<_> = children.map(|child| (child, parent_key)).collect();
        while let Some((wire, parent_key)) = next {
            let address = self.blocks.get_mut(&wire).unwrap();
            let AddressKey::Pending(local) = address.key else {
                unreachable!("resolved child in pending queue")
            };
            let key = prefix_key(&parent_key, &local);
            address.key = AddressKey::Resolved(key);
            if address.has_non_device() {
                record_alias(&mut self.aliases, key, &wire);
            }
            for (owner, observation) in &address.owners {
                for tier in TIERS.into_iter().filter(|tier| observation.contains(*tier)) {
                    pages.store(
                        active,
                        owner,
                        group,
                        tier,
                        None,
                        &[key],
                        tier == StorageTier::Device,
                        usize::MAX,
                    )?;
                }
            }
            if let Some(children) = self.waiting.remove(&wire) {
                let mut children = children.into_iter();
                next = children.next().map(|child| (child, key));
                work.extend(children.map(|child| (child, key)));
            } else {
                next = work.pop();
            }
        }
        Ok(())
    }

    fn has_tier(&self, key: &SequenceHash, owner: &ResidencyOwner, tier: StorageTier) -> bool {
        self.aliases.get(key).is_some_and(|wires| {
            wires.as_slice().iter().any(|wire| {
                self.blocks[wire]
                    .residency(owner)
                    .is_some_and(|r| r.contains(tier))
            })
        })
    }

    fn prune(&mut self, mut wire: Arc<str>) {
        while self
            .blocks
            .get(&wire)
            .is_some_and(|a| a.owners.is_empty() && a.children == 0)
        {
            let address = self.blocks.remove(&wire).unwrap();
            let Some(parent) = address.parent else {
                break;
            };
            if matches!(address.key, AddressKey::Pending(_)) {
                if let Some(children) = self.waiting.get_mut(&parent) {
                    children.remove(&wire);
                    if children.is_empty() {
                        self.waiting.remove(&parent);
                    }
                }
            }
            let Some(previous) = self.blocks.get_mut(&parent) else {
                break;
            };
            previous.children -= 1;
            wire = parent;
        }
    }
}

impl EventAddresses {
    pub(super) fn clear(&mut self, remove: impl Fn(&ResidencyOwner) -> bool) {
        self.groups.retain(|_, addresses| {
            let mut unused = Vec::new();
            for (wire, address) in &mut addresses.blocks {
                let known_alias = !addresses.aliases.is_empty() && address.has_non_device();
                address.owners.retain(|(owner, _)| !remove(owner));
                if known_alias && !address.has_non_device() {
                    if let Some(key) = address.key.resolved() {
                        forget_alias(&mut addresses.aliases, key, wire);
                    }
                }
                if address.owners.is_empty() && address.children == 0 {
                    unused.push(wire.clone());
                }
            }
            for wire in unused {
                addresses.prune(wire);
            }
            !addresses.blocks.is_empty()
        });
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn churn_releases_wire_ids_while_a_shared_parent_stays_live() {
        let index = KvBlockIndexer::new();
        let owner = ResidencyOwner::Worker {
            source: Arc::from("worker"),
            dp_rank: 0,
            incarnation: 1,
        };
        let target = owner.target();
        let parent: Arc<str> = Arc::from("root");
        let tier = StorageTier::Device;
        index.store(0, owner.clone(), tier, None, &[(parent.clone(), [0; 32])]);
        index.with_source_batch(&target, |tx| {
            for i in 1..=256u32 {
                let wire: Arc<str> = Arc::from(format!("b{i}"));
                let mut local = [0; 32];
                local[..4].copy_from_slice(&i.to_le_bytes());
                tx.store_full_pages(
                    0,
                    &owner,
                    tier,
                    Some(&parent),
                    &[(wire.clone(), local)],
                    true,
                )
                .unwrap();
                tx.remove_full_pages(0, &owner, tier, &[wire], true);
            }
            tx.with_records(&target, |records| {
                let addresses = &records.addresses.groups[&0];
                assert_eq!(addresses.blocks.len(), 1);
                assert!(addresses.aliases.is_empty());
                assert!(addresses.waiting.is_empty());
            });
        });
        index.remove(0, &owner, tier, &[parent]);
        assert!(index
            .source_lock(&target)
            .access
            .read()
            .records
            .addresses
            .groups
            .is_empty());
    }
}
