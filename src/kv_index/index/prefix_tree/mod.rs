//! Compressed prefix-tree nodes and source-scoped sequence aliases.

mod insert;
mod query;
mod reclaim;
mod remove;

#[cfg(all(test, feature = "cpu", feature = "ready"))]
pub(super) use query::ClaimSpan;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};

use super::{Fast, IncomingBlock, NodeRef, OwnerTier};
use crate::kv_index::RoutableTarget;

/// A source normally publishes one identity for a tree position.
pub(super) enum AliasHashes {
    One(Arc<str>),
    // Box the rare multi-alias case to keep each single-hash map entry small.
    #[allow(clippy::box_collection)]
    Many(Box<HashSet<Arc<str>, Fast>>),
}

impl AliasHashes {
    fn insert(&mut self, seq: Arc<str>) {
        match self {
            Self::One(known) if *known == seq => {}
            Self::One(known) => {
                let mut hashes = HashSet::default();
                hashes.insert(known.clone());
                hashes.insert(seq);
                *self = Self::Many(Box::new(hashes));
            }
            Self::Many(hashes) => {
                hashes.insert(seq);
            }
        }
    }

    /// Return true when the caller must remove this source entry.
    fn remove(&mut self, seq: &Arc<str>) -> bool {
        match self {
            Self::One(known) => known == seq,
            Self::Many(hashes) => {
                hashes.remove(seq);
                if hashes.len() == 1 {
                    *self = Self::One(hashes.iter().next().unwrap().clone());
                }
                false
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Many(hashes) => hashes.len(),
        }
    }

    fn iter(&self) -> impl Iterator<Item = &Arc<str>> {
        let one = match self {
            Self::One(hash) => Some(hash),
            _ => None,
        };
        let many = match self {
            Self::Many(hashes) => Some(hashes.iter()),
            _ => None,
        };
        one.into_iter().chain(many.into_iter().flatten())
    }

    fn into_one(self) -> Arc<str> {
        match self {
            Self::One(hash) => hash,
            Self::Many(hashes) => (*hashes).into_iter().next().unwrap(),
        }
    }
}

/// Sequence aliases are publisher-scoped: identical opaque hashes
/// at different sources need not describe the same path.
type AliasSources = HashSet<Arc<RoutableTarget>, Fast>;

pub(super) enum Aliases {
    Empty,
    One(Arc<RoutableTarget>, Arc<str>),
    // Equal sequence identities share the source roster across adjacent pages.
    Shared(Arc<AliasSources>, Arc<str>),
    Many(HashMap<Arc<RoutableTarget>, AliasHashes, Fast>),
}

impl Aliases {
    pub(super) fn count(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::One(..) => 1,
            Self::Shared(sources, _) => sources.len(),
            Self::Many(entries) => entries.values().map(AliasHashes::len).sum(),
        }
    }

    pub(super) fn remove_alias(&mut self, target: &RoutableTarget, seq: &Arc<str>) {
        match self {
            Self::One(_, hash) | Self::Shared(_, hash) if hash == seq => {
                self.remove_target(target);
            }
            Self::Many(entries) => {
                if entries
                    .get_mut(target)
                    .is_some_and(|hashes| hashes.remove(seq))
                {
                    entries.remove(target);
                }
                self.compact_many();
            }
            _ => {}
        }
    }
    pub(super) fn insert(&mut self, target: Arc<RoutableTarget>, seq: Arc<str>) {
        match self {
            Self::Empty => *self = Self::One(target, seq),
            Self::One(existing_target, existing_seq)
                if *existing_target == target && *existing_seq == seq => {}
            Self::One(existing, hash) if *hash == seq => {
                let mut sources = AliasSources::default();
                sources.insert(existing.clone());
                sources.insert(target);
                *self = Self::Shared(Arc::new(sources), seq);
            }
            Self::Shared(sources, hash) if *hash == seq => {
                if !sources.contains(&target) {
                    Arc::make_mut(sources).insert(target);
                }
            }
            Self::Shared(sources, hash) => {
                let mut entries: HashMap<_, _, Fast> = sources
                    .iter()
                    .map(|source| (source.clone(), AliasHashes::One(hash.clone())))
                    .collect();
                entries
                    .entry(target)
                    .or_insert_with(|| AliasHashes::One(seq.clone()))
                    .insert(seq);
                *self = Self::Many(entries);
            }
            Self::One(_, _) => {
                let Self::One(old_target, old_seq) = std::mem::replace(self, Self::Empty) else {
                    unreachable!()
                };
                let mut entries: HashMap<Arc<RoutableTarget>, AliasHashes, Fast> =
                    HashMap::default();
                entries.insert(old_target, AliasHashes::One(old_seq));
                entries
                    .entry(target)
                    .or_insert_with(|| AliasHashes::One(seq.clone()))
                    .insert(seq);
                *self = Self::Many(entries);
            }
            Self::Many(entries) => {
                entries
                    .entry(target)
                    .or_insert_with(|| AliasHashes::One(seq.clone()))
                    .insert(seq);
            }
        }
    }

    pub(super) fn has(&self, target: &RoutableTarget, seq: &Arc<str>) -> bool {
        self.any_for(target, |candidate| candidate == seq)
    }

    pub(super) fn any_for(
        &self,
        target: &RoutableTarget,
        f: impl FnMut(&Arc<str>) -> bool,
    ) -> bool {
        match self {
            Self::Empty => false,
            Self::One(owner, seq) => owner.as_ref() == target && std::iter::once(seq).any(f),
            Self::Shared(sources, seq) => sources.contains(target) && std::iter::once(seq).any(f),
            Self::Many(entries) => entries.get(target).is_some_and(|seqs| seqs.iter().any(f)),
        }
    }

    pub(super) fn for_each(&self, mut f: impl FnMut(&RoutableTarget, &Arc<str>)) {
        match self {
            Self::Empty => {}
            Self::One(target, seq) => f(target, seq),
            Self::Shared(sources, seq) => {
                for target in sources.iter() {
                    f(target, seq);
                }
            }
            Self::Many(entries) => {
                for (target, seqs) in entries.iter() {
                    for seq in seqs.iter() {
                        f(target, seq);
                    }
                }
            }
        }
    }

    pub(super) fn remove_target(&mut self, target: &RoutableTarget) {
        match self {
            Self::One(owner, _) if owner.as_ref() == target => *self = Self::Empty,
            Self::Shared(sources, _) if sources.contains(target) => {
                Arc::make_mut(sources).remove(target);
                self.compact_shared();
            }
            Self::Many(entries) => {
                entries.remove(target);
                self.compact_many();
            }
            _ => {}
        }
    }

    fn compact_shared(&mut self) {
        if let Self::Shared(sources, seq) = self {
            if sources.len() == 1 {
                *self = Self::One(sources.iter().next().unwrap().clone(), seq.clone());
            } else if sources.is_empty() {
                *self = Self::Empty;
            }
        }
    }

    fn compact_many(&mut self) {
        if let Self::Many(entries) = self {
            if entries.is_empty() {
                *self = Self::Empty;
            } else if entries.len() == 1
                && entries
                    .values()
                    .next()
                    .is_some_and(|hashes| hashes.len() == 1)
            {
                let (owner, hashes) = std::mem::take(entries).into_iter().next().unwrap();
                *self = Self::One(owner, hashes.into_one());
            }
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        match self {
            Self::Empty => true,
            Self::One(_, _) | Self::Shared(..) => false,
            Self::Many(entries) => entries.is_empty(),
        }
    }
}

/// Reuse an immutable source roster across adjacent pages with the same publishers.
pub(super) struct AliasWriter<'a> {
    target: &'a Arc<RoutableTarget>,
    previous: Option<(Option<Arc<AliasSources>>, Arc<AliasSources>)>,
}

impl<'a> AliasWriter<'a> {
    pub(super) fn new(target: &'a Arc<RoutableTarget>) -> Self {
        Self {
            target,
            previous: None,
        }
    }

    pub(super) fn add(&mut self, aliases: &mut Aliases, seq: &Arc<str>) -> bool {
        if aliases.has(self.target, seq) {
            return false;
        }
        if let Some((before, after)) = &self.previous {
            let same = match &*aliases {
                Aliases::One(target, known) if known == seq => {
                    after.len() == 2 && after.contains(target)
                }
                Aliases::Shared(sources, known) if known == seq => {
                    before
                        .as_ref()
                        .is_some_and(|before| Arc::ptr_eq(before, sources))
                        || (sources.len() + 1 == after.len()
                            && sources.iter().all(|target| after.contains(target)))
                }
                _ => false,
            };
            if same {
                *aliases = Aliases::Shared(after.clone(), seq.clone());
                return true;
            }
        }
        // Keep a pointer for already-shared rosters; a unique roster can be
        // mutated in place. Exact membership reunites pages detached by removal.
        let before = match &*aliases {
            Aliases::Shared(sources, known) if known == seq && Arc::strong_count(sources) > 1 => {
                Some(sources.clone())
            }
            _ => None,
        };
        aliases.insert(self.target.clone(), seq.clone());
        self.previous = match aliases {
            Aliases::Shared(after, _) => Some((before, after.clone())),
            _ => None,
        };
        true
    }
}

/// Reuse the same source removal while reclaiming a batch of pages.
enum RemovedAliasRoster {
    One(Arc<RoutableTarget>),
    Shared(Arc<AliasSources>),
}

pub(super) struct AliasRemover<'a> {
    target: &'a RoutableTarget,
    previous: Option<(Arc<AliasSources>, RemovedAliasRoster)>,
}

impl<'a> AliasRemover<'a> {
    pub(super) fn new(target: &'a RoutableTarget) -> Self {
        Self {
            target,
            previous: None,
        }
    }

    pub(super) fn remove(&mut self, aliases: &mut Aliases, seq: Option<&Arc<str>>) {
        let before = match aliases {
            Aliases::Shared(sources, known)
                if seq.is_none_or(|seq| known == seq)
                    && sources.contains(self.target)
                    && Arc::strong_count(sources) > 1 =>
            {
                Some(sources.clone())
            }
            _ => None,
        };
        if let (Some(before), Some((previous, after))) = (&before, &self.previous) {
            if Arc::ptr_eq(before, previous) {
                let Aliases::Shared(_, known) = aliases else {
                    unreachable!()
                };
                *aliases = match after {
                    RemovedAliasRoster::One(target) => Aliases::One(target.clone(), known.clone()),
                    RemovedAliasRoster::Shared(sources) => {
                        Aliases::Shared(sources.clone(), known.clone())
                    }
                };
                return;
            }
        }
        if let Some(seq) = seq {
            aliases.remove_alias(self.target, seq);
        } else {
            aliases.remove_target(self.target);
        }
        self.previous = before.and_then(|before| match aliases {
            Aliases::One(target, _) => Some((before, RemovedAliasRoster::One(target.clone()))),
            Aliases::Shared(sources, _) => {
                Some((before, RemovedAliasRoster::Shared(sources.clone())))
            }
            _ => None,
        });
    }
}

pub(super) type EdgeBlock = (Aliases, Arc<str>); // (source-scoped aliases, local_hash)

/// Length of the common prefix between an edge and the incoming blocks,
/// compared by local hash only.
pub(super) fn edge_common_prefix(edge: &[EdgeBlock], incoming: &[IncomingBlock]) -> usize {
    edge.iter()
        .zip(incoming.iter())
        .take_while(|(e, b)| e.1 == b.1)
        .count()
}

/// One node in the compressed radix tree. An edge is a run of one or more
/// blocks; the path from root to a node encodes its chain position, so
/// identical tokens at different positions land on different nodes. Each edge
/// block carries its own claim set — residency is per block, so a `remove` of
/// one block never disturbs the claims on its siblings.
pub(super) struct TreeNode {
    pub(super) parent: Weak<parking_lot::RwLock<TreeNode>>,
    pub(super) alias_count: usize,
    pub(super) prune_queued: bool,
    /// Edge blocks with historical identities; empty on the root sentinel.
    /// Ancestors remain while live descendants depend on them.
    pub(super) edge: Vec<EdgeBlock>,
    /// Per-block residency claims, parallel to `edge`; `claims[i]` owns
    /// `edge[i]`. Empty on the root sentinel.
    pub(super) claims: Vec<Arc<Vec<OwnerTier>>>,
    /// Children keyed by the first local hash of each child's edge.
    pub(super) children: HashMap<Arc<str>, NodeRef, Fast>,
}

impl TreeNode {
    pub(super) fn root() -> Self {
        TreeNode {
            parent: Weak::new(),
            alias_count: 0,
            prune_queued: false,
            edge: Vec::new(),
            claims: Vec::new(),
            children: HashMap::default(),
        }
    }
    pub(super) fn with_edge(
        blocks: &[IncomingBlock],
        target: &Arc<RoutableTarget>,
        key: Option<&OwnerTier>,
    ) -> Self {
        let initial_claims = Arc::new(key.cloned().into_iter().collect());
        let claims = vec![initial_claims; blocks.len()];
        TreeNode {
            parent: Weak::new(),
            alias_count: blocks.len(),
            prune_queued: false,
            edge: blocks
                .iter()
                .map(|(seq, local)| (Aliases::One(target.clone(), seq.clone()), local.clone()))
                .collect(),
            claims,
            children: HashMap::default(),
        }
    }
}

pub(super) fn unique_alias_position(
    node: &NodeRef,
    target: &RoutableTarget,
    seq: &Arc<str>,
) -> Option<usize> {
    let node = node.read();
    let mut positions = node
        .edge
        .iter()
        .enumerate()
        .filter_map(|(i, (aliases, _))| aliases.has(target, seq).then_some(i));
    let position = positions.next()?;
    positions.next().is_none().then_some(position)
}
