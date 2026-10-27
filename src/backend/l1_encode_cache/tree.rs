use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::hash::{BuildHasher, Hash, Hasher};
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};

use super::L1HashStrategy;

pub(super) const NO_NODE: usize = usize::MAX;

/// Bytes the allocator hands out for an `n`-byte request: 16-byte chunks
/// with an 8-byte header and a 32-byte minimum, as in glibc malloc.
pub(super) const fn heap_bytes(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let chunk = (n + 8 + 15) & !15;
    if chunk < 32 {
        32
    } else {
        chunk
    }
}

/// Heap bytes a node owns: the boxed node, its edge text, its ID slice with
/// the `Arc` counts, its marks, and a namespace salt.
pub(super) fn node_bytes(text: usize, tokens: usize, marks: usize, salt: Option<usize>) -> usize {
    heap_bytes(size_of::<Node>())
        + heap_bytes(text)
        + heap_bytes(2 * size_of::<usize>() + tokens * size_of::<u32>())
        + heap_bytes(marks * size_of::<(u32, u32)>())
        + salt.map_or(0, |len| heap_bytes(2 * size_of::<usize>() + len))
}

/// Length of the common prefix of `a` and `b`.
pub(super) fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    const CHUNK: usize = 64;
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + CHUNK <= n && a[i..i + CHUNK] == b[i..i + CHUNK] {
        i += CHUNK;
    }
    i + a[i..n]
        .iter()
        .zip(&b[i..n])
        .take_while(|(x, y)| x == y)
        .count()
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub(super) struct Namespace {
    pub(super) instance: u64,
    /// Shared by the node and the roots map, so the bytes are counted once.
    pub(super) salt: Option<Arc<str>>,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub(super) struct SegmentKey {
    pub(super) len: usize,
    pub(super) fingerprint: u64,
}

/// Seeded per process, so prompts cannot be crafted to collide.
pub(super) static SEGMENT_HASH: LazyLock<foldhash::quality::RandomState> =
    LazyLock::new(Default::default);

pub(super) fn segment_key(segment: &str, strategy: L1HashStrategy) -> SegmentKey {
    let bytes = segment.as_bytes();
    let mut hasher = SEGMENT_HASH.build_hasher();
    match strategy {
        L1HashStrategy::Full => bytes.hash(&mut hasher),
        L1HashStrategy::FirstLast64 if bytes.len() <= 128 => bytes.hash(&mut hasher),
        L1HashStrategy::FirstLast64 => {
            bytes[..64].hash(&mut hasher);
            bytes[bytes.len() - 64..].hash(&mut hasher);
        }
    }
    SegmentKey {
        len: bytes.len(),
        fingerprint: hasher.finish(),
    }
}

/// A parent and the key of a child's first segment.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub(super) struct EdgeKey {
    pub(super) parent: usize,
    pub(super) segment: SegmentKey,
}

pub(super) struct Node {
    pub(super) parent: usize,
    /// Next child of `parent` whose first segment has the same key.
    pub(super) next_same_key: usize,
    pub(super) child_count: usize,
    /// The last child added. It is the only child while `child_count` is 1
    /// and it still names a live node whose parent is this one.
    pub(super) child: usize,
    /// Edge text; it ends at a boundary (empty for namespace roots).
    pub(super) text: Box<str>,
    pub(super) ids: Arc<[u32]>,
    /// Boundaries strictly inside the edge as (byte, token) offsets.
    pub(super) marks: Box<[(u32, u32)]>,
    /// Tick of the last insert before this node was last used.
    pub(super) last_used: AtomicU64,
    /// Set on namespace roots.
    pub(super) namespace: Option<Namespace>,
    /// The namespace root this node belongs to (itself for a root).
    pub(super) root: usize,
    /// On roots: the tree version of the namespace's last structural change.
    pub(super) changed: u64,
}

impl Node {
    pub(super) fn new(
        parent: usize,
        text: Box<str>,
        ids: Arc<[u32]>,
        marks: Box<[(u32, u32)]>,
        tick: u64,
        namespace: Option<Namespace>,
    ) -> Self {
        Self {
            parent,
            next_same_key: NO_NODE,
            child_count: 0,
            child: NO_NODE,
            text,
            ids,
            marks,
            last_used: AtomicU64::new(tick),
            namespace,
            root: NO_NODE,
            changed: 0,
        }
    }

    pub(super) fn first_segment(&self) -> &str {
        match self.marks.first() {
            Some(&(byte, _)) => &self.text[..byte as usize],
            None => &self.text,
        }
    }

    pub(super) fn is_leaf(&self) -> bool {
        self.namespace.is_none() && self.child_count == 0
    }

    pub(super) fn bytes(&self) -> usize {
        let salt = self.namespace.as_ref().and_then(|ns| ns.salt.as_ref());
        node_bytes(
            self.text.len(),
            self.ids.len(),
            self.marks.len(),
            salt.map(|salt| salt.len()),
        )
    }

    /// Skips the write when the node is already current, so hot prefixes
    /// are not written by every lookup.
    pub(super) fn touch(&self, tick: u64) {
        if self.last_used.load(Ordering::Relaxed) < tick {
            self.last_used.fetch_max(tick, Ordering::Relaxed);
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Step {
    /// The whole edge matched.
    Full(usize),
    /// The match ends at `marks[mark]` of the node.
    Partial(usize, usize),
}

#[derive(Default)]
pub(super) struct Hit {
    /// The namespace root, if it existed.
    pub(super) root: Option<usize>,
    /// The root's `changed` the steps are valid for.
    pub(super) version: u64,
    pub(super) steps: Vec<Step>,
    pub(super) bytes: usize,
    pub(super) tokens: usize,
    pub(super) pieces: Vec<(Arc<[u32]>, usize)>,
}

pub(super) struct Tree {
    pub(super) nodes: Vec<Option<Box<Node>>>,
    pub(super) free: Vec<usize>,
    pub(super) roots: hashbrown::HashMap<Namespace, usize>,
    /// First child per (parent, first-segment key); collisions chain
    /// through `Node::next_same_key`.
    pub(super) edges: hashbrown::HashMap<EdgeKey, usize>,
    /// Leaves by last use. Lazy: an entry may name a node that is gone, is
    /// no longer a leaf, or was used since; every leaf has an entry.
    pub(super) leaves: BinaryHeap<Reverse<(u64, usize)>>,
    pub(super) live: usize,
    /// Heap bytes owned by live nodes.
    pub(super) content_bytes: usize,
    /// Bumped by every structural change. Roots record it per namespace, so
    /// a lookup's steps stay reusable while only other namespaces change.
    pub(super) version: u64,
    pub(super) strategy: L1HashStrategy,
}

impl Tree {
    pub(super) fn new(strategy: L1HashStrategy) -> Self {
        Self {
            nodes: Vec::new(),
            free: Vec::new(),
            roots: hashbrown::HashMap::new(),
            edges: hashbrown::HashMap::new(),
            leaves: BinaryHeap::new(),
            live: 0,
            content_bytes: 0,
            version: 0,
            strategy,
        }
    }

    /// Estimated heap bytes: node contents plus the capacity of every
    /// container, so slack left behind by evictions is charged too.
    pub(super) fn bytes(&self) -> usize {
        self.content_bytes
            + self.nodes.capacity() * size_of::<Option<Box<Node>>>()
            + self.free.capacity() * size_of::<usize>()
            + self.leaves.capacity() * size_of::<Reverse<(u64, usize)>>()
            + self.edges.allocation_size()
            + self.roots.allocation_size()
    }

    pub(super) fn key(&self, segment: &str) -> SegmentKey {
        segment_key(segment, self.strategy)
    }

    pub(super) fn node(&self, id: usize) -> &Node {
        self.nodes[id].as_deref().expect("live L1 node")
    }

    pub(super) fn node_mut(&mut self, id: usize) -> &mut Node {
        self.nodes[id].as_deref_mut().expect("live L1 node")
    }

    pub(super) fn alloc(&mut self, mut node: Node) -> usize {
        self.live += 1;
        self.content_bytes += node.bytes();
        let id = self.free.pop().unwrap_or(self.nodes.len());
        node.root = if node.namespace.is_some() {
            id
        } else {
            self.node(node.parent).root
        };
        let root = node.root;
        let node = Some(Box::new(node));
        if id == self.nodes.len() {
            self.nodes.push(node);
        } else {
            self.nodes[id] = node;
        }
        self.changed(root);
        id
    }

    pub(super) fn release(&mut self, id: usize) -> Box<Node> {
        let node = self.nodes[id].take().expect("live L1 node");
        self.free.push(id);
        self.live -= 1;
        self.content_bytes -= node.bytes();
        if node.root != id {
            self.changed(node.root);
        }
        node
    }

    fn changed(&mut self, root: usize) {
        self.version += 1;
        let version = self.version;
        self.node_mut(root).changed = version;
    }

    pub(super) fn edge(&self, parent: usize, child: usize) -> EdgeKey {
        EdgeKey {
            parent,
            segment: self.key(self.node(child).first_segment()),
        }
    }

    pub(super) fn child(&self, parent: usize, segment: &str) -> Option<usize> {
        let key = EdgeKey {
            parent,
            segment: self.key(segment),
        };
        let mut id = *self.edges.get(&key)?;
        while self.node(id).first_segment() != segment {
            id = self.node(id).next_same_key;
            if id == NO_NODE {
                return None;
            }
        }
        Some(id)
    }

    pub(super) fn add_child(&mut self, parent: usize, child: usize) {
        let key = self.edge(parent, child);
        let next = self.edges.insert(key, child).unwrap_or(NO_NODE);
        self.node_mut(child).next_same_key = next;
        let parent_node = self.node_mut(parent);
        parent_node.child_count += 1;
        parent_node.child = child;
    }

    fn only_child(&self, parent: usize) -> Option<usize> {
        let node = self.node(parent);
        if node.child_count != 1 {
            return None;
        }
        let child = self.nodes.get(node.child)?.as_deref()?;
        (child.parent == parent).then_some(node.child)
    }

    pub(super) fn unlink(&mut self, parent: usize, child: usize) {
        let key = self.edge(parent, child);
        let next = self.node(child).next_same_key;
        if self.edges[&key] == child {
            if next == NO_NODE {
                self.edges.remove(&key);
            } else {
                self.edges.insert(key, next);
            }
        } else {
            let mut prev = self.edges[&key];
            while self.node(prev).next_same_key != child {
                prev = self.node(prev).next_same_key;
            }
            self.node_mut(prev).next_same_key = next;
        }
        self.node_mut(child).next_same_key = NO_NODE;
        self.node_mut(parent).child_count -= 1;
    }

    /// Follow `text` from `root`, comparing edge text directly.
    /// `first_cut(offset)` returns the end of the first segment of
    /// `text[offset..]` and is only asked at cut positions, to pick among
    /// several children. An only child needs no scan: siblings' first
    /// segments are prefix-free and cuts inside a shared prefix are shared,
    /// so the child matches exactly when `text` continues with its first
    /// segment. Returns the steps and the matched bytes, which end at a cut.
    pub(super) fn walk(
        &self,
        root: usize,
        text: &str,
        mut first_cut: impl FnMut(usize) -> Option<usize>,
    ) -> (Vec<Step>, usize) {
        let mut steps = Vec::new();
        let (mut node, mut offset) = (root, 0);
        loop {
            let child = if let Some(child) = self.only_child(node) {
                let first = self.node(child).first_segment().as_bytes();
                if !text.as_bytes()[offset..].starts_with(first) {
                    break;
                }
                child
            } else if self.node(node).child_count == 0 {
                break;
            } else {
                let Some(end) = first_cut(offset) else {
                    break;
                };
                let Some(child) = self.child(node, &text[offset..end]) else {
                    break;
                };
                child
            };
            let edge = self.node(child);
            let rest = &text.as_bytes()[offset..];
            if rest.starts_with(edge.text.as_bytes()) {
                steps.push(Step::Full(child));
                offset += edge.text.len();
                node = child;
                continue;
            }
            let common = common_prefix(edge.text.as_bytes(), rest);
            // The first segment matched, so a partial match has a mark.
            let mark = edge
                .marks
                .partition_point(|&(byte, _)| byte as usize <= common)
                - 1;
            offset += edge.marks[mark].0 as usize;
            steps.push(Step::Partial(child, mark));
            break;
        }
        (steps, offset)
    }

    pub(super) fn lookup(
        &self,
        root: usize,
        text: &str,
        first_cut: impl FnMut(usize) -> Option<usize>,
        tick: u64,
    ) -> Hit {
        let (steps, _) = self.walk(root, text, first_cut);
        let mut hit = Hit {
            root: Some(root),
            version: self.node(root).changed,
            ..Default::default()
        };
        for &step in &steps {
            let (id, bytes, tokens) = match step {
                Step::Full(id) => {
                    let node = self.node(id);
                    (id, node.text.len(), node.ids.len())
                }
                Step::Partial(id, mark) => {
                    let (byte, token) = self.node(id).marks[mark];
                    (id, byte as usize, token as usize)
                }
            };
            let node = self.node(id);
            node.touch(tick);
            hit.pieces.push((node.ids.clone(), tokens));
            hit.bytes += bytes;
            hit.tokens += tokens;
        }
        hit.steps = steps;
        hit
    }

    pub(super) fn matched_tokens(&self, steps: &[Step]) -> usize {
        steps
            .iter()
            .map(|&step| match step {
                Step::Full(id) => self.node(id).ids.len(),
                Step::Partial(id, mark) => self.node(id).marks[mark].1 as usize,
            })
            .sum()
    }

    pub(super) fn root(&mut self, namespace: &Namespace, tick: u64) -> usize {
        if let Some(&root) = self.roots.get(namespace) {
            return root;
        }
        let root = self.alloc(Node::new(
            NO_NODE,
            Box::from(""),
            Arc::from([]),
            Box::from([]),
            tick,
            Some(namespace.clone()),
        ));
        self.roots.insert(namespace.clone(), root);
        root
    }

    /// Content bytes that splitting `id` at `marks[mark]` adds.
    pub(super) fn split_bytes(&self, id: usize, mark: usize) -> usize {
        let node = self.node(id);
        let (byte, token) = node.marks[mark];
        let (byte, token) = (byte as usize, token as usize);
        let upper = node_bytes(byte, token, mark, None);
        let lower = node_bytes(
            node.text.len() - byte,
            node.ids.len() - token,
            node.marks.len() - mark - 1,
            None,
        );
        (upper + lower).saturating_sub(node.bytes())
    }

    /// Split `id` at `marks[mark]`; returns the new upper node.
    pub(super) fn split(&mut self, id: usize, mark: usize) -> usize {
        let parent = self.node(id).parent;
        self.unlink(parent, id);
        let old_bytes = self.node(id).bytes();
        let node = self.node_mut(id);
        let (byte, token) = node.marks[mark];
        let (byte, token) = (byte as usize, token as usize);
        let upper = Node::new(
            parent,
            Box::from(&node.text[..byte]),
            Arc::from(&node.ids[..token]),
            Box::from(&node.marks[..mark]),
            node.last_used.load(Ordering::Relaxed),
            None,
        );
        node.text = Box::from(&node.text[byte..]);
        node.ids = Arc::from(&node.ids[token..]);
        node.marks = node.marks[mark + 1..]
            .iter()
            .map(|&(b, t)| (b - byte as u32, t - token as u32))
            .collect();
        let new_bytes = node.bytes();
        self.content_bytes = self.content_bytes - old_bytes + new_bytes;
        // The upper node keeps the old first segment, hence the same key.
        let upper = self.alloc(upper);
        self.add_child(parent, upper);
        self.node_mut(id).parent = upper;
        self.add_child(upper, id);
        upper
    }

    pub(super) fn push_leaf(&mut self, id: usize) {
        let tick = self.node(id).last_used.load(Ordering::Relaxed);
        self.leaves.push(Reverse((tick, id)));
    }

    /// Remove a leaf. Single-child chains are kept: merging them would copy
    /// whole edges under the write lock.
    pub(super) fn remove_leaf(&mut self, id: usize) {
        let parent = self.node(id).parent;
        self.unlink(parent, id);
        self.release(id);
        let parent_node = self.node(parent);
        if parent_node.child_count > 0 {
            return;
        }
        if parent_node.namespace.is_some() {
            if let Some(namespace) = self.release(parent).namespace {
                self.roots.remove(&namespace);
            }
        } else {
            // Chains shrink from the end: a parent that became a leaf is next.
            self.push_leaf(parent);
        }
    }

    /// Evict least-recently-used leaves until both budgets hold. Each step
    /// is O(log n): stale heap entries are dropped, or re-queued at the
    /// node's current tick, when they reach the top.
    pub(super) fn evict(&mut self, max_bytes: usize, max_nodes: usize) -> u64 {
        let mut evicted = 0;
        while self.live > 0 && (self.live > max_nodes || self.bytes() > max_bytes) {
            let Some(Reverse((tick, id))) = self.leaves.pop() else {
                debug_assert!(false, "every leaf has a heap entry");
                break;
            };
            let Some(node) = self.nodes.get(id).and_then(Option::as_deref) else {
                continue;
            };
            if !node.is_leaf() {
                continue;
            }
            let now = node.last_used.load(Ordering::Relaxed);
            if now != tick {
                self.leaves.push(Reverse((now, id)));
                continue;
            }
            self.remove_leaf(id);
            evicted += 1;
            self.shrink_sparse();
        }
        if self.live == 0 {
            self.release_storage();
        } else if self.leaves.len() > 2 * self.live + 16 {
            self.compact_leaves();
        }
        evicted
    }

    /// Release container capacity once it is mostly unused, so evictions
    /// also return the slack `bytes` charges. A shrink only follows enough
    /// removals to pay for it.
    pub(super) fn shrink_sparse(&mut self) {
        let sparse = |len: usize, capacity: usize| capacity > 64 && len * 4 < capacity;
        if sparse(self.edges.len(), self.edges.capacity()) {
            self.edges.shrink_to_fit();
        }
        if sparse(self.roots.len(), self.roots.capacity()) {
            self.roots.shrink_to_fit();
        }
        if sparse(self.leaves.len(), self.leaves.capacity()) {
            self.leaves.shrink_to_fit();
        }
        if sparse(self.free.len(), self.free.capacity()) {
            self.free.shrink_to_fit();
        }
    }

    /// Keep one entry per live leaf, at its current tick.
    pub(super) fn compact_leaves(&mut self) {
        let nodes = &self.nodes;
        let mut entries: Vec<(usize, u64)> = self
            .leaves
            .drain()
            .filter_map(|Reverse((_, id))| {
                let node = nodes.get(id)?.as_deref()?;
                node.is_leaf()
                    .then(|| (id, node.last_used.load(Ordering::Relaxed)))
            })
            .collect();
        entries.sort_unstable();
        entries.dedup_by_key(|&mut (id, _)| id);
        self.leaves = entries
            .into_iter()
            .map(|(id, tick)| Reverse((tick, id)))
            .collect();
    }

    pub(super) fn release_storage(&mut self) {
        self.version += 1;
        self.nodes = Vec::new();
        self.free = Vec::new();
        self.leaves = BinaryHeap::new();
        self.edges = hashbrown::HashMap::new();
        self.roots = hashbrown::HashMap::new();
    }
}
