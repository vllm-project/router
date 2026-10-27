//! L1 prefix encoding cache: router-side token IDs, not the worker's KV
//! prefix cache.
//!
//! Hugging Face tokenizers (and fastokens) split the input on added tokens
//! before normalizing, pre-tokenizing and running the model, and each text
//! piece is processed on its own. So when a prefix `P` ends right after a
//! *boundary* added token, `encode(P + S) == encode(P) ++ encode(S)` with
//! `add_special_tokens = false`. For Qwen-style templates the boundaries are
//! `<|im_start|>`, `<|im_end|>`, `<tool_call>`, `</tool_call>`, `<think>`, ...
//!
//! The cache is a radix tree over rendered prompt text. Edges end at
//! boundaries and keep the token IDs of their text plus *marks*: the
//! `(byte, token)` offsets of the boundaries inside the edge. A lookup
//! follows the request text through the tree and may stop at any mark, so a
//! request that edits message `k` still reuses messages `0..k`. Only the
//! remaining text after the matched boundary is encoded, in one call; the
//! token offsets of its boundaries come from the boundary token IDs in that
//! encoding, and the new boundaries up to the insert cap are added to the
//! tree as one edge.
//!
//! Children live in one tree-wide map keyed by (parent, first segment), so
//! nodes carry no per-node maps. Leaves are evicted in LRU order from a heap,
//! and `max_bytes` covers node contents plus container capacity.
//!
//! Cut positions come from an automaton over all added tokens that matches
//! like the tokenizer's added-token split. A token is a boundary only when
//! no longer added token can contain it (see [`boundary_rejections`]) and it
//! has no strip/word flags. A load-time probe compares split and unsplit
//! encodes for each candidate in synthetic contexts and drops any candidate
//! that disagrees.
//!
//! Not covered yet: multimodal placeholders (the gRPC path is text-only),
//! tiktoken/tekken tokenizers, boundaries inside messages, and HTTP workers.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Once};

use anyhow::{bail, Context, Result};
use parking_lot::{Mutex, RwLock};
use vllm_tokenizer::Tokenizer;

use crate::metrics::TokenizerMetrics;

#[cfg(doc)]
use self::boundary::boundary_rejections;
use self::boundary::BoundaryMatcher;
use self::tree::{node_bytes, Hit, Namespace, Node, Step, Tree};

mod boundary;
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests;
mod tree;

pub(crate) use self::boundary::load_boundary_matcher;

/// L1 budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1CacheConfig {
    /// Estimated heap bytes retained across models: node contents plus the
    /// tree's containers. Evictions run after each insert until it holds.
    /// Must be at least 1.
    pub max_bytes: usize,
    /// Tree nodes retained across models. Must be at least 1.
    pub max_nodes: usize,
    /// Estimated bytes one request may add, including a new namespace root,
    /// interior boundary marks, and a split node. At most `max_bytes`.
    pub max_insert_bytes: usize,
    /// Boundary segments one request may add. Must be at least 1.
    pub max_insert_segments: usize,
    /// Child key hashing strategy. The default avoids hashing whole long
    /// segments; `Full` is useful for collision-heavy diagnostics.
    pub hash_strategy: L1HashStrategy,
    /// Re-encode every prompt in full and compare (debugging only).
    pub verify: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L1HashStrategy {
    FirstLast64,
    Full,
}

impl Default for L1CacheConfig {
    fn default() -> Self {
        Self {
            max_bytes: 256 * 1024 * 1024,
            max_nodes: 65_536,
            max_insert_bytes: 32 * 1024 * 1024,
            max_insert_segments: 4096,
            hash_strategy: L1HashStrategy::FirstLast64,
            verify: false,
        }
    }
}

impl L1CacheConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_bytes == 0 || self.max_nodes == 0 || self.max_insert_segments == 0 {
            bail!("L1 limits must be positive");
        }
        if self.max_insert_bytes == 0 || self.max_insert_bytes > self.max_bytes {
            bail!("L1 max_insert_bytes must be positive and at most max_bytes");
        }
        Ok(())
    }
}

/// L1 counters and occupancy across models.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct L1CacheStats {
    /// Prompts that had at least one boundary and went through the tree.
    pub lookups: u64,
    /// Lookups that reused at least one cached token.
    pub hits: u64,
    /// Prompt tokens taken from the tree.
    pub cached_tokens: u64,
    /// Prompt tokens produced by the tokenizer on the L1 path.
    pub encoded_tokens: u64,
    /// Nodes removed to satisfy the byte or node budget.
    pub evictions: u64,
    /// Requests whose next segment alone exceeded `max_insert_bytes`.
    pub oversized: u64,
    /// Verify-mode mismatches (each disables L1 for that model).
    pub verify_mismatches: u64,
    /// Prompts on an L1-enabled model that skipped the tree (any reason).
    pub bypasses: u64,
    pub nodes: usize,
    /// Same estimate `max_bytes` bounds.
    pub bytes: usize,
}

/// Why a prompt on an L1-enabled model did not go through the tree. An L0
/// hit is not a bypass: L1 is not consulted at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum L1Bypass {
    /// The renderer produced token IDs (e.g. Harmony), not text.
    TokenIds,
    /// BOS/EOS post-processing is not prefix-stable.
    AddSpecialTokens,
    /// The prompt contains no boundary token.
    NoBoundary,
    /// L1 was turned off for this model after a verify mismatch.
    Disabled,
}

impl L1Bypass {
    fn as_str(self) -> &'static str {
        match self {
            Self::TokenIds => "token_ids",
            Self::AddSpecialTokens => "add_special_tokens",
            Self::NoBoundary => "no_boundary",
            Self::Disabled => "disabled",
        }
    }
}

/// Occupancy summed over every live L1 cache. The lock orders gauge
/// publications, so a scrape sees one snapshot of both totals.
struct Occupancy {
    nodes: usize,
    bytes: usize,
}

static OCCUPANCY: Mutex<Occupancy> = Mutex::new(Occupancy { nodes: 0, bytes: 0 });

/// Apply one cache's change and publish both gauges. Insert calls this while
/// holding the tree write lock; drop calls it with exclusive access to that
/// cache. Lock order is the tree, then `OCCUPANCY`.
fn publish_occupancy(nodes_delta: isize, bytes_delta: isize) {
    if nodes_delta == 0 && bytes_delta == 0 {
        return;
    }
    let mut total = OCCUPANCY.lock();
    total.nodes = total.nodes.saturating_add_signed(nodes_delta);
    total.bytes = total.bytes.saturating_add_signed(bytes_delta);
    TokenizerMetrics::set_l1_nodes(total.nodes);
    TokenizerMetrics::set_l1_bytes(total.bytes);
}

/// Shared L1 tree with one namespace per (tokenizer instance, cache_salt).
pub(crate) struct L1Cache {
    config: L1CacheConfig,
    tree: RwLock<Tree>,
    /// LRU clock. Only inserts advance it, so lookups never write it and
    /// write a node at most once between inserts.
    tick: AtomicU64,
    lookups: AtomicU64,
    hits: AtomicU64,
    cached_tokens: AtomicU64,
    encoded_tokens: AtomicU64,
    evictions: AtomicU64,
    oversized: AtomicU64,
    verify_mismatches: AtomicU64,
    bypasses: AtomicU64,
}

impl L1Cache {
    pub(crate) fn new(config: L1CacheConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            tree: RwLock::new(Tree::new(config.hash_strategy)),
            tick: AtomicU64::new(0),
            lookups: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            cached_tokens: AtomicU64::new(0),
            encoded_tokens: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            oversized: AtomicU64::new(0),
            verify_mismatches: AtomicU64::new(0),
            bypasses: AtomicU64::new(0),
        })
    }

    pub(crate) fn stats(&self) -> L1CacheStats {
        let (nodes, bytes) = {
            let tree = self.tree.read();
            (tree.live, tree.bytes())
        };
        L1CacheStats {
            lookups: self.lookups.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            cached_tokens: self.cached_tokens.load(Ordering::Relaxed),
            encoded_tokens: self.encoded_tokens.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            oversized: self.oversized.load(Ordering::Relaxed),
            verify_mismatches: self.verify_mismatches.load(Ordering::Relaxed),
            bypasses: self.bypasses.load(Ordering::Relaxed),
            nodes,
            bytes,
        }
    }

    fn lookup(
        &self,
        namespace: &Namespace,
        text: &str,
        first_cut: impl FnMut(usize) -> Option<usize>,
    ) -> Hit {
        let tick = self.tick.load(Ordering::Relaxed);
        let tree = self.tree.read();
        match tree.roots.get(namespace) {
            Some(&root) => tree.lookup(root, text, first_cut, tick),
            None => Hit::default(),
        }
    }

    /// Store the cuts after `hit`. `bounds` are the absolute (byte, token)
    /// offsets of cuts after `hit.bytes`, in order.
    fn insert(
        &self,
        namespace: &Namespace,
        text: &str,
        ids: &[u32],
        hit: Hit,
        bounds: &[(usize, usize)],
        first_cut: impl FnMut(usize) -> Option<usize>,
    ) {
        let tick = self.tick.fetch_add(1, Ordering::Relaxed) + 1;
        let mut tree = self.tree.write();
        let (before_nodes, before_bytes) = (tree.live, tree.bytes());
        let (root, steps, offset, start_token) = match tree.roots.get(namespace) {
            // Versions only grow, so a root recreated in the same slot does
            // not match either.
            Some(&root) if hit.root == Some(root) && tree.node(root).changed == hit.version => {
                (hit.root, hit.steps, hit.bytes, hit.tokens)
            }
            Some(&root) => {
                let (steps, offset) = tree.walk(root, text, first_cut);
                let tokens = tree.matched_tokens(&steps);
                (Some(root), steps, offset, tokens)
            }
            None => (None, Vec::new(), 0, 0),
        };
        // An eviction since the lookup removed part of the path; the cuts
        // in between have no token offsets here, so leave it to a later
        // request.
        if offset >= hit.bytes {
            let start = bounds.partition_point(|&(byte, _)| byte <= offset);
            let end = bounds.len().min(start + self.config.max_insert_segments);
            let fresh = &bounds[start..end];
            let salt = namespace.salt.as_ref().map(|salt| salt.len());
            let fixed = if root.is_none() {
                node_bytes(0, 0, 0, salt)
            } else {
                0
            } + match steps.last() {
                Some(&Step::Partial(id, mark)) => tree.split_bytes(id, mark),
                _ => 0,
            };
            // Cost grows with the number of segments; keep the longest run
            // that fits.
            let cost = |i: usize| {
                let (byte, token) = fresh[i];
                fixed + node_bytes(byte - offset, token - start_token, i, None)
            };
            let (mut stored, mut over) = (0, fresh.len());
            while stored < over {
                let mid = (stored + over) / 2;
                if cost(mid) <= self.config.max_insert_bytes {
                    stored = mid + 1;
                } else {
                    over = mid;
                }
            }
            if stored == 0 && !fresh.is_empty() {
                self.oversized.fetch_add(1, Ordering::Relaxed);
            }
            if stored > 0 {
                let mut parent = root.unwrap_or_else(|| tree.root(namespace, tick));
                for &step in &steps {
                    match step {
                        Step::Full(id) => {
                            tree.node(id).touch(tick);
                            parent = id;
                        }
                        Step::Partial(id, mark) => parent = tree.split(id, mark),
                    }
                }
                let (end_byte, end_token) = fresh[stored - 1];
                let marks: Box<[(u32, u32)]> = fresh[..stored - 1]
                    .iter()
                    .map(|&(b, t)| ((b - offset) as u32, (t - start_token) as u32))
                    .collect();
                // Growth appends a child; existing edges are never copied.
                let leaf = tree.alloc(Node::new(
                    parent,
                    Box::from(&text[offset..end_byte]),
                    Arc::from(&ids[start_token..end_token]),
                    marks,
                    tick,
                    None,
                ));
                tree.add_child(parent, leaf);
                tree.push_leaf(leaf);
            }
        }
        let evicted = tree.evict(self.config.max_bytes, self.config.max_nodes);
        if evicted > 0 {
            self.evictions.fetch_add(evicted, Ordering::Relaxed);
            TokenizerMetrics::record_l1_evictions(evicted);
        }
        publish_occupancy(
            tree.live as isize - before_nodes as isize,
            tree.bytes() as isize - before_bytes as isize,
        );
    }
}

impl Drop for L1Cache {
    fn drop(&mut self) {
        let tree = self.tree.get_mut();
        publish_occupancy(-(tree.live as isize), -(tree.bytes() as isize));
    }
}

/// Tails up to this size are encoded one segment at a time. fastokens sends
/// an input with 16 or more pre-token splits to a thread pool, which costs
/// more than encoding a few short segments directly; longer tails gain from
/// the pool.
const SEGMENTED_TAIL_BYTES: usize = 256;

/// Encode the tail after `hit` into `ids`, one segment per cut plus the text
/// after the last cut. Returns the absolute (byte, token) offsets of the cuts.
fn encode_segments(
    tokenizer: &dyn Tokenizer,
    text: &str,
    hit: &Hit,
    cuts: &[(usize, usize)],
    ids: &mut Vec<u32>,
) -> vllm_tokenizer::Result<Vec<(usize, usize)>> {
    let mut bounds = Vec::with_capacity(cuts.len());
    let mut from = hit.bytes;
    for &(byte, _) in cuts {
        ids.extend(tokenizer.encode(&text[from..byte], false)?);
        bounds.push((byte, hit.tokens + ids.len()));
        from = byte;
    }
    if from < text.len() {
        ids.extend(tokenizer.encode(&text[from..], false)?);
    }
    Ok(bounds)
}

/// Per-model L1 handle used by `PromptEncoder`.
#[derive(Clone)]
pub(crate) struct L1Encoder {
    cache: Arc<L1Cache>,
    matcher: Arc<BoundaryMatcher>,
    model_id: Arc<str>,
    disabled: Arc<AtomicBool>,
    mismatch_warning: Arc<Once>,
    segment_warning: Arc<Once>,
}

impl L1Encoder {
    pub(crate) fn new(model_id: &str, cache: Arc<L1Cache>, matcher: BoundaryMatcher) -> Self {
        Self {
            cache,
            matcher: Arc::new(matcher),
            model_id: Arc::from(model_id),
            disabled: Arc::new(AtomicBool::new(false)),
            mismatch_warning: Arc::new(Once::new()),
            segment_warning: Arc::new(Once::new()),
        }
    }

    /// Encode with L1, or `None` if the prompt has no boundary or L1 was
    /// disabled. Returns the IDs and how many came from the tree.
    pub(crate) fn encode(
        &self,
        tokenizer: &dyn Tokenizer,
        instance: u64,
        text: &str,
        cache_salt: Option<&str>,
    ) -> vllm_tokenizer::Result<Option<(Vec<u32>, usize)>> {
        if self.disabled.load(Ordering::Relaxed) {
            self.record_bypass(L1Bypass::Disabled);
            return Ok(None);
        }
        let namespace = Namespace {
            instance,
            salt: cache_salt.map(Arc::<str>::from),
        };
        let matcher = &*self.matcher;
        let first_cut = |from: usize| matcher.first_cut(text, from);
        let hit = self.cache.lookup(&namespace, text, first_cut);
        // Every cut up to the match is already in the tree; scan the rest.
        let cuts = matcher.cuts(text, hit.bytes);
        if hit.bytes == 0 && cuts.is_empty() {
            self.record_bypass(L1Bypass::NoBoundary);
            return Ok(None);
        }
        // The tail starts at a cut, so one encode of it has the same IDs as
        // encoding each segment.
        let tail = &text[hit.bytes..];
        let (tail_ids, bounds) = if tail.len() <= SEGMENTED_TAIL_BYTES {
            let mut tail_ids = Vec::new();
            let bounds = encode_segments(tokenizer, text, &hit, &cuts, &mut tail_ids)?;
            (tail_ids, bounds)
        } else {
            let tail_ids = tokenizer.encode(tail, false)?;
            let bounds = self.tail_bounds(tokenizer, text, &hit, &cuts, &tail_ids)?;
            (tail_ids, bounds)
        };
        let mut ids = Vec::with_capacity(hit.tokens + tail_ids.len());
        for (piece, take) in &hit.pieces {
            ids.extend_from_slice(&piece[..*take]);
        }
        ids.extend(tail_ids);
        if self.cache.config.verify {
            let full = tokenizer.encode(text, false)?;
            if full != ids {
                self.disabled.store(true, Ordering::Relaxed);
                self.cache.verify_mismatches.fetch_add(1, Ordering::Relaxed);
                TokenizerMetrics::record_l1_verify_mismatch();
                let first = full.iter().zip(&ids).take_while(|(a, b)| a == b).count();
                self.mismatch_warning.call_once(|| {
                    tracing::error!(
                        model_id = %self.model_id,
                        first_mismatch = first,
                        full_len = full.len(),
                        l1_len = ids.len(),
                        "L1 verify mismatch; L1 disabled for this model"
                    );
                });
                return Ok(Some((full, 0)));
            }
        }
        let cached = hit.tokens;
        self.cache.lookups.fetch_add(1, Ordering::Relaxed);
        self.cache
            .cached_tokens
            .fetch_add(cached as u64, Ordering::Relaxed);
        self.cache
            .encoded_tokens
            .fetch_add((ids.len() - cached) as u64, Ordering::Relaxed);
        TokenizerMetrics::record_l1_lookup(cached, ids.len() - cached);
        if cached > 0 {
            self.cache.hits.fetch_add(1, Ordering::Relaxed);
        }
        if !bounds.is_empty() {
            self.cache
                .insert(&namespace, text, &ids, hit, &bounds, first_cut);
        }
        Ok(Some((ids, cached)))
    }

    /// Absolute (byte, token) offsets of the tail's `cuts`, read from the
    /// tail encoding when boundary IDs line up. Otherwise the segments are
    /// encoded one at a time, as far as insert could store them.
    fn tail_bounds(
        &self,
        tokenizer: &dyn Tokenizer,
        text: &str,
        hit: &Hit,
        cuts: &[(usize, usize)],
        tail_ids: &[u32],
    ) -> vllm_tokenizer::Result<Vec<(usize, usize)>> {
        if let Some(offsets) = self.matcher.token_offsets(cuts, tail_ids) {
            return Ok(cuts
                .iter()
                .zip(offsets)
                .map(|(&(byte, _), token)| (byte, hit.tokens + token))
                .collect());
        }
        self.segment_warning.call_once(|| {
            tracing::debug!(
                model_id = %self.model_id,
                "boundary token IDs do not line up with the encoding; L1 encodes segments one at a time"
            );
        });
        let config = &self.cache.config;
        let (mut from, mut tokens) = (hit.bytes, 0);
        let mut bounds = Vec::new();
        for &(byte, _) in cuts.iter().take(config.max_insert_segments) {
            if byte - hit.bytes > config.max_insert_bytes {
                break;
            }
            let piece = tokenizer.encode(&text[from..byte], false)?;
            // Never store a segment that disagrees with the tail encoding.
            if tail_ids.get(tokens..tokens + piece.len()) != Some(&piece[..]) {
                break;
            }
            tokens += piece.len();
            from = byte;
            bounds.push((byte, hit.tokens + tokens));
        }
        Ok(bounds)
    }

    pub(crate) fn record_bypass(&self, reason: L1Bypass) {
        self.cache.bypasses.fetch_add(1, Ordering::Relaxed);
        TokenizerMetrics::record_l1_bypass(reason.as_str());
    }
}

pub(crate) fn config_from_env() -> Result<Option<L1CacheConfig>> {
    parse_config(|name| std::env::var(name).ok())
}

fn parse_flag(name: &str, raw: Option<String>) -> Result<bool> {
    match raw.as_deref() {
        None | Some("0" | "false") => Ok(false),
        Some("1" | "true") => Ok(true),
        _ => bail!("{name} must be 0, 1, false or true"),
    }
}

fn parse_hash_strategy(raw: Option<String>) -> Result<L1HashStrategy> {
    match raw.as_deref() {
        None | Some("first_last_64" | "first-last-64" | "first_last" | "first-last") => {
            Ok(L1HashStrategy::FirstLast64)
        }
        Some("full") | Some("full_segment") | Some("full-segment") => Ok(L1HashStrategy::Full),
        Some(raw) => bail!("VLLM_ROUTER_L1_HASH_STRATEGY must be first_last_64 or full, got {raw}"),
    }
}

fn parse_config(read: impl Fn(&str) -> Option<String>) -> Result<Option<L1CacheConfig>> {
    if !parse_flag("VLLM_ROUTER_L1_CACHE", read("VLLM_ROUTER_L1_CACHE"))? {
        return Ok(None);
    }
    let mut config = L1CacheConfig {
        verify: parse_flag("VLLM_ROUTER_L1_VERIFY", read("VLLM_ROUTER_L1_VERIFY"))?,
        hash_strategy: parse_hash_strategy(read("VLLM_ROUTER_L1_HASH_STRATEGY"))?,
        ..Default::default()
    };
    let explicit_insert = read("VLLM_ROUTER_L1_MAX_INSERT_BYTES").is_some();
    for (name, value) in [
        ("VLLM_ROUTER_L1_MAX_BYTES", &mut config.max_bytes),
        ("VLLM_ROUTER_L1_MAX_NODES", &mut config.max_nodes),
        (
            "VLLM_ROUTER_L1_MAX_INSERT_BYTES",
            &mut config.max_insert_bytes,
        ),
        (
            "VLLM_ROUTER_L1_MAX_INSERT_SEGMENTS",
            &mut config.max_insert_segments,
        ),
    ] {
        if let Some(raw) = read(name) {
            *value = raw
                .parse()
                .with_context(|| format!("invalid {name}: {raw}"))?;
        }
    }
    if !explicit_insert {
        config.max_insert_bytes = config.max_insert_bytes.min(config.max_bytes);
    }
    config.validate()?;
    Ok(Some(config))
}
