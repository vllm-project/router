//! L1 prefix encoding cache.
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
//! remaining text after the matched boundary is encoded. New segments are
//! encoded one at a time up to the insert cap; the rest of the tail is one
//! encode, and the stored boundaries are inserted back into the tree.
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

use std::cmp::Reverse;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Once};

use aho_corasick::{AhoCorasick, MatchKind};
use anyhow::{bail, Context, Result};
use parking_lot::{Mutex, RwLock};
use regex::Regex;
use serde_json::Value;
use vllm_text::backend::hf::{ResolvedModelFiles, TokenizerSource};
use vllm_tokenizer::Tokenizer;

use crate::metrics::TokenizerMetrics;

/// L1 budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixCacheConfig {
    /// Estimated bytes retained across models. Must be at least 1.
    pub max_bytes: usize,
    /// Tree nodes retained across models. Must be at least 1.
    pub max_nodes: usize,
    /// Estimated bytes one request may add. At most `max_bytes`.
    pub max_insert_bytes: usize,
    /// Boundary segments one request may add. Must be at least 1.
    pub max_insert_segments: usize,
    /// Child key hashing strategy. The default avoids hashing whole long
    /// segments; `Full` is useful for collision-heavy diagnostics.
    pub hash_strategy: PrefixCacheHashStrategy,
    /// Re-encode every prompt in full and compare (debugging only).
    pub verify: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixCacheHashStrategy {
    FirstLast64,
    Full,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            max_bytes: 256 * 1024 * 1024,
            max_nodes: 65_536,
            max_insert_bytes: 32 * 1024 * 1024,
            max_insert_segments: 4096,
            hash_strategy: PrefixCacheHashStrategy::FirstLast64,
            verify: false,
        }
    }
}

impl PrefixCacheConfig {
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
pub struct PrefixCacheStats {
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
    pub nodes: usize,
    pub bytes: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct AddedTokenSpec {
    pub content: String,
    pub lstrip: bool,
    pub rstrip: bool,
    pub single_word: bool,
    pub normalized: bool,
}

/// A model's added tokens and whether its normalizer leaves text unchanged.
#[derive(Debug, Clone, Default)]
pub(crate) struct AddedTokenSet {
    pub tokens: Vec<AddedTokenSpec>,
    pub identity_normalizer: bool,
}

/// Finds cut positions: byte offsets right after a boundary token.
///
/// The automaton holds every added token and matches leftmost-longest like
/// the tokenizer's added-token split, so a boundary only counts where the
/// tokenizer itself splits it out.
pub(crate) struct BoundaryMatcher {
    automaton: AhoCorasick,
    patterns: Vec<String>,
    /// Per pattern: whether a cut follows it.
    cut: Vec<bool>,
    tokens: Vec<String>,
}

impl std::fmt::Debug for BoundaryMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundaryMatcher")
            .field("tokens", &self.tokens)
            .finish()
    }
}

impl BoundaryMatcher {
    pub(crate) fn new(added: &AddedTokenSet) -> Result<Self> {
        let mut tokens = Vec::new();
        for (token, rejection) in added.tokens.iter().zip(boundary_rejections(added)) {
            match rejection {
                None => tokens.push(token.content.clone()),
                Some(reason) => {
                    tracing::debug!(token = %token.content, reason, "not an L1 boundary")
                }
            }
        }
        let patterns = added
            .tokens
            .iter()
            .filter(|token| !token.content.is_empty())
            .map(|token| token.content.clone())
            .collect();
        Self::build(patterns, tokens)
    }

    fn build(mut patterns: Vec<String>, mut tokens: Vec<String>) -> Result<Self> {
        tokens.sort();
        tokens.dedup();
        if tokens.is_empty() {
            bail!("no added token is a safe encoding boundary");
        }
        patterns.sort();
        patterns.dedup();
        let cut = patterns
            .iter()
            .map(|pattern| tokens.binary_search(pattern).is_ok())
            .collect();
        let automaton = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)?;
        Ok(Self {
            automaton,
            patterns,
            cut,
            tokens,
        })
    }

    pub(crate) fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// Cut positions in `text`, which must itself start at a cut position.
    pub(crate) fn boundaries(&self, text: &str) -> Vec<usize> {
        self.automaton
            .find_iter(text)
            .filter(|m| self.cut[m.pattern().as_usize()])
            .map(|m| m.end())
            .collect()
    }

    /// Keep only boundaries for which splitting gives the same IDs as a full
    /// encode of synthetic prompts.
    pub(crate) fn probed(self, tokenizer: &dyn Tokenizer) -> Result<Self> {
        const BEFORE: [&str; 5] = ["", "hi", "Hello, world. ", "x\n", "数字 123"];
        const AFTER: [&str; 11] = [
            "",
            "hi",
            " world",
            "\n\nnext",
            "\u{301}e",
            "\u{338}",
            "123 456",
            " \t ",
            "日本語",
            "😀!",
            "'s it",
        ];
        let mut kept = Vec::with_capacity(self.tokens.len());
        'tokens: for token in &self.tokens {
            for before in BEFORE {
                let prefix = format!("{before}{token}");
                let head = tokenizer.encode(&prefix, false)?;
                for after in AFTER {
                    let mut split = head.clone();
                    split.extend(tokenizer.encode(after, false)?);
                    if tokenizer.encode(&format!("{prefix}{after}"), false)? != split {
                        tracing::warn!(token, after, "L1 boundary failed probe; skipping");
                        continue 'tokens;
                    }
                }
            }
            kept.push(token.clone());
        }
        Self::build(self.patterns, kept)
    }
}

/// Why each added token cannot be a boundary (`None` if it can).
///
/// fastokens matches all added tokens in one leftmost-longest pass; Hugging
/// Face matches non-normalized tokens first and normalized ones after
/// normalization. When no token of one group can overlap a token of the
/// other, both split exactly like [`BoundaryMatcher`]. Then a cut after `t`
/// only fails if a longer added token contains `t` before its end: it can
/// win in `P + S` but does not fit in `P`. Otherwise no added token may
/// overlap `t` at all.
pub(crate) fn boundary_rejections(added: &AddedTokenSet) -> Vec<Option<&'static str>> {
    let contents: Vec<&str> = added
        .tokens
        .iter()
        .map(|token| token.content.as_str())
        .filter(|content| !content.is_empty())
        .collect();
    let (normalized, raw): (Vec<&AddedTokenSpec>, Vec<&AddedTokenSpec>) =
        added.tokens.iter().partition(|token| token.normalized);
    let separate = normalized
        .iter()
        .all(|y| raw.iter().all(|x| !overlaps(&x.content, &y.content)));
    let covered = covered_tokens(&contents);
    let any_single_word = added.tokens.iter().any(|token| token.single_word);
    let rejection = |token: &AddedTokenSpec| {
        let t = token.content.as_str();
        if t.is_empty() {
            return Some("empty");
        }
        if token.rstrip {
            return Some("rstrip absorbs whitespace after the cut");
        }
        if token.single_word {
            return Some("single_word depends on the next character");
        }
        if token.lstrip {
            return Some("lstrip (excluded conservatively)");
        }
        if token.normalized && !added.identity_normalizer {
            return Some("normalized and the normalizer changes text");
        }
        if token.normalized && !separate {
            return Some("normalized tokens overlap non-normalized ones");
        }
        // Strip flags on neighbouring tokens absorb adjacent whitespace.
        if t.starts_with(char::is_whitespace) || t.ends_with(char::is_whitespace) {
            return Some("starts or ends with whitespace");
        }
        // A single_word token right after the cut sees `t`'s last character
        // in `P + S` but the start of input in `S`.
        if any_single_word && ends_with_word(t) {
            return Some("a single_word token could follow");
        }
        if covered.contains(t) {
            return Some("a longer added token contains it");
        }
        if !separate
            && (char_offsets(t).any(|k| t.starts_with(&t[k..]))
                || contents.iter().any(|&x| x != t && reaches_into(x, t)))
        {
            return Some("another added token can overlap it");
        }
        None
    };
    added.tokens.iter().map(rejection).collect()
}

/// Hugging Face's `single_word` test: Unicode `\w`, which also covers
/// combining marks, connector punctuation and joiners.
fn ends_with_word(text: &str) -> bool {
    static ENDS_WITH_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w$").unwrap());
    ENDS_WITH_WORD.is_match(text)
}

/// Tokens that occur inside another added token, ending before its end.
fn covered_tokens<'a>(contents: &[&'a str]) -> HashSet<&'a str> {
    let Ok(automaton) = AhoCorasick::new(contents) else {
        return contents.iter().copied().collect();
    };
    let mut covered = HashSet::new();
    for x in contents {
        for m in automaton.find_overlapping_iter(x) {
            if m.end() < x.len() {
                covered.insert(contents[m.pattern().as_usize()]);
            }
        }
    }
    covered
}

/// Whether occurrences of `a` and `b` can overlap in some text.
fn overlaps(a: &str, b: &str) -> bool {
    a == b || reaches_into(a, b) || reaches_into(b, a)
}

/// Whether a match of `x` can start before an occurrence of `t` and reach
/// into it, or start at the same byte and be longer.
fn reaches_into(x: &str, t: &str) -> bool {
    if x.len() > t.len() && x.starts_with(t) {
        return true;
    }
    char_offsets(x).any(|k| {
        let suffix = &x[k..];
        t.starts_with(suffix) || suffix.starts_with(t)
    })
}

fn char_offsets(s: &str) -> impl Iterator<Item = usize> + '_ {
    s.char_indices().map(|(i, _)| i).skip(1)
}

/// Build a boundary matcher from the model's tokenizer files, or explain why
/// L1 cannot be used for it.
pub(crate) async fn load_boundary_matcher(model: &str) -> Result<BoundaryMatcher> {
    BoundaryMatcher::new(&load_added_tokens(model).await?)
}

/// Added tokens as vllm-tokenizer sees them, after config checks.
pub(crate) async fn load_added_tokens(model: &str) -> Result<AddedTokenSet> {
    let files = ResolvedModelFiles::new(model).await?;
    let TokenizerSource::HuggingFace(path) = files.tokenizer else {
        bail!("L1 supports tokenizer.json models only");
    };
    let json: Value = serde_json::from_slice(&std::fs::read(&path)?)
        .with_context(|| format!("parse {}", path.display()))?;
    check_tokenizer_json(&json)?;
    let mut added = added_tokens(&json["added_tokens"], None);
    // vllm-tokenizer merges extra tokens from the sibling tokenizer_config.json.
    let config_path = path
        .parent()
        .unwrap_or(Path::new("."))
        .join("tokenizer_config.json");
    if let Ok(raw) = std::fs::read(&config_path) {
        if let Ok(config) = serde_json::from_slice::<Value>(&raw) {
            let known: Vec<u64> = json["added_tokens"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|token| token["id"].as_u64())
                .collect();
            added.extend(added_tokens(&config["added_tokens_decoder"], Some(&known)));
        }
    }
    Ok(AddedTokenSet {
        tokens: added,
        identity_normalizer: is_identity_normalizer(&json["normalizer"]),
    })
}

fn is_identity_normalizer(normalizer: &Value) -> bool {
    match normalizer["type"].as_str() {
        None => normalizer.is_null(),
        Some("Sequence") => normalizer["normalizers"]
            .as_array()
            .is_some_and(|steps| steps.iter().all(is_identity_normalizer)),
        _ => false,
    }
}

fn check_tokenizer_json(json: &Value) -> Result<()> {
    if !super::encode_cache::deterministic_hf_model(&json["model"]) {
        bail!("tokenizer model is not known to be deterministic");
    }
    if !json["truncation"].is_null() || !json["padding"].is_null() {
        bail!("tokenizer.json enables truncation or padding");
    }
    if uses_first_only_metaspace(&json["pre_tokenizer"]) {
        bail!("Metaspace prepend_scheme=first depends on the split position");
    }
    Ok(())
}

fn uses_first_only_metaspace(pre_tokenizer: &Value) -> bool {
    match pre_tokenizer["type"].as_str() {
        Some("Metaspace") => pre_tokenizer["prepend_scheme"].as_str() == Some("first"),
        Some("Sequence") => pre_tokenizer["pretokenizers"]
            .as_array()
            .into_iter()
            .flatten()
            .any(uses_first_only_metaspace),
        _ => false,
    }
}

/// Parse `added_tokens` (array) or `added_tokens_decoder` (id-keyed map),
/// skipping IDs in `skip`.
fn added_tokens(value: &Value, skip: Option<&[u64]>) -> Vec<AddedTokenSpec> {
    let parse = |token: &Value| {
        Some(AddedTokenSpec {
            content: token["content"].as_str()?.to_string(),
            lstrip: token["lstrip"].as_bool().unwrap_or(false),
            rstrip: token["rstrip"].as_bool().unwrap_or(false),
            single_word: token["single_word"].as_bool().unwrap_or(false),
            // vllm-tokenizer fills a missing field with false before either
            // backend loads. Hugging Face's own default is `!special`.
            normalized: token["normalized"].as_bool().unwrap_or(false),
        })
    };
    match value {
        Value::Array(tokens) => tokens.iter().filter_map(parse).collect(),
        Value::Object(tokens) => tokens
            .iter()
            .filter(|(id, _)| {
                let id = id.parse::<u64>().ok();
                id.is_some_and(|id| !skip.unwrap_or_default().contains(&id))
            })
            .filter_map(|(_, token)| parse(token))
            .collect(),
        _ => Vec::new(),
    }
}

const NO_PARENT: usize = usize::MAX;
const NODE_OVERHEAD_BYTES: usize = 192;

#[derive(Clone, Hash, PartialEq, Eq)]
struct Namespace {
    instance: u64,
    salt: Option<Box<str>>,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct SegmentKey {
    len: usize,
    fingerprint: u64,
}

fn segment_key(segment: &str, strategy: PrefixCacheHashStrategy) -> SegmentKey {
    let bytes = segment.as_bytes();
    let mut hasher = DefaultHasher::new();
    match strategy {
        PrefixCacheHashStrategy::Full => bytes.hash(&mut hasher),
        PrefixCacheHashStrategy::FirstLast64 if bytes.len() <= 128 => bytes.hash(&mut hasher),
        PrefixCacheHashStrategy::FirstLast64 => {
            bytes[..64].hash(&mut hasher);
            bytes[bytes.len() - 64..].hash(&mut hasher);
        }
    }
    SegmentKey {
        len: bytes.len(),
        fingerprint: hasher.finish(),
    }
}

struct Node {
    parent: usize,
    /// Edge text; it ends at a boundary (empty for namespace roots).
    text: Box<str>,
    ids: Arc<[u32]>,
    /// Boundaries strictly inside the edge as (byte, token) offsets.
    marks: Box<[(u32, u32)]>,
    /// Children keyed by their first segment.
    children: HashMap<SegmentKey, Vec<usize>>,
    child_count: usize,
    last_used: AtomicU64,
    /// Set on namespace roots.
    namespace: Option<Namespace>,
}

impl Node {
    fn first_segment(&self) -> &str {
        match self.marks.first() {
            Some(&(byte, _)) => &self.text[..byte as usize],
            None => &self.text,
        }
    }

    fn bytes(&self) -> usize {
        NODE_OVERHEAD_BYTES
            + self.text.len()
            + self.ids.len() * 4
            + self.marks.len() * 8
            + self
                .namespace
                .as_ref()
                .and_then(|ns| ns.salt.as_ref())
                .map_or(0, |salt| salt.len())
    }
}

enum Step {
    /// The whole edge matched.
    Full(usize),
    /// The match ends at `marks[mark]` of the node.
    Partial(usize, usize),
}

#[derive(Default)]
struct Hit {
    bytes: usize,
    tokens: usize,
    pieces: Vec<(Arc<[u32]>, usize)>,
    /// Matched boundaries as absolute (byte, token) offsets.
    bounds: Vec<(usize, usize)>,
}

#[derive(Default)]
struct Tree {
    nodes: Vec<Option<Node>>,
    free: Vec<usize>,
    roots: HashMap<Namespace, usize>,
    live: usize,
    bytes: usize,
}

impl Tree {
    fn node(&self, id: usize) -> &Node {
        self.nodes[id].as_ref().expect("live L1 node")
    }

    fn node_mut(&mut self, id: usize) -> &mut Node {
        self.nodes[id].as_mut().expect("live L1 node")
    }

    fn alloc(&mut self, node: Node) -> usize {
        self.live += 1;
        self.bytes += node.bytes();
        match self.free.pop() {
            Some(id) => {
                self.nodes[id] = Some(node);
                id
            }
            None => {
                self.nodes.push(Some(node));
                self.nodes.len() - 1
            }
        }
    }

    fn release(&mut self, id: usize) -> Node {
        let node = self.nodes[id].take().expect("live L1 node");
        self.free.push(id);
        self.live -= 1;
        self.bytes -= node.bytes();
        node
    }

    fn child(
        &self,
        parent: usize,
        segment: &str,
        strategy: PrefixCacheHashStrategy,
    ) -> Option<usize> {
        self.node(parent)
            .children
            .get(&segment_key(segment, strategy))?
            .iter()
            .copied()
            .find(|&id| self.node(id).first_segment() == segment)
    }

    fn add_child(&mut self, parent: usize, child: usize, strategy: PrefixCacheHashStrategy) {
        let key = segment_key(self.node(child).first_segment(), strategy);
        let node = self.node_mut(parent);
        node.children.entry(key).or_default().push(child);
        node.child_count += 1;
    }

    fn remove_child(&mut self, parent: usize, child: usize, key: SegmentKey) {
        let node = self.node_mut(parent);
        if let Some(ids) = node.children.get_mut(&key) {
            ids.retain(|&id| id != child);
            if ids.is_empty() {
                node.children.remove(&key);
            }
        }
        node.child_count -= 1;
    }

    /// Follow `text` from `root`; returns the steps and the matched bytes.
    fn walk(
        &self,
        root: usize,
        text: &str,
        bounds: &[usize],
        strategy: PrefixCacheHashStrategy,
    ) -> (Vec<Step>, usize) {
        let mut steps = Vec::new();
        let (mut node, mut offset) = (root, 0);
        loop {
            let next = bounds.partition_point(|&bound| bound <= offset);
            let Some(&end) = bounds.get(next) else {
                break;
            };
            let Some(child) = self.child(node, &text[offset..end], strategy) else {
                break;
            };
            let edge = self.node(child);
            let rest = &text.as_bytes()[offset..];
            if rest.starts_with(edge.text.as_bytes()) {
                steps.push(Step::Full(child));
                offset += edge.text.len();
                node = child;
                continue;
            }
            let common = edge
                .text
                .as_bytes()
                .iter()
                .zip(rest)
                .take_while(|(a, b)| a == b)
                .count();
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

    fn lookup(
        &self,
        root: usize,
        text: &str,
        bounds: &[usize],
        tick: u64,
        strategy: PrefixCacheHashStrategy,
    ) -> Hit {
        let mut hit = Hit::default();
        for step in self.walk(root, text, bounds, strategy).0 {
            let (id, bytes, tokens, marks) = match step {
                Step::Full(id) => {
                    let node = self.node(id);
                    (id, node.text.len(), node.ids.len(), &node.marks[..])
                }
                Step::Partial(id, mark) => {
                    let node = self.node(id);
                    let (byte, token) = node.marks[mark];
                    (id, byte as usize, token as usize, &node.marks[..mark])
                }
            };
            let node = self.node(id);
            node.last_used.store(tick, Ordering::Relaxed);
            for &(byte, token) in marks {
                hit.bounds
                    .push((hit.bytes + byte as usize, hit.tokens + token as usize));
            }
            hit.pieces.push((node.ids.clone(), tokens));
            hit.bytes += bytes;
            hit.tokens += tokens;
            hit.bounds.push((hit.bytes, hit.tokens));
        }
        hit
    }

    fn root(&mut self, namespace: &Namespace, tick: u64) -> usize {
        if let Some(&root) = self.roots.get(namespace) {
            return root;
        }
        let root = self.alloc(Node {
            parent: NO_PARENT,
            text: Box::from(""),
            ids: Arc::from([]),
            marks: Box::from([]),
            children: HashMap::new(),
            child_count: 0,
            last_used: AtomicU64::new(tick),
            namespace: Some(namespace.clone()),
        });
        self.roots.insert(namespace.clone(), root);
        root
    }

    /// Split `id` at `marks[mark]`; returns the new upper node.
    fn split(&mut self, id: usize, mark: usize, strategy: PrefixCacheHashStrategy) -> usize {
        let old_bytes = self.node(id).bytes();
        let node = self.node_mut(id);
        let (byte, token) = node.marks[mark];
        let (byte, token) = (byte as usize, token as usize);
        let upper_text: Box<str> = Box::from(&node.text[..byte]);
        let upper_ids: Arc<[u32]> = Arc::from(&node.ids[..token]);
        let upper_marks: Box<[(u32, u32)]> = Box::from(&node.marks[..mark]);
        node.text = Box::from(&node.text[byte..]);
        node.ids = Arc::from(&node.ids[token..]);
        node.marks = node.marks[mark + 1..]
            .iter()
            .map(|&(b, t)| (b - byte as u32, t - token as u32))
            .collect();
        let parent = node.parent;
        let last_used = node.last_used.load(Ordering::Relaxed);
        let new_bytes = node.bytes();
        self.bytes = self.bytes - old_bytes + new_bytes;
        let upper = self.alloc(Node {
            parent,
            text: upper_text,
            ids: upper_ids,
            marks: upper_marks,
            children: HashMap::new(),
            child_count: 0,
            last_used: AtomicU64::new(last_used),
            namespace: None,
        });
        // The upper node keeps the old first segment, hence the same key.
        let key = segment_key(self.node(upper).first_segment(), strategy);
        if let Some(ids) = self.node_mut(parent).children.get_mut(&key) {
            for slot in ids.iter_mut().filter(|slot| **slot == id) {
                *slot = upper;
            }
        }
        self.node_mut(id).parent = upper;
        self.add_child(upper, id, strategy);
        upper
    }

    /// Remove a leaf. Single-child chains are kept: merging them would copy
    /// whole edges under the write lock.
    fn remove_leaf(&mut self, id: usize, strategy: PrefixCacheHashStrategy) {
        let key = segment_key(self.node(id).first_segment(), strategy);
        let parent = self.release(id).parent;
        self.remove_child(parent, id, key);
        let parent_node = self.node(parent);
        if parent_node.namespace.is_some() && parent_node.child_count == 0 {
            if let Some(namespace) = self.release(parent).namespace {
                self.roots.remove(&namespace);
            }
        }
    }

    /// Evict least-recently-used leaves down to 90% of the limits.
    fn evict(&mut self, config: &PrefixCacheConfig) -> u64 {
        if self.bytes <= config.max_bytes && self.live <= config.max_nodes {
            return 0;
        }
        let target_bytes = config.max_bytes - config.max_bytes / 10;
        let target_nodes = config.max_nodes - config.max_nodes / 10;
        let is_leaf = |node: &Node| node.namespace.is_none() && node.child_count == 0;
        let mut heap: BinaryHeap<Reverse<(u64, usize)>> = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(id, node)| {
                let node = node.as_ref()?;
                is_leaf(node).then(|| Reverse((node.last_used.load(Ordering::Relaxed), id)))
            })
            .collect();
        let mut evicted = 0;
        while let Some(Reverse((_, id))) = heap.pop() {
            if self.bytes <= target_bytes && self.live <= target_nodes {
                break;
            }
            let parent = self.node(id).parent;
            self.remove_leaf(id, config.hash_strategy);
            evicted += 1;
            // Chains shrink from the end: a parent that became a leaf is next.
            if let Some(node) = self.nodes[parent].as_ref().filter(|node| is_leaf(node)) {
                heap.push(Reverse((node.last_used.load(Ordering::Relaxed), parent)));
            }
        }
        evicted
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
    TokenizerMetrics::set_prefix_cache_nodes(total.nodes);
    TokenizerMetrics::set_prefix_cache_bytes(total.bytes);
}

/// Shared L1 tree with one namespace per (tokenizer instance, cache_salt).
pub(crate) struct PrefixCache {
    config: PrefixCacheConfig,
    tree: RwLock<Tree>,
    tick: AtomicU64,
    lookups: AtomicU64,
    hits: AtomicU64,
    cached_tokens: AtomicU64,
    encoded_tokens: AtomicU64,
    evictions: AtomicU64,
    oversized: AtomicU64,
    verify_mismatches: AtomicU64,
}

impl PrefixCache {
    pub(crate) fn new(config: PrefixCacheConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            tree: RwLock::new(Tree::default()),
            tick: AtomicU64::new(0),
            lookups: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            cached_tokens: AtomicU64::new(0),
            encoded_tokens: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            oversized: AtomicU64::new(0),
            verify_mismatches: AtomicU64::new(0),
        })
    }

    pub(crate) fn stats(&self) -> PrefixCacheStats {
        let (nodes, bytes) = {
            let tree = self.tree.read();
            (tree.live, tree.bytes)
        };
        PrefixCacheStats {
            lookups: self.lookups.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            cached_tokens: self.cached_tokens.load(Ordering::Relaxed),
            encoded_tokens: self.encoded_tokens.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            oversized: self.oversized.load(Ordering::Relaxed),
            verify_mismatches: self.verify_mismatches.load(Ordering::Relaxed),
            nodes,
            bytes,
        }
    }

    fn next_tick(&self) -> u64 {
        self.tick.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn lookup(&self, namespace: &Namespace, text: &str, bounds: &[usize]) -> Hit {
        let tick = self.next_tick();
        let tree = self.tree.read();
        match tree.roots.get(namespace) {
            Some(&root) => tree.lookup(root, text, bounds, tick, self.config.hash_strategy),
            None => Hit::default(),
        }
    }

    /// Insert boundaries of `text`; `bounds` are absolute (byte, token)
    /// offsets of every boundary in `text`.
    fn insert(&self, namespace: &Namespace, text: &str, ids: &[u32], bounds: &[(usize, usize)]) {
        let tick = self.next_tick();
        let bytes: Vec<usize> = bounds.iter().map(|&(byte, _)| byte).collect();
        let mut tree = self.tree.write();
        let (before_nodes, before_bytes) = (tree.live, tree.bytes);
        let root = tree.root(namespace, tick);
        let (steps, offset) = tree.walk(root, text, &bytes, self.config.hash_strategy);
        let start = bounds.partition_point(|&(byte, _)| byte <= offset);
        let start_token = start.checked_sub(1).map_or(0, |i| bounds[i].1);
        let mut end = None;
        for (i, &(byte, token)) in bounds
            .iter()
            .enumerate()
            .skip(start)
            .take(self.config.max_insert_segments)
        {
            if (byte - offset) + (token - start_token) * 4 + NODE_OVERHEAD_BYTES
                > self.config.max_insert_bytes
            {
                break;
            }
            end = Some(i);
        }
        if start < bounds.len() && end.is_none() {
            self.oversized.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(end) = end {
            let mut parent = root;
            for step in &steps {
                match *step {
                    Step::Full(id) => {
                        tree.node(id).last_used.store(tick, Ordering::Relaxed);
                        parent = id;
                    }
                    Step::Partial(id, mark) => {
                        parent = tree.split(id, mark, self.config.hash_strategy)
                    }
                }
            }
            let (end_byte, end_token) = bounds[end];
            let marks: Vec<(u32, u32)> = bounds[start..end]
                .iter()
                .map(|&(b, t)| ((b - offset) as u32, (t - start_token) as u32))
                .collect();
            // Growth appends a child; existing edges are never copied.
            let leaf = tree.alloc(Node {
                parent,
                text: Box::from(&text[offset..end_byte]),
                ids: Arc::from(&ids[start_token..end_token]),
                marks: marks.into_boxed_slice(),
                children: HashMap::new(),
                child_count: 0,
                last_used: AtomicU64::new(tick),
                namespace: None,
            });
            tree.add_child(parent, leaf, self.config.hash_strategy);
        }
        if tree.node(root).child_count == 0 {
            if let Some(namespace) = tree.release(root).namespace {
                tree.roots.remove(&namespace);
            }
        }
        let evicted = tree.evict(&self.config);
        if evicted > 0 {
            self.evictions.fetch_add(evicted, Ordering::Relaxed);
            TokenizerMetrics::record_prefix_cache_evictions(evicted);
        }
        publish_occupancy(
            tree.live as isize - before_nodes as isize,
            tree.bytes as isize - before_bytes as isize,
        );
    }
}

impl Drop for PrefixCache {
    fn drop(&mut self) {
        let tree = self.tree.get_mut();
        publish_occupancy(-(tree.live as isize), -(tree.bytes as isize));
    }
}

/// Per-model L1 handle used by `PromptEncoder`.
#[derive(Clone)]
pub(crate) struct PrefixEncoder {
    cache: Arc<PrefixCache>,
    matcher: Arc<BoundaryMatcher>,
    model_id: Arc<str>,
    disabled: Arc<AtomicBool>,
    mismatch_warning: Arc<Once>,
}

impl PrefixEncoder {
    pub(crate) fn new(model_id: &str, cache: Arc<PrefixCache>, matcher: BoundaryMatcher) -> Self {
        Self {
            cache,
            matcher: Arc::new(matcher),
            model_id: Arc::from(model_id),
            disabled: Arc::new(AtomicBool::new(false)),
            mismatch_warning: Arc::new(Once::new()),
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
            return Ok(None);
        }
        let bounds = self.matcher.boundaries(text);
        if bounds.is_empty() {
            return Ok(None);
        }
        let namespace = Namespace {
            instance,
            salt: cache_salt.map(Box::from),
        };
        let hit = self.cache.lookup(&namespace, text, &bounds);
        let mut ids = Vec::with_capacity(hit.tokens + (text.len() - hit.bytes) / 3);
        for (piece, take) in &hit.pieces {
            ids.extend_from_slice(&piece[..*take]);
        }
        let matched_bounds = hit.bounds.len();
        let mut all_bounds = hit.bounds;
        let mut offset = hit.bytes;
        // Segment only what insert will keep. A later cut is still a
        // boundary, so one encode of the remainder has the same IDs.
        let origin = offset;
        let start_tokens = ids.len();
        let fresh = &bounds[bounds.partition_point(|&bound| bound <= offset)..];
        for (stored, &bound) in fresh.iter().enumerate() {
            if stored == self.cache.config.max_insert_segments {
                break;
            }
            let piece = tokenizer.encode(&text[offset..bound], false)?;
            let tokens_after = ids.len() + piece.len();
            let cost = (bound - origin) + (tokens_after - start_tokens) * 4 + NODE_OVERHEAD_BYTES;
            if cost > self.cache.config.max_insert_bytes {
                ids.extend(piece);
                offset = bound;
                break;
            }
            ids.extend(piece);
            all_bounds.push((bound, tokens_after));
            offset = bound;
        }
        if offset < text.len() {
            ids.extend(tokenizer.encode(&text[offset..], false)?);
        }
        if self.cache.config.verify {
            let full = tokenizer.encode(text, false)?;
            if full != ids {
                self.disabled.store(true, Ordering::Relaxed);
                self.cache.verify_mismatches.fetch_add(1, Ordering::Relaxed);
                TokenizerMetrics::record_prefix_cache_verify_mismatch();
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
        self.cache.lookups.fetch_add(1, Ordering::Relaxed);
        self.cache
            .cached_tokens
            .fetch_add(hit.tokens as u64, Ordering::Relaxed);
        self.cache
            .encoded_tokens
            .fetch_add((ids.len() - hit.tokens) as u64, Ordering::Relaxed);
        TokenizerMetrics::record_prefix_cache_lookup(hit.tokens, ids.len() - hit.tokens);
        if hit.tokens > 0 {
            self.cache.hits.fetch_add(1, Ordering::Relaxed);
        }
        if all_bounds.len() > matched_bounds {
            self.cache.insert(&namespace, text, &ids, &all_bounds);
        }
        Ok(Some((ids, hit.tokens)))
    }
}

pub(crate) fn config_from_env() -> Result<Option<PrefixCacheConfig>> {
    parse_config(|name| std::env::var(name).ok())
}

fn parse_flag(name: &str, raw: Option<String>) -> Result<bool> {
    match raw.as_deref() {
        None | Some("0" | "false") => Ok(false),
        Some("1" | "true") => Ok(true),
        _ => bail!("{name} must be 0, 1, false or true"),
    }
}

fn parse_hash_strategy(raw: Option<String>) -> Result<PrefixCacheHashStrategy> {
    match raw.as_deref() {
        None | Some("first_last_64" | "first-last-64" | "first_last" | "first-last") => {
            Ok(PrefixCacheHashStrategy::FirstLast64)
        }
        Some("full") | Some("full_segment") | Some("full-segment") => {
            Ok(PrefixCacheHashStrategy::Full)
        }
        Some(raw) => bail!("VLLM_ROUTER_L1_HASH_STRATEGY must be first_last_64 or full, got {raw}"),
    }
}

fn parse_config(read: impl Fn(&str) -> Option<String>) -> Result<Option<PrefixCacheConfig>> {
    if !parse_flag("VLLM_ROUTER_L1_CACHE", read("VLLM_ROUTER_L1_CACHE"))? {
        return Ok(None);
    }
    let mut config = PrefixCacheConfig {
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

#[cfg(test)]
mod tests {
    use super::*;
    use metrics::{
        Counter, Gauge, GaugeFn, Histogram, Key, KeyName, Metadata, Recorder, SharedString, Unit,
    };
    use serde_json::json;
    use std::sync::Arc;
    use std::thread;
    use vllm_tokenizer::HuggingFaceTokenizer;

    fn spec(content: &str) -> AddedTokenSpec {
        AddedTokenSpec {
            content: content.into(),
            lstrip: false,
            rstrip: false,
            single_word: false,
            normalized: false,
        }
    }

    fn token_set(tokens: Vec<AddedTokenSpec>, identity_normalizer: bool) -> AddedTokenSet {
        AddedTokenSet {
            tokens,
            identity_normalizer,
        }
    }

    fn eligible_in(set: &AddedTokenSet) -> Vec<String> {
        BoundaryMatcher::new(set)
            .map(|m| m.tokens().to_vec())
            .unwrap_or_default()
    }

    fn eligible(tokens: &[&str]) -> Vec<String> {
        eligible_in(&token_set(tokens.iter().map(|t| spec(t)).collect(), true))
    }

    fn normalized(content: &str) -> AddedTokenSpec {
        AddedTokenSpec {
            normalized: true,
            ..spec(content)
        }
    }

    #[test]
    fn boundary_tokens_exclude_overlaps_and_flags() {
        let qwen = [
            "<|im_start|>",
            "<|im_end|>",
            "<tool_call>",
            "</tool_call>",
            "<think>",
            "</think>",
            "<tts_text_bos>",
            "<tts_text_bos_single>",
        ];
        assert_eq!(eligible(&qwen).len(), qwen.len());
        // A longer token containing `t` before its end can win in `P + S`.
        assert_eq!(eligible(&["<a>", "<a>b"]), ["<a>b"]);
        assert_eq!(eligible(&["b", "abc"]), ["abc"]);
        // Ending inside or at the end of another token is decided by the
        // matcher, which sees every added token.
        assert_eq!(eligible(&[">x", "a>x"]), [">x", "a>x"]);
        assert_eq!(eligible(&["xy", "ax"]), ["ax", "xy"]);
        assert_eq!(eligible(&["aa", " <x>", "<x> "]), ["aa"]);
        for flag in 0..3 {
            let mut token = spec("<x>");
            match flag {
                0 => token.lstrip = true,
                1 => token.rstrip = true,
                _ => token.single_word = true,
            }
            assert!(BoundaryMatcher::new(&token_set(vec![token], true)).is_err());
        }
        // A single_word token after a word character depends on the cut.
        let word = AddedTokenSpec {
            single_word: true,
            ..spec("w")
        };
        assert_eq!(
            eligible_in(&token_set(vec![spec("<x>"), spec("<x1"), word], true)),
            ["<x>"]
        );
    }

    #[test]
    fn single_word_guard_matches_hugging_face_word_chars() {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

        let vocab = [("[UNK]".to_string(), 0)].into_iter().collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = HfTokenizer::new(model);
        // One candidate boundary per last character, plus a single_word token
        // that can follow the cut.
        let boundaries: Vec<String> = (0x21u32..0x3000)
            .filter_map(char::from_u32)
            .filter(|c| !c.is_whitespace() && !c.is_control())
            .map(|c| format!("§{c}"))
            .collect();
        let mut added: Vec<AddedToken> = boundaries
            .iter()
            .map(|b| AddedToken::from(b.as_str(), false).normalized(false))
            .collect();
        added.push(
            AddedToken::from("rareword", false)
                .single_word(true)
                .normalized(false),
        );
        tokenizer.add_tokens(&added);

        let json: Value = serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap();
        let set = token_set(added_tokens(&json["added_tokens"], None), true);
        let verdicts: HashMap<&str, Option<&str>> = set
            .tokens
            .iter()
            .map(|t| t.content.as_str())
            .zip(boundary_rejections(&set))
            .collect();

        let encode = |text: &str| tokenizer.encode(text, false).unwrap().get_ids().to_vec();
        let suffix = encode("rareword");
        let mut breaking = 0;
        for boundary in &boundaries {
            let mut split = encode(boundary);
            split.extend(&suffix);
            if encode(&format!("{boundary}rareword")) != split {
                breaking += 1;
                assert_eq!(
                    verdicts[boundary.as_str()],
                    Some("a single_word token could follow"),
                    "{boundary:?} changes the IDs but is a boundary"
                );
            }
        }
        assert!(breaking > 0);
        // Combining acute accent: `\w` but not alphanumeric.
        assert!(verdicts["§\u{301}"].is_some());
        assert_eq!(encode("§\u{301}rareword"), [encode("§\u{301}")[0], 0]);
    }

    #[test]
    fn normalized_tokens_need_identity_normalizer_and_separate_groups() {
        let separate = vec![spec("aa"), normalized("<n>")];
        assert_eq!(
            eligible_in(&token_set(separate.clone(), true)),
            ["<n>", "aa"]
        );
        assert_eq!(eligible_in(&token_set(separate, false)), ["aa"]);
        // `s>` overlaps `<s>`: backends may split differently, so normalized
        // tokens are out and the strict rules apply (no self overlap).
        let mixed = vec![spec("<s>"), spec("aa"), normalized("s>")];
        assert_eq!(eligible_in(&token_set(mixed, true)), ["<s>"]);
        assert!(is_identity_normalizer(&Value::Null));
        assert!(is_identity_normalizer(
            &json!({"type": "Sequence", "normalizers": []})
        ));
        assert!(!is_identity_normalizer(&json!({"type": "NFC"})));
    }

    #[test]
    fn matcher_cuts_only_where_the_tokenizer_splits() {
        let matcher =
            BoundaryMatcher::new(&token_set(vec![spec(">x"), spec("a>x")], true)).unwrap();
        assert_eq!(matcher.boundaries("a>x b>x"), [3, 7]);
        // A non-boundary `a<s` hides the first `<s>` from the tokenizer.
        let matcher =
            BoundaryMatcher::build(vec!["<s>".into(), "a<s".into()], vec!["<s>".into()]).unwrap();
        assert_eq!(matcher.boundaries("a<s> <s>"), [8]);
    }

    #[test]
    fn added_tokens_merge_config_and_default_normalized() {
        let json = json!([{"id": 1, "content": "<a>", "special": true, "normalized": false}]);
        let decoder = json!({
            "1": {"content": "<dup>", "special": true},
            "2": {"content": "<b>", "special": true},
            "3": {"content": "plain"}
        });
        let mut tokens = added_tokens(&json, None);
        tokens.extend(added_tokens(&decoder, Some(&[1])));
        let names: Vec<_> = tokens.iter().map(|t| t.content.as_str()).collect();
        assert_eq!(names, ["<a>", "<b>", "plain"]);
        assert!(!tokens[1].normalized);
        // "plain" omits `normalized`. vllm-tokenizer loads that as false.
        assert!(!tokens[2].normalized);
    }

    #[test]
    fn missing_normalized_does_not_share_a_group_with_an_explicit_token() {
        // "ab" is normalized and "bc" omits the field. Treating the omission
        // as normalized puts both in one group, so a one-pass cut after "ab"
        // disagrees with Hugging Face's two-pass encode of "abc".
        let tokens = added_tokens(
            &json!([
                {"content": "ab", "normalized": true},
                {"content": "bc"}
            ]),
            None,
        );
        assert!(tokens[0].normalized);
        assert!(!tokens[1].normalized);
        assert!(eligible_in(&token_set(tokens, true)).is_empty());
    }

    #[test]
    fn rejects_unsupported_tokenizer_configs() {
        let base =
            json!({"model": {"type": "BPE", "dropout": null}, "truncation": null, "padding": null});
        assert!(check_tokenizer_json(&base).is_ok());
        for (path, value) in [
            ("truncation", json!({"max_length": 8})),
            ("padding", json!({"strategy": "BatchLongest"})),
            ("model", json!({"type": "BPE", "dropout": 0.1})),
            (
                "pre_tokenizer",
                json!({"type": "Sequence", "pretokenizers": [{"type": "Metaspace", "prepend_scheme": "first"}]}),
            ),
        ] {
            let mut json = base.clone();
            json[path] = value;
            assert!(check_tokenizer_json(&json).is_err(), "{path}");
        }
        let mut always = base.clone();
        always["pre_tokenizer"] = json!({"type": "Metaspace", "prepend_scheme": "always"});
        assert!(check_tokenizer_json(&always).is_ok());
    }

    fn fixture_tokenizer() -> HuggingFaceTokenizer {
        HuggingFaceTokenizer::new(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tokenizer/byte_level_bpe.json"
        )))
        .unwrap()
    }

    fn encoder(config: PrefixCacheConfig) -> (PrefixEncoder, Arc<PrefixCache>) {
        let cache = Arc::new(PrefixCache::new(config).unwrap());
        let matcher =
            BoundaryMatcher::new(&token_set(vec![spec("<s>"), spec("</s>")], true)).unwrap();
        (
            PrefixEncoder::new("test-model", cache.clone(), matcher),
            cache,
        )
    }

    fn check(encoder: &PrefixEncoder, tokenizer: &dyn Tokenizer, text: &str) -> usize {
        let (ids, cached) = encoder
            .encode(tokenizer, 0, text, None)
            .unwrap()
            .expect("boundaries");
        assert_eq!(ids, tokenizer.encode(text, false).unwrap(), "{text:?}");
        cached
    }

    #[test]
    fn tree_reuses_longest_boundary_prefix_and_splits_edges() {
        let tokenizer = fixture_tokenizer();
        let (encoder, cache) = encoder(Default::default());
        let a = "<s>system hi</s><s>user one</s><s>assistant";
        assert_eq!(check(&encoder, &tokenizer, a), 0);
        let full = tokenizer.encode(a, false).unwrap();
        // Same prompt: everything up to the last boundary is reused.
        let head = tokenizer
            .encode("<s>system hi</s><s>user one</s><s>", false)
            .unwrap();
        assert_eq!(check(&encoder, &tokenizer, a), head.len());
        assert!(head.len() < full.len());
        // Growing conversation reuses the whole old prefix.
        let b = "<s>system hi</s><s>user one</s><s>assistant ok</s><s>user two</s><s>assistant";
        assert_eq!(check(&encoder, &tokenizer, b), head.len());
        // An edited second message stops at the boundary before it (split).
        let c = "<s>system hi</s><s>user ONE</s><s>assistant";
        let first = tokenizer.encode("<s>system hi</s><s>", false).unwrap();
        assert_eq!(check(&encoder, &tokenizer, c), first.len());
        // Both branches still hit after the split.
        assert!(check(&encoder, &tokenizer, b) > head.len());
        assert!(check(&encoder, &tokenizer, c) > first.len());
        // Text without boundaries, a literal partial token, and trailing text.
        assert!(encoder
            .encode(&tokenizer, 0, "plain", None)
            .unwrap()
            .is_none());
        check(&encoder, &tokenizer, "<s>system hi</s><s>user <s</s>x");
        check(&encoder, &tokenizer, "<s>system hi</s>");
        check(&encoder, &tokenizer, "<s>");
        let stats = cache.stats();
        assert!(stats.hits >= 5 && stats.nodes > 0 && stats.bytes > 0);
    }

    #[test]
    fn growth_appends_a_child_without_copying() {
        let tokenizer = fixture_tokenizer();
        let (encoder, cache) = encoder(Default::default());
        let edges = || -> Vec<(usize, Arc<[u32]>)> {
            let tree = cache.tree.read();
            (0..tree.nodes.len())
                .filter_map(|id| tree.nodes[id].as_ref().map(|node| (id, node.ids.clone())))
                .collect()
        };
        let mut text = String::from("<s>system hi</s>");
        check(&encoder, &tokenizer, &format!("{text}<s>"));
        for turn in 0..5 {
            let before = edges();
            text.push_str(&format!("<s>user {turn}</s><s>assistant {turn}</s>"));
            let cached = check(&encoder, &tokenizer, &format!("{text}<s>"));
            assert!(cached > 0);
            let after = edges();
            assert_eq!(after.len(), before.len() + 1, "one new edge per turn");
            for (id, ids) in &before {
                let (_, now) = after.iter().find(|(other, _)| other == id).unwrap();
                assert!(Arc::ptr_eq(ids, now), "existing edges are untouched");
            }
            check_invariants(&cache);
        }
    }

    #[test]
    fn namespaces_are_isolated() {
        let tokenizer = fixture_tokenizer();
        let (encoder, cache) = encoder(Default::default());
        let text = "<s>system hi</s><s>user</s>";
        for (instance, salt) in [(0, None), (1, None), (0, Some("a")), (0, Some(""))] {
            let (_, cached) = encoder
                .encode(&tokenizer, instance, text, salt)
                .unwrap()
                .unwrap();
            assert_eq!(cached, 0);
            let (_, cached) = encoder
                .encode(&tokenizer, instance, text, salt)
                .unwrap()
                .unwrap();
            assert!(cached > 0);
        }
        assert_eq!(cache.stats().hits, 4);
    }

    #[test]
    fn eviction_keeps_budget_and_recent_paths() {
        let tokenizer = fixture_tokenizer();
        let config = PrefixCacheConfig {
            max_bytes: 4096,
            max_nodes: 16,
            max_insert_bytes: 1024,
            max_insert_segments: 8,
            ..Default::default()
        };
        let (encoder, cache) = encoder(config);
        for i in 0..200 {
            check(
                &encoder,
                &tokenizer,
                &format!("<s>system {}</s><s>user {i}</s><s>a", i % 7),
            );
            let stats = cache.stats();
            assert!(stats.bytes <= config.max_bytes && stats.nodes <= config.max_nodes);
        }
        assert!(cache.stats().evictions > 0);
        let recent = "<s>system 3</s><s>user 199</s><s>a";
        assert!(check(&encoder, &tokenizer, recent) > 0);
        // Long segments past max_insert_bytes are encoded but not stored.
        let before = cache.stats();
        check(
            &encoder,
            &tokenizer,
            &format!("<s>{}</s>", "long ".repeat(400)),
        );
        assert_eq!(cache.stats().oversized, before.oversized + 1);
        assert!(cache.stats().bytes <= config.max_bytes);
        // Segment cap stores only the first segments.
        let many: String = (0..20).map(|i| format!("<s>m{i}</s>")).collect();
        check(&encoder, &tokenizer, &many);
        assert!(check(&encoder, &tokenizer, &many) > 0);
    }

    #[test]
    fn concurrent_encodes_match_uncached() {
        let tokenizer = Arc::new(fixture_tokenizer());
        let (encoder, cache) = encoder(PrefixCacheConfig {
            max_bytes: 64 * 1024,
            max_nodes: 64,
            max_insert_bytes: 8 * 1024,
            ..Default::default()
        });
        std::thread::scope(|scope| {
            for t in 0..8 {
                let (encoder, tokenizer) = (encoder.clone(), tokenizer.clone());
                scope.spawn(move || {
                    for i in 0..200 {
                        let turns: String = (0..(i % 9))
                            .map(|k| format!("<s>user {}</s><s>assistant {k}</s>", (k + t) % 3))
                            .collect();
                        check(
                            &encoder,
                            tokenizer.as_ref(),
                            &format!("<s>sys</s>{turns}<s>"),
                        );
                    }
                });
            }
        });
        let stats = cache.stats();
        assert_eq!(stats.lookups, 1600);
        assert!(stats.hits > 0 && stats.nodes <= 64);
    }

    struct RecordingGauge {
        name: &'static str,
        sets: Arc<Mutex<Vec<&'static str>>>,
    }

    impl GaugeFn for RecordingGauge {
        fn increment(&self, _: f64) {}
        fn decrement(&self, _: f64) {}
        fn set(&self, _: f64) {
            self.sets.lock().push(self.name);
        }
    }

    struct PairRecorder(Arc<Mutex<Vec<&'static str>>>);

    impl Recorder for PairRecorder {
        fn describe_counter(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
        fn describe_gauge(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
        fn describe_histogram(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}

        fn register_counter(&self, _: &Key, _: &Metadata<'_>) -> Counter {
            Counter::noop()
        }

        fn register_gauge(&self, key: &Key, _: &Metadata<'_>) -> Gauge {
            let name = if key.name() == "vllm_tokenizer_prefix_cache_nodes" {
                "nodes"
            } else {
                "bytes"
            };
            Gauge::from_arc(Arc::new(RecordingGauge {
                name,
                sets: self.0.clone(),
            }))
        }

        fn register_histogram(&self, _: &Key, _: &Metadata<'_>) -> Histogram {
            Histogram::noop()
        }
    }

    /// Two caches publish both gauges under one lock, so their `set` calls
    /// cannot interleave.
    #[test]
    fn occupancy_publishes_nodes_and_bytes_together() {
        let sets = Arc::new(Mutex::new(Vec::new()));
        let tokenizer = Arc::new(fixture_tokenizer());
        thread::scope(|scope| {
            for t in 0..4 {
                let (sets, tokenizer) = (sets.clone(), tokenizer.clone());
                scope.spawn(move || {
                    metrics::with_local_recorder(&PairRecorder(sets), || {
                        let (encoder, _cache) = encoder(PrefixCacheConfig {
                            max_bytes: 64 * 1024,
                            max_nodes: 64,
                            max_insert_bytes: 8 * 1024,
                            ..Default::default()
                        });
                        check(
                            &encoder,
                            tokenizer.as_ref(),
                            &format!("<s>user {t}</s><s>tail {t}</s>"),
                        );
                    });
                });
            }
        });
        let sets = sets.lock();
        assert!(sets.len() >= 8, "{sets:?}");
        for pair in sets.chunks(2) {
            assert_eq!(pair, ["nodes", "bytes"], "{sets:?}");
        }
    }

    fn lookup_report_node(parent: usize, text: String) -> Node {
        Node {
            parent,
            text: Box::from(text),
            ids: Arc::from([]),
            marks: Box::from([]),
            children: HashMap::new(),
            child_count: 0,
            last_used: AtomicU64::new(0),
            namespace: None,
        }
    }

    fn timed_lookup(
        iters: usize,
        mut lookup: impl FnMut() -> Option<usize>,
    ) -> std::time::Duration {
        let start = std::time::Instant::now();
        let mut hits = 0usize;
        for _ in 0..iters {
            if std::hint::black_box(lookup()).is_some() {
                hits += 1;
            }
        }
        let elapsed = start.elapsed();
        assert_eq!(hits, iters);
        elapsed
    }

    fn timed_encode(
        iters: usize,
        encoder: &PrefixEncoder,
        tokenizer: &dyn Tokenizer,
        text: &str,
    ) -> std::time::Duration {
        let start = std::time::Instant::now();
        let mut total = 0usize;
        for _ in 0..iters {
            let (ids, cached) = encoder
                .encode(tokenizer, 0, text, None)
                .unwrap()
                .expect("text has boundaries");
            total = total.wrapping_add(ids.len()).wrapping_add(cached);
        }
        let elapsed = start.elapsed();
        std::hint::black_box(total);
        elapsed
    }

    fn normal_long_segment(i: usize, len: usize) -> String {
        let head = "h".repeat(64);
        let tail = format!("{i:064x}");
        let middle = format!("normal-{i}-").repeat((len - 128) / 9 + 1);
        format!("{head}{}{tail}", &middle[..len - 128])
    }

    fn adversarial_long_segment(i: usize, len: usize) -> String {
        let head = "h".repeat(64);
        let tail = "t".repeat(64);
        let marker = format!("{i:04x}");
        let middle = format!("{}{}", "m".repeat(len - 128 - marker.len()), marker);
        format!("{head}{middle}{tail}")
    }

    fn build_lookup_tree(segments: &[String], strategy: PrefixCacheHashStrategy) -> (Tree, usize) {
        let mut tree = Tree::default();
        let root = tree.alloc(Node {
            parent: NO_PARENT,
            text: Box::from(""),
            ids: Arc::from([]),
            marks: Box::from([]),
            children: HashMap::new(),
            child_count: 0,
            last_used: AtomicU64::new(0),
            namespace: None,
        });
        for segment in segments {
            let id = tree.alloc(lookup_report_node(root, segment.clone()));
            tree.add_child(root, id, strategy);
        }
        (tree, root)
    }

    fn hash_report_encoder(strategy: PrefixCacheHashStrategy) -> (PrefixEncoder, Arc<PrefixCache>) {
        encoder(PrefixCacheConfig {
            max_bytes: 512 * 1024 * 1024,
            max_nodes: 1_000_000,
            max_insert_bytes: 256 * 1024 * 1024,
            max_insert_segments: 1_000_000,
            hash_strategy: strategy,
            ..Default::default()
        })
    }

    fn hash_report_prompt(segment: &str) -> String {
        format!("<s>{segment}</s>")
    }

    fn warm_hash_report_encoder(
        encoder: &PrefixEncoder,
        tokenizer: &dyn Tokenizer,
        segments: &[String],
    ) {
        for segment in segments {
            check(encoder, tokenizer, &hash_report_prompt(segment));
        }
    }

    fn strategy_name(strategy: PrefixCacheHashStrategy) -> &'static str {
        match strategy {
            PrefixCacheHashStrategy::FirstLast64 => "first_last_64",
            PrefixCacheHashStrategy::Full => "full",
        }
    }

    fn prompt_tokens(tokenizer: &dyn Tokenizer, segment: &str) -> usize {
        tokenizer
            .encode(&hash_report_prompt(segment), false)
            .unwrap()
            .len()
    }

    fn byte_len_for_prompt_tokens(tokenizer: &dyn Tokenizer, target_tokens: usize) -> usize {
        let mut high = 129usize;
        while prompt_tokens(tokenizer, &normal_long_segment(0, high)) < target_tokens {
            high *= 2;
        }
        let mut low = 129usize;
        while low < high {
            let mid = (low + high) / 2;
            if prompt_tokens(tokenizer, &normal_long_segment(0, mid)) < target_tokens {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        low
    }

    /// CPU-only lookup and encode report for the current first/last-64 child
    /// key versus the full-segment hash strategy.
    ///
    /// Run with:
    /// `cargo test --release --lib hash_strategy_lookup_cost_report -- --ignored --nocapture`
    #[test]
    #[ignore = "CPU-only performance report; run in release mode"]
    fn hash_strategy_lookup_cost_report() {
        fn report_lookup_row(
            case: &str,
            len: usize,
            siblings: usize,
            first_last_bucket: usize,
            iters: usize,
            first_last: std::time::Duration,
            full_hash: std::time::Duration,
        ) {
            let first_last_ns = first_last.as_nanos() as f64 / iters as f64;
            let full_hash_ns = full_hash.as_nanos() as f64 / iters as f64;
            println!(
                "lookup {case:11} len={len:6} siblings={siblings:2} first_last_bucket={first_last_bucket:2} iters={iters:5} first_last_64={first_last_ns:9.1}ns full={full_hash_ns:9.1}ns ratio={:.2}",
                first_last_ns / full_hash_ns
            );
        }

        fn report_encode_row(
            case: &str,
            len: usize,
            siblings: usize,
            iters: usize,
            first_last: std::time::Duration,
            full_hash: std::time::Duration,
        ) {
            let first_last_us = first_last.as_secs_f64() * 1_000_000.0 / iters as f64;
            let full_hash_us = full_hash.as_secs_f64() * 1_000_000.0 / iters as f64;
            println!(
                "encode {case:11} len={len:6} siblings={siblings:2} iters={iters:5} first_last_64={first_last_us:9.2}us full={full_hash_us:9.2}us ratio={:.2}",
                first_last_us / full_hash_us
            );
        }

        println!(
            "hash strategy matrix; lookup rows include the exact `first_segment() == segment` check"
        );
        println!(
            "normal: siblings share the first 64 bytes but have different last 64 bytes, so current bucket is 1"
        );
        println!(
            "adversarial: siblings share length, first 64 bytes and last 64 bytes, so current bucket equals siblings"
        );

        let tokenizer = fixture_tokenizer();
        for len in [16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024] {
            for siblings in [1, 4, 8, 16, 32] {
                for (case, segments) in [
                    (
                        "normal",
                        (0..siblings)
                            .map(|i| normal_long_segment(i, len))
                            .collect::<Vec<_>>(),
                    ),
                    (
                        "adversarial",
                        (0..siblings)
                            .map(|i| adversarial_long_segment(i, len))
                            .collect::<Vec<_>>(),
                    ),
                ] {
                    let query = &segments[siblings - 1];
                    assert!(query.len() > 128);

                    let (first_last_tree, first_last_root) =
                        build_lookup_tree(&segments, PrefixCacheHashStrategy::FirstLast64);
                    let (full_tree, full_root) =
                        build_lookup_tree(&segments, PrefixCacheHashStrategy::Full);
                    let first_last_bucket = first_last_tree
                        .node(first_last_root)
                        .children
                        .get(&segment_key(query, PrefixCacheHashStrategy::FirstLast64))
                        .map_or(0, Vec::len);
                    assert_eq!(
                        first_last_bucket,
                        if case == "normal" { 1 } else { siblings }
                    );
                    let lookup_iters = if case == "normal" {
                        (64 * 1024 * 1024 / len).max(512)
                    } else {
                        (64 * 1024 * 1024 / (len * siblings)).max(32)
                    };
                    let first_last_lookup = timed_lookup(lookup_iters, || {
                        first_last_tree.child(
                            first_last_root,
                            query,
                            PrefixCacheHashStrategy::FirstLast64,
                        )
                    });
                    let full_lookup = timed_lookup(lookup_iters, || {
                        full_tree.child(full_root, query, PrefixCacheHashStrategy::Full)
                    });
                    report_lookup_row(
                        case,
                        len,
                        siblings,
                        first_last_bucket,
                        lookup_iters,
                        first_last_lookup,
                        full_lookup,
                    );

                    let (first_last_encoder, _) =
                        hash_report_encoder(PrefixCacheHashStrategy::FirstLast64);
                    let (full_encoder, _) = hash_report_encoder(PrefixCacheHashStrategy::Full);
                    warm_hash_report_encoder(&first_last_encoder, &tokenizer, &segments);
                    warm_hash_report_encoder(&full_encoder, &tokenizer, &segments);
                    let query_text = hash_report_prompt(query);
                    let (ids, cached) = first_last_encoder
                        .encode(&tokenizer, 0, &query_text, None)
                        .unwrap()
                        .expect("query has boundaries");
                    assert_eq!(ids, tokenizer.encode(&query_text, false).unwrap());
                    assert_eq!(cached, ids.len());
                    let (ids, cached) = full_encoder
                        .encode(&tokenizer, 0, &query_text, None)
                        .unwrap()
                        .expect("query has boundaries");
                    assert_eq!(ids, tokenizer.encode(&query_text, false).unwrap());
                    assert_eq!(cached, ids.len());
                    let encode_iters = (8 * 1024 * 1024 / len).max(8);
                    let first_last_encode =
                        timed_encode(encode_iters, &first_last_encoder, &tokenizer, &query_text);
                    let full_encode =
                        timed_encode(encode_iters, &full_encoder, &tokenizer, &query_text);
                    report_encode_row(
                        case,
                        len,
                        siblings,
                        encode_iters,
                        first_last_encode,
                        full_encode,
                    );
                }
            }
        }

        println!(
            "Set VLLM_ROUTER_L1_HASH_STRATEGY={} or {} to switch the runtime strategy.",
            strategy_name(PrefixCacheHashStrategy::FirstLast64),
            strategy_name(PrefixCacheHashStrategy::Full)
        );
    }

    /// Same report as `hash_strategy_lookup_cost_report`, but targets prompt
    /// token counts rather than raw segment byte lengths.
    ///
    /// Run with:
    /// `cargo test --release --lib hash_strategy_token_count_report -- --ignored --nocapture`
    #[test]
    #[ignore = "CPU-only performance report; run in release mode"]
    fn hash_strategy_token_count_report() {
        #[derive(Clone, Copy)]
        struct ReportDims {
            case: &'static str,
            target_tokens: usize,
            actual_tokens: usize,
            bytes: usize,
            siblings: usize,
        }

        fn report_lookup_row(
            dims: ReportDims,
            first_last_bucket: usize,
            iters: usize,
            first_last: std::time::Duration,
            full_hash: std::time::Duration,
        ) {
            let first_last_ns = first_last.as_nanos() as f64 / iters as f64;
            let full_hash_ns = full_hash.as_nanos() as f64 / iters as f64;
            let ReportDims {
                case,
                target_tokens,
                actual_tokens,
                bytes,
                siblings,
            } = dims;
            println!(
                "lookup {case:11} target_tokens={target_tokens:6} actual_tokens={actual_tokens:6} bytes={bytes:7} siblings={siblings:2} first_last_bucket={first_last_bucket:2} iters={iters:4} first_last_64={first_last_ns:10.1}ns full={full_hash_ns:10.1}ns ratio={:.2}",
                first_last_ns / full_hash_ns
            );
        }

        fn report_encode_row(
            dims: ReportDims,
            iters: usize,
            first_last: std::time::Duration,
            full_hash: std::time::Duration,
        ) {
            let first_last_us = first_last.as_secs_f64() * 1_000_000.0 / iters as f64;
            let full_hash_us = full_hash.as_secs_f64() * 1_000_000.0 / iters as f64;
            let ReportDims {
                case,
                target_tokens,
                actual_tokens,
                bytes,
                siblings,
            } = dims;
            println!(
                "encode {case:11} target_tokens={target_tokens:6} actual_tokens={actual_tokens:6} bytes={bytes:7} siblings={siblings:2} iters={iters:4} first_last_64={first_last_us:10.2}us full={full_hash_us:10.2}us ratio={:.2}",
                first_last_us / full_hash_us
            );
        }

        println!("hash strategy token-count matrix; target length is prompt token count, not KiB");
        let tokenizer = fixture_tokenizer();
        for target_tokens in [16_384, 32_768, 65_536, 131_072] {
            let bytes = byte_len_for_prompt_tokens(&tokenizer, target_tokens);
            for siblings in [1, 4, 8, 16, 32] {
                for (case, segments) in [
                    (
                        "normal",
                        (0..siblings)
                            .map(|i| normal_long_segment(i, bytes))
                            .collect::<Vec<_>>(),
                    ),
                    (
                        "adversarial",
                        (0..siblings)
                            .map(|i| adversarial_long_segment(i, bytes))
                            .collect::<Vec<_>>(),
                    ),
                ] {
                    let query = &segments[siblings - 1];
                    let query_text = hash_report_prompt(query);
                    let actual_tokens = tokenizer.encode(&query_text, false).unwrap().len();
                    let dims = ReportDims {
                        case,
                        target_tokens,
                        actual_tokens,
                        bytes,
                        siblings,
                    };
                    let (first_last_tree, first_last_root) =
                        build_lookup_tree(&segments, PrefixCacheHashStrategy::FirstLast64);
                    let (full_tree, full_root) =
                        build_lookup_tree(&segments, PrefixCacheHashStrategy::Full);
                    let first_last_bucket = first_last_tree
                        .node(first_last_root)
                        .children
                        .get(&segment_key(query, PrefixCacheHashStrategy::FirstLast64))
                        .map_or(0, Vec::len);
                    assert_eq!(
                        first_last_bucket,
                        if case == "normal" { 1 } else { siblings }
                    );
                    let lookup_iters = if case == "normal" {
                        (64 * 1024 * 1024 / bytes).max(64)
                    } else {
                        (64 * 1024 * 1024 / (bytes * siblings)).max(16)
                    };
                    let first_last_lookup = timed_lookup(lookup_iters, || {
                        first_last_tree.child(
                            first_last_root,
                            query,
                            PrefixCacheHashStrategy::FirstLast64,
                        )
                    });
                    let full_lookup = timed_lookup(lookup_iters, || {
                        full_tree.child(full_root, query, PrefixCacheHashStrategy::Full)
                    });
                    report_lookup_row(
                        dims,
                        first_last_bucket,
                        lookup_iters,
                        first_last_lookup,
                        full_lookup,
                    );

                    let (first_last_encoder, _) =
                        hash_report_encoder(PrefixCacheHashStrategy::FirstLast64);
                    let (full_encoder, _) = hash_report_encoder(PrefixCacheHashStrategy::Full);
                    warm_hash_report_encoder(&first_last_encoder, &tokenizer, &segments);
                    warm_hash_report_encoder(&full_encoder, &tokenizer, &segments);
                    let (ids, cached) = first_last_encoder
                        .encode(&tokenizer, 0, &query_text, None)
                        .unwrap()
                        .expect("query has boundaries");
                    assert_eq!(ids, tokenizer.encode(&query_text, false).unwrap());
                    assert_eq!(cached, ids.len());
                    let (ids, cached) = full_encoder
                        .encode(&tokenizer, 0, &query_text, None)
                        .unwrap()
                        .expect("query has boundaries");
                    assert_eq!(ids, tokenizer.encode(&query_text, false).unwrap());
                    assert_eq!(cached, ids.len());
                    let encode_iters = (1_000_000 / target_tokens).max(4);
                    let first_last_encode =
                        timed_encode(encode_iters, &first_last_encoder, &tokenizer, &query_text);
                    let full_encode =
                        timed_encode(encode_iters, &full_encoder, &tokenizer, &query_text);
                    report_encode_row(dims, encode_iters, first_last_encode, full_encode);
                }
            }
        }
    }

    fn check_invariants(cache: &PrefixCache) {
        let tree = cache.tree.read();
        let live: Vec<usize> = (0..tree.nodes.len())
            .filter(|&id| tree.nodes[id].is_some())
            .collect();
        assert_eq!(live.len(), tree.live);
        assert_eq!(
            live.iter().map(|&id| tree.node(id).bytes()).sum::<usize>(),
            tree.bytes
        );
        for &id in &live {
            let node = tree.node(id);
            let children: Vec<usize> = node.children.values().flatten().copied().collect();
            assert_eq!(children.len(), node.child_count);
            for (key, ids) in &node.children {
                for &child in ids {
                    assert_eq!(tree.node(child).parent, id);
                    assert!(
                        segment_key(tree.node(child).first_segment(), cache.config.hash_strategy)
                            == *key
                    );
                }
            }
            if let Some(namespace) = &node.namespace {
                assert_eq!(tree.roots[namespace], id);
                assert!(node.child_count > 0, "empty roots are removed");
                continue;
            }
            assert!(!node.text.is_empty());
            let mut prev = (0, 0);
            for &(byte, token) in node.marks.iter() {
                assert!(byte > prev.0 && token > prev.1);
                assert!((byte as usize) < node.text.len() && (token as usize) <= node.ids.len());
                prev = (byte, token);
            }
        }
    }

    #[test]
    fn randomized_conversations_match_uncached_and_keep_invariants() {
        let tokenizer = fixture_tokenizer();
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |n: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % n as u64) as usize
        };
        let words = [
            "hi", " there", "\n", "数", "😀", "<s", "s>", "</", " \t", "1", "é",
        ];
        for budget in [512, 4096, 1 << 20] {
            let (encoder, cache) = encoder(PrefixCacheConfig {
                max_bytes: budget,
                max_nodes: 24,
                max_insert_bytes: budget / 2,
                max_insert_segments: 6,
                ..Default::default()
            });
            let mut conversations: Vec<Vec<String>> = vec![Vec::new(); 4];
            for _ in 0..400 {
                let conversation = &mut conversations[next(4)];
                match next(4) {
                    0 if !conversation.is_empty() => {
                        let at = next(conversation.len());
                        conversation.truncate(at);
                    }
                    1 if !conversation.is_empty() => {
                        let at = next(conversation.len());
                        conversation[at].push_str(words[next(words.len())]);
                    }
                    _ => {
                        conversation.push((0..next(4)).map(|_| words[next(words.len())]).collect())
                    }
                }
                let mut text: String = conversation
                    .iter()
                    .map(|message| format!("<s>{message}</s>"))
                    .collect();
                text.push_str(["", "<s>", "<s>assistant", "tail"][next(4)]);
                if let Some((ids, _)) = encoder.encode(&tokenizer, 0, &text, None).unwrap() {
                    assert_eq!(ids, tokenizer.encode(&text, false).unwrap(), "{text:?}");
                }
                check_invariants(&cache);
                let stats = cache.stats();
                assert!(stats.bytes <= budget && stats.nodes <= 24);
            }
            assert!(cache.stats().hits > 0);
        }
    }

    /// Prints every added token with its L1 verdict for the models in
    /// `VLLM_ROUTER_L1_TOKENIZERS` (local dirs or hub IDs).
    #[tokio::test]
    async fn report_boundary_verdicts() {
        let Ok(models) = std::env::var("VLLM_ROUTER_L1_TOKENIZERS") else {
            eprintln!("skip: set VLLM_ROUTER_L1_TOKENIZERS");
            return;
        };
        for model in models.split(',').filter(|m| !m.is_empty()) {
            let added = match load_added_tokens(model).await {
                Ok(added) => added,
                Err(error) => {
                    eprintln!("== {model}: L1 unavailable: {error:#}");
                    continue;
                }
            };
            let files = ResolvedModelFiles::new(model).await.unwrap();
            let TokenizerSource::HuggingFace(path) = files.tokenizer else {
                unreachable!()
            };
            let tokenizer = HuggingFaceTokenizer::new(&path).unwrap();
            let probed = BoundaryMatcher::new(&added)
                .and_then(|matcher| matcher.probed(&tokenizer))
                .map(|matcher| matcher.tokens().to_vec())
                .unwrap_or_default();
            eprintln!(
                "== {model}: {} added tokens, {} boundaries",
                added.tokens.len(),
                probed.len()
            );
            for (token, rejection) in added.tokens.iter().zip(boundary_rejections(&added)) {
                let verdict = match rejection {
                    Some(reason) => format!("no   {reason}"),
                    None if probed.contains(&token.content) => "YES".into(),
                    None => "no   failed load-time probe".into(),
                };
                eprintln!("  {verdict:<55} {:?}", token.content);
            }
        }
    }

    /// Real tokenizers: `VLLM_ROUTER_L1_TOKENIZERS=a,b` (local dirs or hub
    /// IDs). Random prompts mix boundary tokens, all other added tokens,
    /// fragments of added tokens and adversarial text.
    #[tokio::test]
    async fn randomized_prompts_match_real_tokenizers() {
        let Ok(dirs) = std::env::var("VLLM_ROUTER_L1_TOKENIZERS") else {
            eprintln!("skip: set VLLM_ROUTER_L1_TOKENIZERS");
            return;
        };
        for dir in dirs.split(',').filter(|d| !d.is_empty()) {
            let added = match load_added_tokens(dir).await {
                Ok(added) => added,
                Err(error) => {
                    eprintln!("{dir}: L1 unavailable: {error}");
                    continue;
                }
            };
            let files = ResolvedModelFiles::new(dir).await.unwrap();
            let TokenizerSource::HuggingFace(path) = files.tokenizer else {
                unreachable!()
            };
            let tokenizer = HuggingFaceTokenizer::new(&path).unwrap();
            let matcher = match BoundaryMatcher::new(&added) {
                Ok(matcher) => matcher.probed(&tokenizer).unwrap(),
                Err(error) => {
                    eprintln!("{dir}: L1 unavailable: {error}");
                    continue;
                }
            };
            let added: Vec<String> = added
                .tokens
                .into_iter()
                .map(|token| token.content)
                .filter(|content| !content.is_empty())
                .collect();
            let boundaries = matcher.tokens().to_vec();
            let cache = Arc::new(
                PrefixCache::new(PrefixCacheConfig {
                    max_bytes: 1 << 20,
                    max_nodes: 512,
                    max_insert_bytes: 1 << 18,
                    ..Default::default()
                })
                .unwrap(),
            );
            let encoder = PrefixEncoder::new(dir, cache.clone(), matcher);
            let words = [
                "hello", " world", "  ", "\n", "\n\n", "\t", "数据", "😀", "é", "\u{301}",
                "\u{338}", ">", "<", "|", "▁", "Ġ", "'s", "123", " 4", "<|", "|>", "</", "\u{a0}",
                "ﬁ", "Å",
            ];
            let mut state = 0x9e37_79b9_7f4a_7c15u64;
            let mut next = move |n: usize| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % n as u64) as usize
            };
            let mut history: Vec<String> = Vec::new();
            for _ in 0..3000 {
                let mut text = match next(3) {
                    0 if !history.is_empty() => history[next(history.len())].clone(),
                    _ => String::new(),
                };
                for _ in 0..1 + next(12) {
                    match next(5) {
                        0 => text.push_str(&boundaries[next(boundaries.len())]),
                        1 => text.push_str(&added[next(added.len())]),
                        2 => {
                            // A prefix or suffix of an added token, so that
                            // longer tokens can form across neighbours.
                            let token = &added[next(added.len())];
                            let cuts: Vec<usize> = token.char_indices().map(|(i, _)| i).collect();
                            let at = cuts[next(cuts.len())];
                            text.push_str(if next(2) == 0 {
                                &token[..at]
                            } else {
                                &token[at..]
                            });
                        }
                        _ => text.push_str(words[next(words.len())]),
                    }
                }
                let full = tokenizer.encode(&text, false).unwrap();
                if let Some((ids, _)) = encoder.encode(&tokenizer, 0, &text, None).unwrap() {
                    assert_eq!(ids, full, "{dir}: {text:?}");
                }
                history.push(text);
            }
            let stats = cache.stats();
            eprintln!("{dir}: {} boundaries, {stats:?}", boundaries.len());
            assert!(stats.hits > 0);
        }
    }

    #[test]
    fn tail_beyond_the_insert_cap_is_one_encode() {
        use std::sync::atomic::AtomicUsize;

        struct Counting {
            inner: HuggingFaceTokenizer,
            calls: AtomicUsize,
        }
        impl Tokenizer for Counting {
            fn encode(&self, text: &str, special: bool) -> vllm_tokenizer::Result<Vec<u32>> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                self.inner.encode(text, special)
            }
            fn encode_ordinary(&self, text: &str) -> vllm_tokenizer::Result<Vec<u32>> {
                self.inner.encode_ordinary(text)
            }
            fn decode(&self, ids: &[u32], special: bool) -> vllm_tokenizer::Result<String> {
                self.inner.decode(ids, special)
            }
            fn token_to_id(&self, token: &str) -> Option<u32> {
                self.inner.token_to_id(token)
            }
            fn id_to_token(&self, id: u32) -> Option<String> {
                self.inner.id_to_token(id)
            }
        }

        let tokenizer = Counting {
            inner: fixture_tokenizer(),
            calls: AtomicUsize::new(0),
        };
        let (encoder, _) = encoder(PrefixCacheConfig {
            max_insert_segments: 2,
            ..Default::default()
        });
        let text = "<s>one</s><s>two</s><s>three</s><s>four</s>tail";
        let (ids, cached) = encoder.encode(&tokenizer, 0, text, None).unwrap().unwrap();
        assert_eq!(ids, tokenizer.inner.encode(text, false).unwrap());
        assert_eq!(cached, 0);
        // Two stored segments, then one encode of every later cut.
        assert_eq!(tokenizer.calls.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn probe_drops_boundaries_that_change_encoding() {
        struct Merging(HuggingFaceTokenizer);
        impl Tokenizer for Merging {
            // Pretend `<s>` merges with a following "hi".
            fn encode(&self, text: &str, special: bool) -> vllm_tokenizer::Result<Vec<u32>> {
                self.0.encode(&text.replace("<s>hi", "<s>"), special)
            }
            fn encode_ordinary(&self, text: &str) -> vllm_tokenizer::Result<Vec<u32>> {
                self.0.encode_ordinary(text)
            }
            fn decode(&self, ids: &[u32], special: bool) -> vllm_tokenizer::Result<String> {
                self.0.decode(ids, special)
            }
            fn token_to_id(&self, token: &str) -> Option<u32> {
                self.0.token_to_id(token)
            }
            fn id_to_token(&self, id: u32) -> Option<String> {
                self.0.id_to_token(id)
            }
        }
        let set = token_set(vec![spec("<s>"), spec("</s>")], true);
        let probed = BoundaryMatcher::new(&set)
            .unwrap()
            .probed(&Merging(fixture_tokenizer()))
            .unwrap();
        assert_eq!(probed.tokens(), ["</s>"]);
        let all = BoundaryMatcher::new(&set).unwrap();
        assert_eq!(all.probed(&fixture_tokenizer()).unwrap().tokens().len(), 2);
    }

    #[test]
    fn verify_mode_disables_l1_on_mismatch() {
        struct Drifting(HuggingFaceTokenizer);
        impl Tokenizer for Drifting {
            fn encode(&self, text: &str, special: bool) -> vllm_tokenizer::Result<Vec<u32>> {
                let mut ids = self.0.encode(text, special)?;
                if text.len() > 20 {
                    ids.push(1);
                }
                Ok(ids)
            }
            fn encode_ordinary(&self, text: &str) -> vllm_tokenizer::Result<Vec<u32>> {
                self.0.encode_ordinary(text)
            }
            fn decode(&self, ids: &[u32], special: bool) -> vllm_tokenizer::Result<String> {
                self.0.decode(ids, special)
            }
            fn token_to_id(&self, token: &str) -> Option<u32> {
                self.0.token_to_id(token)
            }
            fn id_to_token(&self, id: u32) -> Option<String> {
                self.0.id_to_token(id)
            }
        }
        let tokenizer = Drifting(fixture_tokenizer());
        let (encoder, cache) = encoder(PrefixCacheConfig {
            verify: true,
            ..Default::default()
        });
        let text = "<s>system hello there</s><s>user hi</s>";
        let (ids, cached) = encoder.encode(&tokenizer, 0, text, None).unwrap().unwrap();
        assert_eq!((ids, cached), (tokenizer.encode(text, false).unwrap(), 0));
        assert_eq!(cache.stats().verify_mismatches, 1);
        assert!(encoder.encode(&tokenizer, 0, text, None).unwrap().is_none());
    }

    #[test]
    fn config_is_opt_in_and_validated() {
        let parse = |values: &[(&str, &str)]| {
            parse_config(|key| {
                values
                    .iter()
                    .find(|(k, _)| *k == key)
                    .map(|(_, v)| v.to_string())
            })
        };
        assert_eq!(parse(&[]).unwrap(), None);
        assert_eq!(parse(&[("VLLM_ROUTER_L1_MAX_BYTES", "bad")]).unwrap(), None);
        assert_eq!(
            parse(&[("VLLM_ROUTER_L1_CACHE", "1")]).unwrap(),
            Some(Default::default())
        );
        let small = parse(&[
            ("VLLM_ROUTER_L1_CACHE", "true"),
            ("VLLM_ROUTER_L1_MAX_BYTES", "1000"),
        ])
        .unwrap()
        .unwrap();
        assert_eq!((small.max_bytes, small.max_insert_bytes), (1000, 1000));
        assert!(
            parse(&[
                ("VLLM_ROUTER_L1_CACHE", "1"),
                ("VLLM_ROUTER_L1_VERIFY", "1")
            ])
            .unwrap()
            .unwrap()
            .verify
        );
        assert_eq!(
            parse(&[
                ("VLLM_ROUTER_L1_CACHE", "1"),
                ("VLLM_ROUTER_L1_HASH_STRATEGY", "full")
            ])
            .unwrap()
            .unwrap()
            .hash_strategy,
            PrefixCacheHashStrategy::Full
        );
        for (name, value) in [
            ("VLLM_ROUTER_L1_CACHE", "maybe"),
            ("VLLM_ROUTER_L1_MAX_NODES", "0"),
            ("VLLM_ROUTER_L1_MAX_INSERT_SEGMENTS", "x"),
            ("VLLM_ROUTER_L1_VERIFY", "yes"),
            ("VLLM_ROUTER_L1_HASH_STRATEGY", "middle"),
        ] {
            let mut values = vec![("VLLM_ROUTER_L1_CACHE", "1")];
            values.retain(|(k, _)| *k != name);
            values.push((name, value));
            assert!(parse(&values).is_err(), "{name}={value}");
        }
        assert!(parse(&[
            ("VLLM_ROUTER_L1_CACHE", "1"),
            ("VLLM_ROUTER_L1_MAX_BYTES", "10"),
            ("VLLM_ROUTER_L1_MAX_INSERT_BYTES", "11"),
        ])
        .is_err());
    }
}
