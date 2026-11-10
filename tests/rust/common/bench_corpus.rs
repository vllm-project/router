//! Deterministic prompts shared by Criterion, the overhead harness and CI.
//!
//! Requests depend only on [`SEED`], input size and request index. Text
//! prompts have exactly the requested byte length; [`IdCorpus`] uses ID counts.
//! A pool of distinct bodies avoids repeatedly exercising the same content
//! in substring searches, hashing and parsing (see [`body_pool_len`]).
//!
//! Corpora:
//! - `hot64`: cycle through 64 prompts.
//! - `cold`: put a unique marker at the start of every prompt.
//! - `mixed90`: 90% hot, 10% cold.
//! - `short_shared_prefix`: share up to 600 bytes, then a unique tail.
//! - `long_shared_prefix`: share everything except a unique 64-byte tail.
//! - `utf8_hot64`: hot prompts with CJK, kana and emoji after the ASCII marker.
//!
//! Other text corpora are ASCII. See `docs/benchmarks/router_overhead.md`.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};
use std::ops::Range;
use vllm_router_rs::protocols::spec::{ChatCompletionRequest, CompletionRequest};

/// Seed for every generator in this module.
pub const SEED: u64 = 0x244;
/// Prompt sizes (bytes) used by the benchmark matrix.
pub const SIZES: [usize; 3] = [200, 2048, 16384];
/// Long prompt sizes (bytes) for the edge-case benchmarks and tests.
pub const LONG_SIZES: [usize; 3] = [128 * 1024, 512 * 1024, 1024 * 1024];
/// Pre-tokenized prompt lengths (token ids) for the edge-case benchmarks.
pub const PROMPT_ID_COUNTS: [usize; 2] = [16_384, 131_072];
/// Range token ids are drawn from. Every id has five decimal digits, so a
/// serialized id prompt's size depends only on how many ids it has.
pub const PROMPT_ID_RANGE: Range<u32> = 10_000..32_000;
/// Number of distinct prompts in the `hot64` set.
pub const HOT_SET_SIZE: usize = 64;
/// Longest shared prefix (bytes) in `short_shared_prefix`.
pub const SHORT_SHARED_PREFIX_MAX_BYTES: usize = 600;
/// Unique tail (bytes) at the end of every `long_shared_prefix` prompt.
pub const LONG_SHARED_PREFIX_TAIL_BYTES: usize = 64;
/// Most distinct bodies that non-shared text is drawn from.
pub const BODY_POOL_SIZE: usize = 256;
/// Fewest distinct bodies, however long the prompts.
pub const BODY_POOL_MIN_SIZE: usize = 8;
/// Target memory budget, subject to [`BODY_POOL_MIN_SIZE`].
pub const BODY_POOL_MAX_BYTES: usize = 64 * 1024 * 1024;
/// Model name carried by every request.
pub const MODEL: &str = "mock-model";
/// `max_tokens` carried by every generation request.
pub const MAX_TOKENS: u32 = 16;

const WORDS: &[&str] = &[
    "router", "worker", "prefix", "cache", "token", "request", "latency", "budget", "policy",
    "stream", "decode", "prefill", "batch", "queue", "health", "retry", "timeout", "session",
    "model", "prompt", "answer", "system", "user", "memory", "block", "index", "hash", "ring",
    "shard", "lease", "metric", "sample", "window", "cursor", "buffer", "socket", "header",
    "payload", "schema", "vector", "matrix", "tensor", "kernel", "device", "engine", "planner",
    "signal", "config", "default", "option", "record", "report", "summary", "detail", "context",
];

/// Words for `utf8_hot64`: three-byte CJK and kana, four-byte emoji. None
/// needs escaping in JSON, so a serialized body grows by exactly the prompt
/// bytes.
const UTF8_WORDS: &[&str] = &[
    "路由",
    "缓存",
    "前缀",
    "令牌",
    "请求",
    "延迟",
    "预算",
    "策略",
    "解码",
    "批次",
    "队列",
    "健康",
    "重试",
    "会话",
    "模型",
    "提示",
    "回答",
    "系统",
    "用户",
    "内存",
    "索引",
    "哈希",
    "分片",
    "指标",
    "窗口",
    "缓冲",
    "向量",
    "张量",
    "内核",
    "设备",
    "引擎",
    "信号",
    "配置",
    "默认",
    "记录",
    "报告",
    "摘要",
    "上下文",
    "キャッシュ",
    "トークン",
    "🚀",
    "🙂",
    "📦",
    "🔁",
];

/// Seed salt for `utf8_hot64`, so adding it left the ASCII corpora's
/// random streams untouched.
const UTF8_SEED_SALT: u64 = 0x7f8_0000;
/// Seed salt for [`IdCorpus`].
const ID_SEED_SALT: u64 = 0x1d5_0000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CorpusKind {
    Hot64,
    Cold,
    Mixed90,
    ShortSharedPrefix,
    LongSharedPrefix,
    Utf8Hot64,
}

impl CorpusKind {
    pub const ALL: [CorpusKind; 6] = [
        CorpusKind::Hot64,
        CorpusKind::Cold,
        CorpusKind::Mixed90,
        CorpusKind::ShortSharedPrefix,
        CorpusKind::LongSharedPrefix,
        CorpusKind::Utf8Hot64,
    ];

    pub fn name(self) -> &'static str {
        match self {
            CorpusKind::Hot64 => "hot64",
            CorpusKind::Cold => "cold",
            CorpusKind::Mixed90 => "mixed90",
            CorpusKind::ShortSharedPrefix => "short_shared_prefix",
            CorpusKind::LongSharedPrefix => "long_shared_prefix",
            CorpusKind::Utf8Hot64 => "utf8_hot64",
        }
    }

    pub fn parse(name: &str) -> Option<CorpusKind> {
        CorpusKind::ALL.into_iter().find(|k| k.name() == name)
    }
}

/// Deterministic word salad of exactly `bytes` ASCII bytes.
fn filler(rng: &mut StdRng, bytes: usize) -> String {
    let mut out = String::with_capacity(bytes + 16);
    while out.len() < bytes {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(WORDS[rng.random_range(0..WORDS.len())]);
    }
    out.truncate(bytes);
    out
}

/// Deterministic ASCII word salad of exactly `bytes` bytes for `seed`.
/// Two calls with the same seed return the same text.
pub fn seeded_text(seed: u64, bytes: usize) -> String {
    filler(&mut StdRng::seed_from_u64(seed), bytes)
}

/// Deterministic CJK and emoji text of exactly `bytes` bytes. The text is
/// cut at the last character boundary that fits and padded with ASCII
/// spaces (at most three) to the exact size.
fn utf8_filler(rng: &mut StdRng, bytes: usize) -> String {
    let mut out = String::with_capacity(bytes + 16);
    while out.len() < bytes {
        out.push_str(UTF8_WORDS[rng.random_range(0..UTF8_WORDS.len())]);
    }
    let mut cut = bytes;
    while !out.is_char_boundary(cut) {
        cut -= 1;
    }
    out.truncate(cut);
    while out.len() < bytes {
        out.push(' ');
    }
    out
}

/// Deterministic CJK and emoji text of exactly `bytes` bytes for `seed`.
pub fn seeded_utf8_text(seed: u64, bytes: usize) -> String {
    utf8_filler(&mut StdRng::seed_from_u64(seed), bytes)
}

/// `n` token ids from [`PROMPT_ID_RANGE`] for `seed`.
pub fn seeded_ids(seed: u64, n: usize) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(PROMPT_ID_RANGE)).collect()
}

/// Distinct pooled bodies for prompts of `size_bytes`: [`BODY_POOL_SIZE`]
/// up to 16 KiB, then as many as fit in [`BODY_POOL_MAX_BYTES`], never
/// fewer than [`BODY_POOL_MIN_SIZE`].
pub fn body_pool_len(size_bytes: usize) -> usize {
    (BODY_POOL_MAX_BYTES / size_bytes).clamp(BODY_POOL_MIN_SIZE, BODY_POOL_SIZE)
}

/// Place `marker` at the start of `body` and cut the result to `size`.
fn with_marker(marker: &str, body: &str, size: usize) -> String {
    let mut s = String::with_capacity(size);
    s.push_str(marker);
    if s.len() < size {
        s.push_str(&body[..size - s.len()]);
    } else {
        s.truncate(size);
    }
    s
}

/// A prompt source of one kind and one size.
#[derive(Clone, Debug)]
pub struct Corpus {
    kind: CorpusKind,
    size_bytes: usize,
    hot: Vec<String>,
    pool: Vec<String>,
    shared_prefix: String,
    long_prefix: String,
}

impl Corpus {
    pub fn new(kind: CorpusKind, size_bytes: usize) -> Corpus {
        assert!(
            size_bytes >= 2 * LONG_SHARED_PREFIX_TAIL_BYTES,
            "prompt size must leave room for a marker and a unique tail"
        );
        let (hot, pool, shared_prefix, long_prefix) = if kind == CorpusKind::Utf8Hot64 {
            let mut rng = StdRng::seed_from_u64(SEED ^ UTF8_SEED_SALT ^ size_bytes as u64);
            let hot = (0..HOT_SET_SIZE)
                .map(|j| {
                    let marker = format!("[hot {j:02}] ");
                    let body = utf8_filler(&mut rng, size_bytes - marker.len());
                    marker + body.as_str()
                })
                .collect();
            (hot, Vec::new(), String::new(), String::new())
        } else {
            let mut rng = StdRng::seed_from_u64(SEED ^ size_bytes as u64);
            let hot = (0..HOT_SET_SIZE)
                .map(|j| {
                    let body = filler(&mut rng, size_bytes);
                    with_marker(&format!("[hot {j:02}] "), &body, size_bytes)
                })
                .collect();
            let pool = (0..body_pool_len(size_bytes))
                .map(|_| filler(&mut rng, size_bytes))
                .collect();
            let prefix_len = SHORT_SHARED_PREFIX_MAX_BYTES.min(size_bytes / 2);
            let shared_prefix = filler(&mut rng, prefix_len);
            let long_prefix = filler(&mut rng, size_bytes - LONG_SHARED_PREFIX_TAIL_BYTES);
            (hot, pool, shared_prefix, long_prefix)
        };
        Corpus {
            kind,
            size_bytes,
            hot,
            pool,
            shared_prefix,
            long_prefix,
        }
    }

    pub fn kind(&self) -> CorpusKind {
        self.kind
    }

    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Shared prefix of the `short_shared_prefix` and `long_shared_prefix`
    /// corpora (empty for others).
    pub fn shared_prefix(&self) -> &str {
        match self.kind {
            CorpusKind::ShortSharedPrefix => &self.shared_prefix,
            CorpusKind::LongSharedPrefix => &self.long_prefix,
            _ => "",
        }
    }

    /// The prompt for request number `i` (0-based, monotonically increasing
    /// across the whole run).
    pub fn prompt(&self, i: usize) -> String {
        match self.kind {
            CorpusKind::Hot64 | CorpusKind::Utf8Hot64 => self.hot[i % HOT_SET_SIZE].clone(),
            CorpusKind::Cold => self.cold(i),
            CorpusKind::Mixed90 => {
                if i.is_multiple_of(10) {
                    self.cold(i)
                } else {
                    self.hot[i % HOT_SET_SIZE].clone()
                }
            }
            CorpusKind::ShortSharedPrefix => self.with_unique_tail(&self.shared_prefix, i),
            CorpusKind::LongSharedPrefix => self.with_unique_tail(&self.long_prefix, i),
        }
    }

    fn with_unique_tail(&self, prefix: &str, i: usize) -> String {
        let body = self.body(i);
        let mut s = String::with_capacity(self.size_bytes);
        s.push_str(prefix);
        s.push_str(&format!(" [turn {i}] "));
        let remaining = self.size_bytes.saturating_sub(s.len());
        s.push_str(&body[..remaining.min(body.len())]);
        s.truncate(self.size_bytes);
        s
    }

    /// The pooled body for request `i`.
    fn body(&self, i: usize) -> &str {
        &self.pool[i % self.pool.len()]
    }

    fn cold(&self, i: usize) -> String {
        with_marker(&format!("[cold {i}] "), self.body(i), self.size_bytes)
    }
}

/// [`HOT_SET_SIZE`] pre-tokenized prompts of `ids_per_prompt` ids each,
/// cycled by request index like `hot64`.
#[derive(Clone, Debug)]
pub struct IdCorpus {
    ids_per_prompt: usize,
    hot: Vec<Vec<u32>>,
}

impl IdCorpus {
    pub fn new(ids_per_prompt: usize) -> IdCorpus {
        let hot = (0..HOT_SET_SIZE as u64)
            .map(|j| {
                seeded_ids(
                    SEED ^ ID_SEED_SALT ^ ((j << 32) | ids_per_prompt as u64),
                    ids_per_prompt,
                )
            })
            .collect();
        IdCorpus {
            ids_per_prompt,
            hot,
        }
    }

    pub fn ids_per_prompt(&self) -> usize {
        self.ids_per_prompt
    }

    /// The ids for request number `i`.
    pub fn prompt_ids(&self, i: usize) -> &[u32] {
        &self.hot[i % HOT_SET_SIZE]
    }
}

/// Non-streaming `/v1/completions` body with a text prompt.
pub fn completion_text(prompt: &str) -> Value {
    json!({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "stream": false,
    })
}

/// Non-streaming `/v1/completions` body with a pre-tokenized prompt.
pub fn completion_ids(ids: &[u32]) -> Value {
    json!({
        "model": MODEL,
        "prompt": ids,
        "max_tokens": MAX_TOKENS,
        "stream": false,
    })
}

/// Non-streaming `/v1/chat/completions` body with an optional system message
/// and one user message.
pub fn chat(system: Option<&str>, user: &str) -> Value {
    let mut messages = Vec::new();
    if let Some(system) = system {
        messages.push(json!({"role": "system", "content": system}));
    }
    messages.push(json!({"role": "user", "content": user}));
    json!({
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "stream": false,
    })
}

/// Like [`chat`] but with `session_params.session_id`, the only field the
/// router uses as chat routing text today.
pub fn chat_with_session(session_id: &str, user: &str) -> Value {
    let mut body = chat(None, user);
    body["session_params"] = json!({"session_id": session_id});
    body
}

pub fn to_completion_request(body: &Value) -> CompletionRequest {
    serde_json::from_value(body.clone()).expect("completion body must deserialize")
}

pub fn to_chat_request(body: &Value) -> ChatCompletionRequest {
    serde_json::from_value(body.clone()).expect("chat body must deserialize")
}
