//! Deterministic prompt corpus and request builders shared by the routing
//! benchmarks (`benches/routing_input.rs`) and the router overhead harness
//! (`tests/router_overhead_bench.rs`).
//!
//! Everything here is derived from [`SEED`], the prompt size and the request
//! index, so two runs on different machines see byte-identical requests.
//! Prompt text is ASCII word salad and every prompt is exactly `size_bytes`
//! long. Text that is not meant to be shared is drawn from a pool of
//! [`BODY_POOL_SIZE`] distinct seeded bodies rather than one repeated body:
//! routing-key code is content dependent (substring search, hashing, JSON
//! parsing), and a single repeated body lets the CPU's branch predictors
//! learn it. With one body, rendezvous-hash selection on 16 KiB prompts
//! measured about three times cheaper than on 64 distinct prompts.
//!
//! Corpus kinds:
//! - `hot64`: 64 fixed prompts cycled by request index; after the first 64
//!   requests every request repeats an earlier one (a cache-hit workload).
//! - `cold`: every request carries a unique marker at the *start* of the
//!   prompt, so no two requests share a prefix (a cache-miss workload).
//! - `mixed90`: 90% `hot64`, 10% `cold` (every tenth request is unique).
//! - `short_shared_prefix`: a shared prefix of at most 600 bytes (roughly
//!   96-160 tokens for English text) followed by a unique tail; models a
//!   shared system prompt with a varying user turn.
//! - `long_shared_prefix`: everything but the last 64 bytes is shared; the
//!   tail is unique. Models "same prefix, varying suffix": an exact-match
//!   cache misses on every request while a prefix router still sees one
//!   prefix.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};
use vllm_router_rs::protocols::spec::{ChatCompletionRequest, CompletionRequest};

/// Seed for every generator in this module.
pub const SEED: u64 = 0x244;
/// Prompt sizes (bytes) used by the benchmark matrix.
pub const SIZES: [usize; 3] = [200, 2048, 16384];
/// Number of distinct prompts in the `hot64` set.
pub const HOT_SET_SIZE: usize = 64;
/// Longest shared prefix (bytes) in `short_shared_prefix`.
pub const SHORT_SHARED_PREFIX_MAX_BYTES: usize = 600;
/// Unique tail (bytes) at the end of every `long_shared_prefix` prompt.
pub const LONG_SHARED_PREFIX_TAIL_BYTES: usize = 64;
/// Distinct bodies that non-shared text is drawn from.
pub const BODY_POOL_SIZE: usize = 256;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CorpusKind {
    Hot64,
    Cold,
    Mixed90,
    ShortSharedPrefix,
    LongSharedPrefix,
}

impl CorpusKind {
    pub const ALL: [CorpusKind; 5] = [
        CorpusKind::Hot64,
        CorpusKind::Cold,
        CorpusKind::Mixed90,
        CorpusKind::ShortSharedPrefix,
        CorpusKind::LongSharedPrefix,
    ];

    pub fn name(self) -> &'static str {
        match self {
            CorpusKind::Hot64 => "hot64",
            CorpusKind::Cold => "cold",
            CorpusKind::Mixed90 => "mixed90",
            CorpusKind::ShortSharedPrefix => "short_shared_prefix",
            CorpusKind::LongSharedPrefix => "long_shared_prefix",
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
        let mut rng = StdRng::seed_from_u64(SEED ^ size_bytes as u64);
        let hot = (0..HOT_SET_SIZE)
            .map(|j| {
                let body = filler(&mut rng, size_bytes);
                with_marker(&format!("[hot {j:02}] "), &body, size_bytes)
            })
            .collect();
        let pool = (0..BODY_POOL_SIZE)
            .map(|_| filler(&mut rng, size_bytes))
            .collect();
        let prefix_len = SHORT_SHARED_PREFIX_MAX_BYTES.min(size_bytes / 2);
        let shared_prefix = filler(&mut rng, prefix_len);
        let long_prefix = filler(&mut rng, size_bytes - LONG_SHARED_PREFIX_TAIL_BYTES);
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
            CorpusKind::Hot64 => self.hot[i % HOT_SET_SIZE].clone(),
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
        &self.pool[i % BODY_POOL_SIZE]
    }

    fn cold(&self, i: usize) -> String {
        with_marker(&format!("[cold {i}] "), self.body(i), self.size_bytes)
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
