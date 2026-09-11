//! Bounded exact-match (L0) cache for tokenizer encode results.
//!
//! [`CachedTokenizer`] wraps any `Arc<dyn Tokenizer>` and memoizes successful
//! [`Encoder::encode`] results keyed by the exact input string. Every other
//! trait method is delegated to the wrapped tokenizer unchanged.
//!
//! # Semantics
//!
//! - A cache belongs to exactly one tokenizer instance. Two `CachedTokenizer`s
//!   never share entries, even when they wrap equivalent tokenizers.
//! - Hits return an owned clone of the stored [`Encoding`], including all
//!   metadata (token strings, offsets, masks) carried by the backend.
//! - Only successful `encode()` results are stored. Errors are returned to the
//!   caller and leave the cache untouched.
//! - `encode_batch()` neither reads from nor populates the cache.
//! - Tokenization runs outside the cache lock. Two concurrent misses on the
//!   same input both tokenize; the later insert replaces the earlier one with
//!   an identical value.
//!
//! # Preconditions
//!
//! Caching is only sound when the wrapped tokenizer is deterministic: the
//! same input must always produce the same encoding, and every setting that
//! affects encoding (normalization, truncation, added tokens, BPE dropout or
//! other sampling) must stay fixed for the cache's lifetime. A fixed
//! configuration does not by itself disable stochastic tokenization; callers
//! must make sure it is off before wrapping.
//!
//! # Bounds and eviction
//!
//! The cache is bounded by an entry count ([`TokenizerCacheConfig::max_entries`])
//! and an estimated retained-byte budget ([`TokenizerCacheConfig::max_bytes`]).
//! When either bound is exceeded, least-recently-used entries are evicted until
//! both hold. Results whose estimate exceeds
//! [`TokenizerCacheConfig::max_entry_bytes`] are returned to the caller but not
//! stored, so a handful of very long inputs cannot flush the working set.
//!
//! # Byte estimation
//!
//! Estimates count heap bytes that an entry keeps alive, using lengths rather
//! than capacities, plus a fixed per-entry bookkeeping overhead
//! ([`ENTRY_OVERHEAD_BYTES`]). Per entry:
//!
//! - the input string: `input.len()`
//! - `Encoding::Sp` and `Encoding::Tiktoken`: the `Vec` header plus
//!   `4 bytes × token count`
//! - `Encoding::Hf`: the boxed struct plus, per token, the four `u32` vectors
//!   (ids, type ids, special-token mask, attention mask), the `Option<u32>` word
//!   index, the `(usize, usize)` offset pair, the `String` header, and the token
//!   text bytes; overflowing encodings are counted recursively.
//!
//! The estimate is a lower bound on real memory use, not a bound on resident
//! memory: it ignores allocator padding, hash-map load factor, unused `Vec`
//! capacity, clones held by callers, and evicted encodings kept alive by an
//! in-flight `Arc` from a concurrent hit.
//!
//! # Metrics
//!
//! Hits, misses, evictions and oversized skips are counted under
//! `vllm_tokenizer_cache_*_total`. `vllm_tokenizer_cache_entries` and
//! `vllm_tokenizer_cache_bytes` are gauges holding the total occupancy of
//! every live `CachedTokenizer` in the process. Each instance applies its
//! change to a process-wide aggregate and publishes the new totals while
//! still holding its own lock, so after any `encode`, `clear` or drop the
//! gauges equal the sum of the live instances' [`stats`](CachedTokenizer::stats).
//! The series carry no labels; per-cache labels can be added when the cache
//! is wired into the request path.

use super::traits::{
    Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer as TokenizerTrait,
};
use crate::metrics::TokenizerMetrics;
use anyhow::{bail, Result};
use lru::LruCache;
use parking_lot::Mutex;
use std::mem::size_of;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Budgets for a [`CachedTokenizer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizerCacheConfig {
    /// Maximum number of entries retained. Must be at least 1.
    pub max_entries: usize,
    /// Maximum estimated bytes retained across all entries. Must be at least 1.
    pub max_bytes: usize,
    /// Encode results whose estimated size exceeds this are not cached.
    /// Must be at least 1 and at most `max_bytes`.
    pub max_entry_bytes: usize,
}

impl Default for TokenizerCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            max_bytes: 64 * 1024 * 1024,
            max_entry_bytes: 1024 * 1024,
        }
    }
}

impl TokenizerCacheConfig {
    /// Check that every budget is usable.
    pub fn validate(&self) -> Result<()> {
        if self.max_entries == 0 {
            bail!("tokenizer cache max_entries must be at least 1");
        }
        if self.max_bytes == 0 {
            bail!("tokenizer cache max_bytes must be at least 1");
        }
        if self.max_entry_bytes == 0 {
            bail!("tokenizer cache max_entry_bytes must be at least 1");
        }
        if self.max_entry_bytes > self.max_bytes {
            bail!(
                "tokenizer cache max_entry_bytes ({}) must not exceed max_bytes ({})",
                self.max_entry_bytes,
                self.max_bytes
            );
        }
        Ok(())
    }
}

/// Point-in-time counters and occupancy of a [`CachedTokenizer`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenizerCacheStats {
    /// `encode()` calls served from the cache.
    pub hits: u64,
    /// `encode()` calls that ran the wrapped tokenizer.
    pub misses: u64,
    /// Entries removed to satisfy the entry or byte budget.
    pub evictions: u64,
    /// Successful encodes not stored because they exceeded `max_entry_bytes`.
    pub oversized: u64,
    /// Entries currently retained.
    pub entries: usize,
    /// Estimated bytes currently retained.
    pub bytes: usize,
}

struct CacheEntry {
    encoding: Arc<Encoding>,
    bytes: usize,
}

struct CacheState {
    lru: LruCache<String, CacheEntry>,
    /// Sum of `CacheEntry::bytes` over all entries in `lru`.
    bytes: usize,
}

/// Approximate fixed cost of one cache entry: the LRU node (key, value and two
/// list links) and its hash-map slot (key reference and node pointer).
pub const ENTRY_OVERHEAD_BYTES: usize =
    size_of::<String>() + size_of::<CacheEntry>() + 4 * size_of::<usize>();

/// Occupancy summed over every live cache in the process. Guarded by a lock
/// so gauge publications are totally ordered and always carry the current
/// total.
struct Occupancy {
    entries: usize,
    bytes: usize,
}

static OCCUPANCY: Mutex<Occupancy> = Mutex::new(Occupancy {
    entries: 0,
    bytes: 0,
});

/// Apply one instance's state change to the process-wide totals and publish
/// them. Callers hold their own state lock while calling this, so an
/// instance's contribution to the gauges always matches its state. Lock order
/// is instance state, then `OCCUPANCY`.
fn publish_occupancy_delta(entries_delta: isize, bytes_delta: isize) {
    if entries_delta == 0 && bytes_delta == 0 {
        return;
    }
    let mut total = OCCUPANCY.lock();
    total.entries = total.entries.saturating_add_signed(entries_delta);
    total.bytes = total.bytes.saturating_add_signed(bytes_delta);
    TokenizerMetrics::set_cache_entries(total.entries);
    TokenizerMetrics::set_cache_bytes(total.bytes);
}

/// Tokenizer wrapper that memoizes `encode()` results by exact input.
///
/// See the [module documentation](self) for semantics and bounds.
pub struct CachedTokenizer {
    inner: Arc<dyn TokenizerTrait>,
    config: TokenizerCacheConfig,
    state: Mutex<CacheState>,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    oversized: AtomicU64,
}

impl CachedTokenizer {
    /// Wrap `inner` with a cache sized by `config`.
    ///
    /// Returns an error if `config` fails [`TokenizerCacheConfig::validate`].
    pub fn new(inner: Arc<dyn TokenizerTrait>, config: TokenizerCacheConfig) -> Result<Self> {
        config.validate()?;
        let capacity =
            NonZeroUsize::new(config.max_entries).expect("validate() guarantees max_entries >= 1");
        Ok(Self {
            inner,
            config,
            state: Mutex::new(CacheState {
                // `sparse` grows the map on demand instead of preallocating
                // `max_entries` slots, which matters when the byte budget is
                // the binding limit.
                lru: LruCache::sparse(capacity),
                bytes: 0,
            }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            oversized: AtomicU64::new(0),
        })
    }

    /// The budgets this cache was built with.
    pub fn config(&self) -> &TokenizerCacheConfig {
        &self.config
    }

    /// The wrapped tokenizer.
    pub fn inner(&self) -> &Arc<dyn TokenizerTrait> {
        &self.inner
    }

    /// Number of entries currently retained.
    pub fn len(&self) -> usize {
        self.state.lock().lru.len()
    }

    /// Whether the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot of counters and occupancy.
    pub fn stats(&self) -> TokenizerCacheStats {
        let (entries, bytes) = {
            let state = self.state.lock();
            (state.lru.len(), state.bytes)
        };
        TokenizerCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            oversized: self.oversized.load(Ordering::Relaxed),
            entries,
            bytes,
        }
    }

    /// Drop every entry. Counters are kept.
    pub fn clear(&self) {
        let mut guard = self.state.lock();
        let state = &mut *guard;
        let entries = state.lru.len();
        let bytes = state.bytes;
        state.lru.clear();
        state.bytes = 0;
        publish_occupancy_delta(-(entries as isize), -(bytes as isize));
    }

    fn lookup(&self, input: &str) -> Option<Arc<Encoding>> {
        let mut state = self.state.lock();
        state
            .lru
            .get(input)
            .map(|entry| Arc::clone(&entry.encoding))
    }

    fn insert(&self, input: &str, encoding: Arc<Encoding>, bytes: usize) {
        let mut evicted: u64 = 0;
        {
            let mut guard = self.state.lock();
            let state = &mut *guard;
            let entries_before = state.lru.len();
            let bytes_before = state.bytes;

            // Another thread may have inserted this input while we were
            // tokenizing. `push` then returns the entry it replaced rather
            // than an LRU victim, so account for it as a replacement.
            let replaced_bytes = state.lru.peek(input).map(|entry| entry.bytes);
            let displaced = state
                .lru
                .push(input.to_owned(), CacheEntry { encoding, bytes });
            match (replaced_bytes, displaced) {
                (Some(previous), Some(_)) => state.bytes -= previous,
                (None, Some((_, victim))) => {
                    state.bytes -= victim.bytes;
                    evicted += 1;
                }
                (_, None) => {}
            }
            state.bytes += bytes;

            // `validate()` bounds every stored entry by `max_bytes`, so this
            // loop never reaches the entry just inserted (the MRU).
            while state.bytes > self.config.max_bytes {
                match state.lru.pop_lru() {
                    Some((_, victim)) => {
                        state.bytes -= victim.bytes;
                        evicted += 1;
                    }
                    None => break,
                }
            }

            publish_occupancy_delta(
                state.lru.len() as isize - entries_before as isize,
                state.bytes as isize - bytes_before as isize,
            );
        }

        if evicted > 0 {
            self.evictions.fetch_add(evicted, Ordering::Relaxed);
            TokenizerMetrics::record_cache_evictions(evicted);
        }
    }
}

impl Drop for CachedTokenizer {
    fn drop(&mut self) {
        let state = self.state.get_mut();
        publish_occupancy_delta(-(state.lru.len() as isize), -(state.bytes as isize));
    }
}

impl Encoder for CachedTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        if let Some(encoding) = self.lookup(input) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            TokenizerMetrics::record_cache_hit();
            // Clone outside the lock so a large encoding does not stall
            // other readers.
            return Ok((*encoding).clone());
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        TokenizerMetrics::record_cache_miss();

        let encoding = self.inner.encode(input)?;
        let bytes = estimate_entry_bytes(input, &encoding);
        if bytes > self.config.max_entry_bytes {
            self.oversized.fetch_add(1, Ordering::Relaxed);
            TokenizerMetrics::record_cache_oversized();
            return Ok(encoding);
        }

        let shared = Arc::new(encoding);
        let result = (*shared).clone();
        self.insert(input, shared, bytes);
        Ok(result)
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        self.inner.encode_batch(inputs)
    }
}

impl Decoder for CachedTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        self.inner.decode(token_ids, skip_special_tokens)
    }
}

impl TokenizerTrait for CachedTokenizer {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        self.inner.get_special_tokens()
    }

    fn token_to_id(&self, token: &str) -> Option<TokenIdType> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: TokenIdType) -> Option<String> {
        self.inner.id_to_token(id)
    }
}

/// Estimated bytes a cache entry for `input` and its `encoding` retains.
///
/// See the [module documentation](self) for what is counted.
pub fn estimate_entry_bytes(input: &str, encoding: &Encoding) -> usize {
    ENTRY_OVERHEAD_BYTES + input.len() + estimate_encoding_bytes(encoding)
}

/// Estimated bytes retained by `encoding` alone.
pub fn estimate_encoding_bytes(encoding: &Encoding) -> usize {
    match encoding {
        Encoding::Sp(ids) | Encoding::Tiktoken(ids) => {
            size_of::<Vec<TokenIdType>>() + ids.len() * size_of::<TokenIdType>()
        }
        Encoding::Hf(inner) => {
            size_of::<tokenizers::tokenizer::Encoding>() + estimate_hf_bytes(inner)
        }
    }
}

fn estimate_hf_bytes(encoding: &tokenizers::tokenizer::Encoding) -> usize {
    // ids, type_ids, special_tokens_mask, attention_mask, words, offsets and
    // the token `String` headers are all one element per token.
    const PER_TOKEN: usize = 4 * size_of::<u32>()
        + size_of::<Option<u32>>()
        + size_of::<(usize, usize)>()
        + size_of::<String>();

    let tokens = encoding.get_ids().len() * PER_TOKEN;
    let token_text: usize = encoding.get_tokens().iter().map(String::len).sum();
    let overflowing: usize = encoding
        .get_overflowing()
        .iter()
        .map(|overflow| size_of::<tokenizers::tokenizer::Encoding>() + estimate_hf_bytes(overflow))
        .sum();
    // `sequence_ranges` is empty for the single-sequence case and holds one
    // `usize -> Range<usize>` pair per sequence otherwise.
    let sequences = encoding.n_sequences();
    let sequence_ranges = if sequences > 1 {
        sequences * (size_of::<usize>() + size_of::<Range<usize>>())
    } else {
        0
    };

    tokens + token_text + overflowing + sequence_ranges
}

#[cfg(test)]
impl CachedTokenizer {
    /// Recompute retained bytes from the entries, to check the running total.
    fn recomputed_bytes(&self) -> usize {
        self.state
            .lock()
            .lru
            .iter()
            .map(|(_, entry)| entry.bytes)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::mock::MockTokenizer;
    use crate::tokenizer::Tokenizer;
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    /// Mock tokenizer that counts calls and fails for inputs containing "fail".
    struct CountingTokenizer {
        inner: MockTokenizer,
        encode_calls: AtomicUsize,
        batch_calls: AtomicUsize,
    }

    impl CountingTokenizer {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                inner: MockTokenizer::new(),
                encode_calls: AtomicUsize::new(0),
                batch_calls: AtomicUsize::new(0),
            })
        }

        fn encode_calls(&self) -> usize {
            self.encode_calls.load(Ordering::SeqCst)
        }

        fn batch_calls(&self) -> usize {
            self.batch_calls.load(Ordering::SeqCst)
        }
    }

    impl Encoder for CountingTokenizer {
        fn encode(&self, input: &str) -> Result<Encoding> {
            self.encode_calls.fetch_add(1, Ordering::SeqCst);
            if input.contains("fail") {
                bail!("simulated encode failure");
            }
            self.inner.encode(input)
        }

        fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
            self.batch_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.encode_batch(inputs)
        }
    }

    impl Decoder for CountingTokenizer {
        fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
            self.inner.decode(token_ids, skip_special_tokens)
        }
    }

    impl TokenizerTrait for CountingTokenizer {
        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }

        fn get_special_tokens(&self) -> &SpecialTokens {
            self.inner.get_special_tokens()
        }

        fn token_to_id(&self, token: &str) -> Option<TokenIdType> {
            self.inner.token_to_id(token)
        }

        fn id_to_token(&self, id: TokenIdType) -> Option<String> {
            self.inner.id_to_token(id)
        }
    }

    fn config(max_entries: usize, max_bytes: usize) -> TokenizerCacheConfig {
        TokenizerCacheConfig {
            max_entries,
            max_bytes,
            max_entry_bytes: max_bytes,
        }
    }

    fn cached(inner: &Arc<CountingTokenizer>, config: TokenizerCacheConfig) -> CachedTokenizer {
        CachedTokenizer::new(inner.clone(), config).unwrap()
    }

    /// Estimated entry size for `input` under the mock tokenizer.
    fn mock_entry_bytes(input: &str) -> usize {
        let encoding = MockTokenizer::new().encode(input).unwrap();
        estimate_entry_bytes(input, &encoding)
    }

    #[test]
    fn hit_returns_equal_encoding_without_calling_inner() {
        let inner = CountingTokenizer::new();
        let cache = cached(&inner, TokenizerCacheConfig::default());

        let first = cache.encode("Hello world").unwrap();
        let second = cache.encode("Hello world").unwrap();

        assert_eq!(first.token_ids(), &[1, 2]);
        assert_eq!(second.token_ids(), first.token_ids());
        assert_eq!(second.get_hash(), first.get_hash());
        assert_eq!(inner.encode_calls(), 1);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.bytes, mock_entry_bytes("Hello world"));
    }

    #[test]
    fn empty_input_is_cached() {
        let inner = CountingTokenizer::new();
        let cache = cached(&inner, TokenizerCacheConfig::default());

        assert!(cache.encode("").unwrap().token_ids().is_empty());
        assert!(cache.encode("").unwrap().token_ids().is_empty());

        assert_eq!(inner.encode_calls(), 1);
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn keys_are_exact_bytes() {
        let inner = CountingTokenizer::new();
        let cache = cached(&inner, TokenizerCacheConfig::default());

        // Case, whitespace and Unicode normalization form all change the key.
        for input in [
            "Hello",
            "hello",
            "Hello ",
            "caf\u{e9}",
            "cafe\u{301}",
            "\u{1f600}",
        ] {
            cache.encode(input).unwrap();
        }
        assert_eq!(inner.encode_calls(), 6);
        assert_eq!(cache.stats().entries, 6);

        cache.encode("caf\u{e9}").unwrap();
        assert_eq!(inner.encode_calls(), 6);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn special_tokens_round_trip_through_cache() {
        let inner = CountingTokenizer::new();
        let cache = cached(&inner, TokenizerCacheConfig::default());

        let first = cache.encode("<bos> Hello <eos>").unwrap();
        let second = cache.encode("<bos> Hello <eos>").unwrap();
        assert_eq!(first.token_ids(), &[1000, 1, 999]);
        assert_eq!(second.token_ids(), &[1000, 1, 999]);
        assert_eq!(inner.encode_calls(), 1);
    }

    #[test]
    fn errors_are_not_cached() {
        let inner = CountingTokenizer::new();
        let cache = cached(&inner, TokenizerCacheConfig::default());

        assert!(cache.encode("please fail").is_err());
        assert!(cache.encode("please fail").is_err());

        assert_eq!(inner.encode_calls(), 2);
        let stats = cache.stats();
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.bytes, 0);
    }

    #[test]
    fn caches_are_isolated_per_instance() {
        let inner_a = CountingTokenizer::new();
        let inner_b = CountingTokenizer::new();
        let cache_a = cached(&inner_a, TokenizerCacheConfig::default());
        let cache_b = cached(&inner_b, TokenizerCacheConfig::default());

        cache_a.encode("Hello").unwrap();
        cache_a.encode("Hello").unwrap();
        assert_eq!(inner_a.encode_calls(), 1);
        assert_eq!(inner_b.encode_calls(), 0);

        cache_b.encode("Hello").unwrap();
        assert_eq!(inner_b.encode_calls(), 1);
        assert_eq!(cache_a.stats().entries, 1);
        assert_eq!(cache_b.stats().entries, 1);
        assert_eq!(cache_b.stats().hits, 0);
    }

    #[test]
    fn evicts_least_recently_used_by_entry_count() {
        let inner = CountingTokenizer::new();
        let cache = cached(&inner, config(2, usize::MAX));

        cache.encode("Hello").unwrap();
        cache.encode("world").unwrap();
        cache.encode("Hello").unwrap(); // touch: "world" is now the LRU entry
        cache.encode("test").unwrap(); // evicts "world"
        assert_eq!(inner.encode_calls(), 3);

        let stats = cache.stats();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.evictions, 1);
        assert_eq!(
            stats.bytes,
            mock_entry_bytes("Hello") + mock_entry_bytes("test")
        );

        cache.encode("Hello").unwrap();
        cache.encode("test").unwrap();
        assert_eq!(inner.encode_calls(), 3, "survivors must still hit");

        cache.encode("world").unwrap();
        assert_eq!(inner.encode_calls(), 4, "evicted entry must miss");
    }

    #[test]
    fn evicts_by_byte_budget() {
        let per_entry = mock_entry_bytes("Hello");
        assert_eq!(mock_entry_bytes("world"), per_entry);
        assert_eq!(mock_entry_bytes("token"), per_entry);
        let inner = CountingTokenizer::new();
        // Room for two entries but not three.
        let cache = cached(&inner, config(100, 2 * per_entry + per_entry / 2));

        cache.encode("Hello").unwrap();
        cache.encode("world").unwrap();
        assert_eq!(cache.stats().evictions, 0);

        cache.encode("token").unwrap();
        let stats = cache.stats();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.bytes, 2 * per_entry);
        assert!(stats.bytes <= cache.config().max_bytes);

        cache.encode("Hello").unwrap();
        assert_eq!(
            inner.encode_calls(),
            4,
            "oldest entry must have been evicted"
        );
        assert_eq!(cache.recomputed_bytes(), cache.stats().bytes);
    }

    #[test]
    fn byte_budget_can_evict_several_small_entries_for_one_larger() {
        let small = mock_entry_bytes("Hello");
        let large_input = "Hello world ".repeat(100);
        let large = mock_entry_bytes(&large_input);
        assert!(large > 3 * small);
        let inner = CountingTokenizer::new();
        let cache = cached(&inner, config(100, large + small / 2));

        cache.encode("Hello").unwrap();
        cache.encode("world").unwrap();
        cache.encode("token").unwrap();
        cache.encode(&large_input).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.evictions, 3);
        assert_eq!(stats.bytes, large);
        assert_eq!(cache.recomputed_bytes(), large);
    }

    #[test]
    fn oversized_results_bypass_cache() {
        let inner = CountingTokenizer::new();
        let cache = cached(
            &inner,
            TokenizerCacheConfig {
                max_entries: 100,
                max_bytes: 1 << 20,
                max_entry_bytes: ENTRY_OVERHEAD_BYTES + 1,
            },
        );

        let encoding = cache.encode("Hello world").unwrap();
        assert_eq!(encoding.token_ids(), &[1, 2]);
        cache.encode("Hello world").unwrap();

        assert_eq!(inner.encode_calls(), 2);
        let stats = cache.stats();
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.oversized, 2);
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.bytes, 0);
    }

    #[test]
    fn batch_encode_bypasses_cache() {
        let inner = CountingTokenizer::new();
        let cache = cached(&inner, TokenizerCacheConfig::default());

        let encodings = cache.encode_batch(&["Hello", "world"]).unwrap();
        assert_eq!(encodings.len(), 2);
        assert_eq!(inner.batch_calls(), 1);
        assert_eq!(cache.stats().entries, 0, "batch must not populate");

        cache.encode("Hello").unwrap();
        assert_eq!(inner.encode_calls(), 1);

        cache.encode_batch(&["Hello"]).unwrap();
        assert_eq!(inner.batch_calls(), 2, "batch must not read");
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn same_key_reinsert_replaces_without_double_counting() {
        let inner = CountingTokenizer::new();
        let cache = cached(&inner, config(10, usize::MAX));
        let encoding = Arc::new(MockTokenizer::new().encode("Hello").unwrap());
        let bytes = mock_entry_bytes("Hello");

        // Models two threads that both missed on the same input.
        cache.insert("Hello", encoding.clone(), bytes);
        cache.insert("Hello", encoding, bytes);

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.bytes, bytes);
        assert_eq!(stats.evictions, 0);
        assert_eq!(cache.recomputed_bytes(), bytes);
    }

    #[test]
    fn concurrent_access_keeps_accounting_consistent() {
        let inner = CountingTokenizer::new();
        let cache = Arc::new(cached(&inner, config(8, usize::MAX)));
        let inputs: Vec<String> = (0..16).map(|i| format!("Hello world {i}")).collect();
        let reference = MockTokenizer::new();
        let expected: Vec<Vec<u32>> = inputs
            .iter()
            .map(|input| reference.encode(input).unwrap().token_ids().to_vec())
            .collect();

        let handles: Vec<_> = (0..8)
            .map(|t| {
                let cache = cache.clone();
                let inputs = inputs.clone();
                let expected = expected.clone();
                thread::spawn(move || {
                    for i in 0..200 {
                        let index = (i * 7 + t) % inputs.len();
                        let encoding = cache.encode(&inputs[index]).unwrap();
                        assert_eq!(encoding.token_ids(), expected[index].as_slice());
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }

        let stats = cache.stats();
        assert_eq!(stats.hits + stats.misses, 8 * 200);
        assert!(stats.entries <= 8);
        assert_eq!(stats.entries, cache.len());
        assert_eq!(stats.bytes, cache.recomputed_bytes());
        assert!(stats.evictions > 0);
    }

    #[test]
    fn clear_drops_entries_and_keeps_counters() {
        let inner = CountingTokenizer::new();
        let cache = cached(&inner, TokenizerCacheConfig::default());

        cache.encode("Hello").unwrap();
        cache.encode("Hello").unwrap();
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());
        let stats = cache.stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.bytes, 0);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);

        cache.encode("Hello").unwrap();
        assert_eq!(inner.encode_calls(), 2);
    }

    #[test]
    fn delegates_decode_and_metadata() {
        let inner = CountingTokenizer::new();
        let cache = cached(&inner, TokenizerCacheConfig::default());

        assert_eq!(cache.decode(&[1, 2], false).unwrap(), "Hello world");
        assert_eq!(cache.decode(&[1000, 1, 999], true).unwrap(), "Hello");
        assert_eq!(cache.vocab_size(), 8);
        assert_eq!(
            cache.get_special_tokens().bos_token.as_deref(),
            Some("<bos>")
        );
        assert_eq!(cache.token_to_id("Hello"), Some(1));
        assert_eq!(cache.token_to_id("nope"), None);
        assert_eq!(cache.id_to_token(2).as_deref(), Some("world"));
        assert_eq!(cache.id_to_token(4242), None);
        assert_eq!(cache.stats().entries, 0);
    }

    #[test]
    fn works_behind_tokenizer_wrapper() {
        let inner = CountingTokenizer::new();
        let tokenizer = Tokenizer::from_arc(Arc::new(cached(&inner, Default::default())));

        let encoding = tokenizer.encode("Hello world").unwrap();
        tokenizer.encode("Hello world").unwrap();
        assert_eq!(inner.encode_calls(), 1);
        assert_eq!(
            tokenizer.decode(encoding.token_ids(), false).unwrap(),
            "Hello world"
        );
    }

    #[test]
    fn config_validation() {
        assert!(TokenizerCacheConfig::default().validate().is_ok());
        assert!(config(0, 1024).validate().is_err());
        assert!(config(1, 0).validate().is_err());
        assert!(TokenizerCacheConfig {
            max_entries: 1,
            max_bytes: 1024,
            max_entry_bytes: 0,
        }
        .validate()
        .is_err());
        assert!(TokenizerCacheConfig {
            max_entries: 1,
            max_bytes: 1024,
            max_entry_bytes: 2048,
        }
        .validate()
        .is_err());

        let inner = CountingTokenizer::new();
        assert!(CachedTokenizer::new(inner.clone(), config(0, 1)).is_err());
        assert!(CachedTokenizer::new(inner, config(1, 1)).is_ok());
    }

    #[test]
    fn estimates_for_id_only_encodings() {
        let header = size_of::<Vec<TokenIdType>>();
        assert_eq!(
            estimate_encoding_bytes(&Encoding::Sp(vec![1, 2, 3])),
            header + 3 * 4
        );
        assert_eq!(
            estimate_encoding_bytes(&Encoding::Tiktoken(vec![7; 10])),
            header + 10 * 4
        );
        assert_eq!(estimate_encoding_bytes(&Encoding::Sp(vec![])), header);
        assert_eq!(
            estimate_entry_bytes("abc", &Encoding::Sp(vec![1])),
            ENTRY_OVERHEAD_BYTES + 3 + header + 4
        );
    }

    #[test]
    fn estimates_for_hf_encodings_count_tokens_text_and_overflow() {
        use tokenizers::tokenizer::Encoding as HfEncoding;

        fn hf(tokens: &[&str], overflowing: Vec<HfEncoding>) -> HfEncoding {
            let n = tokens.len();
            HfEncoding::new(
                (0..n as u32).collect(),
                vec![0; n],
                tokens.iter().map(|t| t.to_string()).collect(),
                vec![None; n],
                vec![(0, 0); n],
                vec![0; n],
                vec![1; n],
                overflowing,
                Default::default(),
            )
        }

        let per_token = 4 * size_of::<u32>()
            + size_of::<Option<u32>>()
            + size_of::<(usize, usize)>()
            + size_of::<String>();
        let struct_size = size_of::<HfEncoding>();

        let plain = Encoding::Hf(Box::new(hf(&["ab", "c"], vec![])));
        assert_eq!(
            estimate_encoding_bytes(&plain),
            struct_size + 2 * per_token + 3
        );

        let nested = Encoding::Hf(Box::new(hf(&["ab", "c"], vec![hf(&["xyz"], vec![])])));
        assert_eq!(
            estimate_encoding_bytes(&nested),
            struct_size + 2 * per_token + 3 + struct_size + per_token + 3
        );

        let empty = Encoding::Hf(Box::new(hf(&[], vec![])));
        assert_eq!(estimate_encoding_bytes(&empty), struct_size);
    }
}
