//! Shared prompt encoding caches: exact-match L0, then prefix L1.
use std::hash::{BuildHasher, Hash, Hasher};
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Once};

use anyhow::{bail, Context, Result};
use vllm_text::Prompt;
use vllm_tokenizer::DynTokenizer;

use super::l1_encode_cache::{L1Bypass, L1Encoder};
use crate::tokenizer::cache::{entry_overhead_bytes, EncodeCache};
use crate::tokenizer::TokenizerCacheConfig;

/// The prompt is hashed once, before the cache lock is taken; the map only
/// rehashes `hash`.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct EncodingKey {
    hash: u64,
    instance: u64,
    add_special_tokens: bool,
    cache_salt: Option<String>,
    text: String,
}

impl EncodingKey {
    fn new(
        instance: u64,
        add_special_tokens: bool,
        cache_salt: Option<String>,
        text: String,
    ) -> Self {
        // Seeded per process, so prompts cannot be crafted to collide.
        static PROMPT_HASH: LazyLock<foldhash::quality::RandomState> =
            LazyLock::new(Default::default);
        Self {
            hash: PROMPT_HASH.hash_one((instance, add_special_tokens, &cache_salt, &text)),
            instance,
            add_special_tokens,
            cache_salt,
            text,
        }
    }
}

impl Hash for EncodingKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

// Hits share this allocation with `TokenizeOut` instead of copying it.
pub(crate) type FrontendCache = EncodeCache<EncodingKey, Arc<[u32]>>;

#[derive(Clone)]
pub(crate) struct PromptEncoder {
    tokenizer: DynTokenizer,
    cache: Option<Arc<FrontendCache>>,
    l1: Option<L1Encoder>,
    instance: u64,
    model_id: Arc<str>,
    oversized_warning: Arc<Once>,
}

/// Prompt IDs and how many of them came from L0 or L1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Encoded {
    pub ids: Arc<[u32]>,
    pub cached_tokens: usize,
}

impl PromptEncoder {
    // Caching requires a deterministic tokenizer with fixed configuration.
    pub(crate) fn new(
        model_id: &str,
        tokenizer: DynTokenizer,
        cache: Option<Arc<FrontendCache>>,
    ) -> Self {
        static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(0);
        Self {
            tokenizer,
            cache,
            l1: None,
            // Clones share an ID; new instances get distinct IDs.
            instance: NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed),
            model_id: Arc::from(model_id),
            oversized_warning: Arc::new(Once::new()),
        }
    }

    pub(crate) fn with_l1_cache(mut self, l1: Option<L1Encoder>) -> Self {
        self.l1 = l1;
        self
    }

    #[cfg(test)]
    pub(crate) fn encode(
        &self,
        prompt: Prompt,
        add_special_tokens: bool,
        cache_salt: Option<&str>,
    ) -> vllm_tokenizer::Result<Arc<[u32]>> {
        self.encode_detailed(prompt, add_special_tokens, cache_salt)
            .map(|encoded| encoded.ids)
    }

    /// L0 exact match first, then L1 prefix reuse, then the tokenizer.
    /// An L0 hit returns the cached `Arc` directly.
    pub(crate) fn encode_detailed(
        &self,
        prompt: Prompt,
        add_special_tokens: bool,
        cache_salt: Option<&str>,
    ) -> vllm_tokenizer::Result<Encoded> {
        let text = match prompt {
            Prompt::TokenIds(ids) => {
                if let Some(l1) = &self.l1 {
                    l1.record_bypass(L1Bypass::TokenIds);
                }
                return Ok(Encoded {
                    ids: Arc::from(ids),
                    cached_tokens: 0,
                });
            }
            Prompt::Text(text) => text,
        };
        let Some(cache) = &self.cache else {
            return self.encode_text(&text, add_special_tokens, cache_salt);
        };
        let key = EncodingKey::new(
            self.instance,
            add_special_tokens,
            cache_salt.map(str::to_owned),
            text,
        );
        let (ids, l1_cached) = cache.get_or_encode_with(
            &key,
            || {
                let encoded = self.encode_text(&key.text, add_special_tokens, cache_salt)?;
                Ok((encoded.ids, encoded.cached_tokens))
            },
            |ids| {
                let bytes = entry_overhead_bytes::<EncodingKey, Arc<[u32]>>()
                    + key.text.len()
                    + key.cache_salt.as_ref().map_or(0, String::len)
                    + size_of::<Arc<[u32]>>()
                    // Strong and weak counts precede the ids.
                    + 2 * size_of::<usize>()
                    + ids.len() * size_of::<u32>();
                if bytes > cache.config().max_entry_bytes {
                    self.oversized_warning.call_once(|| {
                        tracing::warn!(
                            model_id = %self.model_id,
                            estimated_bytes = bytes,
                            max_entry_bytes = cache.config().max_entry_bytes,
                            "L0 entry exceeds cache limit; encoding will not be cached"
                        );
                    });
                }
                bytes
            },
        )?;
        // An L0 hit serves every ID from the cache.
        let cached_tokens = l1_cached.unwrap_or(ids.len());
        Ok(Encoded { ids, cached_tokens })
    }

    fn encode_text(
        &self,
        text: &str,
        add_special_tokens: bool,
        cache_salt: Option<&str>,
    ) -> vllm_tokenizer::Result<Encoded> {
        if let Some(l1) = &self.l1 {
            // BOS/EOS post-processing is not prefix-stable, so L1 skips it.
            if add_special_tokens {
                l1.record_bypass(L1Bypass::AddSpecialTokens);
            } else if let Some((ids, cached_tokens)) =
                l1.encode(self.tokenizer.as_ref(), self.instance, text, cache_salt)?
            {
                return Ok(Encoded {
                    ids: Arc::from(ids),
                    cached_tokens,
                });
            }
        }
        Ok(Encoded {
            ids: Arc::from(self.tokenizer.encode(text, add_special_tokens)?),
            cached_tokens: 0,
        })
    }
}

pub(crate) fn config_from_env() -> Result<Option<TokenizerCacheConfig>> {
    parse_config(|name| std::env::var(name).ok())
}

fn parse_config(read: impl Fn(&str) -> Option<String>) -> Result<Option<TokenizerCacheConfig>> {
    match read("VLLM_ROUTER_L0_CACHE").as_deref() {
        None | Some("0" | "false") => return Ok(None),
        Some("1" | "true") => {}
        _ => bail!("VLLM_ROUTER_L0_CACHE must be 0, 1, false or true"),
    }
    let mut config = TokenizerCacheConfig::default();
    for (name, value) in [
        ("VLLM_ROUTER_L0_MAX_ENTRIES", &mut config.max_entries),
        ("VLLM_ROUTER_L0_MAX_BYTES", &mut config.max_bytes),
        (
            "VLLM_ROUTER_L0_MAX_ENTRY_BYTES",
            &mut config.max_entry_bytes,
        ),
    ] {
        if let Some(raw) = read(name) {
            *value = raw
                .parse()
                .with_context(|| format!("invalid {name}: {raw}"))?;
        }
    }
    config.validate()?;
    Ok(Some(config))
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use vllm_tokenizer::{HuggingFaceTokenizer, Tokenizer, TokenizerError};

    struct CountingTokenizer {
        inner: HuggingFaceTokenizer,
        calls: AtomicUsize,
        suffix: u32,
    }

    impl CountingTokenizer {
        fn new(suffix: u32) -> Arc<Self> {
            Arc::new(Self {
                inner: HuggingFaceTokenizer::new_hf(std::path::Path::new(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/rust/fixtures/tokenizer/byte_level_bpe.json"
                )))
                .unwrap(),
                calls: AtomicUsize::new(0),
                suffix,
            })
        }
    }

    impl Tokenizer for CountingTokenizer {
        fn encode(&self, text: &str, special: bool) -> vllm_tokenizer::Result<Vec<u32>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if text == "fail" {
                return Err(TokenizerError("test failure".into()));
            }
            let mut ids = self.inner.encode(text, special)?;
            ids.extend([self.suffix, u32::from(special)]);
            Ok(ids)
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

    fn encode(encoder: &PromptEncoder, text: &str, special: bool) -> Arc<[u32]> {
        encoder
            .encode(Prompt::Text(text.into()), special, None)
            .unwrap()
    }

    #[test]
    fn hits_skip_encoding_and_isolate_instances_options_and_prompts() {
        let cache = Arc::new(FrontendCache::new(Default::default()).unwrap());
        let inner = CountingTokenizer::new(10);
        let encoder = PromptEncoder::new("test-model", inner.clone(), Some(cache.clone()));
        let uncached = PromptEncoder::new("test-model", CountingTokenizer::new(10), None);
        for text in ["Hello", "hello", "你好 👋", ""] {
            for special in [false, true] {
                let expected = encode(&uncached, text, special);
                assert_eq!(encode(&encoder, text, special), expected);
                assert_eq!(encode(&encoder.clone(), text, special), expected);
            }
        }
        assert_eq!(inner.calls.load(Ordering::Relaxed), 8);
        assert_eq!(cache.stats().hits, 8);
        // A new encoder must miss, even with the same tokenizer Arc.
        let other = PromptEncoder::new("test-model", inner.clone(), Some(cache.clone()));
        encode(&other, "Hello", false);
        assert_eq!(inner.calls.load(Ordering::Relaxed), 9);
        let changed = PromptEncoder::new(
            "test-model",
            CountingTokenizer::new(20),
            Some(cache.clone()),
        );
        assert_ne!(
            encode(&changed, "Hello", false),
            encode(&encoder, "Hello", false)
        );
        assert_eq!(cache.stats().entries, 10);
    }

    #[test]
    fn keys_with_equal_hashes_stay_distinct() {
        let cache = FrontendCache::new(Default::default()).unwrap();
        let key =
            |instance, add_special_tokens, cache_salt: Option<&str>, text: &str| EncodingKey {
                hash: 7,
                instance,
                add_special_tokens,
                cache_salt: cache_salt.map(str::to_owned),
                text: text.to_owned(),
            };
        let keys = [
            key(0, false, None, "a"),
            key(0, false, None, "b"),
            key(1, false, None, "a"),
            key(0, true, None, "a"),
            key(0, false, Some("s"), "a"),
        ];
        for round in 0..2 {
            for (i, key) in keys.iter().enumerate() {
                let (ids, miss) = cache
                    .get_or_encode_with(
                        key,
                        || Ok::<_, ()>((Arc::<[u32]>::from([i as u32]), ())),
                        |_| 64,
                    )
                    .unwrap();
                assert_eq!(&*ids, &[i as u32]);
                assert_eq!(miss.is_some(), round == 0);
            }
        }
        assert_eq!(cache.stats().entries, keys.len());
    }

    #[test]
    fn cache_salt_isolates_entries_and_counts_retained_bytes() {
        let cache = Arc::new(FrontendCache::new(Default::default()).unwrap());
        let inner = CountingTokenizer::new(0);
        let encoder = PromptEncoder::new("test-model", inner.clone(), Some(cache.clone()));
        let expected = encode(
            &PromptEncoder::new("test-model", CountingTokenizer::new(0), None),
            "Hello",
            false,
        );
        let mut unsalted_bytes = 0;
        for salt in [None, Some(""), Some("tenant-a"), Some("tenant-b")] {
            let before = cache.stats().bytes;
            for _ in 0..2 {
                assert_eq!(
                    encoder
                        .encode(Prompt::Text("Hello".into()), false, salt)
                        .unwrap(),
                    expected
                );
            }
            let retained = cache.stats().bytes - before;
            if salt.is_none() {
                unsalted_bytes = retained;
            }
            assert_eq!(retained, unsalted_bytes + salt.map_or(0, str::len));
        }
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses, stats.entries), (4, 4, 4));
        assert_eq!(inner.calls.load(Ordering::Relaxed), 4);

        let small = Arc::new(
            FrontendCache::new(TokenizerCacheConfig {
                max_entry_bytes: unsalted_bytes,
                ..Default::default()
            })
            .unwrap(),
        );
        let encoder = PromptEncoder::new("test-model", inner, Some(small.clone()));
        assert_eq!(encode(&encoder, "Hello", false), expected);
        assert_eq!(
            encoder
                .encode(Prompt::Text("Hello".into()), false, Some("tenant-a"))
                .unwrap(),
            expected
        );
        assert_eq!(small.stats().oversized, 1);
        assert_eq!(small.stats().bytes, unsalted_bytes);
    }

    #[test]
    fn cached_tokens_count_only_l0_hits_without_l1() {
        let cached = |encoder: &PromptEncoder| {
            encoder
                .encode_detailed(Prompt::Text("Hello".into()), false, None)
                .unwrap()
                .cached_tokens
        };
        let cache = Arc::new(FrontendCache::new(Default::default()).unwrap());
        let encoder = PromptEncoder::new("test-model", CountingTokenizer::new(0), Some(cache));
        let len = encode(&encoder, "Hello", false).len();
        assert_eq!(cached(&encoder), len);

        let tiny = Arc::new(
            FrontendCache::new(TokenizerCacheConfig {
                max_entry_bytes: 1,
                ..Default::default()
            })
            .unwrap(),
        );
        let encoder = PromptEncoder::new("test-model", CountingTokenizer::new(0), Some(tiny));
        assert_eq!((cached(&encoder), cached(&encoder)), (0, 0));
    }

    #[test]
    fn errors_and_token_ids_are_never_cached() {
        let cache = Arc::new(FrontendCache::new(Default::default()).unwrap());
        let inner = CountingTokenizer::new(0);
        let encoder = PromptEncoder::new("test-model", inner.clone(), Some(cache.clone()));
        for _ in 0..2 {
            assert!(encoder
                .encode(Prompt::Text("fail".into()), false, None)
                .is_err());
            assert_eq!(
                *encoder
                    .encode(Prompt::TokenIds(vec![5, 4, 3]), true, Some("tenant-a"))
                    .unwrap(),
                [5, 4, 3]
            );
        }
        assert_eq!(inner.calls.load(Ordering::Relaxed), 2);
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().bytes, 0);
    }

    #[test]
    fn shared_budget_evicts_across_models_and_skips_oversized_inputs() {
        let measure = Arc::new(FrontendCache::new(Default::default()).unwrap());
        encode(
            &PromptEncoder::new(
                "test-model",
                CountingTokenizer::new(0),
                Some(measure.clone()),
            ),
            "a",
            false,
        );
        let per_entry = measure.stats().bytes;
        for config in [
            TokenizerCacheConfig {
                max_entries: 1,
                max_bytes: per_entry * 4,
                max_entry_bytes: per_entry * 2,
            },
            TokenizerCacheConfig {
                max_entries: 100,
                max_bytes: per_entry,
                max_entry_bytes: per_entry,
            },
        ] {
            let cache = Arc::new(FrontendCache::new(config).unwrap());
            let a =
                PromptEncoder::new("test-model", CountingTokenizer::new(0), Some(cache.clone()));
            let b =
                PromptEncoder::new("test-model", CountingTokenizer::new(1), Some(cache.clone()));
            encode(&a, "a", false);
            encode(&b, "a", false);
            assert_eq!(cache.stats().entries, 1);
            assert_eq!(cache.stats().bytes, per_entry);
            assert_eq!(cache.stats().evictions, 1);
            encode(&a, "a", false);
            assert_eq!(cache.stats().evictions, 2);
            encode(&a, &"long ".repeat(100), false);
            assert_eq!(cache.stats().oversized, 1);
            assert_eq!(cache.stats().bytes, per_entry);
            cache.clear();
            assert_eq!(cache.stats().bytes, 0);
        }
    }

    #[test]
    fn concurrent_models_stay_within_one_budget() {
        let config = TokenizerCacheConfig {
            max_entries: 8,
            max_bytes: 4096,
            max_entry_bytes: 1024,
        };
        let cache = Arc::new(FrontendCache::new(config).unwrap());
        let a = PromptEncoder::new("test-model", CountingTokenizer::new(0), Some(cache.clone()));
        let b = PromptEncoder::new("test-model", CountingTokenizer::new(1), Some(cache.clone()));
        std::thread::scope(|scope| {
            for t in 0..8 {
                let encoder = if t % 2 == 0 { a.clone() } else { b.clone() };
                scope.spawn(move || {
                    for i in 0..100 {
                        let text = format!("Hello {}", i % 8);
                        let expected = encoder.tokenizer.encode(&text, false).unwrap();
                        assert_eq!(encode(&encoder, &text, false)[..], expected);
                    }
                });
            }
        });
        let stats = cache.stats();
        assert_eq!(stats.hits + stats.misses, 800);
        assert!(stats.entries <= 8);
        assert!(stats.bytes <= 4096);
        assert!(stats.evictions > 0);
    }

    #[test]
    fn hits_share_one_allocation_that_outlives_eviction() {
        let cache = Arc::new(
            FrontendCache::new(TokenizerCacheConfig {
                max_entries: 1,
                ..Default::default()
            })
            .unwrap(),
        );
        let encoder =
            PromptEncoder::new("test-model", CountingTokenizer::new(0), Some(cache.clone()));
        let uncached = PromptEncoder::new("test-model", CountingTokenizer::new(0), None);
        let expected = encode(&uncached, "Hello", false);

        let miss = encode(&encoder, "Hello", false);
        let hit = encode(&encoder, "Hello", false);
        assert!(Arc::ptr_eq(&miss, &hit));
        assert_eq!(hit, expected);
        // Two callers plus the cache entry.
        assert_eq!(Arc::strong_count(&hit), 3);

        encode(&encoder, "evict", false);
        assert_eq!(cache.stats().evictions, 1);
        drop(miss);
        // The in-flight holder keeps the only reference after eviction.
        assert_eq!(Arc::strong_count(&hit), 1);
        assert_eq!(hit, expected);

        let reencoded = encode(&encoder, "Hello", false);
        assert!(!Arc::ptr_eq(&hit, &reencoded));
        assert_eq!(reencoded, expected);
        cache.clear();
        assert_eq!(Arc::strong_count(&reencoded), 1);
        assert_eq!(reencoded, expected);
    }

    #[test]
    fn config_is_opt_in_and_validates_shared_limits() {
        let parse = |values: &[(&str, &str)]| {
            parse_config(|key| {
                values
                    .iter()
                    .find(|(k, _)| *k == key)
                    .map(|(_, v)| v.to_string())
            })
        };
        assert_eq!(parse(&[]).unwrap(), None);
        assert_eq!(parse(&[("VLLM_ROUTER_L0_MAX_BYTES", "bad")]).unwrap(), None);
        assert_eq!(
            parse(&[("VLLM_ROUTER_L0_CACHE", "true")]).unwrap(),
            Some(Default::default())
        );
        assert!(parse(&[("VLLM_ROUTER_L0_CACHE", "maybe")]).is_err());
        for (name, value) in [
            ("VLLM_ROUTER_L0_MAX_ENTRIES", "0"),
            ("VLLM_ROUTER_L0_MAX_BYTES", "1"),
            ("VLLM_ROUTER_L0_MAX_ENTRY_BYTES", "bad"),
        ] {
            assert!(parse(&[("VLLM_ROUTER_L0_CACHE", "1"), (name, value)]).is_err());
        }
    }
}
