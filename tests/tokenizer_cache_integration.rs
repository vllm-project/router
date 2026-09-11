//! Integration tests for `CachedTokenizer` with real HuggingFace encodings.
//!
//! The default tests load `tests/fixtures/tokenizer/byte_level_bpe.json`, a
//! small byte-level BPE tokenizer checked into the repository (256 byte
//! symbols, a few merges, `<s>`/`</s>`/`<unk>` as special tokens). It handles
//! arbitrary UTF-8 and yields `Encoding::Hf` values with tokens, offsets and
//! masks, without network access. One `#[ignore]` test repeats the core
//! comparison with the TinyLlama tokenizer, which is downloaded on first use.

mod common;

use std::sync::Arc;
use tokenizers::{Tokenizer as HfTokenizer, TruncationParams};
use vllm_router_rs::tokenizer::{
    cache::{estimate_entry_bytes, CachedTokenizer, TokenizerCacheConfig},
    huggingface::HuggingFaceTokenizer,
    traits::*,
    Tokenizer as TokenizerWrapper,
};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/tokenizer/byte_level_bpe.json"
);

const PROMPTS: &[&str] = &[
    "",
    " ",
    "deep learning is",
    "Deep learning is",
    "Hello, world!",
    "<s> wrapped in special tokens </s>",
    "<unk> unknown and <s><s> repeated markers",
    "tabs\tand\nnewlines  and  double  spaces",
    "na\u{ef}ve caf\u{e9} \u{2014} accents and dashes",
    "\u{1f600}\u{1f603}\u{1f604}\u{1f601}\u{1f606} emoji \u{1f92a}\u{1f47b}",
    "\u{4f60}\u{597d}\u{ff0c}\u{4e16}\u{754c} CJK text",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor \
     incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud \
     exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
];

fn load_fixture() -> Arc<dyn Tokenizer> {
    Arc::new(HuggingFaceTokenizer::from_file(FIXTURE).expect("Failed to load fixture tokenizer"))
}

/// The fixture with truncation enabled, so long inputs produce overflowing
/// encodings.
fn load_fixture_with_overflow() -> Arc<dyn Tokenizer> {
    let mut tokenizer = HfTokenizer::from_file(FIXTURE).expect("Failed to load fixture tokenizer");
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: 8,
            stride: 2,
            ..TruncationParams::default()
        }))
        .expect("valid truncation");
    Arc::new(HuggingFaceTokenizer::from_tokenizer(tokenizer))
}

fn assert_hf_equal(expected: &Encoding, actual: &Encoding) {
    match (expected, actual) {
        (Encoding::Hf(expected), Encoding::Hf(actual)) => {
            assert_eq!(expected.get_ids(), actual.get_ids());
            assert_eq!(expected.get_tokens(), actual.get_tokens());
            assert_eq!(expected.get_offsets(), actual.get_offsets());
            assert_eq!(expected.get_type_ids(), actual.get_type_ids());
            assert_eq!(expected.get_word_ids(), actual.get_word_ids());
            assert_eq!(
                expected.get_special_tokens_mask(),
                actual.get_special_tokens_mask()
            );
            assert_eq!(expected.get_attention_mask(), actual.get_attention_mask());
            assert_eq!(expected.get_overflowing(), actual.get_overflowing());
            assert_eq!(expected, actual);
        }
        _ => panic!("expected HuggingFace encodings"),
    }
}

fn assert_cached_matches_uncached(inner: Arc<dyn Tokenizer>, prompts: &[&str]) {
    let cache = CachedTokenizer::new(inner.clone(), TokenizerCacheConfig::default()).unwrap();

    for prompt in prompts {
        let uncached = inner.encode(prompt).unwrap();
        let first = cache.encode(prompt).unwrap();
        let second = cache.encode(prompt).unwrap();
        assert_hf_equal(&uncached, &first);
        assert_hf_equal(&uncached, &second);
        assert_eq!(uncached.get_hash(), second.get_hash());
    }

    let stats = cache.stats();
    assert_eq!(stats.misses as usize, prompts.len());
    assert_eq!(stats.hits as usize, prompts.len());
    assert_eq!(stats.entries, prompts.len());
    assert_eq!(stats.evictions, 0);
    assert_eq!(stats.oversized, 0);
}

#[test]
fn fixture_is_the_expected_tokenizer() {
    let tokenizer = load_fixture();
    assert_eq!(tokenizer.vocab_size(), 290);
    assert_eq!(
        tokenizer.get_special_tokens().bos_token.as_deref(),
        Some("<s>")
    );
    assert_eq!(
        tokenizer.get_special_tokens().eos_token.as_deref(),
        Some("</s>")
    );
    assert_eq!(
        tokenizer.get_special_tokens().unk_token.as_deref(),
        Some("<unk>")
    );

    // Pinned output so a silently changed fixture is caught.
    let encoding = tokenizer.encode("Hello world").unwrap();
    assert_eq!(encoding.token_ids(), &[259, 264]);
    let encoding = tokenizer.encode("deep learning is the").unwrap();
    assert_eq!(encoding.token_ids(), &[100, 101, 101, 112, 272, 274, 277]);
    let encoding = tokenizer.encode("<s> hi </s>").unwrap();
    assert_eq!(encoding.token_ids(), &[290, 32, 104, 105, 32, 291]);
    assert_eq!(
        tokenizer.decode(encoding.token_ids(), true).unwrap(),
        " hi "
    );
}

#[test]
fn cached_encodings_match_uncached_including_metadata() {
    assert_cached_matches_uncached(load_fixture(), PROMPTS);
}

#[test]
fn overflowing_encodings_are_preserved() {
    let inner = load_fixture_with_overflow();
    let long = PROMPTS[PROMPTS.len() - 1];
    let Encoding::Hf(reference) = inner.encode(long).unwrap() else {
        panic!("expected HuggingFace encoding");
    };
    assert!(
        !reference.get_overflowing().is_empty(),
        "fixture truncation must produce overflowing encodings"
    );

    assert_cached_matches_uncached(inner.clone(), PROMPTS);

    let cache = CachedTokenizer::new(inner.clone(), TokenizerCacheConfig::default()).unwrap();
    let hit = {
        cache.encode(long).unwrap();
        cache.encode(long).unwrap()
    };
    let Encoding::Hf(hit) = hit else {
        panic!("expected HuggingFace encoding");
    };
    assert_eq!(hit.get_overflowing(), reference.get_overflowing());

    let without_overflow = load_fixture().encode(long).unwrap();
    assert!(
        estimate_entry_bytes(long, &Encoding::Hf(reference))
            > estimate_entry_bytes(long, &without_overflow),
        "overflowing encodings must count toward the estimate"
    );
}

#[test]
fn retained_bytes_track_estimates() {
    let inner = load_fixture();
    let cache = CachedTokenizer::new(inner.clone(), TokenizerCacheConfig::default()).unwrap();

    let mut expected_bytes = 0;
    for prompt in PROMPTS {
        let encoding = inner.encode(prompt).unwrap();
        let estimate = estimate_entry_bytes(prompt, &encoding);
        // Every retained byte we know about must be counted.
        assert!(estimate >= prompt.len() + encoding.token_ids().len() * 4);
        expected_bytes += estimate;
        cache.encode(prompt).unwrap();
        assert_eq!(cache.stats().bytes, expected_bytes);
    }
}

#[test]
fn byte_budget_evicts_oldest_real_encodings() {
    let inner = load_fixture();
    let short = "deep learning is";
    let long = PROMPTS[PROMPTS.len() - 1];
    let short_bytes = estimate_entry_bytes(short, &inner.encode(short).unwrap());
    let long_bytes = estimate_entry_bytes(long, &inner.encode(long).unwrap());
    assert!(long_bytes > short_bytes);

    let cache = CachedTokenizer::new(
        inner,
        TokenizerCacheConfig {
            max_entries: 100,
            max_bytes: long_bytes + short_bytes / 2,
            max_entry_bytes: long_bytes,
        },
    )
    .unwrap();

    cache.encode(short).unwrap();
    cache.encode(long).unwrap();
    let stats = cache.stats();
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.evictions, 1);
    assert_eq!(stats.bytes, long_bytes);

    cache.encode(long).unwrap();
    assert_eq!(cache.stats().hits, 1);
    cache.encode(short).unwrap();
    assert_eq!(cache.stats().misses, 3);
}

#[test]
fn oversized_real_encoding_is_returned_but_not_stored() {
    let inner = load_fixture();
    let long = PROMPTS[PROMPTS.len() - 1];
    let long_bytes = estimate_entry_bytes(long, &inner.encode(long).unwrap());
    let cache = CachedTokenizer::new(
        inner.clone(),
        TokenizerCacheConfig {
            max_entries: 100,
            max_bytes: 1 << 20,
            max_entry_bytes: long_bytes - 1,
        },
    )
    .unwrap();

    assert_hf_equal(&inner.encode(long).unwrap(), &cache.encode(long).unwrap());
    let stats = cache.stats();
    assert_eq!(stats.oversized, 1);
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.bytes, 0);
}

#[test]
fn works_behind_wrapper_with_decode_and_streaming() {
    let inner = load_fixture();
    let cached = Arc::new(CachedTokenizer::new(inner, TokenizerCacheConfig::default()).unwrap());
    let tokenizer = TokenizerWrapper::from_arc(cached.clone());

    let prompt = "The quick brown fox jumps over the lazy dog";
    let encoding = tokenizer.encode(prompt).unwrap();
    tokenizer.encode(prompt).unwrap();
    assert_eq!(cached.stats().hits, 1);

    // Byte-level BPE decodes losslessly.
    assert_eq!(
        tokenizer.decode(encoding.token_ids(), true).unwrap(),
        prompt
    );

    let mut stream = tokenizer.decode_stream(&[], true);
    let mut text = String::new();
    for &id in encoding.token_ids() {
        if let Some(chunk) = stream.step(id).unwrap() {
            text.push_str(&chunk);
        }
    }
    if let Some(chunk) = stream.flush().unwrap() {
        text.push_str(&chunk);
    }
    assert_eq!(text, prompt);

    // Batch encoding is delegated and leaves the cache untouched.
    let batch = tokenizer.encode_batch(&[prompt, "another prompt"]).unwrap();
    assert_eq!(batch.len(), 2);
    assert_eq!(cached.stats().entries, 1);
}

#[test]
fn concurrent_real_encodes_are_consistent() {
    use std::thread;

    let inner = load_fixture();
    let cache = Arc::new(
        CachedTokenizer::new(
            inner.clone(),
            TokenizerCacheConfig {
                max_entries: 6,
                max_bytes: 1 << 20,
                max_entry_bytes: 1 << 20,
            },
        )
        .unwrap(),
    );
    let expected: Vec<Encoding> = PROMPTS.iter().map(|p| inner.encode(p).unwrap()).collect();

    let handles: Vec<_> = (0..8)
        .map(|t| {
            let cache = cache.clone();
            let expected = expected.clone();
            thread::spawn(move || {
                for i in 0..100 {
                    let index = (i * 5 + t) % PROMPTS.len();
                    assert_hf_equal(&expected[index], &cache.encode(PROMPTS[index]).unwrap());
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }

    let stats = cache.stats();
    assert_eq!(stats.hits + stats.misses, 800);
    assert!(stats.entries <= 6);
    assert!(stats.evictions > 0);
}

/// Model-level check with the TinyLlama tokenizer. Downloads the file on
/// first use; run with `cargo test --test tokenizer_cache_integration -- --ignored`.
#[test]
#[ignore = "downloads the TinyLlama tokenizer from Hugging Face"]
fn tinyllama_cached_encodings_match_uncached() {
    let path = common::ensure_tokenizer_cached();
    let inner: Arc<dyn Tokenizer> = Arc::new(
        HuggingFaceTokenizer::from_file(path.to_str().unwrap())
            .expect("Failed to load TinyLlama tokenizer"),
    );
    assert_cached_matches_uncached(inner.clone(), PROMPTS);

    let cache = CachedTokenizer::new(inner, TokenizerCacheConfig::default()).unwrap();
    for _ in 0..2 {
        let hashes: Vec<u64> = common::TEST_PROMPTS
            .iter()
            .map(|prompt| cache.encode(prompt).unwrap().get_hash())
            .collect();
        assert_eq!(hashes, common::EXPECTED_HASHES);
    }
}
