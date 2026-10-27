#![cfg(all(feature = "cpu", feature = "ready"))]

use super::boundary::*;
use super::tree::*;
use super::*;
use metrics::{
    Counter, Gauge, GaugeFn, Histogram, Key, KeyName, Metadata, Recorder, SharedString, Unit,
};
use serde_json::json;
use serde_json::Value;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::thread;
use vllm_text::backend::hf::{ResolvedModelFiles, TokenizerSource};
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
    let matcher = BoundaryMatcher::new(&token_set(vec![spec(">x"), spec("a>x")], true)).unwrap();
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
fn decoder_ids_that_do_not_fit_in_u32_are_not_boundaries() {
    // vllm-tokenizer 0.29 parses the key as u32 and skips a parse failure.
    // "4294967296" is 2^32. Admitting its content would cut where the
    // tokenizer still sees ordinary BPE.
    let tokens = added_tokens(
        &json!({
            "4294967296": {"content": "X"},
            "4294967295": {"content": "<max>"},
            "nope": {"content": "Y"}
        }),
        None,
    );
    let names: Vec<_> = tokens.iter().map(|token| token.content.as_str()).collect();
    assert_eq!(names, ["<max>"]);
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
        "/tests/rust/fixtures/tokenizer/byte_level_bpe.json"
    )))
    .unwrap()
}

fn encoder(config: L1CacheConfig) -> (L1Encoder, Arc<L1Cache>) {
    let cache = Arc::new(L1Cache::new(config).unwrap());
    let matcher = BoundaryMatcher::new(&token_set(vec![spec("<s>"), spec("</s>")], true))
        .unwrap()
        .with_token_ids(&fixture_tokenizer());
    (L1Encoder::new("test-model", cache.clone(), matcher), cache)
}

fn check(encoder: &L1Encoder, tokenizer: &dyn Tokenizer, text: &str) -> usize {
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
fn bytes_count_node_contents_and_container_capacity() {
    let tokenizer = fixture_tokenizer();
    let (encoder, cache) = encoder(Default::default());
    check(&encoder, &tokenizer, "<s>system hi</s><s>user</s>");
    let tree = cache.tree.read();
    let containers = tree.nodes.capacity() * size_of::<Option<Box<Node>>>()
        + tree.free.capacity() * size_of::<usize>()
        + tree.leaves.capacity() * size_of::<Reverse<(u64, usize)>>()
        + tree.edges.allocation_size()
        + tree.roots.allocation_size();
    assert!(tree.edges.allocation_size() > 0 && tree.roots.allocation_size() > 0);
    assert_eq!(tree.bytes(), tree.content_bytes + containers);
    for node in tree.nodes.iter().flatten() {
        // At least the struct, the edge text, the IDs and the `Arc` counts.
        let raw = size_of::<Node>() + node.text.len() + 4 * node.ids.len() + 16;
        assert!(node.bytes() >= raw);
    }
}

#[test]
fn evictions_release_container_capacity() {
    let tokenizer = fixture_tokenizer();
    let (encoder, cache) = encoder(L1CacheConfig::default());
    for i in 0..256 {
        check(&encoder, &tokenizer, &format!("{i}</s>"));
    }
    let (edges, bytes) = {
        let tree = cache.tree.read();
        (tree.edges.capacity(), tree.bytes())
    };
    assert!(edges >= 256);
    {
        let mut tree = cache.tree.write();
        // Keep the root and one leaf.
        assert_eq!(tree.evict(usize::MAX, 2), 255);
        assert!(tree.edges.capacity() < edges / 4);
        assert!(tree.bytes() < bytes / 4);
    }
    check_invariants(&cache);
    // An empty tree holds no storage at all.
    cache.tree.write().evict(0, 0);
    assert_eq!(cache.stats().bytes, 0);
    assert!(check(&encoder, &tokenizer, "0</s>") == 0);
    check_invariants(&cache);
}

#[test]
fn eviction_is_least_recently_used() {
    let tokenizer = fixture_tokenizer();
    let (encoder, cache) = encoder(L1CacheConfig {
        max_nodes: 4,
        ..Default::default()
    });
    for text in ["a</s>", "b</s>", "c</s>"] {
        check(&encoder, &tokenizer, text);
    }
    // A hit refreshes `a`, so `b` is the oldest leaf when `d` arrives.
    assert!(check(&encoder, &tokenizer, "a</s>") > 0);
    check(&encoder, &tokenizer, "d</s>");
    assert_eq!(cache.stats().evictions, 1);
    for text in ["a</s>", "c</s>", "d</s>"] {
        assert!(check(&encoder, &tokenizer, text) > 0, "{text}");
    }
    assert_eq!(check(&encoder, &tokenizer, "b</s>"), 0);
    check_invariants(&cache);
}

#[test]
fn insert_walks_again_when_the_tree_changed_after_lookup() {
    let tokenizer = fixture_tokenizer();
    let (encoder, cache) = encoder(Default::default());
    let namespace = Namespace {
        instance: 0,
        salt: None,
    };
    let insert_after = |text: &str, change: &dyn Fn()| {
        let first_cut = |from| encoder.matcher.first_cut(text, from);
        let hit = cache.lookup(&namespace, text, first_cut);
        change();
        let cuts = encoder.matcher.cuts(text, hit.bytes);
        let tail = tokenizer.encode(&text[hit.bytes..], false).unwrap();
        let bounds = encoder
            .tail_bounds(&tokenizer, text, &hit, &cuts, &tail)
            .unwrap();
        let mut ids: Vec<u32> = hit
            .pieces
            .iter()
            .flat_map(|(piece, take)| piece[..*take].to_vec())
            .collect();
        ids.extend(tail);
        cache.insert(&namespace, text, &ids, hit, &bounds, first_cut);
    };
    // Another request stores a shared prefix in between: the insert
    // walks again, splits that edge and appends only its own turn.
    let text = "<s>sys</s><s>user one</s><s>";
    insert_after(text, &|| {
        check(&encoder, &tokenizer, "<s>sys</s><s>user two</s><s>");
    });
    check_invariants(&cache);
    let all = tokenizer.encode(text, false).unwrap().len();
    assert_eq!(check(&encoder, &tokenizer, text), all);
    // Inserts into another namespace leave this one's steps reusable.
    let next = format!("{text}assistant</s>");
    let hit = cache.lookup(&namespace, &next, |from| {
        encoder.matcher.first_cut(&next, from)
    });
    encoder
        .encode(&tokenizer, 1, "<s>sys</s><s>other model</s><s>", None)
        .unwrap();
    let root = hit.root.unwrap();
    assert_eq!(cache.tree.read().node(root).changed, hit.version);
    // An eviction removes the matched path in between: the cuts before
    // the match have no offsets, so nothing is stored.
    insert_after(&format!("{text}assistant</s>"), &|| {
        cache.tree.write().evict(0, 0);
    });
    check_invariants(&cache);
    assert_eq!(cache.stats().nodes, 0);
}

#[test]
fn lookups_scan_for_cuts_only_where_a_node_has_several_children() {
    let tokenizer = fixture_tokenizer();
    let (encoder, cache) = encoder(Default::default());
    let namespace = Namespace {
        instance: 0,
        salt: None,
    };
    let mut chat = String::new();
    for turn in 0..6 {
        chat.push_str(&format!("<s>turn {turn}</s>"));
        check(&encoder, &tokenizer, &chat);
    }
    let scans = |text: &str| {
        let calls = std::cell::Cell::new(0);
        let hit = cache.lookup(&namespace, text, |from| {
            calls.set(calls.get() + 1);
            encoder.matcher.first_cut(text, from)
        });
        (hit.bytes, calls.get())
    };
    assert_eq!(scans(&chat), (chat.len(), 0));
    // A sibling after turn 2 makes that one step pick by first segment.
    check(
        &encoder,
        &tokenizer,
        "<s>turn 0</s><s>turn 1</s><s>turn 2</s><s>other</s>",
    );
    assert_eq!(scans(&chat), (chat.len(), 1));
    check_invariants(&cache);
}

#[test]
fn walk_finds_the_remaining_child_after_the_last_added_one_is_evicted() {
    let tokenizer = fixture_tokenizer();
    let (encoder, cache) = encoder(Default::default());
    let (x, y) = ("<s>a</s><s>x</s>", "<s>a</s><s>y</s>");
    check(&encoder, &tokenizer, x);
    check(&encoder, &tokenizer, y);
    {
        let mut tree = cache.tree.write();
        let leaf = tree.nodes.iter().position(|node| {
            node.as_deref()
                .is_some_and(|node| node.is_leaf() && &*node.text == "y</s>")
        });
        let leaf = leaf.unwrap();
        let parent = tree.node(leaf).parent;
        tree.remove_leaf(leaf);
        assert_eq!(tree.node(parent).child_count, 1);
        assert_eq!(tree.node(parent).child, leaf, "the hint is stale");
    }
    check_invariants(&cache);
    let all = tokenizer.encode(x, false).unwrap().len();
    assert_eq!(check(&encoder, &tokenizer, x), all);
    // A new child refreshes the hint.
    check(&encoder, &tokenizer, "<s>a</s><s>z</s>");
    assert_eq!(check(&encoder, &tokenizer, x), all);
    check_invariants(&cache);
}

#[test]
fn cut_offsets_fall_back_to_segments_when_boundary_ids_do_not_line_up() {
    use std::sync::atomic::AtomicUsize;

    // Emits `<s>`'s ID for plain "zz" too, which never crosses a cut.
    struct Spurious {
        inner: HuggingFaceTokenizer,
        calls: AtomicUsize,
    }
    impl Tokenizer for Spurious {
        fn encode(&self, text: &str, special: bool) -> vllm_tokenizer::Result<Vec<u32>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.inner.encode(&text.replace("zz", "<s>"), special)
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
    let tokenizer = Spurious {
        inner: fixture_tokenizer(),
        calls: AtomicUsize::new(0),
    };
    let (encoder, cache) = encoder(Default::default());
    let text = &format!("<s>zz one{}</s><s>two</s><s>", " word".repeat(60));
    assert_eq!(check(&encoder, &tokenizer, text), 0);
    // The tail encode, one encode per segment for the offsets, and the
    // reference encode in `check`.
    assert_eq!(tokenizer.calls.load(Ordering::Relaxed), 1 + 5 + 1);
    let all = tokenizer.encode(text, false).unwrap().len();
    assert_eq!(check(&encoder, &tokenizer, text), all);
    check_invariants(&cache);
    assert_eq!(cache.stats().hits, 1);
}

#[test]
fn salt_is_one_allocation_and_the_insert_cap_counts_marks() {
    let tokenizer = fixture_tokenizer();
    let text = "<s>system hi</s><s>user</s>";
    {
        let (encoder, cache) = encoder(Default::default());
        check(&encoder, &tokenizer, text);
        let plain = cache.tree.read().content_bytes;
        let salt = "abcdef";
        // Same text again, so this lookup hits.
        check(&encoder, &tokenizer, text);
        // A new salt is a new root and leaf.
        let (_, _) = encoder
            .encode(&tokenizer, 0, text, Some(salt))
            .unwrap()
            .unwrap();
        assert_eq!(
            cache.tree.read().content_bytes,
            plain * 2 + heap_bytes(2 * size_of::<usize>() + salt.len())
        );
        {
            let tree = cache.tree.read();
            let root = tree
                .nodes
                .iter()
                .flatten()
                .find(|node| {
                    node.namespace
                        .as_ref()
                        .and_then(|ns| ns.salt.as_ref())
                        .is_some_and(|stored| stored.as_ref() == salt)
                })
                .unwrap();
            // The node and the roots map share one string.
            assert_eq!(
                Arc::strong_count(root.namespace.as_ref().unwrap().salt.as_ref().unwrap()),
                2
            );
        }
        check_invariants(&cache);
    }

    // Nine `<s>` tokens: the leaf and the new namespace root fit the
    // insert cap, but not once the leaf's interior marks are counted.
    let nine = "<s>".repeat(9);
    let ids = tokenizer.encode(&nine, false).unwrap();
    let leaf_alone = node_bytes(nine.len(), ids.len(), 0, None) + node_bytes(0, 0, 0, None);
    let (encoder, cache) = encoder(L1CacheConfig {
        max_insert_bytes: leaf_alone,
        max_nodes: 32,
        ..Default::default()
    });
    check(&encoder, &tokenizer, &nine);
    let tree = cache.tree.read();
    assert!(
        tree.nodes
            .iter()
            .flatten()
            .all(|node| node.text.len() != nine.len()),
        "the full leaf is over the insert cap once marks and the root are counted"
    );
    assert!(tree.live > 0 && tree.content_bytes <= leaf_alone);
    drop(tree);
    check_invariants(&cache);
}

#[test]
fn eviction_keeps_budget_and_recent_paths() {
    let tokenizer = fixture_tokenizer();
    let config = L1CacheConfig {
        max_bytes: 8 * 1024,
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
    let (encoder, cache) = encoder(L1CacheConfig {
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
        let name = if key.name() == "vllm_tokenizer_l1_nodes" {
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
                    let (encoder, _cache) = encoder(L1CacheConfig {
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
    Node::new(
        parent,
        Box::from(text),
        Arc::from([]),
        Box::from([]),
        0,
        None,
    )
}

/// Children of `parent` whose first segment shares `segment`'s key.
fn bucket_len(tree: &Tree, parent: usize, segment: &str) -> usize {
    let key = EdgeKey {
        parent,
        segment: tree.key(segment),
    };
    let mut id = tree.edges.get(&key).copied().unwrap_or(NO_NODE);
    let mut len = 0;
    while id != NO_NODE {
        len += 1;
        id = tree.node(id).next_same_key;
    }
    len
}

fn timed_lookup(iters: usize, mut lookup: impl FnMut() -> Option<usize>) -> std::time::Duration {
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
    encoder: &L1Encoder,
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

fn build_lookup_tree(segments: &[String], strategy: L1HashStrategy) -> (Tree, usize) {
    let mut tree = Tree::new(strategy);
    let mut root = lookup_report_node(NO_NODE, String::new());
    root.namespace = Some(Namespace {
        instance: 0,
        salt: None,
    });
    let root = tree.alloc(root);
    for segment in segments {
        let id = tree.alloc(lookup_report_node(root, segment.clone()));
        tree.add_child(root, id);
    }
    (tree, root)
}

fn hash_report_encoder(strategy: L1HashStrategy) -> (L1Encoder, Arc<L1Cache>) {
    encoder(L1CacheConfig {
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

fn warm_hash_report_encoder(encoder: &L1Encoder, tokenizer: &dyn Tokenizer, segments: &[String]) {
    for segment in segments {
        check(encoder, tokenizer, &hash_report_prompt(segment));
    }
}

fn strategy_name(strategy: L1HashStrategy) -> &'static str {
    match strategy {
        L1HashStrategy::FirstLast64 => "first_last_64",
        L1HashStrategy::Full => "full",
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
                    build_lookup_tree(&segments, L1HashStrategy::FirstLast64);
                let (full_tree, full_root) = build_lookup_tree(&segments, L1HashStrategy::Full);
                let first_last_bucket = bucket_len(&first_last_tree, first_last_root, query);
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
                    first_last_tree.child(first_last_root, query)
                });
                let full_lookup = timed_lookup(lookup_iters, || full_tree.child(full_root, query));
                report_lookup_row(
                    case,
                    len,
                    siblings,
                    first_last_bucket,
                    lookup_iters,
                    first_last_lookup,
                    full_lookup,
                );

                let (first_last_encoder, _) = hash_report_encoder(L1HashStrategy::FirstLast64);
                let (full_encoder, _) = hash_report_encoder(L1HashStrategy::Full);
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
        strategy_name(L1HashStrategy::FirstLast64),
        strategy_name(L1HashStrategy::Full)
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
                    build_lookup_tree(&segments, L1HashStrategy::FirstLast64);
                let (full_tree, full_root) = build_lookup_tree(&segments, L1HashStrategy::Full);
                let first_last_bucket = bucket_len(&first_last_tree, first_last_root, query);
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
                    first_last_tree.child(first_last_root, query)
                });
                let full_lookup = timed_lookup(lookup_iters, || full_tree.child(full_root, query));
                report_lookup_row(
                    dims,
                    first_last_bucket,
                    lookup_iters,
                    first_last_lookup,
                    full_lookup,
                );

                let (first_last_encoder, _) = hash_report_encoder(L1HashStrategy::FirstLast64);
                let (full_encoder, _) = hash_report_encoder(L1HashStrategy::Full);
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

fn check_invariants(cache: &L1Cache) {
    let tree = cache.tree.read();
    let live: Vec<usize> = (0..tree.nodes.len())
        .filter(|&id| tree.nodes[id].is_some())
        .collect();
    assert_eq!(live.len(), tree.live);
    assert_eq!(
        live.iter().map(|&id| tree.node(id).bytes()).sum::<usize>(),
        tree.content_bytes
    );
    assert!(tree.bytes() <= cache.config.max_bytes);
    assert!(tree.live <= cache.config.max_nodes);
    // Each edge entry chains children of its parent under its key.
    let mut children = vec![0; tree.nodes.len()];
    for (key, &head) in &tree.edges {
        let mut id = head;
        while id != NO_NODE {
            assert_eq!(tree.node(id).parent, key.parent);
            assert!(tree.edge(key.parent, id) == *key);
            children[key.parent] += 1;
            id = tree.node(id).next_same_key;
        }
    }
    assert_eq!(
        children.iter().sum::<usize>(),
        live.len() - tree.roots.len()
    );
    let queued: HashSet<usize> = tree.leaves.iter().map(|&Reverse((_, id))| id).collect();
    for &id in &live {
        let node = tree.node(id);
        assert_eq!(children[id], node.child_count);
        if let Some(namespace) = &node.namespace {
            assert_eq!(tree.roots[namespace], id);
            assert_eq!(node.root, id);
            assert!(node.child_count > 0, "empty roots are removed");
            continue;
        }
        assert_eq!(node.root, tree.node(node.parent).root);
        assert!(!node.text.is_empty());
        if node.is_leaf() {
            assert!(queued.contains(&id), "every leaf has a heap entry");
        }
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
    for budget in [8 * 1024, 16 * 1024, 1 << 20] {
        let (encoder, cache) = encoder(L1CacheConfig {
            max_bytes: budget,
            max_nodes: 24,
            // The insert cap is the whole budget, so inserts also evict.
            max_insert_bytes: budget,
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
                _ => conversation.push((0..next(4)).map(|_| words[next(words.len())]).collect()),
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
            L1Cache::new(L1CacheConfig {
                max_bytes: 1 << 20,
                max_nodes: 512,
                max_insert_bytes: 1 << 18,
                ..Default::default()
            })
            .unwrap(),
        );
        let encoder = L1Encoder::new(dir, cache.clone(), matcher);
        let words = [
            "hello", " world", "  ", "\n", "\n\n", "\t", "数据", "😀", "é", "\u{301}", "\u{338}",
            ">", "<", "|", "▁", "Ġ", "'s", "123", " 4", "<|", "|>", "</", "\u{a0}", "ﬁ", "Å",
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
fn tail_is_one_encode_unless_short() {
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
    let (capped, _) = encoder(L1CacheConfig {
        max_insert_segments: 2,
        ..Default::default()
    });
    let long = " word".repeat(60);
    let text = &format!("<s>one{long}</s><s>two</s><s>three</s><s>four</s>tail");
    let (ids, cached) = capped.encode(&tokenizer, 0, text, None).unwrap().unwrap();
    assert_eq!(ids, tokenizer.inner.encode(text, false).unwrap());
    assert_eq!(cached, 0);
    // One encode of the tail; boundary IDs give the stored offsets.
    assert_eq!(tokenizer.calls.load(Ordering::Relaxed), 1);

    // A short tail is encoded one segment at a time.
    let (fresh, _) = encoder(L1CacheConfig::default());
    tokenizer.calls.store(0, Ordering::Relaxed);
    let text = "<s>one</s><s>two</s><s>three</s><s>four</s>tail";
    let (ids, _) = fresh.encode(&tokenizer, 0, text, None).unwrap().unwrap();
    assert_eq!(ids, tokenizer.inner.encode(text, false).unwrap());
    let segments = fresh.matcher.cuts(text, 0).len() + 1;
    assert_eq!(tokenizer.calls.load(Ordering::Relaxed), segments);
    let (_, cached) = fresh.encode(&tokenizer, 0, text, None).unwrap().unwrap();
    assert!(cached > 0);
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
    let (encoder, cache) = encoder(L1CacheConfig {
        verify: true,
        ..Default::default()
    });
    let text = &format!("<s>system{}</s><s>user hi</s>", " hello there".repeat(25));
    let expected = (tokenizer.encode(text, false).unwrap(), 0);
    // The cold request is one encode of the whole prompt; the reuse on
    // the next request is what disagrees.
    let encode = || encoder.encode(&tokenizer, 0, text, None).unwrap();
    assert_eq!(encode().unwrap(), expected);
    assert_eq!(cache.stats().verify_mismatches, 0);
    assert_eq!(encode().unwrap(), expected);
    assert_eq!(cache.stats().verify_mismatches, 1);
    assert!(encoder.encode(&tokenizer, 0, text, None).unwrap().is_none());
    assert_eq!(cache.stats().bypasses, 1);
}

#[test]
fn bypasses_are_counted_only_when_l1_is_skipped() {
    use super::super::encode_cache::PromptEncoder;
    use vllm_text::Prompt;

    let (l1, cache) = encoder(Default::default());
    let prompts =
        PromptEncoder::new("fixture", Arc::new(fixture_tokenizer()), None).with_l1_cache(Some(l1));
    let encode = |prompt, add_special_tokens| {
        prompts.encode(prompt, add_special_tokens, None).unwrap();
    };
    encode(Prompt::TokenIds(vec![1, 2]), false);
    encode(Prompt::Text("<s>hi</s>".into()), true);
    encode(Prompt::Text("no boundary here".into()), false);
    assert_eq!((cache.stats().bypasses, cache.stats().lookups), (3, 0));
    encode(Prompt::Text("<s>hi</s>".into()), false);
    assert_eq!((cache.stats().bypasses, cache.stats().lookups), (3, 1));
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
        L1HashStrategy::Full
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

fn long_chat_turn(i: usize) -> String {
    format!(
        "<s>user {i}: {}</s>\n",
        "lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(8)
    )
}

fn timings(mut f: impl FnMut()) -> (std::time::Duration, std::time::Duration) {
    let mut samples: Vec<_> = (0..30)
        .map(|_| {
            let start = std::time::Instant::now();
            f();
            start.elapsed()
        })
        .collect();
    samples.sort();
    (samples[0], samples[samples.len() / 2])
}

fn percentiles(mut samples: Vec<std::time::Duration>) -> String {
    samples.sort();
    let at = |q: f64| samples[((samples.len() - 1) as f64 * q) as usize];
    format!(
        "p50 {:?} p99 {:?} max {:?} (n={})",
        at(0.5),
        at(0.99),
        at(1.0),
        samples.len()
    )
}

/// Synthetic ~440 KB chat with 1,862 boundaries built over 118 growing
/// turns, the shape of the 128k-token review probe.
///
/// `cargo test --release --lib l1_long_chat_report -- --ignored --nocapture`
#[test]
#[ignore = "CPU-only performance report; run in release mode"]
fn l1_long_chat_report() {
    use super::super::encode_cache::{FrontendCache, PromptEncoder};
    use crate::tokenizer::TokenizerCacheConfig;
    use vllm_text::Prompt;

    let tokenizer = Arc::new(fixture_tokenizer());
    let turns: Vec<String> = (0..931).map(long_chat_turn).collect();
    let prompt = |n: usize| format!("{}<s>assistant\n", turns[..n].concat());
    let full = prompt(931);
    let config = L1CacheConfig {
        max_bytes: 1 << 30,
        max_insert_bytes: 1 << 28,
        ..Default::default()
    };
    println!(
        "prompt bytes {} boundaries {} tokens {}",
        full.len(),
        encoder(config).0.matcher.boundaries(&full).len(),
        tokenizer.encode(&full, false).unwrap().len()
    );

    let (single_min, _) = timings(|| {
        std::hint::black_box(tokenizer.encode(&full, false).unwrap());
    });
    let mut cold = Vec::new();
    for _ in 0..5 {
        let (encoder, _) = encoder(config);
        let start = std::time::Instant::now();
        let ids = encoder.encode(tokenizer.as_ref(), 0, &full, None).unwrap();
        cold.push(start.elapsed());
        assert_eq!(ids.unwrap().0, tokenizer.encode(&full, false).unwrap());
    }
    cold.sort();
    println!("cold: L1 {:?} vs single-pass {single_min:?}", cold[0]);

    let (encoder, cache) = encoder(config);
    for step in 1..=118 {
        check(&encoder, tokenizer.as_ref(), &prompt(931 * step / 118));
    }
    let warm = cache.stats();
    let (min, median) = timings(|| {
        encoder.encode(tokenizer.as_ref(), 0, &full, None).unwrap();
    });
    println!(
        "full hit: min {min:?} median {median:?} (nodes {} bytes {})",
        warm.nodes, warm.bytes
    );
    let (min, median) = timings(|| {
        std::hint::black_box(encoder.matcher.boundaries(&full));
    });
    println!("boundaries(full) alone: min {min:?} median {median:?}");
    let mut extra = 0;
    let (min, median) = timings(|| {
        extra += 1;
        let text = format!("{}<s>user extra {extra}</s>\n<s>assistant\n", full);
        encoder.encode(tokenizer.as_ref(), 0, &text, None).unwrap();
    });
    println!("growth by one turn: min {min:?} median {median:?}");

    let l0 = Arc::new(
        FrontendCache::new(TokenizerCacheConfig {
            max_entries: 1024,
            max_bytes: 1 << 30,
            max_entry_bytes: 1 << 26,
        })
        .unwrap(),
    );
    let both = PromptEncoder::new("fixture", tokenizer.clone(), Some(l0))
        .with_l1_cache(Some(encoder.clone()));
    let run = |text: &str| {
        both.encode_detailed(Prompt::Text(text.to_string()), false, None)
            .unwrap()
    };
    run(&full);
    let (min, median) = timings(|| {
        run(&full);
    });
    println!("L0+L1, L0 hit: min {min:?} median {median:?}");
    let mut extra = 0;
    let (min, median) = timings(|| {
        extra += 1;
        run(&format!("{full}<s>user l0 miss {extra}</s>\n"));
    });
    println!("L0+L1, L0 miss then L1 hit: min {min:?} median {median:?}");

    for threads in [1, 8, 16] {
        let start = std::time::Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let (encoder, tokenizer, full) = (&encoder, tokenizer.clone(), &full);
                scope.spawn(move || {
                    for _ in 0..20 {
                        encoder.encode(tokenizer.as_ref(), 0, full, None).unwrap();
                    }
                });
            }
        });
        let per = start.elapsed() / (threads * 20) as u32;
        println!("{threads} threads full hit: {per:?} wall per request");
    }

    // Each thread grows its own conversation in its own namespace, so the
    // only thing they share is the tree and its lock.
    for (models, shared) in [(1, true), (8, true), (16, true), (16, false)] {
        let (shared_encoder, _) = super::tests::encoder(config);
        let own: Vec<_> = (0..models)
            .map(|_| super::tests::encoder(config).0)
            .collect();
        let per_request: std::time::Duration = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..models)
                .map(|t| {
                    let encoder = if shared { &shared_encoder } else { &own[t] };
                    let tokenizer = tokenizer.clone();
                    scope.spawn(move || {
                        let (mut chat, mut busy) = (String::new(), std::time::Duration::ZERO);
                        for i in 0..300 {
                            chat.push_str(&long_chat_turn(t * 1000 + i));
                            let text = format!("{chat}<s>assistant\n");
                            let start = std::time::Instant::now();
                            encoder
                                .encode(tokenizer.as_ref(), t as u64, &text, None)
                                .unwrap();
                            busy += start.elapsed();
                        }
                        busy / 300
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });
        let tree = if shared { "one tree" } else { "a tree each" };
        println!(
            "{models} models growing in parallel, {tree}: {:?} per request",
            per_request / models as u32
        );
    }

    for max_nodes in [2048, 65_536] {
        let (encoder, cache) = super::tests::encoder(L1CacheConfig {
            max_nodes,
            ..config
        });
        let (mut samples, mut alone) = (Vec::new(), Vec::new());
        for i in 0..max_nodes * 3 {
            let text = format!("<s>sys {}</s><s>user {i}</s><s>a", i % 3);
            let start = std::time::Instant::now();
            encoder.encode(tokenizer.as_ref(), 0, &text, None).unwrap();
            let elapsed = start.elapsed();
            if cache.stats().evictions > 0 {
                samples.push(elapsed);
                let start = std::time::Instant::now();
                std::hint::black_box(tokenizer.encode(&text, false).unwrap());
                alone.push(start.elapsed());
            }
        }
        println!(
            "max_nodes {max_nodes} encode once evicting: {}; tokenizer alone: {}",
            percentiles(samples),
            percentiles(alone)
        );
    }
}
