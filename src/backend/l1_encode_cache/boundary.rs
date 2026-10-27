use std::collections::HashSet;
use std::path::Path;
use std::sync::LazyLock;

use aho_corasick::{AhoCorasick, Input, MatchKind};
use anyhow::{bail, Context, Result};
use regex::Regex;
use serde_json::Value;
use vllm_text::backend::hf::{ResolvedModelFiles, TokenizerSource};
use vllm_tokenizer::Tokenizer;

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
    pub(super) automaton: AhoCorasick,
    pub(super) patterns: Vec<String>,
    /// Per pattern: whether a cut follows it.
    pub(super) cut: Vec<bool>,
    pub(super) tokens: Vec<String>,
    /// Per pattern: the token ID of a boundary, once resolved.
    pub(super) cut_ids: Vec<Option<u32>>,
    /// Bitset of every resolved boundary token ID.
    pub(super) cut_id_bits: Vec<u64>,
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

    pub(super) fn build(mut patterns: Vec<String>, mut tokens: Vec<String>) -> Result<Self> {
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
            cut_ids: Vec::new(),
            cut_id_bits: Vec::new(),
        })
    }

    pub(crate) fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// Record each boundary's token ID so cut offsets can be read from one
    /// encode (see [`Self::token_offsets`]).
    pub(crate) fn with_token_ids(mut self, tokenizer: &dyn Tokenizer) -> Self {
        self.cut_ids = self
            .patterns
            .iter()
            .zip(&self.cut)
            .map(|(pattern, &cut)| cut.then(|| tokenizer.token_to_id(pattern)).flatten())
            .collect();
        let mut bits = Vec::new();
        for id in self.cut_ids.iter().flatten().map(|&id| id as usize) {
            if bits.len() <= id / 64 {
                bits.resize(id / 64 + 1, 0u64);
            }
            bits[id / 64] |= 1 << (id % 64);
        }
        self.cut_id_bits = bits;
        self
    }

    /// Cut positions in `text`, which must itself start at a cut position.
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    pub(crate) fn boundaries(&self, text: &str) -> Vec<usize> {
        self.cuts(text, 0).into_iter().map(|(end, _)| end).collect()
    }

    /// Cuts in `text[from..]` as (end, pattern). `from` must be a cut
    /// position: the tokenizer's leftmost-longest split restarts there, so
    /// the matches after it do not depend on the text before it.
    pub(super) fn cuts(&self, text: &str, from: usize) -> Vec<(usize, usize)> {
        self.automaton
            .find_iter(Input::new(text).range(from..))
            .filter(|m| self.cut[m.pattern().as_usize()])
            .map(|m| (m.end(), m.pattern().as_usize()))
            .collect()
    }

    /// The first cut after `from`, which must be a cut position.
    pub(super) fn first_cut(&self, text: &str, from: usize) -> Option<usize> {
        self.automaton
            .find_iter(Input::new(text).range(from..))
            .find(|m| self.cut[m.pattern().as_usize()])
            .map(|m| m.end())
    }

    /// Token offsets right after each of `cuts` in `ids`, the encoding of
    /// text that starts at a cut. The tokenizer splits boundary tokens out
    /// before its model runs, so each cut ends with that token's ID. `None`
    /// if the IDs do not line up one to one with `cuts`, e.g. when the model
    /// can also produce a boundary ID from plain text.
    pub(super) fn token_offsets(&self, cuts: &[(usize, usize)], ids: &[u32]) -> Option<Vec<usize>> {
        if self.cut_id_bits.is_empty() {
            return None;
        }
        let mut offsets = Vec::with_capacity(cuts.len());
        for (i, &id) in ids.iter().enumerate() {
            let id_bit = self.cut_id_bits.get(id as usize / 64).copied().unwrap_or(0);
            if id_bit >> (id % 64) & 1 == 0 {
                continue;
            }
            let &(_, pattern) = cuts.get(offsets.len())?;
            if self.cut_ids[pattern] != Some(id) {
                return None;
            }
            offsets.push(i + 1);
        }
        (offsets.len() == cuts.len()).then_some(offsets)
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
        Ok(Self::build(self.patterns, kept)?.with_token_ids(tokenizer))
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
pub(super) fn ends_with_word(text: &str) -> bool {
    static ENDS_WITH_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w$").unwrap());
    ENDS_WITH_WORD.is_match(text)
}

/// Tokens that occur inside another added token, ending before its end.
pub(super) fn covered_tokens<'a>(contents: &[&'a str]) -> HashSet<&'a str> {
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
pub(super) fn overlaps(a: &str, b: &str) -> bool {
    a == b || reaches_into(a, b) || reaches_into(b, a)
}

/// Whether a match of `x` can start before an occurrence of `t` and reach
/// into it, or start at the same byte and be longer.
pub(super) fn reaches_into(x: &str, t: &str) -> bool {
    if x.len() > t.len() && x.starts_with(t) {
        return true;
    }
    char_offsets(x).any(|k| {
        let suffix = &x[k..];
        t.starts_with(suffix) || suffix.starts_with(t)
    })
}

pub(super) fn char_offsets(s: &str) -> impl Iterator<Item = usize> + '_ {
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

pub(super) fn is_identity_normalizer(normalizer: &Value) -> bool {
    match normalizer["type"].as_str() {
        None => normalizer.is_null(),
        Some("Sequence") => normalizer["normalizers"]
            .as_array()
            .is_some_and(|steps| steps.iter().all(is_identity_normalizer)),
        _ => false,
    }
}

pub(super) fn check_tokenizer_json(json: &Value) -> Result<()> {
    if !super::super::tokenizer_determinism::deterministic_hf_model(&json["model"]) {
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

pub(super) fn uses_first_only_metaspace(pre_tokenizer: &Value) -> bool {
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
pub(super) fn added_tokens(value: &Value, skip: Option<&[u64]>) -> Vec<AddedTokenSpec> {
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
                // vllm-tokenizer parses decoder keys as `u32` and drops the
                // entry when that fails. A wider parse would admit a boundary
                // the tokenizer never installs.
                let Ok(id) = id.parse::<u32>() else {
                    return false;
                };
                !skip.unwrap_or_default().contains(&(u64::from(id)))
            })
            .filter_map(|(_, token)| parse(token))
            .collect(),
        _ => Vec::new(),
    }
}
