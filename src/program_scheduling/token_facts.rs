//! Source-aware input-token facts for Program scheduling.
//!
//! Program admission, context-growth accounting, and the Progress-TTL cost
//! model all consume one number: how many input tokens this request adds to a
//! Program's context. Historically that number came from a byte-count
//! estimator, which is only as good as the estimator's learned tokens-per-byte
//! coefficient for the endpoint.
//!
//! This module keeps *where the number came from* next to the number itself so
//! the scheduler can:
//!
//! - deliberately apply a count that was produced before dispatch
//!   (`ClientTokenIds` for token-ID prompts, `PreparedTokenIds` for Router
//!   request preparation) while labelling every decision with its source;
//! - record a shadow decision when an exact count is available but not trusted
//!   yet, without changing scheduling behavior;
//! - revoke one tokenizer fingerprint when Engine `prompt_info`/usage disagrees
//!   with the prepared count, and fall back to estimation for that fingerprint.
//!
//! "Exact" here means only that the token IDs were known and forwarded before
//! dispatch. Future output length, next tool-return time, physical KV
//! occupancy, and future KV hits stay unknown, and a Router-side encoding-cache
//! hit is never evidence of an Engine KV hit.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

/// Immutable evidence for how one input-token count was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenFactSource {
    /// Byte-count estimator with usage feedback (legacy path).
    Estimated,
    /// Token IDs supplied by the client and forwarded to the Engine verbatim.
    ClientTokenIds,
    /// Token IDs produced by Router request preparation (render + encode).
    PreparedTokenIds,
    /// Engine-reported prompt usage observed at settlement.
    EngineUsage,
}

impl TokenFactSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Estimated => "estimated",
            Self::ClientTokenIds => "client_token_ids",
            Self::PreparedTokenIds => "prepared_token_ids",
            Self::EngineUsage => "engine_usage",
        }
    }

    /// Whether the IDs for this count were known and dispatched before the
    /// Engine tokenized anything.
    pub fn is_prepared(self) -> bool {
        matches!(self, Self::ClientTokenIds | Self::PreparedTokenIds)
    }
}

impl fmt::Display for TokenFactSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Stable identity of the token-equivalence context a prepared count belongs to.
///
/// The fingerprint deliberately covers everything that can change how a count
/// relates to the Engine's own accounting: model pool, request model, Router
/// frontend asset overrides, chat-template rendering options, cache salt, and
/// the encode backend. A mismatch between two fingerprints is never compared as
/// a "close enough" count; each fingerprint is verified or revoked on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct TokenizerFingerprint([u8; 16]);

impl TokenizerFingerprint {
    /// Hash a set of identity parts into a bounded, log-safe fingerprint.
    ///
    /// Parts are joined with a separator that cannot appear in the inputs we
    /// pass (they are model names, JSON snippets, and enum names), so
    /// `["a", "bc"]` and `["ab", "c"]` never collide by concatenation.
    pub fn from_parts(parts: &[&str]) -> Self {
        let mut hasher = Sha256::new();
        for part in parts {
            hasher.update(part.as_bytes());
            hasher.update([0x1f]);
        }
        let digest = hasher.finalize();
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&digest[..16]);
        Self(bytes)
    }

    pub fn as_hex(&self) -> String {
        let mut out = String::with_capacity(32);
        for byte in self.0 {
            use fmt::Write;
            let _ = write!(out, "{byte:02x}");
        }
        out
    }
}

impl fmt::Display for TokenizerFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_hex())
    }
}

/// One candidate input-token count together with its provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenFacts {
    tokens: usize,
    source: TokenFactSource,
    fingerprint: Option<TokenizerFingerprint>,
}

impl TokenFacts {
    pub fn estimated(tokens: usize) -> Self {
        Self {
            tokens,
            source: TokenFactSource::Estimated,
            fingerprint: None,
        }
    }

    pub fn client_token_ids(tokens: usize, fingerprint: TokenizerFingerprint) -> Self {
        Self {
            tokens,
            source: TokenFactSource::ClientTokenIds,
            fingerprint: Some(fingerprint),
        }
    }

    pub fn prepared_token_ids(tokens: usize, fingerprint: TokenizerFingerprint) -> Self {
        Self {
            tokens,
            source: TokenFactSource::PreparedTokenIds,
            fingerprint: Some(fingerprint),
        }
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn source(&self) -> TokenFactSource {
        self.source
    }

    pub fn fingerprint(&self) -> Option<TokenizerFingerprint> {
        self.fingerprint
    }

    /// Whether this count was produced before dispatch and is eligible for
    /// `exact_when_verified` once its fingerprint is accepted.
    pub fn is_prepared(&self) -> bool {
        self.source.is_prepared()
    }
}

/// The counts available for one request at arrival.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrivalTokenFacts {
    estimated_tokens: usize,
    prepared: Option<TokenFacts>,
}

impl ArrivalTokenFacts {
    pub fn new(estimated_tokens: usize, prepared: Option<TokenFacts>) -> Self {
        Self {
            estimated_tokens: estimated_tokens.max(1),
            prepared: prepared.filter(|facts| facts.tokens() > 0),
        }
    }

    pub fn estimated_only(estimated_tokens: usize) -> Self {
        Self::new(estimated_tokens, None)
    }

    /// Legacy byte-estimated count, always available.
    pub fn estimated_tokens(&self) -> usize {
        self.estimated_tokens
    }

    /// Exact pre-dispatch count when one exists.
    pub fn prepared(&self) -> Option<TokenFacts> {
        self.prepared
    }
}

/// Why an available prepared count was not applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenFactFallback {
    /// The model pool is not in the exact-token allowlist.
    PoolNotAllowed,
    /// The fingerprint previously disagreed with Engine usage and was revoked.
    FingerprintRevoked,
    /// No pre-dispatch count was available for this request shape.
    NoPreparedFact,
    /// Router-side text preparation failed, so the estimate was used.
    PreparationFailed,
}

impl TokenFactFallback {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PoolNotAllowed => "pool_not_allowed",
            Self::FingerprintRevoked => "fingerprint_revoked",
            Self::NoPreparedFact => "no_prepared_fact",
            Self::PreparationFailed => "preparation_failed",
        }
    }
}

impl fmt::Display for TokenFactFallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Token-count source selection mode.
///
/// Default is `Estimated`, which preserves pre-RFC behavior exactly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenCountMode {
    /// Byte estimator only; no shadow accounting.
    #[default]
    Estimated,
    /// Apply the estimator, but record what the prepared count would decide.
    Shadow,
    /// Apply the prepared count once its fingerprint is allowed and verified.
    ExactWhenVerified,
}

impl TokenCountMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Estimated => "estimated",
            Self::Shadow => "shadow",
            Self::ExactWhenVerified => "exact_when_verified",
        }
    }

    pub fn uses_prepared_facts(self) -> bool {
        !matches!(self, Self::Estimated)
    }
}

impl fmt::Display for TokenCountMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What the scheduler applied for one arrival, plus the shadow counterfactual.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextTokens {
    pub(crate) tokens: usize,
    pub(crate) source: TokenFactSource,
    pub(crate) fingerprint: Option<TokenizerFingerprint>,
    pub(crate) shadow_tokens: Option<usize>,
    pub(crate) fallback: Option<TokenFactFallback>,
}

impl ContextTokens {
    pub fn estimated(tokens: usize) -> Self {
        Self {
            tokens: tokens.max(1),
            source: TokenFactSource::Estimated,
            fingerprint: None,
            shadow_tokens: None,
            fallback: None,
        }
    }

    pub fn from_prepared(facts: TokenFacts) -> Self {
        Self {
            tokens: facts.tokens().max(1),
            source: facts.source(),
            fingerprint: facts.fingerprint(),
            shadow_tokens: None,
            fallback: None,
        }
    }

    pub fn with_shadow(mut self, shadow_tokens: Option<usize>) -> Self {
        self.shadow_tokens = shadow_tokens.filter(|tokens| *tokens > 0);
        self
    }

    pub fn with_fallback(mut self, fallback: TokenFactFallback) -> Self {
        self.fallback = Some(fallback);
        self
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn source(&self) -> TokenFactSource {
        self.source
    }

    pub fn fingerprint(&self) -> Option<TokenizerFingerprint> {
        self.fingerprint
    }

    pub fn shadow_tokens(&self) -> Option<usize> {
        self.shadow_tokens
    }

    pub fn fallback(&self) -> Option<TokenFactFallback> {
        self.fallback
    }

    /// Keep the larger demand while preserving the strongest provenance.
    ///
    /// A Program's ledger entry is the largest unfinished context demand across
    /// its retained and in-flight attempts, so the merge keeps the larger
    /// count. Ties prefer a prepared count, and the shadow counterfactual also
    /// keeps its larger value so a shadow decision can never shrink because a
    /// later estimated request arrived.
    pub fn max_with(self, other: Self) -> Self {
        let other_wins = other.tokens > self.tokens
            || (other.tokens == self.tokens
                && other.source.is_prepared()
                && !self.source.is_prepared());
        let (tokens, source, fingerprint, fallback) = if other_wins {
            (
                other.tokens,
                other.source,
                other.fingerprint,
                other.fallback,
            )
        } else {
            (self.tokens, self.source, self.fingerprint, self.fallback)
        };
        Self {
            tokens,
            source,
            fingerprint,
            shadow_tokens: match (self.shadow_tokens, other.shadow_tokens) {
                (Some(left), Some(right)) => Some(left.max(right)),
                (Some(value), None) | (None, Some(value)) => Some(value),
                (None, None) => None,
            },
            fallback,
        }
    }
}

/// Resolve the applied count for one arrival.
///
/// `allowed` is true when the model pool is allowlisted for exact counts;
/// `revoked` is true when the request's fingerprint already disagreed with
/// Engine usage. The returned `ContextTokens` is the only value the scheduler
/// may use for admission, growth accounting, and TTL cost.
pub fn resolve_context_tokens(
    mode: TokenCountMode,
    arrival: ArrivalTokenFacts,
    allowed: bool,
    revoked: bool,
) -> ContextTokens {
    let estimated = ContextTokens::estimated(arrival.estimated_tokens());
    let Some(prepared) = arrival.prepared() else {
        return match mode {
            TokenCountMode::ExactWhenVerified => {
                estimated.with_fallback(TokenFactFallback::NoPreparedFact)
            }
            _ => estimated,
        };
    };
    let usable = allowed && !revoked;
    match mode {
        TokenCountMode::Estimated => estimated,
        TokenCountMode::Shadow => {
            if usable {
                estimated.with_shadow(Some(prepared.tokens()))
            } else {
                estimated
            }
        }
        TokenCountMode::ExactWhenVerified => {
            if !allowed {
                return estimated.with_fallback(TokenFactFallback::PoolNotAllowed);
            }
            if revoked {
                return estimated.with_fallback(TokenFactFallback::FingerprintRevoked);
            }
            ContextTokens::from_prepared(prepared)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint() -> TokenizerFingerprint {
        TokenizerFingerprint::from_parts(&["model-a", "{}", "vllm-tokenizer"])
    }

    #[test]
    fn fingerprint_is_stable_and_separator_safe() {
        let first = TokenizerFingerprint::from_parts(&["a", "bc"]);
        let second = TokenizerFingerprint::from_parts(&["a", "bc"]);
        let concatenated = TokenizerFingerprint::from_parts(&["ab", "c"]);
        assert_eq!(first, second);
        assert_ne!(first, concatenated);
        assert_eq!(first.as_hex().len(), 32);
    }

    #[test]
    fn estimated_mode_never_applies_prepared_count() {
        let arrival = ArrivalTokenFacts::new(
            100,
            Some(TokenFacts::prepared_token_ids(98_304, fingerprint())),
        );
        let context = resolve_context_tokens(TokenCountMode::Estimated, arrival, true, false);
        assert_eq!(context.tokens(), 100);
        assert_eq!(context.source(), TokenFactSource::Estimated);
        assert_eq!(context.shadow_tokens(), None);
    }

    #[test]
    fn shadow_mode_records_counterfactual_without_changing_decision() {
        let arrival =
            ArrivalTokenFacts::new(4, Some(TokenFacts::client_token_ids(98_304, fingerprint())));
        let context = resolve_context_tokens(TokenCountMode::Shadow, arrival, true, false);
        assert_eq!(context.tokens(), 4);
        assert_eq!(context.shadow_tokens(), Some(98_304));
    }

    #[test]
    fn exact_mode_applies_prepared_count_when_allowed() {
        let arrival =
            ArrivalTokenFacts::new(4, Some(TokenFacts::client_token_ids(98_304, fingerprint())));
        let context =
            resolve_context_tokens(TokenCountMode::ExactWhenVerified, arrival, true, false);
        assert_eq!(context.tokens(), 98_304);
        assert_eq!(context.source(), TokenFactSource::ClientTokenIds);
        assert_eq!(context.fallback(), None);
    }

    #[test]
    fn exact_mode_falls_back_when_pool_not_allowed_or_revoked() {
        let arrival =
            ArrivalTokenFacts::new(4, Some(TokenFacts::client_token_ids(98_304, fingerprint())));
        let not_allowed =
            resolve_context_tokens(TokenCountMode::ExactWhenVerified, arrival, false, false);
        assert_eq!(not_allowed.tokens(), 4);
        assert_eq!(
            not_allowed.fallback(),
            Some(TokenFactFallback::PoolNotAllowed)
        );
        let revoked =
            resolve_context_tokens(TokenCountMode::ExactWhenVerified, arrival, true, true);
        assert_eq!(revoked.tokens(), 4);
        assert_eq!(
            revoked.fallback(),
            Some(TokenFactFallback::FingerprintRevoked)
        );
    }

    #[test]
    fn exact_mode_labels_missing_prepared_fact() {
        let context = resolve_context_tokens(
            TokenCountMode::ExactWhenVerified,
            ArrivalTokenFacts::estimated_only(512),
            true,
            false,
        );
        assert_eq!(context.tokens(), 512);
        assert_eq!(context.fallback(), Some(TokenFactFallback::NoPreparedFact));
    }

    #[test]
    fn max_with_keeps_larger_demand_and_prefers_prepared_on_ties() {
        let estimated = ContextTokens::estimated(100).with_shadow(Some(900));
        let prepared =
            ContextTokens::from_prepared(TokenFacts::client_token_ids(100, fingerprint()));
        let merged = estimated.max_with(prepared);
        assert_eq!(merged.tokens(), 100);
        assert_eq!(merged.source(), TokenFactSource::ClientTokenIds);
        assert_eq!(merged.shadow_tokens(), Some(900));

        let larger = ContextTokens::estimated(200);
        assert_eq!(merged.max_with(larger).tokens(), 200);
    }
}
