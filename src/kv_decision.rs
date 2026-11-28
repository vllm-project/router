//! Bounded observations of an actual KV-aware routing decision.
//!
//! Credited work is a routing estimate, never engine-confirmed cache reuse.

use serde::{Deserialize, Serialize};
use tracing::{field::Empty, info_span, Span};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvDecisionPolicy {
    KvAware,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvDecisionStage {
    Regular,
    Prefill,
    Decode,
}

impl KvDecisionStage {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Regular => "regular",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

/// A closed vocabulary: caller-supplied strings never become attributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvDecisionOutcome {
    PrefixCost,
    StickyHit,
    NoReusablePrefix,
    MissingTokens,
    CacheSalt,
    UnverifiedIdentity,
    UnverifiedModel,
    RegistrationChurn,
    UnknownPrefillBacklog,
    UnverifiedEffectiveTokens,
    UnsupportedCacheRead,
    UnsupportedCacheIdentity,
    SkipPrefixCache,
    SkipExternalKv,
    UnsupportedPromptScoring,
    PreparationUnavailable,
    ContextTokensMarginal,
    ContextTokens,
    OccupancyUnknownTokens,
    LoadFallback,
    PoolEstimateUnavailable,
    InvalidPrefillCosts,
    PlacementUnavailable,
    PlacementPinFailed,
}

impl KvDecisionOutcome {
    pub(crate) fn from_reason(reason: &str) -> Self {
        match reason {
            "prefix_cost" => Self::PrefixCost,
            "sticky_hit" => Self::StickyHit,
            "missing_tokens" | "empty_input" => Self::MissingTokens,
            "cache_salt" => Self::CacheSalt,
            "unverified_identity" | "unverified_placement" => Self::UnverifiedIdentity,
            "unverified_model" => Self::UnverifiedModel,
            "registration_churn" => Self::RegistrationChurn,
            "unknown_prefill_backlog" => Self::UnknownPrefillBacklog,
            "unverified_effective_tokens" => Self::UnverifiedEffectiveTokens,
            "unsupported_cache_read" => Self::UnsupportedCacheRead,
            "unsupported_cache_identity" => Self::UnsupportedCacheIdentity,
            "skip_prefix_cache" => Self::SkipPrefixCache,
            "skip_external_kv" => Self::SkipExternalKv,
            "unsupported_prompt_scoring" => Self::UnsupportedPromptScoring,
            "context_tokens_marginal" => Self::ContextTokensMarginal,
            "context_tokens" => Self::ContextTokens,
            "occupancy_unknown_tokens" => Self::OccupancyUnknownTokens,
            "load_fallback" => Self::LoadFallback,
            "pool_estimate_unavailable" => Self::PoolEstimateUnavailable,
            "invalid_prefill_costs" => Self::InvalidPrefillCosts,
            "placement_unavailable" => Self::PlacementUnavailable,
            "placement_pin_failed" => Self::PlacementPinFailed,
            _ => Self::PreparationUnavailable,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::PrefixCost => "prefix_cost",
            Self::StickyHit => "sticky_hit",
            Self::NoReusablePrefix => "no_reusable_prefix",
            Self::MissingTokens => "missing_tokens",
            Self::CacheSalt => "cache_salt",
            Self::UnverifiedIdentity => "unverified_identity",
            Self::UnverifiedModel => "unverified_model",
            Self::RegistrationChurn => "registration_churn",
            Self::UnknownPrefillBacklog => "unknown_prefill_backlog",
            Self::UnverifiedEffectiveTokens => "unverified_effective_tokens",
            Self::UnsupportedCacheRead => "unsupported_cache_read",
            Self::UnsupportedCacheIdentity => "unsupported_cache_identity",
            Self::SkipPrefixCache => "skip_prefix_cache",
            Self::SkipExternalKv => "skip_external_kv",
            Self::UnsupportedPromptScoring => "unsupported_prompt_scoring",
            Self::PreparationUnavailable => "preparation_unavailable",
            Self::ContextTokensMarginal => "context_tokens_marginal",
            Self::ContextTokens => "context_tokens",
            Self::OccupancyUnknownTokens => "occupancy_unknown_tokens",
            Self::LoadFallback => "load_fallback",
            Self::PoolEstimateUnavailable => "pool_estimate_unavailable",
            Self::InvalidPrefillCosts => "invalid_prefill_costs",
            Self::PlacementUnavailable => "placement_unavailable",
            Self::PlacementPinFailed => "placement_pin_failed",
        }
    }
}

/// No source recovery state is inferred from a hit or a verified cache identity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvDecisionSourceState {
    #[default]
    Unavailable,
}

/// The application field contract shared by traces and future log consumers.
/// `None` means unavailable or inapplicable, and is distinct from measured zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvDecisionReceipt {
    pub policy: KvDecisionPolicy,
    pub stage: KvDecisionStage,
    pub outcome: KvDecisionOutcome,
    pub credited_tokens: Option<u64>,
    pub new_prefill_tokens: Option<u64>,
    pub limiting_group: Option<u32>,
    pub target_index: usize,
    pub source_state: KvDecisionSourceState,
    pub overhead_micros: u64,
}

impl KvDecisionReceipt {
    pub(crate) fn new(
        stage: KvDecisionStage,
        mut outcome: KvDecisionOutcome,
        target_index: usize,
        credited_tokens: Option<u64>,
        new_prefill_tokens: Option<u64>,
        started: std::time::Instant,
    ) -> Self {
        if credited_tokens == Some(0)
            && matches!(
                outcome,
                KvDecisionOutcome::PrefixCost | KvDecisionOutcome::StickyHit
            )
        {
            outcome = KvDecisionOutcome::NoReusablePrefix;
        }
        Self {
            policy: KvDecisionPolicy::KvAware,
            stage,
            outcome,
            credited_tokens,
            new_prefill_tokens,
            limiting_group: None,
            target_index,
            source_state: KvDecisionSourceState::Unavailable,
            overhead_micros: u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        }
    }
}

/// Request-owned tracing state, separate from the serializable receipt and all
/// load reservations. Clones only keep observation spans alive.
#[derive(Clone, Debug)]
pub struct KvDecisionTrace {
    pub receipt: KvDecisionReceipt,
    span: Span,
}

impl KvDecisionTrace {
    /// Observe an explicitly supplied decision, creating its span only when
    /// OTel is enabled. This does not select a worker, query placement, or
    /// reserve load. The caller owns the decision facts; this factory does not
    /// verify their provenance or replace unavailable values with defaults.
    pub fn new(receipt: KvDecisionReceipt) -> Self {
        let span = if crate::otel_trace::is_otel_enabled() {
            let span = info_span!(
                target: "otel_trace",
                "kv_route",
                policy = "kv_aware",
                stage = receipt.stage.as_str(),
                outcome = receipt.outcome.as_str(),
                credited_tokens = Empty,
                new_prefill_tokens = Empty,
                limiting_group = Empty,
                target_index = Empty,
                source_state = "unavailable",
                overhead_micros = Empty,
            );
            record_optional(&span, "credited_tokens", receipt.credited_tokens);
            record_optional(&span, "new_prefill_tokens", receipt.new_prefill_tokens);
            record_optional(
                &span,
                "limiting_group",
                receipt.limiting_group.map(u64::from),
            );
            record_unsigned(&span, "target_index", receipt.target_index);
            record_unsigned(&span, "overhead_micros", receipt.overhead_micros);
            span
        } else {
            Span::none()
        };
        Self { receipt, span }
    }

    pub(crate) fn dispatch_span(&self, stage: KvDecisionStage) -> Span {
        if self.span.is_none() || !crate::otel_trace::is_otel_enabled() {
            return Span::none();
        }
        match stage {
            KvDecisionStage::Regular | KvDecisionStage::Prefill => {
                info_span!(target: "otel_trace", parent: &self.span, "prefill_dispatch")
            }
            KvDecisionStage::Decode => {
                info_span!(target: "otel_trace", parent: &self.span, "decode_dispatch")
            }
        }
    }
}

fn record_optional(span: &Span, field: &'static str, value: Option<u64>) {
    match value {
        Some(value) => record_unsigned(span, field, value),
        None => {
            span.record(field, "unavailable");
        }
    }
}

fn record_unsigned<T>(span: &Span, field: &'static str, value: T)
where
    T: Copy + TryInto<i64> + std::fmt::Display,
{
    // A filtered span can retain metadata, so is_none() is not sufficient.
    if span.is_disabled() {
        return;
    }
    match value.try_into() {
        Ok(value) => {
            span.record(field, value);
        }
        Err(_) => {
            // OTel integers are signed. Keep the full unsigned value instead
            // of truncating it or confusing a large known value with unknown.
            span.record(field, value.to_string().as_str());
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn receipt_pins_fields_and_distinguishes_unavailable_from_zero() {
        let receipt = KvDecisionReceipt::new(
            KvDecisionStage::Regular,
            KvDecisionOutcome::MissingTokens,
            2,
            None,
            None,
            std::time::Instant::now(),
        );
        let json = serde_json::to_value(receipt).unwrap();
        let keys: Vec<_> = json
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            [
                "policy",
                "stage",
                "outcome",
                "credited_tokens",
                "new_prefill_tokens",
                "limiting_group",
                "target_index",
                "source_state",
                "overhead_micros"
            ]
        );
        assert!(json["credited_tokens"].is_null());
        let zero = KvDecisionReceipt::new(
            KvDecisionStage::Prefill,
            KvDecisionOutcome::PrefixCost,
            0,
            Some(0),
            Some(8),
            std::time::Instant::now(),
        );
        assert_eq!(zero.outcome, KvDecisionOutcome::NoReusablePrefix);
        assert_eq!(serde_json::to_value(zero).unwrap()["credited_tokens"], 0);
        let mut expanded = json;
        expanded["prompt"] = "forbidden".into();
        assert!(serde_json::from_value::<KvDecisionReceipt>(expanded).is_err());
    }

    #[test]
    fn arbitrary_fallback_text_is_bounded() {
        assert_eq!(
            KvDecisionOutcome::from_reason("http://private-worker/prompt"),
            KvDecisionOutcome::PreparationUnavailable
        );
    }
}
