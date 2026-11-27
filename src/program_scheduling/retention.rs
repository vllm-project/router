//! Observational decomposition of the existing return-time TTL objective.
//!
//! These are model terms, not measured cache savings or counterfactual KV
//! benefits. In particular, logical retention does not pin physical KV.

use serde::Serialize;

/// Model terms for the predictor's suggested TTL, not necessarily the armed TTL.
///
/// The legacy objective prices one second of expected retention as one second
/// of cost. This snapshot adds no physical-residency or queue-benefit estimate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ReturnTimeUtilityDiagnostic {
    pub ttl_seconds: f64,
    pub cache_miss_impact_seconds: f64,
    /// Unconditional probability of returning within this TTL. Unlike the
    /// existing return probability, this does not use the diagnostic horizon.
    pub return_probability_within_ttl: f64,
    pub no_return_probability: f64,
    pub expected_retention_seconds: f64,
    pub modeled_cache_benefit_seconds: f64,
    pub modeled_utility_seconds: f64,
}

/// Evaluate precisely the histogram assumptions used by the legacy selector.
/// Unknown overflow returns pay retention cost but receive no finite-TTL hit
/// credit. Terminal mass pays the full TTL. Missing evidence is not zero value.
pub(crate) fn evaluate_legacy_ttl(
    counts: &[f64],
    no_return: f64,
    bounds: &[f64],
    cache_miss_cost: f64,
    ttl: f64,
    max_observed: f64,
) -> Option<ReturnTimeUtilityDiagnostic> {
    let total = counts.iter().sum::<f64>() + no_return;
    if counts.len() > bounds.len() + 1
        || !total.is_finite()
        || total <= 0.0
        || !no_return.is_finite()
        || no_return < 0.0
        || counts
            .iter()
            .any(|count| !count.is_finite() || *count < 0.0)
        || !cache_miss_cost.is_finite()
        || cache_miss_cost < 0.0
        || !ttl.is_finite()
        || ttl < 0.0
    {
        return None;
    }
    let last_bound = bounds.last().copied().unwrap_or(0.0);
    let overflow_upper = if max_observed > last_bound {
        max_observed
    } else {
        f64::INFINITY
    };
    let no_return_probability = no_return / total;
    let mut hit_probability = 0.0;
    let mut retained_cost = no_return_probability * ttl;
    for (index, count) in counts.iter().enumerate() {
        let upper = bounds.get(index).copied().unwrap_or(overflow_upper);
        let probability = count / total;
        if upper <= ttl {
            hit_probability += probability;
        }
        retained_cost += probability * upper.min(ttl);
    }
    let modeled_cache_benefit_seconds = cache_miss_cost * hit_probability;
    Some(ReturnTimeUtilityDiagnostic {
        ttl_seconds: ttl,
        cache_miss_impact_seconds: cache_miss_cost,
        return_probability_within_ttl: hit_probability,
        no_return_probability,
        expected_retention_seconds: retained_cost,
        modeled_cache_benefit_seconds,
        modeled_utility_seconds: modeled_cache_benefit_seconds - retained_cost,
    })
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn finite_returns_and_terminal_mass_reconstruct_utility() {
        let value = evaluate_legacy_ttl(&[3.0, 1.0, 0.0], 1.0, &[0.1, 0.5], 0.6, 0.1, 0.5).unwrap();
        assert!((value.return_probability_within_ttl - 0.6).abs() < 1e-12);
        assert!((value.no_return_probability - 0.2).abs() < 1e-12);
        assert!((value.expected_retention_seconds - 0.1).abs() < 1e-12);
        assert!((value.modeled_cache_benefit_seconds - 0.36).abs() < 1e-12);
        assert!((value.modeled_utility_seconds - 0.26).abs() < 1e-12);
        let json = serde_json::to_value(value).unwrap();
        assert_eq!(json["ttl_seconds"], 0.1);
        assert_eq!(json["cache_miss_impact_seconds"], 0.6);
    }

    #[test]
    fn unknown_overflow_has_no_hit_credit_but_pays_retention_cost() {
        let value = evaluate_legacy_ttl(&[0.0, 0.0, 2.0], 0.0, &[0.1, 0.5], 2.0, 1.0, 0.0).unwrap();
        assert_eq!(value.return_probability_within_ttl, 0.0);
        assert_eq!(value.expected_retention_seconds, 1.0);
        assert_eq!(value.modeled_utility_seconds, -1.0);
        let observed =
            evaluate_legacy_ttl(&[0.0, 0.0, 2.0], 0.0, &[0.1, 0.5], 2.0, 1.0, 0.75).unwrap();
        assert_eq!(observed.return_probability_within_ttl, 1.0);
        assert_eq!(observed.expected_retention_seconds, 0.75);
    }

    #[test]
    fn zero_ttl_is_observed_zero_utility_not_missing_evidence() {
        let value = evaluate_legacy_ttl(&[0.0, 0.0], 2.0, &[1.0], 0.0, 0.0, 0.0).unwrap();
        assert_eq!(value.no_return_probability, 1.0);
        assert_eq!(value.modeled_utility_seconds, 0.0);
        assert!(evaluate_legacy_ttl(&[0.0, 0.0], 0.0, &[1.0], 1.0, 0.0, 0.0).is_none());
    }

    #[test]
    fn invalid_evidence_is_unavailable_not_a_synthetic_score() {
        for counts in [&[-1.0, 2.0][..], &[f64::NAN, 1.0], &[1.0, 0.0, 0.0]] {
            assert!(evaluate_legacy_ttl(counts, 0.0, &[1.0], 1.0, 0.5, 0.0).is_none());
        }
        for cost in [-1.0, f64::INFINITY, f64::NAN] {
            assert!(evaluate_legacy_ttl(&[1.0, 0.0], 0.0, &[1.0], cost, 0.5, 0.0).is_none());
        }
    }
}
