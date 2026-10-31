//! Versioned, bounded PD route/keep-drop estimates. No capacity authority.
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PdPath {
    ColdRemote,
    WarmRemote,
    LocalAppend,
    StoreRemote,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PdCostCell {
    pub path: PdPath,
    pub context_min: usize,
    pub context_max: usize,
    pub miss_min: usize,
    pub miss_max: usize,
    pub running_min: usize,
    pub running_max: usize,
    pub ttft_seconds: f64,
    pub tpot_p95_seconds: f64,
    pub uncertainty_seconds: f64,
    #[serde(default)]
    pub tpot_margin_seconds_per_token: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PdCalibration {
    pub revision: String,
    pub profile: String,
    pub metric: String,
    pub cells: Vec<PdCostCell>,
    pub local_tpot_slo_seconds: f64,
    pub min_route_gain_seconds: f64,
    pub max_prediction_error_seconds: f64,
    /// Independently measured re-admission delay after dropping the logical seat.
    #[serde(default)]
    pub drop_re_admission_seconds: f64,
}

impl PdCalibration {
    pub fn validate(&self) -> Result<(), String> {
        if self.revision.is_empty()
            || self.revision.len() > 128
            || self.profile.is_empty()
            || self.profile.len() > 128
            || self.metric != "ttft_seconds"
            || self.cells.is_empty()
            || self.cells.len() > 4096
        {
            return Err(
                "PD calibration needs bounded revision/profile, ttft_seconds and 1..4096 cells"
                    .into(),
            );
        }
        if ![
            self.local_tpot_slo_seconds,
            self.min_route_gain_seconds,
            self.max_prediction_error_seconds,
            self.drop_re_admission_seconds,
        ]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0)
            || self.local_tpot_slo_seconds <= 0.0
            || self.max_prediction_error_seconds <= 0.0
        {
            return Err(
                "PD calibration thresholds must be finite with positive SLO/error bound".into(),
            );
        }
        for (i, c) in self.cells.iter().enumerate() {
            if c.context_min > c.context_max
                || c.miss_min > c.miss_max
                || c.running_min > c.running_max
                || c.miss_max > c.context_max
                || ![
                    c.ttft_seconds,
                    c.tpot_p95_seconds,
                    c.uncertainty_seconds,
                    c.tpot_margin_seconds_per_token,
                ]
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
                || c.ttft_seconds <= 0.0
                || c.tpot_p95_seconds <= 0.0
            {
                return Err("Invalid PD calibration cell".into());
            }
            if self.cells[..i].iter().any(|p| {
                p.path == c.path
                    && p.context_min <= c.context_max
                    && c.context_min <= p.context_max
                    && p.miss_min <= c.miss_max
                    && c.miss_min <= p.miss_max
                    && p.running_min <= c.running_max
                    && c.running_min <= p.running_max
            }) {
                return Err("Overlapping PD calibration cells are ambiguous".into());
            }
        }
        Ok(())
    }
    pub fn estimate(
        &self,
        path: PdPath,
        context: usize,
        miss: usize,
        running: usize,
        correction: f64,
    ) -> Option<f64> {
        let c = self.cells.iter().find(|c| {
            c.path == path
                && (c.context_min..=c.context_max).contains(&context)
                && (c.miss_min..=c.miss_max).contains(&miss)
                && (c.running_min..=c.running_max).contains(&running)
        })?;
        if !correction.is_finite() || correction.abs() > self.max_prediction_error_seconds {
            return None;
        }
        if path == PdPath::LocalAppend
            && c.tpot_p95_seconds + c.tpot_margin_seconds_per_token > self.local_tpot_slo_seconds
        {
            return None;
        }
        let cost = c.ttft_seconds + c.uncertainty_seconds + correction.max(0.0);
        (cost.is_finite() && cost > 0.0).then_some(cost)
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct PdTtlValue {
    pub seconds: f64,
    pub saving_seconds: f64,
    pub utility_seconds: f64,
}

/// RFC §10 log-normal search with a verified source deadline and same-metric costs.
#[allow(clippy::too_many_arguments)]
pub fn solve_pd_ttl(
    intervals: &[f64],
    keep: f64,
    drop: f64,
    gamma: f64,
    max: f64,
    budget: f64,
    min_samples: usize,
    min_gain: f64,
) -> PdTtlValue {
    let zero = PdTtlValue::default();
    if ![keep, drop, gamma, max, budget, min_gain]
        .iter()
        .all(|v| v.is_finite())
        || keep <= 0.0
        || drop <= keep
        || gamma <= 0.0
        || max <= 0.0
        || budget <= 0.0
        || min_gain < 0.0
    {
        return zero;
    }
    let samples = intervals
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect::<Vec<_>>();
    if samples.len() < min_samples.max(2) {
        return zero;
    }
    let mu = samples.iter().map(|v| v.ln()).sum::<f64>() / samples.len() as f64;
    let sigma = (samples.iter().map(|v| (v.ln() - mu).powi(2)).sum::<f64>() / samples.len() as f64)
        .sqrt()
        .max(1e-6);
    let mean = (mu + sigma * sigma / 2.0).exp();
    if !mean.is_finite() {
        return zero;
    }
    let delta = drop - keep;
    let upper = max.min(delta / gamma).min(budget);
    let lower = 0.05_f64.min(upper);
    if upper <= 0.0 {
        return zero;
    }
    let mut best = PdTtlValue {
        saving_seconds: delta,
        ..zero
    };
    for i in 0..32 {
        let d = (lower.ln() + (upper.ln() - lower.ln()) * i as f64 / 31.0).exp();
        let f = normal_cdf((d.ln() - mu) / sigma);
        let wait = mean * normal_cdf((d.ln() - mu - sigma * sigma) / sigma) + d * (1.0 - f);
        let utility = delta * f - gamma * wait;
        // Ascending grid and strict > give the shorter tie, and zero beats <=0.
        if utility.is_finite() && utility > min_gain && utility > best.utility_seconds {
            best = PdTtlValue {
                seconds: d,
                saving_seconds: delta,
                utility_seconds: utility,
            };
        }
    }
    best
}
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let density = (-x * x / 2.0).exp() / std::f64::consts::TAU.sqrt();
    let tail = density
        * t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if x >= 0.0 {
        1.0 - tail
    } else {
        tail
    }
}
pub fn bounded_pd_duration(seconds: f64) -> Duration {
    if seconds.is_finite() && seconds > 0.0 && seconds <= 86400.0 {
        Duration::from_secs_f64(seconds)
    } else {
        Duration::ZERO
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::config::types::PdProgramSchedulingConfig;
    #[test]
    fn ttl_unknown_invalid_and_cold_window_are_zero() {
        let s = vec![1.0; 32];
        for (keep, drop, budget) in [
            (1.0, 1.0, 10.0),
            (f64::NAN, 5.0, 10.0),
            (1.0, f64::INFINITY, 10.0),
            (1.0, 5.0, 0.0),
        ] {
            assert_eq!(
                solve_pd_ttl(&s, keep, drop, 1.0, 10.0, budget, 16, 0.001).seconds,
                0.0
            );
        }
        assert_eq!(
            solve_pd_ttl(&[1.0], 1.0, 5.0, 1.0, 10.0, 10.0, 16, 0.0).seconds,
            0.0
        );
    }
    #[test]
    fn ttl_respects_value_lease_and_positive_utility() {
        let s = vec![0.1, 0.2, 0.15, 0.1];
        let t = solve_pd_ttl(&s, 0.1, 2.1, 1.0, 10.0, 0.4, 4, 0.001);
        assert!(t.seconds > 0.0 && t.seconds <= 0.4 && t.utility_seconds > 0.0);
    }
    #[test]
    fn ttl_overflow_safe() {
        assert_eq!(
            solve_pd_ttl(&[1e-300, 1e300], 0.1, 2.0, 1.0, 10.0, 10.0, 2, 0.0).seconds,
            0.0
        );
    }
    #[test]
    fn uncalibrated_local_and_unknown_protocol_rejected() {
        let c = PdProgramSchedulingConfig {
            local_append: true,
            ..Default::default()
        };
        assert!(c.validate().is_err());
        let c = PdProgramSchedulingConfig {
            protocol_revision: "kv_both".into(),
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }
    #[test]
    fn warm_remote_bounds_reject_unbounded_or_mismatched_values() {
        let c = PdProgramSchedulingConfig {
            warm_min_p_kv_usage: 1.5,
            ..Default::default()
        };
        assert!(c.validate().is_err());
        let c = PdProgramSchedulingConfig {
            max_candidate_tokens: 1024,
            ..Default::default()
        };
        assert!(c.validate().is_err());
        let c = PdProgramSchedulingConfig {
            capability_cache_seconds: 601.0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
        let c = PdProgramSchedulingConfig {
            max_response_bytes: 1024,
            ..Default::default()
        };
        assert!(c.validate().is_err());
        let c = PdProgramSchedulingConfig {
            max_candidate_tokens: 131072,
            capability_cache_seconds: 0.0,
            warm_min_p_kv_usage: 1.0,
            max_response_bytes: 16_777_216,
            ..Default::default()
        };
        assert!(c.validate().is_ok());
    }
    #[test]
    fn online_ttl_without_calibration_is_bounded_and_valid() {
        let c = PdProgramSchedulingConfig {
            automatic_ttl: true,
            ..Default::default()
        };
        assert!(c.validate().is_err());
        let c = PdProgramSchedulingConfig {
            automatic_ttl: true,
            ttl_online: true,
            ..Default::default()
        };
        assert!(c.validate().is_ok());
        // LocalAppend still needs a versioned calibration even with online TTL.
        let c = PdProgramSchedulingConfig {
            local_append: true,
            automatic_ttl: true,
            ttl_online: true,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }
}
