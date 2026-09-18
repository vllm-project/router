//! Pure Progress-TTL calculations shared by Router lifecycle entry points.
//!
//! Inputs are immutable decision facts. The module never mutates Program
//! lifecycle state, queues, target placement, or backend observations.

use super::ProgressTtlFactors;
use std::time::Duration;

/// Offline-calibrated cold-prefill cost model.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PrefillCostModel {
    /// Fixed request cost in seconds.
    pub intercept_seconds: f64,
    /// Linear cost for each 1,000 uncached prompt tokens.
    pub linear_seconds_per_1k_tokens: f64,
    /// Quadratic cost for each squared 1,000-token unit.
    pub quadratic_seconds_per_1k_tokens_squared: f64,
    /// Decode throughput retained while prefill is mixed into the batch.
    pub decode_throughput_alpha: f64,
}

impl Default for PrefillCostModel {
    fn default() -> Self {
        Self {
            intercept_seconds: 0.06600061907132926,
            linear_seconds_per_1k_tokens: 0.05702024012166613,
            quadratic_seconds_per_1k_tokens_squared: 0.0044057347937978475,
            decode_throughput_alpha: 0.15,
        }
    }
}

/// Offline-calibrated decode throughput surface.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct DecodeThroughputModel {
    /// Fixed per-step latency component.
    pub fixed_step_seconds: f64,
    /// Additional per-step latency per batched request.
    pub batch_step_seconds_per_request: f64,
    /// Additional per-step latency per token in total batch context.
    pub context_step_seconds_per_token: f64,
}

impl Default for DecodeThroughputModel {
    fn default() -> Self {
        Self {
            fixed_step_seconds: 0.010635218423115973,
            batch_step_seconds_per_request: 0.0004192803698834784,
            context_step_seconds_per_token: 1.420458414996201e-7,
        }
    }
}

/// Progress-TTL parameters that affect rank-local scheduling decisions.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgressTtlConfig {
    /// Per-rank logical KV capacity. `None` disables token-capacity decisions.
    pub token_capacity: Option<usize>,
    /// Logical tokens reserved for immediate decode growth.
    pub decode_buffer_tokens: usize,
    /// Maximum request-specific acting TTL.
    pub max_acting_ttl: Duration,
    /// Capacity-repair trigger ratio.
    pub high_watermark_ratio: f64,
    /// Admission and resume capacity ratio.
    pub low_watermark_ratio: f64,
    /// Maximum cache-continuous service rounds before pressure may yield.
    pub max_segment_rounds: usize,
    /// Fixed sample count required before rolling estimates are active.
    pub stats_window_size: usize,
    /// Whether batch gain may bypass only the growth-reserve requirement.
    pub enable_batch_gain_admission: bool,
    /// Offline cold-prefill model.
    pub prefill: PrefillCostModel,
    /// Offline decode throughput surface.
    pub decode: DecodeThroughputModel,
}

impl Default for ProgressTtlConfig {
    fn default() -> Self {
        Self {
            token_capacity: None,
            decode_buffer_tokens: 100,
            max_acting_ttl: Duration::from_secs(10),
            high_watermark_ratio: 1.0,
            low_watermark_ratio: 1.0,
            max_segment_rounds: 14,
            stats_window_size: 100,
            enable_batch_gain_admission: true,
            prefill: PrefillCostModel::default(),
            decode: DecodeThroughputModel::default(),
        }
    }
}

/// Immutable inputs required by the current batch-gain decision.
#[derive(Debug, Clone)]
pub struct BatchGainInputs<'a> {
    /// Rank-local rolling measurements.
    pub factors: &'a ProgressTtlFactors,
    /// Number of active reasoning Programs before admission.
    pub batch_size_before: usize,
    /// Sum of full contexts for active reasoning Programs.
    pub total_context_tokens_before: usize,
    /// Full context of the candidate Program.
    pub candidate_context_tokens: usize,
    /// Current logical active occupancy before the candidate.
    pub used_tokens: f64,
    /// Candidate private tokens required immediately.
    pub required_tokens: f64,
    /// Remaining protected rounds for every active Program except the candidate.
    pub protected_remaining_rounds: &'a [f64],
    /// Cache-miss impact for each corresponding protected Program.
    pub protected_recovery_cost_seconds: &'a [f64],
    /// Candidate cold-prefill or reload recovery cost.
    pub candidate_recovery_cost_seconds: f64,
}

/// Breakdown of the batch-gain comparison used for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchGainEstimate {
    /// Estimated decode benefit from increasing the batch by one request.
    pub batch_gain_seconds: f64,
    /// Candidate recovery cost charged by the current policy.
    pub candidate_recovery_cost_seconds: f64,
    /// Recovery cost caused by shortening protected continuous execution.
    pub continuity_loss_seconds: f64,
    /// Whether benefit covers both costs.
    pub admits: bool,
}

/// Stateless implementation of the formulas used by the current Router PT.
#[derive(Debug, Clone)]
pub struct ProgressTtlPolicyMath {
    config: ProgressTtlConfig,
}

impl ProgressTtlPolicyMath {
    /// Construct the pure policy calculator.
    pub fn new(config: ProgressTtlConfig) -> Self {
        Self { config }
    }

    /// Current immutable parameters.
    pub fn config(&self) -> &ProgressTtlConfig {
        &self.config
    }

    /// Cost of recomputing `uncached_prompt_tokens`, including decode impact.
    pub fn cache_miss_impact_seconds(&self, uncached_prompt_tokens: usize) -> f64 {
        if uncached_prompt_tokens == 0 {
            return 0.0;
        }
        let x = uncached_prompt_tokens as f64 / 1000.0;
        let prefill = self.config.prefill.intercept_seconds
            + self.config.prefill.linear_seconds_per_1k_tokens * x
            + self.config.prefill.quadratic_seconds_per_1k_tokens_squared * x * x;
        prefill * (2.0 - self.config.prefill.decode_throughput_alpha)
    }

    /// Fit one request-specific TTL from the mature rank interval window.
    ///
    /// The cold-start value is zero. Candidate utility is
    /// `P(T < d) C - E[T; T < d] - d P(T >= d)` under a log-normal fit.
    pub fn fitted_acting_ttl(
        &self,
        factors: &ProgressTtlFactors,
        cache_miss_impact_seconds: f64,
    ) -> Duration {
        if !factors.continuity_window_complete(self.config.stats_window_size) {
            return Duration::ZERO;
        }
        let intervals = factors
            .continuity_samples()
            .map(|sample| sample.interval_seconds)
            .filter(|interval| interval.is_finite() && *interval > 0.0)
            .collect::<Vec<_>>();
        if intervals.is_empty() {
            return Duration::ZERO;
        }
        let log_mean =
            intervals.iter().map(|interval| interval.ln()).sum::<f64>() / intervals.len() as f64;
        let log_variance = intervals
            .iter()
            .map(|interval| (interval.ln() - log_mean).powi(2))
            .sum::<f64>()
            / intervals.len() as f64;
        let log_sigma = log_variance.sqrt().max(1e-6);
        let max_ttl = self
            .config
            .max_acting_ttl
            .as_secs_f64()
            .min(cache_miss_impact_seconds);
        if max_ttl <= 0.0 || cache_miss_impact_seconds <= 0.0 {
            return Duration::ZERO;
        }
        let min_ttl = 0.05_f64.min(max_ttl);
        let mut candidates = vec![0.0, min_ttl, max_ttl];
        if max_ttl > min_ttl {
            let log_min = min_ttl.ln();
            let log_max = max_ttl.ln();
            for index in 0..32 {
                let ratio = index as f64 / 31.0;
                candidates.push((log_min + (log_max - log_min) * ratio).exp());
            }
        }
        let lognormal_mean = (log_mean + log_sigma * log_sigma / 2.0).exp();
        let mut best = (0.0, f64::NEG_INFINITY);
        for ttl in candidates {
            let utility = if ttl <= 0.0 {
                0.0
            } else {
                let z = (ttl.ln() - log_mean) / log_sigma;
                let hit_probability = Self::normal_cdf(z);
                let truncated_interval = lognormal_mean
                    * Self::normal_cdf((ttl.ln() - log_mean - log_sigma * log_sigma) / log_sigma);
                cache_miss_impact_seconds * hit_probability
                    - truncated_interval
                    - ttl * (1.0 - hit_probability)
            };
            if utility >= best.1 {
                best = (ttl, utility);
            }
        }
        if best.1 < 0.0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(best.0)
        }
    }

    /// Target cache-continuous rounds used by reserve and fairness estimates.
    pub fn target_rounds(&self, factors: &ProgressTtlFactors) -> f64 {
        if !factors.request_window_complete(self.config.stats_window_size) {
            return 0.0;
        }
        (2.0 * factors.average_rounds_since_ttl_pause())
            .ceil()
            .clamp(1.0, self.config.max_segment_rounds as f64)
    }

    /// Growth reserve for the candidate plus remaining active Program rounds.
    pub fn continuous_growth_reserve_tokens(
        &self,
        factors: &ProgressTtlFactors,
        active_remaining_rounds: impl Iterator<Item = f64>,
    ) -> f64 {
        let target_rounds = self.target_rounds(factors);
        let growth = factors.average_context_growth_tokens();
        if target_rounds <= 0.0 || growth <= 0.0 {
            return 0.0;
        }
        growth * (target_rounds + active_remaining_rounds.sum::<f64>())
    }

    /// Predicted aggregate output throughput for one decode batch.
    pub fn decode_throughput(&self, batch_size: usize, total_context_tokens: usize) -> f64 {
        let batch = batch_size as f64;
        batch
            / (self.config.decode.fixed_step_seconds
                + self.config.decode.batch_step_seconds_per_request * batch
                + self.config.decode.context_step_seconds_per_token * total_context_tokens as f64)
    }

    /// Evaluate the current batch-gain bypass without changing capacity facts.
    pub fn estimate_batch_gain(&self, input: BatchGainInputs<'_>) -> BatchGainEstimate {
        let disabled = || BatchGainEstimate {
            batch_gain_seconds: 0.0,
            candidate_recovery_cost_seconds: input.candidate_recovery_cost_seconds,
            continuity_loss_seconds: 0.0,
            admits: false,
        };
        let Some(capacity) = self.config.token_capacity else {
            return disabled();
        };
        if !self.config.enable_batch_gain_admission
            || !input
                .factors
                .request_window_complete(self.config.stats_window_size)
            || input.batch_size_before == 0
        {
            return disabled();
        }
        let throughput_before =
            self.decode_throughput(input.batch_size_before, input.total_context_tokens_before);
        let throughput_after = self.decode_throughput(
            input.batch_size_before + 1,
            input
                .total_context_tokens_before
                .saturating_add(input.candidate_context_tokens),
        );
        if throughput_before <= 0.0 {
            return disabled();
        }
        let relative_batch_gain = (throughput_after / throughput_before - 1.0).max(0.0);
        let average_decode_seconds = input.factors.average_decode_seconds().unwrap_or_else(|| {
            input.batch_size_before as f64 * input.factors.average_completion_tokens()
                / throughput_before
        });
        if !average_decode_seconds.is_finite() || average_decode_seconds <= 0.0 {
            return disabled();
        }
        let batch_gain_seconds = average_decode_seconds * relative_batch_gain;
        let growth = input.factors.average_context_growth_tokens();
        let continuity_loss_seconds = if growth <= 0.0 {
            0.0
        } else {
            let total_protected_rounds = input.protected_remaining_rounds.iter().sum::<f64>();
            let remaining =
                (capacity as f64 * self.config.low_watermark_ratio - input.used_tokens).max(0.0);
            let before = total_protected_rounds.min(remaining / growth);
            let after = total_protected_rounds.min(
                (remaining - input.required_tokens - growth * self.target_rounds(input.factors))
                    .max(0.0)
                    / growth,
            );
            if before > 0.0 && after <= f64::EPSILON {
                f64::INFINITY
            } else if input.protected_recovery_cost_seconds.is_empty() || after <= f64::EPSILON {
                0.0
            } else {
                let average_recovery = ProgressTtlFactors::average(
                    input.protected_recovery_cost_seconds.iter().copied(),
                );
                ((before - after).max(0.0) / after) * average_recovery
            }
        };
        BatchGainEstimate {
            batch_gain_seconds,
            candidate_recovery_cost_seconds: input.candidate_recovery_cost_seconds,
            continuity_loss_seconds,
            admits: batch_gain_seconds
                >= input.candidate_recovery_cost_seconds + continuity_loss_seconds,
        }
    }

    fn normal_cdf(value: f64) -> f64 {
        let x = value / std::f64::consts::SQRT_2;
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let polynomial =
            (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                + 0.254829592)
                * t;
        let erf = sign * (1.0 - polynomial * (-x * x).exp());
        (0.5 * (1.0 + erf)).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program_scheduling::RequestSample;
    use std::time::Instant;

    fn mature_factors(interval: f64) -> ProgressTtlFactors {
        let mut factors = ProgressTtlFactors::default();
        for index in 0..2 {
            factors.push_request(
                RequestSample {
                    prompt_tokens: 10_000,
                    completion_tokens: 100,
                    cached_prompt_tokens: 9_000,
                    context_growth_tokens: Some(1_000),
                    e2e_seconds: 2.0,
                    decode_seconds: Some(1.0),
                    queue_seconds: 0.0,
                    rounds_since_ttl_pause: 3,
                    finished_at: Instant::now() + Duration::from_secs(index),
                },
                2,
            );
            factors.push_continuity(
                super::super::ContinuitySample {
                    interval_seconds: interval,
                    cache_miss_impact_seconds: 6.0,
                },
                2,
            );
        }
        factors
    }

    #[test]
    fn cold_windows_use_zero_ttl_and_reserve() {
        let policy = ProgressTtlPolicyMath::new(ProgressTtlConfig {
            stats_window_size: 2,
            ..ProgressTtlConfig::default()
        });
        let factors = ProgressTtlFactors::default();
        assert_eq!(policy.fitted_acting_ttl(&factors, 6.0), Duration::ZERO);
        assert_eq!(policy.target_rounds(&factors), 0.0);
        assert_eq!(
            policy.continuous_growth_reserve_tokens(&factors, [3.0].into_iter()),
            0.0
        );
    }

    #[test]
    fn mature_growth_reserve_matches_current_formula() {
        let policy = ProgressTtlPolicyMath::new(ProgressTtlConfig {
            stats_window_size: 2,
            max_segment_rounds: 14,
            ..ProgressTtlConfig::default()
        });
        let factors = mature_factors(1.0);
        assert_eq!(policy.target_rounds(&factors), 6.0);
        assert_eq!(
            policy.continuous_growth_reserve_tokens(&factors, [2.0, 1.0].into_iter()),
            9_000.0
        );
    }

    #[test]
    fn calibrated_cache_miss_and_decode_models_are_finite() {
        let policy = ProgressTtlPolicyMath::new(ProgressTtlConfig::default());
        assert_eq!(policy.cache_miss_impact_seconds(0), 0.0);
        assert!(policy.cache_miss_impact_seconds(100_000) > 0.0);
        assert!(policy.decode_throughput(8, 190_000).is_finite());
    }
}
