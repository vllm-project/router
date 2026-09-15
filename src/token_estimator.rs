//! Router-side prompt-token estimation when exact tokenization is unavailable.
//!
//! Each model and endpoint pair owns an exponentially smoothed tokens-per-byte
//! coefficient. Backend prompt usage updates the coefficient after completion.

use parking_lot::Mutex;
use serde::Serialize;
use std::collections::HashMap;

const DEFAULT_TOKENS_PER_BYTE: f64 = 0.25;
const DEFAULT_COEFFICIENT_MOMENTUM: f64 = 0.9;
const MIN_TOKENS_PER_BYTE: f64 = 0.01;
const MAX_TOKENS_PER_BYTE: f64 = 4.0;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct TokenEstimateScope {
    model_pool: String,
    endpoint: String,
}

impl TokenEstimateScope {
    pub(crate) fn new(model_pool: &str, endpoint: &str) -> Self {
        Self {
            model_pool: model_pool.to_string(),
            endpoint: endpoint.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TokenEstimateCalibration {
    scope: TokenEstimateScope,
    input_bytes: usize,
}

#[derive(Debug)]
pub(crate) struct MomentumTokenEstimator {
    momentum: f64,
    coefficients: Mutex<HashMap<TokenEstimateScope, f64>>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TokenEstimatorDiagnostic {
    model_pool: String,
    endpoint: String,
    tokens_per_byte: f64,
}

impl Default for MomentumTokenEstimator {
    fn default() -> Self {
        Self::new(DEFAULT_COEFFICIENT_MOMENTUM)
    }
}

impl MomentumTokenEstimator {
    fn new(momentum: f64) -> Self {
        assert!((0.0..1.0).contains(&momentum));
        Self {
            momentum,
            coefficients: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn estimate(
        &self,
        scope: TokenEstimateScope,
        input_text: &str,
    ) -> (usize, TokenEstimateCalibration) {
        let input_bytes = input_text.len();
        let coefficient = self
            .coefficients
            .lock()
            .get(&scope)
            .copied()
            .unwrap_or(DEFAULT_TOKENS_PER_BYTE);
        let estimated_tokens = ((input_bytes as f64 * coefficient).ceil() as usize).max(1);
        (
            estimated_tokens,
            TokenEstimateCalibration { scope, input_bytes },
        )
    }

    pub(crate) fn observe(
        &self,
        calibration: &TokenEstimateCalibration,
        actual_prompt_tokens: usize,
    ) -> Option<f64> {
        if calibration.input_bytes == 0 || actual_prompt_tokens == 0 {
            return None;
        }
        let observed = (actual_prompt_tokens as f64 / calibration.input_bytes as f64)
            .clamp(MIN_TOKENS_PER_BYTE, MAX_TOKENS_PER_BYTE);
        let mut coefficients = self.coefficients.lock();
        let previous = coefficients
            .get(&calibration.scope)
            .copied()
            .unwrap_or(DEFAULT_TOKENS_PER_BYTE);
        let updated = self
            .momentum
            .mul_add(previous, (1.0 - self.momentum) * observed);
        coefficients.insert(calibration.scope.clone(), updated);
        Some(updated)
    }

    pub(crate) fn diagnostics(&self) -> Vec<TokenEstimatorDiagnostic> {
        let mut diagnostics = self
            .coefficients
            .lock()
            .iter()
            .map(|(scope, coefficient)| TokenEstimatorDiagnostic {
                model_pool: scope.model_pool.clone(),
                endpoint: scope.endpoint.clone(),
                tokens_per_byte: *coefficient,
            })
            .collect::<Vec<_>>();
        diagnostics.sort_by(|left, right| {
            left.model_pool
                .cmp(&right.model_pool)
                .then_with(|| left.endpoint.cmp(&right.endpoint))
        });
        diagnostics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feedback_updates_coefficient_instead_of_next_token_count() {
        let estimator = MomentumTokenEstimator::new(0.9);
        let scope = TokenEstimateScope::new("model", "/v1/chat/completions");
        let (initial, calibration) = estimator.estimate(scope.clone(), &"x".repeat(400));
        assert_eq!(initial, 100);
        assert_eq!(estimator.observe(&calibration, 200), Some(0.275));
        let (next, _) = estimator.estimate(scope, &"x".repeat(800));
        assert!((220..=221).contains(&next));
    }
}
