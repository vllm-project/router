//! TTFT-aware load balancing policy
//!
//! Estimates Time-to-First-Token (TTFT) for each worker and routes to the one
//! with the lowest estimate. This policy addresses the gap identified in
//! vLLM production-stack issue #583 and vLLM core issue #20962.
//!
//! ## TTFT Estimation Model
//!
//! ```text
//! TTFT = queue_weight × queue_time + (1 - queue_weight) × compute_time
//!
//! queue_time  = pending_requests × avg_task_duration_ms
//! compute_time = estimated_tokens × per_token_cost_ms
//! ```
//!
//! The policy uses observable metrics (worker load, request size) and
//! configurable parameters to estimate TTFT without requiring direct
//! measurement from the engine.

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::info;

/// Configuration for TTFT-aware policy
#[derive(Debug, Clone)]
pub struct TtftAwareConfig {
    /// Average task duration in milliseconds (used for queue time estimation)
    pub avg_task_duration_ms: f64,
    /// Per-token compute cost in milliseconds (used for compute time estimation)
    pub per_token_cost_ms: f64,
    /// Weight for queue time vs compute time (0.0 = only compute, 1.0 = only queue)
    pub queue_weight: f64,
    /// Estimated characters per token for request size estimation
    pub chars_per_token: usize,
}

impl Default for TtftAwareConfig {
    fn default() -> Self {
        Self {
            avg_task_duration_ms: 100.0,
            per_token_cost_ms: 0.5,
            queue_weight: 0.6,
            chars_per_token: 4,
        }
    }
}

/// TTFT-aware routing policy
///
/// Estimates TTFT for each worker using a weighted combination of:
/// - **Queue time**: pending requests × average task duration
/// - **Compute time**: estimated tokens × per-token cost
///
/// This policy is particularly effective for:
/// - Workloads with variable request lengths
/// - Mixed cache hit/miss scenarios
/// - Multi-model serving with different model sizes
#[derive(Debug)]
pub struct TtftAwarePolicy {
    config: TtftAwareConfig,
    /// Cached load information from external monitoring
    cached_loads: RwLock<HashMap<String, isize>>,
}

impl TtftAwarePolicy {
    pub fn new() -> Self {
        Self {
            config: TtftAwareConfig::default(),
            cached_loads: RwLock::new(HashMap::new()),
        }
    }

    pub fn with_config(config: TtftAwareConfig) -> Self {
        Self {
            config,
            cached_loads: RwLock::new(HashMap::new()),
        }
    }

    /// Estimate TTFT for a worker in milliseconds
    fn estimate_ttft(&self, worker: &dyn Worker, request_text: Option<&str>) -> f64 {
        let queue_time = self.estimate_queue_time(worker);
        let compute_time = self.estimate_compute_time(request_text);

        self.config.queue_weight * queue_time + (1.0 - self.config.queue_weight) * compute_time
    }

    /// Estimate queue time based on pending requests
    fn estimate_queue_time(&self, worker: &dyn Worker) -> f64 {
        let pending = if let Ok(loads) = self.cached_loads.read() {
            loads.get(worker.url()).copied().unwrap_or(0)
        } else {
            worker.load() as isize
        };

        pending.max(0) as f64 * self.config.avg_task_duration_ms
    }

    /// Estimate compute time based on request size
    fn estimate_compute_time(&self, request_text: Option<&str>) -> f64 {
        let tokens = request_text
            .map(|t| t.len() / self.config.chars_per_token)
            .unwrap_or(0);
        tokens as f64 * self.config.per_token_cost_ms
    }
}

impl LoadBalancingPolicy for TtftAwarePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        if healthy_indices.len() == 1 {
            return Some(healthy_indices[0]);
        }

        // Find the worker with the lowest estimated TTFT
        let mut best_idx = healthy_indices[0];
        let mut best_ttft = f64::MAX;

        for &idx in &healthy_indices {
            let ttft = self.estimate_ttft(workers[idx].as_ref(), request_text);
            if ttft < best_ttft {
                best_ttft = ttft;
                best_idx = idx;
            }
        }

        info!(
            "TTFT-aware selection: {} (estimated TTFT: {:.2}ms)",
            workers[best_idx].url(),
            best_ttft
        );

        workers[best_idx].increment_processed();
        RouterMetrics::record_processed_request(workers[best_idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[best_idx].url());

        Some(best_idx)
    }

    fn name(&self) -> &'static str {
        "ttft_aware"
    }

    fn update_loads(&self, loads: &HashMap<String, isize>) {
        if let Ok(mut cached) = self.cached_loads.write() {
            *cached = loads.clone();
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for TtftAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn test_ttft_aware_selection_prefers_low_load() {
        let policy = TtftAwarePolicy::new();
        let worker1 = BasicWorker::new("http://w1:8000".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://w2:8000".to_string(), WorkerType::Regular);

        // worker1 has high load, worker2 has low load
        for _ in 0..10 {
            worker1.increment_load();
        }
        for _ in 0..2 {
            worker2.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];

        // Should prefer worker2 (lower load)
        let mut w2_selected = 0;
        for _ in 0..50 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                if idx == 1 {
                    w2_selected += 1;
                }
            }
        }

        // Worker2 should be selected most of the time
        assert!(
            w2_selected > 40,
            "Expected worker2 to be selected most of the time, got {}/50",
            w2_selected
        );
    }

    #[test]
    fn test_ttft_aware_with_cached_loads() {
        let policy = TtftAwarePolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // Update cached loads: w1 has 100 pending, w2 has 10
        let mut loads = HashMap::new();
        loads.insert("http://w1:8000".to_string(), 100);
        loads.insert("http://w2:8000".to_string(), 10);
        policy.update_loads(&loads);

        // Should prefer worker2 with lower cached load
        let mut w2_selected = 0;
        for _ in 0..50 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                if idx == 1 {
                    w2_selected += 1;
                }
            }
        }

        assert!(
            w2_selected > 40,
            "Expected worker2 to be selected most of the time, got {}/50",
            w2_selected
        );
    }

    #[test]
    fn test_ttft_aware_single_worker() {
        let policy = TtftAwarePolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
            "http://w1:8000".to_string(),
            WorkerType::Regular,
        ))];

        assert_eq!(policy.select_worker(&workers, None), Some(0));
    }

    #[test]
    fn test_ttft_aware_estimate() {
        let policy = TtftAwarePolicy::new();
        let worker = BasicWorker::new("http://w1:8000".to_string(), WorkerType::Regular);

        // With no load and no request text, TTFT should be 0
        let ttft = policy.estimate_ttft(&worker, None);
        assert_eq!(ttft, 0.0);

        // With load, TTFT should increase
        for _ in 0..5 {
            worker.increment_load();
        }
        let ttft_with_load = policy.estimate_ttft(&worker, None);
        assert!(ttft_with_load > 0.0);

        // With request text, TTFT should increase
        let ttft_with_text = policy.estimate_ttft(&worker, Some("Hello, world!"));
        assert!(ttft_with_text > ttft_with_load);
    }

    #[test]
    fn test_ttft_aware_config() {
        let config = TtftAwareConfig {
            avg_task_duration_ms: 200.0,
            per_token_cost_ms: 1.0,
            queue_weight: 0.8,
            chars_per_token: 3,
        };
        let policy = TtftAwarePolicy::with_config(config);
        let worker = BasicWorker::new("http://w1:8000".to_string(), WorkerType::Regular);

        for _ in 0..10 {
            worker.increment_load();
        }

        // With high queue weight, load should dominate
        let ttft = policy.estimate_ttft(&worker, Some("short"));
        assert!(ttft > 1000.0); // 10 * 200 * 0.8 = 1600
    }
}
