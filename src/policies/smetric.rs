//! SMetric: session cache affinity with a prefill-work fallback.
use super::{LoadBalancingPolicy, RequestHeaders, RequestObserver, RequestTracker, RoutingContext};
use crate::config::SMetricConfig;
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::protocols::spec::RoutingPrompt;
use crate::tree::Tree;
use dashmap::DashMap;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tracing::warn;

#[derive(Debug)]
pub struct SMetricPolicy {
    config: SMetricConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>,
    round_robin: AtomicUsize,
    states: DashMap<String, Arc<WorkerState>>,
}

impl SMetricPolicy {
    pub fn new(config: SMetricConfig) -> Self {
        let trees: Arc<DashMap<String, Arc<Tree>>> = Arc::new(DashMap::new());
        let eviction_trees = Arc::downgrade(&trees);
        let max_size = config.max_tree_size;
        // Like cache_aware, bound each worker's historical text without scanning per request.
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(30));
            let Some(trees) = eviction_trees.upgrade() else {
                break;
            };
            for entry in trees.iter() {
                entry.value().evict_tenant_by_size(max_size);
            }
        });
        Self {
            config,
            trees,
            states: DashMap::new(),
            round_robin: AtomicUsize::new(0),
        }
    }

    fn select_prefill(
        &self,
        workers: &[Arc<dyn Worker>],
        prompt: &RoutingPrompt,
        passes_turn_gate: bool,
        colocated: bool,
    ) -> Option<(usize, RequestTracker)> {
        let l = prompt.text.chars().count();
        let start = self.round_robin.fetch_add(1, Ordering::Relaxed) % workers.len().max(1);
        let mut min: Option<(usize, u64, u64)> = None; // index, score, own cost
        let mut prev: Option<(usize, usize, u64, bool)> = None; // index, hit, own cost, TTFT
        let mut any_meets_ttft = false;
        for offset in 0..workers.len() {
            let idx = (start + offset) % workers.len();
            let worker = &workers[idx];
            if !worker.is_available() {
                continue;
            }
            let hit = self
                .trees
                .get(worker.model_id())
                .as_ref()
                .map_or(0, |tree| {
                    tree.prefix_match_tenant_char_count(&prompt.text, worker.url())
                });
            let n = (l - hit) as f64;
            let cost = (self.config.c_lin * n + self.config.c_att * n * (l as f64 - n / 2.0)).ceil()
                as u64;
            let state = self.states.get(worker.url());
            let queued = state
                .as_ref()
                .map_or(0, |s| s.pending.load(Ordering::Acquire))
                .saturating_add(cost);
            let rate = self
                .config
                .prefill_rate
                .or_else(|| state.as_ref().and_then(|s| s.rates.lock().rate));
            let meets = rate.is_none_or(|rate| {
                (queued as f64) / rate
                    <= self.config.slack
                        * (self.config.ttft_slo_base + self.config.ttft_slo_per_char * l as f64)
            });
            any_meets_ttft |= meets;
            let score = if colocated {
                queued.saturating_mul(worker.load() as u64)
            } else {
                queued
            };
            if min.is_none_or(|(_, best, _)| score < best) {
                min = Some((idx, score, cost));
            }
            if prev.is_none_or(|(_, best, _, _)| hit > best) {
                prev = Some((idx, hit, cost, meets));
            }
        }
        let (prev_idx, hit, prev_cost, prev_meets) = prev?;
        let (min_idx, _, min_cost) = min?;
        let (idx, cost) = if passes_turn_gate
            && (hit as f64) > self.config.hit_ratio * prompt.est_hit_chars as f64
            && (prev_meets || !any_meets_ttft)
        {
            (prev_idx, prev_cost)
        } else {
            (min_idx, min_cost)
        };
        if !prompt.text.is_empty() {
            self.trees
                .entry(workers[idx].model_id().to_string())
                .or_insert_with(|| Arc::new(Tree::new()))
                .insert(&prompt.text, workers[idx].url());
        }
        RouterMetrics::record_processed_request(workers[idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[idx].url());
        Some((idx, self.track(workers[idx].url(), cost)))
    }

    fn track(&self, worker_url: &str, work: u64) -> RequestTracker {
        let state = self
            .states
            .get(worker_url)
            .map(|s| s.clone())
            .unwrap_or_else(|| {
                self.states
                    .entry(worker_url.to_owned())
                    .or_insert_with(|| {
                        Arc::new(WorkerState {
                            pending: AtomicU64::new(0),
                            rates: Mutex::new(RateHistory::default()),
                            learn_rate: self.config.prefill_rate.is_none(),
                        })
                    })
                    .clone()
            });
        RequestTracker::new(state, work)
    }

    fn balance(&self, workers: &[Arc<dyn Worker>], route: &str) -> Option<usize> {
        let start = self.round_robin.fetch_add(1, Ordering::Relaxed) % workers.len().max(1);
        let idx = (0..workers.len())
            .map(|offset| (start + offset) % workers.len())
            .filter(|&idx| workers[idx].is_available())
            .min_by_key(|&idx| workers[idx].load())?;
        let input = match route {
            "/v1/chat/completions" => "non-text chat",
            "/v1/completions" => "batched or token-ID prompt",
            _ => "unsupported endpoint or missing typed prompt",
        };
        warn!(route, input, fallback = "least_inflight", selected_worker = workers[idx].url(),
            "SMetric does not support this input; falling back to least-inflight balancing (cache affinity disabled)");
        RouterMetrics::record_processed_request(workers[idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[idx].url());
        Some(idx)
    }
}

const RATE_WINDOW: usize = 64;
const RATE_MIN_SAMPLES: usize = 8;

#[derive(Debug, Default)]
struct RateHistory {
    samples: VecDeque<f64>,
    rate: Option<f64>,
}

impl RateHistory {
    fn record(&mut self, rate: f64) {
        if self.samples.len() == RATE_WINDOW {
            self.samples.pop_front();
        }
        self.samples.push_back(rate);
        if self.samples.len() >= RATE_MIN_SAMPLES {
            let mut sorted: Vec<_> = self.samples.iter().copied().collect();
            sorted.sort_by(f64::total_cmp);
            self.rate = Some(sorted[((sorted.len() - 1) as f64 * 0.9).round() as usize]);
        }
    }
}

#[derive(Debug)]
struct WorkerState {
    pending: AtomicU64,
    rates: Mutex<RateHistory>,
    learn_rate: bool,
}

impl RequestObserver for WorkerState {
    fn charge(&self, work: u64) {
        let _ = self
            .pending
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |pending| {
                Some(pending.saturating_add(work))
            });
    }

    fn finish(&self, work: u64, elapsed: Option<Duration>) {
        let _ = self
            .pending
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |pending| {
                Some(pending.saturating_sub(work))
            });
        if let Some(elapsed) =
            elapsed.filter(|elapsed| self.learn_rate && work > 0 && !elapsed.is_zero())
        {
            self.rates
                .lock()
                .record(work as f64 / elapsed.as_secs_f64());
        }
    }
}

impl LoadBalancingPolicy for SMetricPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        self.balance(workers, "<generic-policy-call>")
    }

    fn select_request(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
        context: &RoutingContext<'_>,
    ) -> Option<(usize, Option<RequestTracker>)> {
        if let Some(prompt) = context.prompt {
            self.select_prefill(workers, prompt, context.allow_affinity, context.colocated)
                .map(|(idx, tracker)| (idx, Some(tracker)))
        } else {
            self.balance(workers, context.route).map(|idx| (idx, None))
        }
    }

    fn needs_affinity_prompt(&self) -> bool {
        true
    }

    fn tracks_load(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "smetric"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    fn prompt(text: &str, est_hit_chars: usize) -> RoutingPrompt {
        RoutingPrompt {
            text: text.to_owned(),
            est_hit_chars,
        }
    }

    fn select(
        policy: &SMetricPolicy,
        workers: &[Arc<dyn Worker>],
        prompt: &RoutingPrompt,
        turn_gate: bool,
        colocated: bool,
    ) -> (usize, RequestTracker) {
        policy
            .select_prefill(workers, prompt, turn_gate, colocated)
            .unwrap()
    }

    #[test]
    fn cache_affinity_yields_to_a_feasible_worker_but_survives_when_none_meet_ttft() {
        let config = SMetricConfig {
            c_lin: 1.0,
            c_att: 0.0,
            prefill_rate: Some(1.0),
            slack: 1.0,
            hit_ratio: 0.5,
            ttft_slo_base: 20.0,
            ttft_slo_per_char: 0.0,
            max_tree_size: 1000,
        };
        let policy = SMetricPolicy::new(config);
        let workers: Vec<Arc<dyn Worker>> = ["http://one", "http://two"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        assert_eq!(
            select(&policy, &workers, &prompt("shared", 0), false, false).0,
            0
        );
        let first = policy.track(workers[0].url(), 100);
        assert_eq!(
            select(&policy, &workers, &prompt("shared plus", 6), true, false).0,
            1
        );
        let second = policy.track(workers[1].url(), 1000);
        // Both exceed TTFT. Worker 1 has the longest history even though its q is higher.
        assert_eq!(
            select(
                &policy,
                &workers,
                &prompt("shared plus more", 11),
                true,
                false,
            )
            .0,
            1
        );
        drop(first);
        drop(second);
        assert_eq!(
            policy
                .states
                .get(workers[0].url())
                .unwrap()
                .pending
                .load(Ordering::Acquire),
            0
        );
        assert_eq!(
            policy
                .states
                .get(workers[1].url())
                .unwrap()
                .pending
                .load(Ordering::Acquire),
            0
        );
    }

    #[test]
    fn colocated_balance_counts_inflight_requests_but_pd_prefill_does_not() {
        let config = SMetricConfig {
            c_lin: 1.0,
            c_att: 0.0,
            prefill_rate: Some(1.0),
            slack: 1.0,
            hit_ratio: 0.5,
            ttft_slo_base: 10.0,
            ttft_slo_per_char: 0.0,
            max_tree_size: 1000,
        };
        let workers: Vec<Arc<dyn Worker>> = ["http://one", "http://two"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        for _ in 0..3 {
            workers[0].increment_load();
        }
        workers[1].increment_load();
        let prompt = prompt("first turn", 0);
        assert_eq!(
            select(
                &SMetricPolicy::new(config.clone()),
                &workers,
                &prompt,
                false,
                true
            )
            .0,
            1
        );
        assert_eq!(
            select(&SMetricPolicy::new(config), &workers, &prompt, false, false).0,
            0
        );
    }

    #[test]
    fn learned_rate_is_used_only_without_a_configured_rate() {
        let config = SMetricConfig {
            c_lin: 1.0,
            c_att: 0.0,
            prefill_rate: None,
            slack: 1.0,
            hit_ratio: 0.5,
            ttft_slo_base: 20.0,
            ttft_slo_per_char: 0.0,
            max_tree_size: 1000,
        };
        let workers: Vec<Arc<dyn Worker>> = ["http://one", "http://two"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        let dynamic = SMetricPolicy::new(config.clone());
        assert_eq!(
            select(&dynamic, &workers, &prompt("shared", 0), false, false).0,
            0
        );
        let queued = dynamic.track(workers[0].url(), 100);
        let follow_up = prompt("shared plus", 6);
        assert_eq!(select(&dynamic, &workers, &follow_up, true, false).0, 0);
        let rate_history = dynamic.states.get(workers[0].url()).unwrap();
        for _ in 0..RATE_MIN_SAMPLES {
            rate_history.rates.lock().record(1.0);
        }
        drop(rate_history);
        assert_eq!(select(&dynamic, &workers, &follow_up, true, false).0, 1);
        drop(queued);

        let fixed = SMetricPolicy::new(SMetricConfig {
            prefill_rate: Some(1000.0),
            ..config
        });
        assert_eq!(
            select(&fixed, &workers, &prompt("shared", 0), false, false).0,
            0
        );
        let fixed_queue = fixed.track(workers[0].url(), 100);
        // The configured rate remains authoritative even if observed rates exist.
        fixed
            .states
            .get(workers[0].url())
            .unwrap()
            .rates
            .lock()
            .rate = Some(1.0);
        assert_eq!(select(&fixed, &workers, &follow_up, true, false).0, 0);
        drop(fixed_queue);
    }

    #[test]
    fn first_token_releases_pending_work_and_trains_only_unconfigured_rates() {
        let config = SMetricConfig {
            c_lin: 1.0,
            c_att: 0.0,
            prefill_rate: None,
            slack: 1.0,
            hit_ratio: 0.5,
            ttft_slo_base: 20.0,
            ttft_slo_per_char: 0.0,
            max_tree_size: 1000,
        };
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorker::new("http://one".into(), WorkerType::Regular));
        let workers = vec![worker.clone()];
        let learned = SMetricPolicy::new(config.clone());
        for i in 0..RATE_MIN_SAMPLES {
            let sample = prompt(&format!("prefill-{i}"), 0);
            let other = learned.track(worker.url(), 13);
            let (_, mut tracker) = select(&learned, &workers, &sample, false, true);
            assert!(
                learned
                    .states
                    .get(worker.url())
                    .unwrap()
                    .pending
                    .load(Ordering::Acquire)
                    > 13
            );
            tracker.on_first_token();
            tracker.on_first_token();
            assert_eq!(
                learned
                    .states
                    .get(worker.url())
                    .unwrap()
                    .pending
                    .load(Ordering::Acquire),
                13
            );
            drop(other);
            assert_eq!(
                learned
                    .states
                    .get(worker.url())
                    .unwrap()
                    .pending
                    .load(Ordering::Acquire),
                0
            );
            assert_eq!(
                learned
                    .states
                    .get(worker.url())
                    .unwrap()
                    .rates
                    .lock()
                    .rate
                    .is_some(),
                i + 1 == RATE_MIN_SAMPLES
            );
        }

        let fixed = SMetricPolicy::new(SMetricConfig {
            prefill_rate: Some(1000.0),
            ..config
        });
        let (_, mut tracker) = select(&fixed, &workers, &prompt("fixed prefill", 0), false, true);
        tracker.on_first_token();
        assert_eq!(
            fixed
                .states
                .get(worker.url())
                .unwrap()
                .pending
                .load(Ordering::Acquire),
            0
        );

        let (_, tracker) = select(
            &learned,
            &workers,
            &prompt("cancelled prefill", 0),
            false,
            true,
        );
        drop(tracker);
        assert_eq!(
            learned
                .states
                .get(worker.url())
                .unwrap()
                .pending
                .load(Ordering::Acquire),
            0
        );
    }

    #[test]
    fn fallback_balances_without_seeding_affinity_and_skips_unavailable_workers() {
        let policy = SMetricPolicy::new(SMetricConfig {
            c_lin: 1.0,
            c_att: 0.0,
            prefill_rate: Some(1000.0),
            slack: 1.0,
            hit_ratio: 0.1,
            ttft_slo_base: 20.0,
            ttft_slo_per_char: 0.0,
            max_tree_size: 1000,
        });
        let workers: Vec<Arc<dyn Worker>> = ["http://one", "http://two"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        drop(select(&policy, &workers, &prompt("known", 0), false, false));
        workers[0].increment_load();
        let context = RoutingContext {
            prompt: None,
            allow_affinity: false,
            colocated: true,
            route: "/v1/embeddings",
        };
        assert_eq!(
            policy
                .select_request(&workers, Some("fresh"), None, &context)
                .unwrap()
                .0,
            1
        );
        workers[0].decrement_load();
        // If fallback had inserted its input, this would incorrectly stick to worker 1.
        assert_eq!(
            select(&policy, &workers, &prompt("fresh", 5), true, false).0,
            0
        );
        assert_eq!(
            (
                policy.select_worker(&workers, None),
                policy.select_worker(&workers, None)
            ),
            (Some(1), Some(0))
        );
        workers[1].set_healthy(false);
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        workers[0].set_healthy(false);
        assert_eq!(policy.select_worker(&workers, None), None);
    }
}
