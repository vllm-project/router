//! SMetric: session cache affinity with a prefill-work fallback.
use super::{RequestObserver, RequestTracker, RoutingContext};
mod kv;
use crate::config::SMetricConfig;
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::protocols::spec::RoutingPrompt;
use crate::tree::Tree;
use dashmap::DashMap;
use kv::PrefixMatch;
use parking_lot::{Mutex, RwLock};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Weak};
use std::time::Duration;
use tracing::{debug, warn};

/// Cold-start default for `DECODE_WEIGHT` (`beta`) when omitted from config:
/// places the saturation boundary at `B* = 1 / 0.5 = 2` co-resident decode streams
/// until pure-decode step observations excite online linear slope estimation.
pub(super) const DEFAULT_BETA: f64 = 0.5;
const BETA_MIN: f64 = 0.02;
const BETA_MAX: f64 = 2.0;
const BETA_UNSET: u64 = u64::MAX;
const BETA_SAMPLE_WINDOW: usize = 128;
const BETA_MIN_BUCKET_SAMPLES: usize = 2;

#[derive(Debug)]
pub(super) struct SMetricStrategy {
    config: SMetricConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>,
    model_betas: DashMap<String, Arc<BetaState>>,
    round_robin: AtomicUsize,
    states: DashMap<usize, Arc<WorkerState>>,
    retired: RwLock<HashMap<usize, Weak<dyn Worker>>>,
    fallback_warnings: AtomicU8,
    eviction_shutdown: Option<mpsc::Sender<()>>,
    eviction_handle: Option<std::thread::JoinHandle<()>>,
}

impl SMetricStrategy {
    fn instance_id(worker: &Arc<dyn Worker>) -> usize {
        Arc::as_ptr(worker) as *const () as usize
    }

    fn tenant_key(worker: &Arc<dyn Worker>) -> String {
        format!("{}#{:x}", worker.url(), Self::instance_id(worker))
    }

    pub fn new(config: SMetricConfig) -> Self {
        let trees: Arc<DashMap<String, Arc<Tree>>> = Arc::new(DashMap::new());
        let (eviction_shutdown, eviction_handle) = if config.eviction_interval_secs == 0 {
            (None, None)
        } else {
            let eviction_trees = Arc::downgrade(&trees);
            let max_size = config.max_tree_size;
            let interval = Duration::from_secs(config.eviction_interval_secs);
            let (shutdown, receiver) = mpsc::channel();
            let handle = std::thread::spawn(move || {
                while matches!(
                    receiver.recv_timeout(interval),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ) {
                    let Some(trees) = eviction_trees.upgrade() else {
                        break;
                    };
                    for entry in trees.iter() {
                        if receiver.try_recv().is_ok() {
                            return;
                        }
                        entry.value().evict_tenant_by_size(max_size);
                    }
                }
            });
            (Some(shutdown), Some(handle))
        };
        Self {
            config,
            trees,
            model_betas: DashMap::new(),
            states: DashMap::new(),
            retired: RwLock::new(HashMap::new()),
            round_robin: AtomicUsize::new(0),
            fallback_warnings: AtomicU8::new(0),
            eviction_shutdown,
            eviction_handle,
        }
    }

    fn effective_beta(&self, worker: &Arc<dyn Worker>, state: Option<&WorkerState>) -> f64 {
        self.config.decode_weight.unwrap_or_else(|| {
            state
                .and_then(|s| s.beta.get())
                .or_else(|| {
                    self.model_betas
                        .get(worker.model_id())
                        .and_then(|b| b.get())
                })
                .unwrap_or(DEFAULT_BETA)
        })
    }

    fn select_prefill(
        &self,
        workers: &[Arc<dyn Worker>],
        prompt: &RoutingPrompt,
        passes_turn_gate: bool,
        colocated: bool,
        event_hits: Option<&[PrefixMatch]>,
        loads: Option<&[usize]>,
    ) -> Option<(usize, RequestTracker, u64)> {
        let retired = self.retired.read();
        let l = prompt.text.chars().count();
        let budget = self.config.slack
            * (self.config.ttft_slo_base + self.config.ttft_slo_per_char * l as f64);
        let start = self.round_robin.fetch_add(1, Ordering::Relaxed) % workers.len().max(1);
        // min: (index, meets, phi, prefill_work, coloc_load, queued, score, own_cost)
        let mut min: Option<(usize, bool, Option<f64>, u64, usize, u64, u64, u64)> = None;
        // prev: (index, hit, own_cost, meets, score, prefill_work)
        let mut prev: Option<(usize, usize, u64, bool, u64, u64)> = None;
        let mut any_meets_slo = false;
        for offset in 0..workers.len() {
            let idx = (start + offset) % workers.len();
            let worker = &workers[idx];
            if retired.contains_key(&Self::instance_id(worker)) || !worker.is_available() {
                continue;
            }
            let hit = if let Some(hits) = &event_hits {
                hits[idx].chars
            } else {
                self.trees
                    .get(worker.model_id())
                    .as_ref()
                    .map_or(0, |tree| {
                        tree.prefix_match_tenant_char_count(&prompt.text, &Self::tenant_key(worker))
                    })
            };
            let n = l.saturating_sub(hit) as f64;
            let cost = (self.config.c_lin * n + self.config.c_att * n * (l as f64 - n / 2.0)).ceil()
                as u64;
            let state = self.states.get(&Self::instance_id(worker));
            let pending_work = state
                .as_ref()
                .map_or(0, |s| s.pending.load(Ordering::Acquire));
            let queued = pending_work.saturating_add(cost);
            let worker_load = loads.map_or_else(|| worker.load(), |loads| loads[idx]);
            let coloc_load = if colocated { worker_load } else { 0 };
            let batch_factor = (coloc_load as u64).saturating_add(1);
            let prefill_work = queued.saturating_mul(batch_factor);
            let rate = self
                .config
                .prefill_rate
                .or_else(|| state.as_ref().and_then(|s| s.rates.lock().rate));
            let beta = if colocated {
                self.effective_beta(worker, state.as_deref().map(Arc::as_ref))
            } else {
                0.0
            };
            let decode_fraction = beta * (coloc_load as f64);
            let phi = rate
                .filter(|&r| r > 0.0 && budget > 0.0)
                .map(|r| (prefill_work as f64) / (r * budget) + decode_fraction);
            let meets = phi.map_or(decode_fraction <= 1.0, |p| p <= 1.0);
            any_meets_slo |= meets;
            let score = match phi {
                Some(p) if beta > 0.0 => (p * 1_000_000.0).ceil() as u64,
                _ => prefill_work,
            };
            if min.is_none_or(
                |(_, best_meets, best_phi, best_pwork, best_load, best_q, _, _)| {
                    if meets != best_meets {
                        return meets;
                    }
                    if let (Some(p), Some(best_p)) = (phi, best_phi) {
                        let cmp = p.total_cmp(&best_p);
                        if cmp != std::cmp::Ordering::Equal {
                            return cmp == std::cmp::Ordering::Less;
                        }
                        return queued < best_q;
                    }
                    (prefill_work, coloc_load, queued) < (best_pwork, best_load, best_q)
                },
            ) {
                min = Some((
                    idx,
                    meets,
                    phi,
                    prefill_work,
                    coloc_load,
                    queued,
                    score,
                    cost,
                ));
            }
            if prev.is_none_or(|(_, best_hit, _, _, best_score, _)| {
                hit > best_hit || (hit == best_hit && score < best_score)
            }) {
                prev = Some((idx, hit, cost, meets, score, prefill_work));
            }
        }
        let (prev_idx, hit, prev_cost, prev_meets, prev_score, prev_pwork) = prev?;
        let (min_idx, _, _, min_pwork, _, _, min_score, min_cost) = min?;
        let (idx, cost, score, sample_work) = if passes_turn_gate
            && (hit as f64) > self.config.hit_ratio * prompt.est_hit_chars as f64
            && (prev_meets || !any_meets_slo)
        {
            (prev_idx, prev_cost, prev_score, prev_pwork)
        } else {
            (min_idx, min_cost, min_score, min_pwork)
        };
        if !prompt.text.is_empty() && event_hits.is_none() {
            self.trees
                .entry(workers[idx].model_id().to_string())
                .or_insert_with(|| Arc::new(Tree::new()))
                .insert(&prompt.text, &Self::tenant_key(&workers[idx]));
        }
        RouterMetrics::record_processed_request(workers[idx].url());
        RouterMetrics::record_policy_decision("kv_aware", workers[idx].url());
        let selected_rate = self.config.prefill_rate.or_else(|| {
            self.states
                .get(&Self::instance_id(&workers[idx]))
                .and_then(|s| s.rates.lock().rate)
        });
        let est_prefill = selected_rate.filter(|&r| r > 0.0).map(|r| {
            Duration::from_secs_f64(((sample_work as f64) / r).clamp(0.001, budget.max(0.05)))
        });
        let tracker = self.track_with_sample(&workers[idx], cost, sample_work, est_prefill);
        let skip_rate_sample = event_hits.as_ref().is_some_and(|hits| {
            let h = &hits[idx];
            !h.observed && !(self.config.sample_all_event_requests && h.index_bound)
        });
        let tracker = if skip_rate_sample {
            tracker.without_rate_sample()
        } else {
            tracker
        };
        Some((idx, tracker, score))
    }

    #[cfg(test)]
    fn track(&self, worker: &Arc<dyn Worker>, work: u64) -> RequestTracker {
        self.track_with_sample(worker, work, work, None)
    }

    fn track_with_sample(
        &self,
        worker: &Arc<dyn Worker>,
        work: u64,
        sample_work: u64,
        estimated_prefill: Option<Duration>,
    ) -> RequestTracker {
        let state = self
            .states
            .get(&Self::instance_id(worker))
            .map(|s| s.clone())
            .unwrap_or_else(|| {
                let learn_beta = self.config.decode_weight.is_none();
                let pool_beta = learn_beta.then(|| {
                    self.model_betas
                        .entry(worker.model_id().to_string())
                        .or_insert_with(|| Arc::new(BetaState::new()))
                        .clone()
                });
                self.states
                    .entry(Self::instance_id(worker))
                    .or_insert_with(|| {
                        Arc::new(WorkerState {
                            worker: Arc::downgrade(worker),
                            model_id: worker.model_id().to_string(),
                            pending: AtomicU64::new(0),
                            rates: Mutex::new(RateHistory::default()),
                            beta: BetaState::new(),
                            pool_beta,
                            tpot_slo: self.config.tpot_slo,
                            learn_rate: self.config.prefill_rate.is_none(),
                            learn_beta,
                        })
                    })
                    .clone()
            });
        RequestTracker::new_with_sample(state, work, sample_work, estimated_prefill)
    }

    fn balance(&self, workers: &[Arc<dyn Worker>], route: &str) -> Option<usize> {
        let retired = self.retired.read();
        let start = self.round_robin.fetch_add(1, Ordering::Relaxed) % workers.len().max(1);
        let idx = (0..workers.len())
            .map(|offset| (start + offset) % workers.len())
            .filter(|&idx| {
                !retired.contains_key(&Self::instance_id(&workers[idx]))
                    && workers[idx].is_available()
            })
            .min_by_key(|&idx| workers[idx].load())?;
        let (input, category) = match route {
            "/v1/chat/completions" => ("non-text chat", 1),
            "/v1/completions" => ("batched or token-ID prompt", 2),
            _ => ("unsupported endpoint or missing typed prompt", 4),
        };
        if self.fallback_warnings.fetch_or(category, Ordering::Relaxed) & category == 0 {
            warn!(route, input, fallback = "least_inflight", selected_worker = workers[idx].url(),
                "SMetric does not support this input; falling back to least-inflight balancing (cache affinity disabled)");
        } else {
            debug!(route, input, fallback = "least_inflight", selected_worker = workers[idx].url(),
                "SMetric does not support this input; falling back to least-inflight balancing (cache affinity disabled)");
        }
        RouterMetrics::record_processed_request(workers[idx].url());
        RouterMetrics::record_policy_decision("kv_aware", workers[idx].url());
        Some(idx)
    }
}

impl Drop for SMetricStrategy {
    fn drop(&mut self) {
        if let Some(shutdown) = self.eviction_shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(handle) = self.eviction_handle.take() {
            let _ = handle.join();
        }
    }
}

const RATE_WINDOW: usize = 64;
const RATE_MIN_SAMPLES: usize = 8;

fn sorted_median(sorted: &[f64]) -> f64 {
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        0.5 * (sorted[mid - 1] + sorted[mid])
    }
}

#[derive(Debug, Default)]
struct RateHistory {
    samples: VecDeque<f64>,
    rate: Option<f64>,
}

impl RateHistory {
    fn record(&mut self, rate: f64) {
        if !rate.is_finite() || rate <= 0.0 {
            return;
        }
        if self.samples.len() == RATE_WINDOW {
            self.samples.pop_front();
        }
        self.samples.push_back(rate);
        if self.samples.len() >= RATE_MIN_SAMPLES {
            let mut sorted: Vec<_> = self.samples.iter().copied().collect();
            sorted.sort_by(f64::total_cmp);
            self.rate = Some(sorted_median(&sorted));
        }
    }
}

/// Online estimator of the standing decode contention weight:
/// `TPOT_pure(b) = tau_1 + delta_tau * b  =>  beta = delta_tau / (TPOT_SLO - tau_1)`
/// fitted across per-concurrency bucket medians once >= 2 distinct concurrency
/// buckets `b` (other co-resident requests) have been observed.
#[derive(Debug)]
struct BetaState {
    cached_bits: AtomicU64,
    samples: Mutex<VecDeque<(usize, f64)>>,
}

impl BetaState {
    fn new() -> Self {
        Self {
            cached_bits: AtomicU64::new(BETA_UNSET),
            samples: Mutex::new(VecDeque::with_capacity(BETA_SAMPLE_WINDOW)),
        }
    }

    fn get(&self) -> Option<f64> {
        let bits = self.cached_bits.load(Ordering::Acquire);
        (bits != BETA_UNSET).then(|| f64::from_bits(bits))
    }

    fn record(&self, peers: usize, tpot_secs: f64, tpot_slo: f64) {
        if !tpot_secs.is_finite() || tpot_secs <= 0.0 || !tpot_slo.is_finite() || tpot_slo <= 0.0 {
            return;
        }
        let mut samples = self.samples.lock();
        if samples.len() == BETA_SAMPLE_WINDOW {
            samples.pop_front();
        }
        samples.push_back((peers, tpot_secs));

        let mut by_peers: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        for &(b, tpot) in samples.iter() {
            by_peers.entry(b).or_default().push(tpot);
        }
        let mut buckets: Vec<(f64, f64)> = Vec::with_capacity(by_peers.len());
        let mut isolated_median: Option<f64> = None;
        for (b, mut vals) in by_peers {
            if vals.len() < BETA_MIN_BUCKET_SAMPLES {
                continue;
            }
            vals.sort_by(f64::total_cmp);
            let med = sorted_median(&vals);
            if b == 0 {
                isolated_median = Some(med);
            }
            buckets.push((b as f64, med));
        }

        let learned = if buckets.len() < 2 {
            // Persistence of Excitation not yet met: keep cold-start default unless
            // isolated single-stream decode (b = 0) already exhausts TPOT_SLO.
            isolated_median
                .filter(|&m0| m0 >= tpot_slo)
                .map(|_| BETA_MAX)
        } else {
            let k = buckets.len() as f64;
            let mean_b = buckets.iter().map(|(b, _)| *b).sum::<f64>() / k;
            let mean_m = buckets.iter().map(|(_, m)| *m).sum::<f64>() / k;
            let mut num = 0.0;
            let mut den = 0.0;
            for &(b, m) in &buckets {
                let db = b - mean_b;
                num += db * (m - mean_m);
                den += db * db;
            }
            let delta_tau = if den > 0.0 { (num / den).max(0.0) } else { 0.0 };
            let tau_1 = (mean_m - delta_tau * mean_b).max(0.0);
            let headroom = tpot_slo - tau_1;
            let beta = if headroom <= 1e-6 {
                BETA_MAX
            } else {
                (delta_tau / headroom).clamp(BETA_MIN, BETA_MAX)
            };
            Some(beta)
        };

        let bits = learned.map_or(BETA_UNSET, f64::to_bits);
        self.cached_bits.store(bits, Ordering::Release);
    }
}

#[derive(Debug)]
struct WorkerState {
    worker: Weak<dyn Worker>,
    model_id: String,
    pending: AtomicU64,
    rates: Mutex<RateHistory>,
    beta: BetaState,
    pool_beta: Option<Arc<BetaState>>,
    tpot_slo: f64,
    learn_rate: bool,
    learn_beta: bool,
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
        self.finish_with_sample(work, work, elapsed);
    }

    fn finish_with_sample(&self, work: u64, sample_work: u64, elapsed: Option<Duration>) {
        let _ = self
            .pending
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |pending| {
                Some(pending.saturating_sub(work))
            });
        if let Some(elapsed) =
            elapsed.filter(|elapsed| self.learn_rate && sample_work > 0 && !elapsed.is_zero())
        {
            self.rates
                .lock()
                .record(sample_work as f64 / elapsed.as_secs_f64());
        }
    }

    fn pure_decode_peers(&self) -> Option<usize> {
        if !self.learn_beta || self.pending.load(Ordering::Acquire) != 0 {
            return None;
        }
        let worker = self.worker.upgrade()?;
        Some(worker.load().saturating_sub(1))
    }

    fn record_decode_step(&self, peers: usize, tpot_secs: f64) {
        if !self.learn_beta || self.pending.load(Ordering::Acquire) != 0 {
            return;
        }
        self.beta.record(peers, tpot_secs, self.tpot_slo);
        if let Some(pool_beta) = &self.pool_beta {
            pool_beta.record(peers, tpot_secs, self.tpot_slo);
        }
    }
}

impl SMetricStrategy {
    pub(super) fn on_worker_removed(&self, worker: &Arc<dyn Worker>) {
        let mut retired = self.retired.write();
        retired.retain(|_, worker| worker.strong_count() > 0);
        retired.insert(Self::instance_id(worker), Arc::downgrade(worker));
        self.states.remove(&Self::instance_id(worker));
        let model_id = worker.model_id();
        if !self
            .states
            .iter()
            .any(|entry| entry.value().model_id == model_id)
        {
            self.model_betas.remove(model_id);
        }
        self.trees.remove_if(model_id, |_, tree| {
            tree.remove_tenant(&Self::tenant_key(worker));
            tree.tenant_char_count.is_empty()
        });
    }

    pub(super) fn with_active_worker<T>(
        &self,
        worker: &dyn Worker,
        bind: impl FnOnce() -> T,
    ) -> Option<T> {
        let id = worker as *const dyn Worker as *const () as usize;
        let retired = self.retired.read();
        if retired.contains_key(&id) {
            None
        } else {
            Some(bind())
        }
    }

    fn select_request(
        &self,
        workers: &[Arc<dyn Worker>],
        context: &RoutingContext<'_>,
        hits: Option<&[PrefixMatch]>,
        loads: Option<&[usize]>,
    ) -> Option<(usize, Option<RequestTracker>, u64)> {
        if let Some(prompt) = context.prompt {
            self.select_prefill(
                workers,
                prompt,
                context.allow_affinity,
                context.colocated,
                hits,
                loads,
            )
            .map(|(idx, tracker, score)| (idx, Some(tracker), score))
        } else {
            self.balance(workers, context.route)
                .map(|idx| (idx, None, workers[idx].load() as u64))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn unobserved_event_work_is_released_without_training_the_rate() {
        let policy = SMetricStrategy::new(serde_yaml::from_str("C_LIN: 1\nC_ATT: 0\nSLACK: 1\nHIT_RATIO: 0.1\nTTFT_SLO_BASE: 20\nTTFT_SLO_PER_CHAR: 0\nEVICTION_INTERVAL_SECS: 0\n").unwrap());
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "grpc://worker:1".into(),
            WorkerType::Regular,
        ));
        let prompt = RoutingPrompt {
            text: "abcdefghijklm".into(),
            est_hit_chars: 0,
        };
        let (_, mut tracker, _) = policy
            .select_prefill(
                std::slice::from_ref(&worker),
                &prompt,
                true,
                true,
                Some(&[PrefixMatch::default()]),
                None,
            )
            .unwrap();
        let state = policy
            .states
            .get(&SMetricStrategy::instance_id(&worker))
            .unwrap()
            .clone();
        assert_eq!(state.pending.load(Ordering::Acquire), 13);
        tracker.on_first_token();
        assert_eq!(state.pending.load(Ordering::Acquire), 0);
        assert!(state.rates.lock().samples.is_empty());
    }

    #[derive(Clone)]
    struct LogCapture(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for LogCapture {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn fallback_warns_once_per_bounded_category_and_policy() {
        let _other_dispatch = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let output = LogCapture(Arc::new(Mutex::new(Vec::new())));
        let writer = output.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .without_time()
            .with_max_level(tracing::Level::DEBUG)
            .with_writer(move || writer.clone())
            .finish();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
            "http://fallback-worker".into(),
            WorkerType::Regular,
        ))];
        tracing::subscriber::with_default(subscriber, || {
            let policy = removal_test_policy();
            for route in [
                "/v1/chat/completions",
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/completions",
                "/v1/embeddings",
                "/v1/embeddings",
                "/custom/one",
                "/custom/two",
            ] {
                let context = RoutingContext {
                    prompt: None,
                    allow_affinity: false,
                    colocated: false,
                    route,
                    kv: None,
                };
                assert_eq!(
                    policy
                        .select_request(&workers, &context, None, None)
                        .unwrap()
                        .0,
                    0
                );
            }
            assert_eq!(
                removal_test_policy().balance(&workers, "/v1/chat/completions"),
                Some(0)
            );
        });
        let lines = String::from_utf8(output.0.lock().clone()).unwrap();
        assert!(!lines.contains("private fallback payload"));
        let events: Vec<serde_json::Value> = lines
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            events
                .iter()
                .filter(|event| event["level"] == "WARN")
                .count(),
            4
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event["level"] == "DEBUG")
                .count(),
            5
        );
        assert_eq!(
            events
                .iter()
                .map(|event| event["level"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["WARN", "DEBUG", "WARN", "DEBUG", "WARN", "DEBUG", "DEBUG", "DEBUG", "WARN"]
        );
        for event in &events {
            assert_eq!(event["fields"]["selected_worker"], "http://fallback-worker");
            assert_eq!(event["fields"]["fallback"], "least_inflight");
            assert!(event["fields"]["route"].as_str().unwrap().starts_with('/'));
            assert!([
                "non-text chat",
                "batched or token-ID prompt",
                "unsupported endpoint or missing typed prompt"
            ]
            .contains(&event["fields"]["input"].as_str().unwrap()));
        }
    }

    #[test]
    fn configured_eviction_expires_cache_while_zero_preserves_it() {
        let yaml = "C_LIN: 1\nC_ATT: 0\nSLACK: 1\nHIT_RATIO: 0.5\nTTFT_SLO_BASE: 10\nTTFT_SLO_PER_CHAR: 0\nMAX_TREE_SIZE: 1\n";
        let enabled = SMetricStrategy::new(
            serde_yaml::from_str(&format!("{yaml}EVICTION_INTERVAL_SECS: 1\n")).unwrap(),
        );
        let disabled = SMetricStrategy::new(
            serde_yaml::from_str(&format!("{yaml}EVICTION_INTERVAL_SECS: 0\n")).unwrap(),
        );
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://eviction-worker".into(),
            WorkerType::Regular,
        ));
        for policy in [&enabled, &disabled] {
            drop(select(
                policy,
                std::slice::from_ref(&worker),
                &prompt("cached history", 0),
                false,
                false,
            ));
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(4);
        let enabled_tree = enabled.trees.get(worker.model_id()).unwrap().clone();
        while enabled_tree
            .prefix_match_tenant_char_count("cached history", &SMetricStrategy::tenant_key(&worker))
            != 0
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            enabled_tree.prefix_match_tenant_char_count(
                "cached history",
                &SMetricStrategy::tenant_key(&worker)
            ),
            0
        );
        assert_eq!(
            disabled
                .trees
                .get(worker.model_id())
                .unwrap()
                .prefix_match_tenant_char_count(
                    "cached history",
                    &SMetricStrategy::tenant_key(&worker)
                ),
            14
        );
    }

    #[test]
    fn dropping_policy_stops_idle_eviction_without_waiting_for_its_interval() {
        let mut policy = removal_test_policy();
        assert_eq!(policy.config.eviction_interval_secs, 30);
        let handle = policy.eviction_handle.take().unwrap();

        drop(policy);

        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while !handle.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            handle.is_finished(),
            "idle eviction thread must stop when its policy is dropped"
        );
        handle.join().unwrap();
    }

    fn removal_test_policy() -> SMetricStrategy {
        SMetricStrategy::new(serde_yaml::from_str(
            "C_LIN: 1\nC_ATT: 0\nSLACK: 1\nHIT_RATIO: 0.5\nTTFT_SLO_BASE: 10\nTTFT_SLO_PER_CHAR: 0\n",
        ).unwrap())
    }

    #[test]
    fn worker_removal_rejects_stale_prefill_snapshot_before_first_selection() {
        let policy = removal_test_policy();
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://stale-prefill".into(),
            WorkerType::Regular,
        ));

        policy.on_worker_removed(&worker);

        assert!(policy
            .select_prefill(
                std::slice::from_ref(&worker),
                &prompt("stale request", 0),
                false,
                false,
                None,
                None
            )
            .is_none());
        assert!(policy.states.is_empty());
        assert!(policy.trees.is_empty());
    }

    #[test]
    fn worker_removal_rejects_stale_fallback_snapshot_before_first_selection() {
        let policy = removal_test_policy();
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://stale-fallback".into(),
            WorkerType::Regular,
        ));

        policy.on_worker_removed(&worker);

        assert_eq!(
            policy.balance(std::slice::from_ref(&worker), "/v1/embeddings"),
            None
        );
    }

    #[test]
    fn worker_removal_preserves_replacement_selected_before_old_notification() {
        let policy = removal_test_policy();
        let old: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://same-url".into(),
            WorkerType::Regular,
        ));
        let replacement: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://same-url".into(),
            WorkerType::Regular,
        ));
        let (_, mut old_tracker) = select(
            &policy,
            std::slice::from_ref(&old),
            &prompt("old history", 0),
            false,
            false,
        );
        let (_, replacement_tracker) = select(
            &policy,
            std::slice::from_ref(&replacement),
            &prompt("new history", 0),
            false,
            false,
        );

        policy.on_worker_removed(&old);

        assert!(policy
            .states
            .contains_key(&SMetricStrategy::instance_id(&replacement)));
        let replacement_state = policy
            .states
            .get(&SMetricStrategy::instance_id(&replacement))
            .unwrap()
            .clone();
        assert_eq!(replacement_state.pending.load(Ordering::Acquire), 11);
        assert_eq!(
            policy
                .trees
                .get(replacement.model_id())
                .unwrap()
                .prefix_match_tenant_char_count(
                    "new history",
                    &SMetricStrategy::tenant_key(&replacement)
                ),
            11
        );
        old_tracker.on_first_token();
        assert_eq!(replacement_state.pending.load(Ordering::Acquire), 11);
        assert!(replacement_state.rates.lock().samples.is_empty());
        drop(replacement_tracker);
        assert_eq!(replacement_state.pending.load(Ordering::Acquire), 0);
    }

    #[test]
    fn worker_removal_prunes_dead_markers_but_keeps_live_snapshots() {
        let policy = removal_test_policy();
        let stale: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://retained-snapshot".into(),
            WorkerType::Regular,
        ));
        policy.on_worker_removed(&stale);
        assert_eq!(policy.retired.read().len(), 1);

        for index in 0..16 {
            let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
                format!("http://temporary-{index}"),
                WorkerType::Regular,
            ));
            let weak = Arc::downgrade(&worker);
            policy.on_worker_removed(&worker);
            assert_eq!(policy.retired.read().len(), 2);
            drop(worker);
            assert!(weak.upgrade().is_none());
        }
        assert_eq!(
            policy.balance(std::slice::from_ref(&stale), "/v1/embeddings"),
            None
        );
    }

    #[test]
    fn worker_removal_clears_state_and_only_its_tree_tenant() {
        let policy = removal_test_policy();
        let first: Arc<dyn Worker> =
            Arc::new(BasicWorker::new("http://first".into(), WorkerType::Regular));
        let second: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://second".into(),
            WorkerType::Regular,
        ));
        drop(select(
            &policy,
            std::slice::from_ref(&first),
            &prompt("first history", 0),
            false,
            false,
        ));
        drop(select(
            &policy,
            std::slice::from_ref(&second),
            &prompt("second history", 0),
            false,
            false,
        ));

        policy.on_worker_removed(&first);

        assert!(!policy
            .states
            .contains_key(&SMetricStrategy::instance_id(&first)));
        assert!(policy
            .states
            .contains_key(&SMetricStrategy::instance_id(&second)));
        let tree = policy.trees.get(first.model_id()).unwrap();
        assert_eq!(
            tree.prefix_match_tenant_char_count(
                "first history",
                &SMetricStrategy::tenant_key(&first)
            ),
            0
        );
        assert!(!tree
            .tenant_char_count
            .contains_key(SMetricStrategy::tenant_key(&first).as_str()));
        assert_eq!(
            tree.prefix_match_tenant_char_count(
                "second history",
                &SMetricStrategy::tenant_key(&second)
            ),
            14
        );
        drop(tree);

        policy.on_worker_removed(&second);

        assert!(policy.states.is_empty());
        assert!(policy.trees.is_empty());
    }

    #[test]
    fn worker_removal_keeps_old_tracker_isolated_from_same_url_replacement() {
        let policy = removal_test_policy();
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://reused".into(),
            WorkerType::Regular,
        ));
        let (_, mut old_tracker) = select(
            &policy,
            std::slice::from_ref(&worker),
            &prompt("shared", 0),
            false,
            false,
        );
        let old_state = policy
            .states
            .get(&SMetricStrategy::instance_id(&worker))
            .unwrap()
            .clone();

        policy.on_worker_removed(&worker);
        let replacement: Arc<dyn Worker> =
            Arc::new(BasicWorker::new(worker.url().into(), WorkerType::Regular));
        let (_, replacement_tracker) = select(
            &policy,
            std::slice::from_ref(&replacement),
            &prompt("shared", 0),
            false,
            false,
        );
        let replacement_state = policy
            .states
            .get(&SMetricStrategy::instance_id(&replacement))
            .unwrap()
            .clone();

        assert!(!Arc::ptr_eq(&old_state, &replacement_state));
        assert_eq!(replacement_state.pending.load(Ordering::Acquire), 6);
        old_tracker.on_first_token();
        drop(old_tracker);
        assert_eq!(old_state.pending.load(Ordering::Acquire), 0);
        assert_eq!(replacement_state.pending.load(Ordering::Acquire), 6);
        assert!(replacement_state.rates.lock().samples.is_empty());
        drop(replacement_tracker);
        assert_eq!(replacement_state.pending.load(Ordering::Acquire), 0);
    }

    fn prompt(text: &str, est_hit_chars: usize) -> RoutingPrompt {
        RoutingPrompt {
            text: text.to_owned(),
            est_hit_chars,
        }
    }

    fn select(
        policy: &SMetricStrategy,
        workers: &[Arc<dyn Worker>],
        prompt: &RoutingPrompt,
        turn_gate: bool,
        colocated: bool,
    ) -> (usize, RequestTracker) {
        policy
            .select_prefill(workers, prompt, turn_gate, colocated, None, None)
            .map(|(index, tracker, _)| (index, tracker))
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
            tpot_slo: 0.020,
            decode_weight: None,
            sample_all_event_requests: false,
            max_tree_size: 1000,
            eviction_interval_secs: 30,
        };
        let policy = SMetricStrategy::new(config);
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
        let first = policy.track(&workers[0], 100);
        assert_eq!(
            select(&policy, &workers, &prompt("shared plus", 6), true, false).0,
            1
        );
        let second = policy.track(&workers[1], 1000);
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
                .get(&SMetricStrategy::instance_id(&workers[0]))
                .unwrap()
                .pending
                .load(Ordering::Acquire),
            0
        );
        assert_eq!(
            policy
                .states
                .get(&SMetricStrategy::instance_id(&workers[1]))
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
            tpot_slo: 0.020,
            decode_weight: None,
            sample_all_event_requests: false,
            max_tree_size: 1000,
            eviction_interval_secs: 30,
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
                &SMetricStrategy::new(config.clone()),
                &workers,
                &prompt,
                false,
                true
            )
            .0,
            1
        );
        assert_eq!(
            select(
                &SMetricStrategy::new(config),
                &workers,
                &prompt,
                false,
                false
            )
            .0,
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
            tpot_slo: 0.020,
            decode_weight: None,
            sample_all_event_requests: false,
            max_tree_size: 1000,
            eviction_interval_secs: 30,
        };
        let workers: Vec<Arc<dyn Worker>> = ["http://one", "http://two"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        let dynamic = SMetricStrategy::new(config.clone());
        assert_eq!(
            select(&dynamic, &workers, &prompt("shared", 0), false, false).0,
            0
        );
        let queued = dynamic.track(&workers[0], 100);
        let follow_up = prompt("shared plus", 6);
        assert_eq!(select(&dynamic, &workers, &follow_up, true, false).0, 0);
        let rate_history = dynamic
            .states
            .get(&SMetricStrategy::instance_id(&workers[0]))
            .unwrap();
        for _ in 0..RATE_MIN_SAMPLES {
            rate_history.rates.lock().record(1.0);
        }
        drop(rate_history);
        assert_eq!(select(&dynamic, &workers, &follow_up, true, false).0, 1);
        drop(queued);

        let fixed = SMetricStrategy::new(SMetricConfig {
            prefill_rate: Some(1000.0),
            ..config
        });
        assert_eq!(
            select(&fixed, &workers, &prompt("shared", 0), false, false).0,
            0
        );
        let fixed_queue = fixed.track(&workers[0], 100);
        // The configured rate remains authoritative even if observed rates exist.
        fixed
            .states
            .get(&SMetricStrategy::instance_id(&workers[0]))
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
            tpot_slo: 0.020,
            decode_weight: None,
            sample_all_event_requests: false,
            max_tree_size: 1000,
            eviction_interval_secs: 30,
        };
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorker::new("http://one".into(), WorkerType::Regular));
        let workers = vec![worker.clone()];
        let learned = SMetricStrategy::new(config.clone());
        for i in 0..RATE_MIN_SAMPLES {
            let sample = prompt(&format!("prefill-{i}"), 0);
            let other = learned.track(&worker, 13);
            let (_, mut tracker) = select(&learned, &workers, &sample, false, true);
            assert!(
                learned
                    .states
                    .get(&SMetricStrategy::instance_id(&worker))
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
                    .get(&SMetricStrategy::instance_id(&worker))
                    .unwrap()
                    .pending
                    .load(Ordering::Acquire),
                13
            );
            drop(other);
            assert_eq!(
                learned
                    .states
                    .get(&SMetricStrategy::instance_id(&worker))
                    .unwrap()
                    .pending
                    .load(Ordering::Acquire),
                0
            );
            assert_eq!(
                learned
                    .states
                    .get(&SMetricStrategy::instance_id(&worker))
                    .unwrap()
                    .rates
                    .lock()
                    .rate
                    .is_some(),
                i + 1 == RATE_MIN_SAMPLES
            );
        }

        let fixed = SMetricStrategy::new(SMetricConfig {
            prefill_rate: Some(1000.0),
            ..config
        });
        let (_, mut tracker) = select(&fixed, &workers, &prompt("fixed prefill", 0), false, true);
        tracker.on_first_token();
        assert_eq!(
            fixed
                .states
                .get(&SMetricStrategy::instance_id(&worker))
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
                .get(&SMetricStrategy::instance_id(&worker))
                .unwrap()
                .pending
                .load(Ordering::Acquire),
            0
        );
    }

    #[test]
    fn fallback_balances_without_seeding_affinity_and_skips_unavailable_workers() {
        let policy = SMetricStrategy::new(SMetricConfig {
            c_lin: 1.0,
            c_att: 0.0,
            prefill_rate: Some(1000.0),
            slack: 1.0,
            hit_ratio: 0.1,
            ttft_slo_base: 20.0,
            ttft_slo_per_char: 0.0,
            tpot_slo: 0.020,
            decode_weight: None,
            sample_all_event_requests: false,
            max_tree_size: 1000,
            eviction_interval_secs: 30,
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
            kv: None,
        };
        assert_eq!(
            policy
                .select_request(&workers, &context, None, None)
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
                policy.balance(&workers, "<generic-policy-call>"),
                policy.balance(&workers, "<generic-policy-call>")
            ),
            (Some(1), Some(0))
        );
        workers[1].set_healthy(false);
        assert_eq!(policy.balance(&workers, "<generic-policy-call>"), Some(0));
        workers[0].set_healthy(false);
        assert_eq!(policy.balance(&workers, "<generic-policy-call>"), None);
    }

    #[test]
    fn unified_phi_defaults_to_beta_half_and_sheds_when_decode_saturates() {
        let policy = SMetricStrategy::new(
            serde_yaml::from_str(
                "C_LIN: 1\nC_ATT: 0\nPREFILL_RATE: 1000000\nSLACK: 1\nHIT_RATIO: 0.5\nTTFT_SLO_BASE: 10\nTTFT_SLO_PER_CHAR: 0\nEVICTION_INTERVAL_SECS: 0\n",
            )
            .unwrap(),
        );
        let workers: Vec<Arc<dyn Worker>> = ["http://home", "http://idle"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        assert_eq!(policy.effective_beta(&workers[0], None), 0.5);
        assert_eq!(
            select(&policy, &workers, &prompt("shared prefix", 0), false, true).0,
            0
        );

        // At B_home = 1, Phi(home) ~= 0.5 <= 1.0: session stays on warm home worker.
        workers[0].increment_load();
        assert_eq!(
            select(
                &policy,
                &workers,
                &prompt("shared prefix next", 13),
                true,
                true
            )
            .0,
            0
        );

        // At B_home = 2 with nonzero prefill work, Phi(home) = prefill + 1.0 > 1.0:
        // home exceeds the unified budget and yields to idle worker 1.
        workers[0].increment_load();
        assert_eq!(
            select(
                &policy,
                &workers,
                &prompt("shared prefix turn3", 18),
                true,
                true
            )
            .0,
            1
        );
        workers[0].decrement_load();
        workers[0].decrement_load();
    }

    #[test]
    fn rate_history_uses_p50_median_to_resist_fast_outliers() {
        let mut history = RateHistory::default();
        // Five normal prefill samples at 1000/s and three fast outliers at 50_000/s.
        for &r in &[
            1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 50_000.0, 50_000.0, 50_000.0,
        ] {
            history.record(r);
        }
        assert_eq!(history.rate, Some(1000.0));
    }

    #[test]
    fn beta_estimator_recovers_pure_decode_slope_and_ignores_prefill_stalls() {
        let policy = SMetricStrategy::new(
            serde_yaml::from_str(
                "C_LIN: 1\nC_ATT: 0\nPREFILL_RATE: 1000000\nSLACK: 1\nHIT_RATIO: 0.5\nTTFT_SLO_BASE: 10\nTTFT_SLO_PER_CHAR: 0\nTPOT_SLO: 0.020\nEVICTION_INTERVAL_SECS: 0\n",
            )
            .unwrap(),
        );
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorker::new("http://w0".into(), WorkerType::Regular));
        let workers = vec![worker.clone()];

        // Admit one streaming request (`worker.load() == 1` => `peers = 0`).
        let (_, mut tracker) = select(&policy, &workers, &prompt("turn1", 0), false, true);
        worker.increment_load();
        tracker.on_first_token();
        let state = policy
            .states
            .get(&SMetricStrategy::instance_id(&worker))
            .unwrap()
            .clone();
        assert_eq!(state.pure_decode_peers(), Some(0));

        // Only one concurrency bucket (b = 0) observed: Persistence of Excitation
        // keeps beta at DEFAULT_BETA = 0.5.
        state.record_decode_step(0, 0.012);
        state.record_decode_step(0, 0.012);
        assert_eq!(policy.effective_beta(&worker, Some(&state)), 0.5);

        // Second concurrency bucket (b = 1): tau_1 = 12ms, tau(1) = 14ms => delta_tau = 2ms,
        // headroom = 20ms - 12ms = 8ms => beta = 2 / 8 = 0.25.
        state.record_decode_step(1, 0.014);
        state.record_decode_step(1, 0.014);
        let learned = policy.effective_beta(&worker, Some(&state));
        assert!(
            (learned - 0.25).abs() < 1e-6,
            "expected learned beta 0.25, got {learned}"
        );

        // Third bucket (b = 2) with an outlier inside the bucket: median resists outlier.
        state.record_decode_step(2, 0.016);
        state.record_decode_step(2, 0.016);
        state.record_decode_step(2, 0.250);
        let learned = policy.effective_beta(&worker, Some(&state));
        assert!(
            (learned - 0.25).abs() < 1e-6,
            "bucket median must reject spike outlier, got {learned}"
        );

        // While another request is prefilling (`pending > 0`), `pure_decode_peers` is None
        // and any overlapping decode window is discarded instead of polluting beta.
        let concurrent_prefill = policy.track(&worker, 500);
        assert_eq!(state.pure_decode_peers(), None);
        for _ in 0..16 {
            std::thread::sleep(Duration::from_millis(2));
            tracker.on_output_tokens(1);
        }
        let after_prefill_overlap = policy.effective_beta(&worker, Some(&state));
        assert!(
            (after_prefill_overlap - 0.25).abs() < 1e-6,
            "prefill-contaminated decode tokens must not alter beta, got {after_prefill_overlap}"
        );
        drop(concurrent_prefill);

        // End-to-end streaming window test on clean pure decode (`pending == 0`, `b = 1`):
        worker.increment_load(); // total load = 2 => peers = 1
        let clean_policy = SMetricStrategy::new(
            serde_yaml::from_str(
                "C_LIN: 1\nC_ATT: 0\nPREFILL_RATE: 1000000\nSLACK: 1\nHIT_RATIO: 0.5\nTTFT_SLO_BASE: 10\nTTFT_SLO_PER_CHAR: 0\nTPOT_SLO: 0.050\nEVICTION_INTERVAL_SECS: 0\n",
            )
            .unwrap(),
        );
        let (_, mut stream_tracker) =
            select(&clean_policy, &workers, &prompt("stream", 0), false, true);
        stream_tracker.on_first_token();
        for _ in 0..16 {
            std::thread::sleep(Duration::from_millis(1));
            stream_tracker.on_output_tokens(1);
        }
        let clean_state = clean_policy
            .states
            .get(&SMetricStrategy::instance_id(&worker))
            .unwrap();
        assert_eq!(clean_state.beta.samples.lock().len(), 2);

        worker.decrement_load();
        worker.decrement_load();
    }

    #[test]
    fn sample_all_event_requests_trains_bound_cold_misses_only() {
        let policy = SMetricStrategy::new(
            serde_yaml::from_str(
                "C_LIN: 1\nC_ATT: 0\nSLACK: 1\nHIT_RATIO: 0.1\nTTFT_SLO_BASE: 20\nTTFT_SLO_PER_CHAR: 0\nSAMPLE_ALL_EVENT_REQUESTS: true\nEVICTION_INTERVAL_SECS: 0\n",
            )
            .unwrap(),
        );
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "grpc://worker:1".into(),
            WorkerType::Regular,
        ));
        let p = prompt("cold-prompt", 0);

        // Unbound / suspended source (`index_bound: false`) still skips rate training.
        let (_, mut unbound_tracker, _) = policy
            .select_prefill(
                std::slice::from_ref(&worker),
                &p,
                false,
                true,
                Some(&[PrefixMatch::default()]),
                None,
            )
            .unwrap();
        std::thread::sleep(Duration::from_millis(1));
        unbound_tracker.on_first_token();
        let state = policy
            .states
            .get(&SMetricStrategy::instance_id(&worker))
            .unwrap()
            .clone();
        assert!(state.rates.lock().samples.is_empty());

        // Bound live source (`index_bound: true`) records confirmed 0-block cold miss.
        let (_, mut bound_tracker, _) = policy
            .select_prefill(
                std::slice::from_ref(&worker),
                &p,
                false,
                true,
                Some(&[PrefixMatch {
                    index_bound: true,
                    ..PrefixMatch::default()
                }]),
                None,
            )
            .unwrap();
        std::thread::sleep(Duration::from_millis(1));
        bound_tracker.on_first_token();
        assert_eq!(state.rates.lock().samples.len(), 1);
    }
}
