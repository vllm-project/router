//! SMetric: session-centric, SLO-aware routing with a prefill-work fallback.
use super::{RequestObserver, RequestTracker, RoutingContext};
mod kv;
use crate::config::SMetricConfig;
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::protocols::spec::{CacheIdentity, RoutingPrompt};
use crate::tree::Tree;
use dashmap::DashMap;
pub(in crate::policies) use kv::KvBinding;
use kv::PrefixMatch;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Weak};
use std::time::Duration;
use tracing::{debug, warn};

/// Characters per token used when KV token IDs are absent (HTTP text routing).
const CHARS_PER_TOKEN: f64 = 4.0;
/// Cache-affinity evidence threshold: a warm worker must hold more than 50% of
/// the request's expected reusable history to claim session stickiness.
const HIT_RATIO: f64 = 0.5;
const RATE_WINDOW: usize = 64;
const RATE_MIN_SAMPLES: usize = 8;
/// Rolling P90 tracks peak prefill throughput and avoids the flapping trap
/// where congestion-depressed mean/median rates trigger unnecessary migration.
const RATE_QUANTILE: f64 = 0.9;
/// Safety ceiling on single-request prefill cost so `pending` cannot saturate `u64::MAX`.
const MAX_PREFILL_COST: f64 = (1u64 << 50) as f64;

#[derive(Debug)]
pub(super) struct SMetricStrategy {
    config: SMetricConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>,
    round_robin: AtomicUsize,
    states: DashMap<usize, Arc<WorkerState>>,
    retired: Arc<RwLock<HashMap<usize, Weak<dyn Worker>>>>,
    fallback_warnings: AtomicU8,
    eviction_shutdown: Option<mpsc::Sender<()>>,
    eviction_handle: Option<std::thread::JoinHandle<()>>,
}

#[derive(Clone, Copy)]
struct Candidate {
    idx: usize,
    hit: f64,
    cost: u64,
    queued: u64,
    score: u64,
    meets: bool,
    rate: Option<f64>,
    can_sample: bool,
    used_tree: bool,
}

impl SMetricStrategy {
    fn instance_id(worker: &Arc<dyn Worker>) -> usize {
        Arc::as_ptr(worker) as *const () as usize
    }

    fn tenant_key(worker: &Arc<dyn Worker>) -> String {
        format!("{}#{:x}", worker.url(), Self::instance_id(worker))
    }

    /// Namespace scoped HTTP prompts inside the worker's single per-model `Tree` using a
    /// disjoint fixed-length prefix so all scopes share one unified `MAX_TREE_SIZE` LRU.
    fn scoped_tree_text(prompt: &RoutingPrompt) -> Option<(String, usize)> {
        match &prompt.cache_identity {
            CacheIdentity::Shared => Some((format!("0\0{}", prompt.text), 2)),
            CacheIdentity::Scoped(scope) => Some((
                format!("1{scope}\0{}", prompt.text),
                scope.chars().count() + 2,
            )),
            CacheIdentity::Unknown => None,
        }
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
                    // Clone Arc<Tree> handles first so DashMap shard locks are never held across eviction.
                    let snapshot: Vec<Arc<Tree>> = trees
                        .iter()
                        .map(|entry| Arc::clone(entry.value()))
                        .collect();
                    for tree in snapshot {
                        if receiver.try_recv().is_ok() {
                            return;
                        }
                        tree.evict_tenant_by_size(max_size);
                    }
                }
            });
            (Some(shutdown), Some(handle))
        };
        Self {
            config,
            trees,
            states: DashMap::new(),
            retired: Arc::new(RwLock::new(HashMap::new())),
            round_robin: AtomicUsize::new(0),
            fallback_warnings: AtomicU8::new(0),
            eviction_shutdown,
            eviction_handle,
        }
    }

    /// Prefill cost in linear-token equivalents: `n + n * (L - n / 2) / L_eq`
    /// when `ATTENTION_CROSSOVER_TOKENS > 0`, or pure linear `n` when `0`.
    fn prefill_cost(&self, len: f64, hit: f64) -> f64 {
        let new = (len - hit).max(0.0);
        let raw = if self.config.attention_crossover_tokens > 0.0 {
            new + new * (len - new / 2.0) / self.config.attention_crossover_tokens
        } else {
            new
        };
        raw.clamp(0.0, MAX_PREFILL_COST)
    }

    fn ttft_budget_secs(&self, len: f64) -> f64 {
        self.config.ttft_slo_secs + self.config.ttft_slo_secs_per_1k_tokens * (len / 1000.0)
    }

    fn select_prefill(
        &self,
        workers: &[Arc<dyn Worker>],
        context: &RoutingContext<'_>,
        event_hits: Option<&[PrefixMatch]>,
        loads: Option<&[usize]>,
    ) -> Option<(usize, RequestTracker, u64)> {
        let input_tokens = context.kv.as_ref().map(|kv| kv.token_ids.len());
        let total_chars = context.prompt.map_or(0, |p| p.text.chars().count());
        let len = if let Some(n) = input_tokens {
            n as f64
        } else if context.prompt.is_some() {
            total_chars as f64 / CHARS_PER_TOKEN
        } else {
            return None;
        };
        let chars_to_tokens = if total_chars > 0 {
            len / total_chars as f64
        } else {
            0.0
        };
        // Chat sets `est_hit_chars` through the last assistant turn; `/v1/completions`
        // lacks turn boundaries so `est_hit_chars == total_chars` (cold turns have
        // `hit == 0`, fail `hit > HIT_RATIO * history`, and balance). Token-ID
        // requests without text (`prompt == None`) use `history == len`.
        let history = match context.prompt {
            Some(p) if total_chars > 0 => p.est_hit_chars.min(total_chars) as f64 * chars_to_tokens,
            _ => len,
        };
        let tree_allowed = context.kv.as_ref().is_none_or(|kv| {
            kv.cache_salt.is_none_or(str::is_empty) && kv.lora_name.is_none_or(str::is_empty)
        });
        let scoped_prompt = context
            .prompt
            .filter(|_| tree_allowed)
            .and_then(Self::scoped_tree_text);
        let can_write_tree =
            scoped_prompt.is_some() && context.prompt.is_some_and(|p| !p.text.is_empty());

        let retired = self.retired.read();
        let budget = self.ttft_budget_secs(len);
        let start = self.round_robin.fetch_add(1, Ordering::Relaxed) % workers.len().max(1);
        let mut min: Option<Candidate> = None;
        let mut prev: Option<Candidate> = None;
        let mut any_meets_ttft = false;

        for offset in 0..workers.len() {
            let idx = (start + offset) % workers.len();
            let worker = &workers[idx];
            if retired.contains_key(&Self::instance_id(worker)) || !worker.is_available() {
                continue;
            }
            let (hit, can_sample, used_tree) = match event_hits.map(|hits| hits[idx]) {
                Some(PrefixMatch::Bound(tokens)) => (tokens as f64, input_tokens != Some(0), false),
                Some(PrefixMatch::Unavailable) => (0.0, false, false),
                Some(PrefixMatch::Unmanaged) | None => {
                    if let Some((ref scoped, prefix_chars)) = scoped_prompt {
                        let tree = self.trees.get(worker.model_id()).map(|t| Arc::clone(&t));
                        let tree_chars = tree.map_or(0, |tree| {
                            tree.prefix_match_tenant_char_count(scoped, &Self::tenant_key(worker))
                                .saturating_sub(prefix_chars)
                        });
                        let hit = (tree_chars as f64 * chars_to_tokens).min(len);
                        (hit, event_hits.is_none(), can_write_tree)
                    } else {
                        (0.0, false, false)
                    }
                }
            };
            let cost = self.prefill_cost(len, hit).ceil() as u64;
            let state = self.states.get(&Self::instance_id(worker));
            let queued = state
                .as_ref()
                .map_or(0, |s| s.pending.load(Ordering::Acquire))
                .saturating_add(cost);
            let rate = self
                .config
                .prefill_rate
                .or_else(|| state.as_ref().and_then(|s| s.rates.lock().rate));
            let meets = rate.is_none_or(|rate| (queued as f64) / rate <= budget);
            any_meets_ttft |= meets;
            let inflight = loads.map_or_else(|| worker.load(), |loads| loads[idx]) as u64;
            // Floor `queued` at 1 so a full cache hit (`cost == 0`) does not zero out
            // decode load on a busy worker, while idle workers (`inflight == 0`) still tie at 0.
            let score = if context.colocated {
                queued.max(1).saturating_mul(inflight)
            } else {
                queued
            };
            let candidate = Candidate {
                idx,
                hit,
                cost,
                queued,
                score,
                meets,
                rate,
                can_sample,
                used_tree,
            };
            if min.is_none_or(|b| {
                (!candidate.meets, candidate.score, candidate.queued)
                    < (!b.meets, b.score, b.queued)
            }) {
                min = Some(candidate);
            }
            if candidate.hit > HIT_RATIO * history
                && prev.is_none_or(|b| {
                    candidate
                        .meets
                        .cmp(&b.meets)
                        .then_with(|| candidate.hit.total_cmp(&b.hit))
                        .then_with(|| b.queued.cmp(&candidate.queued))
                        .then_with(|| b.score.cmp(&candidate.score))
                        .is_gt()
                })
            {
                prev = Some(candidate);
            }
        }
        let chosen = match prev {
            Some(p) if context.allow_affinity && (p.meets || !any_meets_ttft) => p,
            _ => min?,
        };
        drop(retired);

        let worker = &workers[chosen.idx];
        RouterMetrics::record_processed_request(worker.url());
        RouterMetrics::record_policy_decision("kv_aware", worker.url());
        let est_prefill = chosen
            .rate
            .filter(|&r| r > 0.0 && chosen.cost > 0)
            .and_then(|r| {
                let cap = (budget * 4.0).clamp(5.0, 300.0);
                Duration::try_from_secs_f64((chosen.queued as f64 / r).clamp(0.001, cap)).ok()
            });
        let mut tracker = self
            .track(worker, chosen.cost)
            .with_estimated_prefill(est_prefill);
        if chosen.used_tree {
            if let Some((scoped_text, _)) = scoped_prompt {
                let trees = Arc::clone(&self.trees);
                let retired = Arc::clone(&self.retired);
                let worker_weak = Arc::downgrade(worker);
                let worker_id = Self::instance_id(worker);
                let model_id = worker.model_id().to_string();
                let tenant = Self::tenant_key(worker);
                tracker = tracker.with_on_commit(move || {
                    let retired = retired.read();
                    if worker_weak.strong_count() > 0 && !retired.contains_key(&worker_id) {
                        let tree =
                            trees
                                .get(&model_id)
                                .map(|t| Arc::clone(&t))
                                .unwrap_or_else(|| {
                                    trees
                                        .entry(model_id)
                                        .or_insert_with(|| Arc::new(Tree::new()))
                                        .clone()
                                });
                        tree.insert(&scoped_text, &tenant);
                    }
                });
            }
        }
        if !chosen.can_sample {
            tracker = tracker.without_rate_sample();
        }
        Some((chosen.idx, tracker, chosen.score))
    }

    fn track(&self, worker: &Arc<dyn Worker>, work: u64) -> RequestTracker {
        let state = self
            .states
            .get(&Self::instance_id(worker))
            .map(|s| s.clone())
            .unwrap_or_else(|| {
                self.states
                    .entry(Self::instance_id(worker))
                    .or_insert_with(|| {
                        Arc::new(WorkerState {
                            _worker: Arc::downgrade(worker),
                            pending: AtomicU64::new(0),
                            rates: Mutex::new(RateHistory::default()),
                            learn_rate: self.config.prefill_rate.is_none(),
                        })
                    })
                    .clone()
            });
        RequestTracker::new(state, work)
    }

    fn balance(
        &self,
        workers: &[Arc<dyn Worker>],
        route: &str,
        loads: Option<&[usize]>,
    ) -> Option<usize> {
        let retired = self.retired.read();
        let start = self.round_robin.fetch_add(1, Ordering::Relaxed) % workers.len().max(1);
        let idx = (0..workers.len())
            .map(|offset| (start + offset) % workers.len())
            .filter(|&idx| {
                !retired.contains_key(&Self::instance_id(&workers[idx]))
                    && workers[idx].is_available()
            })
            .min_by_key(|&idx| loads.map_or_else(|| workers[idx].load(), |ls| ls[idx]))?;
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
            self.rate = Some(sorted[((sorted.len() - 1) as f64 * RATE_QUANTILE).round() as usize]);
        }
    }
}

#[derive(Debug)]
struct WorkerState {
    _worker: Weak<dyn Worker>,
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

impl SMetricStrategy {
    pub(super) fn on_worker_removed(&self, worker: &Arc<dyn Worker>) {
        let mut retired = self.retired.write();
        retired.retain(|_, worker| worker.strong_count() > 0);
        retired.insert(Self::instance_id(worker), Arc::downgrade(worker));
        self.states.remove(&Self::instance_id(worker));
        let tenant = Self::tenant_key(worker);
        self.trees.remove_if(worker.model_id(), |_, tree| {
            tree.remove_tenant(&tenant);
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
        if context.prompt.is_some() || context.kv.is_some_and(|kv| !kv.token_ids.is_empty()) {
            self.select_prefill(workers, context, hits, loads)
                .map(|(idx, tracker, score)| (idx, Some(tracker), score))
        } else {
            self.balance(workers, context.route, loads).map(|idx| {
                (
                    idx,
                    None,
                    loads.map_or_else(|| workers[idx].load(), |ls| ls[idx]) as u64,
                )
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use crate::policies::KvRequestContext;

    fn test_config(slo: f64, prefill_rate: Option<f64>) -> SMetricConfig {
        SMetricConfig {
            ttft_slo_secs: slo,
            ttft_slo_secs_per_1k_tokens: 0.0,
            attention_crossover_tokens: 0.0,
            prefill_rate,
            max_tree_size: 1000,
            eviction_interval_secs: 0,
        }
    }

    fn worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(BasicWorker::new(url.into(), WorkerType::Regular))
    }

    fn workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter().map(|u| worker(u)).collect()
    }

    fn prompt(text: &str, est_hit_chars: usize) -> RoutingPrompt {
        RoutingPrompt {
            text: text.to_owned(),
            est_hit_chars,
            cache_identity: CacheIdentity::Shared,
        }
    }

    fn pending(policy: &SMetricStrategy, w: &Arc<dyn Worker>) -> u64 {
        policy
            .states
            .get(&SMetricStrategy::instance_id(w))
            .unwrap()
            .pending
            .load(Ordering::Acquire)
    }

    fn matched_chars(policy: &SMetricStrategy, w: &Arc<dyn Worker>, p: &RoutingPrompt) -> usize {
        let Some((scoped, prefix)) = SMetricStrategy::scoped_tree_text(p) else {
            return 0;
        };
        policy.trees.get(w.model_id()).map_or(0, |tree| {
            tree.prefix_match_tenant_char_count(&scoped, &SMetricStrategy::tenant_key(w))
                .saturating_sub(prefix)
        })
    }

    fn select(
        policy: &SMetricStrategy,
        ws: &[Arc<dyn Worker>],
        p: &RoutingPrompt,
        allow_affinity: bool,
        colocated: bool,
    ) -> (usize, RequestTracker) {
        let ctx = RoutingContext {
            prompt: Some(p),
            kv: None,
            allow_affinity,
            colocated,
            route: "/v1/chat/completions",
        };
        let (idx, t, _) = policy.select_prefill(ws, &ctx, None, None).unwrap();
        (idx, t)
    }

    fn removal_test_policy() -> SMetricStrategy {
        SMetricStrategy::new(
            serde_yaml::from_str("TTFT_SLO_SECS: 10\nATTENTION_CROSSOVER_TOKENS: 0\n").unwrap(),
        )
    }

    #[test]
    fn bound_cold_miss_trains_rate_while_unavailable_and_unknown_skip_it() {
        let policy = SMetricStrategy::new(test_config(20.0, None));
        let w = worker("grpc://worker:1");
        let p = prompt("abcdefghijklm", 0);
        let tokens: Vec<u32> = (1..=13).collect();
        let kv_ctx = RoutingContext {
            prompt: Some(&p),
            kv: Some(KvRequestContext {
                token_ids: &tokens,
                lora_name: None,
                cache_salt: None,
            }),
            allow_affinity: true,
            colocated: true,
            route: "/v1/chat/completions",
        };
        let (_, mut unavail, _) = policy
            .select_prefill(
                std::slice::from_ref(&w),
                &kv_ctx,
                Some(&[PrefixMatch::Unavailable]),
                None,
            )
            .unwrap();
        assert_eq!(pending(&policy, &w), 13);
        unavail.backdate_for_test(Duration::from_millis(1));
        unavail.on_first_token();
        assert_eq!(pending(&policy, &w), 0);
        let state = policy
            .states
            .get(&SMetricStrategy::instance_id(&w))
            .unwrap()
            .clone();
        assert!(state.rates.lock().samples.is_empty());

        // HTTP CacheIdentity::Unknown must also skip rate training so phantom misses cannot poison P90.
        let unknown = RoutingPrompt {
            text: p.text.clone(),
            est_hit_chars: 0,
            cache_identity: CacheIdentity::Unknown,
        };
        let (_, mut u_tracker) = select(&policy, std::slice::from_ref(&w), &unknown, false, true);
        u_tracker.backdate_for_test(Duration::from_millis(1));
        u_tracker.on_first_token();
        assert!(state.rates.lock().samples.is_empty());

        // Bound live source trains on confirmed 0-block cold misses.
        let (_, mut bound, _) = policy
            .select_prefill(
                std::slice::from_ref(&w),
                &kv_ctx,
                Some(&[PrefixMatch::Bound(0)]),
                None,
            )
            .unwrap();
        bound.backdate_for_test(Duration::from_millis(1));
        bound.on_first_token();
        assert_eq!(state.rates.lock().samples.len(), 1);
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
    fn fallback_warns_once_balances_least_inflight_and_skips_unavailable() {
        let _other_dispatch = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let output = LogCapture(Arc::new(Mutex::new(Vec::new())));
        let writer = output.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .without_time()
            .with_max_level(tracing::Level::DEBUG)
            .with_writer(move || writer.clone())
            .finish();
        let ws = workers(&["http://w0", "http://w1"]);
        ws[0].increment_load();
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
                let ctx = RoutingContext {
                    prompt: None,
                    allow_affinity: false,
                    colocated: false,
                    route,
                    kv: None,
                };
                assert_eq!(policy.select_request(&ws, &ctx, None, None).unwrap().0, 1);
            }
            assert_eq!(
                removal_test_policy().balance(&ws, "/v1/chat/completions", None),
                Some(1)
            );
            ws[0].decrement_load();
            assert_eq!(
                (
                    policy.balance(&ws, "<generic>", None),
                    policy.balance(&ws, "<generic>", None)
                ),
                (Some(0), Some(1))
            );
            ws[1].set_healthy(false);
            assert_eq!(policy.balance(&ws, "/v1/embeddings", None), Some(0));
            ws[0].set_healthy(false);
            assert_eq!(policy.balance(&ws, "/v1/embeddings", None), None);
        });
        let events: Vec<serde_json::Value> = String::from_utf8(output.0.lock().clone())
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(
            events[..9]
                .iter()
                .map(|e| e["level"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["WARN", "DEBUG", "WARN", "DEBUG", "WARN", "DEBUG", "DEBUG", "DEBUG", "WARN"]
        );
        assert_eq!(events[0]["fields"]["fallback"], "least_inflight");
        assert_eq!(events[0]["fields"]["selected_worker"], "http://w1");
        assert_eq!(events[0]["fields"]["input"], "non-text chat");
    }

    #[test]
    fn configured_eviction_expires_cache_while_zero_preserves_it() {
        let yaml = "TTFT_SLO_SECS: 10\nMAX_TREE_SIZE: 1\n";
        let enabled = SMetricStrategy::new(
            serde_yaml::from_str(&format!("{yaml}EVICTION_INTERVAL_SECS: 1\n")).unwrap(),
        );
        let disabled = SMetricStrategy::new(
            serde_yaml::from_str(&format!("{yaml}EVICTION_INTERVAL_SECS: 0\n")).unwrap(),
        );
        let w = worker("http://eviction-worker");
        let p = prompt("cached history", 0);
        for policy in [&enabled, &disabled] {
            let (_, mut t) = select(policy, std::slice::from_ref(&w), &p, false, false);
            t.on_first_token();
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(4);
        while matched_chars(&enabled, &w, &p) != 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(matched_chars(&enabled, &w, &p), 0);
        assert_eq!(matched_chars(&disabled, &w, &p), 14);
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
        assert!(handle.is_finished());
        handle.join().unwrap();
    }

    #[test]
    fn deferred_tree_insert_ignores_failed_attempts_and_retired_workers() {
        let policy = SMetricStrategy::new(test_config(10.0, None));
        let w = worker("http://deferred");
        let (_, failed) = select(
            &policy,
            std::slice::from_ref(&w),
            &prompt("failed attempt", 0),
            false,
            true,
        );
        drop(failed);
        assert_eq!(pending(&policy, &w), 0);
        assert!(policy.trees.is_empty());

        let ok_p = prompt("succeeded prefix", 0);
        let (_, mut ok) = select(&policy, std::slice::from_ref(&w), &ok_p, false, true);
        ok.on_first_token();
        assert_eq!(matched_chars(&policy, &w, &ok_p), 16);

        let (_, mut late) = select(
            &policy,
            std::slice::from_ref(&w),
            &prompt("late prefix", 0),
            false,
            true,
        );
        policy.on_worker_removed(&w);
        late.on_first_token();
        assert!(policy.trees.is_empty());
    }

    #[test]
    fn worker_removal_preserves_replacement_clears_only_its_tenant_and_prunes_dead_markers() {
        let policy = SMetricStrategy::new(test_config(10.0, None));
        let never_selected = worker("http://stale-first");
        policy.on_worker_removed(&never_selected);
        assert!(policy
            .select_prefill(
                std::slice::from_ref(&never_selected),
                &RoutingContext {
                    prompt: Some(&prompt("stale", 0)),
                    ..RoutingContext::default()
                },
                None,
                None
            )
            .is_none());
        assert!(policy.states.is_empty() && policy.trees.is_empty());
        drop(never_selected);

        let old = worker("http://same-url");
        let rep = worker("http://same-url");
        let old_p = prompt("old history", 0);
        let rep_p = prompt("new history", 0);
        let (_, mut old_seed) = select(&policy, std::slice::from_ref(&old), &old_p, false, false);
        old_seed.commit_without_sample();
        let (_, mut old_t) = select(
            &policy,
            std::slice::from_ref(&old),
            &prompt("in flight", 0),
            false,
            false,
        );
        let (_, mut rep_t) = select(&policy, std::slice::from_ref(&rep), &rep_p, false, false);
        rep_t.commit_without_sample();
        let (_, rep_active) = select(
            &policy,
            std::slice::from_ref(&rep),
            &prompt("active pending", 0),
            false,
            false,
        );

        policy.on_worker_removed(&old);
        assert_eq!(matched_chars(&policy, &old, &old_p), 0);
        assert!(!policy
            .trees
            .get(rep.model_id())
            .unwrap()
            .tenant_char_count
            .contains_key(SMetricStrategy::tenant_key(&old).as_str()));
        assert_eq!(matched_chars(&policy, &rep, &rep_p), 11);
        assert_eq!(pending(&policy, &rep), 4);
        old_t.on_first_token();
        assert_eq!(pending(&policy, &rep), 4);
        drop(rep_active);
        assert_eq!(pending(&policy, &rep), 0);

        for i in 0..8 {
            let tmp = worker(&format!("http://tmp-{i}"));
            policy.on_worker_removed(&tmp);
            assert_eq!(policy.retired.read().len(), 2);
        }
        assert_eq!(
            policy.balance(std::slice::from_ref(&old), "/v1/embeddings", None),
            None
        );
        policy.on_worker_removed(&rep);
        assert!(policy.states.is_empty() && policy.trees.is_empty());
    }

    #[test]
    fn single_model_tree_isolates_scopes_without_growing_tree_map() {
        let policy = SMetricStrategy::new(test_config(20.0, None));
        let w = worker("http://worker:1");
        let ws = std::slice::from_ref(&w);
        let shared = prompt("shared prefix across requests", 20);
        let (_, mut t1) = select(&policy, ws, &shared, true, true);
        assert_eq!(t1.work, 8);
        t1.on_first_token();
        assert_eq!(select(&policy, ws, &shared, true, true).1.work, 0);

        let scope_a = RoutingPrompt {
            text: shared.text.clone(),
            est_hit_chars: 20,
            cache_identity: CacheIdentity::Scoped("a".repeat(64)),
        };
        let (_, mut t3) = select(&policy, ws, &scope_a, true, true);
        assert_eq!(t3.work, 8);
        t3.on_first_token();
        assert_eq!(select(&policy, ws, &scope_a, true, true).1.work, 0);

        let scope_b = RoutingPrompt {
            text: shared.text.clone(),
            est_hit_chars: 20,
            cache_identity: CacheIdentity::Scoped("b".repeat(64)),
        };
        assert_eq!(select(&policy, ws, &scope_b, true, true).1.work, 8);
        // All scopes share one per-model Tree (bounded by MAX_TREE_SIZE, no per-salt map growth).
        assert_eq!(policy.trees.len(), 1);

        let unknown = RoutingPrompt {
            text: shared.text.clone(),
            est_hit_chars: 20,
            cache_identity: CacheIdentity::Unknown,
        };
        let (_, mut t6) = select(&policy, ws, &unknown, true, true);
        t6.on_first_token();
        assert_eq!(select(&policy, ws, &unknown, true, true).1.work, 8);

        policy.on_worker_removed(&w);
        assert!(policy.trees.is_empty());
    }

    #[test]
    fn affinity_feasible_migration_and_equal_hit_tie_break() {
        let policy = SMetricStrategy::new(test_config(20.0, Some(1.0)));
        let ws = workers(&["http://one", "http://two", "http://three"]);
        let (idx0, mut t0) = select(&policy, &ws[..2], &prompt("shared", 0), false, false);
        assert_eq!(idx0, 0);
        t0.on_first_token();

        let first = policy.track(&ws[0], 100);
        let (idx1, mut t1) = select(&policy, &ws[..2], &prompt("shared plus", 6), true, false);
        assert_eq!(idx1, 1);
        t1.on_first_token();

        // Both exceed TTFT: worker 1 keeps affinity (longest history).
        let second = policy.track(&ws[1], 1000);
        assert_eq!(
            select(
                &policy,
                &ws[..2],
                &prompt("shared plus more", 11),
                true,
                false
            )
            .0,
            1
        );

        // When worker 0 & 1 are infeasible while worker 2 is feasible (even with higher load),
        // `min` must pick feasible worker 2 instead of infeasible worker 0.
        ws[2].increment_load();
        ws[2].increment_load();
        assert_eq!(
            select(&policy, &ws, &prompt("shared plus more", 11), true, true).0,
            2
        );
        ws[2].decrement_load();
        ws[2].decrement_load();
        drop(first);
        drop(second);

        // Equal-hit tie-breaking in `prev`: lower `queued` wins regardless of round-robin start.
        let p_eq = prompt("shared", 6);
        let (_, mut seed1) = select(&policy, std::slice::from_ref(&ws[1]), &p_eq, false, false);
        seed1.on_first_token();
        let busy0 = policy.track(&ws[0], 10);
        for _ in 0..2 {
            assert_eq!(select(&policy, &ws[..2], &p_eq, true, false).0, 1);
        }
        drop(busy0);
        assert_eq!((pending(&policy, &ws[0]), pending(&policy, &ws[1])), (0, 0));
    }

    #[test]
    fn colocated_score_counts_inflight_even_on_zero_cost_hit_while_pd_ignores_inflight() {
        let config = test_config(10.0, Some(1.0));
        let ws = workers(&["http://one", "http://two"]);
        for _ in 0..3 {
            ws[0].increment_load();
        }
        ws[1].increment_load();
        let p = prompt("first turn", 0);
        assert_eq!(
            select(&SMetricStrategy::new(config.clone()), &ws, &p, false, true).0,
            1
        );
        assert_eq!(
            select(&SMetricStrategy::new(config.clone()), &ws, &p, false, false).0,
            0
        );

        // Full cache hit (`cost == 0`) on heavily loaded worker 0 must NOT zero out inflight load.
        let policy = SMetricStrategy::new(config);
        let (_, mut seed) = select(&policy, std::slice::from_ref(&ws[0]), &p, false, true);
        seed.on_first_token();
        for _ in 0..7 {
            ws[0].increment_load();
        }
        assert_eq!(select(&policy, &ws, &p, false, true).0, 1);
    }

    #[test]
    fn learned_rate_p90_and_config_override() {
        let config = test_config(20.0, None);
        let ws = workers(&["http://one", "http://two"]);
        let dynamic = SMetricStrategy::new(config.clone());
        let (i0, mut t0) = select(&dynamic, &ws, &prompt("shared", 0), false, false);
        assert_eq!(i0, 0);
        t0.on_first_token();
        let queued = dynamic.track(&ws[0], 100);
        let follow_up = prompt("shared plus", 6);
        assert_eq!(select(&dynamic, &ws, &follow_up, true, false).0, 0);
        {
            let h = dynamic
                .states
                .get(&SMetricStrategy::instance_id(&ws[0]))
                .unwrap();
            for _ in 0..RATE_MIN_SAMPLES {
                h.rates.lock().record(1.0);
            }
        }
        assert_eq!(select(&dynamic, &ws, &follow_up, true, false).0, 1);
        drop(queued);

        let learned = SMetricStrategy::new(config.clone());
        for i in 0..RATE_MIN_SAMPLES {
            let other = learned.track(&ws[0], 13);
            let (_, mut t) = select(
                &learned,
                &ws[..1],
                &prompt(&format!("prefill-{i}"), 0),
                false,
                true,
            );
            assert!(pending(&learned, &ws[0]) > 13);
            t.backdate_for_test(Duration::from_millis(1));
            t.on_first_token();
            t.on_first_token();
            assert_eq!(pending(&learned, &ws[0]), 13);
            drop(other);
            assert_eq!(
                learned
                    .states
                    .get(&SMetricStrategy::instance_id(&ws[0]))
                    .unwrap()
                    .rates
                    .lock()
                    .rate
                    .is_some(),
                i + 1 == RATE_MIN_SAMPLES
            );
        }

        let mut history = RateHistory::default();
        for &r in &[100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0] {
            history.record(r);
        }
        assert_eq!(history.rate, Some(700.0));

        let fixed = SMetricStrategy::new(SMetricConfig {
            prefill_rate: Some(1000.0),
            ..config
        });
        let (_, mut ft) = select(&fixed, &ws[..1], &prompt("fixed prefill", 0), false, true);
        ft.backdate_for_test(Duration::from_millis(1));
        ft.on_first_token();
        assert!(fixed
            .states
            .get(&SMetricStrategy::instance_id(&ws[0]))
            .unwrap()
            .rates
            .lock()
            .samples
            .is_empty());
    }

    #[test]
    fn token_id_kv_events_mixed_pool_cjk_units_cold_completion_and_crossover() {
        let policy = SMetricStrategy::new(test_config(20.0, Some(1000.0)));
        let ws = workers(&["grpc://bound", "http://tree-only"]);
        let tokens: Vec<u32> = (1..=100).collect();

        // 1. Token-ID request (`prompt: None`) with bound KV events routes via KV affinity.
        ws[0].increment_load();
        let mm_ctx = RoutingContext {
            prompt: None,
            kv: Some(KvRequestContext {
                token_ids: &tokens,
                lora_name: None,
                cache_salt: None,
            }),
            allow_affinity: true,
            colocated: true,
            route: "/v1/completions",
        };
        let (picked, tracker, _) = policy
            .select_request(
                &ws,
                &mm_ctx,
                Some(&[PrefixMatch::Bound(80), PrefixMatch::Bound(0)]),
                None,
            )
            .unwrap();
        assert_eq!((picked, tracker.unwrap().work), (0, 20));

        // Salted/unavailable token-ID request still enters `select_prefill` (tracks pending work).
        let (_, salted_tracker, _) = policy
            .select_request(
                &ws,
                &mm_ctx,
                Some(&[PrefixMatch::Unavailable, PrefixMatch::Unavailable]),
                None,
            )
            .unwrap();
        assert_eq!(salted_tracker.unwrap().work, 100);
        ws[0].decrement_load();

        // 2. Mixed pool with dense CJK text (~1 char/token): unified `chars_to_tokens` keeps 100% hit.
        let cjk = prompt(&"中".repeat(100), 100);
        let (_, mut seed) = select(&policy, std::slice::from_ref(&ws[1]), &cjk, false, true);
        seed.on_first_token();
        ws[1].increment_load();
        let mixed_ctx = RoutingContext {
            prompt: Some(&cjk),
            kv: Some(KvRequestContext {
                token_ids: &tokens,
                lora_name: None,
                cache_salt: None,
            }),
            allow_affinity: true,
            colocated: true,
            route: "/v1/chat/completions",
        };
        assert_eq!(
            policy
                .select_request(
                    &ws,
                    &mixed_ctx,
                    Some(&[PrefixMatch::Bound(0), PrefixMatch::Unmanaged]),
                    None
                )
                .unwrap()
                .0,
            1
        );

        // 3. Cold first-turn `/v1/completions` (`hit == 0`) fails `hit > HIT_RATIO * history` and balances.
        assert_eq!(
            select(
                &policy,
                &ws,
                &prompt("brand new completion prompt", 27),
                true,
                true
            )
            .0,
            0
        );
        ws[1].decrement_load();

        // 4. ATTENTION_CROSSOVER_TOKENS (0 vs 16384, cold & partial hit) + per-1K SLO scaling.
        let quad = SMetricStrategy::new(SMetricConfig {
            ttft_slo_secs: 1.0,
            ttft_slo_secs_per_1k_tokens: 0.5,
            attention_crossover_tokens: 16_384.0,
            ..test_config(1.0, Some(1000.0))
        });
        assert_eq!(policy.prefill_cost(16_384.0, 0.0), 16_384.0);
        assert_eq!(quad.prefill_cost(16_384.0, 0.0), 24_576.0);
        assert_eq!(quad.prefill_cost(16_384.0, 8_192.0), 14_336.0);
        assert!((quad.ttft_budget_secs(4_000.0) - 3.0).abs() < 1e-9);
    }
}
