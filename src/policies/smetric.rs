//! SMetric: session cache affinity with a prefill-work fallback.
use super::{KvEventConsumer, KvRequestContext, TieredMatchProvider};
use super::{LoadBalancingPolicy, RequestHeaders, RequestObserver, RequestTracker, RoutingContext};
use crate::kv_index::WorkerKvInfo;
mod kv;
use crate::config::SMetricConfig;
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::protocols::spec::RoutingPrompt;
use crate::tree::Tree;
use dashmap::DashMap;
use kv::KvResidency;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Weak};
use std::time::Duration;
use tracing::{debug, warn};

#[derive(Debug)]
pub struct SMetricPolicy {
    config: SMetricConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>,
    kv: KvResidency,
    round_robin: AtomicUsize,
    states: DashMap<usize, Arc<WorkerState>>,
    retired: RwLock<HashMap<usize, Weak<dyn Worker>>>,
    fallback_warnings: AtomicU8,
    eviction_shutdown: Option<mpsc::Sender<()>>,
    eviction_handle: Option<std::thread::JoinHandle<()>>,
}

impl SMetricPolicy {
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
            kv: KvResidency::default(),
            states: DashMap::new(),
            retired: RwLock::new(HashMap::new()),
            round_robin: AtomicUsize::new(0),
            fallback_warnings: AtomicU8::new(0),
            eviction_shutdown,
            eviction_handle,
        }
    }

    fn select_prefill(
        &self,
        workers: &[Arc<dyn Worker>],
        prompt: &RoutingPrompt,
        passes_turn_gate: bool,
        colocated: bool,
        kv_request: Option<&KvRequestContext<'_>>,
    ) -> Option<(usize, RequestTracker)> {
        let retired = self.retired.read();
        let l = prompt.text.chars().count();
        let event_hits = kv_request.map(|request| self.kv.prefix_matches(workers, request, l));
        let start = self.round_robin.fetch_add(1, Ordering::Relaxed) % workers.len().max(1);
        let mut min: Option<(usize, u64, u64)> = None; // index, score, own cost
        let mut prev: Option<(usize, usize, u64, bool)> = None; // index, hit, own cost, TTFT
        let mut any_meets_ttft = false;
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
            let n = (l - hit) as f64;
            let cost = (self.config.c_lin * n + self.config.c_att * n * (l as f64 - n / 2.0)).ceil()
                as u64;
            let state = self.states.get(&Self::instance_id(worker));
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
        if !prompt.text.is_empty() && event_hits.is_none() {
            self.trees
                .entry(workers[idx].model_id().to_string())
                .or_insert_with(|| Arc::new(Tree::new()))
                .insert(&prompt.text, &Self::tenant_key(&workers[idx]));
        }
        RouterMetrics::record_processed_request(workers[idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[idx].url());
        let tracker = self.track(&workers[idx], cost);
        let tracker = if event_hits.as_ref().is_some_and(|hits| !hits[idx].observed) {
            tracker.without_rate_sample()
        } else {
            tracker
        };
        Some((idx, tracker))
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
        RouterMetrics::record_policy_decision(self.name(), workers[idx].url());
        Some(idx)
    }
}

impl Drop for SMetricPolicy {
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

impl KvEventConsumer for SMetricPolicy {
    fn bind_worker(
        &self,
        worker: &dyn Worker,
        info: &WorkerKvInfo,
        matches: Option<Arc<dyn TieredMatchProvider>>,
    ) {
        let id = worker as *const dyn Worker as *const () as usize;
        let retired = self.retired.read();
        if !retired.contains_key(&id) {
            self.kv.register(id, worker, info, matches);
        }
    }
}

impl LoadBalancingPolicy for SMetricPolicy {
    fn on_worker_removed(&self, worker: &Arc<dyn Worker>) {
        let mut retired = self.retired.write();
        retired.retain(|_, worker| worker.strong_count() > 0);
        retired.insert(Self::instance_id(worker), Arc::downgrade(worker));
        self.kv.remove(Self::instance_id(worker));
        self.states.remove(&Self::instance_id(worker));
        self.trees.remove_if(worker.model_id(), |_, tree| {
            tree.remove_tenant(&Self::tenant_key(worker));
            tree.tenant_char_count.is_empty()
        });
    }

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
            self.select_prefill(
                workers,
                prompt,
                context.allow_affinity,
                context.colocated,
                context.kv.as_ref(),
            )
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
                        .select_request(&workers, Some("private fallback payload"), None, &context)
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
        let enabled = SMetricPolicy::new(
            serde_yaml::from_str(&format!("{yaml}EVICTION_INTERVAL_SECS: 1\n")).unwrap(),
        );
        let disabled = SMetricPolicy::new(
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
            .prefix_match_tenant_char_count("cached history", &SMetricPolicy::tenant_key(&worker))
            != 0
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            enabled_tree.prefix_match_tenant_char_count(
                "cached history",
                &SMetricPolicy::tenant_key(&worker)
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
                    &SMetricPolicy::tenant_key(&worker)
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

    fn removal_test_policy() -> SMetricPolicy {
        SMetricPolicy::new(serde_yaml::from_str(
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
            .contains_key(&SMetricPolicy::instance_id(&replacement)));
        let replacement_state = policy
            .states
            .get(&SMetricPolicy::instance_id(&replacement))
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
                    &SMetricPolicy::tenant_key(&replacement)
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
            .contains_key(&SMetricPolicy::instance_id(&first)));
        assert!(policy
            .states
            .contains_key(&SMetricPolicy::instance_id(&second)));
        let tree = policy.trees.get(first.model_id()).unwrap();
        assert_eq!(
            tree.prefix_match_tenant_char_count(
                "first history",
                &SMetricPolicy::tenant_key(&first)
            ),
            0
        );
        assert!(!tree
            .tenant_char_count
            .contains_key(SMetricPolicy::tenant_key(&first).as_str()));
        assert_eq!(
            tree.prefix_match_tenant_char_count(
                "second history",
                &SMetricPolicy::tenant_key(&second)
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
            .get(&SMetricPolicy::instance_id(&worker))
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
            .get(&SMetricPolicy::instance_id(&replacement))
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
        policy: &SMetricPolicy,
        workers: &[Arc<dyn Worker>],
        prompt: &RoutingPrompt,
        turn_gate: bool,
        colocated: bool,
    ) -> (usize, RequestTracker) {
        policy
            .select_prefill(workers, prompt, turn_gate, colocated, None)
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
            eviction_interval_secs: 30,
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
                .get(&SMetricPolicy::instance_id(&workers[0]))
                .unwrap()
                .pending
                .load(Ordering::Acquire),
            0
        );
        assert_eq!(
            policy
                .states
                .get(&SMetricPolicy::instance_id(&workers[1]))
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
            eviction_interval_secs: 30,
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
        let queued = dynamic.track(&workers[0], 100);
        let follow_up = prompt("shared plus", 6);
        assert_eq!(select(&dynamic, &workers, &follow_up, true, false).0, 0);
        let rate_history = dynamic
            .states
            .get(&SMetricPolicy::instance_id(&workers[0]))
            .unwrap();
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
        let fixed_queue = fixed.track(&workers[0], 100);
        // The configured rate remains authoritative even if observed rates exist.
        fixed
            .states
            .get(&SMetricPolicy::instance_id(&workers[0]))
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
            eviction_interval_secs: 30,
        };
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorker::new("http://one".into(), WorkerType::Regular));
        let workers = vec![worker.clone()];
        let learned = SMetricPolicy::new(config.clone());
        for i in 0..RATE_MIN_SAMPLES {
            let sample = prompt(&format!("prefill-{i}"), 0);
            let other = learned.track(&worker, 13);
            let (_, mut tracker) = select(&learned, &workers, &sample, false, true);
            assert!(
                learned
                    .states
                    .get(&SMetricPolicy::instance_id(&worker))
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
                    .get(&SMetricPolicy::instance_id(&worker))
                    .unwrap()
                    .pending
                    .load(Ordering::Acquire),
                13
            );
            drop(other);
            assert_eq!(
                learned
                    .states
                    .get(&SMetricPolicy::instance_id(&worker))
                    .unwrap()
                    .pending
                    .load(Ordering::Acquire),
                0
            );
            assert_eq!(
                learned
                    .states
                    .get(&SMetricPolicy::instance_id(&worker))
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
                .get(&SMetricPolicy::instance_id(&worker))
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
                .get(&SMetricPolicy::instance_id(&worker))
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
