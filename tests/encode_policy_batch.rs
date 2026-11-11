//! Encode assignment must preserve batching without inflating consumer metrics.
use metrics::{
    Counter, CounterFn, Gauge, Histogram, Key, KeyName, Metadata, Recorder, SharedString, Unit,
};
use std::collections::HashMap;
use std::sync::{Arc, Barrier, Mutex};
use vllm_router_rs::{
    config::PolicyConfig,
    core::{BasicWorker, CircuitBreakerConfig, Worker, WorkerType},
    policies::{
        LoadBalancingPolicy, PolicyFactory, PolicyRegistry, RandomPolicy, RoundRobinPolicy,
        SelectionContext,
    },
};

fn assign(
    policy: &dyn LoadBalancingPolicy,
    context: &SelectionContext,
    pool: &[Arc<dyn Worker>],
    count: usize,
) -> Option<Vec<usize>> {
    context.with_selection(|| {
        (0..count)
            .map(|_| policy.select_worker_with_headers(pool, None, None))
            .collect()
    })
}

fn supported_configs() -> Vec<PolicyConfig> {
    vec![
        PolicyConfig::RoundRobin,
        PolicyConfig::Random,
        PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 73,
        },
        PolicyConfig::ConsistentHash { virtual_nodes: 317 },
        PolicyConfig::RendezvousHash,
    ]
}

fn workers() -> Vec<Arc<dyn Worker>> {
    (0..3)
        .map(|index| {
            Arc::new(
                BasicWorker::new(format!("http://e{index}"), WorkerType::Encode)
                    .with_circuit_breaker_config(CircuitBreakerConfig {
                        failure_threshold: 1,
                        ..Default::default()
                    }),
            ) as Arc<dyn Worker>
        })
        .collect()
}

#[test]
fn round_robin_context_keeps_a_contiguous_sequence_per_concurrent_request() {
    let pool = workers();
    let policy = Arc::new(RoundRobinPolicy::new());
    let context = Arc::new(SelectionContext::default());
    let barrier = Arc::new(Barrier::new(32));
    let threads: Vec<_> = (0..32)
        .map(|_| {
            let (pool, policy, barrier, context) = (
                pool.clone(),
                policy.clone(),
                barrier.clone(),
                context.clone(),
            );
            std::thread::spawn(move || {
                barrier.wait();
                assign(policy.as_ref(), &context, &pool, 4).unwrap()
            })
        })
        .collect();
    let mut counts = [0; 3];
    for thread in threads {
        let assignment = thread.join().unwrap();
        for pair in assignment.windows(2) {
            assert_eq!(pair[1], (pair[0] + 1) % 3);
        }
        for index in assignment {
            counts[index] += 1;
        }
    }
    assert_eq!(counts, [43, 43, 42]);
    assert_eq!(
        assign(policy.as_ref(), &context, &pool, 3),
        Some(vec![2, 0, 1])
    );
}

#[test]
fn batch_policies_exclude_unhealthy_and_open_circuit_workers() {
    for config in supported_configs() {
        let policy = PolicyFactory::create_from_config(&config);
        let context = SelectionContext::default();
        let pool = workers();
        pool[0].set_healthy(false);
        pool[1].record_outcome(false);
        assert_eq!(
            assign(policy.as_ref(), &context, &pool, 8),
            Some(vec![2; 8])
        );
        pool[2].set_healthy(false);
        assert!(assign(policy.as_ref(), &context, &pool, 1).is_none());
        assert_eq!(assign(policy.as_ref(), &context, &pool, 0), Some(vec![]));
        assert!(assign(policy.as_ref(), &context, &[], 1).is_none());
    }
}

#[test]
fn empty_assignment_does_not_advance_round_robin() {
    let policy = RoundRobinPolicy::new();
    let context = SelectionContext::default();
    let pool = workers();
    assert_eq!(assign(&policy, &context, &pool, 1), Some(vec![0]));
    assert_eq!(assign(&policy, &context, &pool, 0), Some(vec![]));
    assert!(assign(&policy, &context, &[], 5).is_none());
    assert_eq!(assign(&policy, &context, &pool, 2), Some(vec![1, 2]));
}

#[test]
fn random_assignments_remain_within_the_supplied_encode_pool() {
    let pool = workers();
    let context = SelectionContext::default();
    let policy = RandomPolicy::new();
    let assignment = assign(&policy, &context, &pool, 128).unwrap();
    assert_eq!(assignment.len(), 128);
    assert!(assignment.iter().all(|index| *index < pool.len()));
}

type Counts = Arc<Mutex<HashMap<Key, u64>>>;
struct CapturingCounter {
    key: Key,
    counts: Counts,
}

impl CounterFn for CapturingCounter {
    fn increment(&self, value: u64) {
        *self
            .counts
            .lock()
            .unwrap()
            .entry(self.key.clone())
            .or_default() += value;
    }
    fn absolute(&self, value: u64) {
        let mut counts = self.counts.lock().unwrap();
        let count = counts.entry(self.key.clone()).or_default();
        *count = (*count).max(value);
    }
}

struct CapturingRecorder(Counts);
impl Recorder for CapturingRecorder {
    fn describe_counter(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn describe_gauge(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn describe_histogram(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn register_counter(&self, key: &Key, _: &Metadata<'_>) -> Counter {
        Counter::from_arc(Arc::new(CapturingCounter {
            key: key.clone(),
            counts: self.0.clone(),
        }))
    }
    fn register_gauge(&self, _: &Key, _: &Metadata<'_>) -> Gauge {
        Gauge::noop()
    }
    fn register_histogram(&self, _: &Key, _: &Metadata<'_>) -> Histogram {
        Histogram::noop()
    }
}

#[test]
fn encode_item_assignment_does_not_emit_consumer_processed_or_policy_metrics() {
    for config in supported_configs() {
        let counts = Counts::default();
        let recorder = CapturingRecorder(counts.clone());
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
        registry.set_encode_policy(&config).unwrap();
        let encoders = workers();
        let consumers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
            "http://consumer".into(),
            WorkerType::Regular,
        ))];
        metrics::with_local_recorder(&recorder, || {
            for _ in 0..3 {
                assign(
                    registry.get_encode_policy().unwrap().as_ref(),
                    &registry.get_encode_selection_context(),
                    &encoders,
                    5,
                )
                .unwrap();
            }
            assert!(
                counts.lock().unwrap().is_empty(),
                "Encode item choices emitted consumer counters"
            );
            registry
                .get_default_policy()
                .select_worker(&consumers, None)
                .unwrap();
            registry
                .get_default_policy()
                .select_worker(&consumers, None)
                .unwrap();
        });
        let counts = counts.lock().unwrap();
        assert_eq!(counts.len(), 2);
        for name in [
            "vllm_router_processed_requests_total",
            "vllm_router_policy_decisions_total",
        ] {
            let (key, count) = counts.iter().find(|(key, _)| key.name() == name).unwrap();
            assert_eq!(*count, 2);
            assert!(key
                .labels()
                .any(|label| label.key() == "worker" && label.value() == "http://consumer"));
        }
        assert!(encoders
            .iter()
            .all(|worker| worker.load() == 0 && worker.processed_requests() == 0));
    }
}

#[test]
fn incompatible_roles_are_rejected_before_selection_side_effects() {
    use vllm_router_rs::policies::{CacheAwareConfig, CacheAwarePolicy, KvAwarePolicy};
    let pool = workers();
    let counts = Counts::default();
    let recorder = CapturingRecorder(counts.clone());
    let cache = CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        ..Default::default()
    });
    let kv = KvAwarePolicy::default();
    metrics::with_local_recorder(&recorder, || {
        for name in ["cache_aware", "kv_aware"] {
            let config = name.parse::<PolicyConfig>().unwrap();
            let error = PolicyFactory::validate_role(&config, WorkerType::Encode).unwrap_err();
            assert_eq!(error.policy, name);
            assert_eq!(error.role, WorkerType::Encode);
            assert!(!error.reason.is_empty());
        }
        for policy in [&cache as &dyn LoadBalancingPolicy, &kv] {
            assert!(policy
                .select_worker_with_headers(&pool, Some("affinity-key"), None)
                .is_none());
            assert!(policy
                .select_and_reserve(
                    &SelectionContext::default(),
                    &pool,
                    Some("affinity-key"),
                    None
                )
                .is_none());
        }
    });
    assert!(counts.lock().unwrap().is_empty());
    assert!(pool
        .iter()
        .all(|w| w.load() == 0 && w.processed_requests() == 0));
}

#[test]
fn consumer_selection_keeps_each_policies_existing_accounting() {
    for config in supported_configs() {
        for size in [1, 2] {
            let policy = PolicyFactory::create_from_config(&config);
            let pool: Vec<Arc<dyn Worker>> = (0..size)
                .map(|i| {
                    Arc::new(BasicWorker::new(
                        format!("http://consumer-{i}"),
                        WorkerType::Regular,
                    )) as Arc<dyn Worker>
                })
                .collect();
            let counts = Counts::default();
            let recorder = CapturingRecorder(counts.clone());
            metrics::with_local_recorder(&recorder, || {
                assert!(policy
                    .select_worker_with_headers(&pool, Some("session"), None)
                    .is_some());
            });
            let processed: usize = pool.iter().map(|w| w.processed_requests()).sum();
            let fast_p2c = matches!(config, PolicyConfig::PowerOfTwo { .. }) && size == 1;
            let expected_processed =
                !matches!(config, PolicyConfig::Random | PolicyConfig::RoundRobin) && !fast_p2c;
            assert_eq!(
                processed,
                usize::from(expected_processed),
                "{} size={size}",
                config.name()
            );
            let counts = counts.lock().unwrap();
            assert_eq!(counts.len(), if fast_p2c { 0 } else { 2 });
            assert!(counts.values().all(|count| *count == 1));
        }
    }
}
