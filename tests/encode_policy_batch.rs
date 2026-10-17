//! Encode assignment must preserve batching without inflating consumer metrics.
use metrics::{
    Counter, CounterFn, Gauge, Histogram, Key, KeyName, Metadata, Recorder, SharedString, Unit,
};
use std::collections::HashMap;
use std::sync::{Arc, Barrier, Mutex};
use vllm_router_rs::{
    config::{EncodePolicy, PolicyConfig},
    core::{BasicWorker, CircuitBreakerConfig, Worker, WorkerType},
    policies::{LoadBalancingPolicy, PolicyRegistry, RandomPolicy, RoundRobinPolicy},
};

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
fn round_robin_reserves_a_contiguous_sequence_per_concurrent_request() {
    let pool = workers();
    let policy = Arc::new(RoundRobinPolicy::new());
    let barrier = Arc::new(Barrier::new(32));
    let threads: Vec<_> = (0..32)
        .map(|_| {
            let (pool, policy, barrier) = (pool.clone(), policy.clone(), barrier.clone());
            std::thread::spawn(move || {
                barrier.wait();
                policy.select_encode_batch(&pool, 4).unwrap()
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
    assert_eq!(policy.select_encode_batch(&pool, 3), Some(vec![2, 0, 1]));
}

#[test]
fn batch_policies_exclude_unhealthy_and_open_circuit_workers() {
    for policy in [
        Box::new(RoundRobinPolicy::new()) as Box<dyn LoadBalancingPolicy>,
        Box::new(RandomPolicy::new()),
    ] {
        let pool = workers();
        pool[0].set_healthy(false);
        pool[1].record_outcome(false);
        assert_eq!(policy.select_encode_batch(&pool, 8), Some(vec![2; 8]));
        pool[2].set_healthy(false);
        assert!(policy.select_encode_batch(&pool, 1).is_none());
        assert_eq!(policy.select_encode_batch(&pool, 0), Some(vec![]));
        assert!(policy.select_encode_batch(&[], 1).is_none());
    }
}

#[test]
fn empty_assignment_does_not_advance_round_robin() {
    let policy = RoundRobinPolicy::new();
    let pool = workers();
    assert_eq!(policy.select_encode_batch(&pool, 1), Some(vec![0]));
    assert_eq!(policy.select_encode_batch(&pool, 0), Some(vec![]));
    assert!(policy.select_encode_batch(&[], 5).is_none());
    assert_eq!(policy.select_encode_batch(&pool, 2), Some(vec![1, 2]));
}

#[test]
fn random_assignments_remain_within_the_supplied_encode_pool() {
    let pool = workers();
    let policy = RandomPolicy::new();
    let assignment = policy.select_encode_batch(&pool, 128).unwrap();
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
    for config in [EncodePolicy::RoundRobin, EncodePolicy::Random] {
        let counts = Counts::default();
        let recorder = CapturingRecorder(counts.clone());
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
        registry.set_encode_policy(config);
        let encoders = workers();
        let consumers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
            "http://consumer".into(),
            WorkerType::Regular,
        ))];
        metrics::with_local_recorder(&recorder, || {
            for _ in 0..3 {
                registry
                    .get_encode_policy()
                    .select_encode_batch(&encoders, 5)
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
