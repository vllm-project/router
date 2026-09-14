//! Gauge semantics of `CachedTokenizer`.
//!
//! `vllm_tokenizer_cache_entries` and `vllm_tokenizer_cache_bytes` must equal
//! the total occupancy of every live cache in the process, including after
//! concurrent inserts and clears and after instances are dropped. A capturing
//! `metrics::Recorder` records what the cache publishes.
//!
//! All caches in a process feed one aggregate, so these tests serialize on a
//! lock and drop every cache they create before releasing it.

use metrics::{
    Counter, Gauge, GaugeFn, Histogram, Key, KeyName, Metadata, Recorder, SharedString, Unit,
};
use parking_lot::{Condvar, Mutex};
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use vllm_router_rs::tokenizer::{
    mock::MockTokenizer, traits::Encoder, CachedTokenizer, TokenizerCacheConfig,
};

static SERIAL: Mutex<()> = Mutex::new(());

const ENTRIES: &str = "vllm_tokenizer_cache_entries";
const BYTES: &str = "vllm_tokenizer_cache_bytes";

#[derive(Default)]
struct Captured {
    values: Mutex<HashMap<String, f64>>,
    /// When set, the next gauge `set` call takes this flag and blocks until
    /// `release`. Later calls are unaffected.
    block_next_set: Mutex<bool>,
    released: Mutex<bool>,
    release_signal: Condvar,
}

impl Captured {
    fn gauges(&self) -> (f64, f64) {
        let values = self.values.lock();
        (
            values.get(ENTRIES).copied().unwrap_or(0.0),
            values.get(BYTES).copied().unwrap_or(0.0),
        )
    }

    /// Make the next gauge `set` block until `release` is called.
    fn block_next_set(&self) {
        *self.block_next_set.lock() = true;
    }

    fn release(&self) {
        *self.released.lock() = true;
        self.release_signal.notify_all();
    }
}

struct CapturingGauge {
    captured: Arc<Captured>,
    name: String,
}

impl GaugeFn for CapturingGauge {
    fn increment(&self, value: f64) {
        *self
            .captured
            .values
            .lock()
            .entry(self.name.clone())
            .or_default() += value;
    }

    fn decrement(&self, value: f64) {
        *self
            .captured
            .values
            .lock()
            .entry(self.name.clone())
            .or_default() -= value;
    }

    fn set(&self, value: f64) {
        let blocked = std::mem::take(&mut *self.captured.block_next_set.lock());
        if blocked {
            let mut released = self.captured.released.lock();
            while !*released {
                self.captured.release_signal.wait(&mut released);
            }
        }
        self.captured.values.lock().insert(self.name.clone(), value);
    }
}

struct CapturingRecorder(Arc<Captured>);

impl Recorder for CapturingRecorder {
    fn describe_counter(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn describe_gauge(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn describe_histogram(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}

    fn register_counter(&self, _: &Key, _: &Metadata<'_>) -> Counter {
        Counter::noop()
    }

    fn register_gauge(&self, key: &Key, _: &Metadata<'_>) -> Gauge {
        Gauge::from_arc(Arc::new(CapturingGauge {
            captured: self.0.clone(),
            name: key.name().to_string(),
        }))
    }

    fn register_histogram(&self, _: &Key, _: &Metadata<'_>) -> Histogram {
        Histogram::noop()
    }
}

fn cache(max_entries: usize) -> CachedTokenizer {
    CachedTokenizer::new(
        Arc::new(MockTokenizer::new()),
        TokenizerCacheConfig {
            max_entries,
            ..TokenizerCacheConfig::default()
        },
    )
    .unwrap()
}

fn expected(caches: &[&CachedTokenizer]) -> (f64, f64) {
    let entries: usize = caches.iter().map(|c| c.stats().entries).sum();
    let bytes: usize = caches.iter().map(|c| c.stats().bytes).sum();
    (entries as f64, bytes as f64)
}

#[test]
fn gauges_sum_over_live_instances() {
    let _serial = SERIAL.lock();
    let captured = Arc::new(Captured::default());
    let recorder = CapturingRecorder(captured.clone());

    metrics::with_local_recorder(&recorder, || {
        let a = cache(10);
        let b = cache(10);
        a.encode("Hello").unwrap();
        a.encode("world").unwrap();
        b.encode("Hello").unwrap();
        assert_eq!(captured.gauges(), (3.0, expected(&[&a, &b]).1));

        // Pure hits do not touch the gauges.
        a.encode("Hello").unwrap();
        b.encode("Hello").unwrap();
        assert_eq!(captured.gauges(), expected(&[&a, &b]));

        drop(b);
        assert_eq!(captured.gauges(), expected(&[&a]));

        a.clear();
        assert_eq!(captured.gauges(), (0.0, 0.0));

        a.encode("test").unwrap();
        assert_eq!(captured.gauges(), expected(&[&a]));

        drop(a);
        assert_eq!(captured.gauges(), (0.0, 0.0));
    });
}

#[test]
fn eviction_and_replacement_keep_gauges_exact() {
    let _serial = SERIAL.lock();
    let captured = Arc::new(Captured::default());
    let recorder = CapturingRecorder(captured.clone());

    metrics::with_local_recorder(&recorder, || {
        let a = cache(2);
        for input in ["Hello", "world", "test", "token"] {
            a.encode(input).unwrap();
            assert_eq!(captured.gauges(), expected(&[&a]));
        }
        assert_eq!(a.stats().evictions, 2);

        // Room for two mock entries but not three, so the byte budget evicts.
        let per_entry = a.stats().bytes / 2;
        let b = CachedTokenizer::new(
            Arc::new(MockTokenizer::new()),
            TokenizerCacheConfig {
                max_entries: 100,
                max_bytes: 2 * per_entry + per_entry / 2,
                max_entry_bytes: 2 * per_entry + per_entry / 2,
            },
        )
        .unwrap();
        for input in ["Hello", "world", "test"] {
            b.encode(input).unwrap();
            assert_eq!(captured.gauges(), expected(&[&a, &b]));
        }
        assert!(b.stats().evictions > 0);

        drop(a);
        drop(b);
        assert_eq!(captured.gauges(), (0.0, 0.0));
    });
}

/// A publication that stalls inside the recorder must not be overtaken by a
/// later state change: the stalled operation still holds the cache lock, so
/// the other operation and its publication wait behind it. Only the first
/// `set` is blocked, so the second operation publishes freely; that is
/// exactly the interleaving that left a stale value when publication was
/// not ordered with the state change.
#[test]
fn delayed_publish_cannot_overwrite_a_later_clear() {
    let _serial = SERIAL.lock();
    let captured = Arc::new(Captured::default());
    let recorder = Arc::new(CapturingRecorder(captured.clone()));
    let shared = Arc::new(cache(10));

    captured.block_next_set();
    let inserter = {
        let (recorder, shared) = (recorder.clone(), shared.clone());
        thread::spawn(move || {
            metrics::with_local_recorder(&*recorder, || shared.encode("Hello").unwrap());
        })
    };
    // Let the insert reach the recorder and stall there.
    thread::sleep(Duration::from_millis(50));
    let clearer = {
        let (recorder, shared) = (recorder.clone(), shared.clone());
        thread::spawn(move || metrics::with_local_recorder(&*recorder, || shared.clear()))
    };
    thread::sleep(Duration::from_millis(50));
    captured.release();
    inserter.join().unwrap();
    clearer.join().unwrap();

    let stats = shared.stats();
    assert_eq!(
        captured.gauges(),
        (stats.entries as f64, stats.bytes as f64),
        "gauges must reflect the final state whichever operation ran last"
    );

    metrics::with_local_recorder(&*recorder, || drop(shared));
    assert_eq!(captured.gauges(), (0.0, 0.0));
}

#[test]
fn concurrent_inserts_and_clears_end_consistent() {
    let _serial = SERIAL.lock();
    let captured = Arc::new(Captured::default());
    let recorder = Arc::new(CapturingRecorder(captured.clone()));
    let a = Arc::new(cache(4));
    let b = Arc::new(cache(3));
    let inputs: Vec<String> = (0..16).map(|i| format!("Hello world {i}")).collect();

    let handles: Vec<_> = (0..8)
        .map(|t| {
            let (recorder, a, b, inputs) = (recorder.clone(), a.clone(), b.clone(), inputs.clone());
            thread::spawn(move || {
                metrics::with_local_recorder(&*recorder, || {
                    for i in 0..300 {
                        let target = if (i + t) % 2 == 0 { &a } else { &b };
                        if i % 37 == 0 {
                            target.clear();
                        } else {
                            target.encode(&inputs[(i * 7 + t) % inputs.len()]).unwrap();
                        }
                    }
                })
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(captured.gauges(), expected(&[&a, &b]));
    assert!(a.stats().evictions > 0 && b.stats().evictions > 0);

    metrics::with_local_recorder(&*recorder, || {
        drop(a);
        drop(b);
    });
    assert_eq!(captured.gauges(), (0.0, 0.0));
}
