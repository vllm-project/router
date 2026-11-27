//! Age of the last frame received on the live SUB socket, including rejected events.
use metrics::Gauge;
use parking_lot::Mutex;
use std::time::Duration;
use tokio::time::Instant;

// Reuse labels across worker churn. The high-water mark is the peak number
// of simultaneous subscriptions rather than the number of worker URLs seen.
static SOURCE_SLOTS: Mutex<Vec<bool>> = Mutex::new(Vec::new());

pub(super) struct SourceAge {
    slot: usize,
    last_seen: Mutex<Instant>,
    gauge: Gauge,
}

impl SourceAge {
    pub(super) fn new(source: &str, rank: u32) -> Self {
        let mut slots = SOURCE_SLOTS.lock();
        let slot = slots.iter().position(|used| !*used).unwrap_or(slots.len());
        if slot == slots.len() {
            slots.push(true);
        } else {
            slots[slot] = true;
        }
        drop(slots);
        tracing::debug!(
            source,
            rank,
            metric_slot = slot,
            "assigned KV source age metric"
        );
        let gauge = metrics::gauge!("vllm_router_kv_source_last_event_age_seconds", "source" => slot.to_string());
        gauge.set(0.0);
        Self {
            slot,
            gauge,
            last_seen: Mutex::new(Instant::now()),
        }
    }

    pub(super) fn recv(&self, socket: &zmq::Socket, topic: &[u8]) -> zmq::Result<Vec<Vec<u8>>> {
        let frames = socket.recv_multipart(zmq::DONTWAIT)?;
        if frames.first().is_some_and(|f| f.as_slice() == topic) {
            *self.last_seen.lock() = Instant::now();
        }
        Ok(frames)
    }

    pub(super) async fn report(&self) {
        let mut timer = tokio::time::interval(Duration::from_secs(1));
        timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            timer.tick().await;
            self.gauge
                .set(self.last_seen.lock().elapsed().as_secs_f64());
        }
    }
}

impl Drop for SourceAge {
    fn drop(&mut self) {
        self.gauge.set(f64::NAN);
        SOURCE_SLOTS.lock()[self.slot] = false;
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::kv_index::source::test_support::{
        event_batch, full_binding_with_adapter, full_store, gpu_remove,
    };
    use crate::kv_index::{KVEvent, RoutableTarget};
    use metrics_exporter_prometheus::PrometheusBuilder;
    use std::collections::BTreeSet;

    fn ingestion_series(scrape: &str) -> BTreeSet<String> {
        scrape
            .lines()
            .filter_map(|line| line.rsplit_once(' '))
            .filter_map(|(key, _)| {
                let (name, _) = key.split_once('{').unwrap_or((key, ""));
                matches!(
                    name,
                    "vllm_router_kv_events_total"
                        | "vllm_router_kv_event_warnings_total"
                        | "vllm_router_kv_source_last_event_age_seconds"
                )
                .then(|| key.to_owned())
            })
            .collect()
    }

    #[test]
    fn ingestion_cardinality_stays_bounded_across_workers_and_adapters() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let handle = recorder.handle();
        let mut subscriptions = Vec::new();
        let mut expected: BTreeSet<String> = [
            "vllm_router_kv_events_total{event_type=\"stored\",status=\"ok\"}",
            "vllm_router_kv_events_total{event_type=\"removed\",status=\"ok\"}",
            "vllm_router_kv_event_warnings_total{kind=\"unknown_remove\"}",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        for i in 0..500 {
            let worker = format!("http://worker-{i}:8000");
            let adapter = format!("adapter-{}", i % 100);
            let source = SourceAge::new(&worker, 0);
            expected.insert(format!(
                "vllm_router_kv_source_last_event_age_seconds{{source=\"{}\"}}",
                source.slot
            ));
            subscriptions.push(source);

            let index = full_binding_with_adapter(
                RoutableTarget {
                    instance_id: worker.into(),
                    dp_rank: 0,
                },
                4,
                &adapter,
                &adapter,
            );
            index.begin_empty(0);
            let store = full_store(&[1, 2, 3, 4], 4, Some((&adapter, &adapter)));
            let remove = gpu_remove(Some(0), store.block_hashes.clone());
            index
                .apply_batch(0, &event_batch(vec![remove, KVEvent::BlockStored(store)]))
                .unwrap();
        }
        assert_eq!(expected.len(), 503);
        assert_eq!(ingestion_series(&handle.render()), expected);

        // Negative control: one extra unbounded label must fail the same guard.
        metrics::counter!("vllm_router_kv_event_warnings_total", "kind" => "unknown_remove", "worker" => "http://worker-0:8000").increment(1);
        assert_ne!(ingestion_series(&handle.render()), expected);
    }
}
