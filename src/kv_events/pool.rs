//! Event pool that manages ZMQ subscriptions to multiple vLLM workers.
//!
//! [`KVEventPool`] owns one [`SubscriberHandle`] per worker and provides
//! a single [`mpsc::UnboundedReceiver`] that aggregates all incoming
//! [`KVEventBatch`] messages for downstream index updates.

use super::decoder::KVEventBatch;
use super::subscriber::{self, SubscriberHandle};
use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{info, warn};

/// Configuration for the KV event pool.
#[derive(Debug, Clone)]
pub struct KVEventPoolConfig {
    /// ZMQ topic prefix to subscribe to (e.g., `"kv@"`).
    pub topic_filter: String,
    /// Default KV events ZMQ port on each vLLM worker.
    /// When service discovery provides only HTTP addresses, the pool
    /// derives the ZMQ endpoint by replacing the HTTP port with this port.
    pub default_kv_events_port: u16,
}

impl Default for KVEventPoolConfig {
    fn default() -> Self {
        Self {
            topic_filter: "kv@".to_string(),
            default_kv_events_port: 5556,
        }
    }
}

/// Manages ZMQ SUB connections to all discovered vLLM workers and funnels
/// decoded KV events into a single channel for the index updater.
pub struct KVEventPool {
    config: KVEventPoolConfig,
    event_tx: mpsc::UnboundedSender<KVEventBatch>,
    subscribers: HashMap<String, SubscriberHandle>,
}

impl KVEventPool {
    /// Create a new pool and return the receiver end of the event channel.
    pub fn new(config: KVEventPoolConfig) -> (Self, mpsc::UnboundedReceiver<KVEventBatch>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            Self {
                config,
                event_tx: tx,
                subscribers: HashMap::new(),
            },
            rx,
        )
    }

    /// Subscribe to a worker's KV events using an explicit ZMQ endpoint.
    ///
    /// If a subscription for `worker_id` already exists, it is replaced.
    pub fn subscribe_worker(&mut self, worker_id: String, zmq_endpoint: String) {
        // Tear down existing subscription if present.
        if let Some(mut old) = self.subscribers.remove(&worker_id) {
            info!(
                "KVEventPool: replacing existing subscription for {}",
                worker_id
            );
            old.shutdown();
        }

        let handle = subscriber::spawn_subscriber(
            worker_id.clone(),
            zmq_endpoint.clone(),
            self.config.topic_filter.clone(),
            self.event_tx.clone(),
        );

        info!(
            "KVEventPool: subscribed to worker {} at {}",
            worker_id, zmq_endpoint
        );
        self.subscribers.insert(worker_id, handle);
    }

    /// Subscribe to a worker using its HTTP address; derives the ZMQ
    /// endpoint by replacing the HTTP port with [`KVEventPoolConfig::default_kv_events_port`].
    pub fn subscribe_worker_by_http(&mut self, worker_id: String, http_address: &str) {
        let zmq_endpoint = derive_zmq_endpoint(http_address, self.config.default_kv_events_port);
        self.subscribe_worker(worker_id, zmq_endpoint);
    }

    /// Unsubscribe from a worker and shut down its ZMQ connection.
    pub fn unsubscribe_worker(&mut self, worker_id: &str) {
        if let Some(mut handle) = self.subscribers.remove(worker_id) {
            handle.shutdown();
            info!("KVEventPool: unsubscribed worker {}", worker_id);
        } else {
            warn!(
                "KVEventPool: attempted to unsubscribe unknown worker {}",
                worker_id
            );
        }
    }

    /// Returns true if a subscription exists for the given worker.
    pub fn has_worker(&self, worker_id: &str) -> bool {
        self.subscribers.contains_key(worker_id)
    }

    /// Number of active subscriptions.
    pub fn worker_count(&self) -> usize {
        self.subscribers.len()
    }

    /// Shut down all subscriber threads.
    pub fn shutdown_all(&mut self) {
        let ids: Vec<String> = self.subscribers.keys().cloned().collect();
        for id in ids {
            self.unsubscribe_worker(&id);
        }
    }
}

impl Drop for KVEventPool {
    fn drop(&mut self) {
        self.shutdown_all();
    }
}

/// Derive a ZMQ TCP endpoint from an HTTP address string.
///
/// Example: `"10.0.0.5:8000"` with port 5556 -> `"tcp://10.0.0.5:5556"`
fn derive_zmq_endpoint(http_address: &str, kv_events_port: u16) -> String {
    // Strip scheme if present.
    let addr = http_address
        .trim_start_matches("http://")
        .trim_start_matches("https://");

    // Split host and port.
    if let Some(colon_idx) = addr.rfind(':') {
        let host = &addr[..colon_idx];
        format!("tcp://{}:{}", host, kv_events_port)
    } else {
        format!("tcp://{}:{}", addr, kv_events_port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_zmq_endpoint() {
        assert_eq!(
            derive_zmq_endpoint("10.0.0.5:8000", 5556),
            "tcp://10.0.0.5:5556"
        );
        assert_eq!(
            derive_zmq_endpoint("http://10.0.0.5:8000", 5556),
            "tcp://10.0.0.5:5556"
        );
        assert_eq!(derive_zmq_endpoint("my-host", 5556), "tcp://my-host:5556");
    }
}
