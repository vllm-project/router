//! ZMQ SUB subscriber for a single vLLM worker's KV event stream.
//!
//! Each vLLM instance runs a [`ZmqEventPublisher`] that publishes
//! `[topic, seq_bytes(8B big-endian), msgpack_payload]` frames on a PUB
//! socket.  This module connects a SUB socket, filters by topic prefix,
//! and decodes incoming batches into [`KVEventBatch`] structs.

use super::decoder::{self, KVEventBatch};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Handle to a running subscriber task.  Dropping it signals the background
/// thread to stop.
#[derive(Debug)]
pub struct SubscriberHandle {
    /// Worker identifier (typically HTTP address).
    pub worker_id: String,
    /// Set to `false` to request graceful shutdown.
    running: Arc<AtomicBool>,
    /// Join handle for the blocking subscriber thread.
    join_handle: Option<std::thread::JoinHandle<()>>,
}

impl SubscriberHandle {
    /// Signal the subscriber thread to stop and wait for it to finish.
    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.join_handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for SubscriberHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Spawn a blocking thread that subscribes to a vLLM worker's KV event PUB
/// socket and forwards decoded batches through `event_tx`.
///
/// # Arguments
/// * `worker_id`    – Logical name for this worker (used in decoded batches).
/// * `zmq_endpoint` – Full ZMQ address, e.g. `tcp://10.0.0.5:5557`.
/// * `topic_filter` – Topic prefix for SUB filtering (e.g. `"kv@"`).
/// * `event_tx`     – Channel sender for decoded event batches.
pub fn spawn_subscriber(
    worker_id: String,
    zmq_endpoint: String,
    topic_filter: String,
    event_tx: mpsc::UnboundedSender<KVEventBatch>,
) -> SubscriberHandle {
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = Arc::clone(&running);
    let wid = worker_id.clone();

    let join_handle = std::thread::Builder::new()
        .name(format!("kv-sub-{}", worker_id))
        .spawn(move || {
            subscriber_loop(
                &wid,
                &zmq_endpoint,
                &topic_filter,
                &event_tx,
                &running_clone,
            );
        })
        .expect("failed to spawn KV event subscriber thread");

    SubscriberHandle {
        worker_id,
        running,
        join_handle: Some(join_handle),
    }
}

/// Main loop for a single subscriber thread.
fn subscriber_loop(
    worker_id: &str,
    zmq_endpoint: &str,
    topic_filter: &str,
    event_tx: &mpsc::UnboundedSender<KVEventBatch>,
    running: &AtomicBool,
) {
    let ctx = zmq::Context::new();
    let socket = match ctx.socket(zmq::SUB) {
        Ok(s) => s,
        Err(e) => {
            error!(
                "KV subscriber {}: failed to create ZMQ SUB socket: {}",
                worker_id, e
            );
            return;
        }
    };

    // Set a receive timeout so we can periodically check the `running` flag.
    if let Err(e) = socket.set_rcvtimeo(1000) {
        warn!("KV subscriber {}: failed to set rcvtimeo: {}", worker_id, e);
    }

    if let Err(e) = socket.set_subscribe(topic_filter.as_bytes()) {
        error!(
            "KV subscriber {}: failed to set subscription filter '{}': {}",
            worker_id, topic_filter, e
        );
        return;
    }

    if let Err(e) = socket.connect(zmq_endpoint) {
        error!(
            "KV subscriber {}: failed to connect to {}: {}",
            worker_id, zmq_endpoint, e
        );
        return;
    }

    info!(
        "KV subscriber {}: connected to {} (filter='{}')",
        worker_id, zmq_endpoint, topic_filter
    );

    while running.load(Ordering::Relaxed) {
        match socket.recv_multipart(0) {
            Ok(parts) => {
                // Expected: [topic, seq_bytes(8B), payload]
                if parts.len() < 3 {
                    debug!(
                        "KV subscriber {}: ignoring message with {} parts (expected 3)",
                        worker_id,
                        parts.len()
                    );
                    continue;
                }

                let payload = &parts[2];
                match decoder::decode_event_batch(payload, worker_id) {
                    Ok(batch) => {
                        if event_tx.send(batch).is_err() {
                            info!("KV subscriber {}: channel closed, stopping", worker_id);
                            break;
                        }
                    }
                    Err(e) => {
                        warn!(
                            "KV subscriber {}: failed to decode event batch: {}",
                            worker_id, e
                        );
                    }
                }
            }
            Err(zmq::Error::EAGAIN) => {
                // Receive timeout – loop back to check `running`.
                continue;
            }
            Err(e) => {
                error!("KV subscriber {}: ZMQ recv error: {}", worker_id, e);
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }
    }

    info!("KV subscriber {}: shutting down", worker_id);
}
