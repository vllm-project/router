// MoRIIO Service Discovery Implementation
// This module implements ZMQ-based service discovery for MoRIIO (AMD) P/D connector

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

/// Default ping timeout in seconds (heartbeat TTL)
const DEFAULT_PING_SECONDS: u64 = 5;

/// KV transfer mode for MoRIIO connector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferMode {
    /// Prefill runs first; router waits for its response to get remote_engine_id + remote_block_ids
    Read,
    /// Prefill and decode fire concurrently; decode waits for ZMQ notification from prefill
    Write,
}

impl std::fmt::Display for TransferMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransferMode::Read => write!(f, "READ"),
            TransferMode::Write => write!(f, "WRITE"),
        }
    }
}

impl std::str::FromStr for TransferMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "READ" => Ok(TransferMode::Read),
            "WRITE" => Ok(TransferMode::Write),
            other => Err(format!("Unknown transfer_mode '{}'; expected READ or WRITE", other)),
        }
    }
}

/// Registration message sent by a MoRIIO vLLM instance over ZMQ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoriIORegistration {
    /// Message type: "register" or "HELLO"
    #[serde(rename = "type")]
    pub message_type: String,
    /// Role: "P" (prefill) or "D" (decode)
    pub role: String,
    /// Full HTTP URL including path, e.g. "http://1.2.3.4:8000/v1/completions"
    pub request_address: String,
    pub handshake_port: u16,
    pub notify_port: u16,
    pub dp_size: u32,
    pub tp_size: u32,
    /// Transfer mode string: "READ" or "WRITE"
    pub transfer_mode: String,
}

/// A live MoRIIO instance with its metadata
#[derive(Debug, Clone)]
pub struct MoriIOInstance {
    /// Full HTTP URL, e.g. "http://1.2.3.4:8000/v1/completions"
    pub request_address: String,
    pub handshake_port: u16,
    pub notify_port: u16,
    pub dp_size: u32,
    pub tp_size: u32,
    /// Unix timestamp after which this entry is considered stale
    pub expires_at: u64,
}

/// Registry that tracks live prefill and decode MoRIIO instances
#[derive(Debug)]
pub struct MoriIOServiceRegistry {
    /// keyed by request_address (the canonical identity of a worker)
    prefill_instances: Arc<Mutex<HashMap<String, MoriIOInstance>>>,
    decode_instances: Arc<Mutex<HashMap<String, MoriIOInstance>>>,
    /// The agreed-upon transfer mode for the whole cluster (set on first registration)
    global_transfer_mode: Arc<Mutex<Option<TransferMode>>>,
    shutdown_tx: Option<broadcast::Sender<()>>,
}

impl Default for MoriIOServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MoriIOServiceRegistry {
    pub fn new() -> Self {
        Self {
            prefill_instances: Arc::new(Mutex::new(HashMap::new())),
            decode_instances: Arc::new(Mutex::new(HashMap::new())),
            global_transfer_mode: Arc::new(Mutex::new(None)),
            shutdown_tx: None,
        }
    }

    /// Start the ZMQ ROUTER socket listener for MoRIIO registration messages
    pub async fn start_listener(&mut self, bind_address: &str) -> Result<(), String> {
        info!(
            "Starting MoRIIO service discovery listener on {}",
            bind_address
        );

        let (shutdown_tx, mut shutdown_rx) = broadcast::channel(1);
        self.shutdown_tx = Some(shutdown_tx);

        let prefill_instances = Arc::clone(&self.prefill_instances);
        let decode_instances = Arc::clone(&self.decode_instances);
        let global_transfer_mode = Arc::clone(&self.global_transfer_mode);
        let bind_addr = bind_address.to_string();

        tokio::spawn(async move {
            let context = zmq::Context::new();
            let socket = context.socket(zmq::ROUTER).unwrap();

            if let Err(e) = socket.bind(&format!("tcp://{}", bind_addr)) {
                error!("Failed to bind MoRIIO ZMQ socket to {}: {}", bind_addr, e);
                return;
            }

            info!("MoRIIO ZMQ service discovery bound to tcp://{}", bind_addr);
            socket.set_rcvtimeo(1000).unwrap();

            let mut cleanup_counter: u32 = 0;
            loop {
                if shutdown_rx.try_recv().is_ok() {
                    info!("MoRIIO service discovery shutting down");
                    break;
                }

                match socket.recv_multipart(zmq::DONTWAIT) {
                    Ok(parts) if parts.len() >= 2 => {
                        let message_data = &parts[1];
                        Self::handle_registration(
                            message_data,
                            &prefill_instances,
                            &decode_instances,
                            &global_transfer_mode,
                        );
                    }
                    Ok(_) => {
                        warn!("MoRIIO ZMQ received malformed multipart message (< 2 parts)");
                    }
                    Err(zmq::Error::EAGAIN) => {
                        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                    }
                    Err(e) => {
                        warn!("MoRIIO ZMQ receive error: {}", e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                }

                cleanup_counter += 1;
                if cleanup_counter >= 500 {
                    cleanup_counter = 0;
                    Self::cleanup_expired(&prefill_instances, &decode_instances, &global_transfer_mode).await;
                }
            }
        });

        Ok(())
    }

    fn handle_registration(
        message_data: &[u8],
        prefill_instances: &Arc<Mutex<HashMap<String, MoriIOInstance>>>,
        decode_instances: &Arc<Mutex<HashMap<String, MoriIOInstance>>>,
        global_transfer_mode: &Arc<Mutex<Option<TransferMode>>>,
    ) {
        let reg: MoriIORegistration = match rmp_serde::from_slice(message_data) {
            Ok(r) => r,
            Err(e) => {
                warn!("Failed to parse MoRIIO registration message: {}", e);
                return;
            }
        };

        let mode: TransferMode = match reg.transfer_mode.parse() {
            Ok(m) => m,
            Err(e) => {
                warn!("Invalid transfer_mode in MoRIIO registration: {}", e);
                return;
            }
        };

        // Enforce cluster-wide transfer_mode consistency
        {
            let mut global = global_transfer_mode.lock().unwrap();
            match *global {
                None => {
                    info!("MoRIIO cluster transfer_mode set to {}", mode);
                    *global = Some(mode);
                }
                Some(existing) if existing != mode => {
                    error!(
                        "MoRIIO transfer_mode mismatch: cluster is {}, but {} sent {}. Rejecting registration.",
                        existing, reg.request_address, mode
                    );
                    return;
                }
                Some(_) => {} // consistent, proceed
            }
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let instance = MoriIOInstance {
            request_address: reg.request_address.clone(),
            handshake_port: reg.handshake_port,
            notify_port: reg.notify_port,
            dp_size: reg.dp_size,
            tp_size: reg.tp_size,
            expires_at: now + DEFAULT_PING_SECONDS,
        };

        match reg.role.as_str() {
            "P" => {
                let mut map = prefill_instances.lock().unwrap();
                let is_new = !map.contains_key(&reg.request_address);
                map.insert(reg.request_address.clone(), instance);
                if is_new {
                    info!(
                        "Add MoRIIO Prefill [addr={}, handshake={}, notify={}, dp={}, tp={}, mode={}]",
                        reg.request_address,
                        reg.handshake_port,
                        reg.notify_port,
                        reg.dp_size,
                        reg.tp_size,
                        mode
                    );
                } else {
                    debug!("Update MoRIIO Prefill [addr={}]", reg.request_address);
                }
            }
            "D" => {
                let mut map = decode_instances.lock().unwrap();
                let is_new = !map.contains_key(&reg.request_address);
                map.insert(reg.request_address.clone(), instance);
                if is_new {
                    info!(
                        "Add MoRIIO Decode [addr={}, handshake={}, notify={}, dp={}, tp={}, mode={}]",
                        reg.request_address,
                        reg.handshake_port,
                        reg.notify_port,
                        reg.dp_size,
                        reg.tp_size,
                        mode
                    );
                } else {
                    debug!("Update MoRIIO Decode [addr={}]", reg.request_address);
                }
            }
            other => {
                warn!("Unknown MoRIIO role '{}' from {}", other, reg.request_address);
            }
        }
    }

    async fn cleanup_expired(
        prefill_instances: &Arc<Mutex<HashMap<String, MoriIOInstance>>>,
        decode_instances: &Arc<Mutex<HashMap<String, MoriIOInstance>>>,
        global_mode: &Arc<Mutex<Option<TransferMode>>>,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let prefill_empty;
        {
            let mut map = prefill_instances.lock().unwrap();
            let expired: Vec<_> = map
                .iter()
                .filter(|(_, i)| i.expires_at <= now)
                .map(|(k, _)| k.clone())
                .collect();
            for key in expired {
                info!("Remove MoRIIO Prefill [addr={}, expired]", key);
                map.remove(&key);
            }
            prefill_empty = map.is_empty();
        }

        let decode_empty;
        {
            let mut map = decode_instances.lock().unwrap();
            let expired: Vec<_> = map
                .iter()
                .filter(|(_, i)| i.expires_at <= now)
                .map(|(k, _)| k.clone())
                .collect();
            for key in expired {
                info!("Remove MoRIIO Decode [addr={}, expired]", key);
                map.remove(&key);
            }
            decode_empty = map.is_empty();
        }

        if prefill_empty && decode_empty {
            *global_mode.lock().unwrap() = None;
            info!("All MoRIIO instances expired; transfer_mode reset");
        }
    }

    pub fn get_prefill_instances(&self) -> Vec<MoriIOInstance> {
        self.prefill_instances
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect()
    }

    pub fn get_decode_instances(&self) -> Vec<MoriIOInstance> {
        self.decode_instances
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect()
    }

    /// Returns the cluster-wide transfer mode, or None if no instance has registered yet
    pub fn transfer_mode(&self) -> Option<TransferMode> {
        *self.global_transfer_mode.lock().unwrap()
    }

    pub fn get_instance_counts(&self) -> (usize, usize) {
        (
            self.prefill_instances.lock().unwrap().len(),
            self.decode_instances.lock().unwrap().len(),
        )
    }

    pub fn shutdown(&self) {
        if let Some(ref tx) = self.shutdown_tx {
            let _ = tx.send(());
        }
    }
}

impl Drop for MoriIOServiceRegistry {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_mode_parse() {
        assert_eq!("READ".parse::<TransferMode>().unwrap(), TransferMode::Read);
        assert_eq!("WRITE".parse::<TransferMode>().unwrap(), TransferMode::Write);
        assert_eq!("read".parse::<TransferMode>().unwrap(), TransferMode::Read);
        assert_eq!("write".parse::<TransferMode>().unwrap(), TransferMode::Write);
        assert!("UNKNOWN".parse::<TransferMode>().is_err());
    }

    #[test]
    fn test_transfer_mode_display() {
        assert_eq!(TransferMode::Read.to_string(), "READ");
        assert_eq!(TransferMode::Write.to_string(), "WRITE");
    }

    #[test]
    fn test_handle_registration_mode_mismatch() {
        let prefill = Arc::new(Mutex::new(HashMap::new()));
        let decode = Arc::new(Mutex::new(HashMap::new()));
        let global = Arc::new(Mutex::new(Some(TransferMode::Read)));

        // Build a WRITE-mode registration msgpack
        let reg = MoriIORegistration {
            message_type: "register".to_string(),
            role: "P".to_string(),
            request_address: "http://1.2.3.4:8000/v1/completions".to_string(),
            handshake_port: 8001,
            notify_port: 8002,
            dp_size: 1,
            tp_size: 1,
            transfer_mode: "WRITE".to_string(),
        };
        let data = rmp_serde::to_vec_named(&reg).unwrap();

        MoriIOServiceRegistry::handle_registration(&data, &prefill, &decode, &global);

        // Should have been rejected — no prefill instance added
        assert!(prefill.lock().unwrap().is_empty());
        // Global mode must still be READ
        assert_eq!(*global.lock().unwrap(), Some(TransferMode::Read));
    }

    #[test]
    fn test_handle_registration_sets_global_mode() {
        let prefill = Arc::new(Mutex::new(HashMap::new()));
        let decode = Arc::new(Mutex::new(HashMap::new()));
        let global = Arc::new(Mutex::new(None));

        let reg = MoriIORegistration {
            message_type: "HELLO".to_string(),
            role: "P".to_string(),
            request_address: "http://1.2.3.4:8000/v1/completions".to_string(),
            handshake_port: 8001,
            notify_port: 8002,
            dp_size: 1,
            tp_size: 1,
            transfer_mode: "READ".to_string(),
        };
        let data = rmp_serde::to_vec_named(&reg).unwrap();

        MoriIOServiceRegistry::handle_registration(&data, &prefill, &decode, &global);

        assert_eq!(*global.lock().unwrap(), Some(TransferMode::Read));
        assert_eq!(prefill.lock().unwrap().len(), 1);
    }
}
