// vLLM Service Discovery Implementation
// This module implements service discovery for vLLM P2P NCCL coordination

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

/// Default ping timeout in seconds
const DEFAULT_PING_SECONDS: u64 = 5;

/// Service type for registration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ServiceType {
    Prefill,
    Decode,
}

impl std::fmt::Display for ServiceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceType::Prefill => write!(f, "P"),
            ServiceType::Decode => write!(f, "D"),
        }
    }
}

/// Service registration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceRegistration {
    #[serde(rename = "type")]
    pub service_type: String, // "P" or "D"
    pub http_address: String,
    pub zmq_address: String,
}

/// Service instance with expiration timestamp
#[derive(Debug, Clone)]
pub struct ServiceInstance {
    pub zmq_address: String,
    pub expires_at: u64, // Unix timestamp
}

/// Service registry maintaining prefill and decode instances
#[derive(Debug)]
pub struct ServiceRegistry {
    prefill_instances: Arc<Mutex<HashMap<String, ServiceInstance>>>,
    decode_instances: Arc<Mutex<HashMap<String, ServiceInstance>>>,
    shutdown_tx: Option<broadcast::Sender<()>>,
}

impl ServiceRegistry {
    /// Create a new service registry
    pub fn new() -> Self {
        Self {
            prefill_instances: Arc::new(Mutex::new(HashMap::new())),
            decode_instances: Arc::new(Mutex::new(HashMap::new())),
            shutdown_tx: None,
        }
    }

    /// Start the ZMQ service discovery listener
    pub async fn start_listener(&mut self, bind_address: &str) -> Result<(), String> {
        info!("Starting vLLM service discovery listener on {}", bind_address);

        let (shutdown_tx, mut shutdown_rx) = broadcast::channel(1);
        self.shutdown_tx = Some(shutdown_tx);

        let prefill_instances = Arc::clone(&self.prefill_instances);
        let decode_instances = Arc::clone(&self.decode_instances);
        let bind_addr = bind_address.to_string();

        tokio::spawn(async move {
            // Initialize ZMQ context and socket
            let context = zmq::Context::new();
            let router_socket = context.socket(zmq::ROUTER).unwrap();

            if let Err(e) = router_socket.bind(&format!("tcp://{}", bind_addr)) {
                warn!("Failed to bind ZMQ socket to {}: {}", bind_addr, e);
                return;
            }

            info!("ZMQ service discovery bound to tcp://{}", bind_addr);

            // Set non-blocking mode for graceful shutdown
            router_socket.set_rcvtimeo(1000).unwrap(); // 1 second timeout

            loop {
                // Check for shutdown signal
                if shutdown_rx.try_recv().is_ok() {
                    info!("Service discovery shutting down");
                    break;
                }

                // Try to receive a message
                match router_socket.recv_multipart(zmq::DONTWAIT) {
                    Ok(message_parts) => {
                        if message_parts.len() >= 2 {
                            let remote_address = message_parts[0].clone();
                            let message_data = &message_parts[1];

                            Self::handle_registration_message(
                                message_data,
                                &remote_address,
                                &prefill_instances,
                                &decode_instances,
                            ).await;
                        }
                    }
                    Err(zmq::Error::EAGAIN) => {
                        // No message available, continue
                        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                    }
                    Err(e) => {
                        warn!("ZMQ receive error: {}", e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                }

                // Clean up expired instances periodically
                Self::cleanup_expired_instances(&prefill_instances, &decode_instances).await;
            }
        });

        Ok(())
    }

    /// Handle incoming service registration message
    async fn handle_registration_message(
        message_data: &[u8],
        remote_address: &[u8],
        prefill_instances: &Arc<Mutex<HashMap<String, ServiceInstance>>>,
        decode_instances: &Arc<Mutex<HashMap<String, ServiceInstance>>>,
    ) {
        // Parse MessagePack data
        let data: ServiceRegistration = match rmp_serde::from_slice(message_data) {
            Ok(data) => data,
            Err(e) => {
                warn!("Failed to parse service registration: {}", e);
                return;
            }
        };

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let instance = ServiceInstance {
            zmq_address: data.zmq_address.clone(),
            expires_at: current_time + DEFAULT_PING_SECONDS,
        };

        let remote_addr_str = String::from_utf8_lossy(remote_address);

        match data.service_type.as_str() {
            "P" => {
                let mut prefill = prefill_instances.lock().unwrap();
                let is_new = !prefill.contains_key(&data.http_address);
                prefill.insert(data.http_address.clone(), instance);

                if is_new {
                    info!("🔵Add Prefill [HTTP:{}, ZMQ:{}]", data.http_address, data.zmq_address);
                } else {
                    debug!("🔄Update Prefill [HTTP:{}, ZMQ:{}]", data.http_address, data.zmq_address);
                }
            }
            "D" => {
                let mut decode = decode_instances.lock().unwrap();
                let is_new = !decode.contains_key(&data.http_address);
                decode.insert(data.http_address.clone(), instance);

                if is_new {
                    info!("🔵Add Decode [HTTP:{}, ZMQ:{}]", data.http_address, data.zmq_address);
                } else {
                    debug!("🔄Update Decode [HTTP:{}, ZMQ:{}]", data.http_address, data.zmq_address);
                }
            }
            _ => {
                warn!("Unknown service type '{}' from {}", data.service_type, remote_addr_str);
            }
        }
    }

    /// Clean up expired service instances
    async fn cleanup_expired_instances(
        prefill_instances: &Arc<Mutex<HashMap<String, ServiceInstance>>>,
        decode_instances: &Arc<Mutex<HashMap<String, ServiceInstance>>>,
    ) {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Clean prefill instances
        {
            let mut prefill = prefill_instances.lock().unwrap();
            let expired_keys: Vec<_> = prefill
                .iter()
                .filter(|(_, instance)| instance.expires_at <= current_time)
                .map(|(key, _)| key.clone())
                .collect();

            for key in expired_keys {
                if let Some(instance) = prefill.remove(&key) {
                    info!("🔴Remove Prefill [HTTP:{}, ZMQ:{}, expired]", key, instance.zmq_address);
                }
            }
        }

        // Clean decode instances
        {
            let mut decode = decode_instances.lock().unwrap();
            let expired_keys: Vec<_> = decode
                .iter()
                .filter(|(_, instance)| instance.expires_at <= current_time)
                .map(|(key, _)| key.clone())
                .collect();

            for key in expired_keys {
                if let Some(instance) = decode.remove(&key) {
                    info!("🔴Remove Decode [HTTP:{}, ZMQ:{}, expired]", key, instance.zmq_address);
                }
            }
        }
    }

    /// Register a service manually (fallback mode)
    pub fn register_service(&self, http_address: String, zmq_address: String, service_type: ServiceType) {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let instance = ServiceInstance {
            zmq_address: zmq_address.clone(),
            expires_at: current_time + DEFAULT_PING_SECONDS,
        };

        match service_type {
            ServiceType::Prefill => {
                let mut prefill = self.prefill_instances.lock().unwrap();
                prefill.insert(http_address.clone(), instance);
                info!("🔵Manual register Prefill [HTTP:{}, ZMQ:{}]", http_address, zmq_address);
            }
            ServiceType::Decode => {
                let mut decode = self.decode_instances.lock().unwrap();
                decode.insert(http_address.clone(), instance);
                info!("🔵Manual register Decode [HTTP:{}, ZMQ:{}]", http_address, zmq_address);
            }
        }
    }

    /// Get ZMQ address for a given HTTP address
    pub fn get_zmq_address(&self, http_address: &str, service_type: ServiceType) -> Option<String> {
        let instances = match service_type {
            ServiceType::Prefill => &self.prefill_instances,
            ServiceType::Decode => &self.decode_instances,
        };

        let guard = instances.lock().unwrap();
        guard.get(http_address).map(|instance| instance.zmq_address.clone())
    }

    /// Get all available prefill instances
    pub fn get_prefill_instances(&self) -> Vec<(String, String)> {
        let guard = self.prefill_instances.lock().unwrap();
        guard
            .iter()
            .map(|(http, instance)| (http.clone(), instance.zmq_address.clone()))
            .collect()
    }

    /// Get all available decode instances
    pub fn get_decode_instances(&self) -> Vec<(String, String)> {
        let guard = self.decode_instances.lock().unwrap();
        guard
            .iter()
            .map(|(http, instance)| (http.clone(), instance.zmq_address.clone()))
            .collect()
    }

    /// Get instance count for debugging
    pub fn get_instance_counts(&self) -> (usize, usize) {
        let prefill_count = self.prefill_instances.lock().unwrap().len();
        let decode_count = self.decode_instances.lock().unwrap().len();
        (prefill_count, decode_count)
    }

    /// Shutdown the service discovery
    pub fn shutdown(&self) {
        if let Some(ref tx) = self.shutdown_tx {
            let _ = tx.send(());
        }
    }
}

impl Drop for ServiceRegistry {
    fn drop(&mut self) {
        self.shutdown();
    }
}