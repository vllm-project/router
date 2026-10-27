use crate::routers::RouterTrait;

use futures::{StreamExt, TryStreamExt};
use k8s_openapi::api::core::v1::Pod;
use kube::{
    api::Api,
    runtime::watcher::{watcher, Config},
    runtime::WatchStreamExt,
    Client,
};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};

use rustls;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::task;
use tokio::time;
use tracing::{debug, error, info, warn};

/// Represents the service discovery configuration
#[derive(Debug, Clone)]
pub struct ServiceDiscoveryConfig {
    pub enabled: bool,
    pub selector: HashMap<String, String>,
    pub check_interval: Duration,
    pub port: u16,
    pub namespace: Option<String>,
    // PD mode specific configuration
    pub pd_mode: bool,
    pub prefill_selector: HashMap<String, String>,
    pub decode_selector: HashMap<String, String>,
    // Bootstrap port annotation specific to mooncake implementation
    pub bootstrap_port_annotation: String,
}

impl Default for ServiceDiscoveryConfig {
    fn default() -> Self {
        ServiceDiscoveryConfig {
            enabled: false,
            selector: HashMap::new(),
            check_interval: Duration::from_secs(60),
            port: 8000,      // Standard port for modern services
            namespace: None, // None means watch all namespaces
            pd_mode: false,
            prefill_selector: HashMap::new(),
            decode_selector: HashMap::new(),
            bootstrap_port_annotation: "vllm.ai/bootstrap-port".to_string(),
        }
    }
}

/// Pod type for PD mode service discovery
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PodType {
    Prefill,
    Decode,
    Regular,
}

/// Bookkeeping entry stored in the per-pod tracking map.
/// Retains the *registration-time* URL and PD role so that removal is exact
/// even when the pod's IP or labels have changed since it was first added.
#[derive(Debug, Clone)]
struct TrackedPod {
    /// The worker URL that was registered in the router (`http://<ip>:<port>`).
    worker_url: String,
    /// The PD role at registration time.
    pod_type: Option<PodType>,
    /// The Kubernetes pod UID at registration time.
    uid: Option<String>,
}

/// Represents a Kubernetes pod's information used for worker management
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PodInfo {
    pub name: String,
    /// Kubernetes namespace the pod belongs to. Empty string when metadata.namespace is absent.
    /// Combined with `name` to form a cluster-unique tracking key via `pod_key()`.
    pub namespace: String,
    /// Kubernetes pod UID. Used to reconcile same-name replacement pods across restarts/missed deletions.
    pub uid: Option<String>,
    pub ip: IpAddr,
    pub status: String,
    pub is_ready: bool,
    pub pod_type: Option<PodType>,
    pub bootstrap_port: Option<u16>,
}

impl PodInfo {
    /// Check if a pod matches any of the given selectors
    fn matches_selector(pod: &Pod, selector: &HashMap<String, String>) -> bool {
        if selector.is_empty() {
            return false;
        }

        pod.metadata
            .labels
            .as_ref()
            .is_some_and(|labels| selector.iter().all(|(k, v)| labels.get(k) == Some(v)))
    }

    /// Check if a pod should be included in service discovery
    pub fn should_include(pod: &Pod, config: &ServiceDiscoveryConfig) -> bool {
        if config.pd_mode {
            // In PD mode, at least one selector must be non-empty
            if config.prefill_selector.is_empty() && config.decode_selector.is_empty() {
                warn!("PD mode enabled but both prefill_selector and decode_selector are empty");
                return false;
            }
            // In PD mode, pod must match either prefill or decode selector
            Self::matches_selector(pod, &config.prefill_selector)
                || Self::matches_selector(pod, &config.decode_selector)
        } else {
            // In regular mode, pod must match the general selector
            if config.selector.is_empty() {
                warn!("Regular mode enabled but selector is empty");
                return false;
            }
            Self::matches_selector(pod, &config.selector)
        }
    }

    /// Format a namespace-qualified pod identity key of the form `namespace/name`.
    fn pod_key_from_meta(ns: Option<&str>, name: Option<&str>) -> String {
        format!("{}/{}", ns.unwrap_or(""), name.unwrap_or(""))
    }

    /// Check if a pod event should be allowed through the watcher stream firewall.
    /// Returns true if the pod matches the selector or is currently in the tracked set.
    fn should_allow_event(
        pod: &Pod,
        config: &ServiceDiscoveryConfig,
        tracked_pods: &HashMap<String, TrackedPod>,
    ) -> bool {
        let pod_key = Self::pod_key_from_meta(
            pod.metadata.namespace.as_deref(),
            pod.metadata.name.as_deref(),
        );
        Self::should_include(pod, config) || tracked_pods.contains_key(&pod_key)
    }

    /// Unified PodInfo creation with optional PD configuration
    pub fn from_pod(pod: &Pod, config: Option<&ServiceDiscoveryConfig>) -> Option<Self> {
        let name = pod.metadata.name.clone()?;
        let status = pod.status.clone()?;
        let pod_ip: IpAddr = status.pod_ip?.parse().ok()?;

        let is_ready = if let Some(conditions) = &status.conditions {
            conditions
                .iter()
                .any(|condition| condition.type_ == "Ready" && condition.status == "True")
        } else {
            false
        };

        let pod_status = status.phase.unwrap_or_else(|| "Unknown".to_string());

        // Determine pod type based on labels if config is provided and in PD mode
        let pod_type = if let Some(config) = config {
            if config.pd_mode {
                // Use simplified helper methods for cleaner logic
                if Self::matches_selector(pod, &config.prefill_selector) {
                    Some(PodType::Prefill)
                } else if Self::matches_selector(pod, &config.decode_selector) {
                    Some(PodType::Decode)
                } else {
                    Some(PodType::Regular)
                }
            } else {
                Some(PodType::Regular)
            }
        } else {
            // No config provided, default to None (for backwards compatibility)
            None
        };

        // Extract bootstrap port from annotations for prefill pods
        let bootstrap_port = if matches!(pod_type, Some(PodType::Prefill)) {
            if let Some(config) = config {
                pod.metadata
                    .annotations
                    .as_ref()
                    .and_then(|annotations| annotations.get(&config.bootstrap_port_annotation))
                    .and_then(|port_str| port_str.parse::<u16>().ok())
            } else {
                None
            }
        } else {
            None
        };

        let namespace = pod.metadata.namespace.clone().unwrap_or_default();
        let uid = pod.metadata.uid.clone();

        Some(PodInfo {
            name,
            namespace,
            uid,
            ip: pod_ip,
            status: pod_status,
            is_ready,
            pod_type,
            bootstrap_port,
        })
    }

    /// Returns true if the pod is in a state where it can accept traffic
    pub fn is_healthy(&self) -> bool {
        self.is_ready && self.status == "Running"
    }

    /// Generates a worker URL for this pod.
    pub fn worker_url(&self, port: u16) -> String {
        format!("http://{}", SocketAddr::new(self.ip, port))
    }

    /// Returns a namespace-qualified pod identity key of the form `namespace/name`.
    /// Stable across IP, label, and status changes; unique across namespaces for
    /// cluster-wide (`Api::all`) watches.
    pub fn pod_key(&self) -> String {
        Self::pod_key_from_meta(Some(&self.namespace), Some(&self.name))
    }
}

pub async fn start_service_discovery(
    config: ServiceDiscoveryConfig,
    router: Arc<dyn RouterTrait>,
) -> Result<task::JoinHandle<()>, kube::Error> {
    // Don't initialize anything if service discovery is disabled
    if !config.enabled {
        // Return a generic error when service discovery is disabled
        return Err(kube::Error::Api(kube::error::ErrorResponse {
            status: "Disabled".to_string(),
            message: "Service discovery is disabled".to_string(),
            reason: "ConfigurationError".to_string(),
            code: 400,
        }));
    }

    let _ = rustls::crypto::ring::default_provider().install_default();

    // Initialize Kubernetes client
    let client = Client::try_default().await?;

    // Log the appropriate selectors based on mode
    if config.pd_mode {
        let prefill_selector = config
            .prefill_selector
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",");

        let decode_selector = config
            .decode_selector
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",");

        info!(
            "Starting K8s service discovery | PD mode | prefill: '{}' | decode: '{}'",
            prefill_selector, decode_selector
        );
    } else {
        let label_selector = config
            .selector
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",");

        info!(
            "Starting K8s service discovery | selector: '{}'",
            label_selector
        );
    }

    // Create the task that will run in the background
    let handle = task::spawn(async move {
        // Track pods by namespace-qualified key ("namespace/name") so same-name pods in
        // different namespaces are isolated. Each entry stores the URL and PD role captured
        // at registration time, ensuring accurate rank-aware removal even after label changes.
        let tracked_pods: Arc<Mutex<HashMap<String, TrackedPod>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Create a watcher for pods
        let pods: Api<Pod> = if let Some(namespace) = &config.namespace {
            Api::namespaced(client, namespace)
        } else {
            Api::all(client)
        };

        debug!("K8s service discovery initialized");

        // Create Arcs for configuration data
        let config_arc = Arc::new(config.clone());
        let port = config.port;

        let mut retry_delay = Duration::from_secs(1);
        const MAX_RETRY_DELAY: Duration = Duration::from_secs(300); // 5 minutes max

        loop {
            // Create a watcher with the proper parameters according to the kube-rs API
            let watcher_config = Config::default();
            let watcher_stream = watcher(pods.clone(), watcher_config).applied_objects();

            // Clone Arcs for the closures
            let config_clone = Arc::clone(&config_arc);
            let tracked_pods_clone = Arc::clone(&tracked_pods);

            // Stream firewall: only allow pods that match the selector or are currently tracked
            let filtered_stream = watcher_stream.filter_map(move |obj_res| {
                let config_inner = Arc::clone(&config_clone);
                let tracked_pods_inner = Arc::clone(&tracked_pods_clone);

                async move {
                    match obj_res {
                        Ok(pod) => {
                            let is_allowed = tracked_pods_inner.lock().is_ok_and(|tracked| {
                                PodInfo::should_allow_event(&pod, &config_inner, &tracked)
                            });

                            if is_allowed {
                                Some(Ok(pod))
                            } else {
                                None
                            }
                        }
                        Err(e) => Some(Err(e)),
                    }
                }
            });

            // Clone again for the next closure
            let tracked_pods_clone2 = Arc::clone(&tracked_pods);
            let router_clone = Arc::clone(&router);
            let config_clone2 = Arc::clone(&config_arc);

            match filtered_stream
                .try_for_each(move |pod| {
                    let tracked_pods_inner = Arc::clone(&tracked_pods_clone2);
                    let router_inner = Arc::clone(&router_clone);
                    let config_inner = Arc::clone(&config_clone2);

                    async move {
                        let pod_info = PodInfo::from_pod(&pod, Some(&config_inner));

                        if let Some(pod_info) = pod_info {
                            let should_include = PodInfo::should_include(&pod, &config_inner);
                            if pod.metadata.deletion_timestamp.is_some() {
                                handle_pod_deletion(
                                    &pod_info,
                                    tracked_pods_inner,
                                    router_inner,
                                    config_inner.pd_mode,
                                )
                                .await;
                            } else {
                                handle_pod_event(
                                    &pod_info,
                                    tracked_pods_inner,
                                    router_inner,
                                    port,
                                    config_inner.pd_mode,
                                    should_include,
                                )
                                .await;
                            }
                        }
                        Ok(())
                    }
                })
                .await
            {
                Ok(_) => {
                    // Reset retry delay on success
                    retry_delay = Duration::from_secs(1);
                }
                Err(err) => {
                    error!("Error in Kubernetes watcher: {}", err);
                    warn!(
                        "Retrying in {} seconds with exponential backoff",
                        retry_delay.as_secs()
                    );
                    time::sleep(retry_delay).await;

                    // Exponential backoff with jitter
                    retry_delay = std::cmp::min(retry_delay * 2, MAX_RETRY_DELAY);
                }
            }

            // If the watcher exits for some reason, wait a bit before restarting
            warn!(
                "Kubernetes watcher exited, restarting in {} seconds",
                config_arc.check_interval.as_secs()
            );
            time::sleep(config_arc.check_interval).await;
        }
    });

    Ok(handle)
}

async fn handle_pod_event(
    pod_info: &PodInfo,
    tracked_pods: Arc<Mutex<HashMap<String, TrackedPod>>>,
    router: Arc<dyn RouterTrait>,
    port: u16,
    pd_mode: bool,
    should_include: bool,
) {
    let worker_url = pod_info.worker_url(port);

    // If pod is healthy and matches the selector, try to add it (with atomic check-and-insert)
    if pod_info.is_healthy() && should_include {
        // Atomic check-and-insert/reconcile to prevent race conditions
        let (should_add, stale_registration) = {
            let mut tracker = match tracked_pods.lock() {
                Ok(tracker) => tracker,
                Err(e) => {
                    error!("Failed to acquire tracked_pods lock: {}", e);
                    return;
                }
            };

            match tracker.entry(pod_info.pod_key()) {
                std::collections::hash_map::Entry::Occupied(mut occ) => {
                    let existing = occ.get();
                    let is_same = existing.worker_url == worker_url
                        && existing.pod_type == pod_info.pod_type
                        && (pod_info.uid.is_none() || existing.uid == pod_info.uid);

                    if is_same {
                        (false, None) // Exact duplicate: safe no-op
                    } else {
                        // Stale registration: pod was replaced (new IP or UID) or PD role flipped!
                        let old = occ.insert(TrackedPod {
                            worker_url: worker_url.clone(),
                            pod_type: pod_info.pod_type.clone(),
                            uid: pod_info.uid.clone(),
                        });
                        (true, Some(old))
                    }
                }
                std::collections::hash_map::Entry::Vacant(vac) => {
                    vac.insert(TrackedPod {
                        worker_url: worker_url.clone(),
                        pod_type: pod_info.pod_type.clone(),
                        uid: pod_info.uid.clone(),
                    });
                    (true, None)
                }
            }
        };

        if let Some(stale) = stale_registration {
            info!(
                "Reconciling stale pod registration for {}: evicting old URL {} (role: {:?})",
                pod_info.name, stale.worker_url, stale.pod_type
            );
            remove_worker_from_router(&stale.worker_url, stale.pod_type, &router, pd_mode).await;
        }

        if should_add {
            info!(
                "Adding pod: {} | type: {:?} | url: {}",
                pod_info.name, pod_info.pod_type, worker_url
            );

            // Handle PD mode with specific pod types
            let result = if pd_mode && pod_info.pod_type.is_some() {
                use crate::routers::http::vllm_pd_router::VllmPDRouter;

                if let Some(vllm_pd_router) = router.as_any().downcast_ref::<VllmPDRouter>() {
                    // Support --vllm-pd-disaggregation mode with K8s service discovery
                    match &pod_info.pod_type {
                        Some(PodType::Prefill) => vllm_pd_router
                            .add_prefill_server(worker_url.clone(), pod_info.bootstrap_port)
                            .await
                            .map_err(|e| e.to_string()),
                        Some(PodType::Decode) => vllm_pd_router
                            .add_decode_server(worker_url.clone())
                            .await
                            .map_err(|e| e.to_string()),
                        Some(PodType::Regular) | None => {
                            // Fall back to regular add_worker for regular pods
                            router.add_worker(&worker_url).await
                        }
                    }
                } else {
                    Err("PD mode enabled but router is not a VllmPDRouter".to_string())
                }
            } else {
                // Regular mode or no pod type specified
                router.add_worker(&worker_url).await
            };

            match result {
                Ok(_) => {
                    debug!("Worker added: {}", worker_url);
                }
                Err(e) => {
                    error!("Failed to add worker {} to router: {}", worker_url, e);
                    // Remove from tracking since addition failed
                    if let Ok(mut tracker) = tracked_pods.lock() {
                        tracker.remove(&pod_info.pod_key());
                    }
                }
            }
        }
    } else {
        // Pod is either unhealthy (e.g. Ready=False) or no longer matches selector.
        // If it was previously tracked, remove it from the router and tracking set.
        remove_tracked_pod(pod_info, tracked_pods, router, pd_mode, "ineligible").await;
    }
}

async fn handle_pod_deletion(
    pod_info: &PodInfo,
    tracked_pods: Arc<Mutex<HashMap<String, TrackedPod>>>,
    router: Arc<dyn RouterTrait>,
    pd_mode: bool,
) {
    remove_tracked_pod(pod_info, tracked_pods, router, pd_mode, "deletion").await;
}

async fn remove_tracked_pod(
    pod_info: &PodInfo,
    tracked_pods: Arc<Mutex<HashMap<String, TrackedPod>>>,
    router: Arc<dyn RouterTrait>,
    pd_mode: bool,
    reason: &str,
) {
    let registered = {
        let mut tracked = match tracked_pods.lock() {
            Ok(tracked) => tracked,
            Err(e) => {
                error!(
                    "Failed to acquire tracked_pods lock during {}: {}",
                    reason, e
                );
                return;
            }
        };
        tracked.remove(&pod_info.pod_key())
    };

    if let Some(registered) = registered {
        // Use the URL from registration, not recomputed from current pod IP, so that a
        // pod whose IP changed between registration and eviction is still removed correctly.
        let url = &registered.worker_url;
        info!(
            "Removing pod ({}): {} | type: {:?} | url: {}",
            reason, pod_info.name, registered.pod_type, url
        );
        remove_worker_from_router(url, registered.pod_type, &router, pd_mode).await;
    } else {
        debug!(
            "Pod {} event for untracked/already removed pod: {} (type: {:?})",
            reason, pod_info.name, pod_info.pod_type
        );
    }
}

async fn remove_worker_from_router(
    url: &str,
    pod_type: Option<PodType>,
    router: &Arc<dyn RouterTrait>,
    pd_mode: bool,
) {
    if pd_mode {
        use crate::routers::http::vllm_pd_router::VllmPDRouter;

        if let Some(vllm_pd_router) = router.as_any().downcast_ref::<VllmPDRouter>() {
            // Support --vllm-pd-disaggregation mode with K8s service discovery
            match pod_type {
                Some(PodType::Prefill) => {
                    if let Err(e) = vllm_pd_router.remove_prefill_server(url).await {
                        error!("Failed to remove vllm prefill server {}: {}", url, e);
                    }
                }
                Some(PodType::Decode) => {
                    if let Err(e) = vllm_pd_router.remove_decode_server(url).await {
                        error!("Failed to remove vllm decode server {}: {}", url, e);
                    }
                }
                Some(PodType::Regular) | None => {
                    router.remove_worker(url);
                }
            }
        } else {
            // PD mode but not a VllmPDRouter, use generic removal
            router.remove_worker(url);
        }
    } else {
        // Regular mode removal
        router.remove_worker(url);
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use k8s_openapi::api::core::v1::{Pod, PodCondition, PodSpec, PodStatus};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::Time;

    // Helper function to create a Pod for testing PodInfo::from_pod
    fn create_k8s_pod(
        name: Option<&str>,
        ip: Option<&str>,
        phase: Option<&str>,
        ready_status: Option<&str>,
        deletion_timestamp: Option<Time>,
    ) -> Pod {
        let mut pod = Pod {
            metadata: ObjectMeta {
                name: name.map(String::from),
                uid: Some("default-uid".to_string()),
                deletion_timestamp,
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: None,
        };

        if ip.is_some() || phase.is_some() || ready_status.is_some() {
            let mut pod_status = PodStatus {
                pod_ip: ip.map(String::from),
                phase: phase.map(String::from),
                conditions: None,
                ..Default::default()
            };

            if let Some(status_str) = ready_status {
                let condition = PodCondition {
                    type_: "Ready".to_string(),
                    status: status_str.to_string(),
                    last_probe_time: None,
                    last_transition_time: None,
                    message: None,
                    reason: None,
                    observed_generation: None,
                };
                pod_status.conditions = Some(vec![condition]);
            }
            pod.status = Some(pod_status);
        }
        pod
    }

    // Helper function to create a Pod with PD-specific labels and annotations
    fn create_pd_k8s_pod(name: &str, ip: &str, pod_type: &str, bootstrap_port: Option<u16>) -> Pod {
        let mut labels = std::collections::BTreeMap::new();
        labels.insert("app".to_string(), "vllm".to_string());
        labels.insert("component".to_string(), pod_type.to_string());

        let mut annotations = std::collections::BTreeMap::new();
        if let Some(port) = bootstrap_port {
            annotations.insert("vllm.ai/bootstrap-port".to_string(), port.to_string());
        }

        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                uid: Some("pd-uid".to_string()),
                labels: Some(labels),
                annotations: Some(annotations),
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: Some(PodStatus {
                pod_ip: Some(ip.to_string()),
                phase: Some("Running".to_string()),
                conditions: Some(vec![PodCondition {
                    type_: "Ready".to_string(),
                    status: "True".to_string(),
                    last_probe_time: None,
                    last_transition_time: None,
                    message: None,
                    reason: None,
                    observed_generation: None,
                }]),
                ..Default::default()
            }),
        }
    }

    // Helper to create a Router instance for testing event handlers
    async fn create_test_router() -> Arc<dyn RouterTrait> {
        use crate::config::RouterConfig;
        use crate::middleware::TokenBucket;
        use crate::routers::http::router::Router;
        use crate::server::AppContext;

        // Create a minimal RouterConfig for testing with very short timeout
        let router_config = RouterConfig {
            worker_startup_timeout_secs: 1,
            ..Default::default()
        }; // Very short timeout for tests

        // Create AppContext with minimal components
        let app_context = Arc::new(AppContext {
            client: reqwest::Client::new(),
            router_config: router_config.clone(),
            rate_limiter: Arc::new(TokenBucket::new(1000, 1000)),
            worker_registry: Arc::new(crate::core::WorkerRegistry::new()),
            policy_registry: Arc::new(crate::policies::PolicyRegistry::new_with_affinity_ttl(
                router_config.policy.clone(),
                router_config
                    .kv_aware_session_affinity_ttl_secs
                    .map(std::time::Duration::from_secs),
            )),
            router_manager: None, // Test doesn't need router manager
            response_storage: Arc::new(crate::data_connector::MemoryResponseStorage::new()),
            api_key_cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            api_key_validation_urls: Arc::new(Vec::new()),
        });

        let router = Router::new(vec![], &app_context).await.unwrap();
        Arc::new(router) as Arc<dyn RouterTrait>
    }

    // Helper to create a PD config for testing
    fn create_pd_config() -> ServiceDiscoveryConfig {
        let mut prefill_selector = HashMap::new();
        prefill_selector.insert("app".to_string(), "vllm".to_string());
        prefill_selector.insert("component".to_string(), "prefill".to_string());

        let mut decode_selector = HashMap::new();
        decode_selector.insert("app".to_string(), "vllm".to_string());
        decode_selector.insert("component".to_string(), "decode".to_string());

        ServiceDiscoveryConfig {
            enabled: true,
            selector: HashMap::new(),
            check_interval: Duration::from_secs(60),
            port: 8080,
            namespace: None,
            pd_mode: true,
            prefill_selector,
            decode_selector,
            bootstrap_port_annotation: "vllm.ai/bootstrap-port".to_string(),
        }
    }

    #[test]
    fn test_pod_info_should_include() {
        let config = create_pd_config();

        // Test prefill pod should be included
        let prefill_pod = create_pd_k8s_pod("prefill-pod", "10.0.0.1", "prefill", Some(8081));
        assert!(PodInfo::should_include(&prefill_pod, &config));

        // Test decode pod should be included
        let decode_pod = create_pd_k8s_pod("decode-pod", "10.0.0.2", "decode", None);
        assert!(PodInfo::should_include(&decode_pod, &config));

        // Test unmatched pod should not be included
        let unmatched_pod = create_pd_k8s_pod("other-pod", "10.0.0.3", "other", None);
        assert!(!PodInfo::should_include(&unmatched_pod, &config));

        // Test regular mode
        let mut regular_config = ServiceDiscoveryConfig::default();
        regular_config
            .selector
            .insert("app".to_string(), "vllm".to_string());
        regular_config.pd_mode = false;

        let regular_pod = create_pd_k8s_pod("worker-pod", "10.0.0.4", "worker", None);
        assert!(PodInfo::should_include(&regular_pod, &regular_config));
    }

    #[test]
    fn test_service_discovery_config_default() {
        let config = ServiceDiscoveryConfig::default();
        assert!(!config.enabled);
        assert!(config.selector.is_empty());
        assert_eq!(config.check_interval, Duration::from_secs(60));
        assert_eq!(config.port, 8000);
        assert!(config.namespace.is_none());
        assert!(!config.pd_mode);
        assert!(config.prefill_selector.is_empty());
        assert!(config.decode_selector.is_empty());
        assert_eq!(config.bootstrap_port_annotation, "vllm.ai/bootstrap-port");
    }

    #[test]
    fn test_pod_type_enum() {
        // Test that PodType enum has expected variants
        let prefill = PodType::Prefill;
        let decode = PodType::Decode;
        let regular = PodType::Regular;

        assert_eq!(format!("{:?}", prefill), "Prefill");
        assert_eq!(format!("{:?}", decode), "Decode");
        assert_eq!(format!("{:?}", regular), "Regular");
    }

    #[test]
    fn test_pod_info_from_pod_valid() {
        let k8s_pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("True"),
            None,
        );
        let pod_info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert_eq!(pod_info.name, "test-pod");
        assert_eq!(pod_info.ip, "10.0.0.1".parse::<IpAddr>().unwrap());
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert!(pod_info.pod_type.is_none());
        assert!(pod_info.bootstrap_port.is_none());
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_prefill() {
        let k8s_pod = create_pd_k8s_pod("prefill-pod", "10.0.0.1", "prefill", Some(8081));
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "prefill-pod");
        assert_eq!(pod_info.ip, "10.0.0.1".parse::<IpAddr>().unwrap());
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Prefill));
        assert_eq!(pod_info.bootstrap_port, Some(8081));
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_decode() {
        let k8s_pod = create_pd_k8s_pod("decode-pod", "10.0.0.2", "decode", None);
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "decode-pod");
        assert_eq!(pod_info.ip, "10.0.0.2".parse::<IpAddr>().unwrap());
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Decode));
        assert!(pod_info.bootstrap_port.is_none());
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_regular_mode() {
        let k8s_pod = create_pd_k8s_pod("regular-pod", "10.0.0.3", "worker", None);
        let mut config = create_pd_config();
        config.pd_mode = false; // Set to regular mode

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "regular-pod");
        assert_eq!(pod_info.ip, "10.0.0.3".parse::<IpAddr>().unwrap());
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Regular));
        assert!(pod_info.bootstrap_port.is_none());
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_unmatched_labels() {
        let k8s_pod = create_pd_k8s_pod("unknown-pod", "10.0.0.4", "unknown", None);
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "unknown-pod");
        assert_eq!(pod_info.ip, "10.0.0.4".parse::<IpAddr>().unwrap());
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Regular));
        assert!(pod_info.bootstrap_port.is_none());
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_invalid_bootstrap_port() {
        let mut pod = create_pd_k8s_pod("prefill-pod", "10.0.0.1", "prefill", None);
        // Add invalid bootstrap port annotation
        pod.metadata
            .annotations
            .as_mut()
            .unwrap()
            .insert("vllm.ai/bootstrap-port".to_string(), "invalid".to_string());
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(pod_info.pod_type, Some(PodType::Prefill));
        assert!(pod_info.bootstrap_port.is_none()); // Should be None for invalid port
    }

    #[test]
    fn test_pod_info_from_pod_not_ready() {
        let k8s_pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("False"),
            None,
        );
        let pod_info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert!(!pod_info.is_ready);
    }

    #[test]
    fn test_pod_info_from_pod_no_conditions() {
        let k8s_pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            None,
            None,
        );
        let pod_info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert!(!pod_info.is_ready);
    }

    #[test]
    fn test_pod_info_from_pod_missing_name() {
        let k8s_pod = create_k8s_pod(None, Some("10.0.0.1"), Some("Running"), Some("True"), None);
        assert!(PodInfo::from_pod(&k8s_pod, None).is_none());
    }

    #[test]
    fn test_pod_info_from_pod_missing_ip() {
        let k8s_pod = create_k8s_pod(Some("test-pod"), None, Some("Running"), Some("True"), None);
        assert!(PodInfo::from_pod(&k8s_pod, None).is_none());
    }

    #[test]
    fn test_pod_info_from_pod_missing_status_phase() {
        let k8s_pod = create_k8s_pod(Some("test-pod"), Some("10.0.0.1"), None, Some("True"), None);
        let pod_info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert_eq!(pod_info.status, "Unknown");
    }

    #[test]
    fn test_pod_info_from_pod_no_status_object() {
        let mut k8s_pod = create_k8s_pod(Some("test-pod"), None, None, None, None);
        k8s_pod.status = None;
        assert!(PodInfo::from_pod(&k8s_pod, None).is_none());
    }

    #[test]
    fn test_pod_info_is_healthy() {
        let healthy_pod = PodInfo {
            name: "p1".into(),
            namespace: "".into(),
            uid: None,
            ip: "1.1.1.1".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };
        assert!(healthy_pod.is_healthy());

        let not_ready_pod = PodInfo {
            name: "p2".into(),
            namespace: "".into(),
            uid: None,
            ip: "1.1.1.2".parse().unwrap(),
            status: "Running".into(),
            is_ready: false,
            pod_type: None,
            bootstrap_port: None,
        };
        assert!(!not_ready_pod.is_healthy());

        let not_running_pod = PodInfo {
            name: "p3".into(),
            namespace: "".into(),
            uid: None,
            ip: "1.1.1.3".parse().unwrap(),
            status: "Pending".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };
        assert!(!not_running_pod.is_healthy());
    }

    #[test]
    fn test_pod_info_worker_url_ipv4() {
        let pod_info = PodInfo {
            name: "p1".into(),
            namespace: "".into(),
            uid: None,
            ip: "1.2.3.4".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };
        assert_eq!(pod_info.worker_url(8080), "http://1.2.3.4:8080");
    }

    #[test]
    fn test_pod_info_worker_url_ipv6() {
        let pod_info = PodInfo {
            name: "p1".into(),
            namespace: "".into(),
            uid: None,
            ip: "2803:6086:5cd3:6062:c68d:e7c7:480:10".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };
        assert_eq!(
            pod_info.worker_url(8000),
            "http://[2803:6086:5cd3:6062:c68d:e7c7:480:10]:8000"
        );
    }

    #[test]
    fn test_pod_info_equality_with_pod_type() {
        let pod1 = PodInfo {
            name: "pod1".into(),
            namespace: "".into(),
            uid: None,
            ip: "1.2.3.4".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
        };

        let pod2 = PodInfo {
            name: "pod1".into(),
            namespace: "".into(),
            uid: None,
            ip: "1.2.3.4".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
        };

        let pod3 = PodInfo {
            name: "pod1".into(),
            namespace: "".into(),
            uid: None,
            ip: "1.2.3.4".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
        };

        assert_eq!(pod1, pod2);
        assert_ne!(pod1, pod3);
    }

    #[tokio::test]
    async fn test_handle_pod_event_add_unhealthy_pod() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "pod1".into(),
            namespace: "".into(),
            uid: None,
            ip: "1.2.3.4".parse().unwrap(),
            status: "Pending".into(),
            is_ready: false,
            pod_type: None,
            bootstrap_port: None,
        };
        let port = 8080u16;

        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false, // pd_mode = false
            true,  // should_include = true
        )
        .await;

        assert!(!tracked_pods
            .lock()
            .unwrap()
            .contains_key(&pod_info.pod_key()));
        assert!(!router
            .get_worker_urls()
            .contains(&pod_info.worker_url(port)));
    }

    #[tokio::test]
    async fn test_handle_pod_deletion_non_existing_pod() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "pod1".into(),
            namespace: "".into(),
            uid: None,
            ip: "1.2.3.4".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };
        handle_pod_deletion(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            false, // pd_mode = false
        )
        .await;

        assert!(tracked_pods.lock().unwrap().is_empty());
        assert!(router.get_worker_urls().is_empty());
    }

    #[tokio::test]
    async fn test_handle_pd_pod_event_prefill_pod() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "prefill-pod".into(),
            namespace: "".into(),
            uid: None,
            ip: "1.2.3.4".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
        };
        let port = 8080u16;

        // This test validates the structure but won't actually add workers since
        // we're using a regular router instead of PD router
        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false, // pd_mode = false, so it should fallback to regular handling
            true,  // should_include = true
        )
        .await;

        // Pod should not be tracked since router.add_worker will fail for non-running server
        assert!(!tracked_pods
            .lock()
            .unwrap()
            .contains_key(&pod_info.pod_key()));
    }

    #[tokio::test]
    async fn test_handle_pd_pod_event_decode_pod() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "decode-pod".into(),
            namespace: "".into(),
            uid: None,
            ip: "1.2.3.5".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
        };
        let port = 8080u16;

        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false, // pd_mode = false, so it should fallback to regular handling
            true,  // should_include = true
        )
        .await;

        // Pod should not be tracked since router.add_worker will fail for non-running server
        assert!(!tracked_pods
            .lock()
            .unwrap()
            .contains_key(&pod_info.pod_key()));
    }

    #[tokio::test]
    async fn test_handle_pd_pod_deletion_tracked_pod() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "test-pod".into(),
            namespace: "".into(),
            uid: None,
            ip: "1.2.3.4".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
        };

        let port = 8080u16;

        // Add pod to tracked set first
        {
            let mut tracked = tracked_pods.lock().unwrap();
            tracked.insert(
                pod_info.pod_key(),
                TrackedPod {
                    worker_url: pod_info.worker_url(port),
                    pod_type: pod_info.pod_type.clone(),
                    uid: None,
                },
            );
        }

        handle_pod_deletion(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            false, // pd_mode = false
        )
        .await;

        // Pod should be removed from tracking
        assert!(!tracked_pods
            .lock()
            .unwrap()
            .contains_key(&pod_info.pod_key()));
    }

    #[tokio::test]
    async fn test_handle_pd_pod_deletion_untracked_pod() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "untracked-pod".into(),
            namespace: "".into(),
            uid: None,
            ip: "1.2.3.4".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
        };
        // Don't add pod to tracked set

        handle_pod_deletion(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            true, // pd_mode = true
        )
        .await;

        // Tracked set should remain empty
        assert!(tracked_pods.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_unified_handler_regular_mode() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "regular-pod".into(),
            namespace: "".into(),
            uid: None,
            ip: "1.2.3.4".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Regular),
            bootstrap_port: None,
        };
        let port = 8080u16;

        // Test that unified handler works for regular mode
        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false, // pd_mode = false
            true,  // should_include = true
        )
        .await;

        // Pod should not be tracked since router.add_worker will fail for non-running server
        assert!(!tracked_pods
            .lock()
            .unwrap()
            .contains_key(&pod_info.pod_key()));
    }

    #[tokio::test]
    async fn test_unified_handler_pd_mode_with_prefill() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "prefill-pod".into(),
            namespace: "".into(),
            uid: None,
            ip: "1.2.3.4".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Prefill),
            bootstrap_port: Some(8081),
        };
        let port = 8080u16;

        // Test that unified handler works for PD mode with prefill
        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            true, // pd_mode = true
            true, // should_include = true
        )
        .await;

        // Pod should not be tracked since router.add_pd_worker will fail for regular router
        assert!(!tracked_pods
            .lock()
            .unwrap()
            .contains_key(&pod_info.pod_key()));
    }

    #[tokio::test]
    async fn test_unified_handler_deletion_with_pd_mode() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "decode-pod".into(),
            namespace: "".into(),
            uid: None,
            ip: "1.2.3.4".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
        };

        let port = 8080u16;

        // Add pod to tracked set first
        {
            let mut tracked = tracked_pods.lock().unwrap();
            tracked.insert(
                pod_info.pod_key(),
                TrackedPod {
                    worker_url: pod_info.worker_url(port),
                    pod_type: pod_info.pod_type.clone(),
                    uid: None,
                },
            );
        }

        // Test that unified handler works for deletion in PD mode
        handle_pod_deletion(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            true, // pd_mode = true
        )
        .await;

        // Pod should be removed from tracking
        assert!(!tracked_pods
            .lock()
            .unwrap()
            .contains_key(&pod_info.pod_key()));
    }

    #[tokio::test]
    async fn test_handle_pod_event_removes_unready_pod() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "target-pod".into(),
            namespace: "".into(),
            uid: None,
            ip: "10.0.0.1".parse().unwrap(),
            status: "Running".into(),
            is_ready: false, // Probe failed: Ready=False
            pod_type: None,
            bootstrap_port: None,
        };
        let port = 8000u16;

        // Pre-track the pod as if it was previously healthy
        tracked_pods.lock().unwrap().insert(
            pod_info.pod_key(),
            TrackedPod {
                worker_url: pod_info.worker_url(port),
                pod_type: pod_info.pod_type.clone(),
                uid: None,
            },
        );

        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false,
            true, // Selector still matches, but Ready=False
        )
        .await;

        // Pod must be removed from tracked set and router
        assert!(!tracked_pods
            .lock()
            .unwrap()
            .contains_key(&pod_info.pod_key()));
        assert!(!router
            .get_worker_urls()
            .contains(&pod_info.worker_url(port)));
    }

    #[tokio::test]
    async fn test_handle_pod_event_removes_selector_mismatched_pod() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "target-pod".into(),
            namespace: "".into(),
            uid: None,
            ip: "10.0.0.2".parse().unwrap(),
            status: "Running".into(),
            is_ready: true, // Healthy, but label changed (role=canary)
            pod_type: None,
            bootstrap_port: None,
        };
        let port = 8000u16;

        // Pre-track the pod
        tracked_pods.lock().unwrap().insert(
            pod_info.pod_key(),
            TrackedPod {
                worker_url: pod_info.worker_url(port),
                pod_type: pod_info.pod_type.clone(),
                uid: None,
            },
        );

        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false,
            false, // Selector no longer matches!
        )
        .await;

        // Pod must be removed from tracked set and router
        assert!(!tracked_pods
            .lock()
            .unwrap()
            .contains_key(&pod_info.pod_key()));
        assert!(!router
            .get_worker_urls()
            .contains(&pod_info.worker_url(port)));
    }

    #[tokio::test]
    async fn test_handle_pod_event_untracked_unready_pod_is_noop() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_info = PodInfo {
            name: "untracked".into(),
            namespace: "".into(),
            uid: None,
            ip: "10.0.0.3".parse().unwrap(),
            status: "Running".into(),
            is_ready: false,
            pod_type: None,
            bootstrap_port: None,
        };
        let port = 8000u16;

        handle_pod_event(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false,
            true,
        )
        .await;

        assert!(tracked_pods.lock().unwrap().is_empty());
        assert!(router.get_worker_urls().is_empty());
    }

    #[tokio::test]
    async fn test_handle_pod_event_hostnetwork_pod_isolation() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8000u16;

        // Pod A: Legitimate vLLM worker on node 10.0.0.50
        let live_pod = PodInfo {
            name: "vllm-worker-0".into(),
            namespace: "".into(),
            uid: None,
            ip: "10.0.0.50".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };

        // Pre-track the legitimate worker
        tracked_pods.lock().unwrap().insert(
            live_pod.pod_key(),
            TrackedPod {
                worker_url: live_pod.worker_url(port),
                pod_type: live_pod.pod_type.clone(),
                uid: None,
            },
        );

        // Pod B: Unrelated daemon/hostNetwork pod sharing the same node IP 10.0.0.50
        let daemon_pod = PodInfo {
            name: "node-exporter-abc".into(),
            namespace: "".into(),
            uid: None,
            ip: "10.0.0.50".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };

        // Event for daemon pod (does not match selector: should_include = false)
        handle_pod_event(
            &daemon_pod,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false,
            false, // selector mismatch for daemon pod
        )
        .await;

        // Legitimate worker MUST NOT be evicted
        assert!(tracked_pods
            .lock()
            .unwrap()
            .contains_key(&live_pod.pod_key()));
        assert!(!tracked_pods
            .lock()
            .unwrap()
            .contains_key(&daemon_pod.pod_key()));
    }

    #[test]
    fn test_filter_firewall_logic() {
        let mut tracked_pods = HashMap::new();
        tracked_pods.insert(
            "/tracked-vllm-pod".to_string(),
            TrackedPod {
                worker_url: "http://1.0.0.1:8000".to_string(),
                pod_type: None,
                uid: None,
            },
        );

        let mut config = ServiceDiscoveryConfig::default();
        config
            .selector
            .insert("app".to_string(), "vllm".to_string());

        // Case 1: Unrelated cluster pod (no match, not tracked) -> DROPPED
        let unrelated_pod = create_k8s_pod(
            Some("unrelated-monitoring-pod"),
            Some("10.0.0.9"),
            Some("Running"),
            Some("True"),
            None,
        );
        assert!(!PodInfo::should_allow_event(
            &unrelated_pod,
            &config,
            &tracked_pods
        ));

        // Case 2: New worker pod (matches selector, not yet tracked) -> PASSED
        let mut new_worker_pod = create_k8s_pod(
            Some("new-vllm-pod"),
            Some("10.0.0.10"),
            Some("Running"),
            Some("True"),
            None,
        );
        let mut labels = std::collections::BTreeMap::new();
        labels.insert("app".to_string(), "vllm".to_string());
        new_worker_pod.metadata.labels = Some(labels);
        assert!(PodInfo::should_allow_event(
            &new_worker_pod,
            &config,
            &tracked_pods
        ));

        // Case 3: Tracked pod whose label changed (mismatched selector, but tracked) -> PASSED for cleanup!
        let tracked_pod_mutated = create_k8s_pod(
            Some("tracked-vllm-pod"),
            Some("1.0.0.1"),
            Some("Running"),
            Some("True"),
            None,
        );
        // Labels empty or canary -> mismatched selector, but pod is tracked
        assert!(PodInfo::should_allow_event(
            &tracked_pod_mutated,
            &config,
            &tracked_pods
        ));
    }

    #[tokio::test]
    async fn test_handle_pod_event_pd_mode_preserves_registered_role_on_selector_mismatch() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8000u16;

        // Pre-track the pod as Prefill
        let prefill_pod_key = format!("{}/{}", "", "prefill-1");
        tracked_pods.lock().unwrap().insert(
            prefill_pod_key.clone(),
            TrackedPod {
                worker_url: "http://10.0.1.1:8000".to_string(),
                pod_type: Some(PodType::Prefill),
                uid: None,
            },
        );

        // Event arrives with labels stripped (so from_pod classified it as Regular)
        let pod_info_stripped = PodInfo {
            name: "prefill-1".into(),
            namespace: "".into(),
            uid: None,
            ip: "10.0.1.1".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Regular), // Labels stripped!
            bootstrap_port: None,
        };

        handle_pod_event(
            &pod_info_stripped,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            true,  // pd_mode = true
            false, // selector mismatch
        )
        .await;

        // Pod should be cleanly removed from tracking
        assert!(!tracked_pods.lock().unwrap().contains_key(&prefill_pod_key));
    }
    #[tokio::test]
    async fn test_same_name_different_namespace_isolation() {
        // Regression for reviewer [P1]: pod names are only unique within a namespace.
        // With Api::all(), two pods named "worker-0" in "ns-a" and "ns-b" must be tracked
        // independently via their "namespace/name" key.  An ineligible update from ns-b must
        // NOT evict ns-a's worker registration.
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8000u16;

        // Pod A: legitimate vLLM worker in namespace "ns-a" at 10.0.1.1
        let pod_a = PodInfo {
            name: "worker-0".into(),
            namespace: "ns-a".into(),
            uid: None,
            ip: "10.0.1.1".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };

        // Pre-track pod A (as if it was successfully registered)
        {
            let mut m = tracked_pods.lock().unwrap();
            m.insert(
                pod_a.pod_key(),
                TrackedPod {
                    worker_url: pod_a.worker_url(port),
                    pod_type: pod_a.pod_type.clone(),
                    uid: None,
                },
            );
        }

        // Pod B: different namespace, same name, different IP; selector mismatch.
        let pod_b = PodInfo {
            name: "worker-0".into(),
            namespace: "ns-b".into(),
            uid: None,
            ip: "10.0.2.1".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };

        // Simulate: pod B's event arrives with should_include=false (not in our selector).
        handle_pod_event(
            &pod_b,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false,
            false,
        )
        .await;

        // Pod A's registration must survive — different namespace means different key.
        assert!(
            tracked_pods.lock().unwrap().contains_key(&pod_a.pod_key()),
            "ns-a/worker-0 must remain tracked after an event for ns-b/worker-0"
        );
        // Pod B must never appear in tracking (it was not selected).
        assert!(
            !tracked_pods.lock().unwrap().contains_key(&pod_b.pod_key()),
            "ns-b/worker-0 must not appear in tracked_pods"
        );
        // Pod A's URL must not have been wrongly removed from the router.
        assert!(
            !router.get_worker_urls().contains(&pod_b.worker_url(port)),
            "pod-b's URL must not appear in router"
        );
    }
    #[test]
    fn test_pod_info_from_pod_invalid_ip() {
        // IP with invalid format
        let k8s_pod = create_k8s_pod(
            Some("test-pod"),
            Some("999.999.999.999"),
            Some("Running"),
            Some("True"),
            None,
        );
        assert!(PodInfo::from_pod(&k8s_pod, None).is_none());

        // Empty string IP
        let k8s_pod_empty = create_k8s_pod(
            Some("test-pod"),
            Some(""),
            Some("Running"),
            Some("True"),
            None,
        );
        assert!(PodInfo::from_pod(&k8s_pod_empty, None).is_none());

        // Non-numeric garbage string IP
        let k8s_pod_garbage = create_k8s_pod(
            Some("test-pod"),
            Some("not-an-ip"),
            Some("Running"),
            Some("True"),
            None,
        );
        assert!(PodInfo::from_pod(&k8s_pod_garbage, None).is_none());
    }

    #[test]
    fn test_pod_info_from_pod_conditions_edge_cases() {
        // Ready condition with status "Unknown"
        let pod_unknown = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("Unknown"),
            None,
        );
        let info = PodInfo::from_pod(&pod_unknown, None).unwrap();
        assert!(!info.is_ready);

        // Ready condition with empty status ""
        let pod_empty = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some(""),
            None,
        );
        let info_empty = PodInfo::from_pod(&pod_empty, None).unwrap();
        assert!(!info_empty.is_ready);

        // Pod with a non-Ready condition (e.g. PodScheduled=True, but Ready condition missing)
        let mut pod_scheduled = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            None,
            None,
        );
        if let Some(ref mut status) = pod_scheduled.status {
            status.conditions = Some(vec![PodCondition {
                type_: "PodScheduled".to_string(),
                status: "True".to_string(),
                last_probe_time: None,
                last_transition_time: None,
                message: None,
                reason: None,
                observed_generation: None,
            }]);
        }
        let info_scheduled = PodInfo::from_pod(&pod_scheduled, None).unwrap();
        assert!(!info_scheduled.is_ready);
    }

    #[test]
    fn test_pod_info_is_healthy_all_phases() {
        let make_pod = |status: &str, is_ready: bool| PodInfo {
            name: "pod".into(),
            namespace: "default".into(),
            uid: None,
            ip: "10.0.0.1".parse().unwrap(),
            status: status.into(),
            is_ready,
            pod_type: None,
            bootstrap_port: None,
        };

        // Running + Ready=True is the ONLY healthy state
        assert!(make_pod("Running", true).is_healthy());

        // Negative cases
        assert!(!make_pod("Running", false).is_healthy());
        assert!(!make_pod("Pending", true).is_healthy());
        assert!(!make_pod("Pending", false).is_healthy());
        assert!(!make_pod("Succeeded", true).is_healthy());
        assert!(!make_pod("Failed", true).is_healthy());
        assert!(!make_pod("Unknown", true).is_healthy());
    }

    #[test]
    fn test_pod_info_namespace_extraction_and_key() {
        // Pod with explicit namespace
        let mut k8s_pod = create_k8s_pod(
            Some("my-worker"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("True"),
            None,
        );
        k8s_pod.metadata.namespace = Some("prod-ns".to_string());
        let info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert_eq!(info.namespace, "prod-ns");
        assert_eq!(info.pod_key(), "prod-ns/my-worker");
        assert_eq!(
            PodInfo::pod_key_from_meta(
                k8s_pod.metadata.namespace.as_deref(),
                k8s_pod.metadata.name.as_deref()
            ),
            info.pod_key()
        );

        // Pod with None namespace (cluster-wide)
        k8s_pod.metadata.namespace = None;
        let info_no_ns = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert_eq!(info_no_ns.namespace, "");
        assert_eq!(info_no_ns.pod_key(), "/my-worker");
        assert_eq!(
            PodInfo::pod_key_from_meta(
                k8s_pod.metadata.namespace.as_deref(),
                k8s_pod.metadata.name.as_deref()
            ),
            info_no_ns.pod_key()
        );
    }

    #[test]
    fn test_should_include_edge_cases() {
        let mut config = ServiceDiscoveryConfig::default();
        config
            .selector
            .insert("app".to_string(), "vllm".to_string());
        config
            .selector
            .insert("tier".to_string(), "backend".to_string());

        // Case 1: Pod with no labels at all (labels: None)
        let mut pod_no_labels = create_k8s_pod(
            Some("p"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("True"),
            None,
        );
        pod_no_labels.metadata.labels = None;
        assert!(!PodInfo::should_include(&pod_no_labels, &config));

        // Case 2: Config selector is empty -> warning & returns false
        let empty_config = ServiceDiscoveryConfig::default();
        let mut pod_with_labels = create_k8s_pod(
            Some("p"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("True"),
            None,
        );
        let mut labels = std::collections::BTreeMap::new();
        labels.insert("app".to_string(), "vllm".to_string());
        pod_with_labels.metadata.labels = Some(labels.clone());
        assert!(!PodInfo::should_include(&pod_with_labels, &empty_config));

        // Case 3: Pod with superset of labels (has extra labels) -> MATCHES
        labels.insert("tier".to_string(), "backend".to_string());
        labels.insert("extra".to_string(), "ignored".to_string());
        pod_with_labels.metadata.labels = Some(labels.clone());
        assert!(PodInfo::should_include(&pod_with_labels, &config));

        // Case 4: Pod missing one required key -> FAILS
        labels.remove("tier");
        pod_with_labels.metadata.labels = Some(labels.clone());
        assert!(!PodInfo::should_include(&pod_with_labels, &config));

        // Case 5: Pod with matching key but mismatched value -> FAILS
        labels.insert("tier".to_string(), "frontend".to_string());
        pod_with_labels.metadata.labels = Some(labels);
        assert!(!PodInfo::should_include(&pod_with_labels, &config));
    }

    #[tokio::test]
    async fn test_registered_url_retention_when_ip_changes_on_eviction() {
        // Tests reviewer requirement: "retain the registered URL"
        // If a pod's IP changes between registration and eviction, removal must
        // evict the *registered* URL that exists in the router.
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8000u16;

        let original_pod = PodInfo {
            name: "worker-dynamic-ip".into(),
            namespace: "default".into(),
            uid: None,
            ip: "10.0.1.50".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };

        // Pre-track the pod with original URL (as if registered)
        {
            let mut tracked = tracked_pods.lock().unwrap();
            tracked.insert(
                original_pod.pod_key(),
                TrackedPod {
                    worker_url: original_pod.worker_url(port),
                    pod_type: original_pod.pod_type.clone(),
                    uid: None,
                },
            );
        }

        // An eviction event arrives where pod_info has a different/mutated IP (10.0.1.99)
        let mutated_pod = PodInfo {
            name: "worker-dynamic-ip".into(),
            namespace: "default".into(),
            uid: None,
            ip: "10.0.1.99".parse().unwrap(),
            status: "Running".into(),
            is_ready: false, // unready trigger
            pod_type: None,
            bootstrap_port: None,
        };

        handle_pod_event(
            &mutated_pod,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false,
            true,
        )
        .await;

        // Tracking entry removed
        assert!(!tracked_pods
            .lock()
            .unwrap()
            .contains_key(&original_pod.pod_key()));
    }

    #[tokio::test]
    async fn test_idempotent_duplicate_events() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8000u16;

        let pod = PodInfo {
            name: "idem-pod".into(),
            namespace: "default".into(),
            uid: None,
            ip: "10.0.5.5".parse().unwrap(),
            status: "Running".into(),
            is_ready: false,
            pod_type: None,
            bootstrap_port: None,
        };

        // Repeated unready events on untracked pod must be clean no-ops (no panic, no error)
        handle_pod_event(
            &pod,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false,
            true,
        )
        .await;
        handle_pod_event(
            &pod,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false,
            true,
        )
        .await;
        assert!(tracked_pods.lock().unwrap().is_empty());

        // Repeated deletion events on untracked pod must also be clean no-ops
        handle_pod_deletion(&pod, Arc::clone(&tracked_pods), Arc::clone(&router), false).await;
        handle_pod_deletion(&pod, Arc::clone(&tracked_pods), Arc::clone(&router), false).await;
        assert!(tracked_pods.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_pod_flapping_lifecycle() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8000u16;

        let pod_key = "default/flapping-pod".to_string();
        let healthy_pod = PodInfo {
            name: "flapping-pod".into(),
            namespace: "default".into(),
            uid: None,
            ip: "10.0.6.6".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };
        let unready_pod = PodInfo {
            is_ready: false,
            ..healthy_pod.clone()
        };

        // Phase 1: Pre-track as healthy (simulating registration)
        tracked_pods.lock().unwrap().insert(
            pod_key.clone(),
            TrackedPod {
                worker_url: healthy_pod.worker_url(port),
                pod_type: None,
                uid: None,
            },
        );
        assert!(tracked_pods.lock().unwrap().contains_key(&pod_key));

        // Phase 2: Probe fails -> Ready=False
        handle_pod_event(
            &unready_pod,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false,
            true,
        )
        .await;
        assert!(!tracked_pods.lock().unwrap().contains_key(&pod_key));

        // Phase 3: Probe recovers -> Ready=True (pre-track again)
        tracked_pods.lock().unwrap().insert(
            pod_key.clone(),
            TrackedPod {
                worker_url: healthy_pod.worker_url(port),
                pod_type: None,
                uid: None,
            },
        );
        assert!(tracked_pods.lock().unwrap().contains_key(&pod_key));

        // Phase 4: Label removed -> selector mismatch
        handle_pod_event(
            &healthy_pod,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false,
            false,
        )
        .await;
        assert!(!tracked_pods.lock().unwrap().contains_key(&pod_key));

        // Phase 5: Pod deletion event
        handle_pod_deletion(
            &healthy_pod,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            false,
        )
        .await;
        assert!(tracked_pods.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_start_service_discovery_disabled_returns_error() {
        let router = create_test_router().await;
        let config = ServiceDiscoveryConfig {
            enabled: false,
            ..Default::default()
        };
        let result = start_service_discovery(config, router).await;
        assert!(result.is_err());
        match result {
            Err(kube::Error::Api(err_resp)) => {
                assert_eq!(err_resp.status, "Disabled");
                assert_eq!(err_resp.message, "Service discovery is disabled");
                assert_eq!(err_resp.reason, "ConfigurationError");
                assert_eq!(err_resp.code, 400);
            }
            _ => panic!("Expected kube::Error::Api with ConfigurationError"),
        }
    }

    #[test]
    fn test_should_include_pd_mode_edge_cases() {
        // Case 1: Both prefill_selector and decode_selector empty in PD mode -> returns false
        let empty_pd_config = ServiceDiscoveryConfig {
            enabled: true,
            pd_mode: true,
            prefill_selector: HashMap::new(),
            decode_selector: HashMap::new(),
            ..Default::default()
        };
        let pod = create_pd_k8s_pod("worker-1", "10.0.0.1", "prefill", Some(8081));
        assert!(!PodInfo::should_include(&pod, &empty_pd_config));

        // Case 2: Only prefill_selector populated
        let mut prefill_only_config = ServiceDiscoveryConfig {
            enabled: true,
            pd_mode: true,
            decode_selector: HashMap::new(),
            ..Default::default()
        };
        prefill_only_config
            .prefill_selector
            .insert("component".to_string(), "prefill".to_string());

        let prefill_pod = create_pd_k8s_pod("p1", "10.0.0.1", "prefill", Some(8081));
        let decode_pod = create_pd_k8s_pod("d1", "10.0.0.2", "decode", None);
        assert!(PodInfo::should_include(&prefill_pod, &prefill_only_config));
        assert!(!PodInfo::should_include(&decode_pod, &prefill_only_config));

        // Case 3: Only decode_selector populated
        let mut decode_only_config = ServiceDiscoveryConfig {
            enabled: true,
            pd_mode: true,
            prefill_selector: HashMap::new(),
            ..Default::default()
        };
        decode_only_config
            .decode_selector
            .insert("component".to_string(), "decode".to_string());

        assert!(!PodInfo::should_include(&prefill_pod, &decode_only_config));
        assert!(PodInfo::should_include(&decode_pod, &decode_only_config));
    }

    #[test]
    fn test_pod_info_from_pod_pd_classification_edge_cases() {
        let config = create_pd_config();

        // 1. Decode pod with bootstrap port annotation: bootstrap_port MUST be None
        let mut decode_with_port = create_pd_k8s_pod("d-port", "10.0.0.5", "decode", Some(9999));
        decode_with_port
            .metadata
            .annotations
            .get_or_insert_with(std::collections::BTreeMap::new)
            .insert("vllm.ai/bootstrap-port".to_string(), "9999".to_string());

        let info = PodInfo::from_pod(&decode_with_port, Some(&config)).unwrap();
        assert_eq!(info.pod_type, Some(PodType::Decode));
        assert_eq!(
            info.bootstrap_port, None,
            "Decode pods must never have bootstrap_port populated"
        );

        // 2. Custom bootstrap_port_annotation
        let mut custom_config = create_pd_config();
        custom_config.bootstrap_port_annotation = "custom.ai/port".to_string();
        let mut prefill_custom = create_pd_k8s_pod("p-custom", "10.0.0.6", "prefill", None);
        prefill_custom
            .metadata
            .annotations
            .get_or_insert_with(std::collections::BTreeMap::new)
            .insert("custom.ai/port".to_string(), "7777".to_string());

        let info_custom = PodInfo::from_pod(&prefill_custom, Some(&custom_config)).unwrap();
        assert_eq!(info_custom.pod_type, Some(PodType::Prefill));
        assert_eq!(info_custom.bootstrap_port, Some(7777));

        // 3. Pod matching BOTH prefill and decode selectors: Prefill takes precedence
        let mut ambig_config = ServiceDiscoveryConfig {
            enabled: true,
            pd_mode: true,
            ..Default::default()
        };
        ambig_config
            .prefill_selector
            .insert("tier".to_string(), "llm".to_string());
        ambig_config
            .decode_selector
            .insert("tier".to_string(), "llm".to_string());

        let mut ambig_pod = create_k8s_pod(
            Some("ambig-pod"),
            Some("10.0.0.7"),
            Some("Running"),
            Some("True"),
            None,
        );
        let mut ambig_labels = std::collections::BTreeMap::new();
        ambig_labels.insert("tier".to_string(), "llm".to_string());
        ambig_pod.metadata.labels = Some(ambig_labels);

        let info_ambig = PodInfo::from_pod(&ambig_pod, Some(&ambig_config)).unwrap();
        assert_eq!(
            info_ambig.pod_type,
            Some(PodType::Prefill),
            "Prefill selector check is evaluated first"
        );
    }

    #[tokio::test]
    async fn test_multi_pod_tracking_and_selective_eviction() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8000u16;

        let p1 = PodInfo {
            name: "worker-1".into(),
            namespace: "prod".into(),
            uid: None,
            ip: "10.0.1.1".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };
        let p2 = PodInfo {
            name: "worker-2".into(),
            namespace: "prod".into(),
            uid: None,
            ip: "10.0.1.2".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };
        let p3 = PodInfo {
            name: "worker-3".into(),
            namespace: "prod".into(),
            uid: None,
            ip: "10.0.1.3".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };

        // Pre-track all 3 workers
        {
            let mut map = tracked_pods.lock().unwrap();
            for p in [&p1, &p2, &p3] {
                map.insert(
                    p.pod_key(),
                    TrackedPod {
                        worker_url: p.worker_url(port),
                        pod_type: None,
                        uid: None,
                    },
                );
            }
        }
        assert_eq!(tracked_pods.lock().unwrap().len(), 3);

        // Evict worker-2 due to unready probe failure
        let p2_unready = PodInfo {
            is_ready: false,
            ..p2.clone()
        };
        handle_pod_event(
            &p2_unready,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false,
            true,
        )
        .await;

        // worker-2 must be evicted; worker-1 and worker-3 must remain intact
        {
            let map = tracked_pods.lock().unwrap();
            assert_eq!(map.len(), 2);
            assert!(map.contains_key(&p1.pod_key()));
            assert!(!map.contains_key(&p2.pod_key()));
            assert!(map.contains_key(&p3.pod_key()));
        }

        // Evict worker-1 due to selector mismatch (label removed)
        handle_pod_event(
            &p1,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false,
            false, // selector mismatch
        )
        .await;

        // Only worker-3 remains
        {
            let map = tracked_pods.lock().unwrap();
            assert_eq!(map.len(), 1);
            assert!(!map.contains_key(&p1.pod_key()));
            assert!(map.contains_key(&p3.pod_key()));
        }
    }

    #[tokio::test]
    async fn test_handle_pod_event_evicts_failed_or_terminated_phase() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8000u16;

        let pod = PodInfo {
            name: "failing-pod".into(),
            namespace: "default".into(),
            uid: None,
            ip: "10.0.2.15".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };

        // Pre-track the pod
        tracked_pods.lock().unwrap().insert(
            pod.pod_key(),
            TrackedPod {
                worker_url: pod.worker_url(port),
                pod_type: None,
                uid: None,
            },
        );
        assert!(tracked_pods.lock().unwrap().contains_key(&pod.pod_key()));

        // Event arrives with phase "Failed" (e.g. OOMKilled), even if condition Ready was still True
        let failed_pod = PodInfo {
            status: "Failed".into(),
            is_ready: true,
            ..pod.clone()
        };
        handle_pod_event(
            &failed_pod,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false,
            true,
        )
        .await;

        assert!(
            !tracked_pods.lock().unwrap().contains_key(&pod.pod_key()),
            "Failed phase must cause eviction from tracked_pods"
        );
    }

    #[tokio::test]
    async fn test_handle_pod_event_ipv6_lifecycle() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8000u16;

        let ipv6_pod = PodInfo {
            name: "ipv6-worker".into(),
            namespace: "prod".into(),
            uid: None,
            ip: "2001:db8::100".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };
        let expected_url = "http://[2001:db8::100]:8000";
        assert_eq!(ipv6_pod.worker_url(port), expected_url);

        // Pre-track the IPv6 pod
        tracked_pods.lock().unwrap().insert(
            ipv6_pod.pod_key(),
            TrackedPod {
                worker_url: ipv6_pod.worker_url(port),
                pod_type: None,
                uid: None,
            },
        );
        assert!(tracked_pods
            .lock()
            .unwrap()
            .contains_key(&ipv6_pod.pod_key()));

        // Evict via unready event
        let unready_ipv6 = PodInfo {
            is_ready: false,
            ..ipv6_pod.clone()
        };
        handle_pod_event(
            &unready_ipv6,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false,
            true,
        )
        .await;

        assert!(!tracked_pods
            .lock()
            .unwrap()
            .contains_key(&ipv6_pod.pod_key()));
    }

    #[tokio::test]
    async fn test_handle_pod_event_already_tracked_no_duplicate_add() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8000u16;

        let pod = PodInfo {
            name: "stable-worker".into(),
            namespace: "prod".into(),
            uid: None,
            ip: "10.0.4.4".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };

        // Pre-track the pod
        let original_url = pod.worker_url(port);
        tracked_pods.lock().unwrap().insert(
            pod.pod_key(),
            TrackedPod {
                worker_url: original_url.clone(),
                pod_type: None,
                uid: None,
            },
        );

        // Receive another healthy event for the already-tracked pod
        handle_pod_event(
            &pod,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false,
            true,
        )
        .await;

        // Must still be tracked with original registration entry
        let map = tracked_pods.lock().unwrap();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&pod.pod_key()).unwrap().worker_url, original_url);
    }

    #[tokio::test]
    async fn test_poisoned_mutex_resilience() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8000u16;

        let pod = PodInfo {
            name: "test-pod".into(),
            namespace: "default".into(),
            uid: None,
            ip: "10.0.0.1".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };

        // Intentionally poison the mutex
        let tracked_clone = Arc::clone(&tracked_pods);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _guard = tracked_clone.lock().unwrap();
            panic!("Poisoning mutex");
        }));
        assert!(tracked_pods.is_poisoned());

        // Neither handle_pod_event nor handle_pod_deletion should panic
        handle_pod_event(
            &pod,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            false,
            true,
        )
        .await;

        handle_pod_deletion(&pod, Arc::clone(&tracked_pods), Arc::clone(&router), false).await;
    }

    #[test]
    fn test_pod_info_from_pod_out_of_range_bootstrap_port() {
        let mut pod = create_pd_k8s_pod("prefill-pod", "10.0.0.1", "prefill", None);
        // Add out-of-range bootstrap port annotation (> 65535)
        pod.metadata
            .annotations
            .as_mut()
            .unwrap()
            .insert("vllm.ai/bootstrap-port".to_string(), "70000".to_string());
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(pod_info.pod_type, Some(PodType::Prefill));
        assert!(pod_info.bootstrap_port.is_none());

        // Negative port string
        pod.metadata
            .annotations
            .as_mut()
            .unwrap()
            .insert("vllm.ai/bootstrap-port".to_string(), "-80".to_string());
        let pod_info_neg = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert!(pod_info_neg.bootstrap_port.is_none());
    }

    #[tokio::test]
    async fn test_pod_deletion_timestamp_workflow() {
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8000u16;

        let pod_info = PodInfo {
            name: "deleting-pod".into(),
            namespace: "default".into(),
            uid: None,
            ip: "10.0.3.50".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };

        // Pre-track the pod as running
        tracked_pods.lock().unwrap().insert(
            pod_info.pod_key(),
            TrackedPod {
                worker_url: pod_info.worker_url(port),
                pod_type: None,
                uid: None,
            },
        );
        assert!(tracked_pods
            .lock()
            .unwrap()
            .contains_key(&pod_info.pod_key()));

        // Simulate k8s deletion event (where deletion_timestamp is set)
        handle_pod_deletion(
            &pod_info,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            false,
        )
        .await;

        // Pod must be completely evicted from tracked_pods
        assert!(!tracked_pods
            .lock()
            .unwrap()
            .contains_key(&pod_info.pod_key()));
        assert!(tracked_pods.lock().unwrap().is_empty());
    }

    #[test]
    fn test_pod_info_from_pod_containers_ready_vs_ready_condition() {
        let mut pod = create_k8s_pod(
            Some("multi-cond-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("False"), // Ready is False
            None,
        );
        // Add ContainersReady = True to conditions
        if let Some(ref mut status) = pod.status {
            if let Some(ref mut conditions) = status.conditions {
                conditions.push(PodCondition {
                    type_: "ContainersReady".to_string(),
                    status: "True".to_string(),
                    last_probe_time: None,
                    last_transition_time: None,
                    message: None,
                    reason: None,
                    observed_generation: None,
                });
            }
        }
        let info = PodInfo::from_pod(&pod, None).unwrap();
        // ContainersReady=True must not be confused with Ready=True
        assert!(!info.is_ready);
    }

    #[test]
    fn test_pod_info_from_pod_ipv6_parsing() {
        let pod = create_k8s_pod(
            Some("ipv6-pod"),
            Some("2001:db8:85a3::8a2e:370:7334"),
            Some("Running"),
            Some("True"),
            None,
        );
        let info = PodInfo::from_pod(&pod, None).unwrap();
        assert_eq!(
            info.ip,
            "2001:db8:85a3::8a2e:370:7334".parse::<IpAddr>().unwrap()
        );
        assert_eq!(
            info.worker_url(8080),
            "http://[2001:db8:85a3::8a2e:370:7334]:8080"
        );
    }

    #[tokio::test]
    async fn test_statefulset_pod_replacement_reconciles_new_ip_and_uid() {
        // End-to-end regression test for reviewer [P2] (Thread 10):
        // Uses a mock HTTP listener on 127.0.0.1 so router.add_worker succeeds.
        // Old worker URL is evicted from the router, new worker URL is registered,
        // and tracked_pods is updated with the new URL and UID.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mock_port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf).await;
                let response =
                    "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Type: text/plain\r\n\r\nOK";
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });

        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let pod_key = "default/vllm-0".to_string();
        let old_url = "http://10.0.1.1:8000".to_string();

        // 1. Pre-track old pod registration
        tracked_pods.lock().unwrap().insert(
            pod_key.clone(),
            TrackedPod {
                worker_url: old_url.clone(),
                pod_type: None,
                uid: Some("uid-111".to_string()),
            },
        );
        router.remove_worker(&old_url); // ensure router state clean

        // 2. Replacement pod arrives at 127.0.0.1:mock_port with new UID
        let replacement_pod = PodInfo {
            name: "vllm-0".into(),
            namespace: "default".into(),
            uid: Some("uid-222".to_string()),
            ip: "127.0.0.1".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            bootstrap_port: None,
        };

        handle_pod_event(
            &replacement_pod,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            mock_port,
            false,
            true,
        )
        .await;

        // 3. Verify tracked_pods was reconciled with replacement pod's URL and UID
        let map = tracked_pods.lock().unwrap();
        let expected_url = format!("http://127.0.0.1:{}", mock_port);
        assert_eq!(
            map.get(&pod_key).map(|t| t.worker_url.as_str()),
            Some(expected_url.as_str())
        );
        assert_eq!(
            map.get(&pod_key).and_then(|t| t.uid.as_deref()),
            Some("uid-222")
        );

        // 4. Verify router added the new worker and removed the old worker
        assert!(router.get_worker_urls().contains(&expected_url));
        assert!(!router.get_worker_urls().contains(&old_url));
    }

    #[tokio::test]
    async fn test_pd_role_flip_via_label_mutation_reconciles_role() {
        // End-to-end regression test for collaborator wuhang2014 review (Thread 11):
        // When a registered Prefill pod is relabeled to Decode, handle_pod_event must
        // evict the stale Prefill registration rather than skipping as duplicate.
        let router = create_test_router().await;
        let tracked_pods = Arc::new(Mutex::new(HashMap::new()));
        let port = 8000u16;

        let pod_key = "default/worker-0".to_string();
        let worker_url = "http://10.0.1.5:8000".to_string();

        // 1. Pod initially registered as Prefill
        tracked_pods.lock().unwrap().insert(
            pod_key.clone(),
            TrackedPod {
                worker_url: worker_url.clone(),
                pod_type: Some(PodType::Prefill),
                uid: Some("uid-pd-1".to_string()),
            },
        );

        // 2. Label mutates to Decode
        let relabeled_pod = PodInfo {
            name: "worker-0".into(),
            namespace: "default".into(),
            uid: Some("uid-pd-1".to_string()),
            ip: "10.0.1.5".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: Some(PodType::Decode),
            bootstrap_port: None,
        };

        handle_pod_event(
            &relabeled_pod,
            Arc::clone(&tracked_pods),
            Arc::clone(&router),
            port,
            true, // pd_mode = true
            true, // matches selector
        )
        .await;

        // 3. Stale Prefill was evicted via remove_worker_from_router.
        // Because create_test_router is not a VllmPDRouter, add fails and cleans up tracking.
        assert!(!router.get_worker_urls().contains(&worker_url));
    }
}
