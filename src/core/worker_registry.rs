//! Worker Registry for multi-router support
//!
//! Provides centralized registry for workers with model-based indexing

use crate::{
    core::{ConnectionMode, Worker, WorkerType},
    metrics::RouterMetrics,
};
use dashmap::{mapref::entry::Entry, DashMap};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, RwLock,
};
use uuid::Uuid;

/// Unique identifier for a worker
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct WorkerId(String);

impl WorkerId {
    /// Create a new worker ID
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create a worker ID from a string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Get the ID as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for the model index to reduce complexity
type ModelIndex = Arc<DashMap<String, Arc<RwLock<Vec<Arc<dyn Worker>>>>>>;

/// Worker registry with model-based indexing
#[derive(Debug)]
pub struct WorkerRegistry {
    /// All workers indexed by ID
    workers: Arc<DashMap<WorkerId, Arc<dyn Worker>>>,

    /// Workers indexed by model ID (stores WorkerId for reference)
    model_workers: Arc<DashMap<String, Vec<WorkerId>>>,

    /// Optimized model index for O(1) lookups (stores Arc<dyn Worker> directly)
    model_index: ModelIndex,

    /// Workers indexed by worker type
    type_workers: Arc<DashMap<WorkerType, Vec<WorkerId>>>,

    /// Workers indexed by connection mode
    connection_workers: Arc<DashMap<ConnectionMode, Vec<WorkerId>>>,

    /// Registration order across worker types. Prefill types differ by
    /// bootstrap port, so the per-type lists cannot reconstruct --prefill order.
    registration_order: Arc<RwLock<Vec<WorkerId>>>,

    /// URL to worker ID mapping (for backward compatibility)
    url_to_id: Arc<DashMap<String, WorkerId>>,

    /// Monotonic topology and availability revision for read-side caches.
    revision: Arc<AtomicU64>,
}

impl WorkerRegistry {
    /// Create a new worker registry
    pub fn new() -> Self {
        Self {
            workers: Arc::new(DashMap::new()),
            model_workers: Arc::new(DashMap::new()),
            model_index: Arc::new(DashMap::new()),
            type_workers: Arc::new(DashMap::new()),
            connection_workers: Arc::new(DashMap::new()),
            registration_order: Arc::new(RwLock::new(Vec::new())),
            url_to_id: Arc::new(DashMap::new()),
            revision: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Register a new worker
    pub fn register(&self, worker: Arc<dyn Worker>) -> WorkerId {
        let (worker_id, is_new) = if let Some(existing_id) = self.url_to_id.get(worker.url()) {
            // Worker with this URL already exists, update it
            (existing_id.clone(), false)
        } else {
            (WorkerId::new(), true)
        };
        self.store_worker(&worker, &worker_id, is_new);
        self.url_to_id
            .insert(worker.url().to_string(), worker_id.clone());
        self.index_worker(&worker, &worker_id);
        worker_id
    }

    /// Install a lifecycle-owned target before any registry index can expose
    /// it; hold the URL entry until all indexes are current. No registry access
    /// is allowed in the callback.
    pub(crate) fn register_if_absent_before_publish(
        &self,
        worker: Arc<dyn Worker>,
        before_publish: impl FnOnce(&dyn Worker, &WorkerId),
    ) -> bool {
        let Entry::Vacant(entry) = self.url_to_id.entry(worker.url().to_owned()) else {
            return false;
        };
        let worker_id = WorkerId::new();
        before_publish(worker.as_ref(), &worker_id);
        self.store_worker(&worker, &worker_id, true);
        let _entry = entry.insert(worker_id.clone());
        self.index_worker(&worker, &worker_id);
        true
    }

    fn store_worker(&self, worker: &Arc<dyn Worker>, worker_id: &WorkerId, is_new: bool) {
        if is_new {
            self.registration_order
                .write()
                .expect("RwLock for registration_order is poisoned")
                .push(worker_id.clone());
        }

        // Store worker
        self.workers.insert(worker_id.clone(), worker.clone());
    }

    fn index_worker(&self, worker: &Arc<dyn Worker>, worker_id: &WorkerId) {
        // Update model index (both ID-based and optimized)
        let model_id = worker.model_id().to_string();
        self.model_workers
            .entry(model_id.clone())
            .or_default()
            .push(worker_id.clone());

        // Update optimized model index for O(1) lookups
        self.model_index
            .entry(model_id)
            .or_insert_with(|| Arc::new(RwLock::new(Vec::new())))
            .write()
            .expect("RwLock for model_index is poisoned")
            .push(worker.clone());

        // Update type index
        self.type_workers
            .entry(worker.worker_type())
            .or_default()
            .push(worker_id.clone());

        // Update connection mode index
        self.connection_workers
            .entry(worker.connection_mode())
            .or_default()
            .push(worker_id.clone());

        // Publish the initial state even if no health transition occurs.
        RouterMetrics::set_worker_health(worker.url(), worker.is_healthy());

        self.notify_worker_state_change();
    }

    /// Remove a worker by ID
    pub fn remove(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        let (_, worker) = self.workers.remove(worker_id)?;
        self.url_to_id.remove(worker.url());
        self.unindex_worker(&worker, worker_id);
        Some(worker)
    }

    fn unindex_worker(&self, worker: &Arc<dyn Worker>, worker_id: &WorkerId) {
        // Remove from model index (both ID-based and optimized)
        if let Some(mut model_workers) = self.model_workers.get_mut(worker.model_id()) {
            model_workers.retain(|id| id != worker_id);
        }

        // Remove from optimized model index
        if let Some(model_index_entry) = self.model_index.get(worker.model_id()) {
            let worker_url = worker.url();
            model_index_entry
                .write()
                .expect("RwLock for model_index is poisoned")
                .retain(|w| w.url() != worker_url);
        }

        // Remove from type index
        if let Some(mut type_workers) = self.type_workers.get_mut(&worker.worker_type()) {
            type_workers.retain(|id| id != worker_id);
        }
        self.registration_order
            .write()
            .expect("RwLock for registration_order is poisoned")
            .retain(|id| id != worker_id);

        // Remove from connection mode index
        if let Some(mut conn_workers) = self.connection_workers.get_mut(&worker.connection_mode()) {
            conn_workers.retain(|id| id != worker_id);
        }

        self.notify_worker_state_change();
    }

    /// Keep the URL unavailable for readmission until its registry indexes and
    /// lifecycle-owned target have retired. The callback may not access this
    /// registry.
    pub(crate) fn remove_by_url_with_cleanup(
        &self,
        url: &str,
        cleanup: impl FnOnce(&dyn Worker),
    ) -> Option<Arc<dyn Worker>> {
        let Entry::Occupied(entry) = self.url_to_id.entry(url.to_owned()) else {
            return None;
        };
        let (_, worker) = self.workers.remove(entry.get())?;
        self.unindex_worker(&worker, entry.get());
        cleanup(worker.as_ref());
        entry.remove();
        Some(worker)
    }

    /// Remove a worker by URL
    pub fn remove_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        if let Some((_, worker_id)) = self.url_to_id.remove(url) {
            self.remove(&worker_id)
        } else {
            None
        }
    }

    /// Get a worker by ID
    pub fn get(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        self.workers.get(worker_id).map(|entry| entry.clone())
    }

    /// Get a worker by URL
    pub fn get_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        self.url_to_id.get(url).and_then(|id| self.get(&id))
    }

    /// Fence a short registration update against removal/replacement of this
    /// exact routing Worker. The callback must not access the registry or await.
    pub(crate) fn with_current<T>(
        &self,
        worker: &Arc<dyn Worker>,
        operation: impl FnOnce(&WorkerId) -> T,
    ) -> Option<T> {
        // remove_by_url removes the URL mapping before it removes the ID.
        // Pin both entries until the callback completes.
        let id = self.url_to_id.get(worker.url())?;
        let current = self.workers.get(id.value())?;
        Arc::ptr_eq(current.value(), worker).then(|| operation(id.value()))
    }

    /// Get all workers for a model
    pub fn get_by_model(&self, model_id: &str) -> Vec<Arc<dyn Worker>> {
        self.model_workers
            .get(model_id)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all workers for a model (O(1) optimized version)
    /// This method uses the pre-indexed model_index for fast lookups
    pub fn get_by_model_fast(&self, model_id: &str) -> Vec<Arc<dyn Worker>> {
        self.model_index
            .get(model_id)
            .map(|workers| {
                workers
                    .read()
                    .expect("RwLock for model_index is poisoned")
                    .clone()
            })
            .unwrap_or_default()
    }

    /// Whether the optimized model index contains at least one worker.
    pub fn has_model(&self, model_id: &str) -> bool {
        self.model_index.get(model_id).is_some_and(|workers| {
            !workers
                .read()
                .expect("RwLock for model_index is poisoned")
                .is_empty()
        })
    }

    /// Current worker topology and availability revision.
    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    /// Invalidate target caches after topology or availability changes.
    pub fn notify_worker_state_change(&self) {
        self.revision.fetch_add(1, Ordering::AcqRel);
    }

    /// Get all workers by worker type
    pub fn get_by_type(&self, worker_type: &WorkerType) -> Vec<Arc<dyn Worker>> {
        self.type_workers
            .get(worker_type)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all prefill workers (regardless of bootstrap_port), in registration order.
    pub fn get_prefill_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.registration_order
            .read()
            .expect("RwLock for registration_order is poisoned")
            .iter()
            .filter_map(|id| {
                let worker = self.get(id)?;
                matches!(worker.worker_type(), WorkerType::Prefill { .. }).then_some(worker)
            })
            .collect()
    }

    /// Get all decode workers
    pub fn get_decode_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.get_by_type(&WorkerType::Decode)
    }

    /// Get all workers by connection mode
    pub fn get_by_connection(&self, connection_mode: &ConnectionMode) -> Vec<Arc<dyn Worker>> {
        self.connection_workers
            .get(connection_mode)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all workers
    pub fn get_all(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get all workers with their IDs
    pub fn get_all_with_ids(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        self.workers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Get all worker URLs
    pub fn get_all_urls(&self) -> Vec<String> {
        self.workers
            .iter()
            .map(|entry| entry.value().url().to_string())
            .collect()
    }

    /// Get all model IDs with workers
    pub fn get_models(&self) -> Vec<String> {
        self.model_workers
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get workers filtered by multiple criteria
    ///
    /// This method allows flexible filtering of workers based on:
    /// - model_id: Filter by specific model
    /// - worker_type: Filter by worker type (Regular, Prefill, Decode)
    /// - connection_mode: Filter by connection mode
    /// - healthy_only: Only return healthy workers
    pub fn get_workers_filtered(
        &self,
        model_id: Option<&str>,
        worker_type: Option<WorkerType>,
        connection_mode: Option<ConnectionMode>,
        healthy_only: bool,
    ) -> Vec<Arc<dyn Worker>> {
        // Start with the most efficient collection based on filters
        // Use model index when possible as it's O(1) lookup
        let workers = if let Some(model) = model_id {
            self.get_by_model_fast(model)
        } else {
            self.get_all()
        };

        // Apply remaining filters
        workers
            .into_iter()
            .filter(|w| {
                // Check worker_type if specified
                if let Some(ref wtype) = worker_type {
                    if w.worker_type() != *wtype {
                        return false;
                    }
                }

                // Check connection_mode if specified
                if let Some(ref conn) = connection_mode {
                    if w.connection_mode() != *conn {
                        return false;
                    }
                }

                // Check health if required
                if healthy_only && !w.is_healthy() {
                    return false;
                }

                true
            })
            .collect()
    }

    /// Get worker statistics
    pub fn stats(&self) -> WorkerRegistryStats {
        let total_workers = self.workers.len();
        let total_models = self.get_models().len();

        let mut healthy_count = 0;
        let mut total_load = 0;
        let mut regular_count = 0;
        let mut prefill_count = 0;
        let mut decode_count = 0;

        for worker in self.get_all() {
            if worker.is_healthy() {
                healthy_count += 1;
            }
            total_load += worker.load();

            match worker.worker_type() {
                WorkerType::Regular => regular_count += 1,
                WorkerType::Prefill { .. } => prefill_count += 1,
                WorkerType::Decode => decode_count += 1,
            }
        }

        WorkerRegistryStats {
            total_workers,
            total_models,
            healthy_workers: healthy_count,
            total_load,
            regular_workers: regular_count,
            prefill_workers: prefill_count,
            decode_workers: decode_count,
        }
    }

    /// Start a health checker for all workers in the registry
    /// This should be called once after the registry is populated with workers
    pub fn start_health_checker(&self, check_interval_secs: u64) -> crate::core::HealthChecker {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let workers_ref = self.workers.clone();
        let revision = self.revision.clone();

        let handle = tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(check_interval_secs));

            loop {
                interval.tick().await;

                // Check for shutdown signal
                if shutdown_clone.load(Ordering::Acquire) {
                    tracing::debug!("Registry health checker shutting down");
                    break;
                }

                // Get all workers from registry
                let workers: Vec<Arc<dyn crate::core::Worker>> = workers_ref
                    .iter()
                    .map(|entry| entry.value().clone())
                    .collect();

                // Perform health checks
                for worker in &workers {
                    let was_available = worker.is_available();
                    let _ = worker.check_health_async().await; // Use async version directly
                    if was_available != worker.is_available() {
                        revision.fetch_add(1, Ordering::AcqRel);
                    }
                }
            }
        });

        crate::core::HealthChecker::new(handle, shutdown)
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the worker registry
#[derive(Debug, Clone)]
pub struct WorkerRegistryStats {
    pub total_workers: usize,
    pub total_models: usize,
    pub healthy_workers: usize,
    pub total_load: usize,
    pub regular_workers: usize,
    pub prefill_workers: usize,
    pub decode_workers: usize,
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, CircuitBreakerConfig, WorkerFactory};
    use std::collections::HashMap;

    #[test]
    fn prefill_workers_follow_registration_order() {
        let registry = WorkerRegistry::new();
        for (url, port) in [("http://z", Some(2u16)), ("http://a", Some(1))] {
            registry.register(Arc::new(BasicWorker::new(
                url.to_string(),
                WorkerType::Prefill {
                    bootstrap_port: port,
                },
            )));
        }
        let urls: Vec<_> = registry
            .get_prefill_workers()
            .iter()
            .map(|worker| worker.url().to_string())
            .collect();
        assert_eq!(urls, ["http://z", "http://a"]);
    }

    #[test]
    fn test_worker_registry() {
        let registry = WorkerRegistry::new();

        // Create a worker with labels
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "llama-3-8b".to_string());
        labels.insert("priority".to_string(), "50".to_string());
        labels.insert("cost".to_string(), "0.8".to_string());

        let worker = WorkerFactory::create_regular_with_labels(
            "http://worker1:8080".to_string(),
            labels,
            CircuitBreakerConfig::default(),
        );

        // Register worker (WorkerFactory returns Box<dyn Worker>, convert to Arc)
        let worker_id = registry.register(Arc::from(worker));

        // Verify registration
        assert!(registry.get(&worker_id).is_some());
        assert!(registry.get_by_url("http://worker1:8080").is_some());
        assert_eq!(registry.get_by_model("llama-3-8b").len(), 1);
        assert_eq!(registry.get_by_type(&WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(&ConnectionMode::Http).len(), 1);

        // Test stats
        let stats = registry.stats();
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.total_models, 1);

        // Remove worker
        registry.remove(&worker_id);
        assert!(registry.get(&worker_id).is_none());
    }

    #[test]
    fn registration_update_rejects_removed_or_replaced_worker_arc() {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            "http://worker:8080".to_owned(),
            WorkerType::Regular,
        ));
        let mut initial_id = None;
        assert!(
            registry.register_if_absent_before_publish(worker.clone(), |_, id| {
                initial_id = Some(id.clone());
            })
        );
        assert_eq!(registry.with_current(&worker, Clone::clone), initial_id);
        assert!(
            !registry.register_if_absent_before_publish(worker.clone(), |_, _| {
                panic!("duplicate worker must not overwrite its binding");
            })
        );
        let (retiring, entered) = std::sync::mpsc::channel();
        let (resume, released) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let registry = &registry;
            let url = worker.url();
            let removal = scope.spawn(move || {
                registry
                    .remove_by_url_with_cleanup(url, |_| {
                        retiring.send(()).unwrap();
                        released
                            .recv_timeout(std::time::Duration::from_secs(5))
                            .unwrap();
                    })
                    .unwrap()
            });
            entered
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            let locked = matches!(
                registry.url_to_id.try_get(url),
                dashmap::try_result::TryResult::Locked
            );
            let unindexed = registry.get_all().is_empty()
                && registry.get_by_model_fast(worker.model_id()).is_empty();
            resume.send(()).unwrap();
            assert!(Arc::ptr_eq(&removal.join().unwrap(), &worker));
            assert!(locked, "admission and late probes must wait for retirement");
            assert!(
                unindexed,
                "removed worker remains visible in a routing index"
            );
        });
        assert_eq!(
            registry.with_current(&worker, |_| -> () { panic!("removed worker") }),
            None
        );
        let replacement: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            worker.url().to_owned(),
            WorkerType::Regular,
        ));
        let mut replacement_id = None;
        assert!(
            registry.register_if_absent_before_publish(replacement.clone(), |_, id| {
                replacement_id = Some(id.clone());
            })
        );
        assert_eq!(
            registry.with_current(&worker, |_| -> () { panic!("stale worker") }),
            None
        );
        assert_ne!(initial_id, replacement_id);
        assert_eq!(
            registry.with_current(&replacement, Clone::clone),
            replacement_id
        );
    }

    #[test]
    fn test_model_index_fast_lookup() {
        let registry = WorkerRegistry::new();

        // Create workers for different models
        let mut labels1 = HashMap::new();
        labels1.insert("model_id".to_string(), "llama-3".to_string());
        let worker1 = WorkerFactory::create_regular_with_labels(
            "http://worker1:8080".to_string(),
            labels1,
            CircuitBreakerConfig::default(),
        );

        let mut labels2 = HashMap::new();
        labels2.insert("model_id".to_string(), "llama-3".to_string());
        let worker2 = WorkerFactory::create_regular_with_labels(
            "http://worker2:8080".to_string(),
            labels2,
            CircuitBreakerConfig::default(),
        );

        let mut labels3 = HashMap::new();
        labels3.insert("model_id".to_string(), "gpt-4".to_string());
        let worker3 = WorkerFactory::create_regular_with_labels(
            "http://worker3:8080".to_string(),
            labels3,
            CircuitBreakerConfig::default(),
        );

        // Register workers
        registry.register(Arc::from(worker1));
        registry.register(Arc::from(worker2));
        registry.register(Arc::from(worker3));

        // Test get_by_model_fast for llama-3
        let llama_workers = registry.get_by_model_fast("llama-3");
        assert_eq!(llama_workers.len(), 2);
        let urls: Vec<String> = llama_workers.iter().map(|w| w.url().to_string()).collect();
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));

        // Test get_by_model_fast for gpt-4
        let gpt_workers = registry.get_by_model_fast("gpt-4");
        assert_eq!(gpt_workers.len(), 1);
        assert_eq!(gpt_workers[0].url(), "http://worker3:8080");

        // Test get_by_model_fast for non-existent model
        let unknown_workers = registry.get_by_model_fast("unknown-model");
        assert_eq!(unknown_workers.len(), 0);

        // Test that both get_by_model and get_by_model_fast return same results
        let llama_workers_slow = registry.get_by_model("llama-3");
        assert_eq!(llama_workers.len(), llama_workers_slow.len());

        // Test removal updates the model index
        registry.remove_by_url("http://worker1:8080");
        let llama_workers_after = registry.get_by_model_fast("llama-3");
        assert_eq!(llama_workers_after.len(), 1);
        assert_eq!(llama_workers_after[0].url(), "http://worker2:8080");
    }
}
