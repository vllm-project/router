//! Worker Registry for multi-router support
//!
//! Provides centralized registry for workers with model-based indexing

use crate::{
    core::{ConnectionMode, Worker, WorkerType},
    metrics::RouterMetrics,
};
use dashmap::DashMap;
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

    /// Serialize ownership transfers, including concurrent first registration
    /// and removal/re-registration of the same URL. Load transitions never take it.
    mutation_lock: parking_lot::Mutex<()>,
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
            mutation_lock: parking_lot::Mutex::new(()),
        }
    }

    /// Register a new worker
    pub fn register(&self, worker: Arc<dyn Worker>) -> WorkerId {
        let _mutation = self.mutation_lock.lock();
        let (worker_id, is_new) = if let Some(existing_id) = self.url_to_id.get(worker.url()) {
            // Worker with this URL already exists, update it
            (existing_id.clone(), false)
        } else {
            (WorkerId::new(), true)
        };
        if is_new {
            self.registration_order
                .write()
                .expect("RwLock for registration_order is poisoned")
                .push(worker_id.clone());
        }

        // Retire the previous writer before enabling the replacement. Retirement
        // takes the same lock as publication, so an in-progress old write must
        // finish before the new instance publishes its initial/current load.
        let worker_type = worker.worker_type();
        let connection_mode = worker.connection_mode();
        let (replaces_same_model, replaces_same_type, replaces_same_connection) =
            if let Some(previous) = self.get(&worker_id) {
                previous.set_load_metrics_enabled(false);
                let previous_type = previous.worker_type();
                let previous_connection = previous.connection_mode();
                let same_model = previous.model_id() == worker.model_id();
                let same_type = previous_type == worker_type;
                let same_connection = previous_connection == connection_mode;

                // Keep unchanged entries visible during load publication. Their
                // WorkerId will resolve to the new instance once workers is updated.
                if !same_model {
                    self.remove_model_indexes(&worker_id, previous.as_ref());
                }
                if !same_type {
                    self.remove_type_index(&worker_id, &previous_type);
                }
                if !same_connection {
                    self.remove_connection_index(&worker_id, &previous_connection);
                }
                (same_model, same_type, same_connection)
            } else {
                (false, false, false)
            };
        worker.set_load_metrics_enabled(true);

        // Store worker only after its initial metric has been published.
        self.workers.insert(worker_id.clone(), worker.clone());

        // Update URL mapping
        self.url_to_id
            .insert(worker.url().to_string(), worker_id.clone());

        // Update model index (both ID-based and optimized)
        let model_id = worker.model_id().to_string();
        if !replaces_same_model {
            self.model_workers
                .entry(model_id.clone())
                .or_default()
                .push(worker_id.clone());
        }

        {
            let model_entry = self
                .model_index
                .entry(model_id)
                .or_insert_with(|| Arc::new(RwLock::new(Vec::new())));
            let mut model_workers = model_entry
                .write()
                .expect("RwLock for model_index is poisoned");
            // Same-model replacement must keep the pool visible to readers that
            // do not take mutation_lock. Swap the Arc under the index write lock.
            let existing = if replaces_same_model {
                model_workers.iter_mut().find(|w| w.url() == worker.url())
            } else {
                None
            };
            if let Some(existing) = existing {
                *existing = worker.clone();
            } else {
                model_workers.push(worker.clone());
            }
        }

        if !replaces_same_type {
            self.type_workers
                .entry(worker_type)
                .or_default()
                .push(worker_id.clone());
        }

        if !replaces_same_connection {
            self.connection_workers
                .entry(connection_mode)
                .or_default()
                .push(worker_id.clone());
        }

        // Publish the initial state even if no health transition occurs.
        RouterMetrics::set_worker_health(worker.url(), worker.is_healthy());

        self.notify_worker_state_change();

        worker_id
    }

    /// Caller must hold mutation_lock while changing the secondary indexes.
    fn remove_model_indexes(&self, worker_id: &WorkerId, worker: &dyn Worker) {
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
    }

    /// Caller must hold mutation_lock while changing the secondary indexes.
    fn remove_type_index(&self, worker_id: &WorkerId, worker_type: &WorkerType) {
        if let Some(mut type_workers) = self.type_workers.get_mut(worker_type) {
            type_workers.retain(|id| id != worker_id);
        }
    }

    /// Caller must hold mutation_lock while changing the secondary indexes.
    fn remove_connection_index(&self, worker_id: &WorkerId, connection_mode: &ConnectionMode) {
        if let Some(mut conn_workers) = self.connection_workers.get_mut(connection_mode) {
            conn_workers.retain(|id| id != worker_id);
        }
    }

    /// Remove a worker by ID
    pub fn remove(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        let _mutation = self.mutation_lock.lock();
        self.remove_locked(worker_id)
    }

    /// Caller must hold mutation_lock throughout the topology change.
    fn remove_locked(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        if let Some((_, worker)) = self.workers.remove(worker_id) {
            worker.set_load_metrics_enabled(false);
            // Remove from URL mapping
            self.url_to_id.remove(worker.url());

            self.remove_model_indexes(worker_id, worker.as_ref());
            self.remove_type_index(worker_id, &worker.worker_type());
            self.remove_connection_index(worker_id, &worker.connection_mode());
            self.registration_order
                .write()
                .expect("RwLock for registration_order is poisoned")
                .retain(|id| id != worker_id);

            self.notify_worker_state_change();
            Some(worker)
        } else {
            None
        }
    }

    /// Remove a worker by URL
    pub fn remove_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        let _mutation = self.mutation_lock.lock();
        if let Some((_, worker_id)) = self.url_to_id.remove(url) {
            self.remove_locked(&worker_id)
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
mod tests {
    use super::*;
    use crate::core::{BasicWorker, CircuitBreakerConfig, WorkerFactory};
    use std::collections::HashMap;

    #[test]
    fn replacement_model_lookup_returns_only_the_current_worker() {
        let registry = WorkerRegistry::new();
        let old: Arc<dyn Worker> = Arc::new(
            BasicWorker::new("http://replacement:8000".into(), WorkerType::Regular)
                .with_labels(HashMap::from([("model_id".into(), "model".into())])),
        );
        let current: Arc<dyn Worker> = Arc::new(
            BasicWorker::new(old.url().into(), WorkerType::Regular)
                .with_labels(HashMap::from([("model_id".into(), "model".into())])),
        );
        let id = registry.register(old);
        assert_eq!(registry.register(current.clone()), id);
        // Registering the same live object must not duplicate any index entry.
        registry.register(current.clone());
        for indexed in [
            registry.get_by_model_fast("model"),
            registry.get_by_model("model"),
            registry.get_by_type(&WorkerType::Regular),
            registry.get_by_connection(&ConnectionMode::Http),
        ] {
            assert_eq!(indexed.len(), 1);
            assert!(Arc::ptr_eq(&indexed[0], &current));
        }
    }

    #[test]
    fn replacement_moves_worker_out_of_old_metadata_indexes() {
        let registry = WorkerRegistry::new();
        let old: Arc<dyn Worker> = Arc::new(
            BasicWorker::new("http://replacement:8000".into(), WorkerType::Regular)
                .with_labels(HashMap::from([("model_id".into(), "old-model".into())])),
        );
        let current: Arc<dyn Worker> = Arc::new(
            BasicWorker::with_connection_mode(
                old.url().into(),
                WorkerType::Decode,
                ConnectionMode::Grpc,
            )
            .with_labels(HashMap::from([("model_id".into(), "new-model".into())])),
        );
        let id = registry.register(old);
        assert_eq!(registry.register(current.clone()), id);
        assert!(!registry.has_model("old-model"));
        assert!(registry.get_by_model_fast("old-model").is_empty());
        assert!(registry.get_by_model("old-model").is_empty());
        assert!(registry.get_by_type(&WorkerType::Regular).is_empty());
        assert!(registry.get_by_connection(&ConnectionMode::Http).is_empty());
        for indexed in [
            registry.get_by_model_fast("new-model"),
            registry.get_by_model("new-model"),
            registry.get_by_type(&WorkerType::Decode),
            registry.get_by_connection(&ConnectionMode::Grpc),
        ] {
            assert_eq!(indexed.len(), 1);
            assert!(Arc::ptr_eq(&indexed[0], &current));
        }
        registry.remove(&id);
        assert!(registry.get_by_model_fast("new-model").is_empty());
        assert!(registry.get_by_type(&WorkerType::Decode).is_empty());
        assert!(registry.get_by_connection(&ConnectionMode::Grpc).is_empty());
    }

    fn assert_load_sample(handle: &metrics_exporter_prometheus::PrometheusHandle, load: usize) {
        let sample = handle
            .render()
            .lines()
            .find(|line| line.starts_with("vllm_router_worker_load{"))
            .map(str::to_owned);
        assert_eq!(
            sample,
            Some(format!(
                "vllm_router_worker_load{{worker=\"http://lifecycle:8000\"}} {load}"
            ))
        );
    }

    #[test]
    fn worker_load_gauge_is_published_at_registration() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let registry = WorkerRegistry::new();
        let worker = Arc::new(BasicWorker::new(
            "http://lifecycle:8000".into(),
            WorkerType::Regular,
        ));

        registry.register(worker.clone());
        assert_load_sample(&handle, 0);
        worker.increment_load();
        registry.register(worker.clone());
        assert_eq!(worker.load(), 1);
        assert_load_sample(&handle, 1);
    }

    #[test]
    fn retired_worker_cannot_overwrite_new_worker_load_gauge() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let registry = WorkerRegistry::new();
        let old = Arc::new(BasicWorker::new(
            "http://lifecycle:8000".into(),
            WorkerType::Regular,
        ));
        let old_clone = old.as_ref().clone();
        let id = registry.register(old.clone());
        old.increment_load();
        let current = Arc::new(BasicWorker::new(old.url().into(), WorkerType::Regular));
        assert_eq!(registry.register(current.clone()), id);
        current.increment_load();
        current.increment_load();
        assert_load_sample(&handle, 2);

        old_clone.decrement_load();
        assert_eq!(old.load(), 0);
        assert_load_sample(&handle, 2);
        old.increment_load();
        assert_eq!(old.load(), 1);
        assert_load_sample(&handle, 2);
        old.reset_load();
        assert_eq!(old.load(), 0);
        assert_load_sample(&handle, 2);

        current.decrement_load();
        current.decrement_load();
        assert_load_sample(&handle, 0);
    }

    #[test]
    fn removed_worker_cannot_publish_after_url_is_registered_again() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let registry = WorkerRegistry::new();
        let old = Arc::new(BasicWorker::new(
            "http://lifecycle:8000".into(),
            WorkerType::Regular,
        ));
        registry.register(old.clone());
        old.increment_load();
        registry.remove_by_url(old.url()).unwrap();
        assert_load_sample(&handle, 0);
        let current = Arc::new(BasicWorker::new(old.url().into(), WorkerType::Regular));
        registry.register(current.clone());
        current.increment_load();
        old.decrement_load();
        assert_eq!(old.load(), 0);
        assert_load_sample(&handle, 1);
    }

    #[test]
    fn concurrent_registrations_leave_one_load_gauge_owner() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let registry = WorkerRegistry::new();
        let workers: Vec<Arc<dyn Worker>> = (0..8)
            .map(|_| {
                let worker = BasicWorker::new("http://lifecycle:8000".into(), WorkerType::Regular);
                worker.increment_load();
                Arc::new(worker) as Arc<dyn Worker>
            })
            .collect();
        let start = std::sync::Barrier::new(workers.len());
        std::thread::scope(|scope| {
            for worker in &workers {
                scope.spawn(|| {
                    metrics::with_local_recorder(&recorder, || {
                        start.wait();
                        registry.register(worker.clone());
                    })
                });
            }
        });
        assert_eq!(registry.get_all().len(), 1);
        let current = registry.get_by_url("http://lifecycle:8000").unwrap();
        for worker in &workers {
            if !Arc::ptr_eq(worker, &current) {
                worker.decrement_load();
                assert_eq!(worker.load(), 0);
                assert_load_sample(&handle, 1);
            }
        }
        current.decrement_load();
        assert_load_sample(&handle, 0);
    }

    #[test]
    fn retired_dp_worker_cannot_overwrite_new_worker_load_gauge() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let registry = WorkerRegistry::new();
        let old = Arc::new(crate::core::DPAwareWorker::new(
            "http://lifecycle:8000".into(),
            0,
            2,
            WorkerType::Decode,
        ));
        registry.register(old.clone());
        old.increment_load();
        let current = Arc::new(crate::core::DPAwareWorker::new(
            "http://lifecycle:8000".into(),
            0,
            2,
            WorkerType::Decode,
        ));
        registry.register(current.clone());
        current.increment_load();
        current.increment_load();
        old.decrement_load();
        assert_eq!(old.load(), 0);
        assert!(handle
            .render()
            .lines()
            .any(|line| line == "vllm_router_worker_load{worker=\"http://lifecycle:8000@0\"} 2"));
    }

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
