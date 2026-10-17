//! Worker Registry for multi-router support
//!
//! Provides centralized registry for workers with model-based indexing

use crate::core::{ConnectionMode, Worker, WorkerType};
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

/// Result of registering a worker, including the object actually replaced.
#[derive(Debug)]
pub struct WorkerRegistration {
    pub worker_id: WorkerId,
    pub previous: Option<Arc<dyn Worker>>,
}

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

    /// URL to worker ID mapping (for backward compatibility)
    url_to_id: Arc<DashMap<String, WorkerId>>,

    /// Coordinates updates across all indices. Acquire before any map guard;
    /// release before network calls or awaits. Readers see complete topology updates.
    topology: Arc<parking_lot::RwLock<()>>,

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
            url_to_id: Arc::new(DashMap::new()),
            revision: Arc::new(AtomicU64::new(0)),
            topology: Arc::new(parking_lot::RwLock::new(())),
        }
    }

    /// Insert or replace a worker, retaining the ID for an existing exact URL.
    /// Replaying registration never adds another membership to an index.
    pub fn register(&self, worker: Arc<dyn Worker>) -> WorkerId {
        self.register_with_previous(worker).worker_id
    }

    /// Register and return the previous object for lifecycle notifications.
    /// Replacing an object also replaces its health/load/circuit-breaker state;
    /// callers that want to preserve that state must reuse the existing Arc.
    pub fn register_with_previous(&self, worker: Arc<dyn Worker>) -> WorkerRegistration {
        let _topology = self.topology.write();
        let worker_id = self
            .url_to_id
            .get(worker.url())
            .map(|id| id.clone())
            .unwrap_or_default();
        let previous = self.get_unlocked(&worker_id);
        if let Some(ref old) = previous {
            if Arc::ptr_eq(old, &worker) {
                return WorkerRegistration {
                    worker_id,
                    previous,
                };
            }
            self.remove_from_indices(&worker_id, old);
        }

        self.workers.insert(worker_id.clone(), worker.clone());
        self.url_to_id
            .insert(worker.url().to_string(), worker_id.clone());
        let model_id = worker.model_id().to_string();
        self.model_workers
            .entry(model_id.clone())
            .or_default()
            .push(worker_id.clone());
        self.model_index
            .entry(model_id)
            .or_insert_with(|| Arc::new(RwLock::new(Vec::new())))
            .write()
            .expect("RwLock for model_index is poisoned")
            .push(worker.clone());
        self.type_workers
            .entry(worker.worker_type())
            .or_default()
            .push(worker_id.clone());
        self.connection_workers
            .entry(worker.connection_mode())
            .or_default()
            .push(worker_id.clone());
        self.notify_worker_state_change();
        WorkerRegistration {
            worker_id,
            previous,
        }
    }

    /// Remove index memberships using the old object's keys, before replacement.
    /// The caller holds the topology write lock.
    fn remove_from_indices(&self, worker_id: &WorkerId, worker: &Arc<dyn Worker>) {
        if let Some(mut workers) = self.model_workers.get_mut(worker.model_id()) {
            workers.retain(|id| id != worker_id);
        }
        self.model_workers
            .remove_if(worker.model_id(), |_, workers| workers.is_empty());
        if let Some(workers) = self.model_index.get(worker.model_id()) {
            workers
                .write()
                .expect("RwLock for model_index is poisoned")
                .retain(|w| w.url() != worker.url());
        }
        self.model_index.remove_if(worker.model_id(), |_, workers| {
            workers
                .read()
                .expect("RwLock for model_index is poisoned")
                .is_empty()
        });
        let role = worker.worker_type();
        if let Some(mut workers) = self.type_workers.get_mut(&role) {
            workers.retain(|id| id != worker_id);
        }
        self.type_workers
            .remove_if(&role, |_, workers| workers.is_empty());
        let mode = worker.connection_mode();
        if let Some(mut workers) = self.connection_workers.get_mut(&mode) {
            workers.retain(|id| id != worker_id);
        }
        self.connection_workers
            .remove_if(&mode, |_, workers| workers.is_empty());
    }

    /// Remove a worker by ID.
    pub fn remove(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        let _topology = self.topology.write();
        self.remove_unlocked(worker_id)
    }

    /// The caller holds the topology write lock; never reacquire it here.
    fn remove_unlocked(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        let (_, worker) = self.workers.remove(worker_id)?;
        self.url_to_id.remove(worker.url());
        self.remove_from_indices(worker_id, &worker);
        self.notify_worker_state_change();
        Some(worker)
    }

    /// Remove a worker by URL in the same transaction as its index cleanup.
    pub fn remove_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        let _topology = self.topology.write();
        // Clone and drop the map guard before remove_unlocked mutates this map.
        let worker_id = self.url_to_id.get(url).map(|id| id.clone())?;
        self.remove_unlocked(&worker_id)
    }

    /// The caller holds either topology lock; used to avoid recursive read locks.
    fn get_unlocked(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        self.workers.get(worker_id).map(|entry| entry.clone())
    }

    /// Get a worker by ID
    pub fn get(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        let _topology = self.topology.read();
        self.workers.get(worker_id).map(|entry| entry.clone())
    }

    /// Get a worker by URL
    pub fn get_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        let _topology = self.topology.read();
        self.url_to_id
            .get(url)
            .and_then(|id| self.get_unlocked(&id))
    }

    /// Get all workers for a model
    pub fn get_by_model(&self, model_id: &str) -> Vec<Arc<dyn Worker>> {
        let _topology = self.topology.read();
        self.model_workers
            .get(model_id)
            .map(|ids| ids.iter().filter_map(|id| self.get_unlocked(id)).collect())
            .unwrap_or_default()
    }

    /// Get all workers for a model (O(1) optimized version)
    /// This method uses the pre-indexed model_index for fast lookups
    pub fn get_by_model_fast(&self, model_id: &str) -> Vec<Arc<dyn Worker>> {
        let _topology = self.topology.read();
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
        let _topology = self.topology.read();
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
        let _topology = self.topology.read();
        self.type_workers
            .get(worker_type)
            .map(|ids| ids.iter().filter_map(|id| self.get_unlocked(id)).collect())
            .unwrap_or_default()
    }

    /// Get all prefill workers (regardless of bootstrap_port)
    pub fn get_prefill_workers(&self) -> Vec<Arc<dyn Worker>> {
        let _topology = self.topology.read();
        self.workers
            .iter()
            .filter_map(|entry| {
                let worker = entry.value();
                match worker.worker_type() {
                    WorkerType::Prefill { .. } => Some(worker.clone()),
                    _ => None,
                }
            })
            .collect()
    }

    /// Get all decode workers
    pub fn get_decode_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.get_by_type(&WorkerType::Decode)
    }

    /// Regular inference candidates. Encode-only models must not hide the
    /// unlabeled consumer fallback used by program scheduling.
    pub fn get_regular_workers(&self, model_id: Option<&str>) -> Vec<Arc<dyn Worker>> {
        match model_id {
            Some(model) => {
                self.get_workers_filtered(Some(model), Some(WorkerType::Regular), None, false)
            }
            // The role index preserves registration order for static pools.
            None => self.get_by_type(&WorkerType::Regular),
        }
    }

    /// Inference consumers across routers; management views still use get_all.
    pub fn get_consumer_workers(&self, model_id: Option<&str>) -> Vec<Arc<dyn Worker>> {
        self.get_workers_filtered(model_id, None, None, false)
            .into_iter()
            .filter(|worker| {
                matches!(
                    worker.worker_type(),
                    WorkerType::Regular | WorkerType::Prefill { .. } | WorkerType::Decode
                )
            })
            .collect()
    }

    /// P/D candidates, including prefill workers with any bootstrap port.
    pub fn get_pd_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.get_consumer_workers(None)
            .into_iter()
            .filter(|worker| {
                matches!(
                    worker.worker_type(),
                    WorkerType::Prefill { .. } | WorkerType::Decode
                )
            })
            .collect()
    }

    /// Get all workers by connection mode
    pub fn get_by_connection(&self, connection_mode: &ConnectionMode) -> Vec<Arc<dyn Worker>> {
        let _topology = self.topology.read();
        self.connection_workers
            .get(connection_mode)
            .map(|ids| ids.iter().filter_map(|id| self.get_unlocked(id)).collect())
            .unwrap_or_default()
    }

    /// Get all workers
    pub fn get_all(&self) -> Vec<Arc<dyn Worker>> {
        let _topology = self.topology.read();
        self.workers
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get all workers with their IDs
    pub fn get_all_with_ids(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        let _topology = self.topology.read();
        self.workers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Get all worker URLs
    pub fn get_all_urls(&self) -> Vec<String> {
        let _topology = self.topology.read();
        self.workers
            .iter()
            .map(|entry| entry.value().url().to_string())
            .collect()
    }

    /// Get all model IDs with workers
    pub fn get_models(&self) -> Vec<String> {
        let _topology = self.topology.read();
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
        let _topology = self.topology.read();
        let total_workers = self.workers.len();
        let total_models = self
            .model_workers
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .count();

        let mut healthy_count = 0;
        let mut total_load = 0;
        let mut regular_count = 0;
        let mut prefill_count = 0;
        let mut decode_count = 0;
        let mut encode_count = 0;

        for entry in self.workers.iter() {
            let worker = entry.value();
            if worker.is_healthy() {
                healthy_count += 1;
            }
            total_load += worker.load();

            match worker.worker_type() {
                WorkerType::Regular => regular_count += 1,
                WorkerType::Prefill { .. } => prefill_count += 1,
                WorkerType::Decode => decode_count += 1,
                WorkerType::Encode => encode_count += 1,
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
            encode_workers: encode_count,
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
        let topology = self.topology.clone();

        let handle = tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(check_interval_secs));

            // Counter for periodic load reset (every 10 health check cycles)
            let mut check_count = 0u64;
            const LOAD_RESET_INTERVAL: u64 = 10;

            loop {
                interval.tick().await;

                // Check for shutdown signal
                if shutdown_clone.load(Ordering::Acquire) {
                    tracing::debug!("Registry health checker shutting down");
                    break;
                }

                // Get all workers from registry
                let workers: Vec<Arc<dyn crate::core::Worker>> = {
                    let _topology = topology.read();
                    workers_ref
                        .iter()
                        .map(|entry| entry.value().clone())
                        .collect()
                };

                // Perform health checks
                for worker in &workers {
                    let was_available = worker.is_available();
                    let _ = worker.check_health_async().await; // Use async version directly
                    if was_available != worker.is_available() {
                        revision.fetch_add(1, Ordering::AcqRel);
                    }
                }

                // Reset loads periodically
                check_count += 1;
                if check_count.is_multiple_of(LOAD_RESET_INTERVAL) {
                    tracing::debug!("Resetting worker loads (cycle {})", check_count);
                    for worker in &workers {
                        worker.reset_load();
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
    pub encode_workers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{CircuitBreakerConfig, WorkerFactory};
    use std::collections::HashMap;

    fn registry_worker(
        url: &str,
        model: &str,
        role: WorkerType,
        mode: ConnectionMode,
    ) -> Arc<dyn Worker> {
        Arc::new(
            crate::core::BasicWorker::with_connection_mode(url.to_string(), role, mode)
                .with_labels(HashMap::from([("model_id".to_string(), model.to_string())])),
        )
    }

    #[test]
    fn repeated_registration_has_one_current_worker_in_every_index() {
        let registry = WorkerRegistry::new();
        let first = registry_worker(
            "http://worker",
            "model",
            WorkerType::Regular,
            ConnectionMode::Http,
        );
        let id = registry.register(first.clone());
        assert_eq!(registry.register(first), id);
        let current = registry_worker(
            "http://worker",
            "model",
            WorkerType::Regular,
            ConnectionMode::Http,
        );
        assert_eq!(registry.register(current.clone()), id);

        for workers in [
            registry.get_all(),
            registry.get_by_model("model"),
            registry.get_by_model_fast("model"),
            registry.get_by_type(&WorkerType::Regular),
            registry.get_by_connection(&ConnectionMode::Http),
        ] {
            assert_eq!(workers.len(), 1);
            assert!(Arc::ptr_eq(&workers[0], &current));
        }
        assert!(Arc::ptr_eq(
            &registry.get_by_url("http://worker").unwrap(),
            &current
        ));
    }

    #[test]
    fn registration_reports_replaced_object_and_replay_preserves_revision() {
        let registry = WorkerRegistry::new();
        let first = registry_worker(
            "http://worker",
            "model",
            WorkerType::Regular,
            ConnectionMode::Http,
        );
        let added = registry.register_with_previous(first.clone());
        assert!(added.previous.is_none());
        let before = registry.revision();
        let replay = registry.register_with_previous(first.clone());
        assert_eq!(replay.worker_id, added.worker_id);
        assert!(Arc::ptr_eq(&replay.previous.unwrap(), &first));
        assert_eq!(registry.revision(), before);
        let current = registry_worker(
            "http://worker",
            "model",
            WorkerType::Regular,
            ConnectionMode::Http,
        );
        let replaced = registry.register_with_previous(current.clone());
        assert_eq!(replaced.worker_id, added.worker_id);
        assert!(Arc::ptr_eq(&replaced.previous.unwrap(), &first));
        assert!(registry.revision() > before);
        assert!(Arc::ptr_eq(
            &registry.get_by_model_fast("model")[0],
            &current
        ));
    }

    #[test]
    fn replacement_and_stale_removal_preserve_other_workers() {
        let registry = WorkerRegistry::new();
        let neighbor = registry_worker(
            "http://neighbor",
            "model",
            WorkerType::Regular,
            ConnectionMode::Http,
        );
        registry.register(neighbor.clone());
        let old_id = registry.register(registry_worker(
            "http://worker",
            "model",
            WorkerType::Regular,
            ConnectionMode::Http,
        ));
        registry.register(registry_worker(
            "http://worker",
            "other-model",
            WorkerType::Decode,
            ConnectionMode::Http,
        ));
        assert_eq!(registry.get_by_model_fast("model").len(), 1);
        assert!(Arc::ptr_eq(
            &registry.get_by_model_fast("model")[0],
            &neighbor
        ));
        registry.remove(&old_id).unwrap();
        let current = registry_worker(
            "http://worker",
            "model",
            WorkerType::Regular,
            ConnectionMode::Http,
        );
        let new_id = registry.register(current.clone());
        assert_ne!(new_id, old_id);
        assert!(registry.remove(&old_id).is_none());
        assert!(Arc::ptr_eq(
            &registry.get_by_url("http://worker").unwrap(),
            &current
        ));
        assert_eq!(registry.get_by_model_fast("model").len(), 2);
        assert!(!registry.has_model("other-model"));
    }

    #[test]
    fn concurrent_readers_only_see_members_of_the_requested_index() {
        let registry = Arc::new(WorkerRegistry::new());
        let writer_registry = registry.clone();
        let writer = std::thread::spawn(move || {
            for _ in 0..200 {
                writer_registry.register(registry_worker(
                    "http://worker",
                    "regular",
                    WorkerType::Regular,
                    ConnectionMode::Http,
                ));
                writer_registry.register(registry_worker(
                    "http://worker",
                    "decode",
                    WorkerType::Decode,
                    ConnectionMode::Grpc,
                ));
            }
        });
        for _ in 0..200 {
            for worker in registry.get_by_type(&WorkerType::Regular) {
                assert_eq!(worker.worker_type(), WorkerType::Regular);
            }
            for worker in registry.get_by_connection(&ConnectionMode::Http) {
                assert_eq!(worker.connection_mode(), ConnectionMode::Http);
            }
            for worker in registry.get_by_model("regular") {
                assert_eq!(worker.model_id(), "regular");
            }
        }
        writer.join().unwrap();
    }

    #[test]
    fn replacement_moves_indices_and_removal_clears_old_models() {
        let registry = WorkerRegistry::new();
        let id = registry.register(registry_worker(
            "http://worker",
            "old-model",
            WorkerType::Regular,
            ConnectionMode::Http,
        ));
        let before = registry.revision();
        let current = registry_worker(
            "http://worker",
            "new-model",
            WorkerType::Decode,
            ConnectionMode::Grpc,
        );
        assert_eq!(registry.register(current.clone()), id);
        assert!(registry.revision() > before);
        assert!(registry.get_by_model("old-model").is_empty());
        assert!(registry.get_by_model_fast("old-model").is_empty());
        assert!(!registry.has_model("old-model"));
        assert_eq!(registry.get_models(), vec!["new-model"]);
        assert!(registry.get_by_type(&WorkerType::Regular).is_empty());
        assert!(registry.get_by_connection(&ConnectionMode::Http).is_empty());
        for workers in [
            registry.get_by_model("new-model"),
            registry.get_by_model_fast("new-model"),
            registry.get_decode_workers(),
            registry.get_by_connection(&ConnectionMode::Grpc),
        ] {
            assert_eq!(workers.len(), 1);
            assert!(Arc::ptr_eq(&workers[0], &current));
        }
        let before = registry.revision();
        assert!(Arc::ptr_eq(
            &registry.remove_by_url("http://worker").unwrap(),
            &current
        ));
        assert!(registry.revision() > before);
        assert!(registry.get(&id).is_none());
        assert!(registry.get_by_url("http://worker").is_none());
        assert!(registry.get_all().is_empty());
        assert!(registry.get_models().is_empty());
        for model in ["old-model", "new-model"] {
            assert!(registry.get_by_model(model).is_empty());
            assert!(registry.get_by_model_fast(model).is_empty());
            assert!(!registry.has_model(model));
        }
        assert!(registry.get_decode_workers().is_empty());
        assert!(registry.get_by_connection(&ConnectionMode::Grpc).is_empty());
    }

    #[test]
    fn concurrent_registration_reuses_one_id() {
        let registry = Arc::new(WorkerRegistry::new());
        let barrier = Arc::new(std::sync::Barrier::new(16));
        let tasks: Vec<_> = (0..16)
            .map(|_| {
                let registry = registry.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let worker = registry_worker(
                        "http://worker",
                        "model",
                        WorkerType::Regular,
                        ConnectionMode::Http,
                    );
                    barrier.wait();
                    registry.register(worker)
                })
            })
            .collect();
        let ids: Vec<_> = tasks.into_iter().map(|task| task.join().unwrap()).collect();
        assert!(ids.iter().all(|id| id == &ids[0]));
        assert_eq!(registry.get_all().len(), 1);
        assert_eq!(registry.get_by_model("model").len(), 1);
        assert_eq!(registry.get_by_model_fast("model").len(), 1);
        assert_eq!(registry.get_by_type(&WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(&ConnectionMode::Http).len(), 1);
        registry.remove_by_url("http://worker").unwrap();
        assert!(registry.get_all().is_empty());
    }

    #[test]
    fn concurrent_registration_and_removal_leave_no_orphans() {
        let registry = Arc::new(WorkerRegistry::new());
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let tasks: Vec<_> = (0..8)
            .map(|thread| {
                let registry = registry.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    for iteration in 0..100 {
                        let id = registry.register(registry_worker(
                            "http://worker",
                            "model",
                            WorkerType::Regular,
                            ConnectionMode::Http,
                        ));
                        if (iteration + thread) % 2 == 0 {
                            registry.remove(&id);
                        } else {
                            registry.remove_by_url("http://worker");
                        }
                    }
                })
            })
            .collect();
        for task in tasks {
            task.join().unwrap();
        }
        let current = registry_worker(
            "http://worker",
            "model",
            WorkerType::Regular,
            ConnectionMode::Http,
        );
        registry.register(current.clone());
        for workers in [
            registry.get_all(),
            registry.get_by_model("model"),
            registry.get_by_model_fast("model"),
        ] {
            assert_eq!(workers.len(), 1);
            assert!(Arc::ptr_eq(&workers[0], &current));
        }
        registry.remove_by_url("http://worker").unwrap();
        assert!(registry.get_all().is_empty());
        assert!(!registry.has_model("model"));
    }

    #[test]
    fn encode_registration_tracks_role_stats_and_removal() {
        let registry = WorkerRegistry::new();
        let regular_id = registry.register(Arc::from(WorkerFactory::create_regular(
            "http://consumer:8080".to_string(),
        )));
        let prefill_id = registry.register(Arc::from(WorkerFactory::create_prefill(
            "http://prefill:8080".to_string(),
            Some(8998),
        )));
        let decode_id = registry.register(Arc::from(WorkerFactory::create_decode(
            "http://decode:8080".to_string(),
        )));
        let encoder: Arc<dyn Worker> = Arc::from(WorkerFactory::create_encode(
            "http://encoder:8080".to_string(),
        ));
        encoder.increment_load();
        encoder.set_healthy(false);
        let encode_id = registry.register(encoder);

        let encoders = registry.get_by_type(&WorkerType::Encode);
        assert_eq!(encoders.len(), 1);
        assert_eq!(encoders[0].url(), "http://encoder:8080");
        assert_eq!(registry.get_by_type(&WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_prefill_workers().len(), 1);
        assert_eq!(registry.get_decode_workers().len(), 1);
        let stats = registry.stats();
        assert_eq!(stats.total_workers, 4);
        assert_eq!(stats.healthy_workers, 3);
        assert_eq!(stats.total_load, 1);
        assert_eq!(stats.encode_workers, 1);
        assert_eq!(stats.regular_workers, 1);
        assert_eq!(stats.prefill_workers, 1);
        assert_eq!(stats.decode_workers, 1);

        registry.remove(&encode_id);
        assert!(registry.get_by_type(&WorkerType::Encode).is_empty());
        assert!(registry.get_by_url("http://encoder:8080").is_none());
        assert_eq!(registry.stats().encode_workers, 0);
        assert_eq!(registry.stats().total_workers, 3);
        for id in [regular_id, prefill_id, decode_id] {
            assert!(registry.get(&id).is_some());
        }
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
