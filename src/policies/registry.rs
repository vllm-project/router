/// Policy Registry for managing model-to-policy mappings
///
/// This registry manages the dynamic assignment of load balancing policies to models.
/// When the first worker of a new model is added, it determines the policy for that model.
/// All subsequent workers of the same model use the established policy.
/// When the last worker of a model is removed, the policy mapping is cleaned up.
use super::{
    CacheAwareConfig, CacheAwarePolicy, ConsistentHashPolicy, CostParams, KvAffinityTable,
    KvAwarePolicy, LoadBalancingPolicy, PowerOfTwoPolicy, RandomPolicy, RendezvousHashPolicy,
    RoundRobinPolicy, TransferCredits,
};
use crate::config::types::{EncodePolicy, PolicyConfig};
use crate::core::{Worker, WorkerType};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tracing::{debug, info, warn};

/// Registry for managing model-to-policy mappings
#[derive(Clone)]
pub struct PolicyRegistry {
    /// Model ID -> Policy instance mapping
    model_policies: Arc<RwLock<HashMap<String, Arc<dyn LoadBalancingPolicy>>>>,

    /// Model ID -> Worker count for cleanup tracking
    model_worker_counts: Arc<RwLock<HashMap<String, usize>>>,

    /// Default policy instance (cached)
    default_policy: Arc<dyn LoadBalancingPolicy>,

    /// Router-wide Encode policy, always independent of consumer policies.
    encode_policy: Arc<RwLock<Arc<dyn LoadBalancingPolicy>>>,

    /// Prefill policy for PD mode
    prefill_policy: Arc<RwLock<Option<Arc<dyn LoadBalancingPolicy>>>>,

    /// Decode policy for PD mode
    decode_policy: Arc<RwLock<Option<Arc<dyn LoadBalancingPolicy>>>>,

    /// Affinity table shared by every KV-aware policy this registry creates,
    /// bounding affinity state across models.
    kv_affinity_table: Arc<KvAffinityTable>,
}

impl PolicyRegistry {
    /// Create a new PolicyRegistry with a default policy
    pub fn new(default_policy_config: PolicyConfig) -> Self {
        Self::new_with_affinity_ttl(default_policy_config, None)
    }

    /// Create a policy registry with an optional idle TTL for KV-aware affinity.
    pub fn new_with_affinity_ttl(
        default_policy_config: PolicyConfig,
        affinity_ttl: Option<Duration>,
    ) -> Self {
        let kv_affinity_table = Arc::new(KvAffinityTable::with_ttl(affinity_ttl));
        let default_policy =
            Self::create_policy_from_config(&default_policy_config, &kv_affinity_table);

        Self {
            model_policies: Arc::new(RwLock::new(HashMap::new())),
            model_worker_counts: Arc::new(RwLock::new(HashMap::new())),
            default_policy,
            encode_policy: Arc::new(RwLock::new(Self::create_encode_policy(
                EncodePolicy::default(),
            ))),
            prefill_policy: Arc::new(RwLock::new(None)),
            decode_policy: Arc::new(RwLock::new(None)),
            kv_affinity_table,
        }
    }

    /// Called when a worker is added
    /// Returns the policy that should be used for this worker's model
    pub fn on_worker_added(
        &self,
        model_id: &str,
        policy_hint: Option<&str>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        // Increment worker count
        {
            let mut counts = self.model_worker_counts.write().unwrap();
            *counts.entry(model_id.to_string()).or_insert(0) += 1;
            debug!(
                "Worker added for model {}, count: {}",
                model_id,
                counts.get(model_id).unwrap()
            );
        }

        // Check if model already has a policy
        {
            let policies = self.model_policies.read().unwrap();
            if let Some(existing_policy) = policies.get(model_id) {
                debug!(
                    "Model {} already has policy: {}",
                    model_id,
                    existing_policy.name()
                );
                return Arc::clone(existing_policy);
            }
        }

        // New model - determine policy
        let policy = self.determine_policy_for_model(model_id, policy_hint);

        info!(
            "Assigning policy {} to new model {}",
            policy.name(),
            model_id
        );

        // Store policy for this model
        {
            let mut policies = self.model_policies.write().unwrap();
            policies.insert(model_id.to_string(), Arc::clone(&policy));
        }

        policy
    }

    /// Clean policy state for a successfully removed worker, then update its
    /// consumer count. Call once with the object returned by registry removal.
    /// Encode workers do not participate in consumer policies or accounting.
    pub fn notify_consumer_removed(&self, removed: &dyn Worker) {
        if !removed.worker_type().is_consumer() {
            return;
        }
        self.clear_consumer_cache_state(removed);
        self.on_worker_removed(removed.model_id());
    }

    /// Invalidate affinity for an actual removed or replaced consumer without
    /// changing model accounting. Same-model replacements retain their policy.
    pub fn clear_consumer_cache_state(&self, removed: &dyn Worker) {
        if !removed.worker_type().is_consumer() {
            return;
        }
        let model_id = removed.model_id();
        let mut policies = Vec::new();
        if let Some(policy) = self.get_policy(model_id) {
            policies.push(policy);
        }
        // PD routing uses role-specific instances, even when their configs
        // match the model policy. Clean the actual removed worker's role.
        let routing_policy = match removed.worker_type() {
            WorkerType::Prefill { .. } => Some(self.get_prefill_policy()),
            WorkerType::Decode => Some(self.get_decode_policy()),
            _ => None,
        };
        if let Some(policy) = routing_policy {
            // Unconfigured PD policies fall back to the shared default Arc.
            if !policies
                .iter()
                .any(|existing| Arc::ptr_eq(existing, &policy))
            {
                policies.push(policy);
            }
        }
        // The last-worker notification removes the model's policy mapping,
        // so clean both instances before updating the count.
        for policy in policies {
            if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                cache_aware.remove_worker_by_url(removed.url());
                debug!("Removed worker from cache-aware tree: {}", removed.url());
            }
        }
    }

    /// Decrement model accounting without worker-specific cleanup.
    /// Use `notify_consumer_removed` after removing a consumer from the registry.
    pub fn on_worker_removed(&self, model_id: &str) {
        let should_cleanup = {
            let mut counts = self.model_worker_counts.write().unwrap();
            if let Some(count) = counts.get_mut(model_id) {
                *count = count.saturating_sub(1);
                debug!("Worker removed for model {}, count: {}", model_id, *count);
                if *count == 0 {
                    counts.remove(model_id);
                    true
                } else {
                    false
                }
            } else {
                warn!(
                    "Attempted to remove worker for model {} with no registered workers",
                    model_id
                );
                false
            }
        };

        // Clean up policy if this was the last worker
        if should_cleanup {
            let mut policies = self.model_policies.write().unwrap();
            if let Some(policy) = policies.remove(model_id) {
                info!(
                    "Removed policy {} for model {} (last worker removed)",
                    policy.name(),
                    model_id
                );
                // Policy will be dropped here, cleaning up any resources
                drop(policy);
            }
        }
    }

    /// Get the policy for a model
    pub fn get_policy(&self, model_id: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        self.model_policies.read().unwrap().get(model_id).cloned()
    }

    /// Get the default policy
    pub fn get_default_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        Arc::clone(&self.default_policy)
    }

    /// Get policy for a model, or default if not found
    pub fn get_policy_or_default(&self, model_id: &str) -> Arc<dyn LoadBalancingPolicy> {
        self.get_policy(model_id)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Determine policy for a new model
    fn determine_policy_for_model(
        &self,
        model_id: &str,
        policy_hint: Option<&str>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        // 1. Check policy hint from worker
        if let Some(policy_type) = policy_hint {
            debug!("Using policy hint '{}' for model {}", policy_type, model_id);
            return self.create_policy_from_type(policy_type);
        }

        // 2. Use default policy
        debug!("Using default policy for model {}", model_id);
        Arc::clone(&self.default_policy)
    }

    /// Create a policy from a type string
    fn create_policy_from_type(&self, policy_type: &str) -> Arc<dyn LoadBalancingPolicy> {
        match policy_type {
            "round_robin" => Arc::new(RoundRobinPolicy::new()),
            "random" => Arc::new(RandomPolicy::new()),
            "cache_aware" => Arc::new(CacheAwarePolicy::new()),
            "kv_aware" => Arc::new(
                KvAwarePolicy::default().with_shared_affinity(Arc::clone(&self.kv_affinity_table)),
            ),
            "power_of_two" => Arc::new(PowerOfTwoPolicy::new()),
            "rendezvous_hash" => Arc::new(RendezvousHashPolicy::new()),
            _ => {
                warn!("Unknown policy type '{}', using default", policy_type);
                Arc::clone(&self.default_policy)
            }
        }
    }

    /// Create a policy from a PolicyConfig
    fn create_policy_from_config(
        config: &PolicyConfig,
        kv_affinity_table: &Arc<KvAffinityTable>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        match config {
            PolicyConfig::RoundRobin => Arc::new(RoundRobinPolicy::new()),
            PolicyConfig::Random => Arc::new(RandomPolicy::new()),
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            } => {
                let cache_config = CacheAwareConfig {
                    cache_threshold: *cache_threshold,
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    eviction_interval_secs: *eviction_interval_secs,
                    max_tree_size: *max_tree_size,
                };
                Arc::new(CacheAwarePolicy::with_config(cache_config))
            }
            PolicyConfig::KvAware {
                router_temperature,
                kv_penalty_host,
                kv_penalty_external,
                kv_penalty_disk,
                ..
            } => Arc::new(
                KvAwarePolicy::new(CostParams {
                    router_temperature: *router_temperature,
                    transfer_credits: TransferCredits {
                        device: 1.0,
                        host_pinned: *kv_penalty_host,
                        external: *kv_penalty_external,
                        disk: *kv_penalty_disk,
                    },
                    ..CostParams::default()
                })
                .with_shared_affinity(Arc::clone(kv_affinity_table)),
            ),
            PolicyConfig::PowerOfTwo { .. } => Arc::new(PowerOfTwoPolicy::new()),
            PolicyConfig::ConsistentHash { .. } => Arc::new(ConsistentHashPolicy::new()),
            PolicyConfig::RendezvousHash => Arc::new(RendezvousHashPolicy::new()),
        }
    }

    /// Get current model->policy mappings (for debugging/monitoring)
    pub fn get_all_mappings(&self) -> HashMap<String, String> {
        let policies = self.model_policies.read().unwrap();
        policies
            .iter()
            .map(|(model, policy)| (model.clone(), policy.name().to_string()))
            .collect()
    }

    /// Get worker counts per model
    pub fn get_worker_counts(&self) -> HashMap<String, usize> {
        self.model_worker_counts.read().unwrap().clone()
    }

    /// Clear all policies (useful for testing). Policy rebuilds also drop
    /// every affinity binding recorded against the old policy set.
    pub fn clear(&self) {
        let mut policies = self.model_policies.write().unwrap();
        policies.clear();
        let mut counts = self.model_worker_counts.write().unwrap();
        counts.clear();
        self.kv_affinity_table.clear();
    }

    fn create_encode_policy(config: EncodePolicy) -> Arc<dyn LoadBalancingPolicy> {
        match config {
            EncodePolicy::RoundRobin => Arc::new(RoundRobinPolicy::new()),
            EncodePolicy::Random => Arc::new(RandomPolicy::new()),
        }
    }

    /// Configure the independent E slot using only supported Encode policies.
    pub fn set_encode_policy(&self, config: EncodePolicy) {
        *self.encode_policy.write().unwrap() = Self::create_encode_policy(config);
    }

    /// Get the E policy instance without falling back to any consumer policy.
    pub fn get_encode_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        Arc::clone(&self.encode_policy.read().unwrap())
    }

    /// Set the prefill policy for PD mode
    pub fn set_prefill_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        let mut prefill_policy = self.prefill_policy.write().unwrap();
        *prefill_policy = Some(policy);
    }

    /// Set the decode policy for PD mode
    pub fn set_decode_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        let mut decode_policy = self.decode_policy.write().unwrap();
        *decode_policy = Some(policy);
    }

    /// Get the prefill policy for PD mode, or default if not set
    pub fn get_prefill_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        let prefill_policy = self.prefill_policy.read().unwrap();
        prefill_policy
            .as_ref()
            .map(Arc::clone)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Get the decode policy for PD mode, or default if not set
    pub fn get_decode_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        let decode_policy = self.decode_policy.read().unwrap();
        decode_policy
            .as_ref()
            .map(Arc::clone)
            .unwrap_or_else(|| self.get_default_policy())
    }
}

impl std::fmt::Debug for PolicyRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyRegistry")
            .field("model_policies", &self.model_policies)
            .field("model_worker_counts", &self.model_worker_counts)
            .field("default_policy", &self.default_policy.name())
            .finish()
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    #[test]
    fn encode_default_never_inherits_global_or_pd_policy() {
        for global in [
            PolicyConfig::Random,
            PolicyConfig::RoundRobin,
            PolicyConfig::RendezvousHash,
            PolicyConfig::CacheAware {
                cache_threshold: 0.3,
                balance_abs_threshold: 64,
                balance_rel_threshold: 1.5,
                eviction_interval_secs: 120,
                max_tree_size: 1000,
            },
        ] {
            let registry = PolicyRegistry::new(global);
            let encode = registry.get_encode_policy();
            assert_eq!(encode.name(), "round_robin");
            assert!(!Arc::ptr_eq(&encode, &registry.get_default_policy()));
            assert!(!Arc::ptr_eq(&encode, &registry.get_prefill_policy()));
            assert!(!Arc::ptr_eq(&encode, &registry.get_decode_policy()));
            registry.set_prefill_policy(Arc::new(RandomPolicy::new()));
            registry.set_decode_policy(Arc::new(RoundRobinPolicy::new()));
            assert!(Arc::ptr_eq(&encode, &registry.get_encode_policy()));
        }
    }

    #[test]
    fn encode_round_robin_state_is_independent_and_shared_by_registry_clones() {
        use crate::core::{BasicWorker, Worker, WorkerType};
        let workers: Vec<Arc<dyn Worker>> = ["http://e0:8000", "http://e1:8000"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Encode)) as Arc<dyn Worker>
            })
            .collect();
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let global = registry.get_default_policy();
        let encode = registry.get_encode_policy();
        assert_eq!(global.select_worker(&workers, None), Some(0));
        assert_eq!(encode.select_worker(&workers, None), Some(0));
        global.reset();
        let cloned = registry.clone();
        assert!(Arc::ptr_eq(&encode, &cloned.get_encode_policy()));
        assert_eq!(
            cloned.get_encode_policy().select_worker(&workers, None),
            Some(1)
        );
        assert_eq!(global.select_worker(&workers, None), Some(0));
    }

    #[test]
    fn encode_slot_survives_consumer_policy_cleanup_and_can_be_configured() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let cloned = registry.clone();
        let global = registry.get_default_policy();
        registry.set_encode_policy(EncodePolicy::Random);
        let encode = registry.get_encode_policy();
        assert_eq!(encode.name(), "random");
        assert!(Arc::ptr_eq(&encode, &cloned.get_encode_policy()));
        assert!(Arc::ptr_eq(&global, &registry.get_default_policy()));
        assert!(registry.get_worker_counts().is_empty());
        assert!(registry.get_all_mappings().is_empty());
        registry.on_worker_added("model", None);
        registry.on_worker_removed("model");
        registry.clear();
        assert!(Arc::ptr_eq(&encode, &registry.get_encode_policy()));
        registry.set_encode_policy(EncodePolicy::RoundRobin);
        assert_eq!(registry.get_encode_policy().name(), "round_robin");
        assert!(!Arc::ptr_eq(&registry.get_encode_policy(), &global));
    }

    #[test]
    fn encode_random_selects_only_healthy_encode_candidates() {
        use crate::core::{BasicWorker, Worker, WorkerType};
        let workers: Vec<Arc<dyn Worker>> = ["http://e0:8000", "http://e1:8000"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Encode)) as Arc<dyn Worker>
            })
            .collect();
        let registry = PolicyRegistry::new(PolicyConfig::RendezvousHash);
        registry.set_encode_policy(EncodePolicy::Random);
        let policy = registry.get_encode_policy();
        workers[0].set_healthy(false);
        for _ in 0..8 {
            assert_eq!(policy.select_worker(&workers, None), Some(1));
        }
        workers[1].set_healthy(false);
        assert_eq!(policy.select_worker(&workers, None), None);
        assert_eq!(policy.select_worker(&[], None), None);
    }

    #[test]
    fn test_policy_registry_basic() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // First worker of a model sets the policy
        let policy1 = registry.on_worker_added("llama-3", Some("cache_aware"));
        assert_eq!(policy1.name(), "cache_aware");

        // Second worker of same model uses existing policy
        let policy2 = registry.on_worker_added("llama-3", Some("round_robin"));
        assert_eq!(policy2.name(), "cache_aware"); // Ignores hint, uses existing

        // Different model can have different policy
        let policy3 = registry.on_worker_added("gpt-4", Some("random"));
        assert_eq!(policy3.name(), "random");

        // Check mappings
        let mappings = registry.get_all_mappings();
        assert_eq!(mappings.get("llama-3").unwrap(), "cache_aware");
        assert_eq!(mappings.get("gpt-4").unwrap(), "random");

        // Check worker counts
        let counts = registry.get_worker_counts();
        assert_eq!(*counts.get("llama-3").unwrap(), 2);
        assert_eq!(*counts.get("gpt-4").unwrap(), 1);
    }

    #[test]
    fn test_policy_registry_cleanup() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // Add workers
        registry.on_worker_added("llama-3", Some("cache_aware"));
        registry.on_worker_added("llama-3", None);
        assert_eq!(registry.get_worker_counts().get("llama-3"), Some(&2));

        // Remove one worker - policy should remain
        registry.on_worker_removed("llama-3");
        assert!(registry.get_policy("llama-3").is_some());
        assert_eq!(registry.get_worker_counts().get("llama-3"), Some(&1));

        // Remove last worker - policy should be cleaned up
        registry.on_worker_removed("llama-3");
        assert!(registry.get_policy("llama-3").is_none());
        assert_eq!(registry.get_worker_counts().get("llama-3"), None);
    }

    #[test]
    fn test_default_policy() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // No hint, no template - uses default
        let policy = registry.on_worker_added("unknown-model", None);
        assert_eq!(policy.name(), "round_robin");

        // Get default directly
        let default = registry.get_default_policy();
        assert_eq!(default.name(), "round_robin");
    }
}
