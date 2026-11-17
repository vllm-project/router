/// Policy Registry for managing model-to-policy mappings
///
/// This registry manages the dynamic assignment of load balancing policies to models.
/// When the first worker of a new model is added, it determines the policy for that model.
/// All subsequent workers of the same model use the established policy.
/// When the last worker of a model is removed, the policy mapping is cleaned up.
use super::{
    CacheAwareConfig, CacheAwarePolicy, ConsistentHashPolicy, CostParams, KvAffinityTable,
    KvAwarePolicy, LoadBalancingPolicy, PolicyFactory, PowerOfTwoPolicy, RandomPolicy,
    RendezvousHashPolicy, RoundRobinPolicy, StickyLeastLoadedPolicy, TransferCredits,
};
use crate::config::types::{KvTransferConfig, PolicyConfig};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
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

    /// Prefill policy for PD mode
    prefill_policy: Arc<RwLock<Option<Arc<dyn LoadBalancingPolicy>>>>,

    /// Decode policy for PD mode
    decode_policy: Arc<RwLock<Option<Arc<dyn LoadBalancingPolicy>>>>,

    /// Affinity table shared by every KV-aware policy this registry creates,
    /// bounding affinity state across models.
    kv_affinity_table: Arc<KvAffinityTable>,

    /// `kv_transfer` block of the main `kv_aware` policy config, applied to
    /// hint-created `kv_aware` policies. `None` when the main policy is not
    /// `kv_aware` or has no block.
    kv_transfer: Option<Arc<KvTransferConfig>>,

    /// Active-mode peer fetch kill switch, shared by every active-mode
    /// `kv_aware` policy this registry builds (the default policy and
    /// hint-created ones), so one store flips all of them, including
    /// policies created later. Starts enabled.
    peer_fetch_enabled: Arc<AtomicBool>,

    /// Serializes kill switch writers, so a write and the status it returns
    /// (and anything recorded from that status) are one step: two racing
    /// writers cannot report a state other than the one left in the flag.
    peer_fetch_lock: Arc<Mutex<()>>,
}

/// Peer fetch kill switch state, as reported by the admin endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PeerFetchStatus {
    /// The registry's shared flag.
    pub enabled: bool,
    /// Distinct active-mode `kv_aware` policy instances; empty when no
    /// policy is active.
    pub policies: Vec<PeerFetchPolicyStatus>,
}

/// One active-mode `kv_aware` policy instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PeerFetchPolicyStatus {
    /// Model ids mapped to this instance, sorted.
    pub models: Vec<String>,
    /// Whether this instance is the registry's default policy.
    pub default: bool,
    /// Whether this instance currently grants peer fetches.
    pub enabled: bool,
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
        let peer_fetch_enabled = Arc::new(AtomicBool::new(true));
        let default_policy = Self::create_policy_from_config(
            &default_policy_config,
            &kv_affinity_table,
            &peer_fetch_enabled,
        );
        let kv_transfer = match &default_policy_config {
            PolicyConfig::KvAware { kv_transfer, .. } => {
                kv_transfer.as_deref().cloned().map(Arc::new)
            }
            _ => None,
        };

        Self {
            model_policies: Arc::new(RwLock::new(HashMap::new())),
            model_worker_counts: Arc::new(RwLock::new(HashMap::new())),
            default_policy,
            prefill_policy: Arc::new(RwLock::new(None)),
            decode_policy: Arc::new(RwLock::new(None)),
            kv_affinity_table,
            kv_transfer,
            peer_fetch_enabled,
            peer_fetch_lock: Arc::new(Mutex::new(())),
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

    /// Called when a worker is removed
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

    /// The registry-owned KV affinity table. Its idle TTL carries the
    /// router-level session-affinity setting; policies created outside this
    /// registry must share this table or their affinity gate stays closed.
    pub fn kv_affinity_table(&self) -> Arc<KvAffinityTable> {
        Arc::clone(&self.kv_affinity_table)
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
            "kv_aware" => Arc::new(PolicyFactory::apply_kv_transfer(
                KvAwarePolicy::default().with_shared_affinity(Arc::clone(&self.kv_affinity_table)),
                self.kv_transfer.as_deref(),
                Some(&self.peer_fetch_enabled),
            )),
            "power_of_two" => Arc::new(PowerOfTwoPolicy::new()),
            "consistent_hash" => Arc::new(ConsistentHashPolicy::new()),
            "sticky_least_loaded" => Arc::new(StickyLeastLoadedPolicy::new()),
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
        peer_fetch_enabled: &Arc<AtomicBool>,
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
                kv_transfer,
                ..
            } => Arc::new(PolicyFactory::apply_kv_transfer(
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
                kv_transfer.as_deref(),
                Some(peer_fetch_enabled),
            )),
            PolicyConfig::PowerOfTwo { .. } => Arc::new(PowerOfTwoPolicy::new()),
            PolicyConfig::ConsistentHash { .. } => Arc::new(ConsistentHashPolicy::new()),
            PolicyConfig::StickyLeastLoaded => Arc::new(StickyLeastLoadedPolicy::new()),
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

    /// Runtime kill switch for active-mode peer KV fetch. Stores to the flag
    /// shared by every active-mode `kv_aware` policy this registry built or
    /// will build, so it takes effect for selections made after the call;
    /// in-flight fetches finish normally. Policies not built active are
    /// unaffected. Returns the resulting status.
    pub fn set_peer_fetch_enabled(&self, enabled: bool) -> PeerFetchStatus {
        let _guard = self
            .peer_fetch_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        self.peer_fetch_enabled.store(enabled, Ordering::Release);
        self.peer_fetch_status()
    }

    /// [`set_peer_fetch_enabled`](Self::set_peer_fetch_enabled), but only
    /// when at least one policy is in active mode; `None` (flag unchanged)
    /// otherwise. On success it also sets the
    /// `vllm_router_kv_transfer_peer_fetch_enabled` gauge and logs the new
    /// state at warn. The check, the store, the status, the gauge and the
    /// log happen under one lock, so concurrent callers are applied one after
    /// the other and the gauge always matches the flag.
    pub fn set_peer_fetch_enabled_if_active(&self, enabled: bool) -> Option<PeerFetchStatus> {
        let _guard = self
            .peer_fetch_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if self.peer_fetch_status().policies.is_empty() {
            return None;
        }
        self.peer_fetch_enabled.store(enabled, Ordering::Release);
        let status = self.peer_fetch_status();
        crate::metrics::RouterMetrics::set_kv_transfer_peer_fetch_enabled(status.enabled);
        warn!(
            enabled = status.enabled,
            "KV transfer peer fetch {} for all active-mode kv_aware policies",
            if status.enabled {
                "enabled"
            } else {
                "disabled"
            }
        );
        Some(status)
    }

    /// The kill switch flag and every distinct active-mode `kv_aware`
    /// policy instance (default and per-model), with the models mapped to
    /// each. PD prefill/decode policies are not built by the registry and
    /// are not listed.
    pub fn peer_fetch_status(&self) -> PeerFetchStatus {
        let enabled = self.peer_fetch_enabled.load(Ordering::Acquire);
        let mut entries: Vec<(Arc<dyn LoadBalancingPolicy>, PeerFetchPolicyStatus)> = Vec::new();
        let mut add = |policy: &Arc<dyn LoadBalancingPolicy>, model: Option<&str>| {
            let Some(kv) = policy
                .as_any()
                .downcast_ref::<KvAwarePolicy>()
                .filter(|kv| kv.is_peer_fetch_active_mode())
            else {
                return;
            };
            let index = match entries.iter().position(|(p, _)| Arc::ptr_eq(p, policy)) {
                Some(index) => index,
                None => {
                    entries.push((
                        Arc::clone(policy),
                        PeerFetchPolicyStatus {
                            models: Vec::new(),
                            default: Arc::ptr_eq(policy, &self.default_policy),
                            enabled: kv.peer_fetch_enabled(),
                        },
                    ));
                    entries.len() - 1
                }
            };
            if let Some(model) = model {
                entries[index].1.models.push(model.to_string());
            }
        };
        add(&self.default_policy, None);
        for (model, policy) in self.model_policies.read().unwrap().iter() {
            add(policy, Some(model));
        }
        let policies = entries
            .into_iter()
            .map(|(_, mut status)| {
                status.models.sort();
                status
            })
            .collect();
        PeerFetchStatus { enabled, policies }
    }

    /// Mark a session as finished across all registered policies.
    ///
    /// Session-aware policies (e.g. `sticky_least_loaded`) will release
    /// the replica capacity held by the session; all other policies ignore it.
    /// This fans out to the default policy, every per-model policy, and the
    /// prefill/decode policies (for PD mode), since we don't know which policy
    /// instance is tracking the session.
    pub fn finish_session(&self, session_id: &str) {
        self.default_policy.finish_session(session_id);

        {
            let policies = self.model_policies.read().unwrap();
            for policy in policies.values() {
                policy.finish_session(session_id);
            }
        }

        if let Some(policy) = self.prefill_policy.read().unwrap().as_ref() {
            policy.finish_session(session_id);
        }
        if let Some(policy) = self.decode_policy.read().unwrap().as_ref() {
            policy.finish_session(session_id);
        }
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

    /// Get the decode override, or the main policy when no override is set.
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

    fn kv_aware_main(kv_transfer: Option<crate::config::types::KvTransferConfig>) -> PolicyConfig {
        PolicyConfig::KvAware {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
            static_kv_event_sources: Default::default(),
            kv_transfer: kv_transfer.map(Box::new),
        }
    }

    #[test]
    fn hint_created_kv_aware_inherits_main_shadow_planner() {
        use crate::config::types::KvTransferMode::{Off, Shadow};
        let block = |mode| crate::config::types::KvTransferConfig {
            mode,
            ..Default::default()
        };
        let cases = [
            (kv_aware_main(Some(block(Shadow))), true),
            (kv_aware_main(Some(block(Off))), false),
            (kv_aware_main(None), false),
            (PolicyConfig::RoundRobin, false),
        ];
        for (main, expect_planner) in cases {
            let registry = PolicyRegistry::new(main);
            let policy = registry.on_worker_added("m", Some("kv_aware"));
            let policy = policy
                .as_any()
                .downcast_ref::<KvAwarePolicy>()
                .expect("kv_aware policy");
            assert_eq!(policy.has_transfer_planner(), expect_planner);
        }
    }

    fn active_block() -> crate::config::types::KvTransferConfig {
        crate::config::types::KvTransferConfig {
            mode: crate::config::types::KvTransferMode::Active,
            ..Default::default()
        }
    }

    fn kv(policy: &Arc<dyn LoadBalancingPolicy>) -> &KvAwarePolicy {
        policy
            .as_any()
            .downcast_ref::<KvAwarePolicy>()
            .expect("kv_aware policy")
    }

    #[test]
    fn peer_fetch_flag_is_shared_by_default_and_hint_created_policies() {
        let registry = PolicyRegistry::new(kv_aware_main(Some(active_block())));
        let default = registry.get_default_policy();
        let hinted = registry.on_worker_added("hinted", Some("kv_aware"));
        assert!(!Arc::ptr_eq(&default, &hinted));
        assert!(kv(&default).peer_fetch_enabled());
        assert!(kv(&hinted).peer_fetch_enabled());

        let status = registry.set_peer_fetch_enabled(false);
        assert!(!status.enabled);
        assert!(!kv(&default).peer_fetch_enabled());
        assert!(!kv(&hinted).peer_fetch_enabled());
        assert!(status.policies.iter().all(|policy| !policy.enabled));

        // A policy's own setter stores to the same shared flag.
        assert!(kv(&hinted).set_peer_fetch_enabled(true));
        assert!(kv(&default).peer_fetch_enabled());
        assert!(registry.peer_fetch_status().enabled);
    }

    #[test]
    fn hint_created_policy_after_disable_starts_disabled() {
        let registry = PolicyRegistry::new(kv_aware_main(Some(active_block())));
        registry.set_peer_fetch_enabled(false);
        let late = registry.on_worker_added("late", Some("kv_aware"));
        assert!(kv(&late).is_peer_fetch_active_mode());
        assert!(!kv(&late).peer_fetch_enabled());

        registry.set_peer_fetch_enabled(true);
        assert!(kv(&late).peer_fetch_enabled());
    }

    #[test]
    fn peer_fetch_status_lists_each_active_instance_once() {
        let registry = PolicyRegistry::new(kv_aware_main(Some(active_block())));
        // Two models share the default instance; one has its own.
        registry.on_worker_added("m2", None);
        registry.on_worker_added("m1", None);
        registry.on_worker_added("m1", None);
        registry.on_worker_added("own", Some("kv_aware"));
        // Not kv_aware: not listed.
        registry.on_worker_added("rr", Some("round_robin"));

        let mut status = registry.peer_fetch_status();
        status.policies.sort_by_key(|policy| !policy.default);
        assert_eq!(
            status,
            PeerFetchStatus {
                enabled: true,
                policies: vec![
                    PeerFetchPolicyStatus {
                        models: vec!["m1".into(), "m2".into()],
                        default: true,
                        enabled: true,
                    },
                    PeerFetchPolicyStatus {
                        models: vec!["own".into()],
                        default: false,
                        enabled: true,
                    },
                ],
            }
        );
    }

    #[test]
    fn policies_not_built_active_are_not_listed() {
        use crate::config::types::KvTransferMode::{Off, Shadow};
        let block = |mode| crate::config::types::KvTransferConfig {
            mode,
            ..Default::default()
        };
        for main in [
            kv_aware_main(Some(block(Shadow))),
            kv_aware_main(Some(block(Off))),
            kv_aware_main(None),
            PolicyConfig::RoundRobin,
        ] {
            let registry = PolicyRegistry::new(main);
            registry.on_worker_added("default", None);
            registry.on_worker_added("hinted", Some("kv_aware"));
            let status = registry.set_peer_fetch_enabled(false);
            assert!(status.policies.is_empty(), "{status:?}");
            assert!(!status.enabled);
        }
    }
}
