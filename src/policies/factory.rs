//! Factory for creating load balancing policies

use super::{
    CacheAwareConfig, CacheAwarePolicy, ConsistentHashPolicy, CostParams, KvAffinityTable,
    KvAwarePolicy, LoadBalancingPolicy, PowerOfTwoPolicy, RandomPolicy, RendezvousHashPolicy,
    RoundRobinPolicy, StickyLeastLoadedPolicy, TransferCredits,
};
use crate::config::PolicyConfig;
use std::sync::Arc;

/// Factory for creating policy instances
pub struct PolicyFactory;

/// Build a `kv_aware` policy from its config.
///
/// Shared by [`PolicyFactory`] and `PolicyRegistry` so backend selection lives
/// in exactly one place; duplicating it is how "conductor without a URL" turns
/// into a silent local fallback in one path but not the other.
pub(crate) fn build_kv_aware_policy(
    config: &crate::config::KvAwareConfig,
    kv_affinity_table: Option<&Arc<KvAffinityTable>>,
) -> Arc<dyn LoadBalancingPolicy> {
    let params = CostParams {
        router_temperature: config.router_temperature,
        transfer_credits: TransferCredits {
            device: 1.0,
            host_pinned: config.kv_penalty_host,
            external: config.kv_penalty_external,
            disk: config.kv_penalty_disk,
        },
        ..CostParams::default()
    };
    let policy = match (
        config.kv_index_backend.as_str(),
        config.conductor_url.as_deref(),
    ) {
        ("conductor", Some(url)) => KvAwarePolicy::new_with_conductor(
            params,
            crate::policies::ConductorOptions {
                url: url.to_string(),
                connect_timeout: std::time::Duration::from_millis(
                    config.conductor_connect_timeout_ms,
                ),
                query_timeout: std::time::Duration::from_millis(config.conductor_query_timeout_ms),
                register_timeout: std::time::Duration::from_millis(
                    config.conductor_register_timeout_ms,
                ),
                hash_profile: config.conductor_hash_profile.clone().unwrap_or_default(),
            },
            config.conductor_register,
        ),
        (backend, _) => {
            if backend == "conductor" {
                // Validation rejects this combination; keep a loud fallback for
                // programmatic callers that bypass `ConfigValidator`.
                tracing::warn!(
                    "kv_index_backend=conductor without conductor_url; falling back to the local index"
                );
            }
            KvAwarePolicy::new(params)
        }
    };
    let policy = match kv_affinity_table {
        Some(table) => policy.with_shared_affinity(Arc::clone(table)),
        None => policy,
    };
    Arc::new(policy)
}

impl PolicyFactory {
    /// Create a policy from configuration
    ///
    /// `kv_affinity_table` is the registry-owned table whose idle TTL carries
    /// the router-level `--kv-aware-session-affinity-ttl-secs` setting. KV-aware
    /// policies built without it get a TTL-less table, which silently disables
    /// session affinity no matter what the CLI flag said — so callers that own
    /// a registry must always thread the table through here.
    pub fn create_from_config(
        config: &PolicyConfig,
        kv_affinity_table: Option<&Arc<KvAffinityTable>>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        match config {
            PolicyConfig::Random => Arc::new(RandomPolicy::new()),
            PolicyConfig::RoundRobin => Arc::new(RoundRobinPolicy::new()),
            PolicyConfig::PowerOfTwo { .. } => Arc::new(PowerOfTwoPolicy::new()),
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            } => {
                let config = CacheAwareConfig {
                    cache_threshold: *cache_threshold,
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    eviction_interval_secs: *eviction_interval_secs,
                    max_tree_size: *max_tree_size,
                };
                Arc::new(CacheAwarePolicy::with_config(config))
            }
            PolicyConfig::KvAware(config) => build_kv_aware_policy(config, kv_affinity_table),
            PolicyConfig::ConsistentHash { virtual_nodes: _ } => {
                // Note: virtual_nodes parameter is available but not currently used
                // The consistent hash policy uses a hardcoded value for now
                Arc::new(ConsistentHashPolicy::new())
            }
            PolicyConfig::StickyLeastLoaded => Arc::new(StickyLeastLoadedPolicy::new()),
            PolicyConfig::RendezvousHash => Arc::new(RendezvousHashPolicy::new()),
        }
    }

    /// Create a policy by name (for dynamic loading)
    pub fn create_by_name(name: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        match name.to_lowercase().as_str() {
            "random" => Some(Arc::new(RandomPolicy::new())),
            "round_robin" | "roundrobin" => Some(Arc::new(RoundRobinPolicy::new())),
            "power_of_two" | "poweroftwo" => Some(Arc::new(PowerOfTwoPolicy::new())),
            "cache_aware" | "cacheaware" => Some(Arc::new(CacheAwarePolicy::new())),
            "kv_aware" | "kvaware" => Some(Arc::new(KvAwarePolicy::default())),
            "consistent_hash" | "consistenthash" => Some(Arc::new(ConsistentHashPolicy::new())),
            "sticky_least_loaded" | "stickyleastloaded" => {
                Some(Arc::new(StickyLeastLoadedPolicy::new()))
            }
            "rendezvous_hash" | "rendezvoushash" => Some(Arc::new(RendezvousHashPolicy::new())),
            _ => None,
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::policies::kv_aware_ops;

    #[test]
    fn test_create_from_config() {
        // Test Random
        let policy = PolicyFactory::create_from_config(&PolicyConfig::Random, None);
        assert_eq!(policy.name(), "random");

        // Test RoundRobin
        let policy = PolicyFactory::create_from_config(&PolicyConfig::RoundRobin, None);
        assert_eq!(policy.name(), "round_robin");

        // Test PowerOfTwo
        let policy = PolicyFactory::create_from_config(
            &PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 60,
            },
            None,
        );
        assert_eq!(policy.name(), "power_of_two");

        // Test CacheAware
        let policy = PolicyFactory::create_from_config(
            &PolicyConfig::CacheAware {
                cache_threshold: 0.7,
                balance_abs_threshold: 10,
                balance_rel_threshold: 1.5,
                eviction_interval_secs: 30,
                max_tree_size: 1000,
            },
            None,
        );
        assert_eq!(policy.name(), "cache_aware");

        let config = PolicyConfig::KvAware(Box::new(crate::config::KvAwareConfig {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
            static_kv_event_sources: Default::default(),
            kv_index_backend: "local".to_string(),
            conductor_url: None,
            conductor_register: true,
            conductor_connect_timeout_ms: 100,
            conductor_query_timeout_ms: 10,
            conductor_register_timeout_ms: 1000,
            conductor_hash_profile: None,
        }));
        let policy = PolicyFactory::create_from_config(&config, None);
        assert_eq!(policy.name(), "kv_aware");

        // Without a shared table the factory-built policy must NOT report
        // session affinity (this is exactly the wiring that silently disabled
        // the CLI TTL flag in the PD path).
        let ttlless = PolicyFactory::create_from_config(&config, None);
        assert!(!kv_aware_ops(ttlless.as_ref())
            .expect("factory builds a KvAwarePolicy")
            .session_affinity_enabled());

        // With the registry-owned TTL'd table the gate opens.
        let table = Arc::new(KvAffinityTable::with_ttl(Some(
            std::time::Duration::from_secs(1800),
        )));
        let wired = PolicyFactory::create_from_config(&config, Some(&table));
        assert!(kv_aware_ops(wired.as_ref())
            .expect("factory builds a KvAwarePolicy")
            .session_affinity_enabled());
    }

    #[test]
    fn test_create_by_name() {
        assert!(PolicyFactory::create_by_name("random").is_some());
        assert!(PolicyFactory::create_by_name("RANDOM").is_some());
        assert!(PolicyFactory::create_by_name("round_robin").is_some());
        assert!(PolicyFactory::create_by_name("RoundRobin").is_some());
        assert!(PolicyFactory::create_by_name("power_of_two").is_some());
        assert!(PolicyFactory::create_by_name("PowerOfTwo").is_some());
        assert!(PolicyFactory::create_by_name("cache_aware").is_some());
        assert!(PolicyFactory::create_by_name("CacheAware").is_some());
        assert!(PolicyFactory::create_by_name("kv_aware").is_some());
        assert!(PolicyFactory::create_by_name("consistent_hash").is_some());
        assert!(PolicyFactory::create_by_name("ConsistentHash").is_some());
        assert!(PolicyFactory::create_by_name("sticky_least_loaded").is_some());
        assert!(PolicyFactory::create_by_name("StickyLeastLoaded").is_some());
        assert!(PolicyFactory::create_by_name("rendezvous_hash").is_some());
        assert!(PolicyFactory::create_by_name("RendezvousHash").is_some());
        assert!(PolicyFactory::create_by_name("unknown").is_none());
    }
}
