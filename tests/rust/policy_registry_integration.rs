#![cfg(all(feature = "cpu", feature = "ready"))]

//! Integration tests for PolicyRegistry with RouterManager

use std::collections::HashMap;
use std::sync::Arc;
use vllm_router_rs::config::{PolicyConfig, RouterConfig};
use vllm_router_rs::core::WorkerRegistry;
use vllm_router_rs::policies::PolicyRegistry;
use vllm_router_rs::protocols::worker_spec::WorkerConfigRequest;
use vllm_router_rs::routers::router_manager::RouterManager;

fn manager_fixture() -> (Arc<RouterManager>, Arc<WorkerRegistry>, Arc<PolicyRegistry>) {
    let config = RouterConfig {
        enable_igw: true,
        policy: PolicyConfig::RoundRobin,
        ..Default::default()
    };
    let workers = Arc::new(WorkerRegistry::new());
    let policies = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    let manager = Arc::new(RouterManager::new(
        config,
        reqwest::Client::new(),
        workers.clone(),
        policies.clone(),
    ));
    (manager, workers, policies)
}

fn worker_config(model: &str) -> WorkerConfigRequest {
    WorkerConfigRequest {
        url: "http://worker:8000".to_string(),
        model_id: Some(model.to_string()),
        worker_type: None,
        priority: None,
        cost: None,
        labels: HashMap::new(),
        bootstrap_port: None,
    }
}

#[tokio::test]
async fn test_manager_encode_removal_preserves_consumer_policy() {
    use vllm_router_rs::core::{Worker, WorkerFactory};

    let (manager, workers, policies) = manager_fixture();
    manager.add_worker(worker_config("unknown")).await.unwrap();
    let policy = policies.get_policy("unknown").unwrap();
    let encoder: Arc<dyn Worker> =
        Arc::from(WorkerFactory::create_encode("http://encoder:8000".into()));
    workers.register(encoder);

    manager
        .remove_worker_from_registry("http://encoder:8000")
        .unwrap();
    assert!(workers.get_by_url("http://encoder:8000").is_none());
    assert!(workers.get_by_url("http://worker:8000").is_some());
    assert_eq!(policies.get_worker_counts().get("unknown"), Some(&1));
    assert!(Arc::ptr_eq(
        &policy,
        &policies.get_policy("unknown").unwrap()
    ));

    manager
        .remove_worker_from_registry("http://worker:8000")
        .unwrap();
    assert!(policies.get_worker_counts().is_empty());
    assert!(policies.get_policy("unknown").is_none());
}

async fn check_manager_encode_replacement(replacement_model: &str) {
    use vllm_router_rs::core::{Worker, WorkerFactory};

    // Public-method contract only: EPD + IGW remains unsupported. Exercise
    // both the same-model shortcut and the different-model replacement path.
    let (manager, workers, policies) = manager_fixture();
    let mut consumer = worker_config("unknown");
    consumer.url = "http://consumer:8000".into();
    manager.add_worker(consumer).await.unwrap();
    let policy = policies.get_policy("unknown").unwrap();
    let encoder: Arc<dyn Worker> =
        Arc::from(WorkerFactory::create_encode("http://worker:8000".into()));
    workers.register(encoder);

    manager
        .add_worker(worker_config(replacement_model))
        .await
        .unwrap();
    let expected = if replacement_model == "unknown" {
        HashMap::from([("unknown".to_string(), 2)])
    } else {
        HashMap::from([("unknown".to_string(), 1), ("new-model".to_string(), 1)])
    };
    assert_eq!(policies.get_worker_counts(), expected);
    assert!(Arc::ptr_eq(
        &policy,
        &policies.get_policy("unknown").unwrap()
    ));

    manager
        .remove_worker_from_registry("http://worker:8000")
        .unwrap();
    assert_eq!(
        policies.get_worker_counts(),
        HashMap::from([("unknown".to_string(), 1)])
    );
    assert!(Arc::ptr_eq(
        &policy,
        &policies.get_policy("unknown").unwrap()
    ));
    manager
        .remove_worker_from_registry("http://consumer:8000")
        .unwrap();
    assert!(policies.get_worker_counts().is_empty());
}

#[tokio::test]
async fn test_manager_encode_replacement_same_model_counts_consumer() {
    check_manager_encode_replacement("unknown").await;
}

#[tokio::test]
async fn test_manager_encode_replacement_different_model_preserves_old_consumers() {
    check_manager_encode_replacement("new-model").await;
}

#[tokio::test]
async fn test_policy_registry_with_router_manager() {
    // An explicit model bypasses server-info discovery, so no backend is needed.
    let (manager, workers, policies) = manager_fixture();
    let first = manager
        .add_worker(worker_config("model"))
        .await
        .unwrap()
        .worker
        .unwrap();
    let policy = policies.get_policy("model").unwrap();
    let second = manager
        .add_worker(worker_config("model"))
        .await
        .unwrap()
        .worker
        .unwrap();
    assert_eq!(first.id, second.id);
    assert_eq!(workers.get_by_model_fast("model").len(), 1);
    assert_eq!(policies.get_worker_counts().get("model"), Some(&1));
    assert!(Arc::ptr_eq(&policy, &policies.get_policy("model").unwrap()));
    manager
        .remove_worker_from_registry("http://worker:8000")
        .unwrap();
    assert!(workers.get_all().is_empty());
    assert!(policies.get_worker_counts().is_empty());
    assert!(policies.get_policy("model").is_none());
}

#[tokio::test]
async fn test_model_replacement_moves_policy_count() {
    let (manager, workers, policies) = manager_fixture();
    let first = manager
        .add_worker(worker_config("old-model"))
        .await
        .unwrap()
        .worker
        .unwrap();
    let second = manager
        .add_worker(worker_config("new-model"))
        .await
        .unwrap()
        .worker
        .unwrap();
    assert_eq!(first.id, second.id);
    assert!(!workers.has_model("old-model"));
    assert!(policies.get_policy("old-model").is_none());
    assert_eq!(
        policies.get_worker_counts(),
        HashMap::from([("new-model".to_string(), 1)])
    );
    manager
        .remove_worker_from_registry("http://worker:8000")
        .unwrap();
    assert!(policies.get_worker_counts().is_empty());
    assert!(policies.get_all_mappings().is_empty());
}

#[tokio::test]
async fn igw_replacement_cleans_previous_consumer_affinity_without_changing_counts() {
    use vllm_router_rs::policies::{CacheAwareCandidate, CacheAwarePolicy, PolicyFactory};

    let cache_config = PolicyConfig::CacheAware {
        cache_threshold: 0.3,
        balance_abs_threshold: 32,
        balance_rel_threshold: 1.5,
        eviction_interval_secs: 3600,
        max_tree_size: 1024,
    };
    for keep_old_model in [false, true] {
        for (old_role, new_role, new_model) in [
            ("prefill", "prefill", "old"),
            ("prefill", "prefill", "new"),
            ("prefill", "decode", "old"),
            ("decode", "prefill", "old"),
            ("regular", "prefill", "old"),
            ("prefill", "regular", "old"),
        ] {
            let config = RouterConfig {
                enable_igw: true,
                policy: cache_config.clone(),
                ..Default::default()
            };
            let workers = Arc::new(WorkerRegistry::new());
            let policies = Arc::new(PolicyRegistry::new(cache_config.clone()));
            policies.set_prefill_policy(PolicyFactory::create_from_config(&cache_config));
            policies.set_decode_policy(PolicyFactory::create_from_config(&cache_config));
            let manager = RouterManager::new(
                config,
                reqwest::Client::new(),
                workers.clone(),
                policies.clone(),
            );
            let mut initial = worker_config("old");
            initial.worker_type = Some(old_role.into());
            // The model policy must retain its identity on same-model replacement,
            // including when it was created from a hint instead of the default.
            initial.labels.insert("policy".into(), "cache_aware".into());
            manager.add_worker(initial).await.unwrap();
            if keep_old_model {
                let mut survivor = worker_config("old");
                survivor.url = "http://survivor:8000".into();
                manager.add_worker(survivor).await.unwrap();
            }
            let model_policy = policies.get_policy("old").unwrap();
            let role_policy = match old_role {
                "prefill" => policies.get_prefill_policy(),
                "decode" => policies.get_decode_policy(),
                _ => model_policy.clone(),
            };
            let prompt = "replacement cache affinity";
            for policy in [&model_policy, &role_policy] {
                let cache = policy.as_any().downcast_ref::<CacheAwarePolicy>().unwrap();
                cache.commit_placement("old", prompt, "http://worker:8000");
                cache.commit_placement("old", prompt, "http://survivor:8000");
            }
            let mut replacement = worker_config(new_model);
            replacement.worker_type = Some(new_role.into());
            manager.add_worker(replacement).await.unwrap();
            for policy in [&model_policy, &role_policy] {
                let cache = policy.as_any().downcast_ref::<CacheAwarePolicy>().unwrap();
                for (url, expected) in [("http://worker:8000", 0.0), ("http://survivor:8000", 1.0)]
                {
                    assert_eq!(
                        cache
                            .evaluate_initial_placement(
                                "old",
                                prompt,
                                &[CacheAwareCandidate {
                                    target_id: url,
                                    kv_pressure: 0.0,
                                }]
                            )
                            .unwrap()
                            .match_rate,
                        expected,
                        "{old_role} -> {new_role}, {new_model}, keep_old={keep_old_model}: {url}"
                    );
                }
            }
            let mut expected = HashMap::from([(new_model.to_string(), 1)]);
            if keep_old_model {
                *expected.entry("old".into()).or_insert(0) += 1;
            }
            assert_eq!(policies.get_worker_counts(), expected);
            if new_model == "old" {
                assert!(Arc::ptr_eq(
                    &model_policy,
                    &policies.get_policy("old").unwrap()
                ));
            } else {
                assert_eq!(policies.get_policy("old").is_some(), keep_old_model);
            }
            let current = workers.get_by_url("http://worker:8000").unwrap();
            assert_eq!(current.model_id(), new_model);
            manager.remove_worker_from_registry(current.url()).unwrap();
            if keep_old_model {
                manager
                    .remove_worker_from_registry("http://survivor:8000")
                    .unwrap();
            }
            assert!(policies.get_worker_counts().is_empty());
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_management_add_remove_keeps_policy_count_consistent() {
    let (manager, workers, policies) = manager_fixture();
    let mut tasks = Vec::new();
    for thread in 0..8 {
        let manager = manager.clone();
        tasks.push(tokio::spawn(async move {
            for iteration in 0..20 {
                let model = if (thread + iteration) % 2 == 0 {
                    "model-a"
                } else {
                    "model-b"
                };
                manager.add_worker(worker_config(model)).await.unwrap();
                let _ = manager.remove_worker_from_registry("http://worker:8000");
                tokio::task::yield_now().await;
            }
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }
    manager
        .add_worker(worker_config("final-model"))
        .await
        .unwrap();
    assert_eq!(workers.get_all().len(), 1);
    assert_eq!(
        policies.get_worker_counts(),
        HashMap::from([("final-model".to_string(), 1)])
    );
    manager
        .remove_worker_from_registry("http://worker:8000")
        .unwrap();
    assert!(workers.get_all().is_empty());
    assert!(policies.get_all_mappings().is_empty());
    assert!(policies.get_worker_counts().is_empty());
}

#[tokio::test]
async fn igw_kv_affinity_survives_load_changes_and_rebinds_after_registration() {
    use vllm_router_rs::policies::{kv_aware_ops, AffinityKey};

    let config = RouterConfig {
        enable_igw: true,
        kv_aware_session_affinity_ttl_secs: Some(60),
        ..Default::default()
    };
    let workers = Arc::new(WorkerRegistry::new());
    let policies = Arc::new(PolicyRegistry::new_with_affinity_ttl(
        config.policy.clone(),
        Some(std::time::Duration::from_secs(60)),
    ));
    let manager = RouterManager::new(
        config,
        reqwest::Client::new(),
        workers.clone(),
        policies.clone(),
    );
    let worker_config = |url: &str| WorkerConfigRequest {
        url: url.into(),
        model_id: Some("model".into()),
        worker_type: None,
        priority: None,
        cost: None,
        labels: HashMap::from([("policy".into(), "kv_aware".into())]),
        bootstrap_port: None,
    };
    for url in ["http://a", "http://b"] {
        manager.add_worker(worker_config(url)).await.unwrap();
    }
    let policy = policies.get_policy("model").unwrap();
    let ops = kv_aware_ops(policy.as_ref()).unwrap();
    let body = serde_json::json!({"session_id": "s"});
    let identity = vllm_router_rs::request_identity::resolve_affinity_identity(
        None,
        vllm_router_rs::request_identity::IdentityFields::from_body(&body),
    )
    .unwrap()
    .unwrap();
    let key = AffinityKey::from_identity(identity, "model");
    let select = || {
        let candidates = workers.get_by_model_fast("model");
        let receipt = ops
            .select_worker_with_receipt(&candidates, None, None, None, Some(&key), &[])
            .unwrap();
        let worker = candidates[receipt.worker_index].clone();
        receipt.slot.unwrap().release();
        worker.decrement_load();
        worker
    };
    let first = select();
    first.increment_load();
    assert_eq!(select().url(), first.url());

    manager.remove_worker_from_registry(first.url()).unwrap();
    manager
        .add_worker(worker_config(first.url()))
        .await
        .unwrap();
    workers.get_by_url(first.url()).unwrap().increment_load();
    assert_ne!(select().url(), first.url());
}

#[test]
fn test_policy_registry_cleanup() {
    use vllm_router_rs::config::PolicyConfig;
    use vllm_router_rs::policies::PolicyRegistry;

    let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

    // Add workers for a model
    let policy1 = registry.on_worker_added("model-1", Some("cache_aware"));
    assert_eq!(policy1.name(), "cache_aware");

    // Second worker uses existing policy
    let policy2 = registry.on_worker_added("model-1", Some("random"));
    assert_eq!(policy2.name(), "cache_aware"); // Should still be cache_aware

    // Verify policy exists
    assert!(registry.get_policy("model-1").is_some());

    // Remove first worker - policy should remain
    registry.on_worker_removed("model-1");
    assert!(registry.get_policy("model-1").is_some());

    // Remove second worker - policy should be cleaned up
    registry.on_worker_removed("model-1");
    assert!(registry.get_policy("model-1").is_none());

    println!("✓ PolicyRegistry cleanup test passed");
}

#[test]
fn test_policy_registry_multiple_models() {
    use vllm_router_rs::config::PolicyConfig;
    use vllm_router_rs::policies::PolicyRegistry;

    let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

    // Add workers for different models with different policies
    let llama_policy = registry.on_worker_added("llama-3", Some("cache_aware"));
    let gpt_policy = registry.on_worker_added("gpt-4", Some("random"));
    let mistral_policy = registry.on_worker_added("mistral", None); // Uses default

    assert_eq!(llama_policy.name(), "cache_aware");
    assert_eq!(gpt_policy.name(), "random");
    assert_eq!(mistral_policy.name(), "round_robin"); // Default

    // Verify all policies are stored
    assert!(registry.get_policy("llama-3").is_some());
    assert!(registry.get_policy("gpt-4").is_some());
    assert!(registry.get_policy("mistral").is_some());

    // Get all mappings
    let mappings = registry.get_all_mappings();
    assert_eq!(mappings.len(), 3);
    assert_eq!(mappings.get("llama-3").unwrap(), "cache_aware");
    assert_eq!(mappings.get("gpt-4").unwrap(), "random");
    assert_eq!(mappings.get("mistral").unwrap(), "round_robin");

    println!("✓ PolicyRegistry multiple models test passed");
}
