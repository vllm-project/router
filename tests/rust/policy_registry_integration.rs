#![cfg(all(feature = "cpu", feature = "ready"))]

//! Integration tests for PolicyRegistry with RouterManager

use std::collections::HashMap;
use std::sync::Arc;
use vllm_router_rs::config::RouterConfig;
use vllm_router_rs::core::WorkerRegistry;
use vllm_router_rs::policies::PolicyRegistry;
use vllm_router_rs::protocols::worker_spec::WorkerConfigRequest;
use vllm_router_rs::routers::router_manager::RouterManager;

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

#[tokio::test]
async fn test_igw_cache_aware_routes_repeated_prompt_to_one_worker() {
    use vllm_router_rs::config::PolicyConfig;

    // In IGW mode workers are added through RouterManager. cache_aware needs
    // init_workers() to build the model's tree; without it every pick is random.
    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    let router_manager = RouterManager::new(
        RouterConfig {
            enable_igw: true,
            ..Default::default()
        },
        reqwest::Client::new(),
        worker_registry.clone(),
        policy_registry.clone(),
    );

    for url in ["http://worker1:8000", "http://worker2:8000"] {
        router_manager
            .add_worker(WorkerConfigRequest {
                url: url.to_string(),
                model_id: Some("llama-3".to_string()),
                worker_type: None,
                priority: None,
                cost: None,
                labels: HashMap::from([("policy".to_string(), "cache_aware".to_string())]),
                bootstrap_port: None,
            })
            .await
            .unwrap();
    }

    let policy = policy_registry.get_policy("llama-3").unwrap();
    let workers = worker_registry.get_by_model_fast("llama-3");
    let picks: Vec<_> = (0..20)
        .map(|_| policy.select_worker(&workers, Some("shared system prompt")))
        .collect();
    assert!(picks[0].is_some(), "{picks:?}");
    assert!(picks.iter().all(|pick| *pick == picks[0]), "{picks:?}");
}
