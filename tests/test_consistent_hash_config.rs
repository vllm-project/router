use std::collections::BTreeMap;
use std::sync::Arc;

use vllm_router_rs::config::PolicyConfig;
use vllm_router_rs::core::{BasicWorker, Worker, WorkerType};
use vllm_router_rs::policies::{
    ConsistentHashPolicy, LoadBalancingPolicy, PolicyFactory, PolicyRegistry,
};

fn workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
    urls.iter()
        .map(|url| {
            Arc::new(BasicWorker::new((*url).to_string(), WorkerType::Regular)) as Arc<dyn Worker>
        })
        .collect()
}

fn selected_url(
    policy: &dyn LoadBalancingPolicy,
    workers: &[Arc<dyn Worker>],
    request: &str,
) -> String {
    let index = policy
        .select_worker(workers, Some(request))
        .expect("consistent hash policy should select a worker");
    workers[index].url().to_string()
}

fn expected_url(urls: &[&str], key: &str, virtual_nodes: u32) -> String {
    let ring: BTreeMap<u64, &str> = urls
        .iter()
        .flat_map(|url| {
            (0..virtual_nodes).map(move |index| {
                (
                    ConsistentHashPolicy::fbi_hash(&format!("{url}:{index}")),
                    *url,
                )
            })
        })
        .collect();
    let hash = ConsistentHashPolicy::fbi_hash(key);

    ring.range(hash..)
        .next()
        .or_else(|| ring.iter().next())
        .expect("expected ring should not be empty")
        .1
        .to_string()
}

#[test]
fn factory_forwards_consistent_hash_virtual_nodes() {
    let urls = ["http://a:8000", "http://b:8000", "http://c:8000"];
    let workers = workers(&urls);
    let policy = PolicyFactory::create_from_config(&PolicyConfig::ConsistentHash {
        virtual_nodes: 1,
    });

    assert_eq!(
        selected_url(policy.as_ref(), &workers, r#"{"session_id":"0"}"#),
        expected_url(&urls, "session:0", 1)
    );
}

#[test]
fn registry_forwards_consistent_hash_virtual_nodes() {
    let urls = ["http://a:8000", "http://b:8000", "http://c:8000"];
    let workers = workers(&urls);
    let registry = PolicyRegistry::new(PolicyConfig::ConsistentHash { virtual_nodes: 1 });
    let policy = registry.get_default_policy();

    assert_eq!(
        selected_url(policy.as_ref(), &workers, r#"{"session_id":"0"}"#),
        expected_url(&urls, "session:0", 1)
    );
}
