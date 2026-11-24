use super::*;
use crate::core::{BasicWorker, WorkerType};
use crate::kv_index::{ClearScope, HashMode, KvBlockIndexer, ResidencyOwner};
use crate::policies::KvRequestContext;
use crate::protocols::spec::{CacheIdentity, RoutingPrompt};

fn policy() -> KvAwarePolicy {
    KvAwarePolicy::default().with_smetric(
        serde_yaml::from_str(
            "TTFT_SLO_SECS: 20\nATTENTION_CROSSOVER_TOKENS: 0\nPREFILL_RATE: 1\nEVICTION_INTERVAL_SECS: 0\n",
        )
        .unwrap(),
    )
}

fn worker(url: &str) -> Arc<dyn Worker> {
    Arc::new(BasicWorker::new(url.into(), WorkerType::Regular))
}

fn info(source: &str, model: &str) -> WorkerKvInfo {
    WorkerKvInfo {
        instance_id: source.into(),
        block_size: Some(4),
        ranks: vec![0],
        default_rank: Some(0),
        cache_key: Some(CacheKey {
            model: Arc::from(model),
            hash_mode: HashMode::Sha256,
            block_size: 4,
        }),
    }
}

fn owner(source: &str, rank: u32) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: Arc::from(source),
        dp_rank: rank,
        incarnation: 0,
    }
}

fn store(index: &KvBlockIndexer, source: &str, rank: u32, tier: StorageTier) {
    index.store(
        0,
        owner(source, rank),
        tier,
        None,
        &[
            (
                Arc::from("1"),
                Arc::from("cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72"),
            ),
            (
                Arc::from("2"),
                Arc::from("f3e0813c89d0991f07fbb5027a68c0cb809e1effc0c1e8974664bd2e57b50024"),
            ),
            (
                Arc::from("3"),
                Arc::from("46e7a7e6ce942eae1e13925beded6e6321f99918099a108fdb32bb9510b8e88d"),
            ),
        ],
    );
}

fn context<'a>(prompt: &'a RoutingPrompt, tokens: &'a [u32]) -> RoutingContext<'a> {
    RoutingContext {
        prompt: Some(prompt),
        allow_affinity: true,
        colocated: true,
        route: "/v1/chat/completions",
        kv: Some(KvRequestContext {
            token_ids: tokens,
            lora_name: None,
            cache_salt: None,
        }),
    }
}

fn work(
    policy: &KvAwarePolicy,
    worker: &Arc<dyn Worker>,
    tokens: &[u32],
    chars: usize,
) -> (u64, u64, bool) {
    let prompt = RoutingPrompt {
        text: "x".repeat(chars),
        est_hit_chars: chars,
        cache_identity: CacheIdentity::Shared,
    };
    let (receipt, tracker) = policy
        .select_request_with_receipt(std::slice::from_ref(worker), &context(&prompt, tokens))
        .unwrap();
    let tracker = tracker.unwrap();
    (
        receipt.new_prefill_tokens,
        tracker.work,
        tracker.sample_rate,
    )
}

#[test]
fn block_credit_and_cost_units_survive_the_shared_selection_path() {
    let policy = policy();
    let index = Arc::new(KvBlockIndexer::new());
    let worker = worker("grpc://worker:1");
    policy.register_worker(
        worker.as_ref(),
        &info("engine", "model"),
        Some(index.clone()),
    );
    store(&index, "engine", 0, StorageTier::Device);
    for (n, remaining_tokens, remaining_work, sample_rate) in [
        (13, 1, 1, true),
        (9, 1, 1, true),
        (8, 4, 4, true),
        (7, 3, 3, true),
        (3, 3, 3, true),
        (0, 0, 0, false),
    ] {
        assert_eq!(
            work(&policy, &worker, &(1..=n).collect::<Vec<_>>(), 13),
            (remaining_tokens, remaining_work, sample_rate)
        );
    }
}

#[test]
fn event_removal_clear_and_suspension_change_admitted_work() {
    let policy = policy();
    let index = Arc::new(KvBlockIndexer::new());
    let worker = worker("grpc://worker:1");
    policy.register_worker(
        worker.as_ref(),
        &info("engine", "model"),
        Some(index.clone()),
    );
    let tokens: Vec<_> = (1..=13).collect();
    store(&index, "engine", 0, StorageTier::Device);
    index.remove(
        0,
        &owner("engine", 0),
        StorageTier::Device,
        &[Arc::from("2")],
    );
    assert_eq!(work(&policy, &worker, &tokens, 13), (9, 9, true));
    index.suspend_source(&Arc::from("engine"), 0);
    assert_eq!(work(&policy, &worker, &tokens, 13), (13, 13, false));
    index.resume_source(&Arc::from("engine"), 0);
    assert_eq!(work(&policy, &worker, &tokens, 13).0, 9);
    index.clear(&owner("engine", 0), ClearScope::Worker);
    assert_eq!(work(&policy, &worker, &tokens, 13), (13, 13, true));
}

#[test]
fn lower_tiers_and_other_namespaces_do_not_reduce_smetric_work() {
    let policy = policy();
    let index = Arc::new(KvBlockIndexer::new());
    let worker = worker("grpc://worker:1");
    policy.register_worker(
        worker.as_ref(),
        &info("engine", "model"),
        Some(index.clone()),
    );
    let tokens: Vec<_> = (1..=13).collect();
    store(&index, "engine", 0, StorageTier::HostPinned);
    assert_eq!(work(&policy, &worker, &tokens, 13).0, 13);
    store(&index, "engine", 0, StorageTier::Device);
    let prompt = RoutingPrompt {
        text: "x".repeat(13),
        est_hit_chars: 13,
        cache_identity: CacheIdentity::Shared,
    };
    for (lora_name, cache_salt) in [(Some("adapter"), None), (None, Some("private"))] {
        let mut ctx = context(&prompt, &tokens);
        ctx.kv.as_mut().unwrap().lora_name = lora_name;
        ctx.kv.as_mut().unwrap().cache_salt = cache_salt;
        let (receipt, tracker) = policy
            .select_request_with_receipt(std::slice::from_ref(&worker), &ctx)
            .unwrap();
        assert_eq!(receipt.new_prefill_tokens, 13);
        assert_eq!(tracker.unwrap().work, 13);
    }
}

#[test]
fn shared_queries_keep_model_cache_identities_separate() {
    let policy = policy();
    let cold_index = Arc::new(KvBlockIndexer::new());
    let hot_index = Arc::new(KvBlockIndexer::new());
    let cold = worker("grpc://cold:1");
    let hot = worker("grpc://hot:1");
    policy.register_worker(
        cold.as_ref(),
        &info("engine", "cold-model"),
        Some(cold_index),
    );
    policy.register_worker(
        hot.as_ref(),
        &info("engine", "hot-model"),
        Some(hot_index.clone()),
    );
    store(&hot_index, "engine", 0, StorageTier::Device);
    hot.increment_load();
    let prompt = RoutingPrompt {
        text: "x".repeat(13),
        est_hit_chars: 13,
        cache_identity: CacheIdentity::Shared,
    };
    let tokens: Vec<_> = (1..=13).collect();
    let (receipt, _) = policy
        .select_request_with_receipt(&[cold, hot], &context(&prompt, &tokens))
        .unwrap();
    assert_eq!(receipt.worker_index, 1);
    assert_eq!(receipt.new_prefill_tokens, 1);
}

#[test]
fn retired_worker_cannot_overwrite_or_remove_its_same_url_replacement() {
    let policy = policy();
    let index = Arc::new(KvBlockIndexer::new());
    let old = worker("grpc://same:1");
    let new = worker("grpc://same:1");
    let tokens: Vec<_> = (1..=13).collect();
    policy.register_worker(old.as_ref(), &info("old", "model"), Some(index.clone()));
    store(&index, "old", 0, StorageTier::Device);
    assert_eq!(work(&policy, &new, &tokens, 13).0, 13);
    policy.register_worker(new.as_ref(), &info("new", "model"), Some(index.clone()));
    store(&index, "new", 0, StorageTier::Device);
    policy.on_worker_removed(&old);
    policy.register_worker(old.as_ref(), &info("old", "model"), Some(index));
    assert_eq!(work(&policy, &new, &tokens, 13).0, 1);
    let prompt = RoutingPrompt {
        text: "x".repeat(13),
        est_hit_chars: 13,
        cache_identity: CacheIdentity::Shared,
    };
    assert!(policy
        .select_request_with_receipt(&[old], &context(&prompt, &tokens))
        .is_none());
}

#[test]
fn explicit_rank_and_confirmed_default_rank_are_required_for_binding() {
    let policy = policy();
    let index = Arc::new(KvBlockIndexer::new());
    let pinned = worker("grpc://worker:1@1");
    let unpinned = worker("grpc://worker:1");
    let mut metadata = info("engine", "model");
    metadata.ranks = vec![0, 1];
    metadata.default_rank = None;
    for worker in [&pinned, &unpinned] {
        policy.register_worker(worker.as_ref(), &metadata, Some(index.clone()));
    }
    let tokens: Vec<_> = (1..=13).collect();
    store(&index, "engine", 0, StorageTier::Device);
    assert_eq!(work(&policy, &pinned, &tokens, 13).0, 13);
    assert_eq!(work(&policy, &unpinned, &tokens, 13).0, 13);
    metadata.ranks = vec![0];
    policy.register_worker(unpinned.as_ref(), &metadata, Some(index.clone()));
    assert_eq!(work(&policy, &unpinned, &tokens, 13).0, 13);
    store(&index, "engine", 1, StorageTier::Device);
    assert_eq!(work(&policy, &pinned, &tokens, 13).0, 1);
}

#[test]
fn invalid_metadata_withdraws_cache_credit_without_blocking_routing() {
    let policy = policy();
    let index = Arc::new(KvBlockIndexer::new());
    let worker = worker("grpc://worker:1");
    let mut metadata = info("engine", "model");
    policy.register_worker(worker.as_ref(), &metadata, Some(index.clone()));
    let tokens: Vec<_> = (1..=13).collect();
    store(&index, "engine", 0, StorageTier::Device);
    assert_eq!(work(&policy, &worker, &tokens, 13).0, 1);
    metadata.block_size = Some(0);
    policy.register_worker(worker.as_ref(), &metadata, Some(index));
    assert_eq!(work(&policy, &worker, &tokens, 13), (13, 13, false));
}

#[test]
fn ttft_budget_overrides_event_affinity_until_pending_work_is_released() {
    let policy = policy();
    let index = Arc::new(KvBlockIndexer::new());
    let cold = worker("grpc://cold:1");
    let hot = worker("grpc://hot:1");
    policy.register_worker(hot.as_ref(), &info("hot", "model"), Some(index.clone()));
    store(&index, "hot", 0, StorageTier::Device);
    hot.increment_load();
    let prompt = RoutingPrompt {
        text: "x".repeat(13),
        est_hit_chars: 13,
        cache_identity: CacheIdentity::Shared,
    };
    let tokens: Vec<_> = (1..=13).collect();
    let workers = [cold, hot.clone()];
    assert_eq!(
        policy
            .select_request_with_receipt(&workers, &context(&prompt, &tokens))
            .unwrap()
            .0
            .worker_index,
        1
    );
    let busy_prompt = RoutingPrompt {
        text: "y".repeat(40),
        est_hit_chars: 0,
        cache_identity: CacheIdentity::Shared,
    };
    let busy_tokens: Vec<_> = (100..140).collect();
    let busy = policy
        .select_request_with_receipt(
            std::slice::from_ref(&hot),
            &context(&busy_prompt, &busy_tokens),
        )
        .unwrap()
        .1
        .unwrap();
    assert_eq!(
        policy
            .select_request_with_receipt(&workers, &context(&prompt, &tokens))
            .unwrap()
            .0
            .worker_index,
        0
    );
    drop(busy);
    assert_eq!(
        policy
            .select_request_with_receipt(&workers, &context(&prompt, &tokens))
            .unwrap()
            .0
            .worker_index,
        1
    );
    index.clear(&owner("hot", 0), ClearScope::Worker);
    assert_eq!(
        policy
            .select_request_with_receipt(&workers, &context(&prompt, &tokens))
            .unwrap()
            .0
            .worker_index,
        0
    );
}

#[test]
fn default_mode_does_not_use_an_unidentified_match_provider() {
    let policy = KvAwarePolicy::default();
    let index = Arc::new(KvBlockIndexer::new());
    let worker = worker("grpc://worker:1");
    let mut metadata = info("engine", "model");
    metadata.cache_key = None;
    store(&index, "engine", 0, StorageTier::Device);
    policy.register_worker(worker.as_ref(), &metadata, Some(index));
    let receipt = policy
        .select_worker_with_receipt(&[worker], Some(&[1, 2, 3, 4, 5]), None, None)
        .unwrap();
    assert_eq!(receipt.new_prefill_tokens, 5);
}
