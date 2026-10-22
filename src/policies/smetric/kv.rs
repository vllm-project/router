use crate::core::Worker;
use crate::kv_index::{
    CacheKey, Locality, MatchQuery, RoutableTarget, StorageTier, TierMatch, WorkerKvInfo,
};
use crate::policies::{KvRequestContext, TieredMatchProvider};
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
struct Binding {
    target: RoutableTarget,
    key: CacheKey,
    provider: Arc<dyn TieredMatchProvider>,
}

#[derive(Default)]
pub(super) struct KvResidency {
    bindings: DashMap<usize, Binding>,
}

impl std::fmt::Debug for KvResidency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvResidency")
            .field("targets", &self.bindings.len())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PrefixMatch {
    pub chars: usize,
    pub observed: bool,
}

impl KvResidency {
    pub(super) fn register(
        &self,
        id: usize,
        worker: &dyn Worker,
        info: &WorkerKvInfo,
        provider: Option<Arc<dyn TieredMatchProvider>>,
    ) {
        let rank = crate::backend::parse_dp_rank(worker.url())
            .or_else(|| worker.dp_rank().and_then(|rank| u32::try_from(rank).ok()))
            .or(info.default_rank);
        let Some(rank) = rank else {
            self.bindings.remove(&id);
            return;
        };
        if let (Some(key), Some(provider)) = (info.cache_key.as_ref(), provider) {
            if key.block_size > 0
                && info.block_size == Some(key.block_size)
                && info.ranks.contains(&rank)
                && !info.instance_id.is_empty()
            {
                self.bindings.insert(
                    id,
                    Binding {
                        target: RoutableTarget {
                            instance_id: Arc::from(info.instance_id.as_str()),
                            dp_rank: rank,
                        },
                        key: key.clone(),
                        provider,
                    },
                );
                return;
            }
        }
        self.bindings.remove(&id);
    }

    pub(super) fn remove(&self, id: usize) {
        self.bindings.remove(&id);
    }

    pub(super) fn prefix_matches(
        &self,
        workers: &[Arc<dyn Worker>],
        request: &KvRequestContext<'_>,
        chars: usize,
    ) -> Vec<PrefixMatch> {
        let mut result = vec![PrefixMatch::default(); workers.len()];
        if request.token_ids.is_empty() || request.cache_salt.is_some_and(|salt| !salt.is_empty()) {
            return result;
        }
        let mut queried: HashMap<CacheKey, Vec<TierMatch>> = HashMap::new();
        for (index, worker) in workers.iter().enumerate() {
            let Some(binding) = self
                .bindings
                .get(&super::SMetricPolicy::instance_id(worker))
                .map(|binding| binding.clone())
            else {
                continue;
            };
            let block_size = binding.key.block_size as usize;
            let complete = request.token_ids.len() / block_size * block_size;
            if complete == 0 {
                continue;
            }
            let matches = queried.entry(binding.key.clone()).or_insert_with(|| {
                binding.provider.find_tiered_matches(&MatchQuery {
                    group_idx: 0,
                    local_hashes: crate::kv_index::local_block_hash_strings(
                        &request.token_ids[..complete],
                        binding.key.block_size,
                        request.lora_name,
                    ),
                    tiers_of_interest: vec![StorageTier::Device],
                })
            });
            let blocks = matches
                .iter()
                .filter(|hit| {
                    hit.target == binding.target
                        && hit.tier == StorageTier::Device
                        && hit.locality == Locality::Local
                })
                .flat_map(|hit| hit.segments.iter())
                .filter(|(start, _)| *start == 0)
                .map(|(_, end)| *end as usize)
                .max()
                .unwrap_or(0);
            let tokens =
                blocks.min(request.token_ids.len().saturating_sub(1) / block_size) * block_size;
            result[index] = PrefixMatch {
                chars: ((chars as u128 * tokens as u128) / request.token_ids.len() as u128)
                    as usize,
                observed: blocks > 0,
            };
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use crate::kv_index::{HashMode, KvBlockIndexer, ResidencyOwner};
    use crate::policies::{KvEventConsumer, LoadBalancingPolicy, RoutingContext, SMetricPolicy};
    use crate::protocols::spec::RoutingPrompt;

    fn worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(BasicWorker::new(url.into(), WorkerType::Regular))
    }

    fn owner(source: &str) -> ResidencyOwner {
        ResidencyOwner::Worker {
            source: Arc::from(source),
            dp_rank: 0,
            incarnation: 0,
        }
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

    fn store(index: &KvBlockIndexer, source: &str, tier: StorageTier) {
        let blocks = [
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
        ];
        index.store(0, owner(source), tier, None, &blocks);
    }

    fn observe(
        residency: &KvResidency,
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        chars: usize,
    ) -> Vec<PrefixMatch> {
        residency.prefix_matches(
            workers,
            &KvRequestContext {
                token_ids: tokens,
                lora_name: None,
                cache_salt: None,
            },
            chars,
        )
    }

    #[test]
    fn binding_matches_dispatched_rank_and_rejects_ambiguous_endpoints() {
        let index = Arc::new(KvBlockIndexer::new());
        let residency = KvResidency::default();
        let pinned = worker("grpc://worker:1@1");
        let unpinned = worker("grpc://worker:1");
        let mut info = info("engine", "model");
        info.ranks = vec![0, 1];
        info.default_rank = None;
        for worker in [&pinned, &unpinned] {
            residency.register(
                SMetricPolicy::instance_id(worker),
                worker.as_ref(),
                &info,
                Some(index.clone()),
            );
        }
        store(&index, "engine", StorageTier::Device);
        let ids: Vec<u32> = (1..=13).collect();
        assert_eq!(
            observe(&residency, std::slice::from_ref(&pinned), &ids, 13)[0].chars,
            0
        );
        assert_eq!(
            observe(&residency, std::slice::from_ref(&unpinned), &ids, 13)[0].chars,
            0
        );
        info.ranks = vec![0];
        residency.register(
            SMetricPolicy::instance_id(&unpinned),
            unpinned.as_ref(),
            &info,
            Some(index.clone()),
        );
        assert_eq!(observe(&residency, &[unpinned], &ids, 13)[0].chars, 0);
        index.store(
            0,
            ResidencyOwner::Worker {
                source: Arc::from("engine"),
                dp_rank: 1,
                incarnation: 0,
            },
            StorageTier::Device,
            None,
            &[(
                Arc::from("rank1"),
                Arc::from("cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72"),
            )],
        );
        assert_eq!(observe(&residency, &[pinned], &ids, 13)[0].chars, 4);
    }

    #[test]
    fn device_prefix_conversion_preserves_block_alignment_and_character_units() {
        let index = Arc::new(KvBlockIndexer::new());
        let residency = KvResidency::default();
        let worker = worker("grpc://worker:1");
        residency.register(
            SMetricPolicy::instance_id(&worker),
            worker.as_ref(),
            &info("engine", "model"),
            Some(index.clone()),
        );
        store(&index, "engine", StorageTier::Device);
        let workers = [worker];
        for (tokens, chars, expected) in [
            (13, 13, 12),
            (9, 13, 11),
            (8, 13, 6),
            (7, 13, 7),
            (3, 13, 0),
            (0, 13, 0),
        ] {
            let ids: Vec<u32> = (1..=tokens).collect();
            assert_eq!(
                observe(&residency, &workers, &ids, chars)[0].chars,
                expected
            );
        }
    }

    #[test]
    fn removed_blocks_clear_and_source_invalidation_stop_prefix_credit() {
        let index = Arc::new(KvBlockIndexer::new());
        let residency = KvResidency::default();
        let worker = worker("grpc://worker:1");
        residency.register(
            SMetricPolicy::instance_id(&worker),
            worker.as_ref(),
            &info("engine", "model"),
            Some(index.clone()),
        );
        let workers = [worker];
        let ids: Vec<u32> = (1..=13).collect();
        store(&index, "engine", StorageTier::Device);
        index.remove(0, &owner("engine"), StorageTier::Device, &[Arc::from("2")]);
        assert_eq!(observe(&residency, &workers, &ids, 13)[0].chars, 4);
        index.suspend_source(&Arc::from("engine"), 0);
        let suspended = observe(&residency, &workers, &ids, 13);
        assert_eq!(suspended[0].chars, 0);
        assert!(!suspended[0].observed);
        index.resume_source(&Arc::from("engine"), 0);
        assert_eq!(observe(&residency, &workers, &ids, 13)[0].chars, 4);
        index.clear(&owner("engine"), crate::kv_index::ClearScope::Worker);
        assert_eq!(observe(&residency, &workers, &ids, 13)[0].chars, 0);
    }

    #[test]
    fn lower_tiers_and_other_request_namespaces_receive_no_device_credit() {
        let index = Arc::new(KvBlockIndexer::new());
        let residency = KvResidency::default();
        let worker = worker("grpc://worker:1");
        residency.register(
            SMetricPolicy::instance_id(&worker),
            worker.as_ref(),
            &info("engine", "model"),
            Some(index.clone()),
        );
        let workers = [worker];
        let ids: Vec<u32> = (1..=13).collect();
        store(&index, "engine", StorageTier::HostPinned);
        assert_eq!(observe(&residency, &workers, &ids, 13)[0].chars, 0);
        store(&index, "engine", StorageTier::Device);
        for (lora_name, cache_salt) in [(Some("adapter"), None), (None, Some("private"))] {
            let hit = residency.prefix_matches(
                &workers,
                &KvRequestContext {
                    token_ids: &ids,
                    lora_name,
                    cache_salt,
                },
                13,
            );
            assert_eq!(hit[0].chars, 0);
            assert!(!hit[0].observed);
        }
    }

    #[test]
    fn cache_identity_separates_same_target_names_across_models() {
        let cold_index = Arc::new(KvBlockIndexer::new());
        let hot_index = Arc::new(KvBlockIndexer::new());
        let residency = KvResidency::default();
        let cold = worker("grpc://cold:1");
        let hot = worker("grpc://hot:1");
        residency.register(
            SMetricPolicy::instance_id(&cold),
            cold.as_ref(),
            &info("engine", "cold-model"),
            Some(cold_index),
        );
        residency.register(
            SMetricPolicy::instance_id(&hot),
            hot.as_ref(),
            &info("engine", "hot-model"),
            Some(hot_index.clone()),
        );
        store(&hot_index, "engine", StorageTier::Device);
        let ids: Vec<u32> = (1..=13).collect();
        let hits = observe(&residency, &[cold, hot], &ids, 13);
        assert_eq!(
            hits.iter().map(|hit| hit.chars).collect::<Vec<_>>(),
            vec![0, 12]
        );
    }

    #[test]
    fn replacement_binding_survives_removal_of_same_url_old_worker() {
        let index = Arc::new(KvBlockIndexer::new());
        let residency = KvResidency::default();
        let old = worker("grpc://same:1");
        let new = worker("grpc://same:1");
        residency.register(
            SMetricPolicy::instance_id(&old),
            old.as_ref(),
            &info("old", "model"),
            Some(index.clone()),
        );
        residency.register(
            SMetricPolicy::instance_id(&new),
            new.as_ref(),
            &info("new", "model"),
            Some(index.clone()),
        );
        store(&index, "old", StorageTier::Device);
        let ids: Vec<u32> = (1..=13).collect();
        assert_eq!(
            observe(&residency, std::slice::from_ref(&new), &ids, 13)[0].chars,
            0
        );
        residency.remove(SMetricPolicy::instance_id(&old));
        store(&index, "new", StorageTier::Device);
        assert_eq!(observe(&residency, &[new], &ids, 13)[0].chars, 12);
    }

    #[test]
    fn event_hits_feed_smetric_budget_without_creating_history_tenants() {
        let policy = SMetricPolicy::new(serde_yaml::from_str("C_LIN: 1\nC_ATT: 0\nPREFILL_RATE: 1\nSLACK: 1\nHIT_RATIO: 0.1\nTTFT_SLO_BASE: 20\nTTFT_SLO_PER_CHAR: 0\nEVICTION_INTERVAL_SECS: 0\n").unwrap());
        let index = Arc::new(KvBlockIndexer::new());
        let cold = worker("grpc://cold:1");
        let hot = worker("grpc://hot:1");
        policy.kv.register(
            SMetricPolicy::instance_id(&hot),
            hot.as_ref(),
            &info("hot", "model"),
            Some(index.clone()),
        );
        store(&index, "hot", StorageTier::Device);
        hot.increment_load();
        let workers = [cold, hot.clone()];
        let ids: Vec<u32> = (1..=13).collect();
        let prompt = RoutingPrompt {
            text: "abcdefghijklm".into(),
            est_hit_chars: 10,
        };
        let context = RoutingContext {
            prompt: Some(&prompt),
            allow_affinity: true,
            colocated: true,
            route: "/v1/chat/completions",
            kv: Some(KvRequestContext {
                token_ids: &ids,
                lora_name: None,
                cache_salt: None,
            }),
        };
        let (selected, tracker) = policy
            .select_request(&workers, None, None, &context)
            .unwrap();
        assert_eq!(selected, 1);
        assert_eq!(
            policy
                .states
                .get(&SMetricPolicy::instance_id(&hot))
                .unwrap()
                .pending
                .load(std::sync::atomic::Ordering::Acquire),
            1
        );
        assert!(policy.trees.is_empty());
        drop(tracker);
        let busy = policy.track(&hot, 100);
        assert_eq!(
            policy
                .select_request(&workers, None, None, &context)
                .unwrap()
                .0,
            0
        );
        drop(busy);
        index.clear(&owner("hot"), crate::kv_index::ClearScope::Worker);
        assert_eq!(
            policy
                .select_request(&workers, None, None, &context)
                .unwrap()
                .0,
            0
        );
        assert!(policy.trees.is_empty());
    }

    #[test]
    fn unknown_event_state_releases_work_without_training_or_using_tree_history() {
        use std::sync::atomic::Ordering;
        let policy = SMetricPolicy::new(serde_yaml::from_str(
            "C_LIN: 1\nC_ATT: 0\nSLACK: 1\nHIT_RATIO: 0.1\nTTFT_SLO_BASE: 20\nTTFT_SLO_PER_CHAR: 0\nEVICTION_INTERVAL_SECS: 0\n"
        ).unwrap());
        let worker = worker("grpc://worker:1");
        let workers = [worker.clone()];
        let prompt = RoutingPrompt {
            text: "abcdefghijklm".into(),
            est_hit_chars: 10,
        };
        let mut context = RoutingContext {
            prompt: Some(&prompt),
            allow_affinity: true,
            colocated: true,
            route: "/v1/chat/completions",
            kv: None,
        };
        drop(policy.select_request(&workers, None, None, &context));
        assert!(!policy.trees.is_empty());
        let ids: Vec<u32> = (1..=13).collect();
        context.kv = Some(KvRequestContext {
            token_ids: &ids,
            lora_name: None,
            cache_salt: None,
        });
        let (_, tracker) = policy
            .select_request(&workers, None, None, &context)
            .unwrap();
        let state = policy
            .states
            .get(&SMetricPolicy::instance_id(&worker))
            .unwrap()
            .clone();
        assert_eq!(state.pending.load(Ordering::Acquire), 13);
        tracker.unwrap().on_first_token();
        assert_eq!(state.pending.load(Ordering::Acquire), 0);
        assert!(state.rates.lock().samples.is_empty());

        let index = Arc::new(KvBlockIndexer::new());
        policy.bind_worker(
            worker.as_ref(),
            &info("engine", "model"),
            Some(index.clone()),
        );
        store(&index, "engine", StorageTier::Device);
        let (_, tracker) = policy
            .select_request(&workers, None, None, &context)
            .unwrap();
        assert_eq!(state.pending.load(Ordering::Acquire), 1);
        tracker.unwrap().on_first_token();
        assert_eq!(state.pending.load(Ordering::Acquire), 0);
        assert_eq!(state.rates.lock().samples.len(), 1);

        let mut invalid = info("engine", "model");
        invalid.block_size = Some(0);
        policy.bind_worker(worker.as_ref(), &invalid, Some(index.clone()));
        assert_eq!(observe(&policy.kv, &workers, &ids, 13)[0].chars, 0);
        policy.bind_worker(
            worker.as_ref(),
            &info("engine", "model"),
            Some(index.clone()),
        );
        policy.on_worker_removed(&worker);
        policy.bind_worker(worker.as_ref(), &info("engine", "model"), Some(index));
        assert_eq!(observe(&policy.kv, &workers, &ids, 13)[0].chars, 0);
        assert!(policy.states.is_empty());
        assert!(policy.trees.is_empty());
        assert!(policy
            .select_request(&workers, None, None, &context)
            .is_none());
    }
}
