//! Real-event prefix affinity, with fair least-load fallback on a cold miss.

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::config::KvAwareConfig;
use crate::core::Worker;
use crate::kv_index::{BlockKeyGenerator, KVBlockIndex};
use crate::metrics::RouterMetrics;
use sha2::{Digest, Sha256};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Debug)]
pub struct KvAwarePolicy {
    index: Arc<KVBlockIndex>,
    generator: BlockKeyGenerator,
    cursor: AtomicUsize,
}

impl KvAwarePolicy {
    pub fn new(config: &KvAwareConfig) -> Self {
        Self {
            index: Arc::new(KVBlockIndex::new(config.index_max_entries)),
            generator: BlockKeyGenerator::new(config.block_size, u64::from(config.hash_seed)),
            cursor: AtomicUsize::new(0),
        }
    }

    pub fn index(&self) -> Arc<KVBlockIndex> {
        self.index.clone()
    }
}

impl LoadBalancingPolicy for KvAwarePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        self.select_worker_with_tokens(workers, None, None, headers)
    }

    fn select_worker_with_tokens(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        token_ids: Option<&[u32]>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let keys = token_ids
            .map(|ids| self.generator.generate_block_keys(ids))
            .unwrap_or_default();
        let candidates: Vec<_> = get_healthy_worker_indices(workers)
            .into_iter()
            .map(|i| {
                (
                    i,
                    self.index.prefix_score(workers[i].url(), &keys),
                    workers[i].load(),
                )
            })
            .collect();
        let best_score = candidates.iter().map(|(_, score, _)| *score).max()?;
        let least_load = candidates
            .iter()
            .filter(|(_, score, _)| *score == best_score)
            .map(|(_, _, load)| *load)
            .min()?;
        let mut tied: Vec<_> = candidates
            .iter()
            .filter(|(_, score, load)| *score == best_score && *load == least_load)
            .map(|(i, _, _)| *i)
            .collect();
        tied.sort_unstable_by(|a, b| workers[*a].url().cmp(workers[*b].url()));
        // A unique cache hit must not consume a fallback turn: alternating
        // hot and cold requests would otherwise always send cold requests to
        // the same worker in a two-worker pool.
        let selected = if tied.len() == 1 {
            tied[0]
        } else {
            tied[self.cursor.fetch_add(1, Ordering::Relaxed) % tied.len()]
        };
        if tracing::enabled!(tracing::Level::DEBUG) {
            let scores: Vec<_> = candidates.iter().map(|(i, score, _)|
                serde_json::json!({"worker": workers[*i].url(), "prefix_blocks": score})).collect();
            let token_ids_sha256 = token_ids.map(|ids| {
                let mut digest = Sha256::new();
                for id in ids {
                    digest.update(id.to_be_bytes());
                }
                format!("{:x}", digest.finalize())
            });
            let decision = serde_json::json!({"worker": workers[selected].url(),
                "prefix_blocks": best_score, "complete_blocks": keys.len(), "scores": scores,
                "token_ids_sha256": token_ids_sha256});
            tracing::debug!(decision = %decision, "kv_route_decision");
        }
        workers[selected].increment_processed();
        RouterMetrics::record_processed_request(workers[selected].url());
        RouterMetrics::record_policy_decision(self.name(), workers[selected].url());
        Some(selected)
    }

    fn name(&self) -> &'static str {
        "kv_aware"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn real_ownership_wins_and_cold_misses_rotate_without_learning() {
        let policy = KvAwarePolicy::new(&KvAwareConfig::default());
        let workers: Vec<Arc<dyn Worker>> = ["http://w0:8000", "http://w1:8000"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        let ids: Vec<_> = (0..32).collect();
        let first = policy
            .select_worker_with_tokens(&workers, None, Some(&ids), None)
            .unwrap();
        let second = policy
            .select_worker_with_tokens(&workers, None, Some(&ids), None)
            .unwrap();
        assert_ne!(first, second);
        assert_eq!(policy.index.ownership_count(), 0);
        let generation = policy.index.begin_worker(workers[1].url());
        policy.index.store(
            workers[1].url(),
            generation,
            &policy.generator.generate_block_keys(&ids),
        );
        workers[1].increment_load();
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(1)
        );
        // A legacy string/JSON never becomes guessed prompt tokens.
        assert_eq!(policy.select_worker(&workers, Some("[0,1,2]")), Some(0));
        workers[1].set_healthy(false);
        assert_eq!(
            policy.select_worker_with_tokens(&workers, None, Some(&ids), None),
            Some(0)
        );
    }

    #[test]
    fn unique_hits_do_not_consume_cold_fallback_turns() {
        let policy = KvAwarePolicy::new(&KvAwareConfig::default());
        let workers: Vec<Arc<dyn Worker>> = ["http://w0:8000", "http://w1:8000"]
            .into_iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.into(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect();
        let hot = vec![1; 32];
        let cold = vec![2; 32];
        let generation = policy.index.begin_worker(workers[1].url());
        policy.index.store(
            workers[1].url(),
            generation,
            &policy.generator.generate_block_keys(&hot),
        );
        let mut cold_choices = Vec::new();
        for _ in 0..4 {
            assert_eq!(
                policy.select_worker_with_tokens(&workers, None, Some(&hot), None),
                Some(1)
            );
            cold_choices.push(
                policy
                    .select_worker_with_tokens(&workers, None, Some(&cold), None)
                    .unwrap(),
            );
        }
        assert_eq!(cold_choices, [0, 1, 0, 1]);
    }
}
