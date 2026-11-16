use super::SMetricStrategy;
use crate::core::Worker;
use crate::kv_index::{CacheKey, Locality, MatchQuery, RoutableTarget, StorageTier, WorkerKvInfo};
use crate::policies::{
    CandidateView, CostComponents, CostDecision, KvSelectionReceipt, RequestTracker, RoutingContext,
};
use std::sync::Arc;

impl SMetricStrategy {
    pub(in crate::policies) fn match_query(
        token_ids: Option<&[u32]>,
        block_size: u32,
        lora_name: Option<&str>,
    ) -> MatchQuery {
        let local_hashes = token_ids
            .map(|ids| {
                let b = block_size as usize;
                let complete = ids.len().checked_div(b).unwrap_or(0) * b;
                crate::kv_index::local_block_hash_strings(&ids[..complete], block_size, lora_name)
            })
            .unwrap_or_default();
        MatchQuery {
            group_idx: 0,
            local_hashes,
            tiers_of_interest: vec![StorageTier::Device],
        }
    }

    pub(in crate::policies) fn binding_target(
        worker: &dyn Worker,
        info: &WorkerKvInfo,
    ) -> Option<(RoutableTarget, CacheKey)> {
        let rank = crate::backend::parse_dp_rank(worker.url())
            .or_else(|| worker.dp_rank().and_then(|rank| u32::try_from(rank).ok()))
            .or(info.default_rank)?;
        let key = info.cache_key.as_ref()?;
        if key.block_size == 0
            || info.block_size != Some(key.block_size)
            || !info.ranks.contains(&rank)
            || info.instance_id.is_empty()
        {
            return None;
        }
        Some((
            RoutableTarget {
                instance_id: Arc::from(info.instance_id.as_str()),
                dp_rank: rank,
            },
            key.clone(),
        ))
    }

    pub(in crate::policies) fn select_candidates(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy: &[usize],
        candidates: &[CandidateView<'_>],
        binding_meta: impl Iterator<Item = (u32, bool)>,
        context: Option<&RoutingContext<'_>>,
    ) -> Option<(KvSelectionReceipt, Option<RequestTracker>)> {
        let empty_context = RoutingContext::default();
        let context = context.unwrap_or(&empty_context);
        let chars = context
            .prompt
            .map_or(0, |prompt| prompt.text.chars().count());
        let hits = context.kv.as_ref().map(|_| {
            candidates
                .iter()
                .zip(binding_meta)
                .map(|(candidate, (block_size, index_bound))| {
                    prefix_match(candidate, block_size, index_bound, chars)
                })
                .collect::<Vec<_>>()
        });
        let selected_workers: Vec<_> = healthy.iter().map(|&i| workers[i].clone()).collect();
        let loads: Vec<_> = candidates
            .iter()
            .map(|candidate| candidate.load.active_requests)
            .collect();
        let (picked, tracker, score) =
            self.select_request(&selected_workers, context, hits.as_deref(), Some(&loads))?;
        let candidate = &candidates[picked];
        let new_prefill_tokens = candidate
            .input_tokens
            .saturating_sub(hits.as_ref().map_or(0, |hits| hits[picked].tokens))
            as u64;
        Some((
            KvSelectionReceipt {
                worker_index: healthy[picked],
                new_prefill_tokens,
                decision: CostDecision {
                    target: candidate.target.clone(),
                    cost: score as f64,
                    components: CostComponents {
                        ptoken: new_prefill_tokens,
                        batchsize: loads[picked],
                    },
                },
            },
            tracker,
        ))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PrefixMatch {
    pub(super) chars: usize,
    pub(super) tokens: usize,
    pub(super) observed: bool,
    pub(super) index_bound: bool,
}

fn prefix_match(
    candidate: &CandidateView<'_>,
    block_size: u32,
    index_bound: bool,
    chars: usize,
) -> PrefixMatch {
    let n = candidate.input_tokens;
    let b = block_size as usize;
    if n == 0 || b == 0 {
        return PrefixMatch::default();
    }
    let blocks = candidate
        .matches
        .iter()
        .filter(|hit| {
            hit.target == *candidate.target
                && hit.tier == StorageTier::Device
                && hit.locality == Locality::Local
        })
        .flat_map(|hit| hit.segments.iter())
        .filter(|(start, _)| *start == 0)
        .map(|(_, end)| *end as usize)
        .max()
        .unwrap_or(0);
    let tokens = blocks.min(n.saturating_sub(1) / b) * b;
    PrefixMatch {
        chars: ((chars as u128 * tokens as u128) / n as u128) as usize,
        tokens,
        observed: blocks > 0,
        index_bound,
    }
}
