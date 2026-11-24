use super::SMetricStrategy;
use crate::core::Worker;
use crate::kv_index::{CacheKey, Locality, MatchQuery, RoutableTarget, StorageTier, WorkerKvInfo};
use crate::policies::{
    CandidateView, CostComponents, CostDecision, KvSelectionReceipt, RequestTracker, RoutingContext,
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub(in crate::policies) enum KvBinding {
    /// Worker is not bound to a KV-event index (may fall back to HTTP text tree).
    Unmanaged,
    /// Worker is KV-event managed, but placement is disabled (salted) or source is suspended.
    Unavailable,
    /// Worker is bound to a live KV-event index with the given block size.
    Ready(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PrefixMatch {
    Unmanaged,
    Unavailable,
    Bound(usize),
}

impl PrefixMatch {
    fn tokens(self) -> usize {
        match self {
            Self::Bound(tokens) => tokens,
            Self::Unmanaged | Self::Unavailable => 0,
        }
    }
}

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
        bindings: impl Iterator<Item = KvBinding>,
        context: Option<&RoutingContext<'_>>,
    ) -> Option<(KvSelectionReceipt, Option<RequestTracker>)> {
        let empty_context = RoutingContext::default();
        let context = context.unwrap_or(&empty_context);
        let hits = context.kv.as_ref().map(|_| {
            candidates
                .iter()
                .zip(bindings)
                .map(|(candidate, binding)| prefix_match(candidate, binding))
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
            .saturating_sub(hits.as_ref().map_or(0, |hits| hits[picked].tokens()))
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

fn prefix_match(candidate: &CandidateView<'_>, binding: KvBinding) -> PrefixMatch {
    let block_size = match binding {
        KvBinding::Unmanaged => return PrefixMatch::Unmanaged,
        KvBinding::Unavailable => return PrefixMatch::Unavailable,
        KvBinding::Ready(b) => b as usize,
    };
    let n = candidate.input_tokens;
    if n == 0 || block_size == 0 {
        return PrefixMatch::Bound(0);
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
    PrefixMatch::Bound(blocks.min(n.saturating_sub(1) / block_size) * block_size)
}
