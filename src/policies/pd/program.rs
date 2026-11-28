//! Placement decisions for capacity-first Program sessions on disaggregated workers.
use super::{select_program_kv_prefill, Loads, PdEndpoint, PdRequestContext};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::policies::binding::ProgramBindingCandidate;
use crate::policies::{kv_aware_ops, KvAwareOps, LoadBalancingPolicy};
use std::{collections::HashMap, future::Future, sync::Arc};

/// Price only the live, admitted candidates and only when there is a choice.
pub(crate) async fn recommend_program_decode(
    eligible: &[(&ProgramBindingCandidate, Arc<dyn Worker>)],
    kv: Option<&dyn KvAwareOps>,
    prices: impl Future<Output = Option<HashMap<String, u64>>>,
) -> Option<String> {
    match eligible {
        [] => return None,
        [(only, _)] => {
            RouterMetrics::record_prefix_routing_decision(
                "kv_aware",
                "program_decode",
                "single_candidate",
            );
            return Some(only.target_id.clone());
        }
        _ => {}
    }
    // An unpriced rank makes the entire comparison fall back to counts.
    let marginal: Option<Vec<_>> = prices.await.and_then(|prices| {
        eligible
            .iter()
            .map(|(_, worker)| prices.get(worker.url()).copied())
            .collect()
    });
    let (loads, marginal, outcome): (Vec<usize>, _, _) = if let Some(marginal) = marginal {
        let projected = eligible
            .iter()
            .zip(&marginal)
            .map(|((rank, _), new)| (rank.accounted_tokens as usize).saturating_add(*new as usize))
            .collect();
        (projected, marginal, "prefix_cost")
    } else {
        let loads = eligible
            .iter()
            .map(|(rank, worker)| worker.load().max(rank.accounted_programs))
            .collect();
        (loads, vec![0; eligible.len()], "load_fallback")
    };
    let picked = kv
        .and_then(|kv| kv.pick_decode(&loads, &marginal))
        .or_else(|| (0..loads.len()).min_by_key(|&i| loads[i]));
    RouterMetrics::record_prefix_routing_decision("kv_aware", "program_decode", outcome);
    picked.map(|index| eligible[index].0.target_id.clone())
}

/// A KV-aware P consumes stage load and reuse estimates. Other policies keep
/// their regular worker counters, headers and request text.
pub(crate) fn select_program_prefill(
    load_snapshot: impl FnOnce() -> Loads,
    workers: &[Arc<dyn Worker>],
    policy: &dyn LoadBalancingPolicy,
    context: &PdRequestContext<'_>,
    remote_miss: &HashMap<PdEndpoint, u64>,
) -> Option<usize> {
    if workers.is_empty() {
        return None;
    }
    if kv_aware_ops(policy).is_some() {
        return select_program_kv_prefill(&load_snapshot(), workers, policy, context, remote_miss)
            .map(|decision| decision.worker_index);
    }
    policy
        .select_worker_with_headers(workers, context.legacy_policy_input, context.headers)
        .filter(|index| *index < workers.len())
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use crate::policies::KvAwarePolicy;

    fn rank(url: &str, programs: usize) -> ProgramBindingCandidate {
        ProgramBindingCandidate {
            target_id: url.to_owned(),
            accounted_programs: programs,
            accounted_tokens: 100.0,
            capacity_tokens: None,
            kv_pressure: 0.0,
            immediately_admissible: true,
        }
    }

    #[tokio::test]
    async fn missing_one_rank_price_switches_every_rank_to_counts() {
        let warm = rank("http://warm", 3);
        let idle = rank("http://idle", 1);
        let eligible = [&warm, &idle].map(|rank| {
            let worker: Arc<dyn Worker> =
                Arc::new(BasicWorker::new(rank.target_id.clone(), WorkerType::Decode));
            (rank, worker)
        });
        let kv = KvAwarePolicy::default();
        let mut prices = HashMap::from([(warm.target_id.clone(), 0), (idle.target_id.clone(), 64)]);
        assert_eq!(
            recommend_program_decode(&eligible, Some(&kv), async { Some(prices.clone()) })
                .await
                .as_deref(),
            Some(warm.target_id.as_str())
        );
        prices.remove(&idle.target_id);
        assert_eq!(
            recommend_program_decode(&eligible, Some(&kv), async { Some(prices) })
                .await
                .as_deref(),
            Some(idle.target_id.as_str())
        );
    }

    #[tokio::test]
    async fn sole_eligible_rank_does_not_request_prices() {
        let only = rank("http://only", 1);
        let worker = Arc::new(BasicWorker::new(only.target_id.clone(), WorkerType::Decode));
        let selected = recommend_program_decode(&[(&only, worker)], None, async {
            panic!("one candidate must skip pricing")
        })
        .await;
        assert_eq!(selected.as_deref(), Some(only.target_id.as_str()));
    }
}
