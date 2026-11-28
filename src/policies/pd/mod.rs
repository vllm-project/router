//! Selection for native PD stages and Program placement. All selectors take snapshots.
mod program;
mod types;
use super::{kv_aware_ops, LoadBalancingPolicy};
use crate::core::worker::WorkerMetadata;
use crate::core::{CircuitBreaker, ConnectionMode, Worker, WorkerResult, WorkerType};
use async_trait::async_trait;
pub(crate) use program::{recommend_program_decode, select_program_prefill};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::debug;
pub(crate) use types::{Loads, Stage};
pub use types::{
    PdEndpoint, PdLoadSample, PdPrefillInput, PdRequestContext, PdSelection, StageDecision,
};

/// Whether a decode policy can be re-run at dispatch and still produce a
/// meaningful answer.
///
/// The policy owns the answer — kv_aware reads live occupancy, KV events and
/// affinity; cache_aware re-runs from the prepared prefix selection the caller
/// carries into the commitment — so this helper only forwards the
/// one input the policy cannot see.
pub(crate) fn decode_policy_can_defer(
    policy: &dyn LoadBalancingPolicy,
    context: &PdRequestContext<'_>,
) -> bool {
    context.worker_pairs.is_none()
        && policy.can_redecide_at_dispatch(context.decode_cache.is_some())
}

/// Stage eligibility shared by admission and the deferred commitment.
///
/// Admission must not switch an endpoint between aggregate and per-rank
/// accounting while requests are live, so existing granularities are indexed
/// once instead of rescanning every load key for every candidate. Both entry
/// points filter through this type, so the two cannot drift apart.
struct Eligibility<'a, 'b> {
    active_granularities: HashSet<(Stage, &'a str, bool)>,
    context: &'a PdRequestContext<'b>,
}

impl<'a, 'b> Eligibility<'a, 'b> {
    fn new(loads: &'a Loads, context: &'a PdRequestContext<'b>) -> Self {
        let active_granularities = loads
            .keys()
            .map(|(stage, endpoint)| {
                (
                    *stage,
                    endpoint.http_url.as_str(),
                    endpoint.dp_rank.is_some(),
                )
            })
            .collect();
        Self {
            active_granularities,
            context,
        }
    }

    fn is_eligible(&self, stage: Stage, worker: &dyn Worker, endpoint: &PdEndpoint) -> bool {
        let granularity_matches = !self.active_granularities.contains(&(
            stage,
            endpoint.http_url.as_str(),
            endpoint.dp_rank.is_none(),
        ));
        worker.is_available()
            && granularity_matches
            && model_eligible(worker, self.context.model_id)
    }

    /// Eligible worker indices for `stage`, with their endpoints in pool order.
    fn pool(&self, stage: Stage, workers: &[Arc<dyn Worker>]) -> (Vec<PdEndpoint>, Vec<usize>) {
        let endpoints: Vec<_> = workers
            .iter()
            .map(|worker| PdEndpoint::for_worker(worker.as_ref()))
            .collect();
        let indices = workers
            .iter()
            .zip(&endpoints)
            .enumerate()
            .filter(|(_, (worker, endpoint))| self.is_eligible(stage, worker.as_ref(), endpoint))
            .map(|(index, _)| index)
            .collect();
        (endpoints, indices)
    }
}

/// Admission decision over the supplied load snapshot. Legacy policies
/// retain their own selection semantics and text/header inputs.
pub(crate) fn select_pd_pair(
    loads: &Loads,
    prefill_workers: &[Arc<dyn Worker>],
    decode_workers: &[Arc<dyn Worker>],
    prefill_policy: &dyn LoadBalancingPolicy,
    decode_policy: &dyn LoadBalancingPolicy,
    context: &PdRequestContext<'_>,
    bind_decode: bool,
) -> Option<PdSelection> {
    let eligibility = Eligibility::new(loads, context);
    let (prefill_endpoints, mut prefill_indices) =
        eligibility.pool(Stage::Prefill, prefill_workers);
    let (decode_endpoints, mut decode_indices) = eligibility.pool(Stage::Decode, decode_workers);
    if let Some(pairs) = context.worker_pairs {
        prefill_indices.retain(|&p| {
            decode_indices.iter().any(|&d| {
                pairs.contains(&(prefill_endpoints[p].clone(), decode_endpoints[d].clone()))
            })
        });
    }
    if prefill_indices.is_empty() || decode_indices.is_empty() {
        return None;
    }
    let prefill = select_stage(
        loads,
        Stage::Prefill,
        prefill_workers,
        &prefill_endpoints,
        &prefill_indices,
        prefill_policy,
        context,
        None,
    )?;
    if let Some(pairs) = context.worker_pairs {
        decode_indices
            .retain(|&d| pairs.contains(&(prefill.endpoint.clone(), decode_endpoints[d].clone())));
    }
    let decode = if bind_decode {
        select_stage(
            loads,
            Stage::Decode,
            decode_workers,
            &decode_endpoints,
            &decode_indices,
            decode_policy,
            context,
            None,
        )?
    } else {
        // The deferred pickup is provisional: it only seeds the request identity
        // and the Prefill KV parameters. Scoring it through the policy would
        // consume the decode tie cursor, so the commitment - which advances the
        // cursor itself - would start one tick late and break the tie rotation.
        provisional_decode(loads, &decode_endpoints, &decode_indices)?
    };

    Some(PdSelection { prefill, decode })
}

/// Deterministic stand-in for a decode decision that will be taken later.
///
/// Uses the lowest projected occupancy, first index on a tie, and consults no
/// policy state, so a pick that may be discarded cannot bias later decisions.
fn provisional_decode(
    loads: &Loads,
    endpoints: &[PdEndpoint],
    indices: &[usize],
) -> Option<StageDecision> {
    let (worker_index, endpoint) = indices
        .iter()
        .map(|&index| (index, endpoints[index].clone()))
        .min_by_key(|(_, endpoint)| {
            loads
                .get(&(Stage::Decode, endpoint.clone()))
                .copied()
                .unwrap_or_default()
                .occupancy(Stage::Decode)
        })?;
    Some(StageDecision {
        fallback_reason: None,
        credited_prefix_tokens: None,
        worker_index,
        endpoint,
        score: None,
        new_prefill_tokens: None,
        affinity_outcome: None,
        affinity: None,
    })
}

/// Re-select the decode stage against a live load snapshot.
///
/// Used by the deferred commitment: the request has already been admitted for
/// Prefill, and the decode target is chosen when the decode request is about to
/// be sent.
pub(crate) fn select_decode_stage(
    loads: &Loads,
    decode_workers: &[Arc<dyn Worker>],
    decode_policy: &dyn LoadBalancingPolicy,
    context: &PdRequestContext<'_>,
) -> Option<StageDecision> {
    let (endpoints, indices) = Eligibility::new(loads, context).pool(Stage::Decode, decode_workers);
    if indices.is_empty() {
        return None;
    }
    select_stage(
        loads,
        Stage::Decode,
        decode_workers,
        &endpoints,
        &indices,
        decode_policy,
        context,
        None,
    )
}

/// Score P only after Program admission has committed Decode.
pub(crate) fn select_program_kv_prefill(
    loads: &Loads,
    workers: &[Arc<dyn Worker>],
    policy: &dyn LoadBalancingPolicy,
    context: &PdRequestContext<'_>,
    remote_miss: &HashMap<PdEndpoint, u64>,
) -> Option<StageDecision> {
    let (endpoints, indices) = Eligibility::new(loads, context).pool(Stage::Prefill, workers);
    if indices.is_empty() {
        return None;
    }
    select_stage(
        loads,
        Stage::Prefill,
        workers,
        &endpoints,
        &indices,
        policy,
        context,
        Some(remote_miss),
    )
}

/// Whether a worker can serve this request model: the part of candidate eligibility that
/// does not depend on live load. Selection and pricing both use it, so a worker that cannot
/// be chosen cannot deny credit to the ones that can.
pub(crate) fn model_eligible(worker: &dyn Worker, model_id: Option<&str>) -> bool {
    model_id.is_none_or(|model| worker.model_id() == "unknown" || worker.model_id() == model)
}

#[allow(clippy::too_many_arguments)]
fn select_stage(
    loads: &Loads,
    stage: Stage,
    workers: &[Arc<dyn Worker>],
    endpoints: &[PdEndpoint],
    indices: &[usize],
    policy: &dyn LoadBalancingPolicy,
    context: &PdRequestContext<'_>,
    remote_miss: Option<&HashMap<PdEndpoint, u64>>,
) -> Option<StageDecision> {
    let policy_input = context.legacy_policy_input;
    let kv_aware = kv_aware_ops(policy).is_some();
    // Keep the caller slice order. Static pools pass registry registration
    // order, which is the --prefill / --decode order. Sorting by endpoint URL
    // made round-robin follow port text instead of that order.
    let selected_indices = indices.to_vec();
    // Decode ranks by the logical context it would hold, when every eligible
    // candidate can supply it: one unknown prompt length mixes units across the
    // pool, so the whole stage falls back to request counts instead.
    let decode_tokens_known = stage == Stage::Decode
        && kv_aware
        && selected_indices.iter().all(|&index| {
            loads
                .get(&(stage, endpoints[index].clone()))
                .is_none_or(|load| load.decode_tokens().is_some())
        });
    let projected: Vec<Arc<dyn Worker>> = selected_indices
        .iter()
        .map(|&index| {
            let inner = workers[index].clone();
            let endpoint = endpoints[index].clone();
            let load = loads.get(&(stage, endpoint)).copied().unwrap_or_default();
            let load = if decode_tokens_known {
                load.decode_tokens().unwrap_or(0) as usize
            } else {
                load.occupancy(stage)
            };
            Arc::new(PolicyWorker { inner, load }) as Arc<dyn Worker>
        })
        .collect();
    let unknown_backlog = stage == Stage::Prefill
        && selected_indices.iter().any(|&index| {
            loads
                .get(&(stage, endpoints[index].clone()))
                .is_some_and(|load| load.unknown_prefill_requests > 0)
        });
    if let (Stage::Prefill, Some(kv), false) = (stage, kv_aware_ops(policy), unknown_backlog) {
        let inputs: Option<Vec<_>> = projected
            .iter()
            .map(|worker| {
                let input = context.prefill_inputs?.get(worker.base_url())?;
                Some((
                    input,
                    input.prompts.as_deref()?,
                    *input.work_by_rank.get(&worker.dp_rank())?,
                ))
            })
            .collect();
        // Token costs and request counts have different units. If any
        // eligible P has unknown work, compare the whole pool by occupancy.
        if let Some(inputs) = inputs {
            let work: Vec<_> = inputs
                .iter()
                .map(|(_, _, work)| {
                    let mut estimate = *work;
                    if remote_miss.is_some() {
                        estimate.new_prefill_tokens = estimate.device_miss_tokens;
                    }
                    estimate
                })
                .collect();
            let costs: Vec<_> = selected_indices
                .iter()
                .zip(&work)
                .map(|(&index, work)| {
                    loads
                        .get(&(stage, endpoints[index].clone()))
                        .copied()
                        .unwrap_or_default()
                        .active_prefill_tokens
                        .saturating_add(
                            remote_miss
                                .and_then(|costs| costs.get(&endpoints[index]))
                                .copied()
                                .unwrap_or(u64::MAX)
                                .min(work.new_prefill_tokens),
                        ) as f64
                })
                .collect();
            let occupancy: Vec<_> = projected.iter().map(|worker| worker.load()).collect();
            debug!(version = 1, stage = "prefill", endpoints = ?selected_indices.iter()
                .map(|&index| &endpoints[index]).collect::<Vec<_>>(), ?costs, ?occupancy,
                remaining_work = ?work, "P/D token-cost candidates");
            // P follows actual token work across the full eligible pool;
            // decode session affinity must never filter or pin prefill.
            if let Some(picked) = kv.pick_prefill(&costs, &occupancy) {
                let (input, tokens, _) = inputs[picked];
                let estimate = work[picked];
                let new_tokens = estimate.new_prefill_tokens;
                let outcome = if !estimate.placement_verified {
                    "unverified_identity"
                } else {
                    "prefix_cost"
                };
                crate::metrics::RouterMetrics::record_prefix_routing_decision(
                    "kv_aware", "prefill", outcome,
                );
                return Some(StageDecision {
                    worker_index: selected_indices[picked],
                    endpoint: endpoints[selected_indices[picked]].clone(),
                    score: Some(costs[picked]),
                    new_prefill_tokens: Some(new_tokens),
                    credited_prefix_tokens: Some(
                        tokens
                            .iter()
                            .map(|ids| ids.len() as u64)
                            .sum::<u64>()
                            .saturating_sub(new_tokens),
                    ),
                    fallback_reason: input
                        .fallback_reason
                        .or((!estimate.placement_verified).then_some("unverified_placement")),
                    affinity_outcome: None,
                    affinity: None,
                });
            }
            // pick_prefill refused the candidates (non-finite or misaligned
            // costs): degrade to policy selection, never fail the request.
            debug!(
                stage = "prefill",
                "pick_prefill refused candidates; using policy selection"
            );
        }
    }
    let mut decode_prices_used = false;
    let (picked, affinity_outcome, affinity_binding) = if stage == Stage::Decode && kv_aware {
        // Decode never queries placement or applies the P temperature:
        // deterministic minimum occupancy with rotating equal-load ties, and
        // the optional decode session binding on top.
        let kv = kv_aware_ops(policy)?;
        // `select_decode` itself discards a table that misses any eligible
        // candidate; this mirror exists so `decode_prices_used` reports what the
        // selector actually acted on.
        //
        // The table is keyed by `Worker::url()`, not `base_url()`: DP ranks of
        // one engine have independent KV caches, so rank `@0` must not consume
        // rank `@1`'s price.
        let decode_prices = if !decode_tokens_known {
            None
        } else {
            context.decode_context.filter(|prices| {
                projected
                    .iter()
                    .all(|worker| prices.contains_key(worker.url()))
            })
        };
        decode_prices_used = decode_prices.is_some();
        let pick = kv.select_decode(
            &projected,
            context.affinity_key.as_ref(),
            context.lora_name,
            context.cache_salt,
            decode_prices,
        )?;
        (
            pick.index,
            context.affinity_key.is_some().then_some(pick.outcome),
            pick.binding,
        )
    } else if stage == Stage::Prefill && kv_aware {
        // Unknown token work or refused costs use projected occupancy,
        // without session binding or a request-count affinity limit on P.
        // PD admission owns the candidate lifecycle and load snapshot here;
        // the colocated selector's registration filter can otherwise discard
        // fresh discovery workers before their background KV binding exists.
        let picked = projected
            .iter()
            .enumerate()
            .min_by_key(|(_, worker)| worker.load())
            .map(|(index, _)| index)?;
        (picked, None, None)
    } else if let Some(cache) = policy
        .as_any()
        .downcast_ref::<crate::policies::CacheAwarePolicy>()
    {
        let prepared = if stage == Stage::Prefill {
            context.prefill_cache
        } else {
            context.decode_cache
        };
        let picked = cache.select_prepared(
            &projected,
            prepared,
            if stage == Stage::Prefill {
                "prefill"
            } else {
                "decode"
            },
        )?;
        (picked, None, None)
    } else {
        let picked =
            policy.select_worker_with_headers(&projected, policy_input, context.headers)?;
        (picked, None, None)
    };
    if kv_aware {
        let phase = if stage == Stage::Prefill {
            "prefill"
        } else {
            "decode"
        };
        // Recorded once, here, for every KV-aware decision: what a Decode
        // decision ranked by is part of the answer, not a debugging detail.
        let outcome = if affinity_outcome == Some("sticky_hit") {
            "sticky_hit"
        } else if stage == Stage::Prefill {
            "load_fallback"
        } else {
            // Report what the selector actually used, not whether a table merely
            // existed: a table that failed the completeness filter above fell back
            // to occupancy even though `context.decode_context` was `Some`.
            match (decode_tokens_known, decode_prices_used) {
                (true, true) => "context_tokens_marginal",
                (true, false) => "context_tokens",
                (false, _) => "occupancy_unknown_tokens",
            }
        };
        crate::metrics::RouterMetrics::record_prefix_routing_decision("kv_aware", phase, outcome);
    }
    // A held binding reference must never leak on a late selection failure:
    // release it before any bail-out below.
    let Some((&worker_index, worker)) = selected_indices.get(picked).zip(projected.get(picked))
    else {
        if let Some(record) = &affinity_binding {
            record.release_ref();
        }
        return None;
    };
    let selected_tokens = (stage == Stage::Prefill)
        .then(|| {
            context
                .prefill_inputs
                .and_then(|inputs| inputs.get(worker.base_url()))
                .and_then(|input| input.prompts.as_ref())
                .map(|prompts| prompts.iter().map(|ids| ids.len() as u64).sum())
        })
        .flatten();
    Some(StageDecision {
        fallback_reason: if stage == Stage::Prefill && kv_aware {
            Some(if unknown_backlog {
                "unknown_prefill_backlog"
            } else {
                projected
                    .iter()
                    .find_map(|worker| {
                        context
                            .prefill_inputs
                            .and_then(|inputs| inputs.get(worker.base_url()))
                            .and_then(|input| input.fallback_reason)
                    })
                    .unwrap_or("unverified_effective_tokens")
            })
        } else {
            None
        },
        credited_prefix_tokens: selected_tokens.map(|_| 0),
        worker_index,
        endpoint: endpoints[worker_index].clone(),
        // Unknown P token work falls back to occupancy; Decode records the
        // occupancy plus the same marginal context the selector compared.
        score: kv_aware.then_some(match stage {
            Stage::Prefill => worker.load() as f64,
            Stage::Decode => {
                let additional = if decode_prices_used {
                    context
                        .decode_context
                        .and_then(|prices| prices.get(worker.url()))
                        .copied()
                        .unwrap_or(0)
                } else {
                    0
                };
                (worker.load() as u64).saturating_add(additional) as f64
            }
        }),
        new_prefill_tokens: selected_tokens,
        affinity_outcome,
        affinity: affinity_binding,
    })
}

/// Selection-only view: load is an immutable sample from the admission lock.
/// Worker identity, availability, metadata and policy inputs are preserved.
#[derive(Debug)]
struct PolicyWorker {
    inner: Arc<dyn Worker>,
    load: usize,
}

#[async_trait]
impl Worker for PolicyWorker {
    fn url(&self) -> &str {
        self.inner.url()
    }
    fn worker_type(&self) -> WorkerType {
        self.inner.worker_type()
    }
    fn connection_mode(&self) -> ConnectionMode {
        self.inner.connection_mode()
    }
    fn is_healthy(&self) -> bool {
        self.inner.is_healthy()
    }
    fn set_healthy(&self, healthy: bool) {
        self.inner.set_healthy(healthy);
    }
    async fn check_health_async(&self) -> WorkerResult<()> {
        self.inner.check_health_async().await
    }
    fn load(&self) -> usize {
        self.load
    }
    fn increment_load(&self) {
        self.inner.increment_load();
    }
    fn decrement_load(&self) {
        self.inner.decrement_load();
    }
    fn processed_requests(&self) -> usize {
        self.inner.processed_requests()
    }
    fn increment_processed(&self) {
        self.inner.increment_processed();
    }
    fn metadata(&self) -> &WorkerMetadata {
        self.inner.metadata()
    }
    fn circuit_breaker(&self) -> &CircuitBreaker {
        self.inner.circuit_breaker()
    }
    fn base_url(&self) -> &str {
        self.inner.base_url()
    }
    fn dp_rank(&self) -> Option<usize> {
        self.inner.dp_rank()
    }
    fn dp_size(&self) -> Option<usize> {
        self.inner.dp_size()
    }
    fn is_dp_aware(&self) -> bool {
        self.inner.is_dp_aware()
    }
}
