//! ProgramScheduler orchestration and backend observation epochs.
//!
//! Rank-local and global decision algorithms live in sibling modules. This
//! file owns atomic state access, target discovery, and observation rebasing.

use super::scheduler_state::{ProgramDecisionState, ProgramSchedulerState, RankObservationState};
use super::{
    BackendObservation, ProgramBindingCandidate, ProgramIdentity, ProgramRef,
    ProgramSchedulerConfig, ProgramState, ProgramStatus, ProgramTarget, ProgressTtlPolicyMath,
};
use crate::metrics::RouterMetrics;
use parking_lot::Mutex;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::info;

/// Router-owned Program scheduler.
#[derive(Debug)]
pub struct ProgramScheduler {
    pub(crate) config: ProgramSchedulerConfig,
    pub(crate) policy: ProgressTtlPolicyMath,
    pub(crate) state: Mutex<ProgramSchedulerState>,
}

impl ProgramScheduler {
    /// Construct an opt-in scheduler without changing native Router policy.
    pub fn new(config: ProgramSchedulerConfig) -> Self {
        let policy = ProgressTtlPolicyMath::new(config.progress_ttl.clone());
        let state = ProgramSchedulerState::new(&config);
        Self {
            config,
            policy,
            state: Mutex::new(state),
        }
    }

    /// Backend observation and periodic scheduling interval.
    pub fn metrics_interval(&self) -> Duration {
        self.config.metrics_interval
    }

    /// Metadata source that opts one request into Program scheduling.
    pub fn enable_key(&self) -> super::ProgramSchedulingEnableKey {
        self.config.enable_key
    }

    /// Whether initial Program binding needs the request text for prefix affinity.
    pub fn uses_cache_aware_binding(&self) -> bool {
        self.config.binding_strategy == super::ProgramBindingStrategy::CacheAware
    }

    /// Synchronize one model pool's concrete backend and internal-DP targets.
    pub fn sync_targets(&self, model_pool: &str, targets: &[ProgramTarget]) {
        self.sync_target_snapshot(model_pool, Arc::from(targets));
    }

    /// Synchronize a cached immutable target snapshot.
    pub fn sync_target_snapshot(&self, model_pool: &str, targets: Arc<[ProgramTarget]>) {
        self.sync_target_snapshots([(model_pool.to_string(), targets)]);
    }

    /// Atomically synchronize cached target snapshots for multiple model pools.
    pub fn sync_target_snapshots(
        &self,
        snapshots: impl IntoIterator<Item = (String, Arc<[ProgramTarget]>)>,
    ) {
        let mut state = self.state.lock();
        for (model_pool, targets) in snapshots {
            self.sync_target_snapshot_locked(&mut state, &model_pool, targets);
        }
    }

    fn sync_target_snapshot_locked(
        &self,
        state: &mut ProgramSchedulerState,
        model_pool: &str,
        targets: Arc<[ProgramTarget]>,
    ) {
        if state
            .model_target_snapshots
            .get(model_pool)
            .is_some_and(|current| Arc::ptr_eq(current, &targets))
        {
            return;
        }
        let incoming = targets
            .iter()
            .cloned()
            .map(|target| (target.id.clone(), target))
            .collect::<BTreeMap<_, _>>();
        let current_ids = incoming.keys().cloned().collect::<BTreeSet<_>>();
        let unchanged = state
            .model_targets
            .get(model_pool)
            .is_some_and(|ids| ids == &current_ids)
            && incoming
                .iter()
                .all(|(target_id, target)| state.targets.get(target_id) == Some(target));
        if unchanged {
            state
                .model_target_snapshots
                .insert(model_pool.to_string(), targets);
            return;
        }
        state.target_snapshot_revision = state
            .target_snapshot_revision
            .checked_add(1)
            .expect("Program target snapshot revision exhausted");
        state
            .model_target_snapshots
            .insert(model_pool.to_string(), targets);
        let previous_ids = state
            .model_targets
            .insert(model_pool.to_string(), current_ids.clone())
            .unwrap_or_default();
        state.targets.extend(incoming);
        state
            .global_queues
            .entry(model_pool.to_string())
            .or_default();
        for target_id in &current_ids {
            state.rank_queues.entry(target_id.clone()).or_default();
            state.rank_factors.entry(target_id.clone()).or_default();
            state.observations.entry(target_id.clone()).or_default();
        }

        let removed = previous_ids
            .difference(&current_ids)
            .cloned()
            .collect::<Vec<_>>();
        for target_id in &removed {
            state.bindings.remove_target(target_id);
            state.bindings.remove_affinity_target(model_pool, target_id);
            state.rank_queues.remove(target_id);
            state.rank_factors.remove(target_id);
            state.observations.remove(target_id);
        }
        let affected = state
            .runtime
            .views()
            .into_iter()
            .filter(|program| program.reference.model_pool() == model_pool)
            .filter_map(|program| {
                let decision = state.decisions.get(&program.reference)?;
                let needs_rebinding = decision
                    .last_target
                    .as_ref()
                    .is_none_or(|target| !current_ids.contains(target));
                needs_rebinding.then(|| {
                    (
                        program.reference,
                        program.expected_resume,
                        program.waiting_requests,
                        decision
                            .last_target
                            .as_ref()
                            .is_some_and(|target| removed.contains(target)),
                        decision.placement_key.clone(),
                        decision.placement_hash_key.clone(),
                    )
                })
            })
            .collect::<Vec<_>>();
        for (
            program,
            expected_resume,
            waiting_requests,
            target_disappeared,
            placement_key,
            placement_hash_key,
        ) in affected
        {
            let placement_invalidated = state.runtime.invalidate_placement(&program, &removed);
            if target_disappeared {
                if let Some(target) = state
                    .decisions
                    .get(&program)
                    .and_then(|decision| decision.last_target.as_deref())
                {
                    RouterMetrics::record_agent_aware_transition(target, "health_failover");
                }
            }
            state.bindings.release_program(&program);
            for queue in state.rank_queues.values_mut() {
                queue.retain(|queued| queued != &program);
            }
            for queue in state.global_queues.values_mut() {
                queue.retain(|queued| queued != &program);
            }
            let identity = ProgramIdentity::for_rebinding(
                &program,
                placement_hash_key,
                placement_key,
                expected_resume,
            );
            let candidates = self.binding_candidates(&state, &identity, Instant::now());
            let replacement = state.bindings.bind(&identity, &candidates, None);
            if let Some(decision) = state.decisions.get_mut(&program) {
                decision.home_target = replacement.clone();
                decision.last_target = replacement.clone();
                if placement_invalidated || target_disappeared {
                    let now = Instant::now();
                    decision.paused_at = Some(now);
                    decision.pause_when_idle = false;
                    decision.ttl_deadline = None;
                    decision.segment_started_at = None;
                    Self::restart_shared_prefix_freshness(decision, now);
                }
            }
            if waiting_requests > 0 {
                if self.config.global_queue {
                    state
                        .global_queues
                        .entry(model_pool.to_string())
                        .or_default()
                        .push_back(program);
                } else if let Some(target) = replacement {
                    state
                        .rank_queues
                        .entry(target)
                        .or_default()
                        .push_back(program);
                }
            }
        }
        let referenced = state
            .model_targets
            .values()
            .flat_map(|target_ids| target_ids.iter().cloned())
            .collect::<BTreeSet<_>>();
        state
            .targets
            .retain(|target_id, _| referenced.contains(target_id));
    }

    /// Current target snapshot for one model pool.
    pub fn targets(&self, model_pool: &str) -> Vec<ProgramTarget> {
        let state = self.state.lock();
        state
            .model_targets
            .get(model_pool)
            .into_iter()
            .flatten()
            .filter_map(|target_id| state.targets.get(target_id).cloned())
            .collect()
    }

    /// Snapshot every concrete target across model pools without duplicates.
    pub fn all_targets(&self) -> Vec<ProgramTarget> {
        let state = self.state.lock();
        state.targets.values().cloned().collect()
    }

    /// Model pools whose target snapshots must be refreshed by observation.
    pub fn model_pools(&self) -> Vec<String> {
        let state = self.state.lock();
        let mut pools = state.model_targets.keys().cloned().collect::<Vec<_>>();
        pools.sort();
        pools
    }

    /// Capture the Router ledger before a non-blocking metrics scrape begins.
    pub fn begin_observation(&self, targets: &[ProgramTarget]) -> BackendObservationEpoch {
        let state = self.state.lock();
        let checkpoints = targets
            .iter()
            .map(|target| {
                let views = state.runtime.views();
                let active_reasoning = views
                    .iter()
                    .filter(|program| {
                        program.state == ProgramState::Active
                            && program.status == ProgramStatus::Reasoning
                            && program.placement.as_deref() == Some(target.id.as_str())
                    })
                    .collect::<Vec<_>>();
                let active_acting_private_tokens = views
                    .iter()
                    .filter(|program| {
                        program.state == ProgramState::Active
                            && program.status == ProgramStatus::Acting
                            && program.placement.as_deref() == Some(target.id.as_str())
                    })
                    .filter_map(|program| state.decisions.get(&program.reference))
                    .map(|decision| {
                        decision.private_tokens(self.config.progress_ttl.decode_buffer_tokens)
                    })
                    .sum::<f64>();
                let active_reasoning_private_tokens = active_reasoning
                    .iter()
                    .filter_map(|program| state.decisions.get(&program.reference))
                    .map(|decision| {
                        decision.private_tokens(self.config.progress_ttl.decode_buffer_tokens)
                    })
                    .sum::<f64>();
                let active_reasoning_requests = active_reasoning
                    .iter()
                    .map(|program| program.in_flight_requests)
                    .sum::<usize>();
                let ledger_checkpoint = state
                    .observations
                    .get(&target.id)
                    .map_or(0.0, |observation| observation.active_program_token_delta);
                (
                    target.id.clone(),
                    ObservationCheckpoint {
                        ledger_checkpoint,
                        active_reasoning_programs: active_reasoning.len(),
                        active_reasoning_requests,
                        active_reasoning_private_tokens,
                        active_acting_private_tokens,
                    },
                )
            })
            .collect();
        BackendObservationEpoch {
            target_snapshot_revision: state.target_snapshot_revision,
            checkpoints,
        }
    }

    /// Install raw observations and preserve transitions committed during I/O.
    pub fn apply_observations(
        &self,
        epoch: BackendObservationEpoch,
        observations: impl IntoIterator<Item = BackendObservation>,
    ) {
        let mut state = self.state.lock();
        if epoch.target_snapshot_revision != state.target_snapshot_revision {
            info!(
                event = "stale_capacity_observation_dropped",
                epoch_revision = epoch.target_snapshot_revision,
                current_revision = state.target_snapshot_revision,
                "Dropping backend observations collected against a stale target snapshot"
            );
            return;
        }
        for raw in observations {
            let target_id = raw.target_id.clone();
            let Some(checkpoint) = epoch.checkpoints.get(&raw.target_id) else {
                continue;
            };
            if !state.targets.contains_key(&target_id) {
                continue;
            }
            let native_used_tokens = raw
                .kv_cache_usage
                .zip(self.config.progress_ttl.token_capacity)
                .map(|(usage, capacity)| usage * capacity as f64);
            let estimated_reasoning_tokens = native_used_tokens.map(|native_used| {
                let running = raw.running_requests.unwrap_or(0);
                if running > 0 {
                    native_used * checkpoint.active_reasoning_programs.max(running) as f64
                        / running as f64
                } else if checkpoint.active_reasoning_programs > 0 {
                    checkpoint.active_reasoning_private_tokens
                } else {
                    native_used
                }
            });
            let estimated_active_program_tokens = estimated_reasoning_tokens
                .map(|reasoning| reasoning + checkpoint.active_acting_private_tokens);
            let observation = state.observations.entry(target_id.clone()).or_default();
            observation.active_program_token_delta -= checkpoint.ledger_checkpoint;
            observation.kv_cache_usage = raw.kv_cache_usage;
            observation.running_requests = raw.running_requests;
            observation.waiting_requests = raw.waiting_requests;
            observation.router_active_reasoning_programs = checkpoint.active_reasoning_programs;
            observation.router_active_reasoning_requests = checkpoint.active_reasoning_requests;
            observation.estimated_active_reasoning_tokens = estimated_reasoning_tokens;
            observation.estimated_active_program_tokens = estimated_active_program_tokens;
            observation.observed_at = Some(raw.observed_at);
            let active_program_token_delta = observation.active_program_token_delta;
            let estimated_total_tokens = self.target_usage(&state, &target_id, raw.observed_at);
            info!(
                event = "capacity_observation",
                target = %target_id,
                kv_cache_usage = ?raw.kv_cache_usage,
                native_used_tokens = ?native_used_tokens,
                vllm_running_requests = ?raw.running_requests,
                vllm_waiting_requests = ?raw.waiting_requests,
                router_active_reasoning_programs = checkpoint.active_reasoning_programs,
                router_active_reasoning_requests = checkpoint.active_reasoning_requests,
                estimated_reasoning_tokens = ?estimated_reasoning_tokens,
                active_acting_private_tokens = checkpoint.active_acting_private_tokens,
                observed_active_program_tokens = ?estimated_active_program_tokens,
                active_program_token_delta,
                estimated_total_tokens,
                capacity_tokens = ?self.config.progress_ttl.token_capacity,
                "Program scheduling diagnostic"
            );
        }
    }

    pub(crate) fn binding_candidates(
        &self,
        state: &ProgramSchedulerState,
        identity: &ProgramIdentity,
        now: Instant,
    ) -> Vec<ProgramBindingCandidate> {
        let runtime_views = state.runtime.views();
        state
            .model_targets
            .get(identity.model_pool())
            .into_iter()
            .flatten()
            .map(|target_id| {
                let accounted = runtime_views
                    .iter()
                    .cloned()
                    .into_iter()
                    .filter(|program| {
                        let last_target = state
                            .decisions
                            .get(&program.reference)
                            .and_then(|decision| decision.last_target.as_deref());
                        last_target == Some(target_id.as_str())
                            && (program.state == ProgramState::Active
                                || program.status == ProgramStatus::Reasoning)
                    })
                    .collect::<Vec<_>>();
                let accounted_tokens = accounted
                    .iter()
                    .filter_map(|program| state.decisions.get(&program.reference))
                    .map(|decision| {
                        decision.private_tokens(self.config.progress_ttl.decode_buffer_tokens)
                    })
                    .sum();
                let pending_new_program_tokens = runtime_views
                    .iter()
                    .filter(|program| {
                        program.state == ProgramState::Paused
                            && program.status == ProgramStatus::Reasoning
                            && program.placement.is_none()
                            && program.waiting_requests > 0
                    })
                    .filter_map(|program| state.decisions.get(&program.reference))
                    .filter(|decision| {
                        decision.completed_requests == 0
                            && decision.last_target.as_deref() == Some(target_id.as_str())
                    })
                    .map(|decision| {
                        decision.private_tokens(self.config.progress_ttl.decode_buffer_tokens)
                    })
                    .sum::<f64>();
                let active_programs = runtime_views
                    .iter()
                    .filter(|program| {
                        program.state == ProgramState::Active
                            && program.placement.as_deref() == Some(target_id.as_str())
                    })
                    .count();
                let target_usage = self.target_usage(state, target_id, now);
                let capacity = self.config.progress_ttl.token_capacity;
                let required_tokens = runtime_views
                    .iter()
                    .find(|program| {
                        program.reference.model_pool() == identity.model_pool()
                            && program.reference.program_id() == identity.program_id()
                    })
                    .map_or(0, |program| program.estimated_context_tokens)
                    .saturating_add(self.config.progress_ttl.decode_buffer_tokens)
                    as f64;
                let immediately_admissible = active_programs.saturating_add(1)
                    <= self.config.max_active_programs_per_target
                    && capacity.is_none_or(|capacity| {
                        target_usage + required_tokens
                            <= capacity as f64 * self.config.progress_ttl.low_watermark_ratio
                    });
                ProgramBindingCandidate {
                    target_id: target_id.clone(),
                    accounted_programs: accounted.len(),
                    accounted_tokens,
                    capacity_tokens: self.config.progress_ttl.token_capacity,
                    kv_pressure: capacity.map_or(0.0, |capacity| {
                        if capacity == 0 {
                            1.0
                        } else {
                            ((target_usage + pending_new_program_tokens) / capacity as f64)
                                .clamp(0.0, 1.0)
                        }
                    }),
                    immediately_admissible,
                }
            })
            .collect()
    }

    pub(crate) fn ensure_decision_state(
        &self,
        state: &mut ProgramSchedulerState,
        identity: &ProgramIdentity,
        reference: ProgramRef,
        estimated_context_tokens: usize,
        now: Instant,
        routing_text: Option<&str>,
    ) {
        let _ = state.lineage.observe(
            identity.model_pool(),
            identity.program_id(),
            identity.parent_program_id(),
            identity.root_program_id(),
        );
        if let Some(decision) = state.decisions.get_mut(&reference) {
            if estimated_context_tokens >= decision.estimated_context_tokens {
                decision.estimated_context_tokens = estimated_context_tokens;
                decision.context_shrink_observations = 0;
            }
            decision.task_id = identity.task_id().map(str::to_string);
            decision.session_id = identity.session_id().map(str::to_string);
            decision.agent_id = identity.agent_id().map(str::to_string);
            decision.parent_program_id = identity.parent_program_id().map(str::to_string);
            decision.blocks_parent = identity.blocks_parent();
            decision.agent_role = identity.agent_role().map(str::to_string);
            decision.spawn_reason = identity.spawn_reason().map(str::to_string);
            decision.output_token_reservation = identity.request_hints().expected_output_tokens;
            decision.step_id = identity
                .step_id()
                .unwrap_or_else(|| decision.step_id.saturating_add(1));
            return;
        }
        let candidates = self.binding_candidates(state, identity, now);
        let home_target = state.bindings.bind(identity, &candidates, routing_text);
        let mut decision = ProgramDecisionState::new(
            identity.placement_key().to_string(),
            identity.placement_hash_key().to_string(),
            home_target,
            estimated_context_tokens,
            now,
        );
        decision.task_id = identity.task_id().map(str::to_string);
        decision.session_id = identity.session_id().map(str::to_string);
        decision.agent_id = identity.agent_id().map(str::to_string);
        decision.parent_program_id = identity.parent_program_id().map(str::to_string);
        decision.blocks_parent = identity.blocks_parent();
        decision.agent_role = identity.agent_role().map(str::to_string);
        decision.spawn_reason = identity.spawn_reason().map(str::to_string);
        decision.output_token_reservation = identity.request_hints().expected_output_tokens;
        decision.step_id = identity.step_id().unwrap_or(0);
        state.decisions.insert(reference, decision);
    }

    pub(crate) fn observation_is_fresh(
        &self,
        observation: &RankObservationState,
        now: Instant,
    ) -> bool {
        observation.observed_at.is_some_and(|observed_at| {
            now.saturating_duration_since(observed_at)
                <= self.config.metrics_interval.saturating_mul(3)
        })
    }

    pub(crate) fn target_usage(
        &self,
        state: &ProgramSchedulerState,
        target_id: &str,
        now: Instant,
    ) -> f64 {
        if let Some(observation) = state
            .observations
            .get(target_id)
            .filter(|observation| self.observation_is_fresh(observation, now))
        {
            if let Some(tokens) = observation.estimated_active_program_tokens {
                return (tokens + observation.active_program_token_delta).max(0.0);
            }
        }
        state
            .runtime
            .views()
            .into_iter()
            .filter(|program| {
                program.state == ProgramState::Active
                    && program.placement.as_deref() == Some(target_id)
            })
            .filter_map(|program| state.decisions.get(&program.reference))
            .map(|decision| decision.private_tokens(self.config.progress_ttl.decode_buffer_tokens))
            .sum()
    }

    pub(crate) fn adjust_usage(
        state: &mut ProgramSchedulerState,
        target_id: &str,
        delta_tokens: f64,
    ) {
        state
            .observations
            .entry(target_id.to_string())
            .or_default()
            .active_program_token_delta += delta_tokens;
    }
}

/// Router-side fence captured before backend observation I/O.
#[derive(Debug, Clone, Default)]
pub struct BackendObservationEpoch {
    target_snapshot_revision: u64,
    checkpoints: HashMap<String, ObservationCheckpoint>,
}

#[derive(Debug, Clone, Default)]
struct ObservationCheckpoint {
    ledger_checkpoint: f64,
    active_reasoning_programs: usize,
    active_reasoning_requests: usize,
    active_reasoning_private_tokens: f64,
    active_acting_private_tokens: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program_scheduling::{ProgramBindingStrategy, ProgramIdentity};
    use serde_json::json;

    #[test]
    fn observation_epoch_preserves_post_checkpoint_delta() {
        let mut config = ProgramSchedulerConfig::default();
        config.progress_ttl.token_capacity = Some(1000);
        let scheduler = ProgramScheduler::new(config);
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://worker".into(),
            dp_rank: Some(0),
        };
        scheduler.sync_targets("model", std::slice::from_ref(&target));
        let epoch = scheduler.begin_observation(std::slice::from_ref(&target));
        {
            let mut state = scheduler.state.lock();
            ProgramScheduler::adjust_usage(&mut state, "rank-0", 50.0);
        }
        scheduler.apply_observations(
            epoch,
            [BackendObservation {
                target_id: "rank-0".into(),
                base_url: "http://worker".into(),
                dp_rank: Some(0),
                kv_cache_usage: Some(0.5),
                running_requests: Some(1),
                waiting_requests: Some(0),
                observed_at: Instant::now(),
            }],
        );
        let state = scheduler.state.lock();
        assert_eq!(
            scheduler.target_usage(&state, "rank-0", Instant::now()),
            550.0
        );
    }

    #[test]
    fn stale_observation_epoch_cannot_restore_removed_target() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://worker".into(),
            dp_rank: Some(0),
        };
        scheduler.sync_targets("model", std::slice::from_ref(&target));
        let epoch = scheduler.begin_observation(std::slice::from_ref(&target));
        scheduler.sync_targets("model", &[]);
        scheduler.apply_observations(
            epoch,
            [BackendObservation {
                target_id: target.id.clone(),
                base_url: target.base_url.clone(),
                dp_rank: target.dp_rank,
                kv_cache_usage: Some(0.5),
                running_requests: Some(1),
                waiting_requests: Some(0),
                observed_at: Instant::now(),
            }],
        );
        let state = scheduler.state.lock();
        assert!(!state.targets.contains_key(&target.id));
        assert!(!state.observations.contains_key(&target.id));
    }

    #[test]
    fn stale_observation_epoch_cannot_cross_remove_and_readd() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://worker".into(),
            dp_rank: Some(0),
        };
        scheduler.sync_targets("model", std::slice::from_ref(&target));
        let epoch = scheduler.begin_observation(std::slice::from_ref(&target));
        scheduler.sync_targets("model", &[]);
        scheduler.sync_targets("model", std::slice::from_ref(&target));
        scheduler.apply_observations(
            epoch,
            [BackendObservation {
                target_id: target.id.clone(),
                base_url: target.base_url.clone(),
                dp_rank: target.dp_rank,
                kv_cache_usage: Some(0.5),
                running_requests: Some(1),
                waiting_requests: Some(0),
                observed_at: Instant::now(),
            }],
        );
        let state = scheduler.state.lock();
        assert_eq!(
            state
                .observations
                .get(&target.id)
                .and_then(|observation| observation.kv_cache_usage),
            None
        );
    }

    #[test]
    fn cache_aware_cold_burst_accounts_bound_unadmitted_programs() {
        let mut config = ProgramSchedulerConfig::default();
        config.binding_strategy = ProgramBindingStrategy::CacheAware;
        config.progress_ttl.token_capacity = Some(1_000);
        let scheduler = ProgramScheduler::new(config);
        let targets = [
            ProgramTarget {
                id: "rank-0".into(),
                base_url: "http://worker-0".into(),
                dp_rank: Some(0),
            },
            ProgramTarget {
                id: "rank-1".into(),
                base_url: "http://worker-1".into(),
                dp_rank: Some(1),
            },
        ];
        scheduler.sync_targets("model", &targets);
        let identity = |program: &str| {
            ProgramIdentity::from_request(
                None,
                Some(&json!({"vllm_xargs":{"agentic_context":{
                    "program_id":program,"task_id":null,"expected_resume":true
                }}})),
                Some("model"),
            )
            .unwrap()
            .unwrap()
        };
        let now = Instant::now();
        let mut state = scheduler.state.lock();
        let first = identity("first");
        let first_handle = state
            .runtime
            .retain_request(&first, 100, Some("cold-a"), now);
        scheduler.ensure_decision_state(
            &mut state,
            &first,
            first_handle.program().clone(),
            100,
            now,
            Some("cold-a"),
        );
        let second = identity("second");
        let second_handle = state
            .runtime
            .retain_request(&second, 100, Some("cold-b"), now);
        scheduler.ensure_decision_state(
            &mut state,
            &second,
            second_handle.program().clone(),
            100,
            now,
            Some("cold-b"),
        );
        assert_eq!(
            state.decisions[first_handle.program()]
                .last_target
                .as_deref(),
            Some("rank-0")
        );
        assert_eq!(
            state.decisions[second_handle.program()]
                .last_target
                .as_deref(),
            Some("rank-1")
        );
    }

    #[test]
    fn omitted_step_advances_the_previous_program_step() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://worker".into(),
            dp_rank: Some(0),
        };
        scheduler.sync_targets("model", std::slice::from_ref(&target));
        let identity = |step_id: Option<u64>| {
            ProgramIdentity::from_request(
                None,
                Some(&json!({"vllm_xargs":{"agentic_context":{
                    "program_id":"program",
                    "task_id":null,
                    "expected_resume":true,
                    "step_id":step_id
                }}})),
                Some("model"),
            )
            .unwrap()
            .unwrap()
        };
        let now = Instant::now();
        let mut state = scheduler.state.lock();
        let explicit = identity(Some(7));
        let handle = state.runtime.retain_request(&explicit, 100, None, now);
        scheduler.ensure_decision_state(
            &mut state,
            &explicit,
            handle.program().clone(),
            100,
            now,
            None,
        );
        let inferred = identity(None);
        state.runtime.retain_request(&inferred, 110, None, now);
        scheduler.ensure_decision_state(
            &mut state,
            &inferred,
            handle.program().clone(),
            110,
            now,
            None,
        );
        assert_eq!(state.decisions[handle.program()].step_id, 8);
    }

    #[tokio::test]
    async fn removed_target_invalidates_placement_and_rebinds_program() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let old_target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://old".into(),
            dp_rank: Some(0),
        };
        let new_target = ProgramTarget {
            id: "rank-1".into(),
            base_url: "http://new".into(),
            dp_rank: Some(1),
        };
        let identity = ProgramIdentity::from_request(
            None,
            Some(&json!({"vllm_xargs":{"agentic_context":{
                "program_id":"p","task_id":null,"expected_resume":true
            }}})),
            Some("model"),
        )
        .unwrap()
        .unwrap();
        let dispatch = scheduler
            .acquire(identity, 100, std::slice::from_ref(&old_target), None)
            .await
            .unwrap();
        scheduler.sync_targets("model", std::slice::from_ref(&new_target));
        let state = scheduler.state.lock();
        assert_eq!(state.runtime.placement(dispatch.program()), None);
        assert_eq!(
            state.runtime.state(dispatch.program()),
            Some((ProgramState::Paused, ProgramStatus::Reasoning))
        );
        assert_eq!(
            state.decisions[dispatch.program()].last_target.as_deref(),
            Some("rank-1")
        );
    }
}
