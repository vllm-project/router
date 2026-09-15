//! ProgramScheduler orchestration and backend observation epochs.
//!
//! Rank-local and global decision algorithms live in sibling modules. This
//! file owns atomic state access, target discovery, and observation rebasing.

use super::scheduler_state::{ProgramDecisionState, ProgramSchedulerState, RankObservationState};
use super::{
    BackendObservation, ProgramBindingCandidate, ProgramIdentity, ProgramRef,
    ProgramSchedulerConfig, ProgramState, ProgramStatus, ProgramTarget, ProgressTtlPolicyMath,
};
use parking_lot::Mutex;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::{Duration, Instant};

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

    /// Synchronize one model pool's concrete backend and internal-DP targets.
    pub fn sync_targets(&self, model_pool: &str, targets: &[ProgramTarget]) {
        let mut state = self.state.lock();
        let incoming = targets
            .iter()
            .cloned()
            .map(|target| (target.id.clone(), target))
            .collect::<BTreeMap<_, _>>();
        let current_ids = incoming.keys().cloned().collect::<BTreeSet<_>>();
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
            state.rank_queues.remove(target_id);
            state.rank_factors.remove(target_id);
            state.observations.remove(target_id);
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
        BackendObservationEpoch { checkpoints }
    }

    /// Install raw observations and preserve transitions committed during I/O.
    pub fn apply_observations(
        &self,
        epoch: BackendObservationEpoch,
        observations: impl IntoIterator<Item = BackendObservation>,
    ) {
        let mut state = self.state.lock();
        for raw in observations {
            let Some(checkpoint) = epoch.checkpoints.get(&raw.target_id) else {
                continue;
            };
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
            let observation = state.observations.entry(raw.target_id).or_default();
            observation.active_program_token_delta -= checkpoint.ledger_checkpoint;
            observation.kv_cache_usage = raw.kv_cache_usage;
            observation.running_requests = raw.running_requests;
            observation.waiting_requests = raw.waiting_requests;
            observation.router_active_reasoning_programs = checkpoint.active_reasoning_programs;
            observation.router_active_reasoning_requests = checkpoint.active_reasoning_requests;
            observation.estimated_active_program_tokens = estimated_active_program_tokens;
            observation.observed_at = Some(raw.observed_at);
        }
    }

    pub(crate) fn binding_candidates(
        &self,
        state: &ProgramSchedulerState,
        identity: &ProgramIdentity,
    ) -> Vec<ProgramBindingCandidate> {
        state
            .model_targets
            .get(identity.model_pool())
            .into_iter()
            .flatten()
            .map(|target_id| {
                let accounted = state
                    .runtime
                    .views()
                    .into_iter()
                    .filter(|program| {
                        let home = state
                            .decisions
                            .get(&program.reference)
                            .and_then(|decision| decision.home_target.as_deref());
                        home == Some(target_id.as_str())
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
                ProgramBindingCandidate {
                    target_id: target_id.clone(),
                    accounted_programs: accounted.len(),
                    accounted_tokens,
                    capacity_tokens: self.config.progress_ttl.token_capacity,
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
    ) {
        if let Some(decision) = state.decisions.get_mut(&reference) {
            if estimated_context_tokens >= decision.estimated_context_tokens {
                decision.estimated_context_tokens = estimated_context_tokens;
                decision.context_shrink_observations = 0;
            }
            return;
        }
        let candidates = self.binding_candidates(state, identity);
        let home_target = state.bindings.bind(identity, &candidates);
        state.decisions.insert(
            reference,
            ProgramDecisionState::new(
                identity.placement_key().to_string(),
                home_target,
                estimated_context_tokens,
                now,
            ),
        );
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
}
