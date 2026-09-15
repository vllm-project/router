//! Request completion, acting TTL, lifecycle release, and capacity repair.
//!
//! Completion is the only path that records request-derived rolling factors.
//! Periodic ticks evaluate existing deadlines and never create request facts.

use super::scheduler::ProgramScheduler;
use super::scheduler_state::{ProgramPauseReason, ProgramSchedulerState};
use super::{
    ProgramDispatch, ProgramRef, ProgramState, ProgramStatus, ProgramUsageObservation,
    RequestSample,
};
use crate::metrics::RouterMetrics;
use std::time::Instant;
use tracing::info;

const CACHE_CONTINUITY_HIT_RATIO: f64 = 0.9;

#[derive(Clone, Copy, PartialEq, Eq)]
enum CompletionAction {
    Terminate,
    Pause(ProgramPauseReason),
    StartActingTtl,
}

impl ProgramScheduler {
    /// Record one backend completion and run the same follow-up decisions as a tick.
    pub fn complete(
        &self,
        dispatch: &ProgramDispatch,
        success: bool,
        terminate: bool,
        usage: impl Into<ProgramUsageObservation>,
        decode_seconds: Option<f64>,
    ) {
        let usage = usage.into();
        let now = Instant::now();
        let mut state = self.state.lock();
        let Some(runtime_before) = state.runtime.view(dispatch.program()) else {
            return;
        };
        if runtime_before.placement.as_deref() != Some(dispatch.target_id.as_str()) {
            return;
        }
        let target_id = dispatch.target_id.clone();
        let observed_prompt_tokens = usage
            .prompt_tokens
            .unwrap_or(dispatch.estimated_context_tokens());
        let fallback_cached_tokens = state
            .rank_factors
            .get(&target_id)
            .map(|factors| factors.average_cached_prompt_tokens().round() as usize)
            .unwrap_or(0);
        let cached_prompt_tokens = usage
            .cached_prompt_tokens
            .unwrap_or(fallback_cached_tokens)
            .min(observed_prompt_tokens);
        let explicitly_observed_cache_tokens = usage
            .cached_prompt_tokens
            .unwrap_or(0)
            .min(observed_prompt_tokens);
        let Some(decision_before) = state.decisions.get(dispatch.program()) else {
            return;
        };
        let shared_prefix_for_cost = decision_before
            .shared_prefix_tokens
            .min(observed_prompt_tokens);
        let cache_miss_impact_seconds = self.policy.cache_miss_impact_seconds(
            observed_prompt_tokens.saturating_sub(shared_prefix_for_cost),
        );
        let acting_ttl = state
            .rank_factors
            .get(&target_id)
            .map(|factors| {
                self.policy
                    .fitted_acting_ttl(factors, cache_miss_impact_seconds)
            })
            .unwrap_or_default();
        let ttl_window_samples = state
            .rank_factors
            .get(&target_id)
            .map_or(0, |factors| factors.continuity_sample_count());
        let ttl_window_complete = state.rank_factors.get(&target_id).is_some_and(|factors| {
            factors.continuity_window_complete(self.config.progress_ttl.stats_window_size)
        });
        let uncached_prompt_tokens = observed_prompt_tokens.saturating_sub(cached_prompt_tokens);
        let estimated_cache_miss_tokens =
            observed_prompt_tokens.saturating_sub(shared_prefix_for_cost);
        let freshness = self.shared_prefix_freshness(&state, &target_id);
        if !state.runtime.complete_request(dispatch, usage) {
            return;
        }
        let runtime_after = state.runtime.view(dispatch.program());

        let completion_action = state
            .decisions
            .get_mut(dispatch.program())
            .and_then(|decision| {
                decision.terminate_when_idle |= terminate;
                runtime_after
                    .as_ref()
                    .filter(|runtime| {
                        runtime.in_flight_requests == 0 && runtime.waiting_requests == 0
                    })
                    .map(|_| {
                        if decision.terminate_when_idle {
                            CompletionAction::Terminate
                        } else if decision.pause_when_idle || !success {
                            CompletionAction::Pause(if decision.pause_when_idle {
                                ProgramPauseReason::CapacityRepair
                            } else {
                                ProgramPauseReason::RequestFailed
                            })
                        } else {
                            CompletionAction::StartActingTtl
                        }
                    })
            });
        if let Some(CompletionAction::Pause(reason)) = completion_action {
            self.pause_idle(&mut state, dispatch.program(), reason, now);
        }

        let mut request_sample = None;
        if let Some(decision) = state.decisions.get_mut(dispatch.program()) {
            let previous_context = decision.last_context_tokens;
            let context_growth_tokens = previous_context.and_then(|previous| {
                let growth = observed_prompt_tokens.saturating_sub(previous);
                (previous > 0 && growth > 0 && growth < previous).then_some(growth)
            });
            let cache_continuous = previous_context.is_none_or(|previous| {
                cached_prompt_tokens as f64 >= CACHE_CONTINUITY_HIT_RATIO * previous as f64
            });
            let starts_new_segment = dispatch.placement_start_request()
                && (previous_context.is_none() || !cache_continuous);
            let freshness_due = starts_new_segment
                && (!decision.shared_prefix_observed
                    || decision
                        .shared_prefix_fresh_until
                        .is_none_or(|deadline| now >= deadline));
            if usage.prompt_tokens.is_some() {
                decision.observe_completed_context(observed_prompt_tokens);
            } else if observed_prompt_tokens >= decision.estimated_context_tokens {
                decision.estimated_context_tokens = observed_prompt_tokens;
                decision.context_shrink_observations = 0;
            }
            if success {
                if starts_new_segment {
                    decision.segment_served_rounds = 0;
                    decision.ttl_pause_sampled_segment_rounds = 0;
                    decision.segment_started_at = Some(dispatch.dispatched_at());
                    if freshness_due {
                        decision.shared_prefix_tokens = explicitly_observed_cache_tokens;
                        decision.shared_prefix_observed = true;
                        decision.shared_prefix_fresh_until =
                            Some(decision.shared_prefix_freshness_anchor_at + freshness);
                    }
                }
                let completion_tokens = usage.completion_tokens.unwrap_or(0);
                decision.completed_requests = decision.completed_requests.saturating_add(1);
                decision.segment_served_rounds = decision.segment_served_rounds.saturating_add(1);
                decision.rounds_since_activation = decision
                    .rounds_since_activation
                    .saturating_add(usize::from(cache_continuous));
                decision.rounds_since_ttl_pause = decision
                    .rounds_since_ttl_pause
                    .saturating_add(usize::from(cache_continuous));
                decision.last_context_tokens = Some(observed_prompt_tokens);
                decision.last_completion_tokens = completion_tokens;
                decision.lifetime_generated_tokens = decision
                    .lifetime_generated_tokens
                    .saturating_add(completion_tokens);
                decision.last_request_finished_at = Some(now);
                decision.last_cache_miss_impact_seconds = cache_miss_impact_seconds;
                info!(
                    event = "cache_observation",
                    program = %dispatch.redacted_program_id(),
                    target = %target_id,
                    prompt_tokens = observed_prompt_tokens,
                    cached_prompt_tokens,
                    previous_context_tokens = previous_context.unwrap_or(0),
                    uncached_prompt_tokens,
                    shared_prefix_tokens_for_cost = shared_prefix_for_cost,
                    estimated_cache_miss_tokens,
                    cache_miss_impact_seconds,
                    cache_continuous,
                    placement_start_request = dispatch.placement_start_request(),
                    starts_new_segment,
                    segment_served_rounds = decision.segment_served_rounds,
                    shared_prefix_observation_due = freshness_due,
                    shared_prefix_tokens_after = decision.shared_prefix_tokens,
                    "Program scheduling diagnostic"
                );
                request_sample = Some(RequestSample {
                    prompt_tokens: observed_prompt_tokens,
                    completion_tokens,
                    cached_prompt_tokens,
                    context_growth_tokens,
                    e2e_seconds: now
                        .saturating_duration_since(dispatch.request_arrived_at())
                        .as_secs_f64(),
                    decode_seconds,
                    queue_seconds: dispatch
                        .dispatched_at()
                        .saturating_duration_since(dispatch.request_arrived_at())
                        .as_secs_f64(),
                    rounds_since_ttl_pause: decision.rounds_since_ttl_pause,
                    finished_at: now,
                });
            }
            if completion_action == Some(CompletionAction::StartActingTtl) {
                decision.acting_since = Some(now);
                decision.ttl_deadline = Some(now + acting_ttl);
                info!(
                    event = "ttl_armed",
                    program = %dispatch.redacted_program_id(),
                    target = %target_id,
                    prompt_tokens = observed_prompt_tokens,
                    cached_prompt_tokens,
                    uncached_prompt_tokens,
                    shared_prefix_tokens_for_cost = shared_prefix_for_cost,
                    estimated_cache_miss_tokens,
                    request_cache_miss_impact_seconds = cache_miss_impact_seconds,
                    request_ttl_seconds = acting_ttl.as_secs_f64(),
                    ttl_window_samples,
                    ttl_window_complete,
                    "Program scheduling diagnostic"
                );
            }
        }
        if let Some(sample) = request_sample {
            state
                .rank_factors
                .entry(target_id.clone())
                .or_default()
                .push_request(sample, self.config.progress_ttl.stats_window_size);
        }
        if completion_action == Some(CompletionAction::Terminate) {
            self.release_program(&mut state, dispatch.program(), now);
        }
        if !self.config.binding_only {
            self.yield_completed_segment(&mut state, dispatch.program(), now);
            self.run_periodic_decisions(&mut state, now);
        }
    }

    /// Evaluate deadlines, repair capacity, and admit rank-local waiters.
    pub fn tick(&self) {
        let now = Instant::now();
        let mut state = self.state.lock();
        self.run_periodic_decisions(&mut state, now);
        self.publish_metrics(&state);
    }

    pub(crate) fn run_periodic_decisions(&self, state: &mut ProgramSchedulerState, now: Instant) {
        state.runtime.prune_released_generations(
            now,
            self.config
                .paused_retention_ttl
                .max(std::time::Duration::from_secs(60)),
        );
        self.expire_acting_ttls(state, now);
        self.release_expired_paused(state, now);
        self.reconcile_privileges(state, now);
        self.repair_capacity(state, now);
        self.schedule_waiting(state, now);
    }

    fn expire_acting_ttls(&self, state: &mut ProgramSchedulerState, now: Instant) -> bool {
        let expired = state
            .runtime
            .views()
            .into_iter()
            .filter(|runtime| {
                runtime.state == ProgramState::Active
                    && runtime.status == ProgramStatus::Acting
                    && runtime.in_flight_requests == 0
                    && state
                        .decisions
                        .get(&runtime.reference)
                        .and_then(|decision| decision.ttl_deadline)
                        .is_some_and(|deadline| deadline <= now)
            })
            .map(|runtime| runtime.reference)
            .collect::<Vec<_>>();
        let mut changed = false;
        for program in expired {
            let (target, rounds, segment_rounds, placement_key, was_privileged) = state
                .decisions
                .get(&program)
                .map(|decision| {
                    (
                        decision.last_target.clone(),
                        decision
                            .segment_served_rounds
                            .saturating_sub(decision.ttl_pause_sampled_segment_rounds),
                        decision.segment_served_rounds,
                        decision.placement_key.clone(),
                        decision
                            .privilege_deadline
                            .is_some_and(|deadline| deadline > now),
                    )
                })
                .unwrap_or_default();
            let privilege_target = was_privileged
                .then(|| {
                    let mut related = state
                        .runtime
                        .views()
                        .into_iter()
                        .filter(|candidate| candidate.reference != program)
                        .filter(|candidate| {
                            state
                                .decisions
                                .get(&candidate.reference)
                                .is_some_and(|decision| {
                                    decision.placement_key == placement_key
                                        && decision.last_target == target
                                })
                        })
                        .map(|candidate| candidate.reference)
                        .collect::<Vec<_>>();
                    related.sort_by_key(|candidate| {
                        let runtime = state.runtime.view(candidate).unwrap();
                        let tier = match (runtime.state, runtime.status) {
                            (ProgramState::Active, ProgramStatus::Reasoning) => 0,
                            (ProgramState::Paused, ProgramStatus::Reasoning) => 1,
                            (ProgramState::Active, ProgramStatus::Acting) => 2,
                            _ => 3,
                        };
                        (tier, candidate.program_id().to_string())
                    });
                    related.into_iter().next()
                })
                .flatten();
            if self.pause_idle(state, &program, ProgramPauseReason::TtlExpired, now) {
                if let Some(decision) = state.decisions.get_mut(&program) {
                    decision.ttl_pause_sampled_segment_rounds = segment_rounds;
                    decision.rounds_since_ttl_pause = 0;
                    if was_privileged {
                        decision.privilege_deadline = None;
                    }
                }
                if let Some(target) = privilege_target {
                    if let Some(decision) = state.decisions.get_mut(&target) {
                        decision.privilege_deadline = Some(now + self.config.privileged_ttl);
                        decision.privilege_ttl_expired = false;
                    }
                }
                if let Some(target) = target {
                    state
                        .rank_factors
                        .entry(target)
                        .or_default()
                        .push_ttl_pause_rounds(rounds, self.config.progress_ttl.stats_window_size);
                }
                changed = true;
            }
        }
        changed
    }

    fn release_expired_paused(&self, state: &mut ProgramSchedulerState, now: Instant) -> bool {
        let expired = state
            .runtime
            .views()
            .into_iter()
            .filter(|runtime| {
                runtime.state == ProgramState::Paused
                    && runtime.status == ProgramStatus::Acting
                    && runtime.in_flight_requests == 0
                    && runtime.waiting_requests == 0
                    && state
                        .decisions
                        .get(&runtime.reference)
                        .is_some_and(|decision| {
                            decision.paused_at.is_some_and(|paused_at| {
                                now.saturating_duration_since(paused_at)
                                    >= self.config.paused_retention_ttl
                            })
                        })
            })
            .map(|runtime| runtime.reference)
            .collect::<Vec<_>>();
        for program in &expired {
            if let Some(target_id) = state
                .decisions
                .get(program)
                .and_then(|decision| decision.last_target.as_deref())
            {
                RouterMetrics::record_agent_aware_transition(target_id, "paused_retention_release");
            }
            self.release_program(state, program, now);
        }
        !expired.is_empty()
    }

    pub(crate) fn release_program(
        &self,
        state: &mut ProgramSchedulerState,
        program: &ProgramRef,
        now: Instant,
    ) {
        for queue in state.rank_queues.values_mut() {
            queue.retain(|queued| queued != program);
        }
        for queue in state.global_queues.values_mut() {
            queue.retain(|queued| queued != program);
        }
        if state.runtime.release_idle(program, now) {
            state.decisions.remove(program);
            state.bindings.release_program(program);
        }
    }

    fn shared_prefix_freshness(
        &self,
        state: &ProgramSchedulerState,
        target_id: &str,
    ) -> std::time::Duration {
        let Some(factors) = state.rank_factors.get(target_id) else {
            return self.config.shared_prefix_freshness_warmup;
        };
        if !factors.request_window_complete(self.config.progress_ttl.stats_window_size) {
            return self.config.shared_prefix_freshness_warmup;
        }
        let Some(capacity) = self.config.progress_ttl.token_capacity else {
            return self.config.shared_prefix_freshness_warmup;
        };
        let active_reasoning = state
            .runtime
            .views()
            .into_iter()
            .filter(|program| {
                program.state == ProgramState::Active
                    && program.status == ProgramStatus::Reasoning
                    && program.placement.as_deref() == Some(target_id)
            })
            .count();
        let average_uncached = factors.average_uncached_prompt_tokens();
        let average_latency = factors.average_e2e_seconds();
        if active_reasoning == 0 || average_uncached <= 0.0 || average_latency <= 0.0 {
            return self.config.shared_prefix_freshness_warmup;
        }
        let seconds =
            self.config.shared_prefix_freshness_kv_turnovers * capacity as f64 * average_latency
                / active_reasoning as f64
                / average_uncached;
        if seconds.is_finite() && seconds >= 0.0 {
            std::time::Duration::from_secs_f64(seconds)
        } else {
            self.config.shared_prefix_freshness_warmup
        }
    }

    fn repair_capacity(&self, state: &mut ProgramSchedulerState, now: Instant) -> bool {
        let Some(capacity) = self.config.progress_ttl.token_capacity else {
            return false;
        };
        let target_ids = state.targets.keys().cloned().collect::<Vec<_>>();
        let mut changed = false;
        for target_id in target_ids {
            let high = capacity as f64 * self.config.progress_ttl.high_watermark_ratio;
            let low = capacity as f64 * self.config.progress_ttl.low_watermark_ratio;
            let average_completion = state
                .rank_factors
                .get(&target_id)
                .map_or(0.0, |factors| factors.average_completion_tokens().ceil());
            let views = state.runtime.views();
            let reasoning_count = views
                .iter()
                .filter(|program| {
                    program.state == ProgramState::Active
                        && program.status == ProgramStatus::Reasoning
                        && program.placement.as_deref() == Some(target_id.as_str())
                })
                .count();
            let mut projected = self.target_usage(state, &target_id, now)
                + average_completion * reasoning_count as f64;
            if projected <= high {
                continue;
            }
            let mut victims = views
                .into_iter()
                .filter(|program| {
                    program.state == ProgramState::Active
                        && program.placement.as_deref() == Some(target_id.as_str())
                        && state
                            .decisions
                            .get(&program.reference)
                            .is_some_and(|decision| {
                                decision
                                    .privilege_deadline
                                    .is_none_or(|deadline| deadline <= now)
                                    && match program.status {
                                        ProgramStatus::Acting => program.in_flight_requests == 0,
                                        ProgramStatus::Reasoning => {
                                            !decision.pause_when_idle
                                                && program.in_flight_requests > 0
                                        }
                                    }
                            })
                })
                .collect::<Vec<_>>();
            let average_prompt = state
                .rank_factors
                .get(&target_id)
                .map_or(1.0, |factors| factors.average_prompt_tokens().max(1.0));
            victims.sort_by(|left, right| {
                let score = |program: &super::runtime::RuntimeProgramView| {
                    let decision = &state.decisions[&program.reference];
                    let elapsed = decision.segment_started_at.map_or(0.0, |started| {
                        now.saturating_duration_since(started).as_secs_f64()
                    });
                    let input = if program.status == ProgramStatus::Acting {
                        decision
                            .last_context_tokens
                            .unwrap_or(decision.estimated_context_tokens)
                    } else {
                        decision.estimated_context_tokens
                    };
                    elapsed * (1.0 / (input as f64 / average_prompt).max(1e-6).sqrt()).max(0.5)
                };
                score(right).total_cmp(&score(left)).then_with(|| {
                    left.reference
                        .program_id()
                        .cmp(right.reference.program_id())
                })
            });
            for victim in victims {
                if projected <= low {
                    break;
                }
                let private = state.decisions[&victim.reference]
                    .private_tokens(self.config.progress_ttl.decode_buffer_tokens);
                let relief = private
                    + if victim.status == ProgramStatus::Reasoning {
                        average_completion
                    } else {
                        0.0
                    };
                let relieved = if victim.status == ProgramStatus::Acting {
                    self.pause_idle(
                        state,
                        &victim.reference,
                        ProgramPauseReason::CapacityRepair,
                        now,
                    )
                } else if let Some(decision) = state.decisions.get_mut(&victim.reference) {
                    decision.pause_when_idle = true;
                    RouterMetrics::record_agent_aware_transition(
                        &target_id,
                        "capacity_repair_mark",
                    );
                    true
                } else {
                    false
                };
                if relieved {
                    projected = (projected - relief).max(0.0);
                    changed = true;
                }
            }
            if projected > low {
                let mut privileged = state
                    .runtime
                    .views()
                    .into_iter()
                    .filter(|program| {
                        program.state == ProgramState::Active
                            && program.placement.as_deref() == Some(target_id.as_str())
                            && state
                                .decisions
                                .get(&program.reference)
                                .is_some_and(|decision| {
                                    decision
                                        .privilege_deadline
                                        .is_some_and(|deadline| deadline > now)
                                })
                    })
                    .collect::<Vec<_>>();
                privileged.sort_by(|left, right| {
                    state.decisions[&left.reference]
                        .lifetime_generated_tokens
                        .cmp(&state.decisions[&right.reference].lifetime_generated_tokens)
                        .then_with(|| {
                            left.reference
                                .program_id()
                                .cmp(right.reference.program_id())
                        })
                });
                for victim in privileged {
                    if projected <= low {
                        break;
                    }
                    let private = state.decisions[&victim.reference]
                        .private_tokens(self.config.progress_ttl.decode_buffer_tokens);
                    let relief = private
                        + if victim.status == ProgramStatus::Reasoning {
                            average_completion
                        } else {
                            0.0
                        };
                    state
                        .decisions
                        .get_mut(&victim.reference)
                        .unwrap()
                        .privilege_deadline = None;
                    let relieved = if victim.status == ProgramStatus::Acting {
                        self.pause_idle(
                            state,
                            &victim.reference,
                            ProgramPauseReason::CapacityRepair,
                            now,
                        )
                    } else if let Some(decision) = state.decisions.get_mut(&victim.reference) {
                        decision.pause_when_idle = true;
                        RouterMetrics::record_agent_aware_transition(
                            &target_id,
                            "capacity_repair_mark",
                        );
                        true
                    } else {
                        false
                    };
                    if relieved {
                        projected = (projected - relief).max(0.0);
                        RouterMetrics::record_agent_aware_transition(
                            &target_id,
                            "progress_privilege_demote",
                        );
                        changed = true;
                    }
                }
            }
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program_scheduling::{ProgramIdentity, ProgramSchedulerConfig, ProgramTarget};
    use serde_json::json;

    fn identity() -> ProgramIdentity {
        ProgramIdentity::from_request(
            None,
            Some(&json!({"vllm_xargs":{"agentic_context":{
                "program_id":"p","task_id":null,"expected_resume":true
            }}})),
            Some("model"),
        )
        .unwrap()
        .unwrap()
    }

    #[tokio::test]
    async fn completion_immediately_pauses_zero_ttl_before_window_is_complete() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://rank-0".into(),
            dp_rank: Some(0),
        };
        let dispatch = scheduler
            .acquire(identity(), 100, &[target], None)
            .await
            .unwrap();
        scheduler.complete(
            &dispatch,
            true,
            false,
            ProgramUsageObservation {
                prompt_tokens: Some(100),
                completion_tokens: Some(10),
                cached_prompt_tokens: Some(0),
            },
            Some(0.1),
        );
        let state = scheduler.state.lock();
        let decision = &state.decisions[dispatch.program()];
        assert_eq!(decision.ttl_deadline, None);
        assert_eq!(
            decision.last_pause_reason,
            Some(ProgramPauseReason::TtlExpired)
        );
        assert_eq!(
            state.runtime.state(dispatch.program()),
            Some((ProgramState::Paused, ProgramStatus::Acting))
        );
    }
}
