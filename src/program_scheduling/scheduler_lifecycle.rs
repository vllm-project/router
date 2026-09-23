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
use tracing::{debug, info};

const CACHE_CONTINUITY_HIT_RATIO: f64 = 0.9;

#[derive(Clone, Copy, PartialEq, Eq)]
enum CompletionAction {
    Terminate,
    Pause(ProgramPauseReason),
    StartActingTtl,
}

struct CapacityVictim {
    reference: ProgramRef,
    status: ProgramStatus,
    score: f64,
    private_tokens: f64,
}

impl ProgramScheduler {
    /// Run the historical request-arrival decision sequence before dispatch.
    pub(crate) fn run_request_arrival_decisions(
        &self,
        state: &mut ProgramSchedulerState,
        now: Instant,
    ) {
        self.release_expired_paused(state, now);
        self.expire_acting_ttls(state, now);
        self.repair_capacity(state, now);
        self.schedule_waiting(state, now);
    }

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
        let accounted_tokens_before =
            decision_before.private_tokens(self.config.progress_ttl.decode_buffer_tokens);
        let shared_prefix_for_cost = decision_before
            .shared_prefix_tokens
            .min(observed_prompt_tokens);
        let cache_miss_impact_seconds = self.policy.cache_miss_impact_seconds(
            observed_prompt_tokens.saturating_sub(shared_prefix_for_cost),
        );
        let policy_acting_ttl = state
            .rank_factors
            .get(&target_id)
            .map(|factors| {
                self.policy
                    .fitted_acting_ttl(factors, cache_miss_impact_seconds)
            })
            .unwrap_or_default();
        let acting_ttl = dispatch
            .request_hints()
            .kv_retention_ttl
            .unwrap_or(policy_acting_ttl);
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

        // A retained successor continues the same reasoning placement. Match
        // the historical Router behavior by cancelling a deferred capacity
        // pause; capacity repair can select the Program again after the
        // successor completes if pressure still exists.
        if runtime_after
            .as_ref()
            .is_some_and(|runtime| runtime.in_flight_requests == 0 && runtime.waiting_requests > 0)
        {
            if let Some(decision) = state.decisions.get_mut(dispatch.program()) {
                decision.pause_when_idle = false;
            }
        }

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
                RouterMetrics::record_agent_aware_cache_miss_impact(
                    &target_id,
                    cache_miss_impact_seconds,
                );
                if starts_new_segment {
                    decision.segment_served_rounds = 0;
                    decision.ttl_pause_sampled_segment_rounds = 0;
                    decision.segment_started_at = Some(dispatch.dispatched_at());
                    if freshness_due {
                        decision.shared_prefix_tokens = explicitly_observed_cache_tokens;
                        decision.shared_prefix_observed = true;
                        decision.shared_prefix_fresh_until =
                            Some(decision.shared_prefix_freshness_anchor_at + freshness);
                        if usage.cached_prompt_tokens.is_some() {
                            RouterMetrics::record_agent_aware_shared_prefix_observation(
                                &target_id,
                                explicitly_observed_cache_tokens,
                            );
                        }
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
                if usage.cached_prompt_tokens.is_some()
                    && previous_context.is_some()
                    && !dispatch.placement_start_request()
                    && !cache_continuous
                {
                    let previous_context_tokens = previous_context.unwrap_or_default();
                    info!(
                        event = "unexpected_cache_discontinuity",
                        program = %dispatch.redacted_program_id(),
                        target = %target_id,
                        prompt_tokens = observed_prompt_tokens,
                        cached_prompt_tokens,
                        previous_context_tokens,
                        cache_hit_vs_previous_context = cached_prompt_tokens as f64
                            / previous_context_tokens as f64,
                        uncached_prompt_tokens,
                        estimated_cache_miss_tokens,
                        cache_miss_impact_seconds,
                        segment_served_rounds = decision.segment_served_rounds,
                        placement_start_request = false,
                        "Unexpected cache discontinuity within a Program placement"
                    );
                }
                debug!(
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
                RouterMetrics::record_agent_aware_armed_ttl(
                    &target_id,
                    if dispatch.request_hints().kv_retention_ttl.is_some() {
                        "request_hint"
                    } else {
                        "fitted"
                    },
                    acting_ttl,
                );
                debug!(
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
            decision.output_token_reservation = None;
        }
        if state
            .runtime
            .view(dispatch.program())
            .is_some_and(|runtime| runtime.state == ProgramState::Active)
        {
            let accounted_tokens_after = state.decisions[dispatch.program()]
                .private_tokens(self.config.progress_ttl.decode_buffer_tokens);
            Self::adjust_usage(
                &mut state,
                &target_id,
                accounted_tokens_after - accounted_tokens_before,
            );
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
        }
        self.run_periodic_decisions(&mut state, now);
    }

    /// Evaluate deadlines, repair capacity, and admit rank-local waiters.
    pub fn tick(&self) {
        let now = Instant::now();
        let mut state = self.state.lock();
        self.run_periodic_decisions(&mut state, now);
        self.report_capacity_accounting_transitions(&mut state, now);
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
        if !self.config.binding_only {
            self.repair_capacity(state, now);
            self.schedule_waiting(state, now);
        }
    }

    fn expire_acting_ttls(&self, state: &mut ProgramSchedulerState, now: Instant) -> bool {
        let expired = state
            .runtime
            .iter_views()
            .filter(|runtime| {
                runtime.state == ProgramState::Active
                    && runtime.status == ProgramStatus::Acting
                    && runtime.in_flight_requests == 0
                    && state
                        .decisions
                        .get(runtime.reference)
                        .and_then(|decision| decision.ttl_deadline)
                        .is_some_and(|deadline| deadline <= now)
            })
            .map(|runtime| runtime.reference.clone())
            .collect::<Vec<_>>();
        let mut changed = false;
        for program in expired {
            let (target, rounds, segment_rounds) = state
                .decisions
                .get(&program)
                .map(|decision| {
                    (
                        decision.last_target.clone(),
                        decision
                            .segment_served_rounds
                            .saturating_sub(decision.ttl_pause_sampled_segment_rounds),
                        decision.segment_served_rounds,
                    )
                })
                .unwrap_or_default();
            if self.pause_idle(state, &program, ProgramPauseReason::TtlExpired, now) {
                if let Some(decision) = state.decisions.get_mut(&program) {
                    decision.ttl_pause_sampled_segment_rounds = segment_rounds;
                    decision.rounds_since_ttl_pause = 0;
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
            .iter_views()
            .filter(|runtime| {
                runtime.state == ProgramState::Paused
                    && runtime.status == ProgramStatus::Acting
                    && runtime.in_flight_requests == 0
                    && runtime.waiting_requests == 0
                    && state
                        .decisions
                        .get(runtime.reference)
                        .is_some_and(|decision| {
                            decision.paused_at.is_some_and(|paused_at| {
                                now.saturating_duration_since(paused_at)
                                    >= self.config.paused_retention_ttl
                            })
                        })
            })
            .map(|runtime| runtime.reference.clone())
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
        let active_relief = state
            .runtime
            .view(program)
            .filter(|runtime| runtime.state == ProgramState::Active)
            .and_then(|runtime| runtime.placement)
            .and_then(|target_id| {
                state.decisions.get(program).map(|decision| {
                    (
                        target_id,
                        decision.private_tokens(self.config.progress_ttl.decode_buffer_tokens),
                    )
                })
            });
        if state.runtime.release_idle(program, now) {
            if let Some((target_id, tokens)) = active_relief {
                Self::adjust_usage(state, &target_id, -tokens);
            }
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
            .iter_views()
            .filter(|program| {
                program.state == ProgramState::Active
                    && program.status == ProgramStatus::Reasoning
                    && program.placement == Some(target_id)
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
            let average_prompt = state
                .rank_factors
                .get(&target_id)
                .map_or(1.0, |factors| factors.average_prompt_tokens().max(1.0));
            let mut reasoning_count = 0;
            let mut marked_reasoning_count = 0;
            let mut future_private_relief = 0.0;
            let mut router_usage = 0.0;
            let mut victims = Vec::new();
            for program in state.runtime.iter_views() {
                if program.state != ProgramState::Active
                    || program.placement != Some(target_id.as_str())
                {
                    continue;
                }
                let decision = state.decisions.get(program.reference);
                if let Some(decision) = decision {
                    let private_tokens =
                        decision.private_tokens(self.config.progress_ttl.decode_buffer_tokens);
                    router_usage += private_tokens;
                    if program.status == ProgramStatus::Reasoning && decision.pause_when_idle {
                        marked_reasoning_count += 1;
                        future_private_relief += private_tokens;
                    }
                    let is_victim = match program.status {
                        ProgramStatus::Acting => program.in_flight_requests == 0,
                        ProgramStatus::Reasoning => {
                            !decision.pause_when_idle && program.in_flight_requests > 0
                        }
                    };
                    if is_victim {
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
                        let score = elapsed
                            * (1.0 / (input as f64 / average_prompt).max(1e-6).sqrt()).max(0.5);
                        victims.push(CapacityVictim {
                            reference: program.reference.clone(),
                            status: program.status,
                            score,
                            private_tokens,
                        });
                    }
                }
                if program.status == ProgramStatus::Reasoning {
                    reasoning_count += 1;
                }
            }
            // Admission and resume use current occupancy: a reasoning Program
            // marked to pause still owns its KV until the request finishes.
            // Capacity repair is different: it is selecting any *additional*
            // victims for the same future idle boundary, so relief already
            // committed by earlier repair decisions must be subtracted here.
            let future_completion_relief = average_completion * marked_reasoning_count as f64;
            let target_usage = self
                .observed_target_usage(state, &target_id, now)
                .unwrap_or(router_usage);
            let mut projected = (target_usage + average_completion * reasoning_count as f64
                - future_private_relief
                - future_completion_relief)
                .max(0.0);
            if projected <= high {
                continue;
            }
            victims.sort_by(|left, right| {
                right
                    .score
                    .total_cmp(&left.score)
                    .then_with(|| right.private_tokens.total_cmp(&left.private_tokens))
                    .then_with(|| {
                        left.reference
                            .program_id()
                            .cmp(right.reference.program_id())
                    })
            });
            for victim in victims {
                if projected <= low {
                    break;
                }
                let relief = victim.private_tokens
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
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program_scheduling::{ProgramIdentity, ProgramSchedulerConfig, ProgramTarget};
    use serde_json::json;
    use std::time::Duration;

    fn identity(name: &str) -> ProgramIdentity {
        ProgramIdentity::from_request(
            None,
            Some(&json!({"vllm_xargs":{"agentic_context":{
                "program_id":name,"task_id":null,"expected_resume":true
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
            .acquire(identity("p"), 100, &[target], None)
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

    #[tokio::test]
    async fn request_retention_ttl_overrides_the_policy_ttl_once() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://rank-0".into(),
            dp_rank: Some(0),
        };
        let request = json!({"vllm_xargs":{"agentic_context":{
            "program_id":"ttl-override",
            "task_id":null,
            "expected_resume":true,
            "kv_retention_ttl_seconds":7.0
        }}});
        let identity = ProgramIdentity::from_request(None, Some(&request), Some("model"))
            .unwrap()
            .unwrap();
        let dispatch = scheduler
            .acquire(identity, 100, &[target], None)
            .await
            .unwrap();
        scheduler.complete(&dispatch, true, false, Some(100), Some(0.1));
        let state = scheduler.state.lock();
        let deadline = state.decisions[dispatch.program()].ttl_deadline.unwrap();
        assert!(deadline.saturating_duration_since(Instant::now()) > Duration::from_secs(6));
    }

    #[tokio::test]
    async fn capacity_repair_does_not_remark_committed_future_relief() {
        let mut config = ProgramSchedulerConfig::default();
        config.progress_ttl.token_capacity = Some(400);
        let scheduler = ProgramScheduler::new(config);
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://rank-0".into(),
            dp_rank: Some(0),
        };
        let first = scheduler
            .acquire(identity("first"), 100, std::slice::from_ref(&target), None)
            .await
            .unwrap();
        let second = scheduler
            .acquire(identity("second"), 100, &[target], None)
            .await
            .unwrap();
        let now = Instant::now();
        {
            let mut state = scheduler.state.lock();
            state
                .decisions
                .get_mut(first.program())
                .unwrap()
                .pause_when_idle = true;
            state
                .decisions
                .get_mut(first.program())
                .unwrap()
                .estimated_context_tokens = 140;
            state
                .decisions
                .get_mut(second.program())
                .unwrap()
                .estimated_context_tokens = 140;
            assert!(!scheduler.repair_capacity(&mut state, now));
            assert!(state.decisions[first.program()].pause_when_idle);
            assert!(!state.decisions[second.program()].pause_when_idle);
        }
        scheduler.complete(&first, true, true, Some(140), Some(0.1));
        scheduler.complete(&second, true, true, Some(140), Some(0.1));
    }

    #[tokio::test]
    async fn low_hit_without_a_placement_restart_keeps_the_cache_segment() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://rank-0".into(),
            dp_rank: Some(0),
        };
        let first = scheduler
            .acquire(
                identity("segment"),
                100,
                std::slice::from_ref(&target),
                None,
            )
            .await
            .unwrap();
        let second = scheduler
            .acquire(identity("segment"), 130, &[target], None)
            .await
            .unwrap();
        assert!(first.placement_start_request());
        assert!(!second.placement_start_request());
        scheduler.complete(
            &first,
            true,
            false,
            ProgramUsageObservation {
                prompt_tokens: Some(100),
                completion_tokens: Some(10),
                cached_prompt_tokens: Some(20),
            },
            Some(0.1),
        );
        scheduler.complete(
            &second,
            true,
            false,
            ProgramUsageObservation {
                prompt_tokens: Some(130),
                completion_tokens: Some(5),
                cached_prompt_tokens: Some(30),
            },
            Some(0.1),
        );
        let state = scheduler.state.lock();
        let decision = &state.decisions[second.program()];
        assert_eq!(decision.segment_served_rounds, 2);
        assert_eq!(decision.shared_prefix_tokens, 20);
    }

    #[tokio::test]
    async fn missing_cache_usage_never_becomes_shared_prefix_evidence() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://rank-0".into(),
            dp_rank: Some(0),
        };
        let dispatch = scheduler
            .acquire(identity("missing-cache"), 100, &[target], None)
            .await
            .unwrap();
        scheduler.complete(
            &dispatch,
            true,
            false,
            ProgramUsageObservation {
                prompt_tokens: Some(100),
                completion_tokens: Some(5),
                cached_prompt_tokens: None,
            },
            Some(0.1),
        );
        let state = scheduler.state.lock();
        assert_eq!(state.decisions[dispatch.program()].shared_prefix_tokens, 0);
    }

    #[tokio::test]
    async fn terminal_completion_releases_active_epoch_occupancy() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://rank-0".into(),
            dp_rank: Some(0),
        };
        let dispatch = scheduler
            .acquire(identity("terminal"), 100, &[target], None)
            .await
            .unwrap();
        assert_eq!(
            scheduler.state.lock().observations["rank-0"].active_program_token_delta,
            200.0
        );
        scheduler.complete(&dispatch, true, true, Some(100), Some(0.1));
        let state = scheduler.state.lock();
        assert_eq!(state.observations["rank-0"].active_program_token_delta, 0.0);
        assert!(!state.decisions.contains_key(dispatch.program()));
    }
}
