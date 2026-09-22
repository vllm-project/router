//! Request retention and rank-local Program admission.
//!
//! This module owns request-scoped ordering hints, yield-aware resume order,
//! and capacity checks. Global Queue placement is implemented separately.

use super::scheduler::ProgramScheduler;
use super::scheduler_state::{ProgramPauseReason, ProgramResumeOrder, ProgramSchedulerState};
use super::{
    BatchGainInputs, ContinuitySample, ProgramDispatch, ProgramIdentity, ProgramRef,
    ProgramRequestHandle, ProgramState, ProgramStatus, ProgramTarget, ScheduleError,
};
use crate::metrics::{AgentAwareAdmissionMetrics, RouterMetrics};
use std::cmp::Ordering;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info};

#[derive(Debug, Clone)]
pub(crate) struct RankAdmissionPlan {
    pub(crate) used_tokens: f64,
    pub(crate) required_tokens: f64,
    pub(crate) reserve_tokens: f64,
    pub(crate) capacity_tokens: Option<usize>,
    pub(crate) forced: bool,
    pub(crate) batch_gain: bool,
}

impl ProgramScheduler {
    /// Retain one Program request until rank-local admission permits dispatch.
    pub async fn acquire(
        &self,
        identity: ProgramIdentity,
        estimated_context_tokens: usize,
        targets: &[ProgramTarget],
        routing_text: Option<String>,
    ) -> Result<ProgramDispatch, ScheduleError> {
        self.sync_targets(identity.model_pool(), targets);
        self.acquire_synced(identity, estimated_context_tokens, targets, routing_text)
            .await
    }

    /// Retain a request using a Router-cached immutable target snapshot.
    pub async fn acquire_from_snapshot(
        &self,
        identity: ProgramIdentity,
        estimated_context_tokens: usize,
        targets: Arc<[ProgramTarget]>,
        routing_text: Option<String>,
    ) -> Result<ProgramDispatch, ScheduleError> {
        self.sync_target_snapshot(identity.model_pool(), targets.clone());
        self.acquire_synced(
            identity,
            estimated_context_tokens,
            targets.as_ref(),
            routing_text,
        )
        .await
    }

    async fn acquire_synced(
        &self,
        identity: ProgramIdentity,
        estimated_context_tokens: usize,
        targets: &[ProgramTarget],
        routing_text: Option<String>,
    ) -> Result<ProgramDispatch, ScheduleError> {
        if targets.is_empty() {
            return Err(ScheduleError::NoTargets);
        }
        let arrived_at = Instant::now();
        let handle = {
            let mut state = self.state.lock();
            let handle = state.runtime.retain_request(
                &identity,
                estimated_context_tokens,
                routing_text.as_deref(),
                arrived_at,
            );
            let reference = handle.program().clone();
            let continuity = state
                .runtime
                .view(&reference)
                .filter(|runtime| runtime.in_flight_requests == 0)
                .and_then(|_| {
                    state.decisions.get(&reference).and_then(|decision| {
                        decision.last_request_finished_at.map(|finished_at| {
                            (
                                decision.last_target.clone(),
                                ContinuitySample {
                                    interval_seconds: arrived_at
                                        .saturating_duration_since(finished_at)
                                        .as_secs_f64(),
                                    cache_miss_impact_seconds: decision
                                        .last_cache_miss_impact_seconds,
                                },
                            )
                        })
                    })
                });
            if let Some((Some(target_id), sample)) = continuity {
                if sample.interval_seconds.is_finite() && sample.interval_seconds >= 0.0 {
                    let factors = state.rank_factors.entry(target_id.clone()).or_default();
                    debug!(
                        event = "continuity_sample",
                        program = %reference.redacted_id(),
                        target = %target_id,
                        interval_seconds = sample.interval_seconds,
                        previous_cache_miss_impact_seconds = sample.cache_miss_impact_seconds,
                        window_samples_before = factors.continuity_sample_count(),
                        window_complete_before = factors.continuity_window_complete(
                            self.config.progress_ttl.stats_window_size
                        ),
                        "Program scheduling diagnostic"
                    );
                    RouterMetrics::record_agent_aware_request_interval(
                        &target_id,
                        sample.interval_seconds,
                    );
                    factors.push_continuity(sample, self.config.progress_ttl.stats_window_size);
                }
            }
            let previous_tokens = state.decisions.get(&reference).map_or(0.0, |decision| {
                decision.private_tokens(self.config.progress_ttl.decode_buffer_tokens)
            });
            self.ensure_decision_state(
                &mut state,
                &identity,
                reference.clone(),
                estimated_context_tokens,
                arrived_at,
                routing_text.as_deref(),
            );
            let current_tokens = state.decisions[&reference]
                .private_tokens(self.config.progress_ttl.decode_buffer_tokens);
            if let Some(placement) = state
                .runtime
                .view(&reference)
                .and_then(|program| program.placement)
            {
                let account_growth = state
                    .decisions
                    .get(&reference)
                    .is_some_and(|decision| !decision.pause_when_idle);
                if account_growth && current_tokens > previous_tokens {
                    Self::adjust_usage(&mut state, &placement, current_tokens - previous_tokens);
                }
            } else {
                let target_id = state
                    .decisions
                    .get(&reference)
                    .and_then(|decision| decision.last_target.clone())
                    .ok_or(ScheduleError::NoTargets)?;
                let queue = if self.config.global_queue {
                    state
                        .global_queues
                        .entry(identity.model_pool().to_string())
                        .or_default()
                } else {
                    state.rank_queues.entry(target_id.clone()).or_default()
                };
                if !queue.iter().any(|queued| queued == &reference) {
                    queue.push_back(reference.clone());
                }
                if let Some(decision) = state.decisions.get_mut(&reference) {
                    decision.queued_at.get_or_insert(arrived_at);
                }
                self.set_force_resume_deadline(&mut state, &target_id, &reference, arrived_at);
            }
            if self.config.binding_only {
                self.activate_binding_only(&mut state, &reference);
            } else {
                self.run_request_arrival_decisions(&mut state, arrived_at);
            }
            handle
        };
        let mut guard = AdmissionGuard::new(self, handle);
        let deadline = arrived_at + self.config.queue_timeout;
        loop {
            let wait_handle = guard.handle().clone();
            let notified = wait_handle.notified();
            if let Some(dispatch) = self.claim_dispatch(guard.handle()) {
                guard.disarm();
                return Ok(dispatch);
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(ScheduleError::QueueTimeout);
            }
            let force_deadline = {
                let state = self.state.lock();
                state
                    .decisions
                    .get(guard.handle().program())
                    .and_then(|decision| decision.force_resume_deadline)
            };
            let wake_at = force_deadline.map_or(deadline, |force| force.min(deadline));
            if tokio::time::timeout(wake_at.saturating_duration_since(now), notified)
                .await
                .is_err()
            {
                let mut state = self.state.lock();
                self.run_request_arrival_decisions(&mut state, Instant::now());
            }
        }
    }

    fn activate_binding_only(&self, state: &mut ProgramSchedulerState, program: &ProgramRef) {
        if state
            .runtime
            .view(program)
            .is_some_and(|view| view.state == ProgramState::Active)
        {
            state.runtime.notify_front(program);
            return;
        }
        let Some(target_id) = state
            .decisions
            .get(program)
            .and_then(|decision| decision.last_target.clone())
        else {
            return;
        };
        self.commit_admission(
            state,
            program,
            &target_id,
            RankAdmissionPlan {
                used_tokens: 0.0,
                required_tokens: 0.0,
                reserve_tokens: 0.0,
                capacity_tokens: self.config.progress_ttl.token_capacity,
                forced: false,
                batch_gain: false,
            },
            Instant::now(),
        );
    }

    fn claim_dispatch(&self, handle: &ProgramRequestHandle) -> Option<ProgramDispatch> {
        let mut state = self.state.lock();
        let target = state.runtime.placement(handle.program())?.to_string();
        let dispatch = state.runtime.admit_front(handle, target, Instant::now())?;
        RouterMetrics::record_agent_aware_queue_wait(
            &dispatch.target_id,
            dispatch
                .dispatched_at()
                .saturating_duration_since(dispatch.request_arrived_at()),
        );
        if let Some(text) = dispatch.routing_text() {
            state
                .bindings
                .commit_placement(dispatch.program(), text, &dispatch.target_id);
        }
        debug!(
            event = "request_dispatch",
            program = %dispatch.redacted_program_id(),
            target = %dispatch.target_id,
            router_queue_seconds = dispatch
                .dispatched_at()
                .saturating_duration_since(dispatch.request_arrived_at())
                .as_secs_f64(),
            estimated_context_tokens = dispatch.estimated_context_tokens(),
            placement_epoch = dispatch.placement_epoch(),
            placement_start_request = dispatch.placement_start_request(),
            "Program scheduling diagnostic"
        );
        Some(dispatch)
    }

    pub(crate) fn cancel_retained_request(&self, handle: &ProgramRequestHandle) {
        let mut state = self.state.lock();
        let before = state.runtime.view(handle.program());
        let before_tokens = state
            .decisions
            .get(handle.program())
            .map(|decision| decision.private_tokens(self.config.progress_ttl.decode_buffer_tokens));
        if !state.runtime.cancel_request(handle) {
            return;
        }
        let after = state.runtime.view(handle.program());
        if let (Some(before), Some(after), Some(tokens), Some(target_id)) = (
            before,
            after,
            before_tokens,
            state
                .decisions
                .get(handle.program())
                .and_then(|decision| decision.last_target.clone()),
        ) {
            if before.state == ProgramState::Active && after.state == ProgramState::Paused {
                Self::adjust_usage(&mut state, &target_id, -tokens);
                if let Some(decision) = state.decisions.get_mut(handle.program()) {
                    let now = Instant::now();
                    decision.paused_at = Some(now);
                    decision.last_pause_reason = Some(ProgramPauseReason::RequestCancelled);
                    decision.rounds_since_activation = 0;
                    decision.pause_when_idle = false;
                    Self::restart_shared_prefix_freshness(decision, now);
                }
            }
        }
        let program = handle.program();
        let keep_queued = state
            .runtime
            .view(program)
            .is_some_and(|view| view.waiting_requests > 0);
        if !keep_queued {
            for queue in state.rank_queues.values_mut() {
                queue.retain(|queued| queued != program);
            }
            for queue in state.global_queues.values_mut() {
                queue.retain(|queued| queued != program);
            }
        }
        let remove_unstarted = state
            .runtime
            .view(program)
            .is_some_and(|view| view.in_flight_requests == 0 && view.waiting_requests == 0)
            && state
                .decisions
                .get(program)
                .is_some_and(|decision| decision.completed_requests == 0);
        if remove_unstarted {
            self.release_program(&mut state, program, Instant::now());
        }
        self.schedule_waiting(&mut state, Instant::now());
    }

    pub(crate) fn schedule_rank_local(
        &self,
        state: &mut ProgramSchedulerState,
        now: Instant,
    ) -> bool {
        let target_ids = state.rank_queues.keys().cloned().collect::<Vec<_>>();
        let mut changed = false;
        for target_id in target_ids {
            loop {
                let mut queued = state
                    .rank_queues
                    .get(&target_id)
                    .into_iter()
                    .flatten()
                    .filter(|program| {
                        state.runtime.view(program).is_some_and(|view| {
                            view.state == ProgramState::Paused
                                && view.status == ProgramStatus::Reasoning
                                && view.waiting_requests > 0
                        })
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                self.order_resume_candidates(state, &mut queued, now);
                let Some((candidate, plan)) = queued.into_iter().find_map(|candidate| {
                    self.rank_admission_plan(state, &candidate, &target_id, now)
                        .map(|plan| (candidate, plan))
                }) else {
                    break;
                };
                self.commit_admission(state, &candidate, &target_id, plan, now);
                changed = true;
            }
        }
        changed
    }

    pub(crate) fn schedule_waiting(&self, state: &mut ProgramSchedulerState, now: Instant) -> bool {
        if self.config.global_queue {
            self.schedule_global_waiting(state, now)
        } else {
            self.schedule_rank_local(state, now)
        }
    }

    pub(crate) fn local_resume_cmp(
        &self,
        state: &ProgramSchedulerState,
        left: &ProgramRef,
        right: &ProgramRef,
        now: Instant,
    ) -> Ordering {
        self.resume_tier_cmp(state, left, right, now)
            .then_with(|| self.segment_yield_cmp(state, left, right))
            .then_with(|| self.ordinary_resume_cmp(state, left, right))
    }

    /// Apply force-resume, request priority, one-shot yield, and local order.
    ///
    /// A Program paused after reaching the segment limit ranks behind peers in
    /// the same force/priority tier. Successful admission clears the pause
    /// reason, so this penalty applies to exactly one resume.
    pub(crate) fn order_resume_candidates(
        &self,
        state: &ProgramSchedulerState,
        queued: &mut [ProgramRef],
        now: Instant,
    ) {
        queued.sort_by(|left, right| {
            self.resume_tier_cmp(state, left, right, now)
                .then_with(|| self.segment_yield_cmp(state, left, right))
                .then_with(|| self.ordinary_resume_cmp(state, left, right))
        });
    }

    /// Keep cross-rank escape fair and predictable regardless of local order.
    pub(crate) fn order_cross_rank_candidates(
        &self,
        state: &ProgramSchedulerState,
        queued: &mut [ProgramRef],
        now: Instant,
    ) {
        queued.sort_by(|left, right| {
            self.resume_tier_cmp(state, left, right, now)
                .then_with(|| self.segment_yield_cmp(state, left, right))
                .then_with(|| self.fcfs_resume_cmp(state, left, right))
        });
    }

    fn resume_tier_cmp(
        &self,
        state: &ProgramSchedulerState,
        left: &ProgramRef,
        right: &ProgramRef,
        now: Instant,
    ) -> Ordering {
        let left_state = &state.decisions[left];
        let right_state = &state.decisions[right];
        let left_forced = left_state
            .force_resume_deadline
            .is_some_and(|deadline| deadline <= now);
        let right_forced = right_state
            .force_resume_deadline
            .is_some_and(|deadline| deadline <= now);
        let left_priority = state
            .runtime
            .front_request_hints(left)
            .map_or(0, |hints| hints.priority);
        let right_priority = state
            .runtime
            .front_request_hints(right)
            .map_or(0, |hints| hints.priority);
        right_forced
            .cmp(&left_forced)
            .then_with(|| right_priority.cmp(&left_priority))
    }

    fn segment_yield_cmp(
        &self,
        state: &ProgramSchedulerState,
        left: &ProgramRef,
        right: &ProgramRef,
    ) -> Ordering {
        let yielded = |program: &ProgramRef| {
            state.decisions[program].last_pause_reason == Some(ProgramPauseReason::MaxSegmentYield)
        };
        yielded(left).cmp(&yielded(right))
    }

    fn ordinary_resume_cmp(
        &self,
        state: &ProgramSchedulerState,
        left: &ProgramRef,
        right: &ProgramRef,
    ) -> Ordering {
        let left_state = &state.decisions[left];
        let right_state = &state.decisions[right];
        match self.config.resume_order {
            ProgramResumeOrder::Fcfs => self.fcfs_resume_cmp(state, left, right),
            ProgramResumeOrder::Mru => right_state
                .last_request_finished_at
                .cmp(&left_state.last_request_finished_at)
                .then_with(|| left_state.queued_at.cmp(&right_state.queued_at))
                .then_with(|| left.program_id().cmp(right.program_id())),
        }
    }

    fn fcfs_resume_cmp(
        &self,
        state: &ProgramSchedulerState,
        left: &ProgramRef,
        right: &ProgramRef,
    ) -> Ordering {
        state.decisions[left]
            .queued_at
            .cmp(&state.decisions[right].queued_at)
            .then_with(|| left.program_id().cmp(right.program_id()))
    }

    pub(crate) fn rank_admission_plan(
        &self,
        state: &ProgramSchedulerState,
        program: &ProgramRef,
        target_id: &str,
        now: Instant,
    ) -> Option<RankAdmissionPlan> {
        let runtime = state.runtime.view(program)?;
        let decision = state.decisions.get(program)?;
        if !self.config.global_queue && decision.last_target.as_deref() != Some(target_id) {
            return None;
        }
        if runtime.state != ProgramState::Paused
            || runtime.status != ProgramStatus::Reasoning
            || runtime.waiting_requests == 0
        {
            return None;
        }
        let forced = decision
            .force_resume_deadline
            .is_some_and(|deadline| deadline <= now);
        if self.config.admission_waiting_request_threshold > 0
            && state
                .observations
                .get(target_id)
                .filter(|observation| self.observation_is_fresh(observation, now))
                .and_then(|observation| observation.waiting_requests)
                .is_some_and(|waiting| waiting >= self.config.admission_waiting_request_threshold)
        {
            return None;
        }
        let active = state
            .runtime
            .views()
            .into_iter()
            .filter(|other| {
                other.state == ProgramState::Active && other.placement.as_deref() == Some(target_id)
            })
            .collect::<Vec<_>>();
        let used_tokens = self.target_usage(state, target_id, now);
        let active_count = active.len();
        let expected_output_tokens = state
            .runtime
            .front_request_hints(program)
            .and_then(|hints| hints.expected_output_tokens);
        let required_tokens = decision.private_tokens_with_output(
            self.config.progress_ttl.decode_buffer_tokens,
            expected_output_tokens,
        );
        let capacity = self.config.progress_ttl.token_capacity;
        let count_fits =
            |count: usize| count.saturating_add(1) <= self.config.max_active_programs_per_target;
        let Some(capacity_tokens) = capacity else {
            if !forced && !count_fits(active_count) {
                return None;
            }
            return Some(RankAdmissionPlan {
                used_tokens,
                required_tokens,
                reserve_tokens: 0.0,
                capacity_tokens: None,
                forced,
                batch_gain: false,
            });
        };
        let low = capacity_tokens as f64 * self.config.progress_ttl.low_watermark_ratio;
        let immediate_fits = used_tokens + required_tokens <= low;
        let factors = state.rank_factors.get(target_id);
        let reserve_tokens = factors.map_or(0.0, |factors| {
            let target_rounds = self.policy.target_rounds(factors);
            let active_remaining = active.iter().filter_map(|active| {
                state.decisions.get(&active.reference).map(|active_state| {
                    (target_rounds - active_state.rounds_since_activation as f64).max(0.0)
                })
            });
            self.policy
                .continuous_growth_reserve_tokens(factors, active_remaining)
        });
        let reserve_fits = used_tokens + required_tokens + reserve_tokens <= low;
        if forced || (count_fits(active_count) && immediate_fits && reserve_fits) {
            return Some(RankAdmissionPlan {
                used_tokens,
                required_tokens,
                reserve_tokens,
                capacity_tokens: capacity,
                forced,
                batch_gain: false,
            });
        }
        if !count_fits(active_count) || !immediate_fits {
            return None;
        }
        let factors = factors?;
        let target_rounds = self.policy.target_rounds(factors);
        let running = active
            .iter()
            .filter(|active| active.status == ProgramStatus::Reasoning)
            .collect::<Vec<_>>();
        let protected = active
            .iter()
            .filter_map(|active| {
                let state = state.decisions.get(&active.reference)?;
                let remaining = (target_rounds - state.rounds_since_activation as f64).max(0.0);
                (remaining > 0.0).then_some((remaining, state.last_cache_miss_impact_seconds))
            })
            .collect::<Vec<_>>();
        let protected_rounds = protected
            .iter()
            .map(|(rounds, _)| *rounds)
            .collect::<Vec<_>>();
        let protected_costs = protected.iter().map(|(_, cost)| *cost).collect::<Vec<_>>();
        let estimate = self.policy.estimate_batch_gain(BatchGainInputs {
            factors,
            batch_size_before: running.len(),
            total_context_tokens_before: running
                .iter()
                .filter_map(|active| {
                    state
                        .decisions
                        .get(&active.reference)
                        .map(|decision| decision.estimated_context_tokens)
                })
                .sum(),
            candidate_context_tokens: decision.estimated_context_tokens,
            used_tokens,
            required_tokens,
            protected_remaining_rounds: &protected_rounds,
            protected_recovery_cost_seconds: &protected_costs,
            candidate_recovery_cost_seconds: self.policy.cache_miss_impact_seconds(
                decision
                    .estimated_context_tokens
                    .saturating_sub(decision.shared_prefix_tokens),
            ),
        });
        estimate.admits.then_some(RankAdmissionPlan {
            used_tokens,
            required_tokens,
            reserve_tokens,
            capacity_tokens: capacity,
            forced,
            batch_gain: true,
        })
    }

    pub(crate) fn commit_admission(
        &self,
        state: &mut ProgramSchedulerState,
        program: &ProgramRef,
        target_id: &str,
        plan: RankAdmissionPlan,
        now: Instant,
    ) -> bool {
        let previous_target = state
            .decisions
            .get(program)
            .and_then(|decision| decision.last_target.clone());
        let Some(tokens) = state
            .decisions
            .get(program)
            .map(|decision| decision.private_tokens(self.config.progress_ttl.decode_buffer_tokens))
        else {
            return false;
        };
        if !state.runtime.activate(program, target_id.to_string()) {
            return false;
        }
        if let Some(decision) = state.decisions.get_mut(program) {
            decision.last_target = Some(target_id.to_string());
            decision.rounds_since_activation = 0;
            decision.acting_since = None;
            decision.ttl_deadline = None;
            decision.segment_started_at.get_or_insert(now);
            decision.paused_at = None;
            decision.last_pause_reason = None;
            decision.pause_when_idle = false;
            decision.queued_at = None;
            decision.force_resume_deadline = None;
            decision.force_resume_timeout = None;
            decision.force_resume_active_remaining_rounds = 0.0;
            decision.force_resume_pool_remaining_rounds = 0.0;
            decision.force_resume_request_throughput_per_second = 0.0;
        }
        Self::adjust_usage(state, target_id, tokens);
        for queue in state.rank_queues.values_mut() {
            queue.retain(|queued| queued != program);
        }
        for queue in state.global_queues.values_mut() {
            queue.retain(|queued| queued != program);
        }
        state.runtime.notify_front(program);
        info!(
            event = "program_admit",
            program = %program.redacted_id(),
            target = target_id,
            forced = plan.forced,
            batch_gain = plan.batch_gain,
            used_tokens = plan.used_tokens,
            required_tokens = plan.required_tokens,
            reserve_tokens = plan.reserve_tokens,
            capacity_tokens = ?plan.capacity_tokens,
            "Program scheduling decision"
        );
        if !self.config.binding_only {
            let observation = state.observations.get(target_id);
            RouterMetrics::record_agent_aware_admission(
                target_id,
                AgentAwareAdmissionMetrics {
                    used_tokens: plan.used_tokens,
                    required_tokens: plan.required_tokens,
                    reserve_tokens: plan.reserve_tokens,
                    capacity_tokens: plan.capacity_tokens,
                    backend_kv_usage_ratio: observation.and_then(|value| value.kv_cache_usage),
                    backend_running_requests: observation.and_then(|value| value.running_requests),
                    backend_waiting_requests: observation.and_then(|value| value.waiting_requests),
                    observation_age: observation
                        .and_then(|value| value.observed_at)
                        .map(|observed_at| now.saturating_duration_since(observed_at)),
                },
            );
        }
        let reason = if plan.forced {
            "force_resume"
        } else if plan.batch_gain {
            "batch_gain_admit"
        } else {
            "program_admit"
        };
        RouterMetrics::record_agent_aware_transition(target_id, reason);
        if previous_target
            .as_deref()
            .is_some_and(|previous| previous != target_id)
        {
            RouterMetrics::record_agent_aware_transition(target_id, "cross_rank_resume");
        }
        true
    }

    pub(crate) fn pause_idle(
        &self,
        state: &mut ProgramSchedulerState,
        program: &ProgramRef,
        reason: ProgramPauseReason,
        now: Instant,
    ) -> bool {
        let Some(target_id) = state.runtime.placement(program).map(str::to_string) else {
            return false;
        };
        let Some(tokens) = state
            .decisions
            .get(program)
            .map(|decision| decision.private_tokens(self.config.progress_ttl.decode_buffer_tokens))
        else {
            return false;
        };
        if !state.runtime.pause_idle(program) {
            return false;
        }
        if let Some(decision) = state.decisions.get_mut(program) {
            decision.paused_at = Some(now);
            decision.last_pause_reason = Some(reason);
            decision.rounds_since_activation = 0;
            decision.acting_since = None;
            decision.ttl_deadline = None;
            decision.pause_when_idle = false;
            Self::restart_shared_prefix_freshness(decision, now);
        }
        Self::adjust_usage(state, &target_id, -tokens);
        info!(
            event = "program_pause",
            program = %program.redacted_id(),
            target = target_id,
            reason = reason.as_str(),
            released_private_tokens = tokens,
            "Program scheduling decision"
        );
        RouterMetrics::record_agent_aware_transition(&target_id, reason.as_str());
        true
    }

    pub(crate) fn restart_shared_prefix_freshness(
        decision: &mut super::scheduler_state::ProgramDecisionState,
        pause_at: Instant,
    ) {
        let duration = decision.shared_prefix_fresh_until.map(|deadline| {
            deadline.saturating_duration_since(decision.shared_prefix_freshness_anchor_at)
        });
        decision.shared_prefix_freshness_anchor_at = pause_at;
        if let Some(duration) = duration {
            decision.shared_prefix_fresh_until = Some(pause_at + duration);
        }
    }
}

struct AdmissionGuard<'a> {
    scheduler: &'a ProgramScheduler,
    handle: Option<ProgramRequestHandle>,
}

impl<'a> AdmissionGuard<'a> {
    fn new(scheduler: &'a ProgramScheduler, handle: ProgramRequestHandle) -> Self {
        Self {
            scheduler,
            handle: Some(handle),
        }
    }

    fn handle(&self) -> &ProgramRequestHandle {
        self.handle
            .as_ref()
            .expect("admission guard must remain armed while waiting")
    }

    fn disarm(&mut self) {
        self.handle = None;
    }
}

impl Drop for AdmissionGuard<'_> {
    fn drop(&mut self) {
        if let Some(handle) = &self.handle {
            self.scheduler.cancel_retained_request(handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program_scheduling::scheduler_state::RankObservationState;
    use crate::program_scheduling::ProgramSchedulerConfig;
    use serde_json::json;
    use std::time::Duration;

    fn identity(program: &str) -> ProgramIdentity {
        ProgramIdentity::from_request(
            None,
            Some(&json!({"vllm_xargs":{"agentic_context":{
                "program_id":program,"task_id":null,"expected_resume":true
            }}})),
            Some("model"),
        )
        .unwrap()
        .unwrap()
    }

    fn identity_with_priority(program: &str, priority: i64) -> ProgramIdentity {
        ProgramIdentity::from_request(
            None,
            Some(&json!({"vllm_xargs":{"agentic_context":{
                "program_id":program,
                "task_id":null,
                "expected_resume":true,
                "priority":priority
            }}})),
            Some("model"),
        )
        .unwrap()
        .unwrap()
    }

    fn identity_with_expected_output(program: &str, output_tokens: usize) -> ProgramIdentity {
        ProgramIdentity::from_request(
            None,
            Some(&json!({"vllm_xargs":{"agentic_context":{
                "program_id":program,
                "task_id":null,
                "expected_resume":true,
                "expected_output_length":output_tokens
            }}})),
            Some("model"),
        )
        .unwrap()
        .unwrap()
    }

    fn target() -> ProgramTarget {
        ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://rank-0".into(),
            dp_rank: Some(0),
        }
    }

    #[tokio::test]
    async fn first_program_is_admitted_and_dispatched() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let dispatch = scheduler
            .acquire(identity("p"), 100, &[target()], None)
            .await
            .unwrap();
        assert_eq!(dispatch.target_id, "rank-0");
        assert!(dispatch.placement_start_request());
    }

    #[tokio::test]
    async fn admission_does_not_reclaim_active_acting_programs() {
        let config = ProgramSchedulerConfig {
            max_active_programs_per_target: 1,
            ..ProgramSchedulerConfig::default()
        };
        let scheduler = ProgramScheduler::new(config);
        let target = target();
        let first = scheduler
            .acquire(identity("first"), 100, std::slice::from_ref(&target), None)
            .await
            .unwrap();
        {
            let mut state = scheduler.state.lock();
            assert!(state.runtime.complete_request(&first, Some(100).into()));
        }
        let second = tokio::time::timeout(
            std::time::Duration::from_millis(10),
            scheduler.acquire(identity("second"), 100, &[target], None),
        )
        .await;
        assert!(second.is_err());
        let state = scheduler.state.lock();
        assert_eq!(
            state.runtime.state(first.program()),
            Some((ProgramState::Active, ProgramStatus::Acting))
        );
    }

    #[tokio::test]
    async fn repeated_program_arrival_populates_continuity_window() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let target = target();
        let first = scheduler
            .acquire(
                identity("program"),
                100,
                std::slice::from_ref(&target),
                None,
            )
            .await
            .unwrap();
        scheduler.complete(&first, true, false, Some(100), Some(0.1));
        let second = scheduler
            .acquire(identity("program"), 110, &[target], None)
            .await
            .unwrap();
        let state = scheduler.state.lock();
        assert_eq!(state.rank_factors["rank-0"].continuity_sample_count(), 1);
        drop(state);
        scheduler.complete(&second, true, true, Some(110), Some(0.1));
    }

    #[test]
    fn local_order_prefers_forced_then_non_yielded_then_mru() {
        let config = ProgramSchedulerConfig {
            resume_order: ProgramResumeOrder::Mru,
            ..ProgramSchedulerConfig::default()
        };
        let scheduler = ProgramScheduler::new(config);
        let target = target();
        scheduler.sync_targets("model", std::slice::from_ref(&target));
        let now = Instant::now();
        let mut state = scheduler.state.lock();
        let mut refs = Vec::new();
        for name in ["old", "new", "forced"] {
            let identity = identity(name);
            let handle = state.runtime.retain_request(&identity, 100, None, now);
            let reference = handle.program().clone();
            scheduler.ensure_decision_state(
                &mut state,
                &identity,
                reference.clone(),
                100,
                now,
                None,
            );
            state
                .rank_queues
                .entry("rank-0".into())
                .or_default()
                .push_back(reference.clone());
            refs.push(reference);
        }
        state
            .decisions
            .get_mut(&refs[0])
            .unwrap()
            .last_request_finished_at = Some(now - Duration::from_secs(10));
        state.decisions.get_mut(&refs[0]).unwrap().last_pause_reason =
            Some(ProgramPauseReason::MaxSegmentYield);
        state
            .decisions
            .get_mut(&refs[1])
            .unwrap()
            .last_request_finished_at = Some(now - Duration::from_secs(1));
        state
            .decisions
            .get_mut(&refs[2])
            .unwrap()
            .force_resume_deadline = Some(now);
        let mut ordered = refs.clone();
        ordered.sort_by(|left, right| scheduler.local_resume_cmp(&state, left, right, now));
        assert_eq!(ordered[0], refs[2]);
        assert_eq!(ordered[1], refs[1]);
        assert_eq!(ordered[2], refs[0]);
    }

    #[test]
    fn local_order_defaults_to_mru_after_priority_tiers() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let target = target();
        scheduler.sync_targets("model", std::slice::from_ref(&target));
        let now = Instant::now();
        let mut state = scheduler.state.lock();
        let mut refs = Vec::new();
        for (index, name) in ["first", "second"].into_iter().enumerate() {
            let identity = identity(name);
            let handle = state.runtime.retain_request(&identity, 100, None, now);
            let reference = handle.program().clone();
            scheduler.ensure_decision_state(
                &mut state,
                &identity,
                reference.clone(),
                100,
                now,
                None,
            );
            let decision = state.decisions.get_mut(&reference).unwrap();
            decision.queued_at = Some(now + Duration::from_secs(index as u64));
            decision.last_request_finished_at = Some(now - Duration::from_secs((2 - index) as u64));
            refs.push(reference);
        }
        let mut ordered = refs.clone();
        ordered.sort_by(|left, right| scheduler.local_resume_cmp(&state, left, right, now));
        assert_eq!(ordered, vec![refs[1].clone(), refs[0].clone()]);
    }

    #[test]
    fn resume_order_applies_priority_before_one_shot_yield() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        scheduler.sync_targets("model", &[target()]);
        let now = Instant::now();
        let mut state = scheduler.state.lock();
        let specifications = [
            ("normal", 0, false),
            ("yielded", 0, true),
            ("priority", 9, true),
        ];
        let mut queued = Vec::new();
        for (name, priority, yielded) in specifications {
            let identity = identity_with_priority(name, priority);
            let handle = state.runtime.retain_request(&identity, 100, None, now);
            let reference = handle.program().clone();
            scheduler.ensure_decision_state(
                &mut state,
                &identity,
                reference.clone(),
                100,
                now,
                None,
            );
            let decision = state.decisions.get_mut(&reference).unwrap();
            decision.last_request_finished_at = Some(now - Duration::from_secs(1));
            if yielded {
                decision.last_pause_reason = Some(ProgramPauseReason::MaxSegmentYield);
            }
            queued.push(reference);
        }
        scheduler.order_resume_candidates(&state, &mut queued, now);
        assert_eq!(queued[0].program_id(), "priority");
        assert_eq!(queued[1].program_id(), "normal");
        assert_eq!(queued[2].program_id(), "yielded");
    }

    #[test]
    fn successful_resume_clears_max_segment_yield_marker() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        scheduler.sync_targets("model", &[target()]);
        let now = Instant::now();
        let identity = identity("yielded");
        let mut state = scheduler.state.lock();
        let handle = state.runtime.retain_request(&identity, 100, None, now);
        scheduler.ensure_decision_state(
            &mut state,
            &identity,
            handle.program().clone(),
            100,
            now,
            None,
        );
        state
            .decisions
            .get_mut(handle.program())
            .unwrap()
            .last_pause_reason = Some(ProgramPauseReason::MaxSegmentYield);
        let plan = scheduler
            .rank_admission_plan(&state, handle.program(), "rank-0", now)
            .unwrap();
        assert!(scheduler.commit_admission(&mut state, handle.program(), "rank-0", plan, now));
        assert_eq!(state.decisions[handle.program()].last_pause_reason, None);
    }

    #[test]
    fn expected_output_length_replaces_default_request_capacity() {
        let mut config = ProgramSchedulerConfig::default();
        config.progress_ttl.token_capacity = Some(1_000);
        config.progress_ttl.low_watermark_ratio = 1.0;
        let scheduler = ProgramScheduler::new(config);
        scheduler.sync_targets("model", &[target()]);
        let now = Instant::now();
        let identity = identity_with_expected_output("program", 200);
        let mut state = scheduler.state.lock();
        let handle = state.runtime.retain_request(&identity, 100, None, now);
        scheduler.ensure_decision_state(
            &mut state,
            &identity,
            handle.program().clone(),
            100,
            now,
            None,
        );
        let plan = scheduler
            .rank_admission_plan(&state, handle.program(), "rank-0", now)
            .unwrap();
        assert_eq!(plan.required_tokens, 300.0);
    }

    #[test]
    fn cancelling_unstarted_request_removes_transient_program() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let target = target();
        scheduler.sync_targets("model", std::slice::from_ref(&target));
        let identity = identity("cancelled");
        let now = Instant::now();
        let handle = {
            let mut state = scheduler.state.lock();
            let handle = state.runtime.retain_request(&identity, 100, None, now);
            scheduler.ensure_decision_state(
                &mut state,
                &identity,
                handle.program().clone(),
                100,
                now,
                None,
            );
            state
                .rank_queues
                .entry("rank-0".into())
                .or_default()
                .push_back(handle.program().clone());
            handle
        };
        scheduler.cancel_retained_request(&handle);
        let state = scheduler.state.lock();
        assert!(state.runtime.view(handle.program()).is_none());
        assert!(!state.decisions.contains_key(handle.program()));
    }

    #[test]
    fn backend_waiting_gate_also_blocks_forced_resume() {
        let config = ProgramSchedulerConfig {
            admission_waiting_request_threshold: 1,
            ..ProgramSchedulerConfig::default()
        };
        let scheduler = ProgramScheduler::new(config);
        let target = target();
        scheduler.sync_targets("model", std::slice::from_ref(&target));
        let now = Instant::now();
        let identity = identity("forced");
        let mut state = scheduler.state.lock();
        let handle = state.runtime.retain_request(&identity, 100, None, now);
        scheduler.ensure_decision_state(
            &mut state,
            &identity,
            handle.program().clone(),
            100,
            now,
            None,
        );
        state
            .decisions
            .get_mut(handle.program())
            .unwrap()
            .force_resume_deadline = Some(now);
        state.observations.insert(
            "rank-0".into(),
            RankObservationState {
                waiting_requests: Some(1),
                observed_at: Some(now),
                ..RankObservationState::default()
            },
        );
        assert!(scheduler
            .rank_admission_plan(&state, handle.program(), "rank-0", now)
            .is_none());
    }

    #[tokio::test]
    async fn marked_program_continuation_does_not_double_count_growth() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let target = target();
        let first = scheduler
            .acquire(identity("marked"), 100, std::slice::from_ref(&target), None)
            .await
            .unwrap();
        {
            let mut state = scheduler.state.lock();
            state
                .decisions
                .get_mut(first.program())
                .unwrap()
                .pause_when_idle = true;
            state
                .observations
                .get_mut("rank-0")
                .unwrap()
                .active_program_token_delta = 100.0;
        }
        let second = scheduler
            .acquire(identity("marked"), 200, &[target], None)
            .await
            .unwrap();
        let state = scheduler.state.lock();
        assert_eq!(
            state.observations["rank-0"].active_program_token_delta,
            100.0
        );
        drop(state);
        scheduler.complete(&first, true, false, Some(100), Some(0.1));
        scheduler.complete(&second, true, true, Some(200), Some(0.1));
    }
}
