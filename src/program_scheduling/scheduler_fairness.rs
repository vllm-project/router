//! Progress privilege, dynamic force-resume deadline, and segment yielding.
//!
//! These mechanisms bound starvation while ordinary local resume remains MRU
//! to favor cache-resident Programs.

use super::scheduler::ProgramScheduler;
use super::scheduler_state::{ProgramPauseReason, ProgramSchedulerState};
use super::{ProgramRef, ProgramState, ProgramStatus};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

struct ForceResumeEstimate {
    timeout: Duration,
    active_remaining_rounds: f64,
    pool_remaining_rounds: f64,
    request_throughput_per_second: f64,
}

impl ForceResumeEstimate {
    fn fallback(timeout: Duration) -> Self {
        Self {
            timeout,
            active_remaining_rounds: 0.0,
            pool_remaining_rounds: 0.0,
            request_throughput_per_second: 0.0,
        }
    }
}

impl ProgramScheduler {
    /// Freeze the current work-ahead estimate when a Program enters a rank queue.
    pub(crate) fn set_force_resume_deadline(
        &self,
        state: &mut ProgramSchedulerState,
        target_id: &str,
        candidate: &ProgramRef,
        now: Instant,
    ) {
        let estimate = if self.config.global_queue {
            self.global_force_resume_timeout(state, candidate, now)
        } else {
            self.force_resume_timeout(state, target_id, candidate, now)
        };
        if let Some(decision) = state.decisions.get_mut(candidate) {
            decision.force_resume_deadline = Some(now + estimate.timeout);
            decision.force_resume_timeout = Some(estimate.timeout);
            decision.force_resume_active_remaining_rounds = estimate.active_remaining_rounds;
            decision.force_resume_pool_remaining_rounds = estimate.pool_remaining_rounds;
            decision.force_resume_request_throughput_per_second =
                estimate.request_throughput_per_second;
        }
    }

    fn global_force_resume_timeout(
        &self,
        state: &ProgramSchedulerState,
        candidate: &ProgramRef,
        now: Instant,
    ) -> ForceResumeEstimate {
        let Some(targets) = state.model_targets.get(candidate.model_pool()) else {
            return ForceResumeEstimate::fallback(self.config.force_resume_timeout);
        };
        if targets.is_empty() {
            return ForceResumeEstimate::fallback(self.config.force_resume_timeout);
        }
        let mut target_rounds = std::collections::HashMap::new();
        let mut aggregate_throughput = 0.0;
        for target in targets {
            let Some(factors) = state.rank_factors.get(target) else {
                return ForceResumeEstimate::fallback(self.config.force_resume_timeout);
            };
            let rounds = self.policy.target_rounds(factors);
            let throughput = factors.request_throughput_per_second();
            if rounds <= 0.0 || throughput <= 0.0 {
                return ForceResumeEstimate::fallback(self.config.force_resume_timeout);
            }
            target_rounds.insert(target.as_str(), rounds);
            aggregate_throughput += throughput;
        }
        if aggregate_throughput <= 0.0 {
            return ForceResumeEstimate::fallback(self.config.force_resume_timeout);
        }
        let active_remaining = state
            .runtime
            .views()
            .into_iter()
            .filter(|program| program.state == ProgramState::Active)
            .filter_map(|program| {
                let rounds = target_rounds.get(program.placement.as_deref()?)?;
                let decision = state.decisions.get(&program.reference)?;
                let floor = usize::from(program.status == ProgramStatus::Reasoning) as f64;
                Some((*rounds - decision.rounds_since_activation as f64).max(floor))
            })
            .sum::<f64>();
        let fallback_rounds =
            target_rounds.values().sum::<f64>() / target_rounds.len().max(1) as f64;
        let mut queued = state
            .global_queues
            .get(candidate.model_pool())
            .into_iter()
            .flatten()
            .filter(|program| {
                state.runtime.view(program).is_some_and(|runtime| {
                    runtime.state == ProgramState::Paused
                        && runtime.status == ProgramStatus::Reasoning
                        && runtime.waiting_requests > 0
                })
            })
            .cloned()
            .collect::<Vec<_>>();
        queued.sort_by(|left, right| self.local_resume_cmp(state, left, right, now));
        let pool_remaining = queued
            .into_iter()
            .take_while(|program| program != candidate)
            .filter_map(|program| state.decisions.get(&program))
            .map(|decision| {
                decision
                    .last_target
                    .as_deref()
                    .and_then(|target| target_rounds.get(target))
                    .copied()
                    .unwrap_or(fallback_rounds)
                    .max(1.0)
            })
            .sum::<f64>();
        let maximum = self.config.force_resume_timeout.as_secs_f64();
        let minimum = 30.0_f64.min(maximum);
        ForceResumeEstimate {
            timeout: Duration::from_secs_f64(
                (3.0 * (active_remaining + pool_remaining) / aggregate_throughput)
                    .clamp(minimum, maximum),
            ),
            active_remaining_rounds: active_remaining,
            pool_remaining_rounds: pool_remaining,
            request_throughput_per_second: aggregate_throughput,
        }
    }

    fn force_resume_timeout(
        &self,
        state: &ProgramSchedulerState,
        target_id: &str,
        candidate: &ProgramRef,
        now: Instant,
    ) -> ForceResumeEstimate {
        let Some(factors) = state.rank_factors.get(target_id) else {
            return ForceResumeEstimate::fallback(self.config.force_resume_timeout);
        };
        let throughput = factors.request_throughput_per_second();
        let target_rounds = self.policy.target_rounds(factors);
        if target_rounds <= 0.0 || throughput <= 0.0 {
            return ForceResumeEstimate::fallback(self.config.force_resume_timeout);
        }
        let active_remaining = state
            .runtime
            .views()
            .into_iter()
            .filter(|program| {
                program.state == ProgramState::Active
                    && program.placement.as_deref() == Some(target_id)
            })
            .filter_map(|program| {
                state.decisions.get(&program.reference).map(|decision| {
                    let floor = usize::from(program.status == ProgramStatus::Reasoning) as f64;
                    (target_rounds - decision.rounds_since_activation as f64).max(floor)
                })
            })
            .sum::<f64>();
        let mut queued = state
            .rank_queues
            .get(target_id)
            .into_iter()
            .flatten()
            .filter(|program| {
                state.runtime.view(program).is_some_and(|runtime| {
                    runtime.state == ProgramState::Paused
                        && runtime.status == ProgramStatus::Reasoning
                        && runtime.waiting_requests > 0
                }) && state
                    .decisions
                    .get(program)
                    .and_then(|decision| decision.last_target.as_deref())
                    == Some(target_id)
            })
            .cloned()
            .collect::<Vec<_>>();
        queued.sort_by(|left, right| self.local_resume_cmp(state, left, right, now));
        let pool_remaining = queued
            .into_iter()
            .take_while(|program| program != candidate)
            .count() as f64
            * target_rounds.max(1.0);
        let maximum = self.config.force_resume_timeout.as_secs_f64();
        let minimum = 30.0_f64.min(maximum);
        ForceResumeEstimate {
            timeout: Duration::from_secs_f64(
                (3.0 * (active_remaining + pool_remaining) / throughput).clamp(minimum, maximum),
            ),
            active_remaining_rounds: active_remaining,
            pool_remaining_rounds: pool_remaining,
            request_throughput_per_second: throughput,
        }
    }

    pub(crate) fn reconcile_privileges(
        &self,
        state: &mut ProgramSchedulerState,
        now: Instant,
    ) -> bool {
        let mut changed = false;
        for decision in state.decisions.values_mut() {
            if decision
                .privilege_deadline
                .is_some_and(|deadline| deadline <= now)
            {
                decision.privilege_deadline = None;
                decision.privilege_ttl_expired = true;
                changed = true;
            }
        }
        let target_ids = state.targets.keys().cloned().collect::<Vec<_>>();
        for target_id in target_ids {
            let slots = self
                .config
                .progress_ttl
                .token_capacity
                .map(|capacity| capacity / self.config.privileged_max_context_tokens)
                .unwrap_or(1);
            let limit = (slots / 2).max(1);
            let mut privileged = state
                .decisions
                .iter()
                .filter(|(program, decision)| {
                    decision.last_target.as_deref() == Some(target_id.as_str())
                        && decision
                            .privilege_deadline
                            .is_some_and(|deadline| deadline > now)
                        && state.runtime.view(program).is_some()
                })
                .map(|(program, _)| program.clone())
                .collect::<Vec<_>>();
            privileged.sort_by(|left, right| {
                state.decisions[right]
                    .lifetime_generated_tokens
                    .cmp(&state.decisions[left].lifetime_generated_tokens)
                    .then_with(|| {
                        state.decisions[right]
                            .completed_requests
                            .cmp(&state.decisions[left].completed_requests)
                    })
                    .then_with(|| left.program_id().cmp(right.program_id()))
            });
            let mut retained_tasks = BTreeSet::new();
            let mut retained = 0;
            for program in privileged {
                let task = state.decisions[&program].placement_key.clone();
                if retained >= limit || !retained_tasks.insert(task) {
                    state
                        .decisions
                        .get_mut(&program)
                        .unwrap()
                        .privilege_deadline = None;
                    changed = true;
                } else {
                    retained += 1;
                }
            }
            if retained >= limit || self.config.privileged_ttl.is_zero() {
                continue;
            }
            let mut candidates = state
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
                                decision.privilege_deadline.is_none()
                                    && !decision.privilege_ttl_expired
                                    && decision.completed_requests > 0
                                    && !retained_tasks.contains(&decision.placement_key)
                            })
                })
                .map(|program| program.reference)
                .collect::<Vec<_>>();
            candidates.sort_by(|left, right| {
                state.decisions[right]
                    .lifetime_generated_tokens
                    .cmp(&state.decisions[left].lifetime_generated_tokens)
                    .then_with(|| {
                        state.decisions[right]
                            .completed_requests
                            .cmp(&state.decisions[left].completed_requests)
                    })
                    .then_with(|| {
                        state.decisions[right]
                            .private_tokens(self.config.progress_ttl.decode_buffer_tokens)
                            .total_cmp(
                                &state.decisions[left]
                                    .private_tokens(self.config.progress_ttl.decode_buffer_tokens),
                            )
                    })
                    .then_with(|| left.program_id().cmp(right.program_id()))
            });
            for program in candidates.into_iter().take(limit - retained) {
                let task = state.decisions[&program].placement_key.clone();
                if retained_tasks.insert(task) {
                    state
                        .decisions
                        .get_mut(&program)
                        .unwrap()
                        .privilege_deadline = Some(now + self.config.privileged_ttl);
                    changed = true;
                }
            }
        }
        changed
    }

    /// Yield an idle Program at the soft segment limit only for a local waiter.
    pub(crate) fn yield_completed_segment(
        &self,
        state: &mut ProgramSchedulerState,
        program: &ProgramRef,
        now: Instant,
    ) -> bool {
        let Some(runtime) = state.runtime.view(program) else {
            return false;
        };
        let Some(decision) = state.decisions.get(program) else {
            return false;
        };
        if runtime.state != ProgramState::Active
            || runtime.status != ProgramStatus::Acting
            || decision.rounds_since_activation < self.config.progress_ttl.max_segment_rounds
            || decision
                .privilege_deadline
                .is_some_and(|deadline| deadline > now)
        {
            return false;
        }
        let Some(target_id) = runtime.placement else {
            return false;
        };
        let mut queued = if self.config.global_queue {
            let mut queued = state
                .global_queues
                .get(program.model_pool())
                .into_iter()
                .flatten()
                .filter(|waiter| {
                    state.runtime.view(waiter).is_some_and(|runtime| {
                        runtime.state == ProgramState::Paused
                            && runtime.status == ProgramStatus::Reasoning
                            && runtime.waiting_requests > 0
                    })
                })
                .cloned()
                .collect::<Vec<_>>();
            queued.sort_by(|left, right| self.local_resume_cmp(state, left, right, now));
            queued.retain(|waiter| {
                state.decisions.get(waiter).is_some_and(|candidate| {
                    candidate.last_target.as_deref() == Some(target_id.as_str())
                })
            });
            queued
        } else {
            let mut queued = state
                .rank_queues
                .get(&target_id)
                .into_iter()
                .flatten()
                .filter(|waiter| {
                    state.runtime.view(waiter).is_some_and(|runtime| {
                        runtime.state == ProgramState::Paused
                            && runtime.status == ProgramStatus::Reasoning
                            && runtime.waiting_requests > 0
                    })
                })
                .cloned()
                .collect::<Vec<_>>();
            queued.sort_by(|left, right| self.local_resume_cmp(state, left, right, now));
            queued
        };
        let Some(waiter) = queued.drain(..).find(|waiter| waiter != program) else {
            return false;
        };
        let waiter_required = state.decisions.get(&waiter).map_or(0.0, |decision| {
            decision.private_tokens(self.config.progress_ttl.decode_buffer_tokens)
        });
        let future_relief = state
            .runtime
            .views()
            .into_iter()
            .filter(|runtime| {
                runtime.state == ProgramState::Active
                    && runtime.status == ProgramStatus::Reasoning
                    && runtime.placement.as_deref() == Some(target_id.as_str())
            })
            .filter_map(|runtime| {
                state
                    .decisions
                    .get(&runtime.reference)
                    .filter(|decision| decision.pause_when_idle)
            })
            .map(|decision| decision.private_tokens(self.config.progress_ttl.decode_buffer_tokens))
            .sum::<f64>();
        waiter_required > future_relief
            && self.pause_idle(state, program, ProgramPauseReason::MaxSegmentYield, now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program_scheduling::{ProgramIdentity, ProgramSchedulerConfig, ProgramTarget};
    use serde_json::json;

    #[test]
    fn force_deadline_uses_static_fallback_before_window_matures() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://rank-0".into(),
            dp_rank: Some(0),
        };
        let identity = ProgramIdentity::from_request(
            None,
            Some(&json!({"vllm_xargs":{"agentic_context":{
                "program_id":"p","task_id":"task","expected_resume":true
            }}})),
            Some("model"),
        )
        .unwrap()
        .unwrap();
        scheduler.sync_targets("model", &[target]);
        let now = Instant::now();
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
        scheduler.set_force_resume_deadline(&mut state, "rank-0", handle.program(), now);
        assert_eq!(
            state.decisions[handle.program()].force_resume_deadline,
            Some(now + scheduler.config.force_resume_timeout)
        );
    }
}
