//! Dynamic force-resume deadlines and segment-yield fairness.
//!
//! These mechanisms bound starvation while ordinary local resume remains
//! configurable as FCFS or MRU.

use super::scheduler::ProgramScheduler;
use super::scheduler_state::{ProgramPauseReason, ProgramSchedulerState};
use super::{ProgramRef, ProgramState, ProgramStatus};
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
        let timeout = state
            .runtime
            .front_request_hints(candidate)
            .and_then(|hints| hints.deadline)
            .unwrap_or(estimate.timeout)
            .min(self.config.queue_timeout);
        if let Some(decision) = state.decisions.get_mut(candidate) {
            decision.force_resume_deadline = Some(now + timeout);
            decision.force_resume_timeout = Some(timeout);
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
            .iter_views()
            .filter(|program| program.state == ProgramState::Active)
            .filter_map(|program| {
                let rounds = target_rounds.get(program.placement?)?;
                let decision = state.decisions.get(program.reference)?;
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
        self.order_resume_candidates(state, &mut queued, now);
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
            .iter_views()
            .filter(|program| {
                program.state == ProgramState::Active && program.placement == Some(target_id)
            })
            .filter_map(|program| {
                state.decisions.get(program.reference).map(|decision| {
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
        self.order_resume_candidates(state, &mut queued, now);
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
            self.order_resume_candidates(state, &mut queued, now);
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
            self.order_resume_candidates(state, &mut queued, now);
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
            .iter_views()
            .filter(|runtime| {
                runtime.state == ProgramState::Active
                    && runtime.status == ProgramStatus::Reasoning
                    && runtime.placement == Some(target_id.as_str())
            })
            .filter_map(|runtime| {
                state
                    .decisions
                    .get(runtime.reference)
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
