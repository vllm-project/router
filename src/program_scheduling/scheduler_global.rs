//! Optional model-wide RequestPool scheduling and cross-rank resume placement.
//!
//! Every cycle exhausts feasible cache-affine local resumes before considering
//! relocation. Initial Program binding is never rerun by this module.

use super::scheduler::ProgramScheduler;
use super::scheduler_admission::RankAdmissionPlan;
use super::scheduler_state::ProgramSchedulerState;
use super::{ProgramRef, ProgramState, ProgramStatus};
use std::time::Instant;

impl ProgramScheduler {
    pub(crate) fn schedule_global_waiting(
        &self,
        state: &mut ProgramSchedulerState,
        now: Instant,
    ) -> bool {
        let model_pools = state.global_queues.keys().cloned().collect::<Vec<_>>();
        let mut changed = false;
        for model_pool in model_pools {
            loop {
                let queued = self.global_local_resume_order(state, &model_pool, now);
                let Some((program, target_id, plan)) = queued.into_iter().find_map(|program| {
                    let target_id = state.decisions.get(&program)?.last_target.clone()?;
                    self.rank_admission_plan(state, &program, &target_id, now)
                        .map(|plan| (program, target_id, plan))
                }) else {
                    break;
                };
                self.commit_admission(state, &program, &target_id, plan, now);
                changed = true;
            }

            loop {
                let queued = self.global_cross_rank_order(state, &model_pool, now);
                let Some((program, target_id, plan)) = queued.into_iter().find_map(|program| {
                    self.select_cross_rank_admission(state, &program, now)
                        .map(|(target, plan)| (program, target, plan))
                }) else {
                    break;
                };
                self.commit_admission(state, &program, &target_id, plan, now);
                changed = true;
            }
        }
        changed
    }

    fn global_local_resume_order(
        &self,
        state: &ProgramSchedulerState,
        model_pool: &str,
        now: Instant,
    ) -> Vec<ProgramRef> {
        let mut queued = self.global_reasoning_waiters(state, model_pool);
        queued.sort_by(|left, right| self.local_resume_cmp(state, left, right, now));
        queued
    }

    /// Cross-rank escape uses inverse ordinary MRU while retaining force and privilege tiers.
    fn global_cross_rank_order(
        &self,
        state: &ProgramSchedulerState,
        model_pool: &str,
        now: Instant,
    ) -> Vec<ProgramRef> {
        let mut queued = self
            .global_reasoning_waiters(state, model_pool)
            .into_iter()
            .filter(|program| {
                state
                    .decisions
                    .get(program)
                    .is_some_and(|decision| decision.last_pause_reason.is_some())
            })
            .collect::<Vec<_>>();
        queued.sort_by(|left, right| {
            let left_state = &state.decisions[left];
            let right_state = &state.decisions[right];
            let left_forced = left_state
                .force_resume_deadline
                .is_some_and(|deadline| deadline <= now);
            let right_forced = right_state
                .force_resume_deadline
                .is_some_and(|deadline| deadline <= now);
            let left_privileged = left_state
                .privilege_deadline
                .is_some_and(|deadline| deadline > now);
            let right_privileged = right_state
                .privilege_deadline
                .is_some_and(|deadline| deadline > now);
            right_forced
                .cmp(&left_forced)
                .then_with(|| right_privileged.cmp(&left_privileged))
                .then_with(|| {
                    left_state
                        .last_request_finished_at
                        .cmp(&right_state.last_request_finished_at)
                })
                .then_with(|| left_state.queued_at.cmp(&right_state.queued_at))
                .then_with(|| left.program_id().cmp(right.program_id()))
        });
        queued
    }

    fn global_reasoning_waiters(
        &self,
        state: &ProgramSchedulerState,
        model_pool: &str,
    ) -> Vec<ProgramRef> {
        state
            .global_queues
            .get(model_pool)
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
            .collect()
    }

    fn select_cross_rank_admission(
        &self,
        state: &ProgramSchedulerState,
        program: &ProgramRef,
        now: Instant,
    ) -> Option<(String, RankAdmissionPlan)> {
        let decision = state.decisions.get(program)?;
        if decision.last_pause_reason.is_none() {
            return None;
        }
        let source_target = decision.last_target.as_deref()?;
        let targets = state.model_targets.get(program.model_pool())?;
        let candidates = targets
            .iter()
            .filter(|target| target.as_str() != source_target)
            .filter(|target| !self.destination_has_local_waiter(state, program, target))
            .filter(|target| self.cross_rank_candidate_is_mature(state, program, target, now))
            .filter(|_| self.source_has_resume_pressure(state, program, now))
            .filter(|target| self.destination_has_context_headroom(state, program, target, now))
            .filter_map(|target| {
                self.rank_admission_plan(state, program, target, now)
                    .map(|plan| (target.clone(), plan))
            })
            .collect::<Vec<_>>();
        candidates
            .into_iter()
            .min_by(|(left_id, left), (right_id, right)| {
                let waiting = |target_id: &str| {
                    state
                        .observations
                        .get(target_id)
                        .filter(|observation| self.observation_is_fresh(observation, now))
                        .and_then(|observation| observation.waiting_requests)
                        .unwrap_or(0)
                };
                let pressure = |plan: &RankAdmissionPlan| {
                    plan.capacity_tokens.map_or(
                        plan.used_tokens + plan.required_tokens + plan.reserve_tokens,
                        |capacity| {
                            (plan.used_tokens + plan.required_tokens + plan.reserve_tokens)
                                / capacity.max(1) as f64
                        },
                    )
                };
                left.victims
                    .len()
                    .cmp(&right.victims.len())
                    .then_with(|| waiting(left_id).cmp(&waiting(right_id)))
                    .then_with(|| pressure(left).total_cmp(&pressure(right)))
                    .then_with(|| {
                        self.active_program_count(state, left_id)
                            .cmp(&self.active_program_count(state, right_id))
                    })
                    .then_with(|| left_id.cmp(right_id))
            })
    }

    fn destination_has_local_waiter(
        &self,
        state: &ProgramSchedulerState,
        candidate: &ProgramRef,
        destination: &str,
    ) -> bool {
        state
            .global_queues
            .get(candidate.model_pool())
            .into_iter()
            .flatten()
            .any(|program| {
                program != candidate
                    && state.runtime.view(program).is_some_and(|runtime| {
                        runtime.state == ProgramState::Paused
                            && runtime.status == ProgramStatus::Reasoning
                            && runtime.waiting_requests > 0
                    })
                    && state
                        .decisions
                        .get(program)
                        .and_then(|decision| decision.last_target.as_deref())
                        == Some(destination)
            })
    }

    fn cross_rank_candidate_is_mature(
        &self,
        state: &ProgramSchedulerState,
        program: &ProgramRef,
        destination: &str,
        now: Instant,
    ) -> bool {
        let decision = &state.decisions[program];
        let Some(source) = decision.last_target.as_deref() else {
            return false;
        };
        let limit = self.config.progress_ttl.stats_window_size;
        let source_complete = state
            .rank_factors
            .get(source)
            .is_some_and(|factors| factors.request_window_complete(limit));
        let destination_complete = state
            .rank_factors
            .get(destination)
            .is_some_and(|factors| factors.request_window_complete(limit));
        let waited_for_observation = decision.queued_at.is_some_and(|queued_at| {
            now.saturating_duration_since(queued_at) >= self.config.metrics_interval
        });
        source_complete && destination_complete && waited_for_observation
    }

    fn source_has_resume_pressure(
        &self,
        state: &ProgramSchedulerState,
        program: &ProgramRef,
        now: Instant,
    ) -> bool {
        let threshold = self.config.admission_waiting_request_threshold;
        if threshold == 0 {
            return true;
        }
        let source = state.decisions[program].last_target.as_deref();
        let backend_waiting = source
            .and_then(|target| state.observations.get(target))
            .filter(|observation| self.observation_is_fresh(observation, now))
            .and_then(|observation| observation.waiting_requests)
            .is_some_and(|waiting| waiting >= threshold);
        backend_waiting || self.higher_priority_source_waiters(state, program, now) >= 1
    }

    fn higher_priority_source_waiters(
        &self,
        state: &ProgramSchedulerState,
        candidate: &ProgramRef,
        now: Instant,
    ) -> usize {
        let Some(source) = state.decisions[candidate].last_target.as_deref() else {
            return 0;
        };
        self.global_reasoning_waiters(state, candidate.model_pool())
            .into_iter()
            .filter(|program| program != candidate)
            .filter(|program| {
                state.decisions.get(program).is_some_and(|decision| {
                    decision.last_target.as_deref() == Some(source)
                        && decision.queued_at.is_some_and(|queued_at| {
                            now.saturating_duration_since(queued_at) >= self.config.metrics_interval
                        })
                })
            })
            .filter(|program| {
                self.local_resume_cmp(state, program, candidate, now)
                    .is_lt()
            })
            .count()
    }

    fn destination_has_context_headroom(
        &self,
        state: &ProgramSchedulerState,
        program: &ProgramRef,
        destination: &str,
        now: Instant,
    ) -> bool {
        let Some(capacity) = self.config.progress_ttl.token_capacity else {
            return true;
        };
        let decision = &state.decisions[program];
        let Some(source) = decision.last_target.as_deref() else {
            return false;
        };
        let limit = capacity as f64 * self.config.progress_ttl.low_watermark_ratio;
        let destination_headroom = (limit - self.target_usage(state, destination, now)).max(0.0);
        let source_headroom = (limit - self.target_usage(state, source, now)).max(0.0);
        destination_headroom
            >= source_headroom
                + self.config.cross_rank_headroom_ratio * decision.estimated_context_tokens as f64
    }

    fn active_program_count(&self, state: &ProgramSchedulerState, target_id: &str) -> usize {
        state
            .runtime
            .views()
            .into_iter()
            .filter(|program| {
                program.state == ProgramState::Active
                    && program.placement.as_deref() == Some(target_id)
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program_scheduling::{ProgramIdentity, ProgramSchedulerConfig, ProgramTarget};
    use serde_json::json;

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

    #[test]
    fn new_program_remains_on_initial_binding_in_global_mode() {
        let mut config = ProgramSchedulerConfig::default();
        config.global_queue = true;
        let scheduler = ProgramScheduler::new(config);
        let targets = [
            ProgramTarget {
                id: "rank-0".into(),
                base_url: "http://worker".into(),
                dp_rank: Some(0),
            },
            ProgramTarget {
                id: "rank-1".into(),
                base_url: "http://worker".into(),
                dp_rank: Some(1),
            },
        ];
        scheduler.sync_targets("model", &targets);
        let now = Instant::now();
        let mut state = scheduler.state.lock();
        let identity = identity("p");
        let handle = state.runtime.retain_request(&identity, 100, None, now);
        scheduler.ensure_decision_state(&mut state, &identity, handle.program().clone(), 100, now);
        state
            .global_queues
            .entry("model".into())
            .or_default()
            .push_back(handle.program().clone());
        assert!(scheduler
            .select_cross_rank_admission(&state, handle.program(), now)
            .is_none());
    }
}
