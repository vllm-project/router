//! Direct DP admission: all live waiters participate before any admission policy
//! gate. Native occupancy and request accounting supply the resource costs.

use super::resume_dp::{self, OptionValue};
use super::scheduler_admission::RankAdmissionPlan;
use super::scheduler_state::ProgramSchedulerState;
use super::{ProgramScheduler, ProgramState, ProgramStatus};
use std::time::Instant;

impl ProgramScheduler {
    pub(crate) fn schedule_dp_admission(
        &self,
        state: &mut ProgramSchedulerState,
        now: Instant,
    ) -> bool {
        let mut pools: Vec<_> = state.model_targets.keys().cloned().collect();
        pools.sort();
        let mut changed = false;
        for pool in pools {
            let targets: Vec<_> = state.model_targets[&pool]
                .iter()
                .filter(|id| state.targets.contains_key(*id) && self.capacity_ready(state, id, now))
                .cloned()
                .collect();
            let mut queued: Vec<_> = state
                .runtime
                .iter_views()
                .filter(|view| {
                    view.reference.model_pool() == pool
                        && view.state == ProgramState::Paused
                        && view.status == ProgramStatus::Reasoning
                        && view.waiting_requests > 0
                        && state.decisions.contains_key(view.reference)
                })
                .map(|view| view.reference.clone())
                .collect();
            self.order_resume_candidates(state, &mut queued, now);

            // Priorities remain priorities, not permission to exceed capacity.
            // An unplaceable high-priority waiter does not block all other work.
            let mut start = 0;
            while start < queued.len() {
                let end = (start + 1..queued.len())
                    .find(|&i| {
                        !self
                            .resume_tier_cmp(state, &queued[start], &queued[i], now)
                            .is_eq()
                            || !self
                                .segment_yield_cmp(state, &queued[start], &queued[i])
                                .is_eq()
                    })
                    .unwrap_or(queued.len());
                let budgets: Vec<_> = targets
                    .iter()
                    .map(|target| {
                        let active = state
                            .runtime
                            .iter_views()
                            .filter(|view| {
                                view.state == ProgramState::Active
                                    && view.placement == Some(target.as_str())
                            })
                            .count();
                        let capacity = self.target_capacity(state, target, now).unwrap_or(0);
                        // Use rank capacity, not native low-watermark/growth reserve tests.
                        (
                            (capacity as f64 - self.target_usage(state, target, now))
                                .max(0.0)
                                .floor() as usize,
                            self.config
                                .max_active_programs_per_target
                                .saturating_sub(active),
                        )
                    })
                    .collect();
                let mut assignments = Vec::new();
                let programs: Vec<Vec<OptionValue>> = queued[start..end]
                    .iter()
                    .map(|program| {
                        let decision = &state.decisions[program];
                        // Exactly the charge used by commit_admission; no extra
                        // reserve or separate shared-prefix accounting is added.
                        let cost =
                            decision.private_tokens(self.config.progress_ttl.decode_buffer_tokens);
                        let text = state
                            .runtime
                            .front_routing_text(program)
                            .unwrap_or_default();
                        targets
                            .iter()
                            .enumerate()
                            .filter(|(_, target)| {
                                // This mode changes admission, not routing. Preserve
                                // the existing cache-aware binding in either queue mode.
                                decision.last_target.as_ref() == Some(*target)
                            })
                            .map(|(target_index, target)| {
                                let affinity =
                                    state.bindings.prefix_match_rate(&pool, text, target);
                                let cached = (decision.estimated_context_tokens as f64
                                    * affinity.clamp(0.0, 1.0))
                                .floor() as usize;
                                let index = assignments.len();
                                assignments.push((program.clone(), target.clone(), cost));
                                OptionValue {
                                    target: target_index,
                                    tokens: cost.ceil() as usize,
                                    value: (
                                        cached as u128,
                                        decision.estimated_context_tokens as u128,
                                    ),
                                    index,
                                }
                            })
                            .collect()
                    })
                    .collect();
                // With fixed bindings and nonnegative values, admitting every
                // candidate is already optimal if the complete set fits. This
                // also lets zero-value requests progress without inventing a
                // cache value or constructing an unnecessary DP frontier.
                let mut remaining = budgets.clone();
                let all_fit = programs.iter().all(|options| {
                    let [option] = options.as_slice() else {
                        return false;
                    };
                    let (tokens, slots) = &mut remaining[option.target];
                    if option.tokens > *tokens || *slots == 0 {
                        return false;
                    }
                    *tokens -= option.tokens;
                    *slots -= 1;
                    true
                });
                let selected = if all_fit
                    && self.config.resume_solver_strategy
                        != crate::config::types::ResumeSolverStrategy::Greedy
                {
                    programs.iter().map(|options| options[0].index).collect()
                } else {
                    resume_dp::solve(&programs, &budgets, &self.config)
                };
                tracing::debug!(
                    event = "dp_admission_decision",
                    candidates = end - start,
                    selected = selected.len(),
                    "DP decided the admission set"
                );
                // The scheduler mutex covers collection, solving and commit.
                // No native admission filter is applied before or after DP.
                for index in selected {
                    let (program, target, cost) = &assignments[index];
                    let plan = RankAdmissionPlan {
                        used_tokens: self.target_usage(state, target, now),
                        required_tokens: *cost,
                        reserve_tokens: 0.0,
                        capacity_tokens: self.target_capacity(state, target, now),
                        forced: false,
                        batch_gain: false,
                    };
                    changed |= self.commit_admission(state, program, target, plan, now);
                }
                start = end;
            }
        }
        changed
    }
}
