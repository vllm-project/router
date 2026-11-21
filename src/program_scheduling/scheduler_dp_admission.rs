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
            queued.sort_by_key(|program| {
                let deadline = state.decisions[program]
                    .force_resume_deadline
                    .filter(|deadline| *deadline <= now);
                (deadline.is_none(), deadline)
            });

            // Priorities remain priorities, not permission to exceed capacity.
            // An unplaceable high-priority waiter does not block all other work.
            let mut start = 0;
            while start < queued.len() {
                let overdue = state.decisions[&queued[start]]
                    .force_resume_deadline
                    .is_some_and(|deadline| deadline <= now);
                let end = if overdue {
                    start + 1
                } else {
                    (start + 1..queued.len())
                        .find(|&i| {
                            !self
                                .resume_tier_cmp(state, &queued[start], &queued[i], now)
                                .is_eq()
                                || !self
                                    .segment_yield_cmp(state, &queued[start], &queued[i])
                                    .is_eq()
                        })
                        .unwrap_or(queued.len())
                };
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
                                self.resume_target_allowed(state, program, target)
                            })
                            .map(|(target_index, target)| {
                                let affinity =
                                    state.bindings.prefix_match_rate(&pool, text, target);
                                let cached = (decision.context.tokens() as f64
                                    * affinity.clamp(0.0, 1.0))
                                .floor() as usize;
                                let index = assignments.len();
                                assignments.push((program.clone(), target.clone(), cost));
                                OptionValue {
                                    target: target_index,
                                    tokens: cost.ceil() as usize,
                                    value: (cached as u128, decision.context.tokens() as u128),
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
                    && (overdue
                        || self.config.resume_solver_strategy
                            != crate::config::types::ResumeSolverStrategy::Greedy)
                {
                    programs.iter().map(|options| options[0].index).collect()
                } else if overdue {
                    Vec::new()
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
                        forced: overdue,
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

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::program_scheduling::ProgramResumeOrder;
    use crate::program_scheduling::{
        ProgramBindingStrategy, ProgramIdentity, ProgramRequestHandle, ProgramSchedulerConfig,
        ProgramTarget, ProgressTtlConfig,
    };
    use serde_json::json;

    fn scheduler(global: bool, order: ProgramResumeOrder) -> ProgramScheduler {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig {
            dp_capacity_quantum: 1, // Exact-token fixtures; default quantum is tested separately.
            global_queue: global,
            resume_order: order,
            binding_strategy: ProgramBindingStrategy::CacheAware,
            progress_ttl: ProgressTtlConfig {
                token_capacity: Some(1200),
                ..ProgressTtlConfig::default()
            },
            ..ProgramSchedulerConfig::default()
        });
        scheduler.sync_targets(
            "model",
            &[ProgramTarget {
                id: "rank-0".into(),
                base_url: "http://worker".into(),
                dp_rank: None,
            }],
        );
        scheduler
    }

    fn enqueue(
        scheduler: &ProgramScheduler,
        state: &mut ProgramSchedulerState,
        name: &str,
        tokens: usize,
        priority: i32,
        now: Instant,
    ) -> ProgramRequestHandle {
        let identity = ProgramIdentity::from_request(
            None,
            Some(&json!({
                "vllm_xargs": {"agentic_context": {
                    "program_id": name, "expected_resume": true, "priority": priority,
                }}
            })),
            Some("model"),
        )
        .unwrap()
        .unwrap();
        let handle = state
            .runtime
            .retain_request(&identity, tokens, Some(name), now);
        scheduler.ensure_decision_state(
            state,
            &identity,
            handle.program().clone(),
            tokens,
            now,
            Some(name),
        );
        state
            .bindings
            .commit_placement(handle.program(), name, "rank-0");
        let queue = if scheduler.config.global_queue {
            state.global_queues.entry("model".into()).or_default()
        } else {
            state.rank_queues.entry("rank-0".into()).or_default()
        };
        queue.push_back(handle.program().clone());
        handle
    }

    #[test]
    fn fixed_binding_overdue_requests_use_earliest_deadline_first() {
        let scheduler = scheduler(false, ProgramResumeOrder::DpAdmissionTAO);
        let mut state = scheduler.state.lock();
        let now = Instant::now();
        let later = enqueue(&scheduler, &mut state, "a-later", 1100, 10, now);
        let older = enqueue(&scheduler, &mut state, "z-older", 600, 0, now);
        state
            .decisions
            .get_mut(older.program())
            .unwrap()
            .force_resume_deadline = Some(now - std::time::Duration::from_secs(2));
        state
            .decisions
            .get_mut(later.program())
            .unwrap()
            .force_resume_deadline = Some(now - std::time::Duration::from_secs(1));
        assert!(scheduler.schedule_waiting(&mut state, now));
        assert_eq!(state.runtime.placement(older.program()), Some("rank-0"));
        assert_eq!(state.runtime.placement(later.program()), None);
        assert_eq!(scheduler.target_usage(&state, "rank-0", now), 700.0);
    }

    #[test]
    fn native_resume_preserves_request_priority_and_forced_resume() {
        for forced in [false, true] {
            let scheduler = scheduler(true, ProgramResumeOrder::Fcfs);
            let mut state = scheduler.state.lock();
            let now = Instant::now();
            let large = enqueue(
                &scheduler,
                &mut state,
                "large",
                if forced { 1400 } else { 800 },
                if forced { 0 } else { 10 },
                now,
            );
            enqueue(&scheduler, &mut state, "small-a", 500, 0, now);
            enqueue(&scheduler, &mut state, "small-b", 500, 0, now);
            if forced {
                state
                    .decisions
                    .get_mut(large.program())
                    .unwrap()
                    .force_resume_deadline = Some(now);
            }
            scheduler.schedule_waiting(&mut state, now);
            assert_eq!(state.runtime.placement(large.program()), Some("rank-0"));
        }
    }

    #[test]
    fn cancellation_and_latest_capacity_are_rechecked() {
        let scheduler = scheduler(false, ProgramResumeOrder::Fcfs);
        let mut state = scheduler.state.lock();
        let now = Instant::now();
        let cancelled = enqueue(&scheduler, &mut state, "cancelled", 1000, 0, now);
        let waiting = enqueue(&scheduler, &mut state, "waiting", 500, 0, now);
        assert!(state.runtime.cancel_request(&cancelled));
        // A fresh backend observation reduces capacity before scheduling.
        let observation = state.observations.get_mut("rank-0").unwrap();
        observation.accounting.estimated_active_program_tokens = Some(800.0);
        observation.backend.observed_at = Some(now);
        assert!(!scheduler.schedule_waiting(&mut state, now));
        assert_eq!(state.runtime.placement(waiting.program()), None);
        state
            .observations
            .get_mut("rank-0")
            .unwrap()
            .accounting
            .estimated_active_program_tokens = Some(0.0);
        assert!(scheduler.schedule_waiting(&mut state, now));
        assert_eq!(state.runtime.placement(waiting.program()), Some("rank-0"));
        assert_eq!(state.runtime.placement(cancelled.program()), None);
    }

    #[test]
    fn native_resume_does_not_bypass_growth_admission() {
        use crate::program_scheduling::RequestSample;
        let scheduler = scheduler(false, ProgramResumeOrder::Fcfs);
        let mut state = scheduler.state.lock();
        let now = Instant::now();
        let first = enqueue(&scheduler, &mut state, "a-first", 500, 0, now);
        let second = enqueue(&scheduler, &mut state, "b-second", 500, 0, now);
        let limit = scheduler.config.progress_ttl.stats_window_size;
        for _ in 0..limit {
            state.rank_factors.get_mut("rank-0").unwrap().push_request(
                RequestSample {
                    prompt_tokens: 500,
                    completion_tokens: 100,
                    cached_prompt_tokens: 500,
                    context_growth_tokens: Some(100),
                    e2e_seconds: 1.0,
                    decode_seconds: Some(1.0),
                    queue_seconds: 0.0,
                    rounds_since_ttl_pause: 1,
                    finished_at: now,
                },
                limit,
            );
        }
        for handle in [&first, &second] {
            assert!(scheduler
                .rank_admission_plan(&state, handle.program(), "rank-0", now)
                .is_some());
        }
        assert!(scheduler.schedule_waiting(&mut state, now));
        assert_eq!(state.runtime.placement(first.program()), Some("rank-0"));
        assert_eq!(state.runtime.placement(second.program()), None);
    }

    #[test]
    fn fcfs_keeps_native_first_fit_behavior() {
        let scheduler = scheduler(true, ProgramResumeOrder::Fcfs);
        let mut state = scheduler.state.lock();
        let now = Instant::now();
        let large = enqueue(&scheduler, &mut state, "a-large", 800, 0, now);
        enqueue(&scheduler, &mut state, "b-small", 500, 0, now);
        enqueue(&scheduler, &mut state, "c-small", 500, 0, now);
        scheduler.schedule_waiting(&mut state, now);
        assert_eq!(state.runtime.placement(large.program()), Some("rank-0"));
        assert_eq!(scheduler.target_usage(&state, "rank-0", now), 900.0);
    }

    #[test]
    fn direct_dp_selects_admission_even_when_native_reserve_rejects_every_candidate() {
        use crate::program_scheduling::RequestSample;
        for global in [false, true] {
            for mode in [ProgramResumeOrder::Fcfs, ProgramResumeOrder::DpAdmissionTAO] {
                let scheduler = scheduler(global, mode);
                let mut state = scheduler.state.lock();
                let now = Instant::now();
                let large = enqueue(&scheduler, &mut state, "large", 800, 0, now);
                let a = enqueue(&scheduler, &mut state, "small-a", 500, 0, now);
                let b = enqueue(&scheduler, &mut state, "small-b", 500, 0, now);
                let limit = scheduler.config.progress_ttl.stats_window_size;
                for _ in 0..limit {
                    state.rank_factors.get_mut("rank-0").unwrap().push_request(
                        RequestSample {
                            prompt_tokens: 500,
                            completion_tokens: 100,
                            cached_prompt_tokens: 500,
                            context_growth_tokens: Some(1000),
                            e2e_seconds: 1.0,
                            decode_seconds: Some(1.0),
                            queue_seconds: 0.0,
                            rounds_since_ttl_pause: 1,
                            finished_at: now,
                        },
                        limit,
                    );
                }
                for handle in [&large, &a, &b] {
                    assert!(scheduler
                        .rank_admission_plan(&state, handle.program(), "rank-0", now)
                        .is_none());
                }
                let direct = mode == ProgramResumeOrder::DpAdmissionTAO;
                assert_eq!(scheduler.schedule_waiting(&mut state, now), direct);
                assert_eq!(state.runtime.placement(large.program()), None);
                for handle in [&a, &b] {
                    assert_eq!(
                        state.runtime.placement(handle.program()),
                        direct.then_some("rank-0")
                    );
                }
                assert_eq!(
                    scheduler.target_usage(&state, "rank-0", now),
                    if direct { 1200.0 } else { 0.0 }
                );
            }
        }
    }

    #[test]
    fn direct_dp_uses_capacity_not_backend_waiting_or_low_watermark_gates() {
        let mut scheduler = scheduler(false, ProgramResumeOrder::DpAdmissionTAO);
        scheduler.config.progress_ttl.low_watermark_ratio = 0.5;
        let mut state = scheduler.state.lock();
        let now = Instant::now();
        let a = enqueue(&scheduler, &mut state, "a", 500, 0, now);
        let b = enqueue(&scheduler, &mut state, "b", 500, 0, now);
        let observation = state.observations.get_mut("rank-0").unwrap();
        observation.backend.observed_at = Some(now);
        observation.accounting.estimated_active_program_tokens = Some(0.0);
        observation.backend.waiting_requests = Some(10);
        assert!(scheduler
            .rank_admission_plan(&state, a.program(), "rank-0", now)
            .is_none());
        assert!(scheduler.schedule_waiting(&mut state, now));
        assert_eq!(state.runtime.placement(a.program()), Some("rank-0"));
        assert_eq!(state.runtime.placement(b.program()), Some("rank-0"));
        assert_eq!(scheduler.target_usage(&state, "rank-0", now), 1200.0);
    }

    #[test]
    fn direct_dp_cannot_force_oversized_requests_or_admit_cancelled_requests() {
        let scheduler = scheduler(false, ProgramResumeOrder::DpAdmissionTAO);
        let mut state = scheduler.state.lock();
        let now = Instant::now();
        let oversized = enqueue(&scheduler, &mut state, "oversized", 1400, 0, now);
        let cancelled = enqueue(&scheduler, &mut state, "cancelled", 1100, 0, now);
        let fits = enqueue(&scheduler, &mut state, "fits", 500, 0, now);
        state
            .decisions
            .get_mut(oversized.program())
            .unwrap()
            .force_resume_deadline = Some(now);
        assert!(state.runtime.cancel_request(&cancelled));
        assert!(scheduler.schedule_waiting(&mut state, now));
        assert_eq!(state.runtime.placement(oversized.program()), None);
        assert_eq!(state.runtime.placement(cancelled.program()), None);
        assert_eq!(state.runtime.placement(fits.program()), Some("rank-0"));
        assert_eq!(scheduler.target_usage(&state, "rank-0", now), 600.0);
    }

    #[test]
    fn direct_dp_cross_backend_placement_requires_global_queue() {
        for global in [false, true] {
            let scheduler = scheduler(global, ProgramResumeOrder::DpAdmissionTAO);
            let targets: Vec<_> = (0..2)
                .map(|i| ProgramTarget {
                    id: format!("rank-{i}"),
                    base_url: format!("http://worker-{i}"),
                    dp_rank: None,
                })
                .collect();
            scheduler.sync_targets("model", &targets);
            let mut state = scheduler.state.lock();
            let now = Instant::now();
            let a = enqueue(&scheduler, &mut state, "a", 500, 0, now);
            let b = enqueue(&scheduler, &mut state, "b", 500, 0, now);
            for handle in [&a, &b] {
                state
                    .decisions
                    .get_mut(handle.program())
                    .unwrap()
                    .last_target = Some("rank-0".into());
            }
            let observation = state.observations.get_mut("rank-0").unwrap();
            observation.backend.observed_at = Some(now);
            observation.accounting.estimated_active_program_tokens = Some(1200.0);
            if global {
                assert!(scheduler.schedule_waiting(&mut state, now));
                for handle in [&a, &b] {
                    assert_eq!(state.runtime.placement(handle.program()), Some("rank-1"));
                }
                continue;
            }
            assert!(!scheduler.schedule_waiting(&mut state, now));
            for handle in [&a, &b] {
                assert_eq!(state.runtime.placement(handle.program()), None);
            }
            state.decisions.get_mut(b.program()).unwrap().last_target = Some("rank-1".into());
            assert!(scheduler.schedule_waiting(&mut state, now));
            assert_eq!(state.runtime.placement(a.program()), None);
            assert_eq!(state.runtime.placement(b.program()), Some("rank-1"));
        }
    }

    #[test]
    fn direct_dp_respects_existing_program_count_and_active_occupancy() {
        let mut scheduler = scheduler(false, ProgramResumeOrder::DpAdmissionTAO);
        scheduler.config.max_active_programs_per_target = 1;
        let mut state = scheduler.state.lock();
        let now = Instant::now();
        let large = enqueue(&scheduler, &mut state, "large", 800, 0, now);
        let a = enqueue(&scheduler, &mut state, "a", 500, 0, now);
        let b = enqueue(&scheduler, &mut state, "b", 500, 0, now);
        assert!(scheduler.schedule_waiting(&mut state, now));
        assert_eq!(state.runtime.placement(large.program()), Some("rank-0"));
        assert_eq!(state.runtime.placement(a.program()), None);
        assert_eq!(state.runtime.placement(b.program()), None);
        assert!(!scheduler.schedule_waiting(&mut state, now));
        assert_eq!(scheduler.target_usage(&state, "rank-0", now), 900.0);
    }

    #[tokio::test]
    async fn direct_dp_oversized_acquire_expires_after_priority_deadline() {
        use crate::program_scheduling::ScheduleError;
        use std::time::Duration;
        let mut scheduler = scheduler(false, ProgramResumeOrder::DpAdmissionTAO);
        scheduler.config.queue_timeout = Duration::from_millis(30);
        scheduler.config.force_resume_timeout = Duration::from_millis(1);
        let identity = ProgramIdentity::from_request(
            None,
            Some(&json!({
                "vllm_xargs": {"agentic_context": {"program_id": "oversized"}}
            })),
            Some("model"),
        )
        .unwrap()
        .unwrap();
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://worker".into(),
            dp_rank: None,
        };
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            scheduler.acquire(identity, 1400, &[target], Some("oversized".into())),
        )
        .await
        .unwrap();
        assert!(matches!(result, Err(ScheduleError::QueueTimeout)));
        assert!(scheduler.state.lock().runtime.request_pool_is_empty());
    }

    #[test]
    fn direct_dp_falls_back_to_bfd_and_makes_progress_when_frontier_is_exhausted() {
        for capacity in [200_000, 500_000] {
            let mut scheduler = scheduler(false, ProgramResumeOrder::DpAdmissionTAO);
            scheduler.config.progress_ttl.token_capacity = Some(capacity);
            scheduler.config.dp_state_budget = Some(16);
            let mut state = scheduler.state.lock();
            let now = Instant::now();
            for i in 0..18 {
                enqueue(
                    &scheduler,
                    &mut state,
                    &format!("request-{i:02}"),
                    1 << i,
                    0,
                    now,
                );
            }
            // In the overloaded case, the exact frontier exceeds its existing
            // bound. The all-fit case has a trivial optimal set and needs no DP.
            let all_fit = capacity == 500_000;
            assert!(scheduler.schedule_waiting(&mut state, now));
            let active = state
                .runtime
                .iter_views()
                .filter(|view| view.state == ProgramState::Active)
                .count();
            assert!(active > 0);
            if all_fit {
                assert_eq!(active, 18);
            }
            let used: f64 = state
                .runtime
                .iter_views()
                .filter(|v| v.state == ProgramState::Active)
                .map(|v| {
                    state.decisions[v.reference]
                        .private_tokens(scheduler.config.progress_ttl.decode_buffer_tokens)
                })
                .sum();
            assert!(used <= capacity as f64);
        }
    }

    #[test]
    fn direct_dp_can_admit_a_zero_value_request_when_it_fits() {
        let scheduler = scheduler(false, ProgramResumeOrder::DpAdmissionTAO);
        let mut state = scheduler.state.lock();
        let now = Instant::now();
        let request = enqueue(&scheduler, &mut state, "empty", 0, 0, now);
        assert!(scheduler.schedule_waiting(&mut state, now));
        assert_eq!(state.runtime.placement(request.program()), Some("rank-0"));
    }
}
