//! DP chooses a subset within the existing resume tier. Native admission still
//! owns eligibility, reserves, batch gain, force-resume and committed accounting.

use super::resume_dp::{self, OptionValue};
use super::scheduler_admission::RankAdmissionPlan;
use super::scheduler_state::ProgramSchedulerState;
use super::{ProgramRef, ProgramResumeOrder, ProgramScheduler, ProgramState};
use std::collections::BTreeMap;
use std::time::Instant;

type Admission = (ProgramRef, String, RankAdmissionPlan);

impl ProgramScheduler {
    /// Caller holds the state lock and rechecks native gates before each commit.
    pub(crate) fn select_resume_admissions(
        &self,
        state: &ProgramSchedulerState,
        admissible: impl IntoIterator<Item = Admission>,
        now: Instant,
    ) -> Vec<Admission> {
        let mut admissible = admissible.into_iter();
        let Some(first) = admissible.next() else {
            return Vec::new();
        };
        if self.config.resume_order != ProgramResumeOrder::Dp
            || first.2.forced
            || (self.config.progress_ttl.token_capacity.is_none()
                && !self.config.dynamic_token_capacity)
        {
            return vec![first];
        }
        let admissible: Vec<_> = std::iter::once(first).chain(admissible).collect();
        let first = &admissible[0];
        // Input is already ordered by native force/priority/yield tiers. Do not
        // allow a lower tier's combined value to overtake a higher-tier request.
        let cohort: Vec<_> = admissible
            .iter()
            .enumerate()
            .take_while(|(_, a)| {
                self.resume_tier_cmp(state, &first.0, &a.0, now).is_eq()
                    && self.segment_yield_cmp(state, &first.0, &a.0).is_eq()
            })
            .collect();
        let mut targets = BTreeMap::new();
        let mut capacities = Vec::new();
        let mut programs: Vec<Vec<OptionValue>> = Vec::new();
        let mut previous = None;
        for (index, (program, target, plan)) in cohort {
            let target_index = *targets.entry(target.clone()).or_insert_with(|| {
                let active = state
                    .runtime
                    .iter_views()
                    .filter(|view| {
                        view.state == ProgramState::Active
                            && view.placement == Some(target.as_str())
                    })
                    .count();
                let available = (plan.capacity_tokens.unwrap_or(0) as f64
                    * self.config.progress_ttl.low_watermark_ratio
                    - plan.used_tokens)
                    .max(0.0);
                capacities.push((
                    available.floor() as usize,
                    self.config
                        .max_active_programs_per_target
                        .saturating_sub(active),
                ));
                capacities.len() - 1
            });
            if previous != Some(program) {
                programs.push(Vec::new());
                previous = Some(program);
            }
            let context = state.decisions[program].estimated_context_tokens;
            let rate = state
                .runtime
                .front_routing_text(program)
                .map_or(0.0, |text| {
                    state
                        .bindings
                        .prefix_match_rate(program.model_pool(), text, target)
                });
            let cached = (context as f64 * rate.clamp(0.0, 1.0)).floor() as usize;
            programs.last_mut().unwrap().push(OptionValue {
                target: target_index,
                tokens: plan.required_tokens.ceil() as usize,
                value: (cached as u128, context as u128),
                index,
            });
        }
        let chosen = resume_dp::solve(&programs, &capacities, &self.config);
        admissible
            .into_iter()
            .enumerate()
            .filter_map(|(i, admission)| chosen.binary_search(&i).is_ok().then_some(admission))
            .collect()
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::program_scheduling::{
        ProgramBindingStrategy, ProgramIdentity, ProgramRequestHandle, ProgramSchedulerConfig,
        ProgramStatus, ProgramTarget, ProgressTtlConfig,
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
    fn dp_selects_two_smaller_cached_programs_in_local_and_global_queues() {
        for global in [false, true] {
            let scheduler = scheduler(global, ProgramResumeOrder::Dp);
            let mut state = scheduler.state.lock();
            let now = Instant::now();
            let large = enqueue(&scheduler, &mut state, "large", 800, 0, now);
            let small_a = enqueue(&scheduler, &mut state, "small-a", 500, 0, now);
            let small_b = enqueue(&scheduler, &mut state, "small-b", 500, 0, now);
            assert!(scheduler.schedule_waiting(&mut state, now));
            assert_eq!(
                state.runtime.state(large.program()),
                Some((ProgramState::Paused, ProgramStatus::Reasoning))
            );
            for handle in [&small_a, &small_b] {
                assert_eq!(state.runtime.placement(handle.program()), Some("rank-0"));
            }
            assert_eq!(scheduler.target_usage(&state, "rank-0", now), 1200.0);
        }
    }

    #[test]
    fn dp_preserves_request_priority_and_native_forced_resume() {
        for forced in [false, true] {
            let scheduler = scheduler(true, ProgramResumeOrder::Dp);
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
        let scheduler = scheduler(false, ProgramResumeOrder::Dp);
        let mut state = scheduler.state.lock();
        let now = Instant::now();
        let cancelled = enqueue(&scheduler, &mut state, "cancelled", 1000, 0, now);
        let waiting = enqueue(&scheduler, &mut state, "waiting", 500, 0, now);
        assert!(state.runtime.cancel_request(&cancelled));
        // A fresh backend observation reduces capacity before scheduling.
        let observation = state.observations.get_mut("rank-0").unwrap();
        observation.estimated_active_program_tokens = Some(800.0);
        observation.observed_at = Some(now);
        assert!(!scheduler.schedule_waiting(&mut state, now));
        assert_eq!(state.runtime.placement(waiting.program()), None);
        state
            .observations
            .get_mut("rank-0")
            .unwrap()
            .estimated_active_program_tokens = Some(0.0);
        assert!(scheduler.schedule_waiting(&mut state, now));
        assert_eq!(state.runtime.placement(waiting.program()), Some("rank-0"));
        assert_eq!(state.runtime.placement(cancelled.program()), None);
    }

    #[test]
    fn batch_selection_does_not_bypass_native_growth_admission() {
        use crate::program_scheduling::RequestSample;
        let scheduler = scheduler(false, ProgramResumeOrder::Dp);
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
    fn cross_rank_dp_uses_destination_affinity_after_native_gates() {
        use crate::program_scheduling::{scheduler_state::ProgramPauseReason, RequestSample};
        use std::time::Duration;

        let scheduler = scheduler(true, ProgramResumeOrder::Dp);
        let targets: Vec<_> = (0..3)
            .map(|i| ProgramTarget {
                id: format!("rank-{i}"),
                base_url: format!("http://worker-{i}"),
                dp_rank: None,
            })
            .collect();
        scheduler.sync_targets("model", &targets);
        let mut state = scheduler.state.lock();
        let now = Instant::now();
        let handle = enqueue(&scheduler, &mut state, "request", 100, 0, now);
        let decision = state.decisions.get_mut(handle.program()).unwrap();
        decision.last_target = Some("rank-0".into());
        decision.last_pause_reason = Some(ProgramPauseReason::TtlExpired);
        decision.queued_at = Some(now - Duration::from_secs(2));
        state
            .bindings
            .commit_placement(handle.program(), "request", "rank-2");
        let observation = state.observations.get_mut("rank-0").unwrap();
        observation.observed_at = Some(now);
        observation.estimated_active_program_tokens = Some(1200.0);
        observation.waiting_requests = Some(1);
        // Cross-rank escape requires mature observations on source/destination.
        assert!(!scheduler.schedule_waiting(&mut state, now));
        let limit = scheduler.config.progress_ttl.stats_window_size;
        for factors in state.rank_factors.values_mut() {
            for _ in 0..limit {
                factors.push_request(
                    RequestSample {
                        prompt_tokens: 100,
                        completion_tokens: 0,
                        cached_prompt_tokens: 100,
                        context_growth_tokens: Some(0),
                        e2e_seconds: 1.0,
                        decode_seconds: Some(1.0),
                        queue_seconds: 0.0,
                        rounds_since_ttl_pause: 1,
                        finished_at: now,
                    },
                    limit,
                );
            }
        }
        assert!(scheduler.schedule_waiting(&mut state, now));
        assert_eq!(state.runtime.placement(handle.program()), Some("rank-2"));
    }

    #[test]
    fn direct_dp_selects_admission_even_when_native_reserve_rejects_every_candidate() {
        use crate::program_scheduling::RequestSample;
        for global in [false, true] {
            for mode in [ProgramResumeOrder::Dp, ProgramResumeOrder::DpAdmission] {
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
                let direct = mode == ProgramResumeOrder::DpAdmission;
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
        let mut scheduler = scheduler(false, ProgramResumeOrder::DpAdmission);
        scheduler.config.progress_ttl.low_watermark_ratio = 0.5;
        let mut state = scheduler.state.lock();
        let now = Instant::now();
        let a = enqueue(&scheduler, &mut state, "a", 500, 0, now);
        let b = enqueue(&scheduler, &mut state, "b", 500, 0, now);
        let observation = state.observations.get_mut("rank-0").unwrap();
        observation.observed_at = Some(now);
        observation.estimated_active_program_tokens = Some(0.0);
        observation.waiting_requests = Some(10);
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
        let scheduler = scheduler(false, ProgramResumeOrder::DpAdmission);
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
    fn direct_dp_preserves_bindings_in_both_queue_modes() {
        for global in [false, true] {
            let scheduler = scheduler(global, ProgramResumeOrder::DpAdmission);
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
            observation.observed_at = Some(now);
            observation.estimated_active_program_tokens = Some(1200.0);
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
        let mut scheduler = scheduler(false, ProgramResumeOrder::DpAdmission);
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
        let mut scheduler = scheduler(false, ProgramResumeOrder::DpAdmission);
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
            let mut scheduler = scheduler(false, ProgramResumeOrder::DpAdmission);
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
        let scheduler = scheduler(false, ProgramResumeOrder::DpAdmission);
        let mut state = scheduler.state.lock();
        let now = Instant::now();
        let request = enqueue(&scheduler, &mut state, "empty", 0, 0, now);
        assert!(scheduler.schedule_waiting(&mut state, now));
        assert_eq!(state.runtime.placement(request.program()), Some("rank-0"));
    }
}
