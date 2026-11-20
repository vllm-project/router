//! Live backend headroom expressed in vLLM reference-token units.
//! This is not an exact candidate-specific admission oracle for hybrid models.
use super::scheduler_state::ProgramSchedulerState;
use super::{ProgramScheduler, ProgramState, ProgramStatus};
use std::time::Instant;

impl ProgramScheduler {
    pub(crate) fn dynamic_headroom(
        &self,
        state: &ProgramSchedulerState,
        target: &str,
        now: Instant,
    ) -> Option<f64> {
        self.target_capacity(state, target, now)
            .map(|capacity| (capacity as f64 - self.target_usage(state, target, now)).max(0.0))
    }

    /// None retains legacy count-only behavior only when dynamic mode is disabled.
    pub(crate) fn target_capacity(
        &self,
        state: &ProgramSchedulerState,
        target: &str,
        now: Instant,
    ) -> Option<usize> {
        if self.tree_dp_mode() {
            return self.tree_dp_capacity(state, target, now);
        }
        if !self.config.dynamic_token_capacity {
            return self.config.progress_ttl.token_capacity;
        }
        let observation = state.observations.get(target)?;
        if !self.observation_is_fresh(observation, now) {
            return None;
        }
        observation.backend_available_tokens.map(|available| {
            // Physical usage already reflects idle Acting cache. Keep the full
            // logical ledger for policy scoring, but reserve execution capacity
            // only for programs with pending/in-flight model work. Backend
            // waiters must retain that reservation across metrics scrapes.
            let ledger = self.ledger_usage(state, target);
            let physical_limit = ledger + available - observation.admitted_token_debits;
            let limit = observation
                .backend_usable_token_capacity
                .map_or(physical_limit, |total| {
                    physical_limit
                        .min(ledger + total as f64 - self.reasoning_reservations(state, target))
                });
            limit.max(0.0).floor() as usize
        })
    }

    pub(crate) fn capacity_ready(
        &self,
        state: &ProgramSchedulerState,
        target: &str,
        now: Instant,
    ) -> bool {
        !self.config.dynamic_token_capacity || self.dynamic_headroom(state, target, now).is_some()
    }

    pub(crate) fn reasoning_reservations(
        &self,
        state: &ProgramSchedulerState,
        target: &str,
    ) -> f64 {
        state
            .runtime
            .iter_views()
            .filter(|v| v.state == ProgramState::Active && v.placement == Some(target))
            .filter(|v| {
                v.status == ProgramStatus::Reasoning
                    || v.waiting_requests > 0
                    || v.in_flight_requests > 0
            })
            .filter_map(|v| state.decisions.get(v.reference))
            .map(|d| d.private_tokens(self.config.progress_ttl.decode_buffer_tokens))
            .sum()
    }

    /// An idle placement may retain affinity but cannot bypass execution capacity
    /// when a new turn arrives. Called after retention/size update, before dispatch.
    pub(crate) fn recheck_idle_execution_capacity(
        &self,
        state: &mut ProgramSchedulerState,
        program: &super::ProgramRef,
        now: Instant,
    ) {
        if self.tree_dp_mode() {
            return;
        }
        if !self.config.dynamic_token_capacity || self.config.binding_only {
            return;
        }
        let Some(v) = state.runtime.view(program) else {
            return;
        };
        if v.state != ProgramState::Active || v.in_flight_requests > 0 {
            return;
        }
        let Some(target) = v.placement else {
            return;
        };
        let fits = state.observations.get(&target).is_some_and(|o| {
            self.observation_is_fresh(o, now)
                && o.backend_available_tokens.is_some()
                && o.backend_usable_token_capacity.is_some_and(|total| {
                    self.reasoning_reservations(state, &target) <= total as f64
                })
        });
        if !fits {
            self.pause_idle(
                state,
                program,
                super::scheduler_state::ProgramPauseReason::CapacityRepair,
                now,
            );
        }
    }

    pub(crate) fn ledger_usage(&self, state: &ProgramSchedulerState, target: &str) -> f64 {
        state
            .runtime
            .iter_views()
            .filter(|v| v.state == ProgramState::Active && v.placement == Some(target))
            .filter_map(|v| state.decisions.get(v.reference))
            .map(|d| d.private_tokens(self.config.progress_ttl.decode_buffer_tokens))
            .sum()
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::program_scheduling::{
        BackendObservation, ProgramBindingStrategy, ProgramIdentity, ProgramResumeOrder,
        ProgramSchedulerConfig, ProgramTarget,
    };
    use serde_json::json;
    use std::time::Duration;

    fn targets() -> Vec<ProgramTarget> {
        vec![ProgramTarget {
            id: "rank".into(),
            base_url: "http://worker".into(),
            dp_rank: None,
        }]
    }
    fn scheduler(order: ProgramResumeOrder) -> ProgramScheduler {
        let mut config = ProgramSchedulerConfig {
            dynamic_token_capacity: true,
            dp_capacity_quantum: 1, // Small exact-token test fixtures.
            resume_order: order,
            // These fixtures test legacy physical-headroom accounting.
            // Tree-backed direct DP has its own tests in tree_dp_admission.
            binding_strategy: ProgramBindingStrategy::ConsistentHash,
            ..Default::default()
        };
        config.progress_ttl.decode_buffer_tokens = 0;
        let s = ProgramScheduler::new(config);
        s.sync_targets("m", &targets());
        s
    }
    fn raw(target: &str, usage: f64) -> BackendObservation {
        BackendObservation {
            target_id: target.into(),
            base_url: "http://worker".into(),
            dp_rank: None,
            kv_cache_usage: Some(usage),
            token_capacity: Some(1000),
            num_gpu_blocks: Some(100),
            running_requests: Some(0),
            waiting_requests: Some(0),
            observed_at: Instant::now(),
        }
    }
    fn observe(s: &ProgramScheduler, usage: f64) {
        s.apply_observations(s.begin_observation(&targets()), [raw("rank", usage)]);
    }
    fn enqueue(
        s: &ProgramScheduler,
        id: &str,
        tokens: usize,
    ) -> super::super::ProgramRequestHandle {
        let identity = ProgramIdentity::from_request(None,
            Some(&json!({"vllm_xargs":{"agentic_context":{"program_id":id,"expected_resume":true,"generation":0}}})),
            Some("m")).unwrap().unwrap();
        let mut state = s.state.lock();
        let now = Instant::now();
        let handle = state
            .runtime
            .retain_request(&identity, tokens, Some(id), now);
        s.ensure_decision_state(
            &mut state,
            &identity,
            handle.program().clone(),
            tokens,
            now,
            Some(id),
        );
        state
            .rank_queues
            .entry("rank".into())
            .or_default()
            .push_back(handle.program().clone());
        handle
    }

    #[test]
    fn both_dp_modes_wait_for_backend_then_obey_changing_headroom() {
        for order in [ProgramResumeOrder::Dp, ProgramResumeOrder::DpAdmission] {
            let s = scheduler(order);
            let a = enqueue(&s, "a", 300);
            let b = enqueue(&s, "b", 300);
            {
                let mut state = s.state.lock();
                assert!(!s.schedule_waiting(&mut state, Instant::now()));
            }
            observe(&s, 0.5); // floor(1000 * .5 * 99/100) = 495
            {
                let mut state = s.state.lock();
                assert!(s.schedule_waiting(&mut state, Instant::now()));
                assert_eq!(s.ledger_usage(&state, "rank"), 300.0);
                assert_eq!(
                    s.dynamic_headroom(&state, "rank", Instant::now()),
                    Some(195.0)
                );
                assert!(!s.schedule_waiting(&mut state, Instant::now()));
            }
            observe(&s, 0.9); // only 98/99 equivalent tokens, not enough
            {
                let mut state = s.state.lock();
                assert!(!s.schedule_waiting(&mut state, Instant::now()));
                assert!(s.target_capacity(&state, "rank", Instant::now()).unwrap() < 400);
            }
            observe(&s, 0.0);
            {
                let mut state = s.state.lock();
                assert!(s.schedule_waiting(&mut state, Instant::now()));
                assert_eq!(s.ledger_usage(&state, "rank"), 600.0);
                for p in [a.program(), b.program()] {
                    assert_eq!(state.runtime.view(p).unwrap().state, ProgramState::Active);
                }
            }
        }
    }

    #[test]
    fn fresh_low_usage_does_not_readmit_reserved_backend_queue_capacity() {
        for order in [ProgramResumeOrder::Dp, ProgramResumeOrder::DpAdmission] {
            let s = scheduler(order);
            let a = enqueue(&s, "a", 600);
            observe(&s, 0.0);
            {
                let mut state = s.state.lock();
                s.schedule_waiting(&mut state, Instant::now());
                assert_eq!(
                    state.runtime.view(a.program()).unwrap().state,
                    ProgramState::Active
                );
            }
            let b = enqueue(&s, "b", 600);
            // Backend has not allocated the admitted request's KV yet. Repeated
            // successful scrapes are not evidence that its reservation vanished.
            for _ in 0..3 {
                observe(&s, 0.0);
                let mut state = s.state.lock();
                s.schedule_waiting(&mut state, Instant::now());
                assert_eq!(
                    state.runtime.view(b.program()).unwrap().state,
                    ProgramState::Paused
                );
                assert!(s.ledger_usage(&state, "rank") <= 990.0);
                assert!(s.dynamic_headroom(&state, "rank", Instant::now()).unwrap() <= 390.0);
            }
        }
    }

    #[test]
    fn idle_acting_context_does_not_consume_live_reasoning_reservation_twice() {
        for order in [ProgramResumeOrder::Dp, ProgramResumeOrder::DpAdmission] {
            let s = scheduler(order);
            let a = enqueue(&s, "acting", 600);
            observe(&s, 0.0);
            s.schedule_waiting(&mut s.state.lock(), Instant::now());
            let dispatch = s
                .state
                .lock()
                .runtime
                .admit_front(&a, "rank".to_string(), Instant::now())
                .unwrap();
            s.complete(
                &dispatch,
                true,
                false,
                super::super::ProgramUsageObservation::default(),
                None,
            );
            assert_eq!(
                s.state.lock().runtime.view(a.program()).unwrap().status,
                ProgramStatus::Acting
            );
            let b = enqueue(&s, "next", 600);
            // Idle does not mean physically free: full backend still blocks.
            observe(&s, 1.0);
            s.schedule_waiting(&mut s.state.lock(), Instant::now());
            assert_eq!(
                s.state.lock().runtime.view(b.program()).unwrap().state,
                ProgramState::Paused
            );
            observe(&s, 0.0);
            {
                let mut state = s.state.lock();
                s.schedule_waiting(&mut state, Instant::now());
                assert_eq!(
                    state.runtime.view(b.program()).unwrap().state,
                    ProgramState::Active
                );
                assert_eq!(s.reasoning_reservations(&state, "rank"), 600.0);
            }
            // A subsequent turn of the old Active/Acting session must recheck
            // execution reservations, rather than directly forwarding 1200/990.
            let again = enqueue(&s, "acting", 600);
            let mut state = s.state.lock();
            s.recheck_idle_execution_capacity(&mut state, again.program(), Instant::now());
            assert_eq!(
                state.runtime.view(again.program()).unwrap().state,
                ProgramState::Paused
            );
            assert_eq!(s.reasoning_reservations(&state, "rank"), 600.0);
        }
    }

    #[test]
    fn retained_active_turn_cannot_bypass_invalid_or_stale_usage() {
        for invalid in [true, false] {
            let s = scheduler(ProgramResumeOrder::DpAdmission);
            let a = enqueue(&s, "a", 600);
            observe(&s, 0.0);
            s.schedule_waiting(&mut s.state.lock(), Instant::now());
            if invalid {
                let mut observation = raw("rank", 0.0);
                observation.kv_cache_usage = None;
                s.apply_observations(s.begin_observation(&targets()), [observation]);
            }
            let now = if invalid {
                Instant::now()
            } else {
                Instant::now() + Duration::from_secs(60)
            };
            let mut state = s.state.lock();
            s.recheck_idle_execution_capacity(&mut state, a.program(), now);
            assert_eq!(
                state.runtime.view(a.program()).unwrap().state,
                ProgramState::Paused
            );
        }
    }

    #[test]
    fn epoch_preserves_new_charges_and_does_not_credit_logical_releases() {
        let s = scheduler(ProgramResumeOrder::DpAdmission);
        observe(&s, 0.0);
        {
            let mut state = s.state.lock();
            ProgramScheduler::adjust_usage(&mut state, "rank", 100.0);
        }
        let epoch = s.begin_observation(&targets());
        {
            let mut state = s.state.lock();
            ProgramScheduler::adjust_usage(&mut state, "rank", 200.0);
            ProgramScheduler::adjust_usage(&mut state, "rank", -200.0);
        }
        s.apply_observations(epoch, [raw("rank", 0.0)]);
        let state = s.state.lock();
        assert_eq!(
            s.dynamic_headroom(&state, "rank", Instant::now()),
            Some(790.0)
        );
    }

    #[test]
    fn missing_stale_invalid_and_full_observations_are_distinct_and_safe() {
        let s = scheduler(ProgramResumeOrder::Dp);
        let p = enqueue(&s, "forced", 1);
        observe(&s, 1.0);
        {
            let mut state = s.state.lock();
            state
                .decisions
                .get_mut(p.program())
                .unwrap()
                .force_resume_deadline = Some(Instant::now());
            assert_eq!(
                s.dynamic_headroom(&state, "rank", Instant::now()),
                Some(0.0)
            );
            assert!(!s.schedule_waiting(&mut state, Instant::now())); // force never invents free KV
        }
        observe(&s, 0.0);
        {
            let mut state = s.state.lock();
            state.observations.get_mut("rank").unwrap().observed_at =
                Some(Instant::now() - Duration::from_secs(4));
            assert_eq!(s.dynamic_headroom(&state, "rank", Instant::now()), None);
            assert!(!s.schedule_waiting(&mut state, Instant::now()));
        }
        let mut invalid = raw("rank", 0.0);
        invalid.token_capacity = None;
        s.apply_observations(s.begin_observation(&targets()), [invalid]);
        assert_eq!(
            s.dynamic_headroom(&s.state.lock(), "rank", Instant::now()),
            None
        );
        observe(&s, 0.0);
        assert!(s.schedule_waiting(&mut s.state.lock(), Instant::now()));
    }

    #[test]
    fn overdraw_keeps_pressure_instead_of_inflating_capacity_to_current_usage() {
        let s = scheduler(ProgramResumeOrder::DpAdmission);
        let _p = enqueue(&s, "active", 600);
        observe(&s, 0.0);
        let mut state = s.state.lock();
        assert!(s.schedule_waiting(&mut state, Instant::now()));
        assert_eq!(s.target_capacity(&state, "rank", Instant::now()), Some(990));
        ProgramScheduler::adjust_usage(&mut state, "rank", 500.0);
        assert_eq!(
            s.dynamic_headroom(&state, "rank", Instant::now()),
            Some(0.0)
        );
        assert_eq!(s.target_capacity(&state, "rank", Instant::now()), Some(490));
        assert!(
            s.target_capacity(&state, "rank", Instant::now()).unwrap()
                < s.ledger_usage(&state, "rank") as usize
        );
    }

    #[test]
    fn independent_ranks_do_not_share_a_budget_or_static_override() {
        let s = scheduler(ProgramResumeOrder::DpAdmission);
        let mut targets = targets();
        targets.push(ProgramTarget {
            id: "other".into(),
            base_url: "http://other".into(),
            dp_rank: Some(1),
        });
        s.sync_targets("m", &targets);
        let mut second = raw("other", 0.5);
        second.token_capacity = Some(2000);
        s.apply_observations(s.begin_observation(&targets), [raw("rank", 0.0), second]);
        let state = s.state.lock();
        assert_eq!(s.target_capacity(&state, "rank", Instant::now()), Some(990));
        assert_eq!(
            s.target_capacity(&state, "other", Instant::now()),
            Some(990)
        );
        assert_eq!(s.target_capacity(&state, "absent", Instant::now()), None);
    }

    #[test]
    fn stale_target_epoch_and_out_of_order_samples_cannot_restore_budget() {
        let s = scheduler(ProgramResumeOrder::DpAdmission);
        let old = raw("rank", 0.0);
        observe(&s, 1.0);
        s.apply_observations(s.begin_observation(&targets()), [old]);
        assert_eq!(
            s.dynamic_headroom(&s.state.lock(), "rank", Instant::now()),
            Some(0.0)
        );
        let epoch = s.begin_observation(&targets());
        s.sync_targets("m", &[]);
        s.sync_targets("m", &targets());
        s.apply_observations(epoch, [raw("rank", 0.0)]);
        assert_eq!(
            s.dynamic_headroom(&s.state.lock(), "rank", Instant::now()),
            None
        );
    }
}
