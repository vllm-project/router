//! Tree-affinity DP admission with conservative backend-usage accounting.
//! Reserve each unfinished request's tree-estimated cost in addition to observed
//! backend occupancy; do not assume a metrics scrape has covered backend waiters.
//! The tree supplies pair values, never a binding or a native admission gate.
use super::resume_dp::{self, OptionValue};
use super::scheduler_admission::RankAdmissionPlan;
use super::scheduler_state::{ProgramPauseReason, ProgramSchedulerState};
use super::{
    ProgramBindingStrategy, ProgramResumeOrder, ProgramScheduler, ProgramState, ProgramStatus,
};
use std::time::Instant;

impl ProgramScheduler {
    pub(crate) fn kv_dp_mode(&self) -> bool {
        self.config.dynamic_token_capacity
            && !self.config.binding_only
            && self.config.resume_order == ProgramResumeOrder::DpAdmissionTAO
            && self.config.binding_strategy == ProgramBindingStrategy::CacheAware
    }

    pub(crate) fn tree_dp_capacity(
        &self,
        state: &ProgramSchedulerState,
        target: &str,
        now: Instant,
    ) -> Option<usize> {
        let o = state.observations.get(target)?;
        (self.observation_is_fresh(o, now) && self.tree_dp_backend_used(state, target).is_some())
            .then_some(o.backend.backend_token_capacity)
            .flatten()
            .filter(|c| *c > 0)
    }

    // Freeze the selected cost until the request finishes. Inserting its prefix
    // into the tree must not turn its outstanding reservation into a zero cost.
    pub(crate) fn tree_dp_used(&self, state: &ProgramSchedulerState, target: &str) -> usize {
        state
            .runtime
            .iter_views()
            .filter(|v| {
                v.state == ProgramState::Active
                    && v.placement == Some(target)
                    && (v.in_flight_requests > 0 || v.waiting_requests > 0)
            })
            .filter_map(|v| state.decisions.get(v.reference)?.tree_dp_reserved_tokens) // TODO: predicted length & online observation can be used to better reserve
            .fold(0usize, usize::saturating_add)
    }

    /// Observed unavailable reference tokens, including the null-block correction
    /// already applied by the metrics converter. Missing/invalid usage is unknown,
    /// not an empty backend. Freshness is checked by tree_dp_capacity.
    pub(crate) fn tree_dp_backend_used(
        &self,
        state: &ProgramSchedulerState,
        target: &str,
    ) -> Option<usize> {
        let o = state.observations.get(target)?;
        let total = o.backend.backend_token_capacity.filter(|c| *c > 0)?;
        let available = o.backend.backend_available_tokens?;
        if !available.is_finite() || available < 0.0 || available > total as f64 {
            return None;
        }
        Some(total.saturating_sub(available.floor() as usize))
    }

    /// Conservative trial: some in-flight KV can be counted twice. Keep every
    /// outstanding reservation until completion/cancellation rather than guess
    /// whether an aggregate scrape includes that request's allocations.
    pub(crate) fn tree_dp_accounted_used(
        &self,
        state: &ProgramSchedulerState,
        target: &str,
    ) -> usize {
        self.tree_dp_backend_used(state, target)
            .unwrap_or(0)
            .saturating_add(self.tree_dp_used(state, target))
    }

    pub(crate) fn schedule_tree_dp(&self, state: &mut ProgramSchedulerState, now: Instant) -> bool {
        // Returning active placements retain continuity when their next turn fits.
        // Keep a selected-but-not-yet-dispatched turn active; serialize requests
        // of one program so one reservation cannot cover multiple forwards.
        let returning: Vec<_> = state
            .runtime
            .iter_views()
            .filter(|v| {
                v.state == ProgramState::Active
                    && v.in_flight_requests == 0
                    && v.waiting_requests > 0
                    && state
                        .decisions
                        .get(v.reference)
                        .is_some_and(|d| d.tree_dp_reserved_tokens.is_none())
            })
            .map(|v| v.reference.clone())
            .collect();
        for p in returning {
            // A return can arrive between ticks, after retain_request changed
            // Acting to Reasoning. Do not revive an already expired lease.
            if state.decisions[&p]
                .ttl_deadline
                .is_some_and(|deadline| deadline <= now)
            {
                self.pause_idle(state, &p, ProgramPauseReason::TtlExpired, now);
                continue;
            }
            let target = state.runtime.view(&p).and_then(|v| v.placement).unwrap();
            let total = state.runtime.front_request_context(&p).unwrap_or(0);
            let text = state.runtime.front_routing_text(&p).unwrap_or_default();
            let ratio = state
                .bindings
                .prefix_match_rate(p.model_pool(), text, &target);
            let cached = if ratio.is_finite() {
                (total as f64 * ratio.clamp(0.0, 1.0)).floor() as usize
            } else {
                0
            };
            let cost = total.saturating_sub(cached.min(total));
            let fits = self
                .tree_dp_capacity(state, &target, now)
                .is_some_and(|capacity| {
                    cost <= capacity.saturating_sub(self.tree_dp_accounted_used(state, &target))
                });
            if fits {
                let decision = state.decisions.get_mut(&p).unwrap();
                decision.tree_dp_reserved_tokens = Some(cost);
                decision.ttl_deadline = None;
                state.runtime.notify_front(&p);
            } else {
                self.pause_idle(state, &p, ProgramPauseReason::CapacityRepair, now);
            }
        }
        let mut pools: Vec<_> = state.model_targets.keys().cloned().collect();
        pools.sort();
        let mut changed = false;
        for pool in pools {
            let targets: Vec<_> = state.model_targets[&pool]
                .iter()
                .filter(|t| {
                    state.targets.contains_key(*t) && self.tree_dp_capacity(state, t, now).is_some()
                })
                .cloned()
                .collect();
            let mut queued: Vec<_> = state
                .runtime
                .iter_views()
                .filter(|v| {
                    v.reference.model_pool() == pool
                        && v.state == ProgramState::Paused
                        && v.status == ProgramStatus::Reasoning
                        && v.waiting_requests > 0
                })
                .map(|v| v.reference.clone())
                .collect();
            queued.sort_by(|a, b| a.program_id().cmp(b.program_id()));
            let mut budgets: Vec<_> = targets
                .iter()
                .map(|t| {
                    (
                        self.tree_dp_capacity(state, t, now)
                            .unwrap()
                            .saturating_sub(self.tree_dp_accounted_used(state, t)),
                        queued.len(),
                    )
                })
                .collect();
            let mut assignments = Vec::new();
            let mut options: Vec<Vec<OptionValue>> = queued
                .iter()
                .map(|p| {
                    let total = state.runtime.front_request_context(p).unwrap_or(0);
                    let text = state.runtime.front_routing_text(p).unwrap_or_default();
                    targets
                        .iter()
                        .enumerate()
                        .map(|(target_index, t)| {
                            let ratio = state.bindings.prefix_match_rate(&pool, text, t);
                            let cached = if ratio.is_finite() {
                                (total as f64 * ratio.clamp(0.0, 1.0)).floor() as usize
                            } else {
                                0
                            };
                            let cached = cached.min(total);
                            let cost = total - cached;
                            let index = assignments.len();
                            assignments.push((p.clone(), t.clone(), total, cached, cost));
                            OptionValue {
                                target: target_index,
                                tokens: cost,
                                value: (cached as u128, total as u128),
                                index,
                            }
                        })
                        .collect()
                })
                .collect();
            // Deadlines affect admission order, never permission to exceed
            // capacity. Choose the best feasible backend for each overdue
            // request, then optimize the remaining candidates and budget.
            let mut overdue: Vec<_> = queued
                .iter()
                .enumerate()
                .filter_map(|(index, program)| {
                    state.decisions[program]
                        .force_resume_deadline
                        .filter(|deadline| *deadline <= now)
                        .map(|deadline| (deadline, index))
                })
                .collect();
            overdue.sort_unstable();
            let mut selected = Vec::new();
            for (_, index) in overdue {
                let chosen = options[index]
                    .iter()
                    .filter(|option| {
                        let (tokens, slots) = budgets[option.target];
                        slots > 0 && option.tokens <= tokens
                    })
                    .max_by_key(|option| (option.value, std::cmp::Reverse(option.target)));
                if let Some(option) = chosen {
                    budgets[option.target].0 -= option.tokens;
                    budgets[option.target].1 -= 1;
                    selected.push((option.index, true));
                    options[index].clear();
                }
            }
            selected.extend(
                resume_dp::solve(&options, &budgets, &self.config)
                    .into_iter()
                    .map(|index| (index, false)),
            );
            for (index, overdue) in selected {
                let (p, t, total, cached, cost) = &assignments[index];
                let plan = RankAdmissionPlan {
                    used_tokens: self.tree_dp_accounted_used(state, t) as f64,
                    required_tokens: *cost as f64,
                    reserve_tokens: 0.0,
                    capacity_tokens: self.tree_dp_capacity(state, t, now),
                    forced: overdue,
                    batch_gain: false,
                };
                if self.commit_admission(state, p, t, plan, now) {
                    state.decisions.get_mut(p).unwrap().tree_dp_reserved_tokens = Some(*cost);
                    tracing::info!(event="tree_dp_admission",program=%p.redacted_id(),target=t,
                        total_tokens=total,estimated_cached_tokens=cached,reserved_tokens=cost,
                        backend_token_capacity=?self.tree_dp_capacity(state,t,now),
                        backend_unavailable_tokens=?self.tree_dp_backend_used(state,t),
                        router_reserved_tokens=self.tree_dp_used(state,t),
                        admission_headroom_tokens=?self.dynamic_headroom(state,t,now),
                        "DP selected program/backend pair from prefix-tree value");
                    changed = true;
                }
            }
        }
        changed
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::super::{
        BackendObservation, ProgramIdentity, ProgramRequestHandle, ProgramSchedulerConfig,
        ProgramTarget, ProgramUsageObservation,
    };
    use super::*;
    use serde_json::json;
    fn setup(capacities: &[usize]) -> ProgramScheduler {
        let mut cfg = ProgramSchedulerConfig {
            dynamic_token_capacity: true,
            resume_order: ProgramResumeOrder::DpAdmissionTAO,
            binding_strategy: ProgramBindingStrategy::CacheAware,
            dp_capacity_quantum: 1,
            max_active_programs_per_target: 1,
            ..Default::default()
        };
        cfg.progress_ttl.decode_buffer_tokens = 99999; // Not an input to this cost.
        let s = ProgramScheduler::new(cfg);
        let targets: Vec<_> = capacities
            .iter()
            .enumerate()
            .map(|(i, _)| ProgramTarget {
                id: format!("t{i}"),
                base_url: format!("http://t{i}"),
                dp_rank: None,
            })
            .collect();
        s.sync_targets("m", &targets);
        s.apply_observations(
            s.begin_observation(&targets),
            capacities
                .iter()
                .enumerate()
                .map(|(i, c)| BackendObservation {
                    target_id: format!("t{i}"),
                    base_url: format!("http://t{i}"),
                    dp_rank: None,
                    token_capacity: Some(*c + 1),
                    num_gpu_blocks: Some(*c + 1),
                    kv_cache_usage: Some(0.0),
                    running_requests: Some(100),
                    waiting_requests: Some(100),
                    observed_at: Instant::now(),
                }),
        );
        s
    }
    fn enqueue(
        s: &ProgramScheduler,
        id: &str,
        total: impl Into<super::super::ContextTokens>,
        text: &str,
    ) -> ProgramRequestHandle {
        let total = total.into();
        let identity = ProgramIdentity::from_request(
            None,
            Some(
                &json!({"vllm_xargs":{"agentic_context":{"program_id":id,"expected_resume":true}}}),
            ),
            Some("m"),
        )
        .unwrap()
        .unwrap();
        let mut st = s.state.lock();
        let now = Instant::now();
        let h = st.runtime.retain_request(&identity, total, Some(text), now);
        s.ensure_decision_state(
            &mut st,
            &identity,
            h.program().clone(),
            total,
            now,
            Some(text),
        );
        h
    }
    fn seed(s: &ProgramScheduler, h: &ProgramRequestHandle, text: &str, target: &str) {
        s.state
            .lock()
            .bindings
            .commit_placement(h.program(), text, target);
    }
    #[test]
    fn kv_admission_uses_the_applied_input_fact_not_the_original_estimate() {
        use super::super::{ContextTokens, TokenFacts, TokenizerFingerprint};
        let fingerprint = TokenizerFingerprint::from_parts(&["model", "prepared"]);
        // Arrival resolution has already selected which fact is applicable.
        let exact =
            ContextTokens::from_prepared(TokenFacts::client_token_ids(60, fingerprint), 600);
        for (context, admitted) in [(exact, true), (ContextTokens::estimated(600), false)] {
            let s = setup(&[100]);
            let handle = enqueue(&s, "fact", context, "prompt");
            let mut state = s.state.lock();
            assert_eq!(s.schedule_waiting(&mut state, Instant::now()), admitted);
            assert_eq!(
                state.runtime.placement(handle.program()).is_some(),
                admitted
            );
            assert_eq!(
                state.decisions[handle.program()].tree_dp_reserved_tokens,
                admitted.then_some(60),
            );
        }
    }

    #[test]
    fn overdue_requests_use_deadline_order_before_cache_value() {
        let s = setup(&[100]);
        let older = enqueue(&s, "z-older", 60, "older");
        let later = enqueue(&s, "a-later", 300, "abcdefghij");
        seed(&s, &later, "abcdefgh", "t0");
        let now = Instant::now();
        let mut st = s.state.lock();
        st.decisions
            .get_mut(older.program())
            .unwrap()
            .force_resume_deadline = Some(now - std::time::Duration::from_secs(2));
        st.decisions
            .get_mut(later.program())
            .unwrap()
            .force_resume_deadline = Some(now - std::time::Duration::from_secs(1));
        assert!(s.schedule_waiting(&mut st, now));
        assert_eq!(st.runtime.placement(older.program()), Some("t0"));
        assert_eq!(st.runtime.placement(later.program()), None);
        assert_eq!(s.tree_dp_used(&st, "t0"), 60);
    }

    #[test]
    fn overdue_oversized_request_does_not_block_remaining_dp_budget() {
        let s = setup(&[100]);
        let oversized = enqueue(&s, "oversized", 101, "oversized");
        let due = enqueue(&s, "due", 60, "due");
        let hit = enqueue(&s, "hit", 200, "abcdefghij");
        let miss = enqueue(&s, "miss", 40, "other");
        seed(&s, &hit, "abcdefgh", "t0");
        let now = Instant::now();
        let mut st = s.state.lock();
        st.decisions
            .get_mut(oversized.program())
            .unwrap()
            .force_resume_deadline = Some(now - std::time::Duration::from_secs(2));
        st.decisions
            .get_mut(due.program())
            .unwrap()
            .force_resume_deadline = Some(now);
        st.decisions
            .get_mut(hit.program())
            .unwrap()
            .force_resume_deadline = Some(now + std::time::Duration::from_secs(1));
        assert!(s.schedule_waiting(&mut st, now));
        assert_eq!(st.runtime.placement(oversized.program()), None);
        assert_eq!(st.runtime.placement(due.program()), Some("t0"));
        assert_eq!(st.runtime.placement(hit.program()), Some("t0"));
        assert_eq!(st.runtime.placement(miss.program()), None);
        assert_eq!(s.tree_dp_used(&st, "t0"), 100);
    }

    #[test]
    fn overdue_request_uses_best_feasible_backend_without_dp_quantization() {
        let mut s = setup(&[40, 80, 80]);
        s.config.dp_capacity_quantum = 512;
        let due = enqueue(&s, "due", 100, "abcdefghij");
        seed(&s, &due, "abcde", "t0");
        seed(&s, &due, "abcd", "t1");
        seed(&s, &due, "abc", "t2");
        let now = Instant::now();
        let mut st = s.state.lock();
        st.decisions
            .get_mut(due.program())
            .unwrap()
            .force_resume_deadline = Some(now);
        assert!(s.schedule_waiting(&mut st, now));
        assert_eq!(st.runtime.placement(due.program()), Some("t1"));
        assert_eq!(s.tree_dp_used(&st, "t0"), 0);
        assert_eq!(s.tree_dp_used(&st, "t1"), 60);
        assert_eq!(s.tree_dp_used(&st, "t2"), 0);
    }

    #[test]
    fn overdue_request_still_requires_fresh_backend_capacity() {
        let s = setup(&[100]);
        let due = enqueue(&s, "due", 60, "due");
        let now = Instant::now();
        let mut st = s.state.lock();
        st.decisions
            .get_mut(due.program())
            .unwrap()
            .force_resume_deadline = Some(now);
        assert!(!s.schedule_waiting(&mut st, now + s.config.metrics_interval * 4));
        assert_eq!(st.runtime.placement(due.program()), None);
        assert_eq!(s.tree_dp_used(&st, "t0"), 0);
    }

    #[test]
    fn tree_cost_ignores_prior_cached_feedback_and_native_request_count_gates() {
        let s = setup(&[100]);
        let h = enqueue(&s, "p", 1000, "abcdefghij");
        seed(&s, &h, "abcdefghi", "t0");
        s.state
            .lock()
            .decisions
            .get_mut(h.program())
            .unwrap()
            .shared_prefix_tokens = 0;
        let mut st = s.state.lock();
        assert!(s.schedule_waiting(&mut st, Instant::now()));
        assert_eq!(st.decisions[h.program()].tree_dp_reserved_tokens, Some(100));
        assert_eq!(s.target_capacity(&st, "t0", Instant::now()), Some(101));
        // Once dispatched, insertion of its full prefix cannot shrink its cost.
        st.bindings
            .commit_placement(h.program(), "abcdefghij", "t0");
        assert_eq!(s.tree_dp_used(&st, "t0"), 100);
    }
    #[test]
    fn old_cached_feedback_cannot_admit_a_tree_miss() {
        let s = setup(&[100]);
        let h = enqueue(&s, "p", 1000, "miss");
        s.state
            .lock()
            .decisions
            .get_mut(h.program())
            .unwrap()
            .shared_prefix_tokens = 1000;
        let mut st = s.state.lock();
        s.schedule_waiting(&mut st, Instant::now());
        assert_eq!(
            st.runtime.view(h.program()).unwrap().state,
            ProgramState::Paused
        );
    }
    #[test]
    fn dp_can_choose_backend_other_than_cache_aware_initial_binding() {
        let s = setup(&[5, 100]);
        let h = enqueue(&s, "p", 100, "abcdefghij");
        seed(&s, &h, "abcdefgh", "t0");
        s.state
            .lock()
            .decisions
            .get_mut(h.program())
            .unwrap()
            .last_target = Some("t0".into());
        let mut st = s.state.lock();
        assert!(s.schedule_waiting(&mut st, Instant::now()));
        assert_eq!(
            st.runtime.view(h.program()).unwrap().placement.as_deref(),
            Some("t1")
        );
        assert_eq!(st.decisions[h.program()].tree_dp_reserved_tokens, Some(100));
    }
    #[test]
    fn dp_prefers_tree_hit_value_over_larger_total_and_ignores_count_gate() {
        let s = setup(&[100]);
        let hit = enqueue(&s, "z-hit", 500, "abcdefghij");
        let miss = enqueue(&s, "a-miss", 100, "0123456789");
        seed(&s, &hit, "abcdefgh", "t0");
        let mut st = s.state.lock();
        s.schedule_waiting(&mut st, Instant::now());
        assert_eq!(
            st.runtime.view(hit.program()).unwrap().state,
            ProgramState::Active
        );
        assert_eq!(
            st.runtime.view(miss.program()).unwrap().state,
            ProgramState::Paused
        );
        drop(st);
        let free = enqueue(&s, "free", 100, "abcdefgh");
        let mut st = s.state.lock();
        s.schedule_waiting(&mut st, Instant::now());
        assert_eq!(
            st.runtime.view(free.program()).unwrap().state,
            ProgramState::Active
        );
        assert_eq!(s.tree_dp_used(&st, "t0"), 100);
    }
    #[test]
    fn completion_releases_frozen_cost_and_returning_turn_goes_through_dp() {
        let s = setup(&[100]);
        let a = enqueue(&s, "a", 100, "aaa");
        s.schedule_waiting(&mut s.state.lock(), Instant::now());
        let d = s
            .state
            .lock()
            .runtime
            .admit_front(&a, "t0".into(), Instant::now())
            .unwrap();
        let b = enqueue(&s, "b", 100, "bbb");
        s.schedule_waiting(&mut s.state.lock(), Instant::now());
        assert_eq!(
            s.state.lock().runtime.view(b.program()).unwrap().state,
            ProgramState::Paused
        );
        s.complete(&d, true, false, ProgramUsageObservation::default(), None);
        assert_eq!(
            s.state.lock().runtime.view(b.program()).unwrap().state,
            ProgramState::Active
        );
        let again = enqueue(&s, "a", 100, "new");
        let mut st = s.state.lock();
        s.schedule_waiting(&mut st, Instant::now());
        assert_eq!(
            st.runtime.view(again.program()).unwrap().state,
            ProgramState::Paused
        );
        assert_eq!(s.tree_dp_used(&st, "t0"), 100);
    }
    #[test]
    fn active_return_keeps_backend_even_when_other_tree_has_more_hits() {
        let s = setup(&[200, 200]);
        let first = enqueue(&s, "p", 100, "old");
        s.schedule_waiting(&mut s.state.lock(), Instant::now());
        let target = s
            .state
            .lock()
            .runtime
            .view(first.program())
            .unwrap()
            .placement
            .unwrap();
        let other = if target == "t0" { "t1" } else { "t0" };
        let dispatch = s
            .state
            .lock()
            .runtime
            .admit_front(&first, target.clone(), Instant::now())
            .unwrap();
        // Complete only runtime state to isolate a live, unexpired Acting placement.
        {
            let mut st = s.state.lock();
            assert!(st
                .runtime
                .complete_request(&dispatch, ProgramUsageObservation::default()));
            let d = st.decisions.get_mut(first.program()).unwrap();
            d.tree_dp_reserved_tokens = None;
            d.ttl_deadline = Some(Instant::now() + std::time::Duration::from_secs(10));
        }
        let again = enqueue(&s, "p", 100, "new");
        seed(&s, &again, "new", other);
        let mut st = s.state.lock();
        s.schedule_waiting(&mut st, Instant::now());
        assert_eq!(
            st.runtime
                .view(again.program())
                .unwrap()
                .placement
                .as_deref(),
            Some(target.as_str())
        );
        assert_eq!(
            st.decisions[again.program()].tree_dp_reserved_tokens,
            Some(100)
        );
        assert!(st.decisions[again.program()].last_pause_reason.is_none());
        assert!(st.decisions[again.program()].ttl_deadline.is_none());
        s.schedule_waiting(&mut st, Instant::now());
        assert_eq!(s.tree_dp_used(&st, &target), 100);
    }
    #[test]
    fn expired_acting_placement_still_reenters_dp() {
        let s = setup(&[200, 200]);
        let first = enqueue(&s, "p", 100, "old");
        s.schedule_waiting(&mut s.state.lock(), Instant::now());
        let target = s
            .state
            .lock()
            .runtime
            .view(first.program())
            .unwrap()
            .placement
            .unwrap();
        let other = if target == "t0" { "t1" } else { "t0" };
        let dispatch = s
            .state
            .lock()
            .runtime
            .admit_front(&first, target.clone(), Instant::now())
            .unwrap();
        {
            let mut st = s.state.lock();
            assert!(st
                .runtime
                .complete_request(&dispatch, ProgramUsageObservation::default()));
            let d = st.decisions.get_mut(first.program()).unwrap();
            d.tree_dp_reserved_tokens = None;
            d.ttl_deadline = Some(Instant::now());
        }
        s.tick();
        assert_eq!(
            s.state.lock().runtime.view(first.program()).unwrap().state,
            ProgramState::Paused
        );
        let again = enqueue(&s, "p", 100, "new");
        seed(&s, &again, "new", other);
        let mut st = s.state.lock();
        s.schedule_waiting(&mut st, Instant::now());
        assert_eq!(
            st.runtime
                .view(again.program())
                .unwrap()
                .placement
                .as_deref(),
            Some(other)
        );
    }
    #[test]
    fn stale_capacity_blocks_even_full_tree_hit() {
        let s = setup(&[100]);
        let h = enqueue(&s, "p", 100, "abc");
        seed(&s, &h, "abc", "t0");
        let mut st = s.state.lock();
        s.schedule_waiting(&mut st, Instant::now() + std::time::Duration::from_secs(60));
        assert_eq!(
            st.runtime.view(h.program()).unwrap().state,
            ProgramState::Paused
        );
    }
    #[test]
    fn total_token_value_breaks_equal_hit_ties() {
        let s = setup(&[100]);
        let small = enqueue(&s, "a", 60, "aaa");
        let large = enqueue(&s, "z", 100, "zzz");
        let mut st = s.state.lock();
        s.schedule_waiting(&mut st, Instant::now());
        assert_eq!(
            st.runtime.view(small.program()).unwrap().state,
            ProgramState::Paused
        );
        assert_eq!(
            st.runtime.view(large.program()).unwrap().state,
            ProgramState::Active
        );
    }
    #[test]
    fn repeated_observations_do_not_erase_dp_reservations() {
        let s = setup(&[100]);
        let a = enqueue(&s, "a", 70, "aaa");
        s.schedule_waiting(&mut s.state.lock(), Instant::now());
        let b = enqueue(&s, "b", 60, "bbb");
        for usage in [0.0, 1.0] {
            let targets = vec![ProgramTarget {
                id: "t0".into(),
                base_url: "http://t0".into(),
                dp_rank: None,
            }];
            s.apply_observations(
                s.begin_observation(&targets),
                [BackendObservation {
                    target_id: "t0".into(),
                    base_url: "http://t0".into(),
                    dp_rank: None,
                    token_capacity: Some(101),
                    num_gpu_blocks: Some(101),
                    kv_cache_usage: Some(usage),
                    running_requests: Some(0),
                    waiting_requests: Some(0),
                    observed_at: Instant::now(),
                }],
            );
            let mut st = s.state.lock();
            s.schedule_waiting(&mut st, Instant::now());
            assert_eq!(
                st.runtime.view(a.program()).unwrap().state,
                ProgramState::Active
            );
            assert_eq!(
                st.runtime.view(b.program()).unwrap().state,
                ProgramState::Paused
            );
            assert_eq!(
                s.dynamic_headroom(&st, "t0", Instant::now()),
                Some(if usage == 0.0 { 30.0 } else { 0.0 })
            );
        }
    }

    fn observe_usage(s: &ProgramScheduler, target: &str, capacity: usize, usage: Option<f64>) {
        let targets = [ProgramTarget {
            id: target.into(),
            base_url: format!("http://{target}"),
            dp_rank: None,
        }];
        s.apply_observations(
            s.begin_observation(&targets),
            [BackendObservation {
                target_id: target.into(),
                base_url: format!("http://{target}"),
                dp_rank: None,
                token_capacity: Some(capacity + 1),
                num_gpu_blocks: Some(capacity + 1),
                kv_cache_usage: usage,
                running_requests: Some(0),
                waiting_requests: Some(0),
                observed_at: Instant::now(),
            }],
        );
    }

    #[test]
    fn conservative_backend_usage_blocks_admission_without_router_reservations() {
        let s = setup(&[100]);
        observe_usage(&s, "t0", 100, Some(1.0));
        let h = enqueue(&s, "cold", 1, "cold");
        let mut st = s.state.lock();
        s.schedule_waiting(&mut st, Instant::now());
        assert_eq!(
            st.runtime.view(h.program()).unwrap().state,
            ProgramState::Paused
        );
        assert_eq!(s.tree_dp_used(&st, "t0"), 0);
        assert_eq!(s.dynamic_headroom(&st, "t0", Instant::now()), Some(0.0));
    }

    #[test]
    fn conservative_backend_usage_and_reservations_survive_scrapes() {
        let s = setup(&[100]);
        observe_usage(&s, "t0", 100, Some(0.5));
        let a = enqueue(&s, "a", 30, "aaa");
        s.schedule_waiting(&mut s.state.lock(), Instant::now());
        let b = enqueue(&s, "b", 30, "bbb");
        {
            let mut st = s.state.lock();
            s.schedule_waiting(&mut st, Instant::now());
            assert_eq!(
                st.runtime.view(a.program()).unwrap().state,
                ProgramState::Active
            );
            assert_eq!(
                st.runtime.view(b.program()).unwrap().state,
                ProgramState::Paused
            );
            assert_eq!(s.dynamic_headroom(&st, "t0", Instant::now()), Some(20.0));
        }
        observe_usage(&s, "t0", 100, Some(0.75));
        let st = s.state.lock();
        assert_eq!(s.tree_dp_used(&st, "t0"), 30);
        assert_eq!(s.dynamic_headroom(&st, "t0", Instant::now()), Some(0.0));
    }

    #[test]
    fn conservative_backend_usage_is_target_specific() {
        let s = setup(&[100, 100]);
        observe_usage(&s, "t0", 100, Some(1.0));
        observe_usage(&s, "t1", 100, Some(0.0));
        let h = enqueue(&s, "p", 100, "fresh");
        let mut st = s.state.lock();
        s.schedule_waiting(&mut st, Instant::now());
        assert_eq!(
            st.runtime.view(h.program()).unwrap().placement.as_deref(),
            Some("t1")
        );
    }

    #[test]
    fn conservative_backend_usage_requires_usage_even_on_full_tree_hit() {
        for usage in [None, Some(f64::NAN), Some(-0.1), Some(1.1)] {
            let s = setup(&[100]);
            observe_usage(&s, "t0", 100, usage);
            let h = enqueue(&s, "p", 100, "prefix");
            seed(&s, &h, "prefix", "t0");
            let mut st = s.state.lock();
            s.schedule_waiting(&mut st, Instant::now());
            assert_eq!(
                st.runtime.view(h.program()).unwrap().state,
                ProgramState::Paused
            );
            assert_eq!(s.dynamic_headroom(&st, "t0", Instant::now()), None);
        }
    }

    #[test]
    fn conservative_completion_releases_reservation_but_not_backend_usage() {
        let s = setup(&[100]);
        observe_usage(&s, "t0", 100, Some(0.5));
        let h = enqueue(&s, "a", 30, "aaa");
        s.schedule_waiting(&mut s.state.lock(), Instant::now());
        let dispatch = s
            .state
            .lock()
            .runtime
            .admit_front(&h, "t0".into(), Instant::now())
            .unwrap();
        assert_eq!(s.tree_dp_used(&s.state.lock(), "t0"), 30);
        s.complete(
            &dispatch,
            true,
            false,
            ProgramUsageObservation::default(),
            None,
        );
        let st = s.state.lock();
        assert_eq!(s.tree_dp_used(&st, "t0"), 0);
        assert_eq!(s.dynamic_headroom(&st, "t0", Instant::now()), Some(50.0));
    }

    #[test]
    fn conservative_returning_turn_cannot_bypass_backend_usage() {
        let s = setup(&[100]);
        observe_usage(&s, "t0", 100, Some(0.0));
        let h = enqueue(&s, "a", 30, "aaa");
        s.schedule_waiting(&mut s.state.lock(), Instant::now());
        let dispatch = s
            .state
            .lock()
            .runtime
            .admit_front(&h, "t0".into(), Instant::now())
            .unwrap();
        {
            let mut st = s.state.lock();
            st.runtime
                .complete_request(&dispatch, ProgramUsageObservation::default());
            let d = st.decisions.get_mut(h.program()).unwrap();
            d.tree_dp_reserved_tokens = None;
            d.ttl_deadline = Some(Instant::now() + std::time::Duration::from_secs(10));
        }
        observe_usage(&s, "t0", 100, Some(1.0));
        let next = enqueue(&s, "a", 30, "bbb");
        let mut st = s.state.lock();
        s.schedule_waiting(&mut st, Instant::now());
        assert_eq!(
            st.runtime.view(next.program()).unwrap().state,
            ProgramState::Paused
        );
        assert_eq!(
            st.decisions[next.program()].last_pause_reason,
            Some(ProgramPauseReason::CapacityRepair)
        );
    }
}
