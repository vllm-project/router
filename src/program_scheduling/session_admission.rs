//! Capacity-first placement for the head turn of an opted-in HTTP session.
use super::scheduler_state::ProgramSchedulerState;
use super::{
    ProgramAdmission, ProgramBindingCandidate, ProgramIdentity, ProgramRef, ProgramRequestHandle,
    ProgramRouter, ScheduleError,
};
use std::time::Instant;

/// The HTTP request and KV index stay outside scheduler state.
#[async_trait::async_trait]
pub(crate) trait SessionRankScorer: Sync {
    async fn recommend(
        &self,
        candidates: &[ProgramBindingCandidate],
    ) -> Result<Option<String>, ScheduleError>;
}

pub(super) struct BindingTask {
    id: u64,
    topology: u64,
    pub(super) revision: u64,
    handle: ProgramRequestHandle,
    identity: ProgramIdentity,
    candidates: Vec<ProgramBindingCandidate>,
    deadline: Instant,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Commit {
    Installed,
    AlreadyInstalled,
    Retry,
}

impl ProgramAdmission {
    pub(super) fn enqueue_capacity_waiter(
        &self,
        state: &mut ProgramSchedulerState,
        program: &ProgramRef,
    ) {
        let queue = state
            .capacity_waiters
            .entry(program.model_pool().into())
            .or_default();
        if !queue.contains(program) {
            queue.push_back(program.clone());
        }
    }

    pub(super) fn remove_capacity_waiter(state: &mut ProgramSchedulerState, program: &ProgramRef) {
        if let Some(queue) = state.capacity_waiters.get_mut(program.model_pool()) {
            queue.retain(|p| p != program);
            if queue.is_empty() {
                state.capacity_waiters.remove(program.model_pool());
            }
        }
    }

    pub(super) fn wake_capacity_waiters(state: &mut ProgramSchedulerState) {
        // Rotate a bounded batch, so oversized heads cannot monopolize wakeups.
        state.admission_revision = state.admission_revision.wrapping_add(1);
        for queue in state.capacity_waiters.values_mut() {
            let count = queue.len().min(32);
            for program in queue.iter().take(count) {
                state.runtime.notify_front(program);
            }
            queue.rotate_left(count);
        }
    }

    pub(super) fn binding_task(
        &self,
        handle: &ProgramRequestHandle,
        identity: &ProgramIdentity,
        deadline: Instant,
    ) -> Result<Option<BindingTask>, ScheduleError> {
        let mut state = self.state.lock();
        let now = Instant::now();
        if now >= deadline {
            return Err(ScheduleError::QueueTimeout);
        }
        let program = handle.program();
        if state
            .model_targets
            .get(program.model_pool())
            .is_none_or(|ranks| ranks.is_empty())
        {
            return Err(ScheduleError::NoTargets);
        }
        if !state.runtime.is_front(handle) {
            return Ok(None);
        }
        let decision = &state.decisions[program];
        if decision.last_target.is_some() || decision.binding_ticket.is_some() {
            return Ok(None);
        }
        // Preserve the initial placement key even if a later turn changes task metadata.
        let mut identity = identity.clone();
        identity.restore_placement_keys(&decision.placement_hash_key, &decision.placement_key);
        let mut candidates = self.binding_candidates(&state, &identity, now);
        metrics::counter!("router_program_admission_capacity_scans_total").increment(1);
        candidates.retain(|rank| {
            self.rank_admission_plan(&state, program, &rank.target_id, now)
                .is_some()
        });
        for candidate in &mut candidates {
            candidate.immediately_admissible = true;
        }
        metrics::histogram!("router_program_admission_eligible_ranks")
            .record(candidates.len() as f64);
        if candidates.is_empty() {
            return Ok(None);
        }
        state.next_ticket = state
            .next_ticket
            .checked_add(1)
            .expect("binding ticket exhausted");
        let id = state.next_ticket;
        state.decisions.get_mut(program).unwrap().binding_ticket = Some(id);
        Ok(Some(BindingTask {
            id,
            handle: handle.clone(),
            identity,
            candidates,
            deadline,
            topology: state.target_snapshot_revision,
            revision: state.admission_revision,
        }))
    }

    fn commit_ticket(
        &self,
        state: &mut ProgramSchedulerState,
        task: &BindingTask,
        rank: Option<&str>,
        cursor_valid: bool,
    ) -> Commit {
        let program = task.handle.program();
        let Some(decision) = state.decisions.get_mut(program) else {
            return Commit::Retry;
        };
        if decision.committed_ticket == Some(task.id) {
            return Commit::AlreadyInstalled;
        }
        if decision.binding_ticket != Some(task.id) {
            return Commit::Retry;
        }
        decision.binding_ticket = None;
        let now = Instant::now();
        if !cursor_valid
            || now >= task.deadline
            || state.target_snapshot_revision != task.topology
            || decision.last_target.is_some()
            || !state.runtime.is_front(&task.handle)
        {
            return Commit::Retry;
        }
        let Some(rank) = rank.filter(|id| task.candidates.iter().any(|c| c.target_id == *id))
        else {
            return Commit::Retry;
        };
        let Some(plan) = self.rank_admission_plan(state, program, rank, now) else {
            return Commit::Retry;
        };
        if !self.commit_admission(state, program, rank, plan, now) {
            return Commit::Retry;
        }
        state.bindings.commit(&task.identity, rank);
        let decision = state.decisions.get_mut(program).unwrap();
        decision.home_target = Some(rank.to_owned());
        decision.committed_ticket = Some(task.id);
        Self::remove_capacity_waiter(state, program);
        tracing::debug!(ticket=task.id, generation=?program.generation(), target=rank, "session binding committed");
        Commit::Installed
    }
}

impl ProgramRouter {
    pub(super) async fn select_and_commit_ticket(
        &self,
        task: &BindingTask,
        text: Option<&str>,
        scorer: Option<&dyn SessionRankScorer>,
    ) -> Result<(Option<super::ProgramDispatch>, bool), ScheduleError> {
        let (rank, cursor) = if let Some(scorer) = scorer {
            let scoring = scorer.recommend(&task.candidates);
            tokio::pin!(scoring);
            let rank = loop {
                tokio::select! {
                    result = &mut scoring => break result?,
                    _ = tokio::time::sleep_until(task.deadline.into()) => return Err(ScheduleError::QueueTimeout),
                    _ = task.handle.notified() => {
                        let state = self.admission.state.lock();
                        if state.decisions.get(task.handle.program()).and_then(|d| d.binding_ticket) != Some(task.id) {
                            return Ok((None, true));
                        }
                    }
                }
            };
            (rank, None)
        } else {
            let mut policy = self.policy.lock();
            let cursor = policy.cursor(task.identity.model_pool());
            (
                policy.recommend(&task.identity, &task.candidates, text),
                Some(cursor),
            )
        };
        // No await between commit, cursor confirmation and returning to the claim loop.
        // Topology/placement notifications and legacy cursor steps use this same order.
        let mut state = self.admission.state.lock();
        let mut policy = self.policy.lock();
        let pool = task.identity.model_pool();
        let cursor_valid = cursor.is_none_or(|c| c == policy.cursor(pool));
        let result = self
            .admission
            .commit_ticket(&mut state, task, rank.as_deref(), cursor_valid);
        let reason = match result {
            Commit::Installed => {
                if cursor.is_some() {
                    policy.confirm(pool);
                }
                "committed"
            }
            Commit::AlreadyInstalled => "replayed",
            Commit::Retry if rank.is_none() => "no_recommendation",
            Commit::Retry => "stale",
        };
        metrics::counter!("router_program_admission_selections_total", "result" => reason)
            .increment(1);
        // PD rechecks observation freshness when claiming. Its deadline can
        // expire even while the lock is held. Keep the committed binding and
        // wait for a fresh observation through the ordinary dispatch loop.
        let dispatch = if result == Commit::Installed {
            self.admission.claim_dispatch(&mut state, &task.handle)
        } else {
            None
        };
        if let Some(dispatch) = &dispatch {
            if let Some(text) = dispatch.routing_text() {
                policy.commit_placement(pool, text, &dispatch.target_id);
            }
        }
        Ok((dispatch, rank.is_some()))
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::program_scheduling::{
        ContextTokens, ProgramBindingStrategy, ProgramSchedulerConfig, ProgramSchedulingEnableKey,
        ProgramTarget,
    };
    use serde_json::json;
    use std::time::Duration;

    fn identity(session: &str, request: &str, explicit: bool) -> ProgramIdentity {
        let mut ctx = json!({"session_id":session,"task_id":"task","request_id":request,"expected_resume":true});
        if explicit {
            ctx["program_id"] = json!(session);
        }
        ProgramIdentity::resolve(
            None,
            Some(&json!({"vllm_xargs":{"agentic_context":ctx}})),
            Some("m"),
            ProgramSchedulingEnableKey::AgenticContext,
            true,
        )
        .unwrap()
        .unwrap()
    }
    fn rank(id: &str) -> ProgramTarget {
        ProgramTarget {
            id: id.into(),
            base_url: format!("http://{id}"),
            dp_rank: None,
        }
    }
    fn register(
        router: &ProgramRouter,
        identity: &ProgramIdentity,
        n: usize,
    ) -> ProgramRequestHandle {
        let mut state = router.admission.state.lock();
        let now = Instant::now();
        let (h, context) =
            router.register_attempt(&mut state, identity, ContextTokens::estimated(n), None, now);
        router
            .finish_registration(
                &mut state,
                identity,
                h.program().clone(),
                context,
                None,
                now,
            )
            .unwrap();
        h
    }
    fn router() -> ProgramRouter {
        ProgramRouter::new(ProgramSchedulerConfig {
            placement_policy: ProgramBindingStrategy::ProgramRoundRobin,
            max_active_programs_per_target: 1,
            metrics_interval: Duration::from_millis(10),
            queue_timeout: Duration::from_millis(250),
            ..Default::default()
        })
    }
    fn task(
        router: &ProgramRouter,
        h: &ProgramRequestHandle,
        id: &ProgramIdentity,
    ) -> Option<BindingTask> {
        router
            .binding_task(h, id, Instant::now() + Duration::from_secs(1))
            .unwrap()
    }

    #[tokio::test]
    async fn session_fifo_uses_head_demand_and_one_ticket_across_identity_forms() {
        let router = router();
        let ranks = [rank("b"), rank("a")];
        router.sync_targets("m", &ranks);
        let id = identity("S", "A1", false);
        let tail_id = identity("S", "A2", true);
        assert_eq!(id.program_id(), "S");
        assert_eq!(id.task_id(), Some("task"));
        let h = register(&router, &id, 10);
        let t = task(&router, &h, &id).unwrap();
        let tail = register(&router, &tail_id, 90);
        assert_eq!(h.program(), tail.program());
        assert!(task(&router, &h, &id).is_none());
        assert!(task(&router, &tail, &tail_id).is_none());
        assert!(router.next_dispatch(&tail).is_none());
        let (first, _) = router
            .select_and_commit_ticket(&t, None, None)
            .await
            .unwrap();
        let first = first.unwrap();
        assert_eq!(
            (first.target_id.as_str(), first.context_tokens()),
            ("a", 10)
        );
        assert_eq!(first.request_hints().request_id.as_deref(), Some("A1"));
        let (replay, _) = router
            .select_and_commit_ticket(&t, None, None)
            .await
            .unwrap();
        assert!(replay.is_none());
        assert_eq!(router.policy.lock().cursor("m"), 1);
        let second = router.next_dispatch(&tail).unwrap();
        assert_eq!(
            (
                second.target_id.as_str(),
                second.request_hints().request_id.as_deref()
            ),
            ("a", Some("A2"))
        );
        let other = identity("T", "B1", false);
        let oh = register(&router, &other, 10);
        let only = task(&router, &oh, &other).unwrap();
        assert_eq!(
            only.candidates
                .iter()
                .map(|c| c.target_id.as_str())
                .collect::<Vec<_>>(),
            ["b"]
        );
    }

    #[tokio::test]
    async fn same_sample_last_slot_duplicate_commit_and_cancel_before_claim() {
        let router = router();
        let ranks = [rank("a")];
        router.sync_targets("m", &ranks);
        let epoch = router.begin_observation(&ranks);
        router.apply_observations(
            epoch,
            [super::super::BackendObservation {
                target_id: "a".into(),
                base_url: "http://a".into(),
                dp_rank: None,
                kv_cache_usage: Some(0.0),
                running_requests: Some(0),
                waiting_requests: Some(0),
                observed_at: Instant::now(),
            }],
        );
        let a = identity("a", "a1", false);
        let b = identity("b", "b1", false);
        let ha = register(&router, &a, 50);
        let hb = register(&router, &b, 20);
        let ta = task(&router, &ha, &a).unwrap();
        let tb = task(&router, &hb, &b).unwrap();
        {
            let mut s = router.admission.state.lock();
            assert_eq!(
                router.commit_ticket(&mut s, &ta, Some("a"), true),
                Commit::Installed
            );
            assert_eq!(
                router.commit_ticket(&mut s, &ta, Some("a"), true),
                Commit::AlreadyInstalled
            );
            assert_eq!(
                router.commit_ticket(&mut s, &tb, Some("a"), true),
                Commit::Retry
            );
        }
        assert!(task(&router, &hb, &b).is_none());
        router.cancel_retained_request(&ha);
        let retry = task(&router, &hb, &b).unwrap();
        let (d, _) = router
            .select_and_commit_ticket(&retry, None, None)
            .await
            .unwrap();
        assert_eq!(d.unwrap().target_id, "a");
        assert_eq!(router.policy.lock().cursor("m"), 1);
    }

    #[tokio::test]
    async fn offline_ticket_cannot_bind_readded_target_and_idle_rebind_waits_for_demand() {
        let router = router();
        let ranks = [rank("a"), rank("b")];
        router.sync_targets("m", &ranks);
        let id = identity("s", "first", false);
        let h = register(&router, &id, 10);
        let t = task(&router, &h, &id).unwrap();
        router.sync_targets("m", &ranks[1..]);
        router.sync_targets("m", &ranks);
        let (stale, _) = router
            .select_and_commit_ticket(&t, None, None)
            .await
            .unwrap();
        assert!(stale.is_none());
        assert_eq!(router.policy.lock().cursor("m"), 0);
        let live = task(&router, &h, &id).unwrap();
        let (d, _) = router
            .select_and_commit_ticket(&live, None, None)
            .await
            .unwrap();
        let d = d.unwrap();
        router.complete(&d, true, false, Some(10), None);
        router.sync_targets("m", &ranks[1..]);
        assert_eq!(router.policy.lock().cursor("m"), 1);
        assert!(router.admission.state.lock().decisions[d.program()]
            .last_target
            .is_none());
        let continuation = router.acquire(id, 10, &ranks[1..], None).await.unwrap();
        assert_eq!(continuation.target_id, "b");
        router.complete(&d, true, true, Some(10000), None);
        assert!(router
            .admission
            .state
            .lock()
            .runtime
            .is_live_attempt(&continuation));
    }

    #[tokio::test]
    async fn full_pool_times_out_without_advancing_cursor_or_force_binding() {
        let router = ProgramRouter::new(ProgramSchedulerConfig {
            force_resume_timeout: Duration::from_millis(2),
            queue_timeout: Duration::from_millis(50),
            metrics_interval: Duration::from_millis(5),
            max_active_programs_per_target: 1,
            placement_policy: ProgramBindingStrategy::ProgramRoundRobin,
            ..Default::default()
        });
        let ranks = [rank("a")];
        let d = router
            .acquire(identity("a", "1", false), 10, &ranks, None)
            .await
            .unwrap();
        let err = router
            .acquire(identity("b", "1", false), 10, &ranks, None)
            .await
            .unwrap_err();
        assert_eq!(err, ScheduleError::QueueTimeout);
        assert!(router.admission.state.lock().capacity_waiters.is_empty());
        assert_eq!(router.policy.lock().cursor("m"), 1);
        router.complete(&d, true, true, None, None);
    }

    #[tokio::test]
    async fn waiting_growth_stays_on_current_rank_and_tail_cannot_overtake() {
        let mut config = ProgramSchedulerConfig {
            placement_policy: ProgramBindingStrategy::ProgramRoundRobin,
            ..Default::default()
        };
        config.progress_ttl.token_capacity = Some(200);
        let router = ProgramRouter::new(config);
        let ranks = [rank("a"), rank("b")];
        let a = router
            .acquire(identity("s", "1", false), 20, &ranks, None)
            .await
            .unwrap();
        router.sync_targets("m", &ranks);
        let large = register(&router, &identity("s", "2", true), 150);
        let small = register(&router, &identity("s", "3", false), 10);
        assert!(router.next_dispatch(&large).is_none());
        assert!(router.next_dispatch(&small).is_none());
        router.cancel_retained_request(&large);
        let d = router.next_dispatch(&small).unwrap();
        assert_eq!(a.target_id, d.target_id);
        assert_eq!(router.policy.lock().cursor("m"), 1);
    }
    #[test]
    fn legacy_and_ticket_policies_choose_the_same_rank_for_identical_candidates() {
        use super::super::binding::make_binding_policy;
        use ProgramBindingStrategy::*;
        for strategy in [
            ConsistentHash,
            ProgramRoundRobin,
            LeastProgramCount,
            ReasoningTokenBalance,
            CacheAware,
        ] {
            let mut legacy = make_binding_policy(strategy, 16);
            let mut proposed = make_binding_policy(strategy, 16);
            let ranks = ["a", "b"].map(|target| ProgramBindingCandidate {
                target_id: target.into(),
                accounted_programs: 1,
                accounted_tokens: 10.,
                capacity_tokens: Some(100),
                kv_pressure: 0.1,
                immediately_admissible: true,
            });
            for session in ["one", "two", "three"] {
                let identity = identity(session, "r", false);
                let actual = proposed.recommend(&identity, &ranks, Some("shared prefix"));
                let old = legacy.select(&identity, &ranks, Some("shared prefix"));
                assert_eq!(actual, old, "{strategy:?}");
                proposed.confirm("m");
                for policy in [&legacy, &proposed] {
                    policy.commit_placement("m", "shared prefix", actual.as_deref().unwrap());
                }
            }
        }
    }

    struct BlockingScore {
        called: tokio::sync::Notify,
        calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl SessionRankScorer for BlockingScore {
        async fn recommend(
            &self,
            candidates: &[ProgramBindingCandidate],
        ) -> Result<Option<String>, ScheduleError> {
            use std::sync::atomic::Ordering;
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                self.called.notify_one();
                std::future::pending::<()>().await;
            }
            Ok(candidates.first().map(|c| c.target_id.clone()))
        }
    }

    #[tokio::test]
    async fn topology_interrupts_blocked_policy_and_reuses_original_head() {
        let router = router();
        let ranks = [rank("a"), rank("b")];
        router.sync_targets("m", &ranks);
        let id = identity("s", "head", false);
        let scorer = BlockingScore {
            called: Default::default(),
            calls: Default::default(),
        };
        let future = router.acquire_with_scorer(
            id,
            ContextTokens::estimated(10),
            &ranks,
            None,
            Some(&scorer),
        );
        tokio::pin!(future);
        tokio::select! {
            _ = scorer.called.notified() => {},
            _ = &mut future => panic!("score must stay in flight")
        }
        // sync and registration take the scheduler lock while scoring is suspended.
        let tail = register(&router, &identity("s", "tail", true), 20);
        router.sync_targets("m", &ranks[1..]);
        let dispatch = future.await.unwrap();
        assert_eq!(dispatch.target_id, "b");
        assert_eq!(dispatch.request_hints().request_id.as_deref(), Some("head"));
        assert_eq!(router.next_dispatch(&tail).unwrap().target_id, "b");
    }
}
