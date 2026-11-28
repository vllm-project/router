//! Coordinate Program membership, binding recommendations and fixed-rank dispatch.
//! Full HTTP sessions rank eligible candidates outside admission and commit a fenced ticket.

use super::scheduler::ProgramAdmission;
use super::scheduler_admission::AdmissionGuard;
use super::scheduler_state::ProgramSchedulerState;
use super::{
    ContextTokens, ProgramBindingPolicy, ProgramDispatch, ProgramIdentity, ProgramRef,
    ProgramRequestHandle, ProgramSchedulerConfig, ProgramTarget, ScheduleError,
};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Instant;

/// Router-owned coordinator. Existing transports can continue using the ProgramScheduler alias.
#[derive(Debug)]
pub struct ProgramRouter {
    pub(super) admission: ProgramAdmission,
    pub(super) policy: Mutex<Box<dyn ProgramBindingPolicy>>,
}

// Forward existing admission, observation and settlement APIs to their owner.
impl std::ops::Deref for ProgramRouter {
    type Target = ProgramAdmission;
    fn deref(&self) -> &ProgramAdmission {
        &self.admission
    }
}

impl ProgramRouter {
    pub fn new(config: ProgramSchedulerConfig) -> Self {
        let policy =
            super::binding::make_binding_policy(config.binding_strategy, config.hash_virtual_nodes);
        Self {
            admission: ProgramAdmission::new(config),
            policy: Mutex::new(policy),
        }
    }
    /// Retain one Program request until rank-local admission permits dispatch.
    pub async fn acquire(
        &self,
        identity: ProgramIdentity,
        context: impl Into<ContextTokens>,
        targets: &[ProgramTarget],
        routing_text: Option<String>,
    ) -> Result<ProgramDispatch, ScheduleError> {
        self.sync_targets(identity.model_pool(), targets);
        self.acquire_synced(identity, context.into(), targets, routing_text)
            .await
    }

    /// Synchronize one model pool's concrete backend and internal-DP targets.
    pub fn sync_targets(&self, model_pool: &str, targets: &[ProgramTarget]) {
        self.sync_target_snapshot(model_pool, Arc::from(targets));
    }

    /// Synchronize a cached immutable target snapshot.
    pub fn sync_target_snapshot(&self, model_pool: &str, targets: Arc<[ProgramTarget]>) {
        self.sync_target_snapshots([(model_pool.to_string(), targets)]);
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    pub(crate) fn ensure_decision_state(
        &self,
        state: &mut ProgramSchedulerState,
        identity: &ProgramIdentity,
        reference: ProgramRef,
        context: impl Into<ContextTokens>,
        now: Instant,
        routing_text: Option<&str>,
    ) {
        let home_target = self.initial_binding(state, identity, &reference, now, routing_text);
        self.update_program(state, identity, reference, context, home_target, now);
    }

    pub(super) fn initial_binding(
        &self,
        state: &mut ProgramSchedulerState,
        identity: &ProgramIdentity,
        reference: &ProgramRef,
        now: Instant,
        text: Option<&str>,
    ) -> Option<String> {
        if state.decisions.contains_key(reference) {
            return None;
        }
        self.select_and_commit(state, identity, now, text)
    }

    fn select_and_commit(
        &self,
        state: &mut ProgramSchedulerState,
        identity: &ProgramIdentity,
        now: Instant,
        text: Option<&str>,
    ) -> Option<String> {
        let candidates = self.admission.binding_candidates(state, identity, now);
        if candidates.is_empty() {
            return None;
        }
        // binding_candidates iterates the pool's BTreeSet in target ID order.
        if let Some(bound) = state.bindings.binding(identity) {
            if candidates.iter().any(|c| c.target_id == bound) {
                return Some(bound.to_owned());
            }
        }
        let rank = self.policy.lock().select(identity, &candidates, text)?;
        self.admission.commit_binding(state, identity, rank)
    }

    pub(super) fn next_dispatch(&self, handle: &ProgramRequestHandle) -> Option<ProgramDispatch> {
        let mut state = self.admission.state.lock();
        let dispatch = self.admission.claim_dispatch(&mut state, handle)?;
        if let Some(text) = dispatch.routing_text() {
            self.policy.lock().commit_placement(
                dispatch.program().model_pool(),
                text,
                &dispatch.target_id,
            );
        }
        Some(dispatch)
    }

    pub async fn acquire_from_snapshot(
        &self,
        identity: ProgramIdentity,
        context: ContextTokens,
        targets: Arc<[ProgramTarget]>,
        text: Option<String>,
    ) -> Result<ProgramDispatch, ScheduleError> {
        self.sync_target_snapshots([(identity.model_pool().to_owned(), targets.clone())]);
        self.acquire_synced(identity, context, &targets, text).await
    }

    pub(super) async fn acquire_synced(
        &self,
        identity: ProgramIdentity,
        context: ContextTokens,
        targets: &[ProgramTarget],
        text: Option<String>,
    ) -> Result<ProgramDispatch, ScheduleError> {
        self.acquire_with_scorer(identity, context, targets, text, None)
            .await
    }

    pub(crate) async fn acquire_with_scorer(
        &self,
        identity: ProgramIdentity,
        context: ContextTokens,
        targets: &[ProgramTarget],
        text: Option<String>,
        scorer: Option<&dyn super::SessionRankScorer>,
    ) -> Result<ProgramDispatch, ScheduleError> {
        if targets.is_empty() {
            return Err(ScheduleError::NoTargets);
        }
        let arrived_at = Instant::now();
        let expected_resume = identity.expected_resume();
        let (handle, pending, strict) = {
            let mut state = self.admission.state.lock();
            let (handle, context) = self.admission.register_attempt(
                &mut state,
                &identity,
                context,
                text.as_deref(),
                arrived_at,
            );
            let reference = handle.program().clone();
            let strict = state
                .decisions
                .get(&reference)
                .map_or(identity.session_admission, |d| d.session_admission);
            let rank = if strict {
                None
            } else {
                self.initial_binding(
                    &mut state,
                    &identity,
                    &reference,
                    arrived_at,
                    text.as_deref(),
                )
            };
            let pending = self
                .admission
                .finish_registration(&mut state, &identity, reference, context, rank, arrived_at);
            (handle, pending, strict)
        };
        let mut guard = AdmissionGuard::new(&self.admission, handle);
        let (output_prediction, return_time_keys) = pending?;
        let deadline = arrived_at + self.admission.config.queue_timeout;
        let mut refreshed = false;
        let mut deferred_revision = None;
        loop {
            if strict && Instant::now() >= deadline {
                return Err(ScheduleError::QueueTimeout);
            }
            let wait_handle = guard.handle().clone();
            let notified = wait_handle.notified();
            if let Some(mut dispatch) = self.next_dispatch(guard.handle()) {
                dispatch.attach_prediction_context(
                    expected_resume,
                    output_prediction,
                    return_time_keys,
                );
                guard.disarm();
                return Ok(dispatch);
            }
            if strict
                && deferred_revision
                    .is_none_or(|r| self.admission.state.lock().admission_revision != r)
            {
                if let Some(task) =
                    self.admission
                        .binding_task(guard.handle(), &identity, deadline)?
                {
                    let (dispatch, retry) = self
                        .select_and_commit_ticket(&task, text.as_deref(), scorer)
                        .await?;
                    if let Some(mut dispatch) = dispatch {
                        dispatch.attach_prediction_context(
                            expected_resume,
                            output_prediction,
                            return_time_keys,
                        );
                        guard.disarm();
                        return Ok(dispatch);
                    }
                    deferred_revision = None;
                    if !retry || refreshed {
                        deferred_revision = Some(task.revision);
                    }
                    if retry && !refreshed {
                        refreshed = true;
                        continue;
                    }
                }
            }
            refreshed = false;
            let now = Instant::now();
            if now >= deadline {
                return Err(ScheduleError::QueueTimeout);
            }
            let force_deadline = {
                let state = self.admission.state.lock();
                state
                    .decisions
                    .get(guard.handle().program())
                    .and_then(|decision| decision.force_resume_deadline)
            };
            let wake_at = force_deadline.filter(|force| *force > now).map_or(
                (now + self.admission.config.metrics_interval).min(deadline),
                |force| force.min(deadline),
            );
            if tokio::time::timeout(wake_at.saturating_duration_since(now), notified)
                .await
                .is_err()
            {
                let mut state = self.admission.state.lock();
                self.admission
                    .run_request_arrival_decisions(&mut state, Instant::now());
            }
        }
    }

    pub fn sync_target_snapshots(
        &self,
        snapshots: impl IntoIterator<Item = (String, Arc<[ProgramTarget]>)>,
    ) {
        self.sync_registry_snapshots(None, snapshots);
    }

    pub(crate) fn sync_registry_snapshots(
        &self,
        revision: Option<u64>,
        snapshots: impl IntoIterator<Item = (String, Arc<[ProgramTarget]>)>,
    ) {
        let mut state = self.admission.state.lock();
        for (model_pool, targets) in snapshots {
            if let Some(revision) = revision {
                let installed = state
                    .registry_revisions
                    .entry(model_pool.clone())
                    .or_default();
                if revision < *installed {
                    continue;
                }
                *installed = revision;
            }
            let Some((removed, affected)) =
                self.admission.sync_pool(&mut state, &model_pool, targets)
            else {
                continue;
            };
            ProgramAdmission::wake_capacity_waiters(&mut state);
            for target in &removed {
                self.policy.lock().remove_target(&model_pool, target);
            }
            for work in affected {
                let invalidated = self.admission.begin_rebind(&mut state, &work, &removed);
                let replacement = if state.decisions[&work.program].session_admission {
                    None
                } else {
                    self.select_and_commit(&mut state, &work.identity, Instant::now(), None)
                };
                self.admission
                    .finish_rebind(&mut state, work, replacement, invalidated);
            }
        }
    }

    pub(crate) fn binding_only(&self) -> bool {
        self.admission.config.binding_only
    }

    pub(crate) fn uses_kv_binding(&self) -> bool {
        self.admission.config.binding_strategy.uses_kv_index()
    }

    /// Whether initial Program binding needs the request text for prefix affinity.
    pub fn uses_cache_aware_binding(&self) -> bool {
        self.admission.config.binding_strategy.needs_routing_text()
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::program_scheduling::{ProgramBindingCandidate, ProgramBindingStrategy};
    use serde_json::json;

    type SelectionLog = Arc<Mutex<Vec<(String, Option<String>)>>>;

    #[derive(Debug)]
    struct RecordingPolicy {
        delegate: Box<dyn ProgramBindingPolicy>,
        selections: SelectionLog,
    }

    impl ProgramBindingPolicy for RecordingPolicy {
        fn select(
            &mut self,
            identity: &ProgramIdentity,
            candidates: &[ProgramBindingCandidate],
            text: Option<&str>,
        ) -> Option<String> {
            self.selections
                .lock()
                .push((identity.program_id().into(), text.map(str::to_owned)));
            self.delegate.select(identity, candidates, text)
        }
    }

    fn identity(id: &str) -> ProgramIdentity {
        let request = json!({"vllm_xargs":{"agentic_context":{
            "session_id":"same-chat", "program_id":id, "expected_resume":true
        }}});
        ProgramIdentity::from_request(None, Some(&request), Some("model"))
            .unwrap()
            .unwrap()
    }

    fn rank(id: usize) -> ProgramTarget {
        ProgramTarget {
            id: format!("r{id}"),
            base_url: format!("http://r{id}"),
            dp_rank: Some(id),
        }
    }

    #[tokio::test]
    async fn binding_only_bypasses_rank_capacity_while_full_scheduling_gates_it() {
        for binding_only in [false, true] {
            let router = ProgramRouter::new(ProgramSchedulerConfig {
                binding_only,
                enable_acting_ttl: false,
                max_active_programs_per_target: 1,
                ..Default::default()
            });
            let targets = [rank(0)];
            let a = router
                .acquire(identity("a"), 100, &targets, None)
                .await
                .unwrap();
            let b = router.acquire(identity("b"), 100, &targets, None);
            tokio::pin!(b);
            let dispatch = match futures::poll!(&mut b) {
                std::task::Poll::Ready(result) => {
                    assert!(
                        binding_only,
                        "full scheduling must gate B while A is in flight"
                    );
                    router.complete(&a, true, true, Some(100), None);
                    result.unwrap()
                }
                std::task::Poll::Pending => {
                    assert!(
                        !binding_only,
                        "binding-only mode must bypass the capacity gate"
                    );
                    router.complete(&a, true, true, Some(100), None);
                    tokio::time::timeout(std::time::Duration::from_secs(1), b)
                        .await
                        .unwrap()
                        .unwrap()
                }
            };
            assert_eq!(dispatch.target_id, "r0");
            router.complete(&dispatch, true, true, Some(100), None);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn router_selects_once_per_program_then_rebinds_only_the_lost_current_rank() {
        let router = Arc::new(ProgramRouter::new(ProgramSchedulerConfig::default()));
        let selections = Arc::new(Mutex::new(Vec::new()));
        *router.policy.lock() = Box::new(RecordingPolicy {
            delegate: super::super::binding::make_binding_policy(
                ProgramBindingStrategy::ProgramRoundRobin,
                160,
            ),
            selections: selections.clone(),
        });
        let pool = identity("a").model_pool().to_owned();
        let targets = [rank(1), rank(0)];
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let first_turn = || {
            let router = router.clone();
            let barrier = barrier.clone();
            let targets = targets.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                router
                    .acquire(identity("a"), 100, &targets, Some("first".into()))
                    .await
                    .unwrap()
            })
        };
        let (a, a2) = tokio::join!(first_turn(), first_turn());
        let (a, a2) = (a.unwrap(), a2.unwrap());
        assert_eq!((a.target_id.as_str(), a2.target_id.as_str()), ("r0", "r0"));
        let b = router
            .acquire(identity("b"), 100, &targets, Some("child".into()))
            .await
            .unwrap();
        assert_eq!(b.target_id, "r1");
        assert_eq!(selections.lock().len(), 2);
        router.complete(&a, true, false, Some(100), None);
        router.complete(&a2, true, false, Some(100), None);
        router.complete(&b, true, false, Some(100), None);

        // Topology notification alone initiates the new decision.
        router.sync_targets(&pool, &[rank(1)]);
        assert_eq!(
            &*selections.lock(),
            &[
                ("a".into(), Some("first".into())),
                ("b".into(), Some("child".into())),
                ("a".into(), None)
            ]
        );
        let a3 = router
            .acquire(identity("a"), 100, &[rank(1)], None)
            .await
            .unwrap();
        assert_eq!(a3.target_id, "r1");
        assert_eq!(selections.lock().len(), 3);
    }
}
