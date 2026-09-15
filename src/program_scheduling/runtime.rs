//! Program generations, request retention, and accepted runtime transitions.
//!
//! ProgramRuntime mutates lifecycle facts but makes no scheduling decisions.
//! A policy decides when to call admit, resume, pause, or release.

use super::request_pool::RequestPool;
use super::{
    ProgramDispatch, ProgramIdentity, ProgramRef, ProgramRequestHandle, ProgramState,
    ProgramStatus, ProgramUsageObservation,
};
use std::collections::HashMap;
use std::time::Instant;

/// Mutable protocol-neutral Program runtime owned by ProgramScheduler.
#[derive(Debug, Default)]
pub struct ProgramRuntime {
    programs: HashMap<RuntimeKey, RuntimeProgram>,
    last_generations: HashMap<RuntimeKey, u64>,
    released_generations_at: HashMap<RuntimeKey, Instant>,
    request_pool: RequestPool,
    next_placement_epoch: u64,
}

impl ProgramRuntime {
    /// Materialize or update a Program, then retain the arriving request.
    pub fn retain_request(
        &mut self,
        identity: &ProgramIdentity,
        estimated_context_tokens: usize,
        routing_text: Option<&str>,
        arrived_at: Instant,
    ) -> ProgramRequestHandle {
        let key = RuntimeKey::from(identity);
        self.released_generations_at.remove(&key);
        let program_ref = if let Some(program) = self.programs.get_mut(&key) {
            program.expected_resume = identity.expected_resume();
            program.estimated_context_tokens = program
                .estimated_context_tokens
                .max(estimated_context_tokens);
            program.reference.clone()
        } else {
            let generation = self
                .last_generations
                .get(&key)
                .copied()
                .map_or(0, |generation| generation.saturating_add(1));
            self.last_generations.insert(key.clone(), generation);
            let reference = ProgramRef::new(
                identity.model_pool().to_string(),
                identity.program_id().to_string(),
                generation,
            );
            self.programs.insert(
                key,
                RuntimeProgram {
                    reference: reference.clone(),
                    state: ProgramState::Paused,
                    status: ProgramStatus::Reasoning,
                    expected_resume: identity.expected_resume(),
                    placement: None,
                    placement_epoch: 0,
                    placement_start_request_pending: false,
                    estimated_context_tokens,
                    in_flight_requests: 0,
                },
            );
            reference
        };
        if let Some(program) = self.programs.get_mut(&RuntimeKey::from(identity)) {
            program.status = ProgramStatus::Reasoning;
        }
        self.request_pool.retain(
            program_ref,
            arrived_at,
            estimated_context_tokens,
            routing_text,
        )
    }

    /// Commit the front retained request to `target_id`.
    ///
    /// A paused Program begins a new placement epoch. An active Program may
    /// dispatch only on its existing target and does not begin a new segment.
    pub fn admit_front(
        &mut self,
        handle: &ProgramRequestHandle,
        target_id: String,
        dispatched_at: Instant,
    ) -> Option<ProgramDispatch> {
        let key = RuntimeKey::from(handle.program());
        let program = self.programs.get_mut(&key)?;
        if &program.reference != handle.program() {
            return None;
        }
        let starts_placement = match program.state {
            ProgramState::Paused => {
                self.next_placement_epoch = self.next_placement_epoch.saturating_add(1);
                program.state = ProgramState::Active;
                program.placement = Some(target_id.clone());
                program.placement_epoch = self.next_placement_epoch;
                true
            }
            ProgramState::Active => {
                if program.placement.as_deref() != Some(target_id.as_str()) {
                    return None;
                }
                program.placement_start_request_pending
            }
        };
        program.placement_start_request_pending = false;
        let request = self.request_pool.take_front(handle)?;
        program.status = ProgramStatus::Reasoning;
        program.in_flight_requests = program.in_flight_requests.saturating_add(1);
        self.request_pool.notify_front(&program.reference);
        Some(ProgramDispatch::new(
            program.reference.clone(),
            target_id,
            program.placement_epoch,
            request.arrived_at,
            dispatched_at,
            request.estimated_context_tokens,
            starts_placement,
            request.routing_text,
        ))
    }

    /// Record one request completion if its generation and placement are current.
    pub fn complete_request(
        &mut self,
        dispatch: &ProgramDispatch,
        usage: ProgramUsageObservation,
    ) -> bool {
        let key = RuntimeKey::from(dispatch.program());
        let Some(program) = self.programs.get_mut(&key) else {
            return false;
        };
        if &program.reference != dispatch.program()
            || program.placement_epoch != dispatch.placement_epoch()
            || program.in_flight_requests == 0
        {
            return false;
        }
        let prompt_tokens = usage
            .prompt_tokens
            .unwrap_or(dispatch.estimated_context_tokens());
        program.estimated_context_tokens = program.estimated_context_tokens.max(prompt_tokens);
        program.in_flight_requests -= 1;
        if program.in_flight_requests == 0 && self.request_pool.program_len(&program.reference) == 0
        {
            program.status = ProgramStatus::Acting;
        }
        true
    }

    /// Cancel one retained request and report whether it was present.
    pub fn cancel_request(&mut self, handle: &ProgramRequestHandle) -> bool {
        let Some(was_front) = self.request_pool.cancel(handle) else {
            return false;
        };
        if was_front {
            self.request_pool.notify_front(handle.program());
        }
        let key = RuntimeKey::from(handle.program());
        if let Some(program) = self.programs.get_mut(&key) {
            if &program.reference == handle.program()
                && program.in_flight_requests == 0
                && self.request_pool.program_len(&program.reference) == 0
            {
                program.status = ProgramStatus::Acting;
                if program.state == ProgramState::Active {
                    program.state = ProgramState::Paused;
                    program.placement = None;
                }
            }
        }
        true
    }

    /// Pause an idle active Program and clear its runtime placement.
    pub fn pause_idle(&mut self, program_ref: &ProgramRef) -> bool {
        let key = RuntimeKey::from(program_ref);
        let Some(program) = self.programs.get_mut(&key) else {
            return false;
        };
        if &program.reference != program_ref
            || program.state != ProgramState::Active
            || program.in_flight_requests != 0
        {
            return false;
        }
        program.state = ProgramState::Paused;
        program.placement = None;
        true
    }

    /// Activate one paused Program without consuming its retained request.
    pub(crate) fn activate(&mut self, program_ref: &ProgramRef, target_id: String) -> bool {
        let key = RuntimeKey::from(program_ref);
        let Some(program) = self.programs.get_mut(&key) else {
            return false;
        };
        if &program.reference != program_ref
            || program.state != ProgramState::Paused
            || self.request_pool.program_len(program_ref) == 0
        {
            return false;
        }
        self.next_placement_epoch = self.next_placement_epoch.saturating_add(1);
        program.state = ProgramState::Active;
        program.status = ProgramStatus::Reasoning;
        program.placement = Some(target_id);
        program.placement_epoch = self.next_placement_epoch;
        program.placement_start_request_pending = true;
        true
    }

    /// Release an idle Program generation with no retained requests.
    pub fn release_idle(&mut self, program_ref: &ProgramRef, now: Instant) -> bool {
        let key = RuntimeKey::from(program_ref);
        let releasable = self.programs.get(&key).is_some_and(|program| {
            &program.reference == program_ref
                && program.in_flight_requests == 0
                && self.request_pool.program_len(program_ref) == 0
        });
        if !releasable {
            return false;
        }
        let Some(program) = self.programs.remove(&key) else {
            return false;
        };
        self.last_generations
            .insert(key.clone(), program.reference.generation());
        self.released_generations_at.insert(key, now);
        true
    }

    /// Bound generation fencing history after released Programs age out.
    pub(crate) fn prune_released_generations(
        &mut self,
        now: Instant,
        retention: std::time::Duration,
    ) {
        let expired = self
            .released_generations_at
            .iter()
            .filter(|(key, released_at)| {
                !self.programs.contains_key(*key)
                    && now.saturating_duration_since(**released_at) >= retention
            })
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        for key in expired {
            self.released_generations_at.remove(&key);
            self.last_generations.remove(&key);
        }
    }

    /// Invalidate a placement whose backend disappeared from discovery.
    pub(crate) fn invalidate_placement(
        &mut self,
        program_ref: &ProgramRef,
        removed_targets: &[String],
    ) -> bool {
        let key = RuntimeKey::from(program_ref);
        let Some(program) = self.programs.get_mut(&key) else {
            return false;
        };
        if &program.reference != program_ref
            || program
                .placement
                .as_ref()
                .is_none_or(|target| !removed_targets.contains(target))
        {
            return false;
        }
        self.next_placement_epoch = self.next_placement_epoch.saturating_add(1);
        program.state = ProgramState::Paused;
        program.placement = None;
        program.placement_epoch = self.next_placement_epoch;
        program.placement_start_request_pending = false;
        program.in_flight_requests = 0;
        true
    }

    /// Current state and demand status for diagnostics and policy snapshots.
    pub fn state(&self, program_ref: &ProgramRef) -> Option<(ProgramState, ProgramStatus)> {
        self.programs
            .get(&RuntimeKey::from(program_ref))
            .filter(|program| &program.reference == program_ref)
            .map(|program| (program.state, program.status))
    }

    /// Current target that owns logical capacity for this Program.
    pub fn placement(&self, program_ref: &ProgramRef) -> Option<&str> {
        self.programs
            .get(&RuntimeKey::from(program_ref))
            .filter(|program| &program.reference == program_ref)
            .and_then(|program| program.placement.as_deref())
    }

    /// Number of requests currently retained in RequestPool.
    pub fn retained_request_count(&self) -> usize {
        self.request_pool.len()
    }

    /// Whether no request is currently retained for admission.
    pub fn request_pool_is_empty(&self) -> bool {
        self.request_pool.is_empty()
    }

    /// Immutable lifecycle view used by scheduling decisions under one lock.
    pub(crate) fn view(&self, program_ref: &ProgramRef) -> Option<RuntimeProgramView> {
        self.programs
            .get(&RuntimeKey::from(program_ref))
            .filter(|program| &program.reference == program_ref)
            .map(|program| RuntimeProgramView {
                reference: program.reference.clone(),
                state: program.state,
                status: program.status,
                expected_resume: program.expected_resume,
                placement: program.placement.clone(),
                estimated_context_tokens: program.estimated_context_tokens,
                in_flight_requests: program.in_flight_requests,
                waiting_requests: self.request_pool.program_len(&program.reference),
            })
    }

    /// Snapshot all live Program lifecycle facts without imposing order.
    pub(crate) fn views(&self) -> Vec<RuntimeProgramView> {
        self.programs
            .values()
            .map(|program| RuntimeProgramView {
                reference: program.reference.clone(),
                state: program.state,
                status: program.status,
                expected_resume: program.expected_resume,
                placement: program.placement.clone(),
                estimated_context_tokens: program.estimated_context_tokens,
                in_flight_requests: program.in_flight_requests,
                waiting_requests: self.request_pool.program_len(&program.reference),
            })
            .collect()
    }

    /// Wake the front retained request after an accepted transition.
    pub(crate) fn notify_front(&self, program_ref: &ProgramRef) {
        self.request_pool.notify_front(program_ref);
    }
}

/// Immutable lifecycle facts consumed by Program scheduling policy modules.
#[derive(Debug, Clone)]
pub(crate) struct RuntimeProgramView {
    pub(crate) reference: ProgramRef,
    pub(crate) state: ProgramState,
    pub(crate) status: ProgramStatus,
    pub(crate) expected_resume: bool,
    pub(crate) placement: Option<String>,
    pub(crate) estimated_context_tokens: usize,
    pub(crate) in_flight_requests: usize,
    pub(crate) waiting_requests: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RuntimeKey {
    model_pool: String,
    program_id: String,
}

impl From<&ProgramIdentity> for RuntimeKey {
    fn from(identity: &ProgramIdentity) -> Self {
        Self {
            model_pool: identity.model_pool().to_string(),
            program_id: identity.program_id().to_string(),
        }
    }
}

impl From<&ProgramRef> for RuntimeKey {
    fn from(reference: &ProgramRef) -> Self {
        Self {
            model_pool: reference.model_pool().to_string(),
            program_id: reference.program_id().to_string(),
        }
    }
}

#[derive(Debug)]
struct RuntimeProgram {
    reference: ProgramRef,
    state: ProgramState,
    status: ProgramStatus,
    expected_resume: bool,
    placement: Option<String>,
    placement_epoch: u64,
    placement_start_request_pending: bool,
    estimated_context_tokens: usize,
    in_flight_requests: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn identity(program_id: &str) -> ProgramIdentity {
        ProgramIdentity::from_request(
            None,
            Some(&json!({
                "vllm_xargs": {"agentic_context": {
                    "program_id": program_id,
                    "task_id": null,
                    "expected_resume": true
                }}
            })),
            Some("model"),
        )
        .unwrap()
        .unwrap()
    }

    #[test]
    fn retain_admit_complete_pause_and_release() {
        let mut runtime = ProgramRuntime::default();
        let handle = runtime.retain_request(&identity("p"), 100, None, Instant::now());
        assert_eq!(runtime.retained_request_count(), 1);
        let dispatch = runtime
            .admit_front(&handle, "rank-0".into(), Instant::now())
            .unwrap();
        assert!(dispatch.placement_start_request());
        assert_eq!(runtime.placement(dispatch.program()), Some("rank-0"));
        assert!(runtime.complete_request(
            &dispatch,
            ProgramUsageObservation {
                prompt_tokens: Some(100),
                completion_tokens: Some(10),
                cached_prompt_tokens: Some(0),
            }
        ));
        assert_eq!(
            runtime.state(dispatch.program()),
            Some((ProgramState::Active, ProgramStatus::Acting))
        );
        assert!(runtime.pause_idle(dispatch.program()));
        assert_eq!(runtime.placement(dispatch.program()), None);
        assert!(runtime.release_idle(dispatch.program(), Instant::now()));
    }

    #[test]
    fn stale_generation_cannot_mutate_replacement() {
        let mut runtime = ProgramRuntime::default();
        let old_handle = runtime.retain_request(&identity("p"), 100, None, Instant::now());
        let old_dispatch = runtime
            .admit_front(&old_handle, "rank-0".into(), Instant::now())
            .unwrap();
        assert!(runtime.complete_request(&old_dispatch, None.into()));
        assert!(runtime.pause_idle(old_dispatch.program()));
        assert!(runtime.release_idle(old_dispatch.program(), Instant::now()));

        let new_handle = runtime.retain_request(&identity("p"), 200, None, Instant::now());
        let new_dispatch = runtime
            .admit_front(&new_handle, "rank-1".into(), Instant::now())
            .unwrap();
        assert!(!runtime.complete_request(&old_dispatch, None.into()));
        assert_eq!(runtime.placement(new_dispatch.program()), Some("rank-1"));
        assert_eq!(old_dispatch.program().generation(), 0);
        assert_eq!(new_dispatch.program().generation(), 1);
    }

    #[test]
    fn released_generation_history_is_ttl_bounded() {
        let now = Instant::now();
        let mut runtime = ProgramRuntime::default();
        let identity = identity("bounded");
        let handle = runtime.retain_request(&identity, 10, None, now);
        let dispatch = runtime.admit_front(&handle, "rank-0".into(), now).unwrap();
        assert!(runtime.complete_request(&dispatch, Some(10).into()));
        assert!(runtime.pause_idle(dispatch.program()));
        assert!(runtime.release_idle(dispatch.program(), now));
        runtime.prune_released_generations(
            now + std::time::Duration::from_secs(61),
            std::time::Duration::from_secs(60),
        );
        assert!(runtime.last_generations.is_empty());
        assert!(runtime.released_generations_at.is_empty());
    }

    #[test]
    fn active_consecutive_request_keeps_placement_epoch() {
        let mut runtime = ProgramRuntime::default();
        let program = identity("p");
        let first = runtime.retain_request(&program, 100, None, Instant::now());
        let first_dispatch = runtime
            .admit_front(&first, "rank-0".into(), Instant::now())
            .unwrap();
        let second = runtime.retain_request(&program, 110, None, Instant::now());
        let second_dispatch = runtime
            .admit_front(&second, "rank-0".into(), Instant::now())
            .unwrap();
        assert!(!second_dispatch.placement_start_request());
        assert_eq!(
            first_dispatch.placement_epoch(),
            second_dispatch.placement_epoch()
        );
        assert!(runtime.complete_request(&first_dispatch, None.into()));
        assert!(runtime.complete_request(&second_dispatch, None.into()));
    }

    #[test]
    fn cancelling_only_waiter_releases_active_reasoning_placement() {
        let mut runtime = ProgramRuntime::default();
        let program = identity("p");
        let first = runtime.retain_request(&program, 100, None, Instant::now());
        let dispatch = runtime
            .admit_front(&first, "rank-0".into(), Instant::now())
            .unwrap();
        assert!(runtime.complete_request(&dispatch, None.into()));
        let waiting = runtime.retain_request(&program, 110, None, Instant::now());
        assert!(runtime.cancel_request(&waiting));
        assert_eq!(runtime.placement(dispatch.program()), None);
        assert_eq!(
            runtime.state(dispatch.program()),
            Some((ProgramState::Paused, ProgramStatus::Acting))
        );
    }
}
