//! Initial Program-to-backend binding policies.
//!
//! Binding is evaluated only for a new Program generation. Admission, resume,
//! and cross-rank runtime placement remain ProgramScheduler responsibilities.

use super::ProgramIdentity;
use super::ProgramRef;
pub(crate) use crate::policies::binding::make_binding_policy;
pub use crate::policies::binding::{
    ProgramBindingCandidate, ProgramBindingPolicy, ProgramBindingStrategy,
};
use std::collections::HashMap;

/// Model-pool scoped home bindings. Actual runtime placement lives in the scheduler.
#[derive(Debug, Default)]
pub(crate) struct BindingTable {
    entries: HashMap<ProgramBindingKey, String>,
}

impl BindingTable {
    pub(crate) fn binding(&self, identity: &ProgramIdentity) -> Option<&str> {
        self.entries
            .get(&ProgramBindingKey::from(identity))
            .map(String::as_str)
    }

    pub(crate) fn commit(&mut self, identity: &ProgramIdentity, target: &str) {
        self.entries
            .insert(ProgramBindingKey::from(identity), target.to_owned());
    }

    pub(crate) fn release_program(&mut self, program: &ProgramRef) {
        self.entries.remove(&ProgramBindingKey {
            model_pool: program.model_pool().to_owned(),
            program_id: program.program_id().to_owned(),
        });
    }

    pub(crate) fn remove_target(&mut self, model_pool: &str, target: &str) {
        self.entries
            .retain(|key, bound| key.model_pool != model_pool || bound != target);
    }
}

/// Compatibility API for callers that bind directly. The Router uses a separate
/// policy and binding table so admission can operate without a selector.
#[derive(Debug)]
pub struct ProgramBindings {
    policy: Box<dyn ProgramBindingPolicy>,
    table: BindingTable,
}

impl ProgramBindings {
    pub fn new(strategy: ProgramBindingStrategy, hash_virtual_nodes: u32) -> Self {
        Self {
            policy: make_binding_policy(strategy, hash_virtual_nodes),
            table: BindingTable::default(),
        }
    }

    pub fn bind(
        &mut self,
        identity: &ProgramIdentity,
        candidates: &[ProgramBindingCandidate],
        routing_text: Option<&str>,
    ) -> Option<String> {
        if let Some(bound) = self.binding(identity) {
            if candidates.iter().any(|c| c.target_id == bound) {
                return Some(bound.to_owned());
            }
        }
        let mut ordered = candidates.to_vec();
        ordered.sort_by(|a, b| a.target_id.cmp(&b.target_id));
        let target = self.policy.select(identity, &ordered, routing_text)?;
        self.table.commit(identity, &target);
        Some(target)
    }

    pub fn binding(&self, identity: &ProgramIdentity) -> Option<&str> {
        self.table.binding(identity)
    }
    pub fn release(&mut self, identity: &ProgramIdentity) -> Option<String> {
        self.table
            .entries
            .remove(&ProgramBindingKey::from(identity))
    }
    pub fn remove_target(&mut self, target: &str) -> usize {
        let previous = self.table.entries.len();
        self.table.entries.retain(|_, bound| bound != target);
        previous - self.table.entries.len()
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn commit_placement(&self, program: &ProgramRef, text: &str, target_id: &str) {
        self.policy
            .commit_placement(program.model_pool(), text, target_id);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ProgramBindingKey {
    model_pool: String,
    program_id: String,
}

impl From<&ProgramIdentity> for ProgramBindingKey {
    fn from(identity: &ProgramIdentity) -> Self {
        Self {
            model_pool: identity.model_pool().to_string(),
            program_id: identity.program_id().to_string(),
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
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

    fn candidate(id: &str, programs: usize, used: f64, capacity: usize) -> ProgramBindingCandidate {
        ProgramBindingCandidate {
            target_id: id.to_string(),
            accounted_programs: programs,
            accounted_tokens: used,
            capacity_tokens: Some(capacity),
            kv_pressure: used / capacity as f64,
            immediately_admissible: true,
        }
    }

    #[test]
    fn binding_is_stable_until_release_or_target_removal() {
        let mut bindings = ProgramBindings::new(ProgramBindingStrategy::ProgramRoundRobin, 160);
        let program = identity("program-a");
        let candidates = vec![
            candidate("rank-1", 0, 0.0, 100),
            candidate("rank-0", 0, 0.0, 100),
        ];
        assert_eq!(
            bindings.bind(&program, &candidates, None).as_deref(),
            Some("rank-0")
        );
        assert_eq!(
            bindings.bind(&program, &candidates, None).as_deref(),
            Some("rank-0")
        );
        assert_eq!(bindings.binding(&program), Some("rank-0"));
        assert_eq!(bindings.remove_target("rank-0"), 1);
        assert_eq!(
            bindings.bind(&program, &candidates, None).as_deref(),
            Some("rank-1")
        );
        assert_eq!(bindings.release(&program).as_deref(), Some("rank-1"));
        assert_eq!(bindings.binding(&program), None);
    }

    #[test]
    fn round_robin_uses_sorted_targets_per_program_generation() {
        let mut bindings = ProgramBindings::new(ProgramBindingStrategy::ProgramRoundRobin, 160);
        let candidates = vec![
            candidate("rank-1", 0, 0.0, 100),
            candidate("rank-0", 0, 0.0, 100),
        ];
        assert_eq!(
            bindings
                .bind(&identity("program-a"), &candidates, None)
                .as_deref(),
            Some("rank-0")
        );
        assert_eq!(
            bindings
                .bind(&identity("program-b"), &candidates, None)
                .as_deref(),
            Some("rank-1")
        );
    }

    #[test]
    fn inherited_request_algorithms_select_among_program_candidates() {
        let program = identity("shared-policy");
        let candidates = [candidate("a", 3, 0.0, 100), candidate("b", 1, 0.0, 100)];
        for strategy in [
            ProgramBindingStrategy::Random,
            ProgramBindingStrategy::PowerOfTwo,
            ProgramBindingStrategy::RendezvousHash,
            ProgramBindingStrategy::StickyLeastLoaded,
        ] {
            let mut policy = make_binding_policy(strategy, 160);
            assert_eq!(policy.select(&program, &[], None), None);
            assert_eq!(
                policy.select(&program, &candidates[..1], None).as_deref(),
                Some("a")
            );
            let selected = policy.select(&program, &candidates, None).unwrap();
            match strategy {
                ProgramBindingStrategy::PowerOfTwo | ProgramBindingStrategy::StickyLeastLoaded => {
                    assert_eq!(selected, "b")
                }
                ProgramBindingStrategy::RendezvousHash => {
                    assert_eq!(Some(selected), policy.select(&program, &candidates, None))
                }
                _ => assert!(candidates.iter().any(|c| c.target_id == selected)),
            }
        }
    }

    #[test]
    fn least_program_count_matches_current_tie_break() {
        let mut bindings = ProgramBindings::new(ProgramBindingStrategy::LeastProgramCount, 160);
        let candidates = vec![
            candidate("rank-2", 1, 30.0, 100),
            candidate("rank-1", 1, 90.0, 100),
            candidate("rank-0", 2, 10.0, 100),
        ];
        assert_eq!(
            bindings
                .bind(&identity("program-a"), &candidates, None)
                .as_deref(),
            Some("rank-1")
        );
    }

    #[test]
    fn token_balance_chooses_most_remaining_capacity() {
        let mut bindings = ProgramBindings::new(ProgramBindingStrategy::ReasoningTokenBalance, 160);
        let candidates = vec![
            candidate("rank-0", 1, 80.0, 100),
            candidate("rank-1", 4, 20.0, 100),
        ];
        assert_eq!(
            bindings
                .bind(&identity("program-a"), &candidates, None)
                .as_deref(),
            Some("rank-1")
        );
    }

    #[test]
    fn consistent_hash_is_independent_of_candidate_input_order() {
        let program = identity("program-a");
        let first = vec![
            candidate("rank-1", 0, 0.0, 100),
            candidate("rank-0", 0, 0.0, 100),
        ];
        let second = vec![
            candidate("rank-0", 0, 0.0, 100),
            candidate("rank-1", 0, 0.0, 100),
        ];
        let mut left = ProgramBindings::new(ProgramBindingStrategy::ConsistentHash, 160);
        let mut right = ProgramBindings::new(ProgramBindingStrategy::ConsistentHash, 160);
        assert_eq!(
            left.bind(&program, &first, None),
            right.bind(&program, &second, None)
        );
    }

    #[test]
    fn cache_aware_binding_learns_only_after_committed_dispatch() {
        let mut bindings = ProgramBindings::new(ProgramBindingStrategy::CacheAware, 160);
        let candidates = vec![
            candidate("rank-0", 0, 10.0, 100),
            candidate("rank-1", 0, 10.0, 100),
        ];
        let first = identity("program-a");
        let first_target = bindings
            .bind(&first, &candidates, Some("shared first"))
            .unwrap();
        let first_ref = ProgramRef::new("model".into(), "program-a".into(), 1);
        bindings.commit_placement(&first_ref, "shared first", &first_target);
        assert_eq!(
            bindings
                .bind(&identity("program-b"), &candidates, Some("shared second"))
                .as_deref(),
            Some(first_target.as_str())
        );
    }

    #[test]
    fn cache_aware_prefers_immediately_admissible_candidates() {
        let mut bindings = ProgramBindings::new(ProgramBindingStrategy::CacheAware, 160);
        let mut full = candidate("rank-a", 1, 90.0, 100);
        full.immediately_admissible = false;
        let available = candidate("rank-b", 0, 0.0, 100);
        bindings.commit_placement(
            &ProgramRef::new("model".into(), "owner".into(), 1),
            "shared-prefix/first",
            "rank-a",
        );
        assert_eq!(
            bindings
                .bind(
                    &identity("program-b"),
                    &[full, available],
                    Some("shared-prefix/second")
                )
                .as_deref(),
            Some("rank-b")
        );
    }
}
