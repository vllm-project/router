//! Initial Program-to-backend binding policies.
//!
//! Binding is evaluated only for a new Program generation. Admission, resume,
//! and cross-rank runtime placement remain ProgramScheduler responsibilities.

use super::ProgramIdentity;
use super::ProgramRef;
use crate::policies::{CacheAwareCandidate, CacheAwarePolicy, ConsistentHashPolicy};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;

/// Initial binding algorithm selected for Program generations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgramBindingStrategy {
    /// Preserve stable session/task affinity with a consistent hash ring.
    #[default]
    ConsistentHash,
    /// Assign each new Program generation to the next backend in target order.
    ProgramRoundRobin,
    /// Select the backend with the fewest accounted Programs.
    LeastProgramCount,
    /// Select the backend with the greatest unaccounted token capacity.
    ReasoningTokenBalance,
    /// Combine request-text prefix affinity with current KV headroom.
    CacheAware,
    /// Rank by candidate-qualified KV reuse and current execution load.
    KvAware,
}

/// Program-level load snapshot used only for initial binding.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramBindingCandidate {
    /// Stable backend or DP-rank identifier.
    pub target_id: String,
    /// Active Programs plus paused reasoning Programs bound to the target.
    pub accounted_programs: usize,
    /// Private tokens of active Programs plus paused reasoning Programs.
    pub accounted_tokens: f64,
    /// Explicit per-target token capacity required by token balance.
    pub capacity_tokens: Option<usize>,
    /// Current target KV pressure normalized to [0, 1].
    pub kv_pressure: f64,
    /// Whether the new Program fits the Cache-aware immediate-admission gate.
    pub immediately_admissible: bool,
}

/// One initial Program binding algorithm.
///
/// Implementations receive a target list sorted by `target_id`. They must not
/// apply admission capacity, resume ordering, or cross-rank migration rules.
pub trait ProgramBindingPolicy: Send + Sync + Debug {
    /// Select the home backend for one previously unbound Program generation.
    fn select(
        &mut self,
        identity: &ProgramIdentity,
        candidates: &[ProgramBindingCandidate],
        routing_text: Option<&str>,
    ) -> Option<String>;

    fn cursor(&self, _pool: &str) -> usize {
        0
    }

    fn recommend(
        &mut self,
        identity: &ProgramIdentity,
        candidates: &[ProgramBindingCandidate],
        text: Option<&str>,
    ) -> Option<String> {
        self.select(identity, candidates, text)
    }

    fn confirm(&mut self, _pool: &str) {}

    fn commit_placement(&self, _model_pool: &str, _text: &str, _target_id: &str) {}

    fn remove_target(&self, _model_pool: &str, _target_id: &str) {}
}

/// Construct the binding algorithm separately from admission state.
pub(crate) fn make_binding_policy(
    strategy: ProgramBindingStrategy,
    hash_virtual_nodes: u32,
) -> Box<dyn ProgramBindingPolicy> {
    match strategy {
        ProgramBindingStrategy::ConsistentHash | ProgramBindingStrategy::KvAware => {
            Box::new(ConsistentHashBinding::new(hash_virtual_nodes))
        }
        ProgramBindingStrategy::ProgramRoundRobin => Box::new(RoundRobinBinding::default()),
        ProgramBindingStrategy::LeastProgramCount => Box::new(LeastProgramCountBinding),
        ProgramBindingStrategy::ReasoningTokenBalance => Box::new(ReasoningTokenBalanceBinding),
        ProgramBindingStrategy::CacheAware => Box::new(CacheAwareBinding::default()),
    }
}

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

#[derive(Debug)]
struct ConsistentHashBinding {
    virtual_nodes: u32,
    rings: HashMap<String, (Vec<String>, BTreeMap<u64, String>)>,
}

impl ConsistentHashBinding {
    fn new(virtual_nodes: u32) -> Self {
        Self {
            virtual_nodes,
            rings: HashMap::new(),
        }
    }

    fn ring<'a>(&'a mut self, model_pool: &str, targets: &[String]) -> &'a BTreeMap<u64, String> {
        let rebuild = self
            .rings
            .get(model_pool)
            .is_none_or(|(current, _)| current != targets);
        if rebuild {
            let mut ring = BTreeMap::new();
            for target_id in targets {
                for virtual_node in 0..self.virtual_nodes {
                    let virtual_key = format!("{target_id}:{virtual_node}");
                    ring.insert(
                        ConsistentHashPolicy::fbi_hash(&virtual_key),
                        target_id.clone(),
                    );
                }
            }
            self.rings
                .insert(model_pool.to_string(), (targets.to_vec(), ring));
        }
        &self
            .rings
            .get(model_pool)
            .expect("ring was inserted above")
            .1
    }
}

impl ProgramBindingPolicy for ConsistentHashBinding {
    fn select(
        &mut self,
        identity: &ProgramIdentity,
        candidates: &[ProgramBindingCandidate],
        _routing_text: Option<&str>,
    ) -> Option<String> {
        let targets: Vec<String> = candidates
            .iter()
            .map(|candidate| candidate.target_id.clone())
            .collect();
        let ring = self.ring(identity.model_pool(), &targets);
        if ring.is_empty() {
            return None;
        }
        let hash = ConsistentHashPolicy::fbi_hash(identity.placement_hash_key());
        ring.range(hash..)
            .next()
            .or_else(|| ring.iter().next())
            .map(|(_, target_id)| target_id.clone())
    }
}

#[derive(Debug, Default)]
struct RoundRobinBinding {
    next_by_model: HashMap<String, usize>,
}

impl ProgramBindingPolicy for RoundRobinBinding {
    fn select(
        &mut self,
        identity: &ProgramIdentity,
        candidates: &[ProgramBindingCandidate],
        _routing_text: Option<&str>,
    ) -> Option<String> {
        let selected = self.recommend(identity, candidates, _routing_text)?;
        self.confirm(identity.model_pool());
        Some(selected)
    }

    fn cursor(&self, pool: &str) -> usize {
        self.next_by_model.get(pool).copied().unwrap_or_default()
    }

    fn recommend(
        &mut self,
        identity: &ProgramIdentity,
        candidates: &[ProgramBindingCandidate],
        _: Option<&str>,
    ) -> Option<String> {
        if candidates.is_empty() {
            return None;
        }
        Some(
            candidates[self.cursor(identity.model_pool()) % candidates.len()]
                .target_id
                .clone(),
        )
    }

    fn confirm(&mut self, pool: &str) {
        let next = self.next_by_model.entry(pool.to_owned()).or_default();
        *next = next.wrapping_add(1);
    }
}

#[derive(Debug)]
struct LeastProgramCountBinding;

impl ProgramBindingPolicy for LeastProgramCountBinding {
    fn select(
        &mut self,
        _identity: &ProgramIdentity,
        candidates: &[ProgramBindingCandidate],
        _routing_text: Option<&str>,
    ) -> Option<String> {
        candidates
            .iter()
            .min_by_key(|candidate| (candidate.accounted_programs, candidate.target_id.as_str()))
            .map(|candidate| candidate.target_id.clone())
    }
}

#[derive(Debug)]
struct ReasoningTokenBalanceBinding;

impl ProgramBindingPolicy for ReasoningTokenBalanceBinding {
    fn select(
        &mut self,
        _identity: &ProgramIdentity,
        candidates: &[ProgramBindingCandidate],
        _routing_text: Option<&str>,
    ) -> Option<String> {
        candidates
            .iter()
            .map(|candidate| {
                let capacity = candidate
                    .capacity_tokens
                    .expect("reasoning-token balance requires explicit capacity")
                    as f64;
                (candidate, capacity - candidate.accounted_tokens)
            })
            .max_by(|(left, left_remaining), (right, right_remaining)| {
                left_remaining
                    .total_cmp(right_remaining)
                    .then_with(|| right.target_id.cmp(&left.target_id))
            })
            .map(|(candidate, _)| candidate.target_id.clone())
    }
}

#[derive(Debug, Default)]
struct CacheAwareBinding {
    affinity: CacheAwarePolicy,
}

impl ProgramBindingPolicy for CacheAwareBinding {
    fn select(
        &mut self,
        identity: &ProgramIdentity,
        candidates: &[ProgramBindingCandidate],
        routing_text: Option<&str>,
    ) -> Option<String> {
        let eligible = candidates
            .iter()
            .filter(|candidate| candidate.immediately_admissible)
            .collect::<Vec<_>>();
        let candidates = if eligible.is_empty() {
            candidates.iter().collect::<Vec<_>>()
        } else {
            eligible
        }
        .into_iter()
        .map(|candidate| CacheAwareCandidate {
            target_id: &candidate.target_id,
            kv_pressure: candidate.kv_pressure,
        })
        .collect::<Vec<_>>();
        self.affinity
            .evaluate_initial_placement(
                identity.model_pool(),
                routing_text.unwrap_or_default(),
                &candidates,
            )
            .map(|placement| placement.target_id)
    }

    fn commit_placement(&self, model_pool: &str, text: &str, target_id: &str) {
        self.affinity.commit_placement(model_pool, text, target_id);
    }

    fn remove_target(&self, model_pool: &str, target_id: &str) {
        self.affinity.remove_placement_target(model_pool, target_id);
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
