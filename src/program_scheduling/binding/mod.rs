//! Initial Program-to-backend binding policies.
//!
//! Binding is evaluated only for a new Program generation. Admission, resume,
//! and cross-rank runtime placement remain ProgramScheduler responsibilities.

use super::ProgramIdentity;
use super::ProgramRef;
use crate::policies::ConsistentHashPolicy;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
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
    ) -> Option<String>;
}

/// Stable Program binding table plus one selected initial-binding policy.
///
/// ProgramScheduler serializes calls to this object with its state lock. The
/// table records home affinity only; it does not record runtime placement.
#[derive(Debug)]
pub struct ProgramBindings {
    policy: Box<dyn ProgramBindingPolicy>,
    bindings: HashMap<ProgramBindingKey, String>,
}

impl ProgramBindings {
    /// Construct a binding table using one of the behavior-compatible policies.
    pub fn new(strategy: ProgramBindingStrategy, hash_virtual_nodes: u32) -> Self {
        let policy: Box<dyn ProgramBindingPolicy> = match strategy {
            ProgramBindingStrategy::ConsistentHash => {
                Box::new(ConsistentHashBinding::new(hash_virtual_nodes))
            }
            ProgramBindingStrategy::ProgramRoundRobin => Box::new(RoundRobinBinding::default()),
            ProgramBindingStrategy::LeastProgramCount => Box::new(LeastProgramCountBinding),
            ProgramBindingStrategy::ReasoningTokenBalance => Box::new(ReasoningTokenBalanceBinding),
        };
        Self {
            policy,
            bindings: HashMap::new(),
        }
    }

    /// Return an existing binding, or establish one for a new generation.
    pub fn bind(
        &mut self,
        identity: &ProgramIdentity,
        candidates: &[ProgramBindingCandidate],
    ) -> Option<String> {
        let key = ProgramBindingKey::from(identity);
        let live_targets: BTreeSet<&str> = candidates
            .iter()
            .map(|candidate| candidate.target_id.as_str())
            .collect();
        if let Some(target_id) = self.bindings.get(&key) {
            if live_targets.contains(target_id.as_str()) {
                return Some(target_id.clone());
            }
        }
        let mut ordered = candidates.to_vec();
        ordered.sort_by(|left, right| left.target_id.cmp(&right.target_id));
        let target_id = self.policy.select(identity, &ordered)?;
        self.bindings.insert(key, target_id.clone());
        Some(target_id)
    }

    /// Look up the home backend without making a new policy decision.
    pub fn binding(&self, identity: &ProgramIdentity) -> Option<&str> {
        self.bindings
            .get(&ProgramBindingKey::from(identity))
            .map(String::as_str)
    }

    /// Release the binding for one Program generation.
    pub fn release(&mut self, identity: &ProgramIdentity) -> Option<String> {
        self.bindings.remove(&ProgramBindingKey::from(identity))
    }

    /// Release by exact runtime reference after lifecycle termination.
    pub(crate) fn release_program(&mut self, program: &ProgramRef) -> Option<String> {
        self.bindings.remove(&ProgramBindingKey {
            model_pool: program.model_pool().to_string(),
            program_id: program.program_id().to_string(),
        })
    }

    /// Invalidate bindings whose backend disappeared from discovery.
    ///
    /// The scheduler may rebind the returned Programs using its normal target
    /// synchronization path. This method never chooses a resume placement.
    pub fn remove_target(&mut self, target_id: &str) -> usize {
        let previous_len = self.bindings.len();
        self.bindings.retain(|_, bound| bound != target_id);
        previous_len - self.bindings.len()
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
    ) -> Option<String> {
        if candidates.is_empty() {
            return None;
        }
        let next = self
            .next_by_model
            .entry(identity.model_pool().to_string())
            .or_default();
        let target_id = candidates[*next % candidates.len()].target_id.clone();
        *next = next.wrapping_add(1);
        Some(target_id)
    }
}

#[derive(Debug)]
struct LeastProgramCountBinding;

impl ProgramBindingPolicy for LeastProgramCountBinding {
    fn select(
        &mut self,
        _identity: &ProgramIdentity,
        candidates: &[ProgramBindingCandidate],
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

    fn candidate(id: &str, programs: usize, used: f64, capacity: usize) -> ProgramBindingCandidate {
        ProgramBindingCandidate {
            target_id: id.to_string(),
            accounted_programs: programs,
            accounted_tokens: used,
            capacity_tokens: Some(capacity),
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
            bindings.bind(&program, &candidates).as_deref(),
            Some("rank-0")
        );
        assert_eq!(
            bindings.bind(&program, &candidates).as_deref(),
            Some("rank-0")
        );
        assert_eq!(bindings.binding(&program), Some("rank-0"));
        assert_eq!(bindings.remove_target("rank-0"), 1);
        assert_eq!(
            bindings.bind(&program, &candidates).as_deref(),
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
                .bind(&identity("program-a"), &candidates)
                .as_deref(),
            Some("rank-0")
        );
        assert_eq!(
            bindings
                .bind(&identity("program-b"), &candidates)
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
                .bind(&identity("program-a"), &candidates)
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
                .bind(&identity("program-a"), &candidates)
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
        assert_eq!(left.bind(&program, &first), right.bind(&program, &second));
    }
}
