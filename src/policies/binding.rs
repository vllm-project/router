//! Policies for a new Program generation. Admission owns the binding table and ticket lifecycle.
use super::{CacheAwareCandidate, CacheAwarePolicy, ConsistentHashPolicy};
use crate::program_scheduling::ProgramIdentity;
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

impl ProgramBindingStrategy {
    pub(crate) fn uses_kv_index(self) -> bool {
        self == Self::KvAware
    }

    pub(crate) fn needs_routing_text(self) -> bool {
        self == Self::CacheAware
    }
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
