//! Conservative, single-group KV selection over the existing placement index.
//!
//! This is a library building block, not a registered Router policy. Callers
//! provide engine-owned identities and fresh load observations, and establish
//! trust only after a validated snapshot or clear/reconciliation barrier.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::{Locality, MatchQuery, RoutableTarget, StorageTier, TieredMatchProvider};

#[derive(Debug, Clone)]
struct SourceState {
    incarnation: u64,
    sequence: Option<u64>,
    verified: bool,
    observed_at: Instant,
    reason: &'static str,
}

/// Bounded, rank-scoped lifecycle state. One registry belongs to one cache
/// identity; sharing it across models/tokenizers/salts is not supported.
#[derive(Debug)]
pub struct KvTrustRegistry {
    sources: HashMap<RoutableTarget, SourceState>,
    capacity: usize,
}

impl KvTrustRegistry {
    pub fn new(capacity: usize) -> Result<Self, &'static str> {
        if capacity == 0 {
            return Err("trust registry capacity must be positive");
        }
        Ok(Self {
            sources: HashMap::new(),
            capacity,
        })
    }

    /// Attach/restart a source. Even reattaching the same incarnation revokes
    /// trust: receiving live deltas is not evidence of a complete baseline.
    pub fn register(&mut self, target: RoutableTarget, incarnation: u64, now: Instant) -> bool {
        if !self.sources.contains_key(&target) && self.sources.len() >= self.capacity {
            return false;
        }
        self.sources.insert(
            target,
            SourceState {
                incarnation,
                sequence: None,
                verified: false,
                observed_at: now,
                reason: "unreconciled",
            },
        );
        true
    }

    pub fn remove(&mut self, target: &RoutableTarget) {
        self.sources.remove(target);
    }

    /// Called only after the ingestion owner verifies a complete baseline.
    /// An obsolete incarnation or barrier behind observed deltas is rejected.
    /// This method must never be called from untrusted request metadata.
    pub fn reconcile(
        &mut self,
        target: &RoutableTarget,
        incarnation: u64,
        sequence: u64,
        now: Instant,
    ) -> bool {
        let Some(state) = self.sources.get_mut(target) else {
            return false;
        };
        if state.incarnation != incarnation || state.sequence.is_some_and(|last| sequence < last) {
            return false;
        }
        state.sequence = Some(sequence);
        state.observed_at = now;
        state.verified = true;
        state.reason = "verified";
        true
    }

    /// Observe an already validated/applied engine event. Duplicates do not
    /// refresh freshness; gaps, rollbacks, and epoch mismatches revoke trust.
    /// Replay deltas alone never establish or restore a baseline.
    pub fn advance(
        &mut self,
        target: &RoutableTarget,
        incarnation: u64,
        sequence: u64,
        now: Instant,
    ) -> bool {
        let Some(state) = self.sources.get_mut(target) else {
            return false;
        };
        if state.incarnation != incarnation {
            state.verified = false;
            state.reason = "incarnation_mismatch";
            return false;
        }
        if state.sequence == Some(sequence) {
            return false;
        }
        if let Some(last) = state.sequence {
            if sequence < last {
                state.verified = false;
                state.reason = "sequence_rollback";
                return false;
            }
            if last.checked_add(1) != Some(sequence) {
                state.verified = false;
                state.reason = "sequence_gap_or_rollback";
            }
        }
        state.sequence = Some(sequence);
        state.observed_at = now;
        true
    }

    pub fn mark_unavailable(&mut self, target: &RoutableTarget) {
        if let Some(state) = self.sources.get_mut(target) {
            state.verified = false;
            state.reason = "unavailable";
        }
    }

    fn failure(
        &self,
        target: &RoutableTarget,
        now: Instant,
        max_age: Duration,
    ) -> Option<&'static str> {
        let Some(state) = self.sources.get(target) else {
            return Some("unregistered");
        };
        if !state.verified {
            return Some(state.reason);
        }
        match now.checked_duration_since(state.observed_at) {
            Some(age) if age <= max_age => None,
            _ => Some("stale"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KvRoutingCandidate {
    pub target: RoutableTarget,
    pub active_requests: u64,
    pub healthy: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvRoutingMode {
    Exact,
    Fallback,
}

/// Structured decision evidence; contains no prompt text or token IDs.
#[derive(Debug, Clone)]
pub struct KvCandidateScore {
    pub target: RoutableTarget,
    pub credited_blocks: f64,
    pub normalized_load: f64,
    pub cost: f64,
}

#[derive(Debug, Clone)]
pub struct KvRoutingDecision {
    pub target: Option<RoutableTarget>,
    pub mode: KvRoutingMode,
    pub fallback_reason: Option<&'static str>,
    pub scores: Vec<KvCandidateScore>,
}

impl KvRoutingDecision {
    /// Serialize bounded per-candidate decision evidence, never request tokens
    /// or text. Instance/rank identities are supplied by the ingestion owner.
    pub fn to_json(&self) -> serde_json::Value {
        let target = |t: &RoutableTarget| {
            serde_json::json!({
                "instance_id": t.instance_id.as_ref(), "dp_rank": t.dp_rank
            })
        };
        serde_json::json!({
            "target": self.target.as_ref().map(target),
            "mode": match self.mode { KvRoutingMode::Exact => "exact", KvRoutingMode::Fallback => "fallback" },
            "fallback_reason": self.fallback_reason,
            "scores": self.scores.iter().map(|s| serde_json::json!({
                "target": target(&s.target), "credited_blocks": s.credited_blocks,
                "normalized_load": s.normalized_load, "cost": s.cost
            })).collect::<Vec<_>>()
        })
    }
}

/// Configurable weighting of uncached-prefix fraction and normalized active
/// load. Transfer credit is penalty^hops; it is not a latency prediction.
#[derive(Debug, Clone)]
pub struct TrustedKvSelector {
    cache_weight: f64,
    transfer_penalty: f64,
    max_event_age: Duration,
}

impl TrustedKvSelector {
    pub fn new(
        cache_weight: f64,
        transfer_penalty: f64,
        max_event_age: Duration,
    ) -> Result<Self, &'static str> {
        if !cache_weight.is_finite()
            || !(0.0..=1.0).contains(&cache_weight)
            || !transfer_penalty.is_finite()
            || !(0.0..=1.0).contains(&transfer_penalty)
            || max_event_age.is_zero()
        {
            return Err("weights must be finite values in [0,1] and freshness must be positive");
        }
        Ok(Self {
            cache_weight,
            transfer_penalty,
            max_event_age,
        })
    }

    /// Query a model-scoped index and choose a concrete instance/DP rank.
    /// Any untrusted healthy candidate causes a conservative pool-wide
    /// least-load fallback, without querying or crediting the KV provider.
    pub fn select(
        &self,
        provider: &dyn TieredMatchProvider,
        query: &MatchQuery,
        candidates: &[KvRoutingCandidate],
        trust: &KvTrustRegistry,
        now: Instant,
    ) -> KvRoutingDecision {
        let eligible: Vec<_> = candidates.iter().filter(|c| c.healthy).collect();
        let fallback = if eligible.is_empty() {
            Some("no_healthy_targets")
        } else if query.local_hashes.is_empty() {
            Some("empty_query")
        } else {
            eligible
                .iter()
                .find_map(|c| trust.failure(&c.target, now, self.max_event_age))
        };
        let mode = if fallback.is_some() {
            KvRoutingMode::Fallback
        } else {
            KvRoutingMode::Exact
        };
        let matches = if mode == KvRoutingMode::Exact {
            provider.find_tiered_matches(query)
        } else {
            Vec::new()
        };
        let maximum_load = eligible
            .iter()
            .map(|c| c.active_requests)
            .max()
            .unwrap_or(0) as f64;
        let query_length = query.local_hashes.len() as f64;
        let mut scores: Vec<_> = eligible
            .iter()
            .map(|candidate| {
                let mut credited_blocks = 0.0_f64;
                for hit in matches.iter().filter(|m| {
                    m.target == candidate.target && query.tiers_of_interest.contains(&m.tier)
                }) {
                    // Later held segments cannot be credited across a missing
                    // leading prefix. The multi-segment transfer scorer is later work.
                    let prefix = hit
                        .segments
                        .iter()
                        .filter(|(start, end)| *start == 0 && end > start)
                        .map(|(_, end)| f64::from(*end).min(query_length))
                        .fold(0.0_f64, f64::max);
                    let tier_hops = match hit.tier {
                        StorageTier::Device => 0,
                        StorageTier::HostPinned => 1,
                        StorageTier::Disk => 2,
                        StorageTier::External => 3,
                    };
                    let hops = tier_hops + i32::from(hit.locality == Locality::Remote);
                    credited_blocks =
                        credited_blocks.max(prefix * self.transfer_penalty.powi(hops));
                }
                let normalized_load = candidate.active_requests as f64 / (maximum_load + 1.0);
                let cost = if mode == KvRoutingMode::Fallback {
                    normalized_load
                } else {
                    self.cache_weight * (1.0 - credited_blocks / query_length)
                        + (1.0 - self.cache_weight) * normalized_load
                };
                KvCandidateScore {
                    target: candidate.target.clone(),
                    credited_blocks,
                    normalized_load,
                    cost,
                }
            })
            .collect();
        scores.sort_by(|a, b| {
            a.cost
                .total_cmp(&b.cost)
                .then_with(|| a.target.instance_id.cmp(&b.target.instance_id))
                .then_with(|| a.target.dp_rank.cmp(&b.target.dp_rank))
        });
        KvRoutingDecision {
            target: scores.first().map(|s| s.target.clone()),
            mode,
            fallback_reason: fallback,
            scores,
        }
    }
}
