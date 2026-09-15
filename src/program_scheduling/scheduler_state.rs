//! Internal state shared by Program scheduling decision modules.
//!
//! All values in this module are protected by the ProgramScheduler mutex.
//! They are decision facts, not an additional Program lifecycle state machine.

use super::{
    ProgramBindingStrategy, ProgramBindings, ProgramRef, ProgramRuntime, ProgramTarget,
    ProgressTtlConfig, ProgressTtlFactors,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Complete opt-in Program scheduler configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramSchedulerConfig {
    /// Preserve Program identity and binding but bypass Router admission.
    pub binding_only: bool,
    /// Permit paused reasoning Programs to resume on another target.
    pub global_queue: bool,
    /// Required destination-over-source headroom in complete Program contexts.
    pub cross_rank_headroom_ratio: f64,
    /// Initial Program generation binding policy.
    pub binding_strategy: ProgramBindingStrategy,
    /// Consistent-hash virtual nodes used only by Program binding.
    pub hash_virtual_nodes: u32,
    /// Coarse Program-count guard when explicit token capacity is unavailable.
    pub max_active_programs_per_target: usize,
    /// Backend observation and periodic scheduling interval.
    pub metrics_interval: Duration,
    /// Block admission at or above this native waiting-request count.
    pub admission_waiting_request_threshold: usize,
    /// Maximum Router RequestPool residence time.
    pub queue_timeout: Duration,
    /// Static cold-start upper bound for forced resume.
    pub force_resume_timeout: Duration,
    /// Retain an idle paused Program generation for this duration.
    pub paused_retention_ttl: Duration,
    /// Cold-start reusable-prefix observation lifetime.
    pub shared_prefix_freshness_warmup: Duration,
    /// Estimated whole-KV-pool turnovers before refreshing shared prefix.
    pub shared_prefix_freshness_kv_turnovers: f64,
    /// Context size used to derive bounded per-rank privilege slots.
    pub privileged_max_context_tokens: usize,
    /// Maximum lifetime of one progress privilege lease.
    pub privileged_ttl: Duration,
    /// Allow resume to reclaim active acting Programs before TTL expiry.
    pub resume_reclaim_acting_programs: bool,
    /// Rank-local Progress-TTL formulas and calibration.
    pub progress_ttl: ProgressTtlConfig,
}

impl Default for ProgramSchedulerConfig {
    fn default() -> Self {
        Self {
            binding_only: false,
            global_queue: false,
            cross_rank_headroom_ratio: 1.2,
            binding_strategy: ProgramBindingStrategy::ConsistentHash,
            hash_virtual_nodes: 160,
            max_active_programs_per_target: 64,
            metrics_interval: Duration::from_secs(1),
            admission_waiting_request_threshold: 1,
            queue_timeout: Duration::from_secs(600),
            force_resume_timeout: Duration::from_secs(300),
            paused_retention_ttl: Duration::from_secs(1800),
            shared_prefix_freshness_warmup: Duration::from_secs(100),
            shared_prefix_freshness_kv_turnovers: 2.0,
            privileged_max_context_tokens: 262_144,
            privileged_ttl: Duration::from_secs(5),
            resume_reclaim_acting_programs: false,
            progress_ttl: ProgressTtlConfig::default(),
        }
    }
}

/// Cause of the most recently committed pause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProgramPauseReason {
    TtlExpired,
    CapacityRepair,
    MaxSegmentYield,
    PrivilegeHandoff,
    RequestFailed,
    RequestCancelled,
}

impl ProgramPauseReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::TtlExpired => "ttl_expired",
            Self::CapacityRepair => "capacity_repair",
            Self::MaxSegmentYield => "max_segment_yield",
            Self::PrivilegeHandoff => "privilege_handoff",
            Self::RequestFailed => "request_failed",
            Self::RequestCancelled => "request_cancelled",
        }
    }
}

/// Scheduling facts associated with one exact live Program generation.
#[derive(Debug, Clone)]
pub(crate) struct ProgramDecisionState {
    pub(crate) placement_key: String,
    pub(crate) home_target: Option<String>,
    pub(crate) last_target: Option<String>,
    pub(crate) estimated_context_tokens: usize,
    pub(crate) context_shrink_observations: u8,
    pub(crate) completed_requests: usize,
    pub(crate) segment_served_rounds: usize,
    pub(crate) rounds_since_activation: usize,
    pub(crate) ttl_pause_sampled_segment_rounds: usize,
    pub(crate) rounds_since_ttl_pause: usize,
    pub(crate) segment_started_at: Option<Instant>,
    pub(crate) last_context_tokens: Option<usize>,
    pub(crate) last_completion_tokens: usize,
    pub(crate) shared_prefix_tokens: usize,
    pub(crate) shared_prefix_observed: bool,
    pub(crate) shared_prefix_freshness_anchor_at: Instant,
    pub(crate) shared_prefix_fresh_until: Option<Instant>,
    pub(crate) lifetime_generated_tokens: usize,
    pub(crate) last_request_finished_at: Option<Instant>,
    pub(crate) last_cache_miss_impact_seconds: f64,
    pub(crate) privilege_deadline: Option<Instant>,
    pub(crate) privilege_ttl_expired: bool,
    pub(crate) acting_since: Option<Instant>,
    pub(crate) ttl_deadline: Option<Instant>,
    pub(crate) queued_at: Option<Instant>,
    pub(crate) force_resume_deadline: Option<Instant>,
    pub(crate) paused_at: Option<Instant>,
    pub(crate) last_pause_reason: Option<ProgramPauseReason>,
    pub(crate) pause_when_idle: bool,
    pub(crate) terminate_when_idle: bool,
}

impl ProgramDecisionState {
    pub(crate) fn new(
        placement_key: String,
        home_target: Option<String>,
        estimated_context_tokens: usize,
        now: Instant,
    ) -> Self {
        Self {
            placement_key,
            last_target: home_target.clone(),
            home_target,
            estimated_context_tokens,
            context_shrink_observations: 0,
            completed_requests: 0,
            segment_served_rounds: 0,
            rounds_since_activation: 0,
            ttl_pause_sampled_segment_rounds: 0,
            rounds_since_ttl_pause: 0,
            segment_started_at: None,
            last_context_tokens: None,
            last_completion_tokens: 0,
            shared_prefix_tokens: 0,
            shared_prefix_observed: false,
            shared_prefix_freshness_anchor_at: now,
            shared_prefix_fresh_until: None,
            lifetime_generated_tokens: 0,
            last_request_finished_at: None,
            last_cache_miss_impact_seconds: 0.0,
            privilege_deadline: None,
            privilege_ttl_expired: false,
            acting_since: None,
            ttl_deadline: None,
            queued_at: Some(now),
            force_resume_deadline: None,
            paused_at: Some(now),
            last_pause_reason: None,
            pause_when_idle: false,
            terminate_when_idle: false,
        }
    }

    pub(crate) fn private_tokens(&self, decode_buffer_tokens: usize) -> f64 {
        self.estimated_context_tokens
            .saturating_sub(self.shared_prefix_tokens)
            .saturating_add(decode_buffer_tokens) as f64
    }

    pub(crate) fn observe_completed_context(&mut self, observed_context_tokens: usize) {
        const SHRINK_CONFIRMATIONS: u8 = 2;
        if observed_context_tokens >= self.estimated_context_tokens {
            self.estimated_context_tokens = observed_context_tokens;
            self.context_shrink_observations = 0;
            return;
        }
        self.context_shrink_observations = self.context_shrink_observations.saturating_add(1);
        if self.context_shrink_observations >= SHRINK_CONFIRMATIONS {
            self.estimated_context_tokens = observed_context_tokens;
            self.context_shrink_observations = 0;
        }
    }
}

/// Most recent raw backend observation plus Router epoch accounting.
#[derive(Debug, Clone, Default)]
pub(crate) struct RankObservationState {
    pub(crate) kv_cache_usage: Option<f64>,
    pub(crate) running_requests: Option<usize>,
    pub(crate) waiting_requests: Option<usize>,
    pub(crate) router_active_reasoning_programs: usize,
    pub(crate) router_active_reasoning_requests: usize,
    pub(crate) estimated_active_program_tokens: Option<f64>,
    pub(crate) observed_at: Option<Instant>,
    pub(crate) active_program_token_delta: f64,
}

/// All mutable data committed under one scheduler lock.
#[derive(Debug)]
pub(crate) struct ProgramSchedulerState {
    pub(crate) runtime: ProgramRuntime,
    pub(crate) bindings: ProgramBindings,
    pub(crate) decisions: HashMap<ProgramRef, ProgramDecisionState>,
    pub(crate) targets: BTreeMap<String, ProgramTarget>,
    pub(crate) model_targets: HashMap<String, BTreeSet<String>>,
    pub(crate) rank_queues: HashMap<String, VecDeque<ProgramRef>>,
    pub(crate) global_queues: HashMap<String, VecDeque<ProgramRef>>,
    pub(crate) observations: HashMap<String, RankObservationState>,
    pub(crate) rank_factors: HashMap<String, ProgressTtlFactors>,
}

impl ProgramSchedulerState {
    pub(crate) fn new(config: &ProgramSchedulerConfig) -> Self {
        Self {
            runtime: ProgramRuntime::default(),
            bindings: ProgramBindings::new(config.binding_strategy, config.hash_virtual_nodes),
            decisions: HashMap::new(),
            targets: BTreeMap::new(),
            model_targets: HashMap::new(),
            rank_queues: HashMap::new(),
            global_queues: HashMap::new(),
            observations: HashMap::new(),
            rank_factors: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_current_router_scheduler() {
        let config = ProgramSchedulerConfig::default();
        assert!(!config.binding_only);
        assert!(!config.global_queue);
        assert_eq!(config.cross_rank_headroom_ratio, 1.2);
        assert_eq!(config.progress_ttl.stats_window_size, 100);
        assert_eq!(config.progress_ttl.acting_ttl, Duration::from_secs(10));
        assert_eq!(config.progress_ttl.max_segment_rounds, 14);
        assert!(!config.resume_reclaim_acting_programs);
    }

    #[test]
    fn context_shrink_requires_two_authoritative_observations() {
        let now = Instant::now();
        let mut state = ProgramDecisionState::new("task".into(), Some("rank-0".into()), 100, now);
        state.observe_completed_context(80);
        assert_eq!(state.estimated_context_tokens, 100);
        state.observe_completed_context(80);
        assert_eq!(state.estimated_context_tokens, 80);
        state.observe_completed_context(120);
        assert_eq!(state.estimated_context_tokens, 120);
    }
}
