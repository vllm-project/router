//! Internal state shared by Program scheduling decision modules.
//!
//! All values in this module are protected by the ProgramScheduler mutex.
//! They are decision facts, not an additional Program lifecycle state machine.

use super::lineage::ProgramLineage;
use super::{
    ProgramBindingStrategy, ProgramBindings, ProgramRef, ProgramRuntime,
    ProgramSchedulingEnableKey, ProgramTarget, ProgressTtlConfig, ProgressTtlFactors,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Ordinary paused-Program resume order within each priority tier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgramResumeOrder {
    /// Resume the Program whose retained request entered the RequestPool first.
    Fcfs,
    /// Prefer the Program that most recently completed a request.
    #[default]
    Mru,
}

/// Complete opt-in Program scheduler configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramSchedulerConfig {
    /// PD enforces serial turns and immediate capacity even after force timeout.
    pub pd_mode: bool,
    pub pd_explicit_ttl: bool,
    /// PD treats native occupancy and Program reservations as overlapping KV.
    pub pd_exclusive_capacity_accounting: bool,
    /// Configured P prefill permits; also the tightened cap under P KV pressure.
    pub pd_prefill_permits: usize,
    /// P KV-usage watermark above which the Program-count cap tightens.
    pub pd_prefill_kv_high: f64,
    /// P KV-cache sampling window used for guard freshness.
    pub pd_prefill_kv_window: Duration,
    /// Metadata source that opts an individual request into Program scheduling.
    pub enable_key: ProgramSchedulingEnableKey,
    /// Preserve Program identity and binding but bypass Router admission.
    pub binding_only: bool,
    /// Permit paused reasoning Programs to resume on another target.
    pub global_queue: bool,
    /// Ordinary resume order after force-resume, priority, and one-shot yield.
    pub resume_order: ProgramResumeOrder,
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
    /// Rank-local Progress-TTL formulas and calibration.
    pub progress_ttl: ProgressTtlConfig,
}

impl Default for ProgramSchedulerConfig {
    fn default() -> Self {
        Self {
            pd_mode: false,
            pd_explicit_ttl: false,
            pd_exclusive_capacity_accounting: true,
            pd_prefill_permits: 1,
            pd_prefill_kv_high: 0.85,
            pd_prefill_kv_window: Duration::from_secs(10),
            enable_key: ProgramSchedulingEnableKey::default(),
            binding_only: false,
            global_queue: false,
            resume_order: ProgramResumeOrder::default(),
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
            progress_ttl: ProgressTtlConfig::default(),
        }
    }
}

/// Resolve an optional admission bound: zero means "auto".
///
/// Two Programs per configured P prefill permit covers the short tool gaps
/// between turns; the permit count is the configured proxy for the per-target
/// decode slot count. The clamp keeps pathological configurations bounded.
pub(crate) fn resolve_auto_limit(configured: usize, prefill_permits: usize) -> usize {
    if configured > 0 {
        return configured;
    }
    (2 * prefill_permits.max(1)).clamp(4, 64)
}

impl From<&crate::config::types::ProgramSchedulingConfig> for ProgramSchedulerConfig {
    fn from(config: &crate::config::types::ProgramSchedulingConfig) -> Self {
        // Auto sizing is a PD-mode concept. A regular-mode config that omitted
        // the fields deserializes the zero defaults; keep the historical
        // 64-Program cap and 1-request overload gate for those deployments.
        let pd_mode = config.pd.enabled;
        let permits = config.pd.max_prefill_requests_per_target;
        let max_active_programs_per_target = if config.max_active_programs_per_target > 0 {
            config.max_active_programs_per_target
        } else if pd_mode {
            resolve_auto_limit(0, permits)
        } else {
            64
        };
        let admission_waiting_request_threshold = if config.admission_waiting_request_threshold > 0
        {
            config.admission_waiting_request_threshold
        } else if pd_mode {
            resolve_auto_limit(0, permits)
        } else {
            1
        };
        Self {
            pd_mode: config.pd.enabled,
            pd_explicit_ttl: config.pd.explicit_ttl,
            pd_exclusive_capacity_accounting: config.pd.exclusive_capacity_accounting,
            pd_prefill_permits: config.pd.max_prefill_requests_per_target.max(1),
            pd_prefill_kv_high: config.pd.prefill_kv_high_watermark,
            pd_prefill_kv_window: Duration::from_secs_f64(config.pd.prefill_kv_window_seconds),
            enable_key: config.enable_key,
            binding_only: config.binding_only,
            global_queue: config.global_queue,
            resume_order: config.resume_order,
            cross_rank_headroom_ratio: config.cross_rank_headroom_ratio,
            binding_strategy: config.binding_strategy,
            hash_virtual_nodes: config.hash_virtual_nodes,
            max_active_programs_per_target,
            metrics_interval: Duration::from_secs_f64(config.metrics_interval_seconds),
            admission_waiting_request_threshold,
            queue_timeout: Duration::from_secs_f64(config.queue_timeout_seconds),
            force_resume_timeout: Duration::from_secs_f64(config.force_resume_timeout_seconds),
            paused_retention_ttl: Duration::from_secs_f64(config.paused_retention_ttl_seconds),
            shared_prefix_freshness_warmup: Duration::from_secs_f64(
                config.shared_prefix_freshness_warmup_seconds,
            ),
            shared_prefix_freshness_kv_turnovers: config.shared_prefix_freshness_kv_turnovers,
            progress_ttl: ProgressTtlConfig {
                token_capacity: config.token_capacity_per_dp_rank,
                decode_buffer_tokens: config.decode_buffer_tokens,
                max_acting_ttl: Duration::from_secs_f64(config.max_acting_ttl_seconds),
                high_watermark_ratio: config.high_watermark_ratio,
                low_watermark_ratio: config.low_watermark_ratio,
                max_segment_rounds: config.max_segment_rounds,
                stats_window_size: config.stats_window_size,
                enable_batch_gain_admission: config.enable_batch_gain_admission,
                prefill: config.prefill_cost_model,
                decode: config.decode_throughput_model,
            },
        }
    }
}

/// Cause of the most recently committed pause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProgramPauseReason {
    TtlExpired,
    CapacityRepair,
    MaxSegmentYield,
    RequestFailed,
    RequestCancelled,
}

impl ProgramPauseReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::TtlExpired => "ttl_expired",
            Self::CapacityRepair => "capacity_repair",
            Self::MaxSegmentYield => "max_segment_yield",
            Self::RequestFailed => "request_failed",
            Self::RequestCancelled => "request_cancelled",
        }
    }
}

/// Scheduling facts associated with one exact live Program generation.
#[derive(Debug, Clone)]
pub(crate) struct ProgramDecisionState {
    pub(crate) placement_key: String,
    pub(crate) placement_hash_key: String,
    pub(crate) task_id: Option<String>,
    pub(crate) session_id: Option<String>,
    pub(crate) agent_id: Option<String>,
    pub(crate) parent_program_id: Option<String>,
    pub(crate) blocks_parent: bool,
    pub(crate) agent_role: Option<String>,
    pub(crate) spawn_reason: Option<String>,
    pub(crate) step_id: u64,
    pub(crate) home_target: Option<String>,
    pub(crate) last_target: Option<String>,
    pub(crate) estimated_context_tokens: usize,
    pub(crate) output_token_reservation: Option<usize>,
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
    pub(crate) acting_since: Option<Instant>,
    pub(crate) ttl_deadline: Option<Instant>,
    pub(crate) queued_at: Option<Instant>,
    pub(crate) force_resume_deadline: Option<Instant>,
    pub(crate) force_resume_timeout: Option<Duration>,
    pub(crate) force_resume_active_remaining_rounds: f64,
    pub(crate) force_resume_pool_remaining_rounds: f64,
    pub(crate) force_resume_request_throughput_per_second: f64,
    pub(crate) paused_at: Option<Instant>,
    pub(crate) last_pause_reason: Option<ProgramPauseReason>,
    pub(crate) pause_when_idle: bool,
    pub(crate) terminate_when_idle: bool,
}

impl ProgramDecisionState {
    pub(crate) fn new(
        placement_key: String,
        placement_hash_key: String,
        home_target: Option<String>,
        estimated_context_tokens: usize,
        now: Instant,
    ) -> Self {
        Self {
            placement_key,
            placement_hash_key,
            task_id: None,
            session_id: None,
            agent_id: None,
            parent_program_id: None,
            blocks_parent: false,
            agent_role: None,
            spawn_reason: None,
            step_id: 0,
            last_target: home_target.clone(),
            home_target,
            estimated_context_tokens,
            output_token_reservation: None,
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
            acting_since: None,
            ttl_deadline: None,
            queued_at: Some(now),
            force_resume_deadline: None,
            force_resume_timeout: None,
            force_resume_active_remaining_rounds: 0.0,
            force_resume_pool_remaining_rounds: 0.0,
            force_resume_request_throughput_per_second: 0.0,
            paused_at: Some(now),
            last_pause_reason: None,
            pause_when_idle: false,
            terminate_when_idle: false,
        }
    }

    pub(crate) fn private_tokens(&self, decode_buffer_tokens: usize) -> f64 {
        self.private_tokens_with_output(decode_buffer_tokens, None)
    }

    pub(crate) fn private_tokens_with_output(
        &self,
        decode_buffer_tokens: usize,
        expected_output_tokens: Option<usize>,
    ) -> f64 {
        self.private_tokens_from_context(
            self.estimated_context_tokens,
            decode_buffer_tokens,
            expected_output_tokens,
        )
    }

    /// Private KV footprint for an explicit request-scoped context estimate.
    ///
    /// Admission of a queued turn must charge that turn's own prompt, not the
    /// largest prompt of any later turn already retained behind it.
    pub(crate) fn private_tokens_from_context(
        &self,
        context_tokens: usize,
        decode_buffer_tokens: usize,
        expected_output_tokens: Option<usize>,
    ) -> f64 {
        context_tokens
            .saturating_sub(self.shared_prefix_tokens)
            .saturating_add(
                expected_output_tokens
                    .or(self.output_token_reservation)
                    .unwrap_or(decode_buffer_tokens),
            ) as f64
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
    pub(crate) estimated_active_reasoning_tokens: Option<f64>,
    pub(crate) estimated_active_program_tokens: Option<f64>,
    pub(crate) observed_at: Option<Instant>,
    pub(crate) active_program_token_delta: f64,
    pub(crate) reported_capacity_source: Option<CapacityAccountingSource>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CapacityAccountingSource {
    BackendObservation,
    RouterLedgerMissingObservation,
    RouterLedgerStaleObservation,
    RouterLedgerMissingTokenEstimate,
}

impl CapacityAccountingSource {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::BackendObservation => "backend_observation",
            Self::RouterLedgerMissingObservation => "router_ledger_missing_observation",
            Self::RouterLedgerStaleObservation => "router_ledger_stale_observation",
            Self::RouterLedgerMissingTokenEstimate => "router_ledger_missing_token_estimate",
        }
    }
}

/// All mutable data committed under one scheduler lock.
#[derive(Debug)]
pub(crate) struct ProgramSchedulerState {
    pub(crate) runtime: ProgramRuntime,
    pub(crate) bindings: ProgramBindings,
    pub(crate) decisions: HashMap<ProgramRef, ProgramDecisionState>,
    pub(crate) targets: BTreeMap<String, ProgramTarget>,
    pub(crate) model_targets: HashMap<String, BTreeSet<String>>,
    pub(crate) model_target_snapshots: HashMap<String, Arc<[ProgramTarget]>>,
    /// Monotonic fence for target topology or target-attribute changes.
    pub(crate) target_snapshot_revision: u64,
    pub(crate) rank_queues: HashMap<String, VecDeque<ProgramRef>>,
    pub(crate) global_queues: HashMap<String, VecDeque<ProgramRef>>,
    pub(crate) observations: HashMap<String, RankObservationState>,
    pub(crate) rank_factors: HashMap<String, ProgressTtlFactors>,
    pub(crate) lineage: ProgramLineage,
    /// Most recent P-side KV-cache pressure sample.
    pub(crate) prefill_pressure: Option<PrefillPressure>,
}

/// P KV-cache usage as seen from the prefill engines.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefillPressure {
    pub(crate) kv_usage: f64,
    pub(crate) observed_at: Instant,
}

impl ProgramSchedulerState {
    pub(crate) fn new(config: &ProgramSchedulerConfig) -> Self {
        Self {
            runtime: ProgramRuntime::default(),
            bindings: ProgramBindings::new(config.binding_strategy, config.hash_virtual_nodes),
            decisions: HashMap::new(),
            targets: BTreeMap::new(),
            model_targets: HashMap::new(),
            model_target_snapshots: HashMap::new(),
            target_snapshot_revision: 0,
            rank_queues: HashMap::new(),
            global_queues: HashMap::new(),
            observations: HashMap::new(),
            rank_factors: HashMap::new(),
            lineage: ProgramLineage::default(),
            prefill_pressure: None,
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::program_scheduling::{DecodeThroughputModel, PrefillCostModel};

    #[test]
    fn defaults_match_the_current_router_scheduler() {
        let config = ProgramSchedulerConfig::default();
        assert!(!config.binding_only);
        assert!(!config.global_queue);
        assert_eq!(config.resume_order, ProgramResumeOrder::Mru);
        assert_eq!(config.cross_rank_headroom_ratio, 1.2);
        assert_eq!(config.progress_ttl.stats_window_size, 100);
        assert_eq!(config.progress_ttl.max_acting_ttl, Duration::from_secs(10));
        assert_eq!(config.progress_ttl.max_segment_rounds, 14);
        assert_eq!(config.progress_ttl.prefill, PrefillCostModel::default());
        assert_eq!(config.progress_ttl.decode, DecodeThroughputModel::default());
    }

    #[test]
    fn auto_admission_limits_follow_the_prefill_permits() {
        assert_eq!(resolve_auto_limit(0, 4), 8);
        assert_eq!(resolve_auto_limit(0, 1), 4);
        assert_eq!(resolve_auto_limit(0, 64), 64);
        assert_eq!(resolve_auto_limit(3, 4), 3);
        assert_eq!(resolve_auto_limit(96, 4), 96);

        let parsed: crate::config::types::ProgramSchedulingConfig =
            serde_json::from_str(r#"{"pd":{"enabled":true,"max_prefill_requests_per_target":4}}"#)
                .unwrap();
        let runtime = ProgramSchedulerConfig::from(&parsed);
        assert_eq!(runtime.max_active_programs_per_target, 8);
        assert_eq!(runtime.admission_waiting_request_threshold, 8);
        assert!(runtime.pd_exclusive_capacity_accounting);
        assert_eq!(runtime.pd_prefill_permits, 4);
        assert_eq!(runtime.pd_prefill_kv_high, 0.85);

        // Regular deployments deserialize the zero defaults but must keep the
        // historical admission pool and overload gate.
        let regular: crate::config::types::ProgramSchedulingConfig =
            serde_json::from_str("{}").unwrap();
        let runtime = ProgramSchedulerConfig::from(&regular);
        assert!(!runtime.pd_mode);
        assert_eq!(runtime.max_active_programs_per_target, 64);
        assert_eq!(runtime.admission_waiting_request_threshold, 1);
    }

    #[test]
    fn context_shrink_requires_two_authoritative_observations() {
        let now = Instant::now();
        let mut state = ProgramDecisionState::new(
            "task".into(),
            "hash".into(),
            Some("rank-0".into()),
            100,
            now,
        );
        state.observe_completed_context(80);
        assert_eq!(state.estimated_context_tokens, 100);
        state.observe_completed_context(80);
        assert_eq!(state.estimated_context_tokens, 80);
        state.observe_completed_context(120);
        assert_eq!(state.estimated_context_tokens, 120);
    }
}
