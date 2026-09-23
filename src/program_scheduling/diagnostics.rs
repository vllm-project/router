//! Read-only diagnostic snapshots for Program scheduling operators.
//!
//! Snapshot construction is intentionally on demand and never runs on the
//! request-admission or periodic scheduling hot paths.

use super::scheduler::ProgramScheduler;
use super::scheduler_state::ProgramSchedulerState;
use super::{ProgramState, ProgramStatus};
use crate::metrics::RouterMetrics;
use serde::Serialize;
use std::time::Instant;

/// One live Program generation and the facts used by scheduling decisions.
#[derive(Debug, Clone, Serialize)]
pub struct ProgramDiagnostic {
    pub program: String,
    pub generation: u64,
    pub state: &'static str,
    pub status: &'static str,
    pub expected_resume: bool,
    pub task_id: Option<String>,
    pub session_id: Option<String>,
    pub agent_id: Option<String>,
    pub parent_program_id: Option<String>,
    pub root_program_id: String,
    pub blocks_parent: bool,
    pub agent_role: Option<String>,
    pub spawn_reason: Option<String>,
    pub step_id: u64,
    pub request_id: Option<String>,
    pub request_priority: i64,
    pub request_deadline_seconds: Option<f64>,
    pub expected_output_tokens: Option<usize>,
    pub kv_retention_ttl_seconds: Option<f64>,
    pub home_target: Option<String>,
    pub last_target: Option<String>,
    pub placement: Option<String>,
    pub estimated_context_tokens: usize,
    pub shared_prefix_tokens: usize,
    pub shared_prefix_freshness_remaining_seconds: Option<f64>,
    pub logical_tokens: usize,
    pub private_tokens: usize,
    pub in_flight_requests: usize,
    pub waiting_requests: usize,
    pub segment_served_rounds: usize,
    pub rounds_since_activation: usize,
    pub rounds_since_ttl_pause: usize,
    pub pause_when_idle: bool,
    pub pause_reason: Option<&'static str>,
    pub acting_seconds: Option<f64>,
    pub queued_seconds: Option<f64>,
    pub ttl_remaining_seconds: Option<f64>,
    pub force_resume_remaining_seconds: Option<f64>,
    pub force_resume_timeout_seconds: Option<f64>,
    pub force_resume_active_remaining_rounds: f64,
    pub force_resume_pool_remaining_rounds: f64,
    pub force_resume_request_throughput_per_second: f64,
}

/// Bounded rank-local workload measurements used by Progress-TTL.
#[derive(Debug, Clone, Serialize)]
pub struct RankRollingDiagnostic {
    pub request_samples: usize,
    pub continuity_samples: usize,
    pub ttl_pause_samples: usize,
    pub request_window_complete: bool,
    pub continuity_window_complete: bool,
    pub avg_prompt_tokens: f64,
    pub avg_completion_tokens: f64,
    pub avg_cached_prompt_tokens: f64,
    pub avg_cached_prompt_ratio: f64,
    pub avg_context_growth_tokens: f64,
    pub avg_e2e_seconds: f64,
    pub avg_decode_seconds: Option<f64>,
    pub avg_router_queue_seconds: f64,
    pub avg_rounds_since_ttl_pause: f64,
    pub request_throughput_per_second: f64,
    pub fitted_acting_ttl_seconds: f64,
}

/// One concrete backend or internal-DP rank and its scheduling view.
#[derive(Debug, Clone, Serialize)]
pub struct RankDiagnostic {
    pub target_id: String,
    pub base_url: String,
    pub dp_rank: Option<usize>,
    pub configured_token_capacity: Option<usize>,
    pub router_estimated_used_tokens: usize,
    pub router_estimated_headroom_tokens: Option<usize>,
    pub vllm_usage_scaled_tokens: Option<usize>,
    pub observed_active_reasoning_tokens: Option<usize>,
    pub observed_active_program_tokens: Option<usize>,
    pub tracked_active_reasoning_requests: usize,
    pub observed_router_active_reasoning_programs: Option<usize>,
    pub observed_router_active_reasoning_requests: Option<usize>,
    pub active_program_token_delta_since_observation: Option<f64>,
    pub future_pause_relief_tokens: usize,
    pub active_acting_private_tokens: usize,
    pub router_minus_vllm_used_tokens: Option<i64>,
    pub router_active_programs: usize,
    pub router_active_reasoning_programs: usize,
    pub router_active_acting_programs: usize,
    pub router_paused_reasoning_programs: usize,
    pub router_paused_acting_programs: usize,
    pub router_in_flight_requests: usize,
    pub router_queued_programs: usize,
    pub router_queued_requests: usize,
    pub vllm_kv_cache_usage: Option<f64>,
    pub vllm_running_requests: Option<usize>,
    pub vllm_waiting_requests: Option<usize>,
    pub observation_age_seconds: Option<f64>,
    pub observation_fresh: bool,
    pub capacity_accounting_source: &'static str,
    pub rolling: RankRollingDiagnostic,
    pub programs: Vec<ProgramDiagnostic>,
}

/// Complete read-only diagnostic view returned by an adapter endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct ProgramSchedulerDiagnostic {
    pub enabled: bool,
    pub binding_only: bool,
    pub global_queue: bool,
    pub resume_order: super::ProgramResumeOrder,
    pub cross_rank_headroom_ratio: f64,
    pub retained_requests: usize,
    pub ranks: Vec<RankDiagnostic>,
}

impl ProgramScheduler {
    /// Refresh bounded-cardinality gauges only from the periodic tick path.
    pub(crate) fn publish_metrics(&self, state: &ProgramSchedulerState) {
        let now = Instant::now();
        let views = state.runtime.iter_views().collect::<Vec<_>>();
        for target_id in state.targets.keys() {
            let queued = views
                .iter()
                .filter(|program| {
                    program.state == ProgramState::Paused
                        && program.waiting_requests > 0
                        && state
                            .decisions
                            .get(program.reference)
                            .is_some_and(|decision| {
                                decision.last_target.as_deref() == Some(target_id.as_str())
                            })
                })
                .count();
            let active = views
                .iter()
                .filter(|program| {
                    program.state == ProgramState::Active
                        && program.placement == Some(target_id.as_str())
                })
                .count();
            let factors = state.rank_factors.get(target_id);
            let average_impact = factors.map_or(0.0, |value| {
                super::ProgressTtlFactors::average(
                    value
                        .continuity_samples()
                        .map(|sample| sample.cache_miss_impact_seconds),
                )
            });
            RouterMetrics::set_agent_aware_rank_state(target_id, queued, active);
            RouterMetrics::set_agent_aware_capacity_observation_fresh(
                target_id,
                state
                    .observations
                    .get(target_id)
                    .is_some_and(|observation| self.observation_is_fresh(observation, now)),
            );
            RouterMetrics::set_agent_aware_adaptive_state(
                target_id,
                factors.map_or(std::time::Duration::ZERO, |value| {
                    self.policy.fitted_acting_ttl(value, average_impact)
                }),
                factors.map_or(0.0, |value| value.average_context_growth_tokens()),
                factors.map_or(0, |value| value.request_sample_count()),
                factors.map_or(0, |value| value.continuity_sample_count()),
                average_impact,
            );
        }
    }

    /// Build a stable operator snapshot without affecting scheduling state.
    pub fn diagnostics(&self) -> ProgramSchedulerDiagnostic {
        let now = Instant::now();
        let state = self.state.lock();
        let sample_limit = self.config.progress_ttl.stats_window_size;
        let views = state.runtime.iter_views().collect::<Vec<_>>();
        let mut ranks = Vec::with_capacity(state.targets.len());
        for (target_id, target) in &state.targets {
            let observation = state.observations.get(target_id);
            let capacity_accounting_source = self.capacity_accounting_source(observation, now);
            let mut programs = views
                .iter()
                .copied()
                .filter(|runtime| {
                    state
                        .decisions
                        .get(runtime.reference)
                        .is_some_and(|decision| {
                            decision.last_target.as_deref() == Some(target_id.as_str())
                        })
                })
                .map(|runtime| {
                    let decision = &state.decisions[runtime.reference];
                    let request_hints = state.runtime.front_request_hints(runtime.reference);
                    ProgramDiagnostic {
                        program: runtime.reference.redacted_id(),
                        generation: runtime.reference.generation(),
                        state: match runtime.state {
                            ProgramState::Active => "active",
                            ProgramState::Paused => "paused",
                        },
                        status: match runtime.status {
                            ProgramStatus::Reasoning => "reasoning",
                            ProgramStatus::Acting => "acting",
                        },
                        expected_resume: runtime.expected_resume,
                        task_id: decision.task_id.clone(),
                        session_id: decision.session_id.clone(),
                        agent_id: decision.agent_id.clone(),
                        parent_program_id: decision.parent_program_id.clone(),
                        root_program_id: state.lineage.root_readonly(
                            runtime.reference.model_pool(),
                            runtime.reference.program_id(),
                        ),
                        blocks_parent: decision.blocks_parent,
                        agent_role: decision.agent_role.clone(),
                        spawn_reason: decision.spawn_reason.clone(),
                        step_id: decision.step_id,
                        request_id: request_hints.and_then(|hints| hints.request_id.clone()),
                        request_priority: request_hints.map_or(0, |hints| hints.priority),
                        request_deadline_seconds: request_hints
                            .and_then(|hints| hints.deadline)
                            .map(|deadline| deadline.as_secs_f64()),
                        expected_output_tokens: request_hints
                            .and_then(|hints| hints.expected_output_tokens)
                            .or(decision.output_token_reservation),
                        kv_retention_ttl_seconds: request_hints
                            .and_then(|hints| hints.kv_retention_ttl)
                            .map(|ttl| ttl.as_secs_f64()),
                        home_target: decision.home_target.clone(),
                        last_target: decision.last_target.clone(),
                        placement: runtime.placement.map(str::to_string),
                        estimated_context_tokens: decision.estimated_context_tokens,
                        shared_prefix_tokens: decision.shared_prefix_tokens,
                        shared_prefix_freshness_remaining_seconds: decision
                            .shared_prefix_fresh_until
                            .map(|deadline| deadline.saturating_duration_since(now).as_secs_f64()),
                        logical_tokens: decision
                            .estimated_context_tokens
                            .saturating_add(self.config.progress_ttl.decode_buffer_tokens),
                        private_tokens: decision
                            .private_tokens(self.config.progress_ttl.decode_buffer_tokens)
                            .round() as usize,
                        in_flight_requests: runtime.in_flight_requests,
                        waiting_requests: runtime.waiting_requests,
                        segment_served_rounds: decision.segment_served_rounds,
                        rounds_since_activation: decision.rounds_since_activation,
                        rounds_since_ttl_pause: decision.rounds_since_ttl_pause,
                        pause_when_idle: decision.pause_when_idle,
                        pause_reason: decision.last_pause_reason.map(|reason| reason.as_str()),
                        acting_seconds: decision
                            .acting_since
                            .map(|started| now.saturating_duration_since(started).as_secs_f64()),
                        queued_seconds: decision
                            .queued_at
                            .map(|queued| now.saturating_duration_since(queued).as_secs_f64()),
                        ttl_remaining_seconds: decision
                            .ttl_deadline
                            .map(|deadline| deadline.saturating_duration_since(now).as_secs_f64()),
                        force_resume_remaining_seconds: decision
                            .force_resume_deadline
                            .map(|deadline| deadline.saturating_duration_since(now).as_secs_f64()),
                        force_resume_timeout_seconds: decision
                            .force_resume_timeout
                            .map(|timeout| timeout.as_secs_f64()),
                        force_resume_active_remaining_rounds: decision
                            .force_resume_active_remaining_rounds,
                        force_resume_pool_remaining_rounds: decision
                            .force_resume_pool_remaining_rounds,
                        force_resume_request_throughput_per_second: decision
                            .force_resume_request_throughput_per_second,
                    }
                })
                .collect::<Vec<_>>();
            programs.sort_by(|left, right| left.program.cmp(&right.program));
            let counts = |state_value, status_value| {
                programs
                    .iter()
                    .filter(|program| {
                        program.state == state_value && program.status == status_value
                    })
                    .count()
            };
            let used = self.target_usage(&state, target_id, now).round().max(0.0) as usize;
            let capacity = self.config.progress_ttl.token_capacity;
            let vllm_usage_scaled_tokens = observation
                .and_then(|value| value.kv_cache_usage)
                .zip(capacity)
                .map(|(ratio, tokens)| (ratio * tokens as f64).round() as usize);
            let observed_active_reasoning_tokens = observation
                .and_then(|value| value.estimated_active_reasoning_tokens)
                .map(|tokens| tokens.round().max(0.0) as usize);
            let observed_active_program_tokens = observation
                .and_then(|value| value.estimated_active_program_tokens)
                .map(|tokens| tokens.round().max(0.0) as usize);
            let active_acting_private_tokens = programs
                .iter()
                .filter(|program| program.state == "active" && program.status == "acting")
                .map(|program| program.private_tokens)
                .sum::<usize>();
            let future_pause_relief_tokens = programs
                .iter()
                .filter(|program| {
                    program.state == "active"
                        && program.status == "reasoning"
                        && program.pause_when_idle
                })
                .map(|program| program.private_tokens)
                .sum::<usize>();
            let router_queued_programs = programs
                .iter()
                .filter(|program| program.state == "paused" && program.waiting_requests > 0)
                .count();
            let factors = state.rank_factors.get(target_id);
            let average_impact = factors.map_or(0.0, |value| {
                super::ProgressTtlFactors::average(
                    value
                        .continuity_samples()
                        .map(|sample| sample.cache_miss_impact_seconds),
                )
            });
            let rolling = RankRollingDiagnostic {
                request_samples: factors.map_or(0, |value| value.request_sample_count()),
                continuity_samples: factors.map_or(0, |value| value.continuity_sample_count()),
                ttl_pause_samples: factors.map_or(0, |value| value.ttl_pause_sample_count()),
                request_window_complete: factors
                    .is_some_and(|value| value.request_window_complete(sample_limit)),
                continuity_window_complete: factors
                    .is_some_and(|value| value.continuity_window_complete(sample_limit)),
                avg_prompt_tokens: factors.map_or(0.0, |value| value.average_prompt_tokens()),
                avg_completion_tokens: factors
                    .map_or(0.0, |value| value.average_completion_tokens()),
                avg_cached_prompt_tokens: factors
                    .map_or(0.0, |value| value.average_cached_prompt_tokens()),
                avg_cached_prompt_ratio: factors
                    .map_or(0.0, |value| value.average_cached_prompt_ratio()),
                avg_context_growth_tokens: factors
                    .map_or(0.0, |value| value.average_context_growth_tokens()),
                avg_e2e_seconds: factors.map_or(0.0, |value| value.average_e2e_seconds()),
                avg_decode_seconds: factors.and_then(|value| value.average_decode_seconds()),
                avg_router_queue_seconds: factors
                    .map_or(0.0, |value| value.average_queue_seconds()),
                avg_rounds_since_ttl_pause: factors
                    .map_or(0.0, |value| value.average_rounds_since_ttl_pause()),
                request_throughput_per_second: factors
                    .map_or(0.0, |value| value.request_throughput_per_second()),
                fitted_acting_ttl_seconds: factors.map_or(0.0, |value| {
                    self.policy
                        .fitted_acting_ttl(value, average_impact)
                        .as_secs_f64()
                }),
            };
            ranks.push(RankDiagnostic {
                target_id: target_id.clone(),
                base_url: target.base_url.clone(),
                dp_rank: target.dp_rank,
                configured_token_capacity: capacity,
                router_estimated_used_tokens: used,
                router_estimated_headroom_tokens: capacity
                    .map(|capacity| capacity.saturating_sub(used)),
                vllm_usage_scaled_tokens,
                observed_active_reasoning_tokens,
                observed_active_program_tokens,
                tracked_active_reasoning_requests: programs
                    .iter()
                    .filter(|program| program.state == "active" && program.status == "reasoning")
                    .map(|program| program.in_flight_requests)
                    .sum(),
                observed_router_active_reasoning_programs: observation
                    .map(|value| value.router_active_reasoning_programs),
                observed_router_active_reasoning_requests: observation
                    .map(|value| value.router_active_reasoning_requests),
                active_program_token_delta_since_observation: observation
                    .map(|value| value.active_program_token_delta),
                future_pause_relief_tokens,
                active_acting_private_tokens,
                router_minus_vllm_used_tokens: vllm_usage_scaled_tokens.map(|native| {
                    i64::try_from(used)
                        .unwrap_or(i64::MAX)
                        .saturating_sub(i64::try_from(native).unwrap_or(i64::MAX))
                }),
                router_active_programs: programs
                    .iter()
                    .filter(|program| program.state == "active")
                    .count(),
                router_active_reasoning_programs: counts("active", "reasoning"),
                router_active_acting_programs: counts("active", "acting"),
                router_paused_reasoning_programs: counts("paused", "reasoning"),
                router_paused_acting_programs: counts("paused", "acting"),
                router_in_flight_requests: programs
                    .iter()
                    .map(|program| program.in_flight_requests)
                    .sum(),
                router_queued_programs,
                router_queued_requests: programs
                    .iter()
                    .map(|program| program.waiting_requests)
                    .sum(),
                vllm_kv_cache_usage: observation.and_then(|value| value.kv_cache_usage),
                vllm_running_requests: observation.and_then(|value| value.running_requests),
                vllm_waiting_requests: observation.and_then(|value| value.waiting_requests),
                observation_age_seconds: observation.and_then(|value| {
                    value
                        .observed_at
                        .map(|at| now.saturating_duration_since(at).as_secs_f64())
                }),
                observation_fresh: observation
                    .is_some_and(|value| self.observation_is_fresh(value, now)),
                capacity_accounting_source: capacity_accounting_source.as_str(),
                rolling,
                programs,
            });
        }
        ProgramSchedulerDiagnostic {
            enabled: true,
            binding_only: self.config.binding_only,
            global_queue: self.config.global_queue,
            resume_order: self.config.resume_order,
            cross_rank_headroom_ratio: self.config.cross_rank_headroom_ratio,
            retained_requests: state.runtime.retained_request_count(),
            ranks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program_scheduling::{
        ProgramIdentity, ProgramSchedulerConfig, ProgramTarget, ProgressTtlConfig,
    };
    use serde_json::json;

    #[tokio::test]
    async fn snapshot_exposes_program_and_rank_decision_facts() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://worker".into(),
            dp_rank: Some(0),
        };
        let identity = ProgramIdentity::from_request(
            None,
            Some(&json!({"vllm_xargs":{"agentic_context":{
                "program_id":"program","task_id":null,"expected_resume":true
            }}})),
            Some("model"),
        )
        .unwrap()
        .unwrap();
        let _dispatch = scheduler
            .acquire(identity, 123, std::slice::from_ref(&target), None)
            .await
            .unwrap();

        let snapshot = scheduler.diagnostics();
        assert_eq!(snapshot.ranks.len(), 1);
        assert_eq!(snapshot.ranks[0].target_id, "rank-0");
        assert_eq!(snapshot.ranks[0].router_active_reasoning_programs, 1);
        assert_eq!(snapshot.ranks[0].programs[0].estimated_context_tokens, 123);
        assert!(snapshot.ranks[0].programs[0].expected_resume);
        assert!(!snapshot.ranks[0].observation_fresh);
        assert_eq!(
            snapshot.ranks[0].capacity_accounting_source,
            "router_ledger_missing_observation"
        );
    }

    #[test]
    fn snapshot_identifies_fresh_backend_capacity_accounting() {
        let mut config = ProgramSchedulerConfig::default();
        config.progress_ttl.token_capacity = Some(1_000);
        let scheduler = ProgramScheduler::new(config);
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://worker".into(),
            dp_rank: Some(0),
        };
        scheduler.sync_targets("model", std::slice::from_ref(&target));
        let epoch = scheduler.begin_observation(std::slice::from_ref(&target));
        scheduler.apply_observations(
            epoch,
            [super::super::BackendObservation {
                target_id: target.id.clone(),
                base_url: target.base_url.clone(),
                dp_rank: target.dp_rank,
                kv_cache_usage: Some(0.25),
                running_requests: Some(1),
                waiting_requests: Some(0),
                observed_at: Instant::now(),
            }],
        );

        let snapshot = scheduler.diagnostics();
        assert!(snapshot.ranks[0].observation_fresh);
        assert_eq!(
            snapshot.ranks[0].capacity_accounting_source,
            "backend_observation"
        );
    }

    #[test]
    fn snapshot_identifies_stale_backend_capacity_fallback() {
        let config = ProgramSchedulerConfig {
            metrics_interval: std::time::Duration::from_secs(1),
            progress_ttl: ProgressTtlConfig {
                token_capacity: Some(1_000),
                ..ProgressTtlConfig::default()
            },
            ..ProgramSchedulerConfig::default()
        };
        let scheduler = ProgramScheduler::new(config);
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://worker".into(),
            dp_rank: Some(0),
        };
        scheduler.sync_targets("model", std::slice::from_ref(&target));
        {
            let mut state = scheduler.state.lock();
            let observation = state.observations.get_mut(&target.id).unwrap();
            observation.observed_at = Some(Instant::now() - std::time::Duration::from_secs(4));
            observation.estimated_active_program_tokens = Some(250.0);
        }

        let snapshot = scheduler.diagnostics();
        assert!(!snapshot.ranks[0].observation_fresh);
        assert_eq!(
            snapshot.ranks[0].capacity_accounting_source,
            "router_ledger_stale_observation"
        );
    }
}
