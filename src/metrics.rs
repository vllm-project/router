use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};
use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct PrometheusConfig {
    pub port: u16,
    pub host: String,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            port: 29000,
            host: "0.0.0.0".to_string(),
        }
    }
}

pub fn init_metrics() {
    // Request metrics
    describe_counter!(
        "vllm_router_requests_total",
        "Total number of requests by route and method"
    );
    describe_counter!(
        "vllm_router_request_errors_total",
        "Total number of request errors by route and error type"
    );
    describe_counter!(
        "vllm_router_retries_total",
        "Total number of request retries by route"
    );
    describe_histogram!(
        "vllm_router_retry_backoff_duration_seconds",
        "Backoff duration in seconds by attempt index"
    );
    describe_counter!(
        "vllm_router_retries_exhausted_total",
        "Total number of requests that exhausted retries by route"
    );

    // Circuit breaker metrics
    describe_gauge!(
        "vllm_router_cb_state",
        "Circuit breaker state per worker (0=closed, 1=open, 2=half_open)"
    );
    describe_counter!(
        "vllm_router_cb_state_transitions_total",
        "Total number of circuit breaker state transitions by worker"
    );
    describe_counter!(
        "vllm_router_cb_outcomes_total",
        "Total number of circuit breaker outcomes by worker and outcome type (success/failure)"
    );

    // Worker metrics
    describe_gauge!(
        "vllm_router_active_workers",
        "Number of currently active workers"
    );
    describe_gauge!(
        "vllm_router_worker_health",
        "Worker health status (1=healthy, 0=unhealthy)"
    );
    describe_counter!(
        "vllm_router_processed_requests_total",
        "Total requests processed by each worker"
    );

    // Policy metrics
    describe_counter!(
        "vllm_router_kv_routing_inputs_total",
        "KV-aware token preparation by transport and result"
    );
    describe_counter!(
        "vllm_router_kv_placement_lookups_total",
        "KV-aware placement lookups and reasons for skipping them"
    );
    describe_counter!(
        "vllm_router_kv_source_recovery_events_total",
        "KV event source recovery events by mode and reason"
    );
    describe_counter!(
        "vllm_router_kv_replay_outcomes_total",
        "KV event replay attempts aggregated by outcome"
    );
    describe_counter!(
        "vllm_router_policy_decisions_total",
        "Total routing policy decisions by policy and worker"
    );
    describe_counter!(
        "vllm_router_prefix_routing_decisions_total",
        "Prefix routing decisions by policy, stage and outcome"
    );
    describe_counter!("vllm_router_cache_hits_total", "Total cache hits");
    describe_counter!("vllm_router_cache_misses_total", "Total cache misses");
    describe_counter!(
        "vllm_router_load_balancing_events_total",
        "Total load balancing trigger events"
    );
    describe_gauge!("vllm_router_max_load", "Maximum worker load");
    describe_gauge!("vllm_router_min_load", "Minimum worker load");

    describe_gauge!(
        "vllm_router_agent_aware_rank_queue_size",
        "Queued Programs attributed to each Program-scheduling target"
    );
    describe_gauge!(
        "vllm_router_agent_aware_rank_active_programs",
        "Active Programs on each Program-scheduling target"
    );
    describe_counter!(
        "vllm_router_agent_aware_transitions_total",
        "Committed Program scheduling transitions by target and reason"
    );
    describe_histogram!(
        "vllm_router_agent_aware_queue_wait_seconds",
        "Time retained in the Program RequestPool before dispatch"
    );
    describe_gauge!(
        "vllm_router_agent_aware_fitted_ttl_seconds",
        "Representative fitted acting TTL by target"
    );
    describe_gauge!(
        "vllm_router_agent_aware_context_growth_tokens",
        "Rolling mean Program context growth by target"
    );
    describe_gauge!(
        "vllm_router_agent_aware_rolling_samples",
        "Retained Program-scheduling samples by target and window"
    );
    describe_gauge!(
        "vllm_router_agent_aware_token_estimate_coefficient",
        "Current prompt tokens-per-byte coefficient by model pool and endpoint"
    );
    describe_counter!(
        "vllm_router_agent_aware_token_estimate_feedback_total",
        "Prompt-token estimator feedback outcomes"
    );
    describe_histogram!(
        "vllm_router_agent_aware_cache_miss_impact_seconds",
        "Estimated cache-miss recovery impact for completed Program requests"
    );
    describe_gauge!(
        "vllm_router_agent_aware_cache_miss_impact_rolling_mean_seconds",
        "Rolling mean cache-miss recovery impact by Program-scheduling target"
    );
    describe_gauge!(
        "vllm_router_agent_aware_capacity_observation_fresh",
        "Whether backend capacity accounting is fresh and usable by target"
    );
    describe_histogram!(
        "vllm_router_agent_aware_request_interval_seconds",
        "Program request intervals used by the Progress-TTL fit"
    );
    describe_histogram!(
        "vllm_router_agent_aware_armed_ttl_seconds",
        "Final acting TTL armed after a Program request"
    );
    describe_histogram!(
        "vllm_router_agent_aware_shared_prefix_observed_tokens",
        "Explicit cached-token observation used to refresh a Program shared prefix"
    );
    for (name, description) in [
        (
            "vllm_router_agent_aware_admission_used_tokens",
            "Managed target tokens before Program admission",
        ),
        (
            "vllm_router_agent_aware_admission_required_tokens",
            "Candidate tokens required by a committed Program admission",
        ),
        (
            "vllm_router_agent_aware_admission_growth_reserve_tokens",
            "Growth reserve included in a committed Program admission",
        ),
        (
            "vllm_router_agent_aware_admission_capacity_pressure_ratio",
            "Managed capacity pressure before Program admission",
        ),
        (
            "vllm_router_agent_aware_admission_projected_pressure_ratio",
            "Projected capacity pressure after Program admission and growth reserve",
        ),
        (
            "vllm_router_agent_aware_admission_backend_kv_usage_ratio",
            "Backend KV usage observed at Program admission",
        ),
        (
            "vllm_router_agent_aware_admission_backend_running_requests",
            "Backend running requests observed at Program admission",
        ),
        (
            "vllm_router_agent_aware_admission_backend_waiting_requests",
            "Backend waiting requests observed at Program admission",
        ),
        (
            "vllm_router_agent_aware_admission_capacity_observation_age_seconds",
            "Age of the backend capacity observation at Program admission",
        ),
    ] {
        describe_histogram!(name, description);
    }

    describe_counter!(
        "vllm_router_agent_aware_context_token_facts_total",
        "Applied Program context-token fact sources by model pool and endpoint"
    );
    describe_histogram!(
        "vllm_router_agent_aware_context_token_fact_tokens",
        "Applied Program context-token counts by fact source"
    );
    describe_counter!(
        "vllm_router_agent_aware_context_token_fallbacks_total",
        "Prepared context-token facts not applied, by fallback reason"
    );
    describe_counter!(
        "vllm_router_agent_aware_context_token_shadow_total",
        "Admission counterfactuals recorded by shadow token-count mode"
    );
    describe_histogram!(
        "vllm_router_agent_aware_context_token_observation_delta_ratio",
        "Relative difference between prepared and Engine-observed prompt tokens"
    );
    describe_counter!(
        "vllm_router_agent_aware_context_token_revocations_total",
        "Tokenizer fingerprints revoked after an Engine disagreement"
    );
    describe_gauge!(
        "vllm_router_agent_aware_context_token_revoked_fingerprints",
        "Fingerprints currently revoked in the bounded scheduler map"
    );
    describe_gauge!(
        "vllm_router_agent_aware_context_token_pool_saturated",
        "Whether exact token facts are saturated off for a model pool"
    );
    describe_counter!(
        "vllm_router_agent_aware_context_token_unobserved_total",
        "Prepared facts settled without an Engine prompt count to reconcile"
    );
    describe_gauge!(
        "vllm_router_agent_aware_context_token_usage_unavailable",
        "Whether optional token facts were disabled after missing Engine usage"
    );
    describe_counter!(
        "vllm_router_agent_aware_usage_filter_errors_total",
        "Injected-usage streams rejected for oversized lines or unsupported encoding"
    );
    describe_gauge!(
        "vllm_router_agent_aware_context_token_mode",
        "Active Program context-token count mode (1 for the selected mode)"
    );
    describe_histogram!(
        "vllm_router_agent_aware_token_prepare_seconds",
        "Router-side text tokenization time for prepared exact facts"
    );
    describe_counter!(
        "vllm_router_agent_aware_token_prepare_failures_total",
        "Prepared exact facts skipped because Router-side text preparation failed"
    );
    // PD-specific metrics
    describe_counter!(
        "vllm_router_pd_requests_total",
        "Total PD requests by route"
    );
    describe_counter!(
        "vllm_router_pd_prefill_requests_total",
        "Total prefill requests per worker"
    );
    describe_counter!(
        "vllm_router_pd_decode_requests_total",
        "Total decode requests per worker"
    );
    describe_counter!(
        "vllm_router_pd_errors_total",
        "Total PD errors by error type"
    );
    describe_counter!(
        "vllm_router_pd_prefill_errors_total",
        "Total prefill server errors"
    );
    describe_counter!(
        "vllm_router_pd_decode_errors_total",
        "Total decode server errors"
    );
    describe_histogram!(
        "vllm_router_pd_request_duration_seconds",
        "PD request duration by route"
    );
    for (name, help) in [
        (
            "vllm_router_pd_prefill_decisions_total",
            "Prefill routing decisions by fallback reason",
        ),
        (
            "vllm_router_pd_credited_prefix_tokens_total",
            "Prefix tokens credited by Prefill routing",
        ),
        (
            "vllm_router_pd_affinity_decisions_total",
            "P/D affinity decisions by stage and outcome",
        ),
        (
            "vllm_router_pd_decode_commit_total",
            "Decode commitment outcomes after Prefill",
        ),
        (
            "vllm_router_pd_body_over_limit_total",
            "P/D stage responses that exceeded the Router buffer limit",
        ),
    ] {
        describe_counter!(name, help);
    }
    describe_histogram!(
        "vllm_router_pd_new_prefill_tokens",
        "Uncached prompt tokens estimated at Prefill admission"
    );

    // Generate request specific metrics
    describe_histogram!(
        "vllm_router_generate_duration_seconds",
        "Generate request duration"
    );

    // Embedding request specific metrics
    describe_counter!("vllm_router_embeddings_total", "Total embedding requests");
    describe_histogram!(
        "vllm_router_embeddings_duration_seconds",
        "Embedding request duration"
    );
    describe_counter!(
        "vllm_router_embeddings_errors_total",
        "Embedding request errors"
    );
    describe_gauge!("vllm_router_embeddings_queue_size", "Embedding queue size");

    // Running requests gauge for cache-aware policy
    describe_gauge!(
        "vllm_router_running_requests",
        "Number of running requests per worker"
    );

    // Tokenizer encode cache metrics
    describe_counter!(
        "vllm_tokenizer_cache_hits_total",
        "Total tokenizer encode calls served from the exact-match cache"
    );
    describe_counter!(
        "vllm_tokenizer_cache_misses_total",
        "Total tokenizer encode calls that ran the underlying tokenizer"
    );
    describe_counter!(
        "vllm_tokenizer_cache_evictions_total",
        "Total tokenizer cache entries evicted to satisfy the entry or byte budget"
    );
    describe_counter!(
        "vllm_tokenizer_cache_oversized_total",
        "Total tokenizer encode results not cached because they exceeded the per-entry byte limit"
    );
    describe_gauge!(
        "vllm_tokenizer_cache_entries",
        "Current number of entries in the tokenizer encode cache"
    );
    describe_gauge!(
        "vllm_tokenizer_cache_bytes",
        "Estimated bytes retained by the tokenizer encode cache"
    );

    // L1 prefix encoding cache metrics
    describe_counter!(
        "vllm_tokenizer_l1_lookups_total",
        "Total prompts encoded through the L1 cache"
    );
    describe_counter!(
        "vllm_tokenizer_l1_hits_total",
        "Total L1 lookups that reused at least one cached token"
    );
    describe_counter!(
        "vllm_tokenizer_l1_cached_tokens_total",
        "Total prompt tokens taken from the L1 cache"
    );
    describe_counter!(
        "vllm_tokenizer_l1_encoded_tokens_total",
        "Total prompt tokens encoded by the tokenizer on the L1 path"
    );
    describe_counter!(
        "vllm_tokenizer_l1_evictions_total",
        "Total L1 tree nodes evicted to satisfy the byte or node budget"
    );
    describe_counter!(
        "vllm_tokenizer_l1_verify_mismatches_total",
        "Total L1 verify-mode mismatches"
    );
    describe_counter!(
        "vllm_tokenizer_l1_bypasses_total",
        "Total prompts on an L1-enabled model that skipped L1, by reason"
    );
    describe_gauge!(
        "vllm_tokenizer_l1_nodes",
        "Current number of L1 prefix tree nodes"
    );
    describe_gauge!(
        "vllm_tokenizer_l1_bytes",
        "Estimated bytes retained by the L1 cache"
    );
}

pub fn start_prometheus(config: PrometheusConfig) -> Result<(), Box<dyn std::error::Error>> {
    let ip_addr: IpAddr = config.host.parse().map_err(|error| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("invalid Prometheus host {:?}: {error}", config.host),
        )
    })?;
    let socket_addr = SocketAddr::new(ip_addr, config.port);

    let duration_matcher = Matcher::Suffix(String::from("duration_seconds"));
    let duration_bucket = [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
        60.0, 90.0, 120.0, 180.0, 240.0,
    ];

    let builder = PrometheusBuilder::new()
        .with_http_listener(socket_addr)
        .upkeep_timeout(Duration::from_secs(5 * 60));
    let builder = set_program_scheduling_buckets(builder);
    builder
        .set_buckets_for_metric(duration_matcher, &duration_bucket)?
        .install()?;
    init_metrics();
    Ok(())
}

fn set_program_scheduling_buckets(builder: PrometheusBuilder) -> PrometheusBuilder {
    const CACHE_MISS_SECONDS: &[f64] =
        &[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0];
    const INTERVAL_SECONDS: &[f64] = &[
        0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0,
    ];
    const ARMED_TTL_SECONDS: &[f64] = &[
        0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0,
    ];
    const TOKEN_BUCKETS: &[f64] = &[
        0.0,
        128.0,
        256.0,
        512.0,
        1_024.0,
        2_048.0,
        4_096.0,
        8_192.0,
        16_384.0,
        32_768.0,
        65_536.0,
        131_072.0,
        262_144.0,
        524_288.0,
        1_048_576.0,
    ];
    const RATIO_BUCKETS: &[f64] = &[0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0];
    const REQUEST_BUCKETS: &[f64] = &[0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];

    let exact = |name: &str| Matcher::Full(name.to_string());
    let set = |builder: PrometheusBuilder, name: &str, buckets: &[f64]| {
        builder
            .set_buckets_for_metric(exact(name), buckets)
            .unwrap_or_else(|error| panic!("failed to set buckets for {name}: {error}"))
    };
    let builder = set(
        builder,
        "vllm_router_agent_aware_cache_miss_impact_seconds",
        CACHE_MISS_SECONDS,
    );
    let builder = set(
        builder,
        "vllm_router_agent_aware_request_interval_seconds",
        INTERVAL_SECONDS,
    );
    let builder = set(
        builder,
        "vllm_router_agent_aware_armed_ttl_seconds",
        ARMED_TTL_SECONDS,
    );
    let mut builder = set(
        builder,
        "vllm_router_agent_aware_shared_prefix_observed_tokens",
        TOKEN_BUCKETS,
    );
    for name in [
        "vllm_router_agent_aware_admission_used_tokens",
        "vllm_router_agent_aware_admission_required_tokens",
        "vllm_router_agent_aware_admission_growth_reserve_tokens",
        "vllm_router_pd_new_prefill_tokens",
    ] {
        builder = set(builder, name, TOKEN_BUCKETS);
    }
    for name in [
        "vllm_router_agent_aware_admission_capacity_pressure_ratio",
        "vllm_router_agent_aware_admission_projected_pressure_ratio",
        "vllm_router_agent_aware_admission_backend_kv_usage_ratio",
    ] {
        builder = set(builder, name, RATIO_BUCKETS);
    }
    for name in [
        "vllm_router_agent_aware_admission_backend_running_requests",
        "vllm_router_agent_aware_admission_backend_waiting_requests",
    ] {
        builder = set(builder, name, REQUEST_BUCKETS);
    }
    set(
        builder,
        "vllm_router_agent_aware_admission_capacity_observation_age_seconds",
        INTERVAL_SECONDS,
    )
}

pub(crate) struct AgentAwareAdmissionMetrics {
    pub(crate) used_tokens: f64,
    pub(crate) required_tokens: f64,
    pub(crate) reserve_tokens: f64,
    pub(crate) capacity_tokens: Option<usize>,
    pub(crate) backend_kv_usage_ratio: Option<f64>,
    pub(crate) backend_running_requests: Option<usize>,
    pub(crate) backend_waiting_requests: Option<usize>,
    pub(crate) observation_age: Option<Duration>,
}

/// Keep each lookup result bound to the current stage's recorder.
#[derive(Default)]
pub(crate) struct PlacementLookups {
    counters: smallvec::SmallVec<[(&'static str, metrics::Counter); 2]>,
}

impl PlacementLookups {
    pub(crate) fn record(&mut self, result: &'static str) {
        self.record_count(result, 1);
    }

    pub(crate) fn record_count(&mut self, result: &'static str, count: u64) {
        if let Some((_, counter)) = self.counters.iter().find(|(label, _)| *label == result) {
            counter.increment(count);
        } else {
            let counter = kv_placement_lookup_counter(result);
            counter.increment(count);
            self.counters.push((result, counter));
        }
    }
}

fn kv_placement_lookup_counter(result: &'static str) -> metrics::Counter {
    counter!("vllm_router_kv_placement_lookups_total", "result" => result)
}

pub struct RouterMetrics;

pub struct TokenizerMetrics;

impl RouterMetrics {
    /// Separates real prefix decisions, explicit binding, and load fallback.
    pub fn record_prefix_routing_decision(
        policy: &'static str,
        stage: &'static str,
        outcome: &'static str,
    ) {
        counter!("vllm_router_prefix_routing_decisions_total",
            "policy" => policy, "stage" => stage, "outcome" => outcome)
        .increment(1);
    }

    pub fn record_kv_placement_lookup(result: &'static str) {
        kv_placement_lookup_counter(result).increment(1);
    }

    pub(crate) fn record_kv_source_recovery(strict: bool, reason: &'static str) {
        counter!("vllm_router_kv_source_recovery_events_total",
            "mode" => if strict { "strict" } else { "compatible" },
            "reason" => reason)
        .increment(1);
    }

    pub fn record_kv_replay_outcome(outcome: &'static str) {
        counter!("vllm_router_kv_replay_outcomes_total",
            "outcome" => outcome)
        .increment(1);
    }

    pub fn record_kv_routing_input(transport: &'static str, result: &'static str) {
        counter!("vllm_router_kv_routing_inputs_total", "transport" => transport, "result" => result).increment(1);
    }

    pub fn record_pd_kv_decision(work: Option<u64>, credit: Option<u64>, fallback: Option<&str>) {
        counter!("vllm_router_pd_prefill_decisions_total",
            "fallback" => fallback.unwrap_or("none").to_owned())
        .increment(1);
        if let Some(work) = work {
            histogram!("vllm_router_pd_new_prefill_tokens").record(work as f64);
        }
        if let Some(credit) = credit {
            counter!("vllm_router_pd_credited_prefix_tokens_total").increment(credit);
        }
    }

    /// Per-stage kv_aware session-affinity outcome: bind / sticky_hit / rebind /
    /// no_session / capacity_bypass.
    pub fn record_pd_affinity_decision(stage: &str, outcome: &str) {
        counter!("vllm_router_pd_affinity_decisions_total",
            "stage" => stage.to_owned(),
            "outcome" => outcome.to_owned())
        .increment(1);
    }

    /// Deferred decode commitment result: same / changed / empty.
    pub fn record_pd_decode_commit(result: &str) {
        counter!("vllm_router_pd_decode_commit_total",
            "result" => result.to_owned())
        .increment(1);
    }

    // Request metrics
    pub fn record_request(route: &str) {
        counter!("vllm_router_requests_total",
            "route" => route.to_string()
        )
        .increment(1);
    }

    pub fn record_request_error(route: &str, error_type: &str) {
        counter!("vllm_router_request_errors_total",
            "route" => route.to_string(),
            "error_type" => error_type.to_string()
        )
        .increment(1);
    }

    pub fn record_retry(route: &str) {
        counter!("vllm_router_retries_total",
            "route" => route.to_string()
        )
        .increment(1);
    }

    pub fn record_retry_backoff_duration(duration: Duration, attempt: u32) {
        histogram!("vllm_router_retry_backoff_duration_seconds",
            "attempt" => attempt.to_string()
        )
        .record(duration.as_secs_f64());
    }

    pub fn record_retries_exhausted(route: &str) {
        counter!("vllm_router_retries_exhausted_total",
            "route" => route.to_string()
        )
        .increment(1);
    }

    // Worker metrics
    pub fn set_active_workers(count: usize) {
        gauge!("vllm_router_active_workers").set(count as f64);
    }

    pub fn set_worker_health(worker_url: &str, healthy: bool) {
        gauge!("vllm_router_worker_health",
            "worker" => worker_url.to_string()
        )
        .set(if healthy { 1.0 } else { 0.0 });
    }

    pub fn record_processed_request(worker_url: &str) {
        counter!("vllm_router_processed_requests_total",
            "worker" => worker_url.to_string()
        )
        .increment(1);
    }

    // Policy metrics
    pub fn record_policy_decision(policy: &str, worker: &str) {
        counter!("vllm_router_policy_decisions_total",
            "policy" => policy.to_string(),
            "worker" => worker.to_string()
        )
        .increment(1);
    }

    pub fn record_cache_hit() {
        counter!("vllm_router_cache_hits_total").increment(1);
    }

    pub fn record_cache_miss() {
        counter!("vllm_router_cache_misses_total").increment(1);
    }

    pub fn record_load_balancing_event() {
        counter!("vllm_router_load_balancing_events_total").increment(1);
    }

    pub fn set_load_range(max_load: usize, min_load: usize) {
        gauge!("vllm_router_max_load").set(max_load as f64);
        gauge!("vllm_router_min_load").set(min_load as f64);
    }

    pub fn set_agent_aware_rank_state(target: &str, queued: usize, active: usize) {
        gauge!(
            "vllm_router_agent_aware_rank_queue_size",
            "target" => target.to_string()
        )
        .set(queued as f64);
        gauge!(
            "vllm_router_agent_aware_rank_active_programs",
            "target" => target.to_string()
        )
        .set(active as f64);
    }

    pub fn record_agent_aware_transition(target: &str, reason: &'static str) {
        counter!(
            "vllm_router_agent_aware_transitions_total",
            "target" => target.to_string(),
            "reason" => reason
        )
        .increment(1);
    }

    pub fn record_agent_aware_queue_wait(target: &str, duration: Duration) {
        histogram!(
            "vllm_router_agent_aware_queue_wait_seconds",
            "target" => target.to_string()
        )
        .record(duration.as_secs_f64());
    }

    pub fn record_agent_aware_token_estimate_feedback(
        model_pool: &str,
        endpoint: &str,
        observed: bool,
        coefficient: Option<f64>,
    ) {
        counter!(
            "vllm_router_agent_aware_token_estimate_feedback_total",
            "model_pool" => model_pool.to_string(),
            "endpoint" => endpoint.to_string(),
            "outcome" => if observed { "observed" } else { "missing" }
        )
        .increment(1);
        if let Some(coefficient) = coefficient {
            gauge!(
                "vllm_router_agent_aware_token_estimate_coefficient",
                "model_pool" => model_pool.to_string(),
                "endpoint" => endpoint.to_string()
            )
            .set(coefficient);
        }
    }

    pub fn set_agent_aware_adaptive_state(
        target: &str,
        fitted_ttl: Duration,
        context_growth_tokens: f64,
        request_samples: usize,
        continuity_samples: usize,
        cache_miss_impact_rolling_mean_seconds: f64,
    ) {
        gauge!("vllm_router_agent_aware_fitted_ttl_seconds", "target" => target.to_string())
            .set(fitted_ttl.as_secs_f64());
        gauge!("vllm_router_agent_aware_context_growth_tokens", "target" => target.to_string())
            .set(context_growth_tokens);
        gauge!(
            "vllm_router_agent_aware_rolling_samples",
            "target" => target.to_string(),
            "window" => "request"
        )
        .set(request_samples as f64);
        gauge!(
            "vllm_router_agent_aware_rolling_samples",
            "target" => target.to_string(),
            "window" => "continuity"
        )
        .set(continuity_samples as f64);
        gauge!(
            "vllm_router_agent_aware_cache_miss_impact_rolling_mean_seconds",
            "target" => target.to_string()
        )
        .set(cache_miss_impact_rolling_mean_seconds);
    }

    pub fn set_agent_aware_capacity_observation_fresh(target: &str, fresh: bool) {
        gauge!(
            "vllm_router_agent_aware_capacity_observation_fresh",
            "target" => target.to_string()
        )
        .set(if fresh { 1.0 } else { 0.0 });
    }

    pub fn set_workflow_prediction_profile_keys(return_time: usize, output_length: usize) {
        gauge!(
            "vllm_router_workflow_prediction_profile_keys",
            "predictor" => "return_time"
        )
        .set(return_time as f64);
        gauge!(
            "vllm_router_workflow_prediction_profile_keys",
            "predictor" => "output_length"
        )
        .set(output_length as f64);
    }

    pub fn record_agent_aware_cache_miss_impact(target: &str, seconds: f64) {
        histogram!(
            "vllm_router_agent_aware_cache_miss_impact_seconds",
            "target" => target.to_string()
        )
        .record(seconds);
    }

    pub fn record_agent_aware_request_interval(target: &str, seconds: f64) {
        histogram!(
            "vllm_router_agent_aware_request_interval_seconds",
            "target" => target.to_string()
        )
        .record(seconds);
    }

    pub fn record_agent_aware_armed_ttl(target: &str, source: &'static str, ttl: Duration) {
        histogram!(
            "vllm_router_agent_aware_armed_ttl_seconds",
            "target" => target.to_string(),
            "source" => source
        )
        .record(ttl.as_secs_f64());
    }

    pub fn record_workflow_return_prediction(
        target: &str,
        source: &'static str,
        probability: f64,
        ttl_seconds: f64,
        support: usize,
        backoff_level: usize,
    ) {
        histogram!(
            "vllm_router_workflow_return_probability",
            "target" => target.to_string(),
            "source" => source
        )
        .record(probability);
        histogram!(
            "vllm_router_workflow_predicted_ttl_seconds",
            "target" => target.to_string(),
            "source" => source
        )
        .record(ttl_seconds);
        histogram!(
            "vllm_router_workflow_return_prediction_support",
            "target" => target.to_string(),
            "source" => source
        )
        .record(support as f64);
        counter!(
            "vllm_router_workflow_return_prediction_total",
            "target" => target.to_string(),
            "source" => source,
            "backoff_level" => backoff_level.to_string()
        )
        .increment(1);
    }

    pub fn record_workflow_return_observation(
        target: &str,
        actual_seconds: f64,
        probability: Option<f64>,
        horizon_seconds: f64,
    ) {
        histogram!(
            "vllm_router_workflow_return_actual_seconds",
            "target" => target.to_string()
        )
        .record(actual_seconds);
        if let Some(probability) = probability {
            let outcome = if actual_seconds <= horizon_seconds {
                1.0
            } else {
                0.0
            };
            let calibrated = probability.clamp(1e-9, 1.0 - 1e-9);
            histogram!(
                "vllm_router_workflow_return_brier_score",
                "target" => target.to_string()
            )
            .record((probability - outcome).powi(2));
            histogram!(
                "vllm_router_workflow_return_nll",
                "target" => target.to_string()
            )
            .record(if outcome > 0.0 {
                -calibrated.ln()
            } else {
                -(1.0 - calibrated).ln()
            });
        }
    }

    pub fn record_workflow_output_prediction(
        target: &str,
        source: &'static str,
        tokens: usize,
        support: usize,
        mature: bool,
        backoff_level: usize,
    ) {
        histogram!(
            "vllm_router_workflow_predicted_output_tokens",
            "target" => target.to_string(),
            "source" => source,
            "mature" => if mature { "true" } else { "false" }
        )
        .record(tokens as f64);
        histogram!(
            "vllm_router_workflow_output_prediction_support",
            "target" => target.to_string(),
            "source" => source
        )
        .record(support as f64);
        counter!(
            "vllm_router_workflow_output_prediction_total",
            "target" => target.to_string(),
            "source" => source,
            "backoff_level" => backoff_level.to_string()
        )
        .increment(1);
    }

    pub fn record_workflow_output_observation(
        target: &str,
        actual_tokens: usize,
        p95: Option<usize>,
        p99: Option<usize>,
    ) {
        histogram!(
            "vllm_router_workflow_output_actual_tokens",
            "target" => target.to_string()
        )
        .record(actual_tokens as f64);
        for (quantile, predicted) in [("p95", p95), ("p99", p99)] {
            if let Some(predicted) = predicted {
                let covered = actual_tokens <= predicted;
                counter!(
                    "vllm_router_workflow_output_coverage_total",
                    "target" => target.to_string(),
                    "quantile" => quantile,
                    "covered" => if covered { "true" } else { "false" }
                )
                .increment(1);
                histogram!(
                    "vllm_router_workflow_output_reservation_error_tokens",
                    "target" => target.to_string(),
                    "quantile" => quantile,
                    "direction" => if covered { "over_reserved" } else { "under_reserved" }
                )
                .record(predicted.abs_diff(actual_tokens) as f64);
            }
        }
    }

    pub fn record_agent_aware_shared_prefix_observation(target: &str, tokens: usize) {
        histogram!(
            "vllm_router_agent_aware_shared_prefix_observed_tokens",
            "target" => target.to_string()
        )
        .record(tokens as f64);
    }

    pub fn record_agent_aware_context_token_fact(
        model_pool: &str,
        endpoint: &str,
        source: &'static str,
        tokens: usize,
        fallback: Option<&'static str>,
    ) {
        counter!(
            "vllm_router_agent_aware_context_token_facts_total",
            "model_pool" => model_pool.to_string(),
            "endpoint" => endpoint.to_string(),
            "source" => source
        )
        .increment(1);
        histogram!(
            "vllm_router_agent_aware_context_token_fact_tokens",
            "model_pool" => model_pool.to_string(),
            "source" => source
        )
        .record(tokens as f64);
        if let Some(fallback) = fallback {
            counter!(
                "vllm_router_agent_aware_context_token_fallbacks_total",
                "model_pool" => model_pool.to_string(),
                "reason" => fallback
            )
            .increment(1);
        }
    }

    pub fn record_agent_aware_context_token_shadow(target: &str, outcome: &'static str) {
        counter!(
            "vllm_router_agent_aware_context_token_shadow_total",
            "target" => target.to_string(),
            "outcome" => outcome
        )
        .increment(1);
    }

    pub fn record_agent_aware_context_token_observation_delta(
        model_pool: &str,
        source: &'static str,
        delta_tokens: i64,
        observed_tokens: usize,
    ) {
        let ratio = if observed_tokens == 0 {
            0.0
        } else {
            delta_tokens as f64 / observed_tokens as f64
        };
        histogram!(
            "vllm_router_agent_aware_context_token_observation_delta_ratio",
            "model_pool" => model_pool.to_string(),
            "source" => source
        )
        .record(ratio);
    }

    pub fn record_agent_aware_context_token_revocation(reason: &'static str) {
        counter!(
            "vllm_router_agent_aware_context_token_revocations_total",
            "reason" => reason
        )
        .increment(1);
    }

    pub fn set_agent_aware_context_token_revoked(count: usize) {
        gauge!("vllm_router_agent_aware_context_token_revoked_fingerprints").set(count as f64);
    }

    pub fn set_agent_aware_context_token_pool_saturated(model_pool: &str, saturated: bool) {
        gauge!(
            "vllm_router_agent_aware_context_token_pool_saturated",
            "model_pool" => model_pool.to_string()
        )
        .set(if saturated { 1.0 } else { 0.0 });
    }

    pub fn record_agent_aware_context_token_unobserved(model_pool: &str, source: &'static str) {
        counter!(
            "vllm_router_agent_aware_context_token_unobserved_total",
            "model_pool" => model_pool.to_string(),
            "source" => source
        )
        .increment(1);
    }

    pub fn set_agent_aware_context_token_usage_unavailable(model_pool: &str) {
        gauge!("vllm_router_agent_aware_context_token_usage_unavailable", "model_pool" => model_pool.to_string()).set(1.0);
    }

    pub fn record_agent_aware_usage_filter_error() {
        counter!("vllm_router_agent_aware_usage_filter_errors_total").increment(1);
    }

    pub fn record_agent_aware_token_prepare(seconds: f64) {
        histogram!("vllm_router_agent_aware_token_prepare_seconds").record(seconds);
    }

    pub fn record_agent_aware_token_prepare_failure() {
        counter!("vllm_router_agent_aware_token_prepare_failures_total").increment(1);
    }

    pub fn set_agent_aware_context_token_mode(mode: &'static str) {
        for candidate in ["estimated", "shadow", "exact"] {
            gauge!(
                "vllm_router_agent_aware_context_token_mode",
                "mode" => candidate
            )
            .set(if candidate == mode { 1.0 } else { 0.0 });
        }
    }

    pub(crate) fn record_agent_aware_admission(target: &str, sample: AgentAwareAdmissionMetrics) {
        let record = |name, value| {
            histogram!(name, "target" => target.to_string()).record(value);
        };
        record(
            "vllm_router_agent_aware_admission_used_tokens",
            sample.used_tokens,
        );
        record(
            "vllm_router_agent_aware_admission_required_tokens",
            sample.required_tokens,
        );
        record(
            "vllm_router_agent_aware_admission_growth_reserve_tokens",
            sample.reserve_tokens,
        );
        if let Some(capacity) = sample.capacity_tokens.filter(|capacity| *capacity > 0) {
            let capacity = capacity as f64;
            record(
                "vllm_router_agent_aware_admission_capacity_pressure_ratio",
                sample.used_tokens / capacity,
            );
            record(
                "vllm_router_agent_aware_admission_projected_pressure_ratio",
                (sample.used_tokens + sample.required_tokens + sample.reserve_tokens) / capacity,
            );
        }
        if let Some(value) = sample.backend_kv_usage_ratio {
            record(
                "vllm_router_agent_aware_admission_backend_kv_usage_ratio",
                value,
            );
        }
        if let Some(value) = sample.backend_running_requests {
            record(
                "vllm_router_agent_aware_admission_backend_running_requests",
                value as f64,
            );
        }
        if let Some(value) = sample.backend_waiting_requests {
            record(
                "vllm_router_agent_aware_admission_backend_waiting_requests",
                value as f64,
            );
        }
        if let Some(value) = sample.observation_age {
            record(
                "vllm_router_agent_aware_admission_capacity_observation_age_seconds",
                value.as_secs_f64(),
            );
        }
    }

    // PD-specific metrics
    pub fn record_pd_request(route: &str) {
        counter!("vllm_router_pd_requests_total",
            "route" => route.to_string()
        )
        .increment(1);
    }

    pub fn record_pd_request_duration(route: &str, duration: Duration) {
        histogram!("vllm_router_pd_request_duration_seconds",
            "route" => route.to_string()
        )
        .record(duration.as_secs_f64());
    }

    pub fn record_pd_prefill_request(worker: &str) {
        counter!("vllm_router_pd_prefill_requests_total",
            "worker" => worker.to_string()
        )
        .increment(1);
    }

    pub fn record_pd_decode_request(worker: &str) {
        counter!("vllm_router_pd_decode_requests_total",
            "worker" => worker.to_string()
        )
        .increment(1);
    }

    pub fn record_pd_error(error_type: &str) {
        counter!("vllm_router_pd_errors_total",
            "error_type" => error_type.to_string()
        )
        .increment(1);
    }

    pub fn record_pd_prefill_error(worker: &str) {
        counter!("vllm_router_pd_prefill_errors_total",
            "worker" => worker.to_string()
        )
        .increment(1);
    }

    pub fn record_pd_decode_error(worker: &str) {
        counter!("vllm_router_pd_decode_errors_total",
            "worker" => worker.to_string()
        )
        .increment(1);
    }

    /// A router-buffered stage body exceeded the fixed buffer cap and the
    /// request failed instead of buffering unbounded.
    pub fn record_pd_body_over_limit(stage: &str) {
        counter!("vllm_router_pd_body_over_limit_total",
            "stage" => stage.to_string()
        )
        .increment(1);
    }

    // Generate request metrics
    pub fn record_generate_duration(duration: Duration) {
        histogram!("vllm_router_generate_duration_seconds").record(duration.as_secs_f64());
    }

    // Embeddings metrics
    pub fn record_embeddings_request() {
        counter!("vllm_router_embeddings_total").increment(1);
    }

    pub fn record_embeddings_duration(duration: Duration) {
        histogram!("vllm_router_embeddings_duration_seconds").record(duration.as_secs_f64());
    }

    pub fn record_embeddings_error(error_type: &str) {
        counter!(
            "vllm_router_embeddings_errors_total",
            "error_type" => error_type.to_string()
        )
        .increment(1);
    }

    pub fn set_embeddings_queue_size(size: usize) {
        gauge!("vllm_router_embeddings_queue_size").set(size as f64);
    }

    // Running requests for cache-aware policy
    pub fn set_running_requests(worker: &str, count: usize) {
        gauge!("vllm_router_running_requests",
            "worker" => worker.to_string()
        )
        .set(count as f64);
    }

    // Circuit breaker metrics
    pub fn set_cb_state(worker: &str, state_code: u8) {
        gauge!("vllm_router_cb_state",
            "worker" => worker.to_string()
        )
        .set(state_code as f64);
    }

    pub fn record_cb_state_transition(worker: &str, from: &str, to: &str) {
        counter!("vllm_router_cb_state_transitions_total",
            "worker" => worker.to_string(),
            "from" => from.to_string(),
            "to" => to.to_string()
        )
        .increment(1);
    }

    pub fn record_cb_outcome(worker: &str, outcome: &str) {
        counter!("vllm_router_cb_outcomes_total",
            "worker" => worker.to_string(),
            "outcome" => outcome.to_string()
        )
        .increment(1);
    }
}

impl TokenizerMetrics {
    // Encode cache metrics
    pub fn record_cache_hit() {
        counter!("vllm_tokenizer_cache_hits_total").increment(1);
    }

    pub fn record_cache_miss() {
        counter!("vllm_tokenizer_cache_misses_total").increment(1);
    }

    pub fn record_cache_evictions(count: u64) {
        counter!("vllm_tokenizer_cache_evictions_total").increment(count);
    }

    pub fn record_cache_oversized() {
        counter!("vllm_tokenizer_cache_oversized_total").increment(1);
    }

    pub fn set_cache_entries(entries: usize) {
        gauge!("vllm_tokenizer_cache_entries").set(entries as f64);
    }

    pub fn set_cache_bytes(bytes: usize) {
        gauge!("vllm_tokenizer_cache_bytes").set(bytes as f64);
    }

    // L1 prefix encoding cache metrics
    pub fn record_l1_lookup(cached_tokens: usize, encoded_tokens: usize) {
        counter!("vllm_tokenizer_l1_lookups_total").increment(1);
        if cached_tokens > 0 {
            counter!("vllm_tokenizer_l1_hits_total").increment(1);
        }
        counter!("vllm_tokenizer_l1_cached_tokens_total").increment(cached_tokens as u64);
        counter!("vllm_tokenizer_l1_encoded_tokens_total").increment(encoded_tokens as u64);
    }

    pub fn record_l1_evictions(count: u64) {
        counter!("vllm_tokenizer_l1_evictions_total").increment(count);
    }

    pub fn record_l1_verify_mismatch() {
        counter!("vllm_tokenizer_l1_verify_mismatches_total").increment(1);
    }

    pub fn record_l1_bypass(reason: &'static str) {
        counter!("vllm_tokenizer_l1_bypasses_total", "reason" => reason).increment(1);
    }

    pub fn set_l1_nodes(nodes: usize) {
        gauge!("vllm_tokenizer_l1_nodes").set(nodes as f64);
    }

    pub fn set_l1_bytes(bytes: usize) {
        gauge!("vllm_tokenizer_l1_bytes").set(bytes as f64);
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn publishes_routing_metric_descriptions_and_buckets() {
        let recorder = set_program_scheduling_buckets(PrometheusBuilder::new()).build_recorder();
        metrics::with_local_recorder(&recorder, || {
            init_metrics();
            RouterMetrics::record_prefix_routing_decision("kv_aware", "regular", "hit");
            RouterMetrics::record_pd_kv_decision(Some(256), Some(16), None);
        });
        let rendered = recorder.handle().render();
        for name in [
            "vllm_router_prefix_routing_decisions_total",
            "vllm_router_pd_credited_prefix_tokens_total",
        ] {
            assert!(rendered.contains(&format!("# HELP {name} ")), "{rendered}");
        }
        assert!(
            rendered.contains("vllm_router_pd_new_prefill_tokens_bucket"),
            "{rendered}"
        );
    }

    #[test]
    fn rejects_invalid_prometheus_bind_address() {
        let error = start_prometheus(PrometheusConfig {
            host: "invalid-ip".into(),
            ..PrometheusConfig::default()
        })
        .unwrap_err();
        assert!(error.to_string().contains("invalid Prometheus host"));
    }
}
