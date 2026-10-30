use super::*;
use crate::program_scheduling::ProgramBindingStrategy;

/// Configuration validator
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate a complete router configuration
    pub fn validate(config: &RouterConfig) -> ConfigResult<()> {
        if let Some(ttl_secs) = config.kv_aware_session_affinity_ttl_secs {
            if !(1..=31_536_000).contains(&ttl_secs) {
                return Err(ConfigError::InvalidValue {
                    field: "kv_aware_session_affinity_ttl_secs".to_string(),
                    value: ttl_secs.to_string(),
                    reason: "Must be between 1 and 31536000 seconds".to_string(),
                });
            }
        }
        // Check if service discovery is enabled (either via discovery config or vLLM mode)
        let has_service_discovery = config.discovery.as_ref().is_some_and(|d| d.enabled)
            || matches!(
                &config.mode,
                RoutingMode::VllmPrefillDecode {
                    discovery_address: Some(_),
                    ..
                }
            );

        Self::validate_mode(&config.mode, has_service_discovery)?;
        Self::validate_policy(&config.policy)?;
        Self::validate_server_settings(config)?;
        if let Some(program_scheduling) = &config.program_scheduling {
            Self::validate_program_scheduling(program_scheduling)?;
        }
        Self::validate_epd(config)?;

        if let Some(discovery) = &config.discovery {
            Self::validate_discovery(discovery, &config.mode)?;
        }

        if let Some(metrics) = &config.metrics {
            Self::validate_metrics(metrics)?;
        }

        Self::validate_compatibility(config)?;

        // Validate effective retry/CB configs (respect disable flags)
        let retry_cfg = config.effective_retry_config();
        let cb_cfg = config.effective_circuit_breaker_config();
        Self::validate_retry(&retry_cfg)?;
        Self::validate_circuit_breaker(&cb_cfg)?;

        Ok(())
    }

    fn validate_program_scheduling(config: &ProgramSchedulingConfig) -> ConfigResult<()> {
        if config.binding_strategy == ProgramBindingStrategy::ReasoningTokenBalance
            && config.token_capacity_per_dp_rank.is_none()
        {
            return Err(ConfigError::ValidationFailed {
                reason: "reasoning_token_balance requires token_capacity_per_dp_rank".to_string(),
            });
        }
        let positive_finite = [
            ("metrics_interval_seconds", config.metrics_interval_seconds),
            ("queue_timeout_seconds", config.queue_timeout_seconds),
            (
                "force_resume_timeout_seconds",
                config.force_resume_timeout_seconds,
            ),
            (
                "paused_retention_ttl_seconds",
                config.paused_retention_ttl_seconds,
            ),
            (
                "shared_prefix_freshness_kv_turnovers",
                config.shared_prefix_freshness_kv_turnovers,
            ),
        ];
        for (field, value) in positive_finite {
            if !value.is_finite() || value <= 0.0 {
                return Err(ConfigError::InvalidValue {
                    field: field.to_string(),
                    value: value.to_string(),
                    reason: "Must be finite and > 0".to_string(),
                });
            }
        }
        let nonnegative_finite = [
            ("max_acting_ttl_seconds", config.max_acting_ttl_seconds),
            (
                "shared_prefix_freshness_warmup_seconds",
                config.shared_prefix_freshness_warmup_seconds,
            ),
            (
                "prefill_cost_model.intercept_seconds",
                config.prefill_cost_model.intercept_seconds,
            ),
            (
                "prefill_cost_model.linear_seconds_per_1k_tokens",
                config.prefill_cost_model.linear_seconds_per_1k_tokens,
            ),
            (
                "prefill_cost_model.quadratic_seconds_per_1k_tokens_squared",
                config
                    .prefill_cost_model
                    .quadratic_seconds_per_1k_tokens_squared,
            ),
            (
                "decode_throughput_model.batch_step_seconds_per_request",
                config
                    .decode_throughput_model
                    .batch_step_seconds_per_request,
            ),
            (
                "decode_throughput_model.context_step_seconds_per_token",
                config
                    .decode_throughput_model
                    .context_step_seconds_per_token,
            ),
        ];
        for (field, value) in nonnegative_finite {
            if !value.is_finite() || value < 0.0 {
                return Err(ConfigError::InvalidValue {
                    field: field.to_string(),
                    value: value.to_string(),
                    reason: "Must be finite and >= 0".to_string(),
                });
            }
        }
        if !config
            .prefill_cost_model
            .decode_throughput_alpha
            .is_finite()
            || !(0.0..=1.0).contains(&config.prefill_cost_model.decode_throughput_alpha)
        {
            return Err(ConfigError::InvalidValue {
                field: "prefill_cost_model.decode_throughput_alpha".to_string(),
                value: config
                    .prefill_cost_model
                    .decode_throughput_alpha
                    .to_string(),
                reason: "Must be finite and between 0 and 1 inclusive".to_string(),
            });
        }
        if !config
            .decode_throughput_model
            .fixed_step_seconds
            .is_finite()
            || config.decode_throughput_model.fixed_step_seconds <= 0.0
        {
            return Err(ConfigError::InvalidValue {
                field: "decode_throughput_model.fixed_step_seconds".to_string(),
                value: config
                    .decode_throughput_model
                    .fixed_step_seconds
                    .to_string(),
                reason: "Must be finite and > 0".to_string(),
            });
        }
        // Zero admission bounds mean "auto" and are resolved from the P permit
        // count when the scheduler is built, so they are valid here.
        if config.hash_virtual_nodes == 0
            || config.stats_window_size == 0
            || config.max_segment_rounds == 0
            || config.token_capacity_per_dp_rank == Some(0)
        {
            return Err(ConfigError::ValidationFailed {
                reason: "Program scheduling counts and configured token capacity must be > 0; zero admission bounds mean auto sizing"
                    .to_string(),
            });
        }
        if !config.cross_rank_headroom_ratio.is_finite() || config.cross_rank_headroom_ratio < 1.0 {
            return Err(ConfigError::ValidationFailed {
                reason: "cross_rank_headroom_ratio must be finite and >= 1".to_string(),
            });
        }
        if !(0.0 < config.low_watermark_ratio
            && config.low_watermark_ratio <= config.high_watermark_ratio
            && config.high_watermark_ratio <= 1.0)
        {
            return Err(ConfigError::ValidationFailed {
                reason: "Program scheduling watermarks must satisfy 0 < low <= high <= 1"
                    .to_string(),
            });
        }
        if config.force_resume_timeout_seconds > config.queue_timeout_seconds {
            return Err(ConfigError::ValidationFailed {
                reason: "force_resume_timeout_seconds must not exceed queue_timeout_seconds"
                    .to_string(),
            });
        }
        let prediction = &config.prediction_profile;
        if !prediction.snapshot_interval_seconds.is_finite()
            || prediction.snapshot_interval_seconds <= 0.0
            || !prediction.prior_weight.is_finite()
            || prediction.prior_weight < 0.0
            || prediction.max_keys == 0
            || prediction.return_time_min_support == 0
            || prediction.output_length_min_support == 0
            || prediction.output_length_tail_samples == 0
            || prediction.stability_window_size == 0
            || prediction.step_bucket_size == 0
        {
            return Err(ConfigError::ValidationFailed {
                reason: "Prediction profile intervals, weights, supports, windows, and limits must be valid and non-zero".to_string(),
            });
        }
        if !prediction.return_time_horizon_seconds.is_finite()
            || prediction.return_time_horizon_seconds <= 0.0
            || !prediction.output_length_quantile.is_finite()
            || !(0.0..1.0).contains(&prediction.output_length_quantile)
            || !prediction.stability_tolerance.is_finite()
            || prediction.stability_tolerance < 0.0
        {
            return Err(ConfigError::ValidationFailed {
                reason: "Prediction horizons, quantiles, and stability tolerance are invalid"
                    .to_string(),
            });
        }
        if !strictly_increasing_f64(&prediction.return_time_bucket_bounds_seconds)
            || !strictly_increasing_usize(&prediction.output_length_bucket_bounds_tokens)
        {
            return Err(ConfigError::ValidationFailed {
                reason:
                    "Prediction bucket bounds must be non-empty, positive, and strictly increasing"
                        .to_string(),
            });
        }
        if let Some(path) = prediction.profile_path.as_deref() {
            let raw =
                std::fs::read_to_string(path).map_err(|error| ConfigError::ValidationFailed {
                    reason: format!("Unable to read prediction profile '{path}': {error}"),
                })?;
            crate::program_scheduling::prediction_profile::validate_profile_json(
                &raw,
                &prediction.return_time_bucket_bounds_seconds,
                &prediction.output_length_bucket_bounds_tokens,
                prediction.max_keys,
            )
            .map_err(|error| ConfigError::ValidationFailed {
                reason: format!("Invalid prediction profile '{path}': {error}"),
            })?;
        }
        Ok(())
    }

    fn validate_epd(config: &RouterConfig) -> ConfigResult<()> {
        let Some(epd) = &config.epd else {
            return Ok(());
        };
        let worker_urls: Vec<&String> = match &config.mode {
            RoutingMode::Regular { worker_urls } => worker_urls.iter().collect(),
            RoutingMode::VllmPrefillDecode {
                prefill_urls,
                discovery_address: None,
                ..
            } => prefill_urls.iter().map(|(url, _)| url).collect(),
            _ => {
                return Err(ConfigError::ValidationFailed {
                    reason: "EPD requires regular E+PD or static vLLM E/P/D routing".into(),
                })
            }
        };
        if config.intra_node_data_parallel_size != 1 || config.enable_igw {
            return Err(ConfigError::ValidationFailed {
                reason: "EPD currently requires DP=1 and proxy mode".into(),
            });
        }
        if epd.encoder_urls.is_empty() {
            return Err(ConfigError::MissingRequired {
                field: "epd.encoder_urls".into(),
            });
        }
        Self::validate_urls(&epd.encoder_urls)?;
        for url in epd.encoder_urls.iter().chain(worker_urls.iter().copied()) {
            if !url.starts_with("http://") && !url.starts_with("https://") {
                return Err(ConfigError::InvalidValue {
                    field: "epd.worker_url".into(),
                    value: url.clone(),
                    reason: "EPD requires HTTP(S) encoder and consumer URLs".into(),
                });
            }
        }
        if !epd.consumer_zmq_addrs.is_empty() {
            if worker_urls.is_empty()
                || epd.consumer_zmq_addrs.len() != worker_urls.len()
                || worker_urls
                    .iter()
                    .any(|url| !epd.consumer_zmq_addrs.contains_key(*url))
            {
                return Err(ConfigError::ValidationFailed {
                    reason:
                        "EPD consumer_zmq_addrs must map every embedding consumer URL exactly once"
                            .into(),
                });
            }
            for addr in epd.consumer_zmq_addrs.values() {
                let parsed = url::Url::parse(addr).map_err(|e| ConfigError::ValidationFailed {
                    reason: format!("Invalid EC address: {e}"),
                })?;
                if parsed.scheme() != "tcp"
                    || parsed.host_str().is_none()
                    || parsed.port().is_none()
                {
                    return Err(ConfigError::ValidationFailed {
                        reason: "EC addresses must be tcp://host:port".into(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Validate routing mode configuration
    fn validate_mode(mode: &RoutingMode, has_service_discovery: bool) -> ConfigResult<()> {
        match mode {
            RoutingMode::Regular { worker_urls } => {
                // Validate URLs if any are provided
                if !worker_urls.is_empty() {
                    Self::validate_urls(worker_urls)?;
                    Self::reject_mixed_worker_urls(worker_urls)?;
                }
                // Note: We allow empty worker URLs even without service discovery
                // to let the router start and fail at runtime when routing requests.
                // This matches legacy behavior and test expectations.
            }
            RoutingMode::VllmPrefillDecode {
                prefill_urls,
                decode_urls,
                prefill_policy,
                decode_policy,
                discovery_address: _,
            } => {
                // Only require URLs if service discovery is disabled
                if !has_service_discovery {
                    if prefill_urls.is_empty() {
                        return Err(ConfigError::ValidationFailed {
                            reason: "vLLM PD mode requires at least one prefill worker URL"
                                .to_string(),
                        });
                    }
                    if decode_urls.is_empty() {
                        return Err(ConfigError::ValidationFailed {
                            reason: "vLLM PD mode requires at least one decode worker URL"
                                .to_string(),
                        });
                    }
                }

                // Validate URLs if any are provided
                if !prefill_urls.is_empty() {
                    let prefill_url_strings: Vec<String> =
                        prefill_urls.iter().map(|(url, _)| url.clone()).collect();
                    Self::validate_urls(&prefill_url_strings)?;
                    Self::reject_mixed_worker_urls(&prefill_url_strings)?;
                }
                if !decode_urls.is_empty() {
                    Self::validate_urls(decode_urls)?;
                    Self::reject_mixed_worker_urls(decode_urls)?;
                }
                if !prefill_urls.is_empty() && !decode_urls.is_empty() {
                    let mut all: Vec<String> =
                        prefill_urls.iter().map(|(url, _)| url.clone()).collect();
                    all.extend(decode_urls.iter().cloned());
                    Self::reject_mixed_worker_urls(&all)?;
                }

                // Validate bootstrap ports
                for (_url, port) in prefill_urls {
                    if let Some(port) = port {
                        if *port == 0 {
                            return Err(ConfigError::InvalidValue {
                                field: "bootstrap_port".to_string(),
                                value: port.to_string(),
                                reason: "Port must be between 1 and 65535".to_string(),
                            });
                        }
                    }
                }

                // Validate optional prefill and decode policies
                if let Some(p_policy) = prefill_policy {
                    Self::validate_policy(p_policy)?;
                }
                if let Some(d_policy) = decode_policy {
                    Self::validate_policy(d_policy)?;
                }
            }
            RoutingMode::OpenAI { worker_urls } => {
                // Require exactly one worker URL for OpenAI router
                if worker_urls.len() != 1 {
                    return Err(ConfigError::ValidationFailed {
                        reason: "OpenAI mode requires exactly one --worker-urls entry".to_string(),
                    });
                }
                // Validate URL format
                if let Err(e) = url::Url::parse(&worker_urls[0]) {
                    return Err(ConfigError::ValidationFailed {
                        reason: format!("Invalid OpenAI worker URL '{}': {}", worker_urls[0], e),
                    });
                }
            }
        }
        Ok(())
    }

    /// Validate policy configuration
    fn validate_policy(policy: &PolicyConfig) -> ConfigResult<()> {
        match policy {
            PolicyConfig::Random | PolicyConfig::RoundRobin => {
                // No specific validation needed
            }
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold: _,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            } => {
                if !(0.0..=1.0).contains(cache_threshold) {
                    return Err(ConfigError::InvalidValue {
                        field: "cache_threshold".to_string(),
                        value: cache_threshold.to_string(),
                        reason: "Must be between 0.0 and 1.0".to_string(),
                    });
                }

                if *balance_rel_threshold < 1.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "balance_rel_threshold".to_string(),
                        value: balance_rel_threshold.to_string(),
                        reason: "Must be >= 1.0".to_string(),
                    });
                }

                if *eviction_interval_secs == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "eviction_interval_secs".to_string(),
                        value: eviction_interval_secs.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }

                if *max_tree_size == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "max_tree_size".to_string(),
                        value: max_tree_size.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }
            }
            PolicyConfig::KvAware {
                router_temperature,
                kv_penalty_host,
                kv_penalty_external,
                kv_penalty_disk,
                static_kv_event_sources,
            } => {
                if !router_temperature.is_finite() || *router_temperature < 0.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "router_temperature".to_string(),
                        value: router_temperature.to_string(),
                        reason: "Must be finite and >= 0.0".to_string(),
                    });
                }
                for (field, value) in [
                    ("kv_penalty_host", kv_penalty_host),
                    ("kv_penalty_external", kv_penalty_external),
                    ("kv_penalty_disk", kv_penalty_disk),
                ] {
                    if !value.is_finite() || !(0.0..=1.0).contains(value) {
                        return Err(ConfigError::InvalidValue {
                            field: field.to_string(),
                            value: value.to_string(),
                            reason: "Must be finite and between 0.0 and 1.0".to_string(),
                        });
                    }
                }
                for (worker, source) in static_kv_event_sources.iter() {
                    source.validate(worker)?;
                }
            }
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs,
            } => {
                if *load_check_interval_secs == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "load_check_interval_secs".to_string(),
                        value: load_check_interval_secs.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }
            }
            PolicyConfig::ConsistentHash { virtual_nodes } => {
                if *virtual_nodes == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "virtual_nodes".to_string(),
                        value: virtual_nodes.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }
            }
            PolicyConfig::RendezvousHash => {
                // No specific validation needed
            }
        }
        Ok(())
    }

    /// Validate server configuration
    fn validate_server_settings(config: &RouterConfig) -> ConfigResult<()> {
        if config.port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "port".to_string(),
                value: config.port.to_string(),
                reason: "Port must be > 0".to_string(),
            });
        }

        if config.max_payload_size == 0 {
            return Err(ConfigError::InvalidValue {
                field: "max_payload_size".to_string(),
                value: config.max_payload_size.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        if config.request_timeout_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "request_timeout_secs".to_string(),
                value: config.request_timeout_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        if config.worker_startup_timeout_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "worker_startup_timeout_secs".to_string(),
                value: config.worker_startup_timeout_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        if config.worker_startup_check_interval_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "worker_startup_check_interval_secs".to_string(),
                value: config.worker_startup_check_interval_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        Ok(())
    }

    /// Validate service discovery configuration
    fn validate_discovery(discovery: &DiscoveryConfig, mode: &RoutingMode) -> ConfigResult<()> {
        if !discovery.enabled {
            return Ok(()); // No validation needed if disabled
        }

        if discovery.port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "discovery.port".to_string(),
                value: discovery.port.to_string(),
                reason: "Port must be > 0".to_string(),
            });
        }

        if discovery.check_interval_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "discovery.check_interval_secs".to_string(),
                value: discovery.check_interval_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        // Validate selectors based on mode
        match mode {
            RoutingMode::Regular { .. } => {
                if discovery.selector.is_empty() {
                    return Err(ConfigError::ValidationFailed {
                        reason: "Regular mode with service discovery requires a non-empty selector"
                            .to_string(),
                    });
                }
            }
            RoutingMode::VllmPrefillDecode { .. } => {
                if discovery.prefill_selector.is_empty() && discovery.decode_selector.is_empty() {
                    return Err(ConfigError::ValidationFailed {
                        reason: "vLLM PD mode with service discovery requires at least one non-empty selector (prefill or decode)".to_string(),
                    });
                }
            }
            RoutingMode::OpenAI { .. } => {
                // OpenAI mode doesn't use service discovery
                return Err(ConfigError::ValidationFailed {
                    reason: "OpenAI mode does not support service discovery".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Validate metrics configuration
    fn validate_metrics(metrics: &MetricsConfig) -> ConfigResult<()> {
        if metrics.port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "metrics.port".to_string(),
                value: metrics.port.to_string(),
                reason: "Port must be > 0".to_string(),
            });
        }

        if metrics.host.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "metrics.host".to_string(),
                value: metrics.host.clone(),
                reason: "Host cannot be empty".to_string(),
            });
        }

        Ok(())
    }

    /// Validate retry configuration
    fn validate_retry(retry: &RetryConfig) -> ConfigResult<()> {
        if retry.max_retries < 1 {
            return Err(ConfigError::InvalidValue {
                field: "retry.max_retries".to_string(),
                value: retry.max_retries.to_string(),
                reason: "Must be >= 1 (set to 1 to effectively disable retries)".to_string(),
            });
        }
        if retry.initial_backoff_ms == 0 {
            return Err(ConfigError::InvalidValue {
                field: "retry.initial_backoff_ms".to_string(),
                value: retry.initial_backoff_ms.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        if retry.max_backoff_ms < retry.initial_backoff_ms {
            return Err(ConfigError::InvalidValue {
                field: "retry.max_backoff_ms".to_string(),
                value: retry.max_backoff_ms.to_string(),
                reason: "Must be >= initial_backoff_ms".to_string(),
            });
        }
        if retry.backoff_multiplier < 1.0 {
            return Err(ConfigError::InvalidValue {
                field: "retry.backoff_multiplier".to_string(),
                value: retry.backoff_multiplier.to_string(),
                reason: "Must be >= 1.0".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&retry.jitter_factor) {
            return Err(ConfigError::InvalidValue {
                field: "retry.jitter_factor".to_string(),
                value: retry.jitter_factor.to_string(),
                reason: "Must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(())
    }

    /// Validate circuit breaker configuration
    fn validate_circuit_breaker(cb: &CircuitBreakerConfig) -> ConfigResult<()> {
        if cb.failure_threshold < 1 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.failure_threshold".to_string(),
                value: cb.failure_threshold.to_string(),
                reason: "Must be >= 1 (set to u32::MAX to effectively disable CB)".to_string(),
            });
        }
        if cb.success_threshold < 1 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.success_threshold".to_string(),
                value: cb.success_threshold.to_string(),
                reason: "Must be >= 1".to_string(),
            });
        }
        if cb.timeout_duration_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.timeout_duration_secs".to_string(),
                value: cb.timeout_duration_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        if cb.window_duration_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.window_duration_secs".to_string(),
                value: cb.window_duration_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        Ok(())
    }

    /// Validate compatibility between different configuration sections
    fn validate_compatibility(config: &RouterConfig) -> ConfigResult<()> {
        if let Some(program) = &config.program_scheduling {
            let pd_mode = matches!(config.mode, RoutingMode::VllmPrefillDecode { .. });
            if (!matches!(config.mode, RoutingMode::Regular { .. }) && !pd_mode)
                || pd_mode != program.pd.enabled
                || (pd_mode && config.connection_mode != ConnectionMode::Http)
            {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "Program scheduling requires regular HTTP mode, or vLLM PD HTTP mode with pd.enabled=true".to_string(),
                });
            }
            if pd_mode
                && (!(0.0..=1.0).contains(&program.pd.prefill_kv_high_watermark)
                    || program.pd.prefill_kv_window_seconds <= 0.0
                    || !program.pd.prefill_kv_window_seconds.is_finite())
            {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "PD prefill_kv_high_watermark must be within [0, 1] and prefill_kv_window_seconds must be finite and > 0; a zero watermark disables the guard".to_string(),
                });
            }
            if pd_mode
                && (program.binding_only
                    || program.enable_batch_gain_admission
                    || program.token_capacity_per_dp_rank.is_none()
                    || program.pd.max_prefill_requests_per_target == 0
                    || program.pd.max_prefill_requests_per_target > u32::MAX as usize
                    || !matches!(
                        program.pd.kv_hash_mode.as_str(),
                        "sha256" | "python_hash_seed" | "xxhash_cbor"
                    ))
            {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "PD scheduling requires per-rank capacity, positive bounded P concurrency and a known KV hash mode; binding_only and uncalibrated batch gain are unsupported".to_string(),
                });
            }
        }
        // IGW mode is independent - skip other compatibility checks when enabled
        if config.enable_igw {
            return Ok(());
        }

        if let PolicyConfig::KvAware {
            static_kv_event_sources,
            ..
        } = &config.policy
        {
            if !static_kv_event_sources.is_empty() {
                let RoutingMode::Regular { worker_urls } = &config.mode else {
                    return Err(ConfigError::IncompatibleConfig {
                        reason: "static --kv-event-source is only supported in regular mode"
                            .to_string(),
                    });
                };
                if !config
                    .discovery
                    .as_ref()
                    .is_some_and(|discovery| discovery.enabled)
                {
                    for worker in static_kv_event_sources.keys() {
                        if !worker_urls.contains(worker) {
                            return Err(ConfigError::ValidationFailed {
                                reason: format!(
                                    "static KV event source worker '{worker}' is not in --worker-urls"
                                ),
                            });
                        }
                    }
                }
            }
        }

        // Check if service discovery is enabled for worker count validation.
        // This covers both K8s service discovery (config.discovery) and vLLM ZMQ
        // service discovery (VllmPrefillDecode { discovery_address: Some(_) }).
        let has_vllm_discovery = matches!(
            &config.mode,
            RoutingMode::VllmPrefillDecode {
                discovery_address: Some(_),
                ..
            }
        );
        let has_service_discovery =
            config.discovery.as_ref().is_some_and(|d| d.enabled) || has_vllm_discovery;

        // Only validate worker counts if service discovery is disabled
        if !has_service_discovery {
            // Check if power-of-two policy makes sense with insufficient workers
            if let PolicyConfig::PowerOfTwo { .. } = &config.policy {
                let worker_count = config.mode.worker_count();
                if worker_count < 2 {
                    return Err(ConfigError::IncompatibleConfig {
                        reason: "Power-of-two policy requires at least 2 workers".to_string(),
                    });
                }
            }

            // For vLLM PD mode, validate that policies have sufficient workers
            if let RoutingMode::VllmPrefillDecode {
                prefill_urls,
                decode_urls,
                prefill_policy,
                decode_policy,
                ..
            } = &config.mode
            {
                // Check power-of-two for prefill
                if let Some(PolicyConfig::PowerOfTwo { .. }) = prefill_policy {
                    if prefill_urls.len() < 2 {
                        return Err(ConfigError::IncompatibleConfig {
                            reason: "Power-of-two policy for prefill requires at least 2 prefill workers".to_string(),
                        });
                    }
                }

                // Check power-of-two for decode
                if let Some(PolicyConfig::PowerOfTwo { .. }) = decode_policy {
                    if decode_urls.len() < 2 {
                        return Err(ConfigError::IncompatibleConfig {
                            reason:
                                "Power-of-two policy for decode requires at least 2 decode workers"
                                    .to_string(),
                        });
                    }
                }
            }
        }

        // DP-aware routing is now automatically enabled when data_parallel_size > 1
        // and is compatible with service discovery

        // MoRI-IO requires service discovery: ZMQ addresses are obtained via instance
        // registration and are not available in direct URL mode.
        if config.kv_connector == KvConnector::MoriIO && !has_vllm_discovery {
            return Err(ConfigError::IncompatibleConfig {
                reason: "MoRI-IO KV connector requires service discovery to be enabled \
                         (ZMQ addresses are obtained via instance registration). Please \
                        run with `--vllm-discovery-address ${address}`"
                    .to_string(),
            });
        }

        // Static KV event sources are only meaningful with a KV-aware
        // prefill policy; anything else would silently ignore them.
        if !config.pd_kv_event_sources.is_empty() {
            if let RoutingMode::VllmPrefillDecode { prefill_urls, .. } = &config.mode {
                for (url, sources) in &config.pd_kv_event_sources {
                    // Static sources are looked up by `Worker::base_url()`, which
                    // drops the `@dp_rank` suffix, so a rank-pinned prefill URL
                    // must still be keyed by its base URL.
                    if !prefill_urls
                        .iter()
                        .any(|(prefill, _)| base_url_without_rank(prefill) == url)
                    {
                        return Err(ConfigError::IncompatibleConfig {
                            reason: format!(
                                "pd_kv_event_sources key '{url}' is not a configured prefill URL \
                                 (use the base URL without the '@rank' suffix)"
                            ),
                        });
                    }
                    if sources.endpoints.is_empty() {
                        return Err(ConfigError::IncompatibleConfig {
                            reason: format!("pd_kv_event_sources['{url}'] has no endpoints"),
                        });
                    }
                    if sources.block_size == 0 {
                        return Err(ConfigError::IncompatibleConfig {
                            reason: format!("pd_kv_event_sources['{url}'] block_size must be > 0"),
                        });
                    }
                    for endpoint in sources
                        .endpoints
                        .iter()
                        .chain(sources.replay_endpoints.iter().flatten())
                    {
                        if !is_reachable_tcp_endpoint(endpoint)
                            && endpoint
                                .strip_prefix("ipc://")
                                .is_none_or(|path| path.is_empty())
                        {
                            return Err(ConfigError::IncompatibleConfig {
                                reason: format!(
                                    "pd_kv_event_sources['{url}'] endpoint '{endpoint}' must be \
                                     a reachable ZMQ address (tcp://HOST:PORT or ipc://PATH)"
                                ),
                            });
                        }
                    }
                }
            } else {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "pd_kv_event_sources requires vllm_prefill_decode mode".to_string(),
                });
            }
            // Only kv_aware starts the KV event subscribers; accepting any
            // other policy here would silently drop the configured endpoints.
            if !matches!(
                config.mode.get_prefill_policy(&config.policy),
                PolicyConfig::KvAware { .. }
            ) {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "pd_kv_event_sources requires a kv_aware prefill policy (other \
                             policies never start KV event subscribers)"
                        .to_string(),
                });
            }
        }

        Ok(())
    }

    /// Mixed `http://` + `grpc://` is rejected, not a silent fallback.
    ///
    /// Today gRPC chat is `token_ids` only and HTTP chat is text-only reverse
    /// proxy. `EngineFrontend` is the gRPC path (`prepare` → ids → dispatch);
    /// HTTP never enters that frontend. Those two wires cannot share one
    /// request path, so handling a mixed pool is left as future work.
    fn reject_mixed_worker_urls(urls: &[String]) -> ConfigResult<()> {
        crate::backend::classify_worker_urls(urls)
            .map_err(|reason| ConfigError::ValidationFailed { reason })?;
        Ok(())
    }

    /// Validate URL format
    fn validate_urls(urls: &[String]) -> ConfigResult<()> {
        for url in urls {
            if url.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: "worker_url".to_string(),
                    value: url.clone(),
                    reason: "URL cannot be empty".to_string(),
                });
            }

            let scheme_ok = url.starts_with("http://")
                || url.starts_with("https://")
                || url.starts_with("grpc://");
            if !scheme_ok {
                return Err(ConfigError::InvalidValue {
                    field: "worker_url".to_string(),
                    value: url.clone(),
                    reason:
                        "URL must start with http://, https://, or grpc://; grpcs:// requires tonic TLS support and is not enabled yet"
                            .to_string(),
                });
            }

            // Strip optional `@dp_rank` before URL parse (`grpc://host:port@0`).
            let to_parse = url
                .rsplit_once('@')
                .and_then(|(prefix, rank)| rank.parse::<u32>().ok().map(|_| prefix))
                .unwrap_or(url.as_str());

            // Basic URL validation
            match ::url::Url::parse(to_parse) {
                Ok(parsed) => {
                    // Additional validation
                    if parsed.host_str().is_none() {
                        return Err(ConfigError::InvalidValue {
                            field: "worker_url".to_string(),
                            value: url.clone(),
                            reason: "URL must have a valid host".to_string(),
                        });
                    }
                }
                Err(e) => {
                    return Err(ConfigError::InvalidValue {
                        field: "worker_url".to_string(),
                        value: url.clone(),
                        reason: format!("Invalid URL format: {}", e),
                    });
                }
            }
        }
        Ok(())
    }
}

fn strictly_increasing_f64(values: &[f64]) -> bool {
    !values.is_empty()
        && values.iter().all(|value| value.is_finite() && *value > 0.0)
        && values.windows(2).all(|pair| pair[0] < pair[1])
}

fn strictly_increasing_usize(values: &[usize]) -> bool {
    !values.is_empty()
        && values.iter().all(|value| *value > 0)
        && values.windows(2).all(|pair| pair[0] < pair[1])
}

/// Base URL without the optional `@dp_rank` suffix.
///
/// Mirrors `Worker::normalised_url()`: static KV event sources are looked up by
/// `Worker::base_url()`, so a `http://host:port@1` prefill URL is keyed as
/// `http://host:port`.
fn base_url_without_rank(url: &str) -> &str {
    match url.split_once('@') {
        Some((base, rank)) if rank.parse::<usize>().is_ok() => base,
        _ => url,
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    #[test]
    fn session_affinity_ttl_is_optional_and_bounded() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker:8000".to_string()],
            },
            PolicyConfig::Random,
        );
        assert!(ConfigValidator::validate(&config).is_ok());

        config.kv_aware_session_affinity_ttl_secs = Some(300);
        assert!(ConfigValidator::validate(&config).is_ok());

        for ttl in [0, 31_536_001] {
            config.kv_aware_session_affinity_ttl_secs = Some(ttl);
            assert!(ConfigValidator::validate(&config).is_err());
        }
    }

    #[test]
    fn test_validate_program_scheduling_boundaries() {
        let valid = ProgramSchedulingConfig::default();
        assert!(ConfigValidator::validate_program_scheduling(&valid).is_ok());

        let invalid = [
            ProgramSchedulingConfig {
                cross_rank_headroom_ratio: f64::NAN,
                ..valid.clone()
            },
            ProgramSchedulingConfig {
                low_watermark_ratio: 0.0,
                ..valid.clone()
            },
            ProgramSchedulingConfig {
                high_watermark_ratio: 1.1,
                ..valid.clone()
            },
            ProgramSchedulingConfig {
                low_watermark_ratio: 0.9,
                high_watermark_ratio: 0.8,
                ..valid.clone()
            },
            ProgramSchedulingConfig {
                force_resume_timeout_seconds: valid.queue_timeout_seconds + 1.0,
                ..valid.clone()
            },
            ProgramSchedulingConfig {
                shared_prefix_freshness_warmup_seconds: -1.0,
                ..valid.clone()
            },
            ProgramSchedulingConfig {
                shared_prefix_freshness_kv_turnovers: 0.0,
                ..valid.clone()
            },
            ProgramSchedulingConfig {
                prefill_cost_model: crate::program_scheduling::PrefillCostModel {
                    linear_seconds_per_1k_tokens: f64::NAN,
                    ..valid.prefill_cost_model
                },
                ..valid.clone()
            },
            ProgramSchedulingConfig {
                prefill_cost_model: crate::program_scheduling::PrefillCostModel {
                    decode_throughput_alpha: 1.1,
                    ..valid.prefill_cost_model
                },
                ..valid.clone()
            },
            ProgramSchedulingConfig {
                decode_throughput_model: crate::program_scheduling::DecodeThroughputModel {
                    fixed_step_seconds: 0.0,
                    ..valid.decode_throughput_model
                },
                ..valid.clone()
            },
            ProgramSchedulingConfig {
                decode_throughput_model: crate::program_scheduling::DecodeThroughputModel {
                    context_step_seconds_per_token: -1.0,
                    ..valid.decode_throughput_model
                },
                ..valid.clone()
            },
            ProgramSchedulingConfig {
                max_segment_rounds: 0,
                ..valid
            },
        ];
        for config in invalid {
            assert!(ConfigValidator::validate_program_scheduling(&config).is_err());
        }
    }

    #[test]
    fn test_validate_regular_mode() {
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker:8000".to_string()],
            },
            PolicyConfig::Random,
        );

        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_empty_worker_urls() {
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![],
            },
            PolicyConfig::Random,
        );

        // Empty worker URLs are now allowed to match legacy behavior
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_empty_worker_urls_with_service_discovery() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![],
            },
            PolicyConfig::Random,
        );

        // Enable service discovery
        config.discovery = Some(DiscoveryConfig {
            enabled: true,
            selector: vec![("app".to_string(), "test".to_string())]
                .into_iter()
                .collect(),
            ..Default::default()
        });

        // Should pass validation since service discovery is enabled
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_invalid_urls() {
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["invalid-url".to_string()],
            },
            PolicyConfig::Random,
        );

        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_validate_rejects_grpcs_until_tls_is_enabled() {
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["grpcs://worker:50051".to_string()],
            },
            PolicyConfig::Random,
        );

        let err = ConfigValidator::validate(&config).unwrap_err();
        assert!(err.to_string().contains("grpcs:// requires tonic TLS"));
    }

    #[test]
    fn test_validate_rejects_mixed_http_grpc_workers() {
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![
                    "http://worker:8000".to_string(),
                    "grpc://worker:50051".to_string(),
                ],
            },
            PolicyConfig::Random,
        );

        let err = ConfigValidator::validate(&config).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("mixed"), "{msg}");
    }

    #[test]
    fn test_validate_all_grpc_workers() {
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![
                    "grpc://worker:50051".to_string(),
                    "grpc://worker:50052@0".to_string(),
                ],
            },
            PolicyConfig::Random,
        );

        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_epd_requires_http_workers() {
        for (encoder_scheme, consumer_scheme, supported) in [
            ("http", "http", true),
            ("https", "https", true),
            ("grpc", "http", false),
            ("http", "grpc", false),
        ] {
            let consumer = format!("{consumer_scheme}://consumer:8000");
            for mode in [
                RoutingMode::Regular {
                    worker_urls: vec![consumer.clone()],
                },
                RoutingMode::VllmPrefillDecode {
                    prefill_urls: vec![(consumer.clone(), None)],
                    decode_urls: vec![format!("{consumer_scheme}://decode:8000")],
                    prefill_policy: None,
                    decode_policy: None,
                    discovery_address: None,
                },
            ] {
                let mut config = RouterConfig::new(mode, PolicyConfig::Random);
                config.epd = Some(EpdConfig {
                    encoder_urls: vec![format!("{encoder_scheme}://encoder:8000")],
                    consumer_zmq_addrs: Default::default(),
                });
                let result = ConfigValidator::validate(&config);
                if supported {
                    result.unwrap();
                } else {
                    assert!(
                        result
                            .unwrap_err()
                            .to_string()
                            .contains("EPD requires HTTP(S)"),
                        "encoder={encoder_scheme}, consumer={consumer_scheme}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_validate_cache_aware_thresholds() {
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![
                    "http://worker1:8000".to_string(),
                    "http://worker2:8000".to_string(),
                ],
            },
            PolicyConfig::CacheAware {
                cache_threshold: 1.5, // Invalid: > 1.0
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
            },
        );

        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_validate_cache_aware_single_worker() {
        // Cache-aware with single worker should be allowed (even if not optimal)
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker1:8000".to_string()],
            },
            PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
            },
        );

        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_pd_mode() {
        let config = RouterConfig::new(
            RoutingMode::VllmPrefillDecode {
                prefill_urls: vec![("http://prefill:8000".to_string(), Some(8081))],
                decode_urls: vec!["http://decode:8000".to_string()],
                prefill_policy: None,
                decode_policy: None,
                discovery_address: None,
            },
            PolicyConfig::Random,
        );

        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_roundrobin_with_pd_mode() {
        // RoundRobin with PD mode is now supported
        let config = RouterConfig::new(
            RoutingMode::VllmPrefillDecode {
                prefill_urls: vec![("http://prefill:8000".to_string(), None)],
                decode_urls: vec!["http://decode:8000".to_string()],
                prefill_policy: None,
                decode_policy: None,
                discovery_address: None,
            },
            PolicyConfig::RoundRobin,
        );

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_cache_aware_with_pd_mode() {
        // CacheAware with PD mode is now supported
        let config = RouterConfig::new(
            RoutingMode::VllmPrefillDecode {
                prefill_urls: vec![("http://prefill:8000".to_string(), None)],
                decode_urls: vec!["http://decode:8000".to_string()],
                prefill_policy: None,
                decode_policy: None,
                discovery_address: None,
            },
            PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
            },
        );

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_power_of_two_with_regular_mode() {
        // PowerOfTwo with Regular mode is now supported
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![
                    "http://worker1:8000".to_string(),
                    "http://worker2:8000".to_string(),
                ],
            },
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 60,
            },
        );

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_pd_mode_with_separate_policies() {
        // Test PD mode with different policies for prefill and decode
        let config = RouterConfig::new(
            RoutingMode::VllmPrefillDecode {
                prefill_urls: vec![
                    ("http://prefill1:8000".to_string(), None),
                    ("http://prefill2:8000".to_string(), None),
                ],
                decode_urls: vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
                prefill_policy: Some(PolicyConfig::CacheAware {
                    cache_threshold: 0.5,
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
                    eviction_interval_secs: 60,
                    max_tree_size: 1000,
                }),
                decode_policy: Some(PolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 60,
                }),
                discovery_address: None,
            },
            PolicyConfig::Random, // Main policy as fallback
        );

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_accepts_kv_aware_for_disaggregated_pd_mode() {
        let config = RouterConfig::new(
            RoutingMode::VllmPrefillDecode {
                prefill_urls: vec![("http://prefill:8000".to_string(), None)],
                decode_urls: vec!["http://decode:8000".to_string()],
                prefill_policy: None,
                decode_policy: None,
                discovery_address: None,
            },
            PolicyConfig::KvAware {
                router_temperature: 0.0,
                kv_penalty_host: 0.8,
                kv_penalty_external: 0.6,
                kv_penalty_disk: 0.3,
                static_kv_event_sources: Default::default(),
            },
        );

        ConfigValidator::validate(&config).unwrap();
    }

    #[test]
    fn test_validate_pd_mode_power_of_two_insufficient_workers() {
        // Test that power-of-two policy requires at least 2 workers
        let config = RouterConfig::new(
            RoutingMode::VllmPrefillDecode {
                prefill_urls: vec![("http://prefill1:8000".to_string(), None)], // Only 1 prefill
                decode_urls: vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
                prefill_policy: Some(PolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 60,
                }), // Requires 2+ workers
                decode_policy: None,
                discovery_address: None,
            },
            PolicyConfig::Random,
        );

        let result = ConfigValidator::validate(&config);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("prefill requires at least 2"));
        }
    }

    #[test]
    fn test_moriio_requires_service_discovery() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker:8000".to_string()],
            },
            PolicyConfig::Random,
        );
        config.kv_connector = KvConnector::MoriIO;
        config.discovery = None;

        let result = ConfigValidator::validate(&config);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e
                .to_string()
                .contains("MoRI-IO KV connector requires service discovery"));
        }
    }

    #[test]
    fn test_moriio_with_service_discovery_is_valid() {
        let mut config = RouterConfig::new(
            RoutingMode::VllmPrefillDecode {
                prefill_urls: vec![],
                decode_urls: vec![],
                prefill_policy: None,
                decode_policy: None,
                discovery_address: Some("0.0.0.0:36367".to_string()),
            },
            PolicyConfig::Random,
        );
        config.kv_connector = KvConnector::MoriIO;

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    fn pd_config_with_static_kv_sources() -> RouterConfig {
        let mut config = RouterConfig::new(
            RoutingMode::VllmPrefillDecode {
                prefill_urls: vec![("http://prefill:8000".to_string(), None)],
                decode_urls: vec!["http://decode:8000".to_string()],
                prefill_policy: None,
                decode_policy: None,
                discovery_address: None,
            },
            PolicyConfig::KvAware {
                router_temperature: 0.0,
                kv_penalty_host: 0.8,
                kv_penalty_external: 0.6,
                kv_penalty_disk: 0.3,
                static_kv_event_sources: Default::default(),
            },
        );
        config.pd_kv_event_sources.insert(
            "http://prefill:8000".to_string(),
            PdKvEventSources {
                endpoints: vec!["tcp://prefill:5557".to_string()],
                replay_endpoints: None,
                block_size: 16,
                model: None,
            },
        );
        config
    }

    #[test]
    fn test_pd_kv_event_sources_valid() {
        assert!(ConfigValidator::validate(&pd_config_with_static_kv_sources()).is_ok());
    }

    /// A rank-pinned prefill URL is keyed by its base URL at runtime.
    #[test]
    fn test_pd_kv_event_sources_accepts_rank_pinned_prefill_url() {
        let mut config = pd_config_with_static_kv_sources();
        if let RoutingMode::VllmPrefillDecode { prefill_urls, .. } = &mut config.mode {
            prefill_urls[0].0.push_str("@1");
        }
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_pd_kv_event_sources_rejects_unknown_prefill_url() {
        let mut config = pd_config_with_static_kv_sources();
        let sources = config
            .pd_kv_event_sources
            .remove("http://prefill:8000")
            .unwrap();
        config
            .pd_kv_event_sources
            .insert("http://other:8000".to_string(), sources);
        let err = ConfigValidator::validate(&config).unwrap_err().to_string();
        assert!(err.contains("not a configured prefill URL"), "{err}");
    }

    #[test]
    fn test_pd_kv_event_sources_rejects_bad_endpoints() {
        for (endpoints, block_size, message) in [
            (
                vec!["http://prefill:5557".to_string()],
                16,
                "must be a reachable ZMQ address",
            ),
            (vec!["tcp://*:5557".into()], 16, "reachable ZMQ"),
            (vec!["tcp://0.0.0.0:5557".into()], 16, "reachable ZMQ"),
            (vec!["tcp://[::]:5557".into()], 16, "reachable ZMQ"),
            (vec!["tcp://prefill:0".into()], 16, "reachable ZMQ"),
            (vec!["tcp://prefill".into()], 16, "reachable ZMQ"),
            (vec!["ipc://".into()], 16, "reachable ZMQ"),
            (vec![], 16, "no endpoints"),
            (vec!["tcp://prefill:5557".to_string()], 0, "block_size"),
        ] {
            let mut config = pd_config_with_static_kv_sources();
            let sources = config
                .pd_kv_event_sources
                .get_mut("http://prefill:8000")
                .unwrap();
            sources.endpoints = endpoints;
            sources.block_size = block_size;
            let err = ConfigValidator::validate(&config).unwrap_err().to_string();
            assert!(err.contains(message), "{err}");
        }
    }

    #[test]
    fn test_pd_kv_event_sources_validates_replay_and_accepts_ipc_and_ipv6() {
        let mut config = pd_config_with_static_kv_sources();
        let source = config
            .pd_kv_event_sources
            .get_mut("http://prefill:8000")
            .unwrap();
        source.endpoints = vec!["ipc:///tmp/prefill-kv-events".into()];
        source.replay_endpoints = Some(vec!["tcp://[::1]:5558".into()]);
        assert!(ConfigValidator::validate(&config).is_ok());
        config
            .pd_kv_event_sources
            .get_mut("http://prefill:8000")
            .unwrap()
            .replay_endpoints = Some(vec!["tcp://[::]:5558".into()]);
        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_pd_kv_event_sources_requires_pd_mode_and_kv_aware_policy() {
        let mut config = pd_config_with_static_kv_sources();
        config.mode = RoutingMode::Regular {
            worker_urls: vec!["http://prefill:8000".to_string()],
        };
        let err = ConfigValidator::validate(&config).unwrap_err().to_string();
        assert!(err.contains("requires vllm_prefill_decode mode"), "{err}");

        let mut config = pd_config_with_static_kv_sources();
        config.policy = PolicyConfig::RoundRobin;
        let err = ConfigValidator::validate(&config).unwrap_err().to_string();
        assert!(err.contains("requires a kv_aware prefill policy"), "{err}");

        let mut config = pd_config_with_static_kv_sources();
        config.policy = PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 60,
            max_tree_size: 4096,
        };
        let err = ConfigValidator::validate(&config).unwrap_err().to_string();
        assert!(err.contains("requires a kv_aware prefill policy"), "{err}");
    }
}
