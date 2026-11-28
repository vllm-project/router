#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use clap::{parser::ValueSource, ArgAction, CommandFactory, FromArgMatches, Parser, ValueEnum};
use std::collections::HashMap;
use vllm_router_rs::config::{
    CircuitBreakerConfig, ConfigError, ConfigResult, ConnectionMode, DiscoveryConfig,
    HealthCheckConfig, HistoryBackend, KvConnector, MetricsConfig, PolicyConfig,
    ProgramSchedulingConfig, RetryConfig, RouterConfig, RoutingMode, TraceConfig,
};
use vllm_router_rs::metrics::PrometheusConfig;
use vllm_router_rs::server::{self, ServerConfig};
use vllm_router_rs::service_discovery::ServiceDiscoveryConfig;

type PrefillArgs = (String, Option<u16>);

fn split_prefill_args_from_others(
    args: impl IntoIterator<Item = String>,
) -> Result<(Vec<PrefillArgs>, Vec<String>), String> {
    let mut args = args.into_iter().peekable();
    let mut prefill = vec![];
    let mut other = vec![];

    while let Some(arg) = args.next() {
        if arg != ("--prefill") {
            other.push(arg);
            continue;
        }
        // require a URL after we see the --prefill argument
        let url = args
            .next()
            .ok_or("--prefill requires a URL immediately after")?;
        if url.starts_with("-") {
            return Err(format!("Invalid url {url}"));
        }

        // optional port afterwards
        let port = match args.peek() {
            Some(arg) if arg.eq_ignore_ascii_case("none") => {
                args.next(); // consume it and ignore
                None
            }
            Some(arg) if !arg.starts_with("-") => match arg.parse::<u16>() {
                Ok(port) => {
                    args.next();
                    Some(port)
                }
                Err(_) => None,
            },
            _ => None,
        };
        prefill.push((url, port))
    }

    Ok((prefill, other))
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum Backend {
    #[value(name = "vllm")]
    Vllm,
    #[value(name = "trtllm")]
    Trtllm,
    #[value(name = "openai")]
    Openai,
    #[value(name = "anthropic")]
    Anthropic,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Backend::Vllm => "vllm",
            Backend::Trtllm => "trtllm",
            Backend::Openai => "openai",
            Backend::Anthropic => "anthropic",
        };
        write!(f, "{}", s)
    }
}

#[derive(Parser, Debug)]
#[command(name = "vllm-router")]
#[command(version)]
#[command(about = "VLLM Router - High-performance request distribution across worker nodes")]
#[command(long_about = r#"
VLLM Router - High-performance request distribution across worker nodes

Usage:
This launcher enables starting a router with individual worker instances. It is useful for
multi-node setups or when you want to start workers and router separately.

Examples:
  # Regular mode (HTTP reverse-proxy of OpenAI messages)
  vllm-router --worker-urls http://worker1:8000 http://worker2:8000

  # Regular mode with a vLLM rust Inference worker (router sends token_ids)
  vllm-router --worker-urls grpc://worker1:50051

  # vLLM PD mode with pure service discovery (workers register themselves)
  vllm-router --vllm-pd-disaggregation \
    --vllm-discovery-address 0.0.0.0:30001 \
    --policy consistent_hash

  # Note: In vLLM mode, prefill/decode workers automatically register their
  # HTTP and ZMQ addresses via service discovery. No static --prefill or
  # --decode parameters are needed.

"#)]
struct CliArgs {
    /// Path to a JSON or YAML router configuration file.
    /// Explicit CLI flags override matching file values.
    #[arg(long)]
    config: Option<String>,
    /// E+PD or E/P/D encoder orchestration as JSON (encoder_urls, consumer_zmq_addrs).
    #[arg(long)]
    epd_config: Option<String>,
    /// Host address to bind the router server
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port number to bind the router server
    #[arg(long, default_value_t = 30000)]
    port: u16,

    /// List of worker URLs (`http(s)://` reverse-proxy, or `grpc://` Inference)
    #[arg(long, num_args = 0..)]
    worker_urls: Vec<String>,

    /// Load balancing policy to use. kv_aware uses stage-specific load profiles in PD mode.
    #[arg(long, default_value = "cache_aware", value_parser = ["random", "round_robin", "cache_aware", "kv_aware", "power_of_two", "consistent_hash", "sticky_least_loaded", "rendezvous_hash"])]
    policy: String,

    /// Enable Program-level scheduling independently of the request-level
    /// load-balancing policy. Defaults to verified exact counts with estimation fallback.
    #[arg(long, default_value_t = false)]
    enable_program_scheduling: bool,

    /// Optional JSON object overriding Program-level scheduling defaults.
    /// Requires --enable-program-scheduling. Use {"token_count_mode":"estimated"}
    /// for compatible byte-estimated counts; HTTP text tokenization is opt-in.
    #[arg(long)]
    program_scheduling_config_json: Option<String>,

    /// Enable vLLM PD (Prefill-Decode) disaggregated mode with vLLM-specific two-stage processing
    #[arg(long, default_value_t = false)]
    vllm_pd_disaggregation: bool,

    /// ZMQ service discovery address for vLLM P2P NCCL coordination (e.g., "0.0.0.0:30001")
    /// Required for --vllm-pd-disaggregation mode. Workers register their HTTP and ZMQ addresses here.
    #[arg(long)]
    vllm_discovery_address: Option<String>,

    /// Decode server URL (can be specified multiple times)
    #[arg(long, action = ArgAction::Append)]
    decode: Vec<String>,

    /// Restrict a Prefill worker to its compatible Decode workers in a mixed P/D pool.
    #[arg(long = "pd-pair", action = ArgAction::Append, value_name = "PREFILL=DECODE[,DECODE...]")]
    pd_pair: Vec<String>,

    /// Static KV event publishers for KV-aware prefill or decode workers.
    /// Specify "URL=ZMQ_PUB_ENDPOINT[,ENDPOINT...]" once per worker in DP-rank
    /// order. Block size defaults to 16.
    /// Use the shared KV event replay, block size and model options for static sources.
    #[arg(
        long = "pd-kv-events",
        action = ArgAction::Append,
        value_name = "URL=ENDPOINT[,ENDPOINT...]"
    )]
    pd_kv_events: Vec<String>,
    /// Specific policy for prefill nodes in PD mode. kv_aware uses the prefill load profile.
    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "kv_aware", "power_of_two", "consistent_hash", "sticky_least_loaded", "rendezvous_hash"])]
    prefill_policy: Option<String>,

    /// Specific policy for decode nodes in PD mode (defaults to --policy).
    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "kv_aware", "power_of_two", "consistent_hash", "sticky_least_loaded", "rendezvous_hash"])]
    decode_policy: Option<String>,

    /// Timeout in seconds for worker startup
    #[arg(long, default_value_t = 600)]
    worker_startup_timeout_secs: u64,

    /// Interval in seconds between checks for worker startup
    #[arg(long, default_value_t = 30)]
    worker_startup_check_interval: u64,

    /// Cache threshold (0.0-1.0) for cache-aware routing
    #[arg(long, default_value_t = 0.3)]
    cache_threshold: f32,

    /// Absolute threshold for load balancing
    #[arg(long, default_value_t = 64)]
    balance_abs_threshold: usize,

    /// Relative threshold for load balancing
    #[arg(long, default_value_t = 1.5)]
    balance_rel_threshold: f32,

    /// Interval in seconds between cache eviction operations
    #[arg(long, default_value_t = 120)]
    eviction_interval: u64,

    /// Per-worker character budget in each model tree (enforced at cache eviction)
    #[arg(long, default_value_t = 1 << 25)]
    max_tree_size: usize,

    /// Softmax temperature for kv_aware (0 selects the minimum cost).
    #[arg(long, default_value_t = 0.0)]
    router_temperature: f64,

    /// Enable kv_aware session affinity and set its idle TTL in seconds.
    /// P/D uses decode-only soft affinity based on pending plus active requests.
    /// Omit to disable session affinity while keeping KV-aware routing enabled.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=31_536_000))]
    kv_aware_session_affinity_ttl_secs: Option<u64>,
    /// Static KV event publishers for regular kv_aware workers.
    #[arg(long = "kv-event-source", action = ArgAction::Append, value_name = "WORKER_URL=ENDPOINT[,ENDPOINT...]")]
    kv_event_source: Vec<String>,
    /// Optional ZMQ replay endpoints, one per DP rank, keyed by worker URL.
    #[arg(long = "kv-event-replay-source", action = ArgAction::Append, value_name = "WORKER_URL=ENDPOINT[,ENDPOINT...]")]
    kv_event_replay_source: Vec<String>,
    /// Topics for static KV event publishers in DP-rank order.
    #[arg(long, action = ArgAction::Append, value_name = "WORKER_URL=TOPIC[,TOPIC...]")]
    kv_event_topic: Vec<String>,
    /// KV cache block size for static event publishers.
    #[arg(long, default_value_t = 16)]
    kv_event_block_size: u32,
    /// Shared model identity for workers using static event publishers.
    #[arg(long)]
    kv_event_model: Option<String>,
    /// Require complete KV event recovery or a worker KV clear to restore placement credit.
    #[arg(long, default_value_t = false)]
    kv_event_strict_recovery: bool,
    /// KV cache layouts for colocated or P/D workers, keyed by worker URL.
    #[arg(long)]
    kv_hybrid_layout: Option<std::path::PathBuf>,
    /// Transfer credit for host-resident KV blocks.
    #[arg(long, default_value_t = 0.8)]
    kv_penalty_host: f64,
    #[arg(long, default_value_t = 0.6)]
    kv_penalty_external: f64,
    #[arg(long, default_value_t = 0.3)]
    kv_penalty_disk: f64,

    /// Maximum payload size in bytes
    #[arg(long, default_value_t = 536870912)] // 512MB
    max_payload_size: usize,

    /// Path to a WASM Component Model OnRequest middleware artifact
    #[arg(long)]
    wasm_middleware: Option<String>,

    /// Optional SHA-256 hex digest that must match --wasm-middleware
    #[arg(long)]
    wasm_middleware_sha256: Option<String>,

    /// HTTP paths that invoke the WASM middleware (repeatable).
    /// Defaults to /v1/chat/completions when --wasm-middleware is set.
    #[arg(long = "wasm-middleware-route", action = ArgAction::Append)]
    wasm_middleware_routes: Vec<String>,

    /// Intra-node data parallel size (number of DP replicas per worker URL). When > 1, the router will create multiple worker instances per URL, one for each DP rank.
    #[arg(long, default_value_t = 1)]
    intra_node_data_parallel_size: usize,

    /// API key for worker authorization
    #[arg(long)]
    api_key: Option<String>,

    /// API key validation URLs (defaults to env file)
    #[arg(long, num_args = 0..)]
    api_key_validation_urls: Vec<String>,

    /// Backend to route requests to (vllm, trtllm, openai, anthropic)
    #[arg(long, value_enum, default_value_t = Backend::Vllm, alias = "runtime")]
    backend: Backend,

    /// Directory to store log files
    #[arg(long)]
    log_dir: Option<String>,

    /// Set the logging level
    #[arg(long, default_value = "info", value_parser = ["debug", "info", "warn", "error"])]
    log_level: String,

    /// Enable OpenTelemetry tracing
    #[arg(
        long,
        default_value_t = false,
        help_heading = "Tracing (OpenTelemetry)"
    )]
    enable_trace: bool,

    /// OTLP collector endpoint (format: host:port). If omitted, respects OTEL_EXPORTER_OTLP_ENDPOINT.
    #[arg(long, help_heading = "Tracing (OpenTelemetry)")]
    otlp_traces_endpoint: Option<String>,

    /// Parent-based sampling ratio for OpenTelemetry traces.
    #[arg(long, default_value_t = 1.0, help_heading = "Tracing (OpenTelemetry)")]
    otel_sampling_ratio: f64,

    /// Exact HTTP paths to exclude from OpenTelemetry server spans.
    #[arg(long, num_args = 0.., help_heading = "Tracing (OpenTelemetry)")]
    otel_excluded_paths: Vec<String>,

    /// Enable Kubernetes service discovery
    #[arg(long, default_value_t = false)]
    service_discovery: bool,

    /// Label selector for Kubernetes service discovery (format: key1=value1 key2=value2)
    #[arg(long, num_args = 0..)]
    selector: Vec<String>,

    /// Port to use for discovered worker pods
    #[arg(long, default_value_t = 80)]
    service_discovery_port: u16,

    /// Kubernetes namespace to watch for pods
    #[arg(long)]
    service_discovery_namespace: Option<String>,

    /// Label selector for prefill server pods in PD mode
    #[arg(long, num_args = 0..)]
    prefill_selector: Vec<String>,

    /// Label selector for decode server pods in PD mode
    #[arg(long, num_args = 0..)]
    decode_selector: Vec<String>,

    /// Port to expose Prometheus metrics
    #[arg(long, default_value_t = 29000)]
    prometheus_port: u16,

    /// Host address to bind the Prometheus metrics server
    #[arg(long, default_value = "127.0.0.1")]
    prometheus_host: String,

    /// Custom HTTP headers to check for request IDs
    #[arg(long, num_args = 0..)]
    request_id_headers: Vec<String>,

    /// Request timeout in seconds
    #[arg(long, default_value_t = 1800)]
    request_timeout_secs: u64,

    /// Idle timeout in seconds for pooled backend HTTP connections. Should be
    /// lower than the backend's keep-alive timeout (vLLM defaults to 5s); 0
    /// disables connection reuse entirely
    #[arg(long, default_value_t = vllm_router_rs::server::DEFAULT_POOL_IDLE_TIMEOUT_SECS)]
    pool_idle_timeout_secs: u64,

    /// Maximum number of concurrent requests allowed
    #[arg(long, default_value_t = 32768)]
    max_concurrent_requests: usize,

    /// CORS allowed origins
    #[arg(long, num_args = 0..)]
    cors_allowed_origins: Vec<String>,

    // Retry configuration
    /// Maximum number of retries
    #[arg(long, default_value_t = 5)]
    retry_max_retries: u32,

    /// Initial backoff in milliseconds for retries
    #[arg(long, default_value_t = 50)]
    retry_initial_backoff_ms: u64,

    /// Maximum backoff in milliseconds for retries
    #[arg(long, default_value_t = 30000)]
    retry_max_backoff_ms: u64,

    /// Backoff multiplier for exponential backoff
    #[arg(long, default_value_t = 1.5)]
    retry_backoff_multiplier: f32,

    /// Jitter factor for retry backoff
    #[arg(long, default_value_t = 0.2)]
    retry_jitter_factor: f32,

    /// Disable retries
    #[arg(long, default_value_t = false)]
    disable_retries: bool,

    // Circuit breaker configuration
    /// Number of failures before circuit breaker opens
    #[arg(long, default_value_t = 10)]
    cb_failure_threshold: u32,

    /// Number of successes before circuit breaker closes
    #[arg(long, default_value_t = 3)]
    cb_success_threshold: u32,

    /// Timeout duration in seconds for circuit breaker
    #[arg(long, default_value_t = 60)]
    cb_timeout_duration_secs: u64,

    /// Window duration in seconds for circuit breaker
    #[arg(long, default_value_t = 120)]
    cb_window_duration_secs: u64,

    /// Disable circuit breaker
    #[arg(long, default_value_t = false)]
    disable_circuit_breaker: bool,

    // Health check configuration
    /// Number of consecutive health check failures before marking worker unhealthy
    #[arg(long, default_value_t = 3)]
    health_failure_threshold: u32,

    /// Number of consecutive health check successes before marking worker healthy
    #[arg(long, default_value_t = 2)]
    health_success_threshold: u32,

    /// Timeout in seconds for health check requests
    #[arg(long, default_value_t = 5)]
    health_check_timeout_secs: u64,

    /// Interval in seconds between runtime health checks
    #[arg(long, default_value_t = 60)]
    health_check_interval_secs: u64,

    /// Health check endpoint path
    #[arg(long, default_value = "/health")]
    health_check_endpoint: String,

    // IGW (Inference Gateway) configuration
    /// Enable Inference Gateway mode
    #[arg(long, default_value_t = false)]
    enable_igw: bool,

    /// History backend configuration (memory or none)
    #[arg(long, default_value = "memory", value_parser = ["memory", "none"])]
    history_backend: String,

    /// Enable profiling calls to vLLM workers
    #[arg(long, default_value_t = false)]
    profile: bool,

    /// KV connector type for PD disaggregation (nixl or mooncake)
    #[arg(long, value_enum, default_value_t = KvConnector::Nixl)]
    kv_connector: KvConnector,
}

impl CliArgs {
    /// Parse selector strings into HashMap
    fn parse_selector(selector_list: &[String]) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for item in selector_list {
            if let Some(eq_pos) = item.find('=') {
                let key = item[..eq_pos].to_string();
                let value = item[eq_pos + 1..].to_string();
                map.insert(key, value);
            }
        }
        map
    }

    /// Convert policy string to PolicyConfig
    fn parse_policy(&self, policy_str: &str) -> PolicyConfig {
        match policy_str {
            "random" => PolicyConfig::Random,
            "round_robin" => PolicyConfig::RoundRobin,
            "cache_aware" => PolicyConfig::CacheAware {
                cache_threshold: self.cache_threshold,
                balance_abs_threshold: self.balance_abs_threshold,
                balance_rel_threshold: self.balance_rel_threshold,
                eviction_interval_secs: self.eviction_interval,
                max_tree_size: self.max_tree_size,
            },
            "kv_aware" => PolicyConfig::KvAware {
                router_temperature: self.router_temperature,
                kv_penalty_host: self.kv_penalty_host,
                kv_penalty_external: self.kv_penalty_external,
                kv_penalty_disk: self.kv_penalty_disk,
                static_kv_event_sources: Box::default(),
            },
            "power_of_two" => PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 5, // Default value
            },
            "consistent_hash" => PolicyConfig::ConsistentHash {
                virtual_nodes: 160, // Default value
            },
            "sticky_least_loaded" => PolicyConfig::StickyLeastLoaded,
            "rendezvous_hash" => PolicyConfig::RendezvousHash,
            _ => PolicyConfig::RoundRobin, // Fallback
        }
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn parse_pd_kv_events(
        specs: &[String],
    ) -> ConfigResult<HashMap<String, vllm_router_rs::config::PdKvEventSources>> {
        vllm_router_rs::config::parse_pd_kv_events(specs)
    }

    /// Convert CLI arguments to RouterConfig
    fn to_router_config(
        &self,
        prefill_urls: Vec<(String, Option<u16>)>,
    ) -> ConfigResult<RouterConfig> {
        // Determine routing mode
        let mode = if self.enable_igw {
            // IGW mode - routing mode is not used in IGW, but we need to provide a placeholder
            RoutingMode::Regular {
                worker_urls: vec![],
            }
        } else if matches!(self.backend, Backend::Openai) {
            // OpenAI backend mode - use worker_urls as base(s)
            RoutingMode::OpenAI {
                worker_urls: self.worker_urls.clone(),
            }
        } else if self.vllm_pd_disaggregation {
            // Use decode URLs from CLI arguments (already parsed by clap)
            let decode_urls = &self.decode;

            // Support multiple discovery/configuration modes:
            // 1. Static URLs (--prefill/--decode)
            // 2. vLLM ZMQ discovery (--vllm-discovery-address)
            // 3. K8s service discovery (--service-discovery with --prefill-selector/--decode-selector)
            let use_static_urls = !prefill_urls.is_empty() || !decode_urls.is_empty();
            let use_vllm_discovery = self.vllm_discovery_address.is_some();
            let use_k8s_discovery = self.service_discovery
                && (!self.prefill_selector.is_empty() || !self.decode_selector.is_empty());

            if !use_static_urls && !use_vllm_discovery && !use_k8s_discovery {
                return Err(ConfigError::ValidationFailed {
                    reason: "vLLM PD disaggregation mode requires one of: --vllm-discovery-address, --prefill/--decode URLs, or --service-discovery with --prefill-selector/--decode-selector".to_string(),
                });
            }

            // Use decode URLs directly from CLI
            let final_decode_urls = decode_urls.clone();

            // Log the discovery/configuration mode being used
            if use_k8s_discovery {
                eprintln!("ℹ️  INFO: Using K8s service discovery mode for vLLM PD disaggregation.");
                eprintln!("   Prefill selector: {:?}", self.prefill_selector);
                eprintln!("   Decode selector: {:?}", self.decode_selector);
                if use_static_urls {
                    eprintln!(
                        "   Static fallback URLs - Prefill: {:?}, Decode: {:?}",
                        prefill_urls, final_decode_urls
                    );
                }
            } else if use_static_urls && use_vllm_discovery {
                eprintln!("ℹ️  INFO: Using hybrid mode - static URLs as fallback, vLLM ZMQ discovery for dynamic workers.");
                eprintln!("   Prefill URLs: {:?}", prefill_urls);
                eprintln!("   Decode URLs: {:?}", final_decode_urls);
                eprintln!("   Discovery address: {:?}", self.vllm_discovery_address);
            } else if use_static_urls {
                eprintln!("ℹ️  INFO: Using static URL mode without service discovery.");
                eprintln!("   Prefill URLs: {:?}", prefill_urls);
                eprintln!("   Decode URLs: {:?}", final_decode_urls);
            } else {
                eprintln!("ℹ️  INFO: Using vLLM ZMQ service discovery mode.");
                eprintln!("   Discovery address: {:?}", self.vllm_discovery_address);
            }

            RoutingMode::VllmPrefillDecode {
                prefill_urls: prefill_urls.clone(),
                decode_urls: final_decode_urls,
                prefill_policy: self.prefill_policy.as_ref().map(|p| self.parse_policy(p)),
                decode_policy: self.decode_policy.as_ref().map(|p| self.parse_policy(p)),
                discovery_address: self.vllm_discovery_address.clone(),
            }
        } else {
            // Regular mode
            if !self.service_discovery && self.worker_urls.is_empty() {
                return Err(ConfigError::ValidationFailed {
                    reason: "Regular mode requires --worker-urls when not using service discovery"
                        .to_string(),
                });
            }
            RoutingMode::Regular {
                worker_urls: self.worker_urls.clone(),
            }
        };

        let kv_sources = vllm_router_rs::config::KvSourceOptions {
            layout: self.kv_hybrid_layout.as_deref(),
            publishers: &self.kv_event_source,
            pd_publishers: &self.pd_kv_events,
            replay: &self.kv_event_replay_source,
            topics: &self.kv_event_topic,
            block_size: self.kv_event_block_size,
            model: self.kv_event_model.as_deref(),
        }
        .load(&mode)?;

        // Main policy
        let mut policy = self.parse_policy(&self.policy);
        if let PolicyConfig::KvAware {
            static_kv_event_sources: configured,
            ..
        } = &mut policy
        {
            **configured = kv_sources.regular;
        } else if !kv_sources.regular.is_empty() {
            return Err(ConfigError::IncompatibleConfig {
                reason: "--kv-event-source and --kv-hybrid-layout require --policy kv_aware in regular mode".to_string(),
            });
        }

        // Service discovery configuration
        let discovery = if self.service_discovery {
            Some(DiscoveryConfig {
                enabled: true,
                namespace: self.service_discovery_namespace.clone(),
                port: self.service_discovery_port,
                check_interval_secs: 60,
                selector: Self::parse_selector(&self.selector),
                prefill_selector: Self::parse_selector(&self.prefill_selector),
                decode_selector: Self::parse_selector(&self.decode_selector),
                bootstrap_port_annotation: "vllm.ai/bootstrap-port".to_string(),
            })
        } else {
            None
        };

        // Metrics configuration
        let metrics = Some(MetricsConfig {
            port: self.prometheus_port,
            host: self.prometheus_host.clone(),
        });

        let connection_mode = ConnectionMode::Http;

        let api_key_validation_urls = if !self.api_key_validation_urls.is_empty() {
            self.api_key_validation_urls.clone()
        } else if let Ok(raw_urls) = std::env::var("API_KEY_VALIDATION_URLS") {
            raw_urls
                .split(',')
                .map(|url| url.trim().to_string())
                .filter(|url| !url.is_empty())
                .collect()
        } else {
            Vec::new()
        };

        let program_scheduling = ProgramSchedulingConfig::resolve(
            self.enable_program_scheduling,
            self.program_scheduling_config_json.as_deref(),
        )?;

        // Build RouterConfig
        Ok(RouterConfig {
            epd: self
                .epd_config
                .as_deref()
                .map(serde_json::from_str)
                .transpose()
                .map_err(|e| vllm_router_rs::config::ConfigError::ValidationFailed {
                    reason: format!("Invalid --epd-config: {e}"),
                })?,
            mode,
            pd_kv_event_sources: kv_sources.pd,
            pd_worker_pairs: vllm_router_rs::config::parse_pd_worker_pairs(&self.pd_pair)?,
            kv_event_strict_recovery: self.kv_event_strict_recovery,
            policy,
            connection_mode,
            host: self.host.clone(),
            port: self.port,
            max_payload_size: self.max_payload_size,
            request_timeout_secs: self.request_timeout_secs,
            worker_startup_timeout_secs: self.worker_startup_timeout_secs,
            worker_startup_check_interval_secs: self.worker_startup_check_interval,
            intra_node_data_parallel_size: self.intra_node_data_parallel_size,
            api_key: self.api_key.clone(),
            api_key_validation_urls,
            discovery,
            metrics,
            log_dir: self.log_dir.clone(),
            log_level: Some(self.log_level.clone()),
            request_id_headers: if self.request_id_headers.is_empty() {
                None
            } else {
                Some(self.request_id_headers.clone())
            },
            max_concurrent_requests: self.max_concurrent_requests,
            queue_size: 100,        // Default queue size
            queue_timeout_secs: 60, // Default timeout
            cors_allowed_origins: self.cors_allowed_origins.clone(),
            retry: RetryConfig {
                max_retries: self.retry_max_retries,
                initial_backoff_ms: self.retry_initial_backoff_ms,
                max_backoff_ms: self.retry_max_backoff_ms,
                backoff_multiplier: self.retry_backoff_multiplier,
                jitter_factor: self.retry_jitter_factor,
            },
            circuit_breaker: CircuitBreakerConfig {
                failure_threshold: self.cb_failure_threshold,
                success_threshold: self.cb_success_threshold,
                timeout_duration_secs: self.cb_timeout_duration_secs,
                window_duration_secs: self.cb_window_duration_secs,
            },
            disable_retries: self.disable_retries,
            disable_circuit_breaker: self.disable_circuit_breaker,
            kv_aware_session_affinity_ttl_secs: self.kv_aware_session_affinity_ttl_secs,
            health_check: HealthCheckConfig {
                failure_threshold: self.health_failure_threshold,
                success_threshold: self.health_success_threshold,
                timeout_secs: self.health_check_timeout_secs,
                check_interval_secs: self.health_check_interval_secs,
                endpoint: self.health_check_endpoint.clone(),
            },
            enable_igw: self.enable_igw,
            rate_limit_tokens_per_second: None,
            history_backend: match self.history_backend.as_str() {
                "none" => HistoryBackend::None,
                _ => HistoryBackend::Memory,
            },
            enable_profiling: self.profile,
            profile_timeout_secs: 10, // Default profiling timeout
            kv_connector: self.kv_connector,
            program_scheduling,
        })
    }

    /// Resolve `RouterConfig` from CLI flags alone or a `--config` file with explicit CLI overrides.
    fn resolve_router_config(
        &self,
        matches: &clap::ArgMatches,
        prefill_urls: Vec<(String, Option<u16>)>,
    ) -> ConfigResult<RouterConfig> {
        let Some(config_path) = &self.config else {
            return self.to_router_config(prefill_urls);
        };
        let mut config = RouterConfig::from_file(config_path)?;
        self.apply_cli_overrides(&mut config, matches, prefill_urls)?;
        Ok(config)
    }

    fn apply_cli_overrides(
        &self,
        config: &mut RouterConfig,
        matches: &clap::ArgMatches,
        prefill_urls: Vec<(String, Option<u16>)>,
    ) -> ConfigResult<()> {
        let has_cli = |id: &str| matches.value_source(id) == Some(ValueSource::CommandLine);

        self.apply_mode_overrides(config, &has_cli, prefill_urls)?;

        if has_cli("policy") {
            let parsed = self.parse_policy(&self.policy);
            if std::mem::discriminant(&config.policy) != std::mem::discriminant(&parsed) {
                config.policy = parsed;
            }
        }
        self.apply_policy_param_overrides(config, &has_cli);
        self.apply_kv_event_overrides(config, &has_cli)?;

        if has_cli("host") {
            config.host = self.host.clone();
        }
        if has_cli("port") {
            config.port = self.port;
        }
        if has_cli("max_payload_size") {
            config.max_payload_size = self.max_payload_size;
        }
        if has_cli("request_timeout_secs") {
            config.request_timeout_secs = self.request_timeout_secs;
        }
        if has_cli("worker_startup_timeout_secs") {
            config.worker_startup_timeout_secs = self.worker_startup_timeout_secs;
        }
        if has_cli("worker_startup_check_interval") {
            config.worker_startup_check_interval_secs = self.worker_startup_check_interval;
        }
        if has_cli("intra_node_data_parallel_size") {
            config.intra_node_data_parallel_size = self.intra_node_data_parallel_size;
        }
        if has_cli("api_key") {
            config.api_key = self.api_key.clone();
        }
        if has_cli("api_key_validation_urls") {
            config.api_key_validation_urls = self.api_key_validation_urls.clone();
        } else if config.api_key_validation_urls.is_empty() {
            if let Ok(raw_urls) = std::env::var("API_KEY_VALIDATION_URLS") {
                config.api_key_validation_urls = raw_urls
                    .split(',')
                    .map(|url| url.trim().to_string())
                    .filter(|url| !url.is_empty())
                    .collect();
            }
        }
        if has_cli("log_dir") {
            config.log_dir = self.log_dir.clone();
        }
        if has_cli("log_level") {
            config.log_level = Some(self.log_level.clone());
        }
        if has_cli("request_id_headers") {
            config.request_id_headers = if self.request_id_headers.is_empty() {
                None
            } else {
                Some(self.request_id_headers.clone())
            };
        }
        if has_cli("max_concurrent_requests") {
            config.max_concurrent_requests = self.max_concurrent_requests;
        }
        if has_cli("cors_allowed_origins") {
            config.cors_allowed_origins = self.cors_allowed_origins.clone();
        }
        if has_cli("disable_retries") {
            config.disable_retries = self.disable_retries;
        }
        if has_cli("disable_circuit_breaker") {
            config.disable_circuit_breaker = self.disable_circuit_breaker;
        }
        if has_cli("kv_aware_session_affinity_ttl_secs") {
            config.kv_aware_session_affinity_ttl_secs = self.kv_aware_session_affinity_ttl_secs;
        }
        if has_cli("kv_event_strict_recovery") {
            config.kv_event_strict_recovery = self.kv_event_strict_recovery;
        }
        if has_cli("history_backend") {
            config.history_backend = match self.history_backend.as_str() {
                "none" => HistoryBackend::None,
                _ => HistoryBackend::Memory,
            };
        }
        if has_cli("profile") {
            config.enable_profiling = self.profile;
        }
        if has_cli("kv_connector") {
            config.kv_connector = self.kv_connector;
        }
        if has_cli("epd_config") {
            config.epd = self
                .epd_config
                .as_deref()
                .map(serde_json::from_str)
                .transpose()
                .map_err(|e| ConfigError::ValidationFailed {
                    reason: format!("Invalid --epd-config: {e}"),
                })?;
        }
        if has_cli("pd_pair") {
            config.pd_worker_pairs = vllm_router_rs::config::parse_pd_worker_pairs(&self.pd_pair)?;
        }
        if has_cli("enable_program_scheduling") || has_cli("program_scheduling_config_json") {
            let enabled = if has_cli("enable_program_scheduling") {
                self.enable_program_scheduling
            } else {
                config.program_scheduling.is_some()
            };
            if has_cli("program_scheduling_config_json")
                || !enabled
                || config.program_scheduling.is_none()
            {
                config.program_scheduling = ProgramSchedulingConfig::resolve(
                    enabled,
                    self.program_scheduling_config_json.as_deref(),
                )?;
            }
        }

        self.apply_subconfig_overrides(config, &has_cli);
        Ok(())
    }

    fn apply_mode_overrides(
        &self,
        config: &mut RouterConfig,
        has_cli: &dyn Fn(&str) -> bool,
        prefill_urls: Vec<(String, Option<u16>)>,
    ) -> ConfigResult<()> {
        if has_cli("enable_igw") && self.enable_igw {
            config.enable_igw = true;
            config.mode = RoutingMode::Regular {
                worker_urls: Vec::new(),
            };
            return Ok(());
        }
        if has_cli("enable_igw") {
            config.enable_igw = false;
        }

        if has_cli("backend") && matches!(self.backend, Backend::Openai) {
            let worker_urls = if has_cli("worker_urls") {
                self.worker_urls.clone()
            } else {
                match &config.mode {
                    RoutingMode::Regular { worker_urls } | RoutingMode::OpenAI { worker_urls } => {
                        worker_urls.clone()
                    }
                    _ => Vec::new(),
                }
            };
            config.mode = RoutingMode::OpenAI { worker_urls };
            return Ok(());
        }

        let pd_cli_requested = (has_cli("vllm_pd_disaggregation") && self.vllm_pd_disaggregation)
            || !prefill_urls.is_empty()
            || has_cli("decode")
            || has_cli("vllm_discovery_address")
            || has_cli("prefill_policy")
            || has_cli("decode_policy");

        if pd_cli_requested {
            let (base_prefill, base_decode, base_p_pol, base_d_pol, base_disc) = match &config.mode
            {
                RoutingMode::VllmPrefillDecode {
                    prefill_urls: p,
                    decode_urls: d,
                    prefill_policy: pp,
                    decode_policy: dp,
                    discovery_address: da,
                } => (p.clone(), d.clone(), pp.clone(), dp.clone(), da.clone()),
                _ => (Vec::new(), Vec::new(), None, None, None),
            };
            config.mode = RoutingMode::VllmPrefillDecode {
                prefill_urls: if prefill_urls.is_empty() {
                    base_prefill
                } else {
                    prefill_urls
                },
                decode_urls: if has_cli("decode") {
                    self.decode.clone()
                } else {
                    base_decode
                },
                prefill_policy: self
                    .resolve_role_policy(self.prefill_policy.as_deref(), base_p_pol),
                decode_policy: self.resolve_role_policy(self.decode_policy.as_deref(), base_d_pol),
                discovery_address: if has_cli("vllm_discovery_address") {
                    self.vllm_discovery_address.clone()
                } else {
                    base_disc
                },
            };
            return Ok(());
        }

        if matches!(config.mode, RoutingMode::OpenAI { .. })
            && has_cli("backend")
            && !matches!(self.backend, Backend::Openai)
        {
            let worker_urls = if has_cli("worker_urls") {
                self.worker_urls.clone()
            } else if let RoutingMode::OpenAI { worker_urls } = &config.mode {
                worker_urls.clone()
            } else {
                Vec::new()
            };
            config.mode = RoutingMode::Regular { worker_urls };
            return Ok(());
        }

        if has_cli("worker_urls") {
            match &mut config.mode {
                RoutingMode::Regular { worker_urls } | RoutingMode::OpenAI { worker_urls } => {
                    *worker_urls = self.worker_urls.clone();
                }
                _ => {
                    config.mode = RoutingMode::Regular {
                        worker_urls: self.worker_urls.clone(),
                    };
                }
            }
        }
        Ok(())
    }

    fn apply_kv_event_overrides(
        &self,
        config: &mut RouterConfig,
        has_cli: &dyn Fn(&str) -> bool,
    ) -> ConfigResult<()> {
        let has_kv_cli = has_cli("kv_hybrid_layout")
            || has_cli("kv_event_source")
            || has_cli("pd_kv_events")
            || has_cli("kv_event_replay_source")
            || has_cli("kv_event_topic")
            || has_cli("kv_event_block_size")
            || has_cli("kv_event_model");
        if !has_kv_cli {
            return Ok(());
        }

        let kv_sources = vllm_router_rs::config::KvSourceOptions {
            layout: self.kv_hybrid_layout.as_deref(),
            publishers: &self.kv_event_source,
            pd_publishers: &self.pd_kv_events,
            replay: &self.kv_event_replay_source,
            topics: &self.kv_event_topic,
            block_size: self.kv_event_block_size,
            model: self.kv_event_model.as_deref(),
        }
        .load(&config.mode)?;

        if !kv_sources.pd.is_empty() {
            config.pd_kv_event_sources = kv_sources.pd;
        }
        if !kv_sources.regular.is_empty() {
            if let PolicyConfig::KvAware {
                static_kv_event_sources,
                ..
            } = &mut config.policy
            {
                **static_kv_event_sources = kv_sources.regular;
            } else {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "--kv-event-source and --kv-hybrid-layout require kv_aware policy in regular mode".to_string(),
                });
            }
        }
        Ok(())
    }

    fn resolve_role_policy(
        &self,
        cli_policy: Option<&str>,
        base_policy: Option<PolicyConfig>,
    ) -> Option<PolicyConfig> {
        match cli_policy {
            Some(p) => {
                let parsed = self.parse_policy(p);
                match base_policy {
                    Some(existing)
                        if std::mem::discriminant(&existing) == std::mem::discriminant(&parsed) =>
                    {
                        Some(existing)
                    }
                    _ => Some(parsed),
                }
            }
            None => base_policy,
        }
    }

    fn apply_policy_param_overrides(
        &self,
        config: &mut RouterConfig,
        has_cli: &dyn Fn(&str) -> bool,
    ) {
        self.apply_policy_param_overrides_to(&mut config.policy, has_cli);
        if let RoutingMode::VllmPrefillDecode {
            prefill_policy,
            decode_policy,
            ..
        } = &mut config.mode
        {
            if let Some(policy) = prefill_policy.as_mut() {
                self.apply_policy_param_overrides_to(policy, has_cli);
            }
            if let Some(policy) = decode_policy.as_mut() {
                self.apply_policy_param_overrides_to(policy, has_cli);
            }
        }
    }

    fn apply_policy_param_overrides_to(
        &self,
        policy: &mut PolicyConfig,
        has_cli: &dyn Fn(&str) -> bool,
    ) {
        match policy {
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            } => {
                if has_cli("cache_threshold") {
                    *cache_threshold = self.cache_threshold;
                }
                if has_cli("balance_abs_threshold") {
                    *balance_abs_threshold = self.balance_abs_threshold;
                }
                if has_cli("balance_rel_threshold") {
                    *balance_rel_threshold = self.balance_rel_threshold;
                }
                if has_cli("eviction_interval") {
                    *eviction_interval_secs = self.eviction_interval;
                }
                if has_cli("max_tree_size") {
                    *max_tree_size = self.max_tree_size;
                }
            }
            PolicyConfig::KvAware {
                router_temperature,
                kv_penalty_host,
                kv_penalty_external,
                kv_penalty_disk,
                ..
            } => {
                if has_cli("router_temperature") {
                    *router_temperature = self.router_temperature;
                }
                if has_cli("kv_penalty_host") {
                    *kv_penalty_host = self.kv_penalty_host;
                }
                if has_cli("kv_penalty_external") {
                    *kv_penalty_external = self.kv_penalty_external;
                }
                if has_cli("kv_penalty_disk") {
                    *kv_penalty_disk = self.kv_penalty_disk;
                }
            }
            _ => {}
        }
    }

    fn apply_subconfig_overrides(&self, config: &mut RouterConfig, has_cli: &dyn Fn(&str) -> bool) {
        if has_cli("retry_max_retries") {
            config.retry.max_retries = self.retry_max_retries;
        }
        if has_cli("retry_initial_backoff_ms") {
            config.retry.initial_backoff_ms = self.retry_initial_backoff_ms;
        }
        if has_cli("retry_max_backoff_ms") {
            config.retry.max_backoff_ms = self.retry_max_backoff_ms;
        }
        if has_cli("retry_backoff_multiplier") {
            config.retry.backoff_multiplier = self.retry_backoff_multiplier;
        }
        if has_cli("retry_jitter_factor") {
            config.retry.jitter_factor = self.retry_jitter_factor;
        }

        if has_cli("cb_failure_threshold") {
            config.circuit_breaker.failure_threshold = self.cb_failure_threshold;
        }
        if has_cli("cb_success_threshold") {
            config.circuit_breaker.success_threshold = self.cb_success_threshold;
        }
        if has_cli("cb_timeout_duration_secs") {
            config.circuit_breaker.timeout_duration_secs = self.cb_timeout_duration_secs;
        }
        if has_cli("cb_window_duration_secs") {
            config.circuit_breaker.window_duration_secs = self.cb_window_duration_secs;
        }

        if has_cli("health_failure_threshold") {
            config.health_check.failure_threshold = self.health_failure_threshold;
        }
        if has_cli("health_success_threshold") {
            config.health_check.success_threshold = self.health_success_threshold;
        }
        if has_cli("health_check_timeout_secs") {
            config.health_check.timeout_secs = self.health_check_timeout_secs;
        }
        if has_cli("health_check_interval_secs") {
            config.health_check.check_interval_secs = self.health_check_interval_secs;
        }
        if has_cli("health_check_endpoint") {
            config.health_check.endpoint = self.health_check_endpoint.clone();
        }

        if has_cli("prometheus_port") || has_cli("prometheus_host") {
            let metrics = config.metrics.get_or_insert_with(MetricsConfig::default);
            if has_cli("prometheus_port") {
                metrics.port = self.prometheus_port;
            }
            if has_cli("prometheus_host") {
                metrics.host = self.prometheus_host.clone();
            }
        }

        if has_cli("service_discovery")
            || has_cli("service_discovery_namespace")
            || has_cli("service_discovery_port")
            || has_cli("selector")
            || has_cli("prefill_selector")
            || has_cli("decode_selector")
        {
            let discovery = config
                .discovery
                .get_or_insert_with(DiscoveryConfig::default);
            if has_cli("service_discovery") {
                discovery.enabled = self.service_discovery;
            }
            if has_cli("service_discovery_namespace") {
                discovery.namespace = self.service_discovery_namespace.clone();
            }
            if has_cli("service_discovery_port") {
                discovery.port = self.service_discovery_port;
            }
            if has_cli("selector") {
                discovery.selector = Self::parse_selector(&self.selector);
            }
            if has_cli("prefill_selector") {
                discovery.prefill_selector = Self::parse_selector(&self.prefill_selector);
            }
            if has_cli("decode_selector") {
                discovery.decode_selector = Self::parse_selector(&self.decode_selector);
            }
        }
    }

    /// Create ServerConfig from CLI args and resolved RouterConfig
    fn to_server_config(&self, router_config: RouterConfig) -> ServerConfig {
        let service_discovery_config =
            router_config
                .discovery
                .as_ref()
                .filter(|d| d.enabled)
                .map(|d| ServiceDiscoveryConfig {
                    enabled: true,
                    selector: d.selector.clone(),
                    check_interval: std::time::Duration::from_secs(d.check_interval_secs),
                    port: d.port,
                    namespace: d.namespace.clone(),
                    // HTTP service discovery only supports the vLLM PD router.
                    pd_mode: router_config.mode.is_pd_mode(),
                    prefill_selector: d.prefill_selector.clone(),
                    decode_selector: d.decode_selector.clone(),
                    bootstrap_port_annotation: d.bootstrap_port_annotation.clone(),
                });

        let prometheus_config = router_config.metrics.as_ref().map(|m| PrometheusConfig {
            port: m.port,
            host: m.host.clone(),
        });

        ServerConfig {
            host: router_config.host.clone(),
            port: router_config.port,
            max_payload_size: router_config.max_payload_size,
            wasm_middleware: self.wasm_middleware.clone(),
            wasm_middleware_sha256: self.wasm_middleware_sha256.clone(),
            wasm_middleware_routes: self.wasm_middleware_routes.clone(),
            log_dir: router_config.log_dir.clone(),
            log_level: router_config.log_level.clone(),
            service_discovery_config,
            prometheus_config,
            request_timeout_secs: router_config.request_timeout_secs,
            pool_idle_timeout_secs: self.pool_idle_timeout_secs,
            request_id_headers: router_config.request_id_headers.clone(),
            router_config,
            trace_config: if self.enable_trace {
                Some(TraceConfig {
                    otlp_traces_endpoint: self.otlp_traces_endpoint.clone(),
                    sampling_ratio: self.otel_sampling_ratio,
                    excluded_paths: if self.otel_excluded_paths.is_empty() {
                        TraceConfig::default_excluded_paths()
                    } else {
                        self.otel_excluded_paths.clone()
                    },
                })
            } else {
                None
            },
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    println!("DEBUG: Main function started");

    // Parse prefill arguments manually before clap parsing
    println!("DEBUG: Parsing prefill arguments");
    let (prefill_urls, filtered_args) = split_prefill_args_from_others(std::env::args())?;
    println!("DEBUG: Prefill URLs parsed: {:?}", prefill_urls);
    // Parse CLI arguments with clap using filtered args
    println!("DEBUG: Parsing CLI arguments with clap");
    println!("DEBUG: Filtered args: {:?}", filtered_args);
    let matches = CliArgs::command().get_matches_from(&filtered_args);
    let cli_args = CliArgs::from_arg_matches(&matches)?;
    println!("DEBUG: CLI args parsed successfully");

    // Resolve and validate RouterConfig (from --config file + explicit CLI overrides, or CLI alone)
    println!("DEBUG: Resolving RouterConfig");
    let router_config = cli_args.resolve_router_config(&matches, prefill_urls)?;
    println!("DEBUG: Validating configuration");
    router_config.validate()?;
    println!("DEBUG: Configuration validated successfully");

    // Print startup info from the resolved configuration
    println!("VLLM Router starting...");
    println!("Host: {}:{}", router_config.host, router_config.port);
    let mode_str = if router_config.enable_igw {
        "IGW (Inference Gateway)".to_string()
    } else if matches!(router_config.mode, RoutingMode::OpenAI { .. }) {
        "OpenAI Backend".to_string()
    } else if router_config.mode.is_pd_mode() {
        "vLLM PD Disaggregated".to_string()
    } else if cli_args.config.is_some() {
        "Regular".to_string()
    } else {
        format!("Regular ({})", cli_args.backend)
    };
    println!("Mode: {}", mode_str);

    // Warn for runtimes that are parsed but not yet implemented
    match cli_args.backend {
        Backend::Trtllm | Backend::Anthropic => {
            println!(
                "WARNING: runtime '{}' not implemented yet; falling back to regular routing. \
Provide --worker-urls or PD flags as usual.",
                cli_args.backend
            );
        }
        Backend::Vllm | Backend::Openai => {}
    }

    if !router_config.enable_igw {
        println!("Policy: {}", router_config.policy.name());

        if let RoutingMode::VllmPrefillDecode {
            prefill_urls,
            decode_urls,
            ..
        } = &router_config.mode
        {
            if !prefill_urls.is_empty() {
                println!("Prefill nodes: {:?}", prefill_urls);
                println!("Decode nodes: {:?}", decode_urls);
            }
        }
    }

    // Create ServerConfig
    println!("DEBUG: Creating ServerConfig");
    let server_config = cli_args.to_server_config(router_config);
    println!(
        "DEBUG: ServerConfig created successfully - host: {}, port: {}",
        server_config.host, server_config.port
    );

    // Create a new runtime for the server (like Python binding does)
    println!("DEBUG: Creating Tokio runtime");
    let runtime = tokio::runtime::Runtime::new()?;
    println!("DEBUG: Tokio runtime created successfully");

    // Block on the async startup function
    println!("DEBUG: Starting server startup function");
    runtime.block_on(async move {
        let result = server::startup(server_config).await;
        // Shut down OTel while the Tokio runtime is still alive so the
        // BatchSpanProcessor can flush its final batch.
        if vllm_router_rs::otel_trace::is_otel_enabled() {
            vllm_router_rs::otel_trace::shutdown_otel();
        }
        result
    })?;

    Ok(())
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn documented_kv_layout_examples_pass_cli_validation() {
        let guide = include_str!("../docs/load_balancing/kv_aware.md").replace("\r\n", "\n");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("layout.json");
        let worker = "http://127.0.0.1:8000";
        let example = guide
            .split("\n```json\n")
            .skip(1)
            .map(|block| block.split_once("\n```").expect("unclosed JSON example").0)
            .find(|example| {
                serde_json::from_str::<serde_json::Value>(example)
                    .expect("invalid JSON example")
                    .get(worker)
                    .is_some_and(|layout| layout.get("ranks").is_some())
            })
            .expect("missing worker layout example");
        std::fs::write(&path, example).unwrap();
        for pd in [false, true] {
            let mut cli = vec![
                "vllm-router",
                "--policy",
                "kv_aware",
                "--kv-hybrid-layout",
                path.to_str().unwrap(),
            ];
            let prefill = if pd {
                cli.extend([
                    "--vllm-pd-disaggregation",
                    "--decode",
                    "http://127.0.0.1:8001",
                ]);
                vec![(worker.to_owned(), None)]
            } else {
                cli.extend(["--worker-urls", worker]);
                Vec::new()
            };
            let config = CliArgs::try_parse_from(cli)
                .unwrap()
                .to_router_config(prefill)
                .unwrap();
            config.validate().unwrap_or_else(|error| {
                panic!("invalid worker layout example: {error}\n{example}")
            });
            assert!(config.pd_worker_pairs.is_empty());
        }
        let args = CliArgs::try_parse_from([
            "vllm-router",
            "--vllm-pd-disaggregation",
            "--prefill-policy",
            "kv_aware",
            "--decode",
            "http://127.0.0.1:8001",
            "--pd-pair",
            "http://127.0.0.1:8000=http://127.0.0.1:8001",
        ])
        .unwrap();
        let config = args
            .to_router_config(vec![(worker.to_owned(), None)])
            .unwrap();
        config.validate().unwrap();
        assert_eq!(config.pd_worker_pairs[worker], ["http://127.0.0.1:8001"]);
        assert!(config.pd_kv_event_sources.is_empty());
    }

    #[test]
    fn static_event_options_admit_plain_attention_without_layout() {
        let worker = "http://127.0.0.1:8000";
        for pd in [false, true] {
            let mut cli = vec![
                "vllm-router",
                "--policy",
                "kv_aware",
                "--kv-event-topic",
                "http://127.0.0.1:8000=boot-1",
                "--kv-event-replay-source",
                "http://127.0.0.1:8000=tcp://127.0.0.1:5558",
                "--kv-event-model",
                "qwen3",
            ];
            let prefill = if pd {
                cli.extend([
                    "--vllm-pd-disaggregation",
                    "--decode",
                    "http://127.0.0.1:8001",
                    "--pd-kv-events",
                ]);
                vec![(worker.to_owned(), None)]
            } else {
                cli.extend(["--worker-urls", worker, "--kv-event-source"]);
                Vec::new()
            };
            cli.push("http://127.0.0.1:8000=tcp://127.0.0.1:5557");
            let config = CliArgs::try_parse_from(cli)
                .unwrap()
                .to_router_config(prefill)
                .unwrap();
            config.validate().unwrap();
            let layout = if pd {
                config.pd_kv_event_sources[worker].hybrid.as_ref().unwrap()
            } else {
                let PolicyConfig::KvAware {
                    static_kv_event_sources,
                    ..
                } = &config.policy
                else {
                    unreachable!()
                };
                static_kv_event_sources[worker].hybrid.as_ref().unwrap()
            };
            assert_eq!(layout.ranks[0].publisher_topic.as_deref(), Some("boot-1"));
            assert_eq!(layout.ranks[0].groups[0].block_size, 16);
        }
    }

    #[test]
    fn parses_pd_kv_aware_with_affinity_ttl() {
        let args = CliArgs::try_parse_from([
            "vllm-router",
            "--prefill-policy",
            "kv_aware",
            "--decode-policy",
            "kv_aware",
            "--kv-aware-session-affinity-ttl-secs",
            "1800",
        ])
        .unwrap();
        assert_eq!(args.prefill_policy.as_deref(), Some("kv_aware"));
        assert_eq!(args.decode_policy.as_deref(), Some("kv_aware"));
        assert_eq!(args.kv_aware_session_affinity_ttl_secs, Some(1800));
    }

    #[test]
    fn parses_wasm_middleware_options() {
        let args = CliArgs::try_parse_from([
            "vllm-router",
            "--wasm-middleware",
            "/tmp/example.component.wasm",
            "--wasm-middleware-sha256",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "--wasm-middleware-route",
            "/v1/chat/completions",
            "--wasm-middleware-route",
            "/v1/completions",
        ])
        .unwrap();
        assert_eq!(
            args.wasm_middleware.as_deref(),
            Some("/tmp/example.component.wasm")
        );
        assert_eq!(
            args.wasm_middleware_routes,
            vec!["/v1/chat/completions", "/v1/completions"]
        );
    }

    #[test]
    fn splits_prefill_args_and_preserves_other_args() {
        let args = [
            "vllm-router",
            "--port",
            "3000",
            "--prefill",
            "http://prefill-0:8000",
            "9000",
            "--policy",
            "random",
            "--prefill",
            "http://prefill-1:8000",
            "none",
            "--prefill",
            "http://prefill-2:8000",
            "-h",
        ]
        .map(String::from);

        let (prefill, other) = split_prefill_args_from_others(args).unwrap();

        assert_eq!(
            prefill,
            vec![
                ("http://prefill-0:8000".to_string(), Some(9000)),
                ("http://prefill-1:8000".to_string(), None),
                ("http://prefill-2:8000".to_string(), None),
            ]
        );
        assert_eq!(
            other,
            ["vllm-router", "--port", "3000", "--policy", "random", "-h"]
        );
    }

    #[test]
    fn rejects_missing_or_flag_prefill_urls() {
        let missing_url = ["vllm-router", "--prefill"].map(String::from);
        assert_eq!(
            split_prefill_args_from_others(missing_url).unwrap_err(),
            "--prefill requires a URL immediately after"
        );

        let flag_as_url = ["vllm-router", "--prefill", "--help"].map(String::from);
        assert_eq!(
            split_prefill_args_from_others(flag_as_url).unwrap_err(),
            "Invalid url --help"
        );
    }

    #[test]
    fn preserves_out_of_range_prefill_port_for_clap() {
        let args = ["vllm-router", "--prefill", "http://prefill:8000", "65536"].map(String::from);

        let (prefill, other) = split_prefill_args_from_others(args).unwrap();

        assert_eq!(prefill, vec![("http://prefill:8000".to_string(), None)]);
        assert_eq!(other, ["vllm-router", "65536"]);
    }

    #[test]
    fn parses_pd_kv_events_specs() {
        let specs = [
            "http://prefill:8000=tcp://prefill:5557".to_string(),
            "http://prefill2:8000=tcp://prefill2:5557, tcp://prefill2:5559".to_string(),
        ]
        .to_vec();
        let sources = CliArgs::parse_pd_kv_events(&specs).unwrap();
        assert_eq!(sources.len(), 2);
        assert_eq!(
            sources["http://prefill:8000"].endpoints,
            vec!["tcp://prefill:5557".to_string()]
        );
        assert_eq!(sources["http://prefill:8000"].block_size, 16);
        // The space after ',' is intentional: exercises endpoint trimming.
        assert_eq!(
            sources["http://prefill2:8000"].endpoints,
            vec![
                "tcp://prefill2:5557".to_string(),
                "tcp://prefill2:5559".to_string()
            ]
        );

        for bad in ["no-endpoint", "http://p:8000=", "=tcp://p:5557"] {
            assert!(
                CliArgs::parse_pd_kv_events(&[bad.to_string()]).is_err(),
                "should reject '{bad}'"
            );
        }
    }

    #[test]
    fn loads_config_file_without_clobbering_with_cli_defaults() {
        use std::io::Write;
        let mut file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        writeln!(
            file,
            r#"
mode:
  type: regular
  worker_urls:
    - http://127.0.0.1:8000
policy:
  type: round_robin
host: 0.0.0.0
port: 8080
log_level: warn
metrics:
  port: 29100
  host: 0.0.0.0
retry:
  max_retries: 2
"#
        )
        .unwrap();

        let matches = CliArgs::command()
            .try_get_matches_from(["vllm-router", "--config", file.path().to_str().unwrap()])
            .unwrap();
        let cli_args = CliArgs::from_arg_matches(&matches).unwrap();
        let router_config = cli_args.resolve_router_config(&matches, vec![]).unwrap();
        router_config.validate().unwrap();
        let server_config = cli_args.to_server_config(router_config);

        assert_eq!(server_config.host, "0.0.0.0");
        assert_eq!(server_config.port, 8080);
        assert_eq!(server_config.log_level.as_deref(), Some("warn"));
        assert_eq!(
            server_config.prometheus_config.as_ref().unwrap().port,
            29100
        );
        assert_eq!(
            server_config.prometheus_config.as_ref().unwrap().host,
            "0.0.0.0"
        );
        assert_eq!(server_config.router_config.retry.max_retries, 2);
        assert!(matches!(
            server_config.router_config.policy,
            PolicyConfig::RoundRobin
        ));
    }

    #[test]
    fn explicit_cli_flags_override_config_file_selectively() {
        use std::io::Write;
        let mut file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        writeln!(
            file,
            r#"
mode:
  type: regular
  worker_urls:
    - http://127.0.0.1:8000
policy:
  type: cache_aware
  cache_threshold: 0.45
host: 0.0.0.0
port: 8080
retry:
  max_retries: 9
  initial_backoff_ms: 120
"#
        )
        .unwrap();

        let (prefill_urls, filtered) = split_prefill_args_from_others([
            "vllm-router".to_string(),
            "--config".to_string(),
            file.path().to_str().unwrap().to_string(),
            "--port".to_string(),
            "9090".to_string(),
            "--eviction-interval".to_string(),
            "45".to_string(),
            "--retry-max-retries".to_string(),
            "1".to_string(),
            "--prometheus-port".to_string(),
            "29500".to_string(),
        ])
        .unwrap();

        let matches = CliArgs::command().try_get_matches_from(&filtered).unwrap();
        let cli_args = CliArgs::from_arg_matches(&matches).unwrap();
        let cfg = cli_args
            .resolve_router_config(&matches, prefill_urls)
            .unwrap();
        cfg.validate().unwrap();

        assert_eq!(cfg.host, "0.0.0.0");
        assert_eq!(cfg.port, 9090);
        assert_eq!(cfg.retry.max_retries, 1);
        assert_eq!(cfg.retry.initial_backoff_ms, 120);
        assert_eq!(cfg.metrics.as_ref().unwrap().port, 29500);
        match cfg.policy {
            PolicyConfig::CacheAware {
                cache_threshold,
                eviction_interval_secs,
                ..
            } => {
                assert!((cache_threshold - 0.45).abs() < f32::EPSILON);
                assert_eq!(eviction_interval_secs, 45);
            }
            other => panic!("expected cache_aware, got {other:?}"),
        }
    }

    #[test]
    fn cli_pd_overrides_preserve_unspecified_config_fields() {
        use std::io::Write;
        let mut file = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        writeln!(
            file,
            r#"{{
  "mode": {{
    "type": "vllm_prefill_decode",
    "prefill_urls": [["http://127.0.0.1:8100", 9000]],
    "decode_urls": ["http://127.0.0.1:8200"],
    "discovery_address": "127.0.0.1:30001"
  }},
  "policy": {{ "type": "power_of_two" }},
  "port": 4000
}}"#
        )
        .unwrap();

        let (prefill_urls, filtered) = split_prefill_args_from_others([
            "vllm-router".to_string(),
            "--config".to_string(),
            file.path().to_str().unwrap().to_string(),
            "--decode".to_string(),
            "http://127.0.0.1:8300".to_string(),
        ])
        .unwrap();

        let matches = CliArgs::command().try_get_matches_from(&filtered).unwrap();
        let cli_args = CliArgs::from_arg_matches(&matches).unwrap();
        let cfg = cli_args
            .resolve_router_config(&matches, prefill_urls)
            .unwrap();
        cfg.validate().unwrap();

        assert_eq!(cfg.port, 4000);
        match cfg.mode {
            RoutingMode::VllmPrefillDecode {
                prefill_urls,
                decode_urls,
                discovery_address,
                ..
            } => {
                assert_eq!(
                    prefill_urls,
                    vec![("http://127.0.0.1:8100".to_string(), Some(9000))]
                );
                assert_eq!(decode_urls, vec!["http://127.0.0.1:8300".to_string()]);
                assert_eq!(discovery_address.as_deref(), Some("127.0.0.1:30001"));
            }
            other => panic!("expected pd mode, got {other:?}"),
        }
    }

    #[test]
    fn missing_config_file_returns_error() {
        let matches = CliArgs::command()
            .try_get_matches_from(["vllm-router", "--config", "/nonexistent/router.yaml"])
            .unwrap();
        let cli_args = CliArgs::from_arg_matches(&matches).unwrap();
        assert!(cli_args.resolve_router_config(&matches, vec![]).is_err());
    }

    #[test]
    fn cli_policy_kv_aware_and_kv_event_source_override_non_kv_config_file() {
        use std::io::Write;
        let mut file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        writeln!(
            file,
            r#"
mode:
  type: regular
  worker_urls:
    - http://127.0.0.1:8000
policy:
  type: round_robin
"#
        )
        .unwrap();

        let matches = CliArgs::command()
            .try_get_matches_from([
                "vllm-router",
                "--config",
                file.path().to_str().unwrap(),
                "--policy",
                "kv_aware",
                "--kv-event-source",
                "http://127.0.0.1:8000=tcp://127.0.0.1:5557",
                "--kv-event-model",
                "qwen3",
            ])
            .unwrap();
        let cli_args = CliArgs::from_arg_matches(&matches).unwrap();
        let cfg = cli_args.resolve_router_config(&matches, vec![]).unwrap();
        cfg.validate().unwrap();

        match cfg.policy {
            PolicyConfig::KvAware {
                static_kv_event_sources,
                ..
            } => {
                assert!(
                    static_kv_event_sources.contains_key("http://127.0.0.1:8000"),
                    "expected static_kv_event_sources to retain CLI --kv-event-source"
                );
            }
            other => panic!("expected kv_aware, got {other:?}"),
        }

        // Switching a kv_aware config file away to round_robin with --kv-event-source must error.
        let mut kv_file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        writeln!(
            kv_file,
            r#"
mode:
  type: regular
  worker_urls:
    - http://127.0.0.1:8000
policy:
  type: kv_aware
"#
        )
        .unwrap();
        let conflict_matches = CliArgs::command()
            .try_get_matches_from([
                "vllm-router",
                "--config",
                kv_file.path().to_str().unwrap(),
                "--policy",
                "round_robin",
                "--kv-event-source",
                "http://127.0.0.1:8000=tcp://127.0.0.1:5557",
                "--kv-event-model",
                "qwen3",
            ])
            .unwrap();
        let conflict_args = CliArgs::from_arg_matches(&conflict_matches).unwrap();
        assert!(conflict_args
            .resolve_router_config(&conflict_matches, vec![])
            .is_err());
    }

    #[test]
    fn cli_non_openai_backend_switches_openai_config_file_to_regular() {
        use std::io::Write;
        let mut file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        writeln!(
            file,
            r#"
mode:
  type: openai
  worker_urls:
    - http://127.0.0.1:8000
policy:
  type: round_robin
"#
        )
        .unwrap();

        for backend in ["vllm", "trtllm"] {
            let matches = CliArgs::command()
                .try_get_matches_from([
                    "vllm-router",
                    "--config",
                    file.path().to_str().unwrap(),
                    "--backend",
                    backend,
                ])
                .unwrap();
            let cli_args = CliArgs::from_arg_matches(&matches).unwrap();
            let cfg = cli_args.resolve_router_config(&matches, vec![]).unwrap();
            cfg.validate().unwrap();

            match cfg.mode {
                RoutingMode::Regular { worker_urls } => {
                    assert_eq!(worker_urls, vec!["http://127.0.0.1:8000".to_string()]);
                }
                other => panic!("expected regular mode for --backend {backend}, got {other:?}"),
            }
        }
    }

    #[test]
    fn cli_same_policy_override_preserves_config_file_tuning_and_kv_sources() {
        use std::io::Write;
        let mut file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        writeln!(
            file,
            r#"
mode:
  type: regular
  worker_urls:
    - http://127.0.0.1:8000
policy:
  type: kv_aware
  router_temperature: 0.25
  kv_penalty_host: 0.7
  static_kv_event_sources:
    "http://127.0.0.1:8000":
      endpoints:
        - tcp://127.0.0.1:5557
      block_size: 16
      model: qwen3
"#
        )
        .unwrap();

        let matches = CliArgs::command()
            .try_get_matches_from([
                "vllm-router",
                "--config",
                file.path().to_str().unwrap(),
                "--policy",
                "kv_aware",
            ])
            .unwrap();
        let cli_args = CliArgs::from_arg_matches(&matches).unwrap();
        let cfg = cli_args.resolve_router_config(&matches, vec![]).unwrap();
        cfg.validate().unwrap();

        match cfg.policy {
            PolicyConfig::KvAware {
                router_temperature,
                kv_penalty_host,
                static_kv_event_sources,
                ..
            } => {
                assert!((router_temperature - 0.25).abs() < f64::EPSILON);
                assert!((kv_penalty_host - 0.7).abs() < f64::EPSILON);
                assert!(static_kv_event_sources.contains_key("http://127.0.0.1:8000"));
            }
            other => panic!("expected kv_aware, got {other:?}"),
        }

        let tuned_matches = CliArgs::command()
            .try_get_matches_from([
                "vllm-router",
                "--config",
                file.path().to_str().unwrap(),
                "--policy",
                "kv_aware",
                "--router-temperature",
                "0.5",
            ])
            .unwrap();
        let tuned_args = CliArgs::from_arg_matches(&tuned_matches).unwrap();
        let tuned_cfg = tuned_args
            .resolve_router_config(&tuned_matches, vec![])
            .unwrap();
        tuned_cfg.validate().unwrap();

        match tuned_cfg.policy {
            PolicyConfig::KvAware {
                router_temperature,
                kv_penalty_host,
                static_kv_event_sources,
                ..
            } => {
                assert!((router_temperature - 0.5).abs() < f64::EPSILON);
                assert!((kv_penalty_host - 0.7).abs() < f64::EPSILON);
                assert!(static_kv_event_sources.contains_key("http://127.0.0.1:8000"));
            }
            other => panic!("expected kv_aware, got {other:?}"),
        }
    }

    #[test]
    fn cli_same_role_policy_override_preserves_config_file_settings() {
        use std::io::Write;
        let mut file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        writeln!(
            file,
            r#"
mode:
  type: vllm_prefill_decode
  prefill_urls:
    - ["http://127.0.0.1:8100", 9000]
  decode_urls:
    - http://127.0.0.1:8200
  prefill_policy:
    type: cache_aware
    cache_threshold: 0.72
    balance_abs_threshold: 12
    balance_rel_threshold: 2.0
    eviction_interval_secs: 45
    max_tree_size: 4096
  decode_policy:
    type: kv_aware
    router_temperature: 0.25
    kv_penalty_host: 0.7
    kv_penalty_external: 0.5
    kv_penalty_disk: 0.2
policy:
  type: round_robin
"#
        )
        .unwrap();

        let matches = CliArgs::command()
            .try_get_matches_from([
                "vllm-router",
                "--config",
                file.path().to_str().unwrap(),
                "--prefill-policy",
                "cache_aware",
                "--decode-policy",
                "kv_aware",
            ])
            .unwrap();
        let cli_args = CliArgs::from_arg_matches(&matches).unwrap();
        let cfg = cli_args.resolve_router_config(&matches, vec![]).unwrap();
        cfg.validate().unwrap();

        match &cfg.mode {
            RoutingMode::VllmPrefillDecode {
                prefill_policy:
                    Some(PolicyConfig::CacheAware {
                        cache_threshold,
                        balance_abs_threshold,
                        balance_rel_threshold,
                        eviction_interval_secs,
                        max_tree_size,
                    }),
                decode_policy:
                    Some(PolicyConfig::KvAware {
                        router_temperature,
                        kv_penalty_host,
                        kv_penalty_external,
                        kv_penalty_disk,
                        ..
                    }),
                ..
            } => {
                assert!((*cache_threshold - 0.72).abs() < f32::EPSILON);
                assert_eq!(*balance_abs_threshold, 12);
                assert!((*balance_rel_threshold - 2.0).abs() < f32::EPSILON);
                assert_eq!(*eviction_interval_secs, 45);
                assert_eq!(*max_tree_size, 4096);
                assert!((*router_temperature - 0.25).abs() < f64::EPSILON);
                assert!((*kv_penalty_host - 0.7).abs() < f64::EPSILON);
                assert!((*kv_penalty_external - 0.5).abs() < f64::EPSILON);
                assert!((*kv_penalty_disk - 0.2).abs() < f64::EPSILON);
            }
            other => panic!("expected pd mode with preserved role policies, got {other:?}"),
        }

        let tuned_matches = CliArgs::command()
            .try_get_matches_from([
                "vllm-router",
                "--config",
                file.path().to_str().unwrap(),
                "--prefill-policy",
                "cache_aware",
                "--cache-threshold",
                "0.9",
                "--decode-policy",
                "kv_aware",
                "--router-temperature",
                "0.5",
            ])
            .unwrap();
        let tuned_args = CliArgs::from_arg_matches(&tuned_matches).unwrap();
        let tuned_cfg = tuned_args
            .resolve_router_config(&tuned_matches, vec![])
            .unwrap();
        tuned_cfg.validate().unwrap();

        match &tuned_cfg.mode {
            RoutingMode::VllmPrefillDecode {
                prefill_policy:
                    Some(PolicyConfig::CacheAware {
                        cache_threshold,
                        balance_abs_threshold,
                        balance_rel_threshold,
                        eviction_interval_secs,
                        max_tree_size,
                    }),
                decode_policy:
                    Some(PolicyConfig::KvAware {
                        router_temperature,
                        kv_penalty_host,
                        kv_penalty_external,
                        kv_penalty_disk,
                        ..
                    }),
                ..
            } => {
                assert!((*cache_threshold - 0.9).abs() < f32::EPSILON);
                assert_eq!(*balance_abs_threshold, 12);
                assert!((*balance_rel_threshold - 2.0).abs() < f32::EPSILON);
                assert_eq!(*eviction_interval_secs, 45);
                assert_eq!(*max_tree_size, 4096);
                assert!((*router_temperature - 0.5).abs() < f64::EPSILON);
                assert!((*kv_penalty_host - 0.7).abs() < f64::EPSILON);
                assert!((*kv_penalty_external - 0.5).abs() < f64::EPSILON);
                assert!((*kv_penalty_disk - 0.2).abs() < f64::EPSILON);
            }
            other => panic!("expected pd mode with selectively tuned role policies, got {other:?}"),
        }
    }

    #[test]
    fn cli_different_role_policy_override_replaces_config_file_role_policy() {
        use std::io::Write;
        let mut file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        writeln!(
            file,
            r#"
mode:
  type: vllm_prefill_decode
  prefill_urls:
    - ["http://127.0.0.1:8100", 9000]
  decode_urls:
    - http://127.0.0.1:8200
  prefill_policy:
    type: cache_aware
    cache_threshold: 0.72
  decode_policy:
    type: kv_aware
    router_temperature: 0.25
policy:
  type: round_robin
"#
        )
        .unwrap();

        let matches = CliArgs::command()
            .try_get_matches_from([
                "vllm-router",
                "--config",
                file.path().to_str().unwrap(),
                "--prefill-policy",
                "round_robin",
                "--decode-policy",
                "cache_aware",
            ])
            .unwrap();
        let cli_args = CliArgs::from_arg_matches(&matches).unwrap();
        let cfg = cli_args.resolve_router_config(&matches, vec![]).unwrap();
        cfg.validate().unwrap();

        match &cfg.mode {
            RoutingMode::VllmPrefillDecode {
                prefill_policy: Some(PolicyConfig::RoundRobin),
                decode_policy:
                    Some(PolicyConfig::CacheAware {
                        cache_threshold, ..
                    }),
                ..
            } => {
                assert!((*cache_threshold - 0.3).abs() < f32::EPSILON);
            }
            other => panic!("expected replaced pd role policies, got {other:?}"),
        }
    }

    #[test]
    fn cli_enable_program_scheduling_preserves_file_settings_unless_json_supplied() {
        use std::io::Write;
        let mut file = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        writeln!(
            file,
            r#"{{
  "mode": {{
    "type": "regular",
    "worker_urls": ["http://127.0.0.1:8000"]
  }},
  "policy": {{ "type": "kv_aware" }},
  "program_scheduling": {{
    "decode_buffer_tokens": 4096,
    "max_active_programs_per_target": 8
  }}
}}"#
        )
        .unwrap();

        let matches_a = CliArgs::command()
            .try_get_matches_from([
                "vllm-router",
                "--config",
                file.path().to_str().unwrap(),
                "--enable-program-scheduling",
            ])
            .unwrap();
        let args_a = CliArgs::from_arg_matches(&matches_a).unwrap();
        let cfg_a = args_a.resolve_router_config(&matches_a, vec![]).unwrap();
        cfg_a.validate().unwrap();

        let ps_a = cfg_a
            .program_scheduling
            .expect("program_scheduling should stay enabled");
        assert_eq!(ps_a.decode_buffer_tokens, 4096);
        assert_eq!(ps_a.max_active_programs_per_target, 8);

        let matches_b = CliArgs::command()
            .try_get_matches_from([
                "vllm-router",
                "--config",
                file.path().to_str().unwrap(),
                "--enable-program-scheduling",
                "--program-scheduling-config-json",
                r#"{"decode_buffer_tokens": 2048, "max_active_programs_per_target": 3}"#,
            ])
            .unwrap();
        let args_b = CliArgs::from_arg_matches(&matches_b).unwrap();
        let cfg_b = args_b.resolve_router_config(&matches_b, vec![]).unwrap();
        cfg_b.validate().unwrap();

        let ps_b = cfg_b
            .program_scheduling
            .expect("program_scheduling should be enabled");
        assert_eq!(ps_b.decode_buffer_tokens, 2048);
        assert_eq!(ps_b.max_active_programs_per_target, 3);
    }
}
