//! Backend metrics transport and raw per-DP-rank observation conversion.
//!
//! This module performs HTTP and Prometheus parsing only. Progress-TTL owns
//! freshness, rolling statistics, logical capacity, and scheduling decisions.

use super::ProgramTarget;
use crate::protocols::prometheus_text::{parse_series, split_series_value};
use async_trait::async_trait;
use futures::future::join_all;
use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

/// Raw backend facts for one concrete Program scheduling target.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendObservation {
    /// Stable target identifier including a DP suffix when applicable.
    pub target_id: String,
    /// Base URL from which the shared metrics response was collected.
    pub base_url: String,
    /// Internal data-parallel engine index selected from the response.
    pub dp_rank: Option<usize>,
    /// Fraction of the local KV cache currently used by native vLLM.
    pub kv_cache_usage: Option<f64>,
    /// Backend reference token capacity from cache_config_info.
    pub token_capacity: Option<usize>,
    /// Physical block-pool size, including the reserved null block.
    pub num_gpu_blocks: Option<usize>,
    /// Native requests currently executing on the engine.
    pub running_requests: Option<usize>,
    /// Native requests waiting inside the engine.
    pub waiting_requests: Option<usize>,
    /// Local monotonic time at which the response was converted.
    pub observed_at: Instant,
}

/// One failed backend scrape retained for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendObservationFailure {
    /// Base URL whose metrics endpoint failed.
    pub base_url: String,
    /// Stable failure description suitable for a warning log.
    pub error: String,
}

/// Transport-neutral source of backend scheduling observations.
#[async_trait]
pub trait BackendObservationProvider: Send + Sync + std::fmt::Debug {
    /// Collect observations for the current immutable target snapshot.
    async fn observe(
        &self,
        targets: &[ProgramTarget],
    ) -> (Vec<BackendObservation>, Vec<BackendObservationFailure>);
}

/// Observation provider for vLLM's existing `/metrics` endpoint.
#[derive(Debug, Clone)]
pub struct VllmMetricsObservationProvider {
    client: reqwest::Client,
    api_key: Option<String>,
    request_timeout: Duration,
    /// Metrics endpoint override keyed by Program target base URL.
    ///
    /// A gRPC target's base URL is not an HTTP address, so deployments must map
    /// it to the HTTP `/metrics` endpoint of the same Engine/DP rank. Without a
    /// mapping the target keeps reporting a missing observation; a stale sample
    /// is never reused as live capacity.
    metrics_base_urls: BTreeMap<String, String>,
}

impl VllmMetricsObservationProvider {
    /// Build a provider using the Router's shared HTTP client and API key.
    pub fn new(client: reqwest::Client, api_key: Option<String>) -> Self {
        Self {
            client,
            api_key,
            request_timeout: Duration::from_secs(2),
            metrics_base_urls: BTreeMap::new(),
        }
    }

    /// Map target base URLs (for example `grpc://host:50051`) to the HTTP
    /// metrics endpoint of the same Engine.
    pub fn with_metrics_base_urls(mut self, metrics_base_urls: BTreeMap<String, String>) -> Self {
        self.metrics_base_urls = metrics_base_urls;
        self
    }

    /// Resolve the HTTP metrics address for one target base URL.
    fn metrics_base_url(&self, base_url: &str) -> Result<String, String> {
        if let Some(mapped) = self.metrics_base_urls.get(base_url) {
            return if mapped.starts_with("http://") || mapped.starts_with("https://") {
                Ok(mapped.trim_end_matches('/').to_string())
            } else {
                Err(format!(
                    "configured metrics URL for {base_url} is not HTTP: {mapped}"
                ))
            };
        }
        if base_url.starts_with("grpc://") || base_url.starts_with("grpcs://") {
            return Err(format!(
                "no HTTP metrics URL configured for gRPC target {base_url}"
            ));
        }
        Ok(base_url.trim_end_matches('/').to_string())
    }

    /// Override the per-scrape timeout, primarily for controlled deployments.
    pub fn with_request_timeout(mut self, request_timeout: Duration) -> Self {
        self.request_timeout = request_timeout;
        self
    }
}

#[async_trait]
impl BackendObservationProvider for VllmMetricsObservationProvider {
    async fn observe(
        &self,
        targets: &[ProgramTarget],
    ) -> (Vec<BackendObservation>, Vec<BackendObservationFailure>) {
        let mut targets_by_metrics_url = BTreeMap::<String, Vec<ProgramTarget>>::new();
        let mut unresolved = Vec::new();
        for target in targets {
            match self.metrics_base_url(&target.base_url) {
                Ok(url) => targets_by_metrics_url
                    .entry(url)
                    .or_default()
                    .push(target.clone()),
                Err(error) => unresolved.push(BackendObservationFailure {
                    base_url: target.base_url.clone(),
                    error,
                }),
            }
        }
        let scrapes = targets_by_metrics_url
            .into_iter()
            .map(|(base_url, targets)| {
                let client = self.client.clone();
                let api_key = self.api_key.clone();
                let timeout = self.request_timeout;
                async move {
                    let url = format!("{base_url}/metrics");
                    let mut request = client.get(&url).timeout(timeout);
                    if let Some(api_key) = api_key {
                        request = request.bearer_auth(api_key);
                    }
                    let result = match request.send().await {
                        Ok(response) if response.status().is_success() => response
                            .text()
                            .await
                            .map_err(|error| format!("failed to read response: {error}")),
                        Ok(response) => Err(format!("HTTP {}", response.status())),
                        Err(error) => Err(error.to_string()),
                    };
                    (base_url, targets, result)
                }
            });

        let mut observations = Vec::new();
        let mut failures = unresolved;
        for (base_url, targets, result) in join_all(scrapes).await {
            match result {
                Ok(text) => observations.extend(observations_from_metrics(&targets, &text)),
                Err(error) => failures.push(BackendObservationFailure { base_url, error }),
            }
        }
        (observations, failures)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
struct VllmEngineMetrics {
    kv_cache_usage: Option<f64>,
    token_capacity: Option<usize>,
    num_gpu_blocks: Option<usize>,
    running_requests: Option<usize>,
    waiting_requests: Option<usize>,
}

fn observations_from_metrics(targets: &[ProgramTarget], text: &str) -> Vec<BackendObservation> {
    let parsed = parse_vllm_metrics(text);
    let observed_at = Instant::now();
    targets
        .iter()
        .filter_map(|target| {
            let engine = target.dp_rank.unwrap_or(0);
            let metrics = parsed.get(&engine)?;
            Some(BackendObservation {
                target_id: target.id.clone(),
                base_url: target.base_url.clone(),
                dp_rank: target.dp_rank,
                kv_cache_usage: metrics.kv_cache_usage,
                token_capacity: metrics.token_capacity,
                num_gpu_blocks: metrics.num_gpu_blocks,
                running_requests: metrics.running_requests,
                waiting_requests: metrics.waiting_requests,
                observed_at,
            })
        })
        .collect()
}

/// Parse only the stable vLLM metric families consumed by Program scheduling.
fn parse_vllm_metrics(text: &str) -> HashMap<usize, VllmEngineMetrics> {
    let mut engines: HashMap<usize, VllmEngineMetrics> = HashMap::new();
    let mut seen = std::collections::HashSet::new();
    let mut ambiguous = std::collections::HashSet::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((series, value)) = split_series_value(line) else {
            continue;
        };
        let (name, labels) = parse_series(series);
        if !matches!(
            name,
            "vllm:kv_cache_usage_perc"
                | "vllm:num_requests_running"
                | "vllm:num_requests_waiting"
                | "vllm:cache_config_info"
        ) {
            continue;
        }
        let engine = match labels.get("engine") {
            Some(raw) => match raw.parse::<usize>() {
                Ok(rank) => rank,
                Err(_) => continue,
            },
            None => 0,
        };
        if !seen.insert((engine, name.to_string())) {
            ambiguous.insert(engine);
        }
        let entry = engines.entry(engine).or_default();
        match name {
            "vllm:kv_cache_usage_perc" => {
                entry.kv_cache_usage = value
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
            }
            "vllm:num_requests_running" => entry.running_requests = parse_count(value),
            "vllm:num_requests_waiting" => entry.waiting_requests = parse_count(value),
            "vllm:cache_config_info" if value.parse::<f64>().ok() == Some(1.0) => {
                entry.token_capacity = labels
                    .get("kv_cache_size_tokens")
                    .and_then(|v| parse_count(v))
                    .filter(|v| *v > 0);
                entry.num_gpu_blocks = labels
                    .get("num_gpu_blocks")
                    .and_then(|v| parse_count(v))
                    .filter(|v| *v > 1);
            }
            _ => {}
        }
    }
    // Multiple models/replicas cannot be silently combined into one rank budget.
    for engine in ambiguous {
        if let Some(entry) = engines.get_mut(&engine) {
            entry.token_capacity = None;
            entry.num_gpu_blocks = None;
        }
    }
    engines
}

fn parse_count(value: &str) -> Option<usize> {
    value
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && *v >= 0.0 && v.fract() == 0.0 && *v < usize::MAX as f64)
        .map(|v| v as usize)
}

/// Maximum value of one Prometheus gauge family across all series.
pub(crate) fn max_prometheus_gauge(text: &str, family: &str) -> Option<f64> {
    let mut maximum: Option<f64> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((series, value)) = split_series_value(line) else {
            continue;
        };
        let (name, _labels) = parse_series(series);
        if name != family {
            continue;
        }
        if let Ok(value) = value.parse::<f64>() {
            maximum = Some(maximum.map_or(value, |current| current.max(value)));
        }
    }
    maximum
}

/// Sum one gauge family across its series (per-engine request counts).
pub(crate) fn sum_prometheus_gauge(text: &str, family: &str) -> Option<u64> {
    let mut total: Option<u64> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((series, value)) = split_series_value(line) else {
            continue;
        };
        let (name, _labels) = parse_series(series);
        if name != family {
            continue;
        }
        if let Ok(value) = value.parse::<f64>() {
            let value = value.max(0.0).round() as u64;
            total = Some(total.unwrap_or(0).saturating_add(value));
        }
    }
    total
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    const METRICS: &str = r#"
# HELP vllm:kv_cache_usage_perc GPU KV cache usage. 1 means 100 percent usage.
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
vllm:num_requests_running{engine="0",model_name="m"} 2
vllm:num_requests_waiting{engine="0",model_name="m"} 1
vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.75
vllm:num_requests_running{engine="1",model_name="m"} 4.0
vllm:num_requests_waiting{engine="1",model_name="m"} 3.0
unrelated_metric{engine="1",label="escaped\"comma,value"} 99
"#;

    #[test]
    fn parses_only_required_series_per_engine() {
        let parsed = parse_vllm_metrics(METRICS);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[&0].kv_cache_usage, Some(0.25));
        assert_eq!(parsed[&0].running_requests, Some(2));
        assert_eq!(parsed[&0].waiting_requests, Some(1));
        assert_eq!(parsed[&1].kv_cache_usage, Some(0.75));
        assert_eq!(parsed[&1].running_requests, Some(4));
        assert_eq!(parsed[&1].waiting_requests, Some(3));
    }

    #[test]
    fn maps_internal_dp_rank_to_target() {
        let targets = vec![
            ProgramTarget {
                id: "worker@0".into(),
                base_url: "http://worker".into(),
                dp_rank: Some(0),
            },
            ProgramTarget {
                id: "worker@1".into(),
                base_url: "http://worker".into(),
                dp_rank: Some(1),
            },
        ];
        let observations = observations_from_metrics(&targets, METRICS);
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].target_id, "worker@0");
        assert_eq!(observations[0].kv_cache_usage, Some(0.25));
        assert_eq!(observations[1].target_id, "worker@1");
        assert_eq!(observations[1].waiting_requests, Some(3));
        assert_eq!(observations[0].observed_at, observations[1].observed_at);
    }

    #[test]
    fn max_gauge_picks_the_largest_series() {
        let text = concat!(
            "# HELP vllm:kv_cache_usage_perc usage\n",
            "vllm:kv_cache_usage_perc{engine=\"0\"} 0.25\n",
            "vllm:kv_cache_usage_perc{engine=\"1\"} 0.75\n",
        );
        assert_eq!(
            max_prometheus_gauge(text, "vllm:kv_cache_usage_perc"),
            Some(0.75)
        );
        assert_eq!(max_prometheus_gauge(text, "missing_gauge"), None);
    }

    #[test]
    fn missing_engine_series_does_not_fabricate_observation() {
        let targets = vec![ProgramTarget {
            id: "worker@2".into(),
            base_url: "http://worker".into(),
            dp_rank: Some(2),
        }];
        assert!(observations_from_metrics(&targets, METRICS).is_empty());
    }

    #[test]
    fn grpc_targets_require_an_explicit_http_metrics_url() {
        let provider = VllmMetricsObservationProvider::new(reqwest::Client::new(), None);
        assert!(provider
            .metrics_base_url("grpc://engine:50051")
            .unwrap_err()
            .contains("no HTTP metrics URL"));
        let mapped = provider.with_metrics_base_urls(BTreeMap::from([(
            "grpc://engine:50051".to_string(),
            "http://engine:8000".to_string(),
        )]));
        assert_eq!(
            mapped.metrics_base_url("grpc://engine:50051").unwrap(),
            "http://engine:8000"
        );
        let invalid = VllmMetricsObservationProvider::new(reqwest::Client::new(), None)
            .with_metrics_base_urls(BTreeMap::from([(
                "grpc://engine:1".to_string(),
                "grpc://engine:2".to_string(),
            )]));
        assert!(invalid.metrics_base_url("grpc://engine:1").is_err());
    }

    #[test]
    fn http_targets_scrape_their_own_metrics_endpoint() {
        let provider = VllmMetricsObservationProvider::new(reqwest::Client::new(), None);
        assert_eq!(
            provider.metrics_base_url("http://engine:8000/").unwrap(),
            "http://engine:8000"
        );
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod capacity_tests {
    use super::*;
    #[test]
    fn capacity_is_parsed_per_engine_with_timestamps_and_label_spaces() {
        let metrics = parse_vllm_metrics(
            r#"
vllm:cache_config_info{engine="0",model_name="a b",kv_cache_size_tokens="1000",num_gpu_blocks="100"} 1 123
vllm:cache_config_info{engine="1",kv_cache_size_tokens="2000",num_gpu_blocks="200"} 1
vllm:kv_cache_usage_perc{engine="0"} 0.25 123
vllm:kv_cache_usage_perc{engine="1"} 0.5
"#,
        );
        assert_eq!(metrics[&0].token_capacity, Some(1000));
        assert_eq!(metrics[&1].token_capacity, Some(2000));
        assert_eq!(metrics[&0].kv_cache_usage, Some(0.25));
        assert_eq!(metrics[&1].num_gpu_blocks, Some(200));
    }
    #[test]
    fn malformed_or_ambiguous_capacity_never_invents_a_budget() {
        for value in ["NaN", "inf", "-1", "1.5", "None"] {
            let parsed = parse_vllm_metrics(&format!("vllm:cache_config_info{{kv_cache_size_tokens=\"{value}\",num_gpu_blocks=\"100\"}} 1"));
            assert_eq!(parsed[&0].token_capacity, None);
        }
        for value in ["NaN", "inf", "-0.1", "1.1"] {
            let parsed = parse_vllm_metrics(&format!("vllm:kv_cache_usage_perc {value}"));
            assert_eq!(parsed[&0].kv_cache_usage, None);
        }
        let parsed = parse_vllm_metrics("vllm:cache_config_info{engine=\"0\",model_name=\"a\",kv_cache_size_tokens=\"1000\",num_gpu_blocks=\"100\"} 1\nvllm:cache_config_info{engine=\"0\",model_name=\"b\",kv_cache_size_tokens=\"2000\",num_gpu_blocks=\"100\"} 1");
        assert_eq!(parsed[&0].token_capacity, None);
        assert!(parse_vllm_metrics("vllm:kv_cache_usage_perc{engine=\"bad\"} 0.2").is_empty());
    }
}
