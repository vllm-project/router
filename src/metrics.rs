use http::StatusCode;
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
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

const DURATION_BUCKETS: [f64; 20] = [
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0, 60.0,
    90.0, 120.0, 180.0, 240.0,
];

const NO_STATUS_LABEL: &str = "none";
const HTTP_REQUEST_METHOD_LABEL: &str = "http_request_method";
const HTTP_RESPONSE_STATUS_CODE_LABEL: &str = "http_response_status_code";
const HTTP_RESPONSE_STATUS_CLASS_LABEL: &str = "http_response_status_class";

fn status_code_label(status: StatusCode) -> String {
    status.as_u16().to_string()
}

fn status_class_label(status: StatusCode) -> &'static str {
    match status.as_u16() / 100 {
        1 => "1xx",
        2 => "2xx",
        3 => "3xx",
        4 => "4xx",
        5 => "5xx",
        _ => "other",
    }
}

fn optional_status_labels(status: Option<StatusCode>) -> (String, &'static str) {
    status
        .map(|status| (status_code_label(status), status_class_label(status)))
        .unwrap_or_else(|| (NO_STATUS_LABEL.to_string(), NO_STATUS_LABEL))
}

fn http_response_status_code(status: StatusCode) -> String {
    status_code_label(status)
}

fn http_response_status_class(status: StatusCode) -> &'static str {
    status_class_label(status)
}

fn http_response_status_labels(status: StatusCode) -> (String, &'static str) {
    (
        http_response_status_code(status),
        http_response_status_class(status),
    )
}

pub fn init_metrics() {
    // Request metrics
    describe_counter!(
        "vllm_router_requests_total",
        "Total number of requests by route and method"
    );
    describe_histogram!(
        "vllm_router_request_duration_seconds",
        "Request duration in seconds by route"
    );
    // HTTP middleware metrics (separate from legacy vllm_router_requests_total)
    describe_counter!(
        "vllm_router_http_requests_total",
        "Total number of HTTP responses by route, method, and status"
    );
    describe_histogram!(
        "vllm_router_http_request_duration_seconds",
        "HTTP request duration in seconds by route, method, and status class"
    );
    describe_counter!(
        "vllm_router_request_errors_total",
        "Total number of structured request errors by route, method, error type, and status"
    );
    describe_counter!(
        "vllm_router_http_responses_total",
        "Total number of client-visible HTTP responses by route, method, and status"
    );
    describe_histogram!(
        "vllm_router_http_response_duration_seconds",
        "Client-visible HTTP response duration in seconds by route, method, and status"
    );
    describe_counter!(
        "vllm_router_backend_http_responses_total",
        "Total number of downstream backend HTTP responses by route, worker, phase, method, and status"
    );
    describe_histogram!(
        "vllm_router_backend_http_response_duration_seconds",
        "Downstream backend HTTP response duration in seconds by route, worker, phase, method, and status"
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
        "vllm_router_policy_decisions_total",
        "Total routing policy decisions by policy and worker"
    );
    describe_counter!(
        "vllm_router_cache_hits_total",
        "Cache-aware routing decisions that reused a worker above the cache threshold"
    );
    describe_counter!(
        "vllm_router_cache_misses_total",
        "Cache-aware routing decisions that fell back to low-affinity worker selection"
    );
    describe_gauge!(
        "vllm_router_tree_size",
        "Tracked cache-aware tree size per worker in characters"
    );
    describe_counter!(
        "vllm_router_load_balancing_events_total",
        "Total load balancing trigger events"
    );
    describe_gauge!("vllm_router_max_load", "Maximum worker load");
    describe_gauge!("vllm_router_min_load", "Minimum worker load");

    // PD-specific metrics
    describe_counter!(
        "vllm_router_pd_requests_total",
        "Total number of PD HTTP responses by route, method, and status"
    );
    describe_counter!(
        "vllm_router_pd_prefill_requests_total",
        "Total number of prefill-stage HTTP responses by worker and status"
    );
    describe_counter!(
        "vllm_router_pd_decode_requests_total",
        "Total number of decode-stage HTTP responses by worker and status"
    );
    describe_counter!(
        "vllm_router_pd_errors_total",
        "Total number of structured PD errors by route, method, error type, and status"
    );
    describe_counter!(
        "vllm_router_pd_prefill_errors_total",
        "Total number of structured prefill-stage errors by worker, error type, and status"
    );
    describe_counter!(
        "vllm_router_pd_decode_errors_total",
        "Total number of structured decode-stage errors by worker, error type, and status"
    );
    describe_counter!(
        "vllm_router_pd_stream_errors_total",
        "Total streaming errors per worker"
    );
    describe_histogram!(
        "vllm_router_pd_request_duration_seconds",
        "PD request duration in seconds by route, method, and status class"
    );

    // Service discovery metrics
    describe_counter!(
        "vllm_router_discovery_updates_total",
        "Total successful service discovery change events"
    );
    describe_gauge!(
        "vllm_router_discovery_workers_added",
        "Workers added in the most recent successful service discovery change"
    );
    describe_gauge!(
        "vllm_router_discovery_workers_removed",
        "Workers removed in the most recent successful service discovery change"
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
        "Embedding request errors by HTTP status"
    );
    describe_gauge!("vllm_router_embeddings_queue_size", "Embedding queue size");

    // Running requests gauge for cache-aware policy
    describe_gauge!(
        "vllm_router_running_requests",
        "Number of running requests per worker"
    );
}

pub fn start_prometheus(config: PrometheusConfig) {
    // Initialize metric descriptions
    init_metrics();

    let duration_matcher = Matcher::Suffix(String::from("duration_seconds"));
    let ip_addr: IpAddr = config
        .host
        .parse()
        .unwrap_or(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));
    let socket_addr = SocketAddr::new(ip_addr, config.port);

    PrometheusBuilder::new()
        .with_http_listener(socket_addr)
        .upkeep_timeout(Duration::from_secs(5 * 60))
        .set_buckets_for_metric(duration_matcher, &DURATION_BUCKETS)
        .expect("failed to set duration bucket")
        .install()
        .expect("failed to install Prometheus metrics exporter");
}

pub struct RouterMetrics;

impl RouterMetrics {
    // Request metrics
    pub fn observe_http_request(route: &str, method: &str, status: StatusCode, duration: Duration) {
        let status_code = status_code_label(status);
        let status_class = status_class_label(status);
        counter!("vllm_router_http_requests_total",
            "route" => route.to_string(),
            "method" => method.to_string(),
            "status_code" => status_code,
            "status_class" => status_class.to_string()
        )
        .increment(1);
        histogram!("vllm_router_http_request_duration_seconds",
            "route" => route.to_string(),
            "method" => method.to_string(),
            "status_class" => status_class.to_string()
        )
        .record(duration.as_secs_f64());
    }

    pub fn observe_http_response(
        route: &str,
        method: &str,
        status: StatusCode,
        duration: Duration,
    ) {
        let (status_code, status_class) = http_response_status_labels(status);
        counter!("vllm_router_http_responses_total",
            "route" => route.to_string(),
            HTTP_REQUEST_METHOD_LABEL => method.to_string(),
            HTTP_RESPONSE_STATUS_CODE_LABEL => status_code.clone(),
            HTTP_RESPONSE_STATUS_CLASS_LABEL => status_class.to_string()
        )
        .increment(1);
        histogram!("vllm_router_http_response_duration_seconds",
            "route" => route.to_string(),
            HTTP_REQUEST_METHOD_LABEL => method.to_string(),
            HTTP_RESPONSE_STATUS_CODE_LABEL => status_code,
            HTTP_RESPONSE_STATUS_CLASS_LABEL => status_class.to_string()
        )
        .record(duration.as_secs_f64());
    }

    pub fn observe_backend_http_response(
        route: &str,
        worker: &str,
        request_phase: &str,
        method: &str,
        status: StatusCode,
        duration: Duration,
    ) {
        let (status_code, status_class) = http_response_status_labels(status);
        counter!("vllm_router_backend_http_responses_total",
            "route" => route.to_string(),
            "worker" => worker.to_string(),
            "vllm_request_phase" => request_phase.to_string(),
            HTTP_REQUEST_METHOD_LABEL => method.to_string(),
            HTTP_RESPONSE_STATUS_CODE_LABEL => status_code.clone(),
            HTTP_RESPONSE_STATUS_CLASS_LABEL => status_class.to_string()
        )
        .increment(1);
        histogram!("vllm_router_backend_http_response_duration_seconds",
            "route" => route.to_string(),
            "worker" => worker.to_string(),
            "vllm_request_phase" => request_phase.to_string(),
            HTTP_REQUEST_METHOD_LABEL => method.to_string(),
            HTTP_RESPONSE_STATUS_CODE_LABEL => status_code,
            HTTP_RESPONSE_STATUS_CLASS_LABEL => status_class.to_string()
        )
        .record(duration.as_secs_f64());
    }

    pub fn record_request_error(route: &str, method: &str, status: StatusCode, error_type: &str) {
        let status_code = status_code_label(status);
        let status_class = status_class_label(status);
        counter!("vllm_router_request_errors_total",
            "route" => route.to_string(),
            "method" => method.to_string(),
            "error_type" => error_type.to_string(),
            "status_code" => status_code,
            "status_class" => status_class.to_string()
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

    pub fn set_tree_size(worker: &str, size: usize) {
        gauge!("vllm_router_tree_size",
            "worker" => worker.to_string()
        )
        .set(size as f64);
    }

    pub fn record_load_balancing_event() {
        counter!("vllm_router_load_balancing_events_total").increment(1);
    }

    pub fn set_load_range(max_load: usize, min_load: usize) {
        gauge!("vllm_router_max_load").set(max_load as f64);
        gauge!("vllm_router_min_load").set(min_load as f64);
    }

    // PD-specific metrics
    pub fn observe_pd_request(route: &str, method: &str, status: StatusCode, duration: Duration) {
        let status_code = status_code_label(status);
        let status_class = status_class_label(status);
        counter!("vllm_router_pd_requests_total",
            "route" => route.to_string(),
            "method" => method.to_string(),
            "status_code" => status_code,
            "status_class" => status_class.to_string()
        )
        .increment(1);
        histogram!("vllm_router_pd_request_duration_seconds",
            "route" => route.to_string(),
            "method" => method.to_string(),
            "status_class" => status_class.to_string()
        )
        .record(duration.as_secs_f64());
    }

    pub fn record_pd_prefill_request(worker: &str, status: StatusCode) {
        let status_code = status_code_label(status);
        let status_class = status_class_label(status);
        counter!("vllm_router_pd_prefill_requests_total",
            "worker" => worker.to_string(),
            "status_code" => status_code,
            "status_class" => status_class.to_string()
        )
        .increment(1);
    }

    pub fn record_pd_decode_request(worker: &str, status: StatusCode) {
        let status_code = status_code_label(status);
        let status_class = status_class_label(status);
        counter!("vllm_router_pd_decode_requests_total",
            "worker" => worker.to_string(),
            "status_code" => status_code,
            "status_class" => status_class.to_string()
        )
        .increment(1);
    }

    pub fn record_pd_error(route: &str, method: &str, status: StatusCode, error_type: &str) {
        let status_code = status_code_label(status);
        let status_class = status_class_label(status);
        counter!("vllm_router_pd_errors_total",
            "route" => route.to_string(),
            "method" => method.to_string(),
            "error_type" => error_type.to_string(),
            "status_code" => status_code,
            "status_class" => status_class.to_string()
        )
        .increment(1);
    }

    pub fn record_pd_prefill_error(worker: &str, error_type: &str, status: Option<StatusCode>) {
        let (status_code, status_class) = optional_status_labels(status);
        counter!("vllm_router_pd_prefill_errors_total",
            "worker" => worker.to_string(),
            "error_type" => error_type.to_string(),
            "status_code" => status_code,
            "status_class" => status_class.to_string()
        )
        .increment(1);
    }

    pub fn record_pd_decode_error(worker: &str, error_type: &str, status: Option<StatusCode>) {
        let (status_code, status_class) = optional_status_labels(status);
        counter!("vllm_router_pd_decode_errors_total",
            "worker" => worker.to_string(),
            "error_type" => error_type.to_string(),
            "status_code" => status_code,
            "status_class" => status_class.to_string()
        )
        .increment(1);
    }

    pub fn record_pd_stream_error(worker: &str) {
        counter!("vllm_router_pd_stream_errors_total",
            "worker" => worker.to_string()
        )
        .increment(1);
    }

    // Service discovery metrics
    pub fn record_discovery_update(added: usize, removed: usize) {
        counter!("vllm_router_discovery_updates_total").increment(1);
        gauge!("vllm_router_discovery_workers_added").set(added as f64);
        gauge!("vllm_router_discovery_workers_removed").set(removed as f64);
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

    pub fn record_embeddings_error(status: StatusCode) {
        let status_code = status_code_label(status);
        let status_class = status_class_label(status);
        counter!(
            "vllm_router_embeddings_errors_total",
            "status_code" => status_code,
            "status_class" => status_class.to_string()
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

#[cfg(test)]
mod tests {
    use super::*;
    use metrics::with_local_recorder;
    use std::net::TcpListener;

    fn build_test_recorder() -> metrics_exporter_prometheus::PrometheusRecorder {
        PrometheusBuilder::new()
            .set_buckets_for_metric(
                Matcher::Suffix(String::from("duration_seconds")),
                &DURATION_BUCKETS,
            )
            .expect("failed to configure buckets")
            .build_recorder()
    }

    // ============= PrometheusConfig Tests =============

    #[test]
    fn test_prometheus_config_default() {
        let config = PrometheusConfig::default();
        assert_eq!(config.port, 29000);
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_prometheus_config_custom() {
        let config = PrometheusConfig {
            port: 8080,
            host: "127.0.0.1".to_string(),
        };
        assert_eq!(config.port, 8080);
        assert_eq!(config.host, "127.0.0.1");
    }

    #[test]
    fn test_prometheus_config_clone() {
        let config = PrometheusConfig {
            port: 9090,
            host: "192.168.1.1".to_string(),
        };
        let cloned = config.clone();
        assert_eq!(cloned.port, config.port);
        assert_eq!(cloned.host, config.host);
    }

    // ============= IP Address Parsing Tests =============

    #[test]
    fn test_valid_ipv4_parsing() {
        let test_cases = vec!["127.0.0.1", "192.168.1.1", "0.0.0.0"];

        for ip_str in test_cases {
            let config = PrometheusConfig {
                port: 29000,
                host: ip_str.to_string(),
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            assert!(matches!(ip_addr, IpAddr::V4(_)));
        }
    }

    #[test]
    fn test_valid_ipv6_parsing() {
        let test_cases = vec!["::1", "2001:db8::1", "::"];

        for ip_str in test_cases {
            let config = PrometheusConfig {
                port: 29000,
                host: ip_str.to_string(),
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            assert!(matches!(ip_addr, IpAddr::V6(_)));
        }
    }

    #[test]
    fn test_invalid_ip_parsing() {
        let test_cases = vec!["invalid", "256.256.256.256", "hostname"];

        for ip_str in test_cases {
            let config = PrometheusConfig {
                port: 29000,
                host: ip_str.to_string(),
            };

            let ip_addr: IpAddr = config
                .host
                .parse()
                .unwrap_or(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));

            // Should fall back to 0.0.0.0
            assert_eq!(ip_addr, IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));
        }
    }

    // ============= Socket Address Creation Tests =============

    #[test]
    fn test_socket_addr_creation() {
        let test_cases = vec![("127.0.0.1", 8080), ("0.0.0.0", 29000), ("::1", 9090)];

        for (host, port) in test_cases {
            let config = PrometheusConfig {
                port,
                host: host.to_string(),
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            let socket_addr = SocketAddr::new(ip_addr, config.port);

            assert_eq!(socket_addr.port(), port);
            assert_eq!(socket_addr.ip().to_string(), host);
        }
    }

    #[test]
    fn test_socket_addr_with_different_ports() {
        let ports = vec![0, 80, 8080, 65535];

        for port in ports {
            let config = PrometheusConfig {
                port,
                host: "127.0.0.1".to_string(),
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            let socket_addr = SocketAddr::new(ip_addr, config.port);

            assert_eq!(socket_addr.port(), port);
        }
    }

    // ============= Duration Bucket Tests =============

    #[test]
    fn test_duration_bucket_coverage() {
        let test_cases: [(f64, &str); 7] = [
            (0.0005, "sub-millisecond"),
            (0.005, "5ms"),
            (0.05, "50ms"),
            (1.0, "1s"),
            (10.0, "10s"),
            (60.0, "1m"),
            (240.0, "4m"),
        ];

        let buckets: [f64; 20] = [
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
            60.0, 90.0, 120.0, 180.0, 240.0,
        ];

        for (duration, label) in test_cases {
            let bucket_found = buckets
                .iter()
                .any(|&b| (b - duration).abs() < 0.0001 || b > duration);
            assert!(bucket_found, "No bucket found for {} ({})", duration, label);
        }
    }

    // ============= Matcher Configuration Tests =============

    #[test]
    fn test_duration_suffix_matcher() {
        let matcher = Matcher::Suffix(String::from("duration_seconds"));

        // Test matching behavior
        let _matching_metrics = [
            "request_duration_seconds",
            "response_duration_seconds",
            "vllm_router_request_duration_seconds",
        ];

        let _non_matching_metrics = ["duration_total", "duration_seconds_total", "other_metric"];

        // Note: We can't directly test Matcher matching without the internals,
        // but we can verify the matcher is created correctly
        match matcher {
            Matcher::Suffix(suffix) => assert_eq!(suffix, "duration_seconds"),
            _ => panic!("Expected Suffix matcher"),
        }
    }

    // ============= Builder Configuration Tests =============

    #[test]
    fn test_prometheus_builder_configuration() {
        // This test verifies the builder configuration without actually starting Prometheus
        let _config = PrometheusConfig::default();

        let duration_matcher = Matcher::Suffix(String::from("duration_seconds"));
        let duration_bucket = [
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
            60.0, 90.0, 120.0, 180.0, 240.0,
        ];

        // Verify bucket configuration
        assert_eq!(duration_bucket.len(), 20);

        // Verify matcher is suffix type
        match duration_matcher {
            Matcher::Suffix(s) => assert_eq!(s, "duration_seconds"),
            _ => panic!("Expected Suffix matcher"),
        }
    }

    // ============= Upkeep Timeout Tests =============

    #[test]
    fn test_upkeep_timeout_duration() {
        let timeout = Duration::from_secs(5 * 60);
        assert_eq!(timeout.as_secs(), 300);
    }

    // ============= Custom Bucket Tests =============

    #[test]
    fn test_custom_buckets_for_different_metrics() {
        // Test that we can create different bucket configurations
        let request_buckets = [0.001, 0.01, 0.1, 1.0, 10.0];
        let generate_buckets = [0.1, 0.5, 1.0, 5.0, 30.0, 60.0];

        assert_eq!(request_buckets.len(), 5);
        assert_eq!(generate_buckets.len(), 6);

        // Verify each set is sorted
        for i in 1..request_buckets.len() {
            assert!(request_buckets[i] > request_buckets[i - 1]);
        }

        for i in 1..generate_buckets.len() {
            assert!(generate_buckets[i] > generate_buckets[i - 1]);
        }
    }

    // ============= RouterMetrics Tests =============

    #[test]
    fn test_metrics_static_methods() {
        // Test that all static methods can be called without panic
        RouterMetrics::observe_http_request(
            "/generate",
            "POST",
            StatusCode::OK,
            Duration::from_millis(100),
        );
        RouterMetrics::observe_http_response(
            "/generate",
            "POST",
            StatusCode::OK,
            Duration::from_millis(100),
        );
        RouterMetrics::observe_backend_http_response(
            "/generate",
            "http://worker1",
            "inference",
            "POST",
            StatusCode::BAD_GATEWAY,
            Duration::from_millis(75),
        );
        RouterMetrics::record_request_error(
            "/generate",
            "POST",
            StatusCode::REQUEST_TIMEOUT,
            "timeout",
        );
        RouterMetrics::record_retry("/generate");

        RouterMetrics::set_active_workers(5);
        RouterMetrics::set_worker_health("http://worker1", true);
        RouterMetrics::record_processed_request("http://worker1");

        RouterMetrics::record_policy_decision("random", "http://worker1");
        RouterMetrics::record_cache_hit();
        RouterMetrics::record_cache_miss();
        RouterMetrics::set_tree_size("http://worker1", 1000);
        RouterMetrics::record_load_balancing_event();
        RouterMetrics::set_load_range(20, 5);

        RouterMetrics::observe_pd_request(
            "/v1/chat/completions",
            "POST",
            StatusCode::BAD_GATEWAY,
            Duration::from_secs(1),
        );
        RouterMetrics::record_pd_prefill_request("http://prefill1", StatusCode::OK);
        RouterMetrics::record_pd_decode_request("http://decode1", StatusCode::BAD_GATEWAY);
        RouterMetrics::record_pd_error(
            "/v1/chat/completions",
            "POST",
            StatusCode::SERVICE_UNAVAILABLE,
            "invalid_request",
        );
        RouterMetrics::record_pd_prefill_error("http://prefill1", "transport", None);
        RouterMetrics::record_pd_decode_error(
            "http://decode1",
            "http_response",
            Some(StatusCode::BAD_GATEWAY),
        );
        RouterMetrics::record_pd_stream_error("http://decode1");

        RouterMetrics::record_discovery_update(3, 1);
        RouterMetrics::record_generate_duration(Duration::from_secs(2));
        RouterMetrics::record_embeddings_error(StatusCode::TOO_MANY_REQUESTS);
        RouterMetrics::set_running_requests("http://worker1", 15);
    }

    #[test]
    fn test_http_response_status_helpers() {
        assert_eq!(http_response_status_code(StatusCode::OK), "200");
        assert_eq!(http_response_status_class(StatusCode::OK), "2xx");
        assert_eq!(
            http_response_status_labels(StatusCode::TOO_MANY_REQUESTS),
            ("429".to_string(), "4xx")
        );
        assert_eq!(
            http_response_status_labels(StatusCode::SERVICE_UNAVAILABLE),
            ("503".to_string(), "5xx")
        );
    }

    #[test]
    fn test_rendered_metrics_include_status_aware_http_response_labels() {
        let recorder = build_test_recorder();
        let handle = recorder.handle();

        with_local_recorder(&recorder, || {
            RouterMetrics::observe_http_response(
                "/generate",
                "POST",
                StatusCode::TOO_MANY_REQUESTS,
                Duration::from_millis(25),
            );
            RouterMetrics::observe_backend_http_response(
                "/generate",
                "http://worker1",
                "decode",
                "POST",
                StatusCode::SERVICE_UNAVAILABLE,
                Duration::from_millis(50),
            );
        });

        let rendered = handle.render();

        let http_response_counter = rendered
            .lines()
            .find(|line| line.starts_with("vllm_router_http_responses_total{"))
            .expect("expected rendered final-response counter");
        assert!(http_response_counter.contains("route=\"/generate\""));
        assert!(http_response_counter.contains("http_request_method=\"POST\""));
        assert!(http_response_counter.contains("http_response_status_code=\"429\""));
        assert!(http_response_counter.contains("http_response_status_class=\"4xx\""));

        let http_response_duration = rendered
            .lines()
            .find(|line| line.starts_with("vllm_router_http_response_duration_seconds_count{"))
            .expect("expected rendered final-response duration histogram");
        assert!(http_response_duration.contains("http_request_method=\"POST\""));
        assert!(http_response_duration.contains("http_response_status_code=\"429\""));
        assert!(http_response_duration.contains("http_response_status_class=\"4xx\""));
        assert!(http_response_duration.ends_with(" 1"));

        let backend_response_counter = rendered
            .lines()
            .find(|line| line.starts_with("vllm_router_backend_http_responses_total{"))
            .expect("expected rendered backend-response counter");
        assert!(backend_response_counter.contains("route=\"/generate\""));
        assert!(backend_response_counter.contains("worker=\"http://worker1\""));
        assert!(backend_response_counter.contains("vllm_request_phase=\"decode\""));
        assert!(backend_response_counter.contains("http_request_method=\"POST\""));
        assert!(backend_response_counter.contains("http_response_status_code=\"503\""));
        assert!(backend_response_counter.contains("http_response_status_class=\"5xx\""));

        let backend_response_duration = rendered
            .lines()
            .find(|line| {
                line.starts_with("vllm_router_backend_http_response_duration_seconds_count{")
            })
            .expect("expected rendered backend-response duration histogram");
        assert!(backend_response_duration.contains("worker=\"http://worker1\""));
        assert!(backend_response_duration.contains("vllm_request_phase=\"decode\""));
        assert!(backend_response_duration.contains("http_request_method=\"POST\""));
        assert!(backend_response_duration.contains("http_response_status_code=\"503\""));
        assert!(backend_response_duration.contains("http_response_status_class=\"5xx\""));
        assert!(backend_response_duration.ends_with(" 1"));

        assert!(!rendered.contains("http.request.method"));
        assert!(!rendered.contains("http.response.status_code"));
        assert!(!rendered.contains("http.response.status_class"));
    }

    #[test]
    fn test_rendered_metrics_include_structured_http_labels() {
        let recorder = build_test_recorder();
        let handle = recorder.handle();

        with_local_recorder(&recorder, || {
            RouterMetrics::observe_http_request(
                "/v1/chat/completions",
                "POST",
                StatusCode::TOO_MANY_REQUESTS,
                Duration::from_millis(25),
            );
            RouterMetrics::record_request_error(
                "/v1/chat/completions",
                "POST",
                StatusCode::TOO_MANY_REQUESTS,
                "non_retryable_error",
            );
            RouterMetrics::observe_pd_request(
                "/v1/chat/completions",
                "POST",
                StatusCode::BAD_GATEWAY,
                Duration::from_millis(50),
            );
            RouterMetrics::record_pd_prefill_request("http://prefill1", StatusCode::OK);
            RouterMetrics::record_pd_prefill_error("http://prefill1", "transport", None);
            RouterMetrics::record_pd_decode_request("http://decode1", StatusCode::BAD_GATEWAY);
            RouterMetrics::record_pd_decode_error(
                "http://decode1",
                "http_response",
                Some(StatusCode::BAD_GATEWAY),
            );
            RouterMetrics::record_embeddings_error(StatusCode::TOO_MANY_REQUESTS);
        });

        let rendered = handle.render();

        assert!(rendered.lines().any(|line| {
            line.starts_with("vllm_router_http_requests_total{")
                && line.contains("route=\"/v1/chat/completions\"")
                && line.contains("method=\"POST\"")
                && line.contains("status_code=\"429\"")
                && line.contains("status_class=\"4xx\"")
                && line.ends_with(" 1")
        }));
        assert!(rendered.lines().any(|line| {
            line.starts_with("vllm_router_http_request_duration_seconds_count{")
                && line.contains("route=\"/v1/chat/completions\"")
                && line.contains("method=\"POST\"")
                && line.contains("status_class=\"4xx\"")
                && line.ends_with(" 1")
        }));
        assert!(rendered.lines().any(|line| {
            line.starts_with("vllm_router_request_errors_total{")
                && line.contains("error_type=\"non_retryable_error\"")
                && line.contains("status_code=\"429\"")
                && line.contains("status_class=\"4xx\"")
        }));
        assert!(rendered.lines().any(|line| {
            line.starts_with("vllm_router_pd_requests_total{")
                && line.contains("route=\"/v1/chat/completions\"")
                && line.contains("method=\"POST\"")
                && line.contains("status_code=\"502\"")
                && line.contains("status_class=\"5xx\"")
                && line.ends_with(" 1")
        }));
        assert!(rendered.lines().any(|line| {
            line.starts_with("vllm_router_pd_prefill_errors_total{")
                && line.contains("worker=\"http://prefill1\"")
                && line.contains("error_type=\"transport\"")
                && line.contains("status_code=\"none\"")
                && line.contains("status_class=\"none\"")
        }));
        assert!(rendered.lines().any(|line| {
            line.starts_with("vllm_router_embeddings_errors_total{")
                && line.contains("status_code=\"429\"")
                && line.contains("status_class=\"4xx\"")
        }));
    }

    #[test]
    fn test_rendered_metrics_include_cache_and_discovery_metrics() {
        let recorder = build_test_recorder();
        let handle = recorder.handle();

        with_local_recorder(&recorder, || {
            RouterMetrics::record_cache_hit();
            RouterMetrics::record_cache_miss();
            RouterMetrics::set_tree_size("http://worker1", 12);
            RouterMetrics::record_discovery_update(1, 0);
        });

        let rendered = handle.render();

        assert!(rendered.contains("vllm_router_cache_hits_total 1"));
        assert!(rendered.contains("vllm_router_cache_misses_total 1"));
        assert!(rendered.lines().any(|line| {
            line.starts_with("vllm_router_tree_size{")
                && line.contains("worker=\"http://worker1\"")
                && line.ends_with(" 12")
        }));
        assert!(rendered.contains("vllm_router_discovery_updates_total 1"));
        assert!(rendered.contains("vllm_router_discovery_workers_added 1"));
        assert!(rendered.contains("vllm_router_discovery_workers_removed 0"));
    }

    #[test]
    fn test_rendered_metrics_exclude_removed_dormant_families() {
        let recorder = build_test_recorder();
        let handle = recorder.handle();

        with_local_recorder(&recorder, || {
            RouterMetrics::set_active_workers(1);
            RouterMetrics::observe_http_request(
                "/generate",
                "POST",
                StatusCode::OK,
                Duration::from_millis(10),
            );
        });

        let rendered = handle.render();

        assert!(!rendered.contains("vllm_router_worker_load"));
        assert!(!rendered.contains("vllm_tokenizer_"));
    }

    // ============= Port Availability Tests =============

    #[test]
    fn test_port_already_in_use() {
        // Skip this test if we can't bind to the port
        let port = 29123; // Use a different port to avoid conflicts

        if let Ok(_listener) = TcpListener::bind(("127.0.0.1", port)) {
            // Port is available, we can test
            let config = PrometheusConfig {
                port,
                host: "127.0.0.1".to_string(),
            };

            // Just verify config is created correctly
            assert_eq!(config.port, port);
        }
    }

    // ============= Integration Test Helpers =============

    #[test]
    fn test_metrics_endpoint_accessibility() {
        // This would be an integration test in practice
        // Here we just verify the configuration
        let config = PrometheusConfig {
            port: 29000,
            host: "127.0.0.1".to_string(),
        };

        let ip_addr: IpAddr = config.host.parse().unwrap();
        let socket_addr = SocketAddr::new(ip_addr, config.port);

        assert_eq!(socket_addr.to_string(), "127.0.0.1:29000");
    }

    #[test]
    fn test_concurrent_metric_updates() {
        // Test that metric updates can be called concurrently
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::thread;

        let done = Arc::new(AtomicBool::new(false));
        let mut handles = vec![];

        for i in 0..3 {
            let done_clone = done.clone();
            let handle = thread::spawn(move || {
                let worker = format!("http://worker{}", i);
                while !done_clone.load(Ordering::Relaxed) {
                    RouterMetrics::set_running_requests(&worker, i * 10);
                    RouterMetrics::record_processed_request(&worker);
                    thread::sleep(Duration::from_millis(1));
                }
            });
            handles.push(handle);
        }

        // Let threads run briefly
        thread::sleep(Duration::from_millis(10));
        done.store(true, Ordering::Relaxed);

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
    }

    // ============= Edge Cases Tests =============

    #[test]
    fn test_empty_string_metrics() {
        // Test that empty strings don't cause issues
        RouterMetrics::observe_http_request("", "GET", StatusCode::OK, Duration::from_millis(1));
        RouterMetrics::set_worker_health("", true);
        RouterMetrics::record_policy_decision("", "");
    }

    #[test]
    fn test_very_long_metric_labels() {
        let long_label = "a".repeat(1000);

        RouterMetrics::observe_http_request(
            &long_label,
            "GET",
            StatusCode::OK,
            Duration::from_millis(1),
        );
        RouterMetrics::set_worker_health(&long_label, false);
    }

    #[test]
    fn test_special_characters_in_labels() {
        let special_labels = [
            "test/with/slashes",
            "test-with-dashes",
            "test_with_underscores",
            "test.with.dots",
            "test:with:colons",
        ];

        for label in special_labels {
            RouterMetrics::observe_http_request(
                label,
                "GET",
                StatusCode::OK,
                Duration::from_millis(1),
            );
            RouterMetrics::set_worker_health(label, true);
        }
    }

    #[test]
    fn test_extreme_metric_values() {
        // Test extreme values
        RouterMetrics::set_active_workers(0);
        RouterMetrics::set_active_workers(usize::MAX);

        RouterMetrics::set_running_requests("worker", 0);
        RouterMetrics::set_running_requests("worker", usize::MAX);

        RouterMetrics::observe_http_request(
            "route",
            "GET",
            StatusCode::OK,
            Duration::from_nanos(1),
        );
        // 24 hours
        RouterMetrics::observe_http_request(
            "route",
            "GET",
            StatusCode::OK,
            Duration::from_secs(86400),
        );
    }
}
