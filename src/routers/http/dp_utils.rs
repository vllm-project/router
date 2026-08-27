// Shared DP-aware routing utilities
// This module provides common functions for data-parallel aware routing
// that can be reused across different router implementations.

use std::collections::BTreeSet;
use std::time::Duration;

use tracing::{info, warn};

/// How long to wait for a worker's `/metrics` scrape during DP rank discovery.
const DP_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(5);

/// Parse the set of DP ranks a vLLM server exposes on `/metrics`.
///
/// vLLM emits one series per engine it can address — the same set that
/// `X-data-parallel-rank` selects from — labelled with the engine's **global**
/// data-parallel rank. The ranks are returned sorted, so the caller's ordering
/// is deterministic.
///
/// Returns `None` when the body carries no parseable vLLM engine labels (e.g.
/// a server started with `--disable-log-stats`).
fn parse_engine_ranks(metrics_body: &str) -> Option<Vec<usize>> {
    let ranks: BTreeSet<usize> = metrics_body
        .lines()
        .filter(|line| line.starts_with("vllm:"))
        .filter_map(|line| {
            let (_, after_label) = line.split_once("engine=\"")?;
            let (engine, _) = after_label.split_once('"')?;
            engine.parse::<usize>().ok()
        })
        .collect();

    (!ranks.is_empty()).then(|| ranks.into_iter().collect())
}

/// Ask a worker which global DP ranks it serves, via its `/metrics` endpoint.
///
/// Returns `None` if the worker is unreachable, answers non-2xx, or exposes no
/// engine labels — the caller falls back to assuming ranks `0..dp_size`.
async fn discover_engine_ranks(client: &reqwest::Client, base_url: &str) -> Option<Vec<usize>> {
    // /metrics sits outside vLLM's authenticated path prefixes.
    let metrics_url = format!("{}/metrics", base_url.trim_end_matches('/'));

    let body = match client.get(&metrics_url).send().await {
        Ok(response) if response.status().is_success() => response.text().await.ok()?,
        Ok(response) => {
            warn!(
                "DP rank discovery for {} returned {}",
                metrics_url,
                response.status()
            );
            return None;
        }
        Err(error) => {
            warn!("DP rank discovery for {} failed: {}", metrics_url, error);
            return None;
        }
    };

    parse_engine_ranks(&body)
}

/// Build the DP-aware URLs ("http://host:port@rank") for a single worker.
fn dp_aware_urls(base_url: &str, ranks: &[usize]) -> Vec<String> {
    ranks
        .iter()
        .map(|rank| format!("{}@{}", base_url, rank))
        .collect()
}

/// Given a list of worker URLs, expand them into DP-aware URLs
/// with dp_rank as suffix (format: "http://host:port@rank")
///
/// The rank suffix is the engine's **global** DP rank, which is what vLLM
/// resolves the `X-data-parallel-rank` header against. Each worker is asked for
/// its own ranks via `/metrics` rather than assuming every worker starts at
/// rank 0: under `--data-parallel-hybrid-lb` the API server on the second node
/// of a DP8 deployment owns global ranks 4..7, and addressing it as 0..3 makes
/// it reject every request.
///
/// Workers that expose no engine metrics fall back to `0..dp_size`, preserving
/// the previous behaviour for single-node and internal-LB deployments.
///
/// # Arguments
/// * `worker_urls` - List of base worker URLs
/// * `_api_key` - Unused; `/metrics` is not behind vLLM's API-key prefixes
/// * `dp_size` - Ranks per worker to assume when discovery is unavailable
///
/// # Returns
/// * `Ok(Vec<String>)` - List of expanded worker URLs with dp_rank suffixes
///
/// # Example
/// ```
/// // For worker "http://host:8000" reporting engines 4..7:
/// // Returns: ["http://host:8000@4", ..., "http://host:8000@7"]
/// ```
pub async fn get_dp_aware_workers(
    worker_urls: &[String],
    _api_key: &Option<String>,
    dp_size: usize,
) -> Result<Vec<String>, String> {
    let client = reqwest::Client::builder()
        .timeout(DP_DISCOVERY_TIMEOUT)
        .build()
        .map_err(|e| format!("Failed to create HTTP client for DP rank discovery: {}", e))?;

    // Discover every worker concurrently: a worker that does not answer costs a
    // full DP_DISCOVERY_TIMEOUT, and serialising that over a large deployment
    // would add it to router startup once per node.
    let base_urls: Vec<String> = worker_urls
        .iter()
        .map(|url| parse_worker_url(url).0)
        .collect();
    let discovered = futures::future::join_all(
        base_urls
            .iter()
            .map(|base_url| discover_engine_ranks(&client, base_url)),
    )
    .await;

    let mut dp_aware_workers: Vec<String> = Vec::new();

    for (base_url, ranks) in base_urls.iter().zip(discovered) {
        let ranks = match ranks {
            Some(ranks) => {
                if ranks.len() != dp_size {
                    warn!(
                        "Worker {} reports {} engine(s) {:?}, but \
                         --intra-node-data-parallel-size is {}; using the reported ranks",
                        base_url,
                        ranks.len(),
                        ranks,
                        dp_size
                    );
                }
                ranks
            }
            None => {
                warn!(
                    "Could not read engine ranks from {}/metrics; assuming ranks 0..{}. \
                     If this worker is a non-zero-rank node of a --data-parallel-hybrid-lb \
                     deployment, requests routed to it will be rejected.",
                    base_url,
                    dp_size.saturating_sub(1)
                );
                (0..dp_size).collect()
            }
        };

        info!(
            "Expanding worker {} to {} DP-aware URLs (ranks {:?})",
            base_url,
            ranks.len(),
            ranks
        );

        dp_aware_workers.extend(dp_aware_urls(base_url, &ranks));
    }

    Ok(dp_aware_workers)
}

/// Extract dp_rank from a DP-aware worker URL
///
/// # Arguments
/// * `worker_url` - DP-aware worker URL in format "http://host:port@rank"
///
/// # Returns
/// * `Ok((&str, usize))` - Tuple of (base_url, dp_rank)
/// * `Err(String)` - Error message if the format is invalid
///
/// # Example
/// ```
/// use vllm_router_rs::routers::http::dp_utils::extract_dp_rank;
///
/// let (base_url, rank) = extract_dp_rank("http://worker:8000@3").unwrap();
/// assert_eq!(base_url, "http://worker:8000");
/// assert_eq!(rank, 3);
/// ```
pub fn extract_dp_rank(worker_url: &str) -> Result<(&str, usize), String> {
    let parts: Vec<&str> = worker_url.split('@').collect();
    if parts.len() != 2 {
        return Err(format!("invalid worker_url format: {}", worker_url));
    }

    // Parse the second part (dp_rank) into an integer
    match parts[1].parse::<usize>() {
        Ok(dp_rank) => Ok((parts[0], dp_rank)),
        Err(_) => Err(format!(
            "failed to parse dp_rank from worker_url: {}",
            worker_url
        )),
    }
}

/// Parse a worker URL and extract base URL and optional dp_rank
///
/// This is a convenience function that handles both DP-aware URLs (with @rank suffix)
/// and regular URLs (without @rank suffix).
///
/// # Arguments
/// * `worker_url` - Worker URL which may or may not have @rank suffix
///
/// # Returns
/// * `(String, Option<usize>)` - Tuple of (base_url, optional_dp_rank)
///   - For DP-aware URL "http://host:8000@3": returns ("http://host:8000", Some(3))
///   - For regular URL "http://host:8000": returns ("http://host:8000", None)
///
/// # Example
/// ```
/// use vllm_router_rs::routers::http::dp_utils::parse_worker_url;
///
/// let (base, rank) = parse_worker_url("http://worker:8000@3");
/// assert_eq!(base, "http://worker:8000");
/// assert_eq!(rank, Some(3));
///
/// let (base, rank) = parse_worker_url("http://worker:8000");
/// assert_eq!(base, "http://worker:8000");
/// assert_eq!(rank, None);
/// ```
pub fn parse_worker_url(worker_url: &str) -> (String, Option<usize>) {
    match extract_dp_rank(worker_url) {
        Ok((base, rank)) => (base.to_string(), Some(rank)),
        Err(_) => (worker_url.to_string(), None),
    }
}

/// Add X-data-parallel-rank header to a reqwest RequestBuilder if dp_rank is present
///
/// This is a utility function to standardize how DP rank headers are added to HTTP requests.
///
/// # Arguments
/// * `request` - The reqwest RequestBuilder to add headers to
/// * `dp_rank` - Optional DP rank to add as a header
///
/// # Returns
/// * The RequestBuilder with the header added (if dp_rank was Some)
///
/// # Example
/// ```
/// use vllm_router_rs::routers::http::dp_utils::add_dp_rank_header;
///
/// let client = reqwest::Client::new();
/// let mut request = client.post("http://worker:8000/v1/generate");
/// request = add_dp_rank_header(request, Some(3));
/// // Request now has "X-data-parallel-rank: 3" header
/// ```
pub fn add_dp_rank_header(
    mut request: reqwest::RequestBuilder,
    dp_rank: Option<usize>,
) -> reqwest::RequestBuilder {
    if let Some(rank) = dp_rank {
        request = request.header("X-data-parallel-rank", rank.to_string());
    }
    request
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_dp_rank_valid() {
        let result = extract_dp_rank("http://worker:8000@3");
        assert!(result.is_ok());
        let (base, rank) = result.unwrap();
        assert_eq!(base, "http://worker:8000");
        assert_eq!(rank, 3);
    }

    #[test]
    fn test_extract_dp_rank_no_at() {
        let result = extract_dp_rank("http://worker:8000");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid worker_url format"));
    }

    #[test]
    fn test_extract_dp_rank_invalid_rank() {
        let result = extract_dp_rank("http://worker:8000@abc");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("failed to parse dp_rank"));
    }

    #[test]
    fn test_extract_dp_rank_multiple_at() {
        let result = extract_dp_rank("http://worker:8000@3@5");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid worker_url format"));
    }

    #[test]
    fn test_extract_dp_rank_ipv6() {
        let result = extract_dp_rank("https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009@2");
        assert!(result.is_ok());
        let (base, rank) = result.unwrap();
        assert_eq!(
            base,
            "https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009"
        );
        assert_eq!(rank, 2);
    }

    #[test]
    fn test_extract_dp_rank_ipv6_rank_zero() {
        let result = extract_dp_rank("https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009@0");
        assert!(result.is_ok());
        let (base, rank) = result.unwrap();
        assert_eq!(
            base,
            "https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009"
        );
        assert_eq!(rank, 0);
    }

    #[test]
    fn test_parse_worker_url_ipv6_dp() {
        let (base, rank) =
            parse_worker_url("https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009@3");
        assert_eq!(
            base,
            "https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009"
        );
        assert_eq!(rank, Some(3));
    }

    #[test]
    fn test_parse_worker_url_ipv6_no_dp() {
        let (base, rank) =
            parse_worker_url("https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009");
        assert_eq!(
            base,
            "https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009"
        );
        assert_eq!(rank, None);
    }

    #[test]
    fn test_dp_aware_urls_ipv6() {
        let base = "https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009";
        let result = dp_aware_urls(base, &[0, 1, 2, 3]);
        assert_eq!(result.len(), 4);
        assert_eq!(
            result[0],
            "https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009@0"
        );
        assert_eq!(
            result[3],
            "https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009@3"
        );

        // Verify round-trip: extracting dp_rank from expanded URLs works
        for (i, url) in result.iter().enumerate() {
            let (parsed_base, rank) = extract_dp_rank(url).unwrap();
            assert_eq!(parsed_base, base);
            assert_eq!(rank, i);
        }
    }

    /// The second node of a `--data-parallel-hybrid-lb` DP8 deployment owns
    /// global ranks 4..7, so its URLs must carry those ranks, not 0..3.
    #[test]
    fn test_dp_aware_urls_uses_global_ranks() {
        let result = dp_aware_urls("http://node2:8000", &[4, 5, 6, 7]);
        assert_eq!(
            result,
            vec![
                "http://node2:8000@4",
                "http://node2:8000@5",
                "http://node2:8000@6",
                "http://node2:8000@7",
            ]
        );
    }

    #[test]
    fn test_parse_engine_ranks_hybrid_lb_second_node() {
        // Abridged /metrics from the DP8 hybrid-LB node that owns ranks 4..7.
        let body = r#"# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="4",model_name="m"} 3.0
vllm:num_requests_running{engine="5",model_name="m"} 1.0
vllm:num_requests_running{engine="6",model_name="m"} 0.0
vllm:num_requests_running{engine="7",model_name="m"} 2.0
vllm:num_requests_waiting{engine="4",model_name="m"} 0.0
process_resident_memory_bytes 1.234e+09
"#;
        assert_eq!(parse_engine_ranks(body), Some(vec![4, 5, 6, 7]));
    }

    #[test]
    fn test_parse_engine_ranks_ignores_unparseable_and_foreign_labels() {
        let body = r#"vllm:cache_config_info{engine="",model_name="m"} 1.0
vllm:num_requests_running{engine="1",model_name="m"} 0.0
vllm:num_requests_running{engine="0",model_name="m"} 0.0
other:metric{engine="9"} 1.0
"#;
        // Sorted, deduped, and the empty/foreign labels dropped.
        assert_eq!(parse_engine_ranks(body), Some(vec![0, 1]));
    }

    #[test]
    fn test_parse_engine_ranks_none_without_engine_labels() {
        assert_eq!(parse_engine_ranks(""), None);
        assert_eq!(parse_engine_ranks("process_cpu_seconds_total 1.0\n"), None);
    }
}
