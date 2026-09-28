// Shared DP-aware routing utilities
// This module provides common functions for data-parallel aware routing
// that can be reused across different router implementations.

use std::collections::HashSet;

use tracing::{info, warn};

/// Count the engines a vLLM server exposes on `/metrics`. It emits one series
/// per engine it can address, the same set `X-data-parallel-rank` selects from.
/// Hybrid load balancing labels them with global ranks, so count distinct
/// labels rather than taking the maximum.
fn count_engine_labels(metrics_body: &str) -> Option<usize> {
    let engines: HashSet<&str> = metrics_body
        .lines()
        .filter(|line| line.starts_with("vllm:"))
        .filter_map(|line| {
            let (_, after_label) = line.split_once("engine=\"")?;
            let (engine, _) = after_label.split_once('"')?;
            Some(engine)
        })
        .collect();

    (!engines.is_empty()).then_some(engines.len())
}

/// Number of engines a worker fronts, so prefill and decode can run different
/// data-parallel sizes. Falls back to `fallback_dp_size` for workers exposing
/// no vLLM metrics, e.g. under `--disable-log-stats`.
pub async fn discover_dp_size(
    client: &reqwest::Client,
    worker_url: &str,
    fallback_dp_size: usize,
) -> usize {
    let fallback_dp_size = fallback_dp_size.max(1);
    let (base_url, _) = parse_worker_url(worker_url);
    // /metrics sits outside vLLM's authenticated path prefixes.
    let metrics_url = format!("{}/metrics", base_url.trim_end_matches('/'));

    let body = match client.get(&metrics_url).send().await {
        Ok(response) if response.status().is_success() => response.text().await.ok(),
        Ok(response) => {
            warn!(
                "DP discovery for {} returned {}",
                metrics_url,
                response.status()
            );
            None
        }
        Err(error) => {
            warn!("DP discovery for {} failed: {}", metrics_url, error);
            None
        }
    };

    match body.as_deref().and_then(count_engine_labels) {
        Some(dp_size) => {
            info!("Worker {} reports {} engine(s)", base_url, dp_size);
            dp_size
        }
        None => {
            info!(
                "Worker {} reports no engine metrics; assuming {} engine(s)",
                base_url, fallback_dp_size
            );
            fallback_dp_size
        }
    }
}

/// Given a list of worker URLs, expand them into DP-aware URLs
/// with dp_rank as suffix (format: "http://host:port@rank")
///
/// This function does NOT query the workers - it uses the provided dp_size
/// to expand each worker URL into multiple DP-aware URLs with rank suffixes.
///
/// # Arguments
/// * `worker_urls` - List of base worker URLs
/// * `_api_key` - Unused, kept for API compatibility
/// * `dp_size` - Number of DP ranks to create for each worker
///
/// # Returns
/// * `Ok(Vec<String>)` - List of expanded worker URLs with dp_rank suffixes
///
/// # Example
/// ```
/// // For worker "http://host:8000" with dp_size=2:
/// // Returns: ["http://host:8000@0", "http://host:8000@1"]
/// ```
pub async fn get_dp_aware_workers(
    worker_urls: &[String],
    _api_key: &Option<String>,
    dp_size: usize,
) -> Result<Vec<String>, String> {
    let mut dp_aware_workers: Vec<String> = Vec::new();

    for url in worker_urls {
        info!(
            "Expanding worker {} to {} DP-aware URLs (ranks 0..{})",
            url,
            dp_size,
            dp_size - 1
        );

        // Expand each worker URL to multiple DP-aware URLs
        for rank in 0..dp_size {
            dp_aware_workers.push(format!("{}@{}", url, rank));
        }
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

    #[tokio::test]
    async fn test_get_dp_aware_workers_ipv6() {
        let urls = vec!["https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009".to_string()];
        let result = get_dp_aware_workers(&urls, &None, 4).await.unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(
            result[0],
            "https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009@0"
        );
        assert_eq!(
            result[1],
            "https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009@1"
        );
        assert_eq!(
            result[2],
            "https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009@2"
        );
        assert_eq!(
            result[3],
            "https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009@3"
        );

        // Verify round-trip: extracting dp_rank from expanded URLs works
        for (i, url) in result.iter().enumerate() {
            let (base, rank) = extract_dp_rank(url).unwrap();
            assert_eq!(
                base,
                "https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009"
            );
            assert_eq!(rank, i);
        }
    }
}
