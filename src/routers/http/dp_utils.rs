// Shared DP-aware routing utilities
// This module provides common functions for data-parallel aware routing
// that can be reused across different router implementations.

use tracing::info;

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
    // ============================================================
    // Verify parse_worker_url handles ALL URL formats correctly
    // These tests cover both the #98 regression (hostname:port)
    // and the original #92 concern (IPv6)
    // ============================================================

    #[test]
    fn test_parse_worker_url_hostname_port_dp() {
        let (base, rank) = parse_worker_url("http://node1:8087@3");
        assert_eq!(base, "http://node1:8087");
        assert_eq!(rank, Some(3));
        // Verify the URL we'd send to reqwest is clean
        let request_url = format!("{}/v1/completions", base);
        assert!(
            !request_url.contains('@'),
            "request URL must not contain @rank"
        );
        assert_eq!(request_url, "http://node1:8087/v1/completions");
    }

    #[test]
    fn test_parse_worker_url_hostname_port_no_dp() {
        let (base, rank) = parse_worker_url("http://node1:8087");
        assert_eq!(base, "http://node1:8087");
        assert_eq!(rank, None);
    }

    // ============================================================
    // Critical: prove that reqwest would parse @rank as userinfo
    // This is the actual bug that #98 introduced
    // ============================================================

    #[test]
    fn test_reqwest_userinfo_proof() {
        // Demonstrate WHY @rank must be stripped before reqwest:
        // reqwest/url parses "http://host:port@rank" as userinfo

        // BAD: raw URL with @rank — reqwest sees wrong host
        let bad_url = url::Url::parse("http://node1:8087@3/v1/completions").unwrap();
        assert_eq!(
            bad_url.host_str(),
            Some("0.0.0.3"),
            "proves @rank corrupts host"
        );
        assert_eq!(bad_url.username(), "node1", "host becomes username");

        // GOOD: stripped URL — reqwest sees correct host
        let (base, rank) = parse_worker_url("http://node1:8087@3");
        let good_url_str = format!("{}/v1/completions", base);
        let good_url = url::Url::parse(&good_url_str).unwrap();
        assert_eq!(
            good_url.host_str(),
            Some("node1"),
            "host is correct after stripping"
        );
        assert_eq!(good_url.port(), Some(8087));
        assert_eq!(rank, Some(3));
    }

    #[test]
    fn test_reqwest_userinfo_proof_ipv6() {
        // Same proof for IPv6 — verify our fix works for both formats
        let (base, rank) =
            parse_worker_url("https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009@2");
        let url_str = format!("{}/v1/completions", base);
        let parsed = url::Url::parse(&url_str).unwrap();
        assert_eq!(
            parsed.host_str(),
            Some("[2a03:83e4:5006:90:5f5a:f8c5:400:0]")
        );
        assert_eq!(parsed.port(), Some(20009));
        assert_eq!(rank, Some(2));
        assert!(parsed.username().is_empty(), "no userinfo pollution");
    }

    // ============================================================
    // End-to-end: expand → parse → build URL round-trip
    // ============================================================

    #[tokio::test]
    async fn test_dp_expand_roundtrip_hostname() {
        let urls = vec!["http://node1:8087".to_string()];
        let expanded = get_dp_aware_workers(&urls, &None, 4).await.unwrap();

        for (i, worker_url) in expanded.iter().enumerate() {
            // Simulate what our fixed vllm_pd_router does:
            let (base, rank) = parse_worker_url(worker_url);
            let request_url = format!("{}/v1/completions", base);

            // Verify rank is correct
            assert_eq!(rank, Some(i));

            // Verify URL is clean for reqwest
            assert!(!request_url.contains('@'));
            let parsed = url::Url::parse(&request_url).unwrap();
            assert_eq!(parsed.host_str(), Some("node1"));
            assert_eq!(parsed.port(), Some(8087));
        }
    }

    #[tokio::test]
    async fn test_dp_expand_roundtrip_ipv6() {
        let urls = vec!["https://[2a03:83e4:5006:0090:5f5a:f8c5:0400:0000]:20009".to_string()];
        let expanded = get_dp_aware_workers(&urls, &None, 4).await.unwrap();

        for (i, worker_url) in expanded.iter().enumerate() {
            let (base, rank) = parse_worker_url(worker_url);
            let request_url = format!("{}/v1/completions", base);

            assert_eq!(rank, Some(i));
            assert!(!request_url.contains('@'));
            let parsed = url::Url::parse(&request_url).unwrap();
            assert_eq!(
                parsed.host_str(),
                Some("[2a03:83e4:5006:90:5f5a:f8c5:400:0]")
            );
            assert_eq!(parsed.port(), Some(20009));
        }
    }
}
