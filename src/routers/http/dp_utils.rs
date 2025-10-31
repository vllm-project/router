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
        info!("Expanding worker {} to {} DP-aware URLs (ranks 0..{})", url, dp_size, dp_size - 1);

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
}
