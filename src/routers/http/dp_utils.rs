// Shared DP-aware routing utilities
// This module provides common functions for data-parallel aware routing
// that can be reused across different router implementations.

use tracing::info;

/// Query a worker's /get_server_info endpoint to get its dp_size
///
/// # Arguments
/// * `worker_url` - Base URL of the worker (e.g., "http://host:port")
/// * `api_key` - Optional API key for authentication
///
/// # Returns
/// * `Ok(usize)` - The dp_size value from the worker
/// * `Err(String)` - Error message if the request fails or dp_size is not found
pub async fn get_worker_dp_size(worker_url: &str, api_key: &Option<String>) -> Result<usize, String> {
    let client = reqwest::Client::new();
    let mut req_builder = client.get(format!("{}/get_server_info", worker_url));
    if let Some(key) = api_key {
        req_builder = req_builder.bearer_auth(key);
    }

    match req_builder.send().await {
        Ok(res) => {
            if res.status().is_success() {
                let server_info = res
                    .text()
                    .await
                    .map_err(|e| format!("failed to read text from response: {}", e))?;

                let server_info: serde_json::Value = serde_json::from_str(&server_info)
                    .map_err(|e| format!("failed to decode JSON: {}", e))?;

                let dp_size = server_info
                    .get("dp_size")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| String::from("dp_size not found or not an u64"))?;

                Ok(if dp_size > usize::MAX as u64 {
                    return Err(format!("dp_size is too large: {}", dp_size));
                } else {
                    dp_size as usize
                })
            } else {
                Err(format!("unexpected status code: {}", res.status()))
            }
        }
        Err(e) => Err(format!("error response: {}", e)),
    }
}

/// Given a list of worker URLs, expand them into DP-aware URLs
/// with dp_rank as suffix (format: "http://host:port@rank")
///
/// # Arguments
/// * `worker_urls` - List of base worker URLs
/// * `api_key` - Optional API key for authentication
///
/// # Returns
/// * `Ok(Vec<String>)` - List of expanded worker URLs with dp_rank suffixes
/// * `Err(String)` - Error message if any worker query fails
pub async fn get_dp_aware_workers(
    worker_urls: &[String],
    api_key: &Option<String>,
) -> Result<Vec<String>, String> {
    let mut dp_aware_workers: Vec<String> = Vec::new();

    for url in worker_urls {
        match get_worker_dp_size(url, api_key).await {
            Ok(dp_size) => {
                info!("Worker {} has dp_size={}, expanding to {} DP-aware URLs", url, dp_size, dp_size);
                // Use all available DP ranks for load balancing
                for rank in 0..dp_size {
                    dp_aware_workers.push(format!("{}@{}", url, rank));
                }
            }
            Err(e) => return Err(format!("Failed to get DP size for {}: {}", url, e)),
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
