// Custom error type for PD router operations
#[derive(Debug, thiserror::Error)]
pub enum PDRouterError {
    #[error("Worker already exists: {url}")]
    WorkerAlreadyExists { url: String },

    #[error("Worker not found: {url}")]
    WorkerNotFound { url: String },

    #[error("Lock acquisition failed: {operation}")]
    LockError { operation: String },

    #[error("Health check failed for worker: {url}")]
    HealthCheckFailed { url: String },

    #[error("Invalid worker configuration: {reason}")]
    InvalidConfiguration { reason: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Timeout waiting for worker: {url}")]
    Timeout { url: String },
}

/// Format a full error chain for debugging (walks source() recursively).
/// Produces output like: "outer error caused by: middle error caused by: root cause"
pub fn error_chain(err: &dyn std::error::Error) -> String {
    let mut chain = vec![err.to_string()];
    let mut source = err.source();
    while let Some(s) = source {
        chain.push(s.to_string());
        source = s.source();
    }
    chain.join(" caused by: ")
}

// Helper functions for workers
pub fn api_path(url: &str, api_path: &str) -> String {
    if api_path.starts_with("/") {
        format!("{}{}", url, api_path)
    } else {
        format!("{}/{}", url, api_path)
    }
}

pub fn get_hostname(url: &str) -> String {
    // Simple hostname extraction without external dependencies
    let url = url
        .trim_start_matches("http://")
        .trim_start_matches("https://");
    url.split(':').next().unwrap_or("localhost").to_string()
}

use serde::Serialize;

// Optimized bootstrap wrapper for single requests
#[derive(Serialize)]
pub struct RequestWithBootstrap<'a, T: Serialize> {
    #[serde(flatten)]
    pub original: &'a T,
    pub bootstrap_host: String,
    pub bootstrap_port: Option<u16>,
    pub bootstrap_room: u64,
}

// Optimized bootstrap wrapper for batch requests
#[derive(Serialize)]
pub struct BatchRequestWithBootstrap<'a, T: Serialize> {
    #[serde(flatten)]
    pub original: &'a T,
    pub bootstrap_host: Vec<String>,
    pub bootstrap_port: Vec<Option<u16>>,
    pub bootstrap_room: Vec<u64>,
}

// Helper to generate bootstrap room ID
pub fn generate_room_id() -> u64 {
    // Generate a value in the range [0, 2^63 - 1] to match Python's random.randint(0, 2**63 - 1)
    rand::random::<u64>() & (i64::MAX as u64)
}

// PD-specific routing policies
#[derive(Debug, Clone, PartialEq)]
pub enum PDSelectionPolicy {
    Random,
    PowerOfTwo,
    CacheAware {
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;

    // Simple custom error for testing error chains
    #[derive(Debug)]
    struct TestError {
        msg: String,
        source: Option<Box<dyn std::error::Error>>,
    }

    impl fmt::Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.msg)
        }
    }

    impl std::error::Error for TestError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            self.source.as_deref()
        }
    }

    #[test]
    fn test_error_chain_single_error() {
        let err = TestError {
            msg: "something broke".into(),
            source: None,
        };
        assert_eq!(error_chain(&err), "something broke");
    }

    #[test]
    fn test_error_chain_nested_errors() {
        let inner = TestError {
            msg: "root cause".into(),
            source: None,
        };
        let outer = TestError {
            msg: "outer error".into(),
            source: Some(Box::new(inner)),
        };
        assert_eq!(error_chain(&outer), "outer error caused by: root cause");
    }

    #[test]
    fn test_error_chain_triple_nested() {
        let root = TestError {
            msg: "connection reset".into(),
            source: None,
        };
        let middle = TestError {
            msg: "HTTP send failed".into(),
            source: Some(Box::new(root)),
        };
        let top = TestError {
            msg: "prefill request failed".into(),
            source: Some(Box::new(middle)),
        };
        assert_eq!(
            error_chain(&top),
            "prefill request failed caused by: HTTP send failed caused by: connection reset"
        );
    }
}
