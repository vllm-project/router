pub mod types;
pub mod validation;

pub use types::*;
pub use validation::*;

/// Configuration errors
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("{message}")]
    UnsupportedEncodePolicy {
        policy: &'static str,
        role: crate::core::WorkerType,
        reason: &'static str,
        origin: EncodePolicySource,
        message: String,
    },
    #[error("Validation failed: {reason}")]
    ValidationFailed { reason: String },

    #[error("Invalid value for field '{field}': {value} - {reason}")]
    InvalidValue {
        field: String,
        value: String,
        reason: String,
    },

    #[error("Incompatible configuration: {reason}")]
    IncompatibleConfig { reason: String },

    #[error("Missing required field: {field}")]
    MissingRequired { field: String },
}

/// Result type for configuration operations
pub type ConfigResult<T> = Result<T, ConfigError>;

pub use crate::kv_index::config::{
    parse_pd_kv_events, parse_static_kv_event_sources, KvSourceOptions,
};

/// Parse the explicit Prefill-to-Decode allowlist for a mixed P/D pool.
pub fn parse_pd_worker_pairs(
    specs: &[String],
) -> ConfigResult<std::collections::HashMap<String, Vec<String>>> {
    let mut pairs = std::collections::HashMap::new();
    for spec in specs {
        let invalid = || ConfigError::ValidationFailed {
            reason: format!(
                "--pd-pair expects one PREFILL=DECODE[,DECODE...] entry per Prefill, got '{spec}'"
            ),
        };
        let (prefill, decodes) = spec.split_once('=').ok_or_else(invalid)?;
        let prefill = prefill.trim();
        let decodes: Vec<_> = decodes
            .split(',')
            .map(str::trim)
            .map(str::to_owned)
            .collect();
        if prefill.is_empty()
            || decodes.iter().any(String::is_empty)
            || pairs.insert(prefill.to_owned(), decodes).is_some()
        {
            return Err(invalid());
        }
    }
    Ok(pairs)
}
