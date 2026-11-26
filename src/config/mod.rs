pub mod types;
pub mod validation;

pub use types::*;
pub use validation::*;

/// Configuration errors
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
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

/// Check a whole `worker_url -> capability` table up front, so a typo fails at
/// startup rather than silently disabling transfers for one worker.
pub fn validate_static_transfer_capabilities(
    table: &std::collections::HashMap<String, StaticTransferCapability>,
) -> Result<(), String> {
    table.iter().try_for_each(|(url, capability)| {
        capability
            .validate()
            .map_err(|error| format!("peer_fetch capability for {url}: {error}"))
    })
}

/// Parse repeatable `WORKER_URL=JSON` peer-fetch capabilities, where JSON is a
/// `StaticTransferCapability` object. The URL ends at the first `=`.
pub fn parse_static_transfer_capabilities(
    specs: &[String],
) -> ConfigResult<std::collections::HashMap<String, StaticTransferCapability>> {
    let invalid = |reason: String| ConfigError::ValidationFailed { reason };
    let mut table = std::collections::HashMap::new();
    for spec in specs {
        let (worker_url, json) = spec
            .split_once('=')
            .map(|(url, json)| (url.trim(), json.trim()))
            .filter(|(url, json)| !url.is_empty() && !json.is_empty())
            .ok_or_else(|| {
                invalid(format!(
                    "--peer-fetch-capability '{spec}' must be WORKER_URL=JSON"
                ))
            })?;
        let capability: StaticTransferCapability = serde_json::from_str(json).map_err(|error| {
            invalid(format!(
                "--peer-fetch-capability for {worker_url}: invalid JSON: {error}"
            ))
        })?;
        if table.insert(worker_url.to_owned(), capability).is_some() {
            return Err(invalid(format!(
                "duplicate --peer-fetch-capability worker URL '{worker_url}'"
            )));
        }
    }
    validate_static_transfer_capabilities(&table).map_err(invalid)?;
    Ok(table)
}

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

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod static_transfer_capability_tests {
    use super::*;

    #[test]
    fn static_transfer_capabilities_parse_from_cli_specs() {
        let specs = [
            r#"http://a:8000={"connector":"nixl","connector_version":"1","layout_fingerprint":"fp"}"#
                .to_owned(),
            r#"http://b:8000={"can_export":false,"connector":"nixl","connector_version":"1","layout_fingerprint":"fp"}"#
                .to_owned(),
        ];
        let table = parse_static_transfer_capabilities(&specs).unwrap();
        assert_eq!(table.len(), 2);
        assert!(table["http://a:8000"].can_export && table["http://a:8000"].can_import);
        assert!(!table["http://b:8000"].can_export);
        assert_eq!(table["http://b:8000"].layout_fingerprint, "fp");
        assert!(parse_static_transfer_capabilities(&[]).unwrap().is_empty());
        assert_eq!(table["http://a:8000"].side_channel_host, None);

        let pinned = parse_static_transfer_capabilities(&[r#"http://a:8000={"connector":"nixl","connector_version":"1","layout_fingerprint":"fp","side_channel_host":"192.168.7.1"}"#.to_owned()]).unwrap();
        assert_eq!(
            pinned["http://a:8000"].side_channel_host.as_deref(),
            Some("192.168.7.1")
        );
    }

    #[test]
    fn static_transfer_capabilities_reject_bad_specs() {
        let valid = r#"{"connector":"nixl","connector_version":"1","layout_fingerprint":"fp"}"#;
        for (spec, expected) in [
            ("http://a:8000".to_owned(), "must be WORKER_URL=JSON"),
            (format!("={valid}"), "must be WORKER_URL=JSON"),
            ("http://a:8000=".to_owned(), "must be WORKER_URL=JSON"),
            ("http://a:8000={not json".to_owned(), "invalid JSON"),
            (
                r#"http://a:8000={"connector":"nixl","connector_version":"1","layout_fingerprint":"fp","unknown_field":true}"#
                    .to_owned(),
                "invalid JSON",
            ),
            (
                r#"http://a:8000={"connector":" ","connector_version":"1","layout_fingerprint":"fp"}"#
                    .to_owned(),
                "connector is empty",
            ),
            (
                r#"http://a:8000={"connector":"nixl","connector_version":"1","layout_fingerprint":"fp","side_channel_host":"  "}"#
                    .to_owned(),
                "side_channel_host is empty",
            ),
        ] {
            let error = parse_static_transfer_capabilities(std::slice::from_ref(&spec))
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{spec}: {error}");
        }
        let duplicate = [
            format!("http://a:8000={valid}"),
            format!("http://a:8000={valid}"),
        ];
        let error = parse_static_transfer_capabilities(&duplicate)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("duplicate --peer-fetch-capability"),
            "{error}"
        );
    }
}
