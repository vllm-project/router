//! Declaring a worker's [`TransferCapability`] in router config.
//!
//! Until engines report their own capability (later work), the operator
//! states it per worker URL. Nothing here is verified against the engine: a
//! wrong entry makes the planner's `Incompatible` check wrong, so a fingerprint
//! must be identical on workers with the same model, KV dtype, TP/PP shape,
//! block size and quantization, and different otherwise. Shadow mode does not
//! need this at all when `require_capability` is false.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::kv_transfer::TransferCapability;

/// One worker's declared capability, as written in config.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StaticTransferCapability {
    #[serde(default = "default_true")]
    pub can_export: bool,
    #[serde(default = "default_true")]
    pub can_import: bool,
    /// Transport name, e.g. "nixl".
    pub connector: String,
    pub connector_version: String,
    /// Operator-chosen label; equal labels mean compatible KV layouts.
    pub layout_fingerprint: String,
}

fn default_true() -> bool {
    true
}

impl StaticTransferCapability {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("connector", &self.connector),
            ("connector_version", &self.connector_version),
            ("layout_fingerprint", &self.layout_fingerprint),
        ] {
            if value.trim().is_empty() {
                return Err(format!("{name} is empty"));
            }
        }
        Ok(())
    }
}

impl TryFrom<&StaticTransferCapability> for TransferCapability {
    type Error = String;

    fn try_from(config: &StaticTransferCapability) -> Result<Self, String> {
        config.validate()?;
        Ok(TransferCapability {
            can_export: config.can_export,
            can_import: config.can_import,
            connector: config.connector.trim().into(),
            connector_version: config.connector_version.trim().into(),
            layout_fingerprint: config.layout_fingerprint.trim().into(),
        })
    }
}

/// Validate a whole `worker_url -> capability` table up front, so a typo
/// fails at startup rather than silently disabling transfers for one worker.
pub fn parse_static_capabilities(
    table: &HashMap<String, StaticTransferCapability>,
) -> Result<HashMap<String, TransferCapability>, String> {
    table
        .iter()
        .map(|(url, config)| {
            TransferCapability::try_from(config)
                .map(|capability| (url.clone(), capability))
                .map_err(|error| format!("kv_transfer capability for {url}: {error}"))
        })
        .collect()
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    fn entry(fingerprint: &str) -> StaticTransferCapability {
        StaticTransferCapability {
            can_export: true,
            can_import: true,
            connector: "nixl".into(),
            connector_version: "1".into(),
            layout_fingerprint: fingerprint.into(),
        }
    }

    #[test]
    fn deserializes_with_defaults_and_rejects_unknown_fields() {
        let parsed: StaticTransferCapability = serde_json::from_str(
            r#"{"connector":"nixl","connector_version":"1","layout_fingerprint":"qwen3-8b-bf16-tp1"}"#,
        )
        .unwrap();
        assert!(parsed.can_export && parsed.can_import);

        assert!(serde_json::from_str::<StaticTransferCapability>(
            r#"{"connector":"nixl","connector_version":"1","layout_fingerprint":"x","can_exprot":true}"#,
        )
        .is_err());
    }

    #[test]
    fn empty_fields_are_rejected_with_the_worker_named() {
        let mut bad = entry("fp");
        bad.connector = "  ".into();
        let table = HashMap::from([("http://a".to_string(), bad)]);
        let error = parse_static_capabilities(&table).unwrap_err();
        assert!(error.contains("http://a") && error.contains("connector"));
    }

    #[test]
    fn matching_fingerprints_send_and_mismatches_do_not() {
        let table = HashMap::from([
            ("http://a".to_string(), entry("fp1")),
            ("http://b".to_string(), entry("fp1")),
            ("http://c".to_string(), entry("fp2")),
        ]);
        let parsed = parse_static_capabilities(&table).unwrap();
        assert!(parsed["http://a"].can_send_to(&parsed["http://b"]));
        assert!(!parsed["http://a"].can_send_to(&parsed["http://c"]));

        let mut import_only = entry("fp1");
        import_only.can_export = false;
        let parsed = parse_static_capabilities(&HashMap::from([
            ("http://a".to_string(), import_only),
            ("http://b".to_string(), entry("fp1")),
        ]))
        .unwrap();
        assert!(!parsed["http://a"].can_send_to(&parsed["http://b"]));
        assert!(parsed["http://b"].can_send_to(&parsed["http://a"]));
    }
}
