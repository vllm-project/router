//! Turns the operator-declared capability table (`StaticTransferCapability`,
//! defined and validated in `config`) into planner [`TransferCapability`]s.

use std::collections::HashMap;

use super::peer_fetch::TransferCapability;
use crate::config::types::StaticTransferCapability;
use crate::config::validate_static_transfer_capabilities;

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

/// Convert a whole `worker_url -> capability` table, failing on the first
/// invalid entry with the worker named.
pub fn parse_static_capabilities(
    table: &HashMap<String, StaticTransferCapability>,
) -> Result<HashMap<String, TransferCapability>, String> {
    validate_static_transfer_capabilities(table)?;
    table
        .iter()
        .map(|(url, config)| {
            TransferCapability::try_from(config)
                .map(|capability| (url.clone(), capability))
                .map_err(|error| format!("peer_fetch capability for {url}: {error}"))
        })
        .collect()
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
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
            r#"{"connector":"nixl","connector_version":"1","layout_fingerprint":"x","unknown_field":true}"#,
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
