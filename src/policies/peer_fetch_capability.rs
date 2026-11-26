//! Turns the operator-declared capability table (`StaticTransferCapability`,
//! defined and validated in `config`) into planner [`TransferCapability`]s,
//! each with the side-channel host the operator pinned for that worker.

use std::collections::HashMap;
use std::sync::Arc;

use super::peer_fetch::{normalize_host, TransferCapability};
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

/// One worker's declared capability and its pinned side-channel host
/// (normalized by [`normalize_host`]), if any.
pub type StaticCapability = (TransferCapability, Option<Arc<str>>);

/// Convert a whole `worker_url -> capability` table, failing on the first
/// invalid entry with the worker named.
pub fn parse_static_capabilities(
    table: &HashMap<String, StaticTransferCapability>,
) -> Result<HashMap<String, StaticCapability>, String> {
    validate_static_transfer_capabilities(table)?;
    table
        .iter()
        .map(|(url, config)| {
            TransferCapability::try_from(config)
                .map(|capability| {
                    let pin = config
                        .side_channel_host
                        .as_deref()
                        .map(|host| Arc::from(normalize_host(host)));
                    (url.clone(), (capability, pin))
                })
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
            side_channel_host: None,
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
        assert!(parsed["http://a"].0.can_send_to(&parsed["http://b"].0));
        assert!(!parsed["http://a"].0.can_send_to(&parsed["http://c"].0));

        let mut import_only = entry("fp1");
        import_only.can_export = false;
        let parsed = parse_static_capabilities(&HashMap::from([
            ("http://a".to_string(), import_only),
            ("http://b".to_string(), entry("fp1")),
        ]))
        .unwrap();
        assert!(!parsed["http://a"].0.can_send_to(&parsed["http://b"].0));
        assert!(parsed["http://b"].0.can_send_to(&parsed["http://a"].0));
    }

    #[test]
    fn a_pinned_side_channel_host_is_normalized() {
        let mut pinned = entry("fp");
        pinned.side_channel_host = Some(" [2001:DB8::1] ".into());
        let parsed = parse_static_capabilities(&HashMap::from([
            ("http://a".to_owned(), pinned),
            ("http://b".to_owned(), entry("fp")),
        ]))
        .unwrap();
        assert_eq!(parsed["http://a"].1.as_deref(), Some("2001:db8::1"));
        assert_eq!(parsed["http://b"].1, None);
    }
}
