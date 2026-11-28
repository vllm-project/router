#![cfg(all(test, feature = "cpu", feature = "ready"))]

use super::*;
use std::collections::HashMap;

#[test]
fn parses_static_event_sources_with_shared_model_and_block_size() {
    let specs = vec![
        "http://worker-a:8000=tcp://10.0.0.1:5557,tcp://10.0.0.1:5559".to_string(),
        "grpc://worker-b:50051=tcp://10.0.0.2:5557".to_string(),
    ];
    let replay_specs =
        vec!["http://worker-a:8000=tcp://10.0.0.1:5558,tcp://10.0.0.1:5560".to_string()];
    let sources =
        parse_static_kv_event_sources(&specs, &replay_specs, 32, Some("model-a")).unwrap();
    assert_eq!(sources.len(), 2);
    assert_eq!(sources["http://worker-a:8000"].block_size, 32);
    assert_eq!(sources["http://worker-a:8000"].model, "model-a");
    assert_eq!(sources["http://worker-a:8000"].endpoints.len(), 2);
    assert_eq!(
        sources["http://worker-a:8000"]
            .replay_endpoints
            .as_ref()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(sources["grpc://worker-b:50051"].endpoints.len(), 1);
}

#[test]
fn pd_event_sources_reject_duplicate_workers() {
    let specs = [
        "http://worker:8000=tcp://10.0.0.1:5557".to_owned(),
        "http://worker:8000=tcp://10.0.0.1:9999".to_owned(),
    ];
    let error = parse_pd_kv_events(&specs).unwrap_err().to_string();
    assert!(
        error.contains("duplicate --pd-kv-events worker URL"),
        "{error}"
    );
}

#[test]
fn static_event_sources_require_model_and_reachable_endpoints() {
    let missing_model = vec!["http://worker:8000=tcp://10.0.0.1:5557".to_string()];
    assert!(parse_static_kv_event_sources(&missing_model, &[], 16, None).is_err());
    for endpoint in [
        "tcp://*:5557",
        "tcp://0.0.0.0:5557",
        "tcp://[::]:5557",
        "tcp://10.0.0.1:0",
    ] {
        let specs = vec![format!("http://worker:8000={endpoint}")];
        assert!(parse_static_kv_event_sources(&specs, &[], 16, Some("model")).is_err());
    }
    for (field, value, reason) in [
        ("block_size", serde_json::json!(0), "nonzero block size"),
        ("endpoints", serde_json::json!([]), "at least one endpoint"),
        (
            "replay_endpoints",
            serde_json::json!([]),
            "replay endpoint count",
        ),
    ] {
        let mut value_with_invalid_field = serde_json::json!({
            "endpoints": ["tcp://127.0.0.1:5557"], "block_size": 16, "model": "model"
        });
        value_with_invalid_field[field] = value;
        let source: StaticKvEventSources =
            serde_json::from_value(value_with_invalid_field).unwrap();
        let error = source
            .validate("http://worker:8000")
            .unwrap_err()
            .to_string();
        assert!(error.contains(reason), "{field}: {error}");
    }
}

#[test]
fn generic_static_sources_keep_json_defaults_and_multirank_transports() {
    let source: StaticKvEventSources = serde_json::from_value(serde_json::json!({
        "endpoints": ["tcp://127.0.0.1:5557", "tcp://127.0.0.1:5559"],
        "block_size": 32, "model": "model", "future_generic_field": true
    }))
    .unwrap();
    assert!(source.topic.is_empty());
    for worker in ["grpc://worker:50051", "http://worker:8000/engine"] {
        source.validate(worker).unwrap();
    }
    assert!(source.validate("http://worker:8000@1").is_err());
}

/// A layout without declared endpoints is legal: the declaration still
/// fixes the block size and the model identity, and discovery supplies the
/// publishers.
#[test]
fn hybrid_layout_without_endpoints_validates() {
    let layout: crate::kv_index::CacheCapabilities = serde_json::from_value(serde_json::json!({
        "schema_version": 1,
        "ranks": [{
            "rank": 0, "model": "m", "weight_version": "v1",
            "publisher_topic": "kv-hybrid", "instance_topic": true,
            "hash_algorithm": "sha256_cbor",
            "root_hash": "4e1195df020de59e0d65a33a4279f1183e7ae4e5d980e309f8b55adff2e61c3e",
            "full_hashes": true, "hash_block_size": 64,
            "hash_lookahead_tokens": 0, "speculative_tokens": 0,
            "prefix_caching": true,
            "groups": [
                {"group_idx": 0, "kind": "mamba", "spec_type": "MambaSpec",
                 "block_size": 64, "prefix_cacheable": true,
                 "prefix_replay_tokens": 0, "mamba_cache_mode": "align"},
                {"group_idx": 1, "kind": "full_attention",
                 "spec_type": "FullAttentionSpec", "block_size": 64,
                 "prefix_cacheable": true, "prefix_replay_tokens": 0}
            ]
        }]
    }))
    .unwrap();
    let source: StaticKvEventSources = serde_json::from_value(serde_json::json!({
        "endpoints": [], "block_size": 64, "model": "m"
    }))
    .unwrap();
    let mut source = source;
    source.hybrid = Some(layout);
    source.validate("http://worker:8000").unwrap();

    let mut mismatched = source.clone();
    mismatched.block_size = 16;
    assert!(mismatched.validate("http://worker:8000").is_err());
    let layout = mismatched.hybrid.as_mut().unwrap();
    layout.ranks.clear();
    assert!(!layout.is_plain_full_attention());
    assert!(mismatched.validate("http://worker:8000").is_err());

    // Without a layout the endpoint list is still mandatory.
    let mut generic = source.clone();
    generic.hybrid = None;
    assert!(generic.validate("http://worker:8000").is_err());
}

#[test]
fn loads_one_kv_layout_for_both_pd_stages() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("layout.json");
    for (prefill, decode) in [
        ("http://prefill:8000", "http://decode:8000"),
        ("grpc://prefill:50051", "grpc://decode:50051"),
    ] {
        let layout = || {
            serde_json::json!({
                "schema_version": 1,
                "ranks": [{
                    "rank": 0, "model": "model", "weight_version": "v1",
                    "hash_algorithm": "sha256_cbor", "hash_seed": "0", "full_hashes": true,
                    "hash_block_size": 16, "hash_lookahead_tokens": 0, "declared_groups": 1,
                    "speculative_tokens": 0, "prefix_caching": true,
                    "groups": [{
                        "group_idx": 0, "kind": "full_attention", "spec_type": "FullAttentionSpec",
                        "block_size": 16, "prefix_cacheable": true, "prefix_replay_tokens": 0
                    }]
                }]
            })
        };
        std::fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({
                (prefill): layout(),
                (decode): layout()
            }))
            .unwrap(),
        )
        .unwrap();
        let mode = RoutingMode::VllmPrefillDecode {
            prefill_urls: vec![(prefill.to_owned(), None)],
            decode_urls: vec![decode.to_owned()],
            discovery_address: None,
        };
        let policy: PolicyConfig =
            serde_json::from_value(serde_json::json!({"type": "kv_aware"})).unwrap();
        let mut config = RouterConfig::new(mode, policy);
        apply_kv_layout_file(
            &path,
            &config.mode,
            &mut HashMap::new(),
            &mut config.pd_kv_event_sources,
        )
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.pd_kv_event_sources.len(), 2);
        for worker in [prefill, decode] {
            let source = &config.pd_kv_event_sources[worker];
            assert_eq!(source.model.as_deref(), Some("model"));
            assert_eq!(source.block_size, 16);
            assert!(source.hybrid.is_some());
        }

        std::fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({
                (prefill): {"compatible_decode_urls": [decode]}
            }))
            .unwrap(),
        )
        .unwrap();
        let mut sources = HashMap::new();
        assert!(
            apply_kv_layout_file(&path, &config.mode, &mut HashMap::new(), &mut sources)
                .unwrap_err()
                .to_string()
                .contains("belongs in the KV event source or P/D routing options")
        );
        assert!(sources.is_empty());
    }
}

#[test]
fn rejects_empty_worker_layout_before_creating_a_source() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("layout.json");
    std::fs::write(
        &path,
        r#"{"http://worker:8000":{"schema_version":1,"ranks":[]}}"#,
    )
    .unwrap();
    let mode = RoutingMode::Regular {
        worker_urls: vec!["http://worker:8000".into()],
    };
    assert!(
        apply_kv_layout_file(&path, &mode, &mut HashMap::new(), &mut HashMap::new())
            .unwrap_err()
            .to_string()
            .contains("http://worker:8000' must declare at least one rank")
    );
}
