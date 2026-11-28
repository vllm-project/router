#![cfg(all(feature = "cpu", feature = "ready"))]

use std::io::Write;
use tempfile::NamedTempFile;
use vllm_router_rs::config::{
    ConfigError, ConnectionMode, HistoryBackend, KvConnector, PolicyConfig, RouterConfig,
    RoutingMode,
};

#[test]
fn loads_minimal_yaml_with_defaults() {
    let mut file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
    writeln!(
        file,
        r#"
mode:
  type: regular
  worker_urls:
    - http://127.0.0.1:8000
    - http://127.0.0.1:8001
"#
    )
    .unwrap();

    let cfg = RouterConfig::from_file(file.path()).unwrap();
    cfg.validate().unwrap();

    assert_eq!(cfg.host, "127.0.0.1");
    assert_eq!(cfg.port, 3001);
    assert_eq!(cfg.connection_mode, ConnectionMode::Http);
    assert_eq!(cfg.history_backend, HistoryBackend::Memory);
    assert_eq!(cfg.kv_connector, KvConnector::Nixl);
    assert_eq!(cfg.retry.max_retries, 5);
    assert_eq!(cfg.circuit_breaker.failure_threshold, 10);
    assert_eq!(cfg.health_check.endpoint, "/health");
    assert!(matches!(cfg.policy, PolicyConfig::Random));
    match cfg.mode {
        RoutingMode::Regular { worker_urls } => {
            assert_eq!(
                worker_urls,
                vec![
                    "http://127.0.0.1:8000".to_string(),
                    "http://127.0.0.1:8001".to_string()
                ]
            );
        }
        other => panic!("unexpected mode: {other:?}"),
    }
}

#[test]
fn loads_yml_and_json_configs_with_partial_nested_sections() {
    let mut yml_file = tempfile::Builder::new().suffix(".yml").tempfile().unwrap();
    writeln!(
        yml_file,
        r#"
mode:
  type: regular
  worker_urls:
    - http://127.0.0.1:8000
policy:
  type: cache_aware
  cache_threshold: 0.7
host: 0.0.0.0
port: 8080
metrics:
  port: 29100
retry:
  max_retries: 3
"#
    )
    .unwrap();

    let cfg = RouterConfig::from_file(yml_file.path()).unwrap();
    cfg.validate().unwrap();
    assert_eq!(cfg.host, "0.0.0.0");
    assert_eq!(cfg.port, 8080);
    assert_eq!(cfg.metrics.as_ref().unwrap().port, 29100);
    assert_eq!(cfg.metrics.as_ref().unwrap().host, "127.0.0.1");
    assert_eq!(cfg.retry.max_retries, 3);
    assert_eq!(cfg.retry.initial_backoff_ms, 50);
    match cfg.policy {
        PolicyConfig::CacheAware {
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            eviction_interval_secs,
            max_tree_size,
        } => {
            assert!((cache_threshold - 0.7).abs() < f32::EPSILON);
            assert_eq!(balance_abs_threshold, 64);
            assert!((balance_rel_threshold - 1.5).abs() < f32::EPSILON);
            assert_eq!(eviction_interval_secs, 120);
            assert_eq!(max_tree_size, 1 << 25);
        }
        other => panic!("expected cache_aware policy, got {other:?}"),
    }

    let mut json_file = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
    writeln!(
        json_file,
        r#"{{
  "mode": {{
    "type": "vllm_prefill_decode",
    "prefill_urls": [["http://127.0.0.1:8100", 9000]],
    "decode_urls": ["http://127.0.0.1:8200"]
  }},
  "policy": {{ "type": "kv_aware" }},
  "port": 9090,
  "kv_connector": "mooncake"
}}"#
    )
    .unwrap();

    let json_cfg = RouterConfig::from_file(json_file.path()).unwrap();
    json_cfg.validate().unwrap();
    assert_eq!(json_cfg.port, 9090);
    assert_eq!(json_cfg.kv_connector, KvConnector::Mooncake);
    assert!(matches!(json_cfg.policy, PolicyConfig::KvAware { .. }));
    assert!(json_cfg.mode.is_pd_mode());
}

#[test]
fn loads_extensionless_files_as_json_or_yaml() {
    let mut json_no_ext = NamedTempFile::new().unwrap();
    writeln!(
        json_no_ext,
        r#"{{"mode": {{"type": "regular", "worker_urls": ["http://127.0.0.1:8000"]}}, "policy": {{"type": "consistent_hash"}}, "port": 3100}}"#
    )
    .unwrap();
    let cfg_json = RouterConfig::from_file(json_no_ext.path()).unwrap();
    assert_eq!(cfg_json.port, 3100);
    assert!(matches!(
        cfg_json.policy,
        PolicyConfig::ConsistentHash { virtual_nodes: 160 }
    ));

    let mut yaml_no_ext = NamedTempFile::new().unwrap();
    writeln!(
        yaml_no_ext,
        "mode:\n  type: regular\n  worker_urls:\n    - http://127.0.0.1:8000\npolicy:\n  type: power_of_two\nport: 3200"
    )
    .unwrap();
    let cfg_yaml = RouterConfig::from_file(yaml_no_ext.path()).unwrap();
    assert_eq!(cfg_yaml.port, 3200);
    assert!(matches!(
        cfg_yaml.policy,
        PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 5
        }
    ));
}

#[test]
fn rejects_missing_empty_malformed_and_invalid_configs() {
    let missing = RouterConfig::from_file("/nonexistent/vllm-router-config-does-not-exist.yaml");
    assert!(matches!(missing, Err(ConfigError::ValidationFailed { .. })));

    let empty = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
    let empty_err = RouterConfig::from_file(empty.path());
    assert!(matches!(
        empty_err,
        Err(ConfigError::ValidationFailed { .. })
    ));

    let mut bad_yaml = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
    writeln!(bad_yaml, "mode: [unclosed").unwrap();
    let parse_err = RouterConfig::from_file(bad_yaml.path());
    assert!(matches!(
        parse_err,
        Err(ConfigError::ValidationFailed { .. })
    ));

    let mut invalid_cfg_file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
    writeln!(
        invalid_cfg_file,
        "mode:\n  type: vllm_prefill_decode\nport: 3000"
    )
    .unwrap();
    let loaded = RouterConfig::from_file(invalid_cfg_file.path()).unwrap();
    assert!(loaded.validate().is_err());
}
