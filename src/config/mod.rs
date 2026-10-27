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

/// Parse repeatable `WORKER_URL=ENDPOINT[,ENDPOINT...]` KV event sources.
pub fn parse_static_kv_event_sources(
    specs: &[String],
    replay_specs: &[String],
    block_size: u32,
    model: Option<&str>,
) -> ConfigResult<std::collections::HashMap<String, StaticKvEventSources>> {
    if specs.is_empty() {
        if !replay_specs.is_empty() {
            return Err(ConfigError::ValidationFailed {
                reason: "--kv-event-replay-source requires a matching --kv-event-source"
                    .to_string(),
            });
        }
        return Ok(std::collections::HashMap::new());
    }
    let model = model.unwrap_or_default().trim();

    let mut sources = std::collections::HashMap::new();
    for spec in specs {
        let (worker_url, endpoints) = parse_event_source_spec(spec, "kv-event-source")?;
        let source = StaticKvEventSources {
            endpoints,
            replay_endpoints: None,
            topic: String::new(),
            hwm: None,
            source_contract: StaticKvEventSourceContract::Generic,
            block_size,
            model: model.to_owned(),
        };
        if sources.insert(worker_url.to_owned(), source).is_some() {
            return Err(ConfigError::ValidationFailed {
                reason: format!("duplicate --kv-event-source worker URL '{worker_url}'"),
            });
        }
    }
    for spec in replay_specs {
        let (worker_url, endpoints) = parse_event_source_spec(spec, "kv-event-replay-source")?;
        let source = sources
            .get_mut(worker_url)
            .ok_or_else(|| ConfigError::ValidationFailed {
                reason: format!(
                    "--kv-event-replay-source worker '{worker_url}' has no --kv-event-source"
                ),
            })?;
        if source.replay_endpoints.replace(endpoints).is_some() {
            return Err(ConfigError::ValidationFailed {
                reason: format!("duplicate --kv-event-replay-source worker URL '{worker_url}'"),
            });
        }
    }
    for (worker, source) in &sources {
        source.validate(worker)?;
    }
    Ok(sources)
}

fn parse_event_source_spec<'a>(spec: &'a str, flag: &str) -> ConfigResult<(&'a str, Vec<String>)> {
    let (worker, endpoints) =
        spec.split_once('=')
            .ok_or_else(|| ConfigError::ValidationFailed {
                reason: format!("--{flag} expects WORKER_URL=ENDPOINT[,ENDPOINT...], got '{spec}'"),
            })?;
    let worker = worker.trim();
    let endpoints: Vec<_> = endpoints
        .split(',')
        .map(str::trim)
        .filter(|endpoint| !endpoint.is_empty())
        .map(str::to_owned)
        .collect();
    if worker.is_empty() || endpoints.is_empty() {
        return Err(ConfigError::ValidationFailed {
            reason: format!("--{flag} needs a worker URL and at least one endpoint, got '{spec}'"),
        });
    }
    Ok((worker, endpoints))
}

impl StaticKvEventSources {
    pub(super) fn validate(&self, worker: &str) -> ConfigResult<()> {
        if worker.trim().is_empty() || self.model.trim().is_empty() {
            return Err(ConfigError::ValidationFailed {
                reason: "static KV event sources require worker URLs and model identity"
                    .to_string(),
            });
        }
        if self.block_size == 0 || self.endpoints.is_empty() {
            return Err(ConfigError::ValidationFailed {
                reason: format!(
                    "static KV event source for '{worker}' needs a nonzero block size and at least one endpoint"
                ),
            });
        }
        if self.hwm.is_some_and(|hwm| hwm < 0) {
            return Err(ConfigError::ValidationFailed {
                reason: format!("static KV event HWM must be nonnegative for '{worker}'"),
            });
        }
        if self.source_contract == StaticKvEventSourceContract::HttpDenseDp1 {
            crate::kv_index::canonical_worker(worker)
                .map_err(|reason| ConfigError::ValidationFailed { reason })?;
            if self.endpoints.len() != 1 {
                return Err(ConfigError::ValidationFailed {
                    reason: format!(
                        "static HTTP Dense KV source requires one DP1 endpoint for '{worker}'"
                    ),
                });
            }
        }
        if self
            .replay_endpoints
            .as_ref()
            .is_some_and(|replay| replay.len() != self.endpoints.len())
        {
            return Err(ConfigError::ValidationFailed {
                reason: format!(
                    "static KV replay endpoint count for '{worker}' must match its event endpoint count"
                ),
            });
        }
        for endpoint in self
            .endpoints
            .iter()
            .chain(self.replay_endpoints.iter().flatten())
        {
            crate::kv_index::canonical_endpoint(endpoint)
                .map_err(|reason| ConfigError::ValidationFailed { reason })?;
        }
        Ok(())
    }
}

/// Snapshot compatibility input into the existing source configuration before
/// it is shared with policies or supervisors. Registry keys remain untouched.
pub(crate) fn normalize_static_kv_event_sources(
    policy: &mut PolicyConfig,
    legacy: Option<&str>,
) -> ConfigResult<()> {
    use std::collections::{HashMap, HashSet};

    let PolicyConfig::KvAware {
        static_kv_event_sources: configured,
        ..
    } = policy
    else {
        return Ok(());
    };
    let mut legacy_sources: HashMap<String, StaticKvEventSources> = if let Some(raw) = legacy {
        // The legacy input denied unknown fields. Keep that boundary without
        // narrowing the generic JSON representation or adding a second DTO.
        let value: serde_json::Value =
            serde_json::from_str(raw).map_err(|error| ConfigError::ValidationFailed {
                reason: format!("static KV sources: {error}"),
            })?;
        let entries = value
            .as_object()
            .ok_or_else(|| ConfigError::ValidationFailed {
                reason: "static KV sources must be an object keyed by worker URL".into(),
            })?;
        for source in entries.values().filter_map(serde_json::Value::as_object) {
            if let Some(field) = source
                .keys()
                .filter(|field| {
                    !matches!(
                        field.as_str(),
                        "endpoints" | "replay_endpoints" | "topic" | "hwm" | "block_size" | "model"
                    )
                })
                .min()
            {
                return Err(ConfigError::ValidationFailed {
                    reason: format!("static KV sources: unknown field '{field}'"),
                });
            }
        }
        serde_json::from_value(value).map_err(|error| ConfigError::ValidationFailed {
            reason: format!("static KV sources: {error}"),
        })?
    } else {
        HashMap::new()
    };
    if !configured.is_empty() && !legacy_sources.is_empty() {
        return Err(ConfigError::IncompatibleConfig {
            reason: "explicit static KV sources and VLLM_ROUTER_KV_EVENT_SOURCES cannot both be nonempty".into(),
        });
    }
    for source in legacy_sources.values_mut() {
        source.source_contract = StaticKvEventSourceContract::HttpDenseDp1;
    }
    let mut sources = if legacy_sources.is_empty() {
        configured.as_ref().clone()
    } else {
        legacy_sources
    };
    let mut worker_origins = HashSet::new();
    let mut live_streams = HashSet::new();
    let mut replay_endpoints = HashSet::new();
    for (worker, source) in &mut sources {
        source.validate(worker)?;
        for endpoint in source
            .endpoints
            .iter_mut()
            .chain(source.replay_endpoints.iter_mut().flatten())
        {
            *endpoint = crate::kv_index::canonical_endpoint(endpoint)
                .map_err(|reason| ConfigError::ValidationFailed { reason })?;
        }
        if source.source_contract == StaticKvEventSourceContract::HttpDenseDp1 {
            let origin = crate::kv_index::canonical_worker(worker)
                .map_err(|reason| ConfigError::ValidationFailed { reason })?
                .0;
            if !worker_origins.insert(origin) {
                return Err(ConfigError::ValidationFailed {
                    reason: "static KV worker origins must be independent".into(),
                });
            }
            if !live_streams.insert((source.endpoints[0].clone(), source.topic.clone())) {
                return Err(ConfigError::ValidationFailed {
                    reason: "static live KV streams must be independent".into(),
                });
            }
            if let Some(replay) = source
                .replay_endpoints
                .as_ref()
                .and_then(|replay| replay.first())
            {
                if !replay_endpoints.insert(replay.clone()) {
                    return Err(ConfigError::ValidationFailed {
                        reason: "static KV replay endpoints must be independent".into(),
                    });
                }
            }
        }
    }
    **configured = sources;
    Ok(())
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod static_kv_event_source_tests {
    use super::*;

    fn policy(sources: std::collections::HashMap<String, StaticKvEventSources>) -> PolicyConfig {
        PolicyConfig::KvAware {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
            static_kv_event_sources: Box::new(sources),
        }
    }

    fn configured(
        policy: &PolicyConfig,
    ) -> &std::collections::HashMap<String, StaticKvEventSources> {
        let PolicyConfig::KvAware {
            static_kv_event_sources,
            ..
        } = policy
        else {
            panic!("expected KV policy");
        };
        static_kv_event_sources
    }

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
    fn static_event_sources_require_model_and_reachable_endpoints() {
        let missing_model = vec!["http://worker:8000=tcp://10.0.0.1:5557".to_string()];
        assert!(parse_static_kv_event_sources(&missing_model, &[], 16, None).is_err());
        for endpoint in ["tcp://*:5557", "tcp://0.0.0.0:5557", "tcp://10.0.0.1:0"] {
            let specs = vec![format!("http://worker:8000={endpoint}")];
            assert!(parse_static_kv_event_sources(&specs, &[], 16, Some("model")).is_err());
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
        assert_eq!(source.hwm, None);
        assert_eq!(source.source_contract, StaticKvEventSourceContract::Generic);
        for worker in ["grpc://worker:50051", "http://worker:8000/engine"] {
            let mut policy = policy(std::collections::HashMap::from([(
                worker.into(),
                source.clone(),
            )]));
            normalize_static_kv_event_sources(&mut policy, Some("{}")).unwrap();
            assert_eq!(configured(&policy)[worker], source);
        }
    }

    #[test]
    fn legacy_sources_normalize_once_into_the_canonical_config() {
        let key = "HTTP://WORKER:8000/";
        let raw = serde_json::json!({key: {
            "endpoints": ["tcp://[::ffff:127.0.0.1]:5557/"],
            "replay_endpoints": ["tcp://[::1]:5558/"],
            "topic": "kv/exact", "hwm": 1000, "block_size": 8, "model": "model"
        }})
        .to_string();
        let mut policy = policy(Default::default());
        normalize_static_kv_event_sources(&mut policy, Some(&raw)).unwrap();
        let source = &configured(&policy)[key];
        assert_eq!(source.endpoints, ["tcp://127.0.0.1:5557"]);
        assert_eq!(
            source.replay_endpoints.as_ref().unwrap(),
            &["tcp://[::1]:5558"]
        );
        assert_eq!(source.topic, "kv/exact");
        assert_eq!(source.hwm, Some(1000));
        assert_eq!(
            source.source_contract,
            StaticKvEventSourceContract::HttpDenseDp1
        );
        assert_eq!(configured(&policy).len(), 1);
        let saved = serde_json::to_value(&policy).unwrap();
        normalize_static_kv_event_sources(&mut policy, None).unwrap();
        assert_eq!(serde_json::to_value(&policy).unwrap(), saved);
    }

    #[test]
    fn explicit_and_legacy_sources_have_no_silent_precedence() {
        let explicit = parse_static_kv_event_sources(
            &["grpc://worker:50051=tcp://127.0.0.1:5557".into()],
            &[],
            16,
            Some("model"),
        )
        .unwrap();
        let mut policy = policy(explicit);
        let saved = serde_json::to_value(&policy).unwrap();
        normalize_static_kv_event_sources(&mut policy, Some("{}")).unwrap();
        assert_eq!(serde_json::to_value(&policy).unwrap(), saved);
        let legacy = r#"{"http://other:8000":{"endpoints":["tcp://127.0.0.1:5559"],"block_size":16,"model":"model"}}"#;
        assert!(matches!(
            normalize_static_kv_event_sources(&mut policy, Some(legacy)),
            Err(ConfigError::IncompatibleConfig { .. })
        ));
        assert_eq!(serde_json::to_value(&policy).unwrap(), saved);
        let mut ordinary = PolicyConfig::RoundRobin;
        normalize_static_kv_event_sources(&mut ordinary, Some("invalid JSON")).unwrap();
    }

    #[test]
    fn legacy_invalid_configuration_is_rejected_at_the_snapshot_boundary() {
        let valid = serde_json::json!({"http://worker:8000": {
            "endpoints": ["tcp://127.0.0.1:5557"], "block_size": 16, "model": "model"
        }});
        for raw in [
            "invalid JSON".to_owned(),
            "[]".to_owned(),
            "null".to_owned(),
        ] {
            let mut policy = policy(Default::default());
            assert!(normalize_static_kv_event_sources(&mut policy, Some(&raw)).is_err());
            assert!(configured(&policy).is_empty());
        }
        for (field, value) in [
            ("source_contract", serde_json::json!("generic")),
            ("unknown", serde_json::json!(true)),
            ("hwm", serde_json::json!(-1)),
            ("endpoints", serde_json::json!(["tcp://0.0.0.0:5557"])),
            (
                "endpoints",
                serde_json::json!(["tcp://127.0.0.1:5557", "tcp://127.0.0.1:5558"]),
            ),
        ] {
            let mut invalid = valid.clone();
            invalid["http://worker:8000"][field] = value;
            let mut policy = policy(Default::default());
            assert!(
                normalize_static_kv_event_sources(&mut policy, Some(&invalid.to_string())).is_err(),
                "{invalid}"
            );
            assert!(configured(&policy).is_empty());
        }
        let mut empty = policy(Default::default());
        normalize_static_kv_event_sources(&mut empty, Some("{}")).unwrap();
        assert!(configured(&empty).is_empty());
    }

    #[test]
    fn legacy_bindings_reject_aliases_and_shared_replay_without_rewriting_keys() {
        let entry = |publisher: &str, replay: &str, topic: &str| {
            serde_json::json!({
                "endpoints": [publisher], "replay_endpoints": [replay],
                "block_size": 16, "model": "model", "topic": topic
            })
        };
        for (other_worker, other_publisher, other_replay, topic, message) in [
            (
                "HTTP://WORKER:8000/",
                "tcp://127.0.0.1:5559",
                "tcp://127.0.0.1:5560",
                "other",
                "worker origins",
            ),
            (
                "http://other:8000",
                "tcp://[::ffff:127.0.0.1]:5557",
                "tcp://127.0.0.1:5560",
                "kv",
                "live KV streams",
            ),
            (
                "http://other:8000",
                "tcp://127.0.0.1:5559",
                "tcp://[::ffff:127.0.0.1]:5558",
                "other",
                "replay endpoints",
            ),
        ] {
            let raw = serde_json::json!({
                "http://worker:8000": entry("tcp://127.0.0.1:5557", "tcp://127.0.0.1:5558", "kv"),
                other_worker: entry(other_publisher, other_replay, topic)
            })
            .to_string();
            let mut policy = policy(Default::default());
            assert!(normalize_static_kv_event_sources(&mut policy, Some(&raw))
                .unwrap_err()
                .to_string()
                .contains(message));
        }
        for worker in ["grpc://worker:50051", "http://worker:8000/engine"] {
            let raw = serde_json::json!({worker: entry("tcp://127.0.0.1:5557", "tcp://127.0.0.1:5558", "kv")}).to_string();
            assert!(
                normalize_static_kv_event_sources(&mut policy(Default::default()), Some(&raw))
                    .is_err()
            );
        }
    }
}
