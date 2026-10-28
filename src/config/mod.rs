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

/// Parse repeatable static PD KV-source specifications for both launchers.
pub fn parse_pd_kv_events(
    specs: &[String],
) -> ConfigResult<std::collections::HashMap<String, PdKvEventSources>> {
    let mut sources = std::collections::HashMap::new();
    for spec in specs {
        let (url, endpoints) = parse_event_source_spec(spec, "pd-kv-events")?;
        if sources
            .insert(
                url.to_owned(),
                PdKvEventSources {
                    endpoints,
                    replay_endpoints: None,
                    block_size: 16,
                    model: None,
                },
            )
            .is_some()
        {
            return Err(ConfigError::ValidationFailed {
                reason: format!("duplicate --pd-kv-events worker URL '{url}'"),
            });
        }
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
            if !is_reachable_tcp_endpoint(endpoint) {
                return Err(ConfigError::ValidationFailed {
                    reason: format!(
                        "static KV event endpoint must be reachable tcp://HOST:PORT: '{endpoint}'"
                    ),
                });
            }
        }
        Ok(())
    }
}

fn is_reachable_tcp_endpoint(endpoint: &str) -> bool {
    url::Url::parse(endpoint).is_ok_and(|parsed| {
        parsed.scheme() == "tcp"
            && parsed.host_str().is_some_and(|host| {
                let host = host.trim_start_matches('[').trim_end_matches(']');
                host != "*"
                    && !host
                        .parse::<std::net::IpAddr>()
                        .is_ok_and(|ip| ip.is_unspecified())
            })
            && parsed.port().is_some_and(|port| port != 0)
    })
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod static_kv_event_source_tests {
    use super::*;

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
    }
}
