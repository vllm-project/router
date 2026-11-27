//! KV event-source configuration and cache layouts.

use crate::config::{ConfigError, ConfigResult, PolicyConfig, RouterConfig, RoutingMode};
use serde::{Deserialize, Serialize};

mod layout_file;
pub use layout_file::apply_kv_layout_file;

/// Statically configured KV event sources for one prefill worker.
///
/// This is authoritative for engines that cannot report their own sources:
/// the HTTP `/kv_event_sources` route exists only since vLLM 0.31.0, and
/// OpenAI-compatible frontends never serve it. On v0.29/v0.30 a deployment
/// must declare its sources here (or use the gRPC Control RPC).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PdKvEventSources {
    /// ZMQ PUB endpoints, one per data-parallel rank (index = rank).
    pub endpoints: Vec<String>,
    /// Optional ZMQ ROUTER (replay) endpoints, same ordering as `endpoints`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_endpoints: Option<Vec<String>>,
    /// KV cache block size of the engine. Must match the engine's
    /// `--block-size` (vLLM default: 16).
    #[serde(default = "default_pd_kv_block_size")]
    pub block_size: u32,
    /// Model identity for the cache key. Defaults to the worker URL; the
    /// served-model check then relies on `/v1/models`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Router-managed cache layout for this prefill or decode worker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hybrid: Option<crate::kv_index::CacheCapabilities>,
}

fn default_pd_kv_block_size() -> u32 {
    16
}

/// Manually configured KV event publishers for one regular-mode worker.
///
/// Rust struct literals must supply `topic: String::new()` to retain the
/// previous empty subscription topic. Serde defaults preserve old JSON
/// configurations, but do not provide Rust source compatibility.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StaticKvEventSources {
    /// One externally reachable ZMQ PUB endpoint per DP rank.
    pub endpoints: Vec<String>,
    /// Optional replay ROUTER endpoint per DP rank for loss recovery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_endpoints: Option<Vec<String>>,
    /// Exact ZMQ subscription topic shared by these ranks.
    #[serde(default)]
    pub topic: String,
    /// KV cache block size used to calculate prefix placement.
    pub block_size: u32,
    /// Shared model identity across workers that serve the same model.
    pub model: String,
    /// Router-managed hybrid layout and per-rank publisher identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hybrid: Option<crate::kv_index::CacheCapabilities>,
}

/// Launch-time inputs for KV event sources and the worker cache layouts.
#[doc(hidden)]
pub struct KvSourceOptions<'a> {
    pub layout: Option<&'a std::path::Path>,
    pub publishers: &'a [String],
    pub pd_publishers: &'a [String],
    pub replay: &'a [String],
    pub topics: &'a [String],
    pub model: Option<&'a str>,
    pub block_size: u32,
}

#[doc(hidden)]
pub struct ConfiguredKvSources {
    pub regular: std::collections::HashMap<String, StaticKvEventSources>,
    pub pd: std::collections::HashMap<String, PdKvEventSources>,
}

impl ConfiguredKvSources {
    pub fn install_regular(
        &mut self,
        policy: &mut PolicyConfig,
        program: Option<&mut crate::config::ProgramSchedulingConfig>,
    ) -> ConfigResult<()> {
        if self.regular.is_empty() {
            return Ok(());
        }
        if let PolicyConfig::KvAware {
            static_kv_event_sources,
            ..
        } = policy
        {
            **static_kv_event_sources = std::mem::take(&mut self.regular);
        } else if let Some(program) = program.filter(|p| {
            p.binding_strategy == crate::program_scheduling::ProgramBindingStrategy::KvAware
        }) {
            program.binding_kv_sources = std::mem::take(&mut self.regular);
        } else {
            return Err(ConfigError::IncompatibleConfig {
                reason: "--kv-event-source requires kv_aware routing or Program binding".into(),
            });
        }
        Ok(())
    }
}

impl KvSourceOptions<'_> {
    pub fn load(&self, mode: &RoutingMode) -> ConfigResult<ConfiguredKvSources> {
        let is_pd = matches!(mode, RoutingMode::VllmPrefillDecode { .. });
        let mut regular = parse_static_kv_event_sources(
            self.publishers,
            if is_pd { &[] } else { self.replay },
            self.block_size,
            self.model,
        )?;
        let mut pd = parse_pd_kv_events(self.pd_publishers)?;
        for source in pd.values_mut() {
            source.block_size = self.block_size;
            source.model = self.model.map(str::to_owned);
        }
        if is_pd {
            for spec in self.replay {
                let (worker, endpoints) = parse_event_source_spec(spec, "kv-event-replay-source")?;
                let source = pd
                    .get_mut(worker)
                    .ok_or_else(|| ConfigError::ValidationFailed {
                        reason: format!(
                            "--kv-event-replay-source for '{worker}' requires --pd-kv-events"
                        ),
                    })?;
                if source.replay_endpoints.replace(endpoints).is_some() {
                    return Err(ConfigError::ValidationFailed {
                        reason: format!("duplicate --kv-event-replay-source worker URL '{worker}'"),
                    });
                }
            }
        }
        if let Some(path) = self.layout {
            apply_kv_layout_file(path, mode, &mut regular, &mut pd)?;
        }
        let mut seen = std::collections::HashSet::new();
        for spec in self.topics {
            let (worker, topics) = parse_event_source_spec(spec, "kv-event-topic")?;
            if !seen.insert(worker) {
                return Err(ConfigError::ValidationFailed {
                    reason: format!("duplicate --kv-event-topic worker URL '{worker}'"),
                });
            }
            let missing = || ConfigError::ValidationFailed {
                reason: format!(
                    "--kv-event-topic for '{worker}' requires a static event publisher"
                ),
            };
            let (layout, model, size, ranks) = if is_pd {
                let source = pd.get_mut(worker).ok_or_else(missing)?;
                (
                    &mut source.hybrid,
                    source.model.as_deref().unwrap_or_default(),
                    source.block_size,
                    source.endpoints.len(),
                )
            } else {
                let source = regular.get_mut(worker).ok_or_else(missing)?;
                (
                    &mut source.hybrid,
                    source.model.as_str(),
                    source.block_size,
                    source.endpoints.len(),
                )
            };
            if ranks == 0 || topics.len() != ranks || model.trim().is_empty() {
                return Err(ConfigError::ValidationFailed {
                    reason: format!("static event source '{worker}' needs --kv-event-model and one topic per publisher"),
                });
            }
            let pinned_rank = crate::backend::parse_dp_rank(worker);
            let ranked = topics
                .into_iter()
                .enumerate()
                .map(|(index, topic)| (pinned_rank.unwrap_or(index as u32), topic));
            let layout = layout.get_or_insert_with(|| crate::kv_index::CacheCapabilities {
                schema_version: 1,
                adapters: Vec::new(),
                ranks: ranked
                    .clone()
                    .map(|(rank, topic)| {
                        crate::kv_index::discovery::plain_full_rank(
                            rank,
                            model.to_owned(),
                            topic,
                            size,
                        )
                    })
                    .collect(),
            });
            for (dp_rank, topic) in ranked {
                let Some(rank) = layout.ranks.iter_mut().find(|rank| rank.rank == dp_rank) else {
                    return Err(ConfigError::ValidationFailed {
                        reason: format!(
                            "static KV event source '{worker}' has no layout for DP rank {dp_rank}"
                        ),
                    });
                };
                if rank.model.is_empty() {
                    rank.model = model.to_owned();
                }
                rank.publisher_topic = Some(topic);
            }
        }
        Ok(ConfiguredKvSources { regular, pd })
    }
}

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
            block_size,
            model: model.to_owned(),
            hybrid: None,
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
                    hybrid: None,
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
    let (base, _) = crate::kv_index::discovery::split_worker_rank(worker);
    if url::Url::parse(base).is_ok_and(|url| url.port() == Some(0)) {
        return Err(ConfigError::ValidationFailed {
            reason: format!("--{flag} worker URL '{worker}' must use a nonzero port"),
        });
    }
    Ok((worker, endpoints))
}

impl StaticKvEventSources {
    pub(crate) fn validate(&self, worker: &str) -> ConfigResult<()> {
        if worker.trim().is_empty()
            || (self.model.trim().is_empty()
                && (self.hybrid.is_none() || !self.endpoints.is_empty()))
        {
            return Err(ConfigError::ValidationFailed {
                reason: "static KV event sources require worker URLs and model identity"
                    .to_string(),
            });
        }
        // With a layout, discovery can supply the endpoints and outer model
        // identity. Registration checks the observed model against each rank.
        if self.block_size == 0 || (self.endpoints.is_empty() && self.hybrid.is_none()) {
            return Err(ConfigError::ValidationFailed {
                reason: format!(
                    "static KV event source for '{worker}' needs a nonzero block size and at least one endpoint"
                ),
            });
        }
        if let Some(layout) = &self.hybrid {
            let rank = crate::kv_index::source::discovery::split_worker_rank(worker).1;
            let discover = self.endpoints.is_empty() && !crate::backend::is_grpc_url(worker);
            if !layout.matches_hash_block_size(rank, self.block_size, discover) {
                return Err(ConfigError::ValidationFailed {
                    reason: format!("static KV source for '{worker}' block_size must match the layout hash_block_size"),
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
        if crate::kv_index::source::discovery::split_worker_rank(worker)
            .1
            .is_some()
            && self.endpoints.len() > 1
        {
            return Err(ConfigError::ValidationFailed {
                reason: format!(
                    "rank-pinned KV source '{worker}' can have only one publisher and one replay endpoint"
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

fn validate_pd_kv_endpoint(endpoint: &str) -> Result<(), String> {
    match endpoint.strip_prefix("ipc://") {
        Some(path) if !path.is_empty() => return Ok(()),
        Some(_) => return Err("ipc:// requires a non-empty path".into()),
        None => {}
    }
    crate::kv_index::canonical_endpoint(endpoint).map(|_| ())
}

pub(crate) fn validate_regular_sources(config: &RouterConfig) -> ConfigResult<()> {
    let static_sources = match &config.policy {
        PolicyConfig::KvAware {
            static_kv_event_sources,
            ..
        } => Some(static_kv_event_sources.as_ref()),
        _ => config
            .program_scheduling
            .as_ref()
            .map(|p| &p.binding_kv_sources),
    };
    if let Some(static_kv_event_sources) = static_sources {
        if !static_kv_event_sources.is_empty() {
            let RoutingMode::Regular { worker_urls } = &config.mode else {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "static --kv-event-source is only supported in regular mode"
                        .to_string(),
                });
            };
            if !config
                .discovery
                .as_ref()
                .is_some_and(|discovery| discovery.enabled)
            {
                for worker in static_kv_event_sources.keys() {
                    use crate::kv_index::source::discovery::split_worker_rank;
                    let (base, pinned) = split_worker_rank(worker);
                    let exact = worker_urls.iter().any(|url| url == worker);
                    let covers_ranks = pinned.is_none()
                        && worker_urls.iter().any(|url| {
                            let (url_base, rank) = split_worker_rank(url);
                            rank.is_some() && url_base == worker
                        });
                    let rank_pinned_under_base =
                        pinned.is_some() && !exact && worker_urls.iter().any(|url| url == base);
                    if rank_pinned_under_base {
                        // The KV index resolves static sources by exact
                        // worker URL, so a rank-pinned declaration under a
                        // plain base URL is accepted here and then never
                        // applied. Ask for the rank-pinned URLs instead of
                        // silently disabling the source.
                        return Err(ConfigError::ValidationFailed {
                            reason: format!(
                                "static KV event source '{worker}' is rank-pinned but \
                                 --worker-urls configures only its base URL; declare the \
                                 source under that base URL with one entry per rank, or \
                                 enable discovery"
                            ),
                        });
                    }
                    if !exact && !covers_ranks {
                        return Err(ConfigError::ValidationFailed {
                            reason: format!(
                                "static KV event source worker '{worker}' is not in --worker-urls"
                            ),
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_pd_sources(config: &RouterConfig) -> ConfigResult<()> {
    // Static KV event sources require a KV-aware policy on the owning stage.
    if !config.pd_kv_event_sources.is_empty() {
        if let RoutingMode::VllmPrefillDecode {
            prefill_urls,
            decode_urls,
            ..
        } = &config.mode
        {
            for (url, sources) in &config.pd_kv_event_sources {
                // Sources are keyed by base URL; any KV-aware stage serving
                // a base with disjoint P/D ranks consumes its events.
                let prefill_keyed = prefill_urls
                    .iter()
                    .any(|(prefill, _)| base_url_without_rank(prefill) == url);
                let decode_keyed = decode_urls
                    .iter()
                    .any(|decode| base_url_without_rank(decode) == url);
                if !prefill_keyed && !decode_keyed {
                    return Err(ConfigError::IncompatibleConfig {
                        reason: format!(
                            "pd_kv_event_sources key '{url}' is not a configured prefill or \
                             decode URL (use the base URL without the '@rank' suffix)"
                        ),
                    });
                }
                let prefill_kv = prefill_keyed
                    && matches!(
                        config.mode.get_prefill_policy(&config.policy),
                        PolicyConfig::KvAware { .. }
                    );
                let decode_kv = decode_keyed
                    && matches!(
                        config.mode.get_decode_policy(&config.policy),
                        PolicyConfig::KvAware { .. }
                    );
                if !prefill_kv && !decode_kv {
                    let stage = match (prefill_keyed, decode_keyed) {
                        (true, true) => "prefill and decode",
                        (true, false) => "prefill",
                        _ => "decode",
                    };
                    return Err(ConfigError::IncompatibleConfig {
                        reason: format!(
                            "pd_kv_event_sources['{url}'] belongs to the {stage} stage, \
                             whose policy is not kv_aware; its subscribers would never start"
                        ),
                    });
                }
                if sources.endpoints.is_empty() && sources.hybrid.is_none() {
                    return Err(ConfigError::IncompatibleConfig {
                        reason: format!("pd_kv_event_sources['{url}'] has no endpoints"),
                    });
                }
                if sources.block_size == 0 {
                    return Err(ConfigError::IncompatibleConfig {
                        reason: format!("pd_kv_event_sources['{url}'] block_size must be > 0"),
                    });
                }
                let invalid = |reason: &str| ConfigError::IncompatibleConfig {
                    reason: format!("PD cache layout for '{url}': {reason}"),
                };
                let hybrid_layout = sources
                    .hybrid
                    .as_ref()
                    .is_some_and(|layout| !layout.is_plain_full_attention());
                if hybrid_layout && !config.allows_hybrid_pd_sources() {
                    return Err(invalid(
                        "hybrid Program routing requires NIXL, local-only Prefill scoring, and a non-KV-aware Decode policy",
                    ));
                }
                if let Some(layout) = &sources.hybrid {
                    let discover =
                        sources.endpoints.is_empty() && !crate::backend::is_grpc_url(url);
                    if !layout.matches_hash_block_size(None, sources.block_size, discover) {
                        return Err(invalid("block_size must match the layout hash_block_size"));
                    }
                    if sources.endpoints.is_empty() {
                        if sources.replay_endpoints.is_some() || layout.ranks.is_empty() {
                            return Err(invalid("discovered publishers require a rank layout and no static replay endpoints"));
                        }
                    } else {
                        let model = sources
                            .model
                            .as_deref()
                            .filter(|model| !model.trim().is_empty())
                            .ok_or_else(|| invalid("model must not be empty"))?;
                        let replay = sources
                            .replay_endpoints
                            .as_ref()
                            .filter(|endpoints| endpoints.len() == sources.endpoints.len())
                            .ok_or_else(|| invalid("requires one replay endpoint per rank"))?;
                        if layout.ranks.len() != sources.endpoints.len() {
                            return Err(invalid("layout must declare one capability per endpoint"));
                        }
                        for (rank, replay_endpoint) in replay.iter().enumerate() {
                            layout
                                .validate_declared_source(rank as u32, model, replay_endpoint)
                                .map_err(|reason| ConfigError::IncompatibleConfig {
                                    reason: format!(
                                        "invalid PD layout for '{url}' rank {rank}: {reason}"
                                    ),
                                })?;
                        }
                    }
                }
                for endpoint in sources
                    .endpoints
                    .iter()
                    .chain(sources.replay_endpoints.iter().flatten())
                {
                    if let Err(reason) = validate_pd_kv_endpoint(endpoint) {
                        return Err(ConfigError::IncompatibleConfig {
                            reason: format!(
                                "pd_kv_event_sources['{url}'] endpoint '{endpoint}' is invalid: {reason}"
                            ),
                        });
                    }
                }
            }
        } else {
            return Err(ConfigError::IncompatibleConfig {
                reason: "pd_kv_event_sources requires vllm_prefill_decode mode".to_string(),
            });
        }
    }
    Ok(())
}

/// Base URL without the optional `@dp_rank` suffix.
///
/// Mirrors `Worker::normalised_url()`: static KV event sources are looked up by
/// `Worker::base_url()`, so a `http://host:port@1` prefill URL is keyed as
/// `http://host:port`.
fn base_url_without_rank(url: &str) -> &str {
    match url.split_once('@') {
        Some((base, rank)) if rank.parse::<usize>().is_ok() => base,
        _ => url,
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
