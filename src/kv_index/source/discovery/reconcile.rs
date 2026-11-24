//! Compare admitted sources with current engine metadata.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::lifecycle::{SourceAction, WorkerLifecycle, WorkerRegistration, CONTROL_TIMEOUT};
use super::metrics_meta::EngineCacheConfigs;
use super::probe::{probe_bound_sources, worker_serves_base_model};
use super::{
    events_enabled, resolve_endpoint, split_worker_rank, worker_host, KvEventSourceEntry,
    KvEventSourcesResponse, KvIndexSupervisor, SourcesProbe, WorkerDiscovery,
};
use crate::core::{worker::UNLABELED_WORKER_MODEL_ID, Worker};
use crate::kv_index::subscriber::EventSource;
use crate::kv_index::CacheCapabilities;

impl KvIndexSupervisor {
    pub(crate) fn cache_eligible_workers(
        &self,
        workers: &[Arc<dyn Worker>],
        model: &str,
    ) -> Option<Vec<bool>> {
        let mut allowed: Vec<_> = workers
            .iter()
            .map(|worker| {
                model != UNLABELED_WORKER_MODEL_ID
                    && worker_serves_base_model(worker.as_ref(), model)
            })
            .collect();
        let missing: Vec<_> = workers
            .iter()
            .zip(&allowed)
            .filter(|(_, allowed)| !**allowed)
            .map(|(worker, _)| worker.base_url().to_owned())
            .collect();
        if missing.is_empty() {
            return None;
        }
        let verified = self.worker_model_eligibility(&missing, Some(model));
        for (worker, allowed) in workers.iter().zip(&mut allowed) {
            *allowed |= verified.get(worker.base_url()).copied().unwrap_or(false);
        }
        Some(allowed)
    }

    /// Request preparation needs only model eligibility, not cloned registrations.
    pub(crate) fn worker_model_eligibility(
        &self,
        bases: &[String],
        model: Option<&str>,
    ) -> HashMap<String, bool> {
        let lifecycles = self.lifecycles.lock();
        bases
            .iter()
            .map(|base| {
                let allowed = model.is_some_and(|model| {
                    lifecycles.get(base).is_some_and(|lifecycle| {
                        let state = lifecycle.state.lock();
                        state.base_models.contains(model)
                            || state
                                .info
                                .as_ref()
                                .and_then(|info| info.cache_key.as_ref())
                                .is_some_and(|key| key.model.as_ref() == model)
                    })
                });
                (base.clone(), allowed)
            })
            .collect()
    }

    /// A Decode price requires an installed cache binding for each candidate rank.
    pub(crate) fn workers_have_cache_bindings<'a>(
        &self,
        mut targets: impl Iterator<Item = (&'a str, u32)>,
    ) -> bool {
        let lifecycles = self.lifecycles.lock();
        targets.all(|(base, rank)| {
            lifecycles.get(base).is_some_and(|lifecycle| {
                lifecycle.active.load(Ordering::Acquire)
                    && lifecycle
                        .state
                        .lock()
                        .info
                        .as_ref()
                        .is_some_and(|info| info.cache_key.is_some() && info.ranks.contains(&rank))
            })
        })
    }

    pub(super) fn lifecycle_for_worker(&self, worker_url: &str) -> Arc<WorkerLifecycle> {
        self.lifecycles
            .lock()
            .entry(worker_url.to_owned())
            .or_insert_with(|| {
                Arc::new(WorkerLifecycle::new(
                    Arc::new(AtomicBool::new(false)),
                    false,
                    None,
                ))
            })
            .clone()
    }

    /// Clone policy inputs without exposing or duplicating the source state machine.
    pub(crate) fn worker_registrations(&self) -> HashMap<String, WorkerRegistration> {
        let workers = self.installed.workers.lock();
        self.lifecycles
            .lock()
            .iter()
            .map(|(url, lifecycle)| {
                let state = lifecycle.state.lock();
                let provider = state
                    .info
                    .as_ref()
                    .and_then(|_| workers.get(url))
                    .filter(|worker| Arc::ptr_eq(&worker.active, &lifecycle.active))
                    .map(|worker| {
                        worker.provider.clone() as Arc<dyn crate::policies::TieredMatchProvider>
                    });
                (
                    url.clone(),
                    WorkerRegistration {
                        revision: state.revision,
                        info: state.info.clone(),
                        provider,
                    },
                )
            })
            .collect()
    }

    /// Reconcile one pool outside its request path, sharing registration and trust.
    pub(crate) async fn refresh_workers(
        &self,
        bases: &HashSet<String>,
        plain_only: &HashSet<String>,
        prefill_bases: &HashSet<String>,
    ) -> HashMap<String, WorkerRegistration> {
        let _refresh = self.refresh_lock.lock().await;
        let removed: Vec<_> = self
            .lifecycles
            .lock()
            .iter()
            .filter(|(base, lifecycle)| {
                !bases.contains(*base)
                    || (plain_only.contains(*base)
                        && lifecycle
                            .state
                            .lock()
                            .descriptor
                            .as_ref()
                            .is_some_and(super::probe::WorkerDescriptor::has_nonplain_layout))
            })
            .map(|(base, _)| base.clone())
            .collect();
        for base in removed {
            self.on_worker_removed(&base);
        }
        let refresh: Vec<_> = {
            let workers = self.installed.workers.lock();
            bases
                .iter()
                .filter_map(|base| {
                    let missing = !self.lifecycles.lock().contains_key(base);
                    let lifecycle = self.lifecycle_for_worker(base);
                    (missing || lifecycle.refresh_due()).then(|| {
                        (
                            base.clone(),
                            lifecycle,
                            workers.get(base).map(|worker| worker.active.clone()),
                        )
                    })
                })
                .collect()
        };
        self.refresh_registrations(refresh, plain_only, prefill_bases)
            .await;
        self.worker_registrations()
    }

    /// Ordinary serving re-probes automatically discovered layouts whose
    /// monitor revoked trust.
    pub(crate) async fn refresh_discovered_workers(&self) -> HashMap<String, WorkerRegistration> {
        let _refresh = self.refresh_lock.lock().await;
        let refresh = {
            let workers = self.installed.workers.lock();
            self.lifecycles
                .lock()
                .iter()
                .filter(|(base, lifecycle)| {
                    lifecycle.layout_rebind_due()
                        && self.resolved_static_source(base).is_none_or(|source| {
                            super::layout_only(&source)
                                && source.hybrid.as_ref().is_some_and(|layout| {
                                    layout
                                        .ranks
                                        .iter()
                                        .any(|rank| rank.publisher_topic.is_none())
                                })
                        })
                })
                .map(|(base, lifecycle)| {
                    (
                        base.clone(),
                        lifecycle.clone(),
                        workers.get(base).map(|worker| worker.active.clone()),
                    )
                })
                .collect()
        };
        let changed = self
            .refresh_registrations(refresh, &HashSet::new(), &HashSet::new())
            .await;
        if changed.is_empty() {
            return HashMap::new();
        }
        let mut registrations = self.worker_registrations();
        registrations.retain(|base, _| changed.contains(base));
        registrations
    }

    async fn refresh_registrations(
        &self,
        refresh: Vec<(String, Arc<WorkerLifecycle>, Option<Arc<AtomicBool>>)>,
        plain_only: &HashSet<String>,
        prefill_bases: &HashSet<String>,
    ) -> HashSet<String> {
        use futures_util::{stream, StreamExt};
        let probes: Vec<_> = stream::iter(refresh.into_iter().map(
            |(base, lifecycle, owner)| async move {
                let connector = self.pd_connector.filter(|_| prefill_bases.contains(&base));
                let probe = tokio::time::timeout(
                    CONTROL_TIMEOUT,
                    self.probe_registration(
                        &base,
                        plain_only.contains(&base),
                        connector,
                        &lifecycle,
                    ),
                )
                .await
                .ok()
                .and_then(Result::ok);
                (base, lifecycle, owner, connector, probe)
            },
        ))
        .buffer_unordered(8)
        .collect()
        .await;
        let mut changed = HashSet::new();
        for (base, lifecycle, owner, connector, probe) in probes {
            if !self
                .lifecycles
                .lock()
                .get(&base)
                .is_some_and(|current| Arc::ptr_eq(current, &lifecycle))
            {
                continue;
            }
            let action = lifecycle.observe_descriptor(
                &base,
                probe.as_ref().map(|probe| &probe.descriptor),
                probe
                    .as_ref()
                    .and_then(|probe| probe.worker.engine_cache_configs.as_ref()),
            );
            match action {
                SourceAction::Keep => {
                    if let Some(probe) = probe {
                        lifecycle.state.lock().base_models =
                            probe.worker.base_models.unwrap_or_default();
                    }
                }
                SourceAction::Rebuild => {
                    let probe = probe.expect("rebuilding requires a descriptor");
                    lifecycle.state.lock().base_models =
                        probe.worker.base_models.clone().unwrap_or_default();
                    // Both owner and control-record identities are compared under
                    // the publication lock; an old probe cannot resurrect a source.
                    let installation = self.install_observed_worker(
                        &base,
                        probe.worker,
                        owner,
                        Some(lifecycle.clone()),
                        connector,
                    );
                    if !installation.is_ok_and(|info| info.cache_key.is_some()) {
                        self.installed.retire_if_current(&base, &lifecycle);
                    }
                    changed.insert(base);
                }
                SourceAction::Stop => {
                    self.installed.retire_if_current(&base, &lifecycle);
                    changed.insert(base);
                }
                SourceAction::Recover => {
                    unreachable!("descriptor probe does not recover replay")
                }
            }
        }
        changed
    }
}

/// Outcome of reconciling a declared static hybrid layout with the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StaticHybridMatch {
    /// The engine advertises exactly the declared sources.
    Matches,
    /// The engine advertises different sources than the layout declares.
    Differs,
    /// The engine has no control endpoint to ask (stock vLLM serves none over
    /// HTTP, and the extension may be off), or it is unreachable.
    Unavailable(SourcesProbe),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BoundSource {
    pub(super) endpoint: String,
    pub(super) replay_endpoint: Option<String>,
    pub(super) topic: String,
}

impl From<&EventSource> for BoundSource {
    fn from(source: &EventSource) -> Self {
        Self {
            endpoint: source.pub_endpoint.clone(),
            replay_endpoint: source.replay_endpoint.clone(),
            topic: source.topic.clone(),
        }
    }
}

/// Publisher identities by rank, including publishers fenced before subscription.
pub(super) type BoundSources = HashMap<u32, BoundSource>;

pub(super) fn runtime_hash_mismatch(
    configs: Option<&EngineCacheConfigs>,
    capability: &CacheCapabilities,
    worker_url: &str,
    rank: u32,
) -> bool {
    let reported = configs
        .and_then(|engines| engines.get(&rank))
        .and_then(|engine| engine.prefix_caching_hash_algo.as_deref());
    let expected = capability
        .rank_capability(rank)
        .map(|contract| contract.hash_algorithm.as_str());
    if configs.is_some() && reported.is_none() && expected.is_some() {
        tracing::debug!(
            worker_url,
            rank,
            "worker cache metrics omit the prefix hash algorithm"
        );
    }
    if let (Some(reported), Some(expected)) = (reported, expected) {
        if reported != expected {
            tracing::warn!(
                worker_url,
                rank,
                reported,
                expected,
                "KV credit disabled for incompatible worker prefix hash algorithm"
            );
            return true;
        }
    }
    false
}

/// Check the surviving publisher set against the same source snapshot.
pub(super) fn observed_sources_match<'a>(
    worker: &str,
    mut active: impl Iterator<Item = &'a EventSource>,
    fences: &BoundSources,
    expected: Option<&CacheCapabilities>,
    observed: &Result<(KvEventSourcesResponse, Option<CacheCapabilities>), SourcesProbe>,
) -> StaticHybridMatch {
    let (sources, layout) = match observed {
        Ok(snapshot) => snapshot,
        Err(probe) => return StaticHybridMatch::Unavailable(*probe),
    };
    let Some(host) = worker_host(split_worker_rank(worker).0) else {
        return StaticHybridMatch::Differs;
    };
    if layouts_match(expected, layout.as_ref(), fences)
        && active.all(|source| {
            sources
                .get(&source.dp_rank.to_string())
                .is_some_and(|actual| {
                    source_matches(
                        &host,
                        actual,
                        &source.pub_endpoint,
                        source.replay_endpoint.as_deref(),
                        &source.topic,
                    )
                })
        })
    {
        StaticHybridMatch::Matches
    } else {
        StaticHybridMatch::Differs
    }
}

pub(super) fn layouts_match(
    expected: Option<&CacheCapabilities>,
    observed: Option<&CacheCapabilities>,
    fences: &BoundSources,
) -> bool {
    match (expected, observed) {
        (Some(expected), Some(observed)) => {
            let retained = |rank| !fences.contains_key(&rank);
            expected.schema_version == observed.schema_version
                && expected.adapters == observed.adapters
                && expected
                    .ranks
                    .iter()
                    .filter(|rank| retained(rank.rank))
                    .eq(observed.ranks.iter().filter(|rank| retained(rank.rank)))
        }
        (None, None) => true,
        _ => false,
    }
}

/// Whether a probe result keeps the provider alive. A transient failure is not
/// a contradiction: the worker stays routable and only its credit expires.
pub(super) fn probe_keeps_trust(outcome: &StaticHybridMatch) -> bool {
    matches!(
        outcome,
        StaticHybridMatch::Matches | StaticHybridMatch::Unavailable(_)
    )
}

/// Whether a probe result may extend the freshness window. A confirming probe
/// does, and so does an engine that has no control endpoint at all (the static
/// declaration is authoritative there). A failing probe must let the window
/// expire, otherwise a worker that restarts at the same URL while the endpoint
/// returns 503 would keep crediting claims from before the restart.
pub(super) fn probe_renews_trust(outcome: &StaticHybridMatch) -> bool {
    matches!(
        outcome,
        StaticHybridMatch::Matches | StaticHybridMatch::Unavailable(SourcesProbe::Unsupported)
    )
}

pub(super) async fn static_hybrid_sources_match(
    client: &reqwest::Client,
    worker_url: &str,
    api_key: Option<&str>,
    discovery: &WorkerDiscovery,
    bound: Option<&BoundSources>,
) -> StaticHybridMatch {
    // A rank-pinned worker URL (`base@1`) still queries the engine's HTTP
    // control plane on the base URL.
    let (http_url, _) = split_worker_rank(worker_url);
    let observed = match probe_bound_sources(client, http_url, api_key, true).await {
        Ok((sources, _)) => sources,
        Err(probe) => return StaticHybridMatch::Unavailable(probe),
    };
    let Some(host) = worker_host(http_url) else {
        return StaticHybridMatch::Differs;
    };
    if let Some(bound) = bound {
        return if !bound.is_empty()
            && bound_sources_match(
                &host,
                bound.iter().map(|(&rank, source)| (rank, source)),
                &observed,
            ) {
            StaticHybridMatch::Matches
        } else {
            StaticHybridMatch::Differs
        };
    }
    let Ok(expected) = &discovery.sources else {
        return StaticHybridMatch::Differs;
    };
    // Layout-only sources are verified against their bound publishers above.
    // A rank-pinned declaration only compares the rank it supplies.
    let matches = !expected.is_empty()
        && expected.iter().all(|(rank, expected)| {
            observed.get(rank).is_some_and(|actual| {
                source_matches(
                    &host,
                    actual,
                    &expected.endpoint,
                    expected.replay_endpoint.as_deref(),
                    &expected.topic,
                )
            })
        });
    if matches {
        StaticHybridMatch::Matches
    } else {
        StaticHybridMatch::Differs
    }
}

fn bound_sources_match<'a>(
    host: &str,
    mut bound: impl Iterator<Item = (u32, &'a BoundSource)>,
    sources: &KvEventSourcesResponse,
) -> bool {
    bound.all(|(rank, installed)| {
        sources
            .get(&rank.to_string())
            .is_some_and(|actual| !bound_source_moved(host, installed, actual))
    })
}

/// Whether the engine's current answer for one rank no longer matches the
/// endpoints its subscribers are attached to. The engine reports the addresses
/// it bound (`tcp://*:port`) while the installed source holds the resolved
/// ones, so both sides are resolved against the worker host before comparing.
pub(super) fn bound_source_moved(
    host: &str,
    installed: &BoundSource,
    actual: &KvEventSourceEntry,
) -> bool {
    !source_matches(
        host,
        actual,
        &installed.endpoint,
        installed.replay_endpoint.as_deref(),
        &installed.topic,
    )
}

fn source_matches(
    host: &str,
    actual: &KvEventSourceEntry,
    endpoint: &str,
    replay_endpoint: Option<&str>,
    topic: &str,
) -> bool {
    let same_endpoint = |left: &str, right: &str| {
        resolve_endpoint(left, host)
            .ok()
            .zip(resolve_endpoint(right, host).ok())
            .is_some_and(|(left, right)| left == right)
    };
    events_enabled(actual)
        && actual.topic == topic
        && same_endpoint(&actual.endpoint, endpoint)
        && match (actual.replay_endpoint.as_deref(), replay_endpoint) {
            (None, None) => true,
            (Some(left), Some(right)) => same_endpoint(left, right),
            _ => false,
        }
}
