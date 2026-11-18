//! Compare admitted sources with current engine metadata.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use super::lifecycle::{SourceAction, WorkerLifecycle, WorkerRegistration, CONTROL_TIMEOUT};
use super::probe::{grpc_kv_event_source_entries, query_pd_kv_event_sources};
use super::{
    dp_rank_suffix, events_enabled, layout_topic_mismatch, resolve_endpoint,
    static_worker_discovery, worker_host, KvEventSourceEntry, KvIndexSupervisor, SourcesProbe,
};
use crate::config::StaticKvEventSources;

impl KvIndexSupervisor {
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
        let workers = self.workers.lock();
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
                    .map(|worker| match &worker.hybrid {
                        Some(provider) => {
                            provider.clone() as Arc<dyn crate::policies::TieredMatchProvider>
                        }
                        None => {
                            worker.indexer.clone() as Arc<dyn crate::policies::TieredMatchProvider>
                        }
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
    ) -> HashMap<String, WorkerRegistration> {
        use futures_util::{stream, StreamExt};
        let _refresh = self.refresh_lock.lock().await;
        let removed: Vec<_> = self
            .lifecycles
            .lock()
            .keys()
            .filter(|base| !bases.contains(*base))
            .cloned()
            .collect();
        for base in removed {
            self.on_worker_removed(&base);
        }
        let refresh: Vec<_> = {
            let workers = self.workers.lock();
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
        let probes: Vec<_> = stream::iter(refresh.into_iter().map(
            |(base, lifecycle, owner)| async move {
                let probe = tokio::time::timeout(CONTROL_TIMEOUT, self.probe_registration(&base))
                    .await
                    .ok()
                    .and_then(Result::ok);
                (base, lifecycle, owner, probe)
            },
        ))
        .buffer_unordered(8)
        .collect()
        .await;
        for (base, lifecycle, owner, probe) in probes {
            if !self
                .lifecycles
                .lock()
                .get(&base)
                .is_some_and(|current| Arc::ptr_eq(current, &lifecycle))
            {
                continue;
            }
            let action =
                lifecycle.observe_descriptor(probe.as_ref().map(|probe| &probe.descriptor));
            match action {
                SourceAction::Keep => {
                    if let Some(probe) = probe {
                        lifecycle.state.lock().base_models = probe.base_models;
                    }
                }
                SourceAction::Rebuild => {
                    let probe = probe.expect("rebuilding requires a descriptor");
                    lifecycle.state.lock().base_models = probe.base_models.clone();
                    // Both owner and control-record identities are compared under
                    // the publication lock; an old probe cannot resurrect a source.
                    let installation = self.install_observed_worker(
                        &base,
                        probe.worker,
                        owner,
                        Some(lifecycle.clone()),
                        probe.base_models,
                    );
                    if !installation.is_ok_and(|info| info.cache_key.is_some()) {
                        self.retire_observed_worker(&base, &lifecycle);
                    }
                }
                SourceAction::Stop => self.retire_observed_worker(&base, &lifecycle),
                SourceAction::Expire | SourceAction::Recover => {
                    unreachable!("descriptor probe does not recover replay")
                }
            }
        }
        self.worker_registrations()
    }

    fn retire_observed_worker(&self, base: &str, lifecycle: &Arc<WorkerLifecycle>) {
        let mut workers = self.workers.lock();
        if self
            .lifecycles
            .lock()
            .get(base)
            .is_some_and(|current| Arc::ptr_eq(current, lifecycle))
            && workers
                .get(base)
                .is_some_and(|worker| Arc::ptr_eq(&worker.active, &lifecycle.active))
        {
            if let Some(worker) = workers.remove(base) {
                self.retire_worker(worker);
            }
        }
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

#[derive(Clone, Debug)]
pub(super) struct BoundSource {
    pub(super) endpoint: String,
    pub(super) replay_endpoint: Option<String>,
}

/// The publisher endpoints a worker's hybrid subscribers are attached to, by
/// rank. Used to notice a publisher that moved while keeping its identity.
pub(super) type BoundSources = HashMap<u32, BoundSource>;

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
    config: &StaticKvEventSources,
    bound: Option<&BoundSources>,
) -> StaticHybridMatch {
    // A rank-pinned worker URL (`base@1`) still queries the engine's HTTP
    // control plane on the base URL.
    let http_url = dp_rank_suffix(worker_url)
        .and_then(|_| worker_url.rsplit_once('@').map(|(base, _)| base))
        .unwrap_or(worker_url);
    // Revalidate with the same probe (gRPC or HTTP) discovery used.
    let observed = if crate::backend::is_grpc_url(worker_url) {
        match tokio::time::timeout(
            Duration::from_secs(2),
            crate::backend::control::get_grpc_kv_event_sources(worker_url, Duration::from_secs(2)),
        )
        .await
        {
            Ok(Ok(response)) => grpc_kv_event_source_entries(worker_url, response.sources),
            Ok(Err(crate::backend::control::KvEventSourcesError::Unimplemented)) => {
                return StaticHybridMatch::Unavailable(SourcesProbe::Unsupported);
            }
            Ok(Err(_)) | Err(_) => return StaticHybridMatch::Unavailable(SourcesProbe::Failed),
        }
    } else {
        match tokio::time::timeout(
            Duration::from_secs(2),
            query_pd_kv_event_sources(client, http_url, api_key),
        )
        .await
        {
            Ok(Ok(sources)) => sources,
            Ok(Err(unavailable)) => {
                // A declared layout is authoritative when the engine cannot answer;
                // refusing it would make static configuration unusable on engines
                // without the control endpoint.
                return StaticHybridMatch::Unavailable(unavailable.probe);
            }
            Err(_) => return StaticHybridMatch::Unavailable(SourcesProbe::Failed),
        }
    };
    let Ok(expected) = static_worker_discovery(worker_url, config).sources else {
        return StaticHybridMatch::Differs;
    };
    let Some(host) = worker_host(http_url) else {
        return StaticHybridMatch::Differs;
    };
    // A layout-only binding declares no endpoints, so check the answer
    // against the layout and the sources it was bound to; a new topic or no
    // sources means the engine restarted.
    if expected.is_empty() {
        if observed.is_empty() {
            return StaticHybridMatch::Differs;
        }
        if layout_topic_mismatch(config, &observed).is_some() {
            return StaticHybridMatch::Differs;
        }
        // The subscribers stay attached to the endpoints captured when the
        // layout was installed, so a publisher that moved or disabled events
        // while keeping its rank and topic has to retire the provider.
        if let Some(bound) = bound {
            let moved = bound.iter().any(|(rank, installed)| {
                observed
                    .get(&rank.to_string())
                    .is_none_or(|actual| bound_source_moved(&host, installed, actual))
            });
            if moved {
                return StaticHybridMatch::Differs;
            }
        }
        let missing_rank = config.hybrid.as_ref().is_some_and(|layout| {
            layout.ranks.iter().any(|rank| {
                rank.publisher_topic
                    .as_deref()
                    .is_some_and(|topic| !topic.is_empty())
                    && !observed.contains_key(&rank.rank.to_string())
            })
        });
        return if missing_rank {
            StaticHybridMatch::Differs
        } else {
            StaticHybridMatch::Matches
        };
    }
    // A rank-pinned source declares one rank out of the engine's full set, so
    // only the declared ranks have to match.
    let matches = expected.iter().all(|(rank, expected)| {
        let Some(actual) = observed.get(rank) else {
            return false;
        };
        let endpoints_match = |actual: &str, expected: &str| {
            resolve_endpoint(actual, &host)
                .ok()
                .zip(resolve_endpoint(expected, &host).ok())
                .is_some_and(|(actual, expected)| actual == expected)
        };
        events_enabled(actual)
            && actual.topic == expected.topic
            && endpoints_match(&actual.endpoint, &expected.endpoint)
            && actual
                .replay_endpoint
                .as_deref()
                .zip(expected.replay_endpoint.as_deref())
                .is_some_and(|(actual, expected)| endpoints_match(actual, expected))
    });
    if matches {
        StaticHybridMatch::Matches
    } else {
        StaticHybridMatch::Differs
    }
}
// The full-attention HTTP adapter reads the optional control-metadata contract,
// then falls back to the engine config endpoint and the always-on metrics scrape,
// so a stock vLLM worker needs no static configuration.
/// Whether the engine's current answer for one rank no longer matches the
/// endpoints its subscribers are attached to. The engine reports the addresses
/// it bound (`tcp://*:port`) while the installed source holds the resolved
/// ones, so both sides are resolved against the worker host before comparing.
pub(super) fn bound_source_moved(
    host: &str,
    installed: &BoundSource,
    actual: &KvEventSourceEntry,
) -> bool {
    let same_endpoint = |left: &str, right: &str| {
        resolve_endpoint(left, host)
            .ok()
            .zip(resolve_endpoint(right, host).ok())
            .is_some_and(|(left, right)| left == right)
    };
    let same_optional_endpoint = |left: Option<&str>, right: Option<&str>| match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => same_endpoint(left, right),
        _ => false,
    };
    !events_enabled(actual)
        || !same_endpoint(&actual.endpoint, &installed.endpoint)
        || !same_optional_endpoint(
            actual.replay_endpoint.as_deref(),
            installed.replay_endpoint.as_deref(),
        )
}
