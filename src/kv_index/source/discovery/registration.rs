//! Admit and install one worker and its event sources.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use parking_lot::Mutex;
use tracing::{debug, info, warn};

use super::lifecycle::{HybridWorkerProvider, WorkerLifecycle};
use super::probe::worker_descriptor;
use super::reconcile::{BoundSource, BoundSources};
use super::{
    discovered_block_size, event_source_label, events_enabled, metadata_source_label,
    resolve_endpoint, worker_host, CacheKey, KvIndexSupervisor, LiveSource, SourcesProbe,
    WorkerDiscovery, WorkerKvInfo, WorkerSubs,
};
use crate::kv_index::subscriber::EventSource;
use crate::kv_index::{EventStream, SourceBinding, SourceId};
use crate::protocols::worker_spec::ServerInfo;

impl KvIndexSupervisor {
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(crate) fn install_worker(
        &self,
        worker_url: &str,
        discovery: WorkerDiscovery,
    ) -> Result<WorkerKvInfo, String> {
        let observed = self
            .workers
            .lock()
            .get(worker_url)
            .map(|worker| worker.active.clone());
        self.install_observed_worker(worker_url, discovery, observed, None, HashSet::new())
    }

    pub(super) fn install_observed_worker(
        &self,
        worker_url: &str,
        discovery: WorkerDiscovery,
        observed: Option<Arc<AtomicBool>>,
        expected_lifecycle: Option<Arc<WorkerLifecycle>>,
        base_models: HashSet<String>,
    ) -> Result<WorkerKvInfo, String> {
        let descriptor = worker_descriptor(&discovery).ok();
        let hybrid_capability = discovery.hybrid;
        let grpc = crate::backend::is_grpc_url(worker_url);
        let full_attention_http =
            !grpc && self.full_attention_http && self.resolved_static_source(worker_url).is_none();
        let full_attention = full_attention_http || (self.http_pd && !grpc);
        let server_info = discovery.server_info.unwrap_or_else(|e| {
            warn!(
                "kv_index {}: {} — falling back to URL as instance/model id",
                worker_url, e
            );
            ServerInfo {
                model_id: Some(worker_url.to_string()),
                ..Default::default()
            }
        });

        let instance_id = server_info
            .instance_id
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| worker_url.to_string());
        let block_size = discovered_block_size(&server_info);
        let model_id = match discovery.model_id {
            Ok(Some(model_id)) if !model_id.trim().is_empty() => Some(model_id),
            Ok(_) => {
                warn!(
                    "kv_index {}: GetModelInfo returned an empty model_id — cannot create cache identity",
                    worker_url
                );
                None
            }
            Err(error) => {
                warn!(
                    "kv_index {}: no worker model id: {} — cannot create cache identity; pass --kv-event-model",
                    worker_url, error
                );
                None
            }
        };

        let sources = match discovery.sources {
            Ok(s) => s,
            Err(unavailable) => {
                match unavailable.probe {
                    SourcesProbe::Unsupported => debug!(
                        "kv_index {}: engine does not expose KV event sources; no subscribers",
                        worker_url
                    ),
                    SourcesProbe::Failed => warn!(
                        "kv_index {}: {} probe failed: {} ? will retry",
                        worker_url,
                        event_source_label(worker_url),
                        unavailable.detail
                    ),
                }
                return Ok(WorkerKvInfo {
                    instance_id,
                    block_size,
                    ranks: vec![],
                    cache_key: None,
                });
            }
        };

        let Some(block_size) = block_size else {
            warn!(
                "kv_index {}: no cache metadata from {} — cannot index; pass --kv-event-block-size, or enable a metadata source",
                worker_url,
                metadata_source_label(worker_url)
            );
            return Ok(WorkerKvInfo {
                instance_id,
                block_size: None,
                ranks: vec![],
                cache_key: None,
            });
        };
        let Some(model_id) = model_id else {
            return Ok(WorkerKvInfo {
                instance_id,
                block_size: Some(block_size),
                ranks: vec![],
                cache_key: None,
            });
        };
        // Policy intervals use this unit to translate hits back into tokens.
        // A hybrid hash boundary can lie inside the physical attention page.
        // Use a common divisor when ranks declare different granularities.
        let block_size = hybrid_capability.as_ref().map_or(block_size, |capability| {
            capability
                .ranks
                .iter()
                .filter_map(|rank| rank.hash_block_size)
                .filter(|size| *size > 0)
                .fold(block_size, |mut a, mut b| {
                    while b != 0 {
                        (a, b) = (b, a % b);
                    }
                    a
                })
        });
        let key = CacheKey {
            model: Arc::from(model_id.as_str()),
            hash_mode: self.hash_mode,
            block_size,
        };
        let worker_host = worker_host(worker_url).unwrap_or_else(|| worker_url.to_string());
        let source_id: SourceId = Arc::from(instance_id.as_str());
        let mut event_sources = Vec::with_capacity(sources.len());
        let mut normalized_ranks = HashSet::new();

        for (rank_str, entry) in &sources {
            if !events_enabled(entry) {
                continue;
            }
            let rank: u32 = rank_str.parse().map_err(|_| "invalid KV source rank")?;
            if !normalized_ranks.insert(rank) {
                return Err("duplicate normalized KV source rank".into());
            }
            let pub_endpoint = resolve_endpoint(&entry.endpoint, &worker_host)
                .map_err(|error| format!("invalid KV publisher endpoint: {error}"))?;
            let replay_endpoint = entry
                .replay_endpoint
                .as_deref()
                .map(|e| resolve_endpoint(e, &worker_host))
                .transpose()
                .map_err(|error| format!("invalid KV replay endpoint: {error}"))?;
            let ev = EventSource {
                source: source_id.clone(),
                dp_rank: rank,
                pub_endpoint,
                replay_endpoint,
                topic: entry.topic.clone(),
                hwm: entry.hwm,
            };
            event_sources.push(ev);
        }

        let hybrid_indexes = if let Some(capability) = &hybrid_capability {
            if event_sources.is_empty() {
                return Err("hybrid capability has no event sources".into());
            }
            // Admit each rank on its own evidence; one rank's miss does not
            // revoke its siblings.
            let mut indexes: HashMap<u32, SourceBinding> = HashMap::new();
            for source in &event_sources {
                let Some(rank_capability) = capability.rank_capability(source.dp_rank) else {
                    warn!(
                        "kv_index {}: no hybrid capability for rank {}; rank will receive no cache credit",
                        worker_url, source.dp_rank
                    );
                    continue;
                };
                if rank_capability.model != model_id {
                    warn!(
                        "kv_index {}: hybrid rank {} was captured for model {} but this worker serves {}; rank will receive no cache credit",
                        worker_url, source.dp_rank, rank_capability.model, model_id
                    );
                    continue;
                }
                let target = crate::kv_index::RoutableTarget {
                    instance_id: source_id.clone(),
                    dp_rank: source.dp_rank,
                };
                match capability.index_for_source(source, target, 1_000_000) {
                    Ok(index) => {
                        indexes.insert(source.dp_rank, index);
                    }
                    Err(reason) => warn!(
                        "kv_index {}: hybrid rank {} not admitted: {reason}; rank will receive no cache credit",
                        worker_url, source.dp_rank
                    ),
                }
            }
            if indexes.is_empty() {
                return Err("no hybrid rank has a verified layout".into());
            }
            Some(indexes)
        } else {
            None
        };

        // Full-attention HTTP bindings require plain origins. Generic discovery retains
        // its backend-specific URI scope, including HTTP base paths.
        let worker_origin = if full_attention_http {
            Some(crate::kv_index::source::endpoints::canonical_worker(worker_url)?.0)
        } else {
            None
        };
        // Admission and publication share the existing source-owner lock. Two
        // concurrent discoveries must not both accept an ambiguous binding.
        let mut workers = self.workers.lock();
        let unchanged = match (observed.as_ref(), workers.get(worker_url)) {
            (Some(observed), Some(current)) => Arc::ptr_eq(observed, &current.active),
            (None, None) => true,
            _ => false,
        };
        let lifecycle_unchanged = expected_lifecycle.as_ref().is_none_or(|observed| {
            self.lifecycles
                .lock()
                .get(worker_url)
                .is_some_and(|current| Arc::ptr_eq(observed, current))
        });
        if !unchanged || !lifecycle_unchanged {
            return Err("KV worker was registered while discovery was pending".into());
        }
        if worker_origin.as_ref().is_some_and(|origin| {
            workers.iter().any(|(url, worker)| {
                url != worker_url && worker.worker_origin.as_ref() == Some(origin)
            })
        }) {
            return Err("KV worker origins must be independent".into());
        }
        let mut replay_endpoints = Vec::new();
        for source in &event_sources {
            if let Some(endpoint) = &source.replay_endpoint {
                // This replay protocol sends only start_seq, not a source key.
                if replay_endpoints.contains(endpoint)
                    || workers.iter().any(|(url, worker)| {
                        url != worker_url && worker.replay_endpoints.contains(endpoint)
                    })
                {
                    return Err("KV replay endpoints must be independent".into());
                }
                replay_endpoints.push(endpoint.clone());
            }
            for (url, worker) in workers.iter().filter(|(url, _)| url.as_str() != worker_url) {
                if worker.live_sources.iter().any(|live| {
                    live.publisher == source.pub_endpoint
                        && live.topic == source.topic
                        && live.rank == source.dp_rank
                }) {
                    return Err("live KV streams must be independent".into());
                }
                if worker.key == key
                    && worker.source_id == source_id
                    && worker.ranks.contains(&source.dp_rank)
                {
                    return Err(format!("KV index targets must be independent: {url}"));
                }
            }
        }
        // No admission can fail below this point. Share rank guards by exact
        // publisher/topic and promote them monotonically before publishing B.
        let active = Arc::new(AtomicBool::new(true));
        let lifecycle = Arc::new(WorkerLifecycle::new(
            active.clone(),
            hybrid_indexes.is_some(),
            descriptor,
        ));
        let mut live_sources: Vec<LiveSource> = Vec::with_capacity(event_sources.len());
        for source in &event_sources {
            let strict_rank = workers
                .values()
                .flat_map(|worker| &worker.live_sources)
                .chain(live_sources.iter())
                .find(|live| live.publisher == source.pub_endpoint && live.topic == source.topic)
                .map(|live| live.lease.strict_rank.clone())
                .unwrap_or_else(|| Arc::new(Mutex::new(false)));
            let shared_rank = workers
                .iter()
                .filter(|(url, _)| url.as_str() != worker_url)
                .flat_map(|(_, worker)| &worker.live_sources)
                .any(|live| {
                    live.publisher == source.pub_endpoint
                        && live.topic == source.topic
                        && live.rank != source.dp_rank
                })
                || event_sources.iter().any(|other| {
                    other.pub_endpoint == source.pub_endpoint
                        && other.topic == source.topic
                        && other.dp_rank != source.dp_rank
                });
            if shared_rank {
                let mut strict = strict_rank.lock();
                if !*strict {
                    *strict = true;
                    for worker in workers.values() {
                        for live in &worker.live_sources {
                            if live.publisher == source.pub_endpoint && live.topic == source.topic {
                                worker
                                    .indexer
                                    .invalidate_source(&worker.source_id, live.rank);
                            }
                        }
                    }
                }
            }
            live_sources.push(LiveSource {
                publisher: source.pub_endpoint.clone(),
                topic: source.topic.clone(),
                rank: source.dp_rank,
                lease: crate::kv_index::source::binding::SourceLease {
                    strict_rank,
                    active: active.clone(),
                },
            });
        }
        if let Some(previous) = workers.remove(worker_url) {
            self.retire_worker(previous);
        }
        let indexer = self.acquire_identity(&key);
        let hybrid_indexes = hybrid_indexes.map(|indexes| {
            indexes
                .into_iter()
                .map(|(rank, index)| {
                    let binding = live_sources
                        .iter()
                        .find(|live| live.rank == rank)
                        .expect("admitted rank has a live source")
                        .lease
                        .clone();
                    (
                        rank,
                        Arc::new(
                            index
                                .with_shared_index(indexer.clone())
                                .with_source_binding(binding),
                        ),
                    )
                })
                .collect::<HashMap<_, _>>()
        });
        // Endpoints the subscribers are about to attach to, so the freshness
        // task can notice a publisher that moves while keeping its identity.
        let bound_sources: Arc<BoundSources> = Arc::new(
            event_sources
                .iter()
                .map(|ev| {
                    (
                        ev.dp_rank,
                        BoundSource {
                            endpoint: ev.pub_endpoint.clone(),
                            replay_endpoint: ev.replay_endpoint.clone(),
                        },
                    )
                })
                .collect(),
        );
        let mut ranks = Vec::with_capacity(event_sources.len());
        let mut handles = Vec::with_capacity(event_sources.len());
        let mut hybrid_handles = Vec::new();
        let mut hybrid_targets: Vec<(
            crate::kv_index::subscriber::EventSource,
            Arc<SourceBinding>,
        )> = Vec::new();
        for (ev, live) in event_sources.into_iter().zip(&live_sources) {
            let rank = ev.dp_rank;
            if let Some(indexes) = &hybrid_indexes {
                // A hybrid-configured rank without an admitted index gets no
                // subscriber at all: the legacy single-group index must not
                // credit an unmodeled hybrid layout.
                if let Some(index) = indexes.get(&rank) {
                    // Two ranks on one publisher and one topic cannot be told
                    // apart when a batch omits its rank, and the hybrid
                    // subscriber has no rank fence to apply: give the rank no
                    // subscriber instead of crediting the other rank's cache.
                    if *live.lease.strict_rank.lock() {
                        tracing::warn!(
                            "kv_index {}: rank{} shares a publisher and topic with another rank; no hybrid subscriber",
                            worker_url,
                            rank
                        );
                        ranks.push(rank);
                        continue;
                    }
                    let stream = Arc::new(EventStream::new(index.clone()));
                    let handle =
                        match crate::kv_index::spawn_zmq(ev.clone(), stream, self.ctx.clone()) {
                            Ok(handle) => handle,
                            Err(error) => {
                                warn!(worker_url, rank, error, "KV event subscription rejected");
                                continue;
                            }
                        };
                    hybrid_targets.push((ev, index.clone()));
                    hybrid_handles.push(Arc::new(handle));
                }
            } else {
                let handle = crate::kv_index::subscriber::spawn_bound(
                    ev,
                    indexer.clone(),
                    Some(self.signal_tx.clone()),
                    self.ctx.clone(),
                    full_attention.then_some(key.block_size),
                    live.lease.clone(),
                );
                handles.push(handle);
            }
            info!(
                "kv_index {}: spawned subscriber rank{} (instance {})",
                worker_url, rank, instance_id
            );
            ranks.push(rank);
        }

        let hybrid = hybrid_indexes.map(|indexes| {
            Arc::new(HybridWorkerProvider {
                indexes,
                lifecycle: lifecycle.clone(),
            })
        });
        if let Some(provider) = &hybrid {
            self.monitor_hybrid_source(
                provider,
                worker_url,
                hybrid_handles.clone(),
                hybrid_targets,
                bound_sources,
            );
        }

        workers.insert(
            worker_url.to_string(),
            WorkerSubs {
                active,
                key: key.clone(),
                source_id,
                worker_origin,
                replay_endpoints,
                live_sources,
                ranks: ranks.clone(),
                indexer,
                handles,
                hybrid,
                hybrid_handles,
            },
        );
        let info = WorkerKvInfo {
            instance_id,
            block_size: Some(block_size),
            ranks,
            cache_key: Some(key),
        };
        {
            let mut state = lifecycle.state.lock();
            state.info = (!info.ranks.is_empty()).then(|| info.clone());
            state.base_models = base_models;
        }
        self.lifecycles
            .lock()
            .insert(worker_url.to_owned(), lifecycle);
        self.check_block_size(worker_url, Some(block_size));
        Ok(info)
    }
}
