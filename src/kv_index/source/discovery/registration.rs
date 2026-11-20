//! Admit and install one worker and its event sources.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use parking_lot::Mutex;
use tracing::{debug, info, warn};

use super::lifecycle::{LayoutWorkerProvider, WorkerLifecycle};
use super::probe::{layout_model, worker_descriptor};
use super::{
    discovered_block_size, event_source_label, events_enabled, metadata_source_label,
    resolve_endpoint, worker_host, CacheKey, KvIndexSupervisor, LiveSource, SourcesProbe,
    WorkerDiscovery, WorkerKvInfo, WorkerSubs,
};
use crate::kv_index::subscriber::EventSource;
use crate::kv_index::{EventStream, SourceId};
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
        let assumed_plain_full = discovery.assumed_plain_full;
        let grpc = crate::backend::is_grpc_url(worker_url);
        let full_attention_http =
            !grpc && self.full_attention_http && self.resolved_static_source(worker_url).is_none();
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
        let Some(capability) = discovery.hybrid else {
            debug!(
                worker_url,
                "no verified sha256_cbor cache contract; skipping KV subscription"
            );
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
        let block_size = capability
            .ranks
            .iter()
            .filter_map(|rank| rank.hash_block_size)
            .filter(|size| *size > 0)
            .fold(block_size, |mut a, mut b| {
                while b != 0 {
                    (a, b) = (b, a % b);
                }
                a
            });
        let key = CacheKey {
            model: Arc::from(model_id.as_str()),
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

        // Admit each rank on its own verified contract.
        let engine_model = layout_model(&server_info, &model_id, &capability);
        let mut bindings = HashMap::new();
        event_sources.retain(|source| {
            match capability.bind_worker_source(source, engine_model, 1_000_000) {
                Ok(binding) => {
                    let binding = if assumed_plain_full {
                        binding.require_root_proof()
                    } else {
                        binding
                    };
                    bindings.insert(
                        source.dp_rank,
                        binding.with_mooncake_prefill(self.mooncake_prefill),
                    );
                    true
                }
                Err(reason) => {
                    warn!(
                        worker_url,
                        rank = source.dp_rank,
                        reason,
                        "KV source has no verified cache layout"
                    );
                    false
                }
            }
        });
        if bindings.is_empty() {
            return Err("no KV event rank has a verified layout".into());
        }

        // Full-attention HTTP bindings require plain origins. Generic discovery retains
        // its backend-specific URI scope, including HTTP base paths.
        let worker_origin = if full_attention_http {
            Some(crate::kv_index::source::endpoints::canonical_worker(
                worker_url,
            )?)
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
        let mut live_endpoints = HashSet::new();
        for source in &event_sources {
            // Topic-specific continuity requires one publisher stream per rank.
            if !live_endpoints.insert((&source.pub_endpoint, &source.topic)) {
                return Err("live KV streams must be independent".into());
            }
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
                if worker
                    .live_sources
                    .iter()
                    .any(|live| live.publisher == source.pub_endpoint && live.topic == source.topic)
                {
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
        // Publish the new registration while holding the source-owner lock.
        let active = Arc::new(AtomicBool::new(true));
        let lifecycle = Arc::new(WorkerLifecycle::new(active.clone(), true, descriptor));
        let live_sources: Vec<_> = event_sources
            .iter()
            .map(|source| LiveSource {
                publisher: source.pub_endpoint.clone(),
                topic: source.topic.clone(),
                rank: source.dp_rank,
                lease: crate::kv_index::source::binding::SourceLease {
                    strict_rank: Arc::new(Mutex::new(true)),
                    active: active.clone(),
                },
            })
            .collect();
        if let Some(previous) = workers.remove(worker_url) {
            self.retire_worker(previous);
        }
        let indexer = self.acquire_identity(&key);
        let mut ranks = Vec::with_capacity(event_sources.len());
        let mut handles = Vec::with_capacity(event_sources.len());
        let mut targets = Vec::new();
        let mut indexes = HashMap::default();
        for (ev, live) in event_sources.into_iter().zip(&live_sources) {
            let rank = ev.dp_rank;
            let index = Arc::new(
                bindings
                    .remove(&rank)
                    .expect("admitted rank has a source binding")
                    .with_shared_index(indexer.clone())
                    .with_source_binding(live.lease.clone()),
            );
            let stream = Arc::new(EventStream::new(index.clone()));
            let handle = match super::super::events::event_stream::spawn_zmq_with_signals(
                ev.clone(),
                stream,
                self.ctx.clone(),
                Some(self.signal_tx.clone()),
            ) {
                Ok(handle) => handle,
                Err(error) => {
                    warn!(worker_url, rank, error, "KV event subscription rejected");
                    continue;
                }
            };
            targets.push((ev, index.clone()));
            indexes.insert(rank, index);
            handles.push(Arc::new(handle));
            info!(
                "kv_index {}: spawned subscriber rank{} (instance {})",
                worker_url, rank, instance_id
            );
            ranks.push(rank);
        }

        let provider = Arc::new(LayoutWorkerProvider {
            indexes,
            lifecycle: lifecycle.clone(),
        });
        self.monitor_hybrid_source(
            &provider,
            worker_url,
            handles.clone(),
            targets,
            capability,
            assumed_plain_full,
        );

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
                provider,
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
