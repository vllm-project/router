//! Advisory publishers can report invalidation without a cache lookup contract.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use super::{
    discovered_block_size, event_sources_for_worker, fetch_kv_event_sources,
    query_worker_metadata_and_models, server_instance_id, server_model_id, CacheKey,
    KvIndexSupervisor, SourcesProbe, WorkerKvInfo,
};
use crate::kv_index::source::events::subscriber::{spawn_observer, SubscriberHandle};
use crate::kv_index::SourceId;

pub(super) struct WorkerObservation {
    pub(super) source: SourceId,
    _subscribers: Vec<SubscriberHandle>,
}

impl KvIndexSupervisor {
    pub(super) async fn observe_worker(
        &self,
        worker_url: &str,
        observers: &Mutex<HashMap<String, WorkerObservation>>,
    ) -> Result<WorkerKvInfo, String> {
        let api_key = self.api_key.as_deref();
        let (sources, info) = tokio::join!(
            fetch_kv_event_sources(&self.client, worker_url, api_key),
            query_worker_metadata_and_models(&self.client, worker_url, api_key),
        );
        let sources = match sources {
            Ok(sources) => sources,
            Err(error) if error.probe == SourcesProbe::Unsupported => {
                return Ok(WorkerKvInfo::default());
            }
            Err(error) => return Err(error.detail),
        };
        let server_info = info.server_info?;
        let mut info = WorkerKvInfo {
            instance_id: server_instance_id(&server_info, worker_url),
            block_size: discovered_block_size(&server_info),
            ..Default::default()
        };
        let Some(block_size) = info.block_size else {
            return Ok(info);
        };
        let source: SourceId = Arc::from(format!(
            "{}#observation-{}",
            info.instance_id,
            crate::kv_index::next_owner_incarnation()
        ));
        let mut sources = event_sources_for_worker(worker_url, &source, &sources, false)?;
        if sources.is_empty() {
            return Ok(info);
        }
        sources.sort_unstable_by_key(|entry| entry.dp_rank);
        info.ranks = sources.iter().map(|source| source.dp_rank).collect();
        info.cache_key = Some(CacheKey {
            model: Arc::from(
                server_model_id(&server_info).unwrap_or_else(|| worker_url.to_owned()),
            ),
            block_size,
        });
        let subscribers = sources
            .into_iter()
            .map(|source| spawn_observer(source, self.ctx.clone(), self.signal_tx.clone()))
            .collect();
        observers.lock().insert(
            worker_url.to_owned(),
            WorkerObservation {
                source,
                _subscribers: subscribers,
            },
        );
        Ok(info)
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use axum::{routing::get, Json, Router};
    use serde_json::json;
    use std::time::Duration;
    use tokio::sync::mpsc;

    use crate::kv_index::source::test_support::{event_payload, full_store};
    use crate::kv_index::{wire::ExternalBlockHash, IngestionSignal, KVEvent};

    #[tokio::test]
    async fn advisory_observer_tracks_loss_without_a_cache_contract() {
        let ctx = zmq::Context::new();
        let publisher = ctx.socket(zmq::XPUB).unwrap();
        publisher.set_linger(0).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
        let app = Router::new()
            .route(
                "/get_server_info",
                get(|| async { Json(json!({"model_id": "model", "kv_block_size": 4})) }),
            )
            .route(
                "/kv_event_sources",
                get(move || {
                    let endpoint = endpoint.clone();
                    async move {
                        Json(json!({"0": {
                            "endpoint": endpoint,
                            "publisher": "zmq",
                            "topic": "kv"
                        }}))
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker_url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (supervisor, mut signals) = KvIndexSupervisor::new_observer(reqwest::Client::new());

        let info = supervisor.on_worker_added(&worker_url).await.unwrap();
        assert_eq!(info.ranks, vec![0]);
        assert!(info.cache_key.is_some());
        assert!(supervisor.installed.indexers.is_empty());
        assert!(supervisor.match_provider_for_worker(&worker_url).is_none());

        let subscribed = tokio::time::timeout(Duration::from_secs(4), async {
            loop {
                match publisher.recv_bytes(zmq::DONTWAIT) {
                    Ok(frame) => break frame,
                    Err(zmq::Error::EAGAIN) => tokio::time::sleep(Duration::from_millis(2)).await,
                    Err(error) => panic!("failed to receive subscription: {error}"),
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(subscribed, b"\x01kv");

        let publish = |seq: i64, events: &[KVEvent]| {
            let payload = event_payload(1.0, events, Some(0));
            publisher
                .send_multipart([b"kv".as_slice(), &seq.to_be_bytes(), &payload], 0)
                .unwrap();
        };
        async fn receive_signal(signals: &mut mpsc::Receiver<IngestionSignal>) -> IngestionSignal {
            tokio::time::timeout(Duration::from_secs(4), signals.recv())
                .await
                .unwrap()
                .unwrap()
        }

        let mut store = full_store(&[1, 2, 3, 4], 4, None);
        store.block_hashes[0] = ExternalBlockHash::Int(7);
        publish(15, &[KVEvent::BlockStored(store)]);
        publish(16, &[KVEvent::AllBlocksCleared(Default::default())]);
        let IngestionSignal::Invalidated {
            source, dp_rank: 0, ..
        } = receive_signal(&mut signals).await
        else {
            panic!("expected invalidation from the observed publisher");
        };
        assert_eq!(
            supervisor.worker_urls_for_source(&source),
            vec![worker_url.clone()]
        );

        publish(18, &[]);
        assert!(matches!(
            receive_signal(&mut signals).await,
            IngestionSignal::Gap {
                from_seq: 17,
                to_seq: 17,
                ..
            }
        ));
        publish(0, &[]);
        assert!(matches!(
            receive_signal(&mut signals).await,
            IngestionSignal::IncarnationReset { .. }
        ));
        supervisor.on_worker_removed(&worker_url);
        assert!(supervisor.worker_urls_for_source(&source).is_empty());
        server.abort();
    }
}
