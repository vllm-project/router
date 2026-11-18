#![cfg(all(test, feature = "cpu", feature = "ready"))]

#[test]
fn only_confirming_probes_renew_trust() {
    assert!(probe_renews_trust(&StaticHybridMatch::Matches));
    assert!(probe_renews_trust(&StaticHybridMatch::Unavailable(
        SourcesProbe::Unsupported
    )));
    assert!(!probe_renews_trust(&StaticHybridMatch::Unavailable(
        SourcesProbe::Failed
    )));
    assert!(!probe_renews_trust(&StaticHybridMatch::Differs));
    // A failed probe keeps the worker routable; only the credit expires.
    assert!(probe_keeps_trust(&StaticHybridMatch::Unavailable(
        SourcesProbe::Failed
    )));
    assert!(!probe_keeps_trust(&StaticHybridMatch::Differs));
}

#[test]
fn hybrid_freshness_expires_recovers_and_never_reactivates_retired_worker() {
    let active = Arc::new(AtomicBool::new(true));
    let lifecycle = Arc::new(WorkerLifecycle::new(active.clone(), true, None));
    lifecycle.state.lock().verified_at = Instant::now() - Duration::from_secs(6);
    let provider = HybridWorkerProvider {
        indexes: HashMap::new(),
        lifecycle,
    };
    assert!(!provider.is_trusted());
    assert!(provider.lifecycle.expire_if_needed());
    assert!(active.load(Ordering::Acquire));
    assert_eq!(
        provider
            .lifecycle
            .observe_layout(&StaticHybridMatch::Unavailable(SourcesProbe::Failed)),
        SourceAction::Keep
    );
    assert!(!provider.is_trusted());
    assert_eq!(
        provider
            .lifecycle
            .observe_layout(&StaticHybridMatch::Matches),
        SourceAction::Recover
    );
    assert!(!provider.is_trusted()); // Replay must be started before publication.
    provider.lifecycle.state.lock().trusted = true;
    assert!(provider.is_trusted());
    provider.lifecycle.state.lock().verified_at = Instant::now() - Duration::from_secs(6);
    assert!(provider.lifecycle.expire_if_needed());
    assert_eq!(
        provider
            .lifecycle
            .observe_layout(&StaticHybridMatch::Unavailable(SourcesProbe::Unsupported)),
        SourceAction::Recover
    );
    active.store(false, Ordering::Release);
    assert_eq!(
        provider
            .lifecycle
            .observe_layout(&StaticHybridMatch::Matches),
        SourceAction::Stop
    );
    assert!(!active.load(Ordering::Acquire));
    assert!(!provider.is_trusted());
}

use super::lifecycle::SourceAction;
use super::probe::worker_descriptor;
use super::reconcile::{bound_source_moved, probe_keeps_trust, probe_renews_trust, BoundSource};
use super::*;
use crate::core::worker::worker_get;
use crate::kv_index::source::test_support::{event_payload, stored_block as stored};
use crate::kv_index::subscriber::EventSource;
use crate::kv_index::wire::{ExternalBlockHash, KVEvent};
use crate::kv_index::{MatchQuery, StorageTier};
use std::sync::atomic::Ordering;
use std::time::Instant;

fn capability(
    model: &str,
    topic: &str,
    block_size: u32,
    groups: serde_json::Value,
) -> crate::kv_index::CacheCapabilities {
    serde_json::from_value(serde_json::json!({
        "schema_version": 1,
        "ranks": [{
            "rank": 0, "model": model, "weight_version": "v1",
            "publisher_topic": topic, "instance_topic": true,
            "hash_algorithm": "sha256_cbor", "full_hashes": true,
            "root_hash": "4e1195df020de59e0d65a33a4279f1183e7ae4e5d980e309f8b55adff2e61c3e",
            "hash_block_size": block_size, "hash_lookahead_tokens": 0,
            "speculative_tokens": 0, "prefix_caching": true, "groups": groups
        }]
    }))
    .unwrap()
}

#[tokio::test]
async fn configured_event_sources_skip_unavailable_control_discovery() {
    let publisher_context = zmq::Context::new();
    let publisher = publisher_context.socket(zmq::PUB).unwrap();
    publisher.bind("tcp://127.0.0.1:*").unwrap();
    let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
    for (worker_url, rank_count) in [
        ("grpc://127.0.0.1:1", 2),
        ("http://127.0.0.1:1/proxy", 2),
        ("http://127.0.0.1:1", 1),
    ] {
        let source = StaticKvEventSources {
            endpoints: vec![endpoint.clone(); rank_count],
            replay_endpoints: None,
            block_size: 32,
            model: "shared-model".into(),
            topic: "kv/generic".into(),
            hybrid: None,
        };
        let sources = HashMap::from([(worker_url.to_string(), source)]);
        // An explicit static entry must remain generic when
        // installed by the HTTP Router's full-attention constructor.
        let (supervisor, _signals) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
        let supervisor = supervisor.with_static_sources(sources);

        let mut info = supervisor.on_worker_added(worker_url).await.unwrap();
        assert_eq!(info.block_size, Some(32));
        info.ranks.sort_unstable();
        assert_eq!(info.ranks, (0..rank_count as u32).collect::<Vec<_>>());
        assert_eq!(info.cache_key.unwrap().model.as_ref(), "shared-model");
        assert!(supervisor.match_provider_for_worker(worker_url).is_some());
        assert!(supervisor.workers.lock()[worker_url]
            .worker_origin
            .is_none());
        supervisor.on_worker_removed(worker_url);
    }
}

#[test]
fn worker_auth_uses_only_the_configured_key() {
    let client = reqwest::Client::new();
    for (key, expected) in [
        (None, None),
        (Some("backend-key"), Some("Bearer backend-key")),
    ] {
        let request = worker_get(&client, "https://worker/v1/models", key)
            .build()
            .unwrap();
        assert_eq!(
            request
                .headers()
                .get("authorization")
                .map(|value| value.to_str().unwrap()),
            expected
        );
    }
}

#[test]
fn resolve_endpoint_rewrites_bind_star() {
    assert_eq!(
        resolve_endpoint("tcp://*:5557", "10.0.0.5").unwrap(),
        "tcp://10.0.0.5:5557"
    );
    assert_eq!(
        resolve_endpoint("tcp://0.0.0.0:5558", "10.0.0.5").unwrap(),
        "tcp://10.0.0.5:5558"
    );
}

#[test]
fn resolve_endpoint_keeps_resolved_host() {
    assert_eq!(
        resolve_endpoint("tcp://10.0.0.5:5557", "ignored").unwrap(),
        "tcp://10.0.0.5:5557"
    );
}

#[test]
fn resolve_endpoint_passthrough_non_tcp() {
    assert_eq!(
        resolve_endpoint("ipc:///tmp/x", "h").unwrap(),
        "ipc:///tmp/x"
    );
    assert_eq!(resolve_endpoint("inproc://x", "h").unwrap(), "inproc://x");
}

#[test]
fn bound_source_accepts_the_engines_bind_endpoints() {
    // The engine answers /kv_event_sources with the addresses it bound while
    // the installed source keeps the resolved ones; that is not a move.
    let installed = BoundSource {
        endpoint: "tcp://127.0.0.1:26258".to_string(),
        replay_endpoint: Some("tcp://127.0.0.1:27258".to_string()),
    };
    let actual = KvEventSourceEntry {
        publisher: Some("zmq".to_string()),
        endpoint: "tcp://*:26258".to_string(),
        replay_endpoint: Some("tcp://*:27258".to_string()),
        ..KvEventSourceEntry::default()
    };
    assert!(!bound_source_moved("127.0.0.1", &installed, &actual));

    let moved_publisher = KvEventSourceEntry {
        endpoint: "tcp://*:26259".to_string(),
        ..actual.clone()
    };
    assert!(bound_source_moved(
        "127.0.0.1",
        &installed,
        &moved_publisher
    ));

    let moved_replay = KvEventSourceEntry {
        replay_endpoint: Some("tcp://*:27259".to_string()),
        ..actual.clone()
    };
    assert!(bound_source_moved("127.0.0.1", &installed, &moved_replay));

    let no_replay = KvEventSourceEntry {
        replay_endpoint: None,
        ..actual
    };
    assert!(bound_source_moved("127.0.0.1", &installed, &no_replay));
}

#[test]
fn resolved_tcp_sources_reuse_connect_validation_without_http_worker_admission() {
    let host = worker_host("grpc://[::1]:50051").unwrap();
    assert_eq!(
        resolve_endpoint("tcp://[::]:5557", &host).unwrap(),
        "tcp://[::1]:5557"
    );
    assert_eq!(
        resolve_endpoint("tcp://[::ffff:127.0.0.1]:5557", &host).unwrap(),
        "tcp://127.0.0.1:5557"
    );
    assert!(resolve_endpoint("tcp://*:0", &host).is_err());
}

#[test]
fn worker_host_extracts_from_url() {
    assert_eq!(
        worker_host("http://10.0.0.5:8000").as_deref(),
        Some("10.0.0.5")
    );
    assert_eq!(worker_host("10.0.0.5:8000").as_deref(), Some("10.0.0.5"));
    assert_eq!(
        worker_host("http://host.example:80").as_deref(),
        Some("host.example")
    );
}

#[test]
fn events_enabled_gates_on_publisher_and_endpoint() {
    let mut e = KvEventSourceEntry::default();
    assert!(!events_enabled(&e)); // empty endpoint
    e.endpoint = "tcp://*:5557".into();
    e.publisher = Some("zmq".into());
    // vLLM 0.33 omits the optional wire metadata; "zmq" alone is enough.
    assert!(events_enabled(&e));
    e.encoding = Some("msgpack".into());
    assert!(events_enabled(&e));
    e.schema_version = Some(1);
    assert!(events_enabled(&e));
    e.publisher = Some("null".into());
    assert!(!events_enabled(&e));
    e.publisher = Some("other".into());
    assert!(!events_enabled(&e));
    e.publisher = Some("zmq".into());
    e.enable_kv_cache_events = Some(false);
    assert!(!events_enabled(&e));
    e.enable_kv_cache_events = Some(true);
    assert!(events_enabled(&e));
    e.encoding = Some("json".into());
    assert!(!events_enabled(&e));
    e.encoding = Some("msgpack".into());
    e.schema_version = Some(2);
    assert!(!events_enabled(&e));
}

#[test]
fn grpc_model_id_preserves_errors_and_rejects_empty_ids() {
    let error = "GetModelInfo unavailable".to_string();
    assert_eq!(grpc_model_id(Err(error.clone())), Err(error));
    assert_eq!(
        grpc_model_id(Ok(crate::backend::pb::ModelInfo::default())),
        Ok(None)
    );
    assert_eq!(
        grpc_model_id(Ok(crate::backend::pb::ModelInfo {
            model_id: "   ".into(),
            ..Default::default()
        })),
        Ok(None)
    );
    assert_eq!(
        grpc_model_id(Ok(crate::backend::pb::ModelInfo {
            model_id: "model-a".into(),
            ..Default::default()
        })),
        Ok(Some("model-a".into()))
    );
}

#[test]
fn server_info_parses_kv_fields_and_aliases() {
    let s: ServerInfo = serde_json::from_str(
        r#"{"model_id":"m","kv_block_size":16,"instance_id":"x","data_parallel_size":2}"#,
    )
    .unwrap();
    assert_eq!(s.model_id.as_deref(), Some("m"));
    assert_eq!(s.kv_block_size, Some(16));
    assert_eq!(s.instance_id.as_deref(), Some("x"));
    // `data_parallel_size` is tolerated but unused; rank comes from /kv_event_sources.
    // block_size alias.
    let s: ServerInfo = serde_json::from_str(r#"{"block_size":32}"#).unwrap();
    assert_eq!(s.kv_block_size, Some(32));
    // empty body → defaults, no error.
    let s: ServerInfo = serde_json::from_str("{}").unwrap();
    assert!(s.kv_block_size.is_none());
}

#[test]
fn warnings_name_the_transport_that_was_probed() {
    assert_eq!(
        metadata_source_label("grpc://127.0.0.1:50051"),
        "GetServerInfo"
    );
    assert_eq!(
        metadata_source_label("http://127.0.0.1:8000"),
        "/get_server_info, /server_info or /metrics"
    );
    assert_eq!(
        event_source_label("grpc://127.0.0.1:50051"),
        "GetKvEventSources"
    );
    assert_eq!(
        event_source_label("http://127.0.0.1:8000"),
        "/kv_event_sources"
    );
}

#[test]
fn zero_block_size_is_treated_as_missing() {
    let server_info = ServerInfo {
        kv_block_size: Some(0),
        effective_attention_block_size: Some(16),
        ..Default::default()
    };
    assert_eq!(discovered_block_size(&server_info), Some(16));

    let server_info = ServerInfo {
        kv_block_size: Some(0),
        effective_attention_block_size: Some(0),
        ..Default::default()
    };
    assert_eq!(discovered_block_size(&server_info), None);
}

#[test]
fn kv_event_sources_decodes_rank_map() {
    let body = r#"{
            "0": {"endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558","topic":"","hwm":100000,"publisher":"zmq","encoding":"msgpack","schema_version":1,"enable_kv_cache_events":true},
            "1": {"endpoint":"tcp://*:5559","replay_endpoint":null,"topic":"","publisher":"zmq","encoding":"msgpack","schema_version":1}
        }"#;
    let m: KvEventSourcesResponse = serde_json::from_str(body).unwrap();
    assert_eq!(m.len(), 2);
    assert_eq!(m["0"].hwm, Some(100_000));
    assert_eq!(m["0"].encoding.as_deref(), Some("msgpack"));
    assert_eq!(m["0"].schema_version, Some(1));
    assert!(m["1"].replay_endpoint.is_none());
}

#[test]
fn grpc_sources_without_rank_are_skipped_instead_of_positionally_assigned() {
    let source = |rank| crate::backend::pb::KvEventSource {
        transport: "zmq".to_string(),
        endpoint: "tcp://worker:5557".to_string(),
        topic: String::new(),
        replay_endpoint: String::new(),
        data_parallel_rank: rank,
        encoding: "msgpack".to_string(),
        schema_version: 1,
        buffer_steps: 1,
        hwm: 100,
        max_queue_size: 0,
    };

    let entries =
        grpc_kv_event_source_entries("grpc://worker:8000", vec![source(None), source(Some(3))]);

    assert_eq!(entries.len(), 1);
    assert!(entries.contains_key("3"));
    assert_eq!(entries["3"].publisher.as_deref(), Some("zmq"));
    assert_eq!(entries["3"].enable_kv_cache_events, Some(true));
}

#[test]
fn static_regular_sources_preserve_mapping_alias_and_topic() {
    for (worker_count, block_size, topic) in [
        (2, 16, Some("kv/alias-v2")),
        (1, 8, None),
        (1, 32, Some("")),
    ] {
        let workers: Vec<_> = (0..worker_count)
            .map(|worker| format!("http://worker-{worker}:8000"))
            .collect();
        let mut map = serde_json::Map::new();
        for (worker, url) in workers.iter().enumerate() {
            let mut source = serde_json::json!({
                "endpoints": [format!("tcp://worker-{worker}:5557")],
                "replay_endpoints": [format!("tcp://worker-{worker}:5558")],
                "block_size": block_size,
                "model": "served-alias-v2"
            });
            if let Some(topic) = topic {
                source["topic"] = serde_json::json!(topic);
            }
            map.insert(url.clone(), source);
        }
        let sources: HashMap<String, StaticKvEventSources> =
            serde_json::from_value(serde_json::Value::Object(map)).unwrap();
        assert_eq!(sources.len(), worker_count);
        for worker in &workers {
            let discovery = static_worker_discovery(worker, &sources[worker]);
            assert_eq!(discovery.model_id, Ok(Some("served-alias-v2".into())));
            let info = discovery.server_info.unwrap();
            assert_eq!(info.instance_id.as_deref(), Some(worker.as_str()));
            assert_eq!(discovered_block_size(&info), Some(block_size));
            let entries = discovery.sources.unwrap();
            assert_eq!(entries.len(), 1);
            let entry = &entries["0"];
            assert_eq!(entry.endpoint, sources[worker].endpoints[0]);
            assert_eq!(
                entry.replay_endpoint,
                sources[worker]
                    .replay_endpoints
                    .as_ref()
                    .map(|endpoints| endpoints[0].clone())
            );
            assert_eq!(entry.topic, topic.unwrap_or_default());
            assert_eq!(entry.hwm, None);
            assert!(events_enabled(entry));
        }
    }
}

async fn source_boundary_worker(
    publisher: &str,
    replay: &str,
    instance: &str,
) -> (String, tokio::task::JoinHandle<()>) {
    use serde_json::json;

    let info = json!({"model_id": "boundary-model", "instance_id": instance, "kv_block_size": 8});
    let sources = json!({"0": {
        "endpoint": publisher, "replay_endpoint": replay, "topic": "kv/boundary",
        "publisher": "zmq", "encoding": "msgpack", "schema_version": 1,
        "enable_kv_cache_events": true
    }});
    source_boundary_worker_with_metadata(info, sources).await
}

async fn source_boundary_worker_with_metadata(
    info: serde_json::Value,
    sources: serde_json::Value,
) -> (String, tokio::task::JoinHandle<()>) {
    use axum::{routing::get, Json, Router};

    let app = Router::new()
        .route(
            "/get_server_info",
            get(move || {
                let info = info.clone();
                async move { Json(info) }
            }),
        )
        .route(
            "/kv_event_sources",
            get(move || {
                let sources = sources.clone();
                async move { Json(sources) }
            }),
        );
    let app = app.clone().nest("/proxy", app);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (worker, task)
}

async fn boundary_subscription(publisher: &zmq::Socket, topic: &str, skip_unsubscribe: bool) {
    let subscribed = [b"\x01".as_slice(), topic.as_bytes()].concat();
    let unsubscribed = [b"\x00".as_slice(), topic.as_bytes()].concat();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match publisher.recv_bytes(zmq::DONTWAIT) {
                Ok(frame) if skip_unsubscribe && frame == unsubscribed => continue,
                Ok(frame) => {
                    assert_eq!(frame, subscribed);
                    break;
                }
                Err(zmq::Error::EAGAIN) => tokio::task::yield_now().await,
                Err(error) => panic!("boundary subscription: {error}"),
            }
        }
    })
    .await
    .unwrap();
}

async fn recv_frames(socket: &zmq::Socket) -> Vec<Vec<u8>> {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match socket.recv_multipart(zmq::DONTWAIT) {
                Ok(frames) => break frames,
                Err(zmq::Error::EAGAIN) => tokio::task::yield_now().await,
                Err(error) => panic!("ZMQ receive: {error}"),
            }
        }
    })
    .await
    .unwrap()
}

fn replay_reply(socket: &zmq::Socket, request: &[Vec<u8>], topic: &[u8], seq: i64, payload: &[u8]) {
    let topic = if seq == -1 { &[][..] } else { topic };
    socket
        .send_multipart(
            [
                request[0].as_slice(),
                &[],
                topic,
                &seq.to_be_bytes(),
                payload,
            ],
            0,
        )
        .unwrap();
}

fn live_frame(socket: &zmq::Socket, topic: &[u8], seq: i64, payload: &[u8]) {
    socket
        .send_multipart([topic, &seq.to_be_bytes(), payload], 0)
        .unwrap();
}

async fn finish_empty_startup_replay(replay: &zmq::Socket) {
    let request = recv_frames(replay).await;
    assert_eq!(request.last().unwrap(), &0_i64.to_be_bytes());
    replay_reply(replay, &request, &[], -1, &[]);
}

fn stored_block(hash: i64, tokens: &[u32], block_size: u32, medium: &str) -> KVEvent {
    let mut event = stored(
        vec![ExternalBlockHash::Int(hash.try_into().unwrap())],
        tokens.to_vec(),
        block_size,
    );
    event.medium = Some(medium.into());
    KVEvent::BlockStored(event)
}

fn store_payload(
    hash: i64,
    tokens: &[u32],
    block_size: u32,
    medium: &str,
    rank: Option<u32>,
) -> Vec<u8> {
    event_payload(1.0, &[stored_block(hash, tokens, block_size, medium)], rank)
}

fn block_query(tokens: &[u32], block_size: u32, tiers: &[StorageTier]) -> MatchQuery {
    MatchQuery {
        group_idx: 0,
        local_hashes: crate::kv_index::local_block_hash_strings(tokens, block_size, None),
        tiers_of_interest: tiers.to_vec(),
    }
}

#[tokio::test]
async fn static_hybrid_sources_require_matching_rank_and_generation_topic() {
    use axum::{routing::get, Json, Router};

    let sources = Arc::new(tokio::sync::RwLock::new(serde_json::json!({
        "0": {"endpoint":"tcp://*:5557", "replay_endpoint":"tcp://*:5558", "topic":"boot-1-rank-0", "publisher":"zmq"},
        "1": {"endpoint":"tcp://*:5559", "replay_endpoint":"tcp://*:5560", "topic":"boot-1-rank-1", "publisher":"zmq"}
    })));
    let app = Router::new().route(
        "/kv_event_sources",
        get({
            let sources = sources.clone();
            move || {
                let sources = sources.clone();
                async move { Json(sources.read().await.clone()) }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let mut layout = capability("model", "boot-1-rank-0", 16, serde_json::json!([]));
    layout.ranks[0].root_hash = None;
    let mut second = layout.ranks[0].clone();
    second.rank = 1;
    second.publisher_topic = Some("boot-1-rank-1".into());
    layout.ranks.push(second);
    let config = StaticKvEventSources {
        endpoints: vec!["tcp://127.0.0.1:5557".into(), "tcp://127.0.0.1:5559".into()],
        replay_endpoints: Some(vec![
            "tcp://127.0.0.1:5558".into(),
            "tcp://127.0.0.1:5560".into(),
        ]),
        topic: String::new(),
        block_size: 16,
        model: "model".into(),
        hybrid: Some(layout),
    };
    let client = reqwest::Client::new();
    assert_eq!(
        static_hybrid_sources_match(&client, &worker, None, &config, None).await,
        StaticHybridMatch::Matches
    );
    sources.write().await["1"]["topic"] = serde_json::json!("boot-2-rank-1");
    assert_eq!(
        static_hybrid_sources_match(&client, &worker, None, &config, None).await,
        StaticHybridMatch::Differs
    );
    task.abort();
}

#[tokio::test]
async fn layout_only_sources_retire_when_the_engine_changes_its_topic() {
    use axum::{routing::get, Json, Router};

    let sources = Arc::new(tokio::sync::RwLock::new(serde_json::json!({
        "0": {"endpoint":"tcp://*:5557", "replay_endpoint":"tcp://*:5558", "topic":"kv-hybrid", "publisher":"zmq"}
    })));
    let app = Router::new().route(
        "/kv_event_sources",
        get({
            let sources = sources.clone();
            move || {
                let sources = sources.clone();
                async move { Json(sources.read().await.clone()) }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let mut layout = capability("model", "kv-hybrid", 16, serde_json::json!([]));
    layout.ranks[0].root_hash = None;
    // A layout-only binding declares no endpoints; the engine names them.
    let config = StaticKvEventSources {
        endpoints: Vec::new(),
        replay_endpoints: None,
        topic: String::new(),
        block_size: 16,
        model: "model".into(),
        hybrid: Some(layout),
    };
    let client = reqwest::Client::new();
    assert_eq!(
        static_hybrid_sources_match(&client, &worker, None, &config, None).await,
        StaticHybridMatch::Matches
    );
    // A restarted engine that advertises a new topic belongs to another boot.
    sources.write().await["0"]["topic"] = serde_json::json!("kv-hybrid-boot-2");
    assert_eq!(
        static_hybrid_sources_match(&client, &worker, None, &config, None).await,
        StaticHybridMatch::Differs
    );
    // An engine that advertises nothing cannot keep the old binding alive.
    *sources.write().await = serde_json::json!({});
    assert_eq!(
        static_hybrid_sources_match(&client, &worker, None, &config, None).await,
        StaticHybridMatch::Differs
    );
    task.abort();
}

#[tokio::test]
async fn hybrid_discovery_binds_only_ranks_for_the_worker_model() {
    let rank = |rank: u32, block_size: u32| {
        let mut layout = capability(
            "qwen",
            &format!("rank-{rank}"),
            block_size,
            serde_json::json!([
                {"group_idx":0,"kind":"mamba","spec_type":"MambaSpec","block_size":block_size,
                 "sliding_window":null,"prefix_cacheable":true,"prefix_replay_tokens":0,
                 "mamba_cache_mode":"align","prefill_checkpoint_alignment":block_size},
                {"group_idx":1,"kind":"full_attention","spec_type":"FullAttentionSpec","block_size":block_size,
                 "sliding_window":null,"prefix_cacheable":true,"prefix_replay_tokens":0}
            ]),
        );
        layout.ranks[0].rank = rank;
        layout.ranks[0].weight_version = "default".into();
        layout
    };
    let mut capability = rank(0, 16);
    capability.ranks.extend(rank(1, 32).ranks);
    let sources: KvEventSourcesResponse = serde_json::from_value(serde_json::json!({
            "0":{"publisher":"zmq","encoding":"msgpack","schema_version":1,
                 "endpoint":"inproc://rank0-pub","replay_endpoint":"inproc://rank0-replay","topic":"rank-0"},
            "1":{"publisher":"zmq","encoding":"msgpack","schema_version":1,
                 "endpoint":"inproc://rank1-pub","replay_endpoint":"inproc://rank1-replay","topic":"rank-1"}
        })).unwrap();
    let discovery = |capability| WorkerDiscovery {
        server_info: Ok(ServerInfo {
            instance_id: Some("worker".into()),
            model_id: Some("qwen".into()),
            kv_block_size: Some(576),
            ..Default::default()
        }),
        sources: Ok(sources.clone()),
        model_id: Ok(Some("qwen".into())),
        hybrid: Some(capability),
    };
    let (supervisor, _signals) =
        KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
    let worker_url = "http://127.0.0.1:8080";
    for (model, expected) in [("qwen", &[0, 1][..]), ("other", &[0][..])] {
        capability.ranks[1].model = model.into();
        let info = supervisor
            .install_worker(worker_url, discovery(capability.clone()))
            .unwrap();
        assert_eq!(info.block_size, Some(16));
        let workers = supervisor.workers.lock();
        let provider = workers[worker_url].hybrid.as_ref().unwrap();
        let mut bound: Vec<_> = provider.indexes.keys().copied().collect();
        bound.sort_unstable();
        assert_eq!(bound, expected);
    }
}

#[tokio::test]
async fn static_hybrid_routes_from_live_events_and_retires_changed_source() {
    exercise_static_hybrid(false).await;
}

#[tokio::test]
async fn pd_static_hybrid_routes_from_live_events_and_retires_changed_source() {
    exercise_static_hybrid(true).await;
}

async fn exercise_static_hybrid(pd: bool) {
    use crate::policies::{KvAwareOps, KvAwarePolicy};
    use axum::{routing::get, Json, Router};

    let group = |idx, kind, spec, mode| {
        serde_json::json!({
            "group_idx":idx, "kind":kind, "spec_type":spec, "block_size":528,
            "sliding_window":null, "prefix_cacheable":true,
            "prefix_replay_tokens":0, "mamba_cache_mode":mode,
            "prefill_checkpoint_alignment": if kind == "mamba" { Some(528) } else { None }
        })
    };
    let mut capability = capability(
        "qwen",
        "hybrid/epoch-1",
        528,
        serde_json::json!([
            group(0, "mamba", "MambaSpec", Some("align")),
            group(1, "mamba", "MambaSpec", Some("align")),
            group(2, "mamba", "MambaSpec", Some("align")),
            group(3, "full_attention", "FullAttentionSpec", None)
        ]),
    );
    capability.ranks[0].weight_version = "default".into();
    let sources = Arc::new(tokio::sync::RwLock::new(serde_json::json!({"0":{
        "endpoint":"inproc://hybrid-discovery-pub", "replay_endpoint":"inproc://hybrid-discovery-replay",
        "topic":"hybrid/epoch-1", "publisher":"zmq", "enable_kv_cache_events":true
    }})));
    let fail_probe = Arc::new(AtomicBool::new(false));
    let failed_probes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let app = Router::new().route(
        "/kv_event_sources",
        get({
            let sources = sources.clone();
            let fail_probe = fail_probe.clone();
            let failed_probes = failed_probes.clone();
            move || {
                let sources = sources.clone();
                let fail_probe = fail_probe.clone();
                let failed_probes = failed_probes.clone();
                async move {
                    let status = if fail_probe.load(Ordering::Acquire) {
                        failed_probes.fetch_add(1, Ordering::AcqRel);
                        axum::http::StatusCode::SERVICE_UNAVAILABLE
                    } else {
                        axum::http::StatusCode::OK
                    };
                    (status, Json(sources.read().await.clone()))
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker_url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let config = StaticKvEventSources {
        endpoints: vec!["inproc://hybrid-discovery-pub".into()],
        replay_endpoints: Some(vec!["inproc://hybrid-discovery-replay".into()]),
        topic: String::new(),
        block_size: 528,
        model: "qwen".into(),
        hybrid: Some(capability),
    };
    let (supervisor, _signals) = if pd {
        let source = crate::config::PdKvEventSources {
            endpoints: config.endpoints,
            replay_endpoints: config.replay_endpoints,
            block_size: config.block_size,
            model: Some(config.model),
            hybrid: config.hybrid,
            compatible_decode_urls: vec!["http://decode".into()],
        };
        KvIndexSupervisor::new_pd(
            reqwest::Client::new(),
            HashMode::Sha256,
            HashMap::from([(worker_url.clone(), source)]),
        )
    } else {
        let (supervisor, signals) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
        (
            supervisor.with_static_sources(HashMap::from([(worker_url.clone(), config)])),
            signals,
        )
    };
    let publisher = supervisor.ctx.socket(zmq::XPUB).unwrap();
    let replay = supervisor.ctx.socket(zmq::ROUTER).unwrap();
    publisher.set_linger(0).unwrap();
    replay.set_linger(0).unwrap();
    publisher.bind("inproc://hybrid-discovery-pub").unwrap();
    replay.bind("inproc://hybrid-discovery-replay").unwrap();
    let info = supervisor.on_worker_added(&worker_url).await.unwrap();
    let provider = supervisor.match_provider_for_worker(&worker_url).unwrap();
    boundary_subscription(&publisher, "hybrid/epoch-1", false).await;
    let payload = crate::kv_index::source::binding::tests::qwen35_payload();
    let request = recv_frames(&replay).await;
    replay_reply(&replay, &request, b"hybrid/epoch-1", 0, payload.as_slice());
    live_frame(&publisher, b"hybrid/epoch-1", 1_i64, payload.as_slice());
    replay_reply(&replay, &request, &[], -1, &[]);
    let prompt: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
    tokio::time::timeout(Duration::from_secs(3), async {
        while provider.matches(&prompt, 528, None, 0).is_empty() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    let worker: Arc<dyn crate::core::Worker> = Arc::new(crate::core::BasicWorker::new(
        worker_url.clone(),
        crate::core::WorkerType::Regular,
    ));
    let policy = KvAwarePolicy::default();
    policy.register_worker(worker.as_ref(), Some(&info), Some(provider.clone()));
    let estimate =
        policy.estimate_prefill_work(&[(worker.as_ref(), std::slice::from_ref(&prompt), true)]);
    assert_eq!(estimate[0].new_prefill_tokens, 496);
    let hybrid = supervisor
        .workers
        .lock()
        .get(&worker_url)
        .unwrap()
        .hybrid
        .clone()
        .unwrap();
    fail_probe.store(true, Ordering::Release);
    tokio::time::timeout(Duration::from_secs(5), async {
        while failed_probes.load(Ordering::Acquire) == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    hybrid.lifecycle.state.lock().verified_at = Instant::now() - Duration::from_secs(6);
    tokio::time::timeout(Duration::from_secs(5), async {
        while hybrid.lifecycle.state.lock().trusted {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert!(hybrid.lifecycle.active.load(Ordering::Acquire));
    assert!(provider.matches(&prompt, 528, None, 0).is_empty());
    if pd {
        let revision = supervisor.worker_registrations()[&worker_url].revision;
        hybrid.lifecycle.state.lock().checked = Instant::now() - Duration::from_secs(30);
        let refreshed = supervisor
            .refresh_workers(&HashSet::from([worker_url.clone()]))
            .await;
        assert_eq!(
            refreshed[&worker_url].revision, revision,
            "failed control probe replaced an expired layout registration"
        );
        assert!(
            !hybrid.is_trusted(),
            "descriptor refresh renewed failed layout evidence"
        );
    }
    // XPUB coalesces duplicate subscriptions for a topic. Wait for
    // the old SUB to disconnect before admitting the replacement.
    let unsubscribed = [b"\x00".as_slice(), b"hybrid/epoch-1"].concat();
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            match publisher.recv_bytes(zmq::DONTWAIT) {
                Ok(frame) if frame == unsubscribed => break,
                Ok(_) => continue,
                Err(zmq::Error::EAGAIN) => tokio::time::sleep(Duration::from_millis(1)).await,
                Err(error) => panic!("unsubscribe: {error}"),
            }
        }
    })
    .await
    .unwrap();
    fail_probe.store(false, Ordering::Release);
    tokio::time::timeout(Duration::from_secs(5), async {
        while !hybrid.lifecycle.state.lock().trusted {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    boundary_subscription(&publisher, "hybrid/epoch-1", true).await;
    assert!(provider.matches(&prompt, 528, None, 0).is_empty());
    let request = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            match replay.recv_multipart(zmq::DONTWAIT) {
                // The previous subscriber may have queued an idle check.
                Ok(frames) if frames.last() == Some(&0_u64.to_be_bytes().to_vec()) => break frames,
                Ok(_) => continue,
                Err(zmq::Error::EAGAIN) => tokio::time::sleep(Duration::from_millis(1)).await,
                Err(error) => panic!("replay: {error}"),
            }
        }
    })
    .await
    .unwrap();
    replay_reply(&replay, &request, b"hybrid/epoch-1", 0, payload.as_slice());
    live_frame(&publisher, b"hybrid/epoch-1", 1_i64, payload.as_slice());
    replay_reply(&replay, &request, &[], -1, &[]);
    tokio::time::timeout(Duration::from_secs(3), async {
        while provider.matches(&prompt, 528, None, 0).is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    sources.write().await["0"]["topic"] = serde_json::json!("hybrid/epoch-2");
    tokio::time::timeout(Duration::from_secs(5), async {
        while !provider.matches(&prompt, 528, None, 0).is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    supervisor.shutdown();
    server.abort();
}

#[tokio::test]
async fn source_installation_preserves_topic_rank_and_cache_identity_boundaries() {
    for (case, (topic_b, rank_b, model_b, same_publisher, accepted)) in [
        ("kv/other", 0u32, "boundary-model", true, true),
        ("kv/boundary", 1, "boundary-model", true, true),
        ("kv/boundary", 0, "other-model", false, true),
        ("kv/boundary", 0, "other-model", true, false),
        ("kv/boundary", 0, "boundary-model", true, false),
        ("kv/boundary", 0, "boundary-model", false, false),
    ]
    .into_iter()
    .enumerate()
    {
        let endpoint_a = format!("inproc://source-boundary-{case}-a");
        let endpoint_b = format!("inproc://source-boundary-{case}-b");
        let discovery =
            |model: &str, instance: &str, endpoint: &str, topic: &str, rank: u32| WorkerDiscovery {
                hybrid: None,
                server_info: Ok(ServerInfo {
                    instance_id: Some(instance.into()),
                    kv_block_size: Some(8),
                    ..Default::default()
                }),
                model_id: Ok(Some(model.into())),
                sources: Ok(HashMap::from([(
                    rank.to_string(),
                    KvEventSourceEntry {
                        endpoint: endpoint.into(),
                        topic: topic.into(),
                        publisher: Some("zmq".into()),
                        encoding: Some("msgpack".into()),
                        schema_version: Some(1),
                        enable_kv_cache_events: Some(true),
                        ..Default::default()
                    },
                )])),
            };
        let worker_a = "http://worker-a:8000";
        let worker_b = "http://worker-b:8000";
        let (supervisor, _) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
        let a = supervisor
            .install_worker(
                worker_a,
                discovery(
                    "boundary-model",
                    "shared-target",
                    &endpoint_a,
                    "kv/boundary",
                    0,
                ),
            )
            .unwrap();
        assert_eq!(a.ranks, [0]);
        assert_eq!(
            a.cache_key.as_ref().unwrap().model.as_ref(),
            "boundary-model"
        );
        let b = supervisor.install_worker(
            worker_b,
            discovery(
                model_b,
                if topic_b == "kv/other" {
                    "other-topic-target"
                } else {
                    "shared-target"
                },
                if same_publisher {
                    &endpoint_a
                } else {
                    &endpoint_b
                },
                topic_b,
                rank_b,
            ),
        );
        if accepted {
            let b = b.unwrap();
            assert_eq!(b.ranks, [rank_b]);
            assert_eq!(b.cache_key.as_ref().unwrap().model.as_ref(), model_b);
            assert_eq!(supervisor.workers.lock().len(), 2);
            assert_eq!(
                supervisor.indexers.len(),
                if model_b == "boundary-model" { 1 } else { 2 }
            );
            if same_publisher && rank_b == 1 {
                let workers = supervisor.workers.lock();
                let guard_a = &workers[worker_a].live_sources[0].lease.strict_rank;
                let guard_b = &workers[worker_b].live_sources[0].lease.strict_rank;
                assert!(Arc::ptr_eq(guard_a, guard_b));
                assert!(*guard_a.lock());
            }
        } else {
            let error = b.unwrap_err();
            assert!(
                error.contains(if same_publisher {
                    "live KV streams"
                } else {
                    "KV index targets"
                }),
                "{error}"
            );
            assert_eq!(supervisor.workers.lock().len(), 1);
            assert!(!supervisor.workers.lock().contains_key(worker_b));
            assert_eq!(supervisor.indexers.len(), 1);
        }
        assert!(supervisor.workers.lock().contains_key(worker_a));
        assert_eq!(
            supervisor
                .identity_refs
                .lock()
                .values()
                .copied()
                .sum::<usize>(),
            if accepted { 2 } else { 1 }
        );
        supervisor.shutdown();
        assert!(supervisor.workers.lock().is_empty());
        assert!(supervisor.indexers.is_empty());
        assert!(supervisor.identity_refs.lock().is_empty());
    }
}

#[tokio::test]
async fn source_group_rejects_duplicate_normalized_ranks_without_mutation() {
    let entry = KvEventSourceEntry {
        endpoint: "inproc://duplicate-normalized-rank".into(),
        publisher: Some("zmq".into()),
        encoding: Some("msgpack".into()),
        schema_version: Some(1),
        enable_kv_cache_events: Some(true),
        ..Default::default()
    };
    let (supervisor, _) =
        KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
    assert!(supervisor
        .install_worker(
            "http://worker:8000",
            WorkerDiscovery {
                hybrid: None,
                server_info: Ok(ServerInfo {
                    instance_id: Some("duplicate-rank".into()),
                    kv_block_size: Some(8),
                    ..Default::default()
                }),
                model_id: Ok(Some("boundary-model".into())),
                sources: Ok(HashMap::from([
                    ("0".into(), entry.clone()),
                    ("00".into(), entry)
                ])),
            }
        )
        .unwrap_err()
        .contains("duplicate normalized"));
    assert!(supervisor.workers.lock().is_empty());
    assert!(supervisor.indexers.is_empty());
    assert!(supervisor.identity_refs.lock().is_empty());
}

#[tokio::test]
async fn failed_readd_preserves_registration_then_accepted_readd_retires_it() {
    use axum::{routing::get, Json, Router};
    use serde_json::json;

    let context = zmq::Context::new();
    let publisher_a = context.socket(zmq::XPUB).unwrap();
    let publisher_b = context.socket(zmq::XPUB).unwrap();
    for publisher in [&publisher_a, &publisher_b] {
        publisher.set_xpub_verbose(true).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
    }
    let source = |endpoint: &str| {
        json!({"0": {"endpoint": endpoint, "publisher": "zmq",
            "topic": "kv/boundary", "encoding": "msgpack", "schema_version": 1,
            "enable_kv_cache_events": true}})
    };
    let endpoint_a = publisher_a.get_last_endpoint().unwrap().unwrap();
    let endpoint_b = publisher_b.get_last_endpoint().unwrap().unwrap();
    let sources_a = Arc::new(Mutex::new(source(&endpoint_a)));
    let mutable_sources = sources_a.clone();
    let app = Router::new()
        .route(
            "/get_server_info",
            get(|| async {
                Json(json!({
                    "model_id": "boundary-model", "instance_id": "readd-a", "kv_block_size": 8
                }))
            }),
        )
        .route(
            "/kv_event_sources",
            get(move || {
                let value = mutable_sources.lock().clone();
                async move { Json(value) }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker_a = format!("http://{}", listener.local_addr().unwrap());
    let task_a = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let (worker_b, task_b) = source_boundary_worker_with_metadata(
        json!({"model_id": "boundary-model", "instance_id": "readd-b", "kv_block_size": 8}),
        source(&endpoint_b),
    )
    .await;
    let (supervisor, mut signals) =
        KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
    supervisor.on_worker_added(&worker_a).await.unwrap();
    supervisor.on_worker_added(&worker_b).await.unwrap();
    boundary_subscription(&publisher_a, "kv/boundary", true).await;
    let payload = store_payload(1, &[1; 8], 8, "GPU", Some(0));
    let query = block_query(&[1; 8], 8, &[StorageTier::Device]);
    live_frame(&publisher_a, b"kv/boundary", 0i64, &payload);
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(5), signals.recv())
            .await
            .unwrap(),
        Some(IngestionSignal::Advance { last_seq: 0, .. })
    ));
    let provider = supervisor.match_provider_for_worker(&worker_a).unwrap();
    assert_eq!(provider.find_tiered_matches(&query).len(), 1);
    let original = supervisor.workers.lock()[&worker_a].active.clone();
    *sources_a.lock() = source(&endpoint_b);
    assert!(supervisor
        .on_worker_added(&worker_a)
        .await
        .unwrap_err()
        .contains("live KV streams"));
    assert!(original.load(Ordering::Acquire));
    assert!(Arc::ptr_eq(
        &original,
        &supervisor.workers.lock()[&worker_a].active
    ));
    assert!(supervisor.match_provider_for_worker(&worker_a).is_some());
    assert_eq!(provider.find_tiered_matches(&query).len(), 1);
    let valid_entry = source(&endpoint_a)["0"].clone();
    let mut bad_publisher = valid_entry.clone();
    bad_publisher["endpoint"] = json!("tcp://127.0.0.1:0");
    let mut bad_replay = valid_entry.clone();
    bad_replay["replay_endpoint"] = json!("tcp://127.0.0.1:0");
    for (case, invalid_sources) in [
        ("bad rank", json!({"not-a-rank": valid_entry})),
        ("bad publisher", json!({"0": bad_publisher})),
        ("bad replay", json!({"0": bad_replay})),
        (
            "mixed valid and malformed",
            json!({"0": valid_entry, "1": bad_publisher}),
        ),
    ] {
        *sources_a.lock() = invalid_sources;
        assert!(
            supervisor.on_worker_added(&worker_a).await.is_err(),
            "{case} must reject the complete re-add"
        );
        assert!(original.load(Ordering::Acquire), "{case}");
        assert!(
            Arc::ptr_eq(&original, &supervisor.workers.lock()[&worker_a].active),
            "{case}"
        );
        assert!(
            supervisor.match_provider_for_worker(&worker_a).is_some(),
            "{case}"
        );
        assert_eq!(provider.find_tiered_matches(&query).len(), 1, "{case}");
        assert_eq!(
            supervisor.identity_refs.lock().values().sum::<usize>(),
            2,
            "{case}"
        );
    }
    *sources_a.lock() = source(&endpoint_a);
    supervisor.on_worker_added(&worker_a).await.unwrap();
    assert!(!original.load(Ordering::Acquire));
    assert!(!Arc::ptr_eq(
        &original,
        &supervisor.workers.lock()[&worker_a].active
    ));
    assert!(provider.find_tiered_matches(&query).is_empty());
    boundary_subscription(&publisher_a, "kv/boundary", true).await;
    live_frame(&publisher_a, b"kv/boundary", 0i64, &payload);
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(5), signals.recv())
            .await
            .unwrap(),
        Some(IngestionSignal::Advance { last_seq: 0, .. })
    ));
    assert_eq!(provider.find_tiered_matches(&query).len(), 1);
    assert_eq!(supervisor.identity_refs.lock().values().sum::<usize>(), 2);
    supervisor.shutdown();
    task_a.abort();
    task_b.abort();
}

#[tokio::test]
async fn shared_publisher_rank_promotion_revokes_unranked_credit_before_registration() {
    use serde_json::json;

    let context = zmq::Context::new();
    let publisher = context.socket(zmq::XPUB).unwrap();
    publisher.set_xpub_verbose(true).unwrap();
    publisher.bind("tcp://127.0.0.1:*").unwrap();
    let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
    let source = |rank: u32| {
        json!({rank.to_string(): {
            "endpoint": endpoint, "publisher": "zmq", "topic": "kv/boundary",
            "encoding": "msgpack", "schema_version": 1, "enable_kv_cache_events": true
        }})
    };
    let info = |instance| {
        json!({
            "model_id": "boundary-model", "instance_id": instance, "kv_block_size": 8
        })
    };
    let (worker_a, task_a) =
        source_boundary_worker_with_metadata(info("promotion-a"), source(0)).await;
    let (worker_b, task_b) =
        source_boundary_worker_with_metadata(info("promotion-b"), source(1)).await;
    let (supervisor, mut signals) =
        KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
    let send = |sequence: i64, token: u32, rank: Option<u32>| {
        let payload = store_payload(sequence + 1, &[token; 8], 8, "GPU", rank);
        live_frame(&publisher, b"kv/boundary", sequence, &payload);
    };
    let query = |token| block_query(&[token; 8], 8, &[StorageTier::Device]);
    supervisor.on_worker_added(&worker_a).await.unwrap();
    boundary_subscription(&publisher, "kv/boundary", true).await;
    send(0, 1, None);
    assert!(
        matches!(tokio::time::timeout(Duration::from_secs(5), signals.recv()).await.unwrap(),
            Some(IngestionSignal::Advance { source, last_seq: 0 }) if source.as_ref() == "promotion-a")
    );
    let provider = supervisor.match_provider_for_worker(&worker_a).unwrap();
    let positive = provider.find_tiered_matches(&query(1));
    assert_eq!(positive.len(), 1);
    assert_eq!(positive[0].target.instance_id.as_ref(), "promotion-a");
    supervisor.on_worker_added(&worker_b).await.unwrap();
    assert!(provider.find_tiered_matches(&query(1)).is_empty());
    boundary_subscription(&publisher, "kv/boundary", true).await;

    send(1, 2, None);
    let mut rejected_sources = HashSet::new();
    for _ in 0..2 {
        let signal = tokio::time::timeout(Duration::from_secs(5), signals.recv())
            .await
            .unwrap();
        match signal {
            Some(IngestionSignal::Gap {
                source, to_seq: 1, ..
            }) => {
                rejected_sources.insert(source);
            }
            other => panic!("missing shared rank must reject: {other:?}"),
        }
    }
    assert_eq!(rejected_sources.len(), 2);
    assert!(provider.find_tiered_matches(&query(2)).is_empty());
    for (sequence, token, rank) in [(1, 3, 0), (2, 4, 1), (3, 5, 0)] {
        send(sequence, token, Some(rank));
        let mut advanced_sources = HashSet::new();
        tokio::time::timeout(Duration::from_secs(5), async {
            while advanced_sources.len() != 2 {
                if let Some(IngestionSignal::Advance { source, last_seq }) = signals.recv().await {
                    assert_eq!(last_seq, sequence);
                    advanced_sources.insert(source);
                }
            }
        })
        .await
        .unwrap();
        let hits = provider.find_tiered_matches(&query(token));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].target.dp_rank, rank);
        assert_eq!(
            hits[0].target.instance_id.as_ref(),
            if rank == 0 {
                "promotion-a"
            } else {
                "promotion-b"
            }
        );
        assert_eq!(hits[0].segments, [(0, 1)]);
        // Foreign traffic must preserve already-qualified owned observations.
        assert_eq!(provider.find_tiered_matches(&query(3)).len(), 1);
        if sequence >= 2 {
            assert_eq!(provider.find_tiered_matches(&query(4)).len(), 1);
        }
    }
    assert!(provider.find_tiered_matches(&query(1)).is_empty());
    assert!(provider.find_tiered_matches(&query(2)).is_empty());
    supervisor.shutdown();
    task_a.abort();
    task_b.abort();
}

#[tokio::test]
async fn live_source_and_target_conflicts_reject_before_subscription_without_replay() {
    use serde_json::json;

    for static_a in [false, true] {
        for same_publisher in [true, false] {
            let context = zmq::Context::new();
            let publisher_a = context.socket(zmq::XPUB).unwrap();
            let publisher_b = context.socket(zmq::XPUB).unwrap();
            for publisher in [&publisher_a, &publisher_b] {
                publisher.set_xpub_verbose(true).unwrap();
                publisher.bind("tcp://127.0.0.1:*").unwrap();
            }
            let endpoint_a = publisher_a.get_last_endpoint().unwrap().unwrap();
            let endpoint_b = publisher_b.get_last_endpoint().unwrap().unwrap();
            let source = |endpoint: &str| {
                json!({"0": {
                    "endpoint": endpoint, "topic": "kv/boundary", "publisher": "zmq",
                    "encoding": "msgpack", "schema_version": 1, "enable_kv_cache_events": true
                }})
            };
            let info = |instance: &str| {
                json!({
                    "model_id": "boundary-model", "instance_id": instance, "kv_block_size": 8
                })
            };
            let (worker_a, task_a) =
                source_boundary_worker_with_metadata(info("boundary-a"), source(&endpoint_a)).await;
            let instance_a = if static_a {
                worker_a.as_str()
            } else {
                "boundary-a"
            };
            let (worker_b, task_b) = source_boundary_worker_with_metadata(
                info(if same_publisher {
                    "boundary-b"
                } else {
                    instance_a
                }),
                source(if same_publisher {
                    &endpoint_a
                } else {
                    &endpoint_b
                }),
            )
            .await;
            let (mut supervisor, mut signals) =
                KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
            supervisor.full_attention_http = true;
            if static_a {
                supervisor.static_sources = serde_json::from_value(json!({&worker_a: {
                    "endpoints": [endpoint_a], "topic": "kv/boundary",
                    "block_size": 8, "model": "boundary-model"
                }}))
                .unwrap();
            }
            supervisor.on_worker_added(&worker_a).await.unwrap();
            boundary_subscription(&publisher_a, "kv/boundary", true).await;
            let payload = store_payload(1, &[1; 8], 8, "GPU", Some(0));
            live_frame(&publisher_a, b"kv/boundary", 0i64, &payload);
            assert!(matches!(
                tokio::time::timeout(Duration::from_secs(5), signals.recv())
                    .await
                    .unwrap(),
                Some(IngestionSignal::Advance { last_seq: 0, .. })
            ));
            let query = block_query(&[1; 8], 8, &[StorageTier::Device]);
            let provider = supervisor.match_provider_for_worker(&worker_a).unwrap();
            assert_eq!(provider.find_tiered_matches(&query).len(), 1);
            let admission = supervisor.on_worker_added(&worker_b).await;
            let error =
                admission.expect_err("conflicting worker must be rejected before publication");
            assert!(
                error.contains(if same_publisher {
                    "live KV streams must be independent"
                } else {
                    "KV index targets must be independent"
                }),
                "{error}"
            );
            assert!(supervisor.match_provider_for_worker(&worker_b).is_none());
            assert_eq!(supervisor.workers.lock().len(), 1);
            assert!(matches!(
                publisher_a.recv_bytes(zmq::DONTWAIT),
                Err(zmq::Error::EAGAIN)
            ));
            assert!(matches!(
                publisher_b.recv_bytes(zmq::DONTWAIT),
                Err(zmq::Error::EAGAIN)
            ));
            let retained = provider.find_tiered_matches(&query);
            assert_eq!(retained.len(), 1);
            assert_eq!(retained[0].target.instance_id.as_ref(), instance_a);
            assert_eq!(retained[0].segments, [(0, 1)]);
            supervisor.shutdown();
            task_a.abort();
            task_b.abort();
        }
    }
}

#[tokio::test]
async fn discovered_sources_reject_shared_replay_before_wrong_owner_is_published() {
    let context = zmq::Context::new();
    let a = context.socket(zmq::XPUB).unwrap();
    let b = context.socket(zmq::XPUB).unwrap();
    let replay = context.socket(zmq::ROUTER).unwrap();
    for socket in [&a, &b, &replay] {
        socket.bind("tcp://127.0.0.1:*").unwrap();
    }
    let replay_endpoint = replay.get_last_endpoint().unwrap().unwrap();
    let (worker_a, http_a) = source_boundary_worker(
        &a.get_last_endpoint().unwrap().unwrap(),
        &replay_endpoint,
        "boundary-a",
    )
    .await;
    let (worker_b, http_b) = source_boundary_worker(
        &b.get_last_endpoint().unwrap().unwrap(),
        &replay_endpoint,
        "boundary-b",
    )
    .await;
    let (supervisor, mut signals) =
        KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
    supervisor.on_worker_added(&worker_a).await.unwrap();
    boundary_subscription(&a, "kv/boundary", true).await;
    finish_empty_startup_replay(&replay).await;
    live_frame(
        &a,
        b"kv/boundary",
        0i64,
        &store_payload(1, &[1; 8], 8, "GPU", Some(0)),
    );
    assert!(
        matches!(tokio::time::timeout(Duration::from_secs(5), signals.recv()).await.unwrap(),
            Some(IngestionSignal::Advance { source, last_seq: 0 }) if source.as_ref() == "boundary-a")
    );
    let query = |token| block_query(&[token; 8], 8, &[StorageTier::Device]);
    assert_eq!(
        supervisor
            .match_provider_for_worker(&worker_a)
            .unwrap()
            .find_tiered_matches(&query(1))
            .len(),
        1
    );
    let error = supervisor.on_worker_added(&worker_b).await.unwrap_err();
    assert!(
        error.contains("replay endpoints must be independent"),
        "{error}"
    );
    assert!(supervisor.match_provider_for_worker(&worker_b).is_none());
    assert!(matches!(
        b.recv_bytes(zmq::DONTWAIT),
        Err(zmq::Error::EAGAIN)
    ));
    let retained = supervisor
        .match_provider_for_worker(&worker_a)
        .unwrap()
        .find_tiered_matches(&query(1));
    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0].target.instance_id.as_ref(), "boundary-a");
    assert_eq!(retained[0].segments, [(0, 1)]);
    supervisor.shutdown();
    http_a.abort();
    http_b.abort();
}

#[tokio::test]
async fn discovered_source_bindings_are_atomic_and_released_with_the_worker() {
    use axum::{routing::get, Json, Router};
    use serde_json::json;

    let context = zmq::Context::new();
    let publisher_a = context.socket(zmq::XPUB).unwrap();
    let publisher_b = context.socket(zmq::XPUB).unwrap();
    let replay = context.socket(zmq::ROUTER).unwrap();
    for socket in [&publisher_a, &publisher_b, &replay] {
        socket.bind("tcp://127.0.0.1:*").unwrap();
    }
    let replay_endpoint = replay.get_last_endpoint().unwrap().unwrap();
    let (entered_tx, mut entered) = mpsc::channel(2);
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let mut metadata_workers = Vec::new();
    for (publisher, instance) in [(&publisher_a, "binding-a"), (&publisher_b, "binding-b")] {
        let info =
            json!({"model_id": "boundary-model", "instance_id": instance, "kv_block_size": 8});
        let sources = json!({"0": {
            "endpoint": publisher.get_last_endpoint().unwrap().unwrap(),
            "replay_endpoint": replay_endpoint, "topic": "kv/boundary",
            "publisher": "zmq", "encoding": "msgpack", "schema_version": 1,
            "enable_kv_cache_events": true
        }});
        let first = Arc::new(AtomicBool::new(true));
        let entered_tx = entered_tx.clone();
        let release = release.clone();
        let app = Router::new()
            .route(
                "/get_server_info",
                get(move || {
                    let (info, first, entered_tx, release) = (
                        info.clone(),
                        first.clone(),
                        entered_tx.clone(),
                        release.clone(),
                    );
                    async move {
                        // Only the initial discovery waits; later re-adds use this same server.
                        if first.swap(false, Ordering::AcqRel) {
                            entered_tx.send(instance).await.unwrap();
                            release.acquire().await.unwrap().forget();
                        }
                        Json(info)
                    }
                }),
            )
            .route(
                "/kv_event_sources",
                get(move || {
                    let sources = sources.clone();
                    async move { Json(sources) }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        metadata_workers.push((worker, task));
    }
    let (worker_b, http_b) = metadata_workers.pop().unwrap();
    let (worker_a, http_a) = metadata_workers.pop().unwrap();
    let worker_a = format!("{worker_a}/");
    let (supervisor, _signals) =
        KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
    let discoveries = async {
        tokio::join!(
            supervisor.on_worker_added(&worker_a),
            supervisor.on_worker_added(&worker_b)
        )
    };
    let waiting_metadata = async {
        let mut waiting = [entered.recv().await.unwrap(), entered.recv().await.unwrap()];
        waiting.sort_unstable();
        assert_eq!(waiting, ["binding-a", "binding-b"]);
        assert!(supervisor.workers.lock().is_empty());
        assert!(supervisor.indexers.is_empty());
        assert!(supervisor.identity_refs.lock().is_empty());
        release.add_permits(2);
    };
    let ((a, b), ()) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(discoveries, waiting_metadata)
    })
    .await
    .unwrap();
    assert_ne!(a.is_ok(), b.is_ok());
    let (accepted, rejected) = if a.is_ok() {
        (&worker_a, &worker_b)
    } else {
        (&worker_b, &worker_a)
    };
    assert_eq!(supervisor.workers.lock().len(), 1);
    assert_eq!(
        supervisor
            .identity_refs
            .lock()
            .values()
            .copied()
            .sum::<usize>(),
        1
    );
    assert!(supervisor.match_provider_for_worker(accepted).is_some());
    assert!(supervisor.match_provider_for_worker(rejected).is_none());
    supervisor.on_worker_removed(accepted);
    assert!(supervisor.workers.lock().is_empty());
    assert!(supervisor.identity_refs.lock().is_empty());
    supervisor.on_worker_added(rejected).await.unwrap();
    let alias = if rejected.ends_with('/') {
        rejected.trim_end_matches('/').to_owned()
    } else {
        format!("{rejected}/")
    };
    let error = supervisor.on_worker_added(&alias).await.unwrap_err();
    assert!(error.contains("worker origins must be independent"));
    assert!(supervisor.workers.lock().contains_key(rejected));
    assert!(!supervisor.workers.lock().contains_key(&alias));
    assert_eq!(
        supervisor
            .identity_refs
            .lock()
            .values()
            .copied()
            .sum::<usize>(),
        1
    );
    supervisor.on_worker_removed(rejected);
    let (first, second) = tokio::join!(
        supervisor.on_worker_added(rejected),
        supervisor.on_worker_added(rejected)
    );
    assert_ne!(first.is_ok(), second.is_ok());
    assert_eq!(supervisor.workers.lock().len(), 1);
    assert_eq!(
        supervisor
            .identity_refs
            .lock()
            .values()
            .copied()
            .sum::<usize>(),
        1
    );
    supervisor.shutdown();
    assert!(supervisor.workers.lock().is_empty());
    assert!(supervisor.identity_refs.lock().is_empty());
    http_a.abort();
    http_b.abort();
}

#[tokio::test]
async fn dropping_supervisor_retires_inflight_replay_without_clearing_other_sources() {
    for explicit_shutdown in [false, true] {
        let context = Arc::new(zmq::Context::new());
        let publisher = context.socket(zmq::XPUB).unwrap();
        let replay = context.socket(zmq::ROUTER).unwrap();
        for socket in [&publisher, &replay] {
            socket.bind("tcp://127.0.0.1:*").unwrap();
        }
        let (worker, server) = source_boundary_worker(
            &publisher.get_last_endpoint().unwrap().unwrap(),
            &replay.get_last_endpoint().unwrap().unwrap(),
            "retiring-source",
        )
        .await;
        let (supervisor, mut signals) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
        supervisor.on_worker_added(&worker).await.unwrap();
        boundary_subscription(&publisher, "kv/boundary", true).await;
        finish_empty_startup_replay(&replay).await;
        let (indexer, active) = {
            let workers = supervisor.workers.lock();
            let entry = &workers[&worker];
            (entry.indexer.clone(), entry.active.clone())
        };
        let provider = supervisor.match_provider_for_worker(&worker).unwrap();
        let (neighbor_tx, mut neighbor_signals) = mpsc::channel(8);
        let neighbor = crate::kv_index::subscriber::spawn(
            EventSource {
                source: Arc::from("independent-source"),
                dp_rank: 4,
                pub_endpoint: publisher.get_last_endpoint().unwrap().unwrap(),
                replay_endpoint: None,
                topic: "kv/independent".into(),
                hwm: None,
            },
            indexer,
            Some(neighbor_tx),
            context.clone(),
        );
        boundary_subscription(&publisher, "kv/independent", false).await;
        let payload = |hash, token, rank| store_payload(hash, &[token; 8], 8, "GPU", Some(rank));
        let query = |token| block_query(&[token; 8], 8, &[StorageTier::Device]);
        for (topic, body) in [
            ("kv/boundary", payload(1, 1, 0)),
            ("kv/independent", payload(2, 2, 4)),
        ] {
            live_frame(&publisher, topic.as_bytes(), 0i64, &body);
        }
        for receiver in [&mut signals, &mut neighbor_signals] {
            assert!(matches!(
                tokio::time::timeout(Duration::from_secs(5), receiver.recv())
                    .await
                    .unwrap(),
                Some(IngestionSignal::Advance { last_seq: 0, .. })
            ));
        }
        assert_eq!(provider.find_tiered_matches(&query(1)).len(), 1);
        assert_eq!(provider.find_tiered_matches(&query(2)).len(), 1);
        live_frame(&publisher, b"kv/boundary", 2i64, &payload(3, 3, 0));
        let request = recv_frames(&replay).await;
        assert_eq!(request.last().unwrap(), &1i64.to_be_bytes());
        if explicit_shutdown {
            supervisor.shutdown();
            supervisor.shutdown();
            assert!(supervisor.workers.lock().is_empty());
            assert!(supervisor.identity_refs.lock().is_empty());
        }
        drop(supervisor);
        assert!(!active.load(Ordering::Acquire));
        for (seq, body) in [
            (1i64, payload(4, 4, 0)),
            (2, payload(3, 3, 0)),
            (-1, vec![]),
        ] {
            replay_reply(&replay, &request, b"kv/boundary", seq, &body);
        }
        assert_eq!(recv_frames(&publisher).await, [b"\x00kv/boundary".to_vec()]);
        for token in [1, 3, 4] {
            assert!(provider.find_tiered_matches(&query(token)).is_empty());
        }
        let retained = provider.find_tiered_matches(&query(2));
        assert_eq!(retained.len(), 1);
        assert_eq!(
            retained[0].target.instance_id.as_ref(),
            "independent-source"
        );
        assert_eq!(retained[0].target.dp_rank, 4);
        assert_eq!(retained[0].segments, [(0, 1)]);
        neighbor.shutdown();
        server.abort();
    }
}

#[tokio::test]
async fn generic_static_sources_preserve_worker_base_paths_in_full_attention_router() {
    let context = zmq::Context::new();
    let publisher = context.socket(zmq::XPUB).unwrap();
    let replay = context.socket(zmq::ROUTER).unwrap();
    for socket in [&publisher, &replay] {
        socket.bind("tcp://127.0.0.1:*").unwrap();
    }
    let (worker, http_task) = source_boundary_worker(
        &publisher.get_last_endpoint().unwrap().unwrap(),
        &replay.get_last_endpoint().unwrap().unwrap(),
        "prefixed-source",
    )
    .await;
    let prefixed = format!("{worker}/proxy");
    let (generic, _) =
        KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
    let generic = generic.with_static_sources(HashMap::from([(
        prefixed.clone(),
        StaticKvEventSources {
            endpoints: vec![publisher.get_last_endpoint().unwrap().unwrap()],
            replay_endpoints: Some(vec![replay.get_last_endpoint().unwrap().unwrap()]),
            block_size: 8,
            model: "prefixed-model".into(),
            topic: "kv/boundary".into(),
            hybrid: None,
        },
    )]));
    let result = generic.on_worker_added(&prefixed).await;
    let registered = generic.match_provider_for_worker(&prefixed).is_some();
    let original_absent = generic.match_provider_for_worker(&worker).is_none();
    assert!(generic.workers.lock()[&prefixed].worker_origin.is_none());
    generic.shutdown();
    let (full_attention, _) =
        KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
    let full_attention_result = full_attention.on_worker_added(&prefixed).await;
    let unqualified = full_attention
        .match_provider_for_worker(&prefixed)
        .is_none();
    full_attention.shutdown();
    http_task.abort();
    let info = result.unwrap();
    assert_eq!(info.instance_id, prefixed);
    assert_eq!(info.ranks, [0]);
    assert!(registered);
    assert!(original_absent);
    assert!(full_attention_result
        .unwrap_err()
        .contains("plain HTTP(S) origins"));
    assert!(unqualified);
}

#[tokio::test]
async fn full_attention_source_block_size_mismatch_revokes_only_affected_source() {
    for (expected, actual) in [(16, 8), (8, 16)] {
        check_full_attention_source_block_size(expected, actual).await;
    }
}

async fn check_full_attention_source_block_size(expected: u32, actual: u32) {
    use serde_json::json;

    let context = zmq::Context::new();
    let publisher = context.socket(zmq::XPUB).unwrap();
    publisher.bind("tcp://127.0.0.1:*").unwrap();
    let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
    let (supervisor, mut signals) =
        KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
    let rank = 3;
    let sources = json!({(rank.to_string()): {
        "endpoint": endpoint, "topic": "kv/dense", "publisher": "zmq",
        "encoding": "msgpack", "schema_version": 1, "enable_kv_cache_events": true
    }});
    let (worker, http_task) = source_boundary_worker_with_metadata(
        json!({"model_id": "dense-model", "instance_id": "dense-boot", "kv_block_size": expected}),
        sources,
    )
    .await;
    let info = supervisor.on_worker_added(&worker).await.unwrap();
    assert_eq!(info.cache_key.as_ref().unwrap().block_size, expected);
    assert_eq!(info.ranks, [rank]);
    boundary_subscription(&publisher, "kv/dense", false).await;
    let indexer = supervisor
        .workers
        .lock()
        .get(&worker)
        .unwrap()
        .indexer
        .clone();
    let old_tokens: Vec<u32> = (0..expected).collect();
    let new_tokens: Vec<u32> = (100..100 + actual).collect();
    let query = |tokens: &[u32]| {
        block_query(
            tokens,
            expected,
            &[StorageTier::Device, StorageTier::HostPinned],
        )
    };
    let mut neighbors = Vec::new();
    for (source, neighbor_rank, topic) in [
        ("other-source", rank, "kv/other-source"),
        (info.instance_id.as_str(), rank + 1, "kv/other-rank"),
    ] {
        let (tx, mut rx) = mpsc::channel(8);
        neighbors.push(crate::kv_index::subscriber::spawn(
            EventSource {
                source: Arc::from(source),
                dp_rank: neighbor_rank,
                pub_endpoint: publisher.get_last_endpoint().unwrap().unwrap(),
                replay_endpoint: None,
                topic: topic.into(),
                hwm: None,
            },
            indexer.clone(),
            Some(tx),
            Arc::new(zmq::Context::new()),
        ));
        boundary_subscription(&publisher, topic, false).await;
        let payload = store_payload(70, &old_tokens, expected, "GPU", Some(neighbor_rank));
        assert!(crate::kv_index::decode_batch(&payload).is_ok());
        live_frame(&publisher, topic.as_bytes(), 0i64, &payload);
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .unwrap(),
            Some(IngestionSignal::Advance { last_seq: 0, .. })
        ));
    }
    let source_hits = |tokens: &[u32]| {
        indexer
            .find_matches(&query(tokens))
            .into_iter()
            .filter(|hit| {
                hit.target.instance_id.as_ref() == info.instance_id && hit.target.dp_rank == rank
            })
            .collect::<Vec<_>>()
    };
    for (seq, block_size, tokens) in [(0i64, expected, &old_tokens), (1, actual, &new_tokens)] {
        let payload = store_payload(seq + 7, tokens, block_size, "CPU", Some(rank));
        live_frame(&publisher, b"kv/dense", seq, &payload);
        if seq == 0 {
            assert!(matches!(
                tokio::time::timeout(Duration::from_secs(5), signals.recv())
                    .await
                    .unwrap(),
                Some(IngestionSignal::Advance { last_seq: 0, .. })
            ));
            let hits = source_hits(&old_tokens);
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].segments, [(0, 1)]);
            assert_eq!(hits[0].target.dp_rank, rank);
        } else {
            tokio::time::timeout(Duration::from_secs(5), async {
                    loop {
                        if source_hits(&old_tokens).is_empty() {
                            assert!(indexer.find_matches(&query(&new_tokens)).is_empty());
                            let remaining = indexer.find_matches(&query(&old_tokens));
                            assert_eq!(remaining.len(), 2, "mismatch invalidated another source/rank");
                            assert!(remaining.iter().all(|hit| hit.segments == [(0, 1)]));
                            break;
                        }
                        if let Ok(signal) = signals.try_recv() {
                            assert!(!matches!(signal, IngestionSignal::Advance { last_seq: 1, .. }),
                                "accepted mismatched Dense batch: source={expected}, event={actual}, old={:?}, mismatched={:?}",
                                indexer.find_matches(&query(&old_tokens)),
                                indexer.find_matches(&query(&new_tokens)));
                        }
                        tokio::task::yield_now().await;
                    }
                }).await.expect("mismatch must revoke the previously indexed source");
        }
    }
    supervisor.shutdown();
    drop(neighbors);
    http_task.abort();
}

#[tokio::test]
async fn http_and_static_sources_ingest_nonempty_topics_through_live_zmq() {
    use serde_json::json;

    for (use_static, block_size) in [(false, 8u32), (true, 32)] {
        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
        let rank = if use_static { 0u32 } else { 3 };
        let (mut supervisor, mut signals) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
        let (worker, http_task) = if use_static {
            let worker = "http://127.0.0.1:1".to_string();
            let value = json!({
                &worker: {
                    "endpoints": [endpoint], "block_size": block_size,
                    "model": "served-alias", "topic": "kv/served-alias"
                }
            });
            supervisor.static_sources = serde_json::from_value(value).unwrap();
            (worker, None)
        } else {
            let sources = json!({(rank.to_string()): {
                "endpoint": endpoint, "topic": "kv/served-alias", "hwm": 128,
                "publisher": "zmq", "encoding": "msgpack", "schema_version": 1,
                "enable_kv_cache_events": true
            }});
            let (worker, task) = source_boundary_worker_with_metadata(
                    json!({
                        "model_id": "served-alias", "instance_id": "observed-boot", "kv_block_size": block_size
                    }),
                    sources,
                )
                .await;
            (worker, Some(task))
        };
        let info = supervisor.on_worker_added(&worker).await.unwrap();
        assert_eq!(info.ranks, [rank]);
        assert_eq!(
            info.cache_key.as_ref().unwrap().model.as_ref(),
            "served-alias"
        );
        boundary_subscription(&publisher, "kv/served-alias", false).await;
        let tokens: Vec<u32> = (0..block_size).collect();
        let payload = store_payload(
            7,
            &tokens,
            block_size,
            if use_static { "GPU" } else { "CPU" },
            Some(rank),
        );
        assert!(crate::kv_index::decode_batch(&payload).is_ok());
        live_frame(&publisher, b"kv/served-alias", 0i64, &payload);
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(5), signals.recv())
                .await
                .unwrap(),
            Some(IngestionSignal::Advance { last_seq: 0, .. })
        ));
        let tier = if use_static {
            StorageTier::Device
        } else {
            StorageTier::HostPinned
        };
        let indexer = supervisor
            .workers
            .lock()
            .get(&worker)
            .unwrap()
            .indexer
            .clone();
        let hits = indexer.find_matches(&block_query(&tokens, block_size, &[tier]));
        assert_eq!(hits.len(), 1, "use_static={use_static}");
        assert_eq!(hits[0].target.instance_id.as_ref(), info.instance_id);
        assert_eq!(hits[0].target.dp_rank, rank);
        assert_eq!(hits[0].tier, tier);
        assert_eq!(hits[0].segments, [(0, 1)]);
        supervisor.shutdown();
        if let Some(task) = http_task {
            task.abort();
        }
    }
}

#[test]
fn static_pd_discovery_builds_one_source_per_rank() {
    let config = crate::config::types::PdKvEventSources {
        endpoints: vec!["tcp://10.0.0.5:5557".into(), "tcp://10.0.0.5:5559".into()],
        replay_endpoints: Some(vec!["tcp://10.0.0.5:5558".into()]),
        block_size: 32,
        model: Some("m".into()),
        hybrid: None,
        compatible_decode_urls: Vec::new(),
    };
    let discovery = static_pd_discovery("http://10.0.0.5:8000", &config);

    let server_info = discovery.server_info.unwrap();
    assert_eq!(server_info.model_id.as_deref(), Some("m"));
    assert_eq!(discovered_block_size(&server_info), Some(32));
    assert_eq!(discovery.model_id, Ok(Some("m".into())));

    let sources = discovery.sources.unwrap();
    assert_eq!(sources.len(), 2);
    for rank in ["0", "1"] {
        assert!(
            events_enabled(&sources[rank]),
            "rank {rank} should be enabled"
        );
    }
    assert_eq!(sources["0"].endpoint, "tcp://10.0.0.5:5557");
    assert_eq!(
        sources["0"].replay_endpoint.as_deref(),
        Some("tcp://10.0.0.5:5558")
    );
    // Replay list shorter than endpoints: rank 1 has no replay endpoint.
    assert_eq!(sources["1"].replay_endpoint, None);
    assert_eq!(sources["1"].endpoint, "tcp://10.0.0.5:5559");
}

#[test]
fn static_pd_discovery_falls_back_to_worker_url_as_model() {
    let config = crate::config::types::PdKvEventSources {
        endpoints: vec!["tcp://127.0.0.1:5557".into()],
        replay_endpoints: None,
        block_size: 16,
        model: None,
        hybrid: None,
        compatible_decode_urls: Vec::new(),
    };
    let discovery = static_pd_discovery("http://127.0.0.1:8000", &config);
    assert_eq!(discovery.model_id, Ok(Some("http://127.0.0.1:8000".into())));
    // A blank model also falls back — an empty identity would prevent
    // cache-key creation downstream.
    let config = crate::config::types::PdKvEventSources {
        model: Some("  ".into()),
        ..config
    };
    let discovery = static_pd_discovery("http://127.0.0.1:8000", &config);
    assert_eq!(discovery.model_id, Ok(Some("http://127.0.0.1:8000".into())));
}

/// A worker with static sources must become indexable without any HTTP
/// control plane — the discovery client would fail every request.
#[tokio::test]
async fn static_sources_skip_http_discovery() {
    let publisher_context = zmq::Context::new();
    let publisher = publisher_context.socket(zmq::PUB).unwrap();
    publisher.bind("tcp://127.0.0.1:*").unwrap();
    let endpoint = publisher.get_last_endpoint().unwrap().unwrap();

    let worker_url = "http://127.0.0.1:1"; // nothing listens here
    let static_sources = std::iter::once((
        worker_url.to_string(),
        crate::config::types::PdKvEventSources {
            endpoints: vec![endpoint],
            replay_endpoints: None,
            block_size: 16,
            model: None,
            hybrid: None,
            compatible_decode_urls: Vec::new(),
        },
    ))
    .collect();
    let (supervisor, _signals) =
        KvIndexSupervisor::new_pd(reqwest::Client::new(), HashMode::Sha256, static_sources);

    let info = supervisor.on_worker_added(worker_url).await.unwrap();
    assert_eq!(info.block_size, Some(16));
    assert_eq!(info.ranks, vec![0]);
    let key = info.cache_key.expect("static sources must create a key");
    assert_eq!(key.model.as_ref(), worker_url);
    assert!(supervisor.match_provider_for_worker(worker_url).is_some());
    supervisor.on_worker_removed(worker_url);
}

/// A rank-pinned static source takes its rank from the worker URL, so two
/// DP ranks never share one index even when both declare one endpoint.
#[tokio::test]
async fn rank_pinned_static_source_binds_its_own_rank() {
    let mut layout = layout_only_sources("kv-rank-1", 528, "m").hybrid.unwrap();
    layout.ranks[0].rank = 1;
    let config = crate::config::types::StaticKvEventSources {
        endpoints: vec!["tcp://127.0.0.1:5558".into()],
        replay_endpoints: Some(vec!["tcp://127.0.0.1:5561".into()]),
        topic: String::new(),
        block_size: 528,
        model: "m".into(),
        hybrid: Some(layout),
    };
    let discovery = static_worker_discovery("http://127.0.0.1:8000@1", &config);
    let sources = discovery.sources.unwrap();
    let entry = sources.get("1").expect("rank comes from the URL suffix");
    assert_eq!(entry.endpoint, "tcp://127.0.0.1:5558");
    assert_eq!(
        entry.replay_endpoint.as_deref(),
        Some("tcp://127.0.0.1:5561")
    );
    assert_eq!(entry.topic, "kv-rank-1");
    let hybrid = discovery.hybrid.expect("layout is attached");
    assert_eq!(hybrid.ranks.len(), 1);
    assert_eq!(hybrid.ranks[0].rank, 1);
    assert!(dp_rank_suffix("http://127.0.0.1:8000@1").is_some());
    assert!(dp_rank_suffix("http://127.0.0.1:8000").is_none());
}

/// A layout without declared endpoints is legal on engines that report
/// their own publishers (gRPC `GetKvEventSources`, or the HTTP route vLLM
/// serves since 0.31.0). The declaration still supplies identity, groups
/// and the hash block size.
#[tokio::test]
async fn layout_without_endpoints_discovers_publishers() {
    let (endpoint, replay_endpoint, _publisher, _replay) = layout_endpoints();
    let (worker, task) = layout_worker(endpoint, replay_endpoint, "kv-hybrid").await;
    let config = layout_only_sources("kv-hybrid", 8, "layout-only");
    let (supervisor, _signals) = KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
    let supervisor = supervisor.with_static_sources(HashMap::from([(worker.clone(), config)]));

    let info = supervisor.on_worker_added(&worker).await.unwrap();
    assert_eq!(info.block_size, Some(8));
    assert_eq!(info.ranks, vec![0]);
    assert_eq!(info.cache_key.unwrap().model.as_ref(), "layout-only");
    assert!(supervisor.match_provider_for_worker(&worker).is_some());
    task.abort();
}

/// A publisher advertising another topic belongs to a different boot: the
/// declared layout must not subscribe to it.
#[tokio::test]
async fn layout_rejects_a_publisher_with_another_topic() {
    let (endpoint, replay_endpoint, _publisher, _replay) = layout_endpoints();
    let (worker, task) = layout_worker(endpoint, replay_endpoint, "kv-rebooted").await;

    let config = layout_only_sources("kv-hybrid", 8, "layout-only");
    let (supervisor, _signals) = KvIndexSupervisor::new(reqwest::Client::new(), HashMode::Sha256);
    let supervisor = supervisor.with_static_sources(HashMap::from([(worker.clone(), config)]));

    let info = supervisor.on_worker_added(&worker).await;
    assert!(info.map(|info| info.ranks.is_empty()).unwrap_or(true));
    task.abort();
}

/// Engine stub that reports one publisher for rank 0 under `topic`.
async fn layout_worker(
    endpoint: String,
    replay_endpoint: String,
    topic: &'static str,
) -> (String, tokio::task::JoinHandle<()>) {
    use axum::{routing::get, Json, Router};

    let app = Router::new().route(
        "/kv_event_sources",
        get(move || {
            let endpoint = endpoint.clone();
            let replay_endpoint = replay_endpoint.clone();
            async move {
                Json(serde_json::json!({
                    "0": {"endpoint": endpoint, "replay_endpoint": replay_endpoint,
                          "topic": topic, "publisher": "zmq", "encoding": "msgpack",
                          "schema_version": 1, "enable_kv_cache_events": true}
                }))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (worker, task)
}

/// One PUB and one ROUTER endpoint, like a real rank's ZMQ pair.
fn layout_endpoints() -> (String, String, zmq::Socket, zmq::Socket) {
    let context = zmq::Context::new();
    let publisher = context.socket(zmq::PUB).unwrap();
    publisher.bind("tcp://127.0.0.1:*").unwrap();
    let replay = context.socket(zmq::ROUTER).unwrap();
    replay.bind("tcp://127.0.0.1:*").unwrap();
    (
        publisher.get_last_endpoint().unwrap().unwrap(),
        replay.get_last_endpoint().unwrap().unwrap(),
        publisher,
        replay,
    )
}

fn layout_only_sources(
    topic: &str,
    block_size: u32,
    model: &str,
) -> crate::config::types::StaticKvEventSources {
    let layout = capability(
        model,
        topic,
        block_size,
        serde_json::json!([
            {"group_idx": 0, "kind": "mamba", "spec_type": "MambaSpec",
             "block_size": block_size, "prefix_cacheable": true,
             "prefix_replay_tokens": 0, "mamba_cache_mode": "align"},
            {"group_idx": 1, "kind": "full_attention",
             "spec_type": "FullAttentionSpec", "block_size": block_size,
             "prefix_cacheable": true, "prefix_replay_tokens": 0}
        ]),
    );
    crate::config::types::StaticKvEventSources {
        endpoints: Vec::new(),
        replay_endpoints: None,
        topic: topic.to_owned(),
        block_size,
        model: model.to_owned(),
        hybrid: Some(layout),
    }
}

#[tokio::test]
async fn pd_static_full_attention_rejects_entire_bad_batch_with_source_block_size() {
    use serde_json::json;

    let context = zmq::Context::new();
    let publisher = context.socket(zmq::XPUB).unwrap();
    publisher.bind("tcp://127.0.0.1:*").unwrap();
    let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
    let worker = "http://127.0.0.1:1";
    let (supervisor, mut signals) = KvIndexSupervisor::new_pd(
        reqwest::Client::new(),
        HashMode::Sha256,
        HashMap::from([(
            worker.to_owned(),
            crate::config::types::PdKvEventSources {
                endpoints: vec![endpoint],
                replay_endpoints: None,
                block_size: 8,
                model: Some("pd-model".into()),
                hybrid: None,
                compatible_decode_urls: Vec::new(),
            },
        )]),
    );
    let info = supervisor.on_worker_added(worker).await.unwrap();
    assert_eq!(info.cache_key.unwrap().block_size, 8);
    boundary_subscription(&publisher, "", false).await;
    let stored = |hash, start, size| {
        serde_json::to_value(stored_block(
            hash,
            &(start..start + size).collect::<Vec<_>>(),
            size,
            "GPU",
        ))
        .unwrap()
    };
    let batch = rmp_serde::to_vec(&(
        1.0,
        [
            json!({"type": "BlockRemoved", "block_hashes": [9], "medium": "GPU"}),
            json!({"type": "AllBlocksCleared"}),
            stored(1, 20, 4),
            stored(2, 100, 8),
        ],
        0u32,
    ))
    .unwrap();
    live_frame(&publisher, b"", 0i64, &batch);
    // A subsequent valid frame is a deterministic processing barrier;
    // a rejected earlier batch must not become visible after resume.
    let marker = rmp_serde::to_vec(&(2.0, [stored(3, 200, 8)], 0u32)).unwrap();
    live_frame(&publisher, b"", 1i64, &marker);
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if matches!(
                signals.recv().await,
                Some(IngestionSignal::Advance { last_seq: 1, .. })
            ) {
                break;
            }
        }
    })
    .await
    .unwrap();
    let index = supervisor.workers.lock()[worker].indexer.clone();
    let query = |start| {
        block_query(
            &(start..start + 8).collect::<Vec<_>>(),
            8,
            &[StorageTier::Device],
        )
    };
    assert!(
        index.find_matches(&query(100)).is_empty(),
        "bad block-size batch leaked its valid Store"
    );
    assert_eq!(index.find_matches(&query(200)).len(), 1);
    supervisor.shutdown();
}

/// A stock vLLM worker exposes no engine config endpoint, only `/metrics`.
const METRICS_ONLY: &str = "\
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size=\"16\",effective_attention_block_size=\"16\",enable_prefix_caching=\"True\",prefix_caching_hash_algo=\"sha256_cbor\",hash_block_size=\"None\",kv_cache_layout=\"None\",engine=\"0\"} 1.0
";

fn metrics_and_models_router() -> axum::Router {
    use axum::{routing::get, Router};

    async fn metrics() -> &'static str {
        METRICS_ONLY
    }
    Router::new().route("/metrics", get(metrics)).route(
        "/v1/models",
        get(|| async { axum::Json(serde_json::json!({"data": [{"id": "mock-model"}]})) }),
    )
}

/// Spawn a worker that serves only what a plain `vllm serve` exposes.
async fn spawn_metrics_only_worker(router: axum::Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (worker, server)
}

#[tokio::test]
async fn metrics_only_worker_supplies_cache_metadata() {
    let (worker, server) = spawn_metrics_only_worker(metrics_and_models_router()).await;

    let info = query_worker_metadata(&reqwest::Client::new(), &worker, None)
        .await
        .unwrap();
    assert_eq!(info.kv_block_size, Some(16));
    assert_eq!(info.effective_attention_block_size, Some(16));
    assert_eq!(info.model_id.as_deref(), Some("mock-model"));
    server.abort();
}

#[tokio::test]
async fn metrics_only_worker_creates_a_cache_identity() {
    use axum::routing::get;

    let context = zmq::Context::new();
    let publisher = context.socket(zmq::XPUB).unwrap();
    publisher.bind("tcp://127.0.0.1:*").unwrap();
    let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
    let sources = serde_json::json!({"0": {
        "endpoint": endpoint,
        "topic": "kv/metrics-only",
        "publisher": "zmq",
        "encoding": "msgpack",
        "schema_version": 1,
        "enable_kv_cache_events": true
    }});
    let router = metrics_and_models_router().route(
        "/kv_event_sources",
        get(move || {
            let sources = sources.clone();
            async move { axum::Json(sources) }
        }),
    );
    let (worker, server) = spawn_metrics_only_worker(router).await;

    let (supervisor, _signals) =
        KvIndexSupervisor::new_http_full_attention(reqwest::Client::new(), HashMode::Sha256);
    let info = supervisor.on_worker_added(&worker).await.unwrap();
    assert_eq!(info.block_size, Some(16));
    assert!(
        info.cache_key.is_some(),
        "metadata should produce a cache identity"
    );
    supervisor.shutdown();
    server.abort();
}

#[tokio::test]
async fn worker_without_metadata_names_every_source() {
    let (worker, server) = spawn_metrics_only_worker(axum::Router::new()).await;

    let error = query_worker_metadata(&reqwest::Client::new(), &worker, None)
        .await
        .unwrap_err();
    assert!(error.contains("get_server_info"), "{error}");
    assert!(error.contains("server_info"), "{error}");
    assert!(error.contains("metrics"), "{error}");
    server.abort();
}

#[test]
fn descriptor_refresh_covers_identity_changes_and_failure_threshold() {
    use super::lifecycle::MAX_PROBE_FAILURES;
    let descriptor = |id: &str| {
        worker_descriptor(&WorkerDiscovery {
            server_info: Ok(ServerInfo {
                instance_id: Some(id.into()),
                ..Default::default()
            }),
            sources: Ok(HashMap::new()),
            model_id: Ok(Some("model".into())),
            hybrid: None,
        })
        .unwrap()
    };
    let initial = descriptor("initial");
    let changed = descriptor("replacement");
    for (installed, signature, expected) in [
        (true, Some(&initial), SourceAction::Keep),
        (true, Some(&changed), SourceAction::Rebuild),
        (false, Some(&initial), SourceAction::Rebuild),
        (false, None, SourceAction::Keep),
    ] {
        let lifecycle = WorkerLifecycle::new(
            Arc::new(AtomicBool::new(true)),
            false,
            Some(descriptor("initial")),
        );
        lifecycle.state.lock().info = installed.then(WorkerKvInfo::default);
        assert_eq!(lifecycle.observe_descriptor(signature), expected);
    }
    let lifecycle = WorkerLifecycle::new(
        Arc::new(AtomicBool::new(true)),
        false,
        Some(descriptor("initial")),
    );
    lifecycle.state.lock().info = Some(WorkerKvInfo::default());
    for failures in 1..MAX_PROBE_FAILURES {
        assert_eq!(lifecycle.observe_descriptor(None), SourceAction::Keep);
        assert_eq!(lifecycle.state.lock().probe_failures, failures);
    }
    assert_eq!(lifecycle.observe_descriptor(None), SourceAction::Stop);
    assert!(lifecycle.state.lock().info.is_none());

    let lifecycle = WorkerLifecycle::new(
        Arc::new(AtomicBool::new(true)),
        true,
        Some(descriptor("initial")),
    );
    lifecycle.state.lock().info = Some(WorkerKvInfo::default());
    lifecycle.state.lock().trusted = false;
    assert_eq!(
        lifecycle.observe_descriptor(Some(&initial)),
        SourceAction::Rebuild
    );
}

#[tokio::test]
async fn descriptor_failures_preserve_then_retire_and_rebuild_the_same_source_record() {
    use super::lifecycle::{MAX_PROBE_FAILURES, REFRESH_INTERVAL};
    use axum::{routing::get, Json, Router};

    let healthy = Arc::new(AtomicBool::new(true));
    let context = zmq::Context::new();
    let publisher = context.socket(zmq::XPUB).unwrap();
    publisher.bind("tcp://127.0.0.1:*").unwrap();
    let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
    let metadata = Arc::new(Mutex::new(serde_json::json!({"0": {
        "publisher":"zmq", "endpoint":endpoint, "enable_kv_cache_events":true
    }})));
    let app = Router::new().route("/server_info", get({
        let healthy = healthy.clone();
        move || {
            let healthy = healthy.clone();
            async move {
                if healthy.load(Ordering::Relaxed) {
                    (axum::http::StatusCode::OK, Json(serde_json::json!({
                        "vllm_config":{"model_config":{"model":"m"},"cache_config":{"block_size":16}}
                    })))
                } else {
                    (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({})))
                }
            }
        }
    })).route("/v1/models", get(|| async { Json(serde_json::json!({"data":[{"id":"m","parent":null}]})) }))
      .route("/kv_event_sources", get({
        let metadata = metadata.clone();
        move || { let metadata = metadata.lock().clone(); async move { Json(metadata) } }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let (supervisor, signals) =
        KvIndexSupervisor::new_pd(reqwest::Client::new(), HashMode::Sha256, HashMap::new());
    drop(signals);
    let bases = HashSet::from([base.clone()]);
    let rewind = || {
        supervisor.lifecycles.lock()[&base].state.lock().checked =
            Instant::now() - REFRESH_INTERVAL - CONTROL_TIMEOUT;
    };
    let initial = supervisor.refresh_workers(&bases).await;
    assert!(initial[&base].info.is_some());
    let provider = initial[&base].provider.as_ref().unwrap();
    boundary_subscription(&publisher, "", false).await;
    live_frame(
        &publisher,
        b"",
        0,
        &store_payload(7, &[1; 16], 16, "GPU", Some(0)),
    );
    let query = block_query(&[1; 16], 16, &[StorageTier::Device]);
    tokio::time::timeout(Duration::from_secs(3), async {
        while provider.find_tiered_matches(&query).is_empty() {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap();
    let valid = metadata.lock()["0"].clone();
    *metadata.lock() = serde_json::json!({"not-a-rank": valid.clone()});
    rewind();
    let rejected = supervisor.refresh_workers(&bases).await;
    assert!(rejected[&base].info.is_none());
    assert!(rejected[&base].provider.is_none());
    assert!(provider.find_tiered_matches(&query).is_empty());
    assert!(!supervisor.workers.lock().contains_key(&base));
    *metadata.lock() = serde_json::json!({"0": valid});
    rewind();
    let restored = supervisor.refresh_workers(&bases).await;
    assert!(restored[&base].info.is_some());
    assert_ne!(restored[&base].revision, initial[&base].revision);

    healthy.store(false, Ordering::Relaxed);
    for failures in 1..MAX_PROBE_FAILURES {
        rewind();
        let registration = supervisor.refresh_workers(&bases).await;
        assert!(registration[&base].info.is_some());
        assert!(supervisor.worker_model_eligibility(std::slice::from_ref(&base), Some("m"))[&base]);
        assert_eq!(
            supervisor.lifecycles.lock()[&base]
                .state
                .lock()
                .probe_failures,
            failures
        );
    }
    rewind();
    let retired = supervisor.refresh_workers(&bases).await;
    assert!(retired[&base].info.is_none());
    assert!(retired[&base].provider.is_none());
    assert!(!supervisor.worker_model_eligibility(std::slice::from_ref(&base), Some("m"))[&base]);
    assert_eq!(
        supervisor.lifecycles.lock()[&base]
            .state
            .lock()
            .probe_failures,
        0
    );
    healthy.store(true, Ordering::Relaxed);
    rewind();
    let recovered = supervisor.refresh_workers(&bases).await;
    assert!(recovered[&base].info.is_some());
    assert_ne!(recovered[&base].revision, retired[&base].revision);
    supervisor.shutdown();
    task.abort();
}
