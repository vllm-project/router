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
    let provider = LayoutWorkerProvider {
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
use super::probe::{grpc_kv_event_source_entries, worker_descriptor};
use super::reconcile::{bound_source_moved, probe_keeps_trust, probe_renews_trust, BoundSource};
use super::*;
use crate::core::worker::worker_get;
use crate::kv_index::source::test_support::{event_payload, full_store};
use crate::kv_index::wire::KVEvent;
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

fn plain_full_layout(model: &str, topic: &str, block_size: u32) -> CacheCapabilities {
    let mut layout = capability(
        model,
        topic,
        block_size,
        serde_json::json!([{
            "group_idx": 0, "kind": "full_attention", "spec_type": "FullAttentionSpec",
            "block_size": block_size, "prefix_cacheable": true, "prefix_replay_tokens": 0
        }]),
    );
    layout.ranks[0].declared_groups = Some(1);
    layout
}

fn automatic_layout(layout: CacheCapabilities) -> serde_json::Value {
    let mut response = serde_json::to_value(layout).unwrap();
    response["adapters"] = serde_json::json!([]);
    for rank in response["ranks"].as_array_mut().unwrap() {
        rank["transfer"] = serde_json::Value::Null;
    }
    response
}

#[tokio::test]
async fn endpoints_without_a_verified_layout_do_not_create_a_cache_provider() {
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
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
        let supervisor = supervisor.with_static_sources(sources);

        let info = supervisor.on_worker_added(worker_url).await.unwrap();
        assert_eq!(info.block_size, Some(32));
        assert!(info.ranks.is_empty());
        assert!(info.cache_key.is_none());
        assert!(supervisor.match_provider_for_worker(worker_url).is_none());
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
fn resolves_bind_endpoints_and_preserves_explicit_addresses() {
    let ipv6 = worker_host("grpc://[::1]:50051@3").unwrap();
    for (endpoint, host, expected) in [
        ("tcp://*:5557", "10.0.0.5", "tcp://10.0.0.5:5557"),
        ("tcp://0.0.0.0:5558", "10.0.0.5", "tcp://10.0.0.5:5558"),
        ("tcp://10.0.0.5:5557", "ignored", "tcp://10.0.0.5:5557"),
        ("tcp://[::]:5557", ipv6.as_str(), "tcp://[::1]:5557"),
        ("tcp://[::ffff:127.0.0.1]:5557", "", "tcp://127.0.0.1:5557"),
        ("ipc:///tmp/x", "", "ipc:///tmp/x"),
        ("inproc://x", "", "inproc://x"),
    ] {
        assert_eq!(resolve_endpoint(endpoint, host).unwrap(), expected);
    }
    assert!(resolve_endpoint("tcp://*:0", &ipv6).is_err());
}

#[test]
fn bound_source_accepts_the_engines_bind_endpoints() {
    // The engine answers /kv_event_sources with the addresses it bound while
    // the installed source keeps the resolved ones; that is not a move.
    let installed = BoundSource {
        endpoint: "tcp://127.0.0.1:26258".to_string(),
        replay_endpoint: Some("tcp://127.0.0.1:27258".to_string()),
        topic: "boot-1".to_string(),
    };
    let actual = KvEventSourceEntry {
        publisher: Some("zmq".to_string()),
        endpoint: "tcp://*:26258".to_string(),
        replay_endpoint: Some("tcp://*:27258".to_string()),
        topic: installed.topic.clone(),
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
fn worker_host_extracts_from_url() {
    assert_eq!(
        worker_host("http://10.0.0.5:8000").as_deref(),
        Some("10.0.0.5")
    );
    assert_eq!(worker_host("10.0.0.5:8000").as_deref(), Some("10.0.0.5"));
    assert_eq!(worker_host("10.0.0.5:8000@3").as_deref(), Some("10.0.0.5"));
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
    let layout = automatic_layout(plain_full_layout("boundary-model", "kv/boundary", 8));
    let app = axum::Router::new().route(
        "/kv_routing_capabilities",
        axum::routing::get(move || {
            let layout = layout.clone();
            async move { axum::Json(layout) }
        }),
    );
    source_boundary_worker_with_routes(app, info, Arc::new(Mutex::new(sources))).await
}

async fn source_boundary_worker_with_routes(
    app: axum::Router,
    info: serde_json::Value,
    sources: Arc<Mutex<serde_json::Value>>,
) -> (String, tokio::task::JoinHandle<()>) {
    use axum::{routing::get, Json};

    let app = app
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
                let sources = sources.lock().clone();
                async move { Json(sources) }
            }),
        );
    let app = app.clone().nest("/proxy", app);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (worker, task)
}

#[tokio::test]
async fn http_full_attention_uses_the_worker_hash_contract_when_verified() {
    use axum::{http::StatusCode, routing::get, Json, Router};
    use serde_json::json;

    let topic = "kv/full-boot-1";
    for (algorithm, declared_groups, declare_adapters, native) in [
        (None, None, false, false),
        (Some("sha256"), Some(1), true, false),
        (Some("sha256_cbor"), None, true, false),
        (Some("sha256_cbor"), Some(1), false, false),
        (Some("sha256_cbor"), Some(1), true, true),
    ] {
        let (endpoint, replay_endpoint, _publisher, _replay) = layout_endpoints();
        let sources = Arc::new(Mutex::new(json!({"0": {
            "endpoint": endpoint, "replay_endpoint": replay_endpoint, "topic": topic,
            "publisher": "zmq", "encoding": "msgpack", "schema_version": 1,
            "enable_kv_cache_events": true
        }})));
        let info = json!({
            "model_id": "served-alias", "model_path": "full-model",
            "instance_id": "boot-1", "kv_block_size": 8,
            "effective_attention_block_size": if algorithm.is_none() { 16 } else { 8 }
        });
        let layout = algorithm.map(|algorithm| {
            let mut layout = plain_full_layout("full-model", topic, 8);
            layout.ranks[0].hash_algorithm = algorithm.to_owned();
            layout.ranks[0].declared_groups = declared_groups;
            let mut layout = automatic_layout(layout);
            if !declare_adapters {
                layout.as_object_mut().unwrap().remove("adapters");
            }
            layout
        });
        let capability_response = Arc::new(Mutex::new(layout));
        let app = Router::new().route(
            "/kv_routing_capabilities",
            get({
                let capability_response = capability_response.clone();
                move || {
                    let layout = capability_response.lock().clone();
                    async move { layout.map(Json).ok_or(StatusCode::NOT_FOUND) }
                }
            }),
        );

        let (worker, server) = source_boundary_worker_with_routes(app, info, sources.clone()).await;
        let (supervisor, _signals) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
        let discovered = supervisor.discover(&worker).await;
        let assumed = algorithm.is_none();
        assert_eq!(discovered.assumed_plain_full, assumed);
        if assumed {
            let layout = discovered.hybrid.unwrap();
            let rank = &layout.ranks[0];
            assert!(rank.is_plain_full_attention());
            assert_eq!(rank.hash_seed.as_deref(), Some("vllm-none-hash"));
            assert_eq!(rank.hash_block_size, Some(16));
        }

        let registered = supervisor.on_worker_added(&worker).await.unwrap();
        assert_eq!(
            registered.cache_key.is_some(),
            native || assumed,
            "{algorithm:?}"
        );
        assert_eq!(
            supervisor.match_provider_for_worker(&worker).is_some(),
            native || assumed,
            "{algorithm:?}"
        );
        if assumed {
            assert_eq!(registered.block_size, Some(16));
        }
        if native {
            assert_eq!(registered.ranks, [0]);
            assert_eq!(registered.cache_key.unwrap().model.as_ref(), "served-alias");
            let provider = supervisor.workers.lock()[&worker].provider.clone();
            let verified_at = provider.lifecycle.state.lock().verified_at;
            tokio::time::timeout(Duration::from_secs(5), async {
                while provider.lifecycle.state.lock().verified_at == verified_at {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .expect("monitor did not renew the unchanged hash contract");
            assert!(provider.is_trusted());

            let next_topic = "kv/full-boot-2";
            {
                let mut response = capability_response.lock();
                let rank = &mut response.as_mut().unwrap()["ranks"][0];
                rank["root_hash"] = json!("01".repeat(32));
                rank["publisher_topic"] = json!(next_topic);
            }
            sources.lock()["0"]["topic"] = json!(next_topic);
            tokio::time::timeout(Duration::from_secs(4), async {
                while provider.is_trusted() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .expect("monitor did not revoke the changed hash contract");
            assert!(!supervisor.workers_have_cache_bindings([(worker.as_str(), 0)].into_iter()));
            assert!(supervisor.worker_registrations()[&worker].info.is_none());
            let refreshed = supervisor.refresh_discovered_workers().await;
            assert!(refreshed[&worker]
                .provider
                .as_ref()
                .unwrap()
                .target_scoped());
            let replacement = supervisor.workers.lock()[&worker].provider.clone();
            assert!(replacement.is_trusted());
            assert!(!provider.lifecycle.active.load(Ordering::Acquire));
        }
        supervisor.shutdown();
        server.abort();
    }
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

fn full_payload(tokens: &[u32], block_size: u32, rank: u32) -> Vec<u8> {
    let store = KVEvent::BlockStored(full_store(tokens, block_size, None));
    event_payload(1.0, &[store], Some(rank))
}

async fn wait_for(mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap();
}

async fn event_sources_worker(
    sources: Arc<Mutex<serde_json::Value>>,
) -> (String, tokio::task::JoinHandle<()>) {
    use axum::{routing::get, Json, Router};

    let app = Router::new().route(
        "/kv_event_sources",
        get(move || {
            let sources = sources.lock().clone();
            async move { Json(sources) }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (worker, task)
}

#[tokio::test]
async fn static_hybrid_sources_require_matching_rank_and_generation_topic() {
    let sources = Arc::new(Mutex::new(serde_json::json!({
        "0": {"endpoint":"tcp://*:5557", "replay_endpoint":"tcp://*:5558", "topic":"boot-1-rank-0", "publisher":"zmq"},
        "1": {"endpoint":"tcp://*:5559", "replay_endpoint":"tcp://*:5560", "topic":"boot-1-rank-1", "publisher":"zmq"}
    })));
    let (worker, task) = event_sources_worker(sources.clone()).await;
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
    sources.lock()["1"]["topic"] = serde_json::json!("boot-2-rank-1");
    assert_eq!(
        static_hybrid_sources_match(&client, &worker, None, &config, None).await,
        StaticHybridMatch::Differs
    );
    let installed = HashMap::from([(
        0,
        BoundSource {
            endpoint: config.endpoints[0].clone(),
            replay_endpoint: Some(config.replay_endpoints.as_ref().unwrap()[0].clone()),
            topic: "boot-1-rank-0".to_owned(),
        },
    )]);
    assert_eq!(
        static_hybrid_sources_match(&client, &worker, None, &config, Some(&installed)).await,
        StaticHybridMatch::Matches
    );
    sources.lock()["0"]["topic"] = serde_json::json!("boot-2-rank-0");
    assert_eq!(
        static_hybrid_sources_match(&client, &worker, None, &config, Some(&installed)).await,
        StaticHybridMatch::Differs
    );
    task.abort();
}

#[tokio::test]
async fn layout_only_sources_retire_when_the_engine_changes_its_topic() {
    let sources = Arc::new(Mutex::new(serde_json::json!({
        "0": {"endpoint":"tcp://*:5557", "replay_endpoint":"tcp://*:5558", "topic":"kv-hybrid", "publisher":"zmq"}
    })));
    let (worker, task) = event_sources_worker(sources.clone()).await;
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
    sources.lock()["0"]["topic"] = serde_json::json!("kv-hybrid-boot-2");
    assert_eq!(
        static_hybrid_sources_match(&client, &worker, None, &config, None).await,
        StaticHybridMatch::Differs
    );
    // An engine that advertises nothing cannot keep the old binding alive.
    *sources.lock() = serde_json::json!({});
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
        assumed_plain_full: false,
    };
    let (supervisor, _signals) = KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
    let worker_url = "http://127.0.0.1:8080";
    for (model, expected) in [("qwen", &[0, 1][..]), ("other", &[0][..])] {
        capability.ranks[1].model = model.into();
        let info = supervisor
            .install_worker(worker_url, discovery(capability.clone()))
            .unwrap();
        assert_eq!(info.block_size, Some(16));
        let workers = supervisor.workers.lock();
        let provider = &workers[worker_url].provider;
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
            HashMap::from([(worker_url.clone(), source)]),
        )
    } else {
        let (supervisor, signals) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
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
    let estimate = policy.estimate_prefill_work(
        &[(worker.as_ref(), std::slice::from_ref(&prompt), true)],
        &mut crate::policies::PromptHashes::new(),
    );
    assert_eq!(estimate[0].new_prefill_tokens, 496);
    let hybrid = supervisor
        .workers
        .lock()
        .get(&worker_url)
        .unwrap()
        .provider
        .clone();
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

fn declared_full_worker(
    model: &str,
    instance: &str,
    rank: u32,
    endpoint: &str,
    replay_endpoint: &str,
    topic: &str,
) -> WorkerDiscovery {
    let mut layout = plain_full_layout(model, topic, 8);
    layout.ranks[0].rank = rank;
    WorkerDiscovery {
        server_info: Ok(ServerInfo {
            instance_id: Some(instance.into()),
            kv_block_size: Some(8),
            ..Default::default()
        }),
        sources: Ok(HashMap::from([(
            rank.to_string(),
            KvEventSourceEntry {
                endpoint: endpoint.into(),
                replay_endpoint: Some(replay_endpoint.into()),
                topic: topic.into(),
                publisher: Some("zmq".into()),
                encoding: Some("msgpack".into()),
                schema_version: Some(1),
                enable_kv_cache_events: Some(true),
                ..Default::default()
            },
        )])),
        model_id: Ok(Some(model.into())),
        hybrid: Some(layout),
        assumed_plain_full: false,
    }
}

#[tokio::test]
async fn source_installation_keeps_live_streams_replays_and_index_targets_independent() {
    let worker_a = "http://worker-a:8000";
    let worker_b = "http://worker-b:8000";
    for (model_b, rank_b, endpoint_b, replay_b, topic_b, error) in [
        ("model", 1, "inproc://p1", "inproc://r2", "other", None),
        ("other", 0, "inproc://p2", "inproc://r2", "kv", None),
        (
            "other",
            0,
            "inproc://p1",
            "inproc://r2",
            "kv",
            Some("live KV streams"),
        ),
        (
            "model",
            1,
            "inproc://p1",
            "inproc://r2",
            "kv",
            Some("live KV streams"),
        ),
        (
            "model",
            0,
            "inproc://p2",
            "inproc://r1",
            "kv",
            Some("KV replay endpoints"),
        ),
        (
            "model",
            0,
            "inproc://p2",
            "inproc://r2",
            "kv",
            Some("KV index targets"),
        ),
    ] {
        let (supervisor, _) = KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
        let info = supervisor
            .install_worker(
                worker_a,
                declared_full_worker("model", "shared-id", 0, "inproc://p1", "inproc://r1", "kv"),
            )
            .unwrap();
        assert_eq!(info.ranks, [0]);
        let other = supervisor.install_worker(
            worker_b,
            declared_full_worker(model_b, "shared-id", rank_b, endpoint_b, replay_b, topic_b),
        );
        if let Some(error) = error {
            assert!(other.unwrap_err().contains(error), "{error}");
            assert!(supervisor.match_provider_for_worker(worker_b).is_none());
            assert_eq!(supervisor.workers.lock().len(), 1);
        } else {
            assert_eq!(other.unwrap().ranks, [rank_b]);
            assert!(supervisor.match_provider_for_worker(worker_b).is_some());
            assert_eq!(supervisor.workers.lock().len(), 2);
        }
        assert!(supervisor.match_provider_for_worker(worker_a).is_some());
        supervisor.shutdown();
        assert!(supervisor.workers.lock().is_empty());
        assert!(supervisor.identity_refs.lock().is_empty());
    }
}

#[tokio::test]
async fn source_group_rejects_duplicate_normalized_ranks_without_mutation() {
    let (supervisor, _) = KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
    let mut discovery = declared_full_worker(
        "model",
        "duplicate-rank",
        0,
        "inproc://pub",
        "inproc://replay",
        "kv",
    );
    let sources = discovery.sources.as_mut().unwrap();
    let duplicate = sources["0"].clone();
    sources.insert("00".into(), duplicate);
    let error = supervisor
        .install_worker("http://worker:8000", discovery)
        .unwrap_err();
    assert!(error.contains("duplicate normalized"));
    assert!(supervisor.workers.lock().is_empty());
    assert!(supervisor.identity_refs.lock().is_empty());
}

#[tokio::test]
async fn failed_readd_preserves_the_active_binding_and_success_fences_it() {
    let (endpoint, replay_endpoint, publisher, replay) = layout_endpoints();
    let (second_endpoint, second_replay_endpoint, _second_publisher, _second_replay) =
        layout_endpoints();
    let (supervisor, _) = KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
    let worker = "http://worker-a:8000";
    let neighbor = "http://worker-b:8000";
    let initial = || declared_full_worker("model", "a", 0, &endpoint, &replay_endpoint, "kv");
    supervisor.install_worker(worker, initial()).unwrap();
    supervisor
        .install_worker(
            neighbor,
            declared_full_worker(
                "model",
                "b",
                0,
                &second_endpoint,
                &second_replay_endpoint,
                "kv",
            ),
        )
        .unwrap();

    let provider = supervisor.match_provider_for_worker(worker).unwrap();
    let active = supervisor.workers.lock()[worker].active.clone();
    boundary_subscription(&publisher, "kv", true).await;
    finish_empty_startup_replay(&replay).await;
    live_frame(&publisher, b"kv", 0, &full_payload(&[1; 8], 8, 0));
    let hit = || !provider.matches(&[1; 9], 8, None, 0).is_empty();
    wait_for(hit).await;

    let conflicting = declared_full_worker(
        "model",
        "a",
        0,
        &second_endpoint,
        &second_replay_endpoint,
        "kv",
    );
    let error = supervisor.install_worker(worker, conflicting).unwrap_err();
    assert!(error.contains("live KV streams") || error.contains("KV replay endpoints"));
    assert!(active.load(Ordering::Acquire));
    assert!(hit());

    supervisor.install_worker(worker, initial()).unwrap();
    assert!(!active.load(Ordering::Acquire));
    assert!(!hit());
    assert!(supervisor.match_provider_for_worker(neighbor).is_some());
    supervisor.shutdown();
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
        let layout = automatic_layout(plain_full_layout("boundary-model", "kv/boundary", 8));
        let app = Router::new()
            .route(
                "/kv_routing_capabilities",
                get(move || {
                    let layout = layout.clone();
                    async move { Json(layout) }
                }),
            )
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
    let (supervisor, _signals) = KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
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
async fn verified_plain_full_attention_rejects_mismatched_event_page_size() {
    let (endpoint, replay_endpoint, publisher, replay) = layout_endpoints();
    let (worker, server) = source_boundary_worker(&endpoint, &replay_endpoint, "plain").await;
    let (supervisor, _signals) = KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
    let info = supervisor.on_worker_added(&worker).await.unwrap();
    assert_eq!(info.ranks, [0]);
    let provider = supervisor.match_provider_for_worker(&worker).unwrap();
    boundary_subscription(&publisher, "kv/boundary", false).await;
    finish_empty_startup_replay(&replay).await;
    live_frame(&publisher, b"kv/boundary", 0, &full_payload(&[1; 8], 8, 0));
    let hits = || provider.matches(&[1; 9], 8, None, 0);
    wait_for(|| !hits().is_empty()).await;
    live_frame(&publisher, b"kv/boundary", 1, &full_payload(&[2; 4], 4, 0));
    wait_for(|| hits().is_empty()).await;
    supervisor.shutdown();
    server.abort();
}

#[tokio::test]
async fn generic_static_layout_preserves_the_worker_base_path() {
    let (endpoint, replay_endpoint, _publisher, _replay) = layout_endpoints();
    let (worker, server) = source_boundary_worker(&endpoint, &replay_endpoint, "prefixed").await;
    let prefixed = format!("{worker}/proxy");
    let (supervisor, _) = KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
    let supervisor = supervisor.with_static_sources(HashMap::from([(
        prefixed.clone(),
        StaticKvEventSources {
            endpoints: vec![endpoint],
            replay_endpoints: Some(vec![replay_endpoint]),
            block_size: 8,
            model: "boundary-model".into(),
            topic: "kv/boundary".into(),
            hybrid: Some(plain_full_layout("boundary-model", "kv/boundary", 8)),
        },
    )]));
    let info = supervisor.on_worker_added(&prefixed).await.unwrap();
    assert_eq!(info.instance_id, prefixed);
    assert_eq!(info.ranks, [0]);
    assert!(supervisor.match_provider_for_worker(&prefixed).is_some());
    assert!(supervisor.match_provider_for_worker(&worker).is_none());
    supervisor.shutdown();
    server.abort();
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

/// A complete static layout can index an engine without a discovery endpoint.
#[tokio::test]
async fn static_sources_skip_http_discovery() {
    let (endpoint, replay, _publisher_socket, _replay_socket) = layout_endpoints();
    let (worker, server) = spawn_metrics_only_worker(axum::Router::new()).await;
    let model = "pd-model";
    let static_sources = std::iter::once((
        worker.clone(),
        crate::config::types::PdKvEventSources {
            endpoints: vec![endpoint],
            replay_endpoints: Some(vec![replay]),
            block_size: 16,
            model: Some(model.into()),
            hybrid: Some(plain_full_layout(model, "pd-static", 16)),
            compatible_decode_urls: vec!["http://decode".into()],
        },
    ))
    .collect();
    let (supervisor, _signals) = KvIndexSupervisor::new_pd(reqwest::Client::new(), static_sources);

    let info = supervisor.on_worker_added(&worker).await.unwrap();
    assert_eq!(info.block_size, Some(16));
    assert_eq!(info.ranks, vec![0]);
    let key = info.cache_key.expect("static sources must create a key");
    assert_eq!(key.model.as_ref(), model);
    assert!(supervisor.match_provider_for_worker(&worker).is_some());
    supervisor.on_worker_removed(&worker);
    server.abort();
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
    assert_eq!(
        split_worker_rank("http://127.0.0.1:8000@1"),
        ("http://127.0.0.1:8000", Some(1)),
    );
}

/// A layout without declared endpoints is legal on engines that report
/// their own publishers (gRPC `GetKvEventSources`, or the HTTP route vLLM
/// serves since 0.31.0). The declaration still supplies identity, groups
/// and the hash block size.
#[tokio::test]
async fn layout_without_endpoints_discovers_publishers_for_the_declared_topic() {
    for (topic, matches) in [("kv-hybrid", true), ("kv-rebooted", false)] {
        let (endpoint, replay_endpoint, _publisher, _replay) = layout_endpoints();
        let sources = Arc::new(Mutex::new(serde_json::json!({
            "0": {"endpoint": endpoint, "replay_endpoint": replay_endpoint,
                  "topic": topic, "publisher": "zmq", "encoding": "msgpack",
                  "schema_version": 1, "enable_kv_cache_events": true}
        })));
        let (worker, task) = event_sources_worker(sources).await;
        let config = layout_only_sources("kv-hybrid", 8, "layout-only");
        let (supervisor, _signals) = KvIndexSupervisor::indexed(reqwest::Client::new());
        let supervisor = supervisor.with_static_sources(HashMap::from([(worker.clone(), config)]));

        let info = supervisor.on_worker_added(&worker).await;
        if matches {
            let info = info.unwrap();
            assert_eq!(info.block_size, Some(8));
            assert_eq!(info.ranks, vec![0]);
            assert_eq!(info.cache_key.unwrap().model.as_ref(), "layout-only");
            assert!(supervisor.match_provider_for_worker(&worker).is_some());
        } else {
            assert!(info.map(|info| info.ranks.is_empty()).unwrap_or(true));
            assert!(supervisor.match_provider_for_worker(&worker).is_none());
        }
        task.abort();
    }
}

/// One PUB and one ROUTER endpoint, like a real rank's ZMQ pair.
fn layout_endpoints() -> (String, String, zmq::Socket, zmq::Socket) {
    let context = zmq::Context::new();
    let publisher = context.socket(zmq::XPUB).unwrap();
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

/// A stock vLLM worker exposes no engine config endpoint, only `/metrics`.
const METRICS_ONLY: &str = "\
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size=\"16\",effective_attention_block_size=\"16\",enable_prefix_caching=\"True\",prefix_caching_hash_algo=\"sha256_cbor\",hash_block_size=\"None\",kv_cache_layout=\"None\",engine=\"0\"} 1.0
";

fn metrics_and_models_router(metrics: &'static str) -> axum::Router {
    use axum::{routing::get, Router};

    Router::new()
        .route("/metrics", get(move || async move { metrics }))
        .route(
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
    let metrics = "vllm:cache_config_info{engine=\"0\"} 1.0\n\
        vllm:cache_config_info{block_size=\"16\",effective_attention_block_size=\"16\",engine=\"1\"} 1.0\n";
    let (worker, server) = spawn_metrics_only_worker(metrics_and_models_router(metrics)).await;

    let info = query_worker_metadata(&reqwest::Client::new(), &worker, None)
        .await
        .unwrap();
    assert_eq!(info.kv_block_size, Some(16));
    assert_eq!(info.effective_attention_block_size, Some(16));
    assert_eq!(info.model_id.as_deref(), Some("mock-model"));
    server.abort();
}

#[tokio::test]
async fn cache_metrics_without_a_replay_source_do_not_bind_default_layout() {
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
    let router = metrics_and_models_router(METRICS_ONLY).route(
        "/kv_event_sources",
        get(move || {
            let sources = sources.clone();
            async move { axum::Json(sources) }
        }),
    );
    let (worker, server) = spawn_metrics_only_worker(router).await;

    let (supervisor, _signals) = KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
    let info = supervisor.on_worker_added(&worker).await.unwrap();
    assert_eq!(info.block_size, Some(16));
    assert!(info.cache_key.is_none());
    assert!(supervisor.match_provider_for_worker(&worker).is_none());
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
            assumed_plain_full: false,
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
    let (endpoint, replay_endpoint, publisher, replay) = layout_endpoints();
    let metadata = Arc::new(Mutex::new(serde_json::json!({"0": {
        "publisher":"zmq", "endpoint":endpoint, "enable_kv_cache_events":true,
        "topic":"kv/boot", "replay_endpoint":replay_endpoint
    }})));
    let layout = automatic_layout(plain_full_layout("m", "kv/boot", 16));
    let app = Router::new()
      .route("/kv_routing_capabilities", get(move || {
        let layout = layout.clone();
        async move { Json(layout) }
      }))
      .route("/server_info", get({
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
    let (supervisor, signals) = KvIndexSupervisor::new_pd(reqwest::Client::new(), HashMap::new());
    drop(signals);
    let bases = HashSet::from([base.clone()]);
    let rewind = || {
        supervisor.lifecycles.lock()[&base].state.lock().checked =
            Instant::now() - REFRESH_INTERVAL - CONTROL_TIMEOUT;
    };
    let initial = supervisor.refresh_workers(&bases).await;
    assert!(initial[&base].info.is_some());
    let provider = initial[&base].provider.as_ref().unwrap();
    boundary_subscription(&publisher, "kv/boot", false).await;
    finish_empty_startup_replay(&replay).await;
    live_frame(&publisher, b"kv/boot", 0, &full_payload(&[1; 16], 16, 0));
    let hits = || provider.matches(&[1; 17], 16, None, 0);
    wait_for(|| !hits().is_empty()).await;
    let valid = metadata.lock()["0"].clone();
    *metadata.lock() = serde_json::json!({"not-a-rank": valid.clone()});
    rewind();
    let rejected = supervisor.refresh_workers(&bases).await;
    assert!(rejected[&base].info.is_none());
    assert!(rejected[&base].provider.is_none());
    assert!(hits().is_empty());
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
