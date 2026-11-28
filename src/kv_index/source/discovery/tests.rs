#![cfg(all(test, feature = "cpu", feature = "ready"))]

impl LayoutWorkerProvider {
    fn is_trusted(&self) -> bool {
        self.lifecycle
            .trusted_at(&self.lifecycle.state.lock(), Instant::now)
    }
}

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
        indexes: HashMap::default(),
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
    assert!(provider.lifecycle.finish_layout_recovery(true));
    assert!(provider.is_trusted());
    provider.lifecycle.state.lock().verified_at = Instant::now() - Duration::from_secs(6);
    assert!(provider.lifecycle.expire_if_needed());
    assert_eq!(
        provider
            .lifecycle
            .observe_layout(&StaticHybridMatch::Unavailable(SourcesProbe::Unsupported)),
        SourceAction::Recover
    );
    let revision = {
        let mut state = provider.lifecycle.state.lock();
        state.info = Some(WorkerKvInfo::default());
        state.revision
    };
    assert!(!provider.lifecycle.finish_layout_recovery(false));
    {
        let state = provider.lifecycle.state.lock();
        assert!(state.info.is_none());
        assert_ne!(state.revision, revision);
    }
    assert_eq!(
        provider
            .lifecycle
            .observe_layout(&StaticHybridMatch::Matches),
        SourceAction::Stop
    );
    assert!(!active.load(Ordering::Acquire));
    assert!(!provider.is_trusted());
}

#[test]
fn changed_layout_rebinds_without_waiting_for_the_descriptor_interval() {
    let active = Arc::new(AtomicBool::new(true));
    let lifecycle = WorkerLifecycle::new(active, true, None);
    assert!(!lifecycle.layout_rebind_due());
    assert_eq!(
        lifecycle.observe_layout(&StaticHybridMatch::Differs),
        SourceAction::Stop
    );
    assert!(lifecycle.layout_rebind_due());
    assert!(lifecycle.refresh_due());

    assert_eq!(lifecycle.observe_descriptor(None), SourceAction::Keep);
    assert!(!lifecycle.layout_rebind_due());
    assert!(!lifecycle.refresh_due());
}

use super::lifecycle::SourceAction;
use super::probe::worker_descriptor;
use super::reconcile::{bound_source_moved, probe_keeps_trust, probe_renews_trust, BoundSource};
use super::*;
use crate::kv_index::source::test_support::{event_payload, full_store};
use crate::kv_index::wire::KVEvent;
use std::sync::atomic::Ordering;
use std::time::Instant;

#[tokio::test]
async fn registration_probe_reuses_aliases_until_the_worker_changes() {
    use axum::{routing::get, Json, Router};
    use serde_json::json;
    use std::sync::atomic::AtomicUsize;

    let instance = Arc::new(AtomicUsize::new(1));
    let requests = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route(
            "/get_server_info",
            get({
                let instance = instance.clone();
                move || {
                    let id = instance.load(Ordering::Relaxed).to_string();
                    async move {
                        Json(json!({"model_id": "model", "instance_id": id, "kv_block_size": 16}))
                    }
                }
            }),
        )
        .route(
            "/v1/models",
            get({
                let requests = requests.clone();
                move || {
                    let count = requests.fetch_add(1, Ordering::Relaxed) + 1;
                    async move { Json(json!({"data": [{"id": format!("alias-{count}")}]})) }
                }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let (supervisor, _) = KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
    let lifecycle = supervisor.lifecycle_for_worker(&worker);
    let probe = || {
        supervisor.probe_registration(
            &worker,
            false,
            None,
            &lifecycle,
            tokio::time::Instant::now() + CONTROL_TIMEOUT,
        )
    };
    let first = probe().await.unwrap();
    {
        let mut state = lifecycle.state.lock();
        state.info = Some(WorkerKvInfo::default());
        state.descriptor = Some(first.descriptor);
        state.base_models = first.worker.base_models.unwrap();
    }
    let repeated = probe().await.unwrap();
    assert_eq!(
        repeated.worker.base_models.unwrap(),
        HashSet::from(["alias-1".into()])
    );
    assert_eq!(requests.load(Ordering::Relaxed), 1);

    instance.store(2, Ordering::Relaxed);
    let replaced = probe().await.unwrap();
    assert_eq!(
        replaced.worker.base_models.unwrap(),
        HashSet::from(["alias-2".into()])
    );
    assert_eq!(requests.load(Ordering::Relaxed), 2);
    server.abort();
}

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
fn events_enabled_gates_on_publisher_and_endpoint() {
    let mut e = KvEventSourceEntry::default();
    assert!(!events_enabled(&e)); // empty endpoint
                                  // vLLM 0.33 omits the optional wire metadata; "zmq" alone is enough.
    let mut ranks: KvEventSourcesResponse = serde_json::from_value(serde_json::json!({
        "0": {"endpoint": "tcp://*:5557", "publisher": "zmq", "hwm": 100000},
        "1": {"endpoint": "tcp://*:5559", "publisher": "zmq", "replay_endpoint": null}
    }))
    .unwrap();
    assert!(ranks["1"].replay_endpoint.is_none());
    e = ranks.remove("0").unwrap();
    assert_eq!(e.hwm, Some(100_000));
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
fn zero_block_size_is_treated_as_missing() {
    for (configured, effective, expected) in [
        (8, 16, Some(16)),
        (0, 16, Some(16)),
        (8, 0, Some(8)),
        (0, 0, None),
    ] {
        let server_info = ServerInfo {
            kv_block_size: Some(configured),
            effective_attention_block_size: Some(effective),
            ..Default::default()
        };
        assert_eq!(discovered_block_size(&server_info), expected);
    }
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

#[test]
fn native_layout_rejection_identifies_the_rank_and_reason() {
    let layout = plain_full_layout("model", "boot-1", 16);
    let info = ServerInfo {
        kv_block_size: Some(32),
        ..Default::default()
    };
    let mut sources = HashMap::from([(
        "0".to_owned(),
        KvEventSourceEntry {
            endpoint: "tcp://127.0.0.1:5557".into(),
            replay_endpoint: Some("tcp://127.0.0.1:5558".into()),
            publisher: Some("zmq".into()),
            topic: "boot-1".into(),
            ..Default::default()
        },
    )]);
    assert_eq!(
        validate_native_publishers(&layout, &sources, &info, "model"),
        Err("DP rank 0: layout hash block size differs from worker metadata".into())
    );
    sources.get_mut("0").unwrap().topic = "boot-2".into();
    assert_eq!(
        validate_native_publishers(&layout, &sources, &info, "model"),
        Err(
            "DP rank 0: layout publisher topic does not identify the observed worker instance"
                .into()
        )
    );
}

#[tokio::test]
async fn http_full_attention_uses_the_worker_hash_contract_when_verified() {
    use axum::{http::StatusCode, routing::get, Json, Router};
    use serde_json::json;

    let topic = "kv/full-boot-1";
    for (algorithm, declared_groups, size, declare_empty_fields, native) in [
        (None, None, 16, false, false),
        (Some("sha256"), Some(1), 16, true, false),
        (Some("sha256_cbor"), None, 16, true, false),
        (Some("sha256_cbor"), Some(1), 8, true, false),
        (Some("sha256_cbor"), Some(1), 16, false, true),
        (Some("sha256_cbor"), Some(1), 16, true, true),
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
            "effective_attention_block_size": 16
        });
        let layout = algorithm.map(|algorithm| {
            let mut layout = plain_full_layout("full-model", topic, size);
            layout.ranks[0].hash_algorithm = algorithm.to_owned();
            layout.ranks[0].declared_groups = declared_groups;
            let mut layout = automatic_layout(layout);
            if !declare_empty_fields {
                layout.as_object_mut().unwrap().remove("adapters");
                layout["ranks"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("transfer");
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
            assert!(rank.has_plain_full_attention_shape());
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
        if native || assumed {
            assert_eq!(registered.block_size, Some(16));
        }
        if native {
            assert_eq!(registered.ranks, [0]);
            assert_eq!(registered.cache_key.unwrap().model.as_ref(), "served-alias");
        }
        if native && declare_empty_fields {
            let provider = supervisor.installed.workers.lock()[&worker]
                .provider
                .clone();
            provider
                .lifecycle
                .state
                .lock()
                .base_models
                .insert("old-alias".into());
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
            assert!(!provider.lifecycle.active.load(Ordering::Acquire));
            assert!(!supervisor.workers_have_cache_bindings([(worker.as_str(), 0)].into_iter()));
            assert!(supervisor.worker_registrations()[&worker].info.is_none());
            assert!(
                !supervisor
                    .worker_model_eligibility(std::slice::from_ref(&worker), Some("old-alias"))
                    [&worker]
            );
            let refreshed = supervisor.refresh_discovered_workers().await;
            assert!(refreshed[&worker]
                .provider
                .as_ref()
                .unwrap()
                .target_scoped());
            let replacement = supervisor.installed.workers.lock()[&worker]
                .provider
                .clone();
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
    let discovery = static_worker_discovery(&worker, &config);
    assert_eq!(
        static_hybrid_sources_match(&client, &worker, None, &discovery, None).await,
        StaticHybridMatch::Matches
    );
    let mut live_only = config.clone();
    live_only.replay_endpoints = None;
    let live_only = static_worker_discovery(&worker, &live_only);
    assert_eq!(
        static_hybrid_sources_match(&client, &worker, None, &live_only, None).await,
        StaticHybridMatch::Differs
    );
    let with_replay = sources.lock().clone();
    for rank in ["0", "1"] {
        sources.lock()[rank]
            .as_object_mut()
            .unwrap()
            .remove("replay_endpoint");
    }
    assert_eq!(
        static_hybrid_sources_match(&client, &worker, None, &live_only, None).await,
        StaticHybridMatch::Matches
    );
    *sources.lock() = with_replay;
    sources.lock()["1"]["topic"] = serde_json::json!("boot-2-rank-1");
    assert_eq!(
        static_hybrid_sources_match(&client, &worker, None, &discovery, None).await,
        StaticHybridMatch::Differs
    );
    let (supervisor, _) = KvIndexSupervisor::new_http_full_attention(client.clone());
    let supervisor =
        supervisor.with_static_sources(HashMap::from([(worker.clone(), config.clone())]));
    let rejected = supervisor.discover(&worker).await;
    let unavailable = rejected.sources.as_ref().unwrap_err();
    assert_eq!(unavailable.probe, SourcesProbe::StaticMismatch);
    assert!(!unavailable.probe.retryable());
    assert!(super::probe::worker_descriptor(&rejected).is_ok());
    let installed = HashMap::from([(
        0,
        BoundSource {
            endpoint: config.endpoints[0].clone(),
            replay_endpoint: Some(config.replay_endpoints.as_ref().unwrap()[0].clone()),
            topic: "boot-1-rank-0".to_owned(),
        },
    )]);
    assert_eq!(
        static_hybrid_sources_match(&client, &worker, None, &discovery, Some(&installed)).await,
        StaticHybridMatch::Matches
    );
    sources.lock()["0"]["topic"] = serde_json::json!("boot-2-rank-0");
    assert_eq!(
        static_hybrid_sources_match(&client, &worker, None, &discovery, Some(&installed)).await,
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
                 "mamba_cache_mode":"align"},
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
        base_models: None,
        hybrid: Some(capability),
        assumed_plain_full: false,
        engine_cache_configs: None,
    };
    let (supervisor, _signals) = KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
    let worker_url = "http://127.0.0.1:8080";
    for (model, expected) in [("qwen", &[0, 1][..]), ("other", &[0][..])] {
        capability.ranks[1].model = model.into();
        let info = supervisor
            .install_worker(worker_url, discovery(capability.clone()))
            .unwrap();
        assert_eq!(info.block_size, Some(16));
        let workers = supervisor.installed.workers.lock();
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
        .installed
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
        hybrid.lifecycle.state.lock().checked = Some(Instant::now() - Duration::from_secs(30));
        let refreshed = supervisor
            .refresh_workers(
                &HashSet::from([worker_url.clone()]),
                &HashSet::new(),
                &HashSet::new(),
            )
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
        base_models: None,
        hybrid: Some(layout),
        assumed_plain_full: false,
        engine_cache_configs: None,
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
            assert_eq!(supervisor.installed.workers.lock().len(), 1);
        } else {
            assert_eq!(other.unwrap().ranks, [rank_b]);
            assert!(supervisor.match_provider_for_worker(worker_b).is_some());
            assert_eq!(supervisor.installed.workers.lock().len(), 2);
        }
        assert!(supervisor.match_provider_for_worker(worker_a).is_some());
        supervisor.shutdown();
        assert!(supervisor.installed.workers.lock().is_empty());
        assert!(supervisor.installed.identity_refs.lock().is_empty());
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
    assert!(supervisor.installed.workers.lock().is_empty());
    assert!(supervisor.installed.identity_refs.lock().is_empty());
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
    let active = supervisor.installed.workers.lock()[worker].active.clone();
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
        assert!(supervisor.installed.workers.lock().is_empty());
        assert!(supervisor.installed.indexers.is_empty());
        assert!(supervisor.installed.identity_refs.lock().is_empty());
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
    assert_eq!(supervisor.installed.workers.lock().len(), 1);
    assert_eq!(
        supervisor
            .installed
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
    assert!(supervisor.installed.workers.lock().is_empty());
    assert!(supervisor.installed.identity_refs.lock().is_empty());
    supervisor.on_worker_added(rejected).await.unwrap();
    let alias = if rejected.ends_with('/') {
        rejected.trim_end_matches('/').to_owned()
    } else {
        format!("{rejected}/")
    };
    let error = supervisor.on_worker_added(&alias).await.unwrap_err();
    assert!(error.contains("worker origins must be independent"));
    assert!(supervisor.installed.workers.lock().contains_key(rejected));
    assert!(!supervisor.installed.workers.lock().contains_key(&alias));
    assert_eq!(
        supervisor
            .installed
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
    assert_eq!(supervisor.installed.workers.lock().len(), 1);
    assert_eq!(
        supervisor
            .installed
            .identity_refs
            .lock()
            .values()
            .copied()
            .sum::<usize>(),
        1
    );
    supervisor.shutdown();
    assert!(supervisor.installed.workers.lock().is_empty());
    assert!(supervisor.installed.identity_refs.lock().is_empty());
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

fn static_pd_discovery(
    worker_url: &str,
    config: &crate::config::PdKvEventSources,
) -> WorkerDiscovery {
    static_worker_discovery(worker_url, &pd_static_source(worker_url, config))
}

#[test]
fn static_pd_discovery_falls_back_to_worker_url_as_model() {
    let config = crate::config::types::PdKvEventSources {
        endpoints: vec!["tcp://127.0.0.1:5557".into()],
        replay_endpoints: None,
        block_size: 16,
        model: None,
        hybrid: None,
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

#[tokio::test]
async fn prefill_discovers_mooncake_compressed_truncation() {
    let (endpoint, replay, _publisher, _replay) = layout_endpoints();
    let layout = capability(
        "deepseek-v4",
        "kv-prefill",
        8,
        serde_json::json!([{
            "group_idx": 0, "kind": "mla_attention", "spec_type": "MLAAttentionSpec",
            "block_size": 512, "prefix_cacheable": true, "prefix_replay_tokens": 0,
            "tokens_per_state": "4", "cache_role": "sparse"
        }]),
    );
    for (connector, worker_connector, hf_config, expected) in [
        (
            KvConnector::Nixl,
            "MooncakeHybridConnector",
            serde_json::json!({}),
            true,
        ),
        (
            KvConnector::MooncakeV2,
            "MooncakeConnectorV2",
            serde_json::json!({"compress_ratios":[1, 4]}),
            true,
        ),
        (
            KvConnector::MooncakeV2,
            "MooncakeConnectorV2",
            serde_json::json!({}),
            false,
        ),
    ] {
        let app = axum::Router::new().route(
            "/server_info",
            axum::routing::get(move || {
                let hf_config = hf_config.clone();
                async move {
                    // Model diagnostics can take longer than the initial 750 ms probe.
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    axum::Json(serde_json::json!({"vllm_config": {
                        "kv_transfer_config": {"kv_connector": worker_connector},
                        "model_config": {"hf_config": hf_config}
                    }}))
                }
            }),
        );
        let (worker, server) = spawn_metrics_only_worker(app).await;
        let sources = HashMap::from([(
            worker.clone(),
            crate::config::types::PdKvEventSources {
                endpoints: vec![endpoint.clone()],
                replay_endpoints: Some(vec![replay.clone()]),
                block_size: 8,
                model: Some("deepseek-v4".into()),
                hybrid: Some(layout.clone()),
            },
        )]);
        let (supervisor, _signals) = KvIndexSupervisor::new_pd(reqwest::Client::new(), sources);

        let discovery = supervisor.discover_for_role(&worker, Some(connector)).await;
        assert!(discovery.sources.is_ok());
        assert_eq!(
            discovery.server_info.unwrap().mooncake_truncates_compressed,
            Some(expected),
            "{worker_connector}"
        );
        server.abort();
    }
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
    let mut ambiguous = config;
    ambiguous.endpoints.push("tcp://127.0.0.1:5559".into());
    assert!(
        static_worker_discovery("http://127.0.0.1:8000@1", &ambiguous)
            .sources
            .is_err()
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
        let (worker, task) = event_sources_worker(sources.clone()).await;
        let config = layout_only_sources("kv-hybrid", 8, "layout-only");
        let (supervisor, _signals) = KvIndexSupervisor::indexed(reqwest::Client::new());
        let supervisor = supervisor.with_static_sources(HashMap::from([(worker.clone(), config)]));

        let info = supervisor.on_worker_added(&worker).await;
        if matches {
            let info = info.unwrap();
            assert_eq!(info.block_size, Some(8));
            assert_eq!(info.ranks, vec![0]);
            let key = info.cache_key.unwrap();
            assert_eq!(key.model.as_ref(), "layout-only");
            assert!(supervisor.match_provider_for_worker(&worker).is_some());
            sources.lock()["0"]["topic"] = serde_json::json!("kv-hybrid-next-boot");
            wait_for(|| !supervisor.installed.workers.lock().contains_key(&worker)).await;
            assert!(!supervisor.installed.identity_refs.lock().contains_key(&key));
            assert!(!supervisor.installed.indexers.contains_key(&key));
        } else {
            assert!(info.map(|info| info.ranks.is_empty()).unwrap_or(true));
            assert!(supervisor.match_provider_for_worker(&worker).is_none());
        }
        task.abort();
    }
}

#[tokio::test]
async fn declared_http_layout_uses_the_reported_hash_block_size() {
    use axum::{routing::get, Router};
    use serde_json::json;

    for (metric, expected) in [("8", Some(8)), ("None", None)] {
        let (endpoint, replay, _publisher, _replayer) = layout_endpoints();
        let sources = Arc::new(Mutex::new(json!({"0": {
            "endpoint": endpoint, "replay_endpoint": replay, "topic": "boot-1",
            "publisher": "zmq"
        }})));
        let metrics = format!(
            "vllm:cache_config_info{{block_size=\"64\",hash_block_size=\"{metric}\",engine=\"0\"}} 1.0"
        );
        let app = Router::new().route("/metrics", get(move || async move { metrics }));
        let (worker, task) =
            source_boundary_worker_with_routes(app, json!({"model_id":"m"}), sources).await;
        let mut config = layout_only_sources("boot-1", 64, "m");
        config.block_size = 16;
        config.hybrid.as_mut().unwrap().ranks[0].hash_block_size = None;
        config.validate(&worker).unwrap();
        let (supervisor, _) = KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
        let supervisor = supervisor.with_static_sources(HashMap::from([(worker.clone(), config)]));
        let info = supervisor.on_worker_added(&worker).await.unwrap();
        assert_eq!(info.cache_key.map(|key| key.block_size), expected);
        task.abort();
    }
}

#[tokio::test]
async fn hybrid_discovery_uses_reported_fields_in_colocated_and_pd_modes() {
    use axum::{http::StatusCode, routing::get, Json, Router};
    use serde_json::json;

    let (endpoint, replay, _publisher_socket, _replay_socket) = layout_endpoints();
    let sources = Arc::new(Mutex::new(json!({"0": {
        "endpoint": endpoint, "replay_endpoint": replay, "topic": "boot-1",
        "publisher": "zmq"
    }})));
    let mut config = layout_only_sources("boot-1", 576, "qwen");
    config.block_size = 16;
    let layout = config.hybrid.as_mut().unwrap();
    layout.ranks[0].declared_groups = Some(2);
    layout.ranks[0].hash_block_size = Some(16);
    layout.ranks[0].partial_hash_hits = true;
    let reported = Arc::new(Mutex::new(Some(automatic_layout(layout.clone()))));
    let app = Router::new()
        .route(
            "/kv_routing_capabilities",
            get({
                let reported = reported.clone();
                move || {
                    let reported = reported.lock().clone();
                    async move { reported.map(Json).ok_or(StatusCode::NOT_FOUND) }
                }
            }),
        )
        .route(
            "/v1/models",
            get(|| async { Json(json!({"data": [{"id": "qwen", "root": "qwen"}]})) }),
        );
    let info =
        json!({"model_id":"qwen", "kv_block_size":576, "effective_attention_block_size":576});
    let (worker, server) = source_boundary_worker_with_routes(app, info, sources).await;
    let bases = HashSet::from([worker.clone()]);

    let (regular, _) = KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
    let registered = regular.on_worker_added(&worker).await.unwrap();
    assert_eq!(registered.ranks, [0]);
    assert_eq!(registered.block_size, Some(16));
    let layout = regular.discover(&worker).await.hybrid.unwrap();
    assert!(layout.ranks[0].partial_hash_hits);
    let report = reported.lock().take();
    assert!(regular.discover(&worker).await.hybrid.is_none());
    *reported.lock() = report;
    regular.shutdown();

    let (pd, _) = KvIndexSupervisor::new_pd(reqwest::Client::new(), HashMap::new());
    assert!(pd
        .refresh_workers(&bases, &HashSet::new(), &HashSet::new())
        .await[&worker]
        .provider
        .is_some());
    // Disabling hybrid layouts retires the previously installed provider.
    assert!(
        pd.refresh_workers(&bases, &bases, &HashSet::new()).await[&worker]
            .provider
            .is_none()
    );
    assert!(pd.match_provider_for_worker(&worker).is_none());
    pd.shutdown();

    let source = crate::config::PdKvEventSources {
        endpoints: Vec::new(),
        replay_endpoints: None,
        block_size: 16,
        model: None,
        hybrid: None,
    };
    let (pd, _) = KvIndexSupervisor::new_pd(reqwest::Client::new(), HashMap::new());
    assert!(pd
        .refresh_workers(&bases, &HashSet::new(), &HashSet::new())
        .await[&worker]
        .provider
        .is_some());
    reported.lock().take();
    assert!(pd.discover(&worker).await.hybrid.is_none());
    pd.shutdown();

    let mut manual = config.hybrid.unwrap();
    manual.ranks[0].model.clear();
    manual.ranks[0].publisher_topic = None;
    let (pd, _) = KvIndexSupervisor::new_pd(
        reqwest::Client::new(),
        HashMap::from([(
            worker.clone(),
            crate::config::PdKvEventSources {
                hybrid: Some(manual),
                ..source
            },
        )]),
    );
    let discovered = pd.discover(&worker).await.hybrid.unwrap();
    assert_eq!(discovered.ranks[0].model, "qwen");
    assert_eq!(
        discovered.ranks[0].publisher_topic.as_deref(),
        Some("boot-1")
    );
    assert!(pd
        .refresh_workers(&bases, &HashSet::new(), &HashSet::new())
        .await[&worker]
        .provider
        .is_some());
    server.abort();
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

#[derive(Default)]
struct MetadataRequests {
    metrics: std::sync::atomic::AtomicUsize,
    models: std::sync::atomic::AtomicUsize,
}

fn metrics_and_models_router(
    metrics: impl Fn() -> String + Clone + Send + Sync + 'static,
) -> (axum::Router, Arc<MetadataRequests>) {
    use axum::{routing::get, Json, Router};

    let requests = Arc::new(MetadataRequests::default());
    let app = Router::new()
        .route(
            "/metrics",
            get({
                let requests = requests.clone();
                move || {
                    requests.metrics.fetch_add(1, Ordering::Relaxed);
                    std::future::ready(metrics())
                }
            }),
        )
        .route(
            "/v1/models",
            get({
                let requests = requests.clone();
                move || {
                    requests.models.fetch_add(1, Ordering::Relaxed);
                    async {
                        Json(serde_json::json!({"data": [
                            {"id": "mock-model"}, {"id": "mock-alias"},
                            {"id": "adapter", "parent": "mock-model"}
                        ]}))
                    }
                }
            }),
        );
    (app, requests)
}

/// Spawn a worker that serves only what a plain `vllm serve` exposes.
async fn spawn_metrics_only_worker(router: axum::Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (worker, server)
}

struct AbortTestServer(tokio::task::JoinHandle<()>);

impl Drop for AbortTestServer {
    fn drop(&mut self) {
        self.0.abort();
    }
}

struct HashAdmissionWorker {
    url: String,
    supervisor: KvIndexSupervisor,
    metrics: Arc<Mutex<String>>,
    sources: Arc<Mutex<serde_json::Value>>,
    requests: Arc<MetadataRequests>,
    _server: AbortTestServer,
    sockets: Vec<(String, String, zmq::Socket, zmq::Socket)>,
}

impl HashAdmissionWorker {
    async fn start(ranks: usize, metrics: &str) -> Self {
        use serde_json::json;

        let sockets: Vec<_> = (0..ranks).map(|_| layout_endpoints()).collect();
        let sources = Arc::new(Mutex::new(serde_json::Value::Object(
            sockets
                .iter()
                .enumerate()
                .map(|(rank, (publisher, replay, _, _))| {
                    (
                        rank.to_string(),
                        json!({
                            "endpoint": publisher, "replay_endpoint": replay, "publisher": "zmq",
                            "topic": format!("kv/admission/{rank}")
                        }),
                    )
                })
                .collect(),
        )));
        let metrics = Arc::new(Mutex::new(metrics.to_owned()));
        let (app, requests) = metrics_and_models_router({
            let metrics = metrics.clone();
            move || metrics.lock().clone()
        });
        let (url, server) = source_boundary_worker_with_routes(
            app,
            json!({
                "model_id": "mock-model", "instance_id": "boot", "kv_block_size": 16,
                "served_model_aliases": ["control-alias"]
            }),
            sources.clone(),
        )
        .await;
        let (supervisor, _signals) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
        Self {
            url,
            supervisor,
            metrics,
            sources,
            requests,
            _server: AbortTestServer(server),
            sockets,
        }
    }
}

#[tokio::test]
async fn registration_checks_hashes_per_rank() {
    let worker = HashAdmissionWorker::start(
        3,
        concat!(
            "vllm:cache_config_info{engine=\"0\",prefix_caching_hash_algo=\"sha256_cbor\"} 1.0\n",
            "vllm:cache_config_info{engine=\"1\",prefix_caching_hash_algo=\"sha256\"} 1.0\n",
            "vllm:cache_config_info{engine=\"2\"} 1.0\n",
        ),
    )
    .await;
    let supervisor = &worker.supervisor;
    let mut ranks = supervisor.on_worker_added(&worker.url).await.unwrap().ranks;
    ranks.sort_unstable();
    assert_eq!(ranks, [0, 2]);
    assert_eq!(worker.requests.metrics.load(Ordering::Relaxed), 1);
    assert_eq!(worker.requests.models.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn pinned_static_admission_and_fencing_use_the_single_local_engine_report() {
    use crate::policies::TieredMatchProvider;

    let worker = HashAdmissionWorker::start(1, METRICS_ONLY).await;
    let url = format!("{}@1", worker.url);
    let topic = "kv/admission/0";
    let mut layout = plain_full_layout("mock-model", topic, 16);
    layout.ranks[0].rank = 1;
    let (endpoint, replay, publisher, replay_socket) = &worker.sockets[0];
    let sources = crate::config::types::StaticKvEventSources {
        model: "mock-model".into(),
        block_size: 16,
        endpoints: vec![endpoint.clone()],
        replay_endpoints: Some(vec![replay.clone()]),
        topic: String::new(),
        hybrid: Some(layout),
    };
    let (supervisor, _) = KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
    let supervisor = supervisor.with_static_sources(HashMap::from([(url.clone(), sources)]));
    let source = worker.sources.lock()["0"].clone();
    *worker.sources.lock() = serde_json::json!({"1": source});

    assert_eq!(supervisor.on_worker_added(&url).await.unwrap().ranks, [1]);
    let provider = supervisor.installed.workers.lock()[&url].provider.clone();
    let prompt = [1; 17];
    let mut store = full_store(&prompt[..16], 16, None);
    let root = provider.indexes[&1]
        .keys
        .plain_root_block(&prompt[..16])
        .unwrap();
    store.block_hashes[0] = crate::kv_index::wire::ExternalBlockHash::Bytes(root.to_vec());
    boundary_subscription(publisher, topic, true).await;
    finish_empty_startup_replay(replay_socket).await;
    let payload = event_payload(1.0, &[KVEvent::BlockStored(store)], Some(1));
    live_frame(publisher, topic.as_bytes(), 0, &payload);
    let hit = || !provider.matches(&prompt, 16, None, 1).is_empty();
    wait_for(&hit).await;

    let incompatible = METRICS_ONLY.replace("sha256_cbor", "sha256");
    let (reports, _) = metrics_meta::parse_cache_config_info(&incompatible);
    supervisor.fence_failed_registration(&url, &provider.lifecycle, &reports);
    assert!(!hit());
    let registration = &supervisor.worker_registrations()[&url];
    assert!(registration.info.as_ref().unwrap().ranks.is_empty());
    assert!(!supervisor.workers_have_cache_bindings([(url.as_str(), 1)].into_iter()));

    *worker.metrics.lock() = incompatible;
    let info = supervisor.on_worker_added(&url).await.unwrap();
    assert!(info.cache_key.is_none());
    assert!(info.ranks.is_empty());
    assert_eq!(worker.requests.metrics.load(Ordering::Relaxed), 2);
}

#[tokio::test]
async fn declared_layout_reuses_hash_block_size_metrics_for_admission() {
    let report = METRICS_ONLY
        .replace("sha256_cbor", "sha256")
        .replace("hash_block_size=\"None\"", "hash_block_size=\"16\"");
    let worker = HashAdmissionWorker::start(1, &report).await;
    let mut config = layout_only_sources("kv/admission/0", 16, "mock-model");
    config.hybrid.as_mut().unwrap().ranks[0].hash_block_size = None;
    let (supervisor, _) = KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
    let supervisor = supervisor.with_static_sources(HashMap::from([(worker.url.clone(), config)]));
    let info = supervisor.on_worker_added(&worker.url).await.unwrap();
    assert_eq!(info.block_size, Some(16));
    assert!(info.cache_key.is_none());
    assert!(info.ranks.is_empty());
    assert_eq!(worker.requests.metrics.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn registration_only_scrapes_hashes_for_usable_http_sources() {
    let worker = HashAdmissionWorker::start(1, METRICS_ONLY).await;
    let (supervisor, url) = (&worker.supervisor, &worker.url);
    supervisor.on_worker_added(url).await.unwrap();

    let mut discovered = supervisor.discover(url).await;
    let configs = discovered.engine_cache_configs.take().unwrap();
    assert_eq!(
        configs[&0].prefix_caching_hash_algo.as_deref(),
        Some("sha256_cbor")
    );

    let grpc_url = url.replacen("http://", "grpc://", 1);
    let layout = discovered.hybrid.take();
    assert!(!probe::needs_admission_hashes(url, &discovered));
    discovered.hybrid = layout;
    let mut sources = discovered.sources.as_mut().unwrap();
    let source = sources.get_mut("0").unwrap();
    let enabled = source.enable_kv_cache_events.replace(false);
    assert!(!probe::needs_admission_hashes(url, &discovered));
    sources = discovered.sources.as_mut().unwrap();
    sources.get_mut("0").unwrap().enable_kv_cache_events = enabled;
    assert!(probe::needs_admission_hashes(url, &discovered));
    assert!(!probe::needs_admission_hashes(&grpc_url, &discovered));
    discovered.engine_cache_configs = Some(configs);
    assert!(!probe::needs_admission_hashes(url, &discovered));

    let lifecycle = supervisor.lifecycle_for_worker(url);
    let probe = |budget| {
        supervisor.probe_registration(
            url,
            false,
            None,
            &lifecycle,
            tokio::time::Instant::now() + budget,
        )
    };
    probe(CONTROL_TIMEOUT).await.unwrap();
    let requests = || worker.requests.metrics.load(Ordering::Relaxed);
    assert_eq!(
        requests(),
        2,
        "gRPC and unchanged descriptors skip the scrape"
    );

    let sources = std::mem::replace(&mut *worker.sources.lock(), serde_json::json!([]));
    lifecycle.state.lock().trusted = false;
    assert!(probe(CONTROL_TIMEOUT).await.is_err());
    assert_eq!(requests(), 2, "failed source probe");

    *worker.sources.lock() = sources;
    let without_budget = probe(Duration::ZERO).await.unwrap();
    assert!(without_budget.worker.engine_cache_configs.is_none());
    assert_eq!(requests(), 2, "insufficient refresh budget");
    let with_budget = probe(CONTROL_TIMEOUT).await.unwrap();
    assert!(with_budget.worker.engine_cache_configs.is_some());
    assert_eq!(requests(), 3, "usable source for re-admission");
}

#[tokio::test]
async fn load_only_registration_replaces_binding_and_preserves_model_aliases() {
    let worker = HashAdmissionWorker::start(2, METRICS_ONLY).await;
    let supervisor = &worker.supervisor;
    let first = supervisor.on_worker_added(&worker.url).await.unwrap();
    assert!(first.cache_key.is_some());
    let old = supervisor.lifecycle_for_worker(&worker.url);

    *worker.metrics.lock() = concat!(
        "vllm:cache_config_info{engine=\"0\",prefix_caching_hash_algo=\"sha256\"} 1.0\n",
        "vllm:cache_config_info{engine=\"1\",prefix_caching_hash_algo=\"sha256\"} 1.0\n",
    )
    .to_owned();
    let (info, models) = supervisor
        .on_worker_added_with_models(&worker.url)
        .await
        .unwrap();
    assert!(info.ranks.is_empty());
    assert!(info.cache_key.is_none());
    assert!(!old.active.load(Ordering::Acquire));
    let load_only = supervisor.lifecycle_for_worker(&worker.url);
    assert!(!load_only.active.load(Ordering::Acquire));
    {
        let mut state = load_only.state.lock();
        state.trusted = false;
        state.checked = None;
    }
    assert!(!load_only.layout_rebind_due());
    assert!(supervisor.match_provider_for_worker(&worker.url).is_none());

    let bases = std::slice::from_ref(&worker.url);
    assert!(supervisor.worker_model_eligibility(bases, Some("mock-model"))[&worker.url]);
    assert!(supervisor.worker_model_eligibility(bases, Some("control-alias"))[&worker.url]);
    assert!(models.is_none());
    assert_eq!(worker.requests.models.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn failed_reregistration_revokes_only_the_incompatible_rank() {
    let worker = HashAdmissionWorker::start(2, METRICS_ONLY).await;
    let supervisor = &worker.supervisor;
    let url = worker.url.as_str();
    assert_eq!(
        supervisor.on_worker_added(url).await.unwrap().ranks.len(),
        2
    );
    let previous = supervisor.lifecycle_for_worker(url);
    let provider = supervisor.match_provider_for_worker(url).unwrap();
    let prompt = [1; 17];
    let hit = |rank| !provider.matches(&prompt, 16, None, rank).is_empty();
    let mut store = full_store(&prompt[..16], 16, None);
    store.block_hashes[0] = crate::kv_index::wire::ExternalBlockHash::Bytes(
        crate::kv_index::request_hash::BlockKeyGenerator::from_seed(16, "vllm-none-hash")
            .plain_root_block(&prompt[..16])
            .unwrap()
            .to_vec(),
    );
    let events = [KVEvent::BlockStored(store)];
    for (rank, (_, _, publisher, replay)) in worker.sockets.iter().enumerate() {
        let topic = format!("kv/admission/{rank}");
        boundary_subscription(publisher, &topic, true).await;
        finish_empty_startup_replay(replay).await;
        let payload = event_payload(1.0, &events, Some(rank as u32));
        live_frame(publisher, topic.as_bytes(), 0, &payload);
    }
    wait_for(|| hit(0) && hit(1)).await;

    let incompatible = METRICS_ONLY.replace("sha256_cbor", "sha256");
    *worker.metrics.lock() = incompatible.clone();
    let sources = worker.sources.lock().clone();
    worker.sources.lock()["1"]["replay_endpoint"] = "tcp://127.0.0.1:invalid".into();
    let error = supervisor.on_worker_added(url).await.unwrap_err();
    assert!(error.contains("invalid KV replay endpoint"), "{error}");
    assert!(!hit(0));
    assert!(hit(1));
    assert!(!supervisor.workers_have_cache_bindings([(url, 0)].into_iter()));
    assert!(supervisor.workers_have_cache_bindings([(url, 1)].into_iter()));

    *worker.metrics.lock() = format!(
        "{incompatible}{}",
        incompatible.replace("engine=\"0\"", "engine=\"1\"")
    );
    assert!(supervisor.on_worker_added(url).await.is_err());
    assert!(!hit(1));
    {
        let state = previous.state.lock();
        assert!(state.info.as_ref().unwrap().ranks.is_empty());
    }
    assert!(supervisor.match_provider_for_worker(url).is_some());

    *worker.sources.lock() = sources;
    *worker.metrics.lock() = METRICS_ONLY.to_owned();
    let replacement = supervisor.on_worker_added(url).await.unwrap();
    assert_eq!(replacement.ranks.len(), 2);
    let (reports, _) = metrics_meta::parse_cache_config_info(&incompatible);
    supervisor.fence_failed_registration(url, &previous, &reports);
    assert!(supervisor.workers_have_cache_bindings([(url, 0), (url, 1)].into_iter()));
}

#[tokio::test]
async fn explicit_reregistration_restores_load_only_binding() {
    let incompatible = METRICS_ONLY.replace("sha256_cbor", "sha256");
    let worker = HashAdmissionWorker::start(1, &incompatible).await;
    let (supervisor, url) = (&worker.supervisor, &worker.url);
    assert!(supervisor
        .on_worker_added(url)
        .await
        .unwrap()
        .cache_key
        .is_none());
    assert_eq!(*supervisor.first_block_size.lock(), None);

    *worker.metrics.lock() = METRICS_ONLY.to_owned();
    supervisor.lifecycle_for_worker(url).state.lock().checked = None;
    supervisor
        .refresh_workers(
            &HashSet::from([url.clone()]),
            &HashSet::new(),
            &HashSet::new(),
        )
        .await;
    assert!(supervisor.match_provider_for_worker(url).is_none());
    assert_eq!(worker.requests.metrics.load(Ordering::Relaxed), 1);

    assert_eq!(supervisor.on_worker_added(url).await.unwrap().ranks, [0]);
    assert_eq!(*supervisor.first_block_size.lock(), Some(16));
}

#[tokio::test]
async fn metrics_only_worker_supplies_cache_metadata() {
    let metrics = "vllm:cache_config_info{engine=\"0\"} 1.0\n\
        vllm:cache_config_info{block_size=\"8\",effective_attention_block_size=\"16\",engine=\"1\"} 1.0\n";
    let (router, _) = metrics_and_models_router(move || metrics.to_owned());
    let (worker, server) = spawn_metrics_only_worker(router).await;

    let metadata = query_worker_metadata_and_models(&reqwest::Client::new(), &worker, None).await;
    let info = metadata.server_info.unwrap();
    let models = metadata.base_models;
    assert_eq!(info.kv_block_size, Some(16));
    assert_eq!(discovered_block_size(&info), Some(16));
    assert_eq!(info.model_id.as_deref(), Some("mock-model"));
    assert_eq!(
        models.unwrap(),
        HashSet::from(["mock-model".to_owned(), "mock-alias".to_owned()])
    );
    server.abort();
}

#[tokio::test]
async fn metrics_discovery_requires_replay_and_retains_served_aliases() {
    use axum::routing::get;

    let (endpoint, replay_endpoint, _publisher, _replay) = layout_endpoints();
    for (with_replay, compatible) in [(false, true), (true, true), (true, false)] {
        let sources = serde_json::json!({"0": {
            "endpoint": endpoint,
            "replay_endpoint": with_replay.then_some(&replay_endpoint),
            "topic": "kv/metrics-only",
            "publisher": "zmq"
        }});
        let (router, requests) = metrics_and_models_router(move || {
            if compatible {
                METRICS_ONLY.to_owned()
            } else {
                METRICS_ONLY.replace("sha256_cbor", "sha256")
            }
        });
        let router = router.route(
            "/kv_event_sources",
            get(move || {
                let sources = sources.clone();
                async move { axum::Json(sources) }
            }),
        );
        let (worker, server) = spawn_metrics_only_worker(router).await;
        let _server = AbortTestServer(server);
        let (supervisor, _signals) =
            KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
        let (info, models) = supervisor
            .on_worker_added_with_models(&worker)
            .await
            .unwrap();
        assert_eq!(info.block_size, Some(16));
        let cache_eligible = with_replay && compatible;
        assert_eq!(info.cache_key.is_some(), cache_eligible);
        assert_eq!(info.ranks, if cache_eligible { vec![0] } else { vec![] });
        assert_eq!(
            supervisor.match_provider_for_worker(&worker).is_some(),
            cache_eligible
        );
        assert_eq!(
            supervisor.worker_model_eligibility(std::slice::from_ref(&worker), Some("mock-alias"))
                [&worker],
            with_replay
        );
        let labels = crate::kv_index::discovery::served_model_labels(
            &reqwest::Client::new(),
            &worker,
            None,
            None,
            models,
        )
        .await;
        assert_eq!(labels["served_model_aliases"], "mock-alias,mock-model");
        assert_eq!(requests.models.load(Ordering::Relaxed), 1);
        assert_eq!(requests.metrics.load(Ordering::Relaxed), 1);
    }
}

#[tokio::test]
async fn invalid_metrics_geometry_still_fences_incompatible_ranks() {
    use axum::routing::get;

    let (pub0, replay0, _publisher0, _replay0) = layout_endpoints();
    let (pub1, replay1, _publisher1, _replay1) = layout_endpoints();
    let sources = serde_json::json!({
        "0": {"endpoint": pub0, "replay_endpoint": replay0, "topic": "kv/geometry/0", "publisher": "zmq"},
        "1": {"endpoint": pub1, "replay_endpoint": replay1, "topic": "kv/geometry/1", "publisher": "zmq"}
    });
    let metrics = Arc::new(Mutex::new(METRICS_ONLY.to_owned()));
    let (app, requests) = metrics_and_models_router({
        let metrics = metrics.clone();
        move || metrics.lock().clone()
    });
    let app = app.route(
        "/kv_event_sources",
        get(move || {
            let sources = sources.clone();
            async move { axum::Json(sources) }
        }),
    );
    let (url, server) = spawn_metrics_only_worker(app).await;
    let _server = AbortTestServer(server);
    let (supervisor, _) = KvIndexSupervisor::new_http_full_attention(reqwest::Client::new());
    assert_eq!(
        supervisor.on_worker_added(&url).await.unwrap().ranks.len(),
        2
    );
    let provider = supervisor.match_provider_for_worker(&url).unwrap();
    let has_binding =
        |rank| supervisor.workers_have_cache_bindings([(url.as_str(), rank)].into_iter());
    assert!(has_binding(0) && has_binding(1));

    let missing = "vllm:cache_config_info{engine=\"0\",prefix_caching_hash_algo=\"sha256\"} 1.0\n";
    let conflicting = concat!(
        "vllm:cache_config_info{engine=\"0\",block_size=\"16\"} 1.0\n",
        "vllm:cache_config_info{engine=\"1\",block_size=\"32\",prefix_caching_hash_algo=\"sha256\"} 1.0\n",
    );
    for (report, rank1_stays_bound) in [(missing, true), (conflicting, false)] {
        *metrics.lock() = report.to_owned();
        let info = supervisor.on_worker_added(&url).await.unwrap();
        assert!(info.cache_key.is_none());
        assert!(Arc::ptr_eq(
            &provider,
            &supervisor.match_provider_for_worker(&url).unwrap()
        ));
        assert_eq!((has_binding(0), has_binding(1)), (false, rank1_stays_bound));
    }
    assert_eq!(requests.metrics.load(Ordering::Relaxed), 3);
}

#[tokio::test]
async fn worker_without_metadata_names_every_source() {
    let (worker, server) = spawn_metrics_only_worker(axum::Router::new()).await;

    let error = query_worker_metadata_and_models(&reqwest::Client::new(), &worker, None)
        .await
        .server_info
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
            base_models: None,
            hybrid: None,
            assumed_plain_full: false,
            engine_cache_configs: None,
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
            Some(Instant::now() - REFRESH_INTERVAL - CONTROL_TIMEOUT);
    };
    let initial = supervisor
        .refresh_workers(&bases, &HashSet::new(), &HashSet::new())
        .await;
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
    let rejected = supervisor
        .refresh_workers(&bases, &HashSet::new(), &HashSet::new())
        .await;
    assert!(rejected[&base].info.is_none());
    assert!(rejected[&base].provider.is_none());
    assert!(hits().is_empty());
    assert!(!supervisor.installed.workers.lock().contains_key(&base));
    *metadata.lock() = serde_json::json!({"0": valid});
    rewind();
    let restored = supervisor
        .refresh_workers(&bases, &HashSet::new(), &HashSet::new())
        .await;
    assert!(restored[&base].info.is_some());
    assert_ne!(restored[&base].revision, initial[&base].revision);

    healthy.store(false, Ordering::Relaxed);
    for failures in 1..MAX_PROBE_FAILURES {
        rewind();
        let registration = supervisor
            .refresh_workers(&bases, &HashSet::new(), &HashSet::new())
            .await;
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
    let retired = supervisor
        .refresh_workers(&bases, &HashSet::new(), &HashSet::new())
        .await;
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
    let recovered = supervisor
        .refresh_workers(&bases, &HashSet::new(), &HashSet::new())
        .await;
    assert!(recovered[&base].info.is_some());
    assert_ne!(recovered[&base].revision, retired[&base].revision);
    supervisor.shutdown();
    task.abort();
}

#[test]
fn placement_evidence_comes_from_the_accepted_retry() {
    use crate::kv_index::source::test_support::{event_batch, full_binding, full_store, warm_full};
    use crate::kv_index::{KVEvent, RequestHashCache, RoutableTarget};

    let lifecycle = WorkerLifecycle::new(Arc::new(AtomicBool::new(true)), true, None);
    let source = full_binding(
        RoutableTarget {
            instance_id: "worker".into(),
            dp_rank: 0,
        },
        16,
    );
    let prompt = [42; 33];
    source.begin_empty(0);
    warm_full(&source, 0, &prompt[..32]);
    let mut cache = RequestHashCache::for_single_target();
    cache.placement = Some(Vec::new());

    let mut calls = 0;
    let hits = lifecycle.match_rank(&source, &mut cache, |cache| {
        calls += 1;
        let hit = source.lookup_placement(&prompt, 16, None, 0, cache);
        assert!(hit.is_some());
        if calls == 1 {
            source
                .apply_batch(
                    1,
                    &event_batch(vec![
                        KVEvent::AllBlocksCleared(Default::default()),
                        KVEvent::BlockStored(full_store(&prompt[..32], 16, None)),
                    ]),
                )
                .unwrap();
        }
        hit
    });

    assert_eq!(calls, 2);
    assert_eq!(hits[0].segments, [(0, 2)]);
    let evidence = cache.placement.unwrap();
    assert_eq!(evidence.len(), 1);
    assert!(!evidence[0].is_revoked());
}

#[test]
fn rank_credit_does_not_cross_revocation_and_empty_replay() {
    use crate::kv_index::source::test_support::{full_binding, warm_full};
    use crate::policies::TieredMatchProvider;
    let lifecycle = WorkerLifecycle::new(Arc::new(AtomicBool::new(true)), true, None);
    let source = Arc::new(full_binding(
        crate::kv_index::RoutableTarget {
            instance_id: "p".into(),
            dp_rank: 0,
        },
        16,
    ));
    source.begin_empty(0);
    let tokens = vec![42; 17];
    let mut cache = crate::kv_index::RequestHashCache::for_single_target();
    let mut calls = 0;
    let scored = lifecycle.read_rank(&source, None, Instant::now, || {
        calls += 1;
        if calls == 1 {
            lifecycle.state.lock().trusted = false;
            warm_full(&source, 0, &tokens[..16]);
        }
        let scores = source.matches_with_cache(&tokens, 16, None, 0, &mut cache);
        if calls == 1 {
            assert_eq!(scores[0].segments, [(0, 1)]);
            assert_eq!(
                lifecycle.observe_layout(&StaticHybridMatch::Matches),
                SourceAction::Recover
            );
            let stream = crate::kv_index::EventStream::new(source.clone());
            let lease = stream.begin_replay().unwrap();
            stream.finish_replay(lease, 0).unwrap();
            lifecycle.state.lock().trusted = true;
        }
        (!scores.is_empty()).then_some(scores)
    });
    assert!(scored.is_none());
    assert_eq!(calls, 2);
}

#[test]
fn renewed_grants_do_not_price_pages_seen_only_after_expiry() {
    use crate::kv_index::source::test_support::{event_batch, full_binding, full_store, warm_full};
    use crate::kv_index::{BlockRemoved, KVEvent, RequestHashCache, RoutableTarget};
    use std::cell::Cell;
    for repeat in [false, true] {
        let life = WorkerLifecycle::new(Arc::new(AtomicBool::new(true)), true, None);
        let source = full_binding(
            RoutableTarget {
                instance_id: "p".into(),
                dp_rank: 0,
            },
            16,
        );
        source.begin_empty(0);
        let source_generation = source.control.generation();
        let tokens = vec![16; 17];
        let mut cache = RequestHashCache::for_single_target();
        let clock = Cell::new(life.state.lock().verified_at);
        let mut calls = 0;
        let result = life.read_rank(
            &source,
            Some(clock.get()),
            || clock.get(),
            || {
                calls += 1;
                if calls == 2 && !repeat {
                    return source.lookup_placement(&tokens, 16, None, 0, &mut cache);
                }
                clock.set(clock.get() + Duration::from_secs(6));
                assert!(!life.trusted_at(&life.state.lock(), || clock.get()));
                let seq = 2 * (calls - 1);
                warm_full(&source, seq, &tokens[..16]);
                let hits = source.lookup_placement(&tokens, 16, None, 0, &mut cache);
                assert!(hits.is_some());
                let removed = KVEvent::BlockRemoved(BlockRemoved {
                    block_hashes: full_store(&tokens[..16], 16, None).block_hashes,
                    medium: Some("GPU".into()),
                    group_idx: Some(0),
                    locality: None,
                    ownership: None,
                });
                source
                    .apply_batch(seq + 1, &event_batch(vec![removed]))
                    .unwrap();
                assert_eq!(source.control.generation(), source_generation);
                assert_eq!(
                    life.observe_layout(&StaticHybridMatch::Matches),
                    SourceAction::Keep
                );
                life.state.lock().verified_at = clock.get();
                hits
            },
        );
        assert!(result.is_none());
        assert_eq!(calls, 2);
    }
}

#[test]
fn server_info_accepts_legacy_block_size_and_optional_kv_fields() {
    for field in ["kv_block_size", "block_size"] {
        let info: ServerInfo = serde_json::from_value(serde_json::json!({
            "model_id": "m", (field): 16, "instance_id": "x", "data_parallel_size": 2
        }))
        .unwrap();
        assert_eq!(info.model_id.as_deref(), Some("m"));
        assert_eq!(info.instance_id.as_deref(), Some("x"));
        assert_eq!(info.kv_block_size, Some(16));
    }
    assert_eq!(
        serde_json::from_str::<ServerInfo>("{}").unwrap(),
        ServerInfo::default()
    );
}

#[test]
fn kv_event_discovery_accepts_both_wire_responses() {
    use super::probe::GrpcKvEventSourcesResponse;
    use crate::backend::pb::{GetKvEventSourcesResponse, KvEventSource};
    use prost::Message;

    let legacy = GetKvEventSourcesResponse {
        sources: vec![KvEventSource {
            data_parallel_rank: Some(0),
            topic: "engine-instance".to_owned(),
            ..Default::default()
        }],
    };
    let mut decoded =
        GrpcKvEventSourcesResponse::decode(legacy.encode_to_vec().as_slice()).unwrap();
    assert_eq!(decoded.sources, legacy.sources);
    assert_eq!(decoded.kv_routing_capabilities, None);

    decoded.kv_routing_capabilities = Some(br#"{"schema_version":1}"#.to_vec());
    let payload = decoded.encode_to_vec();
    assert_eq!(
        GrpcKvEventSourcesResponse::decode(payload.as_slice()).unwrap(),
        decoded
    );
    assert_eq!(
        GetKvEventSourcesResponse::decode(payload.as_slice()).unwrap(),
        legacy
    );
}

#[test]
fn wildcard_event_endpoints_preserve_ipv6_and_reject_zero_ports() {
    let host = worker_host("http://[2001:db8::1]:8000@3").unwrap();
    for input in ["tcp://*:5557", "tcp://[::]:5557", "tcp://0.0.0.0:5557"] {
        assert_eq!(
            resolve_endpoint(input, &host).unwrap(),
            "tcp://[2001:db8::1]:5557"
        );
    }
    assert!(resolve_endpoint("tcp://*:0", &host).is_err());
}

#[test]
fn adapter_names_are_not_served_base_models() {
    use crate::core::{BasicWorker, WorkerType};

    let worker = BasicWorker::new("http://127.0.0.1:8000".into(), WorkerType::Regular).with_labels(
        HashMap::from([
            ("model_id".into(), "qwen35".into()),
            ("served_model_aliases".into(), "alias-a,alias-b".into()),
        ]),
    );
    assert!(worker_serves_base_model(&worker, "qwen35"));
    assert!(worker_serves_base_model(&worker, "alias-a"));
    assert!(!worker_serves_base_model(&worker, "lora1"));
}
