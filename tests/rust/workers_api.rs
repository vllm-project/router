#![cfg(all(feature = "cpu", feature = "ready"))]

use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
    routing::get,
    Router,
};
use serde_json::Value;
use std::sync::Arc;
use tower::ServiceExt;
use vllm_router_rs::{
    config::{ConfigValidator, EpdConfig, PolicyConfig, RouterConfig, RoutingMode},
    policies::{CacheAwareCandidate, CacheAwarePolicy, LoadBalancingPolicy},
    routers::{
        router_manager::{RouterId, RouterManager},
        RouterFactory, RouterTrait,
    },
    server::{build_app_with_request_tracing, AppContext, AppState},
};

async fn assert_epd_list_to_detail(pd: bool) {
    let backend = Router::new()
        .route("/pd/health", get(|| async { "ok" }))
        .route("/p/health", get(|| async { "ok" }))
        .route("/d/health", get(|| async { "ok" }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, backend).await.unwrap() });

    let encoder_url = format!("{base}/e");
    let consumer_urls = if pd {
        vec![format!("{base}/p"), format!("{base}/d")]
    } else {
        vec![format!("{base}/pd")]
    };
    let mode = if pd {
        RoutingMode::VllmPrefillDecode {
            prefill_urls: vec![(consumer_urls[0].clone(), None)],
            decode_urls: vec![consumer_urls[1].clone()],
            prefill_policy: None,
            decode_policy: None,
            discovery_address: None,
        }
    } else {
        RoutingMode::Regular {
            worker_urls: consumer_urls.clone(),
        }
    };
    let config = RouterConfig {
        mode,
        policy: PolicyConfig::Random,
        epd: Some(EpdConfig {
            encoder_urls: vec![encoder_url.clone()],
            encode_policy: None,
            consumer_zmq_addrs: Default::default(),
        }),
        worker_startup_timeout_secs: 5,
        ..RouterConfig::default()
    };
    ConfigValidator::validate(&config).unwrap();
    let context =
        Arc::new(AppContext::new(config, reqwest::Client::new(), 64, None, vec![]).unwrap());
    let router: Arc<dyn RouterTrait> =
        Arc::from(RouterFactory::create_router(&context).await.unwrap());
    let mut routing_urls = router.get_worker_urls();
    routing_urls.sort();
    let mut expected_urls = consumer_urls;
    expected_urls.sort();
    assert_eq!(routing_urls, expected_urls);
    assert!(!routing_urls.contains(&encoder_url));

    // Management must still expose an unhealthy encoder excluded from routing.
    context
        .worker_registry
        .get_by_url(&encoder_url)
        .unwrap()
        .set_healthy(false);
    let app = build_app_with_request_tracing(
        Arc::new(AppState {
            router,
            context,
            concurrency_queue_tx: None,
            router_manager: None,
        }),
        1024 * 1024,
        vec![],
        vec![],
        false,
        false,
    );
    let response = app
        .clone()
        .oneshot(Request::get("/workers").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let list: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 16384).await.unwrap()).unwrap();
    expected_urls.push(encoder_url.clone());
    expected_urls.sort();
    let workers = list["workers"].as_array().unwrap();
    let mut listed_urls: Vec<_> = workers
        .iter()
        .map(|worker| worker["url"].as_str().unwrap().to_string())
        .collect();
    listed_urls.sort();
    assert_eq!(listed_urls, expected_urls);
    assert_eq!(list["stats"]["encode_count"], 1);
    for worker in workers {
        let url = worker["url"].as_str().unwrap();
        if url == encoder_url {
            assert_eq!(worker["worker_type"], "encode");
            assert_eq!(worker["is_healthy"], false);
        }
        let encoded_url: String = url::form_urlencoded::byte_serialize(url.as_bytes()).collect();
        let response = app
            .clone()
            .oneshot(
                Request::get(format!("/workers/{encoded_url}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "listed worker: {url}");
        let detail: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 16384).await.unwrap()).unwrap();
        for field in ["url", "model_id", "is_healthy"] {
            assert_eq!(detail[field], worker[field], "{url}: {field}");
        }
    }

    let missing_url: String =
        url::form_urlencoded::byte_serialize(format!("{base}/missing").as_bytes()).collect();
    let response = app
        .oneshot(
            Request::get(format!("/workers/{missing_url}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let error: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 16384).await.unwrap()).unwrap();
    assert_eq!(error["code"], "WORKER_NOT_FOUND");
    server.abort();
}

#[tokio::test]
async fn regular_epd_workers_list_to_detail() {
    assert_epd_list_to_detail(false).await;
}

#[tokio::test]
async fn pd_epd_workers_list_to_detail() {
    assert_epd_list_to_detail(true).await;
}

async fn assert_pd_epd_http_delete_cleans_policy(prefill_first: bool) {
    let cache_config = PolicyConfig::CacheAware {
        cache_threshold: 0.3,
        balance_abs_threshold: 32,
        balance_rel_threshold: 1.5,
        eviction_interval_secs: 3600,
        max_tree_size: 1024,
    };
    for main_policy in [PolicyConfig::RoundRobin, cache_config.clone()] {
        let backend = Router::new()
            .route("/p/health", get(|| async { "ok" }))
            .route("/d/health", get(|| async { "ok" }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, backend).await.unwrap() });
        let prefill_url = format!("{base}/p");
        let decode_url = format!("{base}/d");
        let encoder_url = format!("{base}/e");
        let missing_url = format!("{base}/missing");
        let config = RouterConfig {
            mode: RoutingMode::VllmPrefillDecode {
                prefill_urls: vec![(prefill_url.clone(), None)],
                decode_urls: vec![decode_url.clone()],
                prefill_policy: Some(cache_config.clone()),
                decode_policy: Some(cache_config.clone()),
                discovery_address: None,
            },
            policy: main_policy,
            epd: Some(EpdConfig {
                encoder_urls: vec![encoder_url.clone()],
                encode_policy: None,
                consumer_zmq_addrs: Default::default(),
            }),
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            ..RouterConfig::default()
        };
        ConfigValidator::validate(&config).unwrap();
        let context =
            Arc::new(AppContext::new(config, reqwest::Client::new(), 64, None, vec![]).unwrap());
        // Exercise the production factory and single-router DELETE delegation.
        let router: Arc<dyn RouterTrait> =
            Arc::from(RouterFactory::create_router(&context).await.unwrap());
        let prefill_policy = context.policy_registry.get_prefill_policy();
        let decode_policy = context.policy_registry.get_decode_policy();
        let default_policy = context.policy_registry.get_default_policy();
        assert!(!Arc::ptr_eq(&prefill_policy, &decode_policy));
        assert!(!Arc::ptr_eq(&prefill_policy, &default_policy));
        assert!(!Arc::ptr_eq(&decode_policy, &default_policy));
        let prompt = "cached static EPD HTTP deletion prefix";
        let seed = |policy: &Arc<dyn LoadBalancingPolicy>, urls: &[&str]| {
            if let Some(cache) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                for url in urls {
                    cache.commit_placement("unknown", prompt, url);
                }
            }
        };
        seed(&prefill_policy, &[&prefill_url]);
        seed(&decode_policy, &[&decode_url]);
        let affinity = |policy: &Arc<dyn LoadBalancingPolicy>, url: &str| {
            policy
                .as_any()
                .downcast_ref::<CacheAwarePolicy>()
                .map(|cache| {
                    cache
                        .evaluate_initial_placement(
                            "unknown",
                            prompt,
                            &[CacheAwareCandidate {
                                target_id: url,
                                kv_pressure: 0.0,
                            }],
                        )
                        .unwrap()
                        .match_rate
                })
        };
        let assert_state = |prefill_present: bool, decode_present: bool| {
            let count = usize::from(prefill_present) + usize::from(decode_present);
            for (policy, url, present) in [
                (&prefill_policy, &prefill_url, prefill_present),
                (&decode_policy, &decode_url, decode_present),
            ] {
                let expected = if present { 1.0 } else { 0.0 };
                assert_eq!(context.worker_registry.get_by_url(url).is_some(), present);
                // Read affinity directly; selecting would lazily prune stale tenants.
                assert_eq!(affinity(policy, url), Some(expected));
            }
            assert_eq!(context.worker_registry.get_pd_workers().len(), count);
            // Static PD construction does not establish model-policy accounting.
            // Cleanup must work without a model mapping or inventing counts.
            assert!(context.policy_registry.get_worker_counts().is_empty());
            assert!(context.policy_registry.get_policy("unknown").is_none());
        };
        assert_state(true, true);
        let app = build_app_with_request_tracing(
            Arc::new(AppState {
                router,
                context: context.clone(),
                concurrency_queue_tx: None,
                router_manager: None,
            }),
            1024 * 1024,
            vec![],
            vec![],
            false,
            false,
        );
        let (first, second) = if prefill_first {
            (&prefill_url, &decode_url)
        } else {
            (&decode_url, &prefill_url)
        };
        let mut prefill_present = true;
        let mut decode_present = true;
        for url in [
            &missing_url,
            &encoder_url,
            &encoder_url,
            first,
            first,
            second,
            second,
        ] {
            let encoded: String = url::form_urlencoded::byte_serialize(url.as_bytes()).collect();
            let response = app
                .clone()
                .oneshot(
                    Request::delete(format!("/workers/{encoded}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let response: Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 16384).await.unwrap())
                    .unwrap();
            assert_eq!(response["success"], true);
            assert!(context.worker_registry.get_by_url(url).is_none());
            prefill_present &= url != &prefill_url;
            decode_present &= url != &decode_url;
            assert_state(prefill_present, decode_present);
        }
        server.abort();
    }
}

#[tokio::test]
async fn pd_epd_http_delete_prefill_first_cleans_cache_aware_policy() {
    assert_pd_epd_http_delete_cleans_policy(true).await;
}

#[tokio::test]
async fn pd_epd_http_delete_decode_first_cleans_cache_aware_policy() {
    assert_pd_epd_http_delete_cleans_policy(false).await;
}

async fn assert_igw_http_delete_cleans_policy(prefill_first: bool) {
    let cache_config = PolicyConfig::CacheAware {
        cache_threshold: 0.3,
        balance_abs_threshold: 32,
        balance_rel_threshold: 1.5,
        eviction_interval_secs: 3600,
        max_tree_size: 1024,
    };
    for main_policy in [cache_config.clone(), PolicyConfig::RoundRobin] {
        let config = RouterConfig {
            enable_igw: true,
            policy: main_policy.clone(),
            ..RouterConfig::default()
        };
        ConfigValidator::validate(&config).unwrap();
        let context = Arc::new(
            AppContext::new(config.clone(), reqwest::Client::new(), 64, None, vec![]).unwrap(),
        );
        let manager = Arc::new(RouterManager::new(
            config,
            context.client.clone(),
            context.worker_registry.clone(),
            context.policy_registry.clone(),
        ));
        // Mirror IGW startup: both child routers share the management registries.
        let regular = RouterFactory::create_regular_router(&[], &context)
            .await
            .unwrap();
        manager.register_router(RouterId::new("http-regular".into()), Arc::from(regular));
        // None mirrors production IGW fallback. RoundRobin additionally checks
        // the factory's independent cache-aware role policies without a cache-aware model policy.
        let role_config = matches!(main_policy, PolicyConfig::RoundRobin).then_some(&cache_config);
        let pd = RouterFactory::create_vllm_pd_router(
            &[],
            &[],
            None,
            role_config,
            role_config,
            &main_policy,
            &context,
        )
        .await
        .unwrap();
        manager.register_router(RouterId::new("http-pd".into()), Arc::from(pd));
        let app = build_app_with_request_tracing(
            Arc::new(AppState {
                router: manager.clone(),
                context: context.clone(),
                concurrency_queue_tx: None,
                router_manager: Some(manager),
            }),
            1024 * 1024,
            vec![],
            vec![],
            false,
            false,
        );
        let urls = [
            "http://regular:8000",
            "http://prefill:8000",
            "http://decode:8000",
        ];
        let survivor_url = "http://survivor:8000";
        // Use POST rather than direct registry registration to exercise model
        // policy assignment and the complete IGW HTTP lifecycle.
        for (url, role, model) in [
            (urls[0], "regular", "model"),
            (urls[1], "prefill", "model"),
            (urls[2], "decode", "model"),
            (survivor_url, "regular", "survivor"),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::post("/workers")
                        .header("content-type", "application/json")
                        .body(Body::from(
                            serde_json::json!({
                                "url": url, "worker_type": role, "model_id": model,
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        let policies = &context.policy_registry;
        let model_policy = policies.get_policy("model").unwrap();
        let prefill_policy = policies.get_prefill_policy();
        let decode_policy = policies.get_decode_policy();
        assert!(Arc::ptr_eq(&model_policy, &policies.get_default_policy()));
        assert!(!Arc::ptr_eq(&model_policy, &prefill_policy));
        assert!(!Arc::ptr_eq(&model_policy, &decode_policy));
        assert!(!Arc::ptr_eq(&prefill_policy, &decode_policy));
        let prompt = "cached dynamic IGW HTTP deletion prefix";
        let seed = |policy: &Arc<dyn LoadBalancingPolicy>, model: &str, url: &str| {
            if let Some(cache) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                cache.commit_placement(model, prompt, url);
            }
        };
        for url in urls {
            seed(&model_policy, "model", url);
        }
        seed(&model_policy, "survivor", survivor_url);
        seed(&prefill_policy, "model", urls[1]);
        seed(&decode_policy, "model", urls[2]);
        let affinity = |policy: &Arc<dyn LoadBalancingPolicy>, model: &str, url: &str| {
            policy
                .as_any()
                .downcast_ref::<CacheAwarePolicy>()
                .map(|cache| {
                    cache
                        .evaluate_initial_placement(
                            model,
                            prompt,
                            &[CacheAwareCandidate {
                                target_id: url,
                                kv_pressure: 0.0,
                            }],
                        )
                        .unwrap()
                        .match_rate
                })
        };
        let mut present = [true; 3];
        let assert_state = |present: &[bool; 3]| {
            let count = present.iter().filter(|&&value| value).count();
            assert_eq!(
                policies
                    .get_worker_counts()
                    .get("model")
                    .copied()
                    .unwrap_or(0),
                count
            );
            assert_eq!(policies.get_policy("model").is_some(), count > 0);
            assert_eq!(policies.get_worker_counts().get("survivor"), Some(&1));
            if model_policy.name() == "cache_aware" {
                assert_eq!(affinity(&model_policy, "survivor", survivor_url), Some(1.0));
            }
            for (index, url) in urls.iter().enumerate() {
                let expected = if present[index] { 1.0 } else { 0.0 };
                assert_eq!(
                    context.worker_registry.get_by_url(url).is_some(),
                    present[index]
                );
                if model_policy.name() == "cache_aware" {
                    assert_eq!(
                        affinity(&model_policy, "model", url),
                        Some(expected),
                        "model: {url}"
                    );
                }
                if index > 0 {
                    let role_policy = if index == 1 {
                        &prefill_policy
                    } else {
                        &decode_policy
                    };
                    assert_eq!(
                        affinity(role_policy, "model", url),
                        Some(expected),
                        "role: {url}"
                    );
                }
            }
        };
        assert_state(&present);
        let (first, second) = if prefill_first { (1, 2) } else { (2, 1) };
        // Last consumer cleanup, duplicate deletion, and unknown URL errors
        // must preserve the surviving model sharing the default policy Arc.
        for index in [
            None,
            Some(first),
            Some(first),
            Some(0),
            Some(second),
            Some(second),
        ] {
            let url = index.map_or("http://missing:8000", |index| urls[index]);
            let existed = index.is_some_and(|index| present[index]);
            let encoded: String = url::form_urlencoded::byte_serialize(url.as_bytes()).collect();
            let response = app
                .clone()
                .oneshot(
                    Request::delete(format!("/workers/{encoded}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                if existed {
                    StatusCode::OK
                } else {
                    StatusCode::BAD_REQUEST
                }
            );
            let body: Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 16384).await.unwrap())
                    .unwrap();
            if existed {
                assert_eq!(body["success"], true);
            } else {
                assert_eq!(body["code"], "WORKER_NOT_FOUND");
            }
            if let Some(index) = index {
                present[index] = false;
            }
            assert_state(&present);
        }
    }
}

#[tokio::test]
async fn igw_http_delete_prefill_first_cleans_cache_aware_policy() {
    assert_igw_http_delete_cleans_policy(true).await;
}

#[tokio::test]
async fn igw_http_delete_decode_first_cleans_cache_aware_policy() {
    assert_igw_http_delete_cleans_policy(false).await;
}

#[tokio::test]
async fn openai_worker_detail_without_registry_entry() {
    let url = "http://openai.example/v1";
    let config = RouterConfig {
        mode: RoutingMode::OpenAI {
            worker_urls: vec![format!("{url}/")],
        },
        ..RouterConfig::default()
    };
    let context =
        Arc::new(AppContext::new(config, reqwest::Client::new(), 64, None, vec![]).unwrap());
    let router: Arc<dyn RouterTrait> =
        Arc::from(RouterFactory::create_router(&context).await.unwrap());
    assert!(context.worker_registry.get_all().is_empty());
    assert_eq!(router.get_worker_urls(), vec![url]);
    let app = build_app_with_request_tracing(
        Arc::new(AppState {
            router,
            context,
            concurrency_queue_tx: None,
            router_manager: None,
        }),
        1024 * 1024,
        vec![],
        vec![],
        false,
        false,
    );
    for (worker_url, status) in [
        (url, StatusCode::OK),
        ("http://missing.example", StatusCode::NOT_FOUND),
    ] {
        let encoded: String = url::form_urlencoded::byte_serialize(worker_url.as_bytes()).collect();
        let response = app
            .clone()
            .oneshot(
                Request::get(format!("/workers/{encoded}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), status, "worker: {worker_url}");
        let detail: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 16384).await.unwrap()).unwrap();
        if status == StatusCode::OK {
            assert_eq!(detail["url"], url);
            assert_eq!(detail["model_id"], "unknown");
            assert_eq!(detail["is_healthy"], true);
        } else {
            assert_eq!(detail["code"], "WORKER_NOT_FOUND");
        }
    }
}
