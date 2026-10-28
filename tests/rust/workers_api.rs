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
    routers::{RouterFactory, RouterTrait},
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
