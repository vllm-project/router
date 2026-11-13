#![cfg(all(feature = "cpu", feature = "ready"))]

use std::sync::{Arc, Mutex};

use axum::{
    http::{HeaderMap, StatusCode},
    routing::get,
};
use vllm_router_rs::{
    config::RouterConfig,
    protocols::worker_spec::WorkerConfigRequest,
    routers::{http::pd_router::PdRouterBase, router_manager::RouterManager},
    server::AppContext,
};

struct HealthServer {
    url: String,
    requests: Arc<Mutex<Vec<Option<String>>>>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for HealthServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl HealthServer {
    async fn new(key: Option<&str>) -> Self {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = requests.clone();
        let expected = key.map(|key| format!("Bearer {key}"));
        let app = axum::Router::new().route(
            "/health",
            get(move |headers: HeaderMap| {
                let captured = captured.clone();
                let expected = expected.clone();
                async move {
                    let actual = headers
                        .get("authorization")
                        .map(|v| v.to_str().unwrap().to_owned());
                    let matches = actual == expected;
                    captured.lock().unwrap().push(actual);
                    if matches {
                        StatusCode::OK
                    } else {
                        StatusCode::UNAUTHORIZED
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            url,
            requests,
            task,
        }
    }

    fn assert_requests(&self, key: Option<&str>) {
        let requests = self.requests.lock().unwrap();
        assert!(!requests.is_empty());
        let expected = key.map(|key| format!("Bearer {key}"));
        assert!(requests.iter().all(|actual| actual == &expected));
    }
}

fn context(key: Option<&str>, dp_size: usize) -> Arc<AppContext> {
    Arc::new(
        AppContext::new(
            RouterConfig {
                api_key: key.map(str::to_owned),
                intra_node_data_parallel_size: dp_size,
                worker_startup_timeout_secs: 1,
                worker_startup_check_interval_secs: 1,
                ..Default::default()
            },
            reqwest::Client::new(),
            10,
            None,
            vec![],
        )
        .unwrap(),
    )
}

#[tokio::test]
async fn pd_health_auth_startup_runtime_and_dynamic_workers() {
    for key in [Some("worker-key"), None] {
        for dp_size in [1, 2] {
            let prefill = HealthServer::new(key).await;
            let decode = HealthServer::new(key).await;
            let added_prefill = HealthServer::new(key).await;
            let added_decode = HealthServer::new(key).await;
            let ctx = context(key, dp_size);
            let router = PdRouterBase::new(
                vec![(prefill.url.clone(), None)],
                vec![decode.url.clone()],
                &ctx,
            )
            .await
            .unwrap();
            router
                .add_prefill_server(added_prefill.url.clone(), None)
                .await
                .unwrap();
            router
                .add_decode_server(added_decode.url.clone())
                .await
                .unwrap();
            let workers = ctx.worker_registry.get_all();
            assert_eq!(workers.len(), 4 * dp_size);
            for worker in workers.iter() {
                worker.check_health_async().await.unwrap();
                assert!(!format!("{worker:?}").contains("worker-key"));
            }
            for server in [&prefill, &decode, &added_prefill, &added_decode] {
                server.assert_requests(key);
                assert_eq!(server.requests.lock().unwrap().len(), 1 + dp_size);
            }
        }
    }
}

#[tokio::test]
async fn igw_registered_workers_do_not_forward_router_credentials() {
    for key in [Some("worker-key"), None] {
        let ctx = context(key, 1);
        let manager = RouterManager::new(
            ctx.router_config.clone(),
            ctx.client.clone(),
            ctx.worker_registry.clone(),
            ctx.policy_registry.clone(),
        );
        for role in ["regular", "prefill", "decode"] {
            let server = HealthServer::new(key).await;
            let config: WorkerConfigRequest = serde_json::from_value(serde_json::json!({
                "url": server.url,
                "model_id": "test-model",
                "worker_type": role,
                "labels": {"test": role}
            }))
            .unwrap();
            manager.add_worker(config).await.unwrap();
            let worker = ctx.worker_registry.get_by_url(&server.url).unwrap();
            if key.is_some() {
                assert!(worker.check_health_async().await.is_err());
            } else {
                worker.check_health_async().await.unwrap();
            }
            assert_eq!(worker.metadata().labels.get("test").unwrap(), role);
            server.assert_requests(None);
            assert!(!format!("{worker:?}").contains("worker-key"));
        }
    }
}
