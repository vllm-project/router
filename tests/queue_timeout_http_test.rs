//! Exercise the production queue and admission middleware over loopback HTTP.

use axum::{middleware::from_fn_with_state, routing::post, Router};
use std::{sync::Arc, time::Duration};
use tokio::{sync::mpsc, task::JoinHandle};
use vllm_router_rs::{
    config::RouterConfig,
    middleware::{concurrency_limit_middleware, QueueProcessor},
    routers::RouterFactory,
    server::{AppContext, AppState},
};

struct AbortOnDrop(JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[tokio::test]
async fn queued_http_requests_survive_more_than_one_refill_wait() {
    let context = Arc::new(
        AppContext::new(
            RouterConfig::default(),
            reqwest::Client::new(),
            1,
            Some(1),
            vec![],
        )
        .unwrap(),
    );
    let router = RouterFactory::create_router(&context).await.unwrap();
    let (queue_tx, queue_rx) = mpsc::channel(2);
    let processor = QueueProcessor::new(
        Arc::clone(&context.rate_limiter),
        queue_rx,
        Duration::from_secs(4),
    );
    let mut queue_task = AbortOnDrop(tokio::spawn(processor.run()));
    let state = Arc::new(AppState {
        router: Arc::from(router),
        context: Arc::clone(&context),
        concurrency_queue_tx: Some(queue_tx),
        router_manager: None,
    });
    let app = Router::new()
        .route(
            "/v1/responses",
            post(|| async {
                // Prevent an immediately returned token from masking contention.
                tokio::time::sleep(Duration::from_millis(300)).await;
                "admitted"
            }),
        )
        .layer(from_fn_with_state(state, concurrency_limit_middleware));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let mut server = AbortOnDrop(tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    }));
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(6))
        .build()
        .unwrap();
    context.rate_limiter.try_acquire(1.0).await.unwrap();
    let url = format!("http://{address}/v1/responses");
    let (first, second) = tokio::join!(client.post(&url).send(), client.post(&url).send(),);
    let first = first.unwrap();
    let second = second.unwrap();
    let statuses = (first.status(), second.status());
    let bodies = (first.text().await.unwrap(), second.text().await.unwrap());
    server.0.abort();
    assert!((&mut server.0).await.unwrap_err().is_cancelled());
    queue_task.0.abort();
    let _ = (&mut queue_task.0).await;
    assert_eq!(statuses, (reqwest::StatusCode::OK, reqwest::StatusCode::OK));
    assert_eq!(bodies, ("admitted".to_owned(), "admitted".to_owned()));
}
