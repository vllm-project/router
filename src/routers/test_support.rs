use axum::{
    extract::Request,
    http::{header::AUTHORIZATION, StatusCode},
    middleware::{self, Next},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use tokio::{net::TcpListener, sync::watch, task::JoinHandle};

type RequestLog = watch::Receiver<Vec<(String, StatusCode)>>;

/// Records only the path and response status for each worker request.
pub(super) async fn start_admin_auth_mock_server(
    api_key: Option<&str>,
) -> (String, JoinHandle<()>, RequestLog) {
    let expected = api_key.map(|key| format!("Bearer {key}"));
    let (recorded, requests) = watch::channel(Vec::new());
    let authorize = move |req: Request, next: Next| {
        let authorized = req
            .headers()
            .get_all(AUTHORIZATION)
            .iter()
            .map(|value| value.as_bytes())
            .eq(expected.as_deref().map(str::as_bytes));
        let path = req.uri().path().to_owned();
        let recorded = recorded.clone();
        async move {
            let response = if authorized {
                next.run(req).await
            } else {
                StatusCode::UNAUTHORIZED.into_response()
            };
            recorded.send_modify(|log| log.push((path, response.status())));
            response
        }
    };
    let ok = || async { StatusCode::OK };
    let app = Router::new()
        .route("/health", get(ok))
        .route("/flush_cache", post(ok))
        .route(
            "/get_load",
            get(|| async { Json(serde_json::json!({"load": 7})) }),
        )
        .route("/start_profile", post(ok))
        .route("/stop_profile", post(ok))
        .route("/health_generate", get(ok))
        .route_layer(middleware::from_fn(authorize));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), handle, requests)
}
