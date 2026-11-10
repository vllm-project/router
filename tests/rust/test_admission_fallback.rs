#![cfg(all(feature = "cpu", feature = "ready"))]

//! Unmatched paths go to the router's fallback (the transparent proxy). The
//! fallback must get the same global layers as the named routes: axum's
//! `Router::layer` wraps only the routes and fallback that exist when it is
//! called.

mod common;

use axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN;
use axum::http::StatusCode;
use axum::Router;
use tower::ServiceExt;
use vllm_router_rs::config::RouterConfig;

const LIMIT: usize = 1024;
const ORIGIN: &str = "http://client.example";

/// The app under test, with `--max-payload-size` set to `LIMIT`.
async fn app() -> Router {
    let config = RouterConfig {
        max_payload_size: LIMIT,
        ..Default::default()
    };
    common::test_app::create_test_app_from_config(&config).await
}

/// Request ID, CORS and the body limit apply to an unmatched path the same
/// way they apply to a named route.
#[tokio::test]
async fn fallback_gets_global_layers() {
    let app = app().await;

    for path in ["/v1/completions", "/no/such/route"] {
        let response = app
            .clone()
            .oneshot(common::http::post_with_origin(
                path,
                "{}".to_string(),
                ORIGIN,
            ))
            .await
            .expect("response");
        println!("POST {path}, 2 B: {}", response.status());
        assert!(
            response.headers().contains_key("x-request-id"),
            "POST {path}: no x-request-id header"
        );
        assert!(
            response.headers().contains_key(ACCESS_CONTROL_ALLOW_ORIGIN),
            "POST {path}: no CORS headers"
        );

        let response = app
            .clone()
            .oneshot(common::http::post_with_origin(
                path,
                common::http::json_body_of_len(LIMIT + 1),
                ORIGIN,
            ))
            .await
            .expect("response");
        println!("POST {path}, {} B: {}", LIMIT + 1, response.status());
        assert_eq!(
            response.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "POST {path}: body over --max-payload-size"
        );
    }
}

/// A body with no `Content-Length` passes the limit layer's early check and
/// fails only while it is read. The status must match a named route: 413 when
/// the limit is exceeded, 400 for any other read failure.
#[tokio::test]
async fn streamed_body_errors_match_named_routes() {
    let app = app().await;

    for path in ["/v1/completions", "/no/such/route"] {
        let body = common::http::json_body_of_len(LIMIT + 1);
        let response = app
            .clone()
            .oneshot(common::http::post_stream(path, vec![Ok(body.into())]))
            .await
            .expect("response");
        assert_eq!(
            response.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "POST {path}: streamed body over --max-payload-size"
        );

        let response = app
            .clone()
            .oneshot(common::http::post_stream(
                path,
                vec![Err(std::io::Error::other("injected read failure"))],
            ))
            .await
            .expect("response");
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "POST {path}: body read failure"
        );
    }
}
