#![cfg(all(feature = "cpu", feature = "ready"))]

//! Unmatched paths go to the router's fallback (the transparent proxy). The
//! fallback must get the same global layers as the named routes: axum's
//! `Router::layer` wraps only the routes and fallback that exist when it is
//! called.

mod common;

use axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN;
use axum::http::StatusCode;
use std::sync::Arc;
use tower::ServiceExt;
use vllm_router_rs::config::RouterConfig;
use vllm_router_rs::routers::{RouterFactory, RouterTrait};

const LIMIT: usize = 1024;
const ORIGIN: &str = "http://client.example";

/// Request ID, CORS and the body limit apply to an unmatched path the same
/// way they apply to a named route.
#[tokio::test]
async fn fallback_gets_global_layers() {
    let config = RouterConfig {
        max_payload_size: LIMIT,
        ..Default::default()
    };
    let context = common::create_test_context(config.clone());
    let router: Arc<dyn RouterTrait> = Arc::from(
        RouterFactory::create_router(&context)
            .await
            .expect("router"),
    );
    let app = common::test_app::create_test_app(router, reqwest::Client::new(), &config);

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
