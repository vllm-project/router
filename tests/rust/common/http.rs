use axum::body::Body;
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE, ORIGIN};
use axum::http::{HeaderValue, Request};
use serde_json::json;

/// A JSON body of exactly `len` bytes.
pub fn json_body_of_len(len: usize) -> String {
    let envelope = json!({ "prompt": "" }).to_string().len();
    let body = json!({ "prompt": "x".repeat(len - envelope) }).to_string();
    assert_eq!(body.len(), len);
    body
}

/// A JSON `POST` to `path` with `body`.
pub fn post(path: &str, body: String) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(CONTENT_TYPE, "application/json")
        .header(CONTENT_LENGTH, body.len())
        .body(Body::from(body))
        .expect("request")
}

/// Like [`post`], with an `Origin` header (for CORS checks).
pub fn post_with_origin(path: &str, body: String, origin: &str) -> Request<Body> {
    let mut request = post(path, body);
    request
        .headers_mut()
        .insert(ORIGIN, HeaderValue::from_str(origin).expect("origin"));
    request
}
