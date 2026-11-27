use axum::body::{Body, Bytes};
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE, ORIGIN};
use axum::http::{HeaderValue, Request};
use serde_json::json;
use tokio::io::AsyncReadExt;
use tokio::net::TcpStream;

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

/// A JSON `POST` to `path` with a streamed body: no `Content-Length` and an
/// unknown size hint, like a chunked upload.
pub fn post_stream(path: &str, chunks: Vec<std::io::Result<Bytes>>) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from_stream(futures_util::stream::iter(chunks)))
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

/// Read one HTTP/1.1 request with a `content-length` body; `None` at EOF.
pub async fn read_request(
    socket: &mut TcpStream,
    buffer: &mut Vec<u8>,
) -> Option<(String, Vec<u8>)> {
    loop {
        if let Some(end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buffer[..end]).into_owned();
            let length: usize = head
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    if name.eq_ignore_ascii_case("content-length") {
                        value.trim().parse().ok()
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            let total = end + 4 + length;
            if buffer.len() >= total {
                let body = buffer[end + 4..total].to_vec();
                buffer.drain(..total);
                return Some((head, body));
            }
        }
        let mut chunk = [0; 4096];
        match socket.read(&mut chunk).await {
            Ok(0) | Err(_) => return None,
            Ok(read) => buffer.extend_from_slice(&chunk[..read]),
        }
    }
}
