//! Worker mocks for the E/P/D rig in `rig.rs`.
//!
//! A `Mock` plays one worker (an encoder, P, D or PD). Its `Mode` says how it
//! answers; `serve` puts it on a port. The mock records each request and how
//! it ended, so a test can check what the router sent and whether the router
//! closed the connection.

use super::{CHAT_ROUTE, WAIT_DURATION};
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Json;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;
use tokio::sync::{Notify, Semaphore};
use tokio::task::JoinHandle;

/// How a mock answers a request. The mode is read when the request arrives.
/// "Hold" means: wait for `Mock::release`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The role's normal reply, at once.
    Reply,
    /// Hold, then the normal reply.
    HoldThenReply,
    /// The headers and the first byte of the body at once; hold the rest.
    StallBody,
    /// `StallBody`, but the body is not JSON.
    StallBodyInvalidJson,
    /// `StallBody`, but the JSON has no `kv_transfer_params`.
    StallBodyWithoutKv,
    /// `StallBody`, but the body ends with an error instead of the rest.
    BreakBody,
}

/// The whole body that `Mode::StallBodyInvalidJson` sends. It starts like a
/// JSON object, but it is not JSON.
pub const INVALID_JSON_BODY: &str = "{\"choices\": [not json";

/// Which worker the mock plays. The role picks the normal reply body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Encoder,
    /// `push`: the reply carries P's KV-push address. The router needs it
    /// before it takes the concurrent path.
    Prefill {
        push: bool,
    },
    Decode,
    /// One worker with both roles (E+PD).
    PrefillDecode,
}

/// How a request ended, as the mock saw it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    /// The handler sent its whole reply.
    Completed,
    /// The router closed the connection first, so hyper dropped the handler
    /// or the body.
    Aborted,
}

/// A worker mock. Clones share one state, so the server task and the test
/// hold the same mock.
#[derive(Clone)]
pub struct Mock(Arc<MockState>);

pub struct MockState {
    name: &'static str,
    role: Role,
    mode: Mutex<Mode>,
    requests: Mutex<Vec<Value>>,
    ends: Mutex<Vec<End>>,
    arrived: Notify,
    ended: Notify,
    release: Semaphore,
}

impl Mock {
    pub fn new(name: &'static str, role: Role) -> Self {
        Self(Arc::new(MockState {
            name,
            role,
            mode: Mutex::new(Mode::Reply),
            requests: Mutex::default(),
            ends: Mutex::default(),
            arrived: Notify::new(),
            ended: Notify::new(),
            release: Semaphore::new(0),
        }))
    }

    pub fn name(&self) -> &'static str {
        self.0.name
    }

    /// Sets the mode for the requests that arrive from now on.
    pub fn set_mode(&self, mode: Mode) {
        *self.0.mode.lock().unwrap() = mode;
    }

    fn mode(&self) -> Mode {
        *self.0.mode.lock().unwrap()
    }

    /// Lets one held handler continue (`Hold*`, `Stall*`, `BreakBody`).
    /// A permit given before the request arrives is kept for it.
    pub fn release(&self) {
        self.0.release.add_permits(1);
    }

    async fn hold(&self) {
        self.0.release.acquire().await.unwrap().forget();
    }

    /// The request bodies seen so far, in arrival order.
    pub fn requests(&self) -> Vec<Value> {
        self.0.requests.lock().unwrap().clone()
    }

    fn ends(&self) -> Vec<End> {
        self.0.ends.lock().unwrap().clone()
    }

    /// Forgets the requests and ends seen so far (after the rig's warm-up).
    pub fn reset(&self) {
        self.0.requests.lock().unwrap().clear();
        self.0.ends.lock().unwrap().clear();
    }

    fn observation(&self) -> &'static str {
        if self.0.role == Role::Encoder {
            "encoder abort"
        } else {
            "socket close"
        }
    }

    /// Records a request and starts its `Life`.
    fn arrive(&self, request: Value) -> Life {
        self.0.requests.lock().unwrap().push(request);
        self.0.arrived.notify_one();
        Life {
            mock: self.clone(),
            done: false,
        }
    }

    /// Whether `count` requests arrive within `WAIT_DURATION`.
    pub async fn got_requests(&self, count: usize) -> bool {
        tokio::time::timeout(WAIT_DURATION, async {
            while self.requests().len() < count {
                self.0.arrived.notified().await;
            }
        })
        .await
        .is_ok()
    }

    /// `[dispatch]`: `count` requests arrive.
    pub async fn wait_requests(&self, count: usize) {
        assert!(
            self.got_requests(count).await,
            "[dispatch] {} got {} of {count} requests",
            self.0.name,
            self.requests().len()
        );
    }

    async fn first_end(&self) -> End {
        let waited = tokio::time::timeout(WAIT_DURATION, async {
            loop {
                if let Some(end) = self.ends().first() {
                    return *end;
                }
                self.0.ended.notified().await;
            }
        })
        .await;
        waited.unwrap_or_else(|_| {
            panic!(
                "[{}] {}: the request did not end",
                self.observation(),
                self.0.name
            )
        })
    }

    /// `[socket close]` / `[encoder abort]`: the first request ends because
    /// the router closed the connection.
    pub async fn expect_aborted(&self) {
        assert_eq!(
            self.first_end().await,
            End::Aborted,
            "[{}] {}: expected the router to close the connection",
            self.observation(),
            self.0.name
        );
    }

    /// `[socket close]` / `[encoder abort]`: the first request ends with its
    /// whole reply sent.
    pub async fn expect_completed(&self) {
        assert_eq!(
            self.first_end().await,
            End::Completed,
            "[{}] {}: expected the reply to complete",
            self.observation(),
            self.0.name
        );
    }

    /// `[dispatch]`: no request has arrived. A check of the state now, with
    /// no wait.
    pub fn expect_no_request(&self) {
        assert!(
            self.requests().is_empty(),
            "[dispatch] {} should not get a request: {:?}",
            self.0.name,
            self.requests()
        );
    }

    /// The reply body as (first part, rest). The stall modes send the first
    /// part at once and hold the rest.
    fn body(&self, request: &Value, mode: Mode) -> (String, String) {
        let whole = match (mode, self.0.role) {
            (Mode::StallBodyInvalidJson, _) => INVALID_JSON_BODY.to_owned(),
            (Mode::StallBodyWithoutKv, _) => json!({"choices": []}).to_string(),
            (_, Role::Encoder) => encoder_reply(request).to_string(),
            (_, Role::Prefill { push }) => prefill_reply(push).to_string(),
            (_, Role::Decode | Role::PrefillDecode) => json!({
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "A"}}],
            })
            .to_string(),
        };
        let rest = whole[1..].to_owned();
        (whole[..1].to_owned(), rest)
    }
}

/// One request's life in a mock. `finish` records `Completed`; a drop without
/// `finish` records `Aborted`. The handler keeps it while it works, and the
/// stall modes move it into the body stream.
pub struct Life {
    mock: Mock,
    done: bool,
}

impl Life {
    /// The whole reply is sent.
    fn finish(mut self) {
        self.done = true;
    }
}

impl Drop for Life {
    fn drop(&mut self) {
        let end = if self.done {
            End::Completed
        } else {
            End::Aborted
        };
        self.mock.0.ends.lock().unwrap().push(end);
        self.mock.0.ended.notify_one();
    }
}

/// One EC handle per content part, keyed by the part's `uuid`.
fn encoder_reply(request: &Value) -> Value {
    let mut handles = serde_json::Map::new();
    let parts = request["messages"][0]["content"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for (index, part) in parts.iter().enumerate() {
        let uuid = part["uuid"].as_str().unwrap_or_default();
        handles.insert(
            format!("engine-{uuid}"),
            json!({
                "metadata": {"image_grid_thw": [[1, 2, 3]]},
                "item_indices": [index],
                "peer_host": "encoder",
                "peer_port": 4321,
                "size_bytes": 128,
            }),
        );
    }
    json!({"ec_transfer_params": handles})
}

/// P's `kv_transfer_params`; with `push`, also P's NIXL address.
fn prefill_reply(push: bool) -> Value {
    let mut kv = json!({"remote_engine_id": "P", "do_remote_prefill": true});
    if push {
        kv["remote_host"] = json!("127.0.0.1");
        kv["remote_port"] = json!(9100);
        kv["transfer_mode"] = json!("push");
    }
    json!({"kv_transfer_params": kv})
}

/// The axum handler for `serve`.
async fn handle(State(mock): State<Mock>, Json(request): Json<Value>) -> Response {
    let life = mock.arrive(request.clone());
    let mode = mock.mode();
    let (head, rest) = mock.body(&request, mode);
    let headers = [(CONTENT_TYPE, "application/json")];
    match mode {
        Mode::Reply | Mode::HoldThenReply => {
            if mode == Mode::HoldThenReply {
                mock.hold().await;
            }
            life.finish();
            (headers, format!("{head}{rest}")).into_response()
        }
        Mode::StallBody
        | Mode::StallBodyInvalidJson
        | Mode::StallBodyWithoutKv
        | Mode::BreakBody => {
            // `life` moves into the body: if hyper drops the body before the
            // release, the request ends as `Aborted`.
            let rest = futures_util::stream::once(async move {
                mock.hold().await;
                life.finish();
                if mode == Mode::BreakBody {
                    Err(std::io::Error::other("injected hang-up in the body"))
                } else {
                    Ok(Bytes::from(rest))
                }
            });
            let body = futures_util::stream::once(async move { Ok(Bytes::from(head)) }).chain(rest);
            (headers, Body::from_stream(body)).into_response()
        }
    }
}

/// Serves the mock with axum on a free port: its URL and the server task.
pub async fn serve(mock: &Mock) -> (String, JoinHandle<()>) {
    let app = axum::Router::new()
        .route("/health", get(|| async { "ok" }))
        .route(CHAT_ROUTE, post(handle))
        .with_state(mock.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, server)
}

// Self-tests: the mock sees the ends that the rig's assertions depend on.

/// A client for the self-tests. Its timeout puts a limit on each of them.
fn plain_client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(WAIT_DURATION)
        .build()
        .unwrap()
}

#[tokio::test]
async fn mock_sees_the_peer_leave_before_the_status_line() {
    let mock = Mock::new("PD", Role::PrefillDecode);
    let (url, server) = serve(&mock).await;
    mock.set_mode(Mode::HoldThenReply);
    let request = tokio::spawn(
        plain_client()
            .post(format!("{url}{CHAT_ROUTE}"))
            .json(&json!({}))
            .send(),
    );
    mock.wait_requests(1).await;
    request.abort();
    mock.expect_aborted().await;
    server.abort();
}

#[tokio::test]
async fn mock_sees_the_peer_drop_the_body() {
    let mock = Mock::new("PD", Role::PrefillDecode);
    let (url, server) = serve(&mock).await;
    mock.set_mode(Mode::StallBody);
    let response = plain_client()
        .post(format!("{url}{CHAT_ROUTE}"))
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    let mut body = response.bytes_stream();
    assert_eq!(body.next().await.unwrap().unwrap(), "{");
    drop(body);
    mock.expect_aborted().await;
    server.abort();
}

#[tokio::test]
async fn released_reply_completes() {
    let mock = Mock::new("PD", Role::PrefillDecode);
    let (url, server) = serve(&mock).await;
    mock.set_mode(Mode::HoldThenReply);
    let request = tokio::spawn(
        plain_client()
            .post(format!("{url}{CHAT_ROUTE}"))
            .json(&json!({}))
            .send(),
    );
    mock.wait_requests(1).await;
    mock.release();
    assert!(request.await.unwrap().unwrap().status().is_success());
    mock.expect_completed().await;
    server.abort();
}
