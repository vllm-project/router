//! Mock engine worker that publishes real KV events.
//!
//! Each [`KvEventMock`] is an axum HTTP server plus a ZMQ publisher, shaped like
//! the control surface the router's KV discovery expects from a vLLM worker:
//!
//! * `GET /health` -> 200
//! * `GET /get_server_info` -> `{model_id, instance_id, kv_block_size}`
//! * `GET /server_info?config_format=json` -> native vLLM shape (model, block size)
//! * `GET /kv_event_sources` -> one `zmq`/`msgpack`/`schema_version: 1` entry for the DP rank,
//!   with an optional `peer_lookup` object ([`KvEventMockConfig::peer_lookup`])
//! * `POST /v1/completions` and `POST /v1/chat/completions` -> OpenAI-style replies
//!
//! The ZMQ socket is an XPUB so the mock can observe the router's subscription
//! (the standard cure for the slow-joiner race: wait for the subscribe frame,
//! then publish). Events are multipart `[topic, seq (i64 big-endian), msgpack
//! [ts, [events], dp_rank]]`, built from the router's own wire types so a wire
//! change breaks this mock instead of silently diverging from it.
//!
//! Requests are served after an optional delay. A request whose JSON body carries
//! `"priority": HOLD_PRIORITY` blocks until [`KvEventMock::release_held`] (or
//! drop), which keeps the router's per-worker `active_requests` /
//! `active_prefill_tokens` counters raised for as long as a test needs.
//!
//! The marker is `priority` (a real completions field the router models) rather
//! than an invented one: the router withholds KV placement credit from requests
//! carrying fields it does not understand, which would change the routing under
//! test.

use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;
use tokio::sync::{oneshot, watch};
use vllm_router_rs::kv_index::wire::{
    AllBlocksCleared, BlockRemoved, BlockStored, ExternalBlockHash, KVEvent,
};

/// `priority` value that makes a completion request wait for `release_held`.
pub const HOLD_PRIORITY: i64 = 424_242;

/// Where a stored prefix lives, as the router's index will classify it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// `medium: GPU` -> `StorageTier::Device`.
    Device,
    /// `medium: CPU` -> `StorageTier::HostPinned`.
    HostPinned,
    /// `medium: STORAGE` -> `StorageTier::Disk`.
    Disk,
    /// A shared-pool copy: `ownership` set -> `StorageTier::External`.
    External,
}

impl Residency {
    fn medium(self) -> &'static str {
        match self {
            Residency::Device | Residency::External => "GPU",
            Residency::HostPinned => "CPU",
            Residency::Disk => "STORAGE",
        }
    }

    fn locality(self) -> &'static str {
        match self {
            Residency::External => "REMOTE",
            _ => "LOCAL",
        }
    }

    fn ownership(self) -> Option<String> {
        matches!(self, Residency::External).then(|| "pool".to_owned())
    }
}

/// What the mock reports as the rank's `peer_lookup` record.
#[derive(Debug, Clone)]
pub struct PeerLookupMock {
    pub layout_fingerprint: String,
    pub enabled: bool,
    /// Report `side_channel_port: 0`, which the router must drop.
    pub invalid: bool,
}

impl PeerLookupMock {
    pub fn new(layout_fingerprint: &str) -> Self {
        Self {
            layout_fingerprint: layout_fingerprint.to_owned(),
            enabled: true,
            invalid: false,
        }
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    pub fn invalid(mut self) -> Self {
        self.invalid = true;
        self
    }
}

/// Side-channel address the mock advertises for `dp_rank`.
pub const PEER_HOST: &str = "127.0.0.1";
pub fn peer_port(dp_rank: u32) -> u16 {
    5600 + dp_rank as u16
}

/// What [`KvEventMock::store_prefix`] published, so it can be evicted later.
#[derive(Debug, Clone)]
pub struct StoredPrefix {
    pub residency: Residency,
    pub hashes: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct KvEventMockConfig {
    pub model: String,
    pub instance_id: String,
    pub block_size: u32,
    pub dp_rank: u32,
    pub topic: String,
    /// Added to every non-held completion.
    pub response_delay: Duration,
    /// When set, `/kv_event_sources` carries a `peer_lookup` object for the rank.
    pub peer_lookup: Option<PeerLookupMock>,
}

impl KvEventMockConfig {
    pub fn new(instance_id: &str, model: &str, block_size: u32) -> Self {
        Self {
            model: model.to_owned(),
            instance_id: instance_id.to_owned(),
            block_size,
            dp_rank: 0,
            topic: "kv-events".to_owned(),
            response_delay: Duration::ZERO,
            peer_lookup: None,
        }
    }
}

struct MockState {
    config: KvEventMockConfig,
    endpoint: String,
    /// `false` while held requests must wait; `true` releases them.
    release: watch::Sender<bool>,
    held_inflight: AtomicUsize,
    completions: AtomicUsize,
    fail_next: AtomicUsize,
}

enum Command {
    Publish {
        events: Vec<KVEvent>,
        done: oneshot::Sender<()>,
    },
    /// Consume a sequence number without publishing it (simulates a lost batch).
    SkipSeq {
        done: oneshot::Sender<()>,
    },
    Shutdown,
}

pub struct KvEventMock {
    pub url: String,
    state: Arc<MockState>,
    commands: mpsc::Sender<Command>,
    subscribed: Arc<AtomicBool>,
    next_hash: AtomicU64,
    prefixes: Mutex<Vec<StoredPrefix>>,
    server: tokio::task::JoinHandle<()>,
    publisher: Option<std::thread::JoinHandle<()>>,
}

static HASH_BASE: AtomicU64 = AtomicU64::new(1);

impl KvEventMock {
    pub async fn start(config: KvEventMockConfig) -> Self {
        let context = zmq::Context::new();
        let socket = context.socket(zmq::XPUB).expect("xpub socket");
        // Report subscribe and unsubscribe frames, including duplicates.
        socket.set_xpub_verbose(true).expect("xpub verbose");
        socket.set_linger(0).expect("linger");
        socket.bind("tcp://127.0.0.1:*").expect("bind xpub");
        let endpoint = socket
            .get_last_endpoint()
            .expect("endpoint")
            .expect("utf8 endpoint");

        let (release, _) = watch::channel(false);
        let state = Arc::new(MockState {
            config: config.clone(),
            endpoint,
            release,
            held_inflight: AtomicUsize::new(0),
            completions: AtomicUsize::new(0),
            fail_next: AtomicUsize::new(0),
        });

        let (commands, command_rx) = mpsc::channel();
        let subscribed = Arc::new(AtomicBool::new(false));
        let publisher = {
            let (topic, rank, subscribed) =
                (config.topic.clone(), config.dp_rank, subscribed.clone());
            std::thread::spawn(move || {
                run_publisher(context, socket, command_rx, topic, rank, subscribed)
            })
        };

        let app = Router::new()
            .route("/health", get(|| async { StatusCode::OK }))
            .route("/get_server_info", get(get_server_info))
            .route("/server_info", get(native_server_info))
            .route("/kv_event_sources", get(kv_event_sources))
            .route("/v1/completions", post(completions))
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock http");
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        Self {
            url,
            state,
            commands,
            subscribed,
            next_hash: AtomicU64::new(HASH_BASE.fetch_add(1, Ordering::Relaxed) << 32),
            prefixes: Mutex::new(Vec::new()),
            server,
            publisher: Some(publisher),
        }
    }

    pub fn config(&self) -> &KvEventMockConfig {
        &self.state.config
    }

    pub fn block_size(&self) -> u32 {
        self.state.config.block_size
    }

    /// Wait until the router's subscriber has attached to this publisher.
    /// Publishing earlier would lose the batch (slow joiner).
    pub async fn wait_subscribed(&self) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while !self.subscribed.load(Ordering::Acquire) {
            assert!(
                tokio::time::Instant::now() < deadline,
                "router never subscribed to {}",
                self.state.endpoint
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    async fn send(&self, events: Vec<KVEvent>) {
        let (done, wait) = oneshot::channel();
        self.commands
            .send(Command::Publish { events, done })
            .expect("publisher thread alive");
        wait.await.expect("publisher acked");
    }

    /// Publish a `BlockStored` for the first `blocks` blocks of `tokens`.
    pub async fn store_prefix(
        &self,
        tokens: &[u32],
        blocks: usize,
        residency: Residency,
    ) -> StoredPrefix {
        let block_size = self.block_size() as usize;
        assert!(
            tokens.len() >= blocks * block_size,
            "need {} tokens for {blocks} blocks, got {}",
            blocks * block_size,
            tokens.len()
        );
        let hashes: Vec<u64> = (0..blocks)
            .map(|_| self.next_hash.fetch_add(1, Ordering::Relaxed))
            .collect();
        self.send(vec![KVEvent::BlockStored(BlockStored {
            extra_keys: None,
            block_hashes: hashes.iter().copied().map(ExternalBlockHash::Int).collect(),
            parent_block_hash: None,
            token_ids: tokens[..blocks * block_size].to_vec(),
            block_size: self.block_size(),
            lora_id: None,
            medium: Some(residency.medium().into()),
            lora_name: None,
            group_idx: Some(0),
            kv_cache_spec_kind: Some("full_attention".into()),
            kv_cache_spec_sliding_window: None,
            locality: Some(residency.locality().into()),
            ownership: residency.ownership(),
            session_id: None,
        })])
        .await;
        let stored = StoredPrefix { residency, hashes };
        self.prefixes.lock().unwrap().push(stored.clone());
        stored
    }

    /// Publish a `BlockRemoved` for everything `stored` published.
    pub async fn evict(&self, stored: &StoredPrefix) {
        self.send(vec![KVEvent::BlockRemoved(BlockRemoved {
            block_hashes: stored
                .hashes
                .iter()
                .copied()
                .map(ExternalBlockHash::Int)
                .collect(),
            medium: Some(stored.residency.medium().into()),
            group_idx: Some(0),
            locality: Some(stored.residency.locality().into()),
            ownership: stored.residency.ownership(),
        })])
        .await;
        let hashes = &stored.hashes;
        self.prefixes
            .lock()
            .unwrap()
            .retain(|prefix| prefix.hashes != *hashes);
    }

    /// Publish `AllBlocksCleared`.
    pub async fn clear_all(&self) {
        self.send(vec![KVEvent::AllBlocksCleared(AllBlocksCleared {})])
            .await;
        self.prefixes.lock().unwrap().clear();
    }

    /// Lose one batch: the next published batch arrives with a sequence gap.
    pub async fn skip_seq(&self) {
        let (done, wait) = oneshot::channel();
        self.commands
            .send(Command::SkipSeq { done })
            .expect("publisher thread alive");
        wait.await.expect("publisher acked");
    }

    /// Number of completion requests that have reached this worker (held or not).
    pub fn completions(&self) -> usize {
        self.state.completions.load(Ordering::Acquire)
    }

    /// Held requests currently open on this worker.
    pub fn held(&self) -> usize {
        self.state.held_inflight.load(Ordering::Acquire)
    }

    /// Wait until exactly `count` held requests are open.
    pub async fn wait_held(&self, count: usize) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while self.held() != count {
            assert!(
                tokio::time::Instant::now() < deadline,
                "{}: expected {count} held requests, have {}",
                self.state.config.instance_id,
                self.held()
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Answer the next `count` non-held completions with HTTP 500.
    pub fn fail_next(&self, count: usize) {
        self.state.fail_next.store(count, Ordering::Release);
    }

    /// Let every held request finish.
    pub fn release_held(&self) {
        self.state.release.send_replace(true);
    }
}

impl Drop for KvEventMock {
    fn drop(&mut self) {
        self.state.release.send_replace(true);
        let _ = self.commands.send(Command::Shutdown);
        if let Some(thread) = self.publisher.take() {
            let _ = thread.join();
        }
        self.server.abort();
    }
}

fn run_publisher(
    _context: zmq::Context,
    socket: zmq::Socket,
    commands: mpsc::Receiver<Command>,
    topic: String,
    dp_rank: u32,
    subscribed: Arc<AtomicBool>,
) {
    let subscribe_frame = [b"\x01".as_slice(), topic.as_bytes()].concat();
    let mut seq: i64 = 0;
    loop {
        while let Ok(frame) = socket.recv_bytes(zmq::DONTWAIT) {
            if frame == subscribe_frame {
                subscribed.store(true, Ordering::Release);
            }
        }
        match commands.recv_timeout(Duration::from_millis(2)) {
            Ok(Command::Publish { events, done }) => {
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|elapsed| elapsed.as_secs_f64())
                    .unwrap_or(0.0);
                let payload =
                    rmp_serde::to_vec_named(&(ts, events, Some(dp_rank))).expect("encode batch");
                socket
                    .send_multipart(
                        [
                            topic.as_bytes(),
                            seq.to_be_bytes().as_slice(),
                            payload.as_slice(),
                        ],
                        0,
                    )
                    .expect("publish batch");
                seq += 1;
                let _ = done.send(());
            }
            Ok(Command::SkipSeq { done }) => {
                seq += 1;
                let _ = done.send(());
            }
            Ok(Command::Shutdown) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
    }
}

async fn get_server_info(State(state): State<Arc<MockState>>) -> Json<Value> {
    Json(json!({
        "model_id": state.config.model,
        "instance_id": state.config.instance_id,
        "kv_block_size": state.config.block_size,
    }))
}

async fn native_server_info(State(state): State<Arc<MockState>>) -> Json<Value> {
    Json(json!({"vllm_config": {
        "model_config": {"model": state.config.model},
        "cache_config": {"block_size": state.config.block_size},
    }}))
}

async fn kv_event_sources(State(state): State<Arc<MockState>>) -> Json<Value> {
    let mut entry = json!({
        "endpoint": state.endpoint,
        "topic": state.config.topic,
        "publisher": "zmq",
        "encoding": "msgpack",
        "schema_version": 1,
        "enable_kv_cache_events": true,
    });
    if let Some(peer) = &state.config.peer_lookup {
        let rank = state.config.dp_rank;
        entry["peer_lookup"] = json!({
            "enabled": peer.enabled,
            "engine_id": format!("{}-rank{rank}", state.config.instance_id),
            "side_channel_host": PEER_HOST,
            "side_channel_port": if peer.invalid { 0 } else { u32::from(peer_port(rank)) },
            "tp_size": 1, "dcp_size": 1, "pp_size": 1,
            "connector": "nixl",
            "connector_version": "1",
            "layout_fingerprint": peer.layout_fingerprint,
        });
    }
    Json(json!({ state.config.dp_rank.to_string(): entry }))
}

/// Shared by both completion routes. Returns the failure response when the
/// worker was told to fail, otherwise waits (hold or delay) and returns `None`.
async fn admit(state: &MockState, body: &Bytes) -> Option<Response> {
    state.completions.fetch_add(1, Ordering::AcqRel);
    let held = serde_json::from_slice::<Value>(body)
        .ok()
        .is_some_and(|value| value["priority"].as_i64() == Some(HOLD_PRIORITY));
    if held {
        state.held_inflight.fetch_add(1, Ordering::AcqRel);
        let mut release = state.release.subscribe();
        let _ = release.wait_for(|released| *released).await;
        state.held_inflight.fetch_sub(1, Ordering::AcqRel);
        return None;
    }
    let failing = state
        .fail_next
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |left| {
            left.checked_sub(1)
        })
        .is_ok();
    if failing {
        return Some((StatusCode::INTERNAL_SERVER_ERROR, "mock failure").into_response());
    }
    if !state.config.response_delay.is_zero() {
        tokio::time::sleep(state.config.response_delay).await;
    }
    None
}

async fn completions(State(state): State<Arc<MockState>>, body: Bytes) -> Response {
    if let Some(failure) = admit(&state, &body).await {
        return failure;
    }
    Json(json!({
        "id": "cmpl-mock",
        "object": "text_completion",
        "created": 0,
        "model": state.config.model,
        "choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }))
    .into_response()
}

async fn chat_completions(State(state): State<Arc<MockState>>, body: Bytes) -> Response {
    if let Some(failure) = admit(&state, &body).await {
        return failure;
    }
    Json(json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 0,
        "model": state.config.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }))
    .into_response()
}
