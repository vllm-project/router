use crate::config::EpdConfig;
use axum::http::{HeaderMap, StatusCode};
use futures_util::future::join_all;
use reqwest::{Client, RequestBuilder};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use uuid::Uuid;

type EpdError = (StatusCode, String);

struct MediaItem {
    message_index: usize,
    content_index: usize,
    kind: &'static str,
    content_id: String,
    transfer_id: String,
    media: Value,
}

#[derive(Debug)]
pub(super) struct EncoderStage {
    config: EpdConfig,
    next_encoder: AtomicUsize,
}

/// Cancel abandoned Mooncake pushes when preparation or the P/PD request ends early.
pub(super) struct EcTransferGuard {
    control_addr: String,
    transfer_ids: Vec<String>,
}

impl EcTransferGuard {
    fn retain(&mut self, transfer_ids: Vec<String>) {
        let mut abandoned = Self {
            control_addr: self.control_addr.clone(),
            transfer_ids: std::mem::replace(&mut self.transfer_ids, transfer_ids),
        };
        abandoned
            .transfer_ids
            .retain(|id| !self.transfer_ids.contains(id));
    }

    pub(super) fn disarm(&mut self) {
        self.transfer_ids.clear();
    }
}

impl Drop for EcTransferGuard {
    fn drop(&mut self) {
        if self.transfer_ids.is_empty() {
            return;
        }
        let transfer_ids = std::mem::take(&mut self.transfer_ids);
        let control_addr = self.control_addr.clone();
        // No blocking ZMQ operations on the HTTP runtime threads.
        tokio::task::spawn_blocking(move || {
            let context = zmq::Context::new();
            for transfer_id in transfer_ids {
                let result = (|| -> Result<(), zmq::Error> {
                    let socket = context.socket(zmq::REQ)?;
                    socket.set_linger(0)?;
                    socket.set_sndtimeo(1000)?;
                    socket.set_rcvtimeo(1000)?;
                    socket.connect(&control_addr)?;
                    let body = json!({
                        "op": "cancel",
                        "transfer_id": transfer_id,
                        "abandon": true,
                    })
                    .to_string();
                    socket.send(body.as_bytes(), 0)?;
                    socket.recv_bytes(0)?;
                    Ok(())
                })();
                if let Err(error) = result {
                    tracing::warn!(%error, "Could not cancel abandoned EC transfer");
                }
            }
        });
    }
}

impl EncoderStage {
    pub(super) fn new(config: EpdConfig) -> Self {
        Self {
            config,
            next_encoder: AtomicUsize::new(0),
        }
    }

    pub(super) async fn prepare(
        &self,
        client: &Client,
        mut body: Value,
        consumer_url: &str,
        headers: Option<&HeaderMap>,
        api_key: Option<&str>,
    ) -> Result<(Value, Option<EcTransferGuard>), EpdError> {
        let started = Instant::now();
        let items = collect_media_items(&body)?;
        if items.is_empty() {
            return Ok((body, None));
        }
        let consumer_addr = self.config.consumer_zmq_addrs.get(consumer_url);
        if !self.config.consumer_zmq_addrs.is_empty() && consumer_addr.is_none() {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "Selected embedding consumer has no EC control address".into(),
            ));
        }
        let mut pending = consumer_addr.map(|control_addr| EcTransferGuard {
            control_addr: control_addr.clone(),
            transfer_ids: items.iter().map(|item| item.transfer_id.clone()).collect(),
        });
        let request_id = headers
            .and_then(|h| h.get("x-request-id"))
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let start = self.next_encoder.fetch_add(items.len(), Ordering::Relaxed);
        let replies = join_all(items.iter().enumerate().map(|(index, item)| {
            let encoder_url = &self.config.encoder_urls
                [start.wrapping_add(index) % self.config.encoder_urls.len()];
            let mut request = client
                .post(format!(
                    "{}/v1/chat/completions",
                    encoder_url.trim_end_matches('/')
                ))
                .header(
                    "x-request-id",
                    format!("{request_id}:{index}:{}", item.transfer_id),
                );
            if let Some(key) = api_key {
                request = request.bearer_auth(key);
            } else if let Some(auth) = headers.and_then(|h| h.get("authorization")) {
                request = request.header("authorization", auth);
            }
            encode_item(request, item, &body, consumer_addr.map(String::as_str))
        }))
        .await;
        let replies: Vec<Value> = replies.into_iter().collect::<Result<_, _>>()?;
        let mut handles = Map::new();
        let mut transfers = Vec::new();
        let mut used_transfer_ids = Vec::new();
        for (item, mut reply) in items.iter().zip(replies) {
            let Some(Value::Object(mut params)) =
                reply.get_mut("ec_transfer_params").map(Value::take)
            else {
                continue;
            };
            let entry = if let Some(reported) = params.remove(&item.content_id) {
                Some((item.content_id.clone(), reported))
            } else if params.len() == 1 {
                params.into_iter().next()
            } else {
                None
            };
            let Some((hash, reported)) = entry else {
                continue;
            };
            let Some(metadata) = reported
                .get("metadata")
                .and_then(Value::as_object)
                .filter(|m| !m.is_empty())
            else {
                continue;
            };
            let metadata = flatten_metadata(metadata)?;
            let kind = match item.kind {
                "image_url" => "image_embeds",
                "video_url" => "video_embeds",
                _ => "audio_embeds",
            };
            body["messages"][item.message_index]["content"][item.content_index] = json!({
                "type": kind,
                (kind): metadata,
                "uuid": item.content_id,
            });
            handles.insert(item.content_id.clone(), reported);
            transfers.push(json!({
                "mm_hash": hash,
                "transfer_id": item.transfer_id,
            }));
            used_transfer_ids.push(item.transfer_id.clone());
        }
        let rewritten = transfers.len();
        if rewritten > 0 {
            if let Some(Value::Object(mut existing)) =
                body.get_mut("ec_transfer_params").map(Value::take)
            {
                if let Some(Value::Array(mut previous)) = existing.remove("ec_items") {
                    previous.append(&mut transfers);
                    transfers = previous;
                }
                existing.extend(handles);
                handles = existing;
            }
            handles.insert("ec_items".into(), Value::Array(transfers));
            body["ec_transfer_params"] = Value::Object(handles);
        }
        if let Some(pending) = pending.as_mut() {
            pending.retain(used_transfer_ids);
        }
        tracing::info!(
            %request_id,
            items = items.len(),
            rewritten,
            encode_ms = started.elapsed().as_secs_f64() * 1000.0,
            "EPD encoder stage complete"
        );
        Ok((body, pending))
    }
}

fn collect_media_items(body: &Value) -> Result<Vec<MediaItem>, EpdError> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "messages must be an array".into()))?;
    let mut items = Vec::new();
    for (message_index, message) in messages.iter().enumerate() {
        let Some(content) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        for (content_index, item) in content.iter().enumerate() {
            let kind = match item.get("type").and_then(Value::as_str) {
                Some("image_url") => "image_url",
                Some("video_url") => "video_url",
                Some("audio_url") => "audio_url",
                _ => continue,
            };
            let url = item[kind]["url"]
                .as_str()
                .filter(|v| !v.is_empty())
                .ok_or_else(|| {
                    (
                        StatusCode::BAD_REQUEST,
                        format!("{kind}.url must be a nonempty string"),
                    )
                })?;
            let content_id = format!("{:x}", Sha256::digest(url.as_bytes()));
            let transfer_id = Uuid::new_v4().simple().to_string();
            let mut media = item.clone();
            media["uuid"] = json!(content_id);
            items.push(MediaItem {
                message_index,
                content_index,
                kind,
                content_id,
                transfer_id,
                media,
            });
        }
    }
    Ok(items)
}

async fn encode_item(
    request: RequestBuilder,
    item: &MediaItem,
    body: &Value,
    consumer_addr: Option<&str>,
) -> Result<Value, EpdError> {
    let mut payload = json!({
        "model": body.get("model"),
        "stream": false,
        "messages": [{"role": "user", "content": [item.media]}],
    });
    // Per-request preprocessing options must agree on both sides.
    for key in ["mm_processor_kwargs", "media_io_kwargs"] {
        if let Some(value) = body.get(key) {
            payload[key] = value.clone();
        }
    }
    if let Some(control_addr) = consumer_addr {
        payload["ec_transfer_params"] = json!({
            "consumer_zmq": control_addr,
            "ec_items": [{"transfer_id": item.transfer_id}],
        });
    }
    let request = request.json(&payload);
    drop(payload);
    let response = request.send().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Encoder request failed: {e}"),
        )
    })?;
    if !response.status().is_success() {
        return Err((response.status(), "Encoder request failed".into()));
    }
    response.json::<Value>().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Invalid encoder response: {e}"),
        )
    })
}

fn flatten_metadata(metadata: &Map<String, Value>) -> Result<Map<String, Value>, EpdError> {
    metadata
        .iter()
        .map(|(key, value)| {
            let values = value.as_array().ok_or_else(|| {
                (
                    StatusCode::BAD_GATEWAY,
                    "Encoder metadata must contain numeric arrays".into(),
                )
            })?;
            let mut flat = Vec::new();
            for value in values {
                match value {
                    Value::Array(values) => flat.extend(values.iter().cloned()),
                    value => flat.push(value.clone()),
                }
            }
            if !flat.iter().all(Value::is_number) {
                return Err((
                    StatusCode::BAD_GATEWAY,
                    "Encoder metadata must contain numeric arrays".into(),
                ));
            }
            Ok((key.clone(), Value::Array(flat)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{extract::State, routing::post, Json, Router};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct MockEncoder {
        mode: &'static str,
        seen: Arc<Mutex<Vec<Value>>>,
    }

    async fn encode(
        State(state): State<MockEncoder>,
        Json(body): Json<Value>,
    ) -> (StatusCode, Json<Value>) {
        let count = {
            let mut seen = state.seen.lock().unwrap();
            seen.push(body.clone());
            seen.len()
        };
        if state.mode == "fail" {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error":"unavailable"})),
            );
        }
        let uuid = body["messages"][0]["content"][0]["uuid"].as_str().unwrap();
        let metadata = if state.mode == "empty" || (state.mode == "mixed" && count == 1) {
            json!({})
        } else if state.mode == "invalid" {
            json!({"image_grid_thw": ["invalid"]})
        } else {
            json!({"image_grid_thw":[[1,2,3]]})
        };
        (
            StatusCode::OK,
            Json(json!({
                "ec_transfer_params": {
                    (format!("engine-{uuid}")): {
                        "metadata": metadata,
                        "peer_host": "encoder",
                        "peer_port": 4321,
                        "size_bytes": 128,
                    },
                },
            })),
        )
    }

    async fn mock(
        mode: &'static str,
    ) -> (
        EncoderStage,
        Arc<Mutex<Vec<Value>>>,
        tokio::task::JoinHandle<()>,
    ) {
        let state = MockEncoder {
            mode,
            seen: Arc::new(Mutex::new(Vec::new())),
        };
        let seen = state.seen.clone();
        let app = Router::new()
            .route("/v1/chat/completions", post(encode))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let stage = EncoderStage::new(EpdConfig {
            encoder_urls: vec![url],
            consumer_zmq_addrs: Default::default(),
        });
        (stage, seen, server)
    }

    fn request() -> Value {
        json!({
            "model": "test",
            "stream": true,
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": false},
            "mm_processor_kwargs": {"min_pixels": 64},
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    {"type": "text", "text": "compare"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }],
        })
    }

    #[tokio::test]
    async fn duplicate_images_keep_positions_and_distinct_transfers() {
        let (stage, seen, server) = mock("normal").await;
        let mut input = request();
        input["ec_transfer_params"] = json!({
            "opaque": {"backend_field": 42},
            "ec_items": [{"mm_hash": "existing", "transfer_id": "existing-transfer"}],
        });
        let (body, pending) = stage
            .prepare(&Client::new(), input.clone(), "pd", None, None)
            .await
            .unwrap();
        assert!(pending.is_none());
        let content = &body["messages"][0]["content"];
        assert_eq!(
            content[0]["image_embeds"]["image_grid_thw"],
            json!([1, 2, 3])
        );
        assert_eq!(content[1], input["messages"][0]["content"][1]);
        assert_eq!(content[0]["uuid"], content[2]["uuid"]);
        let transfers = &body["ec_transfer_params"]["ec_items"];
        assert_eq!(transfers.as_array().unwrap().len(), 3);
        assert_eq!(transfers[0], input["ec_transfer_params"]["ec_items"][0]);
        assert_eq!(
            body["ec_transfer_params"]["opaque"],
            input["ec_transfer_params"]["opaque"]
        );
        assert_eq!(transfers[1]["mm_hash"], transfers[2]["mm_hash"]);
        assert_ne!(transfers[1]["transfer_id"], transfers[2]["transfer_id"]);
        assert_eq!(
            transfers[1]["mm_hash"],
            format!("engine-{}", content[0]["uuid"].as_str().unwrap())
        );
        assert_eq!(
            body["ec_transfer_params"][content[0]["uuid"].as_str().unwrap()]["peer_port"],
            4321
        );
        assert_eq!(body["stream"], true);
        assert_eq!(body["chat_template_kwargs"], input["chat_template_kwargs"]);
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests
            .iter()
            .all(|r| r["stream"] == false && r.get("max_tokens").is_none()));
        assert_eq!(
            requests[0]["mm_processor_kwargs"],
            input["mm_processor_kwargs"]
        );
        server.abort();
    }

    #[tokio::test]
    async fn missing_metadata_preserves_original_media() {
        let (stage, _, server) = mock("empty").await;
        let mut input = request();
        input["ec_transfer_params"] = json!({"opaque": {"backend_field": 42}});
        let (body, _) = stage
            .prepare(&Client::new(), input.clone(), "pd", None, None)
            .await
            .unwrap();
        assert_eq!(body, input);
        server.abort();
    }

    #[tokio::test]
    async fn encoder_failure_does_not_forward_metadata_only_input() {
        let (stage, _, server) = mock("fail").await;
        let error = stage
            .prepare(&Client::new(), request(), "pd", None, None)
            .await
            .err()
            .unwrap();
        assert_eq!(error.0, StatusCode::SERVICE_UNAVAILABLE);
        server.abort();
    }

    #[tokio::test]
    async fn existing_embeds_bypass_encoders() {
        let (stage, seen, server) = mock("normal").await;
        let input = json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_embeds",
                    "image_embeds": {"image_grid_thw": [1, 2, 3]},
                    "uuid": "known",
                }],
            }],
        });
        let (body, _) = stage
            .prepare(&Client::new(), input.clone(), "pd", None, None)
            .await
            .unwrap();
        assert_eq!(input, body);
        assert!(seen.lock().unwrap().is_empty());
        server.abort();
    }

    #[tokio::test]
    async fn abandoned_pushes_cancel_the_selected_consumer() {
        for mode in ["normal", "empty", "mixed", "invalid", "fail"] {
            let (mut stage, seen, server) = mock(mode).await;
            let (send_address, recv_address) = tokio::sync::oneshot::channel();
            let control = tokio::task::spawn_blocking(move || {
                let context = zmq::Context::new();
                let socket = context.socket(zmq::REP).unwrap();
                socket.set_rcvtimeo(5000).unwrap();
                socket.set_linger(0).unwrap();
                socket.bind("tcp://127.0.0.1:*").unwrap();
                send_address
                    .send(socket.get_last_endpoint().unwrap().unwrap())
                    .unwrap();
                let mut cancelled = Vec::new();
                let expected = if mode == "mixed" { 1 } else { 2 };
                for _ in 0..expected {
                    let message: Value =
                        serde_json::from_slice(&socket.recv_bytes(0).unwrap()).unwrap();
                    cancelled.push(message);
                    socket.send(b"{\"ok\":true}".as_slice(), 0).unwrap();
                }
                cancelled
            });
            let address = recv_address.await.unwrap();
            stage
                .config
                .consumer_zmq_addrs
                .insert("selected-pd".into(), address.clone());
            let result = stage
                .prepare(&Client::new(), request(), "selected-pd", None, None)
                .await;
            let ids: Vec<Value> = seen
                .lock()
                .unwrap()
                .iter()
                .map(|r| {
                    assert_eq!(r["ec_transfer_params"]["consumer_zmq"], address);
                    r["ec_transfer_params"]["ec_items"][0]["transfer_id"].clone()
                })
                .collect();
            if matches!(mode, "invalid" | "fail") {
                assert!(result.is_err());
            } else {
                let (_, mut pending) = result.unwrap();
                if mode == "mixed" {
                    let pending = pending.as_mut().unwrap();
                    assert_eq!(pending.transfer_ids.len(), 1);
                    pending.disarm();
                }
                drop(pending);
            }
            let cancelled = control.await.unwrap();
            assert!(cancelled.iter().all(|r| r["op"] == "cancel"
                && r["abandon"] == true
                && ids.contains(&r["transfer_id"])));
            if mode == "mixed" {
                assert_eq!(cancelled[0]["transfer_id"], ids[0]);
            } else {
                assert_ne!(cancelled[0]["transfer_id"], cancelled[1]["transfer_id"]);
            }
            server.abort();
        }
    }
}
