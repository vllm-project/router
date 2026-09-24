use crate::config::EpdConfig;
use axum::http::{HeaderMap, StatusCode};
use futures_util::future::join_all;
use reqwest::{Client, RequestBuilder};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
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
            if let Err(error) = cancel_ec_transfers(&control_addr, &transfer_ids) {
                tracing::warn!(%error, %control_addr, "Could not discover EC consumer shards for cancellation");
            }
        });
    }
}

fn ec_control_request(
    context: &zmq::Context,
    address: &str,
    request: &Value,
) -> anyhow::Result<Value> {
    let socket = context.socket(zmq::REQ)?;
    socket.set_linger(0)?;
    socket.set_sndtimeo(1000)?;
    socket.set_rcvtimeo(1000)?;
    socket.connect(address)?;
    socket.send(serde_json::to_vec(request)?.as_slice(), 0)?;
    let mut response: Value = serde_json::from_slice(&socket.recv_bytes(0)?)?;
    anyhow::ensure!(
        response["ok"] == true,
        "EC control request failed: {}",
        response["error"]
    );
    Ok(response["result"].take())
}

fn cancel_ec_transfers(control_addr: &str, transfer_ids: &[String]) -> anyhow::Result<()> {
    let context = zmq::Context::new();
    let peers = ec_control_request(&context, control_addr, &json!({"op": "peers"}))?;
    let ports: Vec<u16> = serde_json::from_value(peers["ports"].clone())?;
    anyhow::ensure!(
        !ports.is_empty() && !ports.contains(&0),
        "Invalid EC consumer shard ports"
    );
    let (prefix, _) = control_addr
        .rsplit_once(':')
        .ok_or_else(|| anyhow::anyhow!("Invalid EC control address: {control_addr}"))?;
    for port in ports {
        let address = format!("{prefix}:{port}");
        for transfer_id in transfer_ids {
            let request = json!({"op": "cancel", "transfer_id": transfer_id, "abandon": true});
            if let Err(error) = ec_control_request(&context, &address, &request) {
                tracing::warn!(%error, %address, %transfer_id, "Could not cancel abandoned EC transfer");
            }
        }
    }
    Ok(())
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
        // Round-robin per item, then batch the images of one request that land
        // on the same encoder into a single call, like the Python EPD proxy.
        // Audio and video items stay singleton groups.
        let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut image_group_of: HashMap<usize, usize> = HashMap::new();
        for (index, item) in items.iter().enumerate() {
            let encoder = start.wrapping_add(index) % self.config.encoder_urls.len();
            if item.kind == "image_url" {
                match image_group_of.get(&encoder) {
                    Some(&group) => groups[group].1.push(index),
                    None => {
                        image_group_of.insert(encoder, groups.len());
                        groups.push((encoder, vec![index]));
                    }
                }
            } else {
                groups.push((encoder, vec![index]));
            }
        }
        let replies = join_all(groups.iter().map(|(encoder, indices)| {
            let encoder_url = &self.config.encoder_urls[*encoder];
            let mut request = client
                .post(format!(
                    "{}/v1/chat/completions",
                    encoder_url.trim_end_matches('/')
                ))
                .header(
                    "x-request-id",
                    format!(
                        "{request_id}:{}:{:.6}",
                        indices[0], items[indices[0]].transfer_id
                    ),
                );
            if let Some(key) = api_key {
                request = request.bearer_auth(key);
            } else if let Some(auth) = headers.and_then(|h| h.get("authorization")) {
                request = request.header("authorization", auth);
            }
            encode_group(
                request,
                &items,
                indices,
                &body,
                consumer_addr.map(String::as_str),
            )
        }))
        .await;
        let replies: Vec<Value> = replies.into_iter().collect::<Result<_, _>>()?;
        let mut handles = Map::new();
        let mut transfers = Vec::new();
        let mut used_transfer_ids = Vec::new();
        let mut rewritten = 0;
        for ((_, indices), mut reply) in groups.iter().zip(replies) {
            let Some(Value::Object(params)) = reply.get_mut("ec_transfer_params").map(Value::take)
            else {
                // No EC report at all: keep the raw media but pin its uuid so
                // the consumer derives the same mm_hash the encoder used.
                for &index in indices {
                    let item = &items[index];
                    body["messages"][item.message_index]["content"][item.content_index] =
                        item.media.clone();
                }
                continue;
            };
            // Position is the reliable correlation when the engine rehashes
            // uuids with processing options; the uuid key and the single-entry
            // shape are fallbacks for encoders without position metadata.
            let mut by_position: HashMap<usize, (&String, &Value)> = HashMap::new();
            if indices.len() > 1 {
                for (mm_hash, reported) in params.iter() {
                    if let Some(item_indices) =
                        reported.get("item_indices").and_then(Value::as_array)
                    {
                        for local in item_indices {
                            let local = local
                                .as_u64()
                                .map(|v| v as usize)
                                .filter(|&v| v < indices.len())
                                .ok_or_else(|| {
                                    (StatusCode::BAD_GATEWAY, "Invalid encoder item index".into())
                                })?;
                            if by_position.insert(local, (mm_hash, reported)).is_some() {
                                return Err((
                                    StatusCode::BAD_GATEWAY,
                                    "Duplicate encoder item index".into(),
                                ));
                            }
                        }
                    }
                }
            }
            for (local, &index) in indices.iter().enumerate() {
                let item = &items[index];
                let entry = by_position
                    .get(&local)
                    .map(|(hash, reported)| ((*hash).clone(), (*reported).clone()))
                    .or_else(|| {
                        params
                            .get_key_value(&item.content_id)
                            .map(|(hash, reported)| (hash.clone(), reported.clone()))
                    })
                    .or_else(|| {
                        if indices.len() == 1 && params.len() == 1 {
                            params
                                .iter()
                                .next()
                                .map(|(hash, reported)| (hash.clone(), reported.clone()))
                        } else {
                            None
                        }
                    });
                let Some((hash, reported)) = entry else {
                    // Unreported item (e.g. an encoder-side processor cache
                    // hit): keep the raw media with its uuid pinned.
                    body["messages"][item.message_index]["content"][item.content_index] =
                        item.media.clone();
                    continue;
                };
                if let Some(metadata) = reported
                    .get("metadata")
                    .and_then(Value::as_object)
                    .filter(|m| !m.is_empty())
                {
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
                    transfers.push(json!({
                        "mm_hash": hash,
                        "transfer_id": item.transfer_id,
                    }));
                    used_transfer_ids.push(item.transfer_id.clone());
                    rewritten += 1;
                } else {
                    // The encoder reported no placeholder metadata (e.g. a
                    // processor cache hit): keep the raw media with its uuid
                    // pinned and hand the transfer to the consumer, which can
                    // then reuse the published embedding after re-preprocessing
                    // instead of abandoning the transfer.
                    body["messages"][item.message_index]["content"][item.content_index] =
                        item.media.clone();
                    transfers.push(json!({
                        "mm_hash": hash,
                        "transfer_id": item.transfer_id,
                    }));
                    used_transfer_ids.push(item.transfer_id.clone());
                }
                handles.insert(hash, reported);
            }
        }
        if !handles.is_empty() {
            if let Some(Value::Object(mut existing)) =
                body.get_mut("ec_transfer_params").map(Value::take)
            {
                if !transfers.is_empty() {
                    if let Some(Value::Array(mut previous)) = existing.remove("ec_items") {
                        previous.append(&mut transfers);
                        transfers = previous;
                    }
                }
                existing.extend(handles);
                handles = existing;
            }
            if !transfers.is_empty() {
                handles.insert("ec_items".into(), Value::Array(transfers));
            }
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
            // A caller-supplied uuid wins (matches the reference EPD proxy):
            // it pins the EC cache identity, e.g. for retrying against an
            // already published embedding. Hash the URL only as a fallback.
            let content_id = item
                .get("uuid")
                .and_then(Value::as_str)
                .filter(|v| !v.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| format!("{:x}", Sha256::digest(url.as_bytes())));
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

async fn encode_group(
    request: RequestBuilder,
    items: &[MediaItem],
    indices: &[usize],
    body: &Value,
    consumer_addr: Option<&str>,
) -> Result<Value, EpdError> {
    let mut payload = json!({
        "model": body.get("model"),
        "stream": false,
        "messages": [{
            "role": "user",
            "content": indices.iter().map(|&i| items[i].media.clone()).collect::<Vec<_>>(),
        }],
    });
    // Per-request preprocessing options must agree on both sides.
    for key in [
        "mm_processor_kwargs",
        "media_io_kwargs",
        "priority",
        "session_id",
    ] {
        if let Some(value) = body.get(key) {
            payload[key] = value.clone();
        }
    }
    if let Some(control_addr) = consumer_addr {
        payload["ec_transfer_params"] = json!({
            "consumer_zmq": control_addr,
            "ec_items": indices
                .iter()
                .map(|&i| json!({"transfer_id": items[i].transfer_id}))
                .collect::<Vec<_>>(),
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
        {
            let mut seen = state.seen.lock().unwrap();
            seen.push(body.clone());
        }
        if state.mode == "fail" {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error":"unavailable"})),
            );
        }
        // Batched requests carry several media parts; dedupe identical media
        // into one entry and report positions via item_indices, like vLLM.
        let content = body["messages"][0]["content"].as_array().unwrap();
        let mut positions: Vec<(String, Vec<usize>)> = Vec::new();
        for (index, part) in content.iter().enumerate() {
            let uuid = part["uuid"].as_str().unwrap().to_owned();
            match positions.iter_mut().find(|(u, _)| *u == uuid) {
                Some((_, indices)) => indices.push(index),
                None => positions.push((uuid, vec![index])),
            }
        }
        let mut params = Map::new();
        for (uuid, mut indices) in positions {
            if state.mode == "mixed" {
                // Report only the later item of a batch, leaving the first
                // unreported (e.g. an encoder-side processor cache hit).
                indices = indices.into_iter().skip(1).collect();
            }
            if indices.is_empty() {
                continue;
            }
            let metadata = match state.mode {
                "empty" => json!({}),
                "invalid" => json!({"image_grid_thw": ["invalid"]}),
                _ => json!({"image_grid_thw": [[1, 2, 3]]}),
            };
            params.insert(
                format!("engine-{uuid}"),
                json!({
                    "metadata": metadata,
                    "item_indices": indices,
                    "peer_host": "encoder",
                    "peer_port": 4321,
                    "size_bytes": 128,
                }),
            );
        }
        (StatusCode::OK, Json(json!({"ec_transfer_params": params})))
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
            body["ec_transfer_params"][transfers[1]["mm_hash"].as_str().unwrap()]["peer_port"],
            4321
        );
        assert!(body["ec_transfer_params"]
            .get(content[0]["uuid"].as_str().unwrap())
            .is_none());
        assert_eq!(body["stream"], true);
        assert_eq!(body["chat_template_kwargs"], input["chat_template_kwargs"]);
        let requests = seen.lock().unwrap();
        // Both images round-robin onto the single encoder and are batched
        // into one request.
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert!(request["stream"] == false && request.get("max_tokens").is_none());
        assert_eq!(request["mm_processor_kwargs"], input["mm_processor_kwargs"]);
        let sent = request["messages"][0]["content"].as_array().unwrap();
        assert_eq!(sent.len(), 2);
        assert!(sent.iter().all(|part| part["uuid"].is_string()));
        server.abort();
    }

    #[tokio::test]
    async fn missing_metadata_keeps_raw_media_with_uuid_and_forwards_transfers() {
        let (stage, seen, server) = mock("empty").await;
        let mut input = request();
        input["ec_transfer_params"] = json!({
            "opaque": {"backend_field": 42},
            "ec_items": [{"mm_hash": "existing", "transfer_id": "existing-transfer"}],
        });
        let (body, _) = stage
            .prepare(&Client::new(), input.clone(), "pd", None, None)
            .await
            .unwrap();
        // Raw media is kept with the content-derived uuid pinned, so the
        // consumer derives the same mm_hash and can reuse the embedding.
        let seen = seen.lock().unwrap();
        let uuid = seen[0]["messages"][0]["content"][0]["uuid"]
            .as_str()
            .unwrap()
            .to_owned();
        let content = &body["messages"][0]["content"];
        for index in [0, 2] {
            let mut expected = input["messages"][0]["content"][index].clone();
            expected["uuid"] = json!(uuid);
            assert_eq!(content[index], expected);
        }
        assert_eq!(content[1], input["messages"][0]["content"][1]);
        assert_eq!(
            body["ec_transfer_params"]["opaque"],
            input["ec_transfer_params"]["opaque"]
        );
        // Raw-fallback transfers are forwarded to the consumer, not abandoned.
        let transfers = body["ec_transfer_params"]["ec_items"].as_array().unwrap();
        assert_eq!(transfers.len(), 3);
        assert_eq!(transfers[0], input["ec_transfer_params"]["ec_items"][0]);
        assert_eq!(transfers[1]["mm_hash"], format!("engine-{uuid}"));
        assert_ne!(transfers[1]["transfer_id"], transfers[2]["transfer_id"]);
        let handle = &body["ec_transfer_params"][format!("engine-{uuid}")];
        assert_eq!(handle["metadata"], json!({}));
        assert_eq!(handle["peer_port"], 4321);
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
    async fn abandoned_pushes_cancel_all_consumer_shards() {
        for mode in [
            "normal",
            "empty",
            "mixed",
            "invalid",
            "fail",
            "cancel_error",
        ] {
            let (mut stage, seen, server) = mock(mode).await;
            let (send_address, recv_address) = tokio::sync::oneshot::channel();
            let control = tokio::task::spawn_blocking(move || {
                let context = zmq::Context::new();
                let sockets = [0, 1].map(|_| {
                    let socket = context.socket(zmq::REP).unwrap();
                    socket.set_rcvtimeo(5000).unwrap();
                    socket.set_sndtimeo(5000).unwrap();
                    socket.set_linger(0).unwrap();
                    socket.bind("tcp://127.0.0.1:*").unwrap();
                    socket
                });
                let addresses = sockets
                    .each_ref()
                    .map(|s| s.get_last_endpoint().unwrap().unwrap());
                let ports = addresses
                    .each_ref()
                    .map(|a| a.rsplit_once(':').unwrap().1.parse::<u16>().unwrap());
                send_address.send(addresses[0].clone()).unwrap();
                let discovery: Value =
                    serde_json::from_slice(&sockets[0].recv_bytes(0).unwrap()).unwrap();
                assert_eq!(discovery, json!({"op": "peers"}));
                sockets[0]
                    .send(
                        json!({"ok": true, "result": {"ports": ports}})
                            .to_string()
                            .as_bytes(),
                        0,
                    )
                    .unwrap();
                let mut cancelled = [Vec::new(), Vec::new()];
                // "mixed" leaves the unreported item's transfer abandoned while
                // the reported one is disarmed below.
                let expected = if mode == "mixed" { 1 } else { 2 };
                for (rank, socket) in sockets.iter().enumerate() {
                    for _ in 0..expected {
                        let message: Value =
                            serde_json::from_slice(&socket.recv_bytes(0).unwrap()).unwrap();
                        cancelled[rank].push(message);
                        let response = if mode == "cancel_error" && rank == 0 {
                            json!({"ok": false, "error": "injected failure"})
                        } else {
                            json!({"ok": true, "result": {"cancelled": true}})
                        };
                        socket.send(response.to_string().as_bytes(), 0).unwrap();
                    }
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
                .flat_map(|r| {
                    assert_eq!(r["ec_transfer_params"]["consumer_zmq"], address);
                    r["ec_transfer_params"]["ec_items"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|entry| entry["transfer_id"].clone())
                        .collect::<Vec<_>>()
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
            assert_eq!(cancelled[0], cancelled[1]);
            assert!(cancelled[0].iter().all(|r| r["op"] == "cancel"
                && r["abandon"] == true
                && ids.contains(&r["transfer_id"])));
            if mode == "mixed" {
                assert_eq!(cancelled[0][0]["transfer_id"], ids[0]);
            } else {
                assert_ne!(
                    cancelled[0][0]["transfer_id"],
                    cancelled[0][1]["transfer_id"]
                );
            }
            server.abort();
        }
    }

    #[test]
    fn control_request_rejects_remote_errors() {
        let context = zmq::Context::new();
        let socket = context.socket(zmq::REP).unwrap();
        socket.set_rcvtimeo(5000).unwrap();
        socket.set_sndtimeo(5000).unwrap();
        socket.set_linger(0).unwrap();
        socket.bind("tcp://127.0.0.1:*").unwrap();
        let address = socket.get_last_endpoint().unwrap().unwrap();
        let server = std::thread::spawn(move || {
            socket.recv_bytes(0).unwrap();
            socket
                .send(
                    b"{\"ok\":false,\"error\":\"injected failure\"}".as_slice(),
                    0,
                )
                .unwrap();
        });
        let result = ec_control_request(&context, &address, &json!({"op": "peers"}));
        server.join().unwrap();
        assert!(result.unwrap_err().to_string().contains("injected failure"));
    }
}
