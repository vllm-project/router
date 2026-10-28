use crate::config::EpdConfig;
use crate::core::{Worker, WorkerLoadGuard, WorkerRegistry, WorkerType};
use crate::policies::PolicyRegistry;
use axum::http::{HeaderMap, StatusCode};
use futures_util::future::join_all;
use reqwest::{Client, RequestBuilder};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
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
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
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
    pub(super) fn new(
        config: EpdConfig,
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        Self {
            config,
            worker_registry,
            policy_registry,
        }
    }

    fn select_encoders(&self, item_count: usize) -> Result<Vec<Arc<dyn Worker>>, EpdError> {
        // Query the live role pool once, then preserve static configuration order.
        // Removed/replaced workers must not fall back to the configured raw URL.
        let pool = self.worker_registry.get_by_type(&WorkerType::Encode);
        let by_url: HashMap<_, _> = pool.iter().map(|worker| (worker.url(), worker)).collect();
        let workers: Vec<_> = self
            .config
            .encoder_urls
            .iter()
            .filter_map(|url| by_url.get(url.as_str()).map(|worker| Arc::clone(worker)))
            .collect();
        let policy = self.policy_registry.get_encode_policy();
        let assignments = policy
            .select_encode_batch(&workers, item_count)
            .ok_or_else(|| {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "No available Encode workers".into(),
                )
            })?;
        Ok(assignments
            .into_iter()
            .map(|index| workers[index].clone())
            .collect())
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
        let mut embed_positions = Vec::new();
        for (message_index, message) in body["messages"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
        {
            for (content_index, item) in message
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .enumerate()
            {
                if matches!(
                    item.get("type").and_then(Value::as_str),
                    Some("image_embeds" | "audio_embeds" | "video_embeds")
                ) {
                    embed_positions.push((message_index, content_index));
                }
            }
        }
        let consumer_addr = self.config.consumer_zmq_addrs.get(consumer_url);
        if !self.config.consumer_zmq_addrs.is_empty() && consumer_addr.is_none() {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "Selected embedding consumer has no EC control address".into(),
            ));
        }
        let encoders = self.select_encoders(items.len())?;
        let mut pending = consumer_addr.map(|control_addr| EcTransferGuard {
            control_addr: control_addr.clone(),
            transfer_ids: items.iter().map(|item| item.transfer_id.clone()).collect(),
        });
        let request_id = headers
            .and_then(|h| h.get("x-request-id"))
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        // Assign per item, then batch the images of one request that land
        // on the same encoder into a single call, like the Python EPD proxy.
        // Audio and video items stay singleton groups.
        let mut groups: Vec<(&Arc<dyn Worker>, Vec<usize>)> = Vec::new();
        let mut image_group_of: HashMap<&str, usize> = HashMap::new();
        for (index, item) in items.iter().enumerate() {
            let encoder = &encoders[index];
            if item.kind == "image_url" {
                match image_group_of.get(encoder.url()) {
                    Some(&group) => groups[group].1.push(index),
                    None => {
                        image_group_of.insert(encoder.url(), groups.len());
                        groups.push((encoder, vec![index]));
                    }
                }
            } else {
                groups.push((encoder, vec![index]));
            }
        }
        let replies = join_all(groups.iter().map(|(encoder, indices)| {
            let encoder_url = encoder.url();
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
                encoder.as_ref(),
                &self.worker_registry,
            )
        }))
        .await;
        let replies: Vec<Value> = replies.into_iter().collect::<Result<_, _>>()?;
        let mut handles = Map::new();
        // Sort transfers by content position, including caller-supplied embeds.
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
            for (mm_hash, reported) in params.iter() {
                let Some(item_indices) = reported.get("item_indices") else {
                    continue;
                };
                let item_indices = item_indices.as_array().ok_or_else(|| {
                    (StatusCode::BAD_GATEWAY, "Invalid encoder item index".into())
                })?;
                for local in item_indices {
                    let local = local
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .filter(|&value| value < indices.len())
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
                    transfers.push((
                        (item.message_index, item.content_index),
                        json!({
                            "mm_hash": hash,
                            "transfer_id": item.transfer_id,
                        }),
                    ));
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
                    transfers.push((
                        (item.message_index, item.content_index),
                        json!({
                            "mm_hash": hash,
                            "transfer_id": item.transfer_id,
                        }),
                    ));
                    used_transfer_ids.push(item.transfer_id.clone());
                }
                handles.insert(hash, reported);
            }
        }
        if !handles.is_empty() {
            let mut previous = Vec::new();
            if let Some(Value::Object(mut existing)) =
                body.get_mut("ec_transfer_params").map(Value::take)
            {
                if let Some(Value::Array(entries)) = existing.remove("ec_items") {
                    previous = entries;
                }
                existing.extend(handles);
                handles = existing;
            }
            if previous.len() == embed_positions.len() {
                transfers.extend(embed_positions.into_iter().zip(previous.drain(..)));
            }
            transfers.sort_by_key(|(position, _)| *position);
            previous.extend(transfers.into_iter().map(|(_, entry)| entry));
            if !previous.is_empty() {
                handles.insert("ec_items".into(), Value::Array(previous));
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
    worker: &dyn Worker,
    registry: &WorkerRegistry,
) -> Result<Value, EpdError> {
    // A batch stays in flight until its response body/JSON has been consumed.
    // Dropping the future releases load without inventing a completion/outcome.
    let _load = WorkerLoadGuard::new(worker);
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
    let result = async {
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
    .await;
    worker.increment_processed();
    // Match consumer circuit semantics: client errors are reachable workers;
    // transport failures, 5xx and unreadable JSON are failed batch attempts.
    let success = result
        .as_ref()
        .map_or_else(|(status, _)| status.is_client_error(), |_| true);
    let was_available = worker.is_available();
    worker.record_outcome(success);
    if was_available != worker.is_available() {
        registry.notify_worker_state_change();
    }
    tracing::debug!(
        worker = worker.url(),
        items = indices.len(),
        success,
        "EPD encoder batch complete"
    );
    result
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

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use axum::{extract::State, routing::post, Json, Router};
    use std::sync::{Arc, Mutex};

    fn test_stage(config: EpdConfig) -> EncoderStage {
        let workers = Arc::new(WorkerRegistry::new());
        for url in &config.encoder_urls {
            workers.register(Arc::from(crate::core::WorkerFactory::create_encode(
                url.clone(),
            )));
        }
        let policies = Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin));
        policies.set_encode_policy(config.effective_encode_policy());
        EncoderStage::new(config, workers, policies)
    }

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
        if state.mode == "no_report" {
            return (StatusCode::OK, Json(json!({})));
        }
        if state.mode == "missing_entry" {
            return (
                StatusCode::OK,
                Json(json!({"ec_transfer_params": {"opaque": {"backend_field": 42}}})),
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
            let item_indices = match state.mode {
                "bad_indices_type" => json!("0"),
                "bad_index_fractional" => json!([0.5]),
                "bad_index_out_of_range" => json!([indices.len()]),
                "duplicate_index" => json!([0, 0]),
                _ => json!(indices),
            };
            params.insert(
                format!("engine-{uuid}"),
                json!({
                    "metadata": metadata,
                    "item_indices": item_indices,
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
        let stage = test_stage(EpdConfig {
            encoder_urls: vec![url],
            encode_policy: None,
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
    async fn encoder_health_checks_preserve_trailing_slash_url_identity() {
        use crate::core::{HealthConfig, WorkerFactory};
        use axum::routing::get;

        for (suffix, configured_endpoint, route_path) in [
            ("", "/health", "/health"),
            ("/", "/health", "/health"),
            ("///", "/health", "/health"),
            ("/", "/encoder-health", "/encoder-health"),
            ("/", "health", "/health"),
            ("", "health", "/health"),
            ("///", "health", "/health"),
            ("", "///health", "/health"),
            ("/", "///health", "/health"),
            ("///", "///health", "/health"),
        ] {
            let state = MockEncoder {
                mode: "normal",
                seen: Arc::new(Mutex::new(Vec::new())),
            };
            let app = Router::new()
                .route(route_path, get(|| async { StatusCode::OK }))
                .route("/v1/chat/completions", post(encode))
                .with_state(state.clone());
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base_url = format!("http://{}", listener.local_addr().unwrap());
            let url = format!("{base_url}{suffix}");
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let stage = test_stage(EpdConfig {
                encoder_urls: vec![url.clone()],
                encode_policy: None,
                consumer_zmq_addrs: Default::default(),
            });
            let health_config = HealthConfig {
                endpoint: configured_endpoint.into(),
                ..Default::default()
            };
            let failure_threshold = health_config.failure_threshold;
            let worker: Arc<dyn Worker> = Arc::from(WorkerFactory::create_encode_with_config(
                url.clone(),
                Default::default(),
                health_config,
            ));
            stage.worker_registry.register(worker.clone());
            let client = Client::new();

            // Encoding already works before the first health check.
            stage
                .prepare(&client, request(), "pd", None, None)
                .await
                .unwrap();
            let mut health_results = Vec::new();
            for _ in 0..failure_threshold {
                health_results.push(worker.check_health_async().await);
            }
            let next_request = stage.prepare(&client, request(), "pd", None, None).await;
            assert!(
                health_results.iter().all(Result::is_ok),
                "Health checks failed for {url} with endpoint {configured_endpoint:?}: {health_results:?}; next request: {:?}",
                next_request.as_ref().err()
            );
            assert!(worker.is_healthy());
            assert!(worker.is_available());
            next_request.unwrap();
            assert_eq!(state.seen.lock().unwrap().len(), 2);

            // Only the request URL is normalized; config and registry keys stay exact.
            assert_eq!(stage.config.encoder_urls, vec![url.clone()]);
            assert_eq!(worker.url(), url);
            assert_eq!(
                worker.metadata().health_config.endpoint,
                configured_endpoint
            );
            assert!(Arc::ptr_eq(
                &stage.worker_registry.get_by_url(&url).unwrap(),
                &worker
            ));
            if !suffix.is_empty() {
                assert!(stage.worker_registry.get_by_url(&base_url).is_none());
            }
            server.abort();
        }
    }

    #[test]
    fn selection_uses_live_encode_pool_in_config_order_and_independent_slot() {
        use crate::{config::EncodePolicy, core::BasicWorker};
        let stage = test_stage(EpdConfig {
            encoder_urls: vec!["http://e0".into(), "http://e1".into(), "http://e2".into()],
            encode_policy: None,
            consumer_zmq_addrs: Default::default(),
        });
        let urls = |count| {
            stage
                .select_encoders(count)
                .unwrap()
                .iter()
                .map(|w| w.url().to_owned())
                .collect::<Vec<_>>()
        };
        assert_eq!(urls(3), ["http://e0", "http://e1", "http://e2"]);
        let second_stage = EncoderStage::new(
            stage.config.clone(),
            stage.worker_registry.clone(),
            stage.policy_registry.clone(),
        );
        assert_eq!(urls(1), ["http://e0"]);
        assert_eq!(
            second_stage.select_encoders(1).unwrap()[0].url(),
            "http://e1"
        );
        // Reinsert E0 last in the registry. Config order still determines RR.
        stage.worker_registry.remove_by_url("http://e0");
        let e0: Arc<dyn Worker> =
            Arc::new(BasicWorker::new("http://e0".into(), WorkerType::Encode));
        stage.worker_registry.register(e0.clone());
        stage
            .policy_registry
            .set_encode_policy(EncodePolicy::RoundRobin);
        assert_eq!(urls(3), ["http://e0", "http://e1", "http://e2"]);
        let e1 = stage.worker_registry.get_by_url("http://e1").unwrap();
        e1.set_healthy(false);
        let e2 = stage.worker_registry.get_by_url("http://e2").unwrap();
        for _ in 0..5 {
            e2.record_outcome(false);
        }
        assert_eq!(urls(4), ["http://e0"; 4]);
        // A healthy consumer is not an Encode fallback, even at a configured URL.
        stage.worker_registry.register(Arc::new(BasicWorker::new(
            "http://e2".into(),
            WorkerType::Regular,
        )));
        assert_eq!(urls(4), ["http://e0"; 4]);
        stage
            .policy_registry
            .set_encode_policy(EncodePolicy::Random);
        assert_eq!(urls(8), ["http://e0"; 8]);
        // The independent global consumer RR has not advanced at all.
        assert_eq!(
            stage
                .policy_registry
                .get_default_policy()
                .select_worker(&[e0.clone(), e1], None),
            Some(0)
        );
        stage.worker_registry.remove_by_url("http://e0");
        assert_eq!(
            stage.select_encoders(1).unwrap_err().0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(e0.processed_requests(), 0);
    }

    #[tokio::test]
    async fn text_and_existing_embeds_bypass_an_empty_encode_pool() {
        let stage = test_stage(EpdConfig {
            encoder_urls: vec![],
            encode_policy: None,
            consumer_zmq_addrs: Default::default(),
        });
        for content in [
            json!("hello"),
            json!([{"type":"image_embeds", "uuid":"existing", "image_embeds":[1,2]}]),
        ] {
            let body = json!({"messages":[{"role":"user", "content":content}]});
            assert_eq!(
                stage
                    .prepare(&Client::new(), body.clone(), "pd", None, None)
                    .await
                    .unwrap()
                    .0,
                body
            );
        }
        assert_eq!(
            stage
                .prepare(&Client::new(), request(), "pd", None, None)
                .await
                .err()
                .unwrap()
                .0,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn image_batches_and_audio_video_singletons_count_completed_attempts() {
        let (stage, seen, server) = mock("normal").await;
        let worker = stage.worker_registry.get_by_type(&WorkerType::Encode)[0].clone();
        let mut input = request();
        input["messages"][0]["content"]
            .as_array_mut()
            .unwrap()
            .extend([
                json!({"type":"audio_url", "audio_url":{"url":"audio-a"}}),
                json!({"type":"audio_url", "audio_url":{"url":"audio-b"}}),
                json!({"type":"video_url", "video_url":{"url":"video-a"}}),
            ]);
        stage
            .prepare(&Client::new(), input, "pd", None, None)
            .await
            .unwrap();
        let requests = seen.lock().unwrap();
        let sizes: Vec<_> = requests
            .iter()
            .map(|b| b["messages"][0]["content"].as_array().unwrap().len())
            .collect();
        assert_eq!(sizes.len(), 4);
        assert_eq!(sizes.iter().filter(|&&size| size == 2).count(), 1);
        assert_eq!(sizes.iter().filter(|&&size| size == 1).count(), 3);
        assert_eq!(worker.processed_requests(), 4);
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.circuit_breaker().total_successes(), 4);
        server.abort();
    }

    #[tokio::test]
    async fn batch_outcomes_cover_http_json_and_transport_failures() {
        use axum::response::IntoResponse;
        for (status, response_body, success) in [
            (StatusCode::OK, "{}", true),
            (StatusCode::BAD_REQUEST, "bad input", true),
            (StatusCode::INTERNAL_SERVER_ERROR, "failed", false),
            (StatusCode::OK, "invalid json", false),
        ] {
            let app = Router::new().route(
                "/v1/chat/completions",
                post(move || async move { (status, response_body).into_response() }),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let stage = test_stage(EpdConfig {
                encoder_urls: vec![url.clone()],
                encode_policy: None,
                consumer_zmq_addrs: Default::default(),
            });
            let worker: Arc<dyn Worker> =
                Arc::from(crate::core::WorkerFactory::create_encode_with_config(
                    url,
                    crate::core::CircuitBreakerConfig {
                        failure_threshold: 1,
                        ..Default::default()
                    },
                    Default::default(),
                ));
            stage.worker_registry.register(worker.clone());
            let revision = stage.worker_registry.revision();
            let result = stage
                .prepare(&Client::new(), request(), "pd", None, None)
                .await;
            assert_eq!(result.is_ok(), status.is_success() && success);
            if let Err((actual, _)) = result {
                assert_eq!(
                    actual,
                    if status.is_success() {
                        StatusCode::BAD_GATEWAY
                    } else {
                        status
                    }
                );
            }
            assert_eq!(worker.load(), 0);
            assert_eq!(worker.processed_requests(), 1);
            assert_eq!(
                worker.circuit_breaker().total_successes(),
                u64::from(success)
            );
            assert_eq!(
                worker.circuit_breaker().total_failures(),
                u64::from(!success)
            );
            assert_eq!(stage.worker_registry.revision() > revision, !success);
            if !success {
                assert_eq!(
                    stage.select_encoders(1).unwrap_err().0,
                    StatusCode::SERVICE_UNAVAILABLE
                );
            }
            server.abort();
        }
        // Bind without listening to keep a local refused port reserved.
        let socket = tokio::net::TcpSocket::new_v4().unwrap();
        socket.bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let url = format!("http://{}", socket.local_addr().unwrap());
        let stage = test_stage(EpdConfig {
            encoder_urls: vec![url],
            encode_policy: None,
            consumer_zmq_addrs: Default::default(),
        });
        let worker = stage.worker_registry.get_by_type(&WorkerType::Encode)[0].clone();
        assert_eq!(
            stage
                .prepare(&Client::new(), request(), "pd", None, None)
                .await
                .err()
                .unwrap()
                .0,
            StatusCode::BAD_GATEWAY
        );
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.processed_requests(), 1);
        assert_eq!(worker.circuit_breaker().total_failures(), 1);
    }

    #[tokio::test]
    async fn blocked_response_body_holds_load_until_completion_timeout_or_drop() {
        use axum::{body::Body, response::Response};
        use futures_util::stream;
        use std::time::Duration;
        use tokio::sync::{Notify, Semaphore};
        for mode in ["complete", "timeout", "drop"] {
            let entered = Arc::new(Notify::new());
            let release = Arc::new(Semaphore::new(0));
            let (entered_body, release_body) = (entered.clone(), release.clone());
            let app = Router::new().route(
                "/v1/chat/completions",
                post(move || {
                    let (entered, release) = (entered_body.clone(), release_body.clone());
                    async move {
                        let body = stream::once(async move {
                            entered.notify_one();
                            let _permit = release.acquire().await.unwrap();
                            Ok::<_, std::io::Error>("{}")
                        });
                        Response::new(Body::from_stream(body))
                    }
                }),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let stage = Arc::new(test_stage(EpdConfig {
                encoder_urls: vec![url],
                encode_policy: None,
                consumer_zmq_addrs: Default::default(),
            }));
            let worker = stage.worker_registry.get_by_type(&WorkerType::Encode)[0].clone();
            let client = if mode == "timeout" {
                Client::builder()
                    .timeout(Duration::from_millis(250))
                    .build()
                    .unwrap()
            } else {
                Client::new()
            };
            let task_stage = stage.clone();
            let task = tokio::spawn(async move {
                task_stage
                    .prepare(&client, request(), "pd", None, None)
                    .await
                    .map(|(body, _)| body)
            });
            tokio::time::timeout(Duration::from_secs(5), entered.notified())
                .await
                .unwrap();
            assert_eq!(worker.load(), 1, "{mode}");
            assert_eq!(worker.processed_requests(), 0);
            worker.reset_load();
            assert_eq!(worker.load(), 1);
            match mode {
                "complete" => {
                    release.add_permits(1);
                    task.await.unwrap().unwrap();
                }
                "timeout" => {
                    assert_eq!(task.await.unwrap().unwrap_err().0, StatusCode::BAD_GATEWAY);
                }
                _ => {
                    task.abort();
                    assert!(task.await.unwrap_err().is_cancelled());
                }
            }
            assert_eq!(worker.load(), 0, "{mode}");
            assert_eq!(worker.processed_requests(), usize::from(mode != "drop"));
            assert_eq!(
                worker.circuit_breaker().total_successes(),
                u64::from(mode == "complete")
            );
            assert_eq!(
                worker.circuit_breaker().total_failures(),
                u64::from(mode == "timeout")
            );
            worker.reset_load();
            assert_eq!(worker.load(), 0);
            server.abort();
        }
    }

    #[tokio::test]
    async fn failed_batch_still_joins_sibling_and_releases_each_load() {
        use std::time::Duration;
        use tokio::sync::{Notify, Semaphore};
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Semaphore::new(0));
        let (notice, gate) = (entered.clone(), release.clone());
        let app = Router::new()
            .route(
                "/e0/v1/chat/completions",
                post(|| async { StatusCode::SERVICE_UNAVAILABLE }),
            )
            .route(
                "/e1/v1/chat/completions",
                post(move || {
                    let (notice, gate) = (notice.clone(), gate.clone());
                    async move {
                        notice.notify_one();
                        let _permit = gate.acquire().await.unwrap();
                        Json(json!({}))
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let stage = Arc::new(test_stage(EpdConfig {
            encoder_urls: vec![format!("{base}/e0"), format!("{base}/e1")],
            encode_policy: None,
            consumer_zmq_addrs: Default::default(),
        }));
        let workers = stage.worker_registry.get_by_type(&WorkerType::Encode);
        let task_stage = stage.clone();
        let task = tokio::spawn(async move {
            task_stage
                .prepare(&Client::new(), request(), "pd", None, None)
                .await
                .map(|(body, _)| body)
        });
        tokio::time::timeout(Duration::from_secs(5), entered.notified())
            .await
            .unwrap();
        assert!(!task.is_finished());
        assert_eq!(workers[1].load(), 1);
        release.add_permits(1);
        assert_eq!(
            task.await.unwrap().unwrap_err().0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        for worker in workers {
            assert_eq!(worker.load(), 0);
            assert_eq!(worker.processed_requests(), 1);
        }
        server.abort();
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
    async fn ec_items_follow_media_order_across_encoders() {
        // Images [A, B, A] round-robin onto two encoders: E0 gets [A, A]
        // batched, E1 gets [B]. ec_items must stay in media order
        // [A, B, A] so the consumer's positional transfer mapping cancels
        // every reservation on request finish.
        let spawn = || async {
            let state = MockEncoder {
                mode: "normal",
                seen: Arc::new(Mutex::new(Vec::new())),
            };
            let seen = state.seen.clone();
            let app = Router::new()
                .route("/v1/chat/completions", post(encode))
                .with_state(state);
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            (url, seen, server)
        };
        let (url0, seen0, server0) = spawn().await;
        let (url1, seen1, server1) = spawn().await;
        let stage = test_stage(EpdConfig {
            encoder_urls: vec![url0, url1],
            encode_policy: None,
            consumer_zmq_addrs: Default::default(),
        });
        let input = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,def"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }],
        });
        let (body, _) = stage
            .prepare(&Client::new(), input, "pd", None, None)
            .await
            .unwrap();
        // The split across two encoders actually happened.
        assert_eq!(seen0.lock().unwrap().len(), 1);
        assert_eq!(seen1.lock().unwrap().len(), 1);
        assert_eq!(
            seen0.lock().unwrap()[0]["messages"][0]["content"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        let content = &body["messages"][0]["content"];
        let uuid_a = content[0]["uuid"].as_str().unwrap();
        let uuid_b = content[1]["uuid"].as_str().unwrap();
        assert_ne!(uuid_a, uuid_b);
        assert_eq!(content[2]["uuid"].as_str().unwrap(), uuid_a);
        let hashes: Vec<String> = body["ec_transfer_params"]["ec_items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["mm_hash"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(
            hashes,
            [
                format!("engine-{uuid_a}"),
                format!("engine-{uuid_b}"),
                format!("engine-{uuid_a}"),
            ]
        );
        server0.abort();
        server1.abort();
    }

    #[tokio::test]
    async fn existing_and_new_transfers_follow_content_order() {
        let (stage, _, server) = mock("normal").await;
        let embed = json!({
            "type": "image_embeds",
            "image_embeds": {"image_grid_thw": [1, 2, 3]},
            "uuid": "cached",
        });
        let raw = json!({"type": "image_url", "image_url": {"url": "new-image"}});
        let cached = json!({"mm_hash": "cached-hash", "transfer_id": "cached-transfer"});
        for content in [json!([raw, embed]), json!([embed, raw])] {
            let raw_index = usize::from(content[0]["type"] == "image_embeds");
            let input = json!({
                "messages": [{"role": "user", "content": content}],
                "ec_transfer_params": {
                    "cached-hash": {"peer_port": 42},
                    "ec_items": [cached],
                },
            });
            let (body, _) = stage
                .prepare(&Client::new(), input, "pd", None, None)
                .await
                .unwrap();
            let transfers = body["ec_transfer_params"]["ec_items"].as_array().unwrap();
            assert_eq!(transfers.len(), 2);
            assert_eq!(transfers[1 - raw_index], cached);
            assert_ne!(transfers[raw_index]["transfer_id"], cached["transfer_id"]);
            assert_eq!(body["messages"][0]["content"][1 - raw_index], embed);
            assert_eq!(body["ec_transfer_params"]["cached-hash"]["peer_port"], 42);
        }
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
    async fn missing_report_and_missing_entry_keep_raw_media_with_uuid() {
        for mode in ["no_report", "missing_entry"] {
            let (stage, seen, server) = mock(mode).await;
            let input = request();
            let (body, pending) = stage
                .prepare(&Client::new(), input.clone(), "pd", None, None)
                .await
                .unwrap();
            assert!(pending.is_none());

            let encoded = seen.lock().unwrap();
            let uuid = encoded[0]["messages"][0]["content"][0]["uuid"]
                .as_str()
                .unwrap();
            for index in [0, 2] {
                let mut expected = input["messages"][0]["content"][index].clone();
                expected["uuid"] = json!(uuid);
                assert_eq!(body["messages"][0]["content"][index], expected);
            }
            assert_eq!(
                body["messages"][0]["content"][1],
                input["messages"][0]["content"][1]
            );
            server.abort();
        }
    }

    #[tokio::test]
    async fn malformed_encoder_item_indices_are_rejected() {
        for mode in [
            "bad_indices_type",
            "bad_index_fractional",
            "bad_index_out_of_range",
            "duplicate_index",
        ] {
            let (stage, _, server) = mock(mode).await;
            let mut input = request();
            input["messages"][0]["content"] = json!([input["messages"][0]["content"][0].clone()]);
            let error = stage
                .prepare(&Client::new(), input, "pd", None, None)
                .await
                .err()
                .unwrap();
            assert_eq!(error.0, StatusCode::BAD_GATEWAY, "mode: {mode}");
            server.abort();
        }
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
