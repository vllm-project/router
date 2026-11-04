//! EXPERIMENTAL: E→PD over gRPC on the released vLLM contract (vllm-proto 0.2,
//! vLLM >= 0.30.0), with no vLLM changes.
//!
//! Mirrors [`EncoderStage::prepare`]: the same item identity (caller uuid, else
//! the URL's SHA-256), transfer ids, per-item round-robin, Mooncake transfer
//! guard, report correlation and raw-media fallbacks.
//!
//! Each item gets its own encoder call, which returns its expanded prompt ids.
//! Splicing those expansions into PD's marker prompt yields PD's exact prompt,
//! used for KV-aware selection and admission. Without Mooncake the stage runs
//! before selection; Mooncake needs the consumer's address first, so it still
//! runs after selection and routes on the marker prompt.
//!
//! What differs, because of the released proto/frontend:
//! - PD always receives the raw media (uuid pinned) and preprocesses it again on
//!   CPU. vllm-proto 0.2 carries placeholder metadata only as engine MessagePack
//!   `PreprocessedMediaFeatures.kwargs` and the encoder reply has no placeholder
//!   ranges, so the HTTP `*_embeds` rewrite has no gRPC equivalent yet. PD's
//!   frontend still drops the pixels of every item listed with metadata in
//!   `ec_transfer_params.ec_items`, and its scheduler loads the published
//!   embedding instead of running the vision encoder.
//! - `mm_processor_kwargs` / `media_io_kwargs` have no proto field, and caller
//!   `*_embeds` parts need the MessagePack kwargs above: both are rejected.

use super::*;
use crate::backend::frontend::{EngineFrontend, PreparedRequest};
use crate::backend::pb;
use crate::protocols::spec::ChatCompletionRequest;
use std::sync::Arc;

/// Request fields vllm-proto 0.2 cannot carry to the encoder or PD.
const UNSUPPORTED_FIELDS: [&str; 3] = [
    "mm_processor_kwargs",
    "media_io_kwargs",
    "ec_transfer_params",
];

/// Response extension marking an encoder-stage failure. As in HTTP EPD, it is
/// returned as-is: not retried and not charged to the PD worker.
#[derive(Clone, Copy, Debug)]
pub(in crate::routers::http) struct EncoderStageFailure;

/// A media part as PD sees it, in placeholder order.
struct Placed {
    message_index: usize,
    content_index: usize,
    item: pb::MediaItem,
}

impl EncoderStage {
    /// Whether the encoder stage must wait for the consumer to be selected:
    /// Mooncake producers push to the consumer's address.
    pub(in crate::routers::http) fn encodes_after_selection(&self) -> bool {
        !self.config.consumer_zmq_addrs.is_empty()
    }

    /// Encode `prepared`'s media and fill `prepared.media`; a no-op once
    /// encoded. `consumer_url` matters only for Mooncake. Returns the guard to
    /// disarm once PD accepted the request.
    pub(in crate::routers::http) async fn prepare_grpc(
        &self,
        frontend: &EngineFrontend,
        prepared: &mut PreparedRequest,
        consumer_url: &str,
        headers: Option<&HeaderMap>,
    ) -> Result<Option<EcTransferGuard>, EpdError> {
        let started = Instant::now();
        let Some(media) = prepared.media.as_mut().filter(|media| !media.encoded) else {
            return Ok(None);
        };
        let request = prepared.request.as_ref();
        if let Some(field) = UNSUPPORTED_FIELDS
            .iter()
            .find(|field| request.other.contains_key(**field))
        {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("{field} is not supported by experimental gRPC EPD: vllm-proto 0.2 has no field for it"),
            ));
        }
        let body = serde_json::to_value(request)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let items = collect_media_items(&body)?;
        let placed = place_media(&body, &items, media.order.as_deref())?;
        if items.is_empty() {
            media.items = placed.into_iter().map(|placed| placed.item).collect();
            media.encoded = true;
            return Ok(None);
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

        // Round-robin per item as in `prepare`, but one call per item (HTTP
        // batches images per encoder): an item's expansion is then exactly
        // what its encoder's prompt ids add, giving PD's exact prompt.
        let start = self.next_encoder.fetch_add(items.len(), Ordering::Relaxed);
        let groups: Vec<(usize, Vec<usize>)> = (0..items.len())
            .map(|index| {
                let encoder = start.wrapping_add(index) % self.config.encoder_urls.len();
                (encoder, vec![index])
            })
            .collect();

        let replies = join_all(groups.iter().map(|(encoder, indices)| {
            encode_group_grpc(
                frontend,
                &self.config.encoder_urls[*encoder],
                format!(
                    "{request_id}:{}:{:.6}",
                    indices[0], items[indices[0]].transfer_id
                ),
                &items,
                indices,
                request,
                consumer_addr.map(String::as_str),
            )
        }))
        .await;
        let replies: Vec<EncoderReply> = replies.into_iter().collect::<Result<_, _>>()?;

        let mut expansions: HashMap<(usize, usize), Expansion> = HashMap::new();
        for ((_, indices), reply) in groups.iter().zip(&replies) {
            if let (&[index], Some(expanded)) = (indices.as_slice(), reply.expanded.as_deref()) {
                if let Some(expansion) = Expansion::between(&reply.prompt, expanded) {
                    let item = &items[index];
                    expansions.insert((item.message_index, item.content_index), expansion);
                }
            }
        }
        let routing = splice_expansions(&prepared.tokenized.token_ids, &placed, &expansions);
        if routing.is_none() {
            tracing::debug!(%request_id, "EPD exact prompt unavailable; routing on marker prompt");
        }

        let mut handles = Map::new();
        let mut transfers: Vec<((usize, usize), Value)> = Vec::new();
        let mut used_transfer_ids = Vec::new();
        let mut published = 0;
        for ((_, indices), reply) in groups.iter().zip(replies) {
            let params = reply.params;
            // No EC report at all: PD keeps the raw media with the uuid pinned.
            let Some(params) = params else {
                continue;
            };
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
                        (indices.len() == 1 && params.len() == 1)
                            .then(|| params.iter().next())
                            .flatten()
                            .map(|(hash, reported)| (hash.clone(), reported.clone()))
                    });
                // Unreported item (e.g. an encoder processor-cache hit): PD
                // keeps the raw media and computes it if the cache misses.
                let Some((hash, reported)) = entry else {
                    continue;
                };
                let mut transfer = json!({
                    "mm_hash": hash,
                    "transfer_id": item.transfer_id,
                });
                if let Some(metadata) = reported
                    .get("metadata")
                    .and_then(Value::as_object)
                    .filter(|m| !m.is_empty())
                {
                    // Validated like the HTTP rewrite. PD only reads the keys:
                    // it drops every other kwarg (the pixels) of this item.
                    flatten_metadata(metadata)?;
                    for (key, value) in metadata {
                        transfer[key] = value.clone();
                    }
                    published += 1;
                }
                transfers.push(((item.message_index, item.content_index), transfer));
                used_transfer_ids.push(item.transfer_id.clone());
                handles.insert(hash, reported);
            }
        }
        if !handles.is_empty() {
            transfers.sort_by_key(|(position, _)| *position);
            handles.insert(
                "ec_items".into(),
                Value::Array(transfers.into_iter().map(|(_, entry)| entry).collect()),
            );
            media.ec_transfer_params = Some(Value::Object(handles));
        }
        media.items = placed.into_iter().map(|placed| placed.item).collect();
        media.encoded = true;
        let exact_prompt_tokens = routing.as_ref().map(|ids| ids.len());
        let reported = used_transfer_ids.len();
        media.routing_token_ids = routing;
        if let Some(pending) = pending.as_mut() {
            pending.retain(used_transfer_ids);
        }
        tracing::info!(
            %request_id,
            transport = "grpc",
            items = items.len(),
            encoder_calls = groups.len(),
            // reported: PD loads these from the EC; published: of those, the
            // ones whose placeholder metadata also lets PD skip preprocessing.
            reported,
            published,
            marker_prompt_tokens = prepared.tokenized.token_ids.len(),
            exact_prompt_tokens,
            encode_ms = started.elapsed().as_secs_f64() * 1000.0,
            "EPD encoder stage complete (experimental gRPC: PD re-preprocesses raw media)"
        );
        Ok(pending)
    }
}

/// PD's media list in the renderer's placeholder order (request order when the
/// renderer reports none). Encoded items carry their identity uuid; inline
/// `input_audio` is never sent to an encoder, as on HTTP.
fn place_media(
    body: &Value,
    items: &[MediaItem],
    order: Option<&[crate::backend::frontend::MediaPartSource]>,
) -> Result<Vec<Placed>, EpdError> {
    let mut placed = Vec::new();
    for (message_index, message) in body["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        for (content_index, part) in message
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            let item = match part.get("type").and_then(Value::as_str) {
                Some("image_url" | "video_url" | "audio_url") => {
                    let encoded = items
                        .iter()
                        .find(|item| {
                            item.message_index == message_index
                                && item.content_index == content_index
                        })
                        .ok_or_else(|| {
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                "Unindexed media part".into(),
                            )
                        })?;
                    media_item(encoded.kind, &encoded.media, &encoded.content_id)?
                }
                Some("input_audio") => {
                    let data = part
                        .pointer("/input_audio/data")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            (
                                StatusCode::BAD_REQUEST,
                                "input_audio content part is missing data".into(),
                            )
                        })?;
                    let format = part
                        .pointer("/input_audio/format")
                        .and_then(Value::as_str)
                        .unwrap_or("wav");
                    pb::MediaItem {
                        modality: pb::Modality::Audio as i32,
                        source: Some(pb::media_item::Source::DataUri(format!(
                            "data:audio/{format};base64,{data}"
                        ))),
                        ..Default::default()
                    }
                }
                _ => continue,
            };
            placed.push(Placed {
                message_index,
                content_index,
                item,
            });
        }
    }
    if let Some(order) = order {
        let mut by_source: HashMap<(usize, usize), Placed> = placed
            .into_iter()
            .map(|placed| ((placed.message_index, placed.content_index), placed))
            .collect();
        placed = order
            .iter()
            .map(|source| {
                by_source
                    .remove(&(source.message_index, source.content_part_index))
                    .ok_or_else(|| {
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "Renderer media order names an unknown media part".into(),
                        )
                    })
            })
            .collect::<Result<_, _>>()?;
    }
    Ok(placed)
}

fn media_item(kind: &str, media: &Value, uuid: &str) -> Result<pb::MediaItem, EpdError> {
    let modality = match kind {
        "image_url" => pb::Modality::Image,
        "video_url" => pb::Modality::Video,
        _ => pb::Modality::Audio,
    };
    let url = media[kind]["url"].as_str().unwrap_or_default().to_string();
    let source = if url.starts_with("data:") {
        pb::media_item::Source::DataUri(url)
    } else if url.starts_with("http://") || url.starts_with("https://") {
        pb::media_item::Source::Url(url)
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("{kind}.url must be an http(s) or data URL on gRPC"),
        ));
    };
    Ok(pb::MediaItem {
        modality: modality as i32,
        source: Some(source),
        uuid: uuid.to_string(),
        ..Default::default()
    })
}

/// What one encoder call returns: its `ec_transfer_params`, the marker prompt
/// the router sent, and the encoder's expanded prompt ids.
struct EncoderReply {
    params: Option<Map<String, Value>>,
    prompt: Vec<u32>,
    expanded: Option<Vec<u32>>,
}

/// One media item's placeholder expansion: `marker` in the router's prompt
/// becomes `expanded` in the engine's.
#[derive(Debug, PartialEq)]
struct Expansion {
    marker: Vec<u32>,
    expanded: Vec<u32>,
}

impl Expansion {
    /// The span where a one-item encoder prompt and its expansion differ. The
    /// marker is kept non-empty so it can be located in PD's prompt.
    fn between(prompt: &[u32], expanded: &[u32]) -> Option<Self> {
        let mut prefix = prompt
            .iter()
            .zip(expanded)
            .take_while(|(a, b)| a == b)
            .count();
        let suffix = prompt
            .iter()
            .rev()
            .zip(expanded.iter().rev())
            .take_while(|(a, b)| a == b)
            .count()
            .min(prompt.len() - prefix)
            .min(expanded.len() - prefix);
        if prefix + suffix == prompt.len() {
            prefix = prefix.checked_sub(1)?;
        }
        let expansion = Self {
            marker: prompt[prefix..prompt.len() - suffix].to_vec(),
            expanded: expanded[prefix..expanded.len() - suffix].to_vec(),
        };
        (expansion.marker != expansion.expanded).then_some(expansion)
    }
}

/// PD's expanded prompt: each placed item's marker, found left to right in
/// renderer order, replaced by its expansion. `None` unless every media item
/// has an expansion that matches.
fn splice_expansions(
    prompt: &[u32],
    placed: &[Placed],
    expansions: &HashMap<(usize, usize), Expansion>,
) -> Option<Arc<[u32]>> {
    if placed.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(prompt.len());
    let mut cursor = 0;
    for item in placed {
        let expansion = expansions.get(&(item.message_index, item.content_index))?;
        let at = cursor
            + prompt[cursor..]
                .windows(expansion.marker.len())
                .position(|window| window == expansion.marker.as_slice())?;
        out.extend_from_slice(&prompt[cursor..at]);
        out.extend_from_slice(&expansion.expanded);
        cursor = at + expansion.marker.len();
    }
    out.extend_from_slice(&prompt[cursor..]);
    Some(out.into())
}

/// One encoder `Generate`: a chat holding only this group's media, one
/// output token, with the encoder's expanded prompt ids returned.
#[allow(clippy::too_many_arguments)]
async fn encode_group_grpc(
    frontend: &EngineFrontend,
    encoder_url: &str,
    engine_request_id: String,
    items: &[MediaItem],
    indices: &[usize],
    request: &ChatCompletionRequest,
    consumer_addr: Option<&str>,
) -> Result<EncoderReply, EpdError> {
    let chat: ChatCompletionRequest = serde_json::from_value(json!({
        "model": request.model,
        "messages": [{
            "role": "user",
            "content": indices.iter().map(|&i| items[i].media.clone()).collect::<Vec<_>>(),
        }],
    }))
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let token_ids = frontend
        .tokenize_with_media(&chat)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Encoder prompt: {e}")))?;
    let mut generate = pb::GenerateRequest {
        request_id: engine_request_id,
        model: request.model.clone().unwrap_or_default(),
        prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
            ids: token_ids.clone(),
        })),
        response: Some(pb::ResponseOptions {
            prompt_token_ids: true,
            ..Default::default()
        }),
        media: indices
            .iter()
            .map(|&i| media_item(items[i].kind, &items[i].media, &items[i].content_id))
            .collect::<Result<_, _>>()?,
        stopping: Some(pb::StoppingCriteria {
            max_new_tokens: 1,
            ..Default::default()
        }),
        priority: request
            .other
            .get("priority")
            .and_then(Value::as_i64)
            .and_then(|priority| i32::try_from(priority).ok())
            .unwrap_or(0),
        session_id: request
            .session_params
            .as_ref()
            .and_then(|params| params.get("session_id"))
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_string),
        ..Default::default()
    };
    if let Some(control_addr) = consumer_addr {
        crate::backend::convert::set_ec_transfer_params(
            &mut generate,
            &json!({
                "consumer_zmq": control_addr,
                "ec_items": indices
                    .iter()
                    .map(|&i| json!({"transfer_id": items[i].transfer_id}))
                    .collect::<Vec<_>>(),
            }),
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    }
    let response = frontend
        .generate(encoder_url, generate)
        .await
        .map_err(|error| match error {
            Ok(status) => (
                encoder_status(&status),
                format!("Encoder request failed: {}", status.message()),
            ),
            Err(connect) => (
                StatusCode::BAD_GATEWAY,
                format!("Encoder request failed: {connect}"),
            ),
        })?;
    let expanded = response
        .prompt_info
        .map(|info| info.token_ids)
        .filter(|ids| !ids.is_empty());
    let params = response
        .outputs
        .and_then(|outputs| outputs.finish_info)
        .and_then(|finish| finish.ec_transfer_params)
        .map(|params| crate::backend::convert::proto_struct_to_json(&params))
        .and_then(|params| match params {
            Value::Object(params) => Some(params),
            _ => None,
        });
    Ok(EncoderReply {
        params,
        prompt: token_ids,
        expanded,
    })
}

/// The encoder's own verdict, as HTTP EPD forwards an encoder's HTTP status.
fn encoder_status(status: &tonic::Status) -> StatusCode {
    match status.code() {
        tonic::Code::InvalidArgument | tonic::Code::OutOfRange => StatusCode::BAD_REQUEST,
        tonic::Code::NotFound => StatusCode::NOT_FOUND,
        tonic::Code::ResourceExhausted => StatusCode::TOO_MANY_REQUESTS,
        tonic::Code::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        tonic::Code::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
        _ => StatusCode::BAD_GATEWAY,
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    fn body() -> Value {
        json!({"messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "http://x/a.png"}},
                {"type": "text", "text": "and"},
                {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}, "uuid": "vid"},
            ]}
        ]})
    }

    #[test]
    fn placed_media_pins_identity_and_keeps_inline_audio_raw() {
        let body = body();
        let items = collect_media_items(&body).unwrap();
        let placed = place_media(&body, &items, None).unwrap();
        assert_eq!(placed.len(), 3);
        assert_eq!(placed[0].item.uuid, items[0].content_id);
        assert!(matches!(
            placed[0].item.source,
            Some(pb::media_item::Source::Url(_))
        ));
        assert!(placed[1].item.uuid.is_empty());
        assert_eq!(placed[1].item.modality, pb::Modality::Audio as i32);
        assert_eq!(placed[2].item.uuid, "vid");
        assert!(matches!(
            placed[2].item.source,
            Some(pb::media_item::Source::DataUri(_))
        ));
    }

    #[test]
    fn renderer_order_reorders_and_rejects_unknown_parts() {
        use crate::backend::frontend::MediaPartSource;
        let body = body();
        let items = collect_media_items(&body).unwrap();
        let order = [
            MediaPartSource {
                message_index: 0,
                content_part_index: 3,
            },
            MediaPartSource {
                message_index: 0,
                content_part_index: 0,
            },
        ];
        let placed = place_media(&body, &items, Some(&order)).unwrap();
        assert_eq!(placed.len(), 2);
        assert_eq!(placed[0].item.uuid, "vid");
        let unknown = [MediaPartSource {
            message_index: 0,
            content_part_index: 1,
        }];
        assert!(place_media(&body, &items, Some(&unknown)).is_err());
    }

    // Qwen-style ids: 1 = vision_start, 2 = image_pad, 3 = vision_end,
    // 4 = video_pad, 9 = timestamp text; others are text.
    #[test]
    fn expansion_of_repeated_pad_keeps_one_marker() {
        let prompt = [7, 8, 1, 2, 3, 5];
        let expanded = [7, 8, 1, 2, 2, 2, 2, 3, 5];
        assert_eq!(
            Expansion::between(&prompt, &expanded).unwrap(),
            Expansion {
                marker: vec![2],
                expanded: vec![2, 2, 2, 2],
            }
        );
        assert!(Expansion::between(&prompt, &prompt).is_none());
    }

    #[test]
    fn expansion_of_video_frames_wraps_from_first_vision_start() {
        let prompt = [7, 1, 4, 3, 5];
        let expanded = [7, 9, 1, 4, 4, 3, 9, 1, 4, 4, 3, 5];
        assert_eq!(
            Expansion::between(&prompt, &expanded).unwrap(),
            Expansion {
                marker: vec![1],
                expanded: vec![9, 1, 4, 4, 3, 9, 1, 4],
            }
        );
    }

    fn placed_at(content_index: usize) -> Placed {
        Placed {
            message_index: 0,
            content_index,
            item: pb::MediaItem::default(),
        }
    }

    #[test]
    fn splice_walks_items_in_renderer_order() {
        // image, text, video, text, image (same image twice).
        let prompt = [6, 1, 2, 3, 8, 1, 4, 3, 8, 1, 2, 3, 6];
        let image = || Expansion {
            marker: vec![2],
            expanded: vec![2, 2, 2],
        };
        let expansions = HashMap::from([
            ((0, 0), image()),
            (
                (0, 2),
                Expansion {
                    marker: vec![1],
                    expanded: vec![9, 1, 4, 4, 3, 9, 1],
                },
            ),
            ((0, 4), image()),
        ]);
        let placed = [placed_at(0), placed_at(2), placed_at(4)];
        let spliced = splice_expansions(&prompt, &placed, &expansions).unwrap();
        assert_eq!(
            &spliced[..],
            &[6, 1, 2, 2, 2, 3, 8, 9, 1, 4, 4, 3, 9, 1, 4, 3, 8, 1, 2, 2, 2, 3, 6]
        );
        // An item without an expansion (e.g. raw inline audio) disables it.
        let placed = [placed_at(0), placed_at(1)];
        assert!(splice_expansions(&prompt, &placed, &expansions).is_none());
    }

    #[test]
    fn non_http_urls_are_rejected() {
        let media = json!({"type": "image_url", "image_url": {"url": "file:///etc/x"}});
        assert_eq!(
            media_item("image_url", &media, "u").unwrap_err().0,
            StatusCode::BAD_REQUEST
        );
    }
}
