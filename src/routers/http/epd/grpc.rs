//! EXPERIMENTAL: E→PD over gRPC, against the vLLM worker contract proposed
//! alongside this router (placeholder ranges in `PromptInfo`, metadata-only
//! `PreprocessedMediaFeatures`, `MediaItem.marker_offset`). In E→P→D, "PD"
//! below is P; D gets the same media and prompt without `ec_transfer_params`.
//!
//! Each encoder gets a prompt of bare placeholder markers, one per item, and
//! returns where each item expanded (`PromptInfo.mm_placeholders`) plus its
//! `ec_transfer_params`. Images landing on one encoder share a call; audio and
//! video stay singleton. Splicing those ranges into PD's marker prompt gives
//! PD's exact prompt for KV-aware selection and admission. Without Mooncake
//! the stage runs before selection; Mooncake needs the consumer's address
//! first, so it still runs after selection and routes on the marker prompt.
//!
//! PD gets, per item in placeholder order:
//! - a metadata-only feature at its expanded range when the encoder published
//!   placeholder metadata: PD skips preprocessing and loads the embedding
//!   from the EC connector;
//! - otherwise the raw media at its marker (`marker_offset`). PD preprocesses
//!   it; if the encoder still reported the item, PD loads its embedding.
//!
//! Item identity is the engine's: only a caller's uuid is forwarded, and the
//! encoder-cache key comes from the encoder's placeholder. A raw item without
//! uuid rehashes to the same key on PD because both see the same bytes and
//! options. An encoder that rejects the request fails it as the encoder's
//! verdict; an unreachable or failing encoder leaves its items to PD.

use super::*;
use crate::backend::frontend::{EngineFrontend, MediaModality, MediaPartSource, PreparedRequest};
use crate::backend::pb;
use crate::protocols::spec::ChatCompletionRequest;
use std::collections::VecDeque;
use std::sync::Arc;

/// Response extension marking an encoder-stage failure. As in HTTP EPD, it is
/// returned as-is: not retried and not charged to the PD worker.
#[derive(Clone, Copy, Debug)]
pub(in crate::routers::http) struct EncoderStageFailure;

/// One media part, in placeholder order.
#[derive(Clone, Debug)]
struct Item {
    message_index: usize,
    content_index: usize,
    /// The part as the worker takes it raw: source, caller uuid, detail.
    raw: pb::MediaItem,
    transfer_id: String,
}

impl Item {
    fn modality(&self) -> pb::Modality {
        pb::Modality::try_from(self.raw.modality).unwrap_or(pb::Modality::Image)
    }
}

/// What an encoder reported for one item.
#[derive(Clone, Debug, Default, PartialEq)]
struct Published {
    /// Encoder-cache key, which `ec_transfer_params` entries are keyed by.
    identifier: String,
    mm_hash: String,
    /// The item's expansion, when the encoder's ranges tile its prompt.
    tokens: Option<Vec<u32>>,
    is_embed: Vec<bool>,
    /// Placeholder metadata, flattened; `None` means PD must preprocess.
    metadata: Option<Map<String, Value>>,
}

/// One encoder call's outcome for its items, in call order. `None` items were
/// not published, so PD computes them.
type GroupOutcome = (Vec<Option<Published>>, Map<String, Value>);

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
        let Some(media) = prepared.media.as_ref().filter(|media| !media.encoded) else {
            return Ok(None);
        };
        let request = prepared.request.as_ref();
        if request.other.contains_key("ec_transfer_params") {
            return Err((
                StatusCode::BAD_REQUEST,
                "ec_transfer_params is not supported by experimental gRPC EPD".into(),
            ));
        }
        let body = serde_json::to_value(request)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let items = place_items(collect_items(&body)?, media.order.as_deref())?;
        let markers: Vec<Option<Vec<u32>>> = items
            .iter()
            .map(|item| {
                prepared
                    .handle
                    .media_marker(media_modality(item.modality()))
            })
            .collect();
        let request_id = headers
            .and_then(|h| h.get("x-request-id"))
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        self.encode_items(
            frontend,
            prepared,
            items,
            &markers,
            consumer_url,
            &request_id,
        )
        .await
    }

    /// [`Self::prepare_grpc`] once the items and their template markers are
    /// known; split out so tests can pin the markers.
    async fn encode_items(
        &self,
        frontend: &EngineFrontend,
        prepared: &mut PreparedRequest,
        items: Vec<Item>,
        markers: &[Option<Vec<u32>>],
        consumer_url: &str,
        request_id: &str,
    ) -> Result<Option<EcTransferGuard>, EpdError> {
        let started = Instant::now();
        let request = prepared.request.clone();
        let options = request_options(&request)?;
        if items.is_empty() {
            prepared
                .media
                .as_mut()
                .expect("checked by prepare_grpc")
                .encoded = true;
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

        // Without a marker per item no encoder prompt exists; PD computes all.
        let markers: Option<Vec<Vec<u32>>> = markers.iter().cloned().collect();
        let groups = match &markers {
            Some(_) => {
                let start = self.next_encoder.fetch_add(items.len(), Ordering::Relaxed);
                encoder_groups(&items, self.config.encoder_urls.len(), start)
            }
            None => {
                tracing::warn!(%request_id, "model has no placeholder marker for a media item; PD computes every item");
                Vec::new()
            }
        };
        let replies = join_all(groups.iter().map(|(encoder, indices)| {
            let markers = markers.as_deref().unwrap_or_default();
            let ids = indices
                .iter()
                .flat_map(|&index| markers[index].iter().copied())
                .collect();
            let generate = encoder_request(
                &request,
                format!(
                    "{request_id}:{}:{:.6}",
                    indices[0], items[indices[0]].transfer_id
                ),
                ids,
                indices.iter().map(|&index| &items[index]),
                consumer_addr.map(String::as_str),
                &options,
            );
            async move {
                let generate = generate?;
                Ok::<_, EpdError>(
                    frontend
                        .generate(&self.config.encoder_urls[*encoder], generate)
                        .await,
                )
            }
        }))
        .await;

        let mut outcomes: Vec<Option<Published>> = vec![None; items.len()];
        let mut handles = Map::new();
        let mut failed_calls = 0;
        for ((encoder, indices), reply) in groups.iter().zip(replies) {
            let outcome = match reply? {
                Ok(reply) => split_reply(&reply, indices.len()),
                Err(Ok(status)) if rejects_request(&status) => {
                    return Err((
                        encoder_status(&status),
                        format!("Encoder rejected the request: {}", status.message()),
                    ));
                }
                Err(Ok(status)) => Err(format!("{:?}: {}", status.code(), status.message())),
                Err(Err(error)) => Err(error),
            };
            match outcome {
                Ok((published, params)) => {
                    for (&index, published) in indices.iter().zip(published) {
                        outcomes[index] = published;
                    }
                    handles.extend(params);
                }
                Err(error) => {
                    failed_calls += 1;
                    tracing::warn!(
                        %request_id,
                        encoder = %self.config.encoder_urls[*encoder],
                        items = indices.len(),
                        %error,
                        "gRPC EPD encoder call failed; PD computes its items"
                    );
                }
            }
        }

        let prompt = prepared.tokenized.token_ids.clone();
        let positions = markers.as_ref().and_then(|markers| {
            marker_positions(&prompt, markers)
                .inspect_err(|error| {
                    tracing::warn!(%request_id, %error, "EPD markers not found in PD's prompt; PD preprocesses every item");
                })
                .ok()
        });
        let assembled = assemble(
            &prompt,
            &items,
            markers.as_deref().unwrap_or_default(),
            positions.as_deref(),
            &outcomes,
            handles,
        );
        let media = prepared.media.as_mut().expect("checked by prepare_grpc");
        media.items = assembled.media;
        media.ec_transfer_params = assembled.ec_transfer_params;
        media.prompt_token_ids = assembled.prompt_token_ids;
        media.routing_token_ids = assembled.routing_token_ids;
        media.encoded = true;
        if let Some(pending) = pending.as_mut() {
            pending.retain(assembled.used_transfer_ids);
        }
        tracing::info!(
            %request_id,
            transport = "grpc",
            items = items.len(),
            encoder_calls = groups.len(),
            failed_calls,
            // reported: PD loads these from the EC; featured: of those, the
            // ones PD also does not preprocess.
            reported = assembled.reported,
            featured = assembled.featured,
            marker_prompt_tokens = prompt.len(),
            exact_prompt_tokens = media.routing_token_ids.as_ref().map(|ids| ids.len()),
            encode_ms = started.elapsed().as_secs_f64() * 1000.0,
            "EPD encoder stage complete"
        );
        Ok(pending)
    }
}

fn media_modality(modality: pb::Modality) -> MediaModality {
    match modality {
        pb::Modality::Video => MediaModality::Video,
        pb::Modality::Audio => MediaModality::Audio,
        _ => MediaModality::Image,
    }
}

type MediaOptions = (Option<prost_types::Struct>, Option<prost_types::Struct>);

/// `mm_processor_kwargs` and `media_io_kwargs`, which E and PD must share:
/// they change preprocessing and so the engine's item hash. PD gets them
/// from the request itself.
fn request_options(request: &ChatCompletionRequest) -> Result<MediaOptions, EpdError> {
    let option = |key| {
        crate::backend::convert::other_struct(request, key)
            .map_err(|e| (StatusCode::BAD_REQUEST, e))
    };
    Ok((option("mm_processor_kwargs")?, option("media_io_kwargs")?))
}

/// Media parts in request order. Only a caller's uuid is forwarded.
fn collect_items(body: &Value) -> Result<Vec<Item>, EpdError> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "messages must be an array".into()))?;
    let mut items = Vec::new();
    for (message_index, message) in messages.iter().enumerate() {
        let Some(content) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        for (content_index, part) in content.iter().enumerate() {
            let kind = part.get("type").and_then(Value::as_str).unwrap_or_default();
            let (modality, source) = match kind {
                "image_url" | "video_url" | "audio_url" => {
                    let modality = match kind {
                        "image_url" => pb::Modality::Image,
                        "video_url" => pb::Modality::Video,
                        _ => pb::Modality::Audio,
                    };
                    let url = part[kind]["url"]
                        .as_str()
                        .filter(|url| !url.is_empty())
                        .ok_or_else(|| {
                            (
                                StatusCode::BAD_REQUEST,
                                format!("{kind}.url must be a nonempty string"),
                            )
                        })?;
                    (modality, media_source(kind, url)?)
                }
                "input_audio" => {
                    let data = part
                        .pointer("/input_audio/data")
                        .and_then(Value::as_str)
                        .filter(|data| !data.is_empty())
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
                    (
                        pb::Modality::Audio,
                        pb::media_item::Source::DataUri(format!(
                            "data:audio/{format};base64,{data}"
                        )),
                    )
                }
                "image_embeds" | "audio_embeds" | "video_embeds" => {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        format!("{kind} is not supported by experimental gRPC EPD"),
                    ));
                }
                _ => continue,
            };
            items.push(Item {
                message_index,
                content_index,
                raw: pb::MediaItem {
                    modality: modality as i32,
                    source: Some(source),
                    uuid: part
                        .get("uuid")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    detail: part
                        .pointer("/image_url/detail")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    ..Default::default()
                },
                transfer_id: Uuid::new_v4().simple().to_string(),
            });
        }
    }
    Ok(items)
}

fn media_source(kind: &str, url: &str) -> Result<pb::media_item::Source, EpdError> {
    if url.starts_with("data:") {
        Ok(pb::media_item::Source::DataUri(url.to_string()))
    } else if ["http://", "https://", "file:"]
        .iter()
        .any(|scheme| url.starts_with(scheme))
    {
        // The encoder applies --allowed-media-domains and
        // --allowed-local-media-path.
        Ok(pb::media_item::Source::Url(url.to_string()))
    } else {
        Err((
            StatusCode::BAD_REQUEST,
            format!("{kind}.url must be an http(s), file or data URL on gRPC"),
        ))
    }
}

/// Reorder to the renderer's placeholder order; parts it dropped have no
/// marker and are not encoded. `None` keeps request order.
fn place_items(items: Vec<Item>, order: Option<&[MediaPartSource]>) -> Result<Vec<Item>, EpdError> {
    let Some(order) = order else {
        return Ok(items);
    };
    let mut by_source: HashMap<(usize, usize), Item> = items
        .into_iter()
        .map(|item| ((item.message_index, item.content_index), item))
        .collect();
    order
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
        .collect()
}

/// Per-item round-robin from `start`; images landing on one encoder share a call.
fn encoder_groups(items: &[Item], encoders: usize, start: usize) -> Vec<(usize, Vec<usize>)> {
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut image_group_of: HashMap<usize, usize> = HashMap::new();
    for (index, item) in items.iter().enumerate() {
        let encoder = start.wrapping_add(index) % encoders;
        if item.modality() == pb::Modality::Image {
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
    groups
}

/// A prompt of the items' bare markers, one output token, prompt ids and
/// placeholder ranges returned: the encoder needs the media, not the chat.
fn encoder_request<'a>(
    request: &ChatCompletionRequest,
    engine_request_id: String,
    ids: Vec<u32>,
    items: impl Iterator<Item = &'a Item>,
    consumer_addr: Option<&str>,
    (mm_processor_kwargs, media_io_kwargs): &MediaOptions,
) -> Result<pb::GenerateRequest, EpdError> {
    let items: Vec<&Item> = items.collect();
    let mut generate = pb::GenerateRequest {
        request_id: engine_request_id,
        model: request.model.clone().unwrap_or_default(),
        prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds { ids })),
        media: items.iter().map(|item| item.raw.clone()).collect(),
        mm_processor_kwargs: mm_processor_kwargs.clone(),
        media_io_kwargs: media_io_kwargs.clone(),
        response: Some(pb::ResponseOptions {
            prompt_token_ids: true,
            ..Default::default()
        }),
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
                "ec_items": items
                    .iter()
                    .map(|item| json!({"transfer_id": item.transfer_id}))
                    .collect::<Vec<_>>(),
            }),
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    }
    Ok(generate)
}

/// Split one encoder reply over its `items` items. Placeholder ranges come in
/// item order; an item whose key the encoder did not report is left to PD.
fn split_reply(reply: &pb::GenerateResponse, items: usize) -> Result<GroupOutcome, String> {
    let prompt = reply
        .prompt_info
        .as_ref()
        .ok_or("encoder response has no prompt info")?;
    let mut spans: Vec<&pb::PromptPlaceholder> = prompt.mm_placeholders.iter().collect();
    spans.sort_by_key(|span| span.offset);
    if spans.len() != items {
        return Err(format!(
            "encoder returned {} placeholder ranges for {items} items",
            spans.len()
        ));
    }
    // A marker-only prompt expands into exactly its items' ranges.
    let mut cursor = 0u64;
    let tiles = spans.iter().all(|span| {
        let starts_here = span.offset == cursor && span.length > 0;
        cursor = span.offset.saturating_add(span.length);
        starts_here
    }) && cursor == prompt.token_ids.len() as u64;
    let params = match reply
        .outputs
        .as_ref()
        .and_then(|outputs| outputs.finish_info.as_ref())
        .and_then(|finish| finish.ec_transfer_params.as_ref())
        .map(crate::backend::convert::proto_struct_to_json)
    {
        Some(Value::Object(params)) => params,
        _ => Map::new(),
    };
    let published = spans
        .iter()
        .map(|span| {
            let identifier = if span.identifier.is_empty() {
                span.mm_hash.clone()
            } else {
                span.identifier.clone()
            };
            let reported = params.get(&identifier)?;
            let metadata = reported
                .get("metadata")
                .and_then(Value::as_object)
                .filter(|metadata| !metadata.is_empty())
                .and_then(|metadata| flatten_metadata(metadata).ok());
            Some(Published {
                tokens: tiles.then(|| {
                    let start = span.offset as usize;
                    prompt.token_ids[start..start + span.length as usize].to_vec()
                }),
                is_embed: span.is_embed.clone(),
                mm_hash: if span.mm_hash.is_empty() {
                    identifier.clone()
                } else {
                    span.mm_hash.clone()
                },
                identifier,
                metadata,
            })
        })
        .collect();
    Ok((published, params))
}

/// Start of each item's marker in `prompt`. Items of one modality take that
/// modality's markers left to right, as the worker expands them.
fn marker_positions(prompt: &[u32], markers: &[Vec<u32>]) -> Result<Vec<usize>, String> {
    let mut lanes: Vec<(&[u32], VecDeque<usize>)> = Vec::new();
    for (index, marker) in markers.iter().enumerate() {
        if marker.is_empty() {
            return Err("empty placeholder marker".into());
        }
        match lanes
            .iter_mut()
            .find(|(lane, _)| *lane == marker.as_slice())
        {
            Some((_, indices)) => indices.push_back(index),
            None => lanes.push((marker, VecDeque::from([index]))),
        }
    }
    let mut positions = vec![0; markers.len()];
    let mut cursor = 0;
    while cursor < prompt.len() {
        let Some((marker, indices)) = lanes
            .iter_mut()
            .find(|(marker, _)| prompt[cursor..].starts_with(marker))
        else {
            cursor += 1;
            continue;
        };
        let index = indices
            .pop_front()
            .ok_or("prompt has more media placeholders than media items")?;
        positions[index] = cursor;
        cursor += marker.len();
    }
    if lanes.iter().any(|(_, indices)| !indices.is_empty()) {
        return Err("prompt has fewer media placeholders than media items".into());
    }
    let mut sorted = positions.clone();
    sorted.sort_unstable();
    if sorted != positions {
        return Err("media placeholders are not in placeholder order".into());
    }
    Ok(positions)
}

#[derive(Debug, Default)]
struct Assembled {
    media: Vec<pb::MediaItem>,
    ec_transfer_params: Option<Value>,
    prompt_token_ids: Option<Arc<[u32]>>,
    routing_token_ids: Option<Arc<[u32]>>,
    used_transfer_ids: Vec<String>,
    reported: usize,
    featured: usize,
}

/// PD's request. An item with metadata and a known expansion becomes that
/// expansion plus a metadata-only feature; the rest stay raw at their marker.
/// `positions` is `None` when PD's markers could not be located, and the
/// worker then expands every raw item itself.
fn assemble(
    prompt: &[u32],
    items: &[Item],
    markers: &[Vec<u32>],
    positions: Option<&[usize]>,
    outcomes: &[Option<Published>],
    handles: Map<String, Value>,
) -> Assembled {
    let mut featured: Vec<bool> = outcomes
        .iter()
        .map(|outcome| {
            positions.is_some()
                && outcome.as_ref().is_some_and(|published| {
                    published.tokens.is_some() && published.metadata.is_some()
                })
        })
        .collect();
    // The worker places mixed raw items at a single marker token only.
    let raw_need_one_token = featured.iter().any(|&f| f)
        && featured
            .iter()
            .zip(markers)
            .any(|(&featured, marker)| !featured && marker.len() != 1);
    if raw_need_one_token {
        featured.fill(false);
    }
    let mixed = featured.iter().any(|&f| f);

    let mut assembled = Assembled::default();
    let mut ec_items = Vec::new();
    let mut used_handles = Map::new();
    for (item, outcome) in items.iter().zip(outcomes) {
        if let Some(published) = outcome {
            ec_items.push(json!({
                "mm_hash": published.identifier,
                "transfer_id": item.transfer_id,
            }));
            if let Some(handle) = handles.get(&published.identifier) {
                used_handles.insert(published.identifier.clone(), handle.clone());
            }
            assembled.used_transfer_ids.push(item.transfer_id.clone());
            assembled.reported += 1;
        }
    }
    if !ec_items.is_empty() {
        used_handles.insert("ec_items".into(), Value::Array(ec_items));
        assembled.ec_transfer_params = Some(Value::Object(used_handles));
    }

    let Some(positions) = positions else {
        assembled.media = items.iter().map(|item| item.raw.clone()).collect();
        return assembled;
    };
    let mut dispatch = Vec::with_capacity(prompt.len());
    let mut routing = Vec::with_capacity(prompt.len());
    let mut routing_known = true;
    let mut cursor = 0;
    for (index, item) in items.iter().enumerate() {
        let (at, marker) = (positions[index], &markers[index]);
        dispatch.extend_from_slice(&prompt[cursor..at]);
        routing.extend_from_slice(&prompt[cursor..at]);
        cursor = at + marker.len();
        let published = outcomes[index].as_ref();
        match published.and_then(|published| published.tokens.as_deref()) {
            Some(tokens) => routing.extend_from_slice(tokens),
            None => {
                routing_known = false;
                routing.extend_from_slice(marker);
            }
        }
        if featured[index] {
            let published = published.expect("featured items are published");
            let tokens = published.tokens.as_deref().expect("featured items expand");
            let metadata = published
                .metadata
                .as_ref()
                .expect("featured items have metadata");
            assembled.media.push(pb::MediaItem {
                modality: item.raw.modality,
                source: Some(pb::media_item::Source::Features(
                    pb::PreprocessedMediaFeatures {
                        kwargs: None,
                        identifier: published.identifier.clone(),
                        offset: dispatch.len() as u64,
                        length: tokens.len() as u64,
                        mm_hash: Some(published.mm_hash.clone()),
                        is_embed: published.is_embed.clone(),
                        metadata: crate::backend::convert::json_to_proto_struct(&Value::Object(
                            metadata.clone(),
                        ))
                        .ok(),
                        tensors: Default::default(),
                    },
                )),
                ..Default::default()
            });
            dispatch.extend_from_slice(tokens);
            assembled.featured += 1;
        } else {
            assembled.media.push(pb::MediaItem {
                marker_offset: mixed.then_some(dispatch.len() as u64),
                ..item.raw.clone()
            });
            dispatch.extend_from_slice(marker);
        }
    }
    dispatch.extend_from_slice(&prompt[cursor..]);
    routing.extend_from_slice(&prompt[cursor..]);
    if mixed {
        assembled.prompt_token_ids = Some(dispatch.into());
    }
    if routing_known {
        assembled.routing_token_ids = Some(routing.into());
    }
    assembled
}

/// Encoder verdicts PD would reach too: the request, not the encoder, is at
/// fault, so PD computing the items would fail the same way.
fn rejects_request(status: &tonic::Status) -> bool {
    matches!(
        status.code(),
        tonic::Code::InvalidArgument | tonic::Code::OutOfRange | tonic::Code::FailedPrecondition
    )
}

/// The encoder's own verdict, as HTTP EPD forwards an encoder's HTTP status.
fn encoder_status(status: &tonic::Status) -> StatusCode {
    match status.code() {
        tonic::Code::InvalidArgument | tonic::Code::OutOfRange => StatusCode::BAD_REQUEST,
        tonic::Code::FailedPrecondition => StatusCode::UNPROCESSABLE_ENTITY,
        tonic::Code::NotFound => StatusCode::NOT_FOUND,
        tonic::Code::ResourceExhausted => StatusCode::TOO_MANY_REQUESTS,
        tonic::Code::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        tonic::Code::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
        _ => StatusCode::BAD_GATEWAY,
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests;
