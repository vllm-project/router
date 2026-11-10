use super::*;
use crate::backend::pb::inference_server::{Inference, InferenceServer};
use std::sync::Mutex;

// Marker ids: 1 = image pad, 2 = video pad; 50.. are text.
const IMAGE: u32 = 1;
const VIDEO: u32 = 2;

fn item(modality: pb::Modality, uuid: &str) -> Item {
    Item {
        message_index: 0,
        content_index: 0,
        raw: pb::MediaItem {
            modality: modality as i32,
            source: Some(pb::media_item::Source::Url(format!("http://x/{uuid}"))),
            uuid: uuid.into(),
            ..Default::default()
        },
        transfer_id: format!("t-{uuid}"),
    }
}

fn published(id: &str, tokens: Option<Vec<u32>>, metadata: Option<Value>) -> Option<Published> {
    Some(Published {
        identifier: id.into(),
        mm_hash: format!("h-{id}"),
        tokens,
        is_embed: Vec::new(),
        metadata: metadata.map(|metadata| metadata.as_object().unwrap().clone()),
    })
}

fn features(media: &pb::MediaItem) -> &pb::PreprocessedMediaFeatures {
    match &media.source {
        Some(pb::media_item::Source::Features(features)) => features,
        other => panic!("expected features, got {other:?}"),
    }
}

#[test]
fn items_forward_only_caller_uuids_and_reject_embeds() {
    let body = json!({"messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://x/a.png", "detail": "low"}},
            {"type": "text", "text": "and"},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "ogg"}},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}, "uuid": "vid"},
        ]}
    ]});
    let items = collect_items(&body).unwrap();
    assert_eq!(items.len(), 3);
    assert_eq!(
        items
            .iter()
            .map(|item| item.raw.uuid.as_str())
            .collect::<Vec<_>>(),
        ["", "", "vid"]
    );
    assert_eq!(items[0].raw.detail.as_deref(), Some("low"));
    assert_eq!((items[0].message_index, items[0].content_index), (1, 0));
    assert_eq!(items[1].modality(), pb::Modality::Audio);
    assert!(matches!(
        &items[1].raw.source,
        Some(pb::media_item::Source::DataUri(uri)) if uri == "data:audio/ogg;base64,AAAA"
    ));
    assert_ne!(items[0].transfer_id, items[2].transfer_id);

    let embeds = json!({"messages": [{"role": "user", "content": [
        {"type": "image_embeds", "image_embeds": {"image_grid_thw": [1, 2, 2]}, "uuid": "u"}
    ]}]});
    assert_eq!(
        collect_items(&embeds).unwrap_err().0,
        StatusCode::BAD_REQUEST
    );
    let file = json!({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "file:///media/x.png"}}
    ]}]});
    assert!(matches!(
        &collect_items(&file).unwrap()[0].raw.source,
        Some(pb::media_item::Source::Url(url)) if url == "file:///media/x.png"
    ));
    let ftp = json!({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "ftp://host/x.png"}}
    ]}]});
    assert_eq!(collect_items(&ftp).unwrap_err().0, StatusCode::BAD_REQUEST);
}

#[test]
fn renderer_order_reorders_drops_and_rejects_unknown_parts() {
    let mut items = vec![
        item(pb::Modality::Image, "a"),
        item(pb::Modality::Video, "v"),
    ];
    items[1].content_index = 3;
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
    let placed = place_items(items.clone(), Some(&order)).unwrap();
    assert_eq!(placed[0].raw.uuid, "v");
    assert_eq!(placed[1].raw.uuid, "a");
    assert_eq!(
        place_items(items.clone(), Some(&order[..1])).unwrap().len(),
        1
    );
    let unknown = [MediaPartSource {
        message_index: 0,
        content_part_index: 1,
    }];
    assert!(place_items(items, Some(&unknown)).is_err());
}

#[test]
fn images_batch_per_encoder_and_audio_video_stay_singleton() {
    let items = [
        item(pb::Modality::Image, "a"),
        item(pb::Modality::Image, "b"),
        item(pb::Modality::Video, "v"),
        item(pb::Modality::Image, "c"),
        item(pb::Modality::Audio, "s"),
    ];
    assert_eq!(
        encoder_groups(&items, 2, 0),
        [(0, vec![0]), (1, vec![1, 3]), (0, vec![2]), (0, vec![4])]
    );
    assert_eq!(
        encoder_groups(&items, 1, 7),
        [(0, vec![0, 1, 3]), (0, vec![2]), (0, vec![4])]
    );
}

#[test]
fn markers_fill_per_modality_lanes_left_to_right() {
    let markers = [vec![IMAGE], vec![VIDEO], vec![IMAGE]];
    assert_eq!(
        marker_positions(&[50, IMAGE, 51, VIDEO, IMAGE, 52], &markers).unwrap(),
        [1, 3, 4]
    );
    // Multi-token markers match as a unit.
    assert_eq!(
        marker_positions(&[50, 7, 8, 51, 7, 8], &[vec![7, 8], vec![7, 8]]).unwrap(),
        [1, 4]
    );
    for (prompt, error) in [
        (&[IMAGE, VIDEO][..], "fewer"),
        (&[IMAGE, VIDEO, IMAGE, IMAGE][..], "more"),
        // The renderer listed the video first, the template put it second.
        (&[IMAGE, IMAGE, VIDEO][..], "order"),
    ] {
        let message = marker_positions(prompt, &markers).unwrap_err();
        assert!(message.contains(error), "{message}");
    }
}

fn reply(token_ids: Vec<u32>, spans: &[(&str, u64, u64)], params: Value) -> pb::GenerateResponse {
    pb::GenerateResponse {
        prompt_info: Some(pb::PromptInfo {
            token_ids,
            mm_placeholders: spans
                .iter()
                .map(|(id, offset, length)| pb::PromptPlaceholder {
                    modality: pb::Modality::Image as i32,
                    mm_hash: format!("h-{id}"),
                    offset: *offset,
                    length: *length,
                    identifier: (*id).into(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }),
        outputs: Some(pb::SequenceOutput {
            finish_info: Some(pb::FinishInfo {
                ec_transfer_params: crate::backend::convert::json_to_proto_struct(&params).ok(),
                ..Default::default()
            }),
            ..Default::default()
        }),
    }
}

#[test]
fn encoder_reply_splits_by_placeholder_ranges() {
    let params = json!({
        "a": {"metadata": {"image_grid_thw": [[1, 2, 4]]}, "item_indices": [0]},
        "b": {"metadata": {}, "item_indices": [1]},
    });
    // Out-of-order ranges are sorted; "c" was never reported.
    let response = reply(
        vec![1, 1, 1, 1, 1, 1, 1],
        &[("b", 2, 3), ("a", 0, 2), ("c", 5, 2)],
        params,
    );
    let (items, handles) = split_reply(&response, 3).unwrap();
    assert_eq!(
        items[0],
        published(
            "a",
            Some(vec![1, 1]),
            Some(json!({"image_grid_thw": [1, 2, 4]}))
        )
    );
    assert_eq!(items[1], published("b", Some(vec![1, 1, 1]), None));
    assert_eq!(items[2], None);
    assert_eq!(handles.len(), 2);

    // Ranges that do not tile the prompt give no expansion, only the report.
    let gap = reply(
        vec![1, 1, 9, 1],
        &[("a", 0, 2), ("b", 3, 1)],
        json!({"a": {}, "b": {}}),
    );
    let (items, _) = split_reply(&gap, 2).unwrap();
    assert!(items.iter().all(|p| p.as_ref().unwrap().tokens.is_none()));

    let short = reply(vec![1], &[("a", 0, 1)], json!({}));
    assert!(split_reply(&short, 2)
        .unwrap_err()
        .contains("1 placeholder ranges for 2"));
}

#[test]
fn featured_items_splice_and_raw_items_shift_to_their_marker() {
    // [text, image a, text, video v, text, image b]
    let prompt = [50, IMAGE, 51, VIDEO, 52, IMAGE];
    let items = [
        item(pb::Modality::Image, "a"),
        item(pb::Modality::Video, "v"),
        item(pb::Modality::Image, "b"),
    ];
    let markers = [vec![IMAGE], vec![VIDEO], vec![IMAGE]];
    let positions = marker_positions(&prompt, &markers).unwrap();
    let grid = Some(json!({"image_grid_thw": [1, 2, 2]}));
    let outcomes = [
        published("a", Some(vec![IMAGE; 4]), grid.clone()),
        // Reported without metadata: raw, but PD still loads it.
        published("v", Some(vec![60, VIDEO, VIDEO, 61]), None),
        published("b", Some(vec![IMAGE; 2]), grid.clone()),
    ];
    let handles = json!({"a": {"peer": 1}, "v": {"peer": 2}, "b": {"peer": 3}, "x": {}});
    let assembled = assemble(
        &prompt,
        &items,
        &markers,
        Some(&positions),
        &outcomes,
        handles.as_object().unwrap().clone(),
    );
    assert_eq!(
        assembled.prompt_token_ids.as_deref().unwrap(),
        [50, IMAGE, IMAGE, IMAGE, IMAGE, 51, VIDEO, 52, IMAGE, IMAGE]
    );
    assert_eq!(
        assembled.routing_token_ids.as_deref().unwrap(),
        [50, IMAGE, IMAGE, IMAGE, IMAGE, 51, 60, VIDEO, VIDEO, 61, 52, IMAGE, IMAGE]
    );
    let a = features(&assembled.media[0]);
    assert_eq!((a.offset, a.length, a.kwargs.is_none()), (1, 4, true));
    assert_eq!(a.mm_hash.as_deref(), Some("h-a"));
    assert_eq!(
        crate::backend::convert::proto_struct_to_json(a.metadata.as_ref().unwrap()),
        json!({"image_grid_thw": [1, 2, 2]})
    );
    // The raw video sits at its marker in the dispatched prompt.
    assert_eq!(assembled.media[1].marker_offset, Some(6));
    assert_eq!(assembled.media[1].uuid, "v");
    assert_eq!(features(&assembled.media[2]).offset, 8);
    assert_eq!((assembled.reported, assembled.featured), (3, 2));
    let params = assembled.ec_transfer_params.unwrap();
    assert_eq!(
        params["ec_items"],
        json!([
            {"mm_hash": "a", "transfer_id": "t-a"},
            {"mm_hash": "v", "transfer_id": "t-v"},
            {"mm_hash": "b", "transfer_id": "t-b"},
        ])
    );
    assert_eq!(params["v"], json!({"peer": 2}));
    assert!(params.get("x").is_none());
    assert_eq!(assembled.used_transfer_ids, ["t-a", "t-v", "t-b"]);
}

#[test]
fn failed_items_stay_raw_without_transfer_and_hide_the_exact_prompt() {
    let prompt = [50, IMAGE, IMAGE];
    let items = [
        item(pb::Modality::Image, "a"),
        item(pb::Modality::Image, "b"),
    ];
    let markers = [vec![IMAGE], vec![IMAGE]];
    let grid = Some(json!({"image_grid_thw": [1, 2, 2]}));
    let outcomes = [published("a", Some(vec![IMAGE; 4]), grid), None];
    let assembled = assemble(
        &prompt,
        &items,
        &markers,
        Some(&[1, 2]),
        &outcomes,
        Map::new(),
    );
    assert_eq!(
        assembled.prompt_token_ids.as_deref().unwrap(),
        [50, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE]
    );
    assert!(assembled.routing_token_ids.is_none());
    assert_eq!(assembled.media[1].marker_offset, Some(5));
    assert_eq!(
        assembled.ec_transfer_params.unwrap()["ec_items"],
        json!([{"mm_hash": "a", "transfer_id": "t-a"}])
    );
    assert_eq!(assembled.used_transfer_ids, ["t-a"]);

    let nothing = assemble(
        &prompt,
        &items,
        &markers,
        Some(&[1, 2]),
        &[None, None],
        Map::new(),
    );
    assert!(nothing.ec_transfer_params.is_none() && nothing.prompt_token_ids.is_none());
    assert!(nothing
        .media
        .iter()
        .all(|media| media.marker_offset.is_none()));
}

#[test]
fn unplaceable_raw_items_send_every_item_raw() {
    let grid = Some(json!({"image_grid_thw": [1, 2, 2]}));
    let items = [
        item(pb::Modality::Image, "a"),
        item(pb::Modality::Video, "v"),
    ];
    let outcomes = [published("a", Some(vec![IMAGE; 4]), grid.clone()), None];
    // A raw item behind a two-token marker cannot ride next to features.
    let markers = [vec![IMAGE], vec![8, 9]];
    let assembled = assemble(
        &[IMAGE, 8, 9],
        &items,
        &markers,
        Some(&[0, 1]),
        &outcomes,
        Map::new(),
    );
    assert_eq!(assembled.featured, 0);
    assert!(assembled.prompt_token_ids.is_none());
    assert!(assembled
        .media
        .iter()
        .all(|media| media.marker_offset.is_none()));
    assert_eq!(assembled.reported, 1);

    // Markers missing from PD's prompt: raw, but the reports still go.
    let markers = [vec![IMAGE], vec![VIDEO]];
    let assembled = assemble(&[50], &items, &markers, None, &outcomes, Map::new());
    assert_eq!(assembled.featured, 0);
    assert!(assembled.routing_token_ids.is_none());
    assert_eq!(assembled.media[0], items[0].raw);
    assert!(assembled.ec_transfer_params.is_some());
}

/// Encoder that expands each marker into `expansion` pads and reports
/// whatever `mode` says.
#[derive(Clone)]
struct MockEncoder {
    mode: &'static str,
    seen: Arc<Mutex<Vec<pb::GenerateRequest>>>,
}

const EXPANSION: usize = 3;

#[tonic::async_trait]
impl Inference for MockEncoder {
    type GenerateStreamStream = std::pin::Pin<
        Box<dyn futures_util::Stream<Item = Result<pb::GenerateResponse, tonic::Status>> + Send>,
    >;

    async fn generate(
        &self,
        request: tonic::Request<pb::GenerateRequest>,
    ) -> Result<tonic::Response<pb::GenerateResponse>, tonic::Status> {
        let request = request.into_inner();
        self.seen.lock().unwrap().push(request.clone());
        match self.mode {
            "invalid" => return Err(tonic::Status::invalid_argument("bad image")),
            "unavailable" => return Err(tonic::Status::unavailable("loading")),
            _ => {}
        }
        let Some(pb::generate_request::Prompt::TokenIds(prompt)) = request.prompt else {
            return Err(tonic::Status::invalid_argument("token ids required"));
        };
        let mut token_ids = Vec::new();
        let mut spans = Vec::new();
        let mut params = Map::new();
        for (index, (marker, media)) in prompt.ids.iter().zip(&request.media).enumerate() {
            let identifier = if media.uuid.is_empty() {
                let Some(pb::media_item::Source::Url(url)) = &media.source else {
                    unreachable!()
                };
                format!("hash-{url}")
            } else {
                media.uuid.clone()
            };
            spans.push(pb::PromptPlaceholder {
                modality: media.modality,
                mm_hash: format!("mm-{identifier}"),
                offset: token_ids.len() as u64,
                length: EXPANSION as u64,
                identifier: identifier.clone(),
                ..Default::default()
            });
            token_ids.extend(std::iter::repeat_n(*marker, EXPANSION));
            let entry = params.entry(identifier).or_insert_with(
                || json!({"metadata": {"image_grid_thw": [1, 2, 6]}, "item_indices": []}),
            );
            entry["item_indices"]
                .as_array_mut()
                .unwrap()
                .push(json!(index));
        }
        Ok(tonic::Response::new(pb::GenerateResponse {
            prompt_info: Some(pb::PromptInfo {
                token_ids,
                mm_placeholders: spans,
                ..Default::default()
            }),
            outputs: Some(pb::SequenceOutput {
                finish_info: Some(pb::FinishInfo {
                    ec_transfer_params: crate::backend::convert::json_to_proto_struct(
                        &Value::Object(params),
                    )
                    .ok(),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        }))
    }

    async fn generate_stream(
        &self,
        _request: tonic::Request<pb::GenerateRequest>,
    ) -> Result<tonic::Response<Self::GenerateStreamStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("encoders are called unary"))
    }
}

async fn spawn_encoder(mode: &'static str) -> (String, Arc<Mutex<Vec<pb::GenerateRequest>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("grpc://{}", listener.local_addr().unwrap());
    let service = InferenceServer::new(MockEncoder {
        mode,
        seen: seen.clone(),
    });
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(service)
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    (url, seen)
}

async fn prepared(prompt: Vec<u32>, body: Value) -> (EngineFrontend, PreparedRequest) {
    let frontend = EngineFrontend::new();
    frontend.pin_test_token_ids(prompt);
    let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
    let prepared = frontend.prepare_with_media(request).await.unwrap();
    (frontend, prepared)
}

fn chat() -> Value {
    json!({
        "model": "m",
        "mm_processor_kwargs": {"max_pixels": 50176},
        "media_io_kwargs": {"video": {"num_frames": 4}},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://x/a.png"}},
            {"type": "video_url", "video_url": {"url": "http://x/v.mp4"}, "uuid": "vid"},
            {"type": "image_url", "image_url": {"url": "http://x/a.png"}},
            {"type": "text", "text": "compare"},
        ]}],
    })
}

async fn run(
    encoders: Vec<String>,
) -> (PreparedRequest, Result<Option<EcTransferGuard>, EpdError>) {
    let prompt = vec![50, IMAGE, VIDEO, IMAGE, 51];
    let (frontend, mut prepared) = prepared(prompt, chat()).await;
    let stage = EncoderStage::new(EpdConfig {
        encoder_urls: encoders,
        consumer_zmq_addrs: Default::default(),
    });
    let body = serde_json::to_value(prepared.request.as_ref()).unwrap();
    let items = collect_items(&body).unwrap();
    let markers = [Some(vec![IMAGE]), Some(vec![VIDEO]), Some(vec![IMAGE])];
    let result = stage
        .encode_items(&frontend, &mut prepared, items, &markers, "", "req")
        .await;
    (prepared, result)
}

#[tokio::test]
async fn stage_sends_marker_prompts_and_pd_gets_metadata_only_features() {
    let (url, seen) = spawn_encoder("normal").await;
    let (prepared, result) = run(vec![url]).await;
    assert!(result.unwrap().is_none());

    let seen = seen.lock().unwrap();
    // Images [a, a] share a call; the video has its own.
    assert_eq!(seen.len(), 2);
    let images = seen.iter().find(|r| r.media.len() == 2).unwrap();
    let Some(pb::generate_request::Prompt::TokenIds(ids)) = &images.prompt else {
        panic!("token ids expected");
    };
    assert_eq!(ids.ids, [IMAGE, IMAGE]);
    assert!(images.media.iter().all(|media| media.uuid.is_empty()));
    assert_eq!(images.stopping.as_ref().unwrap().max_new_tokens, 1);
    assert!(images.response.as_ref().unwrap().prompt_token_ids);
    assert_eq!(
        crate::backend::convert::proto_struct_to_json(images.mm_processor_kwargs.as_ref().unwrap()),
        json!({"max_pixels": 50176})
    );
    let video = seen.iter().find(|r| r.media.len() == 1).unwrap();
    assert_eq!(video.media[0].uuid, "vid");
    assert!(video.media_io_kwargs.is_some());

    let media = prepared.media.as_ref().unwrap();
    assert!(media.encoded);
    let expected: Vec<u32> = [
        vec![50],
        vec![IMAGE; 3],
        vec![VIDEO; 3],
        vec![IMAGE; 3],
        vec![51],
    ]
    .concat();
    assert_eq!(media.prompt_token_ids.as_deref().unwrap(), expected);
    assert_eq!(media.routing_token_ids.as_deref().unwrap(), expected);
    let offsets: Vec<u64> = media.items.iter().map(|m| features(m).offset).collect();
    assert_eq!(offsets, [1, 4, 7]);
    assert_eq!(features(&media.items[1]).identifier, "vid");
    let params = media.ec_transfer_params.as_ref().unwrap();
    let hashes: Vec<&str> = params["ec_items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["mm_hash"].as_str().unwrap())
        .collect();
    assert_eq!(
        hashes,
        ["hash-http://x/a.png", "vid", "hash-http://x/a.png"]
    );

    // What dispatch sends: the spliced prompt, features and options.
    let mut proto = crate::backend::convert::chat_to_generate_request(
        &prepared.request,
        prepared.tokenized.token_ids.to_vec(),
        "r".into(),
    )
    .unwrap();
    media.apply(&mut proto).unwrap();
    assert!(matches!(
        proto.prompt,
        Some(pb::generate_request::Prompt::TokenIds(ref ids)) if ids.ids == expected
    ));
    assert!(proto.kv.unwrap().ec_transfer_params.is_some());
    assert!(proto.mm_processor_kwargs.is_some());
}

#[tokio::test]
async fn unavailable_encoders_leave_their_items_to_pd() {
    let (down, _) = spawn_encoder("unavailable").await;
    let (up, _) = spawn_encoder("normal").await;
    // Round-robin from 0: images a and a' on encoder 0 and 0, video on 1.
    let (prepared, result) = run(vec![down, up]).await;
    assert!(result.unwrap().is_none());
    let media = prepared.media.as_ref().unwrap();
    // a (index 0) and a' (index 2) land on the failed encoder 0.
    assert_eq!(media.items[0].marker_offset, Some(1));
    assert!(matches!(
        media.items[1].source,
        Some(pb::media_item::Source::Features(_))
    ));
    assert_eq!(media.items[2].marker_offset, Some(5));
    assert!(media.routing_token_ids.is_none());
    let ec_items = &media.ec_transfer_params.as_ref().unwrap()["ec_items"];
    assert_eq!(ec_items.as_array().unwrap().len(), 1);
    assert_eq!(ec_items[0]["mm_hash"], "vid");
}

#[tokio::test]
async fn encoder_rejections_fail_the_request() {
    let (url, _) = spawn_encoder("invalid").await;
    let (prepared, result) = run(vec![url]).await;
    let error = result.err().unwrap();
    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(error.1.contains("bad image"));
    assert!(!prepared.media.as_ref().unwrap().encoded);
}

#[tokio::test]
async fn unreachable_encoders_degrade_to_raw_media() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("grpc://{}", listener.local_addr().unwrap());
    drop(listener);
    let (prepared, result) = run(vec![url]).await;
    assert!(result.unwrap().is_none());
    let media = prepared.media.as_ref().unwrap();
    assert!(media.ec_transfer_params.is_none());
    assert!(media.prompt_token_ids.is_none());
    assert!(media.items.iter().all(|item| item.marker_offset.is_none()
        && !matches!(item.source, Some(pb::media_item::Source::Features(_)))));
}

#[tokio::test]
async fn mooncake_pushes_to_the_selected_dp_rank() {
    let (url, seen) = spawn_encoder("normal").await;
    let stage = EncoderStage::new(EpdConfig {
        encoder_urls: vec![url],
        consumer_zmq_addrs: [
            ("grpc://p:9000@0".into(), "tcp://p:5000".into()),
            ("grpc://p:9000@1".into(), "tcp://p:5001".into()),
        ]
        .into(),
    });
    let encode = |consumer: &'static str| {
        let stage = &stage;
        async move {
            let (frontend, mut prepared) = prepared(
                vec![50, IMAGE, 51],
                json!({
                    "model": "m",
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "http://x/a.png"}},
                    ]}],
                }),
            )
            .await;
            let body = serde_json::to_value(prepared.request.as_ref()).unwrap();
            let items = collect_items(&body).unwrap();
            stage
                .encode_items(
                    &frontend,
                    &mut prepared,
                    items,
                    &[Some(vec![IMAGE])],
                    consumer,
                    "req",
                )
                .await
        }
    };

    let mut guard = encode("grpc://p:9000@1").await.unwrap().unwrap();
    guard.disarm();
    let sent = seen.lock().unwrap()[0]
        .kv
        .clone()
        .unwrap()
        .ec_transfer_params;
    let sent = crate::backend::convert::proto_struct_to_json(sent.as_ref().unwrap());
    assert_eq!(sent["consumer_zmq"], "tcp://p:5001");

    // The unranked URL names no replica, so it has no control address.
    let error = encode("grpc://p:9000").await.err().unwrap();
    assert_eq!(error.0, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn caller_ec_transfer_params_are_rejected() {
    let mut body = chat();
    body["ec_transfer_params"] = json!({"ec_items": []});
    let (frontend, mut prepared) = prepared(vec![IMAGE], body).await;
    let stage = EncoderStage::new(EpdConfig {
        encoder_urls: vec!["grpc://127.0.0.1:1".into()],
        consumer_zmq_addrs: Default::default(),
    });
    let error = stage
        .prepare_grpc(&frontend, &mut prepared, "", None)
        .await
        .err()
        .unwrap();
    assert_eq!(error.0, StatusCode::BAD_REQUEST);
}
