/// Regression tests for #322: user message content parts other than
/// `text`/`image_url` (vLLM's `audio_url`, `input_audio`, `video_url`) must be
/// forwarded verbatim instead of being silently replaced with `""`.
use serde_json::{json, Value};
use vllm_router_rs::protocols::spec::{ChatCompletionRequest, ChatMessage, UserMessageContent};

fn parse_user_content(content: Value) -> UserMessageContent {
    let raw = json!({
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "Follow the user instruction precisely."},
            {"role": "user", "content": content}
        ]
    })
    .to_string();

    let req: ChatCompletionRequest = serde_json::from_str(&raw).unwrap();
    req.messages
        .iter()
        .find_map(|m| match m {
            ChatMessage::User { content, .. } => Some(content.clone()),
            _ => None,
        })
        .expect("user message missing")
}

#[test]
fn test_multimodal_user_parts_survive_roundtrip() {
    let parts = [
        json!({"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAAA"}}),
        json!({"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}),
        json!({"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}}),
        json!({"type": "image_url", "image_url": {"url": "https://example.com/i.png"}}),
    ];

    for part in parts {
        let content = json!([part, {"type": "text", "text": "Return W0_TANGERINE_4931"}]);
        let parsed = parse_user_content(content.clone());

        assert!(
            matches!(&parsed, UserMessageContent::Parts(_)),
            "expected Parts for {content}, got {parsed:?}"
        );

        let reserialized = serde_json::to_value(&parsed).unwrap();
        assert_eq!(
            reserialized, content,
            "content parts were not preserved for {content}"
        );
    }
}

#[test]
fn test_multimodal_user_message_passes_validation() {
    let content = json!([
        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAAA"}},
        {"type": "text", "text": "hello"}
    ]);
    let raw = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": content}]
    })
    .to_string();

    let req: ChatCompletionRequest = serde_json::from_str(&raw).unwrap();
    req.validate_messages().unwrap();
}

#[test]
fn test_empty_user_parts_still_rejected() {
    let raw = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": []}]
    })
    .to_string();

    let req: ChatCompletionRequest = serde_json::from_str(&raw).unwrap();
    assert!(req.validate_messages().is_err());
}
