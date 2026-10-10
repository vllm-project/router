//! Reconcile Prefill-owned prefix-cache usage in vLLM PD responses.

use bytes::Bytes;
use futures_util::{stream, Stream, StreamExt};
use serde_json::Value;
use std::io;
use std::pin::Pin;

// Chat/Completions usage frames are small, but a malformed upstream must not
// make the router retain an unbounded partial SSE event.
const MAX_SSE_FRAME_BYTES: usize = 8 * 1024 * 1024;

pub(super) fn is_supported_path(path: &str) -> bool {
    matches!(path, "/v1/chat/completions" | "/v1/completions")
}

pub(super) fn prefill_cached_tokens(prefill: Option<&Value>) -> Option<u64> {
    let value = prefill?.pointer("/usage/prompt_tokens_details/cached_tokens")?;
    let count = value.as_u64();
    if count.is_none() {
        tracing::warn!("Prefill cached_tokens has an invalid type or value");
    }
    count
}

/// Change only the cached-token detail. A missing Prefill value removes a
/// potentially misleading Decode value while retaining all sibling fields.
pub(super) fn reconcile_usage(
    response: &mut Value,
    cached: Option<u64>,
) -> Result<bool, &'static str> {
    let Some(usage) = response.get_mut("usage").and_then(Value::as_object_mut) else {
        return if cached.is_some() {
            Err("Decode response has no usage object")
        } else {
            Ok(false)
        };
    };

    match usage.get_mut("prompt_tokens_details") {
        Some(Value::Object(details)) => match cached {
            Some(count) => {
                let new_value = Value::from(count);
                if details.get("cached_tokens") == Some(&new_value) {
                    Ok(false)
                } else {
                    details.insert("cached_tokens".into(), new_value);
                    Ok(true)
                }
            }
            None => Ok(details.remove("cached_tokens").is_some()),
        },
        Some(Value::Null) | None => {
            if let Some(count) = cached {
                usage.insert(
                    "prompt_tokens_details".into(),
                    serde_json::json!({"cached_tokens": count}),
                );
                Ok(true)
            } else {
                Ok(false)
            }
        }
        Some(_) => Err("Decode prompt_tokens_details is not an object"),
    }
}

fn frame_end(buffer: &[u8]) -> Option<usize> {
    let mut line_start = 0;
    for (index, byte) in buffer.iter().enumerate() {
        if *byte == b'\n' {
            let content_end = if index > line_start && buffer[index - 1] == b'\r' {
                index - 1
            } else {
                index
            };
            if content_end == line_start {
                return Some(index + 1);
            }
            line_start = index + 1;
        }
    }
    None
}

fn rewrite_frame(frame: &[u8], cached: Option<u64>) -> Bytes {
    let mut data = Vec::new();
    let mut has_data = false;
    for line in frame.split_inclusive(|byte| *byte == b'\n') {
        let line = line.strip_suffix(b"\n").unwrap_or(line);
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if let Some(value) = line.strip_prefix(b"data:") {
            if has_data {
                data.push(b'\n');
            }
            data.extend_from_slice(value.strip_prefix(b" ").unwrap_or(value));
            has_data = true;
        }
    }

    let Ok(mut json) = serde_json::from_slice::<Value>(&data) else {
        return Bytes::copy_from_slice(frame);
    };
    let has_usage = json.get("usage").is_some_and(Value::is_object);
    if !has_usage {
        return Bytes::copy_from_slice(frame);
    }
    let is_final_usage = json
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty);
    let effective_cached = if is_final_usage { cached } else { None };
    match reconcile_usage(&mut json, effective_cached) {
        Ok(true) => {}
        Ok(false) => return Bytes::copy_from_slice(frame),
        Err(reason) => {
            tracing::warn!("Cannot rewrite PD SSE usage: {}", reason);
            return Bytes::copy_from_slice(frame);
        }
    }
    let Ok(encoded) = serde_json::to_vec(&json) else {
        return Bytes::copy_from_slice(frame);
    };

    let mut result = Vec::with_capacity(frame.len() + encoded.len());
    let mut wrote_data = false;
    for line in frame.split_inclusive(|byte| *byte == b'\n') {
        let content = line.strip_suffix(b"\n").unwrap_or(line);
        let content = content.strip_suffix(b"\r").unwrap_or(content);
        if content.starts_with(b"data:") {
            if !wrote_data {
                result.extend_from_slice(b"data: ");
                result.extend_from_slice(&encoded);
                if line.ends_with(b"\r\n") {
                    result.extend_from_slice(b"\r\n");
                } else if line.ends_with(b"\n") {
                    result.push(b'\n');
                }
                wrote_data = true;
            }
        } else {
            result.extend_from_slice(line);
        }
    }
    Bytes::from(result)
}

struct SseState<S> {
    source: Pin<Box<S>>,
    buffer: Vec<u8>,
    cached: Option<u64>,
    done: bool,
}

pub(super) fn rewrite_sse_stream<S, E>(
    source: S,
    cached: Option<u64>,
) -> impl Stream<Item = Result<Bytes, io::Error>> + Send
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    let state = SseState {
        source: Box::pin(source),
        buffer: Vec::new(),
        cached,
        done: false,
    };
    stream::unfold(state, |mut state| async move {
        loop {
            if state.done {
                return None;
            }
            if let Some(end) = frame_end(&state.buffer) {
                if end > MAX_SSE_FRAME_BYTES {
                    state.done = true;
                    return Some((
                        Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "PD SSE frame exceeds limit",
                        )),
                        state,
                    ));
                }
                let remainder = state.buffer.split_off(end);
                let frame = std::mem::replace(&mut state.buffer, remainder);
                let rewritten = rewrite_frame(&frame, state.cached);
                return Some((Ok(rewritten), state));
            }
            if state.buffer.len() > MAX_SSE_FRAME_BYTES {
                state.done = true;
                return Some((
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "PD SSE frame exceeds limit",
                    )),
                    state,
                ));
            }
            match state.source.as_mut().next().await {
                Some(Ok(chunk)) => state.buffer.extend_from_slice(&chunk),
                Some(Err(error)) => {
                    state.done = true;
                    return Some((Err(io::Error::other(error)), state));
                }
                None if state.buffer.is_empty() => return None,
                None => {
                    state.done = true;
                    return Some((Ok(Bytes::from(std::mem::take(&mut state.buffer))), state));
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::TryStreamExt;
    use serde_json::json;

    #[test]
    fn prefill_zero_and_missing_are_distinct() {
        assert_eq!(
            prefill_cached_tokens(Some(
                &json!({"usage": {"prompt_tokens_details": {"cached_tokens": 0}}})
            )),
            Some(0)
        );
        for value in [
            json!({}),
            json!({"usage": {"prompt_tokens_details": {"cached_tokens": -1}}}),
            json!({"usage": {"prompt_tokens_details": {"cached_tokens": "3"}}}),
            serde_json::from_str::<Value>(
                r#"{"usage":{"prompt_tokens_details":{"cached_tokens":18446744073709551616}}}"#,
            )
            .unwrap(),
        ] {
            assert_eq!(prefill_cached_tokens(Some(&value)), None);
        }
    }

    #[test]
    fn only_chat_and_completions_are_reconciled() {
        assert!(is_supported_path("/v1/chat/completions"));
        assert!(is_supported_path("/v1/completions"));
        assert!(!is_supported_path("/v1/responses"));
    }

    #[test]
    fn reconcile_preserves_decode_owned_usage_and_unknown_fields() {
        let mut response = json!({"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "prompt_tokens_details": {"cached_tokens": 100, "audio_tokens": 2}, "future_field": 7}, "metrics": {"time_to_first_token_ms": 12.5}});
        assert_eq!(reconcile_usage(&mut response, Some(64)), Ok(true));
        assert_eq!(
            response["usage"]["prompt_tokens_details"],
            json!({"cached_tokens": 64, "audio_tokens": 2})
        );
        assert_eq!(response["usage"]["future_field"], 7);
        assert_eq!(response["metrics"], json!({"time_to_first_token_ms": 12.5}));
        assert_eq!(reconcile_usage(&mut response, None), Ok(true));
        assert_eq!(
            response["usage"]["prompt_tokens_details"],
            json!({"audio_tokens": 2})
        );
    }

    #[test]
    fn zero_cache_creates_details_when_decode_does_not_report_them() {
        let mut response = json!({"usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6, "prompt_tokens_details": null}});
        assert_eq!(reconcile_usage(&mut response, Some(0)), Ok(true));
        assert_eq!(
            response["usage"]["prompt_tokens_details"],
            json!({"cached_tokens": 0})
        );
        assert_eq!(response["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn fragmented_sse_rewrites_only_final_usage_and_preserves_other_frames() {
        let input = b": keep\r\nevent: message\r\ndata: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":99}}}\r\n\r\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"prompt_tokens_details\":{\"cached_tokens\":100,\"audio_tokens\":2}},\"metrics\":{\"x\":1}}\r\n\r\ndata: [DONE]\r\n\r\n";
        let chunks = input
            .chunks(7)
            .map(|part| Ok::<_, io::Error>(Bytes::copy_from_slice(part)))
            .collect::<Vec<_>>();
        let output = rewrite_sse_stream(stream::iter(chunks), Some(64))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let output = output.concat();
        let output = String::from_utf8(output).unwrap();
        assert!(output.starts_with(": keep\r\nevent: message\r\n"));
        assert!(output.contains("\"audio_tokens\":2"));
        assert!(output.contains("\"cached_tokens\":64"));
        assert!(!output.contains("\"cached_tokens\":99"));
        assert!(!output.contains("\"cached_tokens\":100"));
        assert!(output.ends_with("data: [DONE]\r\n\r\n"));
    }

    #[tokio::test]
    async fn missing_prefill_cache_removes_decode_cache_from_usage_frame() {
        let input = Bytes::from_static(b"data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":30,\"audio_tokens\":1}}}\n\n");
        let output = rewrite_sse_stream(stream::iter([Ok::<_, io::Error>(input)]), None)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let output = String::from_utf8(output.concat()).unwrap();
        assert!(!output.contains("cached_tokens"));
        assert!(output.contains("audio_tokens"));
    }
}
