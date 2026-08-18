//! Usage merging utilities for prefill/decode disaggregation.
//!
//! A decode worker reports KV transferred from the prefill worker as cached
//! tokens. That is correct from the decode worker's point of view, but it makes
//! a cold end-to-end PD request look like a 100% prefix-cache hit. The router
//! has both responses, so it must take input/cache accounting from prefill and
//! output accounting from decode.

use bytes::Bytes;
use futures_util::{stream, Stream, StreamExt};
use serde_json::{Map, Value};
use std::pin::Pin;

type UpstreamByteStream =
    Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static>>;

struct SseRewriteState {
    upstream: UpstreamByteStream,
    buffer: Vec<u8>,
    prefill_response: Option<Value>,
    finished: bool,
}

fn usage_object(value: &Value) -> Option<&Map<String, Value>> {
    value.get("usage").and_then(Value::as_object).or_else(|| {
        value
            .get("response")
            .and_then(|response| response.get("usage"))
            .and_then(Value::as_object)
    })
}

fn usage_object_mut(value: &mut Value) -> Option<&mut Map<String, Value>> {
    if value.get("usage").and_then(Value::as_object).is_some() {
        return value.get_mut("usage").and_then(Value::as_object_mut);
    }

    value
        .get_mut("response")
        .and_then(|response| response.get_mut("usage"))
        .and_then(Value::as_object_mut)
}

fn merge_pd_usage_inner(
    prefill_response: &Value,
    decode_response: &mut Value,
    copy_details: bool,
) -> bool {
    let Some(prefill_usage) = usage_object(prefill_response) else {
        return false;
    };
    let Some(decode_usage) = usage_object_mut(decode_response) else {
        return false;
    };

    let (input_key, details_key, output_key) = if prefill_usage.contains_key("prompt_tokens") {
        (
            "prompt_tokens",
            "prompt_tokens_details",
            "completion_tokens",
        )
    } else if prefill_usage.contains_key("input_tokens") {
        ("input_tokens", "input_tokens_details", "output_tokens")
    } else {
        return false;
    };

    let Some(input_tokens) = prefill_usage.get(input_key).cloned() else {
        return false;
    };
    decode_usage.insert(input_key.to_string(), input_tokens.clone());

    if copy_details {
        if let Some(details) = prefill_usage.get(details_key) {
            decode_usage.insert(details_key.to_string(), details.clone());
        } else {
            // Never leave decode-side external KV transfer masquerading as an
            // end-to-end prefix-cache hit when prefill did not report details.
            decode_usage.remove(details_key);
        }
    }

    if let (Some(input), Some(output)) = (
        input_tokens.as_u64(),
        decode_usage.get(output_key).and_then(Value::as_u64),
    ) {
        if let Some(total) = input.checked_add(output) {
            decode_usage.insert("total_tokens".to_string(), Value::from(total));
        }
    }

    true
}

/// Merge end-to-end PD usage into a non-streaming decode response.
///
/// Input counts and cache details come from prefill. Output counts stay on the
/// decode response, so the prefill worker's forced one-token output is never
/// exposed or billed.
pub fn merge_pd_usage(prefill_response: &Value, decode_response: &mut Value) -> bool {
    merge_pd_usage_inner(prefill_response, decode_response, true)
}

/// Keep only the small usage portion of a prefill response for the lifetime of
/// a decode stream. In particular, do not retain large KV-transfer block lists.
pub fn usage_snapshot(prefill_response: &Value) -> Option<Value> {
    usage_object(prefill_response).cloned().map(|usage| {
        Value::Object(Map::from_iter([(
            "usage".to_string(),
            Value::Object(usage),
        )]))
    })
}

fn is_final_usage_event(value: &Value) -> bool {
    // Chat/Completions emits a final usage-only chunk with an empty choices
    // array. Responses API carries the final usage under `response.usage`.
    value
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty)
        || value.get("response").is_some()
        || usage_object(value).is_some_and(|usage| {
            usage.contains_key("prompt_tokens_details")
                || usage.contains_key("input_tokens_details")
        })
}

fn find_sse_event_end(buffer: &[u8]) -> Option<usize> {
    let lf = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|index| index + 2);
    let crlf = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4);

    match (lf, crlf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn rewrite_sse_event(event: &[u8], prefill_response: &Value) -> Vec<u8> {
    let mut line_start = 0;

    while line_start < event.len() {
        let line_end = event[line_start..]
            .iter()
            .position(|byte| *byte == b'\n')
            .map(|offset| line_start + offset)
            .unwrap_or(event.len());
        let content_end = if line_end > line_start && event[line_end - 1] == b'\r' {
            line_end - 1
        } else {
            line_end
        };
        let line = &event[line_start..content_end];

        if let Some(payload) = line.strip_prefix(b"data:") {
            let leading_spaces = payload.iter().take_while(|byte| **byte == b' ').count();
            let json_start = line_start + b"data:".len() + leading_spaces;
            let json_bytes = &event[json_start..content_end];

            if json_bytes != b"[DONE]" {
                if let Ok(mut decode_json) = serde_json::from_slice::<Value>(json_bytes) {
                    let copy_details = is_final_usage_event(&decode_json);
                    if merge_pd_usage_inner(prefill_response, &mut decode_json, copy_details) {
                        let mut rewritten = Vec::with_capacity(event.len() + 64);
                        rewritten.extend_from_slice(&event[..json_start]);
                        if serde_json::to_writer(&mut rewritten, &decode_json).is_ok() {
                            rewritten.extend_from_slice(&event[content_end..]);
                            return rewritten;
                        }
                    }
                }
            }
        }

        if line_end == event.len() {
            break;
        }
        line_start = line_end + 1;
    }

    event.to_vec()
}

/// Rewrite usage in an SSE byte stream while preserving streaming behavior.
///
/// Network chunks are not guaranteed to align with SSE events, so this keeps a
/// small buffer until a complete blank-line-delimited event is available.
pub fn rewrite_sse_usage_stream<S>(
    upstream: S,
    prefill_response: Option<Value>,
) -> impl Stream<Item = Result<Bytes, reqwest::Error>>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let state = SseRewriteState {
        upstream: Box::pin(upstream),
        buffer: Vec::new(),
        prefill_response,
        finished: false,
    };

    stream::unfold(state, |mut state| async move {
        loop {
            if let Some(event_end) = find_sse_event_end(&state.buffer) {
                let event: Vec<u8> = state.buffer.drain(..event_end).collect();
                let event = state
                    .prefill_response
                    .as_ref()
                    .map(|prefill| rewrite_sse_event(&event, prefill))
                    .unwrap_or(event);
                return Some((Ok(Bytes::from(event)), state));
            }

            if state.finished {
                if state.buffer.is_empty() {
                    return None;
                }
                let tail = std::mem::take(&mut state.buffer);
                let tail = state
                    .prefill_response
                    .as_ref()
                    .map(|prefill| rewrite_sse_event(&tail, prefill))
                    .unwrap_or(tail);
                return Some((Ok(Bytes::from(tail)), state));
            }

            match state.upstream.next().await {
                Some(Ok(chunk)) => state.buffer.extend_from_slice(&chunk),
                Some(Err(error)) => {
                    state.finished = true;
                    state.buffer.clear();
                    return Some((Err(error), state));
                }
                None => state.finished = true,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;
    use serde_json::json;

    #[test]
    fn merges_chat_and_completions_usage_from_prefill_and_decode() {
        let prefill = json!({
            "usage": {
                "prompt_tokens": 186,
                "completion_tokens": 1,
                "total_tokens": 187,
                "prompt_tokens_details": {"cached_tokens": 64}
            }
        });
        let mut decode = json!({
            "usage": {
                "prompt_tokens": 999,
                "completion_tokens": 8,
                "total_tokens": 1007,
                "prompt_tokens_details": {"cached_tokens": 999}
            }
        });

        assert!(merge_pd_usage(&prefill, &mut decode));
        assert_eq!(decode["usage"]["prompt_tokens"], 186);
        assert_eq!(decode["usage"]["completion_tokens"], 8);
        assert_eq!(decode["usage"]["total_tokens"], 194);
        assert_eq!(
            decode["usage"]["prompt_tokens_details"]["cached_tokens"],
            64
        );
    }

    #[test]
    fn merges_responses_api_usage() {
        let prefill = json!({
            "usage": {
                "input_tokens": 98,
                "output_tokens": 1,
                "total_tokens": 99,
                "input_tokens_details": {"cached_tokens": 32}
            }
        });
        let mut decode = json!({
            "usage": {
                "input_tokens": 98,
                "output_tokens": 8,
                "total_tokens": 106,
                "input_tokens_details": {"cached_tokens": 98},
                "output_tokens_details": {"reasoning_tokens": 3}
            }
        });

        assert!(merge_pd_usage(&prefill, &mut decode));
        assert_eq!(decode["usage"]["input_tokens"], 98);
        assert_eq!(decode["usage"]["output_tokens"], 8);
        assert_eq!(decode["usage"]["total_tokens"], 106);
        assert_eq!(decode["usage"]["input_tokens_details"]["cached_tokens"], 32);
        assert_eq!(
            decode["usage"]["output_tokens_details"]["reasoning_tokens"],
            3
        );
    }

    #[test]
    fn removes_decode_cache_details_when_prefill_did_not_report_them() {
        let prefill = json!({
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
        });
        let mut decode = json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 10}
            }
        });

        assert!(merge_pd_usage(&prefill, &mut decode));
        assert!(decode["usage"].get("prompt_tokens_details").is_none());
    }

    #[tokio::test]
    async fn rewrites_split_chat_or_completions_streaming_usage_event() {
        let prefill = json!({
            "usage": {
                "prompt_tokens": 127,
                "completion_tokens": 1,
                "total_tokens": 128,
                "prompt_tokens_details": {"cached_tokens": 0}
            }
        });
        let upstream = stream::iter(vec![
            Ok::<Bytes, reqwest::Error>(Bytes::from_static(
                b"data: {\"choices\":[{\"index\":0}],\"usage\":{\"prompt_tokens\":999,\"completion_tokens\":1,\"total_tokens\":1000}}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":999,",
            )),
            Ok::<Bytes, reqwest::Error>(Bytes::from_static(
                b"\"completion_tokens\":8,\"total_tokens\":135,\"prompt_tokens_details\":{\"cached_tokens\":127}}}\n\ndata: [DONE]\n\n",
            )),
        ]);

        let output = rewrite_sse_usage_stream(upstream, Some(prefill))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(Result::unwrap)
            .fold(Vec::new(), |mut output, chunk| {
                output.extend_from_slice(&chunk);
                output
            });
        let output = String::from_utf8(output).unwrap();

        assert!(output.contains("\"completion_tokens\":8"));
        assert!(output.contains("\"prompt_tokens\":127"));
        assert!(output.contains("\"total_tokens\":135"));
        assert!(output.contains("\"cached_tokens\":0"));
        assert!(!output.contains("\"cached_tokens\":127"));
        assert!(!output.contains("\"prompt_tokens\":999"));
        assert!(output.ends_with("data: [DONE]\n\n"));
    }

    #[test]
    fn rewrites_nested_responses_api_streaming_event() {
        let prefill = json!({
            "usage": {
                "input_tokens": 50,
                "output_tokens": 1,
                "total_tokens": 51,
                "input_tokens_details": {"cached_tokens": 16}
            }
        });
        let event = b"event: response.completed\r\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":999,\"output_tokens\":7,\"total_tokens\":1006,\"input_tokens_details\":{\"cached_tokens\":999},\"output_tokens_details\":{\"reasoning_tokens\":2}}}}\r\n\r\n";

        let rewritten = rewrite_sse_event(event, &prefill);
        let rewritten = String::from_utf8(rewritten).unwrap();

        assert!(rewritten.contains("\"cached_tokens\":16"));
        assert!(rewritten.contains("\"input_tokens\":50"));
        assert!(rewritten.contains("\"output_tokens\":7"));
        assert!(rewritten.contains("\"total_tokens\":57"));
        assert!(rewritten.contains("\"reasoning_tokens\":2"));
        assert!(!rewritten.contains("\"input_tokens\":999"));
        assert!(rewritten.ends_with("\r\n\r\n"));
    }

    #[test]
    fn usage_snapshot_drops_kv_transfer_metadata() {
        let prefill = json!({
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            "kv_transfer_params": {"remote_block_ids": [1, 2, 3]}
        });

        let snapshot = usage_snapshot(&prefill).unwrap();
        assert_eq!(snapshot["usage"]["prompt_tokens"], 10);
        assert!(snapshot.get("kv_transfer_params").is_none());
    }
}
