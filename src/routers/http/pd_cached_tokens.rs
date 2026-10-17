//! Reconcile Prefill-owned prefix-cache usage in vLLM PD responses.

use bytes::{Buf, Bytes};
use futures_util::{stream, Stream, StreamExt};
use serde_json::Value;
use std::io;
use std::pin::Pin;

// Count upstream bytes per frame, including its LF/CRLF blank-line delimiter.
// Exactly the limit is accepted; the next byte terminates the body stream with
// InvalidData (the committed HTTP status cannot change). This bounds only the
// frame accumulator: transport chunks, JSON and rewritten output also use memory.
// Use a 1 MiB frame budget for supported PD streaming workloads.
// HTTP error bodies have a separate policy; they never enter this transformer.
const MAX_SSE_FRAME_BYTES: usize = 1024 * 1024;

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

#[derive(Default)]
struct FrameScanner {
    line_bytes: usize,
    last_was_cr: bool,
}

impl FrameScanner {
    // Scan only newly received bytes; remember a partial line across chunks.
    fn frame_end(&mut self, bytes: &[u8]) -> Option<usize> {
        let mut offset = 0;
        while let Some(newline) = bytes[offset..].iter().position(|byte| *byte == b'\n') {
            self.line_bytes += newline;
            if newline > 0 {
                self.last_was_cr = bytes[offset + newline - 1] == b'\r';
            }
            let blank = self.line_bytes == 0 || (self.line_bytes == 1 && self.last_was_cr);
            self.line_bytes = 0;
            self.last_was_cr = false;
            offset += newline + 1;
            if blank {
                return Some(offset);
            }
        }
        self.line_bytes += bytes.len() - offset;
        if offset < bytes.len() {
            self.last_was_cr = bytes.last() == Some(&b'\r');
        }
        None
    }
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
    pending: Bytes,
    scanner: FrameScanner,
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
    rewrite_sse_stream_with_limit(source, cached, MAX_SSE_FRAME_BYTES)
}

// An explicit limit lets tests exercise frame boundaries with smaller inputs.
fn rewrite_sse_stream_with_limit<S, E>(
    source: S,
    cached: Option<u64>,
    max_frame_bytes: usize,
) -> impl Stream<Item = Result<Bytes, io::Error>> + Send
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    let state = SseState {
        source: Box::pin(source),
        buffer: Vec::new(),
        pending: Bytes::new(),
        scanner: FrameScanner::default(),
        cached,
        done: false,
    };
    stream::unfold(state, move |mut state| async move {
        loop {
            if state.done {
                return None;
            }
            if !state.pending.is_empty() {
                let remaining = max_frame_bytes - state.buffer.len();
                if remaining == 0 {
                    state.done = true;
                    state.buffer = Vec::new();
                    state.pending = Bytes::new();
                    return Some((
                        Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "PD SSE frame exceeds limit",
                        )),
                        state,
                    ));
                }
                let scan_len = remaining.min(state.pending.len());
                let end = state.scanner.frame_end(&state.pending[..scan_len]);
                let consumed = end.unwrap_or(scan_len);
                let needed = state.buffer.len() + consumed;
                if needed > state.buffer.capacity() {
                    let capacity = needed
                        .max(state.buffer.capacity().saturating_mul(2))
                        .min(max_frame_bytes);
                    // Prevent Vec's geometric growth from exceeding the limit.
                    state.buffer.reserve_exact(capacity - state.buffer.len());
                }
                state.buffer.extend_from_slice(&state.pending[..consumed]);
                state.pending.advance(consumed);
                if state.pending.is_empty() {
                    // An empty Bytes slice can still own the whole transport
                    // allocation. Release it before waiting for another chunk.
                    state.pending = Bytes::new();
                }
                if end.is_some() {
                    let rewritten = rewrite_frame(&state.buffer, state.cached);
                    state.buffer.clear();
                    return Some((Ok(rewritten), state));
                }
                continue;
            }
            match state.source.as_mut().next().await {
                Some(Ok(chunk)) => state.pending = chunk,
                Some(Err(error)) => {
                    state.done = true;
                    state.buffer = Vec::new();
                    return Some((Err(io::Error::other(error)), state));
                }
                None if state.buffer.is_empty() => return None,
                None => {
                    // Preserve the existing EOF behavior: an in-limit final
                    // fragment is forwarded verbatim, without synthesizing a
                    // delimiter or trying to rewrite an incomplete event.
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

    fn comment_frame(size: usize, newline: &[u8]) -> Bytes {
        let mut frame = vec![b'x'; size - newline.len() * 2];
        frame[0] = b':';
        frame.extend_from_slice(newline);
        frame.extend_from_slice(newline);
        Bytes::from(frame)
    }

    fn chunks(
        bytes: &Bytes,
        size: usize,
    ) -> impl Stream<Item = Result<Bytes, io::Error>> + Send + 'static {
        let bytes = bytes.clone();
        stream::unfold(0, move |offset| {
            let bytes = bytes.clone();
            async move {
                if offset == bytes.len() {
                    None
                } else {
                    let end = (offset + size).min(bytes.len());
                    Some((Ok(bytes.slice(offset..end)), end))
                }
            }
        })
    }

    #[tokio::test]
    async fn sse_limit_counts_delimiters_and_accepts_exact_boundary() {
        let limit = MAX_SSE_FRAME_BYTES;
        for newline in [b"\n".as_slice(), b"\r\n".as_slice()] {
            for size in [limit - 1, limit, limit + 1] {
                let frame = comment_frame(size, newline);
                for chunk_size in [8191, size] {
                    let output =
                        rewrite_sse_stream_with_limit(chunks(&frame, chunk_size), None, limit)
                            .try_collect::<Vec<_>>()
                            .await;
                    if size <= limit {
                        assert_eq!(output.unwrap().concat(), frame);
                    } else {
                        assert_eq!(output.unwrap_err().kind(), io::ErrorKind::InvalidData);
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn sse_limits_each_frame_not_the_chunk_or_total_stream() {
        let limit = 128;
        let frame = comment_frame(limit, b"\r\n");
        let input = Bytes::from(frame.repeat(1024));
        let output = rewrite_sse_stream_with_limit(chunks(&input, input.len()), None, limit)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(output.len(), 1024);
        assert_eq!(output.concat(), input);
    }

    #[tokio::test]
    async fn sse_overflow_delivers_previous_frames_then_one_error_and_stops_reading() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let limit = 128;
        let prefix = Bytes::from_static(b"data: {\"choices\":[{\"text\":\"ok\"}]}\n\n");
        let oversized = comment_frame(limit + 1, b"\n");
        let input =
            Bytes::from([prefix.as_ref(), oversized.as_ref(), b"data: [DONE]\n\n"].concat());
        let polls = Arc::new(AtomicUsize::new(0));
        let observed = polls.clone();
        let source = stream::iter([
            Ok::<_, io::Error>(input),
            Ok(Bytes::from_static(b"never read")),
        ])
        .inspect(move |_| {
            observed.fetch_add(1, Ordering::Relaxed);
        });
        let output = rewrite_sse_stream_with_limit(source, None, limit);
        futures_util::pin_mut!(output);
        assert_eq!(output.next().await.unwrap().unwrap(), prefix);
        assert_eq!(
            output.next().await.unwrap().unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert!(output.next().await.is_none());
        assert_eq!(polls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn sse_unterminated_frames_are_bounded_and_in_limit_eof_is_verbatim() {
        let limit = 128;
        for size in [limit - 1, limit, limit + 1] {
            let input = Bytes::from(vec![b'x'; size]);
            for chunk_size in [1, size] {
                let output = rewrite_sse_stream_with_limit(chunks(&input, chunk_size), None, limit)
                    .try_collect::<Vec<_>>()
                    .await;
                if size <= limit {
                    assert_eq!(output.unwrap().concat(), input);
                } else {
                    assert_eq!(output.unwrap_err().kind(), io::ErrorKind::InvalidData);
                }
            }
        }
        let incomplete = Bytes::from_static(
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":99}}}",
        );
        let output = rewrite_sse_stream_with_limit(chunks(&incomplete, 1), Some(64), limit)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(output.concat(), incomplete);
    }

    #[tokio::test]
    async fn sse_split_utf8_delimiters_and_fields_preserve_every_other_byte() {
        let prefix = ": heartbeat\r\nevent: message\r\nid: 7\r\nretry: 100\r\ndata: {\"choices\":[{\"text\":\"中文🙂\"}]}\r\n\r\n";
        let usage = "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":99,\"audio_tokens\":2}},\"metrics\":{\"future\":\"中文🙂\"}}\n\n";
        let suffix = "data: not-json\n\ndata: [DONE]\r\n\r\n";
        let input = Bytes::from(format!("{prefix}{usage}{suffix}"));
        for chunk_size in [1, 2, 3, 7, input.len()] {
            let output = rewrite_sse_stream(chunks(&input, chunk_size), Some(64))
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            assert_eq!(output.len(), 4);
            assert_eq!(output[0].as_ref(), prefix.as_bytes());
            let json: Value = serde_json::from_slice(&output[1][6..output[1].len() - 2]).unwrap();
            assert_eq!(
                json["usage"]["prompt_tokens_details"],
                json!({"cached_tokens":64,"audio_tokens":2})
            );
            assert_eq!(json["metrics"], json!({"future":"中文🙂"}));
            assert_eq!(
                [output[2].as_ref(), output[3].as_ref()].concat(),
                suffix.as_bytes()
            );
        }
    }

    #[tokio::test]
    async fn sse_transport_error_does_not_forward_partial_frame() {
        let source = stream::iter([
            Ok(Bytes::from_static(b"data: [DONE]\n\ndata: partial")),
            Err(io::Error::new(io::ErrorKind::ConnectionReset, "reset")),
        ]);
        let output = rewrite_sse_stream(source, None);
        futures_util::pin_mut!(output);
        assert_eq!(
            output.next().await.unwrap().unwrap(),
            b"data: [DONE]\n\n".as_slice()
        );
        assert!(output.next().await.unwrap().is_err());
        assert!(output.next().await.is_none());
    }

    #[tokio::test]
    async fn sse_releases_consumed_transport_chunk_before_waiting_for_more() {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };

        struct OwnedChunk {
            bytes: Vec<u8>,
            dropped: Arc<AtomicBool>,
        }
        impl AsRef<[u8]> for OwnedChunk {
            fn as_ref(&self) -> &[u8] {
                &self.bytes
            }
        }
        impl Drop for OwnedChunk {
            fn drop(&mut self) {
                self.dropped.store(true, Ordering::Relaxed);
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let input = Bytes::from_owner(OwnedChunk {
            bytes: b"data: [DONE]\n\n".to_vec(),
            dropped: dropped.clone(),
        });
        let source = stream::iter([Ok::<_, io::Error>(input)]).chain(stream::pending());
        let output = rewrite_sse_stream(source, None);
        futures_util::pin_mut!(output);
        assert_eq!(
            output.next().await.unwrap().unwrap(),
            b"data: [DONE]\n\n".as_slice()
        );
        assert!(dropped.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn sse_rejects_large_echo_and_logprobs_frames_above_one_mib() {
        // Construct large echo/logprobs payloads to verify defensive rejection.
        // These are synthetic events, not supported-workload engine captures.
        let tokens = 16_384;
        let logprobs = json!({
            "tokens": vec![" token"; tokens],
            "token_logprobs": vec![-0.123456789; tokens],
            "text_offset": (0..tokens).map(|index| index * 6).collect::<Vec<_>>(),
            "top_logprobs": vec![json!({" token": -0.123456789, " other": -1.23456789, " word": -2.3456789, " text": -3.456789, " item": -4.56789}); tokens],
        });
        for choice in [
            json!({"index":0,"text":"token ".repeat(200_000),"finish_reason":null}),
            json!({"index":0,"text":" token".repeat(tokens),"logprobs":logprobs,"finish_reason":null}),
        ] {
            let frame = Bytes::from(format!(
                "data: {}\n\n",
                json!({"id":"cmpl-sizing","object":"text_completion","created":0,"model":"sizing","choices":[choice]})
            ));
            assert!(frame.len() > 1024 * 1024);
            let output = rewrite_sse_stream(chunks(&frame, 8191), Some(64))
                .try_collect::<Vec<_>>()
                .await;
            assert!(output.is_err(), "oversized SSE frame must be rejected");
            assert_eq!(output.unwrap_err().kind(), io::ErrorKind::InvalidData);
        }
    }

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
