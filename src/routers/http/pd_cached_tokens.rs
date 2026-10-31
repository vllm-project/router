//! Reconcile Prefill-owned prefix-cache usage in vLLM PD responses.

use super::{
    sse::{sse_lines, FrameScanner},
    usage_merge,
};
use bytes::{Buf, Bytes};
use futures_util::{stream, Stream, StreamExt};
use serde_json::Value;
use std::io;
use std::pin::Pin;

// Limit the rewrite accumulator, not the supported SSE event size. Larger
// events switch to incremental passthrough until their delimiter; the following
// event resumes reconciliation. Transport chunks also occupy memory.
const MAX_SSE_FRAME_BYTES: usize = 1024 * 1024;

pub(super) fn is_supported_path(path: &str) -> bool {
    matches!(
        path,
        "/v1/chat/completions" | "/v1/completions" | "/v1/responses"
    )
}

pub(super) fn prefill_cached_tokens(prefill: Option<&Value>) -> Option<u64> {
    let usage = usage_merge::usage_object(prefill?)?;
    // Validate each candidate against its matching input count, so an invalid
    // legacy spelling cannot hide a valid Responses spelling. Keep support for
    // details-only snapshots; an explicit input count must be valid and bound
    // the cache count. Zero remains a valid cache count.
    [
        ("prompt_tokens_details", "prompt_tokens"),
        ("input_tokens_details", "input_tokens"),
    ]
    .iter()
    .find_map(|(details_key, input_key)| {
        let cached = usage.get(*details_key)?.get("cached_tokens")?.as_u64()?;
        if let Some(input) = usage.get(*input_key) {
            if cached > input.as_u64()? {
                return None;
            }
        }
        Some(cached)
    })
}

/// Change only the cached-token detail. A missing Prefill value removes a
/// potentially misleading Decode value while retaining all sibling fields.
pub(super) fn reconcile_usage(
    response: &mut Value,
    cached: Option<u64>,
    prefill: Option<&Value>,
) -> Result<bool, &'static str> {
    let (_, details_key, _) = usage_merge::usage_dialect(response, prefill).keys();
    let Some(usage) = usage_merge::usage_object_mut(response) else {
        return if cached.is_some() {
            Err("Decode response has no usage object")
        } else {
            Ok(false)
        };
    };

    match usage.get_mut(details_key) {
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
                    details_key.into(),
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

fn rewrite_frame(frame: &[u8], cached: Option<u64>, prefill: Option<&Value>) -> Bytes {
    // Most delta frames carry no usage. Avoid collecting data or allocating a
    // JSON tree for them. Any escape conservatively takes the parse path, since
    // a valid JSON key can spell "usage" with Unicode escapes.
    if !frame.contains(&b'\\')
        && !frame
            .windows(7)
            .any(|window| window[0] == b'"' && &window[1..] == b"usage\"")
    {
        return Bytes::copy_from_slice(frame);
    }
    let mut data = Vec::new();
    let mut has_data = false;
    for (line, _) in sse_lines(frame) {
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
    if usage_merge::usage_object(&json).is_none() {
        return Bytes::copy_from_slice(frame);
    }
    // Running/per-chunk usage is corrected as soon as its event completes;
    // neither final-frame heuristics nor a following event are required.
    let mut changed = prefill.is_some_and(|value| usage_merge::merge_pd_usage(value, &mut json));
    match reconcile_usage(&mut json, cached, prefill) {
        Ok(rewritten) => changed |= rewritten,
        Err(reason) => tracing::warn!("Cannot rewrite PD SSE usage: {}", reason),
    }
    if !changed {
        return Bytes::copy_from_slice(frame);
    }
    let Ok(encoded) = serde_json::to_vec(&json) else {
        return Bytes::copy_from_slice(frame);
    };

    let mut result = Vec::with_capacity(frame.len() + encoded.len());
    let mut wrote_data = false;
    for (line, ending) in sse_lines(frame) {
        if line.starts_with(b"data:") {
            if !wrote_data {
                result.extend_from_slice(b"data: ");
                result.extend_from_slice(&encoded);
                wrote_data = true;
            } else {
                // Preserve every line ending. Removing a continuation line can
                // join a CR and LF that originally ended different lines, hiding
                // the event's blank line. Empty data adds only JSON whitespace.
                result.extend_from_slice(b"data:");
            }
        } else {
            result.extend_from_slice(line);
        }
        result.extend_from_slice(ending);
    }
    Bytes::from(result)
}

struct SseState<S> {
    source: Pin<Box<S>>,
    buffer: Vec<u8>,
    pending: Bytes,
    scanner: FrameScanner,
    cached: Option<u64>,
    prefill: Option<Value>,
    passthrough: bool,
    error: Option<io::Error>,
    done: bool,
}

pub(super) fn rewrite_sse_stream<S, E>(
    source: S,
    cached: Option<u64>,
    prefill: Option<Value>,
) -> impl Stream<Item = Result<Bytes, io::Error>> + Send
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    rewrite_sse_stream_with_limit(source, cached, MAX_SSE_FRAME_BYTES, prefill)
}

// An explicit limit lets tests exercise frame boundaries with smaller inputs.
fn rewrite_sse_stream_with_limit<S, E>(
    source: S,
    cached: Option<u64>,
    max_frame_bytes: usize,
    prefill: Option<Value>,
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
        prefill,
        passthrough: false,
        error: None,
        done: false,
    };
    stream::unfold(state, move |mut state| async move {
        loop {
            if let Some(error) = state.error.take() {
                return Some((Err(error), state));
            }
            if state.done {
                return None;
            }
            if !state.pending.is_empty() {
                if state.buffer.is_empty()
                    && !state.passthrough
                    && state.scanner.skip_lf
                    && state.pending[0] == b'\n'
                {
                    // The preceding event ended at a CR in another chunk.
                    // Preserve its LF without treating it as a new blank line
                    // or holding it until the next event/EOF.
                    state.scanner.skip_lf = false;
                    let suffix = state.pending.split_to(1);
                    if state.pending.is_empty() {
                        state.pending = Bytes::new();
                    }
                    return Some((Ok(suffix), state));
                }

                if state.passthrough {
                    let end = state.scanner.frame_end(&state.pending);
                    let consumed = end.unwrap_or(state.pending.len());
                    let output = state.pending.split_to(consumed);
                    if end.is_some() {
                        state.passthrough = false;
                    }
                    if state.pending.is_empty() {
                        state.pending = Bytes::new();
                    }
                    return Some((Ok(output), state));
                }
                let remaining = max_frame_bytes.saturating_sub(state.buffer.len());
                if remaining == 0 {
                    state.passthrough = true;
                    return Some((Ok(Bytes::from(std::mem::take(&mut state.buffer))), state));
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
                    let rewritten =
                        rewrite_frame(&state.buffer, state.cached, state.prefill.as_ref());
                    state.buffer.clear();
                    return Some((Ok(rewritten), state));
                }
                continue;
            }
            match state.source.as_mut().next().await {
                Some(Ok(chunk)) => state.pending = chunk,
                Some(Err(error)) => {
                    state.done = true;
                    let error = io::Error::other(error);
                    if !state.buffer.is_empty() {
                        // Preserve every received byte before reporting the
                        // source failure, without rewriting incomplete data.
                        state.error = Some(error);
                        return Some((Ok(Bytes::from(std::mem::take(&mut state.buffer))), state));
                    }
                    return Some((Err(error), state));
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

#[cfg(all(test, feature = "cpu", feature = "ready"))]
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
        for newline in [b"\n".as_slice(), b"\r\n".as_slice(), b"\r".as_slice()] {
            for size in [limit - 1, limit, limit + 1] {
                let frame = comment_frame(size, newline);
                for chunk_size in [8191, size] {
                    let output = rewrite_sse_stream_with_limit(
                        chunks(&frame, chunk_size),
                        None,
                        limit,
                        None,
                    )
                    .try_collect::<Vec<_>>()
                    .await;
                    assert_eq!(output.unwrap().concat(), frame);
                }
            }
        }
    }

    #[tokio::test]
    async fn sse_limits_each_frame_not_the_chunk_or_total_stream() {
        let limit = 128;
        let frame = comment_frame(limit, b"\r\n");
        let input = Bytes::from(frame.repeat(1024));
        let output = rewrite_sse_stream_with_limit(chunks(&input, input.len()), None, limit, None)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(output.len(), 1024);
        assert_eq!(output.concat(), input);
    }

    #[tokio::test]
    async fn oversized_frame_is_passthrough_and_next_frame_is_reconciled() {
        let limit = 128;
        let oversized = comment_frame(limit * 4, b"\r\n");
        let usage = b"data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":99}}}\n\n";
        let input = Bytes::from([oversized.as_ref(), usage, b"data: [DONE]\n\n"].concat());
        for chunk_size in [1, 7, limit, input.len()] {
            let output =
                rewrite_sse_stream_with_limit(chunks(&input, chunk_size), Some(4), limit, None)
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap()
                    .concat();
            assert!(output.starts_with(&oversized));
            let tail = String::from_utf8(output[oversized.len()..].to_vec()).unwrap();
            assert!(tail.contains("\"cached_tokens\":4"));
            assert!(!tail.contains("\"cached_tokens\":99"));
            assert!(tail.ends_with("data: [DONE]\n\n"));
        }
    }

    #[tokio::test]
    async fn sse_unterminated_frames_are_bounded_and_in_limit_eof_is_verbatim() {
        let limit = 128;
        for size in [limit - 1, limit, limit + 1] {
            let input = Bytes::from(vec![b'x'; size]);
            for chunk_size in [1, size] {
                let output =
                    rewrite_sse_stream_with_limit(chunks(&input, chunk_size), None, limit, None)
                        .try_collect::<Vec<_>>()
                        .await;
                assert_eq!(output.unwrap().concat(), input);
            }
        }
        let incomplete = Bytes::from_static(
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":99}}}",
        );
        let output = rewrite_sse_stream_with_limit(chunks(&incomplete, 1), Some(64), limit, None)
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
            let output = rewrite_sse_stream(chunks(&input, chunk_size), Some(64), None)
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            let output = output.concat();
            assert!(output.starts_with(prefix.as_bytes()));
            assert!(output.ends_with(suffix.as_bytes()));
            let usage = &output[prefix.len()..output.len() - suffix.len()];
            let json: Value = serde_json::from_slice(&usage[6..usage.len() - 2]).unwrap();
            assert_eq!(
                json["usage"]["prompt_tokens_details"],
                json!({"cached_tokens":64,"audio_tokens":2})
            );
            assert_eq!(json["metrics"], json!({"future":"中文🙂"}));
        }
    }

    #[tokio::test]
    async fn sse_transport_error_flushes_partial_frame_before_error() {
        let source = stream::iter([
            Ok(Bytes::from_static(b"data: [DONE]\n\ndata: partial")),
            Err(io::Error::new(io::ErrorKind::ConnectionReset, "reset")),
        ]);
        let output = rewrite_sse_stream(source, None, None);
        futures_util::pin_mut!(output);
        assert_eq!(
            output.next().await.unwrap().unwrap(),
            b"data: [DONE]\n\n".as_slice()
        );
        assert_eq!(
            output.next().await.unwrap().unwrap(),
            b"data: partial".as_slice()
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
        let output = rewrite_sse_stream(source, None, None);
        futures_util::pin_mut!(output);
        assert_eq!(
            output.next().await.unwrap().unwrap(),
            b"data: [DONE]\n\n".as_slice()
        );
        assert!(dropped.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn sse_preserves_large_echo_and_logprobs_frames_above_one_mib() {
        // Regression fixtures for legitimate large echo/logprobs events.
        // Synthetic event bodies; no live engine is required.
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
            let output = rewrite_sse_stream(chunks(&frame, 8191), Some(64), None)
                .try_collect::<Vec<_>>()
                .await;
            assert_eq!(output.unwrap().concat(), frame);
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
    fn prefill_cache_is_bounded_by_the_matching_input_count() {
        for (input_key, details_key) in [
            ("prompt_tokens", "prompt_tokens_details"),
            ("input_tokens", "input_tokens_details"),
        ] {
            for nested in [false, true] {
                for (input, cached, expected) in [
                    (70, 90, None),
                    (0, 1, None),
                    (0, 0, Some(0)),
                    (70, 70, Some(70)),
                    (70, 4, Some(4)),
                    (u64::MAX, u64::MAX, Some(u64::MAX)),
                ] {
                    let mut prefill = json!({"usage": {input_key: input, details_key: {"cached_tokens": cached}}});
                    if nested {
                        prefill = json!({"response": prefill});
                    }
                    assert_eq!(prefill_cached_tokens(Some(&prefill)), expected, "{prefill}");
                }
                for invalid_input in [json!(null), json!(-1), json!("70"), json!(1.5)] {
                    let mut prefill = json!({"usage": {input_key: invalid_input, details_key: {"cached_tokens": 4}}});
                    if nested {
                        prefill = json!({"response": prefill});
                    }
                    assert_eq!(prefill_cached_tokens(Some(&prefill)), None, "{prefill}");
                }
            }
        }
        let prefill = json!({"usage": {
            "prompt_tokens": 70, "prompt_tokens_details": {"cached_tokens": 90},
            "input_tokens": 100, "input_tokens_details": {"cached_tokens": 80}
        }});
        assert_eq!(prefill_cached_tokens(Some(&prefill)), Some(80));
    }

    #[test]
    fn invalid_prefill_cache_does_not_hide_valid_responses_fallback() {
        for invalid in [json!(-1), json!(null), json!("4"), json!(1.5)] {
            let prefill = json!({"usage": {
                "prompt_tokens_details": {"cached_tokens": invalid},
                "input_tokens_details": {"cached_tokens": 0}
            }});
            assert_eq!(prefill_cached_tokens(Some(&prefill)), Some(0));
        }
    }

    #[test]
    fn responses_details_are_created_at_the_correct_level() {
        for mut response in [
            json!({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            json!({"type": "response.completed", "response": {"usage": {"input_tokens": 10, "output_tokens": 2}}}),
        ] {
            assert_eq!(reconcile_usage(&mut response, Some(4), None), Ok(true));
            let usage = usage_merge::usage_object(&response).unwrap();
            assert_eq!(usage["input_tokens_details"], json!({"cached_tokens": 4}));
            assert!(!usage.contains_key("prompt_tokens_details"));
        }
    }

    #[tokio::test]
    async fn complete_usage_event_is_released_without_waiting_for_next_chunk() {
        let prefill = json!({"usage": {"prompt_tokens": 70, "completion_tokens": 1}});
        let frame = Bytes::from_static(b"data: {\"choices\":[{\"text\":\"A\"}],\"usage\":{\"prompt_tokens\":999,\"completion_tokens\":2,\"total_tokens\":1001,\"prompt_tokens_details\":{\"cached_tokens\":99}}}\n\n");
        let source = stream::iter([Ok::<_, io::Error>(frame)]).chain(stream::pending());
        let output = rewrite_sse_stream(source, Some(4), Some(prefill));
        futures_util::pin_mut!(output);
        let output = tokio::time::timeout(std::time::Duration::from_secs(1), output.next())
            .await
            .expect("completed events must not need lookahead")
            .unwrap()
            .unwrap();
        let json: Value = serde_json::from_slice(&output[6..output.len() - 2]).unwrap();
        assert_eq!(json["usage"]["prompt_tokens"], 70);
        assert_eq!(json["usage"]["total_tokens"], 72);
        assert_eq!(json["usage"]["prompt_tokens_details"]["cached_tokens"], 4);
        assert_eq!(json["choices"][0]["text"], "A");
    }

    #[tokio::test]
    async fn bare_cr_usage_event_is_reconciled_without_waiting_for_more_input() {
        let frame = Bytes::from_static(b"event: message\rdata: {\"choices\":[],\"usage\":{\"prompt_tokens\":999,\"completion_tokens\":2,\"total_tokens\":1001,\"prompt_tokens_details\":{\"cached_tokens\":99}}}\r\r");
        let source = stream::iter([Ok::<_, io::Error>(frame)]).chain(stream::pending());
        let prefill = json!({"usage": {"prompt_tokens": 70}});
        let output = rewrite_sse_stream(source, Some(4), Some(prefill));
        futures_util::pin_mut!(output);
        let bytes = tokio::time::timeout(std::time::Duration::from_secs(1), output.next())
            .await
            .expect("a bare-CR delimiter must release the event immediately")
            .unwrap()
            .unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.starts_with("event: message\rdata: "));
        assert!(text.ends_with("\r\r"));
        let payload = text.strip_prefix("event: message\rdata: ").unwrap().trim();
        let value: Value = serde_json::from_str(payload).unwrap();
        assert_eq!(value["usage"]["prompt_tokens"], 70);
        assert_eq!(value["usage"]["total_tokens"], 72);
        assert_eq!(value["usage"]["prompt_tokens_details"]["cached_tokens"], 4);
    }

    #[tokio::test]
    async fn mixed_line_endings_reconcile_multiline_data_at_every_chunk_split() {
        let input = Bytes::from_static(concat!(
            ": heartbeat\rid: 7\nevent: response.completed\r\n",
            "data: {\"type\":\"response.completed\",\r",
            "data: \"response\":{\"usage\":{\"input_tokens\":999,\"output_tokens\":7,\"total_tokens\":1006}}}\r\n",
            "\ndata: [DONE]\r\r"
        ).as_bytes());
        let prefill = json!({"usage": {"input_tokens": 50}});
        for split in 0..=input.len() {
            let source = stream::iter([
                Ok::<_, io::Error>(input.slice(..split)),
                Ok(input.slice(split..)),
            ]);
            let output = rewrite_sse_stream(source, Some(16), Some(prefill.clone()))
                .try_collect::<Vec<_>>()
                .await
                .unwrap()
                .concat();
            // Retain even mixed original endings: removing a data line must
            // not accidentally combine a CR and LF into one terminator.
            let endings = |bytes: &[u8]| {
                bytes
                    .iter()
                    .copied()
                    .filter(|byte| *byte == b'\r' || *byte == b'\n')
                    .collect::<Vec<_>>()
            };
            assert_eq!(endings(&output), endings(&input), "split={split}");
            let wire = std::str::from_utf8(&output)
                .unwrap()
                .replace("\r\n", "\n")
                .replace('\r', "\n");
            let events: Vec<_> = wire
                .split("\n\n")
                .filter(|event| !event.is_empty())
                .collect();
            assert_eq!(events.len(), 2, "split={split}");
            assert!(events[0].starts_with(": heartbeat\nid: 7\nevent: response.completed\n"));
            let payload = events[0]
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|value| value.strip_prefix(' ').unwrap_or(value))
                .collect::<Vec<_>>()
                .join("\n");
            let value: Value = serde_json::from_str(&payload).unwrap();
            assert_eq!(
                value["response"]["usage"],
                json!({"input_tokens": 50, "output_tokens": 7,
                "total_tokens": 57, "input_tokens_details": {"cached_tokens": 16}})
            );
            assert_eq!(events[1], "data: [DONE]");
        }
    }

    #[tokio::test]
    async fn split_crlf_releases_event_at_cr_and_forwards_optional_lf() {
        let frame = Bytes::from_static(b"data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":99}}}\r\n\r");
        let (finish, ready) = tokio::sync::oneshot::channel::<()>();
        let source = stream::iter([Ok::<_, io::Error>(frame)]).chain(stream::once(async move {
            ready.await.unwrap();
            Ok(Bytes::from_static(b"\ndata: [DONE]\r\n\r\n"))
        }));
        let output = rewrite_sse_stream(source, Some(4), None);
        futures_util::pin_mut!(output);
        let first = tokio::time::timeout(std::time::Duration::from_secs(1), output.next())
            .await
            .expect("CR must release the event before an optional LF arrives")
            .unwrap()
            .unwrap();
        assert!(std::str::from_utf8(&first)
            .unwrap()
            .contains("\"cached_tokens\":4"));
        assert!(first.ends_with(b"\r\n\r"));
        finish.send(()).unwrap();
        assert_eq!(output.next().await.unwrap().unwrap().as_ref(), b"\n");
        assert_eq!(
            output.next().await.unwrap().unwrap().as_ref(),
            b"data: [DONE]\r\n\r\n"
        );
        assert!(output.next().await.is_none());
    }

    #[tokio::test]
    async fn nested_responses_multiline_data_and_null_usage_keep_framing() {
        let prefill = json!({"usage": {"input_tokens": 50, "output_tokens": 1}});
        let input = Bytes::from_static(concat!(
            "event: response.created\r\ndata: {\"type\":\"response.created\",\"response\":{\"usage\":null}}\r\n\r\n",
            "id: 7\r\nevent: response.completed\r\ndata: {\"type\":\"response.completed\",\r\n",
            "data: \"response\":{\"usage\":{\"input_tokens\":999,\"output_tokens\":7,\"total_tokens\":1006}}}\r\n\r\n",
            ": heartbeat\n\ndata: [DONE]\n\n"
        ).as_bytes());
        for chunk_size in [1, 2, 7, input.len()] {
            let output =
                rewrite_sse_stream(chunks(&input, chunk_size), Some(16), Some(prefill.clone()))
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
            let output = output.concat();
            let wire = std::str::from_utf8(&output).unwrap();
            let normalized = wire.replace("\r\n", "\n").replace('\r', "\n");
            let frames: Vec<_> = normalized
                .split("\n\n")
                .filter(|frame| !frame.is_empty())
                .collect();
            assert_eq!(frames.len(), 4);
            assert!(frames[0].ends_with("\"usage\":null}}"));
            assert!(frames[1].starts_with("id: 7\nevent: response.completed\ndata: "));
            let payload = frames[1]
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|value| value.strip_prefix(' ').unwrap_or(value))
                .collect::<Vec<_>>()
                .join("\n");
            let json: Value = serde_json::from_str(&payload).unwrap();
            assert!(json.get("usage").is_none());
            assert_eq!(
                json["response"]["usage"],
                json!({"input_tokens": 50, "output_tokens": 7, "total_tokens": 57, "input_tokens_details": {"cached_tokens": 16}})
            );
            assert_eq!(frames[2], ": heartbeat");
            assert_eq!(frames[3], "data: [DONE]");
        }
    }

    #[test]
    fn all_openai_pd_usage_paths_are_reconciled() {
        assert!(is_supported_path("/v1/chat/completions"));
        assert!(is_supported_path("/v1/completions"));
        assert!(is_supported_path("/v1/responses"));
    }

    #[test]
    fn reconcile_preserves_decode_owned_usage_and_unknown_fields() {
        let mut response = json!({"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "prompt_tokens_details": {"cached_tokens": 100, "audio_tokens": 2}, "future_field": 7}, "metrics": {"time_to_first_token_ms": 12.5}});
        assert_eq!(reconcile_usage(&mut response, Some(64), None), Ok(true));
        assert_eq!(
            response["usage"]["prompt_tokens_details"],
            json!({"cached_tokens": 64, "audio_tokens": 2})
        );
        assert_eq!(response["usage"]["future_field"], 7);
        assert_eq!(response["metrics"], json!({"time_to_first_token_ms": 12.5}));
        assert_eq!(reconcile_usage(&mut response, None, None), Ok(true));
        assert_eq!(
            response["usage"]["prompt_tokens_details"],
            json!({"audio_tokens": 2})
        );
    }

    #[test]
    fn zero_cache_creates_details_when_decode_does_not_report_them() {
        let mut response = json!({"usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6, "prompt_tokens_details": null}});
        assert_eq!(reconcile_usage(&mut response, Some(0), None), Ok(true));
        assert_eq!(
            response["usage"]["prompt_tokens_details"],
            json!({"cached_tokens": 0})
        );
        assert_eq!(response["usage"]["completion_tokens"], 2);
    }

    #[test]
    fn usage_and_cache_merging_choose_the_same_dialect_in_either_order() {
        let cases = [
            (
                json!({"usage": {"input_tokens": 50, "input_tokens_details": {"cached_tokens": 4, "audio_tokens": 2}}}),
                json!({"usage": {"output_tokens": 7, "total_tokens": 1006}}),
                "input_tokens_details",
            ),
            (
                json!({"usage": {"input_tokens_details": {"cached_tokens": 4, "audio_tokens": 2}}}),
                json!({"usage": {"output_tokens": 7}}),
                "input_tokens_details",
            ),
            (
                json!({"usage": {"input_tokens": 50}}),
                json!({"response": {"usage": {"output_tokens": 7}}}),
                "input_tokens_details",
            ),
            (
                json!({"usage": {"prompt_tokens": 50, "prompt_tokens_details": {"cached_tokens": 4}}}),
                json!({"usage": {"completion_tokens": 7}}),
                "prompt_tokens_details",
            ),
        ];
        for (prefill, original, details_key) in cases {
            let mut usage_first = original.clone();
            usage_merge::merge_pd_usage(&prefill, &mut usage_first);
            reconcile_usage(&mut usage_first, Some(4), Some(&prefill)).unwrap();
            let mut cache_first = original;
            reconcile_usage(&mut cache_first, Some(4), Some(&prefill)).unwrap();
            usage_merge::merge_pd_usage(&prefill, &mut cache_first);
            assert_eq!(usage_first, cache_first);
            let usage = usage_merge::usage_object(&cache_first).unwrap();
            assert_eq!(usage[details_key]["cached_tokens"], 4);
            let other_key = if details_key == "input_tokens_details" {
                "prompt_tokens_details"
            } else {
                "input_tokens_details"
            };
            assert!(!usage.contains_key(other_key));
        }
    }

    #[tokio::test]
    async fn no_usage_and_null_usage_deltas_remain_byte_identical() {
        let input = Bytes::from_static(concat!(
            ": heartbeat\r\n\r\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"usage\\n\\\"quoted\\\"\"}}],\"usage\":null}\r\r",
            "data: [DONE]\n\n"
        ).as_bytes());
        for size in [1, 7, input.len()] {
            let output = rewrite_sse_stream(chunks(&input, size), Some(4), None)
                .try_collect::<Vec<_>>()
                .await
                .unwrap()
                .concat();
            assert_eq!(output, input);
        }
    }

    #[tokio::test]
    async fn escaped_usage_keys_still_remove_decode_cache_without_prefill() {
        for key in [r#""us\u0061ge""#, r#""\u0075\u0073\u0061\u0067\u0065""#] {
            for nested in [false, true] {
                let usage = format!(
                    r#"{{{key}:{{"input_tokens":10,"output_tokens":2,"input_tokens_details":{{"cached_tokens":99,"audio_tokens":1}}}}}}"#
                );
                let payload = if nested {
                    format!(r#"{{"type":"response.completed","response":{usage}}}"#)
                } else {
                    usage
                };
                let input = Bytes::from(format!("data: {payload}\n\n"));
                let output = rewrite_sse_stream(chunks(&input, 1), None, None)
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap()
                    .concat();
                let value: Value = serde_json::from_slice(&output[6..output.len() - 2]).unwrap();
                let usage = usage_merge::usage_object(&value).unwrap();
                assert_eq!(usage["input_tokens_details"], json!({"audio_tokens": 1}));
                assert_eq!(usage["output_tokens"], 2);
            }
        }
    }

    #[tokio::test]
    async fn fragmented_sse_rewrites_running_and_final_usage_and_preserves_metadata() {
        let input = b": keep\r\nevent: message\r\ndata: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":99}}}\r\n\r\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"prompt_tokens_details\":{\"cached_tokens\":100,\"audio_tokens\":2}},\"metrics\":{\"x\":1}}\r\n\r\ndata: [DONE]\r\n\r\n";
        let chunks = input
            .chunks(7)
            .map(|part| Ok::<_, io::Error>(Bytes::copy_from_slice(part)))
            .collect::<Vec<_>>();
        let output = rewrite_sse_stream(stream::iter(chunks), Some(64), None)
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
        let output = rewrite_sse_stream(stream::iter([Ok::<_, io::Error>(input)]), None, None)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let output = String::from_utf8(output.concat()).unwrap();
        assert!(!output.contains("cached_tokens"));
        assert!(output.contains("audio_tokens"));
    }
}
