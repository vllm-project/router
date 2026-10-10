//! Logprobs merging utilities for PD disaggregation
//!
//! This module provides utilities for merging logprobs from prefill and decode responses
//! in prefill-decode disaggregation mode.

use serde_json::Value;
use tracing::{debug, info};

/// Locate the object that holds the usage inside a **decode** payload, creating the slot (as
/// `null`) when it is absent so the caller can fill it in. `None` means the payload cannot
/// carry usage at all.
///
/// Two API surfaces share this module and they disagree on where usage lives:
///
/// - **Chat Completions / Completions** carry it at the top level: `{"usage": {...}}`.
/// - A **Responses API streaming event** wraps a whole response, so it sits one level down:
///   `{"type": "response.completed", "response": {..., "usage": {...}}}`. `/v1/responses` is
///   routed through `route_transparent` → `process_vllm_two_stage_request`, i.e. through the
///   very same [`SseUsageMerger`], so it reaches this module too.
///
/// ⚠️ The nesting must be detected **before** falling back to the top level: a Responses
/// event has no top-level `usage`, so the fallback would happily fabricate one — adding
/// prefill's whole usage at a level no client reads while leaving the real
/// `response.usage.input_tokens_details.cached_tokens` at the decode-side placeholder.
///
/// The top level still wins whenever it already holds a usage **object**, so payloads of
/// the existing surfaces keep their behaviour exactly.
fn usage_slot_mut(decode_json: &mut Value) -> Option<&mut Value> {
    let nested_usage = !decode_json.get("usage").is_some_and(Value::is_object)
        && decode_json.get("response").is_some_and(Value::is_object);

    let decode_obj = decode_json.as_object_mut()?;
    if nested_usage {
        let response = decode_obj.get_mut("response")?.as_object_mut()?;
        return Some(response.entry("usage".to_string()).or_insert(Value::Null));
    }
    Some(decode_obj.entry("usage".to_string()).or_insert(Value::Null))
}

/// Merge usage metadata from prefill response into decode response.
///
/// Prefer the prefill-side cached_tokens, because vLLM decode-side cached token
/// accounting may temporarily report invalid placeholder values such as -1.
pub fn merge_usage_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
    let Some(prefill_usage) = prefill_json.get("usage") else {
        return false;
    };

    let prefill_cached = prefill_usage
        .get("prompt_tokens_details")
        .and_then(|v| v.get("cached_tokens"))
        .or_else(|| {
            prefill_usage
                .get("input_tokens_details")
                .and_then(|v| v.get("cached_tokens"))
        });

    let Some(prefill_cached) = prefill_cached else {
        return false;
    };

    // Which key holds the usage depends on the API surface — see `usage_slot_mut`.
    let Some(usage_slot) = usage_slot_mut(decode_json) else {
        return false;
    };

    // ⚠️ "Missing" is not the only unusable shape: engines legitimately emit
    // `"usage": null` (and `"prompt_tokens_details": null`, see below). Both mean "no
    // decode-side accounting here", so prefill's usage is filled in — an early `return
    // false` on a `null` would leave the client with the unusable value it already had,
    // which is precisely the bug this function exists to fix.
    if !usage_slot.is_object() {
        *usage_slot = prefill_usage.clone();
        debug!("[USAGE MERGE] Filled in decode usage from prefill (missing or null)");
        return true;
    }

    let Some(decode_usage_obj) = usage_slot.as_object_mut() else {
        return false;
    };

    let details_target = if decode_usage_obj.contains_key("prompt_tokens_details") {
        "prompt_tokens_details"
    } else if decode_usage_obj.contains_key("input_tokens_details") {
        "input_tokens_details"
    } else {
        "prompt_tokens_details"
    };

    // Same class of shape problem one level down. `Map::entry(..).or_insert(..)` is not
    // enough here: for `"prompt_tokens_details": null` the key **is** present, `entry`
    // hands back that `Null`, `as_object_mut()` fails and the function bails out with
    // nothing repaired. Treat any non-object value exactly like an absent key.
    if !decode_usage_obj
        .get(details_target)
        .is_some_and(Value::is_object)
    {
        decode_usage_obj.insert(
            details_target.to_string(),
            Value::Object(serde_json::Map::new()),
        );
    }

    let Some(decode_details_obj) = decode_usage_obj
        .get_mut(details_target)
        .and_then(|v| v.as_object_mut())
    else {
        return false;
    };

    let should_replace = match decode_details_obj.get("cached_tokens") {
        Some(existing) => existing.as_i64().is_none_or(|v| v <= 0),
        None => true,
    };

    if should_replace {
        decode_details_obj.insert("cached_tokens".to_string(), prefill_cached.clone());
        debug!(
            "[USAGE MERGE] Replaced decode cached_tokens with prefill value {}",
            prefill_cached
        );
        return true;
    }

    false
}

/// Prefix of an SSE data line (excluding the space after the colon).
const SSE_DATA_PREFIX: &str = "data:";

/// End-of-stream marker, emitted by vLLM after the last data event.
const SSE_DONE: &str = "[DONE]";

/// Extract the payload of a `data:` line (`[DONE]` and other non-JSON payloads are
/// returned verbatim).
fn sse_payload(line: &str) -> Option<&str> {
    let rest = line.strip_prefix(SSE_DATA_PREFIX)?;
    Some(rest.strip_prefix(' ').unwrap_or(rest))
}

/// Whether the line is a JSON event carrying a **non-null usage object**.
///
/// ⚠️ A plain `contains("\"usage\"")` check does not work here: the **first chunk of a
/// vLLM streaming response already carries `"usage":null`**, and holding it back as if it
/// were a usage event would delay the first chunk until the end of the stream, which
/// breaks streaming. The colon must therefore be followed by `{` (an empty `{}` object
/// qualifies as well; the merge fills it in as needed, which is expected).
///
/// The match runs on every line, but only as one pre-compiled regex match — no full JSON
/// parsing.
fn is_usage_event(payload: &str) -> bool {
    if payload == SSE_DONE {
        return false;
    }
    static RE: once_cell::sync::Lazy<regex::Regex> = once_cell::sync::Lazy::new(|| {
        // ⚠️ `\{` must **not** be followed by `?` — that makes `{` optional, so
        // `"usage": null` would match too, this function would degenerate into
        // `contains("\"usage\"")`, and the trap warned about above would come back to
        // life.
        regex::Regex::new(r#""usage"\s*:\s*\{"#).expect("valid usage pattern")
    });
    RE.is_match(payload)
}

/// Merge prefill usage into one SSE data line.
///
/// ⚠️ **Every trailing newline must be preserved verbatim** (`\n`, `\n\n` or `\r\n`).
///
/// In SSE an event ends with a blank line, so the bytes of a `data: {...}` line look like
/// `...}\n\n` — **two** `\n`. An earlier version kept only one of them
/// (`else if ends_with('\n') => "\n"`), which meant:
///   - the remaining `\n` was cut out by `feed()` as a stand-alone blank line and passed
///     through immediately;
///   - once the held line reached the end of the stream, the `\n\n` boundary used to
///     decide "is this the last event?" was gone.
///
/// The symptom was `feed()` returning `b"\n"` when it should have returned nothing.
///
/// What this does: strip all trailing newlines to get the body, then append **the stripped
/// sequence unchanged**, keeping the bytes identical for `\n` / `\n\n` / `\r\n`.
///
/// Returns `None` when parsing fails; the caller should forward the line verbatim —
/// **never swallow content just because the merge failed**.
fn merge_usage_into_sse_line(line: &str, prefill_json: &Value) -> Option<String> {
    let payload = sse_payload(line)?;
    let mut event: Value = serde_json::from_str(payload).ok()?;
    if !merge_usage_in_json(prefill_json, &mut event) {
        return None;
    }
    let body = line.trim_end_matches(['\r', '\n']);
    let trailing = &line[body.len()..];
    Some(format!("{SSE_DATA_PREFIX} {event}{trailing}"))
}

/// Incremental SSE stream transformer: buffers line by line and merges prefill usage into
/// the events that carry usage.
///
/// # Why one event of delay is required
///
/// vLLM's `stream_options.include_usage` semantics put usage in the **last** chunk. But
/// `merge_usage_in_json` has a rule that copies prefill usage wholesale when decode has
/// none — with **immediate line-by-line processing**, any intermediate chunk without usage
/// (usage is null there) would be stamped with usage by that rule, inventing usage for
/// every chunk out of thin air.
///
/// A line carrying usage is therefore **held until the next line arrives**, which confirms
/// whether it really was the last one. Intermediate chunks are parsed normally and do not
/// trigger that rule.
///
/// # Byte order and event boundaries are part of the contract
///
/// Holding a line deliberately keeps its terminating blank line inside `held` (see
/// `handle_line`), which leaves two traps that both corrupt the stream instead of merely
/// reordering it:
///
///   1. Emitting a later line while an earlier one is still held. The held line's event has
///      not been terminated yet, so the later event's terminating blank line gets appended
///      to `held` and the client receives **both payloads inside one SSE event**
///      (`data: {content}\ndata: {error}\n\n`) — that is not a valid JSON document and
///      cannot be parsed.
///   2. Releasing `held` *after* a later line, which puts bytes on the wire out of order.
///
/// Both are avoided by one rule: **any line other than the blank line that terminates the
/// held event releases `held` first**.
///
/// Buffering is bounded: at most one SSE line is held at any moment (splitting on `\n`
/// never accumulates across lines); a stream without usage holds exactly one line.
#[derive(Default)]
pub struct SseUsageMerger {
    buf: Vec<u8>,
    held: Option<Vec<u8>>,
    /// Whether the blank line terminating the held event has already been taken into
    /// `held`. A blank line belongs to the event it terminates, so only the **first** one
    /// after a held data line may be absorbed; any further blank line belongs to whatever
    /// comes next and must not be glued on.
    held_terminated: bool,
    merged: bool,
}

impl SseUsageMerger {
    /// Process one chunk of bytes and return the bytes that **may be sent on
    /// immediately**.
    pub fn feed(&mut self, chunk: &[u8], prefill_json: &Value) -> Vec<u8> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        // Split on `\n`; the last segment has no newline yet, so it stays in buf until the
        // next chunk
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            self.handle_line(&line, prefill_json, &mut out);
        }
        out
    }

    /// End of stream: release the held line and rebuild the trailing line buffer.
    pub fn finish(&mut self, prefill_json: &Value) -> Vec<u8> {
        let mut out = Vec::new();
        if !self.buf.is_empty() {
            let line = std::mem::take(&mut self.buf);
            self.handle_line(&line, prefill_json, &mut out);
        }
        self.release_held(&mut out);
        out
    }

    fn handle_line(&mut self, line: &[u8], prefill_json: &Value, out: &mut Vec<u8>) {
        let Ok(text) = std::str::from_utf8(line) else {
            // Invalid UTF-8: cannot be decided, pass through verbatim — but never ahead of
            // an earlier line that is still held
            self.release_held(out);
            out.extend_from_slice(line);
            return;
        };

        // While an event is held, take the blank line that follows it into held as well.
        //
        // A blank line terminates an SSE event and belongs to the same event unit, so it
        // **must not be sent on its own**: `feed()` splits line by line on `\n`, so
        // `data: {...}\n\n` is always cut into a "data line" plus "a blank line holding a
        // single \n". Sending that blank line through on its own would order the bytes as
        // "blank line → held → [DONE]", while held itself still carries the first `\n` —
        // the position would be wrong.
        //
        // Only the **first** blank line is taken (`held_terminated`); a second one would
        // otherwise be appended to the held event as a stray newline.
        if self.held.is_some() && !self.held_terminated && text.trim().is_empty() {
            if let Some(held) = self.held.as_mut() {
                held.extend_from_slice(line);
            }
            self.held_terminated = true;
            return;
        }

        // Every other line — another data event (`data: [DONE]`, an error frame, ...), an
        // SSE comment, an `event:` field, even a second blank line — comes **after** the
        // held one, so the held event is released first. See the type-level docs: skipping
        // this step fuses two JSON payloads into a single unparseable SSE event.
        self.release_held(out);

        if sse_payload(text).is_some_and(is_usage_event) {
            match merge_usage_into_sse_line(text, prefill_json) {
                Some(merged) => {
                    self.merged = true;
                    debug!("[USAGE MERGE] merged prefill usage into streaming chunk");
                    self.held = Some(merged.into_bytes());
                }
                None => {
                    // Parse failure or nothing to change: keep holding it and let the next
                    // line decide whether it is the last one
                    self.held = Some(line.to_vec());
                }
            }
            return;
        }

        out.extend_from_slice(line);
    }

    /// Put the held event on the wire (no-op when nothing is held, so it is safe to call
    /// on every path).
    fn release_held(&mut self, out: &mut Vec<u8>) {
        if let Some(held) = self.held.take() {
            out.extend_from_slice(&held);
        }
        self.held_terminated = false;
    }

    /// Whether a merge actually happened (for caller logging/metrics).
    pub fn did_merge(&self) -> bool {
        self.merged
    }
}

/// Wrap a decode-side byte stream so that prefill usage is merged into every event that
/// carries usage.
///
/// This is the whole streaming integration, shared by both PD streaming paths
/// (`handle_decode_response` and `process_vllm_two_stage_request`). Keeping one copy is
/// deliberate: it used to be two identical `unfold` state machines, and a fix applied to
/// one of them silently missed the other.
///
/// `label` only feeds the log lines.
///
/// # An upstream failure must not swallow bytes that were already received
///
/// The merger holds at most one line. When the upstream stream fails mid-flight, that line
/// is **flushed first** and the error is propagated right after it. In "usage on every
/// chunk" mode the held line carries generated text, not just usage counters, so surfacing
/// the error immediately would truncate content the client had already been sent — and,
/// `data: [DONE]` never having arrived, the truncation would not even look like one.
///
/// ⚠️ Flushing into the body stream is **not enough on its own**: an item returned by a
/// body stream does not reach the client until the connection task writes it out, and when
/// the next item is an `Err` the response is aborted (chunked body cut off / connection
/// reset) — the frame that was just handed over is discarded with it. This was verified
/// end to end against a mock decode worker that cut its connection mid-stream: the router
/// logged `tail_bytes=267` while the client received none of them.
///
/// The salvaged tail is therefore given a short window ([`TAIL_FLUSH_WINDOW`]) to leave the
/// socket before the error is handed over; the error itself is still propagated, so the
/// client sees both the salvaged event and an aborted response.
pub fn merge_usage_stream<S, E>(
    stream: S,
    prefill: Option<Value>,
    label: &'static str,
) -> impl futures::Stream<Item = Result<bytes::Bytes, std::io::Error>>
where
    S: futures::Stream<Item = Result<bytes::Bytes, E>> + Unpin,
    E: std::error::Error + Send + Sync + 'static,
{
    let state = UsageMergeState {
        stream,
        merger: SseUsageMerger::default(),
        prefill,
        label,
        upstream_done: false,
        pending_error: None,
        flush_tail_before_error: false,
    };
    futures::stream::unfold(state, |mut state| async move {
        usage_merge_next(&mut state).await.map(|item| (item, state))
    })
}

/// How long a salvaged tail is allowed to leave the socket before an upstream failure is
/// propagated.
///
/// The window only ever costs latency on the failure path, and only when there actually is
/// a tail to salvage (no tail ⇒ the error is handed over immediately). See
/// [`merge_usage_stream`] for why the delay is needed at all.
const TAIL_FLUSH_WINDOW: std::time::Duration = std::time::Duration::from_millis(100);

/// State of [`merge_usage_stream`] between polls.
struct UsageMergeState<S> {
    stream: S,
    merger: SseUsageMerger,
    prefill: Option<Value>,
    label: &'static str,
    /// Upstream is exhausted or failed; only a flushed tail and/or a pending error are
    /// left to deliver.
    upstream_done: bool,
    /// Failure to deliver **after** the flushed tail (if any).
    pending_error: Option<std::io::Error>,
    /// Whether a tail was just salvaged and therefore needs [`TAIL_FLUSH_WINDOW`] before
    /// the abort.
    flush_tail_before_error: bool,
}

/// Produce the next item of a [`merge_usage_stream`].
async fn usage_merge_next<S, E>(
    state: &mut UsageMergeState<S>,
) -> Option<Result<bytes::Bytes, std::io::Error>>
where
    S: futures::Stream<Item = Result<bytes::Bytes, E>> + Unpin,
    E: std::error::Error + Send + Sync + 'static,
{
    loop {
        if state.upstream_done {
            // Nothing left except a possible failure, delivered only now that any flushed
            // tail has gone out
            if let Some(err) = state.pending_error.take() {
                if state.flush_tail_before_error {
                    state.flush_tail_before_error = false;
                    // Returning `Pending` here is what makes the connection task flush
                    // what it already holds; without it the salvaged tail dies with the
                    // aborted response.
                    tokio::time::sleep(TAIL_FLUSH_WINDOW).await;
                }
                return Some(Err(err));
            }
            return None;
        }

        match futures_util::StreamExt::next(&mut state.stream).await {
            Some(Ok(chunk)) => {
                let passthrough = match state.prefill.as_ref() {
                    Some(prefill) => state.merger.feed(&chunk, prefill),
                    None => chunk.to_vec(),
                };
                if passthrough.is_empty() {
                    continue;
                }
                return Some(Ok(bytes::Bytes::from(passthrough)));
            }
            Some(Err(e)) => {
                state.upstream_done = true;
                // ⚠️ Flush *before* recording the error, and keep the error pending rather
                // than returning it here: the client must first receive the bytes that
                // were already read off the wire from decode.
                let tail = flush_merge_tail(
                    &mut state.merger,
                    state.prefill.as_ref(),
                    state.label,
                    "aborted by an upstream stream error",
                );
                state.pending_error = Some(std::io::Error::other(e));
                if tail.is_empty() {
                    continue;
                }
                // The tail needs a moment on the socket before the abort that follows it
                state.flush_tail_before_error = true;
                return Some(Ok(bytes::Bytes::from(tail)));
            }
            None => {
                state.upstream_done = true;
                let tail = flush_merge_tail(
                    &mut state.merger,
                    state.prefill.as_ref(),
                    state.label,
                    "finished",
                );
                if tail.is_empty() {
                    return None;
                }
                return Some(Ok(bytes::Bytes::from(tail)));
            }
        }
    }
}

/// Release whatever the merger still holds and log the outcome.
fn flush_merge_tail(
    merger: &mut SseUsageMerger,
    prefill: Option<&Value>,
    label: &str,
    reason: &str,
) -> Vec<u8> {
    let tail = match prefill {
        Some(prefill) => merger.finish(prefill),
        None => Vec::new(),
    };
    // info! rather than debug!: "did merging happen at all" is the conclusion of this
    // whole path and debug! is invisible at the default level.
    info!(
        "{} {}: merged={}, tail_bytes={}",
        label,
        reason,
        merger.did_merge(),
        tail.len()
    );
    tail
}

/// Merge prompt_logprobs from prefill response into decode response.
///
/// Handles both Completions API (prompt_logprobs in choices) and
/// Chat Completions API (prompt_logprobs at top level).
///
/// For Completions API with echo=true and logprobs, we need to merge:
/// 1. choices[].prompt_logprobs - top-level per-choice field
/// 2. choices[].logprobs.token_logprobs - flattened array of all logprobs
/// 3. choices[].logprobs.tokens - token strings
/// 4. choices[].logprobs.text_offset - text offsets (with adjustment)
/// 5. choices[].logprobs.top_logprobs - alternative tokens with logprobs
///
/// # Arguments
/// * `prefill_json` - The prefill response JSON
/// * `decode_json` - The decode response JSON (will be modified in place)
///
/// # Returns
/// * `bool` - Whether any logprobs were merged
pub fn merge_logprobs_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
    let mut merged = merge_usage_in_json(prefill_json, decode_json);

    // 1. Try to merge meta_info/input_token_logprobs (for Generate API)
    if let (Some(prefill_meta), Some(decode_meta)) = (
        prefill_json.get("meta_info"),
        decode_json.get_mut("meta_info"),
    ) {
        if let (Some(prefill_logprobs), Some(decode_logprobs)) = (
            prefill_meta.get("input_token_logprobs"),
            decode_meta.get_mut("input_token_logprobs"),
        ) {
            if let (Some(prefill_arr), Some(decode_arr)) =
                (prefill_logprobs.as_array(), decode_logprobs.as_array_mut())
            {
                let mut merged_logprobs = prefill_arr.clone();
                merged_logprobs.extend(decode_arr.clone());
                decode_meta["input_token_logprobs"] = Value::Array(merged_logprobs);
                merged = true;
            }
        }
    }

    // 2. Try to merge prompt_logprobs (for Chat Completions API)
    // Chat Completions: prompt_logprobs is at top level
    if let Some(prefill_prompt_logprobs) = prefill_json.get("prompt_logprobs") {
        // Insert into decode response at top level
        if let Some(decode_obj) = decode_json.as_object_mut() {
            decode_obj.insert(
                "prompt_logprobs".to_string(),
                prefill_prompt_logprobs.clone(),
            );
            merged = true;
        }
    }

    // 3. Try to merge prompt_logprobs in choices (for Completions API)
    // Completions: prompt_logprobs is inside each choice
    if let Some(choices) = decode_json
        .get_mut("choices")
        .and_then(|v| v.as_array_mut())
    {
        if let Some(prefill_choices) = prefill_json.get("choices").and_then(|v| v.as_array()) {
            // Merge prompt_logprobs from prefill choices into decode choices
            for (decode_choice, prefill_choice) in choices.iter_mut().zip(prefill_choices.iter()) {
                if let (Some(decode_obj), Some(prefill_obj)) =
                    (decode_choice.as_object_mut(), prefill_choice.as_object())
                {
                    // 3.1. Merge top-level prompt_logprobs field
                    if let Some(prefill_prompt_logprobs) = prefill_obj.get("prompt_logprobs") {
                        debug!(
                            "[LOGPROBS MERGE] Merging prompt_logprobs from prefill choice into decode choice: {} items",
                            prefill_prompt_logprobs.as_array().map(|a| a.len()).unwrap_or(0)
                        );
                        decode_obj.insert(
                            "prompt_logprobs".to_string(),
                            prefill_prompt_logprobs.clone(),
                        );
                        merged = true;
                    } else {
                        debug!("[LOGPROBS MERGE] No prompt_logprobs found in prefill choice");
                    }

                    // 3.2. Merge logprobs object (token_logprobs, tokens, text_offset, top_logprobs)
                    if let (Some(prefill_logprobs), Some(decode_logprobs)) =
                        (prefill_obj.get("logprobs"), decode_obj.get_mut("logprobs"))
                    {
                        if let (Some(prefill_logprobs_obj), Some(decode_logprobs_obj)) = (
                            prefill_logprobs.as_object(),
                            decode_logprobs.as_object_mut(),
                        ) {
                            // Determine how many prompt tokens there are from prompt_logprobs
                            // Prefill generates with max_tokens=1, so it has [prompt_tokens] + [1 output token]
                            // We only want the prompt tokens, not prefill's output token
                            let num_prompt_tokens = prefill_obj
                                .get("prompt_logprobs")
                                .and_then(|v| v.as_array())
                                .map(|arr| arr.len())
                                .unwrap_or(0);

                            // Merge token_logprobs: [prefill_PROMPT_logprobs_only] + [decode_ALL_logprobs]
                            if let (Some(prefill_token_logprobs), Some(decode_token_logprobs)) = (
                                prefill_logprobs_obj
                                    .get("token_logprobs")
                                    .and_then(|v| v.as_array()),
                                decode_logprobs_obj
                                    .get("token_logprobs")
                                    .and_then(|v| v.as_array()),
                            ) {
                                // Extract only prompt logprobs from prefill (exclude the 1 output token)
                                let prefill_prompt_only = &prefill_token_logprobs
                                    [..num_prompt_tokens.min(prefill_token_logprobs.len())];
                                let prefill_prompt_len = prefill_prompt_only.len();
                                let decode_len = decode_token_logprobs.len();
                                let mut merged_token_logprobs = prefill_prompt_only.to_vec();
                                merged_token_logprobs.extend(decode_token_logprobs.clone());
                                let merged_len = merged_token_logprobs.len();
                                decode_logprobs_obj.insert(
                                    "token_logprobs".to_string(),
                                    Value::Array(merged_token_logprobs),
                                );
                                debug!(
                                    "[LOGPROBS MERGE] Merged token_logprobs: {} prompt (from prefill) + {} all (from decode) = {} total",
                                    prefill_prompt_len,
                                    decode_len,
                                    merged_len
                                );
                                merged = true;
                            }

                            // Merge tokens: [prefill_PROMPT_tokens_only] + [decode_ALL_tokens]
                            if let (Some(prefill_tokens), Some(decode_tokens)) = (
                                prefill_logprobs_obj
                                    .get("tokens")
                                    .and_then(|v| v.as_array()),
                                decode_logprobs_obj.get("tokens").and_then(|v| v.as_array()),
                            ) {
                                // Extract only prompt tokens from prefill (exclude the 1 output token)
                                let prefill_prompt_tokens_only =
                                    &prefill_tokens[..num_prompt_tokens.min(prefill_tokens.len())];
                                let prefill_prompt_len = prefill_prompt_tokens_only.len();
                                let decode_len = decode_tokens.len();
                                let mut merged_tokens = prefill_prompt_tokens_only.to_vec();
                                merged_tokens.extend(decode_tokens.clone());
                                let merged_len = merged_tokens.len();
                                decode_logprobs_obj
                                    .insert("tokens".to_string(), Value::Array(merged_tokens));
                                debug!(
                                    "[LOGPROBS MERGE] Merged tokens: {} prompt + {} all = {} total",
                                    prefill_prompt_len, decode_len, merged_len
                                );
                                merged = true;
                            }

                            // Merge text_offset: [prefill_PROMPT_offsets_only] + [decode_ALL_offsets_adjusted]
                            if let (Some(prefill_offsets), Some(decode_offsets)) = (
                                prefill_logprobs_obj
                                    .get("text_offset")
                                    .and_then(|v| v.as_array()),
                                decode_logprobs_obj
                                    .get("text_offset")
                                    .and_then(|v| v.as_array()),
                            ) {
                                // Extract only prompt offsets from prefill (exclude the 1 output token)
                                let prefill_prompt_offsets_only = &prefill_offsets
                                    [..num_prompt_tokens.min(prefill_offsets.len())];

                                let mut merged_offsets = prefill_prompt_offsets_only.to_vec();

                                // Decode offsets need to be adjusted by the last prefill prompt offset
                                if !prefill_prompt_offsets_only.is_empty() {
                                    let last_prefill_offset = prefill_prompt_offsets_only
                                        .last()
                                        .and_then(|v| v.as_i64())
                                        .unwrap_or(0);

                                    // Get the length of the last prefill prompt token to compute the base offset
                                    let base_offset = if let Some(prefill_tokens_arr) =
                                        prefill_logprobs_obj
                                            .get("tokens")
                                            .and_then(|v| v.as_array())
                                    {
                                        if num_prompt_tokens > 0
                                            && prefill_tokens_arr.len() >= num_prompt_tokens
                                        {
                                            let last_token =
                                                &prefill_tokens_arr[num_prompt_tokens - 1];
                                            let last_token_len = last_token
                                                .as_str()
                                                .map(|s| s.len() as i64)
                                                .unwrap_or(0);
                                            last_prefill_offset + last_token_len
                                        } else {
                                            last_prefill_offset
                                        }
                                    } else {
                                        last_prefill_offset
                                    };

                                    // Adjust decode offsets by adding base_offset
                                    let adjusted_decode_offsets: Vec<Value> = decode_offsets
                                        .iter()
                                        .filter_map(|v| {
                                            v.as_i64()
                                                .map(|offset| Value::from(offset + base_offset))
                                        })
                                        .collect();

                                    merged_offsets.extend(adjusted_decode_offsets);
                                    debug!(
                                        "[LOGPROBS MERGE] Merged text_offset: {} prompt + {} all (adjusted by {}) = {} total",
                                        prefill_prompt_offsets_only.len(),
                                        decode_offsets.len(),
                                        base_offset,
                                        merged_offsets.len()
                                    );
                                } else {
                                    merged_offsets.extend(decode_offsets.clone());
                                }

                                decode_logprobs_obj.insert(
                                    "text_offset".to_string(),
                                    Value::Array(merged_offsets),
                                );
                                merged = true;
                            }

                            // Merge top_logprobs: [prefill_PROMPT_top_logprobs_only] + [decode_ALL_top_logprobs]
                            if let (Some(prefill_top_logprobs), Some(decode_top_logprobs)) = (
                                prefill_logprobs_obj
                                    .get("top_logprobs")
                                    .and_then(|v| v.as_array()),
                                decode_logprobs_obj
                                    .get("top_logprobs")
                                    .and_then(|v| v.as_array()),
                            ) {
                                // Extract only prompt top_logprobs from prefill (exclude the 1 output token)
                                let prefill_prompt_top_only = &prefill_top_logprobs
                                    [..num_prompt_tokens.min(prefill_top_logprobs.len())];
                                let prefill_prompt_len = prefill_prompt_top_only.len();
                                let decode_len = decode_top_logprobs.len();
                                let mut merged_top_logprobs = prefill_prompt_top_only.to_vec();
                                merged_top_logprobs.extend(decode_top_logprobs.clone());
                                let merged_len = merged_top_logprobs.len();
                                decode_logprobs_obj.insert(
                                    "top_logprobs".to_string(),
                                    Value::Array(merged_top_logprobs),
                                );
                                debug!(
                                    "[LOGPROBS MERGE] Merged top_logprobs: {} prompt + {} all = {} total",
                                    prefill_prompt_len,
                                    decode_len,
                                    merged_len
                                );
                                merged = true;
                            }
                        }
                    }
                }
            }
        }
    }

    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt as _;
    use serde_json::json;

    #[test]
    fn test_merge_completions_api_logprobs() {
        let prefill_json = json!({
            "choices": [{
                "prompt_logprobs": [null, -0.5, -1.2],
                "logprobs": {
                    "token_logprobs": [null, -0.5, -1.2, -2.1],
                    "tokens": ["Hello", " world", " test", " extra"],
                    "text_offset": [0, 5, 11, 16],
                    "top_logprobs": [null, {" world": -0.5}, {" test": -1.2}, {" extra": -2.1}]
                }
            }]
        });

        let mut decode_json = json!({
            "choices": [{
                "logprobs": {
                    "token_logprobs": [-3.5, -4.2],
                    "tokens": [" output", " token"],
                    "text_offset": [0, 7],
                    "top_logprobs": [{" output": -3.5}, {" token": -4.2}]
                }
            }]
        });

        let merged = merge_logprobs_in_json(&prefill_json, &mut decode_json);
        assert!(merged);

        // Check prompt_logprobs was added
        assert_eq!(
            decode_json["choices"][0]["prompt_logprobs"],
            json!([null, -0.5, -1.2])
        );

        // Check token_logprobs merged correctly (3 prompt + 2 decode)
        let merged_token_logprobs = decode_json["choices"][0]["logprobs"]["token_logprobs"]
            .as_array()
            .unwrap();
        assert_eq!(merged_token_logprobs.len(), 5);

        // Check tokens merged correctly
        let merged_tokens = decode_json["choices"][0]["logprobs"]["tokens"]
            .as_array()
            .unwrap();
        assert_eq!(merged_tokens.len(), 5);
        assert_eq!(merged_tokens[0], "Hello");
        assert_eq!(merged_tokens[4], " token");

        // Check text_offset adjusted correctly
        let merged_offsets = decode_json["choices"][0]["logprobs"]["text_offset"]
            .as_array()
            .unwrap();
        assert_eq!(merged_offsets.len(), 5);
        // We take first 3 prompt offsets [0, 5, 11]. Last prompt offset is 11,
        // last prompt token " test" has length 5, so base is 11 + 5 = 16
        assert_eq!(merged_offsets[0].as_i64().unwrap(), 0); // Prompt token "Hello"
        assert_eq!(merged_offsets[1].as_i64().unwrap(), 5); // Prompt token " world"
        assert_eq!(merged_offsets[2].as_i64().unwrap(), 11); // Prompt token " test"
        assert_eq!(merged_offsets[3].as_i64().unwrap(), 16); // Decode token " output" (0 + 16)
        assert_eq!(merged_offsets[4].as_i64().unwrap(), 23); // Decode token " token" (7 + 16)
    }

    #[test]
    fn test_merge_usage_prefill_cached_tokens_over_decode_invalid_value() {
        let prefill_json = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 1,
                "total_tokens": 101,
                "prompt_tokens_details": {
                    "cached_tokens": 50
                }
            }
        });

        let mut decode_json = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_tokens_details": {
                    "cached_tokens": -1
                }
            }
        });

        let merged = merge_usage_in_json(&prefill_json, &mut decode_json);
        assert!(merged);
        assert_eq!(
            decode_json["usage"]["prompt_tokens_details"]["cached_tokens"],
            json!(50)
        );
    }

    #[test]
    fn test_merge_usage_keeps_valid_decode_cached_tokens() {
        let prefill_json = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 1,
                "total_tokens": 101,
                "prompt_tokens_details": {
                    "cached_tokens": 50
                }
            }
        });

        let mut decode_json = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_tokens_details": {
                    "cached_tokens": 12
                }
            }
        });

        let merged = merge_usage_in_json(&prefill_json, &mut decode_json);
        assert!(!merged);
        assert_eq!(
            decode_json["usage"]["prompt_tokens_details"]["cached_tokens"],
            json!(12)
        );
    }

    #[test]
    fn test_merge_usage_fills_missing_cached_tokens() {
        let prefill_json = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 1,
                "total_tokens": 101,
                "prompt_tokens_details": {
                    "cached_tokens": 50
                }
            }
        });

        let mut decode_json = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_tokens_details": {}
            }
        });

        let merged = merge_usage_in_json(&prefill_json, &mut decode_json);
        assert!(merged);
        assert_eq!(
            decode_json["usage"]["prompt_tokens_details"]["cached_tokens"],
            json!(50)
        );
    }

    /// A **legitimately shaped `null`** in decode must be repaired like a missing key.
    ///
    /// `"prompt_tokens_details": null` is a real engine shape (not a corrupt one), and it
    /// is exactly where the old `entry(..).or_insert(..)` implementation gave up: the key
    /// is present, so `entry` returns the `Null` value, `as_object_mut()` fails, and the
    /// function returns without filling anything in — the client keeps seeing the invalid
    /// decode-side value.
    #[test]
    fn test_merge_usage_repairs_null_prompt_tokens_details() {
        let prefill_json = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 1,
                "total_tokens": 101,
                "prompt_tokens_details": { "cached_tokens": 50 }
            }
        });

        let mut decode_json = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_tokens_details": null
            }
        });

        let merged = merge_usage_in_json(&prefill_json, &mut decode_json);
        assert!(
            merged,
            "a null details object must be repaired, not treated as final"
        );
        assert_eq!(
            decode_json["usage"]["prompt_tokens_details"],
            json!({ "cached_tokens": 50 }),
            "the null must be replaced by a usable details object"
        );
    }

    /// Same `null` shape, but reported under the `input_tokens_details` spelling.
    #[test]
    fn test_merge_usage_repairs_null_input_tokens_details() {
        let prefill_json = json!({
            "usage": {
                "input_tokens": 100,
                "output_tokens": 1,
                "total_tokens": 101,
                "input_tokens_details": { "cached_tokens": 41 }
            }
        });

        let mut decode_json = json!({
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "input_tokens_details": null
            }
        });

        let merged = merge_usage_in_json(&prefill_json, &mut decode_json);
        assert!(merged);
        assert_eq!(
            decode_json["usage"]["input_tokens_details"],
            json!({ "cached_tokens": 41 }),
            "decode's spelling must be kept when it already has the key"
        );
    }

    /// A `usage` that is present but `null` is the same class of shape problem one level
    /// up: prefill's usage is copied in rather than leaving the client with `null`.
    #[test]
    fn test_merge_usage_fills_null_usage() {
        let prefill_json = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 1,
                "total_tokens": 101,
                "prompt_tokens_details": { "cached_tokens": 50 }
            }
        });

        let mut decode_json = json!({
            "choices": [{ "message": { "content": "ok" } }],
            "usage": null
        });

        let merged = merge_usage_in_json(&prefill_json, &mut decode_json);
        assert!(merged);
        assert_eq!(
            decode_json["usage"]["prompt_tokens_details"]["cached_tokens"],
            json!(50)
        );
    }

    /// A **Responses API streaming event** carries the usage one level down, under
    /// `response`:
    ///
    /// ```text
    /// {"type":"response.completed","response":{...,"usage":{"input_tokens_details":{...}}}}
    /// ```
    ///
    /// `/v1/responses` goes through the same transformer (`route_transparent` →
    /// `process_vllm_two_stage_request`), so writing prefill's usage at the event's **top
    /// level** both invents a field no client reads and leaves the real
    /// `response.usage.input_tokens_details.cached_tokens` at the decode-side placeholder.
    #[test]
    fn test_merge_usage_repairs_responses_api_event_nested_usage() {
        let prefill_json = json!({
            "usage": {
                "input_tokens": 100,
                "output_tokens": 1,
                "total_tokens": 101,
                "input_tokens_details": { "cached_tokens": 50 }
            }
        });

        let mut decode_json = json!({
            "type": "response.completed",
            "sequence_number": 3,
            "response": {
                "id": "resp_1",
                "object": "response",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "total_tokens": 110,
                    "input_tokens_details": { "cached_tokens": -1 }
                }
            }
        });

        let merged = merge_usage_in_json(&prefill_json, &mut decode_json);
        assert!(merged);
        assert_eq!(
            decode_json["response"]["usage"]["input_tokens_details"]["cached_tokens"],
            json!(50),
            "the nested usage must be the one repaired"
        );
        assert_eq!(
            decode_json["response"]["usage"]["output_tokens"],
            json!(10),
            "decode's own counters must survive"
        );
        assert!(
            decode_json.get("usage").is_none(),
            "no top-level usage may be added to a Responses event: {decode_json}"
        );
    }

    /// The same shape, end to end through the SSE merger: a Responses stream of deltas plus
    /// a final `response.completed`.
    #[test]
    fn test_sse_merger_merges_responses_api_stream() {
        let prefill = json!({
            "usage": {
                "input_tokens": 100,
                "output_tokens": 1,
                "total_tokens": 101,
                "input_tokens_details": { "cached_tokens": 50 }
            }
        });
        let stream = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,",
            "\"delta\":\"Hel\"}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":2,",
            "\"delta\":\"lo\"}\n\n",
            "data: {\"type\":\"response.completed\",\"sequence_number\":3,\"response\":{",
            "\"id\":\"resp_1\",\"usage\":{\"input_tokens\":100,\"output_tokens\":2,",
            "\"total_tokens\":102,\"input_tokens_details\":{\"cached_tokens\":-1}}}}\n\n",
            "data: [DONE]\n\n",
        );

        let mut merger = SseUsageMerger::default();
        let out = feed_all(&mut merger, stream.as_bytes(), &prefill);
        let events = sse_events(&out);

        assert!(merger.did_merge());
        assert_eq!(events.len(), 4, "3 data events + [DONE]");

        // Deltas are untouched and still flow through
        let first_delta = sse_payload_json(&events[0]);
        assert_eq!(first_delta["type"], json!("response.output_text.delta"));
        assert_eq!(first_delta["delta"], json!("Hel"));

        let completed = sse_payload_json(&events[2]);
        assert_eq!(completed["type"], json!("response.completed"));
        assert_eq!(
            completed["response"]["usage"]["input_tokens_details"]["cached_tokens"],
            json!(50),
            "the client reads usage from response.usage, so that is where 50 must land"
        );
        assert!(
            completed.get("usage").is_none(),
            "no top-level usage may be added to a Responses event: {completed}"
        );
        assert_eq!(events[3], "data: [DONE]");
    }

    #[test]
    fn test_merge_usage_uses_input_tokens_details_fallback() {
        let prefill_json = json!({
            "usage": {
                "input_tokens": 100,
                "output_tokens": 1,
                "total_tokens": 101,
                "input_tokens_details": {
                    "cached_tokens": 41
                }
            }
        });

        let mut decode_json = json!({
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "prompt_tokens_details": {
                    "cached_tokens": -1
                }
            }
        });

        let merged = merge_usage_in_json(&prefill_json, &mut decode_json);
        assert!(merged);
        assert_eq!(
            decode_json["usage"]["prompt_tokens_details"]["cached_tokens"],
            json!(41)
        );
    }

    #[test]
    fn test_merge_usage_copies_usage_when_decode_missing() {
        let prefill_json = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 1,
                "total_tokens": 101,
                "prompt_tokens_details": {
                    "cached_tokens": 50
                }
            }
        });

        let mut decode_json = json!({
            "choices": [{ "message": { "content": "ok" } }]
        });

        let merged = merge_usage_in_json(&prefill_json, &mut decode_json);
        assert!(merged);
        assert_eq!(
            decode_json["usage"]["prompt_tokens_details"]["cached_tokens"],
            json!(50)
        );
    }

    #[test]
    fn test_merge_usage_noop_without_prefill_usage() {
        let prefill_json = json!({ "choices": [{ "message": { "content": "ok" } }] });
        let mut decode_json = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_tokens_details": {
                    "cached_tokens": -1
                }
            }
        });

        let merged = merge_usage_in_json(&prefill_json, &mut decode_json);
        assert!(!merged);
        assert_eq!(
            decode_json["usage"]["prompt_tokens_details"]["cached_tokens"],
            json!(-1)
        );
    }

    #[test]
    fn test_merge_chat_completions_api_logprobs() {
        let prefill_json = json!({
            "prompt_logprobs": [null, -0.5, -1.2]
        });

        let mut decode_json = json!({
            "choices": [{
                "message": {"content": "response"}
            }]
        });

        let merged = merge_logprobs_in_json(&prefill_json, &mut decode_json);
        assert!(merged);

        assert_eq!(decode_json["prompt_logprobs"], json!([null, -0.5, -1.2]));
    }

    // ---------------------------------------------------------------- SSE streaming merge
    //
    // These cases cover the fix for "cached_tokens is not corrected by the prefill value on
    // streaming requests". Key invariant: **intermediate chunks must be sent immediately**
    // and must not be held back until the end of the stream — otherwise streaming
    // degenerates into "buffer the whole response, then send". vLLM's first chunk carries
    // `"usage":null`, which is the easiest trap to fall into.

    /// Build a prefill-side response (with cached_tokens).
    fn prefill_with_cached(cached: i64) -> Value {
        json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 1,
                "total_tokens": 101,
                "prompt_tokens_details": { "cached_tokens": cached }
            }
        })
    }

    /// Split the bytes of a whole stream into event texts on `\n`, so the order can be
    /// asserted.
    fn sse_lines(bytes: &[u8]) -> Vec<String> {
        String::from_utf8_lossy(bytes)
            .split('\n')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn feed_all(merger: &mut SseUsageMerger, input: &[u8], prefill: &Value) -> Vec<u8> {
        let mut out = merger.feed(input, prefill);
        out.extend(merger.finish(prefill));
        out
    }

    #[test]
    fn test_sse_merger_merges_last_chunk_usage() {
        let prefill = prefill_with_cached(50);
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}],\"usage\":null}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}],\"usage\":null}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,",
            "\"prompt_tokens_details\":{\"cached_tokens\":-1}}}\n\n",
            "data: [DONE]\n\n",
        );

        let mut merger = SseUsageMerger::default();
        let out = feed_all(&mut merger, stream.as_bytes(), &prefill);
        let lines = sse_lines(&out);

        assert!(merger.did_merge(), "a merge should have happened");
        assert_eq!(lines.len(), 4);
        assert!(lines[0].contains("\"Hel\""));
        assert!(lines[1].contains("\"lo\""));
        assert!(
            lines[2].contains("\"cached_tokens\":50"),
            "the last chunk should be rewritten: {}",
            lines[2]
        );
        assert!(
            !lines[2].contains("\"cached_tokens\":-1"),
            "the placeholder should be replaced"
        );
        assert_eq!(lines[3], "data: [DONE]");
    }

    /// **The most important one**: the first chunk's `"usage":null` must not trigger
    /// holding, otherwise streaming breaks.
    ///
    /// ⚠️ This has to check **whether the chunk is sent immediately**, not merely that "the
    /// content shows up in the output eventually". Even if the first chunk were wrongly
    /// held, it would be released when `[DONE]` arrives and the content would still appear
    /// — a content-only assertion would **pass falsely** and would not catch the "first
    /// chunk pushed to the end of the stream" regression.
    #[test]
    fn test_sse_merger_does_not_hold_null_usage_chunk() {
        let prefill = prefill_with_cached(50);
        let mut merger = SseUsageMerger::default();

        let first = "data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}],\"usage\":null}\n\n";
        let out = merger.feed(first.as_bytes(), &prefill);

        assert!(
            String::from_utf8_lossy(&out).contains("\"A\""),
            "a first chunk carrying \"usage\":null must be sent immediately, not held"
        );
        // The whole first chunk (including its trailing blank line) should be consumed,
        // leaving no residue for later output
        assert_eq!(
            out,
            first.as_bytes(),
            "the first chunk should be sent immediately byte for byte, got: {:?}",
            String::from_utf8_lossy(&out)
        );

        // Feed a real usage line afterwards: if the first chunk had been held by mistake,
        // it would surface here
        let tail = "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":\
                    {\"cached_tokens\":-1}}}\n\ndata: [DONE]\n\n";
        let rest = feed_all(&mut merger, tail.as_bytes(), &prefill);
        let text = String::from_utf8_lossy(&rest);

        assert!(
            !text.contains("\"A\""),
            "the first chunk must not be deferred to here: {text:?}"
        );
        assert_eq!(
            text.matches("data: ").count(),
            2,
            "there should be exactly two data events (the usage line + DONE): {text:?}"
        );
        assert!(merger.did_merge());
    }

    /// The byte stream can split at any position; lines must be reassembled correctly
    /// across chunks.
    #[test]
    fn test_sse_merger_handles_chunk_boundary_split() {
        let prefill = prefill_with_cached(50);
        let json = "{\"choices\":[],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":-1}}}";
        let stream = format!("data: {json}\n\ndata: [DONE]\n\n");

        let cut = stream.find("cached_tokens").unwrap() + 3;
        let mut merger = SseUsageMerger::default();
        let mut out = merger.feed(&stream.as_bytes()[..cut], &prefill);
        out.extend(merger.feed(&stream.as_bytes()[cut..], &prefill));
        out.extend(merger.finish(&prefill));

        let lines = sse_lines(&out);
        assert!(merger.did_merge());
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"cached_tokens\":50"), "{}", lines[0]);
        assert_eq!(lines[1], "data: [DONE]");
    }

    /// Feed one byte at a time, stressing the state machine.
    #[test]
    fn test_sse_merger_byte_by_byte() {
        let prefill = prefill_with_cached(50);
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}],\"usage\":null}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":",
            "{\"cached_tokens\":-1}}}\n\n",
            "data: [DONE]\n\n",
        );

        let mut merger = SseUsageMerger::default();
        let mut out = Vec::new();
        for b in stream.as_bytes() {
            out.extend(merger.feed(std::slice::from_ref(b), &prefill));
        }
        out.extend(merger.finish(&prefill));

        assert!(merger.did_merge());
        let lines = sse_lines(&out);
        assert_eq!(lines.len(), 3);
        assert!(lines[1].contains("\"cached_tokens\":50"));
        assert_eq!(lines[2], "data: [DONE]");
    }

    /// Stream without usage: nothing is altered and no usage is produced.
    #[test]
    fn test_sse_merger_passthrough_without_usage() {
        let prefill = prefill_with_cached(50);
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"only\"}}]}\n\n",
            "data: [DONE]\n\n",
        );

        let mut merger = SseUsageMerger::default();
        let out = feed_all(&mut merger, stream.as_bytes(), &prefill);

        assert!(!merger.did_merge());
        assert_eq!(
            out,
            stream.as_bytes(),
            "without usage the stream must pass through verbatim"
        );
    }

    /// Prefill provides no usage: nothing may be changed based on prefill.
    #[test]
    fn test_sse_merger_noop_without_prefill_usage() {
        let prefill = json!({"choices": []});
        let stream = concat!(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":",
            "{\"cached_tokens\":-1}}}\n\n",
            "data: [DONE]\n\n",
        );

        let mut merger = SseUsageMerger::default();
        let out = feed_all(&mut merger, stream.as_bytes(), &prefill);

        assert!(!merger.did_merge());
        assert!(
            String::from_utf8_lossy(&out).contains("\"cached_tokens\":-1"),
            "decode's value must not be changed when prefill has no usage"
        );
    }

    /// A valid value reported by decode is kept (consistent with "valid value wins" on the
    /// non-streaming path).
    #[test]
    fn test_sse_merger_keeps_valid_decode_value() {
        let prefill = prefill_with_cached(50);
        let stream = concat!(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":",
            "{\"cached_tokens\":12}}}\n\n",
            "data: [DONE]\n\n",
        );

        let mut merger = SseUsageMerger::default();
        let out = feed_all(&mut merger, stream.as_bytes(), &prefill);

        assert!(!merger.did_merge(), "a valid value must not be overwritten");
        assert!(String::from_utf8_lossy(&out).contains("\"cached_tokens\":12"));
    }

    /// Stream truncated without waiting for `[DONE]`: the held line must not be lost.
    #[test]
    fn test_sse_merger_flushes_held_line_when_stream_truncated() {
        let prefill = prefill_with_cached(50);
        let mut merger = SseUsageMerger::default();

        let stream = "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":\
                      {\"cached_tokens\":-1}}}\n\n";
        let out = merger.feed(stream.as_bytes(), &prefill);
        assert!(
            out.is_empty(),
            "the usage line should be held until it is confirmed to be last"
        );

        let tail = merger.finish(&prefill);
        let lines = sse_lines(&tail);
        assert_eq!(
            lines.len(),
            1,
            "the held line must be flushed at end of stream, otherwise data is lost"
        );
        assert!(lines[0].contains("\"cached_tokens\":50"));
    }

    /// A complete trailing data line with no newline: it must be handled too, rather than
    /// dropped as garbage.
    #[test]
    fn test_sse_merger_handles_unterminated_trailing_line() {
        let prefill = prefill_with_cached(50);
        let mut merger = SseUsageMerger::default();

        // Note there is no trailing `\n`
        let stream = "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":\
                      {\"cached_tokens\":-1}}}";
        let out = feed_all(&mut merger, stream.as_bytes(), &prefill);
        let lines = sse_lines(&out);

        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("\"cached_tokens\":50"), "{}", lines[0]);
    }

    /// Regression: a rewritten line must keep its line terminator, otherwise it fuses with
    /// `data: [DONE]` into a single line.
    ///
    /// This asserts the **raw bytes** rather than the split lines — fusing shows up exactly
    /// as "only one line remains", which a per-line assertion would miss.
    #[test]
    fn test_sse_merger_preserves_line_terminator_after_merge() {
        let prefill = prefill_with_cached(50);
        let stream = concat!(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":",
            "{\"cached_tokens\":-1}}}\n\n",
            "data: [DONE]\n\n",
        );

        let mut merger = SseUsageMerger::default();
        let out = feed_all(&mut merger, stream.as_bytes(), &prefill);

        assert!(merger.did_merge());
        // Assert the complete byte sequence instead of a hand-written needle.
        //
        // Lesson: a hand-written needle got this wrong twice for this case (first one `}`
        // too few; then "add one `}`" was **still not enough** — after `cached_tokens`
        // there are actually **three** `}` closing the cached_tokens object, the
        // prompt_tokens_details object and the usage object in turn). Hand-written needles
        // are extremely error-prone, and a wrong one is constantly false, staying red and
        // hiding whether the implementation is actually correct. Comparing the whole
        // string lets the implementation define "correct output" itself.
        //
        // Note that adjacent byte literals in Rust are **not** concatenated automatically
        // (that is C behaviour; here it would produce a tuple) — `concat!` is required.
        let expected = concat!(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":",
            "{\"cached_tokens\":50}}}\n\ndata: [DONE]\n\n"
        );
        assert_eq!(
            out.as_slice(),
            expected.as_bytes(),
            "\n  expected: {:?}\n  actual:   {:?}",
            expected,
            String::from_utf8_lossy(&out)
        );
    }

    /// Same as above, but covering CRLF line endings.
    #[test]
    fn test_sse_merger_preserves_crlf_terminator() {
        let prefill = prefill_with_cached(50);
        let stream = "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":\
                      {\"cached_tokens\":-1}}}\r\ndata: [DONE]\r\n";

        let mut merger = SseUsageMerger::default();
        let out = feed_all(&mut merger, stream.as_bytes(), &prefill);

        assert!(merger.did_merge());
        // Whole-string comparison again; adjacent literals need concat!
        let expected = concat!(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":",
            "{\"cached_tokens\":50}}}\r\ndata: [DONE]\r\n"
        );
        assert_eq!(
            out.as_slice(),
            expected.as_bytes(),
            "\n  expected: {:?}\n  actual:   {:?}",
            expected,
            String::from_utf8_lossy(&out)
        );
    }

    /// **Worst case** of a truncated stream whose held line has not been released yet: the
    /// cut lands right after the usage line and before the blank line. Here held contains
    /// only the data line, and finish must flush it.
    ///
    /// Differs from `flushes_held_line_when_stream_truncated`, where the cut lands after
    /// the blank line.
    #[test]
    fn test_sse_merger_flushes_on_truncation_before_blank_line() {
        let prefill = prefill_with_cached(50);
        let mut merger = SseUsageMerger::default();

        // Feed only the data line itself, without its trailing blank line
        let stream = b"data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":-1}}}\n";
        let out = merger.feed(stream, &prefill);
        assert!(
            out.is_empty(),
            "should be held, got: {:?}",
            String::from_utf8_lossy(&out)
        );

        let tail = merger.finish(&prefill);
        let text = String::from_utf8_lossy(&tail);
        assert!(
            text.contains("\"cached_tokens\":50"),
            "the held line must be flushed: {text:?}"
        );
    }

    // ------------------------------------------------ SSE event separation
    //
    // A held line has not been terminated on the wire yet (it is `data: {...}\n` plus, at
    // most, the blank line that closes it). If a **later** line is emitted while it is
    // still held, that later event's terminating blank line lands in `held` instead, and
    // the client receives both JSON documents inside a **single** SSE event:
    //
    //   data: {content}\n
    //   data: {error}\n
    //   \n              ← one event, two payloads, not valid JSON
    //
    // These tests therefore assert **event boundaries**, not just content: every event must
    // carry exactly one `data:` line whose payload parses as JSON on its own.

    /// Split raw SSE bytes into events (blank-line separated) after checking that each one
    /// carries exactly one `data:` line — i.e. that a client can parse every event as a
    /// single JSON document.
    fn sse_events(bytes: &[u8]) -> Vec<String> {
        let text = String::from_utf8_lossy(bytes).replace("\r\n", "\n");
        text.split("\n\n")
            .map(|event| event.trim().to_string())
            .filter(|event| !event.is_empty())
            .map(|event| {
                assert_eq!(
                    event.matches("data:").count(),
                    1,
                    "an SSE event must carry exactly one data line, got: {event:?}"
                );
                event
            })
            .collect()
    }

    /// The payload of an event, parsed as JSON (panics with the event text when it is not a
    /// single valid JSON document).
    fn sse_payload_json(event: &str) -> Value {
        let payload = event
            .strip_prefix("data: ")
            .unwrap_or_else(|| panic!("not a data event: {event:?}"));
        serde_json::from_str(payload)
            .unwrap_or_else(|e| panic!("payload is not one JSON document ({e}): {event:?}"))
    }

    /// A usage-carrying content chunk followed by an error frame **without** usage: the
    /// held content chunk must be released first, so the two payloads end up in two
    /// separate events (the old code emitted the error line while the content line was
    /// still held, gluing both into one unparseable event).
    #[test]
    fn test_sse_merger_releases_held_line_before_error_event() {
        let prefill = prefill_with_cached(50);
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}],\"usage\":",
            "{\"prompt_tokens_details\":{\"cached_tokens\":-1}}}\n\n",
            "data: {\"error\":{\"message\":\"upstream failed\"}}\n\n",
        );

        let mut merger = SseUsageMerger::default();
        let out = feed_all(&mut merger, stream.as_bytes(), &prefill);

        let events = sse_events(&out);
        assert_eq!(
            events.len(),
            2,
            "content and error must be two events, got: {:?}",
            String::from_utf8_lossy(&out)
        );

        let content = sse_payload_json(&events[0]);
        assert_eq!(content["choices"][0]["delta"]["content"], json!("Hel"));
        assert_eq!(
            content["usage"]["prompt_tokens_details"]["cached_tokens"],
            json!(50)
        );

        let error = sse_payload_json(&events[1]);
        assert_eq!(error["error"]["message"], json!("upstream failed"));
        assert!(merger.did_merge());
    }

    /// The per-chunk usage mode the bug report describes: **every** content chunk is held
    /// until the next line, and an error frame closes the stream. All three events must
    /// arrive in order, each on its own.
    #[test]
    fn test_sse_merger_keeps_event_boundaries_with_per_chunk_usage() {
        let prefill = prefill_with_cached(50);
        let mut merger = SseUsageMerger::default();

        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}],\"usage\":",
            "{\"prompt_tokens_details\":{\"cached_tokens\":-1}}}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}],\"usage\":",
            "{\"prompt_tokens_details\":{\"cached_tokens\":-1}}}\n\n",
            "data: {\"error\":{\"message\":\"boom\"}}\n\n",
        );

        // Feed it in two pieces as well, so the held/release dance also crosses a chunk
        // boundary (the realistic case: the error arrives in its own network read)
        let cut = stream.find("\"B\"").unwrap();
        let mut out = merger.feed(&stream.as_bytes()[..cut], &prefill);
        out.extend(merger.feed(&stream.as_bytes()[cut..], &prefill));
        out.extend(merger.finish(&prefill));

        let events = sse_events(&out);
        assert_eq!(
            events.len(),
            3,
            "expected content A, content B and the error as three events, got: {:?}",
            String::from_utf8_lossy(&out)
        );

        let first = sse_payload_json(&events[0]);
        let second = sse_payload_json(&events[1]);
        assert_eq!(first["choices"][0]["delta"]["content"], json!("A"));
        assert_eq!(second["choices"][0]["delta"]["content"], json!("B"));
        assert_eq!(
            first["usage"]["prompt_tokens_details"]["cached_tokens"],
            json!(50)
        );
        assert_eq!(
            second["usage"]["prompt_tokens_details"]["cached_tokens"],
            json!(50)
        );
        assert_eq!(
            sse_payload_json(&events[2])["error"]["message"],
            json!("boom")
        );
    }

    /// Ordering must survive any other kind of line as well: an SSE comment (`: ping`) is
    /// not a data event but still comes after the held one.
    #[test]
    fn test_sse_merger_releases_held_line_before_comment() {
        let prefill = prefill_with_cached(50);
        let stream = concat!(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":",
            "{\"cached_tokens\":-1}}}\n\n",
            ": keep-alive\n",
            "data: [DONE]\n\n",
        );

        let mut merger = SseUsageMerger::default();
        let out = feed_all(&mut merger, stream.as_bytes(), &prefill);

        assert_eq!(
            out.as_slice(),
            concat!(
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":",
                "{\"cached_tokens\":50}}}\n\n: keep-alive\ndata: [DONE]\n\n"
            )
            .as_bytes(),
            "the held event must be released before the comment, got: {:?}",
            String::from_utf8_lossy(&out)
        );
    }

    // ------------------------------------------------ stream wrapper (upstream failure)

    /// Build a decode-side stream out of byte chunks, optionally ending in a transport
    /// error.
    fn chunk_stream(
        chunks: Vec<&'static [u8]>,
        error: Option<&'static str>,
    ) -> impl futures::Stream<Item = Result<bytes::Bytes, std::io::Error>> + Unpin {
        let mut items: Vec<Result<bytes::Bytes, std::io::Error>> = chunks
            .into_iter()
            .map(|c| Ok(bytes::Bytes::from_static(c)))
            .collect();
        if let Some(message) = error {
            items.push(Err(std::io::Error::other(message)));
        }
        futures::stream::iter(items)
    }

    /// **The connection drops while a usage-carrying chunk is held**: the held bytes (which
    /// in per-chunk usage mode contain generated text) must be delivered **before** the
    /// error, otherwise the response is silently truncated.
    ///
    /// ⚠️ Order matters here, so this asserts the sequence rather than partitioning the
    /// items: the error must be the **last** item, and everything before it must be data.
    #[tokio::test]
    async fn test_merge_usage_stream_flushes_held_tail_before_error() {
        let prefill = prefill_with_cached(50);
        let stream = chunk_stream(
            vec![
                // Hold on: every chunk here carries usage
                b"data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":-1}}}\n\n",
                b"data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":-1}}}\n\n",
            ],
            Some("connection reset by peer"),
        );

        let items = merge_usage_stream(stream, Some(prefill), "test")
            .collect::<Vec<Result<bytes::Bytes, std::io::Error>>>()
            .await;

        assert_eq!(
            items.len(),
            3,
            "content A, the salvaged tail with B, and then the error"
        );
        assert!(items[0].is_ok(), "content A is delivered first");
        assert!(items[1].is_ok(), "the salvaged tail follows it");
        let error = items[2]
            .as_ref()
            .expect_err("the upstream failure must still be reported to the client");
        assert!(
            error.to_string().contains("connection reset"),
            "the original error must be preserved, got: {error}"
        );

        let delivered: Vec<u8> = items[..2]
            .iter()
            .flat_map(|item| item.as_ref().unwrap().to_vec())
            .collect();
        let text = String::from_utf8_lossy(&delivered);
        assert!(
            text.contains("\"A\""),
            "content A must not be lost when the upstream breaks: {text:?}"
        );
        assert!(
            text.contains("\"B\""),
            "the held content B must be flushed before the error: {text:?}"
        );
        assert!(
            text.contains("\"cached_tokens\":50"),
            "the flushed tail keeps the merged usage: {text:?}"
        );
        assert!(
            text.ends_with("\n\n"),
            "the flushed bytes must still be a complete event: {text:?}"
        );
    }

    /// A failure with nothing held: the error is passed through on its own.
    #[tokio::test]
    async fn test_merge_usage_stream_propagates_error_without_tail() {
        let prefill = prefill_with_cached(50);
        let stream = chunk_stream(
            vec![b"data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}],\"usage\":null}\n\n"],
            Some("upstream gone"),
        );

        let items = merge_usage_stream(stream, Some(prefill), "test")
            .collect::<Vec<Result<bytes::Bytes, std::io::Error>>>()
            .await;

        assert_eq!(items.len(), 2);
        assert!(items[0].is_ok(), "the pass-through chunk comes first");
        assert!(items[1].is_err(), "the error must terminate the stream");
    }

    /// Happy path through the wrapper: content flows in order, the merged usage event is
    /// the last data event, and the stream ends without an error.
    #[tokio::test]
    async fn test_merge_usage_stream_normal_completion() {
        let prefill = prefill_with_cached(50);
        let stream = chunk_stream(
            vec![
                b"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}],\"usage\":null}\n\n",
                b"data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":-1}}}\n\n",
                b"data: [DONE]\n\n",
            ],
            None,
        );

        let items = merge_usage_stream(stream, Some(prefill), "test")
            .collect::<Vec<Result<bytes::Bytes, std::io::Error>>>()
            .await;

        let delivered: Vec<u8> = items
            .into_iter()
            .flat_map(|item| item.expect("no error on the happy path").to_vec())
            .collect();
        let events = sse_events(&delivered);

        assert_eq!(events.len(), 3, "two data events plus [DONE]");
        assert_eq!(
            sse_payload_json(&events[0])["choices"][0]["delta"]["content"],
            json!("Hel")
        );
        assert_eq!(
            sse_payload_json(&events[1])["usage"]["prompt_tokens_details"]["cached_tokens"],
            json!(50)
        );
        assert_eq!(events[2], "data: [DONE]");
    }

    /// The wrapper must forward bytes verbatim when prefill carries no response JSON.
    #[tokio::test]
    async fn test_merge_usage_stream_passthrough_without_prefill() {
        let chunk = b"data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":-1}}}\n\n";
        let stream = chunk_stream(vec![chunk, b"data: [DONE]\n\n"], None);

        let items = merge_usage_stream(stream, None, "test")
            .collect::<Vec<Result<bytes::Bytes, std::io::Error>>>()
            .await;

        let delivered: Vec<u8> = items
            .into_iter()
            .flat_map(|item| item.expect("no error").to_vec())
            .collect();
        let mut expected = chunk.to_vec();
        expected.extend_from_slice(b"data: [DONE]\n\n");
        assert_eq!(delivered, expected, "without prefill nothing is rewritten");
    }
}
