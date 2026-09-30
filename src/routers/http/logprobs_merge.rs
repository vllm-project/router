//! Logprobs merging utilities for PD disaggregation
//!
//! This module provides utilities for merging logprobs from prefill and decode responses
//! in prefill-decode disaggregation mode.

use serde_json::Value;
use tracing::debug;

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

    let decode_usage = match decode_json.get_mut("usage") {
        Some(v) => v,
        None => {
            decode_json["usage"] = prefill_usage.clone();
            return true;
        }
    };

    let Some(decode_usage_obj) = decode_usage.as_object_mut() else {
        return false;
    };

    let details_target = if decode_usage_obj.contains_key("prompt_tokens_details") {
        "prompt_tokens_details"
    } else if decode_usage_obj.contains_key("input_tokens_details") {
        "input_tokens_details"
    } else {
        "prompt_tokens_details"
    };

    let decode_details = decode_usage_obj
        .entry(details_target.to_string())
        .or_insert(Value::Object(serde_json::Map::new()));

    let Some(decode_details_obj) = decode_details.as_object_mut() else {
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
/// A line carrying usage is therefore **held until the next data event or `[DONE]`
/// arrives**, which confirms it really is the last one. Intermediate chunks are parsed
/// normally and do not trigger that rule.
///
/// Buffering is bounded: at most one SSE line is held at any moment (splitting on `\n`
/// never accumulates across lines); a stream without usage holds exactly one line.
#[derive(Default)]
pub struct SseUsageMerger {
    buf: Vec<u8>,
    held: Option<Vec<u8>>,
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
        if let Some(held) = self.held.take() {
            out.extend_from_slice(&held);
        }
        out
    }

    fn handle_line(&mut self, line: &[u8], prefill_json: &Value, out: &mut Vec<u8>) {
        let Ok(text) = std::str::from_utf8(line) else {
            // Invalid UTF-8: cannot be decided, pass through verbatim
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
        if self.held.is_some() && text.trim().is_empty() {
            if let Some(held) = self.held.as_mut() {
                held.extend_from_slice(line);
            }
            return;
        }

        if sse_payload(text).is_some_and(is_usage_event) {
            // The next line is still a data event → the previously held line was not the
            // last one, so release it first
            if let Some(held) = self.held.take() {
                out.extend_from_slice(&held);
            }
            match merge_usage_into_sse_line(text, prefill_json) {
                Some(merged) => {
                    self.merged = true;
                    debug!("[USAGE MERGE] merged prefill usage into streaming chunk");
                    self.held = Some(merged.into_bytes());
                }
                None => {
                    // Parse failure or nothing to change: keep holding it and let the next
                    // event decide whether it is the last one
                    self.held = Some(line.to_vec());
                }
            }
            return;
        }

        if text.trim() == "data: [DONE]" {
            // The last data event is confirmed: release the held line, then emit [DONE]
            if let Some(held) = self.held.take() {
                out.extend_from_slice(&held);
            }
        }
        out.extend_from_slice(line);
    }

    /// Whether a merge actually happened (for caller logging/metrics).
    pub fn did_merge(&self) -> bool {
        self.merged
    }
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
}
