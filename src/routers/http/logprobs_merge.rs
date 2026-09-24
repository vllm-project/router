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

/// SSE 数据行的前缀（不含冒号后的空格）。
const SSE_DATA_PREFIX: &str = "data:";

/// 流结束标记，由 vLLM 在最后一个 data 事件后发出。
const SSE_DONE: &str = "[DONE]";

/// 取出 `data:` 行的负载（`[DONE]` 这类非 JSON 负载原样返回）。
fn sse_payload(line: &str) -> Option<&str> {
    let rest = line.strip_prefix(SSE_DATA_PREFIX)?;
    Some(rest.strip_prefix(' ').unwrap_or(rest))
}

/// 该行是否为**携带非 null usage 对象**的 JSON 事件。
///
/// ⚠️ 这里不能简单判 `contains("\"usage\"")`：vLLM 流式响应的**第一个 chunk 就带
/// `"usage":null`**，若把它当成 usage 事件扣住，会把首块压到流末尾才发出去，
/// 等于破坏流式。所以必须要求冒号后面是 `{`（空对象 `{}` 也满足条件，触发合并后
/// 会按需填充，属预期）。
///
/// 匹配在每行上都会跑，但只做一次预编译规则匹配，不做完整 JSON 解析。
fn is_usage_event(payload: &str) -> bool {
    if payload == SSE_DONE {
        return false;
    }
    static RE: once_cell::sync::Lazy<regex::Regex> = once_cell::sync::Lazy::new(|| {
        // ⚠️ `\{` 后面**不能**加 `?`——那会让 `{` 变成可选，于是 `"usage": null`
        // 也会命中，本函数就退化成 `contains("\"usage\"")`，注释里警告的坑会原地复活。
        regex::Regex::new(r#""usage"\s*:\s*\{"#).expect("valid usage pattern")
    });
    RE.is_match(payload)
}

/// 把 prefill 的 usage 合并进一个 SSE data 行。
///
/// ⚠️ **必须原样保留行尾的所有换行符**（`\n`、`\n\n` 或 `\r\n`）。
///
/// SSE 里一个事件以空行结束，所以 `data: {...}` 行的字节形如 `...}\n\n`——**两个** `\n`。
/// 早期版本只保留了其中一个（`else if ends_with('\n') => "\n"`），于是：
///   - 剩下的那个 `\n` 会被 `feed()` 当成独立空行切出来并立即透传；
///   - 扣留的行落到流末尾时，判定「是否为最后一个事件」要看的 `\n\n` 边界就没了。
///
/// 症状是 `feed()` 在应当返回空的时候返回了 `b"\n"`。
///
/// 这里的做法：剥掉全部行尾换行符得到正文，再把**剥掉的那串原样接回**，
/// 从而对 `\n` / `\n\n` / `\r\n` 都保持字节级不变。
///
/// 解析失败时返回 `None`，调用方应原样转发该行——**绝不因为合并失败而吞掉内容**。
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

/// 增量式 SSE 流转换器：按行缓冲，把 prefill 的 usage 合并进携带 usage 的事件。
///
/// # 为什么必须延迟一个事件
///
/// vLLM 的 `stream_options.include_usage` 语义是「usage 在**最后一个** chunk 里」。
/// 而 `merge_usage_in_json` 里有一条规则是「decode 没有 usage 时整体复制 prefill 的
/// usage」——如果**逐行立即处理**，任何一个不带 usage 的中间 chunk（usage 为 null）
/// 都会被这条规则填上 usage，等于凭空给每个 chunk 塞了 usage。
///
/// 所以携带 usage 的行要**留到下一个 data 事件或 `[DONE]` 到来时**才输出，
/// 以此确认「它就是最后一个」。中间 chunk 正常解析、不触发那条规则。
///
/// 缓冲量有界：任何时刻最多只留一个 SSE 行（按 `\n` 切分，不会跨行累积）；
/// 没有 usage 的流则只留一个被扣住的行。
#[derive(Default)]
pub struct SseUsageMerger {
    buf: Vec<u8>,
    held: Option<Vec<u8>>,
    merged: bool,
}

impl SseUsageMerger {
    /// 处理一块字节，返回**可以立即下发**的字节。
    pub fn feed(&mut self, chunk: &[u8], prefill_json: &Value) -> Vec<u8> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        // 按 `\n` 切；最后一段没有换行符，留在 buf 里等下一块
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            self.handle_line(&line, prefill_json, &mut out);
        }
        out
    }

    /// 流结束：把还扣着的行放出去，并重建尾部行缓冲。
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
            // 非法 UTF-8：无法判定，原样透传
            out.extend_from_slice(line);
            return;
        };

        // 正在扣留事件时，把紧随其后的空行也收进 held。
        //
        // 空行是 SSE 事件的结束符，属于同一个事件单位，**不能单独下发**：
        // `feed()` 是按 `\n` 逐行切的，所以 `data: {...}\n\n` 必然被切成
        // 「数据行」+「单个 \n 的空行」两段。若让那个空行单独透传，字节顺序会变成
        // 「空行 → held → [DONE]」，而 held 自己还带着第一个 `\n` —— 位置就错了。
        if self.held.is_some() && text.trim().is_empty() {
            if let Some(held) = self.held.as_mut() {
                held.extend_from_slice(line);
            }
            return;
        }

        if sse_payload(text).is_some_and(is_usage_event) {
            // 下一行仍是 data 事件 → 说明先前扣住的那行不是最后一个，先放出去
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
                    // 解析失败或无需改动：仍然扣住它，交给下一个事件判定是否为最后一个
                    self.held = Some(line.to_vec());
                }
            }
            return;
        }

        if text.trim() == "data: [DONE]" {
            // 最后一个 data 事件已确认，释放扣住的行，再输出 [DONE]
            if let Some(held) = self.held.take() {
                out.extend_from_slice(&held);
            }
        }
        out.extend_from_slice(line);
    }

    /// 是否真正合并过（供调用方记日志/指标）。
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

    // ---------------------------------------------------------------- SSE 流式合并
    //
    // 这些用例覆盖的是「流式请求下 cached_tokens 不被 prefill 值纠正」这个缺陷的修复。
    // 关键不变量：**中间 chunk 必须立即下发**，不能被扣到流末尾——否则流式就退化成
    // 「攒完一整段再发」。vLLM 的首个 chunk 带 `"usage":null`，是最容易踩的坑。

    /// 组装 prefill 侧响应（含 cached_tokens）。
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

    /// 把整条流的字节按 `\n` 切成事件文本，便于断言顺序。
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

        assert!(merger.did_merge(), "应当发生合并");
        assert_eq!(lines.len(), 4);
        assert!(lines[0].contains("\"Hel\""));
        assert!(lines[1].contains("\"lo\""));
        assert!(lines[2].contains("\"cached_tokens\":50"), "末块应被改写: {}", lines[2]);
        assert!(!lines[2].contains("\"cached_tokens\":-1"), "占位值应被替换");
        assert_eq!(lines[3], "data: [DONE]");
    }

    /// **最重要的一条**：首块的 `"usage":null` 不能触发扣留，否则流式失效。
    ///
    /// ⚠️ 这里必须检查**是否立即下发**，不能只检查"内容最终在不在输出里"。
    /// 因为即使首块被错误扣住，`[DONE]` 到来时也会把它放出去，内容照样出现——
    /// 只查内容的断言会**假通过**，测不出"首块被压到流末尾"这个退化。
    #[test]
    fn test_sse_merger_does_not_hold_null_usage_chunk() {
        let prefill = prefill_with_cached(50);
        let mut merger = SseUsageMerger::default();

        let first = "data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}],\"usage\":null}\n\n";
        let out = merger.feed(first.as_bytes(), &prefill);

        assert!(
            String::from_utf8_lossy(&out).contains("\"A\""),
            "带 \"usage\":null 的首块必须立即下发，不能被扣住"
        );
        // 整个首块（含尾部空行）都应被消费掉，不为后续输出留残留
        assert_eq!(
            out, first.as_bytes(),
            "首块应逐字节原样立即下发，实际: {:?}",
            String::from_utf8_lossy(&out)
        );

        // 后续再喂一个真实 usage 行：若首块曾被误扣，这里会冒出来
        let tail = "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":\
                    {\"cached_tokens\":-1}}}\n\ndata: [DONE]\n\n";
        let rest = feed_all(&mut merger, tail.as_bytes(), &prefill);
        let text = String::from_utf8_lossy(&rest);

        assert!(!text.contains("\"A\""), "首块不应被推迟到这里: {text:?}");
        assert_eq!(
            text.matches("data: ").count(),
            2,
            "只应有两个 data 事件（usage 行 + DONE）: {text:?}"
        );
        assert!(merger.did_merge());
    }

    /// 字节流可能在任意位置断开，必须跨 chunk 正确拼行。
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

    /// 逐字节喂入，考验状态机的健壮性。
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

    /// 没有 usage 的流：内容一字不改，也不产生任何 usage。
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
        assert_eq!(out, stream.as_bytes(), "无 usage 时必须原样透传");
    }

    /// prefill 没给 usage：不该据 prefill 改动任何东西。
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
            "prefill 无 usage 时不得改动 decode 的值"
        );
    }

    /// decode 给出合法值时保留（与非流式路径的「合法值优先」一致）。
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

        assert!(!merger.did_merge(), "合法值不应被覆盖");
        assert!(String::from_utf8_lossy(&out).contains("\"cached_tokens\":12"));
    }

    /// 流中断、没等到 `[DONE]`：扣住的行不能丢。
    #[test]
    fn test_sse_merger_flushes_held_line_when_stream_truncated() {
        let prefill = prefill_with_cached(50);
        let mut merger = SseUsageMerger::default();

        let stream = "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":\
                      {\"cached_tokens\":-1}}}\n\n";
        let out = merger.feed(stream.as_bytes(), &prefill);
        assert!(out.is_empty(), "usage 行应被扣住等待确认是否为最后一个");

        let tail = merger.finish(&prefill);
        let lines = sse_lines(&tail);
        assert_eq!(lines.len(), 1, "流结束时扣住的行必须补发，否则丢数据");
        assert!(lines[0].contains("\"cached_tokens\":50"));
    }

    /// 尾部残留没有换行的完整 data 行：也要处理，不能当垃圾丢掉。
    #[test]
    fn test_sse_merger_handles_unterminated_trailing_line() {
        let prefill = prefill_with_cached(50);
        let mut merger = SseUsageMerger::default();

        // 注意结尾没有 `\n`
        let stream = "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":\
                      {\"cached_tokens\":-1}}}";
        let out = feed_all(&mut merger, stream.as_bytes(), &prefill);
        let lines = sse_lines(&out);

        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("\"cached_tokens\":50"), "{}", lines[0]);
    }

    /// 回归：被改写的行必须保留行尾换行，否则会与 `data: [DONE]` 粘成一行。
    ///
    /// 这里断言**原始字节**而不是切分后的行——因为粘行的症状正是「只剩一行」，
    /// 按行断言恰好会漏掉它。
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
        // 断言完整字节序列，而不是手写 needle。
        //
        // 教训：这条用例手写 needle 时错过两次（先少一个 `}`；改成"补一个 `}`"后**仍不够**——
        // `cached_tokens` 后面实际有 **三个** `}`，依次关掉 cached_tokens 对象、
        // prompt_tokens_details 对象、usage 对象）。手写 needle 极易错，一旦错就恒为假、
        // 一直红着，反而掩盖实现是否真的正确。整串比对让实现自身定义"正确输出"。
        //
        // 注意 Rust 里相邻字节字面量**不会**自动拼接（那是 C 的行为，这里会变成元组），
        // 必须用 `concat!`。
        let expected = concat!(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":",
            "{\"cached_tokens\":50}}}\n\ndata: [DONE]\n\n"
        );
        assert_eq!(
            out.as_slice(),
            expected.as_bytes(),
            "\n  期望: {:?}\n  实际: {:?}",
            expected,
            String::from_utf8_lossy(&out)
        );
    }

    /// 同上，但覆盖 CRLF 行尾。
    #[test]
    fn test_sse_merger_preserves_crlf_terminator() {
        let prefill = prefill_with_cached(50);
        let stream = "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":\
                      {\"cached_tokens\":-1}}}\r\ndata: [DONE]\r\n";

        let mut merger = SseUsageMerger::default();
        let out = feed_all(&mut merger, stream.as_bytes(), &prefill);

        assert!(merger.did_merge());
        // 同样整串比对；相邻字面量须用 concat!
        let expected = concat!(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":",
            "{\"cached_tokens\":50}}}\r\ndata: [DONE]\r\n"
        );
        assert_eq!(
            out.as_slice(),
            expected.as_bytes(),
            "\n  期望: {:?}\n  实际: {:?}",
            expected,
            String::from_utf8_lossy(&out)
        );
    }

    /// 流中断且扣住的行尚未放出的**最坏情形**：截断点正好在 usage 行之后、
    /// 空行之前。此时 held 只含数据行、finish 必须把它补发。
    ///
    /// 与 `flushes_held_line_when_stream_truncated` 的区别：那条的截断点在空行之后。
    #[test]
    fn test_sse_merger_flushes_on_truncation_before_blank_line() {
        let prefill = prefill_with_cached(50);
        let mut merger = SseUsageMerger::default();

        // 只喂数据行本身，不带结尾的空行
        let stream = b"data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":-1}}}\n";
        let out = merger.feed(stream, &prefill);
        assert!(out.is_empty(), "应被扣住，实际: {:?}", String::from_utf8_lossy(&out));

        let tail = merger.finish(&prefill);
        let text = String::from_utf8_lossy(&tail);
        assert!(text.contains("\"cached_tokens\":50"), "扣住的行必须补发: {text:?}");
    }
}
