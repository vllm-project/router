//! Logprobs merging utilities for PD disaggregation
//!
//! This module provides utilities for merging logprobs from prefill and decode responses
//! in prefill-decode disaggregation mode.

use serde_json::Value;
use tracing::debug;

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
    let mut merged = false;

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
}
