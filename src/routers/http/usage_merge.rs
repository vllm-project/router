//! End-to-end usage accounting for prefill/decode disaggregation.
//!
//! Input counts/details belong to Prefill. Decode owns output counts/details;
//! Prefill's forced one-token output must never replace unknown Decode output.

use serde_json::{Map, Value};

pub(super) fn usage_object(value: &Value) -> Option<&Map<String, Value>> {
    value
        .get("usage")
        .and_then(Value::as_object)
        .or_else(|| value.pointer("/response/usage").and_then(Value::as_object))
}

pub(super) fn usage_object_mut(value: &mut Value) -> Option<&mut Map<String, Value>> {
    if value.get("usage").is_some_and(Value::is_object) {
        value.get_mut("usage").and_then(Value::as_object_mut)
    } else {
        value
            .pointer_mut("/response/usage")
            .and_then(Value::as_object_mut)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum UsageDialect {
    Completions,
    Responses,
}

impl UsageDialect {
    pub(super) fn keys(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::Completions => (
                "prompt_tokens",
                "prompt_tokens_details",
                "completion_tokens",
            ),
            Self::Responses => ("input_tokens", "input_tokens_details", "output_tokens"),
        }
    }
}

/// Resolve the payload spelling before mutation. Both usage and cache merging
/// receive the same Prefill context, so neither relies on the other running first.
pub(super) fn usage_dialect(response: &Value, prefill: Option<&Value>) -> UsageDialect {
    fn detect(usage: &Map<String, Value>) -> Option<UsageDialect> {
        if usage.contains_key("input_tokens") || usage.contains_key("input_tokens_details") {
            Some(UsageDialect::Responses)
        } else if usage.contains_key("prompt_tokens") {
            Some(UsageDialect::Completions)
        } else {
            None
        }
    }

    if !response.get("usage").is_some_and(Value::is_object)
        && response
            .pointer("/response/usage")
            .is_some_and(Value::is_object)
    {
        return UsageDialect::Responses;
    }
    usage_object(response)
        .and_then(detect)
        .or_else(|| prefill.and_then(usage_object).and_then(detect))
        .unwrap_or(UsageDialect::Completions)
}

/// Retain only usage, releasing potentially large KV-transfer metadata before
/// consuming the Decode stream.
pub(super) fn usage_snapshot(prefill: &Value) -> Option<Value> {
    usage_object(prefill).map(|usage| serde_json::json!({"usage": usage}))
}

fn replace_if_changed(usage: &mut Map<String, Value>, key: &str, value: Value) -> bool {
    if usage.get(key) == Some(&value) {
        false
    } else {
        usage.insert(key.into(), value);
        true
    }
}

/// Merge valid Prefill input counts and input details into existing Decode
/// usage. Preserve unknown Decode fields and never synthesize missing usage.
/// Cache-token ownership/validation is handled by `pd_cached_tokens`.
pub(super) fn merge_pd_usage(prefill: &Value, decode: &mut Value) -> bool {
    let (input_key, details_key, output_key) = usage_dialect(decode, Some(prefill)).keys();
    let Some(source) = usage_object(prefill) else {
        return false;
    };
    let Some(target) = usage_object_mut(decode) else {
        return false;
    };
    let source_input = source.get(input_key).and_then(Value::as_u64);
    let mut changed = false;
    if let Some(input) = source_input {
        changed |= replace_if_changed(target, input_key, Value::from(input));
        if let Some(total) = target
            .get(output_key)
            .and_then(Value::as_u64)
            .and_then(|output| input.checked_add(output))
        {
            changed |= replace_if_changed(target, "total_tokens", Value::from(total));
        } else {
            // The old total describes Decode's input, not the Prefill count we
            // just adopted. Unknown output or overflow makes it unrecomputable.
            changed |= target.remove("total_tokens").is_some();
        }
    }

    if let Some(details) = source.get(details_key).and_then(Value::as_object) {
        // Overlay Prefill's input details, retaining Decode extension fields.
        // A malformed details container is left untouched rather than erased.
        if target.get(details_key).is_none_or(Value::is_null)
            && details.keys().any(|key| key != "cached_tokens")
        {
            target.insert(details_key.into(), Value::Object(Map::new()));
            changed = true;
        }
        if let Some(target_details) = target.get_mut(details_key).and_then(Value::as_object_mut) {
            for (key, value) in details
                .iter()
                .filter(|(key, _)| key.as_str() != "cached_tokens")
            {
                changed |= replace_if_changed(target_details, key, value.clone());
            }
        }
    }
    changed
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn input_from_prefill_output_from_decode_and_totals_recomputed() {
        for responses in [false, true] {
            let (input, output, details, output_details) = if responses {
                (
                    "input_tokens",
                    "output_tokens",
                    "input_tokens_details",
                    "output_tokens_details",
                )
            } else {
                (
                    "prompt_tokens",
                    "completion_tokens",
                    "prompt_tokens_details",
                    "completion_tokens_details",
                )
            };
            let prefill = json!({"usage": {input: 186, output: 1, "total_tokens": 187, details: {"audio_tokens": 3}}});
            let mut decode = json!({"usage": {input: 999, output: 8, "total_tokens": 1007, details: {"audio_tokens": 2, "future": 7}, output_details: {"reasoning_tokens": 4}, "future": 9}, "metrics": {"x": 1}});
            assert!(merge_pd_usage(&prefill, &mut decode));
            assert_eq!(decode["usage"][input], 186);
            assert_eq!(decode["usage"][output], 8);
            assert_eq!(decode["usage"]["total_tokens"], 194);
            assert_eq!(
                decode["usage"][details],
                json!({"audio_tokens": 3, "future": 7})
            );
            assert_eq!(decode["usage"][output_details]["reasoning_tokens"], 4);
            assert_eq!(decode["usage"]["future"], 9);
            assert_eq!(decode["metrics"]["x"], 1);
            assert!(!merge_pd_usage(&prefill, &mut decode));
        }
    }

    #[test]
    fn missing_usage_or_output_never_borrows_prefill_output() {
        let prefill =
            json!({"usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}});
        for mut decode in [
            json!({}),
            json!({"usage": null}),
            json!({"usage": "opaque"}),
        ] {
            let original = decode.clone();
            assert!(!merge_pd_usage(&prefill, &mut decode));
            assert_eq!(decode, original);
        }
        let mut decode = json!({"usage": {"prompt_tokens": 99}});
        assert!(merge_pd_usage(&prefill, &mut decode));
        assert_eq!(decode["usage"], json!({"prompt_tokens": 10}));
    }

    #[test]
    fn invalid_input_and_overflow_do_not_create_totals() {
        for input in [json!(-1), json!(null), json!("10"), json!(1.5)] {
            let prefill = json!({"usage": {"prompt_tokens": input}});
            let mut decode =
                json!({"usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}});
            let original = decode.clone();
            assert!(!merge_pd_usage(&prefill, &mut decode));
            assert_eq!(decode, original);
        }
        let prefill = json!({"usage": {"prompt_tokens": u64::MAX}});
        let mut decode = json!({"usage": {"completion_tokens": 1}});
        assert!(merge_pd_usage(&prefill, &mut decode));
        assert!(decode["usage"].get("total_tokens").is_none());
    }

    #[test]
    fn stale_totals_are_removed_when_decode_output_is_unavailable_or_overflows() {
        for (input_key, output_key, nested) in [
            ("prompt_tokens", "completion_tokens", false),
            ("input_tokens", "output_tokens", false),
            ("input_tokens", "output_tokens", true),
        ] {
            for original_input in [999, 70] {
                for output in [
                    None,
                    Some(json!(null)),
                    Some(json!(-1)),
                    Some(json!("8")),
                    Some(json!(1.5)),
                ] {
                    let prefill = json!({"usage": {input_key: 70, output_key: 1}});
                    let mut decode = json!({"usage": {input_key: original_input, "total_tokens": 1007, "future": 9}});
                    if let Some(output) = &output {
                        decode["usage"][output_key] = output.clone();
                    }
                    if nested {
                        decode = json!({"type": "response.completed", "response": decode});
                    }
                    assert!(merge_pd_usage(&prefill, &mut decode));
                    let usage = usage_object(&decode).unwrap();
                    assert_eq!(usage[input_key], 70);
                    assert!(
                        usage.get("total_tokens").is_none(),
                        "stale total retained: {decode}"
                    );
                    assert_eq!(usage.get(output_key), output.as_ref());
                    assert_eq!(usage["future"], 9);
                    assert!(!merge_pd_usage(&prefill, &mut decode));
                }
            }
            let prefill = json!({"usage": {input_key: u64::MAX}});
            let mut decode =
                json!({"usage": {input_key: 999, output_key: 1, "total_tokens": 1000}});
            if nested {
                decode = json!({"response": decode});
            }
            assert!(merge_pd_usage(&prefill, &mut decode));
            let usage = usage_object(&decode).unwrap();
            assert_eq!(usage[input_key], u64::MAX);
            assert_eq!(usage[output_key], 1);
            assert!(usage.get("total_tokens").is_none());
            assert!(!merge_pd_usage(&prefill, &mut decode));
        }
    }

    #[test]
    fn nested_responses_usage_and_snapshot_without_kv_metadata() {
        let prefill = json!({"usage": {"input_tokens": 50, "output_tokens": 1}, "kv_transfer_params": {"remote_block_ids": [1,2,3]}});
        let snapshot = usage_snapshot(&prefill).unwrap();
        assert!(snapshot.get("kv_transfer_params").is_none());
        let mut decode = json!({"type": "response.completed", "response": {"usage": {"input_tokens": 999, "output_tokens": 7, "total_tokens": 1006}}});
        assert!(merge_pd_usage(&snapshot, &mut decode));
        assert_eq!(decode["response"]["usage"]["input_tokens"], 50);
        assert_eq!(decode["response"]["usage"]["total_tokens"], 57);
        assert!(decode.get("usage").is_none());
    }
}
