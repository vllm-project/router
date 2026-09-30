//! The vLLM v0.29 msgspec wire contract. A malformed or unsupported event
//! invalidates the entire batch; silently skipping a removal can retain stale
//! ownership indefinitely.

use std::{collections::HashMap, io::Cursor};

use rmpv::Value;

use crate::kv_index::{BlockHash, OwnershipEvent};

pub(super) const MAX_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

pub(super) fn decode_batch(
    payload: &[u8],
    block_size: usize,
) -> Result<Vec<OwnershipEvent>, String> {
    if payload.len() > MAX_PAYLOAD_BYTES || block_size == 0 {
        return Err("invalid KV event payload size or block size".into());
    }
    let mut cursor = Cursor::new(payload);
    let value = rmpv::decode::read_value_with_max_depth(&mut cursor, 16)
        .map_err(|error| format!("invalid MessagePack: {error}"))?;
    if cursor.position() as usize != payload.len() {
        return Err("trailing bytes after KV event batch".into());
    }
    let batch = value.as_array().ok_or("KV event batch must be an array")?;
    if !(2..=3).contains(&batch.len()) {
        return Err("KV event batch must contain timestamp, events, and optional DP rank".into());
    }
    if !batch[0]
        .as_f64()
        .is_some_and(|timestamp| timestamp.is_finite() && timestamp >= 0.0)
    {
        return Err("invalid KV event timestamp".into());
    }
    if batch
        .get(2)
        .is_some_and(|rank| !rank.is_nil() && rank.as_u64() != Some(0))
    {
        return Err("only independent DP=1 KV event publishers are supported".into());
    }
    let events = batch[1].as_array().ok_or("KV events must be an array")?;
    if events.len() > 4096 {
        return Err("KV event batch exceeds the event limit".into());
    }
    events
        .iter()
        .map(|event| decode_event(event, block_size))
        .collect()
}

fn decode_event(event: &Value, block_size: usize) -> Result<OwnershipEvent, String> {
    let pairs = event.as_map().ok_or("KV event must be a tagged map")?;
    let mut fields = HashMap::with_capacity(pairs.len());
    for (key, value) in pairs {
        let key = key.as_str().ok_or("KV event keys must be strings")?;
        if fields.insert(key, value).is_some() {
            return Err(format!("duplicate KV event field: {key}"));
        }
    }
    let kind = required(&fields, "type")?
        .as_str()
        .ok_or("invalid KV event type")?;
    match kind {
        "AllBlocksCleared" => {
            reject_unknown(&fields, &["type"])?;
            Ok(OwnershipEvent::Clear)
        }
        "BlockRemoved" => {
            reject_unknown(&fields, &["type", "block_hashes", "medium", "group_idx"])?;
            validate_medium_and_group(&fields)?;
            Ok(OwnershipEvent::Remove(hashes(required(
                &fields,
                "block_hashes",
            )?)?))
        }
        "BlockStored" => {
            reject_unknown(
                &fields,
                &[
                    "type",
                    "block_hashes",
                    "parent_block_hash",
                    "token_ids",
                    "block_size",
                    "lora_id",
                    "medium",
                    "lora_name",
                    "extra_keys",
                    "group_idx",
                    "kv_cache_spec_kind",
                    "kv_cache_spec_sliding_window",
                ],
            )?;
            validate_medium_and_group(&fields)?;
            for key in ["lora_id", "lora_name"] {
                if !required(&fields, key)?.is_nil() {
                    return Err(format!("unsupported KV event {key}"));
                }
            }
            let parent = required(&fields, "parent_block_hash")?;
            if !parent.is_nil() {
                hash(parent)?;
            }
            if required(&fields, "block_size")?.as_u64() != Some(block_size as u64) {
                return Err("KV event block size does not match router configuration".into());
            }
            let blocks = hashes(required(&fields, "block_hashes")?)?;
            let tokens = required(&fields, "token_ids")?
                .as_array()
                .ok_or("invalid token IDs")?;
            if blocks.len().checked_mul(block_size) != Some(tokens.len())
                || tokens
                    .iter()
                    .any(|token| token.as_u64().is_none_or(|id| id > u32::MAX.into()))
            {
                return Err("KV stored event must contain complete blocks of u32 token IDs".into());
            }
            if let Some(extra_keys) = fields.get("extra_keys").filter(|value| !value.is_nil()) {
                let keys = extra_keys.as_array().ok_or("invalid KV extra keys")?;
                if keys.len() != blocks.len() || keys.iter().any(|key| !key.is_nil()) {
                    return Err("KV extra keys are outside the normal-dense contract".into());
                }
            }
            if fields
                .get("kv_cache_spec_kind")
                .is_some_and(|kind| !kind.is_nil() && kind.as_str() != Some("full_attention"))
                || fields
                    .get("kv_cache_spec_sliding_window")
                    .is_some_and(|window| !window.is_nil())
            {
                return Err("only normal dense full-attention KV events are supported".into());
            }
            Ok(OwnershipEvent::Store(blocks))
        }
        _ => Err(format!("unsupported KV event type: {kind}")),
    }
}

fn required<'a>(fields: &HashMap<&str, &'a Value>, key: &str) -> Result<&'a Value, String> {
    fields
        .get(key)
        .copied()
        .ok_or_else(|| format!("missing KV event field: {key}"))
}

fn reject_unknown(fields: &HashMap<&str, &Value>, allowed: &[&str]) -> Result<(), String> {
    if let Some(key) = fields.keys().find(|key| !allowed.contains(key)) {
        return Err(format!("unsupported KV event field: {key}"));
    }
    Ok(())
}

fn validate_medium_and_group(fields: &HashMap<&str, &Value>) -> Result<(), String> {
    if required(fields, "medium")?.as_str() != Some("GPU") {
        return Err("only GPU KV events are supported".into());
    }
    if fields
        .get("group_idx")
        .is_some_and(|group| !group.is_nil() && group.as_u64() != Some(0))
    {
        return Err("only normal-dense cache group 0 is supported".into());
    }
    Ok(())
}

fn hash(value: &Value) -> Result<BlockHash, String> {
    match value {
        Value::Binary(bytes) => bytes
            .as_slice()
            .try_into()
            .map_err(|_| "KV hash must have 32 bytes".into()),
        _ => Err("KV hash must be sha256_cbor bytes, not an integer or string".into()),
    }
}

fn hashes(value: &Value) -> Result<Vec<BlockHash>, String> {
    let values = value.as_array().ok_or("KV hashes must be an array")?;
    if values.is_empty() {
        return Err("KV event hashes cannot be empty".into());
    }
    values.iter().map(hash).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(fields: Vec<(&str, Value)>) -> Value {
        Value::Map(
            fields
                .into_iter()
                .map(|(key, value)| (key.into(), value))
                .collect(),
        )
    }

    fn stored() -> Value {
        map(vec![
            ("type", "BlockStored".into()),
            (
                "block_hashes",
                Value::Array(vec![Value::Binary(vec![7; 32])]),
            ),
            ("parent_block_hash", Value::Nil),
            ("token_ids", Value::Array(vec![1.into(), 2.into()])),
            ("block_size", 2.into()),
            ("lora_id", Value::Nil),
            ("medium", "GPU".into()),
            ("lora_name", Value::Nil),
            ("extra_keys", Value::Array(vec![Value::Nil])),
            ("group_idx", 0.into()),
            ("kv_cache_spec_kind", "full_attention".into()),
        ])
    }

    fn encode(events: Vec<Value>) -> Vec<u8> {
        let mut payload = Vec::new();
        rmpv::encode::write_value(
            &mut payload,
            &Value::Array(vec![1.0.into(), Value::Array(events), 0.into()]),
        )
        .unwrap();
        payload
    }

    fn change(event: &mut Value, key: &str, value: Value) {
        let Value::Map(fields) = event else {
            panic!("map expected")
        };
        if let Some((_, old)) = fields
            .iter_mut()
            .find(|(field, _)| field.as_str() == Some(key))
        {
            *old = value;
        } else {
            fields.push((key.into(), value));
        }
    }

    #[test]
    fn decodes_vllm_tagged_normal_dense_batch_in_order() {
        let events = vec![
            stored(),
            map(vec![("type", "AllBlocksCleared".into())]),
            map(vec![
                ("type", "BlockRemoved".into()),
                (
                    "block_hashes",
                    Value::Array(vec![Value::Binary(vec![7; 32])]),
                ),
                ("medium", "GPU".into()),
            ]),
        ];
        assert_eq!(
            decode_batch(&encode(events), 2).unwrap(),
            vec![
                OwnershipEvent::Store(vec![[7; 32]]),
                OwnershipEvent::Clear,
                OwnershipEvent::Remove(vec![[7; 32]]),
            ]
        );
    }

    #[test]
    fn entire_batch_rejects_unsupported_or_incomplete_event() {
        let cases = vec![
            ("block_hashes", Value::Array(vec![42.into()])),
            ("medium", "CPU".into()),
            ("group_idx", 1.into()),
            ("lora_id", 3.into()),
            ("lora_name", "adapter".into()),
            ("kv_cache_spec_kind", "mamba".into()),
            ("kv_cache_spec_sliding_window", 128.into()),
            (
                "extra_keys",
                Value::Array(vec![Value::Array(vec!["salt".into()])]),
            ),
            ("token_ids", Value::Array(vec![1.into()])),
            ("parent_block_hash", Value::Binary(vec![1; 31])),
            ("block_size", 16.into()),
            ("unknown_future_field", true.into()),
        ];
        for (key, value) in cases {
            let mut bad = stored();
            change(&mut bad, key, value);
            assert!(
                decode_batch(&encode(vec![stored(), bad]), 2).is_err(),
                "{key}"
            );
        }
        assert!(decode_batch(
            &encode(vec![stored(), map(vec![("type", "BlockRemoved".into())])]),
            2
        )
        .is_err());
        assert!(decode_batch(
            &encode(vec![stored(), map(vec![("type", "FutureEvent".into())])]),
            2
        )
        .is_err());
    }

    #[test]
    fn rejects_duplicate_fields_and_trailing_payload() {
        let mut bad = stored();
        let Value::Map(fields) = &mut bad else {
            unreachable!()
        };
        fields.push(("medium".into(), "CPU".into()));
        assert!(decode_batch(&encode(vec![bad]), 2).is_err());
        let mut bytes = encode(vec![stored()]);
        bytes.push(0);
        assert!(decode_batch(&bytes, 2).is_err());
    }

    #[test]
    fn malformed_removal_or_clear_invalidates_all_preceding_stores() {
        let removed = map(vec![
            ("type", "BlockRemoved".into()),
            (
                "block_hashes",
                Value::Array(vec![Value::Binary(vec![7; 32])]),
            ),
            ("medium", "GPU".into()),
        ]);
        for (key, value) in [
            (
                "block_hashes",
                Value::Array(vec![Value::Binary(vec![7; 31])]),
            ),
            ("medium", Value::Nil),
            ("group_idx", 1.into()),
            ("locality", "REMOTE".into()),
        ] {
            let mut event = removed.clone();
            change(&mut event, key, value);
            assert!(
                decode_batch(&encode(vec![stored(), event]), 2).is_err(),
                "{key}"
            );
        }
        let malformed_clear = map(vec![
            ("type", "AllBlocksCleared".into()),
            ("group_idx", 1.into()),
        ]);
        assert!(decode_batch(&encode(vec![stored(), malformed_clear]), 2).is_err());
    }

    #[test]
    fn independent_dp_rank_and_timestamp_are_required() {
        for rank in [1.into(), (-1).into(), "0".into()] {
            let mut payload = Vec::new();
            rmpv::encode::write_value(
                &mut payload,
                &Value::Array(vec![1.0.into(), Value::Array(vec![stored()]), rank]),
            )
            .unwrap();
            assert!(decode_batch(&payload, 2).is_err());
        }
        for timestamp in [f64::NAN, f64::INFINITY, -1.0] {
            let mut payload = Vec::new();
            rmpv::encode::write_value(
                &mut payload,
                &Value::Array(vec![timestamp.into(), Value::Array(vec![stored()])]),
            )
            .unwrap();
            assert!(decode_batch(&payload, 2).is_err());
        }
    }
}
