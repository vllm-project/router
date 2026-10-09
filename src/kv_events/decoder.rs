//! Bounded structural decoding into the shared PR326 wire contract.
use std::{collections::HashSet, io::Cursor};

use crate::kv_index::wire::{ExternalBlockHash, KVEvent, KVEventBatch};
use rmpv::Value;
use serde::Deserialize;

const MAX_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

/// Decode atomically without conflating structural validity with device admission.
pub fn decode_batch(payload: &[u8]) -> Result<KVEventBatch, String> {
    if payload.len() > MAX_PAYLOAD_BYTES {
        return Err("KV event payload exceeds 16 MiB".into());
    }
    // rmpv accepts Reserved as nil; rmp-serde can turn invalid strings into
    // bytes. Use it only for syntax, then retain rmpv's wire type fidelity.
    let mut syntax = rmp_serde::Deserializer::new(Cursor::new(payload));
    syntax.set_max_depth(16);
    serde::de::IgnoredAny::deserialize(&mut syntax)
        .map_err(|error| format!("invalid MessagePack: {error}"))?;
    let mut cursor = Cursor::new(payload);
    let value = rmpv::decode::read_value_with_max_depth(&mut cursor, 16)
        .map_err(|error| format!("invalid MessagePack: {error}"))?;
    if cursor.position() as usize != payload.len() {
        return Err("trailing bytes after KV event batch".into());
    }
    let batch = value.as_array().ok_or("KV event batch must be an array")?;
    if !(2..=3).contains(&batch.len()) {
        return Err("KV batch requires timestamp, events, and optional rank".into());
    }
    let ts = batch[0]
        .as_f64()
        .filter(|ts| ts.is_finite() && *ts >= 0.0)
        .ok_or("invalid KV event timestamp")?;
    let data_parallel_rank = match batch.get(2) {
        None | Some(Value::Nil) => None,
        Some(rank) => Some(
            u32::try_from(rank.as_u64().ok_or("invalid DP rank")?)
                .map_err(|_| "invalid DP rank")?,
        ),
    };
    let events = batch[1].as_array().ok_or("KV events must be an array")?;
    if events.len() > 4096 {
        return Err("KV event batch exceeds the event limit".into());
    }
    let events = events.iter().map(decode_event).collect::<Result<_, _>>()?;
    Ok(KVEventBatch {
        ts,
        events,
        data_parallel_rank,
    })
}

fn decode_event(event: &Value) -> Result<KVEvent, String> {
    let pairs = event.as_map().ok_or("KV event must be a tagged map")?;
    let mut keys = HashSet::new();
    for (key, _) in pairs {
        let key = key.as_str().ok_or("KV event keys must be strings")?;
        if !keys.insert(key) {
            return Err(format!("duplicate KV event field: {key}"));
        }
    }
    let kind = pairs
        .iter()
        .find(|(key, _)| key.as_str() == Some("type"))
        .and_then(|(_, value)| value.as_str())
        .ok_or("missing or invalid KV event type")?;
    let (allowed, required): (&[&str], &[&str]) = match kind {
        "BlockStored" => (
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
                "locality",
                "ownership",
                "session_id",
            ],
            &[
                "block_hashes",
                "parent_block_hash",
                "token_ids",
                "block_size",
                "lora_id",
                "medium",
                "lora_name",
            ],
        ),
        "BlockRemoved" => (
            &[
                "type",
                "block_hashes",
                "medium",
                "group_idx",
                "locality",
                "ownership",
            ],
            &["block_hashes", "medium"],
        ),
        "AllBlocksCleared" => (&["type"], &[]),
        _ => return Err(format!("unsupported KV event type: {kind}")),
    };
    if let Some(key) = keys.iter().find(|key| !allowed.contains(key)) {
        return Err(format!("unsupported KV event field: {key}"));
    }
    if let Some(key) = required.iter().find(|key| !keys.contains(**key)) {
        return Err(format!("missing KV event field: {key}"));
    }
    for (key, value) in pairs {
        match key.as_str() {
            Some("block_hashes") => {
                let hashes = value
                    .as_array()
                    .filter(|hashes| !hashes.is_empty())
                    .ok_or("KV event hashes must be a nonempty array")?;
                for hash in hashes {
                    validate_wire_hash(hash)?;
                }
            }
            Some("parent_block_hash") if !value.is_nil() => validate_wire_hash(value)?,
            Some(
                "medium" | "lora_name" | "kv_cache_spec_kind" | "locality" | "ownership"
                | "session_id",
            ) if !matches!(value, Value::String(_) | Value::Nil) => {
                return Err(format!("KV event field {key} must be a string or nil"));
            }
            _ => {}
        }
    }
    // The Value deserializer turns invalid UTF-8 strings into bytes, including
    // opaque extra-key values and map keys. Reject them before DTO conversion.
    validate_utf8_strings(event)?;
    rmpv::ext::from_value(event.clone()).map_err(|error| format!("invalid KV event: {error}"))
}

fn validate_utf8_strings(value: &Value) -> Result<(), String> {
    match value {
        Value::String(text) if text.as_str().is_none() => {
            Err("invalid UTF-8 KV event string".into())
        }
        Value::Array(values) => {
            for value in values {
                validate_utf8_strings(value)?;
            }
            Ok(())
        }
        Value::Map(pairs) => {
            for (key, value) in pairs {
                validate_utf8_strings(key)?;
                validate_utf8_strings(value)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn validate_wire_hash(hash: &Value) -> Result<(), String> {
    if matches!(hash, Value::Binary(_)) || hash.as_u64().is_some() {
        Ok(())
    } else {
        Err("KV wire hash must be bytes or an unsigned integer".into())
    }
}

/// Admit only the narrow normal-dense, independently addressed GPU contract.
/// This checks event shape, not the publisher's hash algorithm, cache namespace,
/// remote incarnation or freshness. The source adapter must establish those
/// facts independently before publishing observations to the shared index.
pub fn validate_device_dp1_batch(
    batch: &KVEventBatch,
    expected_block_size: usize,
) -> Result<(), String> {
    if expected_block_size == 0 || batch.data_parallel_rank.is_some_and(|rank| rank != 0) {
        return Err(
            "only independent DP=1 publishers with nonzero block size are supported".into(),
        );
    }
    for event in &batch.events {
        let (hashes, medium, group, locality, ownership) = match event {
            KVEvent::AllBlocksCleared(_) => continue,
            KVEvent::BlockRemoved(event) => (
                &event.block_hashes,
                &event.medium,
                event.group_idx,
                &event.locality,
                &event.ownership,
            ),
            KVEvent::BlockStored(event) => {
                if event.block_size as usize != expected_block_size
                    || event.block_hashes.len().checked_mul(expected_block_size)
                        != Some(event.token_ids.len())
                    || event.lora_id.is_some()
                    || event.lora_name.is_some()
                    || event.extra_keys.as_ref().is_some_and(|keys| {
                        keys.len() != event.block_hashes.len()
                            || keys.iter().any(|key| !key.is_nil())
                    })
                    || event
                        .kv_cache_spec_kind
                        .as_deref()
                        .is_some_and(|kind| kind != "full_attention")
                    || event.kv_cache_spec_sliding_window.is_some()
                    || event.session_id.is_some()
                {
                    return Err(
                        "stored event is outside the normal-dense complete-block contract".into(),
                    );
                }
                if let Some(parent) = &event.parent_block_hash {
                    validate_hash(parent)?;
                }
                (
                    &event.block_hashes,
                    &event.medium,
                    event.group_idx,
                    &event.locality,
                    &event.ownership,
                )
            }
        };
        if medium.as_deref() != Some("GPU")
            || group.is_some_and(|group| group != 0)
            || locality
                .as_deref()
                .is_some_and(|locality| locality != "LOCAL")
            || ownership.is_some()
        {
            return Err("event is outside the GPU group-0 local ownership contract".into());
        }
        if hashes.is_empty() {
            return Err("KV event hashes cannot be empty".into());
        }
        for hash in hashes {
            validate_hash(hash)?;
        }
    }
    Ok(())
}

fn validate_hash(hash: &ExternalBlockHash) -> Result<(), String> {
    match hash {
        ExternalBlockHash::Bytes(bytes) if bytes.len() == 32 => Ok(()),
        _ => Err("KV hash must be 32-byte sha256_cbor bytes".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_index::wire::BlockStored;

    fn stored() -> Value {
        let event = KVEvent::BlockStored(BlockStored {
            block_hashes: vec![ExternalBlockHash::Bytes(vec![7; 32])],
            parent_block_hash: Some(ExternalBlockHash::Bytes(vec![8; 32])),
            token_ids: vec![1, 2],
            block_size: 2,
            lora_id: None,
            medium: Some("GPU".into()),
            lora_name: None,
            extra_keys: Some(vec![Value::Nil]),
            group_idx: Some(0),
            kv_cache_spec_kind: Some("full_attention".into()),
            kv_cache_spec_sliding_window: None,
            locality: Some("LOCAL".into()),
            ownership: None,
            session_id: None,
        });
        // vLLM emits tagged maps; generic Value serialization uses positional
        // structs. Use the common DTO's named MessagePack representation.
        let bytes = rmp_serde::to_vec_named(&event).unwrap();
        rmpv::decode::read_value(&mut Cursor::new(bytes)).unwrap()
    }

    fn encode(events: Vec<Value>, rank: Value) -> Vec<u8> {
        let mut bytes = Vec::new();
        rmpv::encode::write_value(
            &mut bytes,
            &Value::Array(vec![1.25.into(), Value::Array(events), rank]),
        )
        .unwrap();
        bytes
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
    fn common_wire_retains_raw_facts_before_admission() {
        let mut event = stored();
        for (key, value) in [
            ("medium", "CPU".into()),
            ("ownership", "TRANSFERRED".into()),
            ("locality", "REMOTE".into()),
            ("session_id", "session".into()),
            ("lora_id", 9.into()),
            ("lora_name", "adapter".into()),
            ("extra_keys", Value::Array(vec!["salt".into()])),
        ] {
            change(&mut event, key, value);
        }
        let batch = decode_batch(&encode(vec![event], 3.into())).unwrap();
        assert_eq!(batch.ts, 1.25);
        assert_eq!(batch.data_parallel_rank, Some(3));
        let KVEvent::BlockStored(event) = &batch.events[0] else {
            panic!("stored expected")
        };
        assert_eq!(
            event.block_hashes,
            vec![ExternalBlockHash::Bytes(vec![7; 32])]
        );
        assert_eq!(
            event.parent_block_hash,
            Some(ExternalBlockHash::Bytes(vec![8; 32]))
        );
        assert_eq!(event.token_ids, vec![1, 2]);
        assert_eq!(event.medium.as_deref(), Some("CPU"));
        assert_eq!(event.ownership.as_deref(), Some("TRANSFERRED"));
        assert_eq!(event.locality.as_deref(), Some("REMOTE"));
        assert_eq!(event.session_id.as_deref(), Some("session"));
        assert_eq!(event.lora_id, Some(9));
        assert_eq!(event.lora_name.as_deref(), Some("adapter"));
        assert_eq!(event.extra_keys, Some(vec!["salt".into()]));
        assert!(validate_device_dp1_batch(&batch, 2).is_err());
    }

    #[test]
    fn device_admission_is_separate_and_conservative() {
        let batch = decode_batch(&encode(vec![stored()], Value::Nil)).unwrap();
        assert!(validate_device_dp1_batch(&batch, 2).is_ok());
        for (key, value) in [
            ("medium", "CPU".into()),
            ("medium", Value::Nil),
            ("ownership", "UNKNOWN".into()),
            ("locality", "UNKNOWN".into()),
            ("group_idx", 1.into()),
            ("token_ids", Value::Array(vec![1.into()])),
            ("parent_block_hash", Value::Binary(vec![8; 31])),
        ] {
            let mut event = stored();
            change(&mut event, key, value);
            let batch = decode_batch(&encode(vec![event], 0.into())).unwrap();
            assert!(validate_device_dp1_batch(&batch, 2).is_err(), "{key}");
        }
    }

    #[test]
    fn malformed_lifecycle_rejects_whole_batch() {
        for event in [
            Value::Map(vec![
                ("type".into(), "AllBlocksCleared".into()),
                ("medium".into(), "GPU".into()),
            ]),
            Value::Map(vec![
                ("type".into(), "BlockRemoved".into()),
                ("medium".into(), "GPU".into()),
            ]),
            Value::Map(vec![
                ("type".into(), "AllBlocksCleared".into()),
                ("type".into(), "AllBlocksCleared".into()),
            ]),
        ] {
            assert!(decode_batch(&encode(vec![stored(), event], 0.into())).is_err());
        }
        let removed = Value::Map(vec![
            ("type".into(), "BlockRemoved".into()),
            ("medium".into(), "CPU".into()),
            (
                "block_hashes".into(),
                Value::Array(vec![Value::Binary(vec![7; 32])]),
            ),
        ]);
        let batch = decode_batch(&encode(vec![removed], 0.into())).unwrap();
        assert!(
            matches!(&batch.events[0], KVEvent::BlockRemoved(event) if event.medium.as_deref() == Some("CPU"))
        );
        assert!(validate_device_dp1_batch(&batch, 2).is_err());
    }

    #[test]
    fn reserved_marker_is_rejected_but_binary_bytes_are_preserved() {
        let mut event = stored();
        change(
            &mut event,
            "block_hashes",
            Value::Array(vec![Value::Binary(vec![0xc1; 32])]),
        );
        change(&mut event, "parent_block_hash", Value::Nil);
        let mut payload = encode(vec![event], 0.into());
        let batch = decode_batch(&payload).unwrap();
        assert!(validate_device_dp1_batch(&batch, 2).is_ok());

        let key = b"parent_block_hash";
        let parent = payload
            .windows(key.len() + 1)
            .position(|field| &field[..key.len()] == key && field[key.len()] == 0xc0)
            .unwrap()
            + key.len();
        payload[parent] = 0xc1;
        assert!(decode_batch(&payload).is_err());
    }

    #[test]
    fn invalid_utf8_string_hash_is_not_reclassified_as_binary() {
        let mut event = stored();
        change(
            &mut event,
            "block_hashes",
            Value::Array(vec![Value::Binary(vec![0xff; 32])]),
        );
        let mut payload = encode(vec![event], 0.into());
        assert!(decode_batch(&payload).is_ok());
        let hash = payload
            .windows(34)
            .position(|field| field[..2] == [0xc4, 32] && field[2..] == [0xff; 32])
            .unwrap();
        payload[hash] = 0xd9; // Str8 instead of Bin8, preserving length and bytes.
        assert!(decode_batch(&payload).is_err());
    }

    #[test]
    fn invalid_utf8_nested_extra_keys_are_not_reclassified_as_binary() {
        for (key, value) in [
            ("opaque".into(), Value::Binary(vec![0xff; 32])),
            (Value::Binary(vec![0xff; 32]), "opaque".into()),
        ] {
            let extras = vec![Value::Map(vec![(key, value)])];
            let mut event = stored();
            change(&mut event, "extra_keys", Value::Array(extras.clone()));
            let mut payload = encode(vec![event], 0.into());
            let batch = decode_batch(&payload).unwrap();
            let KVEvent::BlockStored(event) = &batch.events[0] else {
                panic!("stored expected")
            };
            assert_eq!(event.extra_keys.as_ref(), Some(&extras));
            assert!(validate_device_dp1_batch(&batch, 2).is_err());

            let opaque = payload
                .windows(34)
                .position(|field| field[..2] == [0xc4, 32] && field[2..] == [0xff; 32])
                .unwrap();
            payload[opaque] = 0xd9; // Same bytes and length, but Str8 rather than Bin8.
            assert!(decode_batch(&payload).is_err());
        }
    }

    #[test]
    fn bounded_messagepack_and_trailing_bytes() {
        assert!(decode_batch(&vec![0; MAX_PAYLOAD_BYTES + 1]).is_err());
        let clear = Value::Map(vec![("type".into(), "AllBlocksCleared".into())]);
        assert!(decode_batch(&encode(vec![clear; 4097], 0.into())).is_err());
        let mut payload = encode(vec![stored()], 0.into());
        payload.push(0);
        assert!(decode_batch(&payload).is_err());
        let mut deep = Value::Nil;
        for _ in 0..20 {
            deep = Value::Array(vec![deep]);
        }
        let mut event = stored();
        change(&mut event, "extra_keys", deep);
        assert!(decode_batch(&encode(vec![event], 0.into())).is_err());
    }

    #[test]
    fn string_metadata_requires_string_or_nil_wire_values() {
        let removed = Value::Map(vec![
            ("type".into(), "BlockRemoved".into()),
            ("medium".into(), "GPU".into()),
            (
                "block_hashes".into(),
                Value::Array(vec![Value::Binary(vec![7; 32])]),
            ),
        ]);
        for (template, fields) in [
            (
                stored(),
                &[
                    "medium",
                    "lora_name",
                    "kv_cache_spec_kind",
                    "locality",
                    "ownership",
                    "session_id",
                ][..],
            ),
            (removed, &["medium", "locality", "ownership"][..]),
        ] {
            for field in fields {
                for valid in [Value::Nil, "text".into()] {
                    let mut event = template.clone();
                    change(&mut event, field, valid);
                    assert!(
                        decode_batch(&encode(vec![event], 0.into())).is_ok(),
                        "{field}"
                    );
                }
                for invalid in [
                    Value::Binary(b"GPU".to_vec()),
                    42.into(),
                    true.into(),
                    Value::Array(vec![]),
                    Value::Map(vec![]),
                ] {
                    let mut event = template.clone();
                    change(&mut event, field, invalid.clone());
                    assert!(
                        decode_batch(&encode(vec![stored(), event], 0.into())).is_err(),
                        "{field}: {invalid:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn unsigned_and_binary_hash_contract_is_preserved() {
        for hash in [0_u64.into(), u64::MAX.into(), Value::Binary(vec![0xc1; 32])] {
            let mut event = stored();
            change(&mut event, "block_hashes", Value::Array(vec![hash.clone()]));
            change(&mut event, "parent_block_hash", hash);
            assert!(decode_batch(&encode(vec![event], 0.into())).is_ok());
        }
        for field in ["block_hashes", "parent_block_hash"] {
            let mut event = stored();
            let value = if field == "block_hashes" {
                Value::Array(vec![(-1).into()])
            } else {
                (-1).into()
            };
            change(&mut event, field, value);
            assert!(decode_batch(&encode(vec![event], 0.into())).is_err());
        }
    }

    #[test]
    fn truncated_maximum_messagepack_lengths_return_errors() {
        for marker in [0xc6, 0xdb, 0xc9, 0xdd, 0xdf] {
            let payload = [marker, 0xff, 0xff, 0xff, 0xff];
            assert!(decode_batch(&payload).is_err(), "marker {marker:#x}");
            let mut nested = vec![0x92, 0xcb, 0, 0, 0, 0, 0, 0, 0, 0];
            nested.extend_from_slice(&payload);
            assert!(decode_batch(&nested).is_err(), "nested marker {marker:#x}");
        }
    }
}
