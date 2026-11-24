//! Structural decoding into the shared wire contract.
use std::{collections::HashSet, io::Cursor};

use crate::kv_index::wire::{KVEvent, KVEventBatch};
use rmpv::Value;
use serde::Deserialize;

/// Decode atomically without conflating structural validity with device admission.
pub fn decode_batch(payload: &[u8]) -> Result<KVEventBatch, String> {
    if payload.len() > super::MAX_KV_EVENT_BYTES {
        return Err("KV event payload exceeds 64 MiB".into());
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
    let Value::Array(mut batch) = value else {
        return Err("KV event batch must be an array".into());
    };
    if !(2..=3).contains(&batch.len()) {
        return Err("KV batch requires timestamp, events, and optional rank".into());
    }
    // Ordering comes from the sequence frame. The first slot is publisher time.
    let data_parallel_rank = match batch.get(2) {
        None | Some(Value::Nil) => None,
        Some(rank) => Some(
            u32::try_from(rank.as_u64().ok_or("invalid DP rank")?)
                .map_err(|_| "invalid DP rank")?,
        ),
    };
    let Value::Array(events) = batch.remove(1) else {
        return Err("KV events must be an array".into());
    };
    let events = events
        .into_iter()
        .map(decode_event)
        .collect::<Result<_, _>>()?;
    Ok(KVEventBatch {
        events,
        data_parallel_rank,
    })
}

fn decode_event(event: Value) -> Result<KVEvent, String> {
    const STORED_FIELDS: &[&str] = &[
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
        "session_id", // Accepted for older publishers. Residency does not use it.
        "lookahead_token_ids",
    ];
    let pairs = event.as_map().ok_or("KV event must be a tagged map")?;
    let mut keys = HashSet::with_capacity(pairs.len().min(STORED_FIELDS.len()));
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
    type Decode = fn(Value) -> Result<KVEvent, rmpv::ext::Error>;
    let (allowed, required, decode): (&[&str], &[&str], Decode) = match kind {
        "BlockStored" => (
            STORED_FIELDS,
            &[
                "block_hashes",
                "parent_block_hash",
                "token_ids",
                "block_size",
                "lora_id",
                "medium",
                "lora_name",
            ],
            |value| rmpv::ext::from_value(value).map(KVEvent::BlockStored),
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
            |value| rmpv::ext::from_value(value).map(KVEvent::BlockRemoved),
        ),
        "AllBlocksCleared" => (&["type"], &[], |value| {
            rmpv::ext::from_value(value).map(KVEvent::AllBlocksCleared)
        }),
        _ => return Err(format!("unsupported KV event type: {kind}")),
    };
    if let Some(key) = keys.iter().filter(|key| !allowed.contains(key)).min() {
        return Err(format!("unsupported KV event field: {key}"));
    }
    if let Some(key) = required.iter().find(|key| !keys.contains(**key)) {
        return Err(format!("missing KV event field: {key}"));
    }
    for (key, value) in pairs {
        match key.as_str() {
            Some("block_hashes") => {
                // Structure only. vLLM emits an empty list for a Mamba checkpoint
                // group whose states in a chunk were all skipped, so whether an
                // empty list is acceptable depends on the group and is decided by
                // each consumer, not by the shared decoder.
                let hashes = value.as_array().ok_or("KV event hashes must be an array")?;
                for hash in hashes {
                    validate_wire_hash(hash)?;
                }
            }
            Some("parent_block_hash") if !value.is_nil() => validate_wire_hash(value)?,
            Some("lookahead_token_ids") if !value.is_nil() => {
                let blocks = value
                    .as_array()
                    .ok_or("KV lookahead tokens must be an array of token arrays")?;
                for block in blocks {
                    let tokens = block
                        .as_array()
                        .ok_or("KV lookahead tokens must be an array of token arrays")?;
                    if !tokens
                        .iter()
                        .all(|token| token.as_u64().is_some_and(|id| u32::try_from(id).is_ok()))
                    {
                        return Err("KV lookahead tokens must be unsigned 32-bit integers".into());
                    }
                }
            }
            Some("medium" | "lora_name" | "kv_cache_spec_kind" | "locality" | "ownership")
                if !matches!(value, Value::String(_) | Value::Nil) =>
            {
                return Err(format!("KV event field {key} must be a string or nil"));
            }
            _ => {}
        }
    }
    // The Value deserializer turns invalid UTF-8 strings into bytes, including
    // opaque extra-key values and map keys. Reject them before DTO conversion.
    validate_utf8_strings(&event)?;
    decode(event).map_err(|error| format!("invalid KV event: {error}"))
}

fn validate_utf8_strings(value: &Value) -> Result<(), String> {
    match value {
        Value::String(text) if text.as_str().is_none() => {
            Err("invalid UTF-8 KV event string".into())
        }
        Value::Array(values) => {
            for value in values {
                if matches!(value, Value::String(_) | Value::Map(_) | Value::Array(_)) {
                    validate_utf8_strings(value)?;
                }
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

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::kv_index::wire::{BlockStored, ExternalBlockHash};

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
            lookahead_token_ids: None,
        });
        // vLLM emits tagged maps; generic Value serialization uses positional
        // structs. Use the common DTO's named MessagePack representation.
        let bytes = rmp_serde::to_vec_named(&event).unwrap();
        rmpv::decode::read_value(&mut Cursor::new(bytes)).unwrap()
    }

    fn encode(events: Vec<Value>, rank: impl Into<Option<Value>>) -> Vec<u8> {
        let mut batch = vec![1.25.into(), Value::Array(events)];
        if let Some(rank) = rank.into() {
            batch.push(rank);
        }
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &Value::Array(batch)).unwrap();
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
            ("session_id", 42.into()),
            ("lora_id", 9.into()),
            ("lora_name", "adapter".into()),
            ("extra_keys", Value::Array(vec!["salt".into()])),
        ] {
            change(&mut event, key, value);
        }
        let batch = decode_batch(&encode(vec![event], Value::from(3))).unwrap();
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
        assert_eq!(event.lora_id, Some(9));
        assert_eq!(event.lora_name.as_deref(), Some("adapter"));
        assert_eq!(event.extra_keys, Some(vec!["salt".into()]));
    }

    #[test]
    fn lookahead_tokens_are_preserved_as_one_array_per_hash() {
        let mut event = stored();
        change(
            &mut event,
            "lookahead_token_ids",
            Value::Array(vec![Value::Array(vec![42.into(), Value::from(u32::MAX)])]),
        );
        let batch = decode_batch(&encode(vec![event.clone()], None)).unwrap();
        let KVEvent::BlockStored(store) = &batch.events[0] else {
            panic!("stored expected")
        };
        assert_eq!(store.lookahead_token_ids, Some(vec![vec![42, u32::MAX]]));

        for malformed in [
            Value::Binary(vec![42]),
            Value::Array(vec![42.into()]),
            Value::Array(vec![Value::Array(vec![(-1).into()])]),
            Value::Array(vec![Value::Array(vec![Value::from(
                u64::from(u32::MAX) + 1,
            )])]),
        ] {
            change(&mut event, "lookahead_token_ids", malformed);
            assert!(decode_batch(&encode(vec![event.clone()], None)).is_err());
        }
    }

    #[test]
    fn event_rank_preserves_absent_nil_zero_and_nonzero() {
        for event in [
            stored(),
            Value::Map(vec![
                ("type".into(), "BlockRemoved".into()),
                (
                    "block_hashes".into(),
                    Value::Array(vec![Value::Binary(vec![7; 32])]),
                ),
                ("medium".into(), "GPU".into()),
            ]),
            Value::Map(vec![("type".into(), "AllBlocksCleared".into())]),
        ] {
            for rank in [None, Some(Value::Nil), Some(0.into()), Some(7.into())] {
                let expected = rank
                    .as_ref()
                    .and_then(Value::as_u64)
                    .map(|rank| rank as u32);
                let batch = decode_batch(&encode(vec![event.clone()], rank)).unwrap();
                assert_eq!(batch.data_parallel_rank, expected);
            }
        }
    }

    #[test]
    fn extra_key_shapes_are_preserved_before_admission() {
        let extension = Value::Array(vec![Value::Ext(7, vec![0xc1, 0xff])]);
        for (wire, expected) in [
            (None, None),
            (Some(Value::Nil), None),
            (Some(Value::Array(vec![Value::Nil])), Some(vec![Value::Nil])),
            (
                Some(Value::Array(vec![extension.clone()])),
                Some(vec![extension]),
            ),
            (
                Some(Value::Array(vec![
                    Value::Nil,
                    Value::Binary(vec![0xc1, 0xff]),
                ])),
                Some(vec![Value::Nil, Value::Binary(vec![0xc1, 0xff])]),
            ),
        ] {
            let mut event = stored();
            let Value::Map(fields) = &mut event else {
                unreachable!()
            };
            fields.retain(|(key, _)| key.as_str() != Some("extra_keys"));
            if let Some(wire) = wire {
                change(&mut event, "extra_keys", wire);
            }
            let batch = decode_batch(&encode(vec![event], Value::Nil)).unwrap();
            let KVEvent::BlockStored(event) = &batch.events[0] else {
                unreachable!()
            };
            assert_eq!(event.extra_keys, expected);
        }
    }

    #[test]
    fn unsupported_event_fields_report_smallest_key() {
        for (kind, template) in [
            ("BlockStored", stored()),
            (
                "BlockRemoved",
                Value::Map(vec![
                    ("type".into(), "BlockRemoved".into()),
                    ("medium".into(), "GPU".into()),
                    (
                        "block_hashes".into(),
                        Value::Array(vec![Value::Binary(vec![7; 32])]),
                    ),
                ]),
            ),
            (
                "AllBlocksCleared",
                Value::Map(vec![("type".into(), "AllBlocksCleared".into())]),
            ),
        ] {
            assert!(decode_batch(&encode(vec![template.clone()], Value::from(0))).is_ok());
            for (unknown_keys, expected) in [
                (&["zzzz", "aaaa"][..], "aaaa"),
                (&["aaaa", "zzzz"][..], "aaaa"),
                (&["zzzz"][..], "zzzz"),
            ] {
                let mut event = template.clone();
                for key in unknown_keys {
                    change(&mut event, key, Value::Nil);
                }
                assert_eq!(
                    decode_batch(&encode(vec![stored(), event], Value::from(0))).unwrap_err(),
                    format!("unsupported KV event field: {expected}"),
                    "{kind}: {unknown_keys:?}"
                );
            }
        }
    }

    #[test]
    fn malformed_lifecycle_rejects_whole_batch() {
        for (event, expected) in [
            (
                Value::Map(vec![
                    ("type".into(), "AllBlocksCleared".into()),
                    ("medium".into(), "GPU".into()),
                ]),
                "unsupported KV event field: medium",
            ),
            (
                Value::Map(vec![
                    ("type".into(), "BlockRemoved".into()),
                    ("medium".into(), "GPU".into()),
                ]),
                "missing KV event field: block_hashes",
            ),
            (
                Value::Map(vec![
                    ("type".into(), "AllBlocksCleared".into()),
                    ("zzzz".into(), Value::Nil),
                    ("aaaa".into(), Value::Nil),
                    ("type".into(), "AllBlocksCleared".into()),
                ]),
                "duplicate KV event field: type",
            ),
        ] {
            assert_eq!(
                decode_batch(&encode(vec![stored(), event], Value::from(0))).unwrap_err(),
                expected
            );
        }
        let removed = Value::Map(vec![
            ("type".into(), "BlockRemoved".into()),
            ("medium".into(), "CPU".into()),
            (
                "block_hashes".into(),
                Value::Array(vec![Value::Binary(vec![7; 32])]),
            ),
        ]);
        let batch = decode_batch(&encode(vec![removed], Value::from(0))).unwrap();
        assert!(
            matches!(&batch.events[0], KVEvent::BlockRemoved(event) if event.medium.as_deref() == Some("CPU"))
        );
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
        let mut payload = encode(vec![event], Value::from(0));
        let batch = decode_batch(&payload).unwrap();
        assert!(matches!(&batch.events[0], KVEvent::BlockStored(event)
            if event.block_hashes == [ExternalBlockHash::Bytes(vec![0xc1; 32])]));

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
        let mut payload = encode(vec![event], Value::from(0));
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
            let mut payload = encode(vec![event], Value::from(0));
            let batch = decode_batch(&payload).unwrap();
            let KVEvent::BlockStored(event) = &batch.events[0] else {
                panic!("stored expected")
            };
            assert_eq!(event.extra_keys.as_ref(), Some(&extras));

            let opaque = payload
                .windows(34)
                .position(|field| field[..2] == [0xc4, 32] && field[2..] == [0xff; 32])
                .unwrap();
            payload[opaque] = 0xd9; // Same bytes and length, but Str8 rather than Bin8.
            assert!(decode_batch(&payload).is_err());
        }
    }

    #[test]
    fn malformed_messagepack_and_trailing_bytes() {
        let mut payload = encode(vec![stored()], Value::from(0));
        payload.push(0);
        assert!(decode_batch(&payload).is_err());
        let mut deep = Value::Nil;
        for _ in 0..20 {
            deep = Value::Array(vec![deep]);
        }
        let mut event = stored();
        change(&mut event, "extra_keys", deep);
        assert!(decode_batch(&encode(vec![event], Value::from(0))).is_err());
    }

    #[test]
    fn large_eviction_batch_decodes() {
        let removed = Value::Map(vec![
            ("type".into(), "BlockRemoved".into()),
            ("medium".into(), "GPU".into()),
            (
                "block_hashes".into(),
                Value::Array(vec![Value::Binary(vec![7; 32])]),
            ),
        ]);
        let batch = decode_batch(&encode(vec![removed; 4712], Value::from(0))).unwrap();
        assert_eq!(batch.events.len(), 4712);
    }

    #[test]
    fn oversized_payload_is_rejected_before_decoding() {
        let payload = vec![0; super::super::MAX_KV_EVENT_BYTES + 1];
        assert_eq!(
            decode_batch(&payload).unwrap_err(),
            "KV event payload exceeds 64 MiB"
        );
    }

    #[test]
    fn large_event_payload_decodes() {
        let hash_count = 16 * 1024 * 1024 / 34 + 1;
        let removed = Value::Map(vec![
            ("type".into(), "BlockRemoved".into()),
            ("medium".into(), "GPU".into()),
            (
                "block_hashes".into(),
                Value::Array(vec![Value::Binary(vec![7; 32]); hash_count]),
            ),
        ]);
        let payload = encode(vec![removed], Value::from(0));
        assert!(payload.len() > 16 * 1024 * 1024);
        let batch = decode_batch(&payload).unwrap();
        let KVEvent::BlockRemoved(event) = &batch.events[0] else {
            panic!("removed expected")
        };
        assert_eq!(event.block_hashes.len(), hash_count);
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
                ][..],
            ),
            (removed, &["medium", "locality", "ownership"][..]),
        ] {
            for field in fields {
                for valid in [Value::Nil, "text".into()] {
                    let mut event = template.clone();
                    change(&mut event, field, valid);
                    assert!(
                        decode_batch(&encode(vec![event], Value::from(0))).is_ok(),
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
                        decode_batch(&encode(vec![stored(), event], Value::from(0))).is_err(),
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
            assert!(decode_batch(&encode(vec![event], Value::from(0))).is_ok());
        }
        for field in ["block_hashes", "parent_block_hash"] {
            let mut event = stored();
            let value = if field == "block_hashes" {
                Value::Array(vec![(-1).into()])
            } else {
                (-1).into()
            };
            change(&mut event, field, value);
            assert!(decode_batch(&encode(vec![event], Value::from(0))).is_err());
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

    #[test]
    fn empty_hash_list_decodes_for_consumers_to_judge() {
        let mut event = stored();
        change(&mut event, "block_hashes", Value::Array(vec![]));
        change(&mut event, "extra_keys", Value::Nil);
        let batch = decode_batch(&encode(vec![event], Value::from(0))).unwrap();
        assert_eq!(batch.events.len(), 1);
        // Non-array hashes are still structurally invalid.
        let mut bad = stored();
        change(&mut bad, "block_hashes", Value::Nil);
        assert!(decode_batch(&encode(vec![bad], Value::from(0))).is_err());
    }
}
