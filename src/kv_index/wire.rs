//! msgpack wire types mirroring vLLM's KV cache events.

use serde::{Deserialize, Serialize};

/// Engine-emitted block hash — `bytes` or `int` on the wire. Opaque key for
/// parent chaining and `remove` lookup only.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExternalBlockHash {
    Bytes(Vec<u8>),
    Int(u64),
}

impl<'de> Deserialize<'de> for ExternalBlockHash {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{self, Visitor};
        use std::fmt;

        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = ExternalBlockHash;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("bytes or int")
            }
            fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                Ok(ExternalBlockHash::Bytes(v.to_vec()))
            }
            fn visit_borrowed_bytes<E: de::Error>(self, v: &'de [u8]) -> Result<Self::Value, E> {
                Ok(ExternalBlockHash::Bytes(v.to_vec()))
            }
            fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
                Ok(ExternalBlockHash::Bytes(v))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(ExternalBlockHash::Int(v))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(ExternalBlockHash::Int(v as u64))
            }
        }
        deserializer.deserialize_any(V)
    }
}

impl Serialize for ExternalBlockHash {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            ExternalBlockHash::Bytes(b) => serializer.serialize_bytes(b),
            ExternalBlockHash::Int(i) => serializer.serialize_u64(*i),
        }
    }
}

/// A KV cache event. vLLM tags each with the class name under `"type"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum KVEvent {
    #[serde(rename = "BlockStored")]
    BlockStored(BlockStored),
    #[serde(rename = "BlockRemoved")]
    BlockRemoved(BlockRemoved),
    #[serde(rename = "AllBlocksCleared")]
    AllBlocksCleared(AllBlocksCleared),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockStored {
    pub block_hashes: Vec<ExternalBlockHash>,
    pub parent_block_hash: Option<ExternalBlockHash>,
    pub token_ids: Vec<u32>,
    pub block_size: u32,
    pub lora_id: Option<i64>,
    pub medium: Option<String>,
    pub lora_name: Option<String>,
    #[serde(default)]
    pub group_idx: Option<u32>,
    #[serde(default)]
    pub kv_cache_spec_kind: Option<String>,
    #[serde(default)]
    pub kv_cache_spec_sliding_window: Option<u32>,
    #[serde(default)]
    pub locality: Option<String>,
    #[serde(default)]
    pub ownership: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockRemoved {
    pub block_hashes: Vec<ExternalBlockHash>,
    pub medium: Option<String>,
    #[serde(default)]
    pub group_idx: Option<u32>,
    #[serde(default)]
    pub locality: Option<String>,
    #[serde(default)]
    pub ownership: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AllBlocksCleared {}

/// A batch of events. vLLM encodes with `array_like=True` as a positional
/// array `[ts, events, (data_parallel_rank)?]`; `seq` rides in its own ZMQ
/// frame, not here. Derived `Deserialize` reads maps, so a custom positional
/// impl is required.
#[derive(Debug, Clone, Serialize)]
pub struct KVEventBatch {
    pub ts: f64,
    pub events: Vec<KVEvent>,
    #[serde(default)]
    pub data_parallel_rank: Option<u32>,
}

impl<'de> Deserialize<'de> for KVEventBatch {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{self, SeqAccess, Visitor};
        use std::fmt;

        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = KVEventBatch;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a 2- or 3-element array [ts, events, (rank)?]")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let ts: f64 = seq.next_element()?.ok_or_else(|| de::Error::custom("missing ts"))?;
                let events: Vec<KVEvent> =
                    seq.next_element()?.ok_or_else(|| de::Error::custom("missing events"))?;
                let data_parallel_rank: Option<u32> = seq.next_element()?;
                Ok(KVEventBatch { ts, events, data_parallel_rank })
            }
        }
        deserializer.deserialize_seq(V)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmp_serde::{from_slice, to_vec};

    #[test]
    fn block_stored_with_int_hashes_round_trips() {
        let ev = KVEvent::BlockStored(BlockStored {
            block_hashes: vec![ExternalBlockHash::Int(42), ExternalBlockHash::Int(43)],
            parent_block_hash: Some(ExternalBlockHash::Int(42)),
            token_ids: vec![1, 2, 3, 4],
            block_size: 4,
            lora_id: None,
            medium: Some("device".into()),
            lora_name: None,
            group_idx: Some(0),
            kv_cache_spec_kind: Some("full".into()),
            kv_cache_spec_sliding_window: None,
            locality: Some("LOCAL".into()),
            ownership: None,
            session_id: None,
        });
        let buf = to_vec(&ev).unwrap();
        match from_slice::<KVEvent>(&buf).unwrap() {
            KVEvent::BlockStored(b) => {
                assert_eq!(b.block_hashes.len(), 2);
                assert_eq!(b.block_size, 4);
                assert_eq!(b.token_ids, vec![1, 2, 3, 4]);
                assert_eq!(b.group_idx, Some(0));
            }
            _ => panic!("expected BlockStored"),
        }
    }

    #[test]
    fn block_stored_with_byte_hashes_round_trips() {
        let ev = KVEvent::BlockStored(BlockStored {
            block_hashes: vec![ExternalBlockHash::Bytes(vec![0u8; 32])],
            parent_block_hash: None,
            token_ids: vec![5],
            block_size: 16,
            lora_id: None,
            medium: None,
            lora_name: None,
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            locality: None,
            ownership: None,
            session_id: None,
        });
        let buf = to_vec(&ev).unwrap();
        match from_slice::<KVEvent>(&buf).unwrap() {
            KVEvent::BlockStored(b) => match &b.block_hashes[0] {
                ExternalBlockHash::Bytes(b) => assert_eq!(b.len(), 32),
                _ => panic!("expected bytes hash"),
            },
            _ => panic!("expected BlockStored"),
        }
    }

    #[test]
    fn block_removed_optionals_default_when_missing() {
        #[derive(Serialize)]
        #[serde(tag = "type")]
        enum Tag<'a> {
            #[serde(rename = "BlockRemoved")]
            BlockRemoved(&'a BlockRemoved),
        }
        let minimal = BlockRemoved {
            block_hashes: vec![ExternalBlockHash::Int(42)],
            medium: None,
            group_idx: None,
            locality: None,
            ownership: None,
        };
        let buf = to_vec(&Tag::BlockRemoved(&minimal)).unwrap();
        match from_slice::<KVEvent>(&buf).unwrap() {
            KVEvent::BlockRemoved(b) => {
                assert_eq!(b.block_hashes.len(), 1);
                assert!(b.group_idx.is_none());
                assert!(b.locality.is_none());
                assert!(b.ownership.is_none());
            }
            _ => panic!("expected BlockRemoved"),
        }
    }

    #[test]
    fn all_blocks_cleared_round_trips() {
        let ev = KVEvent::AllBlocksCleared(AllBlocksCleared {});
        let buf = to_vec(&ev).unwrap();
        assert!(matches!(from_slice::<KVEvent>(&buf).unwrap(), KVEvent::AllBlocksCleared(_)));
    }

    #[test]
    fn batch_decodes_two_element_array() {
        let buf = to_vec(&(1.5f64, Vec::<KVEvent>::new())).unwrap();
        let back: KVEventBatch = from_slice(&buf).unwrap();
        assert_eq!(back.ts, 1.5);
        assert!(back.events.is_empty());
        assert!(back.data_parallel_rank.is_none());
    }

    #[test]
    fn batch_decodes_three_element_array() {
        let buf = to_vec(&(2.5f64, Vec::<KVEvent>::new(), 1u32)).unwrap();
        let back: KVEventBatch = from_slice(&buf).unwrap();
        assert_eq!(back.ts, 2.5);
        assert_eq!(back.data_parallel_rank, Some(1));
    }
}
