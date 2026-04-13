//! Msgpack decoder for vLLM KV cache events.
//!
//! vLLM serializes events using `msgspec` with `array_like=True` and
//! `tag=True`, which produces compact positional arrays with an integer
//! discriminator tag for polymorphic event types.
//!
//! Wire format (msgpack):
//!   EventBatch -> [ts: f64, events: [...], data_parallel_rank: int | nil]
//!   BlockStored (tag 0) -> [0, block_hashes, parent_hash, token_ids, block_size, ...]
//!   BlockRemoved (tag 1) -> [1, block_hashes, medium]
//!   AllBlocksCleared (tag 2) -> [2]

use tracing::warn;

/// A single KV cache mutation event from a vLLM worker.
#[derive(Debug, Clone)]
pub enum KVEvent {
    /// One or more KV blocks were computed and stored in GPU cache.
    BlockStored {
        /// Content-based hashes identifying each block.
        block_hashes: Vec<u64>,
        /// Hash of the parent block (for prefix chain verification).
        parent_block_hash: Option<u64>,
        /// Token IDs contained in these blocks.
        token_ids: Vec<u32>,
        /// Number of tokens per block.
        block_size: u32,
        /// Optional LoRA adapter name.
        lora_name: Option<String>,
    },
    /// One or more KV blocks were evicted from GPU cache.
    BlockRemoved {
        /// Content-based hashes of the removed blocks.
        block_hashes: Vec<u64>,
    },
    /// The entire KV cache of a worker was cleared (e.g., model reload).
    AllBlocksCleared,
}

/// A timestamped batch of [`KVEvent`]s originating from a single vLLM worker.
#[derive(Debug, Clone)]
pub struct KVEventBatch {
    /// Identifier of the source worker (typically its HTTP address).
    pub worker_id: String,
    /// Event timestamp (seconds since epoch, from vLLM scheduler clock).
    pub timestamp: f64,
    /// Individual events in this batch.
    pub events: Vec<KVEvent>,
    /// Data-parallel rank (when DP > 1 each rank has its own publisher).
    pub data_parallel_rank: Option<u32>,
}

/// Decode a raw msgpack payload into a [`KVEventBatch`].
///
/// The format mirrors vLLM's `EventBatch` (`msgspec.Struct, array_like=True`):
///   `[ts, [event, event, ...], dp_rank?]`
///
/// Each event is a tagged array where the first element is the type
/// discriminator assigned by `msgspec` (`tag=True`).
pub fn decode_event_batch(payload: &[u8], worker_id: &str) -> Result<KVEventBatch, String> {
    let value: rmpv::Value = rmpv::decode::read_value(&mut std::io::Cursor::new(payload))
        .map_err(|e| format!("msgpack decode error: {}", e))?;

    let arr = value
        .as_array()
        .ok_or_else(|| "EventBatch: expected array at top level".to_string())?;

    if arr.len() < 2 {
        return Err(format!(
            "EventBatch: expected >= 2 elements, got {}",
            arr.len()
        ));
    }

    let timestamp = arr[0]
        .as_f64()
        .ok_or_else(|| "EventBatch: ts must be float".to_string())?;

    let raw_events = arr[1]
        .as_array()
        .ok_or_else(|| "EventBatch: events must be array".to_string())?;

    let data_parallel_rank = arr.get(2).and_then(|v| v.as_u64()).map(|v| v as u32);

    let mut events = Vec::with_capacity(raw_events.len());
    for raw in raw_events {
        match decode_single_event(raw) {
            Ok(ev) => events.push(ev),
            Err(e) => {
                warn!("Skipping malformed KV event from {}: {}", worker_id, e);
            }
        }
    }

    Ok(KVEventBatch {
        worker_id: worker_id.to_string(),
        timestamp,
        events,
        data_parallel_rank,
    })
}

/// Decode a single tagged event array.
///
/// Tag values follow the declaration order in vLLM's `kv_events.py`:
///   BlockStored = 0, BlockRemoved = 1, AllBlocksCleared = 2
fn decode_single_event(value: &rmpv::Value) -> Result<KVEvent, String> {
    let arr = value
        .as_array()
        .ok_or_else(|| "event: expected array".to_string())?;

    if arr.is_empty() {
        return Err("event: empty array".to_string());
    }

    // The tag field is the first element (msgspec tag=True).
    // For Struct subclasses, msgspec assigns tags as strings by default:
    //   "BlockStored", "BlockRemoved", "AllBlocksCleared"
    // It can also be an integer depending on configuration.
    let tag = if let Some(s) = arr[0].as_str() {
        s.to_string()
    } else if let Some(n) = arr[0].as_u64() {
        n.to_string()
    } else {
        return Err(format!("event: unrecognized tag type: {:?}", arr[0]));
    };

    match tag.as_str() {
        "BlockStored" | "0" => decode_block_stored(arr),
        "BlockRemoved" | "1" => decode_block_removed(arr),
        "AllBlocksCleared" | "2" => Ok(KVEvent::AllBlocksCleared),
        other => Err(format!("event: unknown tag '{}'", other)),
    }
}

/// Decode a `BlockStored` event.
///
/// Expected layout (after tag):
///   [tag, block_hashes, parent_hash, token_ids, block_size, lora_id, medium, lora_name, extra_keys?]
fn decode_block_stored(arr: &[rmpv::Value]) -> Result<KVEvent, String> {
    if arr.len() < 8 {
        return Err(format!(
            "BlockStored: expected >= 8 fields, got {}",
            arr.len()
        ));
    }

    let block_hashes = extract_u64_array(&arr[1], "block_hashes")?;

    let parent_block_hash = if arr[2].is_nil() {
        None
    } else {
        Some(
            arr[2]
                .as_u64()
                .ok_or_else(|| "BlockStored: parent_hash must be uint or nil".to_string())?,
        )
    };

    let token_ids = extract_u32_array(&arr[3], "token_ids")?;

    let block_size = arr[4]
        .as_u64()
        .ok_or_else(|| "BlockStored: block_size must be uint".to_string())?
        as u32;

    // arr[5] = lora_id (deprecated, skip)
    // arr[6] = medium (skip for now)

    let lora_name = if arr.len() > 7 {
        arr[7].as_str().map(|s| s.to_string())
    } else {
        None
    };

    Ok(KVEvent::BlockStored {
        block_hashes,
        parent_block_hash,
        token_ids,
        block_size,
        lora_name,
    })
}

/// Decode a `BlockRemoved` event.
///
/// Expected layout: [tag, block_hashes, medium]
fn decode_block_removed(arr: &[rmpv::Value]) -> Result<KVEvent, String> {
    if arr.len() < 2 {
        return Err(format!(
            "BlockRemoved: expected >= 2 fields, got {}",
            arr.len()
        ));
    }

    let block_hashes = extract_u64_array(&arr[1], "block_hashes")?;

    Ok(KVEvent::BlockRemoved { block_hashes })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_u64_array(val: &rmpv::Value, field: &str) -> Result<Vec<u64>, String> {
    let arr = val
        .as_array()
        .ok_or_else(|| format!("{}: expected array", field))?;
    arr.iter()
        .map(|v| {
            // vLLM ExternalBlockHash can be a signed i64 stored as msgpack int.
            if let Some(n) = v.as_u64() {
                Ok(n)
            } else if let Some(n) = v.as_i64() {
                Ok(n as u64)
            } else {
                Err(format!("{}: element is not integer: {:?}", field, v))
            }
        })
        .collect()
}

fn extract_u32_array(val: &rmpv::Value, field: &str) -> Result<Vec<u32>, String> {
    let arr = val
        .as_array()
        .ok_or_else(|| format!("{}: expected array", field))?;
    arr.iter()
        .map(|v| {
            v.as_u64()
                .map(|n| n as u32)
                .ok_or_else(|| format!("{}: element is not integer: {:?}", field, v))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_empty_batch() {
        // Minimal valid batch: [0.0, []]
        let data = rmpv::Value::Array(vec![
            rmpv::Value::F64(1234567890.0),
            rmpv::Value::Array(vec![]),
        ]);
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &data).unwrap();

        let batch = decode_event_batch(&buf, "test-worker").unwrap();
        assert_eq!(batch.worker_id, "test-worker");
        assert!((batch.timestamp - 1234567890.0).abs() < f64::EPSILON);
        assert!(batch.events.is_empty());
        assert!(batch.data_parallel_rank.is_none());
    }

    #[test]
    fn test_decode_block_removed() {
        let event = rmpv::Value::Array(vec![
            rmpv::Value::String("BlockRemoved".into()),
            rmpv::Value::Array(vec![
                rmpv::Value::Integer(42.into()),
                rmpv::Value::Integer(99.into()),
            ]),
            rmpv::Value::Nil,
        ]);
        let batch_val =
            rmpv::Value::Array(vec![rmpv::Value::F64(1.0), rmpv::Value::Array(vec![event])]);
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &batch_val).unwrap();

        let batch = decode_event_batch(&buf, "w1").unwrap();
        assert_eq!(batch.events.len(), 1);
        match &batch.events[0] {
            KVEvent::BlockRemoved { block_hashes } => {
                assert_eq!(block_hashes, &[42, 99]);
            }
            other => panic!("expected BlockRemoved, got {:?}", other),
        }
    }
}
