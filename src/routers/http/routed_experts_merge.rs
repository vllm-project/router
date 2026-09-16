//! Routed-experts merging for P/D disaggregation.
//!
//! With `--enable-return-routed-experts` each response choice carries a
//! `routed_experts` field: a base64-encoded NumPy `.npy` blob of shape
//! `(num_tokens-1, num_layers, num_experts_per_tok)`. Under P/D the decode
//! replica pulls the prompt KV and never forwards the prompt, so its
//! prompt-region rows are invalid; we splice the prefill replica's rows over
//! them. Connector-agnostic, non-streaming only.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::Value;
use tracing::{debug, warn};

struct NpyHeader {
    descr: String,
    shape: Vec<usize>,
    data_offset: usize,
}

impl NpyHeader {
    fn itemsize(&self) -> usize {
        self.descr
            .chars()
            .filter(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .unwrap_or(1)
    }

    fn row_bytes(&self) -> usize {
        self.shape.iter().skip(1).product::<usize>() * self.itemsize()
    }
}

/// Parse a numpy v1.0 `.npy` header, assuming the well-formed payload vLLM emits.
fn parse_npy_header(buf: &[u8]) -> NpyHeader {
    let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let data_offset = 10 + header_len;
    let header = std::str::from_utf8(&buf[10..data_offset]).unwrap_or("");

    let inner = header
        .split_once('(')
        .and_then(|(_, r)| r.split_once(')'))
        .map(|(l, _)| l)
        .unwrap_or("");
    let shape = inner
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();

    NpyHeader {
        descr: quoted(header, "descr"),
        shape,
        data_offset,
    }
}

fn quoted(header: &str, key: &str) -> String {
    let pat = format!("'{key}':");
    header
        .find(&pat)
        .and_then(|i| header[i + pat.len()..].split('\'').nth(1))
        .unwrap_or("")
        .to_string()
}

/// Splice prefill routed-experts rows over the decode array's invalid
/// prompt-region prefix. Inputs are base64 `.npy` blobs from a P/D pair (same
/// dtype/trailing shape, prefill rows <= decode rows); output reuses decode's
/// header (shape unchanged) with its leading rows replaced by prefill's.
pub fn splice_npy(prefill_b64: &str, decode_b64: &str) -> Result<String, String> {
    let p_buf = STANDARD.decode(prefill_b64).map_err(|e| e.to_string())?;
    let d_buf = STANDARD.decode(decode_b64).map_err(|e| e.to_string())?;
    let ph = parse_npy_header(&p_buf);
    let dh = parse_npy_header(&d_buf);

    let split = ph.shape[0] * dh.row_bytes();
    let mut out = Vec::with_capacity(d_buf.len());
    out.extend_from_slice(&d_buf[..dh.data_offset]);
    out.extend_from_slice(&p_buf[ph.data_offset..ph.data_offset + split]);
    out.extend_from_slice(&d_buf[dh.data_offset + split..]);
    Ok(STANDARD.encode(&out))
}

/// Whether any choice in `resp` carries a non-null `routed_experts` field.
pub fn has_routed_experts(resp: &Value) -> bool {
    resp.get("choices")
        .and_then(Value::as_array)
        .map(|choices| {
            choices.iter().any(|c| {
                c.get("routed_experts")
                    .map(|v| !v.is_null())
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

/// Merge `routed_experts` from a prefill response into a decode response,
/// in place. Choices are matched by their `index` field. Returns whether any
/// choice was merged.
pub fn merge_routed_experts_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
    let prefill_choices = match prefill_json.get("choices").and_then(Value::as_array) {
        Some(c) => c,
        None => return false,
    };
    let decode_choices = match decode_json.get_mut("choices").and_then(Value::as_array_mut) {
        Some(c) => c,
        None => return false,
    };

    let mut merged = false;
    for decode_choice in decode_choices.iter_mut() {
        let idx = decode_choice.get("index").and_then(Value::as_u64);

        // Find the prefill choice with the same index; fall back to positional
        // alignment when no usable index is present.
        let prefill_choice = prefill_choices
            .iter()
            .find(|c| c.get("index").and_then(Value::as_u64) == idx)
            .or_else(|| prefill_choices.first());

        let prefill_re = prefill_choice
            .and_then(|c| c.get("routed_experts"))
            .and_then(Value::as_str);
        let decode_re = decode_choice.get("routed_experts").and_then(Value::as_str);

        let (prefill_re, decode_re) = match (prefill_re, decode_re) {
            (Some(p), Some(d)) => (p, d),
            _ => continue,
        };

        match splice_npy(prefill_re, decode_re) {
            Ok(spliced) => {
                if let Some(obj) = decode_choice.as_object_mut() {
                    obj.insert("routed_experts".to_string(), Value::String(spliced));
                    merged = true;
                    debug!("[ROUTED_EXPERTS MERGE] spliced choice index {:?}", idx);
                }
            }
            Err(e) => {
                warn!(
                    "[ROUTED_EXPERTS MERGE] skipped choice index {:?}: {}",
                    idx, e
                );
            }
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny C-order `.npy` v1.0 blob for a `(n, l, k)` uint8 array whose
    /// element at (row, layer, expert) == base + row (so rows are distinguishable).
    fn make_npy_u8(n: usize, l: usize, k: usize, base: u8) -> Vec<u8> {
        let header = format!(
            "{{'descr': '|u1', 'fortran_order': False, 'shape': ({}, {}, {}), }}",
            n, l, k
        );
        // pad header so total (10 + len) is a multiple of 64, ending in '\n'.
        let mut h = header.into_bytes();
        let total = 10 + h.len() + 1;
        let pad = (64 - (total % 64)) % 64;
        h.resize(h.len() + pad, b' ');
        h.push(b'\n');

        let mut buf = Vec::new();
        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1);
        buf.push(0);
        buf.extend_from_slice(&(h.len() as u16).to_le_bytes());
        buf.extend_from_slice(&h);
        for row in 0..n {
            for _ in 0..(l * k) {
                buf.push(base.wrapping_add(row as u8));
            }
        }
        buf
    }

    fn b64(v: &[u8]) -> String {
        STANDARD.encode(v)
    }

    #[test]
    fn parse_header_roundtrip() {
        let buf = make_npy_u8(3, 2, 4, 0);
        let h = parse_npy_header(&buf);
        assert_eq!(h.descr, "|u1");
        assert_eq!(h.shape, vec![3, 2, 4]);
        assert_eq!(h.itemsize(), 1);
        assert_eq!(h.row_bytes(), 8);
    }

    #[test]
    fn splice_replaces_prefix_only() {
        // prefill: 2 rows valued 100,101 ; decode: 5 rows valued 0..4.
        let l = 2;
        let k = 3;
        let prefill = make_npy_u8(2, l, k, 100);
        let decode = make_npy_u8(5, l, k, 0);
        let merged_b64 = splice_npy(&b64(&prefill), &b64(&decode)).unwrap();
        let merged = STANDARD.decode(merged_b64).unwrap();

        let h = parse_npy_header(&merged);
        assert_eq!(h.shape, vec![5, l, k]); // keeps decode shape
        let row_bytes = l * k;
        let data = &merged[h.data_offset..];
        // rows 0,1 from prefill (100,101); rows 2,3,4 from decode (2,3,4).
        assert_eq!(data[0], 100);
        assert_eq!(data[row_bytes], 101);
        assert_eq!(data[2 * row_bytes], 2);
        assert_eq!(data[3 * row_bytes], 3);
        assert_eq!(data[4 * row_bytes], 4);
    }

    #[test]
    fn splice_equal_lengths_uses_all_prefill() {
        let prefill = make_npy_u8(4, 1, 1, 50);
        let decode = make_npy_u8(4, 1, 1, 0);
        let merged = STANDARD
            .decode(splice_npy(&b64(&prefill), &b64(&decode)).unwrap())
            .unwrap();
        let h = parse_npy_header(&merged);
        let data = &merged[h.data_offset..];
        assert_eq!(data, &[50, 51, 52, 53]); // entirely prefill
    }

    #[test]
    fn merge_in_json_splices_matching_choices() {
        let l = 2;
        let k = 2;
        let prefill_re = b64(&make_npy_u8(2, l, k, 100));
        let decode_re = b64(&make_npy_u8(5, l, k, 0));
        let prefill = serde_json::json!({"choices": [{"index": 0, "routed_experts": prefill_re}]});
        let mut decode =
            serde_json::json!({"choices": [{"index": 0, "routed_experts": decode_re}]});
        assert!(merge_routed_experts_in_json(&prefill, &mut decode));

        let merged_b64 = decode["choices"][0]["routed_experts"].as_str().unwrap();
        let merged = STANDARD.decode(merged_b64).unwrap();
        let h = parse_npy_header(&merged);
        assert_eq!(h.shape, vec![5, l, k]);
        assert_eq!(merged[h.data_offset], 100);
    }

    #[test]
    fn merge_no_op_when_absent() {
        let prefill = serde_json::json!({"choices": [{"index": 0}]});
        let mut decode = serde_json::json!({"choices": [{"index": 0}]});
        assert!(!merge_routed_experts_in_json(&prefill, &mut decode));
    }

    #[test]
    fn has_routed_experts_detects() {
        assert!(has_routed_experts(
            &serde_json::json!({"choices": [{"routed_experts": "abc"}]})
        ));
        assert!(!has_routed_experts(
            &serde_json::json!({"choices": [{"routed_experts": null}]})
        ));
        assert!(!has_routed_experts(&serde_json::json!({"choices": []})));
    }
}
