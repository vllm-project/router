//! Engine capacity and pressure, discovered from the engine's own `/metrics`.
//!
//! Admission needs a ceiling per engine, and there are two independent sources: `vllm:cache_config_info` carries `num_gpu_blocks` and `block_size`, so capacity in tokens is their product; and `vllm:kv_cache_usage_perc`, `vllm:num_requests_running` and `vllm:num_requests_waiting_by_reason{reason="capacity"}` describe pressure, which every recent build publishes even when it publishes no absolute capacity. `max_num_seqs` is a scheduler setting rather than a `CacheConfig` field, so it is read only when an engine happens to publish it.
//!
//! Both are parsed per engine. A pool that serves several DP ranks reports one series per rank, so samples are keyed by the `engine` label rather than merged: overwriting them would let a full rank pass, and merging would let one rank's pressure exclude its siblings.

use std::collections::BTreeMap;

use crate::protocols::prometheus_text::for_each_sample;

/// Share of an engine's KV pool a request may be admitted into.
pub const DEFAULT_MAX_KV_USAGE: f64 = 0.90;

/// What one engine reports about its capacity and its pressure.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EngineSample {
    /// Router-visible DP rank this sample belongs to (`engine` label, 0 when absent).
    pub engine: u32,
    /// Total KV in tokens, from `num_gpu_blocks x block_size`.
    pub capacity_tokens: Option<u64>,
    /// Slot limit, only when the engine publishes one.
    pub max_sequences: Option<u64>,
    /// KV pool usage as a fraction.
    pub kv_usage: Option<f64>,
    /// Requests queued specifically because the engine ran out of KV capacity.
    pub waiting_capacity: u64,
    /// Requests in model-execution batches.
    pub running: u64,
}

impl EngineSample {
    /// Whether this rank should stop receiving new Decode work. A queue that already names capacity as its reason is stronger evidence than the usage gauge: the engine has decided it cannot place another sequence.
    pub fn is_saturated(&self, max_usage: f64) -> bool {
        self.waiting_capacity > 0 || self.kv_usage.is_some_and(|usage| usage >= max_usage)
    }

    /// Whether one more request of `tokens` fits under this rank's capacity. `None` means the engine published no capacity, so the ceiling cannot be evaluated and the caller must not guess.
    pub fn fits_tokens(&self, tokens: u64, watermark_percent: u64) -> Option<bool> {
        self.capacity_tokens.map(|capacity| {
            tokens.saturating_mul(100) <= capacity.saturating_mul(watermark_percent)
        })
    }

    /// Whether `sequences` more slots fit. Integer counts, so the comparison is exact: a percentage watermark would make a single-slot engine never eligible.
    pub fn fits_sequences(&self, sequences: u64) -> Option<bool> {
        self.max_sequences.map(|slots| sequences <= slots)
    }
}

/// Parse one `/metrics` body into per-engine samples.
pub fn parse_metrics(text: &str) -> BTreeMap<u32, EngineSample> {
    let mut engines: BTreeMap<u32, EngineSample> = BTreeMap::new();
    for_each_sample(text, |name, labels, value| {
        let engine = labels
            .get("engine")
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0);
        let sample = engines.entry(engine).or_insert(EngineSample {
            engine,
            ..EngineSample::default()
        });
        match name {
            "vllm:cache_config_info" => {
                let blocks = labels
                    .get("num_gpu_blocks")
                    .and_then(|value| value.parse::<u64>().ok())
                    .filter(|blocks| *blocks > 0);
                let unit = labels
                    .get("block_size")
                    .and_then(|value| value.parse::<u64>().ok())
                    .filter(|size| *size > 0);
                sample.capacity_tokens = blocks
                    .zip(unit)
                    .map(|(blocks, unit)| blocks.saturating_mul(unit));
                sample.max_sequences = labels
                    .get("max_num_seqs")
                    .and_then(|value| value.parse::<u64>().ok())
                    .filter(|slots| *slots > 0);
            }
            "vllm:kv_cache_usage_perc" => {
                sample.kv_usage = value.parse::<f64>().ok().map(|usage| usage.clamp(0.0, 1.0));
            }
            "vllm:num_requests_running" => {
                sample.running = value
                    .parse::<f64>()
                    .map_or(0, |parsed| parsed.max(0.0) as u64);
            }
            // Only the capacity reason means the engine could not place the request;
            // `deferred` is the ordinary wait for a batch slot.
            "vllm:num_requests_waiting_by_reason"
                if labels.get("reason").map(String::as_str) == Some("capacity") =>
            {
                sample.waiting_capacity = value
                    .parse::<f64>()
                    .map_or(0, |parsed| parsed.max(0.0) as u64);
            }
            _ => {}
        }
    });
    engines
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    const TWO_RANKS: &str = r#"
# HELP vllm:cache_config_info engine cache config
vllm:cache_config_info{engine="0",block_size="16",num_gpu_blocks="1000"} 1
vllm:cache_config_info{engine="1",block_size="16",num_gpu_blocks="2000"} 1
vllm:kv_cache_usage_perc{engine="0"} 0.42
vllm:kv_cache_usage_perc{engine="1"} 0.97
vllm:num_requests_running{engine="0"} 2
vllm:num_requests_running{engine="1"} 7
vllm:num_requests_waiting_by_reason{engine="1",reason="capacity"} 1
vllm:num_requests_waiting_by_reason{engine="1",reason="deferred"} 3
"#;

    #[test]
    fn keeps_each_rank_separate() {
        let engines = parse_metrics(TWO_RANKS);
        assert_eq!(engines.len(), 2);
        assert_eq!(engines[&0].capacity_tokens, Some(16_000));
        assert_eq!(engines[&1].capacity_tokens, Some(32_000));
        assert_eq!(engines[&0].kv_usage, Some(0.42));
        assert_eq!(engines[&1].kv_usage, Some(0.97));
        assert_eq!(engines[&1].running, 7);
        assert_eq!(engines[&1].waiting_capacity, 1);
        assert!(!engines[&0].is_saturated(DEFAULT_MAX_KV_USAGE));
        assert!(engines[&1].is_saturated(DEFAULT_MAX_KV_USAGE));
    }

    #[test]
    fn missing_capacity_is_not_a_guess() {
        let engines = parse_metrics("vllm:kv_cache_usage_perc{engine=\"0\"} 0.10\n");
        let sample = engines[&0];
        assert_eq!(sample.capacity_tokens, None);
        assert_eq!(sample.fits_tokens(1, 95), None);
        assert_eq!(sample.max_sequences, None);
        assert_eq!(sample.fits_sequences(1), None);
    }

    #[test]
    fn slot_ceiling_is_exact() {
        let mut sample = EngineSample {
            max_sequences: Some(1),
            ..EngineSample::default()
        };
        assert_eq!(sample.fits_sequences(1), Some(true));
        assert_eq!(sample.fits_sequences(2), Some(false));
        sample.max_sequences = Some(4);
        assert_eq!(sample.fits_sequences(4), Some(true));
    }

    #[test]
    fn capacity_ceiling_uses_the_watermark() {
        let sample = EngineSample {
            capacity_tokens: Some(1_000),
            ..EngineSample::default()
        };
        assert_eq!(sample.fits_tokens(900, 95), Some(true));
        assert_eq!(sample.fits_tokens(960, 95), Some(false));
    }

    #[test]
    fn queueing_by_capacity_saturates_without_a_usage_gauge() {
        let sample = EngineSample {
            waiting_capacity: 1,
            ..EngineSample::default()
        };
        assert!(sample.is_saturated(DEFAULT_MAX_KV_USAGE));
    }
}
