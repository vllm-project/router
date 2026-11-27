//! Engine cache-layout metadata carried by vLLM's `/metrics` endpoint.
//!
//! `vllm:cache_config_info` is an info-style gauge: one series per engine whose
//! labels mirror that engine's `CacheConfig`. Unlike `/server_info` it is not
//! gated behind `VLLM_SERVER_DEV_MODE`, so it is the metadata channel a stock
//! vLLM worker always exposes.

use std::collections::BTreeMap;

use crate::protocols::prometheus_text::for_each_sample;

/// The metric family that carries one engine's `CacheConfig`.
pub(crate) const CACHE_CONFIG_INFO: &str = "vllm:cache_config_info";

pub(super) type EngineCacheConfigs = BTreeMap<u32, EngineCacheConfig>;

pub(super) fn reported_prefix_hash<'a>(
    configs: &'a EngineCacheConfigs,
    worker_url: &str,
    rank: u32,
) -> Option<&'a str> {
    configs
        .get(&rank)
        .or_else(|| {
            // A pinned worker can expose a single engine with a local rank.
            (configs.len() == 1 && super::split_worker_rank(worker_url).1 == Some(rank))
                .then(|| configs.values().next())
                .flatten()
        })?
        .prefix_caching_hash_algo
        .as_deref()
}

/// The parts of one engine's `CacheConfig` the KV index needs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct EngineCacheConfig {
    /// Tokens per cache block.
    pub(crate) block_size: Option<u32>,
    /// Attention block size, which a hybrid layout may report separately.
    pub(crate) effective_attention_block_size: Option<u32>,
    /// Resolved hash granularity. Older engines may leave it unset.
    pub(crate) hash_block_size: Option<u32>,
    pub(crate) prefix_caching_hash_algo: Option<String>,
    /// Engine-declared cache layout, once vLLM populates it.
    pub(crate) kv_cache_layout: Option<String>,
}

impl EngineCacheConfig {
    /// Effective attention unit, or the configured unit when absent.
    pub(crate) fn hash_unit(&self) -> Option<u32> {
        super::effective_block_size(self.effective_attention_block_size, self.block_size)
    }
}

/// The hash unit every engine agrees on, or the sizes that conflict.
///
/// A scrape that mixes layouts cannot be trusted for indexing: crediting
/// blocks at the wrong granularity overstates cache hits, so the caller
/// refuses the worker instead of picking one engine's size.
pub(crate) fn shared_hash_unit(
    engines: &BTreeMap<u32, EngineCacheConfig>,
) -> Result<Option<u32>, Vec<u32>> {
    let units: std::collections::BTreeSet<u32> = engines
        .values()
        .filter_map(EngineCacheConfig::hash_unit)
        .collect();
    match units.len() {
        0 => Ok(None),
        1 => Ok(units.into_iter().next()),
        _ => Err(units.into_iter().collect()),
    }
}

/// Engine configs in rank order, with the number of samples skipped for an invalid rank.
pub(super) fn parse_cache_config_info(text: &str) -> (EngineCacheConfigs, usize) {
    let mut engines = EngineCacheConfigs::new();
    let mut malformed_ranks = 0;
    for_each_sample(text, |name, labels, _value| {
        if name != CACHE_CONFIG_INFO {
            return;
        }
        let engine = match labels.get("engine").filter(|value| !is_unset(value)) {
            Some(value) => {
                let Ok(rank) = value.parse::<u32>() else {
                    malformed_ranks += 1;
                    return;
                };
                rank
            }
            None => 0,
        };
        let entry = engines.entry(engine).or_default();
        entry.block_size = labels
            .get("block_size")
            .and_then(|value| value.parse().ok());
        entry.effective_attention_block_size = labels
            .get("effective_attention_block_size")
            .and_then(|value| value.parse().ok());
        entry.hash_block_size = labels
            .get("hash_block_size")
            .and_then(|value| value.parse().ok())
            .filter(|size| *size > 0);
        entry.prefix_caching_hash_algo = labels
            .get("prefix_caching_hash_algo")
            .filter(|algorithm| !is_unset(algorithm))
            .cloned();
        entry.kv_cache_layout = labels
            .get("kv_cache_layout")
            .filter(|layout| !is_unset(layout))
            .cloned();
    });
    (engines, malformed_ranks)
}

/// vLLM serializes an unset `CacheConfig` field as the string `None`.
fn is_unset(value: &str) -> bool {
    let value = value.trim();
    value.is_empty() || value.eq_ignore_ascii_case("none")
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    const METRICS: &str = r#"
    # HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
    # TYPE vllm:cache_config_info gauge
    vllm:cache_config_info{block_size="16",effective_attention_block_size="16",cache_dtype="auto",num_gpu_blocks="12989",num_cpu_blocks="None",enable_prefix_caching="True",prefix_caching_hash_algo="sha256_cbor",kv_cache_layout="None",hash_block_size="None",prefix_match_unit="None",cache_hit_alignment_tokens="None",swa_bounded_replay="True",mamba_block_size="None",mamba_cache_mode="none",engine="0"} 1.0
    vllm:num_requests_running{engine="0",model_name="m"} 2
    "#;

    #[test]
    fn parses_one_engine() {
        let (engines, malformed_ranks) = parse_cache_config_info(METRICS);
        assert_eq!(malformed_ranks, 0);
        assert_eq!(engines.len(), 1);
        let engine = &engines[&0];
        assert_eq!(engine.block_size, Some(16));
        assert_eq!(engine.effective_attention_block_size, Some(16));
        assert_eq!(engine.kv_cache_layout, None);
        assert_eq!(
            engine.prefix_caching_hash_algo.as_deref(),
            Some("sha256_cbor")
        );
        assert_eq!(engine.hash_unit(), Some(16));
    }

    #[test]
    fn parses_each_dp_engine_separately() {
        let metrics = r#"
    vllm:cache_config_info{block_size="16",engine="1"} 1.0
    vllm:cache_config_info{block_size="32",engine="0"} 1.0
    "#;
        let (engines, _) = parse_cache_config_info(metrics);
        assert_eq!(engines.len(), 2);
        assert_eq!(engines[&0].block_size, Some(32));
        assert_eq!(engines[&1].block_size, Some(16));
        assert_eq!(engines[&0].hash_unit(), Some(32));
    }

    #[test]
    fn missing_or_unset_engine_label_defaults_to_rank_zero() {
        let expected = parse_cache_config_info(METRICS);
        for label in ["", r#",engine="""#, r#",engine="  ""#, r#",engine="None""#] {
            let metrics = METRICS.replacen(r#",engine="0""#, label, 1);
            assert_eq!(parse_cache_config_info(&metrics), expected, "{label:?}");
        }
    }

    #[test]
    fn malformed_engine_labels_do_not_overwrite_rank_zero() {
        let malformed = r#"vllm:cache_config_info{block_size="32",engine="unknown",prefix_caching_hash_algo="sha256"} 1.0"#;
        let (valid, _) = parse_cache_config_info(METRICS);
        assert_eq!(
            parse_cache_config_info(malformed),
            (EngineCacheConfigs::new(), 1)
        );
        assert_eq!(
            parse_cache_config_info(&format!("{METRICS}\n{malformed}")),
            (valid, 1)
        );
    }

    #[test]
    fn reports_declared_hybrid_layout() {
        let metrics = r#"
    vllm:cache_config_info{block_size="64",hash_block_size="4",kv_cache_layout="hybrid-v1",engine="0"} 1.0
    "#;
        let (engines, _) = parse_cache_config_info(metrics);
        let engine = &engines[&0];
        assert_eq!(engine.kv_cache_layout.as_deref(), Some("hybrid-v1"));
    }

    #[test]
    fn refuses_mixed_block_sizes() {
        for metrics in [
            r#"
vllm:cache_config_info{block_size="16",engine="0"} 1.0
vllm:cache_config_info{block_size="32",engine="1"} 1.0
"#,
            r#"
vllm:cache_config_info{block_size="8",effective_attention_block_size="16",engine="0"} 1.0
vllm:cache_config_info{block_size="8",effective_attention_block_size="32",engine="1"} 1.0
"#,
        ] {
            assert_eq!(
                shared_hash_unit(&parse_cache_config_info(metrics).0),
                Err(vec![16, 32])
            );
        }
        let (agreed, _) =
            parse_cache_config_info(r#"vllm:cache_config_info{block_size="16",engine="0"} 1.0"#);
        assert_eq!(shared_hash_unit(&agreed), Ok(Some(16)));
        assert_eq!(shared_hash_unit(&BTreeMap::new()), Ok(None));
    }

    #[test]
    fn missing_family_yields_no_engines() {
        assert_eq!(
            parse_cache_config_info("vllm:num_requests_running{engine=\"0\"} 1\n"),
            (EngineCacheConfigs::new(), 0)
        );
        assert_eq!(parse_cache_config_info(""), (EngineCacheConfigs::new(), 0));
    }
}
