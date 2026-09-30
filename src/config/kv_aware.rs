use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Opt-in contract for the first static, single-model KV-aware deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KvAwareConfig {
    /// Must match every worker's block size and sha256_cbor configuration.
    pub block_size: usize,
    pub hash_seed: u32,
    /// Local tokenizer.json (or directory), with neighboring Qwen3 Dense
    /// config.json and tokenizer_config.json matching the workers.
    pub tokenizer_path: String,
    /// Accepted request model alias; omitted request.model is also accepted.
    pub model: String,
    pub topic: String,
    pub default_port: u16,
    pub worker_endpoints: HashMap<String, String>,
    /// Hard bound on (worker, block) ownership records.
    pub index_max_entries: usize,
}

impl Default for KvAwareConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            hash_seed: 0,
            tokenizer_path: String::new(),
            model: "Qwen/Qwen3-0.6B".into(),
            topic: String::new(),
            default_port: 5557,
            worker_endpoints: HashMap::new(),
            index_max_entries: 100_000,
        }
    }
}
