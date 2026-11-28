//! Engine-compatible `sha256_cbor` prefix chains for token and named-LoRA inputs.
//!
//! Use the verified engine root or its exact resolved seed. The numeric seed
//! constructor validates the Python range; `from_seed` also supports vLLM's
//! verified fixed default. Unsupported extra-key contracts remain uncredited.

use sha2::{Digest, Sha256};

/// Full 32-byte, engine-compatible `sha256_cbor` sequence digest.
///
/// The decoder preserves the publisher's bytes. This module generates the
/// corresponding request chain for the supported `sha256_cbor` contracts.
pub type SequenceHash = [u8; 32];

/// Worker-rendered identity for a single image in a chat prompt.
#[derive(Debug, Clone)]
pub(crate) struct ImagePrefix {
    hash: String,
    offset: usize,
    end: usize,
}

impl ImagePrefix {
    pub(crate) fn new(hash: &str, offset: usize, length: usize, tokens: usize) -> Option<Self> {
        let end = offset.checked_add(length)?;
        (length != 0 && end <= tokens && is_image_hash(hash)).then(|| Self {
            hash: hash.to_owned(),
            offset,
            end,
        })
    }

    fn key(&self, start: usize, end: usize) -> Option<(&str, i64)> {
        (start < self.end && self.offset < end)
            .then_some((self.hash.as_str(), self.offset as i64 - start as i64))
    }
}

pub(crate) fn is_image_hash(hash: &str) -> bool {
    let digest = hash.strip_prefix("content:").unwrap_or(hash);
    digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockKeyGenerator {
    block_size: usize,
    none_hash: SequenceHash,
    lookahead_tokens: usize,
}

impl BlockKeyGenerator {
    /// Require a positive block size and a canonical numeric Python hash seed.
    /// Panics if the block size is zero or the seed exceeds `u32::MAX`.
    pub fn new(block_size: usize, hash_seed: u64) -> Self {
        assert!(block_size > 0, "block_size must be greater than zero");
        assert!(
            hash_seed <= u64::from(u32::MAX),
            "hash_seed must fit PYTHONHASHSEED"
        );
        let seed = hash_seed.to_string();
        Self::from_seed(block_size, &seed)
    }

    /// Derive the root from vLLM's resolved seed string, including the fixed
    /// `vllm-none-hash` seed when the worker uses it.
    pub fn from_seed(block_size: usize, seed: &str) -> Self {
        assert!(block_size > 0 && !seed.is_empty(), "invalid hash contract");
        let mut encoded = Vec::with_capacity(seed.len() + 9);
        encode_text(&mut encoded, seed);
        Self {
            block_size,
            none_hash: digest(&encoded),
            lookahead_tokens: 0,
        }
    }

    pub(crate) fn root_hash(&self) -> SequenceHash {
        self.none_hash
    }

    /// Use the engine's authenticated root digest without exposing its seed.
    pub fn from_root_hash(block_size: usize, root_hash: SequenceHash) -> Self {
        assert!(block_size > 0, "block_size must be greater than zero");
        Self {
            block_size,
            none_hash: root_hash,
            lookahead_tokens: 0,
        }
    }

    /// Apply the verified engine's future-token hash contract.
    pub(crate) fn with_lookahead(mut self, tokens: u32) -> Self {
        self.lookahead_tokens = tokens as usize;
        self
    }

    pub(crate) fn lookahead_tokens(&self) -> usize {
        self.lookahead_tokens
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Return one key per complete block followed by its required lookahead.
    pub fn generate_block_keys(&self, token_ids: &[u32]) -> Vec<SequenceHash> {
        self.keys_after_with_extra(self.none_hash, token_ids, None, None)
    }

    pub(crate) fn generate_block_keys_for_image(
        &self,
        token_ids: &[u32],
        image: &ImagePrefix,
    ) -> Vec<SequenceHash> {
        if self.lookahead_tokens != 0 {
            return Vec::new();
        }
        self.keys_after_with_extra(self.none_hash, token_ids, None, Some(image))
    }

    /// Recompute a plain first block with the same encoding as request lookup.
    pub(crate) fn plain_root_block(&self, token_ids: &[u32]) -> Option<SequenceHash> {
        self.root_block(token_ids, &[])
    }

    pub(crate) fn root_block(
        &self,
        token_ids: &[u32],
        image_keys: &[(&str, i64)],
    ) -> Option<SequenceHash> {
        if self.lookahead_tokens != 0 {
            return None;
        }
        let block = token_ids.get(..self.block_size)?;
        Some(hash_block(
            &mut Vec::new(),
            &self.none_hash,
            block,
            None,
            image_keys,
            &[],
        ))
    }

    /// The same chain for one LoRA adapter. vLLM hashes each adapter block with
    /// `[["lora", name, path]]` as extra keys, where `path` is the string the
    /// engine was launched with, so this reproduces the adapter's chain instead
    /// of the base chain.
    pub fn generate_block_keys_for_adapter(
        &self,
        token_ids: &[u32],
        name: &str,
        path: &str,
    ) -> Vec<SequenceHash> {
        self.keys_after_with_extra(self.none_hash, token_ids, Some((name, path)), None)
    }

    fn keys_after_with_extra(
        &self,
        parent: SequenceHash,
        token_ids: &[u32],
        adapter: Option<(&str, &str)>,
        image: Option<&ImagePrefix>,
    ) -> Vec<SequenceHash> {
        let mut parent = parent;
        let mut encoded = Vec::new();
        let hashable_tokens = token_ids.len().saturating_sub(self.lookahead_tokens);
        token_ids[..hashable_tokens]
            .chunks_exact(self.block_size)
            .enumerate()
            .map(|(index, tokens)| {
                let end = (index + 1) * self.block_size;
                let lookahead = &token_ids[end..end + self.lookahead_tokens];
                let image_key = image.and_then(|image| image.key(end - self.block_size, end));
                parent = hash_block(
                    &mut encoded,
                    &parent,
                    tokens,
                    adapter,
                    image_key.as_slice(),
                    lookahead,
                );
                parent
            })
            .collect()
    }
}

fn hash_block(
    encoded: &mut Vec<u8>,
    parent: &SequenceHash,
    token_ids: &[u32],
    adapter: Option<(&str, &str)>,
    image_keys: &[(&str, i64)],
    lookahead: &[u32],
) -> SequenceHash {
    encoded.clear();
    encoded.reserve(40 + token_ids.len() * 3);

    // Canonical CBOR for [parent_hash, token_ids, extra_keys].
    encode_major(encoded, 4, 3);
    encode_major(encoded, 2, parent.len() as u64);
    encoded.extend_from_slice(parent);
    encode_major(encoded, 4, token_ids.len() as u64);
    for &token_id in token_ids {
        encode_major(encoded, 0, u64::from(token_id));
    }
    if !lookahead.is_empty() {
        // vLLM wraps the original extra keys and appends the lookahead IDs.
        encode_major(encoded, 4, 1 + lookahead.len() as u64);
    }
    if adapter.is_none() && image_keys.is_empty() {
        encoded.push(0xf6);
    } else {
        encode_major(
            encoded,
            4,
            image_keys.len() as u64 + u64::from(adapter.is_some()),
        );
        if let Some((name, path)) = adapter {
            encode_major(encoded, 4, 3);
            encode_text(encoded, "lora");
            encode_text(encoded, name);
            encode_text(encoded, path);
        }
        for &(hash, offset) in image_keys {
            encode_major(encoded, 4, 3);
            encode_text(encoded, "mm");
            encode_text(encoded, hash);
            if offset < 0 {
                encode_major(encoded, 1, (!offset) as u64);
            } else {
                encode_major(encoded, 0, offset as u64);
            }
        }
    }
    for &token_id in lookahead {
        encode_major(encoded, 0, u64::from(token_id));
    }

    digest(encoded)
}

fn encode_text(output: &mut Vec<u8>, value: &str) {
    encode_major(output, 3, value.len() as u64);
    output.extend_from_slice(value.as_bytes());
}

fn encode_major(output: &mut Vec<u8>, major: u8, value: u64) {
    debug_assert!(major <= 7);
    let prefix = major << 5;
    match value {
        0..=23 => output.push(prefix | value as u8),
        24..=0xff => {
            output.push(prefix | 24);
            output.push(value as u8);
        }
        0x100..=0xffff => {
            output.push(prefix | 25);
            output.extend_from_slice(&(value as u16).to_be_bytes());
        }
        0x1_0000..=0xffff_ffff => {
            output.push(prefix | 26);
            output.extend_from_slice(&(value as u32).to_be_bytes());
        }
        _ => {
            output.push(prefix | 27);
            output.extend_from_slice(&value.to_be_bytes());
        }
    }
}

fn digest(bytes: &[u8]) -> SequenceHash {
    Sha256::digest(bytes).into()
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn from_hex(value: &str) -> SequenceHash {
        assert_eq!(value.len(), 64);
        let mut output = [0_u8; 32];
        for (index, byte) in output.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).unwrap();
        }
        output
    }

    #[test]
    fn default_seed_matches_vllm_033() {
        assert_eq!(
            BlockKeyGenerator::from_seed(16, "vllm-none-hash").root_hash(),
            from_hex("9bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef0")
        );
    }

    #[test]
    fn sha256_cbor_matches_independent_vllm_goldens() {
        let generator = BlockKeyGenerator::new(128, 0);
        assert_eq!(
            generator.root_hash(),
            from_hex("4e1195df020de59e0d65a33a4279f1183e7ae4e5d980e309f8b55adff2e61c3e")
        );

        let tokens: Vec<u32> = (0..256).collect();
        assert_eq!(
            generator.generate_block_keys(&tokens),
            vec![
                from_hex("41b04e727e2efe6a5e35a707a47e3e11872b3215e1056d068d9bde11d7e22ce7"),
                from_hex("ad8f0385e9091ed10027f18191a8d1e7935ed4a2bf1768f047380def604d3987"),
            ]
        );
    }

    #[test]
    fn adapter_chain_matches_vllm_sha256_cbor() {
        // Generated with vLLM 0.33's LoRA extra keys and hash_block_tokens.
        let generator = BlockKeyGenerator::from_root_hash(4, std::array::from_fn(|i| i as u8));
        let tokens = [1, 23, 24, 256, 2, 255, 65536, 3];
        assert_eq!(
            generator.generate_block_keys_for_adapter(&tokens, "adapter-a", "/models/适配"),
            [
                from_hex("d7660db4552cadaf15169e06db24842cebb24c34973a48ac614a9b14801cab54"),
                from_hex("67998e38bee1644dfa2edf152c70f8d512209d8b4802ae1e704c80136b5a89eb"),
            ]
        );
    }

    #[test]
    fn lookahead_chains_match_vllm_sha256_cbor() {
        // Python hashlib over canonical CBOR for the vLLM 0.33 hash layout.
        let generator = BlockKeyGenerator::from_root_hash(2, std::array::from_fn(|i| i as u8));
        let lookahead = generator.clone().with_lookahead(1);
        let image = ImagePrefix::new(
            crate::kv_index::source::test_support::TEST_IMAGE_HASH,
            0,
            2,
            5,
        )
        .unwrap();
        assert!(image.key(2, 4).is_none());
        assert!(lookahead
            .generate_block_keys_for_image(&[1, 2, 3, 4, 5], &image)
            .is_empty());
        let hashes = lookahead.generate_block_keys(&[1, 2, 3, 4, 5]);
        assert_eq!(
            hashes,
            [
                from_hex("965f0a94f2a1c261565764dd67239779cb02a723253fcd6168feb4a6f1a5e75d"),
                from_hex("11ce81d6c93598c4a80f87fde74bc9a75235fc0bfc158f50a63c05969c8934c9"),
            ]
        );
        assert_eq!(lookahead.generate_block_keys(&[1, 2, 3, 4]), &hashes[..1]);
        let changed = lookahead.generate_block_keys(&[1, 2, 3, 4, 6]);
        assert_eq!(changed[0], hashes[0]);
        assert_ne!(changed[1], hashes[1]);
        assert_eq!(
            generator.with_lookahead(2).generate_block_keys_for_adapter(
                &[1, 2, 3, 4],
                "a",
                "/lora"
            ),
            [from_hex(
                "64def2c18e799b00f00514594e50aee78e32625338d10b3ff5e92445e8fc03b9"
            )]
        );
    }

    #[test]
    fn sha256_cbor_matches_python_integer_boundaries_and_seeds() {
        #[derive(serde::Deserialize)]
        struct OracleCase {
            seed: u64,
            none_hash: String,
            block_hashes: Vec<String>,
        }
        #[derive(serde::Deserialize)]
        struct Oracle {
            algorithm: String,
            cbor2_version: String,
            block_size: usize,
            token_ids: Vec<u32>,
            trailing_partial_tokens: usize,
            cases: Vec<OracleCase>,
        }

        // Python hashlib + cbor2 5.6.5 oracle, independent of this encoder.
        // Synthetic inputs: Apache-2.0; serialization cases, not inference
        // requests. Contract: vLLM v0.29.0 vllm/utils/hashing.py
        // (sha256_cbor) and vllm/v1/core/kv_cache_utils.py
        // (init_none_hash / hash_block_tokens). No engine run is implied.
        // Canonical-CBOR inputs are retained for representation review.
        let oracle: Oracle = serde_json::from_str(include_str!(
            "../../tests/rust/fixtures/kv_cbor/python_oracle.json"
        ))
        .unwrap();
        assert_eq!(oracle.algorithm, "sha256_cbor");
        assert_eq!(oracle.cbor2_version, "5.6.5");
        assert_eq!(oracle.block_size, 16);
        assert_eq!(oracle.token_ids.len(), 35);
        assert_eq!(oracle.trailing_partial_tokens, 3);
        assert_eq!(
            oracle
                .cases
                .iter()
                .map(|case| case.seed)
                .collect::<Vec<_>>(),
            [0, 42]
        );
        for case in oracle.cases {
            let generator = BlockKeyGenerator::new(oracle.block_size, case.seed);
            assert_eq!(generator.root_hash(), from_hex(&case.none_hash));
            let expected: Vec<_> = case
                .block_hashes
                .iter()
                .map(|hash| from_hex(hash))
                .collect();
            assert_eq!(expected.len(), 2);
            assert_eq!(
                generator.generate_block_keys(&oracle.token_ids),
                expected,
                "canonical-CBOR boundaries and hash chain for seed {}",
                case.seed
            );
            assert_eq!(
                generator.generate_block_keys(&oracle.token_ids[..32]),
                expected,
                "trailing partial tokens must not alter complete block hashes"
            );
        }
    }

    #[test]
    fn identical_suffix_tokens_with_different_prefixes_have_distinct_sequence_hashes() {
        let generator = BlockKeyGenerator::new(2, 0);
        let first = generator.generate_block_keys(&[1, 2, 5, 6]);
        let second = generator.generate_block_keys(&[3, 4, 5, 6]);
        assert_ne!(first[1], second[1]);
    }

    #[test]
    fn maximum_python_seed_matches_canonical_cbor() {
        // Independent SHA-256 of CBOR text 0x6a || "4294967295".
        assert_eq!(
            BlockKeyGenerator::new(16, u64::from(u32::MAX)).root_hash(),
            from_hex("177f280a5695322a18f16c96a26dc99d9c03f905940103dfe24a9c646fe446a8")
        );
    }

    #[test]
    #[should_panic(expected = "hash_seed must fit PYTHONHASHSEED")]
    fn seed_above_python_range_is_rejected() {
        BlockKeyGenerator::new(16, u64::from(u32::MAX) + 1);
    }

    #[test]
    fn trailing_partial_block_is_not_hashed() {
        let generator = BlockKeyGenerator::new(16, 0);
        assert!(generator.generate_block_keys(&[1; 15]).is_empty());
        assert_eq!(generator.generate_block_keys(&[1; 31]).len(), 1);
    }
}
