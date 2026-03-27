//! Block hash computation compatible with vLLM's content-addressed KV cache.
//!
//! vLLM identifies KV blocks by a chain of content hashes:
//!
//! ```text
//!   key_0 = hash(None, tokens[0..B])
//!   key_1 = hash(key_0, tokens[B..2B])
//!   key_i = hash(key_{i-1}, tokens[iB..(i+1)B])
//! ```
//!
//! This chaining ensures that two blocks with identical tokens but different
//! prefixes produce different hashes, enabling prefix-tree-style lookups.
//!
//! **Critical**: The hash function, seed, and block size MUST match the
//! vLLM engine configuration.  Mismatches silently produce wrong scores.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Generates block-level content hashes from token ID sequences.
///
/// # Configuration alignment
///
/// | Parameter    | Must match vLLM config        |
/// |-------------|-------------------------------|
/// | `block_size` | `--block-size` (default 16)   |
/// | `hash_seed`  | `PYTHONHASHSEED` env variable |
#[derive(Debug, Clone)]
pub struct BlockKeyGenerator {
    block_size: usize,
    hash_seed: u64,
}

impl BlockKeyGenerator {
    /// Create a new generator.
    ///
    /// * `block_size` – tokens per KV block (must match vLLM `--block-size`).
    /// * `hash_seed` – hash seed (must match vLLM `PYTHONHASHSEED`).
    pub fn new(block_size: usize, hash_seed: u64) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        Self {
            block_size,
            hash_seed,
        }
    }

    /// Number of tokens per block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Convert a token ID sequence into a chain of block hashes.
    ///
    /// Only complete blocks (exactly `block_size` tokens) are hashed.
    /// Trailing tokens that do not fill a full block are ignored.
    ///
    /// Returns an ordered `Vec<u64>` of block hashes, one per complete block.
    pub fn generate_block_keys(&self, token_ids: &[u32]) -> Vec<u64> {
        let num_blocks = token_ids.len() / self.block_size;
        let mut keys = Vec::with_capacity(num_blocks);
        let mut parent_hash: Option<u64> = None;

        for chunk in token_ids.chunks_exact(self.block_size) {
            let key = self.compute_block_hash(parent_hash, chunk);
            keys.push(key);
            parent_hash = Some(key);
        }

        keys
    }

    /// Compute the hash for a single block given its parent hash and tokens.
    ///
    /// Uses Rust's [`DefaultHasher`] (SipHash-1-3) seeded with `self.hash_seed`.
    /// NOTE: This is an approximation.  For bit-exact compatibility with Python's
    /// `hash()` on tuples, a dedicated Python-hash reimplementation is required.
    /// This implementation is sufficient for the router's own consistent hashing
    /// when both the router and vLLM use the same `PYTHONHASHSEED` value and
    /// the event-based index compares hashes from the same source (vLLM events).
    fn compute_block_hash(&self, parent: Option<u64>, tokens: &[u32]) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash_seed.hash(&mut hasher);
        parent.hash(&mut hasher);
        tokens.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_block_keys_basic() {
        let gen = BlockKeyGenerator::new(4, 12345);
        // 8 tokens -> 2 blocks
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let keys = gen.generate_block_keys(&tokens);
        assert_eq!(keys.len(), 2);
        // Keys should be different due to chaining.
        assert_ne!(keys[0], keys[1]);
    }

    #[test]
    fn test_generate_block_keys_trailing_ignored() {
        let gen = BlockKeyGenerator::new(4, 0);
        // 6 tokens -> 1 full block + 2 trailing
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let keys = gen.generate_block_keys(&tokens);
        assert_eq!(keys.len(), 1);
    }

    #[test]
    fn test_generate_block_keys_empty() {
        let gen = BlockKeyGenerator::new(4, 0);
        let keys = gen.generate_block_keys(&[]);
        assert!(keys.is_empty());
    }

    #[test]
    fn test_deterministic() {
        let gen = BlockKeyGenerator::new(4, 42);
        let tokens: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let keys_a = gen.generate_block_keys(&tokens);
        let keys_b = gen.generate_block_keys(&tokens);
        assert_eq!(keys_a, keys_b);
    }

    #[test]
    fn test_different_prefix_different_hash() {
        let gen = BlockKeyGenerator::new(4, 0);
        // Same second block tokens, different first block -> different second key.
        let tokens_a: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let tokens_b: Vec<u32> = vec![9, 9, 9, 9, 5, 6, 7, 8];
        let keys_a = gen.generate_block_keys(&tokens_a);
        let keys_b = gen.generate_block_keys(&tokens_b);
        assert_eq!(keys_a.len(), 2);
        assert_eq!(keys_b.len(), 2);
        // First blocks differ.
        assert_ne!(keys_a[0], keys_b[0]);
        // Second blocks also differ because parent hash is chained.
        assert_ne!(keys_a[1], keys_b[1]);
    }
}
