//! Prefix-match scorer for KV-cache-aware worker selection.
//!
//! Given a sequence of block hashes (representing a tokenized prompt), the
//! scorer queries the [`KVBlockIndex`] to determine how many *contiguous*
//! prefix blocks each worker already has cached.  The worker with the
//! longest prefix match can skip the most prefill computation.

use super::index::KVBlockIndex;
use std::collections::HashMap;

/// Scores workers by the length of their longest contiguous prefix match
/// in the global KV block index.
#[derive(Debug)]
pub struct PrefixScorer<'a> {
    index: &'a KVBlockIndex,
}

/// Result of scoring a set of block hashes against the index.
#[derive(Debug, Clone)]
pub struct PrefixScoreResult {
    /// Map from worker URL to the number of contiguously matched prefix blocks.
    pub scores: HashMap<String, usize>,
    /// Total number of block hashes that were queried.
    pub total_blocks: usize,
}

impl PrefixScoreResult {
    /// Returns the worker URL with the highest prefix score, or `None` if
    /// no worker matched any block.
    pub fn best_worker(&self) -> Option<(&str, usize)> {
        self.scores
            .iter()
            .max_by_key(|(_, &score)| score)
            .map(|(url, &score)| (url.as_str(), score))
    }

    /// Returns the prefix match ratio for a specific worker (0.0 to 1.0).
    pub fn match_ratio(&self, worker_url: &str) -> f32 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        let score = self.scores.get(worker_url).copied().unwrap_or(0);
        score as f32 / self.total_blocks as f32
    }

    /// Number of uncached tokens for a given worker, calculated from
    /// the cache miss region.
    pub fn uncached_tokens(&self, worker_url: &str, block_size: usize) -> usize {
        let cached_blocks = self.scores.get(worker_url).copied().unwrap_or(0);
        self.total_blocks
            .saturating_sub(cached_blocks)
            .saturating_mul(block_size)
    }
}

impl<'a> PrefixScorer<'a> {
    /// Create a scorer backed by the given index.
    pub fn new(index: &'a KVBlockIndex) -> Self {
        Self { index }
    }

    /// Score all workers for a given ordered sequence of block hashes.
    ///
    /// The algorithm counts how many leading (prefix) blocks each worker
    /// holds without gaps.  A gap at position `i` means the worker's
    /// score is capped at `i` even if it holds later blocks.
    ///
    /// # Example
    ///
    /// If `block_hashes = [A, B, C, D]` and worker W1 holds `{A, B, D}`,
    /// W1's score is 2 (blocks A and B are contiguous; D is not counted
    /// because C is missing).
    pub fn score(&self, block_hashes: &[u64]) -> PrefixScoreResult {
        let mut scores: HashMap<String, usize> = HashMap::new();
        // Track which workers are still "alive" in the contiguous prefix.
        // Once a worker misses a block, it is removed from this set.
        let mut active_workers: HashMap<String, usize> = HashMap::new();

        for (i, &hash) in block_hashes.iter().enumerate() {
            let workers_holding = self.index.get_workers_for_block(hash);

            if i == 0 {
                // Initialize: all workers holding the first block start with score 1.
                for w in &workers_holding {
                    active_workers.insert(w.clone(), 1);
                    scores.insert(w.clone(), 1);
                }
            } else {
                // For subsequent blocks, only workers that (a) were active and
                // (b) hold this block continue to extend their prefix.
                let mut still_active = HashMap::new();
                for w in &workers_holding {
                    if let Some(&prev_score) = active_workers.get(w) {
                        // This worker had a contiguous match up to the previous block.
                        if prev_score == i {
                            let new_score = i + 1;
                            still_active.insert(w.clone(), new_score);
                            scores.insert(w.clone(), new_score);
                        }
                    } else {
                        // Worker appears for the first time at a non-zero index.
                        // Not a prefix match, so we don't add it.
                    }
                }
                active_workers = still_active;
            }
        }

        PrefixScoreResult {
            scores,
            total_blocks: block_hashes.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_index() -> KVBlockIndex {
        KVBlockIndex::new(10000)
    }

    #[test]
    fn test_empty_blocks() {
        let idx = make_index();
        let scorer = PrefixScorer::new(&idx);
        let result = scorer.score(&[]);
        assert!(result.scores.is_empty());
        assert_eq!(result.total_blocks, 0);
    }

    #[test]
    fn test_single_worker_full_match() {
        let idx = make_index();
        idx.on_block_stored(1, "http://w1:8000");
        idx.on_block_stored(2, "http://w1:8000");
        idx.on_block_stored(3, "http://w1:8000");

        let scorer = PrefixScorer::new(&idx);
        let result = scorer.score(&[1, 2, 3]);

        assert_eq!(result.scores.get("http://w1:8000"), Some(&3));
        assert_eq!(result.total_blocks, 3);

        let (best_url, best_score) = result.best_worker().unwrap();
        assert_eq!(best_url, "http://w1:8000");
        assert_eq!(best_score, 3);
    }

    #[test]
    fn test_gap_breaks_prefix() {
        let idx = make_index();
        // Worker holds blocks 1, 2, 4 (gap at 3).
        idx.on_block_stored(1, "http://w1:8000");
        idx.on_block_stored(2, "http://w1:8000");
        idx.on_block_stored(4, "http://w1:8000");

        let scorer = PrefixScorer::new(&idx);
        let result = scorer.score(&[1, 2, 3, 4]);

        // Score should be 2 (blocks 1, 2 contiguous; gap at 3).
        assert_eq!(result.scores.get("http://w1:8000"), Some(&2));
    }

    #[test]
    fn test_multiple_workers() {
        let idx = make_index();
        // W1: holds all 3 blocks.
        idx.on_block_stored(10, "http://w1:8000");
        idx.on_block_stored(20, "http://w1:8000");
        idx.on_block_stored(30, "http://w1:8000");
        // W2: holds first 2 blocks only.
        idx.on_block_stored(10, "http://w2:8000");
        idx.on_block_stored(20, "http://w2:8000");

        let scorer = PrefixScorer::new(&idx);
        let result = scorer.score(&[10, 20, 30]);

        assert_eq!(result.scores.get("http://w1:8000"), Some(&3));
        assert_eq!(result.scores.get("http://w2:8000"), Some(&2));

        let (best, _) = result.best_worker().unwrap();
        assert_eq!(best, "http://w1:8000");
    }

    #[test]
    fn test_no_match() {
        let idx = make_index();
        idx.on_block_stored(999, "http://w1:8000");

        let scorer = PrefixScorer::new(&idx);
        let result = scorer.score(&[1, 2, 3]);

        // W1 doesn't hold any of the queried blocks.
        assert!(result.scores.is_empty() || result.scores.get("http://w1:8000").is_none());
    }

    #[test]
    fn test_uncached_tokens() {
        let idx = make_index();
        idx.on_block_stored(1, "http://w1:8000");
        idx.on_block_stored(2, "http://w1:8000");

        let scorer = PrefixScorer::new(&idx);
        let result = scorer.score(&[1, 2, 3, 4]);

        // 2 blocks cached, 2 uncached, block_size=64 -> 128 uncached tokens.
        assert_eq!(result.uncached_tokens("http://w1:8000", 64), 128);
    }

    #[test]
    fn test_match_ratio() {
        let idx = make_index();
        idx.on_block_stored(1, "http://w1:8000");
        idx.on_block_stored(2, "http://w1:8000");

        let scorer = PrefixScorer::new(&idx);
        let result = scorer.score(&[1, 2, 3, 4]);

        let ratio = result.match_ratio("http://w1:8000");
        assert!((ratio - 0.5).abs() < f32::EPSILON);
    }
}
