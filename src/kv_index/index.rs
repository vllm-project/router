//! Global KV block index: maps block content hashes to worker locations.
//!
//! This is the core data structure that records which vLLM workers currently
//! hold which KV cache blocks.  It is updated in real time by the
//! [`run_kv_index_updater`](super::updater::run_kv_index_updater) task
//! consuming events from the [`KVEventPool`](crate::kv_events::KVEventPool).
//!
//! Thread safety is provided by [`DashMap`], which uses fine-grained sharded
//! locking to allow concurrent reads and writes from multiple event consumer
//! threads and scoring queries.

use dashmap::DashMap;
use std::time::{Duration, Instant};

/// Metadata about a single block's presence on a specific worker.
#[derive(Debug, Clone)]
struct BlockLocation {
    /// URL or identifier of the worker holding this block.
    worker_url: String,
    /// When this entry was created or last refreshed.
    stored_at: Instant,
    /// Whether this entry was inserted speculatively (before the real
    /// KV event arrives from vLLM).
    is_speculative: bool,
    /// TTL for speculative entries; ignored for non-speculative ones.
    speculative_ttl: Option<Duration>,
}

impl BlockLocation {
    /// Returns `true` if this speculative entry has expired.
    fn is_expired(&self) -> bool {
        if !self.is_speculative {
            return false;
        }
        match self.speculative_ttl {
            Some(ttl) => self.stored_at.elapsed() > ttl,
            None => false,
        }
    }
}

/// Thread-safe global index: `block_hash -> Vec<BlockLocation>`.
///
/// Each entry records one or more workers that hold a KV block with the
/// given content hash.  The index supports three mutation operations
/// corresponding to the three vLLM event types, plus a speculative insert
/// for latency-hiding optimistic scheduling.
pub struct KVBlockIndex {
    index: DashMap<u64, Vec<BlockLocation>>,
    max_entries: usize,
}

impl std::fmt::Debug for KVBlockIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KVBlockIndex")
            .field("entries", &self.index.len())
            .field("max_entries", &self.max_entries)
            .finish()
    }
}

impl KVBlockIndex {
    /// Create a new index with a capacity hint.
    ///
    /// `max_entries` is advisory and used for pre-allocation; the index
    /// does not currently enforce hard eviction (vLLM's `BlockRemoved`
    /// events handle cleanup).
    pub fn new(max_entries: usize) -> Self {
        Self {
            index: DashMap::with_capacity(max_entries.min(1_000_000)),
            max_entries,
        }
    }

    /// Record that a block with the given hash exists on `worker_url`.
    ///
    /// Called when a `BlockStored` event is received.  Duplicate entries
    /// for the same (hash, worker) pair are deduplicated.
    pub fn on_block_stored(&self, block_hash: u64, worker_url: &str) {
        let mut entry = self.index.entry(block_hash).or_insert_with(Vec::new);
        let locations = entry.value_mut();

        // Deduplicate: if the worker is already recorded, refresh timestamp
        // and clear any speculative flag.
        for loc in locations.iter_mut() {
            if loc.worker_url == worker_url {
                loc.stored_at = Instant::now();
                loc.is_speculative = false;
                loc.speculative_ttl = None;
                return;
            }
        }

        locations.push(BlockLocation {
            worker_url: worker_url.to_string(),
            stored_at: Instant::now(),
            is_speculative: false,
            speculative_ttl: None,
        });
    }

    /// Remove a block from a specific worker.
    ///
    /// Called when a `BlockRemoved` event is received.
    pub fn on_block_removed(&self, block_hash: u64, worker_url: &str) {
        if let Some(mut locations) = self.index.get_mut(&block_hash) {
            locations.retain(|loc| loc.worker_url != worker_url);
            if locations.is_empty() {
                drop(locations);
                self.index.remove(&block_hash);
            }
        }
    }

    /// Remove ALL blocks belonging to a specific worker.
    ///
    /// Called when an `AllBlocksCleared` event is received (e.g., engine restart).
    /// This is O(N) over the entire index but is expected to be rare.
    pub fn on_all_blocks_cleared(&self, worker_url: &str) {
        // Collect keys to avoid holding DashMap shard locks across iterations.
        let keys: Vec<u64> = self.index.iter().map(|e| *e.key()).collect();
        for key in keys {
            if let Some(mut locations) = self.index.get_mut(&key) {
                locations.retain(|loc| loc.worker_url != worker_url);
            }
        }
        // Prune empty entries.
        self.index.retain(|_, v| !v.is_empty());
    }

    /// Speculatively insert block entries for a worker.
    ///
    /// This is called immediately after a routing decision to "predict"
    /// that the selected worker will soon hold the uncached blocks.
    /// The entries carry a TTL and are replaced by real events when they
    /// arrive.  This prevents duplicate work when multiple requests with
    /// the same prefix arrive in quick succession.
    pub fn speculative_insert(
        &self,
        block_hashes: &[u64],
        worker_url: &str,
        ttl: Duration,
    ) {
        for &hash in block_hashes {
            let mut entry = self.index.entry(hash).or_insert_with(Vec::new);
            let locations = entry.value_mut();

            // Skip if the worker already has a real (non-speculative) entry.
            let already_real = locations
                .iter()
                .any(|loc| loc.worker_url == worker_url && !loc.is_speculative);
            if already_real {
                continue;
            }

            // Remove stale speculative entry for this worker if present.
            locations.retain(|loc| {
                !(loc.worker_url == worker_url && loc.is_speculative)
            });

            locations.push(BlockLocation {
                worker_url: worker_url.to_string(),
                stored_at: Instant::now(),
                is_speculative: true,
                speculative_ttl: Some(ttl),
            });
        }
    }

    /// Query which workers hold a given block hash (excluding expired
    /// speculative entries).
    ///
    /// Returns an iterator of worker URLs for external callers that need
    /// per-block lookups.
    pub fn get_workers_for_block(&self, block_hash: u64) -> Vec<String> {
        self.index
            .get(&block_hash)
            .map(|locations| {
                locations
                    .iter()
                    .filter(|loc| !loc.is_expired())
                    .map(|loc| loc.worker_url.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Total number of distinct block hashes in the index.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Maximum configured entries (advisory).
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_store_and_lookup() {
        let idx = KVBlockIndex::new(1000);
        idx.on_block_stored(42, "http://w1:8000");
        idx.on_block_stored(42, "http://w2:8000");

        let workers = idx.get_workers_for_block(42);
        assert_eq!(workers.len(), 2);
        assert!(workers.contains(&"http://w1:8000".to_string()));
        assert!(workers.contains(&"http://w2:8000".to_string()));
    }

    #[test]
    fn test_remove_block() {
        let idx = KVBlockIndex::new(1000);
        idx.on_block_stored(42, "http://w1:8000");
        idx.on_block_stored(42, "http://w2:8000");
        idx.on_block_removed(42, "http://w1:8000");

        let workers = idx.get_workers_for_block(42);
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0], "http://w2:8000");
    }

    #[test]
    fn test_clear_all_blocks_for_worker() {
        let idx = KVBlockIndex::new(1000);
        idx.on_block_stored(1, "http://w1:8000");
        idx.on_block_stored(2, "http://w1:8000");
        idx.on_block_stored(2, "http://w2:8000");
        idx.on_block_stored(3, "http://w2:8000");

        idx.on_all_blocks_cleared("http://w1:8000");

        assert!(idx.get_workers_for_block(1).is_empty());
        assert_eq!(idx.get_workers_for_block(2), vec!["http://w2:8000"]);
        assert_eq!(idx.get_workers_for_block(3), vec!["http://w2:8000"]);
    }

    #[test]
    fn test_deduplication() {
        let idx = KVBlockIndex::new(1000);
        idx.on_block_stored(42, "http://w1:8000");
        idx.on_block_stored(42, "http://w1:8000");
        idx.on_block_stored(42, "http://w1:8000");

        let workers = idx.get_workers_for_block(42);
        assert_eq!(workers.len(), 1);
    }

    #[test]
    fn test_speculative_insert_and_expiry() {
        let idx = KVBlockIndex::new(1000);

        // Insert with a very short TTL.
        idx.speculative_insert(&[100], "http://w1:8000", Duration::from_millis(1));

        // Should be visible immediately.
        assert_eq!(idx.get_workers_for_block(100).len(), 1);

        // Wait for expiry.
        std::thread::sleep(Duration::from_millis(10));

        // Should be filtered out.
        assert!(idx.get_workers_for_block(100).is_empty());
    }

    #[test]
    fn test_speculative_replaced_by_real() {
        let idx = KVBlockIndex::new(1000);

        idx.speculative_insert(&[200], "http://w1:8000", Duration::from_millis(1));
        // Real event replaces speculative.
        idx.on_block_stored(200, "http://w1:8000");

        std::thread::sleep(Duration::from_millis(10));

        // Should still be visible (real entry, not expired).
        assert_eq!(idx.get_workers_for_block(200).len(), 1);
    }
}
