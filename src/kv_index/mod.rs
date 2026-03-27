//! Precise KV block index and scoring for cache-aware routing.
//!
//! This module maintains a global, near-real-time index that maps KV block
//! content hashes to the set of workers that hold each block.  Combined
//! with a token-to-block-hash generator, it enables the router to compute
//! exact prefix-cache hit scores for every candidate worker and route
//! requests to the one that maximizes KV cache reuse.
//!
//! # Components
//!
//! * [`BlockKeyGenerator`] – Converts token sequences into block hashes
//!   compatible with vLLM's content-addressed block keys.
//! * [`KVBlockIndex`] – Thread-safe map from block hash → worker locations,
//!   updated via [`KVEvent`]s from the event pool.
//! * [`PrefixScorer`] – Given a sequence of block keys, scores each worker
//!   by the length of the longest contiguous prefix match.
//! * [`run_kv_index_updater`] – Background task that consumes events from
//!   the [`KVEventPool`] channel and applies them to the index.

pub mod block_hash;
pub mod index;
pub mod scorer;
pub mod updater;

pub use block_hash::BlockKeyGenerator;
pub use index::KVBlockIndex;
pub use scorer::PrefixScorer;
pub use updater::run_kv_index_updater;
