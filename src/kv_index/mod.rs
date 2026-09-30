//! Exact vLLM prefix-block hashing and a generation-safe ownership index.

mod block_hash;
mod index;

pub use block_hash::{BlockHash, BlockKeyGenerator};
pub use index::{KVBlockIndex, OwnershipEvent};
