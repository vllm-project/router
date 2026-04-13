//! KV Cache Events ingestion from vLLM workers.
//!
//! This module subscribes to real-time KV cache change events published by
//! vLLM instances over ZMQ PUB sockets.  Each vLLM worker emits
//! [`BlockStored`] / [`BlockRemoved`] / [`AllBlocksCleared`] events whenever
//! its KV cache state changes, enabling the router to maintain an accurate,
//! near-real-time global index of which KV blocks reside on which workers.
//!
//! # Architecture
//!
//! ```text
//! vLLM Pod₁  ──ZMQ PUB──►  ┌─────────────────┐
//! vLLM Pod₂  ──ZMQ PUB──►  │  KVEventPool     │──► mpsc ──► KVBlockIndex
//! vLLM Pod₃  ──ZMQ PUB──►  └─────────────────┘
//! ```
//!
//! The [`KVEventPool`] manages per-worker ZMQ SUB connections, decodes
//! incoming msgpack payloads (compatible with vLLM's `msgspec` encoding),
//! and forwards typed [`KVEventBatch`] structs through a tokio mpsc channel
//! for index updates.

pub mod decoder;
pub mod pool;
pub mod subscriber;

pub use decoder::{KVEvent, KVEventBatch};
pub use pool::KVEventPool;
