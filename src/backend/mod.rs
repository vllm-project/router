//! Switchable worker backends.
//!
//! Worker pool is all-`http(s)://` or all-`grpc(s)://`. Mixed schemes fail
//! at init / `add_worker` (gRPC is `token_ids` only).
//!
//! - All-HTTP: policy selects, then `router.rs` reverse-proxies OpenAI JSON
//!   (no chat/tokenizer frontend on the critical path).
//! - All-gRPC: `EngineFrontend::prepare(chat)` → `token_ids`, then policy
//!   selects, then `EngineFrontend::dispatch` (`convert` + GenerateStream +
//!   detok).
//!
//! gRPC wire types come from crates.io `vllm-proto`. Chat+tokenize is
//! `vllm-chat` + `vllm-tokenizer` only (Cargo git, not `pip install`).
//!
//! `VLLM_ROUTER_STAGES=1` emits stage clocks. gRPC always tokenizes; HTTP
//! remains a transparent proxy and reports worker header/first-byte timings.

pub mod convert;
pub mod detect;
pub mod frontend;
pub mod grpc;
pub mod health;
pub mod openai;
pub mod preprocess;
mod vllm_frontend;

/// Opt-in router stage clocks.
pub fn stages_enabled() -> bool {
    matches!(
        std::env::var("VLLM_ROUTER_STAGES").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

pub use detect::{
    classify_worker_urls, connection_mode_from_url, grpc_connect_uri, is_grpc_url, parse_dp_rank,
    strip_dp_suffix, WorkerPoolKind,
};
pub use frontend::{EngineFrontend, PreparedChat};
pub use grpc::GrpcEngineBackend;
pub use health::check_grpc_health;
pub use preprocess::TokenizerCache;

/// Prost/Tonic bindings for vLLM Inference + Control (`vllm-proto` on crates.io).
pub use vllm_proto as pb;
