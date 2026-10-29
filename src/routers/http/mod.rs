//! HTTP router implementations

pub mod dp_utils;
mod epd;
pub mod logprobs_merge;
pub mod openai_router;
mod pd_cached_tokens;
mod pd_kv_state;
mod pd_program_adapter;
pub mod pd_router;
pub mod pd_types;
mod program_adapter;
pub mod router;
mod usage_merge;
pub mod vllm_pd_router;
pub mod vllm_service_discovery;
