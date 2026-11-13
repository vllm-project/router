//! HTTP router implementations

mod completion_output;
mod completion_route;
pub mod dp_utils;
mod epd;
mod kv_observation;
pub mod logprobs_merge;
pub mod openai_router;
mod pd_kv_state;
mod pd_program_adapter;
pub mod pd_router;
pub mod pd_types;
mod pd_warm_remote;
mod program_adapter;
pub mod router;
mod sse;
mod usage_merge;
mod usage_stream;
pub mod vllm_pd_router;
pub mod vllm_service_discovery;
