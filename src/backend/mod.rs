//! Switchable worker backends.
//!
//! Worker pool is all-`http(s)://` or all-`grpc://`. Mixed schemes fail
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

pub mod completion;
pub mod control;
pub mod convert;
pub mod detect;
pub(crate) mod encode_cache;
pub mod frontend;
pub mod grpc;
pub mod health;
pub(crate) mod l1_encode_cache;
pub mod openai;
pub mod preprocess;
mod tokenizer_determinism;
mod vllm_frontend;

/// Opt-in router stage clocks.
pub fn stages_enabled() -> bool {
    matches!(
        std::env::var("VLLM_ROUTER_STAGES").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

pub use completion::{
    prepare_completion, prepare_completion_classified, validate_completion_tokenizer_definition,
    CompletionPreparationError,
};
pub use control::{
    get_grpc_model_info, get_grpc_server_info, model_info_json, openai_models_json,
    server_info_json,
};
pub use convert::prepare_grpc_completion;
pub use detect::{
    classify_worker_urls, connection_mode_from_url, grpc_connect_uri, is_grpc_url, parse_dp_rank,
    strip_dp_suffix, WorkerPoolKind,
};
pub use frontend::{EngineFrontend, PrepareError, PreparedRequest};
pub use grpc::GrpcEngineBackend;
pub use health::check_grpc_health;
pub use l1_encode_cache::{L1CacheConfig, L1CacheStats, L1HashStrategy};
pub use preprocess::TokenizerCache;

/// Prost/Tonic bindings for vLLM Inference + Control (`vllm-proto` on crates.io).
pub use vllm_proto as pb;

// Shared by gRPC emission and HTTP response consumption.
pub(crate) mod stream_output {
    //! First model output tracking across a queued response body.
    //!
    //! Each successful producer send must yield exactly one successful consumer
    //! data frame, in order. Metadata, errors and terminal chunks count as frames
    //! but do not mark model output. Publishing before enqueueing prevents a fast
    //! consumer from missing the transition; counting consumption prevents queued
    //! output from transitioning an earlier metadata frame into decode.

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use tokio::sync::mpsc::UnboundedSender;

    pub(crate) struct OutputFrameProducer {
        first_output: Arc<AtomicUsize>,
        emitted: usize,
    }

    #[derive(Clone)]
    pub(crate) struct OutputFrameConsumer {
        first_output: Arc<AtomicUsize>,
        consumed: usize,
    }

    pub(crate) fn output_frame_tracking() -> (OutputFrameProducer, OutputFrameConsumer) {
        // Zero means no generated output; actual frame indices are one-based.
        let first_output = Arc::new(AtomicUsize::new(0));
        (
            OutputFrameProducer {
                first_output: first_output.clone(),
                emitted: 0,
            },
            OutputFrameConsumer {
                first_output,
                consumed: 0,
            },
        )
    }

    impl OutputFrameProducer {
        /// Own both indexing and enqueueing so every emitted chunk is counted.
        pub(crate) fn send<T, E>(
            &mut self,
            tx: &UnboundedSender<Result<T, E>>,
            chunk: T,
            has_generated_output: bool,
        ) -> bool {
            let next = self.emitted.saturating_add(1);
            if has_generated_output {
                let _ = self.first_output.compare_exchange(
                    0,
                    next,
                    Ordering::Release,
                    Ordering::Relaxed,
                );
            }
            if tx.send(Ok(chunk)).is_err() {
                return false;
            }
            self.emitted = next;
            true
        }
    }

    impl OutputFrameConsumer {
        /// Call exactly once per successful data frame consumed from the body.
        pub(crate) fn on_frame(&mut self) -> bool {
            self.consumed = self.consumed.saturating_add(1);
            let first_output = self.first_output.load(Ordering::Acquire);
            first_output != 0 && self.consumed >= first_output
        }
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    mod tests {
        use super::*;

        #[test]
        fn queued_output_does_not_transition_an_earlier_metadata_frame() {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Result<&str, ()>>();
            let (mut producer, mut consumer) = output_frame_tracking();
            assert!(producer.send(&tx, "prompt info", false));
            assert!(producer.send(&tx, "first token", true));
            assert!(producer.send(&tx, "second token", true));
            assert!(producer.send(&tx, "usage", false));

            for (chunk, started) in [
                ("prompt info", false),
                ("first token", true),
                ("second token", true),
                ("usage", true),
            ] {
                assert_eq!(rx.try_recv().unwrap(), Ok(chunk));
                assert_eq!(consumer.on_frame(), started);
            }
        }

        #[test]
        fn metadata_and_terminal_chunks_never_signal_model_output() {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Result<&str, ()>>();
            let (mut producer, mut consumer) = output_frame_tracking();
            for chunk in ["stages", "prompt info", "error", "done"] {
                assert!(producer.send(&tx, chunk, false));
                assert_eq!(rx.try_recv().unwrap(), Ok(chunk));
                assert!(!consumer.on_frame());
            }
        }

        #[test]
        fn consumer_can_catch_up_before_the_producer_generates_output() {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Result<&str, ()>>();
            let (mut producer, mut consumer) = output_frame_tracking();
            assert!(producer.send(&tx, "prompt info", false));
            assert_eq!(rx.try_recv().unwrap(), Ok("prompt info"));
            assert!(!consumer.on_frame());
            assert!(producer.send(&tx, "first token", true));
            assert_eq!(rx.try_recv().unwrap(), Ok("first token"));
            assert!(consumer.on_frame());
            drop(rx);
            assert!(!producer.send(&tx, "cancelled", true));
        }
    }
}
