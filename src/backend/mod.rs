//! Switchable worker backends.
//!
//! Worker pool is all-`http(s)://` or all-`grpc://`. Mixed schemes fail
//! at init / `add_worker` (gRPC is `token_ids` only).
//!
//! - All-HTTP: KV-aware routing obtains exact inputs from token arrays or
//!   the workers' render endpoints, then `router.rs` proxies OpenAI JSON.
//!   Other policies do not prepare tokens.
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
    get_grpc_model_info, get_grpc_server_info, grpc_engine_version_older_than_0_30,
    model_info_json, openai_models_json, server_info_json, warn_old_grpc_pd_workers,
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

    impl OutputFrameProducer {
        /// Reserve capacity before publishing the output frame marker.
        pub(crate) async fn send_bounded<T>(
            &mut self,
            tx: &tokio::sync::mpsc::Sender<T>,
            chunk: T,
            has_generated_output: bool,
        ) -> bool {
            let Ok(permit) = tx.reserve().await else {
                return false;
            };
            let next = self.emitted.saturating_add(1);
            if has_generated_output {
                let _ = self.first_output.compare_exchange(
                    0,
                    next,
                    Ordering::Release,
                    Ordering::Relaxed,
                );
            }
            permit.send(chunk);
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

        /// A bounded hand-off must not publish the first-output marker while it
        /// is still waiting for capacity: the marker is an ordered position, and
        /// the frame it points at has not been enqueued yet.
        #[tokio::test]
        async fn bounded_send_keeps_the_marker_unpublished_until_a_slot_frees() {
            let (tx, mut rx) = tokio::sync::mpsc::channel::<&str>(2);
            let (mut producer, mut consumer) = output_frame_tracking();
            assert!(producer.send_bounded(&tx, "meta", false).await);
            assert!(producer.send_bounded(&tx, "first token", true).await);
            assert_eq!(producer.first_output.load(Ordering::Acquire), 2);

            // Queue is full: the third frame cannot be reserved, and the marker
            // must stay where it is while the send waits.
            let blocked = tokio::time::timeout(
                std::time::Duration::from_millis(100),
                producer.send_bounded(&tx, "second token", true),
            )
            .await;
            assert!(
                blocked.is_err(),
                "the bounded send did not wait for capacity"
            );
            assert_eq!(producer.emitted, 2);
            assert_eq!(producer.first_output.load(Ordering::Acquire), 2);

            assert_eq!(rx.recv().await, Some("meta"));
            assert!(!consumer.on_frame());
            assert_eq!(rx.recv().await, Some("first token"));
            assert!(consumer.on_frame());
        }

        #[tokio::test]
        async fn bounded_send_publishes_the_marker_once_capacity_arrives() {
            let (tx, mut rx) = tokio::sync::mpsc::channel::<&str>(1);
            let (mut producer, _consumer) = output_frame_tracking();
            assert!(producer.send_bounded(&tx, "meta", false).await);
            let pending = tokio::spawn(async move {
                let sent = producer.send_bounded(&tx, "first token", true).await;
                (
                    sent,
                    producer.emitted,
                    producer.first_output.load(Ordering::Acquire),
                )
            });
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            assert_eq!(rx.recv().await, Some("meta"));
            let (sent, emitted, first_output) = pending.await.unwrap();
            assert!(sent);
            assert_eq!(emitted, 2);
            assert_eq!(
                first_output, 2,
                "capacity arrived but the marker was not published"
            );
        }

        /// Closing the receiver while a reservation is pending must fail the
        /// hand-off without publishing a marker for a frame nobody will see.
        #[tokio::test]
        async fn bounded_send_reports_closed_receiver_without_publishing() {
            let (tx, rx) = tokio::sync::mpsc::channel::<&str>(1);
            let (mut producer, _consumer) = output_frame_tracking();
            assert!(producer.send_bounded(&tx, "meta", false).await);
            let pending = tokio::spawn(async move {
                let sent = producer.send_bounded(&tx, "first token", true).await;
                (
                    sent,
                    producer.emitted,
                    producer.first_output.load(Ordering::Acquire),
                )
            });
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            drop(rx);
            let (sent, emitted, first_output) = pending.await.unwrap();
            assert!(!sent, "a closed receiver accepted a frame");
            assert_eq!(emitted, 1, "a dropped frame counted as emitted");
            assert_eq!(first_output, 0, "a dropped frame published the marker");
        }
    }
}
