//! Source leases and visibility for a declared-layout event stream.
//!
//! A caller must authenticate the publisher, its complete cache layout, and
//! replay/live watermark. A local generation is not a publisher epoch.

use std::sync::Arc;
#[cfg(all(test, feature = "cpu", feature = "ready"))]
use std::time::Duration;

use tokio::sync::mpsc;

use crate::kv_index::source::endpoints::EventSource;
#[cfg(all(test, feature = "cpu", feature = "ready"))]
use crate::kv_index::source::events::transport::retry_backoff;
use crate::kv_index::storage::source_transaction::{StreamPhase as Phase, StreamState};
use crate::kv_index::storage::IndexBatch;
use crate::kv_index::{IngestionSignal, KVEvent, SourceBinding, SourceError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamLease(pub(crate) u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamError {
    StaleLease,
    InvalidState,
    Index(SourceError),
}

/// Lease changes and cache mutations use the index's single source transaction.
pub struct EventStream {
    index: Arc<SourceBinding>,
}

impl EventStream {
    /// Fence any previous subscriber and hide its cache before starting a new replay.
    pub fn new(index: Arc<SourceBinding>) -> Self {
        index.invalidate();
        Self { index }
    }

    /// The caller must first establish that sequence zero represents an empty
    /// cache for the authenticated publisher's current incarnation.
    pub fn begin_replay(&self) -> Result<StreamLease, StreamError> {
        self.index
            .with_active_binding(|| {
                self.index.with_stream_batch(|state, tx| {
                    let lease = StreamLease(state.begin_lease());
                    self.index.reset_stream(state, tx, Some(0), false);
                    lease
                })
            })
            .ok_or(StreamError::Index(SourceError::Untrusted))
    }

    pub(super) fn restart_replay(&self, lease: StreamLease, loss: bool) -> Result<(), StreamError> {
        self.with_state(lease, |state, tx| {
            if loss {
                crate::kv_index::record_signal_loss();
            }
            self.index.reset_stream(state, tx, Some(0), false);
            Ok(())
        })
    }

    pub(super) fn suspend_for_replay(
        &self,
        lease: StreamLease,
        loss: bool,
    ) -> Result<(), StreamError> {
        self.with_state(lease, |state, tx| {
            if loss {
                crate::kv_index::record_signal_loss();
            }
            if state.phase == Phase::Replaying {
                return Ok(());
            }
            if !state.visible() {
                return Err(StreamError::Index(SourceError::Untrusted));
            }
            tx.suspend_source(&self.index.target.instance_id, self.index.target.dp_rank);
            state.hide();
            state.phase = Phase::Replaying;
            Ok(())
        })
    }

    pub fn apply_payload(
        &self,
        lease: StreamLease,
        seq: u64,
        payload: &[u8],
    ) -> Result<(), StreamError> {
        self.apply_publisher_payload(lease, seq, payload)
            .map(|_| ())
    }

    pub(super) fn apply_publisher_payload(
        &self,
        lease: StreamLease,
        seq: u64,
        payload: &[u8],
    ) -> Result<bool, StreamError> {
        let batch = crate::kv_index::decode_batch(payload);
        let prepared = batch
            .as_ref()
            .ok()
            .map(|batch| self.index.prepare_batch(batch));
        self.with_state(lease, |state, tx| {
            let result = match (&batch, prepared) {
                (Ok(batch), Some(prepared)) => {
                    self.index.apply_prepared(state, tx, seq, batch, prepared)
                }
                _ => SourceBinding::fail(state, tx, &self.index.owner, SourceError::InvalidBatch),
            };
            if let Err(error) = result {
                // A rejected batch means the publisher's layout and the declared
                // one disagree; naming the reason helps fix a layout
                // from a live system.
                match error {
                    SourceError::IntBlockHashes => {
                        // The engine's default event encoding, so the fix belongs to
                        // the engine invocation and the operator has to be told once.
                        if !state.rejection_reported {
                            state.rejection_reported = true;
                            tracing::error!(
                                seq,
                                "KV event source publishes integer block hashes, which the sha256 \
                             index cannot address; no prefix will be credited for this engine. \
                             Start it with VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0."
                            );
                        }
                    }
                    _ => tracing::warn!(seq, error = ?error, "KV event batch rejected"),
                }
                state.phase = Phase::Replaying;
                return Err(StreamError::Index(error));
            }
            Ok(batch.as_ref().is_ok_and(|batch| {
                batch
                    .events
                    .iter()
                    .any(|event| matches!(event, KVEvent::AllBlocksCleared(_)))
            }))
        })
    }

    /// Publish a continuous replay; its caller must keep the cursor caught up
    /// until the live subscription has been confirmed.
    pub fn finish_replay(&self, lease: StreamLease, next_seq: u64) -> Result<(), StreamError> {
        self.with_state(lease, |state, tx| {
            if state.phase != Phase::Replaying {
                return Err(StreamError::InvalidState);
            }
            self.index
                .publish_stream(state, tx, next_seq)
                .map_err(|error| {
                    state.phase = Phase::Replaying;
                    StreamError::Index(error)
                })
        })
    }

    pub fn retire(&self, lease: StreamLease) -> Result<(), StreamError> {
        self.index
            .index
            .with_existing_stream_batch(
                &self.index.target,
                &self.index.owner,
                &self.index.control,
                |state, tx| {
                    Self::check(state, lease)?;
                    self.index.reset_stream(state, tx, None, false);
                    state.lease_id = 0;
                    Ok(())
                },
            )
            .unwrap_or(Ok(()))
    }

    fn check(state: &StreamState, lease: StreamLease) -> Result<(), StreamError> {
        if state.lease_id != lease.0 {
            Err(StreamError::StaleLease)
        } else {
            Ok(())
        }
    }

    fn with_state<R>(
        &self,
        lease: StreamLease,
        operation: impl FnOnce(&mut StreamState, &mut IndexBatch<'_>) -> Result<R, StreamError>,
    ) -> Result<R, StreamError> {
        self.index
            .with_active_binding(|| {
                self.index.with_stream_batch(|state, tx| {
                    Self::check(state, lease)?;
                    if state.phase == Phase::Retired {
                        return Err(StreamError::InvalidState);
                    }
                    operation(state, tx)
                })
            })
            .unwrap_or(Err(StreamError::Index(SourceError::Untrusted)))
    }

    pub(super) fn is_active(&self, lease: StreamLease) -> bool {
        self.index
            .binding
            .as_ref()
            .is_none_or(|binding| binding.is_active())
            && self
                .index
                .index
                .stream_lease_is_active(&self.index.owner, lease.0)
    }
}

/// Owns a live subscription. Dropping it stops the task and retires its view.
pub struct EventSubscriberHandle {
    subscriber: super::subscriber::SubscriberHandle,
    stream: Arc<EventStream>,
    lease: StreamLease,
}

impl EventSubscriberHandle {
    pub fn shutdown(&self) {
        let _ = self.stream.retire(self.lease);
        self.subscriber.shutdown();
    }
}

impl Drop for EventSubscriberHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Attach to one publisher incarnation using its replay stream.
pub fn spawn_zmq(
    source: EventSource,
    stream: Arc<EventStream>,
    ctx: Arc<zmq::Context>,
) -> Result<EventSubscriberHandle, &'static str> {
    spawn_zmq_with_signals(source, stream, ctx, None)
}

pub(in crate::kv_index::source) fn spawn_zmq_with_signals(
    source: EventSource,
    stream: Arc<EventStream>,
    ctx: Arc<zmq::Context>,
    signals: Option<mpsc::Sender<IngestionSignal>>,
) -> Result<EventSubscriberHandle, &'static str> {
    if source.replay_endpoint.as_deref().is_none_or(str::is_empty) {
        return Err("replay endpoint required");
    }
    if source.topic.is_empty() || source.pub_endpoint.is_empty() {
        return Err("an incarnation-bound topic and publisher are required");
    }
    let lease = stream
        .begin_replay()
        .map_err(|_| "KV source is no longer active")?;
    let subscriber = super::subscriber::spawn(source, stream.clone(), lease, ctx, signals);
    Ok(EventSubscriberHandle {
        subscriber,
        stream,
        lease,
    })
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::kv_index::source::binding::tests::qwen_index;
    use crate::kv_index::RoutableTarget;

    async fn recv(socket: &zmq::Socket) -> Vec<Vec<u8>> {
        tokio::time::timeout(Duration::from_secs(4), async {
            loop {
                match socket.recv_multipart(zmq::DONTWAIT) {
                    Ok(m) => break m,
                    Err(zmq::Error::EAGAIN) => tokio::time::sleep(Duration::from_millis(2)).await,
                    Err(e) => panic!("{e}"),
                }
            }
        })
        .await
        .expect("missing frame")
    }

    async fn wait_for(index: &SourceBinding, prompt: &[u32], expected: u64) {
        tokio::time::timeout(Duration::from_secs(4), async {
            while index.reusable_tokens(prompt, usize::MAX) != expected {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn queued_live_store_cannot_replace_failed_startup_replay() {
        let ctx = Arc::new(zmq::Context::new());
        let publisher = ctx.socket(zmq::XPUB).unwrap();
        let replay = ctx.socket(zmq::ROUTER).unwrap();
        publisher.set_linger(0).unwrap();
        replay.set_linger(0).unwrap();
        publisher.bind("inproc://hybrid-failed-start-pub").unwrap();
        replay.bind("inproc://hybrid-failed-start-replay").unwrap();
        let index = Arc::new(qwen_index(
            RoutableTarget {
                instance_id: Arc::from("worker"),
                dp_rank: 0,
            },
            8,
        ));
        let handle = spawn_zmq(
            EventSource {
                source: Arc::from("worker"),
                dp_rank: 0,
                pub_endpoint: "inproc://hybrid-failed-start-pub".into(),
                replay_endpoint: Some("inproc://hybrid-failed-start-replay".into()),
                topic: "epoch".into(),
                hwm: None,
            },
            Arc::new(EventStream::new(index.clone())),
            ctx,
        )
        .unwrap();
        recv(&publisher).await;
        let request = recv(&replay).await;
        let payload = crate::kv_index::source::binding::tests::qwen35_payload();
        publisher
            .send_multipart([b"epoch".as_slice(), &0_i64.to_be_bytes(), &payload], 0)
            .unwrap();
        // Missing sequence zero rejects the replay. The queued live Store
        // alone cannot prove whether a later eviction was also missed.
        replay
            .send_multipart(
                [
                    request[0].as_slice(),
                    &[],
                    b"epoch",
                    &1_i64.to_be_bytes(),
                    &payload,
                ],
                0,
            )
            .unwrap();
        let request = recv(&replay).await;
        assert_eq!(request[2], 0_u64.to_be_bytes());
        let prompt: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 0);
        handle.shutdown();
    }

    #[test]
    fn replay_backoff_doubles_and_caps() {
        assert_eq!(retry_backoff(0), Duration::from_millis(100));
        assert_eq!(retry_backoff(1), Duration::from_millis(200));
        assert_eq!(retry_backoff(5), Duration::from_millis(3_200));
        assert_eq!(retry_backoff(6), Duration::from_millis(5_000));
        assert_eq!(retry_backoff(30), Duration::from_millis(5_000));
    }

    #[test]
    fn inactive_binding_rejects_replay_before_allocating_a_lease() {
        use crate::kv_index::source::binding::SourceLease;

        let index = qwen_index(
            RoutableTarget {
                instance_id: Arc::from("worker"),
                dp_rank: 0,
            },
            8,
        )
        .with_source_binding(SourceLease {
            strict_rank: Arc::new(parking_lot::Mutex::new(true)),
            active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        });
        assert_eq!(
            EventStream::new(Arc::new(index)).begin_replay(),
            Err(StreamError::Index(SourceError::Untrusted))
        );
    }

    #[test]
    fn repeated_retire_does_not_clear_a_replacement_stream() {
        let index = Arc::new(qwen_index(
            RoutableTarget {
                instance_id: Arc::from("worker"),
                dp_rank: 0,
            },
            8,
        ));
        let old = EventStream::new(index.clone());
        let old_lease = old.begin_replay().unwrap();
        old.retire(old_lease).unwrap();
        old.retire(old_lease).unwrap();
        assert!(!old.is_active(old_lease));
        let replacement = EventStream::new(index.clone());
        let replacement_lease = replacement.begin_replay().unwrap();
        assert_ne!(old_lease, replacement_lease);
        let payload = &crate::kv_index::source::binding::tests::qwen35_payload();
        replacement
            .apply_payload(replacement_lease, 0, payload)
            .unwrap();
        replacement.finish_replay(replacement_lease, 1).unwrap();
        let prompt: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 528);
        assert_eq!(old.retire(old_lease), Err(StreamError::StaleLease));
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 528);
    }

    #[test]
    fn retired_subscriber_cannot_change_replacement_replay() {
        let index = Arc::new(qwen_index(
            RoutableTarget {
                instance_id: Arc::from("worker"),
                dp_rank: 0,
            },
            8,
        ));
        let stream = EventStream::new(index.clone());
        let first = stream.begin_replay().unwrap();
        let current = stream.begin_replay().unwrap();
        let payload = &crate::kv_index::source::binding::tests::qwen35_payload();
        let prompt: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
        assert_eq!(
            stream.apply_payload(first, 0, payload),
            Err(StreamError::StaleLease)
        );
        stream.apply_payload(current, 0, payload).unwrap();
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 0);
        assert_eq!(stream.retire(first), Err(StreamError::StaleLease));
        stream.finish_replay(current, 1).unwrap();
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 528);
        assert_eq!(
            stream.apply_payload(first, 1, &[0xc1]),
            Err(StreamError::StaleLease)
        );
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 528);
        let clear = crate::kv_index::source::test_support::event_payload(
            1.0,
            &[crate::kv_index::KVEvent::AllBlocksCleared(
                Default::default(),
            )],
            Some(0),
        );
        stream.apply_payload(current, 1, &clear).unwrap();
        assert!(
            stream.is_active(current),
            "a cache clear must not revoke its subscriber lease"
        );
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 0);
        stream.apply_payload(current, 2, payload).unwrap();
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 528);
        assert_eq!(
            stream.apply_payload(current, 3, &[0xc1]),
            Err(StreamError::Index(SourceError::InvalidBatch))
        );
        assert!(
            stream.is_active(current),
            "a rejected batch must remain recoverable"
        );
        let loss_epoch = crate::kv_index::signal_loss_epoch();
        stream.restart_replay(current, true).unwrap();
        assert!(crate::kv_index::signal_loss_epoch() > loss_epoch);
        stream.apply_payload(current, 0, payload).unwrap();
        stream.finish_replay(current, 1).unwrap();
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 528);
        let loss_epoch = crate::kv_index::signal_loss_epoch();
        stream.suspend_for_replay(current, true).unwrap();
        assert!(crate::kv_index::signal_loss_epoch() > loss_epoch);
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 0);
        stream.finish_replay(current, 1).unwrap();
        stream.retire(current).unwrap();
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 0);
    }

    #[tokio::test]
    async fn quiet_replay_recovers_and_polls_until_live_handoff() {
        use crate::kv_hash::BlockKeyGenerator;
        use crate::kv_index::wire::{BlockRemoved, ExternalBlockHash};
        use crate::kv_index::{AllBlocksCleared, KVEvent};

        let ctx = Arc::new(zmq::Context::new());
        let publisher = ctx.socket(zmq::XPUB).unwrap();
        let replay = ctx.socket(zmq::ROUTER).unwrap();
        publisher.set_linger(0).unwrap();
        replay.set_linger(0).unwrap();
        publisher.bind("inproc://hybrid-quiet-pub").unwrap();
        replay.bind("inproc://hybrid-quiet-replay").unwrap();
        let index = Arc::new(qwen_index(
            RoutableTarget {
                instance_id: Arc::from("worker"),
                dp_rank: 0,
            },
            8,
        ));
        let stream = Arc::new(EventStream::new(index.clone()));
        let handle = spawn_zmq(
            EventSource {
                source: Arc::from("worker"),
                dp_rank: 0,
                pub_endpoint: "inproc://hybrid-quiet-pub".into(),
                replay_endpoint: Some("inproc://hybrid-quiet-replay".into()),
                topic: "epoch-quiet".into(),
                hwm: None,
            },
            stream,
            ctx,
        )
        .unwrap();
        let subscription = recv(&publisher).await;
        assert_eq!(
            subscription[0],
            [b"\x01".as_slice(), b"epoch-quiet"].concat()
        );
        let request = recv(&replay).await;
        assert_eq!(request[2], 0_u64.to_be_bytes());
        let payload = &crate::kv_index::source::binding::tests::qwen35_payload();
        replay
            .send_multipart(
                [
                    request[0].as_slice(),
                    &[],
                    b"epoch-quiet",
                    &0_i64.to_be_bytes(),
                    payload,
                ],
                0,
            )
            .unwrap();
        replay
            .send_multipart(
                [
                    request[0].as_slice(),
                    &[],
                    &[],
                    &(-1_i64).to_be_bytes(),
                    &[],
                ],
                0,
            )
            .unwrap();
        let prompt: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
        // No live frame was sent: the replay must still restore the cache.
        wait_for(&index, &prompt, 528).await;
        let request = recv(&replay).await;
        assert_eq!(request[2], 1_u64.to_be_bytes());
        let clear = rmp_serde::to_vec_named(&(
            2.0_f64,
            vec![KVEvent::AllBlocksCleared(AllBlocksCleared::default())],
            Some(0_u32),
        ))
        .unwrap();
        // Simulate a frame missed before SUB became active. Incremental replay
        // must remove the old claim even when no live event arrives.
        replay
            .send_multipart(
                [
                    request[0].as_slice(),
                    &[],
                    b"epoch-quiet",
                    &1_i64.to_be_bytes(),
                    &clear,
                ],
                0,
            )
            .unwrap();
        wait_for(&index, &prompt, 0).await;
        replay
            .send_multipart(
                [
                    request[0].as_slice(),
                    &[],
                    b"epoch-quiet",
                    &2_i64.to_be_bytes(),
                    payload,
                ],
                0,
            )
            .unwrap();
        replay
            .send_multipart(
                [
                    request[0].as_slice(),
                    &[],
                    &[],
                    &(-1_i64).to_be_bytes(),
                    &[],
                ],
                0,
            )
            .unwrap();
        wait_for(&index, &prompt, 528).await;
        // Replayed frames can reach SUB later in separate batches.
        publisher
            .send_multipart([b"epoch-quiet".as_slice(), &1_i64.to_be_bytes(), &clear], 0)
            .unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;
        publisher
            .send_multipart(
                [b"epoch-quiet".as_slice(), &2_i64.to_be_bytes(), payload],
                0,
            )
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 528);
        publisher
            .send_multipart([b"epoch-quiet".as_slice(), &3_i64.to_be_bytes(), &clear], 0)
            .unwrap();
        wait_for(&index, &prompt, 0).await;

        // Once live handoff is complete, the last event may still be lost
        // without a subsequent sequence gap to expose it.
        publisher
            .send_multipart(
                [b"epoch-quiet".as_slice(), &4_i64.to_be_bytes(), payload],
                0,
            )
            .unwrap();
        wait_for(&index, &prompt, 528).await;
        let hash = BlockKeyGenerator::new(528, 0).generate_block_keys(&prompt)[0];
        let missed_remove = rmp_serde::to_vec_named(&(
            3.0_f64,
            vec![KVEvent::BlockRemoved(BlockRemoved {
                block_hashes: vec![ExternalBlockHash::Bytes(hash.to_vec())],
                medium: Some("GPU".into()),
                group_idx: Some(3),
                locality: None,
                ownership: None,
            })],
            Some(0_u32),
        ))
        .unwrap();
        let request = recv(&replay).await;
        assert_eq!(request[2], 5_u64.to_be_bytes());
        // vLLM services replay between queue polls. A pending empty replay
        // must not hide a verified prefix during that response delay.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 528);
        replay
            .send_multipart(
                [
                    request[0].as_slice(),
                    &[],
                    &[],
                    &(-1_i64).to_be_bytes(),
                    &[],
                ],
                0,
            )
            .unwrap();
        let request = recv(&replay).await;
        assert_eq!(request[2], 5_u64.to_be_bytes());
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 528);
        replay
            .send_multipart(
                [
                    request[0].as_slice(),
                    &[],
                    b"epoch-quiet",
                    &5_i64.to_be_bytes(),
                    &missed_remove,
                ],
                0,
            )
            .unwrap();
        replay
            .send_multipart(
                [
                    request[0].as_slice(),
                    &[],
                    &[],
                    &(-1_i64).to_be_bytes(),
                    &[],
                ],
                0,
            )
            .unwrap();
        wait_for(&index, &prompt, 0).await;

        // Replay/live overlap must not treat the already-applied removal as a
        // publisher restart or apply it twice.
        publisher
            .send_multipart(
                [
                    b"epoch-quiet".as_slice(),
                    &5_i64.to_be_bytes(),
                    &missed_remove,
                ],
                0,
            )
            .unwrap();
        publisher
            .send_multipart(
                [b"epoch-quiet".as_slice(), &6_i64.to_be_bytes(), payload],
                0,
            )
            .unwrap();
        wait_for(&index, &prompt, 528).await;

        // A replay endpoint that stops responding must not leave the old
        // credit visible, and the source must recover after replay resumes.
        let request = recv(&replay).await;
        assert_eq!(request[2], 7_u64.to_be_bytes());
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 528);
        let request = tokio::time::timeout(Duration::from_secs(8), async {
            loop {
                match replay.recv_multipart(zmq::DONTWAIT) {
                    Ok(frames) => break frames,
                    Err(zmq::Error::EAGAIN) => tokio::time::sleep(Duration::from_millis(1)).await,
                    Err(error) => panic!("socket: {error}"),
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(request[2], 0_u64.to_be_bytes());
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 0);
        for (seq, event) in [
            (0_i64, payload.as_slice()),
            (1, clear.as_slice()),
            (2, payload.as_slice()),
            (3, clear.as_slice()),
            (4, payload.as_slice()),
            (5, missed_remove.as_slice()),
            (6, payload.as_slice()),
        ] {
            replay
                .send_multipart(
                    [
                        request[0].as_slice(),
                        &[],
                        b"epoch-quiet",
                        &seq.to_be_bytes(),
                        event,
                    ],
                    0,
                )
                .unwrap();
        }
        replay
            .send_multipart(
                [
                    request[0].as_slice(),
                    &[],
                    &[],
                    &(-1_i64).to_be_bytes(),
                    &[],
                ],
                0,
            )
            .unwrap();
        wait_for(&index, &prompt, 528).await;
        drop(handle);
    }

    #[tokio::test]
    async fn subscriber_replays_then_applies_live_events_and_retires() {
        use crate::kv_index::{AllBlocksCleared, KVEvent};

        let ctx = Arc::new(zmq::Context::new());
        let publisher = ctx.socket(zmq::XPUB).unwrap();
        let replay = ctx.socket(zmq::ROUTER).unwrap();
        publisher.set_linger(0).unwrap();
        replay.set_linger(0).unwrap();
        publisher.bind("inproc://hybrid-test-pub").unwrap();
        replay.bind("inproc://hybrid-test-replay").unwrap();
        let index = Arc::new(qwen_index(
            RoutableTarget {
                instance_id: Arc::from("worker"),
                dp_rank: 0,
            },
            8,
        ));
        let stream = Arc::new(EventStream::new(index.clone()));
        let source = EventSource {
            source: Arc::from("worker"),
            dp_rank: 0,
            pub_endpoint: "inproc://hybrid-test-pub".into(),
            replay_endpoint: Some("inproc://hybrid-test-replay".into()),
            topic: "epoch-1".into(),
            hwm: None,
        };
        let handle = spawn_zmq(source, stream, ctx.clone()).unwrap();
        let subscription = recv(&publisher).await;
        assert_eq!(subscription[0], [b"\x01".as_slice(), b"epoch-1"].concat());
        let request = recv(&replay).await;
        let payload = &crate::kv_index::source::binding::tests::qwen35_payload();
        let seq0 = 0_i64.to_be_bytes();
        let end = (-1_i64).to_be_bytes();
        assert_eq!(request.len(), 3);
        replay
            .send_multipart(
                [
                    request[0].as_slice(),
                    &[],
                    b"epoch-1",
                    &seq0,
                    payload.as_slice(),
                ],
                0,
            )
            .unwrap();
        replay
            .send_multipart([request[0].as_slice(), &[], &[], &end, &[]], 0)
            .unwrap();
        let prompt: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
        wait_for(&index, &prompt, 528).await;
        let workers: Vec<Arc<dyn crate::core::Worker>> = (0..2)
            .map(|i| {
                Arc::new(crate::core::BasicWorker::new(
                    format!("http://worker-{i}"),
                    crate::core::WorkerType::Regular,
                )) as Arc<dyn crate::core::Worker>
            })
            .collect();
        let policy = crate::policies::KvAwarePolicy::default();
        policy.register_target(
            workers[0].url(),
            RoutableTarget {
                instance_id: Arc::from("cold"),
                dp_rank: 0,
            },
        );
        policy.register_target_with_cache_identity(
            workers[1].url(),
            RoutableTarget {
                instance_id: Arc::from("worker"),
                dp_rank: 0,
            },
            crate::kv_index::CacheKey {
                model: Arc::from("qwen"),
                block_size: 528,
            },
            index.clone(),
        );
        assert_eq!(
            policy.select_worker_with_token_ids(&workers, Some(&prompt)),
            Some(1)
        );
        let clear = rmp_serde::to_vec_named(&(
            2.0_f64,
            vec![KVEvent::AllBlocksCleared(AllBlocksCleared::default())],
            Some(0_u32),
        ))
        .unwrap();
        publisher
            .send_multipart([b"epoch-1".as_slice(), &1_i64.to_be_bytes(), &clear], 0)
            .unwrap();
        wait_for(&index, &prompt, 0).await;
        // After a clear both workers tie; the credit is gone, the tie-break
        // winner is not part of this assertion.
        assert!(policy
            .select_worker_with_token_ids(&workers, Some(&prompt))
            .is_some());
        publisher
            .send_multipart(
                [
                    b"epoch-1".as_slice(),
                    &2_i64.to_be_bytes(),
                    payload.as_slice(),
                ],
                0,
            )
            .unwrap();
        wait_for(&index, &prompt, 528).await;
        // A live gap requires a replay through the observed sequence. An end
        // marker for a truncated history must not make its partial state visible.
        publisher
            .send_multipart(
                [
                    b"epoch-1".as_slice(),
                    &4_i64.to_be_bytes(),
                    payload.as_slice(),
                ],
                0,
            )
            .unwrap();
        wait_for(&index, &prompt, 0).await;
        let truncated = recv(&replay).await;
        replay
            .send_multipart(
                [
                    truncated[0].as_slice(),
                    &[],
                    b"epoch-1",
                    &seq0,
                    payload.as_slice(),
                ],
                0,
            )
            .unwrap();
        replay
            .send_multipart([truncated[0].as_slice(), &[], &[], &end, &[]], 0)
            .unwrap();
        let complete = recv(&replay).await;
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 0);
        for (seq, event) in [
            (0_i64, payload.as_slice()),
            (1, clear.as_slice()),
            (2, payload.as_slice()),
            (3, payload.as_slice()),
            (4, payload.as_slice()),
        ] {
            replay
                .send_multipart(
                    [
                        complete[0].as_slice(),
                        &[],
                        b"epoch-1",
                        &seq.to_be_bytes(),
                        event,
                    ],
                    0,
                )
                .unwrap();
        }
        replay
            .send_multipart([complete[0].as_slice(), &[], &[], &end, &[]], 0)
            .unwrap();
        wait_for(&index, &prompt, 528).await;
        // A restarted publisher that reuses the topic restarts at sequence 0.
        // Treating that as a stale duplicate would keep the dead epoch's cache
        // visible, so credit must be revoked until a new replay is verified.
        publisher
            .send_multipart([b"epoch-1".as_slice(), &0_i64.to_be_bytes(), &clear], 0)
            .unwrap();
        wait_for(&index, &prompt, 0).await;
        let restarted = recv(&replay).await;
        assert_eq!(restarted.len(), 3);
        replay
            .send_multipart([restarted[0].as_slice(), &[], b"epoch-1", &seq0, &clear], 0)
            .unwrap();
        replay
            .send_multipart([restarted[0].as_slice(), &[], &[], &end, &[]], 0)
            .unwrap();
        publisher
            .send_multipart(
                [
                    b"epoch-1".as_slice(),
                    &1_i64.to_be_bytes(),
                    payload.as_slice(),
                ],
                0,
            )
            .unwrap();
        wait_for(&index, &prompt, 528).await;
        // Negative live cursors cannot be retained as an unsigned replay target.
        for invalid_seq in [-1_i64, -2] {
            publisher
                .send_multipart(
                    [
                        b"epoch-1".as_slice(),
                        &invalid_seq.to_be_bytes(),
                        payload.as_slice(),
                    ],
                    0,
                )
                .unwrap();
            wait_for(&index, &prompt, 0).await;
            let request = recv(&replay).await;
            assert_eq!(request[2], 0_u64.to_be_bytes());
            replay
                .send_multipart(
                    [
                        request[0].as_slice(),
                        &[],
                        b"epoch-1",
                        &seq0,
                        payload.as_slice(),
                    ],
                    0,
                )
                .unwrap();
            replay
                .send_multipart([request[0].as_slice(), &[], &[], &end, &[]], 0)
                .unwrap();
            wait_for(&index, &prompt, 528).await;
        }
        handle.shutdown();
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 0);
    }
}
