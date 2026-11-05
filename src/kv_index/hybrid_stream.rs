//! Source-generation fencing for an experimental hybrid event stream.
//!
//! A caller must authenticate the publisher, its complete cache layout, and
//! replay/live watermark. A local generation is not a publisher epoch.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::watch;

use super::hybrid_events::{HybridEventError, HybridEventIndex};
use super::subscriber::EventSource;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridSourceLease(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridStreamError {
    StaleLease,
    InvalidState,
    Index(HybridEventError),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Retired,
    Replaying,
    Live,
}

struct SourceState {
    generation: u64,
    phase: Phase,
}

/// Serializes lease changes and index mutations, so a retired subscriber
/// cannot invalidate or publish state belonging to a replacement subscriber.
pub struct HybridEventStream {
    index: Arc<HybridEventIndex>,
    source: Mutex<SourceState>,
}

impl HybridEventStream {
    pub fn new(index: Arc<HybridEventIndex>) -> Self {
        index.invalidate();
        Self {
            index,
            source: Mutex::new(SourceState {
                generation: 0,
                phase: Phase::Retired,
            }),
        }
    }

    /// The caller must first establish that sequence zero represents an empty
    /// cache for the authenticated publisher's current incarnation.
    pub fn begin_replay(&self) -> HybridSourceLease {
        let mut source = self.source.lock();
        source.generation = source
            .generation
            .checked_add(1)
            .expect("source generation exhausted");
        source.phase = Phase::Replaying;
        self.index.begin_replay(0);
        HybridSourceLease(source.generation)
    }

    fn restart_replay(&self, lease: HybridSourceLease) -> Result<(), HybridStreamError> {
        let mut source = self.source.lock();
        Self::check(&source, lease)?;
        if source.phase == Phase::Retired {
            return Err(HybridStreamError::InvalidState);
        }
        source.phase = Phase::Replaying;
        self.index.begin_replay(0);
        Ok(())
    }

    fn invalidate_replay(&self, lease: HybridSourceLease) -> Result<(), HybridStreamError> {
        let mut source = self.source.lock();
        Self::check(&source, lease)?;
        if source.phase == Phase::Retired {
            return Err(HybridStreamError::InvalidState);
        }
        self.index.invalidate();
        source.phase = Phase::Replaying;
        Ok(())
    }

    pub fn apply_payload(
        &self,
        lease: HybridSourceLease,
        seq: u64,
        payload: &[u8],
    ) -> Result<(), HybridStreamError> {
        let mut source = self.source.lock();
        Self::check(&source, lease)?;
        if source.phase == Phase::Retired {
            return Err(HybridStreamError::InvalidState);
        }
        if let Err(error) = self.index.apply_payload(seq, payload) {
            // A rejected batch means the publisher's layout and the declared
            // one disagree; naming the reason is the only way to fix a layout
            // from a live system.
            tracing::warn!(seq, error = ?error, "hybrid event batch rejected");
            source.phase = Phase::Replaying;
            return Err(HybridStreamError::Index(error));
        }
        Ok(())
    }

    /// Publish only after an independently verified replay/live handoff.
    pub fn finish_replay(
        &self,
        lease: HybridSourceLease,
        next_seq: u64,
    ) -> Result<(), HybridStreamError> {
        let mut source = self.source.lock();
        Self::check(&source, lease)?;
        if source.phase != Phase::Replaying {
            return Err(HybridStreamError::InvalidState);
        }
        if let Err(error) = self.index.finish_replay(next_seq) {
            source.phase = Phase::Replaying;
            return Err(HybridStreamError::Index(error));
        }
        source.phase = Phase::Live;
        Ok(())
    }

    pub fn retire(&self, lease: HybridSourceLease) -> Result<(), HybridStreamError> {
        let mut source = self.source.lock();
        Self::check(&source, lease)?;
        self.index.invalidate();
        source.phase = Phase::Retired;
        Ok(())
    }

    fn check(source: &SourceState, lease: HybridSourceLease) -> Result<(), HybridStreamError> {
        if source.generation != lease.0 {
            Err(HybridStreamError::StaleLease)
        } else {
            Ok(())
        }
    }

    fn is_active(&self, lease: HybridSourceLease) -> bool {
        let source = self.source.lock();
        source.generation == lease.0 && source.phase != Phase::Retired
    }
}

/// Owns a live subscription. Dropping it stops the task and retires its view.
pub struct HybridSubscriberHandle {
    shutdown: watch::Sender<bool>,
    stream: Arc<HybridEventStream>,
    lease: HybridSourceLease,
}

impl HybridSubscriberHandle {
    pub fn shutdown(&self) {
        let _ = self.stream.retire(self.lease);
        let _ = self.shutdown.send(true);
    }
}

impl Drop for HybridSubscriberHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Attach to one publisher incarnation using its replay stream. The first
/// live event acts as a subscription barrier.
pub fn spawn_zmq(
    source: EventSource,
    stream: Arc<HybridEventStream>,
    ctx: Arc<zmq::Context>,
) -> Result<HybridSubscriberHandle, &'static str> {
    let replay_endpoint = source
        .replay_endpoint
        .clone()
        .ok_or("replay endpoint required")?;
    if source.topic.is_empty() || source.pub_endpoint.is_empty() {
        return Err("an incarnation-bound topic and publisher are required");
    }
    let (shutdown, shutdown_rx) = watch::channel(false);
    let lease = stream.begin_replay();
    let task_stream = stream.clone();
    tokio::spawn(async move {
        let stream = task_stream;
        let result: Result<(), ()> = async {
            let sub = ctx.socket(zmq::SUB).map_err(|_| ())?;
            sub.set_linger(0).map_err(|_| ())?;
            if let Some(hwm) = source.hwm {
                sub.set_rcvhwm(hwm).map_err(|_| ())?;
            }
            sub.set_subscribe(source.topic.as_bytes()).map_err(|_| ())?;
            sub.connect(&source.pub_endpoint).map_err(|_| ())?;
            tracing::debug!(
                endpoint = %source.pub_endpoint,
                replay = %replay_endpoint,
                topic = %source.topic,
                rank = source.dp_rank,
                "hybrid subscriber connecting"
            );
            // A SUB socket can connect before its subscription reaches the
            // publisher. Do not expose a replay until a live event proves the
            // subscription is active and replay reaches at least that event.
            let mut required_through = loop {
                if stopped(&shutdown_rx) || !stream.is_active(lease) {
                    return Ok(());
                }
                match sub.recv_multipart(zmq::DONTWAIT) {
                    Ok(frames)
                        if frames
                            .first()
                            .is_some_and(|topic| topic != source.topic.as_bytes()) =>
                    {
                        tokio::task::yield_now().await;
                        continue;
                    }
                    Ok(frames) if frames.len() == 3 && frames[1].len() == 8 => {
                        let seq =
                            i64::from_be_bytes(frames[1].as_slice().try_into().map_err(|_| ())?);
                        if seq < 0 {
                            return Err(());
                        }
                        break seq as u64;
                    }
                    Ok(_) => return Err(()),
                    Err(zmq::Error::EAGAIN) => tokio::time::sleep(Duration::from_millis(10)).await,
                    Err(_) => return Err(()),
                }
            };
            tracing::debug!(
                endpoint = %source.pub_endpoint,
                rank = source.dp_rank,
                required_through,
                "hybrid subscription barrier reached"
            );
            loop {
                if stopped(&shutdown_rx) || !stream.is_active(lease) {
                    return Ok(());
                }
                stream.restart_replay(lease).map_err(|_| ())?;
                let next = replay_from_empty(
                    &ctx,
                    &replay_endpoint,
                    &source.topic,
                    &stream,
                    lease,
                    &shutdown_rx,
                    required_through,
                )
                .await;
                let Some(mut next) = next else {
                    tracing::debug!("hybrid replay produced no usable history");
                    stream.invalidate_replay(lease).map_err(|_| ())?;
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                };
                tracing::debug!(next, "hybrid replay complete");
                let mut replaying = true;
                loop {
                    if stopped(&shutdown_rx) || !stream.is_active(lease) {
                        return Ok(());
                    }
                    match sub.recv_multipart(zmq::DONTWAIT) {
                        Ok(frames)
                            if frames
                                .first()
                                .is_some_and(|topic| topic != source.topic.as_bytes()) =>
                        {
                            tokio::task::yield_now().await;
                            continue;
                        }
                        Ok(frames) if frames.len() == 3 && frames[1].len() == 8 => {
                            let seq = i64::from_be_bytes(
                                frames[1].as_slice().try_into().map_err(|_| ())?,
                            );
                            if seq < 0 {
                                return Err(());
                            }
                            let seq = seq as u64;
                            if seq < next {
                                // Frames queued during the replay/live overlap
                                // are harmless duplicates. Once the replay is
                                // published, a lower sequence can only come
                                // from a restarted publisher reusing the topic,
                                // so the whole view belongs to a dead epoch.
                                if replaying {
                                    continue;
                                }
                                required_through = seq;
                                break;
                            }
                            if seq != next || stream.apply_payload(lease, seq, &frames[2]).is_err()
                            {
                                required_through = seq;
                                break;
                            }
                            tracing::debug!(
                                seq,
                                rank = source.dp_rank,
                                "hybrid live batch applied"
                            );
                            next = next.checked_add(1).ok_or(())?;
                        }
                        Ok(_) => return Err(()),
                        Err(zmq::Error::EAGAIN) if replaying => {
                            stream.finish_replay(lease, next).map_err(|_| ())?;
                            replaying = false;
                        }
                        Err(zmq::Error::EAGAIN) => {
                            tokio::time::sleep(Duration::from_millis(10)).await
                        }
                        Err(_) => return Err(()),
                    }
                    tokio::task::yield_now().await;
                }
                stream.invalidate_replay(lease).map_err(|_| ())?;
            }
        }
        .await;
        if result.is_err() {
            tracing::warn!(source = %source.source, "hybrid event subscriber stopped");
        }
        let _ = stream.retire(lease);
    });
    Ok(HybridSubscriberHandle {
        shutdown,
        stream,
        lease,
    })
}

fn stopped(rx: &watch::Receiver<bool>) -> bool {
    *rx.borrow() || rx.has_changed().is_err()
}

async fn replay_from_empty(
    ctx: &zmq::Context,
    endpoint: &str,
    topic: &str,
    stream: &HybridEventStream,
    lease: HybridSourceLease,
    shutdown: &watch::Receiver<bool>,
    required_through: u64,
) -> Option<u64> {
    let dealer = ctx.socket(zmq::DEALER).ok()?;
    dealer.set_linger(0).ok()?;
    dealer.connect(endpoint).ok()?;
    dealer
        .send_multipart([&[][..], &0_u64.to_be_bytes()], zmq::DONTWAIT)
        .ok()?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut next = 0_u64;
    while tokio::time::Instant::now() < deadline && !stopped(shutdown) && stream.is_active(lease) {
        match dealer.recv_multipart(zmq::DONTWAIT) {
            Ok(frames) if frames.len() == 4 && frames[0].is_empty() && frames[2].len() == 8 => {
                let seq = i64::from_be_bytes(frames[2].as_slice().try_into().ok()?);
                if seq == -1 && frames[1].is_empty() && frames[3].is_empty() {
                    if next <= required_through {
                        return None;
                    }
                    return Some(next);
                }
                if seq < 0 || seq as u64 != next || frames[1] != topic.as_bytes() {
                    return None;
                }
                stream.apply_payload(lease, next, &frames[3]).ok()?;
                next = next.checked_add(1)?;
            }
            Ok(_) => return None,
            Err(zmq::Error::EAGAIN) => tokio::time::sleep(Duration::from_millis(1)).await,
            Err(_) => return None,
        }
    }
    None
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::kv_index::hybrid::HybridGroupKind;
    use crate::kv_index::hybrid_events::HybridEventGroup;
    use crate::kv_index::RoutableTarget;

    #[test]
    fn retired_subscriber_cannot_change_replacement_replay() {
        let groups = (0..4)
            .map(|group_idx| HybridEventGroup {
                group_idx,
                block_size: 528,
                kind: if group_idx == 3 {
                    HybridGroupKind::FullAttention
                } else {
                    HybridGroupKind::Checkpoint
                },
            })
            .collect();
        let index = Arc::new(
            HybridEventIndex::new(
                RoutableTarget {
                    instance_id: Arc::from("worker"),
                    dp_rank: 0,
                },
                groups,
                528,
                0,
                8,
            )
            .unwrap(),
        );
        let stream = HybridEventStream::new(index.clone());
        let first = stream.begin_replay();
        let current = stream.begin_replay();
        let payload = &crate::kv_index::hybrid_events::tests::qwen35_payload();
        let prompt: Vec<u32> = (0..1024).map(|i| 100 + i % 100).collect();
        assert_eq!(
            stream.apply_payload(first, 0, payload),
            Err(HybridStreamError::StaleLease)
        );
        stream.apply_payload(current, 0, payload).unwrap();
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 0);
        assert_eq!(stream.retire(first), Err(HybridStreamError::StaleLease));
        stream.finish_replay(current, 1).unwrap();
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 528);
        assert_eq!(
            stream.apply_payload(first, 1, &[0xc1]),
            Err(HybridStreamError::StaleLease)
        );
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 528);
        stream.retire(current).unwrap();
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 0);
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
        let groups = (0..4)
            .map(|group_idx| HybridEventGroup {
                group_idx,
                block_size: 528,
                kind: if group_idx == 3 {
                    HybridGroupKind::FullAttention
                } else {
                    HybridGroupKind::Checkpoint
                },
            })
            .collect();
        let index = Arc::new(
            HybridEventIndex::new(
                RoutableTarget {
                    instance_id: Arc::from("worker"),
                    dp_rank: 0,
                },
                groups,
                528,
                0,
                8,
            )
            .unwrap(),
        );
        let stream = Arc::new(HybridEventStream::new(index.clone()));
        let source = EventSource {
            source: Arc::from("worker"),
            dp_rank: 0,
            pub_endpoint: "inproc://hybrid-test-pub".into(),
            replay_endpoint: Some("inproc://hybrid-test-replay".into()),
            topic: "epoch-1".into(),
            hwm: None,
        };
        let handle = spawn_zmq(source, stream, ctx.clone()).unwrap();
        async fn recv(socket: &zmq::Socket) -> Vec<Vec<u8>> {
            tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    match socket.recv_multipart(zmq::DONTWAIT) {
                        Ok(frames) => break frames,
                        Err(zmq::Error::EAGAIN) => {
                            tokio::time::sleep(Duration::from_millis(1)).await
                        }
                        Err(error) => panic!("socket: {error}"),
                    }
                }
            })
            .await
            .unwrap()
        }
        let subscription = recv(&publisher).await;
        assert_eq!(subscription[0], [b"\x01".as_slice(), b"epoch-1"].concat());
        assert_eq!(
            replay.recv_multipart(zmq::DONTWAIT),
            Err(zmq::Error::EAGAIN)
        );
        let payload = &crate::kv_index::hybrid_events::tests::qwen35_payload();
        let seq0 = 0_i64.to_be_bytes();
        let end = (-1_i64).to_be_bytes();
        publisher
            .send_multipart([b"epoch-1".as_slice(), &seq0, payload.as_slice()], 0)
            .unwrap();
        let request = recv(&replay).await;
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
        async fn wait_for(index: &HybridEventIndex, prompt: &[u32], expected: u64) {
            tokio::time::timeout(Duration::from_secs(3), async {
                while index.reusable_tokens(prompt, usize::MAX) != expected {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            })
            .await
            .unwrap();
        }
        wait_for(&index, &prompt, 528).await;
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
        handle.shutdown();
        assert_eq!(index.reusable_tokens(&prompt, usize::MAX), 0);
    }
}
