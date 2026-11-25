//! Shared event transport, frame parsing and connection readiness.

use std::collections::{hash_map::Entry, HashMap, VecDeque};
use std::io;
use std::time::Duration;

use super::subscriber::ReplayError;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};

pub(super) const IDLE_REPLAY_INTERVAL: Duration = Duration::from_secs(1);

/// Bound retries while an event source or its replay endpoint is unavailable.
pub(super) fn retry_backoff(failures: u32) -> Duration {
    Duration::from_millis((100u64 << failures.min(6)).min(5_000))
}

/// Own the socket so its signalling descriptor outlives its registration.
pub(in crate::kv_index) enum ReadableSocket {
    #[cfg(unix)]
    Ready(tokio::io::unix::AsyncFd<zmq::Socket>),
    Poll(zmq::Socket),
}

impl ReadableSocket {
    pub(in crate::kv_index) fn new(socket: zmq::Socket) -> Self {
        #[cfg(unix)]
        {
            match tokio::io::unix::AsyncFd::try_with_interest(socket, tokio::io::Interest::READABLE)
            {
                Ok(socket) => Self::Ready(socket),
                Err(error) => {
                    let (socket, error) = error.into_parts();
                    tracing::warn!(%error, "ZMQ fd registration failed; falling back to polling");
                    Self::Poll(socket)
                }
            }
        }
        #[cfg(not(unix))]
        {
            Self::Poll(socket)
        }
    }

    pub(in crate::kv_index) fn socket(&self) -> &zmq::Socket {
        match self {
            #[cfg(unix)]
            Self::Ready(socket) => socket.get_ref(),
            Self::Poll(socket) => socket,
        }
    }

    /// Call after a nonblocking receive returns EAGAIN. ZMQ_FD is an
    /// edge-triggered signal; ZMQ_EVENTS determines whether input is ready.
    pub(in crate::kv_index) async fn wait(&mut self) -> io::Result<()> {
        match self {
            #[cfg(unix)]
            Self::Ready(socket) => loop {
                if socket
                    .get_ref()
                    .get_events()
                    .map_err(io::Error::other)?
                    .contains(zmq::POLLIN)
                {
                    return Ok(());
                }
                let mut ready = socket.readable_mut().await?;
                if ready
                    .get_inner()
                    .get_events()
                    .map_err(io::Error::other)?
                    .contains(zmq::POLLIN)
                {
                    return Ok(());
                }
                ready.clear_ready();
            },
            Self::Poll(_) => {
                tokio::time::sleep(Duration::from_millis(10)).await;
                Ok(())
            }
        }
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn pair(polling: bool) -> (ReadableSocket, zmq::Socket) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let ctx = zmq::Context::new();
        let endpoint = format!("inproc://kv-ready-{}", NEXT.fetch_add(1, Ordering::Relaxed));
        let receiver = ctx.socket(zmq::PAIR).unwrap();
        receiver.set_linger(0).unwrap();
        receiver.bind(&endpoint).unwrap();
        let sender = ctx.socket(zmq::PAIR).unwrap();
        sender.set_linger(0).unwrap();
        sender.connect(&endpoint).unwrap();
        (
            if polling {
                ReadableSocket::Poll(receiver)
            } else {
                ReadableSocket::new(receiver)
            },
            sender,
        )
    }

    #[tokio::test]
    async fn replay_buffers_more_than_1024_live_batches() {
        let ctx = zmq::Context::new();
        let endpoint = "inproc://kv-live-buffer-during-replay";
        let receiver = ctx.socket(zmq::PAIR).unwrap();
        receiver.set_linger(0).unwrap();
        receiver.set_rcvhwm(8192).unwrap();
        receiver.bind(endpoint).unwrap();
        let mut monitor = ConnectionMonitor::new(&ctx, &receiver).unwrap();
        let sender = ctx.socket(zmq::PAIR).unwrap();
        sender.set_linger(0).unwrap();
        sender.set_sndhwm(8192).unwrap();
        sender.connect(endpoint).unwrap();
        for seq in 0_i64..1025 {
            sender
                .send_multipart([b"topic".as_slice(), &seq.to_be_bytes(), b"event"], 0)
                .unwrap();
        }
        let mut receiver = ReadableSocket::new(receiver);
        let mut pending = PendingLive::default();
        let overlap = ReplayFingerprints::default();
        let replay = async {
            tokio::time::sleep(Duration::from_secs(1)).await;
            Ok(0)
        };
        let result = while_connected(
            &mut monitor,
            replay,
            false,
            &mut receiver,
            &mut pending,
            b"topic",
            &overlap,
        )
        .await
        .unwrap();
        assert!(matches!(result, ReplayPoll::Finished(Ok(0))));
        for seq in 0_i64..1025 {
            assert_eq!(
                live_sequence(&pending.pop(b"topic", None, &overlap).unwrap().0, b"topic"),
                Ok(Some(seq))
            );
        }
        assert!(pending.pop(b"topic", None, &overlap).is_none());
    }

    #[tokio::test]
    async fn queued_and_later_messages_wake_without_losing_edges() {
        let (mut receiver, sender) = pair(false);
        // An event arriving before the wait must not be lost.
        sender.send("first", 0).unwrap();
        tokio::time::timeout(Duration::from_secs(1), receiver.wait())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            receiver.socket().recv_bytes(zmq::DONTWAIT).unwrap(),
            b"first"
        );
        // Repeated empty -> ready transitions exercise signal rearming.
        for n in 0..100_u8 {
            assert_eq!(
                receiver.socket().recv_bytes(zmq::DONTWAIT),
                Err(zmq::Error::EAGAIN)
            );
            sender.send(&[n][..], 0).unwrap();
            tokio::time::timeout(Duration::from_secs(1), receiver.wait())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(receiver.socket().recv_bytes(zmq::DONTWAIT).unwrap(), [n]);
        }
    }

    #[test]
    fn replay_fingerprint_stays_until_last_queued_copy_leaves() {
        let mut overlap = ReplayFingerprints::default();
        overlap.track_queued_live_frame(0);
        overlap.track_queued_live_frame(0);
        overlap.record_payload(0, b"old replay").unwrap();

        // A replay reset clears fingerprints while leaving both frames queued.
        overlap.clear_fingerprints();
        assert!(!overlap.matches_payload(0, b"old replay"));
        overlap.record_payload(0, b"new replay").unwrap();
        overlap.entries.lock().recent_by_seq.remove(&0);

        overlap.release_queued_live_frame(0);
        assert!(overlap.matches_payload(0, b"new replay"));
        overlap.release_queued_live_frame(0);
        assert!(!overlap.matches_payload(0, b"new replay"));

        overlap.track_queued_live_frame(1);
        overlap.record_payload(1, b"replay").unwrap();
        overlap.release_queued_live_frame(1);
        assert!(overlap.matches_payload(1, b"replay"));
        assert!(overlap.entries.lock().queued_live_by_seq.is_empty());
    }

    #[tokio::test]
    async fn a_waiting_receiver_wakes_for_a_later_message() {
        let (mut receiver, sender) = pair(false);
        let writer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            sender.send("later", 0).unwrap();
        });
        tokio::time::timeout(Duration::from_secs(1), receiver.wait())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            receiver.socket().recv_bytes(zmq::DONTWAIT).unwrap(),
            b"later"
        );
        writer.join().unwrap();
    }

    #[tokio::test]
    async fn a_burst_does_not_require_one_edge_per_message() {
        let (mut receiver, sender) = pair(false);
        for n in 0..32_u8 {
            sender.send(&[n][..], 0).unwrap();
        }
        for n in 0..32_u8 {
            tokio::time::timeout(Duration::from_secs(1), receiver.wait())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(receiver.socket().recv_bytes(zmq::DONTWAIT).unwrap(), [n]);
        }
    }

    #[tokio::test]
    async fn waiting_can_run_in_a_spawned_send_future() {
        let (mut receiver, sender) = pair(false);
        let task = tokio::spawn(async move {
            receiver.wait().await.unwrap();
            receiver.socket().recv_bytes(zmq::DONTWAIT).unwrap()
        });
        sender.send("spawned", 0).unwrap();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), task)
                .await
                .unwrap()
                .unwrap(),
            b"spawned"
        );
    }

    #[tokio::test]
    async fn polling_fallback_keeps_receiving() {
        let (mut receiver, sender) = pair(true);
        sender.send("poll", 0).unwrap();
        tokio::time::timeout(Duration::from_secs(1), receiver.wait())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            receiver.socket().recv_bytes(zmq::DONTWAIT).unwrap(),
            b"poll"
        );
    }

    #[tokio::test]
    async fn a_disconnected_peer_does_not_keep_readiness_set() {
        let (mut receiver, sender) = pair(false);
        drop(sender);
        assert!(
            tokio::time::timeout(Duration::from_millis(30), receiver.wait())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn receiver_wakes_after_peer_disconnects_and_reconnects() {
        let ctx = zmq::Context::new();
        let endpoint = "inproc://kv-ready-reconnect";
        let socket = ctx.socket(zmq::PAIR).unwrap();
        socket.set_linger(0).unwrap();
        socket.bind(endpoint).unwrap();
        let sender = ctx.socket(zmq::PAIR).unwrap();
        sender.set_linger(0).unwrap();
        sender.connect(endpoint).unwrap();
        let mut receiver = ReadableSocket::new(socket);
        sender.send("before", 0).unwrap();
        tokio::time::timeout(Duration::from_secs(1), receiver.wait())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            receiver.socket().recv_bytes(zmq::DONTWAIT).unwrap(),
            b"before"
        );
        sender.disconnect(endpoint).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(30), receiver.wait())
                .await
                .is_err()
        );
        sender.connect(endpoint).unwrap();
        sender.send("after", 0).unwrap();
        tokio::time::timeout(Duration::from_secs(1), receiver.wait())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            receiver.socket().recv_bytes(zmq::DONTWAIT).unwrap(),
            b"after"
        );
    }

    #[tokio::test]
    async fn cancelled_wait_can_be_rearmed() {
        let (mut receiver, sender) = pair(false);
        assert!(
            tokio::time::timeout(Duration::from_millis(30), receiver.wait())
                .await
                .is_err()
        );
        sender.send("after-cancel", 0).unwrap();
        tokio::time::timeout(Duration::from_secs(1), receiver.wait())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            receiver.socket().recv_bytes(zmq::DONTWAIT).unwrap(),
            b"after-cancel"
        );
    }
}

/// Signals transport continuity for a single SUB socket, independently from
/// engine HTTP health. This cannot detect TCP half-open or PUB HWM drops.
pub(in crate::kv_index) struct ConnectionMonitor {
    socket: ReadableSocket,
    #[cfg(test)]
    endpoint: String,
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(test)]
fn injected_failures() -> &'static std::sync::Mutex<std::collections::HashSet<String>> {
    static FAILURES: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    FAILURES.get_or_init(Default::default)
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(test)]
pub(super) fn fail_next_monitor_drain(endpoint: &str) {
    injected_failures()
        .lock()
        .unwrap()
        .insert(endpoint.to_owned());
}

impl ConnectionMonitor {
    pub(in crate::kv_index) fn new(
        ctx: &zmq::Context,
        socket: &zmq::Socket,
    ) -> Result<Self, zmq::Error> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let endpoint = format!(
            "inproc://kv-monitor-{}",
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let mask = (zmq::SocketEvent::CONNECTED as i32) | (zmq::SocketEvent::DISCONNECTED as i32);
        socket.monitor(&endpoint, mask)?;
        let pair = ctx.socket(zmq::PAIR)?;
        pair.set_linger(0)?;
        pair.connect(&endpoint)?;
        Ok(Self {
            socket: ReadableSocket::new(pair),
            #[cfg(test)]
            endpoint: String::new(),
        })
    }
    /// Whether any disconnect occurred, and the newest connection state observed.
    pub(in crate::kv_index) fn drain(&self) -> Result<(bool, Option<bool>), zmq::Error> {
        #[cfg(test)]
        if injected_failures().lock().unwrap().remove(&self.endpoint) {
            return Err(zmq::Error::EINVAL);
        }
        let mut lost = false;
        let mut last = None;
        loop {
            match self.socket.socket().recv_multipart(zmq::DONTWAIT) {
                Ok(frames) => {
                    if frames.first().is_none_or(|f| f.len() < 2) {
                        return Err(zmq::Error::EINVAL);
                    }
                    let event = u16::from_ne_bytes([frames[0][0], frames[0][1]]);
                    if event == zmq::SocketEvent::DISCONNECTED.to_raw() {
                        lost = true;
                        last = Some(false)
                    } else if event == zmq::SocketEvent::CONNECTED.to_raw() {
                        last = Some(true)
                    }
                }
                Err(zmq::Error::EAGAIN) => return Ok((lost, last)),
                Err(e) => return Err(e),
            }
        }
    }
    pub(in crate::kv_index) async fn wait(&mut self) -> std::io::Result<()> {
        self.socket.wait().await
    }

    pub(in crate::kv_index) fn fresh_sub(
        ctx: &zmq::Context,
        endpoint: &str,
        topic: &str,
        hwm: Option<i32>,
    ) -> Result<(ReadableSocket, Self), zmq::Error> {
        let sub = ctx.socket(zmq::SUB)?;
        sub.set_maxmsgsize(super::MAX_KV_EVENT_BYTES as i64)?;
        sub.set_linger(0)?;
        if let Some(hwm) = hwm {
            sub.set_rcvhwm(hwm)?;
        }
        sub.set_subscribe(topic.as_bytes())?;
        let monitor = Self::new(ctx, &sub)?;
        #[cfg(test)]
        let monitor = {
            let mut monitor = monitor;
            monitor.endpoint = endpoint.to_owned();
            monitor
        };
        sub.connect(endpoint)?;
        Ok((ReadableSocket::new(sub), monitor))
    }
}

#[derive(Default)]
pub(super) struct PendingLive {
    frames: VecDeque<Vec<Vec<u8>>>,
    bytes: usize,
}

pub(super) const MAX_PENDING_LIVE_FRAMES: usize = 4096;

impl PendingLive {
    fn push(
        &mut self,
        frames: Vec<Vec<u8>>,
        topic: &[u8],
        replay_fingerprints: &ReplayFingerprints,
    ) -> bool {
        if let Some(seq) = valid_live_seq(&frames, topic) {
            replay_fingerprints.track_queued_live_frame(seq as u64);
        }
        self.bytes += frames.iter().map(Vec::len).sum::<usize>();
        self.frames.push_back(frames);
        self.frames.len() >= MAX_PENDING_LIVE_FRAMES || self.bytes >= super::MAX_KV_EVENT_BYTES
    }

    pub(super) fn pop(
        &mut self,
        topic: &[u8],
        duplicate_check_through: Option<i64>,
        replay_fingerprints: &ReplayFingerprints,
    ) -> Option<(Vec<Vec<u8>>, bool)> {
        let frames = self.frames.pop_front()?;
        let is_duplicate =
            replay_fingerprints.is_replayed_duplicate(&frames, topic, duplicate_check_through);
        if let Some(seq) = valid_live_seq(&frames, topic) {
            replay_fingerprints.release_queued_live_frame(seq as u64);
        }
        self.bytes -= frames.iter().map(Vec::len).sum::<usize>();
        Some((frames, is_duplicate))
    }

    pub(super) fn discard_all(
        &mut self,
        topic: &[u8],
        replay_fingerprints: &ReplayFingerprints,
    ) -> Option<i64> {
        let mut highest_discarded_seq = None;
        while let Some((frames, _)) = self.pop(topic, None, replay_fingerprints) {
            highest_discarded_seq = highest_discarded_seq.max(valid_live_seq(&frames, topic));
        }
        highest_discarded_seq
    }

    /// Remove invalid frames immediately and superseded frames only after verification.
    pub(super) fn validate_replay_handoff(
        &mut self,
        topic: &[u8],
        replay_clear_seq: Option<i64>,
        last_applied_seq: i64,
        replay_fingerprints: &ReplayFingerprints,
    ) -> bool {
        let mut verified = true;
        self.frames.retain(|frames| {
            if let Some(seq) = valid_live_seq(frames, topic) {
                let superseded = replay_clear_seq.is_some_and(|first| seq < first);
                verified &= seq > last_applied_seq
                    || replay_fingerprints
                        .compare_payload(seq as u64, &frames[2])
                        .unwrap_or(superseded);
                return true;
            }
            verified = false;
            self.bytes -= frames.iter().map(Vec::len).sum::<usize>();
            false
        });
        if let Some(first) = replay_clear_seq.filter(|_| verified) {
            self.frames.retain(|frames| {
                let Some(seq) = valid_live_seq(frames, topic).filter(|&seq| seq < first) else {
                    return true;
                };
                replay_fingerprints.release_queued_live_frame(seq as u64);
                self.bytes -= frames.iter().map(Vec::len).sum::<usize>();
                false
            });
        }
        verified
    }
}

fn valid_live_seq(frames: &[Vec<u8>], topic: &[u8]) -> Option<i64> {
    live_sequence(frames, topic)
        .ok()
        .flatten()
        .filter(|seq| (0..i64::MAX).contains(seq))
}

pub(super) enum ReplayPoll {
    Finished(Result<i64, ReplayError>),
    Disconnected,
    BufferFull,
}

/// Watch the connection and drain the live socket while replay is in progress.
pub(super) async fn while_connected(
    connection: &mut ConnectionMonitor,
    replay: impl std::future::Future<Output = Result<i64, ReplayError>>,
    watch: bool,
    sub: &mut ReadableSocket,
    pending: &mut PendingLive,
    topic: &[u8],
    replay_fingerprints: &ReplayFingerprints,
) -> io::Result<ReplayPoll> {
    tokio::pin!(replay);
    loop {
        tokio::select! {
            result = &mut replay => return Ok(ReplayPoll::Finished(result)),
            result = connection.wait(), if watch => {
                result?;
                let (lost, _) = connection.drain().map_err(io::Error::other)?;
                if lost {
                    return Ok(ReplayPoll::Disconnected);
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(1)) => {
                for _ in 0..64 {
                    let frames = match sub.socket().recv_multipart(zmq::DONTWAIT) {
                        Ok(frames) => frames,
                        Err(zmq::Error::EAGAIN) => break,
                        Err(error) => return Err(io::Error::other(error)),
                    };
                    if frames.first().is_some_and(|frame| frame.as_slice() != topic) {
                        continue;
                    }
                    if pending.push(frames, topic, replay_fingerprints) {
                        return Ok(ReplayPoll::BufferFull);
                    }
                }
            }
        }
    }
}

/// `None` is a different topic; `Err` is a malformed frame for this topic.
pub(in crate::kv_index) fn live_sequence(
    frames: &[Vec<u8>],
    topic: &[u8],
) -> Result<Option<i64>, ()> {
    if frames
        .first()
        .is_some_and(|frame| frame.as_slice() != topic)
    {
        return Ok(None);
    }
    let [_, seq, _] = frames else {
        return Err(());
    };
    Ok(Some(i64::from_be_bytes(
        seq.as_slice().try_into().map_err(|_| ())?,
    )))
}

/// A replay reply carries `[empty, topic, sequence, payload]`.
pub(in crate::kv_index) fn replay_frame(frames: &[Vec<u8>]) -> Option<(i64, &[u8], &[u8])> {
    let [envelope, topic, seq, payload] = frames else {
        return None;
    };
    if !envelope.is_empty() {
        return None;
    }
    Some((
        i64::from_be_bytes(seq.as_slice().try_into().ok()?),
        topic,
        payload,
    ))
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod frame_tests {
    use super::*;

    #[test]
    fn live_topic_and_envelope_are_validated_before_sequence() {
        assert_eq!(live_sequence(&[], b"kv"), Err(()));
        assert_eq!(live_sequence(&[b"other".to_vec()], b"kv"), Ok(None));
        assert_eq!(live_sequence(&[b"kv".to_vec()], b"kv"), Err(()));
        let mut frames = vec![b"kv".to_vec(), 42_i64.to_be_bytes().to_vec(), vec![]];
        assert_eq!(live_sequence(&frames, b"kv"), Ok(Some(42)));
        assert_eq!(live_sequence(&frames[..2], b"kv"), Err(()));
        frames[1].pop();
        assert_eq!(live_sequence(&frames, b"kv"), Err(()));
        frames[1].resize(9, 0);
        assert_eq!(live_sequence(&frames, b"kv"), Err(()));
    }

    #[test]
    fn replay_envelope_and_end_sentinel_are_preserved() {
        let mut frames = vec![vec![], vec![], (-1_i64).to_be_bytes().to_vec(), vec![]];
        assert_eq!(replay_frame(&frames), Some((-1, &[][..], &[][..])));
        assert_eq!(replay_frame(&frames[..3]), None);
        frames[0].push(1);
        assert_eq!(replay_frame(&frames), None);
        frames[0].clear();
        frames[2].pop();
        assert_eq!(replay_frame(&frames), None);
    }
}

/// Recent replay payload fingerprints and fingerprints kept for queued live frames.
/// A queued sequence keeps its fingerprint until its last live copy leaves the queue.
#[derive(Default)]
pub(super) struct ReplayFingerprints {
    entries: Mutex<ReplayFingerprintEntries>,
}

#[derive(Default)]
struct ReplayFingerprintEntries {
    recent_by_seq: HashMap<u64, [u8; 32]>,
    queued_live_by_seq: HashMap<u64, QueuedLiveSequence>,
}

struct QueuedLiveSequence {
    frame_count: usize,
    replay_fingerprint: Option<[u8; 32]>,
}

impl ReplayFingerprintEntries {
    fn fingerprint(&self, seq: u64) -> Option<[u8; 32]> {
        self.queued_live_by_seq
            .get(&seq)
            .and_then(|queued| queued.replay_fingerprint)
            .or_else(|| self.recent_by_seq.get(&seq).copied())
    }
}

#[derive(Debug)]
pub(super) struct ConflictingReplayFingerprint;

impl ReplayFingerprints {
    fn track_queued_live_frame(&self, seq: u64) {
        let mut entries = self.entries.lock();
        let fingerprint = entries.recent_by_seq.get(&seq).copied();
        let queued = entries
            .queued_live_by_seq
            .entry(seq)
            .or_insert(QueuedLiveSequence {
                frame_count: 0,
                replay_fingerprint: fingerprint,
            });
        queued.frame_count += 1;
    }

    fn release_queued_live_frame(&self, seq: u64) {
        if let Entry::Occupied(mut entry) = self.entries.lock().queued_live_by_seq.entry(seq) {
            let queued = entry.get_mut();
            queued.frame_count -= 1;
            if queued.frame_count == 0 {
                entry.remove();
            }
        }
    }

    pub(super) fn record_payload(
        &self,
        seq: u64,
        payload: &[u8],
    ) -> Result<(), ConflictingReplayFingerprint> {
        let fingerprint = Sha256::digest(payload).into();
        let mut entries = self.entries.lock();
        if entries
            .fingerprint(seq)
            .is_some_and(|known| known != fingerprint)
        {
            return Err(ConflictingReplayFingerprint);
        }
        if let Some(queued) = entries.queued_live_by_seq.get_mut(&seq) {
            queued.replay_fingerprint = Some(fingerprint);
        }
        entries.recent_by_seq.insert(seq, fingerprint);
        const RECENT_FINGERPRINT_LIMIT: usize = 4_096;
        if entries.recent_by_seq.len() > RECENT_FINGERPRINT_LIMIT {
            let floor = seq.saturating_sub((RECENT_FINGERPRINT_LIMIT / 2) as u64);
            entries.recent_by_seq.retain(|known, _| *known >= floor);
        }
        Ok(())
    }

    pub(super) fn is_replayed_duplicate(
        &self,
        frames: &[Vec<u8>],
        topic: &[u8],
        duplicate_check_through: Option<i64>,
    ) -> bool {
        duplicate_check_through.is_some_and(|last_applied_seq| {
            valid_live_seq(frames, topic).is_some_and(|seq| {
                seq <= last_applied_seq && self.matches_payload(seq as u64, &frames[2])
            })
        })
    }

    fn matches_payload(&self, seq: u64, payload: &[u8]) -> bool {
        self.compare_payload(seq, payload) == Some(true)
    }

    /// Return `None` when no fingerprint exists for this sequence.
    fn compare_payload(&self, seq: u64, payload: &[u8]) -> Option<bool> {
        let fingerprint = self.entries.lock().fingerprint(seq)?;
        Some(fingerprint == <[u8; 32]>::from(Sha256::digest(payload)))
    }

    pub(super) fn clear_fingerprints(&mut self) {
        let entries = self.entries.get_mut();
        if !entries.recent_by_seq.is_empty() {
            entries.recent_by_seq = HashMap::new();
            // Keep the live frame counts for the next replay.
            for queued in entries.queued_live_by_seq.values_mut() {
                queued.replay_fingerprint = None;
            }
        }
    }
}
