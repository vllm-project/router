//! Shared event transport, frame parsing and connection readiness.

use std::io;
use std::time::Duration;

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
        let mut received = Vec::new();
        let replay = async {
            tokio::time::sleep(Duration::from_secs(1)).await;
            0
        };
        let result = while_connected(
            &mut monitor,
            replay,
            false,
            &mut receiver,
            b"topic",
            |frames, _topic| {
                received.push(frames);
                false
            },
        )
        .await
        .unwrap();
        assert!(matches!(result, ReplayPoll::Finished(0)));
        assert_eq!(received.len(), 1025);
        for (seq, frames) in (0_i64..).zip(received) {
            assert_eq!(live_sequence(&frames, b"topic"), Ok(Some(seq)));
        }
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

pub(super) enum ReplayPoll<T> {
    Finished(T),
    Disconnected,
    BufferFull,
}

/// Watch the connection and drain the live socket while replay is in progress.
pub(super) async fn while_connected<T>(
    connection: &mut ConnectionMonitor,
    replay: impl std::future::Future<Output = T>,
    watch: bool,
    sub: &mut ReadableSocket,
    topic: &[u8],
    mut queue_live: impl FnMut(Vec<Vec<u8>>, &[u8]) -> bool,
) -> io::Result<ReplayPoll<T>> {
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
                    if queue_live(frames, topic) {
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
