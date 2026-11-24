//! Observe one Mooncake Store publisher. Shared object keys alone cannot prove
//! that an engine can restore a prompt, so only aggregate metrics leave here.
use std::{
    collections::HashSet,
    io::Cursor,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use serde::Deserialize;

const MAX_FRAME_BYTES: usize = 1024 * 1024;
const MAX_COPIES: usize = 16_384;
const MAX_KEY_BYTES: usize = 1024;
const MAX_SCOPE_BYTES: usize = 256;
const MAX_EVENTS: usize = 64;

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Kind {
    Stored,
    Removed,
    Cleared,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Medium {
    Cpu,
    Disk,
}

#[derive(Deserialize)]
struct Event {
    event_id: u64,
    event_type: Kind,
    tenant_id: String,
    backend_id: String,
    dp_rank: u32,
    #[serde(deserialize_with = "Option::deserialize")]
    medium: Option<Medium>,
    object_key: Option<String>,
}

type Object = (String, String, String);

#[derive(Default)]
struct Shadow {
    cpu: HashSet<Object>,
    disk: HashSet<Object>,
    cursor: Option<(u64, u64)>,
}

impl Shadow {
    fn reset(&mut self, reason: &'static str) {
        *self = Self::default();
        metrics::counter!("vllm_router_mooncake_shadow_invalidations_total", "reason" => reason)
            .increment(1);
        self.publish_counts();
        metrics::gauge!("vllm_router_mooncake_shadow_state").set(2.0);
    }

    fn publish_counts(&self) {
        metrics::gauge!("vllm_router_mooncake_shadow_copies", "medium" => "cpu")
            .set(self.cpu.len() as f64);
        metrics::gauge!("vllm_router_mooncake_shadow_copies", "medium" => "disk")
            .set(self.disk.len() as f64);
    }

    fn observe(&mut self, seq: u64, data: &[u8], capacity: usize) -> Result<(), &'static str> {
        if data.len() > MAX_FRAME_BYTES {
            return Err("frame");
        }
        let mut reader = Cursor::new(data);
        let (_, events, rank): (i64, Vec<Event>, u32) =
            rmp_serde::from_read(&mut reader).map_err(|_| "payload")?;
        if reader.position() != data.len() as u64 || events.is_empty() || events.len() > MAX_EVENTS
        {
            return Err("payload");
        }
        let first = events[0].event_id;
        let last = events.last().unwrap().event_id;
        if events
            .windows(2)
            .any(|pair| pair[1].event_id <= pair[0].event_id)
        {
            return Err("event_order");
        }
        if let Some((previous_seq, previous_event)) = self.cursor {
            if previous_seq.checked_add(1) != Some(seq) || first <= previous_event {
                // A missed event might be a removal in any tenant on this stream.
                self.reset("sequence");
            }
        }
        let event_count = events.len() as u64;
        for event in events {
            if event.dp_rank != rank
                || event.backend_id.len() > MAX_SCOPE_BYTES
                || event.tenant_id.len() > MAX_SCOPE_BYTES
            {
                return Err("scope");
            }
            match event.event_type {
                Kind::Cleared => {
                    if event.medium.is_some() || event.object_key.is_some() {
                        return Err("payload");
                    }
                    let keep = |key: &Object| key.0 != event.backend_id || key.1 != event.tenant_id;
                    self.cpu.retain(keep);
                    self.disk.retain(keep);
                }
                kind => {
                    let key = event.object_key.ok_or("object_key")?;
                    if key.is_empty() || key.len() > MAX_KEY_BYTES {
                        return Err("object_key");
                    }
                    let at_capacity = self.cpu.len() + self.disk.len() >= capacity;
                    let copies = match event.medium.ok_or("medium")? {
                        Medium::Cpu => &mut self.cpu,
                        Medium::Disk => &mut self.disk,
                    };
                    let object = (event.backend_id, event.tenant_id, key);
                    match kind {
                        Kind::Stored => {
                            if at_capacity && !copies.contains(&object) {
                                return Err("capacity");
                            }
                            copies.insert(object);
                        }
                        Kind::Removed => {
                            copies.remove(&object);
                        }
                        Kind::Cleared => unreachable!(),
                    }
                }
            }
        }
        self.cursor = Some((seq, last));
        metrics::gauge!("vllm_router_mooncake_shadow_state").set(1.0);
        self.publish_counts();
        metrics::counter!("vllm_router_mooncake_shadow_events_total").increment(event_count);
        Ok(())
    }

    fn apply(&mut self, seq: u64, data: &[u8], capacity: usize) {
        if let Err(reason) = self.observe(seq, data, capacity) {
            // Counts are published only after a complete batch or after this reset.
            self.reset(reason);
        }
    }
}

fn receive(sub: &zmq::Socket, state: &mut Shadow) -> zmq::Result<()> {
    let topic = sub.recv_msg(0)?;
    let mut valid = topic.is_empty() && sub.get_rcvmore()?;
    let mut seq = None;
    let mut payload = None;
    let mut part = 1;
    while sub.get_rcvmore()? {
        let frame = sub.recv_msg(0)?;
        match part {
            1 => seq = <[u8; 8]>::try_from(&frame[..]).ok().map(u64::from_be_bytes),
            2 => payload = Some(frame),
            _ => valid = false,
        }
        part += 1;
    }
    match (valid, seq, payload) {
        (true, Some(seq), Some(frame)) => state.apply(seq, &frame, MAX_COPIES),
        _ => state.reset("frame"),
    }
    Ok(())
}

struct Connection {
    sub: zmq::Socket,
    monitor: zmq::Socket,
}

impl Connection {
    fn open(endpoint: &str) -> zmq::Result<Self> {
        let ctx = zmq::Context::new();
        let sub = ctx.socket(zmq::SUB)?;
        sub.set_linger(0)?;
        sub.set_rcvhwm(32)?;
        sub.set_maxmsgsize(MAX_FRAME_BYTES as i64)?;
        sub.set_subscribe(b"")?;
        sub.monitor(
            "inproc://store-monitor",
            zmq::SocketEvent::DISCONNECTED as i32,
        )?;
        let monitor = ctx.socket(zmq::PAIR)?;
        monitor.set_linger(0)?;
        monitor.connect("inproc://store-monitor")?;
        sub.connect(endpoint)?;
        Ok(Self { sub, monitor })
    }

    fn run(&self, state: &mut Shadow, stop: &AtomicBool) -> zmq::Result<()> {
        while !stop.load(Ordering::Relaxed) {
            let mut items = [
                self.monitor.as_poll_item(zmq::POLLIN),
                self.sub.as_poll_item(zmq::POLLIN),
            ];
            if let Err(err) = zmq::poll(&mut items, 100) {
                if err == zmq::Error::EINTR {
                    continue;
                }
                return Err(err);
            }
            if items[0].is_readable() {
                // Drop the old socket and all queued messages on disconnect.
                return Ok(());
            }
            if items[1].is_readable() {
                receive(&self.sub, state)?;
            }
        }
        Ok(())
    }
}

pub(crate) struct Observer {
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Observer {
    pub(crate) fn start(endpoint: String) -> anyhow::Result<Self> {
        let connection = Connection::open(&endpoint)?;
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let handle = thread::Builder::new()
            .name("mooncake-shadow".into())
            .spawn(move || {
                let mut connection = Some(connection);
                let mut state = Shadow::default();
                state.publish_counts();
                metrics::gauge!("vllm_router_mooncake_shadow_state").set(0.0);
                while !worker_stop.load(Ordering::Relaxed) {
                    if let Some(active) = connection.take() {
                        let outcome = active.run(&mut state, &worker_stop);
                        if worker_stop.load(Ordering::Relaxed) {
                            break;
                        }
                        state.reset(if outcome.is_ok() {
                            "disconnect"
                        } else {
                            "transport"
                        });
                    }
                    if !worker_stop.load(Ordering::Relaxed) {
                        thread::sleep(Duration::from_millis(100));
                        connection = Connection::open(&endpoint).ok();
                    }
                }
            })?;
        Ok(Self {
            stop,
            thread: Some(handle),
        })
    }
}

impl Drop for Observer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn event(id: u64, action: &str, tenant: &str, key: &str, medium: &str) -> Value {
        let mut value = json!({
            "event_id": id, "timestamp": 1730000000000_i64, "event_type": action,
            "backend_id": "store", "tenant_id": tenant, "dp_rank": 0,
            "model_name": null, "block_size": null, "additional_salt": null, "lora_name": null,
            "medium": null
        });
        if action != "cleared" {
            value["medium"] = json!(medium);
            value["object_key"] = json!(key);
            value["group_id"] = Value::Null;
            value["seq_hashes"] = json!([]);
            value["base_block_idx"] = Value::Null;
        }
        value
    }

    fn batch(events: &[Value]) -> Vec<u8> {
        rmp_serde::to_vec_named(&(1730000000000_i64, events, 0)).unwrap()
    }

    #[test]
    fn receives_publisher_framing_over_real_zmq() {
        let context = zmq::Context::new();
        let publisher = context.socket(zmq::XPUB).unwrap();
        let subscriber = context.socket(zmq::SUB).unwrap();
        publisher.set_rcvtimeo(2000).unwrap();
        subscriber.set_rcvtimeo(2000).unwrap();
        publisher.bind("inproc://store").unwrap();
        subscriber.set_subscribe(b"").unwrap();
        subscriber.connect("inproc://store").unwrap();
        assert_eq!(publisher.recv_bytes(0).unwrap(), [1]); // subscription barrier

        let state = &mut Shadow::default();
        let send = |seq: u64, events: &[Value], state: &mut Shadow| {
            publisher
                .send_multipart([&[][..], &seq.to_be_bytes(), &batch(events)], 0)
                .unwrap();
            receive(&subscriber, state).unwrap();
        };
        send(
            17,
            &[
                event(90, "stored", "a", "one", "cpu"),
                event(91, "stored", "a", "one", "cpu"),
                event(92, "stored", "a", "one", "disk"),
                event(93, "stored", "b", "one", "disk"),
            ],
            state,
        );
        assert_eq!((state.cpu.len(), state.disk.len()), (1, 2));

        send(
            18,
            &[
                event(94, "removed", "a", "one", "cpu"),
                event(95, "removed", "a", "one", "cpu"),
                event(96, "cleared", "a", "", ""),
            ],
            state,
        );
        assert_eq!((state.cpu.len(), state.disk.len()), (0, 1));
        assert!(state
            .disk
            .contains(&("store".into(), "b".into(), "one".into())));

        send(20, &[event(103, "stored", "c", "two", "cpu")], state);
        assert_eq!((state.cpu.len(), state.disk.len()), (1, 0)); // lost cross-tenant removal
        send(1, &[event(1, "stored", "d", "three", "disk")], state);
        assert_eq!((state.cpu.len(), state.disk.len()), (0, 1)); // publisher restarted

        publisher
            .send_multipart([&[][..], &[1], &batch(&[])], 0)
            .unwrap();
        receive(&subscriber, state).unwrap();
        assert_eq!(
            (state.cpu.len(), state.disk.len(), state.cursor),
            (0, 0, None)
        );
    }

    #[test]
    fn reads_frames_emitted_by_mooncake_store() {
        // Produced with the upstream KVEventPublisher at Mooncake 10e9ecba.
        // The fixture wraps the unmodified three-part messages in a MsgPack array.
        let frames: Vec<(&[u8], &[u8], &[u8])> = rmp_serde::from_slice(include_bytes!(
            "../../tests/rust/fixtures/mooncake_store_frames.msgpack"
        ))
        .unwrap();
        let expected = [(1, 1), (0, 1), (1, 1), (1, 0)];
        assert_eq!(frames.len(), expected.len());
        let mut state = Shadow::default();
        for ((topic, sequence, payload), counts) in frames.into_iter().zip(expected) {
            assert!(topic.is_empty());
            let sequence = u64::from_be_bytes(sequence.try_into().unwrap());
            state.apply(sequence, payload, MAX_COPIES);
            assert_eq!((state.cpu.len(), state.disk.len()), counts);
        }
    }

    #[test]
    fn rejects_whole_batch_on_capacity_or_ambiguous_removal() {
        let state = &mut Shadow::default();
        let stored = event(1, "stored", "", "key", "cpu");
        state.apply(1, &batch(std::slice::from_ref(&stored)), 1);
        state.apply(
            2,
            &batch(&[
                event(2, "stored", "", "key", "cpu"),
                event(3, "stored", "", "other", "cpu"),
            ]),
            1,
        );
        assert!(state.cursor.is_none() && state.cpu.is_empty());
        let mut bad = event(2, "removed", "", "key", "cpu");
        bad.as_object_mut().unwrap().remove("object_key");
        state.apply(4, &batch(&[stored, bad]), 1);
        assert!(state.cursor.is_none() && state.cpu.is_empty());
    }
}
