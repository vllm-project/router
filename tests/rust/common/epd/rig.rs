//! The E/P/D rig: a real router in front of mock workers and an EC control
//! recorder. `mock.rs` has the workers, `client.rs` the request and the checks
//! on the router's answer.

use axum::http::{Method, StatusCode};
use axum::response::Response;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use vllm_router_rs::config::{
    ConfigValidator, EpdConfig, KvConnector, PolicyConfig, RouterConfig, RoutingMode,
};
use vllm_router_rs::routers::{RouterFactory, RouterTrait};
use vllm_router_rs::server::AppContext;

pub const CHAT_ROUTE: &str = "/v1/chat/completions";
/// How long a check waits for something that must happen.
pub const WAIT_DURATION: Duration = Duration::from_secs(5);
/// How long a check waits for something that must *not* happen (`QUIET` in the
/// test file). Such a check can miss a late event, but it cannot fail while the
/// behavior is as described.
pub const QUIET_DURATION: Duration = Duration::from_millis(300);

mod mock;
pub use mock::*;

mod client;
pub use client::*;

/// The consumer's EC control address: a ZMQ REP socket on its own thread.
pub struct EcRecorder {
    address: String,
    log: mpsc::UnboundedReceiver<Value>,
    seen: Vec<Value>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl EcRecorder {
    pub fn start() -> Self {
        let (address_tx, address_rx) = std::sync::mpsc::channel();
        let (log_tx, log) = mpsc::unbounded_channel();
        let stop = Arc::new(AtomicBool::new(false));
        let thread = std::thread::spawn({
            let stop = stop.clone();
            move || {
                let context = zmq::Context::new();
                let socket = context.socket(zmq::REP).unwrap();
                socket.set_linger(0).unwrap();
                socket.set_rcvtimeo(20).unwrap();
                socket.set_sndtimeo(1000).unwrap();
                socket.bind("tcp://127.0.0.1:*").unwrap();
                let address = socket.get_last_endpoint().unwrap().unwrap();
                let port: u16 = address.rsplit_once(':').unwrap().1.parse().unwrap();
                address_tx.send(address).unwrap();
                while !stop.load(Ordering::Relaxed) {
                    let message = match socket.recv_bytes(0) {
                        Ok(message) => message,
                        Err(zmq::Error::EAGAIN) => continue,
                        Err(error) => panic!("EC recorder: {error}"),
                    };
                    let message: Value = serde_json::from_slice(&message).unwrap();
                    let result = if message["op"] == "peers" {
                        json!({"ports": [port]})
                    } else {
                        json!({"cancelled": true})
                    };
                    let reply = json!({"ok": true, "result": result}).to_string();
                    socket.send(reply.as_bytes(), 0).unwrap();
                    let _ = log_tx.send(message);
                }
            }
        });
        Self {
            address: address_rx.recv().unwrap(),
            log,
            seen: Vec::new(),
            stop,
            thread: Some(thread),
        }
    }

    /// Moves the messages logged so far into `seen`.
    pub fn take_new(&mut self) {
        while let Ok(message) = self.log.try_recv() {
            self.seen.push(message);
        }
    }

    /// The transfer IDs with a `cancel` in `seen`.
    pub fn cancelled(&self) -> Vec<&str> {
        self.seen
            .iter()
            .filter(|message| message["op"] == "cancel")
            .filter_map(|message| message["transfer_id"].as_str())
            .collect()
    }

    /// Rig assumption: the recorder thread is still running. A thread that
    /// panicked logs nothing more, so without this check `expect_no_cancel`
    /// could pass with a recorder that sees no traffic.
    fn expect_running(&self) {
        assert!(
            self.thread
                .as_ref()
                .is_some_and(|thread| !thread.is_finished()),
            "[rig assumption] the recorder thread stopped, so its log is not complete: {:?}",
            self.seen
        );
    }

    /// `[EC cancel]`: after a `peers` query, every transfer in `ids` gets a
    /// cancel with `abandon: true`, and no other transfer does.
    pub async fn expect_cancel(&mut self, ids: &[String]) {
        let deadline = tokio::time::Instant::now() + WAIT_DURATION;
        loop {
            self.take_new();
            let cancelled = self.cancelled();
            if ids.iter().all(|id| cancelled.contains(&id.as_str())) {
                break;
            }
            match tokio::time::timeout_at(deadline, self.log.recv()).await {
                Ok(Some(message)) => self.seen.push(message),
                _ => {
                    self.expect_running();
                    panic!(
                        "[EC cancel] expected a cancel for each of {ids:?}, the control address got {:?}",
                        self.seen
                    )
                }
            }
        }
        assert_eq!(
            self.seen[0]["op"], "peers",
            "[EC cancel] shard discovery comes first: {:?}",
            self.seen
        );
        for message in self.seen.iter().filter(|message| message["op"] == "cancel") {
            assert_eq!(message["abandon"], true, "[EC cancel] {message}");
            assert!(
                ids.iter().any(|id| message["transfer_id"] == id.as_str()),
                "[EC cancel] unexpected transfer: {message}"
            );
        }
    }

    /// `[EC cancel]`: the control address gets no traffic during `QUIET`.
    pub async fn expect_no_cancel(&mut self) {
        tokio::time::sleep(QUIET_DURATION).await;
        self.take_new();
        assert!(
            self.seen.is_empty(),
            "[EC cancel] expected no EC control traffic, got {:?}",
            self.seen
        );
        self.expect_running();
    }
}

impl Drop for EcRecorder {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    /// One worker (PD) does prefill and decode.
    CombinedPd,
    /// Separate P and D workers (NIXL). P runs, then D.
    SplitPdSequential,
    /// Separate P and D workers (NIXL). P and D run at the same time.
    SplitPdConcurrent,
}

/// A router with Mooncake-push EPD (a consumer EC control address, so prepare
/// creates the cleanup guard), two encoders, the consumer and Decode.
pub struct Rig {
    pub router: Arc<dyn RouterTrait>,
    _context: Arc<AppContext>,
    pub e0: Mock,
    pub e1: Mock,
    /// The EC consumer: PD on `CombinedPd`, else P.
    pub consumer: Mock,
    /// Unused on `CombinedPd`.
    pub decode: Mock,
    pub ec: EcRecorder,
    /// The mocks' server tasks, aborted with the rig.
    pub servers: Vec<JoinHandle<()>>,
}

impl Rig {
    /// Starts the mocks and the recorder and builds the router. On
    /// `SplitPdConcurrent`, also warms the router up, so that the tests run on
    /// the concurrent path from their first request.
    pub async fn new(path: Path) -> Self {
        let ec = EcRecorder::start();
        let e0 = Mock::new("E0", Role::Encoder);
        let e1 = Mock::new("E1", Role::Encoder);
        let (consumer, decode) = match path {
            Path::CombinedPd => (
                Mock::new("PD", Role::PrefillDecode),
                Mock::new("D", Role::Decode),
            ),
            Path::SplitPdSequential => (
                Mock::new("P", Role::Prefill { push: false }),
                Mock::new("D", Role::Decode),
            ),
            Path::SplitPdConcurrent => (
                Mock::new("P", Role::Prefill { push: true }),
                Mock::new("D", Role::Decode),
            ),
        };
        let mut urls = Vec::new();
        let mut servers = Vec::new();
        for mock in [&e0, &e1, &consumer, &decode] {
            let (url, server) = serve(mock).await;
            urls.push(url);
            servers.push(server);
        }
        let [e0_url, e1_url, consumer_url, decode_url] = <[String; 4]>::try_from(urls).unwrap();
        let mode = match path {
            Path::CombinedPd => RoutingMode::Regular {
                worker_urls: vec![consumer_url.clone()],
            },
            Path::SplitPdSequential | Path::SplitPdConcurrent => RoutingMode::VllmPrefillDecode {
                prefill_urls: vec![(consumer_url.clone(), None)],
                decode_urls: vec![decode_url],
                prefill_policy: None,
                decode_policy: None,
                discovery_address: None,
            },
        };
        let config = RouterConfig {
            mode,
            policy: PolicyConfig::Random,
            kv_connector: KvConnector::Nixl,
            epd: Some(EpdConfig {
                encoder_urls: vec![e0_url, e1_url],
                consumer_zmq_addrs: HashMap::from([(consumer_url, ec.address.clone())]),
            }),
            disable_retries: true,
            disable_circuit_breaker: true,
            worker_startup_timeout_secs: 5,
            ..RouterConfig::default()
        };
        ConfigValidator::validate(&config).expect("valid E/P/D config");
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap();
        let context = Arc::new(AppContext::new(config, client, 64, None, vec![]).unwrap());
        let router: Arc<dyn RouterTrait> =
            Arc::from(RouterFactory::create_router(&context).await.unwrap());
        let rig = Self {
            router,
            _context: context,
            e0,
            e1,
            consumer,
            decode,
            ec,
            servers,
        };
        if path == Path::SplitPdConcurrent {
            // The router runs P and D together only after a P reply has given
            // it P's NIXL address. Text only, so the encoders see nothing.
            let warm_up = json!({
                "model": "test",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "warm up"}],
            });
            let response = answer(rig.send(warm_up.clone())).await;
            assert_eq!(response.status(), StatusCode::OK, "[client] warm-up");
            text(response).await;
            rig.expect_concurrent_path(warm_up).await;
            rig.consumer.reset();
            rig.decode.reset();
        }
        rig
    }

    /// Rig assumption: after the warm-up, D gets its request while P is held.
    /// Without this check, concurrent tests could quietly run sequentially.
    async fn expect_concurrent_path(&self, probe: Value) {
        self.consumer.reset();
        self.decode.reset();
        self.consumer.set_mode(Mode::HoldThenReply);
        let client = self.send(probe);
        self.consumer.wait_requests(1).await;
        let decode_reached = self.decode.got_requests(1).await;
        self.consumer.set_mode(Mode::Reply);
        self.consumer.release();
        assert!(
            decode_reached,
            "[rig assumption] D got no request while P was held: after the \
             warm-up, the router did not take the concurrent path"
        );
        let response = answer(client).await;
        assert_eq!(response.status(), StatusCode::OK, "[client] probe");
        text(response).await;
    }

    /// Rig assumption: a new router sends image 0 to E0 and image 1 to E1
    /// (round-robin per item in `EncoderStage::prepare`), so E0's comes first.
    fn expect_encoder_assignment(&self) {
        for (mock, image) in [(&self.e0, "base64,AAAA"), (&self.e1, "base64,BBBB")] {
            let requests = mock.requests();
            assert!(
                requests.len() == 1 && requests[0]["messages"].to_string().contains(image),
                "[rig assumption] {} should get only the image {image}; the encoder \
                 assignment changed: {requests:?}",
                mock.name()
            );
        }
    }

    /// Sends `body` to the router on its own task. The handle stands for the
    /// client: `answer`, `expect_waiting` or `disconnect` it.
    pub fn send(&self, body: Value) -> JoinHandle<Response> {
        let router = self.router.clone();
        tokio::spawn(async move {
            router
                .route_transparent(None, CHAT_ROUTE, &Method::POST, body)
                .await
        })
    }

    /// The transfer IDs that the router gave the two images, as the encoder
    /// requests carry them.
    pub fn transfer_ids(&self) -> Vec<String> {
        self.expect_encoder_assignment();
        let mut ids = Vec::new();
        for request in self.e0.requests().into_iter().chain(self.e1.requests()) {
            let params = &request["ec_transfer_params"];
            assert_eq!(
                params["consumer_zmq"],
                self.ec.address.as_str(),
                "[dispatch] Mooncake push target"
            );
            for item in params["ec_items"].as_array().unwrap() {
                ids.push(item["transfer_id"].as_str().unwrap().to_owned());
            }
        }
        assert_eq!(ids.len(), 2, "[dispatch] one transfer per image: {ids:?}");
        ids
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        for server in &self.servers {
            server.abort();
        }
    }
}
