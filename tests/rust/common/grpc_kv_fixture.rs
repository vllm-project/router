//! Single-rank vLLM event publisher with a deterministic startup replay.

use serde_json::json;
use std::time::Duration;
use vllm_router_rs::kv_index::request_hash::BlockKeyGenerator;
use vllm_router_rs::kv_index::wire::ExternalBlockHash;
use vllm_router_rs::kv_index::{AllBlocksCleared, BlockStored, CacheCapabilities, KVEvent};

pub const TOPIC: &str = "grpc-kv-start";

pub struct KvPublisher {
    live: zmq::Socket,
    replay: zmq::Socket,
    seed: u64,
    pub endpoint: String,
    pub replay_endpoint: String,
}

impl KvPublisher {
    pub fn bind() -> Self {
        Self::bind_with_seed(0)
    }

    pub fn bind_with_seed(seed: u64) -> Self {
        let context = zmq::Context::new();
        let (live, endpoint) = bind_socket(&context, zmq::XPUB);
        let (replay, replay_endpoint) = bind_socket(&context, zmq::ROUTER);
        Self {
            live,
            replay,
            seed,
            endpoint,
            replay_endpoint,
        }
    }

    pub fn layout(&self, model: &str, block_size: u32) -> CacheCapabilities {
        serde_json::from_value(json!({
            "schema_version": 1,
            "ranks": [{
                "rank": 0, "model": model, "weight_version": "default",
                "publisher_topic": TOPIC, "instance_topic": true,
                "hash_algorithm": "sha256_cbor", "hash_seed": self.seed.to_string(), "full_hashes": true,
                "hash_block_size": block_size, "declared_groups": 1,
                "hash_lookahead_tokens": 0, "speculative_tokens": 0, "prefix_caching": true,
                "groups": [{
                    "group_idx": 0, "kind": "full_attention", "spec_type": "FullAttentionSpec",
                    "block_size": block_size, "prefix_cacheable": true, "prefix_replay_tokens": 0
                }]
            }]
        }))
        .unwrap()
    }

    pub async fn restore(&self, tokens: &[u32], block_size: u32) {
        let event = if tokens.is_empty() {
            KVEvent::AllBlocksCleared(AllBlocksCleared {})
        } else {
            let block_hashes = BlockKeyGenerator::new(block_size as usize, self.seed)
                .generate_block_keys(tokens)
                .into_iter()
                .map(|hash| ExternalBlockHash::Bytes(hash.to_vec()))
                .collect();
            KVEvent::BlockStored(BlockStored {
                extra_keys: None,
                block_hashes,
                parent_block_hash: None,
                token_ids: tokens.to_vec(),
                block_size,
                lora_id: None,
                medium: Some("GPU".into()),
                lora_name: None,
                group_idx: Some(0),
                kv_cache_spec_kind: Some("full_attention".into()),
                kv_cache_spec_sliding_window: None,
                locality: None,
                ownership: None,
                lookahead_token_ids: None,
            })
        };
        let payload = batch(&[event]);
        let subscription = frames(&self.live).await;
        assert_eq!(subscription[0], [b"\x01", TOPIC.as_bytes()].concat());
        let request = frames(&self.replay).await;
        assert_eq!(request[2], 0_u64.to_be_bytes());
        self.reply(&request[0], TOPIC.as_bytes(), 0, &payload);
        self.reply(&request[0], b"", -1, b"");

        // Requesting sequence 1 proves that the Router published sequence 0.
        let request = frames(&self.replay).await;
        assert_eq!(request[2], 1_u64.to_be_bytes());
        self.reply(&request[0], b"", -1, b"");
        self.live
            .send_multipart([TOPIC.as_bytes(), &1_u64.to_be_bytes(), &batch(&[])], 0)
            .unwrap();
    }

    fn reply(&self, identity: &[u8], topic: &[u8], seq: i64, payload: &[u8]) {
        self.replay
            .send_multipart([identity, b"", topic, &seq.to_be_bytes(), payload], 0)
            .unwrap();
    }
}

fn bind_socket(context: &zmq::Context, kind: zmq::SocketType) -> (zmq::Socket, String) {
    let socket = context.socket(kind).unwrap();
    socket.set_linger(0).unwrap();
    socket.bind("tcp://127.0.0.1:*").unwrap();
    let endpoint = socket.get_last_endpoint().unwrap().unwrap();
    (socket, endpoint)
}

fn batch(events: &[KVEvent]) -> Vec<u8> {
    rmp_serde::to_vec_named(&(1.0_f64, events, 0_u32)).unwrap()
}

async fn frames(socket: &zmq::Socket) -> Vec<Vec<u8>> {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match socket.recv_multipart(zmq::DONTWAIT) {
                Ok(frames) => return frames,
                Err(zmq::Error::EAGAIN) => tokio::time::sleep(Duration::from_millis(2)).await,
                Err(error) => panic!("KV publisher socket: {error}"),
            }
        }
    })
    .await
    .expect("Router did not contact KV publisher")
}
