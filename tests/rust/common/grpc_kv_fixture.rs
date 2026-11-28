//! Single-rank vLLM event publisher with a deterministic startup replay.

use serde_json::json;
use sha2::{Digest, Sha256};
use std::time::Duration;
use vllm_router_rs::kv_index::request_hash::BlockKeyGenerator;
use vllm_router_rs::kv_index::wire::ExternalBlockHash;
use vllm_router_rs::kv_index::{AllBlocksCleared, BlockStored, CacheCapabilities, KVEvent};

pub const TOPIC: &str = "grpc-kv-start";
const XPUB_SUBSCRIBE: u8 = 1;
const REPLAY_END: i64 = -1;

pub struct KvPublisher {
    live: zmq::Socket,
    replay: zmq::Socket,
    seed: u64,
    pub groups: &'static [&'static str],
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
            groups: &["full_attention"],
            endpoint,
            replay_endpoint,
        }
    }

    pub fn layout(&self, model: &str, block_size: u32) -> CacheCapabilities {
        let groups: Vec<_> = self
            .groups
            .iter()
            .enumerate()
            .map(|(index, &kind)| {
                let mut group = json!({"group_idx":index,"kind":kind,"block_size":block_size,
                "prefix_cacheable":true,"prefix_replay_tokens":0,"spec_type":"FullAttentionSpec"});
                if kind == "mamba" {
                    group["spec_type"] = json!("MambaSpec");
                    group["mamba_cache_mode"] = json!("align");
                }
                group
            })
            .collect();
        serde_json::from_value(json!({
            "schema_version": 1,
            "ranks": [{
                "rank": 0, "model": model, "weight_version": "default",
                "publisher_topic": TOPIC, "instance_topic": true,
                "hash_algorithm": "sha256_cbor", "root_hash": self.root_hash_hex(), "full_hashes": true,
                "hash_block_size": block_size, "declared_groups": groups.len(),
                "hash_lookahead_tokens": 0, "speculative_tokens": 0, "prefix_caching": true,
                "groups": groups
            }]
        }))
        .unwrap()
    }

    fn root_hash_hex(&self) -> String {
        use std::fmt::Write;
        let seed = self.seed.to_string();
        // The decimal representation of a u64 uses the CBOR short string form.
        let mut encoded = vec![0x60 | seed.len() as u8];
        encoded.extend_from_slice(seed.as_bytes());
        let mut hex = String::with_capacity(64);
        for byte in Sha256::digest(encoded) {
            write!(hex, "{byte:02x}").unwrap();
        }
        hex
    }

    pub async fn restore(&self, tokens: &[u32], block_size: u32) {
        let events = if tokens.is_empty() {
            vec![KVEvent::AllBlocksCleared(AllBlocksCleared {})]
        } else {
            let block_hashes: Vec<_> = BlockKeyGenerator::new(block_size as usize, self.seed)
                .generate_block_keys(tokens)
                .into_iter()
                .map(|hash| ExternalBlockHash::Bytes(hash.to_vec()))
                .collect();
            self.groups
                .iter()
                .enumerate()
                .map(|(index, &kind)| {
                    KVEvent::BlockStored(BlockStored {
                        extra_keys: None,
                        block_hashes: block_hashes.clone(),
                        parent_block_hash: None,
                        token_ids: tokens.to_vec(),
                        block_size,
                        lora_id: None,
                        medium: Some("GPU".into()),
                        lora_name: None,
                        group_idx: Some(index as u32),
                        kv_cache_spec_kind: Some(kind.into()),
                        kv_cache_spec_sliding_window: None,
                        locality: None,
                        ownership: None,
                        lookahead_token_ids: None,
                    })
                })
                .collect()
        };
        let payload = batch(&events);
        let subscription = frames(&self.live).await;
        let [subscription] = subscription.as_slice() else {
            panic!("expected an XPUB subscription");
        };
        assert_eq!(
            subscription.split_first(),
            Some((&XPUB_SUBSCRIBE, TOPIC.as_bytes()))
        );
        let peer = self.replay_peer(0).await;
        self.reply(&peer, TOPIC.as_bytes(), 0, &payload);
        self.reply(&peer, b"", REPLAY_END, b"");

        // Requesting sequence 1 proves that the Router published sequence 0.
        let peer = self.replay_peer(1).await;
        self.reply(&peer, b"", REPLAY_END, b"");
        self.live
            .send_multipart([TOPIC.as_bytes(), &1_u64.to_be_bytes(), &batch(&[])], 0)
            .unwrap();
    }

    async fn replay_peer(&self, expected_seq: u64) -> Vec<u8> {
        let request = frames(&self.replay).await;
        let [peer, delimiter, sequence]: [Vec<u8>; 3] = request
            .try_into()
            .expect("expected identity, delimiter and sequence");
        assert!(delimiter.is_empty());
        assert_eq!(sequence, expected_seq.to_be_bytes());
        peer
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
