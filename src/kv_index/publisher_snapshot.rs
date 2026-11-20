//! Experimental #57789 interop: bounded reference normalization, not a second index.
use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rmpv::Value;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::binding::SourceLease;
use super::subscriber::{apply_batch, SubscriberHandle};
use super::wire::ExternalBlockHash;
use super::{
    BlockRemoved, BlockStored, KVEvent, KVEventBatch, KvBlockIndexer, SourceId,
    SourceRecoveryReason,
};
use crate::config::types::{KvSnapshotSource, PublisherSnapshotSource};
use crate::metrics::RouterMetrics;

const LIMIT: usize = 4096;
const BYTES: usize = 1024 * 1024;
const LIVE_TIMEOUT: Duration = Duration::from_secs(3);
const STRICT_REQUEST: &[u8] = b"snapshot-full-attention-v1";

pub(super) fn validate_config(config: &PublisherSnapshotSource) -> Result<(), String> {
    for endpoint in [&config.endpoint, &config.snapshot_endpoint] {
        let url = reqwest::Url::parse(endpoint).map_err(|_| "invalid publisher endpoint")?;
        // tcp is a non-special URL scheme: a literal IPv4 host may be exposed
        // as Domain, not Host::Ipv4. Parse the literal without DNS resolution.
        let local = url
            .host_str()
            .and_then(|host| {
                host.trim_start_matches('[')
                    .trim_end_matches(']')
                    .parse::<std::net::IpAddr>()
                    .ok()
            })
            .is_some_and(|ip| ip.is_loopback());
        if url.scheme() != "tcp"
            || !local
            || url.port().is_none_or(|p| p == 0)
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || !matches!(url.path(), "" | "/")
        {
            return Err("experimental publisher endpoints require literal loopback TCP".into());
        }
    }
    if config.topic.len() > 256 {
        return Err("publisher topic too large".into());
    }
    Ok(())
}

#[derive(Deserialize)]
struct Proof {
    protocol: String,
    reporting: String,
    rank: u32,
    block_size: u32,
    complete: bool,
    live_blocks: usize,
    live_references: usize,
}

#[derive(Default)]
struct NormalizedBudget {
    operations: usize,
    bytes: usize,
}

impl NormalizedBudget {
    fn charge(&mut self, operations: usize, bytes: usize) -> Result<(), &'static str> {
        self.operations += operations;
        self.bytes += bytes;
        if self.operations > LIMIT || self.bytes > BYTES {
            return Err("normalized publisher capacity");
        }
        Ok(())
    }
}

fn hash_bytes(hash: &ExternalBlockHash) -> usize {
    match hash {
        ExternalBlockHash::Int(_) => 8,
        ExternalBlockHash::Bytes(bytes) => bytes.len(),
    }
}

fn description_bytes(
    width: usize,
    hash: &ExternalBlockHash,
    parent: Option<&ExternalBlockHash>,
) -> usize {
    width * std::mem::size_of::<u32>() + hash_bytes(hash) + parent.map_or(0, hash_bytes)
}

struct Normalizer {
    descriptions: HashMap<ExternalBlockHash, BlockStored>,
    references: HashMap<ExternalBlockHash, usize>,
    bytes: usize,
    total: usize,
    block_size: u32,
    rank: u32,
}

impl Normalizer {
    fn new(config: &KvSnapshotSource) -> Self {
        Self {
            descriptions: HashMap::new(),
            references: HashMap::new(),
            bytes: 0,
            total: 0,
            block_size: config.block_size,
            rank: config.rank,
        }
    }

    fn apply(
        &mut self,
        batch: KVEventBatch,
        snapshot: bool,
        budget: &mut NormalizedBudget,
    ) -> Result<KVEventBatch, &'static str> {
        if batch.data_parallel_rank != Some(self.rank) {
            return Err("publisher rank");
        }
        let mut changes = Vec::new();
        for event in batch.events {
            match event {
                KVEvent::AllBlocksCleared(clear) => {
                    budget.charge(1, 0)?;
                    self.descriptions.clear();
                    self.references.clear();
                    self.bytes = 0;
                    self.total = 0;
                    changes.push(KVEvent::AllBlocksCleared(clear));
                }
                KVEvent::BlockStored(store) => {
                    if store.medium.as_deref() != Some("GPU")
                        || store.group_idx != Some(0)
                        || store.ownership.is_some()
                        || store.locality.is_some()
                        || store.lora_id.is_some()
                        || store.lora_name.is_some()
                        || store.block_hashes.is_empty()
                        || store
                            .kv_cache_spec_kind
                            .as_deref()
                            .is_some_and(|kind| kind != "full_attention")
                        || store.kv_cache_spec_sliding_window.is_some()
                        || store.extra_keys.as_ref().is_some_and(|keys| {
                            keys.len() != store.block_hashes.len()
                                || keys.iter().any(|v| !v.is_nil())
                        })
                        || store.block_hashes.iter().collect::<HashSet<_>>().len()
                            != store.block_hashes.len()
                    {
                        return Err("unsupported publisher scope");
                    }
                    let teaching = store.block_size != 0;
                    if teaching && store.kv_cache_spec_kind.as_deref() != Some("full_attention") {
                        return Err("publisher full-attention declaration");
                    }
                    if !snapshot
                        && teaching
                        && store
                            .block_hashes
                            .iter()
                            .any(|hash| self.references.get(hash).is_some_and(|n| *n != 0))
                    {
                        // A full cache report is indistinguishable from another
                        // real reference. Refuse credit rather than inflate it.
                        return Err("publisher repeated full report");
                    }
                    if (teaching
                        && (store.block_size != self.block_size
                            || store.token_ids.len()
                                != store.block_hashes.len() * self.block_size as usize))
                        || (!teaching
                            && (!store.token_ids.is_empty() || store.parent_block_hash.is_some()))
                    {
                        return Err("publisher descriptor shape");
                    }
                    let mut parent = store.parent_block_hash.clone();
                    for (offset, hash) in store.block_hashes.iter().enumerate() {
                        if teaching {
                            if parent
                                .as_ref()
                                .is_some_and(|p| !self.descriptions.contains_key(p))
                            {
                                return Err("unknown publisher parent");
                            }
                            let width = self.block_size as usize;
                            let tokens = &store.token_ids[offset * width..(offset + 1) * width];
                            if let Some(old) = self.descriptions.get(hash) {
                                if old.parent_block_hash != parent || old.token_ids != tokens {
                                    return Err("conflicting publisher descriptor");
                                }
                            } else {
                                self.bytes += description_bytes(width, hash, parent.as_ref())
                                    + hash_bytes(hash);
                                if self.descriptions.len() >= LIMIT || self.bytes > BYTES {
                                    return Err("publisher history capacity");
                                }
                                self.descriptions.insert(
                                    hash.clone(),
                                    BlockStored {
                                        block_hashes: vec![hash.clone()],
                                        parent_block_hash: parent,
                                        token_ids: tokens.to_vec(),
                                        block_size: store.block_size,
                                        extra_keys: None,
                                        medium: store.medium.clone(),
                                        group_idx: store.group_idx,
                                        locality: None,
                                        ownership: None,
                                        lora_id: None,
                                        lora_name: None,
                                        kv_cache_spec_kind: store.kv_cache_spec_kind.clone(),
                                        kv_cache_spec_sliding_window: None,
                                        // Session annotates the request, not index residency.
                                        session_id: None,
                                    },
                                );
                            }
                            parent = Some(hash.clone());
                        }
                        let description = self.descriptions.get(hash).ok_or("unknown reference")?;
                        let count = self.references.entry(hash.clone()).or_default();
                        if self.total >= LIMIT {
                            return Err("publisher reference capacity");
                        }
                        if *count == 0 {
                            budget.charge(
                                1,
                                description_bytes(
                                    description.token_ids.len(),
                                    hash,
                                    description.parent_block_hash.as_ref(),
                                ),
                            )?;
                            changes.push(KVEvent::BlockStored(description.clone()));
                        }
                        *count += 1;
                        self.total += 1;
                    }
                }
                KVEvent::BlockRemoved(remove) => {
                    if remove.medium.as_deref() != Some("GPU")
                        || remove.group_idx != Some(0)
                        || remove.ownership.is_some()
                        || remove.locality.is_some()
                        || remove.block_hashes.is_empty()
                        || remove.block_hashes.iter().collect::<HashSet<_>>().len()
                            != remove.block_hashes.len()
                    {
                        return Err("unsupported publisher removal");
                    }
                    let mut removed = Vec::new();
                    for hash in &remove.block_hashes {
                        let count = self
                            .references
                            .get_mut(hash)
                            .ok_or("unmatched publisher removal")?;
                        if *count == 0 {
                            return Err("unmatched publisher removal");
                        }
                        *count -= 1;
                        self.total -= 1;
                        if *count == 0 {
                            budget.charge(1, hash_bytes(hash))?;
                            removed.push(hash.clone());
                        }
                    }
                    if !removed.is_empty() {
                        changes.push(KVEvent::BlockRemoved(BlockRemoved {
                            block_hashes: removed,
                            ..remove
                        }));
                    }
                }
            }
        }
        let normalized = KVEventBatch {
            ts: batch.ts,
            data_parallel_rank: Some(0),
            events: changes,
        };
        super::admit::validate_full_attention_batch(&normalized, self.block_size)
            .map_err(|_| "normalized admission")?;
        Ok(normalized)
    }
}

// Only this opt-in protocol understands the fourth publisher UUID and empty
// reference Store. Generic decoding/admission remains strict and unchanged.
fn envelope(payload: &[u8]) -> Result<Vec<Value>, &'static str> {
    if payload.len() > BYTES {
        return Err("publisher payload capacity");
    }
    let mut syntax = rmp_serde::Deserializer::new(Cursor::new(payload));
    syntax.set_max_depth(16);
    serde::de::IgnoredAny::deserialize(&mut syntax).map_err(|_| "publisher syntax")?;
    let mut cursor = Cursor::new(payload);
    let Value::Array(fields) =
        rmpv::decode::read_value_with_max_depth(&mut cursor, 16).map_err(|_| "publisher wire")?
    else {
        return Err("publisher array");
    };
    if cursor.position() as usize != payload.len() {
        return Err("publisher envelope");
    }
    Ok(fields)
}

fn decode(payload: &[u8], identity: Option<&[u8]>) -> Result<KVEventBatch, &'static str> {
    let mut fields = envelope(payload)?;
    if fields.len() != if identity.is_some() { 4 } else { 3 } {
        return Err("publisher envelope");
    }
    if let Some(expected) = identity {
        let Value::Binary(actual) = fields.pop().ok_or("publisher identity")? else {
            return Err("publisher identity");
        };
        if actual.len() != 16 || actual != expected {
            return Err("publisher replacement");
        }
    }
    let Some(Value::Array(events)) = fields.get_mut(1) else {
        return Err("publisher events");
    };
    for event in events {
        let Value::Map(pairs) = event else {
            return Err("publisher event map");
        };
        if pairs
            .iter()
            .filter(|(key, _)| key.as_str() == Some("lookahead_token_ids"))
            .count()
            > 1
        {
            return Err("duplicate publisher lookahead");
        }
        if let Some((_, ahead)) = pairs
            .iter()
            .find(|(key, _)| key.as_str() == Some("lookahead_token_ids"))
        {
            if !ahead.is_nil() {
                return Err("publisher lookahead");
            }
        }
        pairs.retain(|(key, _)| key.as_str() != Some("lookahead_token_ids"));
    }
    let mut original_contract = Vec::new();
    rmpv::encode::write_value(&mut original_contract, &Value::Array(fields))
        .map_err(|_| "publisher wire")?;
    super::decode_batch(&original_contract).map_err(|_| "publisher event")
}

fn recv(
    socket: &zmq::Socket,
    deadline: Instant,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<Vec<u8>>, &'static str> {
    while Instant::now() < deadline {
        if cancelled() {
            return Err("publisher cancelled");
        }
        if socket.poll(zmq::POLLIN, 50).map_err(|_| "publisher poll")? == 0 {
            continue;
        }
        let mut parts = Vec::new();
        let mut size = 0;
        loop {
            let frame = socket
                .recv_bytes(zmq::DONTWAIT)
                .map_err(|_| "publisher receive")?;
            size += frame.len();
            if parts.len() >= LIMIT || size > BYTES {
                return Err("publisher multipart capacity");
            }
            parts.push(frame);
            if !socket.get_rcvmore().map_err(|_| "publisher receive")? {
                return Ok(parts);
            }
        }
    }
    Err("publisher deadline")
}

struct Connection {
    _context: zmq::Context,
    live: zmq::Socket,
    snapshot: zmq::Socket,
    publisher: [u8; 16],
    first: Vec<Vec<u8>>,
    first_seq: i64,
}

struct StagedCut {
    watermark: i64,
    plan: Vec<KVEventBatch>,
    state: Normalizer,
    budget: NormalizedBudget,
}

type SeenSequences = HashMap<i64, [u8; 32]>;

impl Connection {
    fn new(config: &KvSnapshotSource, cancelled: &dyn Fn() -> bool) -> Result<Self, &'static str> {
        let protocol = &config.publisher_snapshot;
        let context = zmq::Context::new();
        let live = context.socket(zmq::SUB).map_err(|_| "publisher socket")?;
        live.set_ipv6(true).map_err(|_| "publisher socket")?;
        live.set_linger(0).map_err(|_| "publisher socket")?;
        live.set_rcvhwm(LIMIT as i32)
            .map_err(|_| "publisher socket")?;
        live.set_maxmsgsize(BYTES as i64)
            .map_err(|_| "publisher socket")?;
        live.set_subscribe(protocol.topic.as_bytes())
            .map_err(|_| "publisher socket")?;
        live.connect(&protocol.endpoint)
            .map_err(|_| "publisher connect")?;
        let first = recv(&live, Instant::now() + LIVE_TIMEOUT, cancelled)?;
        if first.len() != 3 || first[0] != protocol.topic.as_bytes() {
            return Err("publisher topic");
        }
        let fields = envelope(&first[2])?;
        let Some(Value::Binary(id)) = fields.get(3) else {
            return Err("missing publisher identity");
        };
        let publisher: [u8; 16] = id.as_slice().try_into().map_err(|_| "publisher identity")?;
        let (first_seq, _) = live_batch(&first, &publisher, config)?;
        let snapshot = context.socket(zmq::REQ).map_err(|_| "publisher socket")?;
        snapshot.set_ipv6(true).map_err(|_| "publisher socket")?;
        snapshot.set_linger(0).map_err(|_| "publisher socket")?;
        snapshot
            .set_sndtimeo(1000)
            .map_err(|_| "publisher socket")?;
        snapshot
            .set_maxmsgsize(BYTES as i64)
            .map_err(|_| "publisher socket")?;
        snapshot
            .connect(&protocol.snapshot_endpoint)
            .map_err(|_| "snapshot connect")?;
        Ok(Self {
            _context: context,
            live,
            snapshot,
            publisher,
            first,
            first_seq,
        })
    }

    fn cut(
        &self,
        config: &KvSnapshotSource,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<StagedCut, &'static str> {
        self.snapshot
            .send(STRICT_REQUEST, 0)
            .map_err(|_| "snapshot request")?;
        let parts = recv(
            &self.snapshot,
            Instant::now() + Duration::from_secs(2),
            cancelled,
        )?;
        let cut = stage(&parts, &self.publisher, config)?;
        if cut.watermark < self.first_seq {
            return Err("snapshot predates live coverage");
        }
        Ok(cut)
    }

    /// Keep this exact subscription through W+1..H and into live ingestion.
    /// Frames covered by the cut are never applied a second time.
    fn catch_up(
        &self,
        cut: &mut StagedCut,
        config: &KvSnapshotSource,
        deadline: Instant,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(i64, SeenSequences), &'static str> {
        let mut cursor = cut.watermark;
        let mut frames = 1;
        let mut bytes = self.first[2].len();
        let mut seen = HashMap::from([(self.first_seq, Sha256::digest(&self.first[2]).into())]);
        loop {
            let live = recv(&self.live, deadline, cancelled)?;
            let (seq, batch) = live_batch(&live, &self.publisher, config)?;
            bytes += live[2].len();
            frames += 1;
            if frames > LIMIT || bytes > BYTES {
                return Err("publisher buffer capacity");
            }
            if seq <= cut.watermark {
                let hash = Sha256::digest(&live[2]).into();
                if seen
                    .insert(seq, hash)
                    .is_some_and(|previous| previous != hash)
                {
                    return Err("conflicting publisher sequence");
                }
            } else if sequence(&mut seen, seq, &mut cursor, &live[2])? {
                cut.plan
                    .push(cut.state.apply(batch, false, &mut cut.budget)?);
            }
            // At least one verified post-cut frame establishes H. Drain the
            // buffered suffix before publishing, including final duplicates.
            if cursor > cut.watermark
                && !self
                    .live
                    .get_events()
                    .map_err(|_| "publisher poll")?
                    .contains(zmq::POLLIN)
            {
                return Ok((cursor, seen));
            }
        }
    }
}

fn stage(
    parts: &[Vec<u8>],
    publisher: &[u8],
    config: &KvSnapshotSource,
) -> Result<StagedCut, &'static str> {
    if parts.len() < 3 || parts[0].len() != 8 || parts[1] != publisher {
        return Err("snapshot identity");
    }
    let watermark = i64::from_be_bytes(
        parts[0]
            .as_slice()
            .try_into()
            .map_err(|_| "snapshot sequence")?,
    );
    if watermark < -1 {
        return Err("snapshot unavailable");
    }
    let proof: Proof = serde_json::from_slice(&parts[2]).map_err(|_| "snapshot proof")?;
    if proof.protocol != "full-attention-v1"
        || proof.reporting != "incremental"
        || !proof.complete
        || proof.rank != config.rank
        || proof.block_size != config.block_size
        || proof.live_blocks > LIMIT
        || proof.live_references > LIMIT
    {
        return Err("snapshot incomplete profile");
    }
    let mut state = Normalizer::new(config);
    let mut plan = Vec::new();
    let mut budget = NormalizedBudget::default();
    for bytes in &parts[3..] {
        plan.push(state.apply(decode(bytes, None)?, true, &mut budget)?);
    }
    if state.references.values().filter(|v| **v > 0).count() != proof.live_blocks
        || state.total != proof.live_references
    {
        return Err("snapshot inventory mismatch");
    }
    Ok(StagedCut {
        watermark,
        plan,
        state,
        budget,
    })
}
fn publisher_id(publisher: &[u8; 16]) -> String {
    publisher.iter().map(|byte| format!("{byte:02x}")).collect()
}

struct CancelProbe(Arc<AtomicBool>);

impl Drop for CancelProbe {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

pub(super) async fn discover(config: KvSnapshotSource) -> Result<String, &'static str> {
    let probe = CancelProbe(Arc::new(AtomicBool::new(false)));
    let flag = probe.0.clone();
    tokio::task::spawn_blocking(move || {
        let cancelled = || flag.load(Ordering::Acquire);
        let connection = Connection::new(&config, &cancelled)?;
        connection.cut(&config, &cancelled)?;
        Ok(publisher_id(&connection.publisher))
    })
    .await
    .map_err(|_| "publisher discovery task")?
}

struct Ingestion {
    indexer: Arc<KvBlockIndexer>,
    source: SourceId,
    lease: SourceLease,
    token: u64,
    published: bool,
    failure: SourceRecoveryReason,
}
impl Drop for Ingestion {
    fn drop(&mut self) {
        self.lease.with_active(|_| {
            self.indexer.fail_snapshot_if_current(
                &self.source,
                0,
                self.token,
                self.published,
                self.failure,
            )
        });
    }
}

pub(super) fn spawn(
    config: KvSnapshotSource,
    remote_source: String,
    indexer: Arc<KvBlockIndexer>,
    source: SourceId,
    lease: SourceLease,
) -> SubscriberHandle {
    let (shutdown_tx, shutdown) = tokio::sync::broadcast::channel(1);
    tokio::task::spawn_blocking(move || {
        let Some((_, token)) = lease.with_active(|_| indexer.begin_replay(&source, 0, -1, true))
        else {
            return;
        };
        let mut owned = Ingestion {
            indexer,
            source,
            lease: lease.clone(),
            token,
            published: false,
            failure: SourceRecoveryReason::ReplayFailed,
        };
        let cancelled = || !lease.is_active() || !shutdown.is_empty() || shutdown.is_closed();
        let mut run = || -> Result<(), &'static str> {
            let mut connection = Connection::new(&config, &cancelled)?;
            if publisher_id(&connection.publisher) != remote_source {
                return Err("publisher replacement");
            }
            let mut cut = connection.cut(&config, &cancelled)?;
            let deadline = Instant::now() + LIVE_TIMEOUT;
            let (mut cursor, mut seen) =
                connection.catch_up(&mut cut, &config, deadline, &cancelled)?;
            connection.first.clear();
            let StagedCut {
                plan, mut state, ..
            } = cut;
            // Staging and publication share the same lease and mutation fence.
            // No await (or socket read) may intervene between publication and
            // adopting the fresh token for live apply and cleanup.
            owned.token = owned
                .lease
                .with_active(|_| {
                    owned
                        .indexer
                        .install_snapshot_if_current(&owned.source, 0, owned.token, || {
                            if cancelled() || Instant::now() >= deadline {
                                return false;
                            }
                            for batch in &plan {
                                apply_batch(
                                    &owned.indexer,
                                    &owned.source,
                                    0,
                                    0,
                                    batch,
                                    Some(config.block_size),
                                )
                                .expect("snapshot batch preflight");
                            }
                            !cancelled() && Instant::now() < deadline
                        })
                })
                .flatten()
                .ok_or_else(|| {
                    if Instant::now() >= deadline {
                        "publisher deadline"
                    } else {
                        "stale publisher install"
                    }
                })?;
            owned.published = true;
            RouterMetrics::record_kv_replay_outcome("success");
            drop(plan);
            let mut expires = Instant::now() + LIVE_TIMEOUT;
            while !cancelled() {
                let live = recv(&connection.live, expires, &cancelled)?;
                let (seq, batch) = live_batch(&live, &connection.publisher, &config)?;
                if !sequence(&mut seen, seq, &mut cursor, &live[2])? {
                    continue;
                }
                let normalized = state.apply(batch, false, &mut NormalizedBudget::default())?;
                let applied = owned
                    .lease
                    .with_active(|_| {
                        owned
                            .indexer
                            .with_current_snapshot(&owned.source, 0, owned.token, || {
                                apply_batch(
                                    &owned.indexer,
                                    &owned.source,
                                    0,
                                    0,
                                    &normalized,
                                    Some(config.block_size),
                                )
                                .expect("normalized batch preflight");
                            })
                    })
                    .flatten()
                    .is_some();
                if !applied {
                    return Err("stale publisher live");
                }
                expires = Instant::now() + LIVE_TIMEOUT;
            }
            Ok(())
        };
        if let Err(reason) = run() {
            if !owned.published {
                let outcome = if cancelled() || reason == "stale publisher install" {
                    "superseded"
                } else {
                    "failed"
                };
                RouterMetrics::record_kv_replay_outcome(outcome);
            }
            owned.failure = match reason {
                "publisher replacement" => SourceRecoveryReason::IncarnationReset,
                "publisher gap" => SourceRecoveryReason::SequenceGap,
                _ => SourceRecoveryReason::ReplayFailed,
            };
            tracing::debug!(source = %owned.source, publisher = %remote_source, rank = config.rank, reason, "publisher snapshot stopped");
        }
    });
    SubscriberHandle { shutdown_tx }
}

fn sequence(
    seen: &mut SeenSequences,
    seq: i64,
    cursor: &mut i64,
    payload: &[u8],
) -> Result<bool, &'static str> {
    let fingerprint: [u8; 32] = Sha256::digest(payload).into();
    if seq <= *cursor {
        return if seen.get(&seq) == Some(&fingerprint) {
            Ok(false)
        } else {
            Err("conflicting publisher sequence")
        };
    }
    if seq != *cursor + 1 {
        return Err("publisher gap");
    }
    *cursor = seq;
    seen.insert(seq, fingerprint);
    seen.remove(&(seq - LIMIT as i64));
    Ok(true)
}

fn live_batch(
    parts: &[Vec<u8>],
    identity: &[u8],
    config: &KvSnapshotSource,
) -> Result<(i64, KVEventBatch), &'static str> {
    if parts.len() != 3
        || parts[0] != config.publisher_snapshot.topic.as_bytes()
        || parts[1].len() != 8
    {
        return Err("publisher frames");
    }
    let seq = i64::from_be_bytes(
        parts[1]
            .as_slice()
            .try_into()
            .map_err(|_| "publisher seq")?,
    );
    if seq < 0 || seq == i64::MAX {
        return Err("publisher seq");
    }
    let batch = decode(&parts[2], Some(identity))?;
    if batch.data_parallel_rank != Some(config.rank) {
        return Err("publisher rank");
    }
    Ok((seq, batch))
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) mod tests {
    use super::*;

    /// Socket peer replaying captured upstream wire; it does not emulate vLLM.
    pub(crate) struct CapturedPublisher {
        pub(crate) config: PublisherSnapshotSource,
        commands: std::sync::mpsc::Sender<Option<Vec<u8>>>,
        thread: Option<std::thread::JoinHandle<()>>,
        publisher: Vec<u8>,
        serve_snapshot: Arc<AtomicBool>,
    }

    impl CapturedPublisher {
        pub(crate) fn new(ipv6: bool) -> Self {
            Self::start(ipv6, true)
        }

        pub(crate) fn paused_ipv4() -> Self {
            Self::start(false, false)
        }

        fn start(ipv6: bool, serve_snapshot: bool) -> Self {
            let context = zmq::Context::new();
            let live = context.socket(zmq::XPUB).unwrap();
            let snapshot = context.socket(zmq::REP).unwrap();
            for socket in [&live, &snapshot] {
                socket.set_ipv6(ipv6).unwrap();
                socket.set_linger(0).unwrap();
                socket.set_sndtimeo(1000).unwrap();
                socket
                    .bind(if ipv6 {
                        "tcp://[::1]:*"
                    } else {
                        "tcp://127.0.0.1:*"
                    })
                    .unwrap();
            }
            let config = PublisherSnapshotSource {
                endpoint: live.get_last_endpoint().unwrap().unwrap(),
                snapshot_endpoint: snapshot.get_last_endpoint().unwrap().unwrap(),
                topic: String::new(),
            };
            let mut parts = self::snapshot("two");
            let publisher = parts[1].clone();
            let id = publisher.clone();
            let (commands, receiver) = std::sync::mpsc::channel::<Option<Vec<u8>>>();
            let serve_snapshot = Arc::new(AtomicBool::new(serve_snapshot));
            let serve = serve_snapshot.clone();
            let thread = std::thread::spawn(move || {
                let _context = context;
                let deadline = Instant::now() + Duration::from_secs(30);
                let mut heartbeat = Instant::now();
                let mut seq = 2i64;
                let empty = wire(vec![], &id);
                while Instant::now() < deadline {
                    let mut items = [
                        live.as_poll_item(zmq::POLLIN),
                        snapshot.as_poll_item(if serve.load(Ordering::Acquire) {
                            zmq::POLLIN
                        } else {
                            zmq::PollEvents::empty()
                        }),
                    ];
                    zmq::poll(&mut items, 20).unwrap();
                    if items[0].is_readable() {
                        let subscription = live.recv_bytes(0).unwrap();
                        if subscription.first() == Some(&1) {
                            seq += 1;
                            live.send_multipart([b"".as_slice(), &seq.to_be_bytes(), &empty], 0)
                                .unwrap();
                        }
                    }
                    if items[1].is_readable() {
                        assert_eq!(snapshot.recv_bytes(0).unwrap(), STRICT_REQUEST);
                        parts[0] = seq.to_be_bytes().to_vec();
                        snapshot.send_multipart(&parts, 0).unwrap();
                    }
                    match receiver.try_recv() {
                        Ok(Some(payload)) => {
                            seq += 1;
                            live.send_multipart([b"".as_slice(), &seq.to_be_bytes(), &payload], 0)
                                .unwrap();
                        }
                        Ok(None) | Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
                        Err(std::sync::mpsc::TryRecvError::Empty) => {}
                    }
                    if Instant::now() >= heartbeat {
                        seq += 1;
                        live.send_multipart([b"".as_slice(), &seq.to_be_bytes(), &empty], 0)
                            .unwrap();
                        heartbeat = Instant::now() + Duration::from_millis(100);
                    }
                }
            });
            Self {
                config,
                commands,
                thread: Some(thread),
                publisher,
                serve_snapshot,
            }
        }

        pub(crate) fn resume_snapshot(&self) {
            self.serve_snapshot.store(true, Ordering::Release);
        }

        pub(crate) fn remove_and_marker(&self, first: u32) {
            let captured: Vec<Vec<u8>> = serde_json::from_str(include_str!(
                "../../tests/rust/fixtures/kv_snapshot/publisher-57789-remove.json"
            ))
            .unwrap();
            let removed = rmpv::decode::read_value(&mut Cursor::new(&captured[2])).unwrap();
            let remove = removed.as_array().unwrap()[1].as_array().unwrap()[0].clone();
            self.publish_with_marker(first, remove);
        }

        pub(crate) fn clear_and_store(&self, first: u32) {
            let clear = Value::Map(vec![("type".into(), "AllBlocksCleared".into())]);
            self.publish_with_marker(first, clear);
        }

        fn publish_with_marker(&self, first: u32, event: Value) {
            let seed: Vec<Vec<u8>> = serde_json::from_str(include_str!(
                "../../tests/rust/fixtures/kv_snapshot/publisher-57789-seed.json"
            ))
            .unwrap();
            let seed = rmpv::decode::read_value(&mut Cursor::new(&seed[2])).unwrap();
            let mut marker = seed.as_array().unwrap()[1].as_array().unwrap()[0].clone();
            let Value::Map(fields) = &mut marker else {
                unreachable!()
            };
            for (key, value) in fields {
                match key.as_str() {
                    Some("block_hashes") => {
                        *value = Value::Array(vec![Value::from(first), Value::from(first + 1)])
                    }
                    Some("token_ids") => {
                        *value = Value::Array((first..first + 32).map(Value::from).collect())
                    }
                    _ => {}
                }
            }
            self.commands
                .send(Some(wire(vec![event, marker], &self.publisher)))
                .unwrap();
        }
    }

    impl Drop for CapturedPublisher {
        fn drop(&mut self) {
            let _ = self.commands.send(None);
            self.thread.take().unwrap().join().unwrap();
        }
    }

    fn wire(events: Vec<Value>, publisher: &[u8]) -> Vec<u8> {
        let value = Value::Array(vec![
            Value::from(1.0),
            Value::Array(events),
            Value::from(0),
            Value::Binary(publisher.to_vec()),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &value).unwrap();
        bytes
    }

    #[test]
    fn publisher_ipv4_exchanges_snapshot_and_live_frames() {
        exchange(false);
    }

    #[test]
    fn publisher_ipv6_exchanges_snapshot_and_live_frames() {
        match std::net::TcpListener::bind("[::1]:0") {
            Ok(listener) => drop(listener),
            Err(error) if error.kind() == std::io::ErrorKind::AddrNotAvailable => {
                eprintln!(
                    "N/A IPv6 transport: OS has no configured ::1 ({error}); not an IPv6 PASS"
                );
                return;
            }
            Err(error) => panic!("IPv6 OS probe failed: {error}"),
        }
        exchange(true);
    }

    fn exchange(ipv6: bool) {
        let peer = CapturedPublisher::new(ipv6);
        if ipv6 {
            let context = zmq::Context::new();
            for (kind, endpoint) in [
                (zmq::SUB, &peer.config.endpoint),
                (zmq::REQ, &peer.config.snapshot_endpoint),
            ] {
                let legacy = context.socket(kind).unwrap();
                legacy.set_linger(0).unwrap();
                legacy.set_sndtimeo(100).unwrap();
                assert!(!legacy.is_ipv6().unwrap());
                if kind == zmq::SUB {
                    legacy.set_subscribe(b"").unwrap();
                }
                let connected = legacy.connect(endpoint);
                eprintln!("IPv6 default-disabled {kind:?} connect: {connected:?}");
                if connected.is_ok() {
                    if kind == zmq::REQ {
                        legacy.send(STRICT_REQUEST, 0).unwrap();
                    }
                    assert_eq!(
                        legacy.poll(zmq::POLLIN, 250).unwrap(),
                        0,
                        "default-disabled IPv6 socket unexpectedly exchanged data"
                    );
                    eprintln!("IPv6 default-disabled {kind:?}: no reply/frame within 250 ms");
                }
            }
        }
        let config = KvSnapshotSource {
            publisher_snapshot: peer.config.clone(),
            ..config()
        };
        let connection = Connection::new(&config, &|| false).unwrap();
        assert!(connection.live.is_ipv6().unwrap());
        assert!(connection.snapshot.is_ipv6().unwrap());
        let StagedCut {
            watermark, state, ..
        } = connection.cut(&config, &|| false).unwrap();
        assert!(watermark >= connection.first_seq);
        assert_eq!(state.total, 4);
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let live = recv(&connection.live, deadline, &|| false).unwrap();
            if live_batch(&live, &connection.publisher, &config).unwrap().0 > watermark {
                break;
            }
        }
        eprintln!(
            "IPv{} SUB + REQ exchange: W={watermark}, references={}",
            if ipv6 { 6 } else { 4 },
            state.total
        );
    }
    fn config() -> KvSnapshotSource {
        KvSnapshotSource {
            namespace: "actual-public-synthetic-b16".into(),
            rank: 0,
            block_size: 16,
            publisher_snapshot: PublisherSnapshotSource {
                endpoint: "tcp://127.0.0.1:5557".into(),
                snapshot_endpoint: "tcp://127.0.0.1:5558".into(),
                topic: String::new(),
            },
        }
    }

    fn snapshot(name: &str) -> Vec<Vec<u8>> {
        serde_json::from_str(match name {
            "one" => {
                include_str!("../../tests/rust/fixtures/kv_snapshot/publisher-57789-baseline.json")
            }
            "two" => include_str!(
                "../../tests/rust/fixtures/kv_snapshot/publisher-57789-two-references.json"
            ),
            "removed" => {
                include_str!("../../tests/rust/fixtures/kv_snapshot/publisher-57789-removed.json")
            }
            _ => panic!("unknown captured snapshot"),
        })
        .unwrap()
    }

    #[test]
    fn actual_publisher_teaching_references_do_not_evict_residency() {
        // Exact multipart bytes from ZmqEventPublisher/recorder/export, with
        // two public B16 token blocks; no synthetic ideal snapshot envelope.
        let config = config();
        for (name, refs) in [("one", 2), ("two", 4), ("removed", 0)] {
            let parts = snapshot(name);
            let StagedCut { state, .. } = stage(&parts, &parts[1], &config).unwrap();
            assert_eq!(state.total, refs, "{name}");
            assert_eq!(
                state.references.values().filter(|n| **n > 0).count(),
                if refs == 0 { 0 } else { 2 }
            );
        }
        let two = snapshot("two");
        let StagedCut { mut state, .. } = stage(&two, &two[1], &config).unwrap();
        let mut budget = NormalizedBudget::default();
        let live: Vec<Vec<u8>> = serde_json::from_str(include_str!(
            "../../tests/rust/fixtures/kv_snapshot/publisher-57789-remove.json"
        ))
        .unwrap();
        let remove = decode(&live[2], Some(&two[1])).unwrap();
        assert!(state
            .apply(remove.clone(), false, &mut budget)
            .unwrap()
            .events
            .is_empty());
        assert_eq!(state.total, 2);
        let final_remove = state.apply(remove, false, &mut budget).unwrap();
        assert!(
            matches!(&final_remove.events[..], [KVEvent::BlockRemoved(event)] if event.block_hashes.len() == 2)
        );
        assert_eq!(state.total, 0);
    }

    #[test]
    fn actual_publisher_proof_and_live_identity_are_fail_closed() {
        let config = config();
        for (endpoint, accepted) in [
            ("tcp://127.0.0.1:5557", true),
            ("tcp://[::1]:5557", true),
            ("tcp://localhost:5557", false),
            ("tcp://192.0.2.1:5557", false),
        ] {
            let source = PublisherSnapshotSource {
                endpoint: endpoint.into(),
                snapshot_endpoint: endpoint.into(),
                topic: "kv".into(),
            };
            assert_eq!(validate_config(&source).is_ok(), accepted, "{endpoint}");
        }
        let parts = snapshot("one");
        for reason in [
            "missing",
            "full-report",
            "subset",
            "wrong-count",
            "truncated",
        ] {
            let mut bad = parts.clone();
            let mut proof: serde_json::Value = serde_json::from_slice(&bad[2]).unwrap();
            match reason {
                "missing" => {
                    bad.remove(2);
                }
                "full-report" => proof["reporting"] = "full".into(),
                "subset" => proof["complete"] = false.into(),
                "wrong-count" => proof["live_references"] = 3.into(),
                "truncated" => {
                    bad[3].pop();
                }
                _ => unreachable!(),
            }
            if reason != "missing" {
                bad[2] = serde_json::to_vec(&proof).unwrap();
            }
            assert!(stage(&bad, &parts[1], &config).is_err(), "{reason}");
        }
        let seed: Vec<Vec<u8>> = serde_json::from_str(include_str!(
            "../../tests/rust/fixtures/kv_snapshot/publisher-57789-seed.json"
        ))
        .unwrap();
        for case in [
            "missing-group",
            "locality",
            "missing-layout",
            "duplicate-hash",
            "extra-count",
        ] {
            let mut batch = decode(&seed[2], Some(&parts[1])).unwrap();
            let KVEvent::BlockStored(store) = &mut batch.events[0] else {
                panic!("actual seed Store");
            };
            match case {
                "missing-group" => store.group_idx = None,
                "locality" => store.locality = Some("LOCAL".into()),
                "missing-layout" => store.kv_cache_spec_kind = None,
                "duplicate-hash" => store.block_hashes[1] = store.block_hashes[0].clone(),
                "extra-count" => store.extra_keys = Some(vec![Value::Nil]),
                _ => unreachable!(),
            }
            assert!(
                Normalizer::new(&config)
                    .apply(batch, false, &mut NormalizedBudget::default())
                    .is_err(),
                "{case}"
            );
        }
        let mut wire = rmpv::decode::read_value(&mut Cursor::new(&seed[2])).unwrap();
        let Value::Array(fields) = &mut wire else {
            unreachable!()
        };
        let Value::Array(events) = &mut fields[1] else {
            unreachable!()
        };
        let Value::Map(pairs) = &mut events[0] else {
            unreachable!()
        };
        pairs.retain(|(key, _)| key.as_str() != Some("lookahead_token_ids"));
        pairs.push((
            "lookahead_token_ids".into(),
            Value::Array(vec![Value::Array(vec![]), Value::Array(vec![])]),
        ));
        let mut altered = Vec::new();
        rmpv::encode::write_value(&mut altered, &wire).unwrap();
        assert!(decode(&altered, Some(&parts[1])).is_err());
        assert!(decode(&seed[2], Some(&[0; 16])).is_err());
        let StagedCut { mut state, .. } = stage(&parts, &parts[1], &config).unwrap();
        assert_eq!(
            state
                .apply(
                    decode(&seed[2], Some(&parts[1])).unwrap(),
                    false,
                    &mut NormalizedBudget::default()
                )
                .unwrap_err(),
            "publisher repeated full report"
        );
    }

    #[test]
    fn cheap_reference_cycles_count_expanded_snapshot_plan_bytes() {
        let mut config = config();
        config.block_size = 1024;
        let captured = snapshot("one");
        let mut cut = decode(&captured[3], None).unwrap();
        cut.events.truncate(1);
        let KVEvent::BlockStored(seed) = &mut cut.events[0] else {
            panic!("captured cut begins with a Store");
        };
        seed.block_hashes.truncate(1);
        seed.extra_keys = None;
        seed.token_ids = vec![7; config.block_size as usize];
        seed.block_size = config.block_size;
        let mut reference = seed.clone();
        reference.token_ids.clear();
        reference.block_size = 0;
        let remove = KVEvent::BlockRemoved(BlockRemoved {
            block_hashes: reference.block_hashes.clone(),
            medium: reference.medium.clone(),
            group_idx: reference.group_idx,
            locality: None,
            ownership: None,
        });
        let change = KVEventBatch {
            ts: cut.ts,
            data_parallel_rank: cut.data_parallel_rank,
            events: vec![remove, KVEvent::BlockStored(reference)],
        };
        let mut state = Normalizer::new(&config);
        let mut budget = NormalizedBudget::default();
        state.apply(cut, true, &mut budget).unwrap();
        let mut accepted = 0;
        for _ in 0..LIMIT / 2 {
            match state.apply(change.clone(), false, &mut budget) {
                Ok(batch) => {
                    assert_eq!(batch.events.len(), 2);
                    accepted += 1;
                }
                Err(reason) => {
                    assert_eq!(reason, "normalized publisher capacity");
                    break;
                }
            }
        }
        assert!(accepted > 0);
        assert!(budget.operations < LIMIT && budget.bytes > BYTES);
    }

    #[test]
    fn staged_message_stream_proves_continuity_or_terminates() {
        let config = config();
        let captured = snapshot("two");
        let publisher: [u8; 16] = captured[1].as_slice().try_into().unwrap();
        let heartbeat = wire(Vec::new(), &publisher);
        let clear = wire(
            vec![Value::Map(vec![("type".into(), "AllBlocksCleared".into())])],
            &publisher,
        );
        let large = vec![0; BYTES + 1];
        for (name, frames, cancel, expected) in [
            (
                "duplicate",
                vec![(3i64, &heartbeat), (3, &heartbeat)],
                false,
                Ok(3),
            ),
            ("gap", vec![(4, &heartbeat)], false, Err("publisher gap")),
            (
                "conflict",
                vec![(3, &heartbeat), (3, &clear)],
                false,
                Err("conflicting publisher sequence"),
            ),
            (
                "oversized",
                vec![(3, &large)],
                false,
                Err("publisher multipart capacity"),
            ),
            ("cancelled", vec![], true, Err("publisher cancelled")),
            ("deadline", vec![], false, Err("publisher deadline")),
        ] {
            let ctx = zmq::Context::new();
            let live = ctx.socket(zmq::PAIR).unwrap();
            let peer = ctx.socket(zmq::PAIR).unwrap();
            let snapshot = ctx.socket(zmq::REQ).unwrap();
            live.bind("inproc://publisher-test").unwrap();
            peer.connect("inproc://publisher-test").unwrap();
            peer.set_linger(0).unwrap();
            let first = vec![vec![], 2i64.to_be_bytes().to_vec(), heartbeat.clone()];
            let connection = Connection {
                _context: ctx,
                live,
                snapshot,
                publisher,
                first,
                first_seq: 2,
            };
            for (seq, payload) in frames {
                peer.send_multipart([b"".as_slice(), &seq.to_be_bytes(), payload], 0)
                    .unwrap();
            }
            let mut cut = stage(&captured, &publisher, &config).unwrap();
            let deadline =
                Instant::now() + Duration::from_millis(if name == "deadline" { 75 } else { 1000 });
            assert_eq!(
                connection
                    .catch_up(&mut cut, &config, deadline, &|| cancel)
                    .map(|(cursor, _)| cursor),
                expected,
                "{name}"
            );
        }
    }
}
