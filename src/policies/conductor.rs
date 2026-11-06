//! External KV-placement provider backed by the Mooncake Conductor.
//!
//! This is the first slice of the Conductor-integration RFC
//! (vllm-project/router#414). It implements the existing synchronous
//! [`TieredMatchProvider`] port so that `kv_aware` reuses its load accounting,
//! scorer and picker untouched.
//!
//! Conductor is a standalone global prefix-cache indexer: it consumes KV
//! events from each engine over ZMQ, keeps a tiered
//! (device / host / shared-pool / disk) prefix table, and answers `/query`
//! with "how many prefix tokens does each instance hold at each tier".
//!
//! ## Query protocol (msgpack, `application/msgpack`)
//!
//! Request to `POST {base_url}/query`:
//! ```text
//! { "model": <str>, "block_size": <u32>,
//!   "token_ids": <bin of little-endian int32> | <array of int> }
//! ```
//! Response:
//! ```text
//! { "instances": {
//!     "<instance_id>": {
//!       "longest_matched": <i64>,
//!       "dp": { "<rank>": <i64 tokens> },
//!       "rank_matches": {
//!         "<rank>": { "device":<i64>, "cpu_local":<i64>,
//!                     "cpu_share":<i64>, "disk":<i64> } },
//!       "device":<i64>, "cpu_local":<i64>, "cpu_share":<i64>, "disk":<i64>
//!     } } }
//! ```
//! The per-tier token counts are **cumulative** ceilings
//! (`device <= cpu_local <= cpu_share <= disk`). This provider converts them
//! to held block intervals (floor-division by `block_size`) and emits one flat
//! `(target, tier)` row per DP rank, matching the shape `KvBlockIndexer`
//! produces, so the `kv_aware` cost model's two-segment (and future chained)
//! reuse rule applies unchanged.
//!
//! ## Degradation
//! Any transport or contract failure is treated as "no placement credit": the
//! provider returns an empty match set, so `kv_aware` falls back to pure load
//! scoring and never fails a request because Conductor is unreachable.

use std::collections::HashSet;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use reqwest::blocking::Client;
use reqwest::header::{HeaderValue, CONTENT_TYPE};

use crate::config::types::ConductorHashProfile;
use crate::kv_index::{Locality, MatchQuery, RoutableTarget, StorageTier, TierMatch, WorkerKvInfo};
use crate::metrics::ConductorMetrics;
use crate::policies::{MatchError, TieredMatchProvider};
use rmpv::Value;

/// Practical ceiling so a misbehaving Conductor cannot stall selection.
const DEFAULT_QUERY_TIMEOUT: Duration = Duration::from_millis(10);
/// Control-path budget for `/register` and `/unregister`. These calls create or
/// tear down a ZMQ subscription on the Conductor, so they need far more than the
/// request-path budget.
const DEFAULT_REGISTER_TIMEOUT: Duration = Duration::from_millis(1000);
/// TCP connect budget, shared by both clients.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_millis(100);

/// Everything needed to construct a [`ConductorMatchProvider`].
#[derive(Debug, Clone)]
pub struct ConductorOptions {
    pub url: String,
    pub connect_timeout: Duration,
    pub query_timeout: Duration,
    pub register_timeout: Duration,
    pub hash_profile: ConductorHashProfile,
    /// Breaker around `/query`. `None` leaves the provider unwrapped.
    pub circuit_breaker: Option<crate::policies::CircuitBreakerConfig>,
    /// Heartbeat period for registration renewal; `0` disables it.
    pub heartbeat_secs: u64,
}

/// Why a Conductor call failed. A caller with no request yields an empty
/// success, so there is no `NoRequest` variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConductorQueryError {
    /// Transport failure other than a timeout (connect/read/reset).
    Io(String),
    /// The request exceeded its timeout budget.
    Timeout,
    /// Non-2xx HTTP status.
    Status(u16),
    /// Response body was not valid msgpack / not the expected shape.
    Decode(String),
}

impl ConductorQueryError {
    /// Stable, low-cardinality label for `conductor_query_errors_total`.
    pub fn reason(&self) -> &'static str {
        match self {
            ConductorQueryError::Io(_) => "transport",
            ConductorQueryError::Timeout => "timeout",
            ConductorQueryError::Status(_) => "status",
            ConductorQueryError::Decode(_) => "decode",
        }
    }
}

/// Conductor placement provider.
///
/// The base URL is the Conductor HTTP endpoint (e.g. `http://127.0.0.1:13333`).
/// Block size and model come from each query's `request`, so one client serves
/// every model/identity group. It also registers/unregisters workers with
/// Conductor so Conductor can run with zero static configuration.
///
/// Two blocking clients are kept on purpose: the `/query` client is bounded by
/// the request-path budget, while `/register` and `/unregister` get a much
/// larger control-path budget.
#[derive(Debug)]
pub struct ConductorMatchProvider {
    base_url: String,
    client: Client,
    control_client: Client,
    hash_profile: ConductorHashProfile,
    /// Every `(instance_id, dp_rank)` this router successfully registered.
    /// Feeds the `conductor_registered_sources` gauge; re-registering the same
    /// key is idempotent, so it never double counts.
    registered: Mutex<HashSet<(String, u32)>>,
}

fn blocking_client(connect_timeout: Duration, timeout: Duration) -> Client {
    Client::builder()
        .connect_timeout(connect_timeout)
        .timeout(timeout)
        .build()
        .expect("blocking HTTP client")
}

fn classify_io(error: &reqwest::Error) -> ConductorQueryError {
    if error.is_timeout() {
        ConductorQueryError::Timeout
    } else {
        ConductorQueryError::Io(error.to_string())
    }
}

fn read_response(
    response: Result<reqwest::blocking::Response, reqwest::Error>,
) -> Result<Vec<u8>, ConductorQueryError> {
    match response {
        Ok(resp) if resp.status().is_success() => resp
            .bytes()
            .map(|b| b.to_vec())
            .map_err(|e| classify_io(&e)),
        Ok(resp) => Err(ConductorQueryError::Status(resp.status().as_u16())),
        Err(e) => Err(classify_io(&e)),
    }
}

/// Parse the `/services` response into `(instance_id, dp_rank)` pairs.
///
/// `/services` uses the exported `ServiceConfig` field names (capitalized),
/// unlike `/query` (lowercase). Getting this wrong would make reconciliation
/// believe every service is missing, so it is covered by a unit test.
fn parse_services(value: &Value) -> HashSet<(String, u32)> {
    let mut out = HashSet::new();
    let Some(map) = value.as_map() else {
        return out;
    };
    let Some((_, Value::Array(services))) =
        map.iter().find(|(key, _)| key.as_str() == Some("services"))
    else {
        return out;
    };
    for service in services {
        let Some(fields) = service.as_map() else {
            continue;
        };
        let field = |name: &str| {
            fields
                .iter()
                .find(|(key, _)| key.as_str() == Some(name))
                .map(|(_, value)| value)
        };
        let Some(instance_id) = field("InstanceID").and_then(|v| v.as_str()) else {
            continue;
        };
        if instance_id.is_empty() {
            continue;
        }
        let dp_rank = field("DPRank")
            .and_then(|v| v.as_i64())
            .filter(|rank| *rank >= 0)
            .unwrap_or(0) as u32;
        out.insert((instance_id.to_string(), dp_rank));
    }
    out
}

impl ConductorMatchProvider {
    /// Build a provider pointing at Conductor's HTTP base URL.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::with_timeout(base_url, DEFAULT_QUERY_TIMEOUT)
    }

    /// Build a provider whose query/register/connect budgets are all `timeout`.
    /// Convenience for tests; production code uses [`Self::with_options`].
    pub fn with_timeout(base_url: impl Into<String>, timeout: Duration) -> Self {
        Self::with_options(ConductorOptions {
            url: base_url.into(),
            connect_timeout: DEFAULT_CONNECT_TIMEOUT.min(timeout),
            query_timeout: timeout,
            register_timeout: DEFAULT_REGISTER_TIMEOUT,
            hash_profile: ConductorHashProfile::default(),
            circuit_breaker: None,
            heartbeat_secs: 0,
        })
    }

    /// Build a provider with explicit per-path budgets and the engine hash
    /// profile used when registering workers with Conductor.
    pub fn with_options(options: ConductorOptions) -> Self {
        Self {
            base_url: options.url.trim_end_matches('/').to_string(),
            client: blocking_client(options.connect_timeout, options.query_timeout),
            control_client: blocking_client(options.connect_timeout, options.register_timeout),
            hash_profile: options.hash_profile,
            registered: Mutex::new(HashSet::new()),
        }
    }

    fn post(
        &self,
        client: &Client,
        path: &str,
        body: Vec<u8>,
    ) -> Result<Vec<u8>, ConductorQueryError> {
        let response = client
            .post(format!("{}{}", self.base_url, path))
            .header(
                CONTENT_TYPE,
                HeaderValue::from_static("application/msgpack"),
            )
            .body(body)
            .send();
        read_response(response)
    }

    fn get(&self, client: &Client, path: &str) -> Result<Vec<u8>, ConductorQueryError> {
        let response = client
            .get(format!("{}{}", self.base_url, path))
            .header(
                reqwest::header::ACCEPT,
                HeaderValue::from_static("application/msgpack"),
            )
            .send();
        read_response(response)
    }

    /// Every `(instance_id, dp_rank)` the Conductor currently knows about.
    ///
    /// The heartbeat uses it to re-register only what is actually missing
    /// instead of blindly re-posting every service (RFC #414 §4 reconciliation).
    pub fn list_services(&self) -> Result<HashSet<(String, u32)>, ConductorQueryError> {
        let bytes = self.get(&self.control_client, "/services")?;
        let value = rmpv::decode::read_value(&mut std::io::Cursor::new(bytes.as_slice()))
            .map_err(|e| ConductorQueryError::Decode(e.to_string()))?;
        Ok(parse_services(&value))
    }

    /// Register a discovered worker (per DP rank) with Conductor so it can run
    /// with zero static service configuration. Registers each source endpoint
    /// under `instance_id`; a no-op when the worker has no cache identity or no
    /// ZMQ sources. Failures are reported but never fatal to the caller.
    pub fn register_worker(&self, info: &WorkerKvInfo) -> Result<(), String> {
        let Some(cache_key) = info.cache_key.as_ref() else {
            return Ok(());
        };
        let model = cache_key.model.to_string();
        let block_size = info.block_size.unwrap_or(cache_key.block_size);
        if block_size == 0 {
            return Ok(());
        }
        let mut last_error = String::new();
        for src in &info.sources {
            if src.pub_endpoint.is_empty() {
                continue;
            }
            let body = encode_register(
                &self.hash_profile,
                &info.instance_id,
                src.dp_rank,
                &model,
                block_size,
                &src.pub_endpoint,
                src.replay_endpoint.clone(),
            );
            match self.post(&self.control_client, "/register", body) {
                Ok(_) => {
                    self.registered
                        .lock()
                        .unwrap()
                        .insert((info.instance_id.clone(), src.dp_rank));
                }
                Err(error) => last_error = format!("register failed: {error:?}"),
            }
        }
        ConductorMetrics::record_register(last_error.is_empty());
        ConductorMetrics::set_registered_sources(self.registered_sources());
        if last_error.is_empty() {
            Ok(())
        } else {
            Err(last_error)
        }
    }

    /// Unregister a worker (per DP rank) from Conductor.
    pub fn unregister_worker(&self, instance_id: &str, dp_ranks: &[u32]) -> Result<(), String> {
        let mut last_error = String::new();
        for &rank in dp_ranks {
            let body = encode_unregister(instance_id, rank);
            match self.post(&self.control_client, "/unregister", body) {
                Ok(_) => {
                    self.registered
                        .lock()
                        .unwrap()
                        .remove(&(instance_id.to_string(), rank));
                }
                Err(error) => last_error = format!("unregister failed: {error:?}"),
            }
        }
        ConductorMetrics::set_registered_sources(self.registered_sources());
        if last_error.is_empty() {
            Ok(())
        } else {
            Err(last_error)
        }
    }

    fn registered_sources(&self) -> usize {
        self.registered.lock().map(|set| set.len()).unwrap_or(0)
    }

    /// Issue a `/query` against Conductor and translate the response into flat
    /// `(target, tier)` rows. Any error yields an empty set (degrade to
    /// load-only).
    pub fn query(&self, query: &MatchQuery<'_>) -> Vec<TierMatch> {
        self.try_query(query).unwrap_or_default()
    }

    /// Same as [`query`](Self::query), but distinguishes a successful (possibly
    /// empty) answer from a transport/status/decode failure. The circuit-breaker
    /// (a follow-up) needs this to trip on real failures only.
    pub fn try_query(&self, query: &MatchQuery<'_>) -> Result<Vec<TierMatch>, ConductorQueryError> {
        let Some(request) = query.request.as_ref() else {
            return Ok(Vec::new());
        };
        if request.token_ids.is_empty() {
            return Ok(Vec::new());
        }

        let body = encode_query(request);
        let started = Instant::now();
        let result = self.post(&self.client, "/query", body);
        ConductorMetrics::record_query_duration(started.elapsed());

        let bytes = match result {
            Ok(bytes) => bytes,
            Err(error) => {
                ConductorMetrics::record_query_error(error.reason());
                return Err(error);
            }
        };
        let value = match rmpv::decode::read_value(&mut std::io::Cursor::new(bytes.as_slice())) {
            Ok(value) => value,
            Err(e) => {
                let error = ConductorQueryError::Decode(e.to_string());
                ConductorMetrics::record_query_error(error.reason());
                return Err(error);
            }
        };
        Ok(translate_response(
            &value,
            request,
            &query.tiers_of_interest,
        ))
    }
}

impl TieredMatchProvider for ConductorMatchProvider {
    fn find_tiered_matches(&self, query: &MatchQuery<'_>) -> Vec<TierMatch> {
        self.query(query)
    }

    fn try_find_tiered_matches(
        &self,
        query: &MatchQuery<'_>,
    ) -> Result<Vec<TierMatch>, MatchError> {
        self.try_query(query).map_err(|error| match error {
            // A 4xx is deterministic: the backend is alive but rejects this
            // request, so it must not open the breaker for everybody.
            ConductorQueryError::Status(status) if (400..500).contains(&status) => {
                MatchError::Rejected
            }
            _ => MatchError::Unavailable,
        })
    }
}

/// Encode a `/query` request body in msgpack. `token_ids` goes as a binary
/// blob of little-endian int32 (Conductor's preferred compact form).
fn encode_query(request: &crate::kv_index::MatchRequest<'_>) -> Vec<u8> {
    let mut tokens: Vec<u8> = Vec::with_capacity(request.token_ids.len() * 4);
    for &t in request.token_ids {
        tokens.extend_from_slice(&t.to_le_bytes());
    }

    let mut map: Vec<(Value, Value)> = vec![
        (
            Value::String("model".into()),
            Value::String(request.model_id.clone().unwrap_or_default().into()),
        ),
        (
            Value::String("block_size".into()),
            Value::Integer(request.block_size.into()),
        ),
        (Value::String("token_ids".into()), Value::Binary(tokens)),
    ];
    if let Some(salt) = &request.cache_salt {
        if !salt.is_empty() {
            map.push((
                Value::String("cache_salt".into()),
                Value::String(salt.clone().into()),
            ));
        }
    }
    // `lora_name` is part of Conductor's `{tenant, model, lora, block_size}`
    // context key. Sending it is what keeps a LoRA request from matching the
    // base model's prefix and getting credit for a cache it cannot reuse.
    // TODO(RFC #414 §4): the router registers one service per source with the
    // base-model context, so a LoRA request currently finds no context at all
    // (zero hits, never wrong hits). Per-LoRA registration is a follow-up.
    if let Some(lora) = &request.lora_name {
        if !lora.is_empty() {
            map.push((
                Value::String("lora_name".into()),
                Value::String(lora.clone().into()),
            ));
        }
    }

    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &Value::Map(map)).expect("encode query");
    out
}

/// Encode a `/register` request body in msgpack matching Conductor's
/// `ParseServiceConfigRequest` contract. `hash_profile` carries the engine
/// hash recipe; `modelname` must equal what `/query` sends (`model_id`).
fn encode_register(
    profile: &ConductorHashProfile,
    instance_id: &str,
    dp_rank: u32,
    model: &str,
    block_size: u32,
    pub_endpoint: &str,
    replay_endpoint: Option<String>,
) -> Vec<u8> {
    let mut hash_profile = vec![
        (
            Value::String("strategy".into()),
            Value::String(profile.strategy.clone().into()),
        ),
        (
            Value::String("algorithm".into()),
            Value::String(profile.algorithm.clone().into()),
        ),
        (
            Value::String("index_projection".into()),
            Value::String(profile.index_projection.clone().into()),
        ),
    ];
    // "Unset" is expressed by *omitting* the field, never by sending an empty
    // string (RFC #414 §4). Conductor currently still requires it for
    // `vllm_v1`, which the config validator enforces; once Conductor accepts an
    // absent seed it will apply vLLM's "vllm-none-hash" root.
    if let Some(seed) = &profile.python_hash_seed {
        hash_profile.push((
            Value::String("python_hash_seed".into()),
            Value::String(seed.clone().into()),
        ));
    }

    let mut fields: Vec<(Value, Value)> = vec![
        (
            Value::String("endpoint".into()),
            Value::String(pub_endpoint.into()),
        ),
        (Value::String("type".into()), Value::String("vLLM".into())),
        (
            Value::String("modelname".into()),
            Value::String(model.into()),
        ),
        (
            Value::String("instance_id".into()),
            Value::String(instance_id.into()),
        ),
        (
            Value::String("block_size".into()),
            Value::Integer(block_size.into()),
        ),
        (
            Value::String("dp_rank".into()),
            Value::Integer(dp_rank.into()),
        ),
        (
            Value::String("hash_profile".into()),
            Value::Map(hash_profile),
        ),
    ];
    if let Some(replay) = replay_endpoint {
        if !replay.is_empty() {
            fields.push((
                Value::String("replay_endpoint".into()),
                Value::String(replay.into()),
            ));
        }
    }

    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &Value::Map(fields)).expect("encode register");
    out
}

/// Encode a `/unregister` request body matching Conductor's handler contract.
fn encode_unregister(instance_id: &str, dp_rank: u32) -> Vec<u8> {
    let map: Vec<(Value, Value)> = vec![
        (
            Value::String("instance_id".into()),
            Value::String(instance_id.into()),
        ),
        (
            Value::String("dp_rank".into()),
            Value::Integer(dp_rank.into()),
        ),
    ];
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &Value::Map(map)).expect("encode unregister");
    out
}

/// Translate Conductor's `/query` response into flat `(target, tier)` rows.
///
/// Per-instance per-rank cumulative token ceilings are floored to block
/// intervals; an invariant violation (descending ceilings, or a ceiling past
/// the query length) drops the impacted rows rather than over-crediting.
pub fn translate_response(
    response: &rmpv::Value,
    request: &crate::kv_index::MatchRequest<'_>,
    tiers_of_interest: &[StorageTier],
) -> Vec<TierMatch> {
    let block_size = request.block_size.max(1) as i64;
    let len = request.token_ids.len() as i64;
    let query_blocks = ((len + block_size - 1) / block_size).max(1);
    let mut out: Vec<TierMatch> = Vec::new();

    // Top-level "instances": map of instance_id -> hit.
    let instances = match response.as_map() {
        Some(map) => map.iter().find(|(k, _)| k.as_str() == Some("instances")),
        None => None,
    };
    let instances_map = match instances {
        Some((_, Value::Map(entries))) => entries,
        _ => return out,
    };

    for (instance_id_value, hit_value) in instances_map {
        let instance_id = match instance_id_value.as_str() {
            Some(s) => s,
            None => continue,
        };
        if instance_id.is_empty() {
            continue;
        }
        // Prefer per-rank rows when present; otherwise fall back to the
        // aggregate using dp_rank 0.
        let dp_hits = collect_dp_hits(hit_value);
        let rank_hits = collect_rank_hits(hit_value);
        let mut violated = false;
        for (rank, (device, cpu_local, cpu_share, disk)) in rank_hits {
            // RFC #414 §2 invariant: the aggregate `dp[rank]` must agree with
            // the per-rank device ceiling. A mismatch means the two maps come
            // from different index states, so drop the whole instance rather
            // than credit an untrustworthy split.
            if let Some(&dp_device) = dp_hits.get(&rank) {
                if dp_device != device {
                    violated = true;
                    break;
                }
            }
            let target = RoutableTarget {
                instance_id: instance_id.into(),
                dp_rank: rank,
            };
            if !append_tier(
                &mut out,
                &target,
                tiers_of_interest,
                query_blocks,
                (device, cpu_local, cpu_share, disk, block_size),
            ) {
                violated = true;
                break;
            }
        }
        if violated {
            // Do not leave a partial instance in the output.
            out.retain(|hit| hit.target.instance_id.as_ref() != instance_id);
            ConductorMetrics::record_invariant_violation();
        }
    }
    out
}

/// Extract the aggregate `dp` map (`rank -> device tokens`). Used only for the
/// cross-check against `rank_matches`; absent maps skip the check.
fn collect_dp_hits(hit: &rmpv::Value) -> std::collections::HashMap<u32, i64> {
    let mut out = std::collections::HashMap::new();
    let Some(map) = hit.as_map() else {
        return out;
    };
    if let Some((_, Value::Map(dp))) = map.iter().find(|(k, _)| k.as_str() == Some("dp")) {
        for (rank_value, tokens) in dp {
            if let (Some(rank), Some(tokens)) = (
                rank_value.as_str().and_then(|s| s.parse::<u32>().ok()),
                tokens.as_i64(),
            ) {
                out.insert(rank, tokens);
            }
        }
    }
    out
}

/// Extract per-rank cumulative token ceilings.
///
/// Returns `(rank, (device, cpu_local, cpu_share, disk))`. When the hit carries
/// a `rank_matches` map we use it (each rank has its own ceilings); otherwise
/// the single aggregate is attributed to dp_rank 0.
fn collect_rank_hits(hit: &rmpv::Value) -> Vec<(u32, (i64, i64, i64, i64))> {
    let Some(map) = hit.as_map() else {
        return Vec::new();
    };

    // Prefer rank_matches for DP fidelity.
    if let Some((_, Value::Map(ranks))) =
        map.iter().find(|(k, _)| k.as_str() == Some("rank_matches"))
    {
        let mut rows = Vec::with_capacity(ranks.len());
        for (rank_value, match_value) in ranks {
            // An unparseable rank cannot be mapped onto a router target, and
            // attributing it to rank 0 could over-credit the wrong engine, so
            // drop that row instead.
            let Some(rank) = rank_value.as_str().and_then(|s| s.parse().ok()) else {
                continue;
            };
            let Tiers4(device, cpu_local, cpu_share, disk) = tiers_from_map(match_value);
            rows.push((rank, (device, cpu_local, cpu_share, disk)));
        }
        if !rows.is_empty() {
            // Deterministic ordering for tests.
            rows.sort_by_key(|(r, _)| *r);
            return rows;
        }
    }

    // Fallback: aggregate ceilings attributed to dp_rank 0.
    let Tiers4(device, cpu_local, cpu_share, disk) = tiers_from_map(hit);
    vec![(0, (device, cpu_local, cpu_share, disk))]
}

struct Tiers4(i64, i64, i64, i64);

fn tiers_from_map(value: &rmpv::Value) -> Tiers4 {
    let Some(map) = value.as_map() else {
        return Tiers4(0, 0, 0, 0);
    };
    let get = |key: &str| -> i64 {
        map.iter()
            .find(|(k, _)| k.as_str() == Some(key))
            .and_then(|(_, v)| v.as_i64())
            .unwrap_or(0)
    };
    Tiers4(
        get("device"),
        get("cpu_local"),
        get("cpu_share"),
        get("disk"),
    )
}

/// Append one flat `(target, tier)` row per tier from cumulative ceilings.
///
/// `(device, cpu_local, cpu_share, disk)` are cumulative token ceilings with
/// `device <= cpu_local <= cpu_share <= disk`. We split into disjoint held
/// intervals per tier, floor-divided by `block_size`, capped at `query_blocks`.
///
/// Returns `false` when the ceilings violate the monotonic invariant (the
/// caller drops the whole instance). `device`/`cpu_local` are engine-local and
/// reported as `Locality::Local`; `cpu_share` (Mooncake Store) and the shared
/// `disk` replicas are not owned by the target engine, so they are
/// `Locality::Remote`.
fn append_tier(
    out: &mut Vec<TierMatch>,
    target: &RoutableTarget,
    tiers_of_interest: &[StorageTier],
    query_blocks: i64,
    (device, cpu_local, cpu_share, disk, block_size): (i64, i64, i64, i64, i64),
) -> bool {
    if block_size <= 0 {
        return true;
    }
    // Validate monotonic invariant once per target.
    if !(device <= cpu_local && cpu_local <= cpu_share && cpu_share <= disk) {
        return false;
    }
    let clamp = |v: i64| -> u32 { (v / block_size).clamp(0, query_blocks) as u32 };

    let d = clamp(device);
    let l = clamp(cpu_local);
    let s = clamp(cpu_share);
    let k = clamp(disk);

    let mut segments_of = |start: u32, end: u32, tier: StorageTier, locality: Locality| {
        if !tiers_of_interest.contains(&tier) {
            return;
        }
        if end > start {
            out.push(TierMatch {
                target: target.clone(),
                tier,
                segments: vec![(start, end)],
                locality,
            });
        }
    };

    segments_of(0, d, StorageTier::Device, Locality::Local);
    segments_of(d, l, StorageTier::HostPinned, Locality::Local);
    segments_of(l, s, StorageTier::External, Locality::Remote);
    segments_of(s, k, StorageTier::Disk, Locality::Remote);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_index::MatchRequest;
    use rmpv::Value;

    fn hit(device: i64, cpu_local: i64, cpu_share: i64, disk: i64) -> Value {
        Value::Map(vec![
            (
                Value::String("longest_matched".into()),
                Value::Integer(disk.into()),
            ),
            (
                Value::String("device".into()),
                Value::Integer(device.into()),
            ),
            (
                Value::String("cpu_local".into()),
                Value::Integer(cpu_local.into()),
            ),
            (
                Value::String("cpu_share".into()),
                Value::Integer(cpu_share.into()),
            ),
            (Value::String("disk".into()), Value::Integer(disk.into())),
        ])
    }

    fn req(token_count: usize) -> MatchRequest<'static> {
        // Boxed slice leaks are fine in tests.
        let ids: Box<[u32]> = (0..token_count as u32).collect();
        let ids: &'static [u32] = Box::leak(ids);
        MatchRequest {
            model_id: Some("qwen3-8b".to_string()),
            block_size: 128,
            token_ids: ids,
            lora_name: None,
            cache_salt: None,
        }
    }

    const TIERS: [StorageTier; 4] = [
        StorageTier::Device,
        StorageTier::HostPinned,
        StorageTier::External,
        StorageTier::Disk,
    ];

    fn resp(entries: Vec<(&str, Value)>) -> Value {
        Value::Map(vec![(
            Value::String("instances".into()),
            Value::Map(
                entries
                    .into_iter()
                    .map(|(k, v)| (Value::String(k.into()), v))
                    .collect(),
            ),
        )])
    }

    #[test]
    fn aggregates_into_flat_tier_rows() {
        // 512 tokens -> 4 device blocks; host continues to 6 blocks; share to 7;
        // disk to 8. Query is 1280 tokens = 10 blocks, so nothing truncated.
        let r = req(10 * 128);
        let v = resp(vec![("inst-0", hit(512, 768, 896, 1024))]);
        let rows = translate_response(&v, &r, &TIERS);
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].target.instance_id.as_ref(), "inst-0");
        assert_eq!(rows[0].tier, StorageTier::Device);
        assert_eq!(rows[0].segments, vec![(0, 4)]);
        assert_eq!(rows[1].tier, StorageTier::HostPinned);
        assert_eq!(rows[1].segments, vec![(4, 6)]);
        assert_eq!(rows[2].tier, StorageTier::External);
        assert_eq!(rows[2].segments, vec![(6, 7)]);
        assert_eq!(rows[3].tier, StorageTier::Disk);
        assert_eq!(rows[3].segments, vec![(7, 8)]);
    }

    #[test]
    fn caps_at_query_blocks() {
        // 128 tokens = 1 block query; Conductor claims 8 device blocks.
        let r = req(128);
        let v = resp(vec![("inst-0", hit(1024, 1024, 1024, 1024))]);
        let rows = translate_response(&v, &r, &TIERS);
        assert_eq!(rows[0].segments, vec![(0, 1)]);
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn drops_non_monotonic_ceilings() {
        // disk < cpu_share violates the invariant -> no rows for this instance.
        let r = req(1024);
        let v = resp(vec![("inst-0", hit(256, 512, 1024, 384))]);
        let rows = translate_response(&v, &r, &TIERS);
        assert!(rows.is_empty());
    }

    #[test]
    fn empty_or_missing_token_ids_yield_no_matches() {
        let provider = ConductorMatchProvider::new("http://127.0.0.1:1");
        // No request.
        let q = MatchQuery {
            group_idx: 0,
            local_hashes: Vec::new(),
            tiers_of_interest: TIERS.to_vec(),
            request: None,
        };
        assert!(provider.find_tiered_matches(&q).is_empty());
    }

    #[test]
    fn register_payload_carries_endpoint_model_profile_and_optional_replay() {
        let profile = ConductorHashProfile {
            strategy: "vllm_v1".into(),
            algorithm: "sha256".into(),
            index_projection: "low64_be".into(),
            python_hash_seed: Some("1450105824".into()),
        };
        let body = encode_register(
            &profile,
            "inst-7",
            0,
            "/mnt/weight/Qwen3-8B",
            128,
            "tcp://127.0.0.1:5957",
            Some("tcp://127.0.0.1:16500".into()),
        );
        let value = rmpv::decode::read_value(&mut std::io::Cursor::new(body.as_slice())).unwrap();
        let map = value.as_map().unwrap();
        let get = |k: &str| {
            map.iter()
                .find(|(kk, _)| kk.as_str() == Some(k))
                .map(|(_, v)| v.clone())
        };
        assert_eq!(
            get("endpoint"),
            Some(Value::String("tcp://127.0.0.1:5957".into()))
        );
        assert_eq!(get("type"), Some(Value::String("vLLM".into())));
        assert_eq!(
            get("modelname"),
            Some(Value::String("/mnt/weight/Qwen3-8B".into()))
        );
        assert_eq!(get("instance_id"), Some(Value::String("inst-7".into())));
        assert_eq!(get("block_size"), Some(Value::Integer(128.into())));
        assert_eq!(get("dp_rank"), Some(Value::Integer(0.into())));
        assert_eq!(
            get("replay_endpoint"),
            Some(Value::String("tcp://127.0.0.1:16500".into()))
        );
        let hp_binding = get("hash_profile").unwrap();
        let hp = hp_binding.as_map().unwrap();
        let hpget = |k: &str| {
            hp.iter()
                .find(|(kk, _)| kk.as_str() == Some(k))
                .map(|(_, v)| v.clone())
        };
        assert_eq!(hpget("strategy"), Some(Value::String("vllm_v1".into())));
        assert_eq!(hpget("algorithm"), Some(Value::String("sha256".into())));
        assert_eq!(
            hpget("index_projection"),
            Some(Value::String("low64_be".into()))
        );
        assert_eq!(
            hpget("python_hash_seed"),
            Some(Value::String("1450105824".into()))
        );
    }

    #[test]
    fn unregister_payload_carries_instance_and_rank() {
        let body = encode_unregister("inst-7", 2);
        let value = rmpv::decode::read_value(&mut std::io::Cursor::new(body.as_slice())).unwrap();
        let map = value.as_map().unwrap();
        assert_eq!(
            map.iter()
                .find(|(k, _)| k.as_str() == Some("instance_id"))
                .map(|(_, v)| v.as_str().unwrap()),
            Some("inst-7")
        );
        assert_eq!(
            map.iter()
                .find(|(k, _)| k.as_str() == Some("dp_rank"))
                .map(|(_, v)| v.as_i64().unwrap()),
            Some(2)
        );
    }

    #[test]
    fn drops_instance_when_dp_disagrees_with_rank_matches() {
        // RFC #414 §2: `dp[r]` must equal `rank_matches[r].device`. When they
        // disagree the two maps describe different index states, so nothing from
        // this instance may be credited.
        let r = req(2 * 128);
        let d = 2i64;
        let v = resp(vec![(
            "inst-0",
            Value::Map(vec![
                (
                    Value::String("dp".into()),
                    Value::Map(vec![(Value::String("0".into()), Value::Integer(1.into()))]),
                ),
                (
                    Value::String("rank_matches".into()),
                    Value::Map(vec![(
                        Value::String("0".into()),
                        Value::Map(vec![
                            (Value::String("device".into()), Value::Integer(d.into())),
                            (Value::String("cpu_local".into()), Value::Integer(d.into())),
                            (Value::String("cpu_share".into()), Value::Integer(d.into())),
                            (Value::String("disk".into()), Value::Integer(d.into())),
                        ]),
                    )]),
                ),
            ]),
        )]);
        assert!(translate_response(&v, &r, &TIERS).is_empty());
    }

    #[test]
    fn shared_tiers_are_reported_as_remote() {
        // `cpu_share` (Mooncake Store) and the shared disk replicas are not
        // owned by the target engine, unlike device / cpu_local.
        let r = req(4 * 128);
        // One block per tier: device [0,1), host [1,2), external [2,3),
        // disk [3,4).
        let v = resp(vec![("inst-0", hit(128, 256, 384, 512))]);
        let rows = translate_response(&v, &r, &TIERS);
        let locality = |tier: StorageTier| {
            rows.iter()
                .find(|row| row.tier == tier)
                .map(|row| row.locality)
                .unwrap()
        };
        assert_eq!(locality(StorageTier::Device), Locality::Local);
        assert_eq!(locality(StorageTier::External), Locality::Remote);
        assert_eq!(locality(StorageTier::Disk), Locality::Remote);
    }

    #[test]
    fn register_omits_python_hash_seed_when_unset() {
        // "Unset" is omission, never an empty string (RFC #414 §4).
        let profile = ConductorHashProfile {
            strategy: "vllm_v1".into(),
            algorithm: "sha256".into(),
            index_projection: "low64_be".into(),
            python_hash_seed: None,
        };
        let body = encode_register(
            &profile,
            "inst-1",
            0,
            "model",
            128,
            "tcp://127.0.0.1:5557",
            None,
        );
        let value = rmpv::decode::read_value(&mut std::io::Cursor::new(body.as_slice())).unwrap();
        let map = value.as_map().unwrap();
        let hash_profile = map
            .iter()
            .find(|(k, _)| k.as_str() == Some("hash_profile"))
            .map(|(_, v)| v.as_map().unwrap())
            .unwrap();
        assert!(hash_profile
            .iter()
            .all(|(k, _)| k.as_str() != Some("python_hash_seed")));
        assert!(hash_profile
            .iter()
            .any(|(k, _)| k.as_str() == Some("index_projection")));
    }

    #[test]
    fn query_timeout_is_classified_as_timeout() {
        use std::io::Read;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        // Accept, read the request, then stall: the client must time out and
        // report `Timeout`, not a generic transport error.
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);
                std::thread::sleep(Duration::from_millis(500));
            }
        });

        let provider = ConductorMatchProvider::with_timeout(
            format!("http://{addr}"),
            Duration::from_millis(30),
        );
        let ids: Vec<u32> = (0..8).collect();
        let request = MatchRequest {
            model_id: Some("model".into()),
            block_size: 4,
            token_ids: &ids,
            lora_name: None,
            cache_salt: None,
        };
        let query = MatchQuery {
            group_idx: 0,
            local_hashes: Vec::new(),
            tiers_of_interest: vec![StorageTier::Device],
            request: Some(request),
        };
        assert_eq!(
            provider.try_query(&query).unwrap_err(),
            ConductorQueryError::Timeout
        );
    }

    #[test]
    fn parses_services_response_with_exported_field_names() {
        // `/services` uses the exported ServiceConfig names (capitalized),
        // unlike `/query`. A lowercase parser would silently think every
        // service is missing and re-register on every heartbeat.
        let value = Value::Map(vec![
            (Value::String("count".into()), Value::Integer(1.into())),
            (
                Value::String("services".into()),
                Value::Array(vec![Value::Map(vec![
                    (
                        Value::String("InstanceID".into()),
                        Value::String("inst-1".into()),
                    ),
                    (Value::String("DPRank".into()), Value::Integer(2.into())),
                    (
                        Value::String("Endpoint".into()),
                        Value::String("tcp://127.0.0.1:5557".into()),
                    ),
                ])]),
            ),
        ]);
        let services = parse_services(&value);
        assert!(services.contains(&("inst-1".to_string(), 2)));
        assert_eq!(services.len(), 1);
    }
}
