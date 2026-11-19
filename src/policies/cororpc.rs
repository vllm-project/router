//! coro_rpc backend for the Conductor KV index (P3).
//!
//! Talks to Conductor over Mooncake's native coro_rpc channel by linking the C++
//! `ConductorClient` (which already speaks coro_rpc) through its stable C ABI
//! (`conductor_c.h`). This mirrors the "embed a native binding" form the router
//! already uses for vllm-rs.
//!
//! Compile-gated behind the `conductor-rpc` cargo feature; `build.rs` documents
//! the required Mooncake build tree (`MOONCAKE_CONDUCTOR_LIB_DIR`).
//!
//! The RFC suggests the `mooncake_conductor` Rust crate (a bindgen wrapper over
//! the same C ABI). This module declares the C ABI by hand because the build
//! environment cannot fetch crates.io; the structs below mirror
//! `conductor_c.h` field for field, so switching to the crate later is a
//! drop-in replacement.
//!
//! Only the byte-level wire details live here. Tier segmentation, invariant
//! checks and registration/metrics bookkeeping are shared with the HTTP
//! transport (`conductor::{tier_matches_from_ceilings, plan_registrations,
//! RegisteredSources}`), so a semantic rule cannot be fixed on one transport
//! and forgotten on the other.

#![cfg(feature = "conductor-rpc")]

use std::collections::HashSet;
use std::ffi::{c_char, c_int, c_long, c_void, CStr, CString};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::conductor::{
    plan_registrations, tier_matches_from_ceilings, ConductorQueryError, ConductorRegistrar,
    RankCeilings, RegisterPlan, RegisteredSources,
};
use super::kv_aware::{AsyncTieredMatchProvider, MatchError, TieredMatchProvider};
use crate::config::types::ConductorHashProfile;
use crate::kv_index::{MatchQuery, MatchRequest, StorageTier, TierMatch, WorkerKvInfo};
use crate::metrics::ConductorMetrics;

const MOONCAKE_CONDUCTOR_OK: c_int = 0;
const MOONCAKE_CONDUCTOR_PUBLISHER_VLLM: c_int = 0;

/// Everything needed to construct a [`CoroRpcMatchProvider`].
#[derive(Debug, Clone)]
pub struct ConductorRpcOptions {
    /// `host:port` of Conductor's coro_rpc channel (`rpc_server_port`).
    pub addr: String,
    pub connect_timeout: Duration,
    pub query_timeout: Duration,
    pub register_timeout: Duration,
    pub hash_profile: ConductorHashProfile,
    pub circuit_breaker: Option<crate::policies::CircuitBreakerConfig>,
    pub heartbeat_secs: u64,
}

// ---------------------------------------------------------------------------
// C ABI (mirrors mooncake-conductor/include/conductor/client/conductor_c.h)
// ---------------------------------------------------------------------------

#[repr(C)]
struct CQueryRequest {
    model_name: *const c_char,
    lora_name: *const c_char,
    block_size: c_long,
    tenant_id: *const c_char,
    token_ids: *const i32,
    token_count: usize,
    cache_salt: *const c_char,
    instance_filter: *const c_char,
}

#[repr(C)]
struct CHashProfileConfig {
    strategy: *const c_char,
    algorithm: *const c_char,
    python_hash_seed: *const c_char,
    index_projection: *const c_char,
}

#[repr(C)]
struct CServiceConfig {
    instance_id: *const c_char,
    endpoint: *const c_char,
    replay_endpoint: *const c_char,
    publisher_kind: c_int,
    model_name: *const c_char,
    lora_name: *const c_char,
    tenant_id: *const c_char,
    block_size: c_long,
    dp_rank: c_int,
    has_cache_group: c_int,
    cache_group: c_long,
    hash_profile: CHashProfileConfig,
}

#[repr(C)]
struct CDpHit {
    rank: c_long,
    tokens: c_long,
}

#[repr(C)]
struct CRankMatch {
    rank: c_long,
    device: c_long,
    cpu_local: c_long,
    cpu_share: c_long,
    disk: c_long,
}

#[repr(C)]
struct CCacheHit {
    instance_id: *mut c_char,
    longest_matched: c_long,
    dp: *mut CDpHit,
    dp_count: usize,
    rank_matches: *mut CRankMatch,
    rank_match_count: usize,
    device: c_long,
    cpu_local: c_long,
    cpu_share: c_long,
    disk: c_long,
}

#[repr(C)]
struct CQueryResult {
    hits: *mut CCacheHit,
    hit_count: usize,
}

/// `mooncake_conductor_hash_profile_t` (resolved output flavour).
#[repr(C)]
struct CResolvedHashProfile {
    strategy: *mut c_char,
    algorithm: *mut c_char,
    python_hash_seed: *mut c_char,
    root_digest: *mut c_char,
    index_projection: *mut c_char,
}

/// `mooncake_conductor_service_info_t` as returned by `list_services`.
#[repr(C)]
struct CServiceInfo {
    endpoint: *mut c_char,
    replay_endpoint: *mut c_char,
    publisher_kind: c_int,
    model_name: *mut c_char,
    lora_name: *mut c_char,
    tenant_id: *mut c_char,
    instance_id: *mut c_char,
    block_size: c_long,
    dp_rank: c_int,
    has_cache_group: c_int,
    cache_group: c_long,
    hash_profile: CResolvedHashProfile,
}

#[repr(C)]
struct CServiceList {
    services: *mut CServiceInfo,
    service_count: usize,
}

extern "C" {
    fn mooncake_conductor_create() -> *mut c_void;
    fn mooncake_conductor_destroy(conductor: *mut c_void);
    fn mooncake_conductor_setup(
        conductor: *mut c_void,
        addr: *const c_char,
        connect_timeout_ms: c_long,
        request_timeout_ms: c_long,
    ) -> c_int;
    fn mooncake_conductor_query(
        conductor: *mut c_void,
        request: *const CQueryRequest,
        result: *mut CQueryResult,
    ) -> c_int;
    fn mooncake_conductor_query_result_free(result: *mut CQueryResult);
    fn mooncake_conductor_register(
        conductor: *mut c_void,
        config: *const CServiceConfig,
        is_new: *mut c_int,
    ) -> c_int;
    fn mooncake_conductor_unregister(
        conductor: *mut c_void,
        instance_id: *const c_char,
        tenant_id: *const c_char,
        dp_rank: c_int,
    ) -> c_int;
    fn mooncake_conductor_list_services(conductor: *mut c_void, list: *mut CServiceList) -> c_int;
    fn mooncake_conductor_service_list_free(list: *mut CServiceList);
}

#[inline]
fn cstring_or_empty(s: &str) -> CString {
    CString::new(s).unwrap_or_else(|_| CString::new("").unwrap())
}

#[inline]
fn rc_to_error(rc: c_int, op: &'static str) -> ConductorQueryError {
    ConductorQueryError::Io(format!("{op} failed rc={rc}"))
}

/// Low-level coro_rpc client handle over the C++ `ConductorClient`.
///
/// `Sync`/`Send` justification: the C ABI contract (`conductor_c.h`) states that
/// calls on one handle are serialized by an internal mutex and that
/// `mooncake_conductor_destroy` must not race any other call. Both hold here:
/// every call borrows the handle through an `Arc<CoroTcpTransport>`, and the
/// handle is destroyed only from `Drop`, which runs after the last `Arc` is
/// gone — i.e. when no call can be in flight.
pub struct CoroTcpTransport {
    handle: *mut c_void,
    addr: CString,
    connect_timeout_ms: c_long,
    request_timeout_ms: c_long,
}

unsafe impl Send for CoroTcpTransport {}
unsafe impl Sync for CoroTcpTransport {}

impl std::fmt::Debug for CoroTcpTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoroTcpTransport")
            .field("addr", &self.addr)
            .finish_non_exhaustive()
    }
}

impl CoroTcpTransport {
    pub fn new(
        addr: &str,
        connect_timeout: Duration,
        request_timeout: Duration,
    ) -> Result<Self, ConductorQueryError> {
        let handle = unsafe { mooncake_conductor_create() };
        if handle.is_null() {
            return Err(ConductorQueryError::Io(
                "mooncake_conductor_create returned NULL".into(),
            ));
        }
        let transport = Self {
            handle,
            addr: cstring_or_empty(addr),
            connect_timeout_ms: connect_timeout.as_millis().min(i64::MAX as u128) as c_long,
            request_timeout_ms: request_timeout.as_millis().min(i64::MAX as u128) as c_long,
        };
        let rc = unsafe {
            mooncake_conductor_setup(
                transport.handle,
                transport.addr.as_ptr(),
                transport.connect_timeout_ms,
                transport.request_timeout_ms,
            )
        };
        if rc != MOONCAKE_CONDUCTOR_OK {
            // `transport` drops here and destroys the handle.
            return Err(rc_to_error(rc, "mooncake_conductor_setup"));
        }
        Ok(transport)
    }

    fn handle(&self) -> Result<*mut c_void, ConductorQueryError> {
        if self.handle.is_null() {
            Err(ConductorQueryError::Io("conductor handle closed".into()))
        } else {
            Ok(self.handle)
        }
    }

    /// Issue a coro_rpc `/query` and translate the result into `TierMatch` rows.
    pub fn query(
        &self,
        request: &MatchRequest<'_>,
        tiers_of_interest: &[StorageTier],
    ) -> Result<Vec<TierMatch>, ConductorQueryError> {
        let handle = self.handle()?;

        let model = cstring_or_empty(request.model_id.as_deref().unwrap_or(""));
        // `lora_name` is part of Conductor's context key; omitting it would make
        // a LoRA request match (and get credit for) the base model's prefix.
        let lora = request
            .lora_name
            .as_deref()
            .filter(|name| !name.is_empty())
            .map(cstring_or_empty);
        let cache_salt = request
            .cache_salt
            .as_deref()
            .filter(|salt| !salt.is_empty())
            .map(cstring_or_empty);
        let tokens: Vec<i32> = request.token_ids.iter().map(|&t| t as i32).collect();
        let token_ptr = if tokens.is_empty() {
            std::ptr::null()
        } else {
            tokens.as_ptr()
        };

        let creq = CQueryRequest {
            model_name: model.as_ptr(),
            lora_name: lora.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            block_size: request.block_size as c_long,
            tenant_id: std::ptr::null(),
            token_ids: token_ptr,
            token_count: tokens.len(),
            cache_salt: cache_salt.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            instance_filter: std::ptr::null(),
        };

        let mut result = CQueryResult {
            hits: std::ptr::null_mut(),
            hit_count: 0,
        };
        let rc = unsafe { mooncake_conductor_query(handle, &creq, &mut result) };
        if rc != MOONCAKE_CONDUCTOR_OK {
            return Err(rc_to_error(rc, "mooncake_conductor_query"));
        }

        // SAFETY: on success the C ABI filled `result` with C-allocated arrays
        // that stay valid until `query_result_free`; the rows are copied out
        // before freeing.
        let ceilings = unsafe { ceilings_from_result(&result) };
        unsafe { mooncake_conductor_query_result_free(&mut result) };
        Ok(tier_matches_from_ceilings(
            ceilings,
            request,
            tiers_of_interest,
        ))
    }

    /// Register one DP-rank endpoint with Conductor.
    pub fn register(
        &self,
        plan: &RegisterPlan,
        profile: &ConductorHashProfile,
    ) -> Result<(), ConductorQueryError> {
        let handle = self.handle()?;

        let c_instance = cstring_or_empty(&plan.instance_id);
        let c_model = cstring_or_empty(&plan.model);
        let c_endpoint = cstring_or_empty(&plan.endpoint);
        let c_replay = plan
            .replay_endpoint
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(cstring_or_empty);
        let c_strategy = cstring_or_empty(&profile.strategy);
        let c_algorithm = cstring_or_empty(&profile.algorithm);
        let c_projection = cstring_or_empty(&profile.index_projection);
        let c_seed = profile.python_hash_seed.as_deref().map(cstring_or_empty);

        let cfg = CServiceConfig {
            instance_id: c_instance.as_ptr(),
            endpoint: c_endpoint.as_ptr(),
            replay_endpoint: c_replay.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            publisher_kind: MOONCAKE_CONDUCTOR_PUBLISHER_VLLM,
            model_name: c_model.as_ptr(),
            lora_name: std::ptr::null(),
            tenant_id: std::ptr::null(),
            block_size: plan.block_size as c_long,
            dp_rank: plan.dp_rank as c_int,
            has_cache_group: 0,
            cache_group: 0,
            hash_profile: CHashProfileConfig {
                strategy: c_strategy.as_ptr(),
                algorithm: c_algorithm.as_ptr(),
                // "Unset" is omission, never an empty string (RFC #414 §4).
                python_hash_seed: c_seed.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                index_projection: c_projection.as_ptr(),
            },
        };
        let mut is_new: c_int = 0;
        let rc = unsafe { mooncake_conductor_register(handle, &cfg, &mut is_new) };
        if rc != MOONCAKE_CONDUCTOR_OK {
            return Err(rc_to_error(rc, "mooncake_conductor_register"));
        }
        Ok(())
    }

    pub fn unregister(&self, instance_id: &str, dp_rank: u32) -> Result<(), ConductorQueryError> {
        let handle = self.handle()?;
        let c_instance = cstring_or_empty(instance_id);
        let rc = unsafe {
            mooncake_conductor_unregister(
                handle,
                c_instance.as_ptr(),
                std::ptr::null(),
                dp_rank as c_int,
            )
        };
        if rc != MOONCAKE_CONDUCTOR_OK {
            return Err(rc_to_error(rc, "mooncake_conductor_unregister"));
        }
        Ok(())
    }

    /// `(instance_id, dp_rank)` pairs Conductor currently knows about.
    pub fn list_services(&self) -> Result<HashSet<(String, u32)>, ConductorQueryError> {
        let handle = self.handle()?;
        let mut list = CServiceList {
            services: std::ptr::null_mut(),
            service_count: 0,
        };
        let rc = unsafe { mooncake_conductor_list_services(handle, &mut list) };
        if rc != MOONCAKE_CONDUCTOR_OK {
            return Err(rc_to_error(rc, "mooncake_conductor_list_services"));
        }
        let mut out = HashSet::new();
        if !list.services.is_null() && list.service_count > 0 {
            // SAFETY: the C ABI returns `service_count` contiguous entries that
            // stay valid until `service_list_free`.
            let services = unsafe { std::slice::from_raw_parts(list.services, list.service_count) };
            for service in services {
                let instance_id = unsafe { read_c_string(service.instance_id) };
                if instance_id.is_empty() {
                    continue;
                }
                out.insert((instance_id, service.dp_rank.max(0) as u32));
            }
        }
        unsafe { mooncake_conductor_service_list_free(&mut list) };
        Ok(out)
    }
}

impl Drop for CoroTcpTransport {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { mooncake_conductor_destroy(self.handle) };
        }
    }
}

/// Copy a `CQueryResult` into transport-independent rows.
///
/// Uses `dp` for the RFC #414 §2 cross-check, which the shared translation then
/// enforces; the FFI path previously dropped that information entirely.
unsafe fn ceilings_from_result(result: &CQueryResult) -> Vec<RankCeilings> {
    let mut out = Vec::new();
    if result.hits.is_null() || result.hit_count == 0 {
        return out;
    }
    let hits = unsafe { std::slice::from_raw_parts(result.hits, result.hit_count) };
    for hit in hits {
        let instance_id = unsafe { read_c_string(hit.instance_id) };
        if instance_id.is_empty() {
            continue;
        }
        let mut dp: std::collections::HashMap<u32, i64> = std::collections::HashMap::new();
        if !hit.dp.is_null() && hit.dp_count > 0 {
            for entry in unsafe { std::slice::from_raw_parts(hit.dp, hit.dp_count) } {
                if entry.rank >= 0 {
                    dp.insert(entry.rank as u32, entry.tokens);
                }
            }
        }

        if !hit.rank_matches.is_null() && hit.rank_match_count > 0 {
            for rank in
                unsafe { std::slice::from_raw_parts(hit.rank_matches, hit.rank_match_count) }
            {
                if rank.rank < 0 {
                    continue;
                }
                let rank_index = rank.rank as u32;
                out.push(RankCeilings {
                    instance_id: instance_id.clone(),
                    rank: rank_index,
                    device: rank.device,
                    cpu_local: rank.cpu_local,
                    cpu_share: rank.cpu_share,
                    disk: rank.disk,
                    dp_device: dp.get(&rank_index).copied(),
                });
            }
        } else {
            out.push(RankCeilings {
                instance_id: instance_id.clone(),
                rank: 0,
                device: hit.device,
                cpu_local: hit.cpu_local,
                cpu_share: hit.cpu_share,
                disk: hit.disk,
                dp_device: dp.get(&0).copied(),
            });
        }
    }
    out
}

unsafe fn read_c_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// Concurrent `/query` calls the coro_rpc channel serves at once.
///
/// Mooncake's `ConductorClient` serialises calls made on one handle: `Invoke`
/// takes the client's mutex *before* it starts the RPC timeout. Sharing a single
/// handle would therefore queue callers outside their own deadline — a healthy
/// Conductor would still show multi-second tail latency under load, and the
/// circuit breaker would never see a failure to open on. Parallel query capacity
/// is thus the handle count, kept small because each handle is one connection
/// and (while a call is in flight) one blocking thread.
const QUERY_HANDLES: usize = 4;

/// A bounded set of query handles with one admission permit per handle.
///
/// `idle` is a plain stack of free handles: a caller only pops after it holds a
/// permit, and returns the handle before releasing that permit, so admission and
/// handle availability cannot drift apart.
#[derive(Debug)]
struct TransportSlots<T> {
    idle: Mutex<Vec<Arc<T>>>,
    permits: Arc<tokio::sync::Semaphore>,
}

/// One admitted call: owns a handle and its admission permit. Dropping it
/// returns the handle first and releases the permit second (the permit is a
/// field, so it drops after `Drop::drop` returns), which keeps admission from
/// running ahead of handle availability — including when the blocking call
/// panics.
struct AdmittedTransport<T> {
    slots: Arc<TransportSlots<T>>,
    transport: Option<Arc<T>>,
    _permit: tokio::sync::OwnedSemaphorePermit,
}

impl<T> TransportSlots<T> {
    fn new(transports: Vec<Arc<T>>) -> Arc<Self> {
        Arc::new(Self {
            permits: Arc::new(tokio::sync::Semaphore::new(transports.len())),
            idle: Mutex::new(transports),
        })
    }

    /// Take a free handle. Only called while holding a permit, which guarantees
    /// one is available; `None` is returned defensively if that invariant is
    /// ever broken.
    fn take_idle(&self) -> Option<Arc<T>> {
        self.idle
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .pop()
    }

    /// Admission without waiting, for callers on a blocking selection path that
    /// must not queue: either a handle is free right now or the placement query
    /// is reported as unavailable.
    fn try_admit(self: &Arc<Self>) -> Option<AdmittedTransport<T>> {
        let permit = self.permits.clone().try_acquire_owned().ok()?;
        let transport = self.take_idle()?;
        Some(AdmittedTransport {
            slots: Arc::clone(self),
            transport: Some(transport),
            _permit: permit,
        })
    }

    /// Admission bounded by `budget`. `None` means nothing freed up in time, so
    /// the caller reports a placement miss instead of queueing behind other
    /// queries.
    async fn admit(self: &Arc<Self>, budget: Duration) -> Option<AdmittedTransport<T>> {
        let permit = tokio::time::timeout(budget, self.permits.clone().acquire_owned())
            .await
            .ok()?
            .ok()?;
        let transport = self.take_idle()?;
        Some(AdmittedTransport {
            slots: Arc::clone(self),
            transport: Some(transport),
            _permit: permit,
        })
    }
}

impl<T> AdmittedTransport<T> {
    fn transport(&self) -> &T {
        self.transport
            .as_deref()
            .expect("an admitted call always holds a handle")
    }
}

impl<T> Drop for AdmittedTransport<T> {
    fn drop(&mut self) {
        if let Some(transport) = self.transport.take() {
            self.slots
                .idle
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .push(transport);
        }
    }
}

/// Placement provider over the coro_rpc channel.
///
/// Implements the same traits as the HTTP provider, so the policy wiring
/// (breaker, async selection, registration/heartbeat) is shared. Queries run on
/// a small bounded pool of handles ([`QUERY_HANDLES`]) so concurrency is real
/// rather than queueing behind the client's internal mutex, while
/// `/register`/`/unregister`/`/services` keep a separate control handle with the
/// larger register budget (same split as the HTTP transports).
#[derive(Debug)]
pub struct CoroRpcMatchProvider {
    query_slots: Arc<TransportSlots<CoroTcpTransport>>,
    /// Whole budget for one placement query: admission *and* the RPC call.
    query_budget: Duration,
    control_transport: Arc<CoroTcpTransport>,
    hash_profile: ConductorHashProfile,
    registered: RegisteredSources,
}

impl CoroRpcMatchProvider {
    pub fn new(options: ConductorRpcOptions) -> Result<Self, String> {
        let mut query_transports = Vec::with_capacity(QUERY_HANDLES);
        for _ in 0..QUERY_HANDLES {
            let transport = CoroTcpTransport::new(
                &options.addr,
                options.connect_timeout,
                options.query_timeout,
            )
            .map_err(|error| format!("conductor rpc setup failed: {error:?}"))?;
            query_transports.push(Arc::new(transport));
        }
        let control_transport = CoroTcpTransport::new(
            &options.addr,
            options.connect_timeout,
            options.register_timeout,
        )
        .map_err(|error| format!("conductor rpc setup failed: {error:?}"))?;
        Ok(Self {
            query_slots: TransportSlots::new(query_transports),
            query_budget: options.query_timeout,
            control_transport: Arc::new(control_transport),
            hash_profile: options.hash_profile,
            registered: RegisteredSources::default(),
        })
    }

    pub fn query(&self, query: &MatchQuery<'_>) -> Vec<TierMatch> {
        self.try_query(query).unwrap_or_default()
    }

    pub fn try_query(&self, query: &MatchQuery<'_>) -> Result<Vec<TierMatch>, ConductorQueryError> {
        let Some(request) = query.request.as_ref() else {
            return Ok(Vec::new());
        };
        if request.token_ids.is_empty() {
            return Ok(Vec::new());
        }
        // Synchronous selection cannot await: admit only if a handle is free
        // right now, so a busy Conductor degrades to load-only instead of
        // queueing this caller behind other queries.
        let Some(admitted) = self.query_slots.try_admit() else {
            ConductorMetrics::record_query_error("admission_timeout");
            return Err(ConductorQueryError::Timeout);
        };
        let started = Instant::now();
        let result = admitted.transport().query(request, &query.tiers_of_interest);
        ConductorMetrics::record_query_duration(started.elapsed());
        if let Err(error) = &result {
            ConductorMetrics::record_query_error(error.reason());
        }
        result
    }
}

impl TieredMatchProvider for CoroRpcMatchProvider {
    fn find_tiered_matches(&self, query: &MatchQuery<'_>) -> Vec<TierMatch> {
        self.query(query)
    }

    fn try_find_tiered_matches(
        &self,
        query: &MatchQuery<'_>,
    ) -> Result<Vec<TierMatch>, MatchError> {
        self.try_query(query).map_err(|error| match error {
            ConductorQueryError::Status(status) if (400..500).contains(&status) => {
                MatchError::Rejected
            }
            _ => MatchError::Unavailable,
        })
    }
}

#[async_trait::async_trait]
impl AsyncTieredMatchProvider for CoroRpcMatchProvider {
    async fn find_tiered_matches_async(
        &self,
        query: &MatchQuery<'_>,
    ) -> Result<Vec<TierMatch>, MatchError> {
        // The C++ client is blocking; copy the query identity into owned data
        // and run the call off the async workers.
        let Some(request) = query.request.as_ref() else {
            return Ok(Vec::new());
        };
        if request.token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let model_id = request.model_id.clone();
        let block_size = request.block_size;
        let lora_name = request.lora_name.clone();
        let cache_salt = request.cache_salt.clone();
        let tokens = request.token_ids.to_vec();
        let tiers = query.tiers_of_interest.clone();
        let budget = self.query_budget;

        // One budget covers admission *and* the RPC: the client's own request
        // timeout only starts once it owns the handle, so without a bound here a
        // caller could wait for every query queued ahead of it.
        let Some(admitted) = self.query_slots.admit(budget).await else {
            ConductorMetrics::record_query_error("admission_timeout");
            return Err(MatchError::Unavailable);
        };
        let started = Instant::now();
        let blocking = tokio::task::spawn_blocking(move || {
            let request = MatchRequest {
                model_id,
                block_size,
                token_ids: &tokens,
                lora_name,
                cache_salt,
            };
            let result = admitted.transport().query(&request, &tiers);
            ConductorMetrics::record_query_duration(started.elapsed());
            if let Err(error) = &result {
                ConductorMetrics::record_query_error(error.reason());
            }
            result
            // `admitted` drops here: the handle returns to the pool and only
            // then is the admission permit released.
        });
        let remaining = budget.saturating_sub(started.elapsed());
        match tokio::time::timeout(remaining, blocking).await {
            Ok(Ok(result)) => result.map_err(|_| MatchError::Unavailable),
            Ok(Err(join_error)) => {
                tracing::warn!("spawn_blocking coro_rpc query join error: {join_error}");
                Err(MatchError::Unavailable)
            }
            Err(_deadline) => {
                // The C call cannot be cancelled and keeps running on the
                // blocking pool; its handle and permit return when it finishes,
                // so a timed-out caller never admits extra work.
                ConductorMetrics::record_query_error("deadline");
                Err(MatchError::Unavailable)
            }
        }
    }
}

impl ConductorRegistrar for CoroRpcMatchProvider {
    fn register_worker(&self, info: &WorkerKvInfo) -> Result<(), String> {
        let mut last_error = String::new();
        for plan in plan_registrations(info) {
            match self.control_transport.register(&plan, &self.hash_profile) {
                Ok(()) => self.registered.insert(&plan.instance_id, plan.dp_rank),
                Err(error) => last_error = format!("register failed: {error:?}"),
            }
        }
        ConductorMetrics::record_register(last_error.is_empty());
        if last_error.is_empty() {
            Ok(())
        } else {
            Err(last_error)
        }
    }

    fn unregister_worker(&self, instance_id: &str, dp_ranks: &[u32]) -> Result<(), String> {
        let mut last_error = String::new();
        for &rank in dp_ranks {
            match self.control_transport.unregister(instance_id, rank) {
                Ok(()) => self.registered.remove(instance_id, rank),
                Err(error) => last_error = format!("unregister failed: {error:?}"),
            }
        }
        if last_error.is_empty() {
            Ok(())
        } else {
            Err(last_error)
        }
    }

    fn list_services(&self) -> Result<HashSet<(String, u32)>, String> {
        self.control_transport
            .list_services()
            .map_err(|error| format!("{error:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_rank_matches_and_dp_cross_check() {
        let instance = CString::new("inst-0").unwrap();
        let mut ranks = vec![CRankMatch {
            rank: 0,
            device: 256,
            cpu_local: 256,
            cpu_share: 256,
            disk: 256,
        }];
        let mut dp = vec![CDpHit {
            rank: 0,
            tokens: 256,
        }];
        let mut hit = CCacheHit {
            instance_id: instance.as_ptr() as *mut c_char,
            longest_matched: 256,
            dp: dp.as_mut_ptr(),
            dp_count: dp.len(),
            rank_matches: ranks.as_mut_ptr(),
            rank_match_count: ranks.len(),
            device: 256,
            cpu_local: 256,
            cpu_share: 256,
            disk: 256,
        };
        let result = CQueryResult {
            hits: &mut hit,
            hit_count: 1,
        };

        let rows = unsafe { ceilings_from_result(&result) };
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].instance_id, "inst-0");
        assert_eq!(rows[0].rank, 0);
        assert_eq!(rows[0].device, 256);
        assert_eq!(rows[0].dp_device, Some(256));

        // A disagreement between `dp` and `rank_matches` must drop the instance.
        let mut dp_mismatch = vec![CDpHit {
            rank: 0,
            tokens: 128,
        }];
        hit.dp = dp_mismatch.as_mut_ptr();
        let result = CQueryResult {
            hits: &mut hit,
            hit_count: 1,
        };
        let rows = unsafe { ceilings_from_result(&result) };
        assert_eq!(rows[0].dp_device, Some(128));
        let request = MatchRequest {
            model_id: Some("model".into()),
            block_size: 128,
            token_ids: &[0u32; 512],
            lora_name: None,
            cache_salt: None,
        };
        let tiers = [
            StorageTier::Device,
            StorageTier::HostPinned,
            StorageTier::External,
            StorageTier::Disk,
        ];
        assert!(tier_matches_from_ceilings(rows, &request, &tiers).is_empty());
    }

    #[test]
    fn falls_back_to_aggregate_when_rank_matches_are_absent() {
        let instance = CString::new("inst-1").unwrap();
        let mut hit = CCacheHit {
            instance_id: instance.as_ptr() as *mut c_char,
            longest_matched: 256,
            dp: std::ptr::null_mut(),
            dp_count: 0,
            rank_matches: std::ptr::null_mut(),
            rank_match_count: 0,
            device: 256,
            cpu_local: 256,
            cpu_share: 256,
            disk: 256,
        };
        let result = CQueryResult {
            hits: &mut hit,
            hit_count: 1,
        };
        let rows = unsafe { ceilings_from_result(&result) };
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].rank, 0);
        assert_eq!(rows[0].dp_device, None);
    }
    #[test]
    fn query_admission_is_bounded_by_the_handle_count() {
        let slots = TransportSlots::new(vec![Arc::new(1_u8), Arc::new(2_u8)]);
        let first = slots.try_admit().expect("one handle is free");
        let second = slots.try_admit().expect("two handles are free");
        // Both callers got distinct handles, and a third is refused rather than
        // queued behind them.
        assert_ne!(first.transport(), second.transport());
        assert!(slots.try_admit().is_none());

        // Releasing one handle admits exactly one more caller.
        drop(first);
        let third = slots.try_admit().expect("the released handle is reusable");
        assert!(slots.try_admit().is_none());
        drop(second);
        drop(third);
        assert!(slots.try_admit().is_some());
    }

    #[tokio::test]
    async fn async_admission_gives_up_at_the_deadline() {
        let slots = TransportSlots::new(vec![Arc::new(1_u8)]);
        let held = slots.try_admit().expect("the only handle is free");

        let started = Instant::now();
        assert!(
            slots.admit(Duration::from_millis(20)).await.is_none(),
            "a busy pool must refuse instead of blocking past the budget"
        );
        assert!(
            started.elapsed() >= Duration::from_millis(15),
            "admission waits for the budget before giving up"
        );

        // The permit is returned with the handle, so the next caller proceeds.
        drop(held);
        assert!(slots.admit(Duration::from_millis(20)).await.is_some());
    }
}
