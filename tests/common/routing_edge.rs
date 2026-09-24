//! Fixtures shared by the edge benchmarks and CI tests.
//!
//! With [`BASE_LOADS`], W1 identifies a cache hit, W3 a low-match or
//! imbalanced selection, and W0 the fallback when W1 is unhealthy.
//! Routing inserts the probe, so benchmarks reset the fixture outside
//! each timed call.

use super::bench_corpus::{seeded_text, seeded_utf8_text, SEED};
use std::collections::HashMap;
use std::sync::Arc;
use vllm_router_rs::core::{BasicWorker, Worker, WorkerType};
use vllm_router_rs::policies::{
    CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, RendezvousHashPolicy, RequestHeaders,
};

/// Workers in branch cases and the rendezvous pair.
pub const WORKERS: usize = 4;
/// Worker counts shared by the scaling test and benchmark.
pub const WORKER_COUNTS: [usize; 4] = [1, 4, 16, 64];
/// Holds the warmed key.
pub const TENANT: usize = 1;
/// The only least-loaded worker under [`BASE_LOADS`].
pub const MIN_LOAD: usize = 3;
/// The first healthy worker while [`TENANT`] is down.
pub const FIRST_HEALTHY: usize = 0;
/// Balanced loads with W3 as the only least-loaded worker.
pub const BASE_LOADS: [usize; WORKERS] = [1, 1, 1, 0];

/// Absent from the corpus, so a prefix match stops here.
const FORK: char = '~';
/// CJK characters for the hand-built UTF-8 cases.
const CJK: &str = "路由缓存前缀令牌请求延迟预算策略解码批次队列健康重试会话模型提示";

/// Explicit thresholds for boundary tests; eviction is disabled.
/// These differ from both library and CLI defaults (see the benchmark docs).
pub fn fixture_config() -> CacheAwareConfig {
    CacheAwareConfig {
        cache_threshold: 0.5,
        balance_abs_threshold: 5,
        balance_rel_threshold: 2.0,
        eviction_interval_secs: 0,
        max_tree_size: 10_000,
    }
}

pub fn workers(n: usize) -> Vec<Arc<dyn Worker>> {
    (0..n)
        .map(|i| {
            Arc::new(BasicWorker::new(
                format!("http://127.0.0.1:{}", 30_000 + i),
                WorkerType::Regular,
            )) as Arc<dyn Worker>
        })
        .collect()
}

fn set_loads(workers: &[Arc<dyn Worker>], loads: &[usize]) {
    for (w, &target) in workers.iter().zip(loads) {
        while w.load() < target {
            w.increment_load();
        }
        while w.load() > target {
            w.decrement_load();
        }
    }
}

/// Restores warmed keys, worker loads and health before each measurement.
pub struct CacheAwareFixture {
    policy: CacheAwarePolicy,
    workers: Vec<Arc<dyn Worker>>,
    warm: Vec<(usize, String)>,
    loads: Vec<usize>,
    down: Vec<usize>,
}

impl CacheAwareFixture {
    pub fn new(warm: Vec<(usize, String)>, loads: &[usize], down: &[usize]) -> CacheAwareFixture {
        let fixture = CacheAwareFixture {
            policy: CacheAwarePolicy::with_config(fixture_config()),
            workers: workers(loads.len()),
            warm,
            loads: loads.to_vec(),
            down: down.to_vec(),
        };
        fixture.reset();
        fixture
    }

    /// Warm prompts round-robin across workers, with no outstanding load.
    pub fn round_robin(prompts: &[String], n: usize) -> Self {
        let warm = prompts
            .iter()
            .enumerate()
            .map(|(i, p)| (i % n, p.clone()))
            .collect();
        Self::new(warm, &vec![0; n], &[])
    }

    /// Clear and re-warm the tree, then restore loads and health.
    /// Each key's tenant is the only least-loaded worker during warming.
    pub fn reset(&self) {
        self.clear();
        for w in &self.workers {
            w.set_healthy(true);
            self.policy.add_worker(w.as_ref());
        }
        for (tenant, key) in &self.warm {
            let mut loads = vec![1; self.workers.len()];
            loads[*tenant] = 0;
            set_loads(&self.workers, &loads);
            assert_eq!(
                self.select(key),
                Some(*tenant),
                "warming a key must place it on its tenant"
            );
        }
        set_loads(&self.workers, &self.loads);
        for &i in &self.down {
            self.workers[i].set_healthy(false);
        }
    }

    /// Removing all tenants prunes the tree to its root.
    fn clear(&self) {
        for w in &self.workers {
            self.policy.remove_worker(w.as_ref());
        }
    }

    pub fn select(&self, probe: &str) -> Option<usize> {
        self.policy.select_worker(&self.workers, Some(probe))
    }

    /// Change a worker's health until the next [`reset`](Self::reset).
    pub fn set_healthy(&self, worker: usize, healthy: bool) {
        self.workers[worker].set_healthy(healthy);
    }

    pub fn warm_keys(&self) -> impl Iterator<Item = &str> {
        self.warm.iter().map(|(_, key)| key.as_str())
    }
}

impl Drop for CacheAwareFixture {
    /// Prune nodes before dropping the tree to break parent/child Arc cycles.
    fn drop(&mut self) {
        self.clear();
    }
}

/// A ready-to-run fixture, probe, and expected `(matched, input)` character counts.
pub struct BuiltCase {
    pub fixture: CacheAwareFixture,
    pub probe: String,
    pub matched_chars: (usize, usize),
}

/// Builds long inputs on demand so only one case is held at a time.
pub struct CacheAwareCase {
    pub name: String,
    /// The worker selected when the intended branch runs.
    pub expect: usize,
    /// Size of the warmed key in bytes; sets the benchmark's sample count.
    pub key_bytes: usize,
    /// Timed by `edge/cache_aware`; every case is asserted in the test.
    pub bench: bool,
    build: Box<dyn Fn() -> BuiltCase>,
}

impl CacheAwareCase {
    fn new(
        name: String,
        expect: usize,
        key_bytes: usize,
        bench: bool,
        build: impl Fn() -> BuiltCase + 'static,
    ) -> CacheAwareCase {
        CacheAwareCase {
            name,
            expect,
            key_bytes,
            bench,
            build: Box::new(build),
        }
    }

    pub fn build(&self) -> BuiltCase {
        (self.build)()
    }
}

/// Name for a worker index in assertion messages.
pub fn branch(worker: usize) -> &'static str {
    match worker {
        TENANT => "W1 (cache hit)",
        MIN_LOAD => "W3 (least loaded: low match or imbalanced)",
        FIRST_HEALTHY => "W0 (stale-tenant fallback)",
        _ => "an unexpected worker",
    }
}

pub fn size_label(bytes: usize) -> String {
    if bytes >= 1024 * 1024 && bytes.is_multiple_of(1024 * 1024) {
        format!("{}MiB", bytes / (1024 * 1024))
    } else if bytes >= 1024 && bytes.is_multiple_of(1024) {
        format!("{}KiB", bytes / 1024)
    } else {
        format!("{bytes}B")
    }
}

/// Equal-length ASCII keys sharing exactly `shared` bytes before [`FORK`].
/// When `shared == total`, the keys are identical.
pub fn ascii_fork(seed: u64, total: usize, shared: usize) -> (String, String) {
    let warm = seeded_text(seed, total);
    let mut probe = String::with_capacity(total);
    probe.push_str(&warm[..shared]);
    if shared < total {
        probe.push(FORK);
        probe.push_str(&seeded_text(seed ^ 0x5eed, total - shared - 1));
    }
    (warm, probe)
}

/// CJK keys with equal character counts and a controlled shared prefix.
/// Returns `(warm, probe, shared_chars, total_chars)`. Only `warm` has the
/// requested byte size.
fn utf8_fork(
    seed: u64,
    total_bytes: usize,
    shared_percent: usize,
) -> (String, String, usize, usize) {
    let warm = seeded_utf8_text(seed, total_bytes);
    let total_chars = warm.chars().count();
    let shared = total_chars * shared_percent / 100;
    let mut probe: String = warm.chars().take(shared).collect();
    if shared < total_chars {
        probe.push(FORK);
        let tail = seeded_utf8_text(seed ^ 0x5eed, 4 * total_chars);
        probe.extend(tail.chars().take(total_chars - shared - 1));
    }
    (warm, probe, shared, total_chars)
}

fn cjk(n: usize) -> String {
    CJK.chars().cycle().take(n).collect()
}

fn ascii_case(
    name: String,
    expect: usize,
    bench: bool,
    total: usize,
    shared: usize,
    loads: [usize; WORKERS],
) -> CacheAwareCase {
    CacheAwareCase::new(name, expect, total, bench, move || {
        let (warm, probe) = ascii_fork(SEED ^ total as u64, total, shared);
        BuiltCase {
            fixture: CacheAwareFixture::new(vec![(TENANT, warm)], &loads, &[]),
            probe,
            matched_chars: (shared, total),
        }
    })
}

fn utf8_case(
    name: &str,
    expect: usize,
    warm: String,
    probe: String,
    matched: usize,
) -> CacheAwareCase {
    let key_bytes = warm.len();
    let input = probe.chars().count();
    CacheAwareCase::new(name.to_string(), expect, key_bytes, false, move || {
        BuiltCase {
            fixture: CacheAwareFixture::new(vec![(TENANT, warm.clone())], &BASE_LOADS, &[]),
            probe: probe.clone(),
            matched_chars: (matched, input),
        }
    })
}

/// All CI cases; `bench` marks the subset timed by Criterion.
pub fn cache_aware_cases() -> Vec<CacheAwareCase> {
    const ONE_MIB: usize = 1024 * 1024;
    const SMALL: usize = 2048;
    let mut cases = Vec::new();

    for size in super::bench_corpus::LONG_SIZES {
        for (name, shared, expect) in [("long_hit", size, TENANT), ("long_cold", 0, MIN_LOAD)] {
            cases.push(ascii_case(
                format!("{name}/{}", size_label(size)),
                expect,
                true,
                size,
                shared,
                BASE_LOADS,
            ));
        }
    }

    for size in [16 * 1024, ONE_MIB] {
        for (percent, expect) in [(45, MIN_LOAD), (55, TENANT)] {
            cases.push(ascii_case(
                format!("threshold_{percent}pct/{}", size_label(size)),
                expect,
                true,
                size,
                size * percent / 100,
                BASE_LOADS,
            ));
        }
    }

    // `match_rate > 0.5`: exactly half is still a miss.
    for size in [2000, ONE_MIB] {
        let label = size_label(size);
        let half = size / 2;
        for (suffix, shared, expect) in [
            ("below", half - 1, MIN_LOAD),
            ("equal", half, MIN_LOAD),
            ("above", half + 1, TENANT),
        ] {
            cases.push(ascii_case(
                format!("threshold_exact_{suffix}/{label}"),
                expect,
                false,
                size,
                shared,
                BASE_LOADS,
            ));
        }
    }

    // Imbalanced only when `max - min > 5 && max > 2 * min`.
    // A full hit selects W1 if balanced, W3 otherwise.
    cases.push(ascii_case(
        "imbalanced/16KiB".to_string(),
        MIN_LOAD,
        true,
        16 * 1024,
        16 * 1024,
        [1, 6, 1, 0],
    ));
    for (name, loads, expect) in [
        ("balance_both_exceeded", [1, 6, 1, 0], MIN_LOAD),
        ("balance_abs_only", [11, 16, 11, 10], TENANT),
        ("balance_rel_only", [2, 6, 2, 1], TENANT),
        ("balance_abs_equal", [1, 5, 1, 0], TENANT),
        ("balance_rel_equal", [7, 12, 7, 6], TENANT),
    ] {
        cases.push(ascii_case(
            format!("{name}/{}", size_label(SMALL)),
            expect,
            false,
            SMALL,
            SMALL,
            loads,
        ));
    }

    // Long non-ASCII keys: a full hit and a 45% character match.
    cases.push(CacheAwareCase::new(
        "utf8_hit/1MiB".to_string(),
        TENANT,
        ONE_MIB,
        true,
        || {
            let warm = seeded_utf8_text(SEED ^ 0x8, ONE_MIB);
            let chars = warm.chars().count();
            BuiltCase {
                fixture: CacheAwareFixture::new(vec![(TENANT, warm.clone())], &BASE_LOADS, &[]),
                probe: warm,
                matched_chars: (chars, chars),
            }
        },
    ));
    cases.push(CacheAwareCase::new(
        "utf8_45pct/1MiB".to_string(),
        MIN_LOAD,
        ONE_MIB,
        true,
        || {
            let (warm, probe, shared, total) = utf8_fork(SEED ^ 0x8, ONE_MIB, 45);
            BuiltCase {
                fixture: CacheAwareFixture::new(vec![(TENANT, warm)], &BASE_LOADS, &[]),
                probe,
                matched_chars: (shared, total),
            }
        },
    ));

    // The match rate counts characters, not bytes. Each case puts the
    // character ratio and the byte ratio on opposite sides of 0.5.
    // 40 CJK shared + 60 ASCII: characters 40/100, bytes 120/180.
    let shared = cjk(40);
    cases.push(utf8_case(
        "utf8_chars_below_bytes_above",
        MIN_LOAD,
        format!("{shared}a{}", seeded_text(SEED, 59)),
        format!("{shared}{FORK}{}", seeded_text(SEED, 59)),
        40,
    ));
    // 5 CJK + 55 ASCII shared + 40 emoji: characters 60/100, bytes 70/230.
    // The two emoji at the fork also share their first two bytes.
    let shared = format!("{}{}", cjk(5), seeded_text(SEED, 55));
    cases.push(utf8_case(
        "utf8_chars_above_bytes_below",
        TENANT,
        format!("{shared}🚀{}", "📦".repeat(39)),
        format!("{shared}🙂{}", "📦".repeat(39)),
        60,
    ));
    // Forks inside a multi-byte sequence: '中' (E4 B8 AD) and '丰'
    // (E4 B8 B0) share two bytes. Exactly half the characters match, so
    // counting either shared byte would turn this miss into a hit.
    let shared = cjk(50);
    cases.push(utf8_case(
        "utf8_fork_inside_char_exact_half",
        MIN_LOAD,
        format!("{shared}中{}", cjk(49)),
        format!("{shared}丰{}", cjk(49)),
        50,
    ));

    cases
}

/// Headers carrying `x-session-id: value`.
pub fn session_headers(value: &str) -> RequestHeaders {
    let mut headers = HashMap::new();
    headers.insert("x-session-id".to_string(), value.to_string());
    headers
}

/// Two ASCII prompts of `size` bytes that `rendezvous_hash` sends to
/// different workers when no header names a session.
pub fn rendezvous_pair(size: usize) -> (String, String) {
    let policy = RendezvousHashPolicy::new();
    let workers = workers(WORKERS);
    let pick = |p: &str| policy.select_worker(&workers, Some(p));
    let first = seeded_text(SEED ^ 0xe6, size);
    let first_worker = pick(&first);
    for j in 1..32u64 {
        let other = seeded_text(SEED ^ 0xe6 ^ (j << 32), size);
        if pick(&other) != first_worker {
            return (first, other);
        }
    }
    panic!("no two of 32 seeded {size}-byte prompts hash to different workers");
}

/// A prompt containing a JSON-like `"user"` field, which rendezvous uses as its key.
pub fn json_like_prompt(size: usize) -> String {
    const FIELD: &str = "\"user\": \"u-1\"";
    let before = (size - FIELD.len()) / 2;
    let mut prompt = seeded_text(SEED ^ 0xe7, before);
    prompt.push_str(FIELD);
    prompt.push_str(&seeded_text(SEED ^ 0xe7 ^ 1, size - before - FIELD.len()));
    prompt
}
