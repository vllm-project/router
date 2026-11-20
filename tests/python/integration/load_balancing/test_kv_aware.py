"""KV-aware process STs: pytest -> router binary -> workers + KV events.

Run after `cargo build --bin vllm-router` and `pip install -e '.[dev]'`:
    pytest tests/python/integration/load_balancing/test_kv_aware.py -v

Colocated cases drive the router through the public chat API and observe the
selected worker. The P/D section at the bottom does the same for prefill/decode
pools, on both HTTP and gRPC workers. CPU only, with a local tokenizer; no
policy/scorer/index hooks or pinned token IDs.
"""

import concurrent.futures
import collections
import json
import time

import pytest
import requests

pytestmark = [pytest.mark.ready, pytest.mark.cpu]
PROMPT = " ".join(f"t{i}" for i in range(17))


class KvScenario:
    def __init__(self, workers, url):
        self.workers = workers
        self.url = url
        self.marker = 64
        self.session = requests.Session()
        self.session.trust_env = False
        self.stream_session = requests.Session()
        self.stream_session.trust_env = False

    def request(self, text=PROMPT, **extra):
        client = self.stream_session if extra.get("stream") else self.session
        return client.post(
            f"{self.url}/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 2,
                **extra,
            },
            timeout=10,
            stream=extra.get("stream", False),
        )

    def choose(self, text=PROMPT, **extra):
        with self.request(text, **extra) as response:
            assert response.status_code == 200, response.text
            worker_id = response.json()["choices"][0]["message"]["content"]
        return next(w for w in self.workers if w.worker_id == worker_id)

    def wait_choice(self, expected, text=PROMPT, **extra):
        deadline = time.monotonic() + 10
        seen = []
        while time.monotonic() < deadline:
            actual = self.choose(text, **extra)
            seen.append(actual.worker_id)
            if actual is expected:
                return
            time.sleep(0.02)
        pytest.fail(f"Expected {expected.worker_id}, last observations: {seen[-20:]}")

    def wait_stable(self, expected, text=PROMPT, rounds=3, **extra):
        """Require ``rounds`` consecutive picks of ``expected``.

        Equal-cost ties spread across workers, so a single matching probe can
        be luck; a streak cannot, which makes this a real apply acknowledgement.
        """
        deadline = time.monotonic() + 10
        seen = []
        streak = 0
        while time.monotonic() < deadline:
            actual = self.choose(text, **extra)
            seen.append(actual.worker_id)
            streak = streak + 1 if actual is expected else 0
            if streak >= rounds:
                return
            time.sleep(0.02)
        pytest.fail(
            f"Expected {rounds} consecutive {expected.worker_id} picks; "
            f"last observations: {seen[-20:]}"
        )

    def tokens(self, text=PROMPT):
        worker = self.choose(text)
        request = worker.captured[-1]
        assert request.WhichOneof("prompt") == "token_ids"
        tokens = list(request.token_ids.ids)
        assert len(tokens) == 17, tokens
        return tokens

    @staticmethod
    def first_output(response):
        assert response.status_code == 200, response.text
        lines = response.iter_lines()
        for line in lines:
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                chunk = json.loads(line[6:])
                if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                    # Keep the iterator alive while the request is held. Its
                    # destruction can close urllib3's underlying HTTP stream.
                    return lines
        pytest.fail("No first model output")

    @staticmethod
    def next_chunk(lines):
        """Next parsed SSE chunk; fails when the stream ends early."""
        for line in lines:
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                return json.loads(line[6:])
        pytest.fail("Stream ended before the next chunk")

    @classmethod
    def consume_until_content(cls, lines):
        """Advance past metadata-only chunks to the first content delta."""
        while True:
            chunk = cls.next_chunk(lines)
            if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                return

    def publish_applied(self, worker, events, sequence=None):
        """A fresh sentinel in the SAME batch must change its selected worker.

        If cold routing already chooses the producer, first cache a shorter
        sentinel on its rival. Thus waiting cannot succeed on an unchanged
        cold tie, and negative assertions run only after the batch was applied.
        """
        marker = " ".join(f"t{i}" for i in range(self.marker, self.marker + 17))
        self.marker += 17
        baseline = self.choose(marker)
        tokens = list(baseline.captured[-1].token_ids.ids)
        if baseline is worker:
            rival = next(w for w in self.workers if w is not worker)
            rival.publish([rival.store(tokens[:8])])
            self.wait_stable(rival, marker)
        worker.publish(events + [worker.store(tokens[:16])], sequence=sequence)
        self.wait_stable(worker, marker)


@pytest.fixture
def kv_scenario(kv_workers, kv_st_runtime, kv_tokenizer, router_manager):
    scenarios = []

    def start(extra=None, worker_options=None):
        workers = [kv_workers(**options) for options in (worker_options or [{}, {}])]
        router = router_manager.start_router(
            worker_urls=[w.url for w in workers],
            policy="kv_aware",
            router_bin=str(kv_st_runtime[1]),
            extra={
                "router_temperature": 0,
                "disable_retries": True,
                "disable_circuit_breaker": True,
                **(extra or {}),
            },
            env={
                "VLLM_ROUTER_TOKENIZER": str(kv_tokenizer),
                "NO_PROXY": "127.0.0.1,localhost",
            },
        )
        for worker in workers:
            if worker.block_size and worker.model and worker.events_enabled:
                worker.wait_subscribed()
        scenario = KvScenario(workers, router.url)
        scenarios.append(scenario)
        return scenario

    yield start
    for scenario in scenarios:
        scenario.session.close()
        scenario.stream_session.close()


@pytest.mark.integration
@pytest.mark.parametrize(
    "invalidation", ["remove", "clear", "gap", "restart", "malformed"]
)
def test_kv_aware_cache_events_change_routing(kv_scenario, invalidation):
    st = kv_scenario()
    tokens = st.tokens()
    a, b = st.workers
    # N=17, B=2: A reuses 16 -> q=1; B reuses 14 -> q=3.
    cached = a.store(tokens[:16])
    st.publish_applied(b, [b.store(tokens[:14])])
    st.publish_applied(a, [cached])
    for _ in range(5):
        assert st.choose() is a

    if invalidation == "restart":
        # A lower sequence signals restart; repeating the initial 0 is a
        # duplicate, so first advance this publisher beyond 0.
        st.publish_applied(a, [cached])

    if invalidation == "remove":
        events = [
            {
                "type": "BlockRemoved",
                "medium": "device",
                "block_hashes": cached["block_hashes"],
            }
        ]
        sequence = None
    elif invalidation == "clear":
        events = [{"type": "AllBlocksCleared"}]
        sequence = None
    elif invalidation == "malformed":
        a.publish([{"type": "UnknownEvent"}])
        # Observe the invalidation before sending a fresh valid batch, so a
        # subsequent sequence gap cannot mask a broken malformed decoder.
        st.wait_choice(b)
        events, sequence = [], None
    else:
        events = []
        sequence = a.sequence + 1 if invalidation == "gap" else 0
    st.publish_applied(a, events, sequence=sequence)
    # B still has valid cache: there is no cold-tie expectation after eviction.
    for _ in range(5):
        assert st.choose() is b


@pytest.mark.integration
@pytest.mark.parametrize("tier", ["host", "external", "disk"])
@pytest.mark.parametrize("credit,expected", [(0.8, "b"), (0.2, "a")])
def test_kv_aware_transfer_credit(kv_scenario, tier, credit, expected):
    st = kv_scenario(extra={f"kv_penalty_{tier}": credit})
    tokens = st.tokens()
    a, b = st.workers
    st.publish_applied(a, [a.store(tokens[:8])])
    st.publish_applied(
        b,
        [
            b.store(
                tokens[:16],
                medium=tier,
                ownership="shared-pool" if tier == "external" else None,
            )
        ],
    )
    # GPU A: q=9. Transfer B: ceil(17-16*.8)=5, or ceil(17-16*.2)=14.
    winner = a if expected == "a" else b
    for _ in range(5):
        assert st.choose() is winner


@pytest.mark.integration
def test_kv_aware_block_boundary_keeps_last_token_work(kv_scenario):
    st = kv_scenario(worker_options=[{"block_size": 4}, {"block_size": 2}])
    tokens = st.tokens()
    a, b = st.workers
    st.publish_applied(a, [a.store(tokens[:16])])
    st.publish_applied(b, [b.store(tokens[:14])])
    shorter = " ".join(PROMPT.split()[:16])
    # N=16: A can reuse only 12 (q=4); B reuses 14 (q=2).
    assert st.choose(shorter) is b
    # N=17: A reuses 16 (q=1); B reuses 14 (q=3).
    assert st.choose() is a


@pytest.mark.integration
@pytest.mark.parametrize("connected", [True, False])
def test_kv_aware_only_credits_connected_tier_continuation(kv_scenario, connected):
    st = kv_scenario()
    tokens = st.tokens()
    a, b = st.workers
    st.publish_applied(a, [a.store(tokens[:8])])
    device = b.store(tokens[:4])
    # A missing parent leaves the host chain orphaned; it cannot contribute
    # to the request's continuous prefix even though its tokens are present.
    host = b.store(
        tokens[4:12],
        medium="host",
        parent=(device["block_hashes"][-1] if connected else b"missing-parent"),
    )
    st.publish_applied(b, [device, host])
    # A q=9; connected B q=ceil(17-4-8*.8)=7; orphaned B q=13.
    for _ in range(5):
        assert st.choose() is (b if connected else a)


@pytest.mark.integration
@pytest.mark.parametrize("discovery", ["no_events", "empty_model", "zero_block"])
def test_kv_aware_missing_kv_metadata_uses_load(kv_scenario, discovery):
    options = {
        "no_events": {"events_enabled": False},
        "empty_model": {"model": ""},
        "zero_block": {"block_size": 0},
    }[discovery]
    st = kv_scenario(worker_options=[options, options])
    # Without KV metadata routing is load-only, and equal idle load spreads, so
    # the first request is not pinned to a fixed worker. Hold whichever worker
    # takes it, then the other one must win every later probe.
    gates = [(worker, worker.hold_next()) for worker in st.workers]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(st.request, stream=True)
        held = None
        try:
            deadline = time.time() + 10
            while time.time() < deadline and held is None:
                for worker, gate in gates:
                    if gate.entered.is_set():
                        held = (worker, gate)
                        break
                time.sleep(0.02)
            assert held is not None, "no worker took the request"
            for worker, gate in gates:
                if gate is not held[1]:
                    worker.drop_gate(gate)
            other = next(w for w in st.workers if w is not held[0])
            for _ in range(5):
                assert st.choose() is other
        finally:
            for _, gate in gates:
                gate.output.set()
                gate.finish.set()
            pending.result(timeout=5).close()


@pytest.mark.integration
@pytest.mark.parametrize("termination", ["error", "cancel", "retry"])
def test_kv_aware_failure_and_cancel_release_prefill(kv_scenario, termination):
    st = kv_scenario(
        extra=(
            {
                "disable_retries": False,
                "retry_max_retries": 2,
                "retry_initial_backoff_ms": 1,
                "retry_max_backoff_ms": 1,
                "retry_jitter_factor": 0,
            }
            if termination == "retry"
            else None
        )
    )
    tokens = st.tokens()
    a, b = st.workers
    st.publish_applied(b, [b.store(tokens[:14])])
    st.publish_applied(a, [a.store(tokens[:16])])
    assert st.choose() is a
    if termination in ("error", "retry"):
        total_before = sum(len(worker.captured) for worker in st.workers)
        a.fail_next()
        with st.request() as response:
            if termination == "retry":
                assert response.status_code == 200, response.text
                # The failed prefill is still accounted while the retry picks,
                # so the retry lands on the other (cheaper) worker. What must
                # hold is that the failed attempt is actually retried once.
                assert response.json()["choices"][0]["message"]["content"] in {
                    worker.worker_id for worker in st.workers
                }
                assert (
                    sum(len(worker.captured) for worker in st.workers)
                    == total_before + 2
                )
            else:
                assert response.status_code == 503, response.text
    else:
        gate = a.hold_next()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(st.request, stream=True)
            try:
                assert gate.entered.wait(5)
                response = pending.result(timeout=5)
                response.close()
                assert gate.done.wait(5), "Cancelled gRPC request stayed active"
            finally:
                gate.output.set()
                gate.finish.set()
    # Clean A costs 1 against B's 3, and a leaked prefill slot lifts A to
    # (1+1)*2 = 4, which still flips the winner.
    st.wait_choice(a)
    for _ in range(5):
        assert st.choose() is a


@pytest.mark.integration
def test_kv_aware_prefill_decode_and_finish_change_routing(kv_scenario):
    st = kv_scenario()
    tokens = st.tokens()
    a, b = st.workers
    st.publish_applied(b, [b.store(tokens[:14])])
    st.publish_applied(a, [a.store(tokens[:16])])
    assert st.choose() is a
    gate = a.hold_next()
    # Use a separate client: the held streaming request overlaps probe requests.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(st.request, stream=True)
        try:
            assert gate.entered.wait(5), "Cached worker did not receive request"
            response = pending.result(timeout=5)
            lines = response.iter_lines()
            # The worker yields PromptInfo before the output gate opens, so the
            # router's metadata-only chunk is already in flight. Consume it
            # first: otherwise the prefill probe below can win the race against
            # the first content delta, and a decode-started-on-metadata
            # regression would still pass.
            delta = (st.next_chunk(lines).get("choices") or [{}])[0].get("delta", {})
            assert not delta.get("content"), delta
            # A prefill: (1+1)*2=4; B idle: 3 -> B.
            st.wait_choice(b)
            gate.output.set()
            st.consume_until_content(lines)
            # A decode: 1*2=2; B idle: 3 -> A.
            st.wait_choice(a)
            gate.finish.set()
            assert b"data: [DONE]" in list(lines)
            response.close()
            assert gate.done.wait(5)
            # Reduce B's prefix to 12 -> q=5. A second held prefill costs
            # clean A (1+1)*2=4, but 6 if the first decode slot leaked.
            # The winners differ strictly; no equal-cost tie is assumed.
            st.publish_applied(b, [{"type": "AllBlocksCleared"}, b.store(tokens[:12])])
            second = a.hold_next()
            second_pending = executor.submit(st.request, stream=True)
            assert second.entered.wait(5)
            second_response = second_pending.result(timeout=5)
            try:
                st.wait_choice(a)
                second.output.set()
                second.finish.set()
                assert b"data: [DONE]" in list(second_response.iter_lines())
            finally:
                second.finish.set()
                second_response.close()
            assert second.done.wait(5)
            st.wait_choice(a)
        finally:
            gate.output.set()
            gate.finish.set()


@pytest.mark.integration
@pytest.mark.parametrize("namespace", ["cache_salt", "lora"])
def test_kv_aware_cache_namespace_isolation(kv_scenario, namespace):
    st = kv_scenario()
    tokens = st.tokens()
    a, b = st.workers
    st.publish_applied(b, [b.store(tokens[:14])])
    st.publish_applied(a, [a.store(tokens[:16])])
    assert st.choose() is a
    if namespace == "lora":
        st.publish_applied(b, [b.store(tokens[:14], lora_name="adapter-b")])
        # A's longer base-model cache must not match a LoRA request.
        for _ in range(5):
            assert st.choose(lora_name="adapter-b") is b
    else:
        # With salt neither worker can claim placement credit. Hold an A
        # decode slot: load-only chooses B (17 vs 34), whereas wrongly using
        # cache credit still chooses A (2 vs 3).
        gate = a.hold_next()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(st.request, stream=True)
            response = None
            try:
                assert gate.entered.wait(5)
                gate.output.set()
                response = pending.result(timeout=5)
                lines = st.first_output(response)
                st.wait_choice(a)
                for _ in range(5):
                    assert st.choose(cache_salt="tenant-b") is b
                gate.finish.set()
                assert b"data: [DONE]" in list(lines)
            finally:
                gate.finish.set()
                if response is not None:
                    response.close()


@pytest.mark.integration
def test_kv_aware_positive_temperature_samples_equal_cost_workers(kv_scenario):
    st = kv_scenario(extra={"router_temperature": 100})
    # Equal costs give equal softmax weights. A broad bound makes accidental
    # statistical failure negligible while rejecting deterministic picking.
    counts = collections.Counter(st.choose().worker_id for _ in range(200))
    for worker in st.workers:
        assert counts[worker.worker_id] >= 20, counts


# ---------------------------------------------------------------------------
# P/D disaggregation
#
# The prefix-credit rule must pick the prefill worker on both transports the
# router supports: HTTP pools (`/tokenize` + `/kv_event_sources` probed over
# HTTP) and gRPC pools (control RPCs plus the unary `Generate` prefill leg).
#
# Synchronization and assertion are deliberately separate. Publishing is
# acknowledged through the router's own telemetry: it exports
# `vllm_router_pd_credited_prefix_tokens_total`, so the test republishes until a
# probe's decision reports the expected credited tokens. That is an apply ack,
# not a routing guess: no pick is inspected to decide whether to continue.
# The assertion is then a fixed sample that must all pick the credited worker.
# With credit disabled the ack itself fails, and a tie would still only pass the
# sample with probability `2**-PD_SAMPLE`.
# ---------------------------------------------------------------------------

PD_BLOCK_SIZE = 2
PD_PROMPT_TOKENS = [index + 1 for index in range(17)]
# A caches 14 tokens (3 left to compute) and B caches 12 (5 left). Once the
# batch is applied the winner is deterministic; without credit it is a tie.
PD_PREFIX_A = PD_PROMPT_TOKENS[:14]
PD_PREFIX_B = PD_PROMPT_TOKENS[:12]
# Publish replay and the fixed settle that follows each attempt.
PD_REPLAYS = 3
PD_SETTLE = 0.2
# Fixed sample size: a uniform tie passes it with probability 2**-PD_SAMPLE.
PD_SAMPLE = 10
# Counter the router increments by the credited prefix tokens of a decision.
PD_CREDIT_METRIC = "vllm_router_pd_credited_prefix_tokens_total"
# Counter that only moves once a decision used a verified worker identity.
PD_DECISIONS_METRIC = "vllm_router_prefix_routing_decisions_total"
PD_VERIFIED_OUTCOME = 'stage="prefill",outcome="prefix_cost"'


def _pd_post(base: str, path: str, payload: dict, timeout: float = 30):
    session = requests.Session()
    session.trust_env = False
    try:
        return session.post(f"{base}{path}", json=payload, timeout=timeout)
    finally:
        session.close()


def _pd_completion(base: str, content: str):
    return _pd_post(
        base,
        "/v1/completions",
        {"model": "test-model", "prompt": content, "max_tokens": 1},
    )


def _pd_chat(base: str, content: str):
    return _pd_post(
        base,
        "/v1/chat/completions",
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 2,
        },
    )


def _pd_served(url: str) -> int:
    try:
        return requests.get(f"{url}/request_count", timeout=2).json().get("count", 0)
    except (requests.RequestException, ValueError):
        return 0


def _pd_store(url: str, tokens) -> None:
    response = _pd_post(url, "/kv_store", {"token_ids": tokens})
    response.raise_for_status()


def _pd_subscribe(url: str) -> None:
    """Wait for the router's KV subscriber on this worker (XPUB handshake)."""
    response = _pd_post(url, "/kv_wait_subscribed", {}, timeout=15)
    if response.status_code != 200:
        raise AssertionError(f"{url}: KV subscriber did not attach: {response.text}")


def _pd_kv_status(url: str) -> dict:
    try:
        return requests.get(f"{url}/kv_status", timeout=2).json()
    except (requests.RequestException, ValueError):
        return {}


def _pd_publish(url: str, tokens, rounds: int = PD_REPLAYS) -> None:
    """Publish a prefix. This is synchronization only, never an assertion.

    The subscriber handshake is awaited first (PUB drops frames for a peer that
    is still joining), then the batch is replayed because BlockStored is
    idempotent, and a fixed settle follows so the index can apply it. No
    routing result is inspected here.
    """
    _pd_subscribe(url)
    for _ in range(rounds):
        _pd_store(url, tokens)
        time.sleep(0.05)
    time.sleep(PD_SETTLE)


def _pd_probe(router_url: str, prefill_urls, content: str):
    """One completion probe: the prefill that served it, plus the reply."""
    before = {url: _pd_served(url) for url in prefill_urls}
    response = _pd_completion(router_url, content)
    assert response.status_code == 200, response.text
    deadline = time.time() + 10
    while time.time() < deadline:
        for url in prefill_urls:
            if _pd_served(url) > before[url]:
                return url, response
        time.sleep(0.02)
    loads = [_pd_served(url) for url in prefill_urls]
    raise AssertionError(f"no prefill worker served the probe: {loads}")


def _pd_metric(prom_url: str, name: str, contains: str = "") -> float:
    """Sum one Prometheus counter (optionally only matching lines)."""
    try:
        text = requests.get(f"{prom_url}/metrics", timeout=2).text
    except requests.RequestException:
        return 0.0
    total = 0.0
    for line in text.splitlines():
        if not line.startswith(name):
            continue
        if contains and contains not in line:
            continue
        try:
            total += float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
    return total


def _pd_credited_tokens(prom_url: str) -> float:
    """Credited prefix tokens the router has reported so far."""
    return _pd_metric(prom_url, PD_CREDIT_METRIC)


def _pd_wait_verified_identity(prom_url: str, probe, timeout: float = 20.0) -> None:
    """Wait until a prefill decision used a verified worker identity.

    ``PdKvState`` binds workers on a background tick (1 s refresh, 3 s control
    timeout) and denies placement credit until the binding exists, so a credit
    ack issued straight after `start_router` can race it. The decision counter
    moves to ``prefix_cost`` only once the identity is verified, so this blocks
    on telemetry and not on a routing guess.
    """
    before = _pd_metric(prom_url, PD_DECISIONS_METRIC, PD_VERIFIED_OUTCOME)
    deadline = time.time() + timeout
    while time.time() < deadline:
        probe()
        if _pd_metric(prom_url, PD_DECISIONS_METRIC, PD_VERIFIED_OUTCOME) > before:
            return
        time.sleep(0.2)
    raise AssertionError("router never reported a verified prefill identity")


def _pd_publish_until_credit(
    prom_url: str,
    publish,
    probe,
    tokens,
    attempts: int = 5,
    status=None,
) -> None:
    """Publish until the router reports ``len(tokens)`` credited prefix tokens.

    This is the apply acknowledgement: the router increments
    ``PD_CREDIT_METRIC`` by the credited tokens of each prefill decision, so a
    probe that reports the full prefix proves the batch reached the index. The
    probe's own pick is not consulted, and once the ack lands the sample below
    is fixed and retry-free.
    """
    for _ in range(attempts):
        publish()
        before = _pd_credited_tokens(prom_url)
        probe()
        if _pd_credited_tokens(prom_url) - before >= len(tokens):
            return
        time.sleep(0.05)
    detail = status() if status is not None else "n/a"
    raise AssertionError(
        f"router never credited {len(tokens)} prefix tokens; "
        f"{PD_CREDIT_METRIC}={_pd_credited_tokens(prom_url)}; mock={detail}"
    )


def _pd_assert_picks(router_url: str, prefill_urls, expected, sample: int = PD_SAMPLE):
    """Every probe in a fixed sample must pick ``expected``. No retries."""
    seen = []
    response = None
    for _ in range(sample):
        picked, response = _pd_probe(router_url, prefill_urls, PROMPT)
        seen.append(picked)
    assert seen == [expected] * sample, seen
    return response


@pytest.mark.integration
def test_kv_aware_pd_http_prefill_follows_prefix_credit(
    mock_workers, kv_st_runtime, router_manager
):
    _, prefill_urls, _ = mock_workers(n=2, args=["--kv-block-size", str(PD_BLOCK_SIZE)])
    _, decode_urls, decode_ids = mock_workers(n=1)
    # Launch the same pinned binary as the other cases: the Python launcher
    # runs the engine in-process and raced its own bind on L1 (`EADDRINUSE`).
    router = router_manager.start_router(
        policy="kv_aware",
        vllm_pd_disaggregation=True,
        prefill_urls=[(url, None) for url in prefill_urls],
        decode_urls=decode_urls,
        router_bin=str(kv_st_runtime[1]),
        extra={"worker_startup_check_interval": 1},
        env={"NO_PROXY": "127.0.0.1,localhost"},
    )

    prom = f"http://127.0.0.1:{router.prometheus_port}"
    # Bindings land on a background tick; wait for a verified identity before
    # the credit ack, otherwise the ack races the refresh and reads 0.
    _pd_wait_verified_identity(
        prom, lambda: _pd_probe(router.url, prefill_urls, PROMPT)
    )
    # B holds the shorter prefix, so a cold tie cannot reach B by luck: without
    # its batch every pick would be uniform, and the sample would catch it.
    _pd_publish_until_credit(
        prom,
        lambda: _pd_publish(prefill_urls[1], PD_PREFIX_B),
        lambda: _pd_probe(router.url, prefill_urls, PROMPT),
        PD_PREFIX_B,
        status=lambda: {url: _pd_kv_status(url) for url in prefill_urls},
    )
    _pd_assert_picks(router.url, prefill_urls, prefill_urls[1])

    # A holds the longer prefix of the same prompt and must win every pick.
    _pd_publish_until_credit(
        prom,
        lambda: _pd_publish(prefill_urls[0], PD_PREFIX_A),
        lambda: _pd_probe(router.url, prefill_urls, PROMPT),
        PD_PREFIX_A,
        status=lambda: {url: _pd_kv_status(url) for url in prefill_urls},
    )
    response = _pd_assert_picks(router.url, prefill_urls, prefill_urls[0])
    assert response.headers.get("X-Worker-Id") in decode_ids, response.headers


def _pd_grpc_probe(router_url: str, prefill, content: str):
    """One chat probe: the prefill index that took it, plus the reply."""
    before = [len(worker.captured) for worker in prefill]
    response = _pd_chat(router_url, content)
    assert response.status_code == 200, response.text
    deadline = time.time() + 10
    while time.time() < deadline:
        after = [len(worker.captured) for worker in prefill]
        picked = [
            index for index in range(len(prefill)) if after[index] > before[index]
        ]
        if picked:
            return picked[0], response
        time.sleep(0.01)
    counts = [len(worker.captured) for worker in prefill]
    raise AssertionError(f"no prefill worker took the probe: {counts}")


def _pd_grpc_assert_picks(router_url: str, prefill, expected, sample: int = PD_SAMPLE):
    """Every probe in a fixed sample must pick prefill ``expected``."""
    seen = []
    response = None
    for _ in range(sample):
        picked, response = _pd_grpc_probe(router_url, prefill, PROMPT)
        seen.append(picked)
    assert seen == [expected] * sample, seen
    return response


def _pd_grpc_publish(worker, tokens, rounds: int = PD_REPLAYS) -> None:
    """Publish a prefix on a gRPC worker. Synchronization only, no assertion.

    The subscriber handshake is already done by ``wait_subscribed``, but the
    index applies batches asynchronously; replaying the batch with a fixed
    settle keeps the sample below free of that race without retrying on a
    routing result.
    """
    for _ in range(rounds):
        worker.publish([worker.store(tokens)])
        time.sleep(0.05)
    time.sleep(PD_SETTLE)


@pytest.mark.integration
def test_kv_aware_pd_grpc_prefill_follows_prefix_credit(
    kv_workers, kv_st_runtime, kv_tokenizer, router_manager
):
    prefill = [kv_workers(block_size=PD_BLOCK_SIZE) for _ in range(2)]
    # Decode publishes no cache events; only the prefill leg is scored here.
    decode = [kv_workers(block_size=PD_BLOCK_SIZE, events_enabled=False)]
    router = router_manager.start_router(
        policy="kv_aware",
        vllm_pd_disaggregation=True,
        prefill_urls=[(worker.url, None) for worker in prefill],
        decode_urls=[worker.url for worker in decode],
        router_bin=str(kv_st_runtime[1]),
        extra={
            "router_temperature": 0,
            "disable_retries": True,
            "disable_circuit_breaker": True,
            "worker_startup_check_interval": 1,
        },
        env={
            "VLLM_ROUTER_TOKENIZER": str(kv_tokenizer),
            "NO_PROXY": "127.0.0.1,localhost",
        },
    )
    for worker in prefill:
        worker.wait_subscribed()

    # `wait_subscribed` covers the subscriber handshake; the credit counter is
    # the apply ack for the batch itself, and the samples are the assertions.
    prom = f"http://127.0.0.1:{router.prometheus_port}"
    _pd_wait_verified_identity(
        prom, lambda: _pd_grpc_probe(router.url, prefill, PROMPT)
    )
    _pd_publish_until_credit(
        prom,
        lambda: _pd_grpc_publish(prefill[1], PD_PREFIX_B),
        lambda: _pd_grpc_probe(router.url, prefill, PROMPT),
        PD_PREFIX_B,
    )
    _pd_grpc_assert_picks(router.url, prefill, 1)

    _pd_publish_until_credit(
        prom,
        lambda: _pd_grpc_publish(prefill[0], PD_PREFIX_A),
        lambda: _pd_grpc_probe(router.url, prefill, PROMPT),
        PD_PREFIX_A,
    )
    response = _pd_grpc_assert_picks(router.url, prefill, 0)
    assert response.json()["choices"][0]["message"]["content"] in {
        worker.worker_id for worker in decode
    }
