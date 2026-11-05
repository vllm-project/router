"""The #427 prototype's retain and release lifecycle and its report, with a fake clock and no engine.

benches/kv_retention/kv_retention_lifecycle.py and report.py import no vLLM, so these run wherever the Router's tests
do. They do not validate vLLM's behavior: the connector and the policy on a real engine are the live probes in
benches/kv_retention/run_eval.py.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ready, pytest.mark.cpu]

MODULE = (
    Path(__file__).parents[3] / "benches" / "kv_retention" / "kv_retention_lifecycle.py"
)
SPEC = importlib.util.spec_from_file_location("kv_retention_lifecycle", MODULE)
assert SPEC and SPEC.loader
lifecycle = importlib.util.module_from_spec(SPEC)
sys.modules["kv_retention_lifecycle"] = lifecycle
SPEC.loader.exec_module(lifecycle)

POOL = 1000
GENERATION = "worker-a"


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def ledger(clock):
    return lifecycle.RetentionLedger(POOL, 0.5, 0.3, 10.0, GENERATION, clock)


@pytest.fixture
def book(clock):
    return lifecycle.LeaseBook(
        100, 0.5, 0.3, 120.0, GENERATION, tokens_per_chunk=16, clock=clock
    )


def raw_hint(session="s", generation=0, **fields):
    return {
        "version": "0.1",
        "hint_id": f"{session}:{generation}",
        "op": "retain",
        "scope": {"tenant": "t", "cache_namespace": "default", "session": session},
        "generation": generation,
        "tiers": ["gpu"],
        "ttl_seconds": 60.0,
        "priority": 0,
        **fields,
    }


def make_hint(session="s", generation=0, tenant=None, **fields):
    raw = raw_hint(session, generation, **fields)
    if tenant is not None:
        raw["scope"]["tenant"] = tenant
    hint, ignored = lifecycle.parse_hint({lifecycle.HINT_KEY: raw})
    assert ignored is None
    return hint


def outcomes(ledger):
    return [(end.hint_id, end.outcome, end.reason) for end in ledger.take_ended()]


class TestParse:
    def test_no_hint_is_nothing_to_ignore(self):
        assert lifecycle.parse_hint(None) == (None, None)
        assert lifecycle.parse_hint({"other": 1}) == (None, None)

    def test_unsupported_version_is_ignored_without_raising(self):
        assert lifecycle.parse_hint({lifecycle.HINT_KEY: raw_hint(version="9.9")}) == (
            None,
            "unsupported_version",
        )

    @pytest.mark.parametrize(
        "fields",
        [
            {"op": "pin"},
            {"tiers": ["disk"]},
            {"tiers": "gpu"},
            {"generation": -1},
            {"generation": True},
            {"generation": "1"},
            {"ttl_seconds": "soon"},
            {"ttl_seconds": float("inf")},
            {"priority": 1.5},
            {"hint_id": ""},
            {"scope": {"tenant": "t"}},
            {"scope": "s"},
            {"worker_generation": 3},
        ],
    )
    def test_malformed_is_ignored_without_raising(self, fields):
        assert lifecycle.parse_hint({lifecycle.HINT_KEY: raw_hint(**fields)}) == (
            None,
            "malformed",
        )

    def test_a_hint_that_is_not_an_object_is_malformed(self):
        assert lifecycle.parse_hint({lifecycle.HINT_KEY: "retain"}) == (
            None,
            "malformed",
        )

    def test_a_clients_hint_body_parses(self):
        body = lifecycle.hint_body("a", 3, ["gpu", "cpu"], 30.0, tenant="x")
        hint, ignored = lifecycle.parse_hint({lifecycle.HINT_KEY: body})
        assert ignored is None
        assert hint.hint_id == "a:3"
        assert hint.key == ("x", "default", "a")
        assert hint.tiers == frozenset({"gpu", "cpu"})

    def test_scope_makes_the_key(self):
        hint = make_hint("a", 3, tenant="x")
        assert hint.key == ("x", "default", "a")
        assert hint.generation == 3
        assert hint.tiers == frozenset({"gpu"})


class TestHold:
    def test_accepts_inside_the_budget_and_clamps_the_ttl(self, ledger):
        ack = ledger.retain(make_hint(ttl_seconds=60.0), "r0", 100, 1600)
        assert (ack.state, ack.reason, ack.ttl_seconds, ack.tokens) == (
            "accepted",
            None,
            10.0,
            1600,
        )
        assert ack.worker_generation == GENERATION
        assert ledger.held_blocks == 100

    def test_retain_is_idempotent_and_renews(self, ledger, clock):
        hint = make_hint()
        assert ledger.retain(hint, "r0", 100, 1600).reason is None
        clock.now += 9.0
        renewed = ledger.retain(hint, "r0-retry", 100, 1600)
        assert (renewed.state, renewed.reason) == ("accepted", lifecycle.RENEWED)
        assert ledger.held_blocks == 100, "a duplicate holds nothing more"
        clock.now += 9.0
        ledger.expire()
        assert ledger.take_releases() == set(), "the renewal moved the deadline"
        clock.now += 2.0
        ledger.expire()
        assert ledger.take_releases() == {"r0"}, "the hold stays the first request's"

    def test_duplicate_after_the_end_does_not_bring_a_hold_back(self, ledger):
        hint = make_hint()
        ledger.retain(hint, "r0", 100, 1600)
        ledger.release(make_hint(op="release"))
        ack = ledger.retain(hint, "r0-late", 100, 1600)
        assert (ack.state, ack.reason) == ("rejected", "already_ended")
        assert ledger.held_blocks == 0

    def test_a_newer_generation_supersedes(self, ledger):
        ledger.retain(make_hint(generation=0), "r0", 100, 1600)
        ack = ledger.retain(make_hint(generation=1), "r1", 120, 1920)
        assert ack.state == "accepted"
        assert ledger.held_blocks == 120
        assert ledger.take_releases() == {"r0"}
        assert outcomes(ledger) == [("s:0", "released", "superseded")]

    def test_a_stale_generation_is_refused(self, ledger):
        ledger.retain(make_hint(generation=5), "r5", 100, 1600)
        ack = ledger.retain(make_hint(generation=4), "r4", 100, 1600)
        assert (ack.state, ack.reason) == ("rejected", "stale_generation")
        assert ledger.held_blocks == 100
        release = ledger.release(make_hint(generation=4, op="release"))
        assert (release.state, release.reason) == ("rejected", "stale_generation")
        assert ledger.held_blocks == 100, "a delayed release cannot end a newer hold"

    def test_the_generation_is_remembered_after_the_hold_ended(self, ledger):
        ledger.retain(make_hint(generation=5), "r5", 100, 1600)
        ledger.reset()
        ack = ledger.retain(make_hint(generation=4), "r4", 100, 1600)
        assert (ack.state, ack.reason) == ("rejected", "stale_generation")

    def test_one_generation_has_one_hint(self, ledger):
        ledger.retain(make_hint(generation=1), "r1", 100, 1600)
        other = make_hint(generation=1, hint_id="other")
        ack = ledger.retain(other, "r1b", 100, 1600)
        assert (ack.state, ack.reason) == ("rejected", "generation_conflict")

    @pytest.mark.parametrize("ttl", [0.0, -1.0])
    def test_no_ttl_is_refused(self, ledger, ttl):
        ack = ledger.retain(make_hint(ttl_seconds=ttl), "r0", 100, 1600)
        assert (ack.state, ack.reason) == ("rejected", "bad_ttl")

    def test_expiry_ends_the_hold_once(self, ledger, clock):
        ledger.retain(make_hint(), "r0", 100, 1600)
        clock.now += 9.9
        ledger.expire()
        assert ledger.held_blocks == 100
        clock.now += 0.1
        ledger.expire()
        ledger.expire()
        assert ledger.held_blocks == 0
        assert ledger.take_releases() == {"r0"}
        assert ledger.take_releases() == set()
        ended = ledger.take_ended()
        assert [(end.outcome, end.held_seconds) for end in ended] == [("expired", 10.0)]

    def test_over_budgets_is_refused_and_holds_nothing(self, ledger):
        too_big_for_tenant = ledger.retain(make_hint("a"), "ra", 301, 4816)
        assert (too_big_for_tenant.state, too_big_for_tenant.reason) == (
            "rejected",
            "over_tenant_budget",
        )
        assert (
            ledger.retain(make_hint("a", tenant="t1"), "ra", 300, 4800).state
            == "accepted"
        )
        assert (
            ledger.retain(make_hint("b", tenant="t2"), "rb", 200, 3200).state
            == "accepted"
        )
        over_engine = ledger.retain(make_hint("c", tenant="t3"), "rc", 1, 16)
        assert (over_engine.state, over_engine.reason) == (
            "rejected",
            "over_engine_budget",
        )
        assert ledger.held_blocks == 500

    def test_a_tenants_budget_is_its_own(self, ledger):
        assert (
            ledger.retain(make_hint("a", tenant="t1"), "ra", 300, 4800).state
            == "accepted"
        )
        refused = ledger.retain(make_hint("b", tenant="t1"), "rb", 1, 16)
        assert (refused.state, refused.reason) == ("rejected", "over_tenant_budget")
        assert (
            ledger.retain(make_hint("c", tenant="t2"), "rc", 1, 16).state == "accepted"
        )

    def test_a_superseded_holds_blocks_do_not_count_against_its_successor(self, ledger):
        assert ledger.retain(make_hint("a", 0), "r0", 300, 4800).state == "accepted"
        assert ledger.retain(make_hint("a", 1), "r1", 300, 4800).state == "accepted"
        assert ledger.held_blocks == 300

    def test_refused_generation_can_try_again(self, ledger):
        ledger.retain(make_hint("big"), "r0", 300, 4800)
        hint = make_hint("next")
        assert ledger.retain(hint, "r1", 10, 160).reason == "over_tenant_budget"
        ledger.release(make_hint("big", op="release"))
        assert ledger.retain(hint, "r1-again", 10, 160).state == "accepted"

    def test_next_turn_ends_the_hold_as_used(self, ledger):
        ledger.retain(make_hint(generation=0), "r0", 100, 1600)
        ledger.on_next_turn(("t", "default", "s"), 1, 1600)
        assert ledger.take_releases() == {"r0"}
        ended = ledger.take_ended()
        assert [(end.outcome, end.hit_tokens) for end in ended] == [("used", 1600)]

    def test_next_turn_without_a_hit_did_not_use_it(self, ledger):
        ledger.retain(make_hint(generation=0), "r0", 100, 1600)
        ledger.on_next_turn(("t", "default", "s"), 1, 0)
        assert outcomes(ledger) == [("s:0", "released", "no_hit")]

    def test_a_retry_of_the_same_turn_leaves_the_hold(self, ledger):
        ledger.retain(make_hint(generation=3), "r3", 100, 1600)
        ledger.on_next_turn(("t", "default", "s"), 3, 1600)
        assert ledger.held_blocks == 100

    def test_next_turn_of_another_tenant_touches_nothing(self, ledger):
        ledger.retain(make_hint(tenant="t1"), "r0", 100, 1600)
        ledger.on_next_turn(("t2", "default", "s"), 1, 1600)
        assert ledger.held_blocks == 100


class TestRelease:
    def test_release_is_idempotent(self, ledger):
        ledger.retain(make_hint(), "r0", 100, 1600)
        first = ledger.release(make_hint(generation=1, op="release"))
        assert (first.state, first.reason, first.tokens) == ("released", None, 1600)
        again = ledger.release(make_hint(generation=1, op="release"))
        assert (again.state, again.reason, again.tokens) == (
            "released",
            "no_live_hold",
            0,
        )
        assert ledger.take_releases() == {"r0"}

    def test_unknown_release_succeeds_as_a_no_op(self, ledger):
        ack = ledger.release(make_hint("never", op="release"))
        assert (ack.state, ack.reason) == ("released", "no_live_hold")
        assert ledger.take_releases() == set()

    def test_another_tenants_release_reaches_nothing(self, ledger):
        ledger.retain(make_hint("s", tenant="owner"), "r0", 100, 1600)
        ack = ledger.release(
            make_hint("s", generation=9, tenant="intruder", op="release")
        )
        assert (ack.state, ack.reason, ack.tokens) == ("released", "no_live_hold", 0)
        assert ledger.held_blocks == 100
        assert ledger.take_releases() == set()

    def test_another_namespaces_release_reaches_nothing(self, ledger):
        ledger.retain(make_hint("s"), "r0", 100, 1600)
        raw = raw_hint("s", 9, op="release")
        raw["scope"]["cache_namespace"] = "other"
        hint, _ = lifecycle.parse_hint({lifecycle.HINT_KEY: raw})
        ledger.release(hint)
        assert ledger.held_blocks == 100


class TestDemand:
    def test_yield_releases_oldest_first_until_enough(self, ledger):
        for index, session in enumerate("abc"):
            ledger.retain(
                make_hint(session, tenant=f"t{index}"), f"r{session}", 100, 1600
            )
        assert ledger.yield_blocks(150) == 200
        assert ledger.take_releases() == {"ra", "rb"}
        assert [end.outcome for end in ledger.take_ended()] == ["evicted", "evicted"]
        assert ledger.held_blocks == 100

    def test_yield_spares_the_requesters_own_hold(self, ledger):
        ledger.retain(make_hint("a"), "ra", 100, 1600)
        ledger.retain(make_hint("b"), "rb", 100, 1600)
        assert ledger.yield_blocks(1000, exclude=("t", "default", "a")) == 100
        assert ledger.take_releases() == {"rb"}

    def test_a_renewed_hold_is_younger(self, ledger):
        ledger.retain(make_hint("a"), "ra", 100, 1600)
        ledger.retain(make_hint("b"), "rb", 100, 1600)
        ledger.retain(make_hint("a"), "ra-retry", 100, 1600)
        ledger.yield_blocks(1)
        assert ledger.take_releases() == {"rb"}

    def test_with_nothing_to_yield_it_says_so(self, ledger):
        assert ledger.yield_blocks(10) == 0

    def test_released_blocks_count_until_they_are_freed(self, ledger):
        ledger.retain(make_hint(), "r0", 100, 1600)
        ledger.yield_blocks(10)
        assert ledger.held_blocks == 0
        assert ledger.releasing_blocks == 100
        ledger.take_releases()
        assert ledger.releasing_blocks == 100, "handed to the worker is not yet free"
        ledger.on_freed({"r0"})
        assert ledger.releasing_blocks == 0

    def test_budget_room_comes_back_with_the_release(self, ledger):
        ledger.retain(make_hint("a"), "ra", 300, 4800)
        ledger.yield_blocks(1)
        assert ledger.retain(make_hint("b"), "rb", 300, 4800).state == "accepted"


class TestWorkerGeneration:
    def test_every_answer_names_the_generation(self, ledger):
        ledger.retain(make_hint("a"), "ra", 100, 1600)
        ledger.yield_blocks(1)
        assert ledger.take_ended()[0].worker_generation == GENERATION
        assert (
            ledger.release(make_hint("a", op="release")).worker_generation == GENERATION
        )

    def test_a_restarted_worker_refuses_a_hint_for_the_old_one(self, clock):
        restarted = lifecycle.RetentionLedger(POOL, 0.5, 0.3, 10.0, "worker-b", clock)
        hint = make_hint(worker_generation=GENERATION)
        ack = restarted.retain(hint, "r0", 100, 1600)
        assert (ack.state, ack.reason, ack.worker_generation) == (
            "rejected",
            "stale_worker_generation",
            "worker-b",
        )
        release = restarted.release(
            make_hint(op="release", worker_generation=GENERATION)
        )
        assert (release.state, release.reason) == (
            "rejected",
            "stale_worker_generation",
        )

    def test_a_restarted_worker_has_no_old_generations(self, clock):
        first = lifecycle.RetentionLedger(POOL, 0.5, 0.3, 10.0, "worker-a", clock)
        first.retain(make_hint(generation=7), "r7", 100, 1600)
        restarted = lifecycle.RetentionLedger(POOL, 0.5, 0.3, 10.0, "worker-b", clock)
        assert (
            restarted.retain(make_hint(generation=2), "r2", 100, 1600).state
            == "accepted"
        )

    def test_reset_ends_every_hold(self, ledger):
        ledger.retain(make_hint("a"), "ra", 100, 1600)
        ledger.retain(make_hint("b"), "rb", 100, 1600)
        ledger.reset()
        assert ledger.take_releases() == {"ra", "rb"}
        assert {end.reason for end in ledger.take_ended()} == {"reset"}


class TestLease:
    def test_open_acknowledges_with_the_tiers_ttl(self, book):
        ack = book.open(make_hint(tiers=["cpu"], ttl_seconds=600.0))
        assert (ack.state, ack.tier, ack.ttl_seconds) == ("accepted", "cpu", 120.0)

    def test_covered_chunks_rank_above_unleased(self, book):
        hint = make_hint(tiers=["cpu"], priority=1)
        book.open(hint)
        assert book.tag(hint.key, ["c0", "c1"]) == ["c0", "c1"]
        assert book.rank("c0") == 2
        assert book.rank("elsewhere") == 0

    def test_no_lease_covers_nothing(self, book):
        assert book.tag(("t", "default", "s"), ["c0"]) == []
        assert book.rank("c0") == 0

    def test_a_turns_end_restarts_the_clock(self, book, clock):
        hint = make_hint(tiers=["cpu"], ttl_seconds=120.0)
        book.open(hint)
        book.tag(hint.key, ["c0"])
        clock.now += 100.0
        book.touch(hint.key)
        clock.now += 100.0
        assert book.expire() == []
        clock.now += 20.0
        assert book.expire() == ["c0"]
        assert book.rank("c0") == 0
        ended = book.take_ended()
        assert [(end.outcome, end.tokens) for end in ended] == [("expired", 16)]

    def test_a_chunk_two_sessions_share_outlives_one_lease(self, book):
        a, b = (
            make_hint("a", tiers=["cpu"], priority=2),
            make_hint("b", tiers=["cpu"], priority=0),
        )
        book.open(a)
        book.open(b)
        book.tag(a.key, ["shared"])
        book.tag(b.key, ["shared"])
        assert book.rank("shared") == 3
        book.release(make_hint("a", op="release", tiers=["cpu"]))
        assert book.rank("shared") == 1

    def test_the_budgets_clip_the_tail(self, book):
        hint = make_hint(tiers=["cpu"])
        book.open(hint)
        covered = book.tag(hint.key, [f"c{index}" for index in range(40)])
        assert covered == [f"c{index}" for index in range(30)], (
            "the tenant's 30 chunks, from the head"
        )
        assert book.chunks_clipped == 10
        other = make_hint("o", tenant="t2", tiers=["cpu"])
        book.open(other)
        assert len(book.tag(other.key, [f"o{index}" for index in range(40)])) == 20, (
            "the engine's 50 chunks"
        )
        assert book.leased_chunks == 50

    def test_stale_generation_and_conflict_are_refused(self, book):
        book.open(make_hint(generation=5, tiers=["cpu"]))
        stale = book.open(make_hint(generation=4, tiers=["cpu"]))
        assert (stale.state, stale.reason) == ("rejected", "stale_generation")
        conflict = book.open(make_hint(generation=5, tiers=["cpu"], hint_id="other"))
        assert (conflict.state, conflict.reason) == ("rejected", "generation_conflict")
        assert (
            book.release(make_hint(generation=4, op="release", tiers=["cpu"]))[0].reason
            == "stale_generation"
        )

    def test_the_same_hint_again_is_a_renewal(self, book, clock):
        hint = make_hint(tiers=["cpu"], ttl_seconds=120.0)
        book.open(hint)
        book.tag(hint.key, ["c0"])
        clock.now += 100.0
        assert book.open(hint).state == "accepted"
        clock.now += 100.0
        assert book.expire() == []
        assert book.rank("c0") == 1

    def test_release_is_idempotent_and_scoped(self, book):
        hint = make_hint(tiers=["cpu"])
        book.open(hint)
        book.tag(hint.key, ["c0"])
        intruder, moved = book.release(
            make_hint(tenant="other", op="release", tiers=["cpu"])
        )
        assert (intruder.state, intruder.reason, moved) == (
            "released",
            "no_live_lease",
            [],
        )
        assert book.rank("c0") == 1
        ack, moved = book.release(make_hint(op="release", tiers=["cpu"]))
        assert (ack.state, ack.reason, moved) == ("released", None, ["c0"])
        again, moved = book.release(make_hint(op="release", tiers=["cpu"]))
        assert (again.reason, moved) == ("no_live_lease", [])

    def test_an_evicted_leased_chunk_is_reported(self, book):
        hint = make_hint(tiers=["cpu"])
        book.open(hint)
        book.tag(hint.key, ["c0", "c1"])
        book.forget("c0", evicted=True)
        book.forget("c1", evicted=False)
        book.forget("not-leased", evicted=True)
        assert book.take_evictions() == {hint.hint_id: 1}
        assert book.take_evictions() == {}
        assert book.leased_chunks == 0

    def test_an_evicted_chunk_frees_budget(self, book):
        hint = make_hint(tiers=["cpu"])
        book.open(hint)
        book.tag(hint.key, [f"c{index}" for index in range(30)])
        book.forget("c0", evicted=True)
        assert book.tag(hint.key, ["again"]) == ["again"]

    def test_a_restarted_worker_refuses_a_hint_for_the_old_one(self, clock):
        restarted = lifecycle.LeaseBook(100, 0.5, 0.3, 120.0, "worker-b", clock=clock)
        hint = make_hint(tiers=["cpu"], worker_generation=GENERATION)
        ack = restarted.open(hint)
        assert (ack.state, ack.reason) == ("rejected", "stale_worker_generation")


def test_receipts_are_json_lines(tmp_path):
    path = tmp_path / "receipts.jsonl"
    log = lifecycle.ReceiptLog(path)
    log.write("ack", hint_id="h", state="accepted")
    log.write("end", hint_id="h", outcome="used")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["ack", "end"]
    assert rows[0]["hint_id"] == "h"
    assert isinstance(rows[0]["t"], float)


def test_no_receipt_path_writes_nothing():
    lifecycle.ReceiptLog(None).write("ack", hint_id="h")


REPORT = MODULE.with_name("report.py")
REPORT_SPEC = importlib.util.spec_from_file_location("kv_retention_report", REPORT)
assert REPORT_SPEC and REPORT_SPEC.loader
report = importlib.util.module_from_spec(REPORT_SPEC)
REPORT_SPEC.loader.exec_module(report)


def turn(session, index, start, end, isl, osl, cached, ttft, phase="profiling"):
    return {
        "metadata": {
            "x_correlation_id": session,
            "turn_index": index,
            "request_start_ns": int(start * 1e9),
            "request_end_ns": int(end * 1e9),
            "benchmark_phase": phase,
            "was_cancelled": False,
        },
        "metrics": {
            "time_to_first_token": {"value": ttft},
            "input_sequence_length": {"value": isl},
            "output_sequence_length": {"value": osl},
            "usage_prompt_cache_read_tokens": {"value": cached},
        },
    }


def by_source(gpu, cpu, computed, evicted=0, preemptions=0):
    prefix = 'vllm:prompt_tokens_by_source_total{engine="0",source="'
    return "\n".join(
        [
            "# HELP not a sample",
            f'{prefix}local_cache_hit"}} {gpu}',
            f'{prefix}external_kv_transfer"}} {cpu}',
            f'{prefix}local_compute"}} {computed}',
            f'vllm:num_preemptions_total{{engine="0"}} {preemptions}',
            f'vllm:kv_block_idle_before_evict_seconds_count{{engine="0"}} {evicted}',
            f'vllm:kv_block_idle_before_evict_seconds_bucket{{engine="0",le="1.0"}} {evicted / 2}',
            f'vllm:kv_block_idle_before_evict_seconds_bucket{{engine="0",le="10.0"}} {evicted}',
            f'vllm:kv_block_idle_before_evict_seconds_bucket{{engine="0",le="+Inf"}} {evicted}',
        ]
    )


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / "gpu10-4x" / "11"
    (run / "aiperf").mkdir(parents=True)
    records = [
        turn("a", 0, 0.0, 1.0, 1000, 10, 0, 50.0, phase="warmup"),
        turn(
            "a", 1, 10.0, 11.0, 1000, 10, 0, 500.0
        ),  # the first profiled turn: all cold
        turn("a", 2, 11.5, 12.5, 1100, 10, 1000, 60.0),  # 100 new, no pause
        turn(
            "a", 3, 20.0, 21.0, 1200, 10, 600, 300.0
        ),  # 100 new, 500 recomputed, a resume
        turn("b", 0, 0.0, 1.0, 400, 10, 0, 100.0),
    ]
    (run / "aiperf" / "profile_export.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records)
    )
    (run / "metrics_start.txt").write_text(by_source(0, 0, 0))
    (run / "metrics_end.txt").write_text(
        by_source(600, 300, 100, evicted=21, preemptions=2)
    )
    (run / "samples.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "vllm:kv_cache_usage_perc": usage,
                    "vllm:num_requests_running": 1,
                    "vllm:num_requests_waiting": waiting,
                }
            )
            for usage, waiting in ((0.2, 0), (0.4, 2))
        )
    )
    receipts = [
        {"event": "ack", "tier": "gpu", "state": "accepted", "reason": None},
        {
            "event": "ack",
            "tier": "gpu",
            "state": "rejected",
            "reason": "over_tenant_budget",
        },
        {
            "event": "end",
            "tier": "gpu",
            "outcome": "used",
            "tokens": 1000,
            "held_seconds": 2.1,
            "hit_tokens": 900,
        },
        {
            "event": "end",
            "tier": "gpu",
            "outcome": "expired",
            "tokens": 3000,
            "held_seconds": 1.0,
            "hit_tokens": 0,
        },
        {"event": "request", "gpu_hit_tokens": 900, "cpu_hit_tokens": 16},
        {
            "event": "cpu_tier",
            "chunks_evicted": 42,
            "leased_chunks_evicted": 7,
            "chunks_clipped": 3,
        },
    ]
    (run / "receipts.jsonl").write_text("\n".join(json.dumps(row) for row in receipts))
    (run / "run.json").write_text(
        json.dumps({"arm": "gpu10-4x", "seed": 11, "valid": True})
    )
    return run


class TestReport:
    def test_percentile_interpolates(self):
        assert report.percentile([1.0, 2.0, 3.0], 0.5) == 2.0
        assert report.percentile([1.0, 2.0], 0.5) == 1.5
        assert report.percentile([], 0.5) is None

    def test_client_side_keeps_profiling_turns_and_splits_computed_tokens(
        self, run_dir
    ):
        client = report.client_side(run_dir)
        assert client["requests"] == 4
        assert client["resumes"] == 1, "only the pause of at least a second is a resume"
        assert client["resume_p50"] == 300.0
        assert client["cold_tokens"] == 1000 + 400
        assert client["new_tokens"] == 100 + 100
        assert client["recomputed_tokens"] == 500
        assert client["seconds"] == 21.0

    def test_engine_side_reads_the_runs_change(self, run_dir):
        engine = report.engine_side(run_dir, 21.0)
        assert engine["gpu_share"] == 0.6
        assert engine["cpu_share"] == 0.3
        assert engine["computed_share"] == 0.1
        assert engine["preemptions"] == 2
        assert engine["gpu_evictions_per_s"] == pytest.approx(21 / 0.05 / 21.0)
        assert engine["idle_before_evict_p50"] == 1.0

    def test_receipts_count_what_was_asked_and_how_holds_ended(self, run_dir):
        receipts = report.receipts(run_dir, 10.0)
        assert (receipts["gpu_accepted"], receipts["gpu_refused"]) == (1, 1)
        assert (receipts["gpu_used"], receipts["gpu_expired"]) == (1, 1)
        assert receipts["gpu_used_share"] == 0.5
        assert receipts["gpu_used_token_share"] == 0.25
        assert receipts["held_pool_share"] == pytest.approx(
            (2100 + 3000) / 10 / report.POOL_TOKENS
        )
        assert receipts["cpu_evictions_per_s"] == 4.2
        assert receipts["leased_chunks_evicted"] == 7

    def test_a_stock_run_has_no_receipts(self, run_dir):
        (run_dir / "receipts.jsonl").unlink()
        assert report.receipts(run_dir, 10.0)["gpu_accepted"] == 0

    def test_a_cell_is_the_median_and_the_range(self):
        assert report.cell([1.0, 2.0, 9.0]) == "2 (1–9)"
        assert report.cell([0.5], digits=1, scale=100) == "50.0"
        assert report.cell([None]) == "—"
