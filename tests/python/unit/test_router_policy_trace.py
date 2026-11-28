"""CPU checks for replay input preparation before GPU startup."""

import hashlib
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from tests.python.e2e.perf.router_policy import case_loader
from tests.python.e2e.perf.router_policy.data import extract_agentx_ci

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


def test_default_trace_uses_the_case_path(tmp_path, monkeypatch):
    monkeypatch.delenv("TRACELAB_JSONL_URL", raising=False)
    trace = tmp_path / "case-trace.jsonl"
    trace.write_text('{"session": "default"}\n', encoding="utf-8")
    assert case_loader.prepare_trace(trace, out_dir=tmp_path / "run") == trace


def test_missing_default_trace_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("TRACELAB_JSONL_URL", raising=False)
    with pytest.raises(FileNotFoundError, match="TraceLab replay file"):
        case_loader.prepare_trace(tmp_path / "missing.jsonl", out_dir=tmp_path / "run")


def test_trace_override_is_saved_with_the_cell(tmp_path, monkeypatch):
    trace = tmp_path / "case-trace.jsonl"
    trace.write_bytes(b"original trace\n")
    monkeypatch.setenv("TRACELAB_JSONL_URL", "https://example.test/trace.jsonl")
    monkeypatch.setattr(
        case_loader,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(b"downloaded trace\n"),
    )
    result = case_loader.prepare_trace(trace, out_dir=tmp_path / "run")
    assert result == tmp_path / "run" / "trace.jsonl"
    assert result.read_bytes() == b"downloaded trace\n"
    assert trace.read_bytes() == b"original trace\n"


def test_failed_download_preserves_the_previous_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACELAB_JSONL_URL", "https://example.test/trace.jsonl")
    out = tmp_path / "run"
    out.mkdir()
    previous = out / "trace.jsonl"
    previous.write_bytes(b"previous trace\n")

    class InterruptedDownload(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError("download interrupted")
            return super().read(3)

    monkeypatch.setattr(
        case_loader, "urlopen", lambda *args, **kwargs: InterruptedDownload(b"partial")
    )
    with pytest.raises(OSError, match="download interrupted"):
        case_loader.prepare_trace(tmp_path / "default.jsonl", out_dir=out)
    assert previous.read_bytes() == b"previous trace\n"
    assert list(out.iterdir()) == [previous]


def test_http_error_fails_trace_preparation(tmp_path, monkeypatch):
    url = "https://example.test/missing.jsonl"
    monkeypatch.setenv("TRACELAB_JSONL_URL", url)

    def missing(*args, **kwargs):
        raise HTTPError(url, 404, "Not found", None, None)

    monkeypatch.setattr(case_loader, "urlopen", missing)
    out = tmp_path / "run"
    with pytest.raises(HTTPError, match="404"):
        case_loader.prepare_trace(tmp_path / "default.jsonl", out_dir=out)
    assert not list(out.iterdir())


@pytest.mark.parametrize("trace_type", ["agentX", "tracelab"])
def test_pinned_trace_ignores_tracelab_override(tmp_path, monkeypatch, trace_type):
    trace = tmp_path / "pinned.jsonl"
    trace.write_bytes(b"pinned replay data\n")
    expected = hashlib.sha256(trace.read_bytes()).hexdigest()
    monkeypatch.setenv("TRACELAB_JSONL_URL", "https://example.test/tracelab.jsonl")

    def unexpected_download(*args, **kwargs):
        pytest.fail("pinned traces must not use the TraceLab URL override")

    monkeypatch.setattr(case_loader, "urlopen", unexpected_download)
    assert (
        case_loader.prepare_trace(
            trace,
            out_dir=tmp_path / "run",
            trace_type=trace_type,
            expected_sha256=expected,
        )
        == trace
    )


@pytest.mark.parametrize("expected", ["", "0" * 64], ids=["missing_hash", "wrong_hash"])
def test_agentx_trace_requires_a_matching_hash(tmp_path, expected):
    trace = tmp_path / "agentx.jsonl"
    trace.write_bytes(b"changed AgentX data\n")
    with pytest.raises(ValueError, match="trace_sha256|SHA256 mismatch"):
        case_loader.prepare_trace(
            trace,
            out_dir=tmp_path / "run",
            trace_type="agentX",
            expected_sha256=expected,
        )
    assert not (tmp_path / "run").exists()


def test_pinned_tracelab_rejects_changed_data(tmp_path, monkeypatch):
    monkeypatch.delenv("TRACELAB_JSONL_URL", raising=False)
    trace = tmp_path / "pinned.jsonl"
    trace.write_bytes(b"changed replay data\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        case_loader.prepare_trace(
            trace,
            out_dir=tmp_path / "run",
            expected_sha256="0" * 64,
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("trace_type", ["agentX", "tracelab"])
@pytest.mark.parametrize("tampered", [False, True])
def test_pinned_trace_accepts_crlf_but_rejects_content_changes(
    tmp_path, trace_type, tampered, monkeypatch
):
    monkeypatch.delenv("TRACELAB_JSONL_URL", raising=False)
    original = b'{"session": "first"}\n{"session": "second"}\n'
    expected = hashlib.sha256(original).hexdigest()
    body = original.replace(b"\n", b"\r\n")
    if tampered:
        body = body.replace(b"first", b"changed")
    trace = tmp_path / "trace.jsonl"
    trace.write_bytes(body)
    if tampered:
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            case_loader.prepare_trace(
                trace,
                out_dir=tmp_path / "run",
                trace_type=trace_type,
                expected_sha256=expected,
            )
    else:
        assert (
            case_loader.prepare_trace(
                trace,
                out_dir=tmp_path / "run",
                trace_type=trace_type,
                expected_sha256=expected,
            )
            == trace
        )
    assert trace.read_bytes() == body


def test_pd_agentx_fits_all_pd_cases():
    suite = Path(__file__).parents[1] / "e2e/perf/router_policy"
    trace = suite / "data" / extract_agentx_ci.TRACE_FILE
    body = trace.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(body).hexdigest() == extract_agentx_ci.TRACE_SHA256
    sessions = [json.loads(line) for line in body.splitlines()]
    assert len(sessions) == len({session["id"] for session in sessions}) == 12
    requests = []
    for session in sessions:
        assert session["hash_id_scope"] == "local"
        assert len(session["requests"]) == 12
        for request in session["requests"]:
            assert request.get("type") != "subagent"
            assert request["in"] > 0 and request["out"] > 1
            assert (
                len(request["hash_ids"])
                == (request["in"] + session["block_size"] - 1) // session["block_size"]
            )
            requests.append(request)
    assert len(requests) == 144
    longest = max(request["in"] + request["out"] for request in requests)
    assert longest == 65477
    for policy in ("cache_aware", "kv_aware", "round_robin"):
        case = case_loader.load_case(suite / "cases" / f"pd_disagg_{policy}.json")
        assert suite.parents[4] / case["runtime"]["trace_path"] == trace
        assert case["runtime"]["trace_type"] == "agentX"
        assert case["runtime"]["trace_sha256"] == extract_agentx_ci.TRACE_SHA256
        assert case["runtime"]["endpoint"] == "/v1/completions"
        assert case["benchmark_params"]["task-num"] == 12
        assert case["benchmark_params"]["max-concurrency"] == [12]
        assert case["benchmark_params"]["max_inflight"] == 1
        assert case["workers"]["count"] == 4
        assert case["workers"]["prefill_count"] == 2
        worker = case["workers"]["per_worker"]
        assert longest <= worker["max_model_len"] == 65536
        assert worker["kv_cache_memory_bytes"] == 12 * 1024**3
