"""CPU checks for replay input preparation before GPU startup."""

import hashlib
import io
from urllib.error import HTTPError

import pytest

from tests.python.e2e.perf.router_policy import case_loader

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


def test_agentx_trace_is_pinned_and_ignores_tracelab_override(tmp_path, monkeypatch):
    trace = tmp_path / "agentx.jsonl"
    trace.write_bytes(b"pinned AgentX data\n")
    expected = hashlib.sha256(trace.read_bytes()).hexdigest()
    monkeypatch.setenv("TRACELAB_JSONL_URL", "https://example.test/tracelab.jsonl")

    def unexpected_download(*args, **kwargs):
        pytest.fail("AgentX must not use the TraceLab URL override")

    monkeypatch.setattr(case_loader, "urlopen", unexpected_download)
    assert (
        case_loader.prepare_trace(
            trace,
            out_dir=tmp_path / "run",
            trace_type="agentX",
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
