"""Standard-library tests; no Cargo build, server, model, or GPU is started."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("pd_pressure_runner", Path(__file__).parents[1] / "run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class CommandsAndResults(unittest.TestCase):
    def test_pr1_cargo_command_has_features_filter_and_artifact_metadata(self):
        command = runner.cargo_command("/tools/cargo", 1)
        self.assertEqual(command[0], "/tools/cargo")
        self.assertEqual(command[command.index("--features") + 1], "cpu,ready")
        self.assertIn("routers::http::pd_router::admission_tests::pd_cache_aware_pr1_stress", command)
        self.assertIn("--ignored", command)
        self.assertIn("--exact", command)
        self.assertIn("--message-format=json", command)
        self.assertIn("--locked", command)
        http = runner.cargo_command("/tools/cargo", 1, 1)
        self.assertEqual(http[http.index("--test") + 1], "test_pd_routing")
        self.assertIn("test_pd_routing::http_pd_cache_aware_pr1_stress", http)
        self.assertIn("--exact", http)
        self.assertIn("--ignored", http)
        integration = runner.cargo_command("/tools/cargo", 1, 2)
        self.assertEqual(integration[integration.index("--test") + 1], "grpc_pd_routing")
        self.assertIn("grpc_pd_cache_aware_pr1_stress", integration)
        self.assertNotIn("--release", runner.cargo_command("cargo", 1, profile="test"))

    def test_future_stages_are_not_implemented(self):
        for stage in (2, 3, 4):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "not implemented"):
                runner.cargo_command("cargo", stage)

    def test_serve_targets_public_chat_endpoint_with_identical_workload(self):
        args = SimpleNamespace(vllm="/tools with spaces/vllm", model="small/model", input_len=16,
                               output_len=4, num_prompts=2, request_rate=1.5)
        first = runner.serve_command(args, "http://localhost:30001", Path("results/base"), 23)
        second = runner.serve_command(args, "http://localhost:30002", Path("results/head"), 23)
        self.assertEqual(first[0], "/tools with spaces/vllm")
        self.assertEqual(first[first.index("--backend") + 1], "openai-chat")
        self.assertEqual(first[first.index("--endpoint") + 1], "/v1/chat/completions")
        for flag in ("--model", "--seed", "--num-prompts", "--request-rate", "--random-input-len"):
            self.assertEqual(first[first.index(flag) + 1], second[second.index(flag) + 1])
        self.assertNotIn("grpc://", " ".join(first))
        self.assertIn("--save-detailed", first)
        self.assertEqual(first[first.index("--metric-percentiles") + 1], "50,95,99")

    def test_cargo_zero_tests_never_pass(self):
        fixtures = {
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out;": "passed",
            "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out;": "zero_tests",
            "test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 12 filtered out;": "zero_tests",
            "Finished release profile": "zero_tests",
            "test result: FAILED. 1 passed; 1 failed; 0 ignored;": "failed",
        }
        for text, expected in fixtures.items():
            with self.subTest(text=text):
                self.assertEqual(runner.cargo_verdict(0, text), expected)
        self.assertEqual(runner.cargo_verdict(101, next(iter(fixtures))), "failed")

    def test_artifact_extraction_ignores_diagnostics_and_non_test_binaries(self):
        records = [
            {"reason": "compiler-artifact", "profile": {"test": True}, "executable": "/tmp/test"},
            {"reason": "compiler-artifact", "profile": {"test": False}, "executable": "/tmp/app"},
            {"reason": "build-finished", "success": True},
        ]
        raw = "\n".join(json.dumps(record) for record in records) + "\nrunning 1 test\n"
        self.assertEqual(runner.test_artifacts(raw), ["/tmp/test"])

    def test_stress_json_checks_completion_and_sentinel_invariants(self):
        common = {"rounds": 2, "concurrency": 3, "wall_seconds": 0.1}
        admission = {**common, "kind": "pair_admission", "attempts": 6,
                     "final_load_entries": 0, "sentinel_preserved": True}
        grpc = {**common, "kind": "grpc_pd", "requests": 6, "completed": 6,
                "handoffs_verified": 6, "final_occupancy": 0}
        http = {**grpc, "kind": "http_pd"}
        for suite, record in (("admission", admission), ("http", http), ("grpc", grpc)):
            raw = "test sample ... PD_PRESSURE_STRESS_RESULT " + json.dumps(record)
            self.assertEqual(runner.stress_result(raw, suite), record)
        for bad in ({**grpc, "completed": 5}, {**grpc, "final_occupancy": 1},
                    {**grpc, "rounds": "bad"}, {**grpc, "wall_seconds": None}):
            with self.assertRaises(ValueError):
                runner.stress_result("PD_PRESSURE_STRESS_RESULT " + json.dumps(bad), "grpc")
        with self.assertRaises(ValueError):
            runner.stress_result("test result: ok. 1 passed; 0 failed;", "grpc")
        for suite, wrong_kind in (("http", grpc), ("grpc", http)):
            with self.subTest(suite=suite), self.assertRaises(ValueError):
                runner.stress_result("PD_PRESSURE_STRESS_RESULT " + json.dumps(wrong_kind), suite)

    def test_stress_count_must_equal_full_requested_workload(self):
        common = {"rounds": 2, "concurrency": 3, "wall_seconds": 0.1}
        for count in (1, 5, 7):
            records = {
                "admission": {**common, "kind": "pair_admission", "attempts": count,
                              "final_load_entries": 0, "sentinel_preserved": True},
                "http": {**common, "kind": "http_pd", "requests": count, "completed": count,
                         "handoffs_verified": count, "final_occupancy": 0},
                "grpc": {**common, "kind": "grpc_pd", "requests": count, "completed": count,
                         "handoffs_verified": count, "final_occupancy": 0},
            }
            for suite, record in records.items():
                with self.subTest(count=count, suite=suite), self.assertRaisesRegex(ValueError, "count"):
                    runner.stress_result("PD_PRESSURE_STRESS_RESULT " + json.dumps(record), suite)

    def test_invalid_urls_and_parameters_are_rejected(self):
        for value in ("grpc://host:5000", "http://user:pass@host", "http://host?api_key=x", "http://host:bad"):
            with self.subTest(value=value), self.assertRaises(Exception):
                runner.base_url(value)
        self.assertEqual(runner.base_url("http://localhost:30000/"), "http://localhost:30000")
        for option, value in (("--repeat", "0"), ("--timeout", "nan"), ("--request-rate", "inf"),
                              ("--input-len", "-1"), ("--stage", "5")):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.parser().parse_args(["stress", "--base-ref", "HEAD", "--out-dir", "unused", option, value])
                self.assertEqual(error.exception.code, 2)


class RecordingAndCapabilities(unittest.TestCase):
    def test_linux_machine_fingerprint_uses_observed_values(self):
        with mock.patch.object(runner.sys, "platform", "linux"), mock.patch.object(
                runner.Path, "read_text", return_value="processor: 0\nmodel name: Fixture CPU\n"), \
                mock.patch.object(runner.os, "sched_getaffinity", return_value={0, 1, 2, 3}, create=True), \
                mock.patch.object(runner.os, "sysconf", side_effect=[1024, 4096], create=True):
            self.assertEqual(runner.machine_fingerprint(), {
                "cpu_model": "Fixture CPU", "visible_logical_cpus": 4, "memory_bytes": 4194304})

    def test_windows_unavailable_machine_fields_are_null(self):
        with mock.patch.object(runner.sys, "platform", "win32"), \
                mock.patch.object(runner.os, "cpu_count", return_value=8):
            self.assertEqual(runner.machine_fingerprint(), {
                "cpu_model": None, "visible_logical_cpus": 8, "memory_bytes": None})

    def test_unavailable_linux_machine_fields_are_null(self):
        with mock.patch.object(runner.sys, "platform", "linux"), \
                mock.patch.object(runner.os, "cpu_count", return_value=None), \
                mock.patch.object(runner.Path, "read_text", side_effect=OSError), \
                mock.patch.object(runner.os, "sched_getaffinity", side_effect=OSError, create=True), \
                mock.patch.object(runner.os, "sysconf", side_effect=ValueError, create=True):
            self.assertEqual(runner.machine_fingerprint(), {
                "cpu_model": None, "visible_logical_cpus": None, "memory_bytes": None})

    def test_default_repeat_records_admission_http_and_grpc_each_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = {"rounds": 2, "concurrency": 3, "wall_seconds": 0.01}
            pd_result = {**common, "requests": 6, "completed": 6,
                         "handoffs_verified": 6, "final_occupancy": 0}
            fixtures = [
                {**common, "kind": "pair_admission", "attempts": 6,
                 "final_load_entries": 0, "sentinel_preserved": True},
                {**pd_result, "kind": "http_pd"}, {**pd_result, "kind": "grpc_pd"},
            ]
            artifact = json.dumps({"reason": "compiler-artifact", "profile": {"test": True},
                                   "executable": sys.executable})
            outputs = [(0, artifact + "\nPD_PRESSURE_STRESS_RESULT " + json.dumps(result)
                        + "\ntest result: ok. 1 passed; 0 failed; 0 ignored;")
                       for result in fixtures * 2]
            args = ["stress", "--repo", tmp, "--out-dir", str(root / "out"), "--base-ref", "HEAD",
                    "--cargo", sys.executable, "--profile", "test", "--stress-rounds", "2",
                    "--stress-concurrency", "3"]
            with mock.patch.object(runner, "provenance"), mock.patch.object(
                    runner.Recorder, "run", side_effect=outputs) as execute:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(args), 0)
            self.assertEqual(execute.call_count, 6)
            saved = json.loads((root / "out/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([result["suite"] for result in saved["stress_results"]],
                             ["admission", "http", "grpc"] * 2)
            self.assertEqual(len(list((root / "out").glob("stress-*.result.json"))), 6)
            binary = Path(sys.executable)
            self.assertEqual(saved["test_binaries"][sys.executable], runner.sha256(binary))
            self.assertEqual(saved["test_binary_metadata"][sys.executable], {
                "size_bytes": binary.stat().st_size, "mtime_ns": binary.stat().st_mtime_ns})
            self.assertIn("machine", saved)

    def test_provenance_failure_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(base_ref="HEAD", repo=Path(tmp))
            record = runner.Recorder(Path(tmp) / "evidence", Path(tmp), 5)
            with mock.patch.object(record, "run", side_effect=[(0, "base"), (0, "head"), (1, "failure")]) as execute:
                with self.assertRaisesRegex(ValueError, "git-status"):
                    runner.provenance(record, args)
            self.assertTrue(all(call.args[1][:3] == ["git", "-c", "core.autocrlf=true"]
                                for call in execute.call_args_list))

    def test_untracked_source_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "tests/new_test.rs"
            source.parent.mkdir()
            source.write_bytes(b"// new regression\r\n")
            args = SimpleNamespace(base_ref="HEAD", repo=root, action="stress", cargo=sys.executable)
            record = runner.Recorder(root / "evidence", root, 5)
            responses = [(0, "base"), (0, "head"), (0, "?? tests/new_test.rs"), (0, "patch"),
                         (0, "tests/new_test.rs\0"), (0, "tool version"), (0, "rust version")]
            with mock.patch.object(record, "run", side_effect=responses):
                runner.provenance(record, args)
            snapshot = root / "evidence/source-snapshot/tests/new_test.rs"
            self.assertEqual(snapshot.read_bytes(), source.read_bytes())
            self.assertEqual(record.data["untracked_source_sha256"]["tests/new_test.rs"], runner.sha256(source))

    def test_grouped_help_is_expanded_without_benchmark_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = runner.Recorder(Path(tmp) / "evidence", Path(tmp), 5)
            with mock.patch.object(record, "run", side_effect=[
                    (0, "usage: vllm bench serve [options]\nFor full list: --help=all"),
                    (0, "usage: vllm bench serve --model --metric-percentiles")]) as execute:
                self.assertIn("--model", runner.probe(record, "vllm", "serve"))
            self.assertEqual(execute.call_args.args[1], ["vllm", "bench", "serve", "--help=all"])

    def test_small_process_preserves_argv_raw_output_and_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = runner.Recorder(root / "evidence", root, 5)
            argv = [sys.executable, "-B", "-c", "print('two sample requests')"]
            code, raw = record.run("tiny", argv)
            with contextlib.redirect_stdout(io.StringIO()):
                record.finish("passed", code)
            saved = json.loads((root / "evidence/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertIn("two sample requests", raw)
            self.assertEqual(saved["commands"][0]["argv"], argv)
            self.assertGreaterEqual(saved["commands"][0]["elapsed_seconds"], 0)
            self.assertIn("ended_at", saved["commands"][0])
            self.assertEqual(runner.sha256(root / "evidence/manifest.json"),
                             hashlib.sha256((root / "evidence/manifest.json").read_bytes()).hexdigest())

    def test_existing_evidence_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                runner.Recorder(Path(tmp), Path(tmp), 5)

    def test_unsupported_forward_does_not_execute_gpu_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ["forward", "--repo", tmp, "--out-dir", str(Path(tmp) / "out"),
                    "--base-ref", "HEAD", "--vllm", sys.executable, "--model", "tiny"]
            with mock.patch.object(runner, "provenance"), mock.patch.object(
                    runner.Recorder, "run", side_effect=[(0, "usage: vllm bench {serve}"),
                                                         (2, "invalid choice: forward")]) as execute:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(args), 3)
            self.assertEqual(execute.call_count, 2)
            self.assertTrue(all("--help" in call.args[1] for call in execute.call_args_list))
            saved = json.loads((Path(tmp) / "out/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "unsupported")

    def test_unimplemented_stage_records_failure_without_invoking_cargo(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner.Recorder, "run") as execute:
            with contextlib.redirect_stdout(io.StringIO()):
                result = runner.main(["stress", "--stage", "2", "--repo", tmp,
                                      "--out-dir", str(Path(tmp) / "out"), "--base-ref", "HEAD"])
            self.assertEqual(result, 3)
            execute.assert_not_called()
            saved = json.loads((Path(tmp) / "out/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "not_implemented")


if __name__ == "__main__":
    unittest.main()
