"""Unit tests for the prefix-KV-hit benchmark client.

This file does not validate vLLM prefix-cache behavior and does not start a
router or worker. It tests `benches/backend/bench_prefix_kvhit.py` itself: SSE
parsing, error handling, completion validation, and TPOT timing boundaries. The
live prefix-cache check is the benchmark client script against a running router.
"""

import importlib.util
import json
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "benches" / "backend" / "bench_prefix_kvhit.py"
SPEC = importlib.util.spec_from_file_location("bench_prefix_kvhit", SCRIPT)
assert SPEC and SPEC.loader
BENCH_PREFIX_KVHIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH_PREFIX_KVHIT)
HELLO = [{"role": "user", "content": "hello"}]


@contextmanager
def sse_server(events):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for delay, event in events:
                time.sleep(delay)
                self.wfile.write(f"data: {event}\n\n".encode())
                self.wfile.flush()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def chunk(content=None, finish_reason=None):
    return json.dumps(
        {
            "choices": [
                {
                    "delta": {"content": content} if content is not None else {},
                    "finish_reason": finish_reason,
                }
            ]
        }
    )


class PrefixKvhitSseTests(unittest.TestCase):
    def test_tpot_stops_at_last_output_not_delayed_trailer(self):
        usage = json.dumps(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }
        )
        events = [
            (0, chunk("a")),
            (0, chunk("b")),
            (0.15, chunk(finish_reason="stop")),
            (0, usage),
            (0, "[DONE]"),
        ]
        with sse_server(events) as url:
            result = BENCH_PREFIX_KVHIT.post_chat(url, "model", HELLO, 2, 2)
        self.assertLess(result["tpot_ms"], 50)
        self.assertGreater(result["total_duration_ms"], 100)

    def test_top_level_sse_error_fails_measurement(self):
        with sse_server([(0, chunk("partial")), (0, '{"error":"worker failed"}')]) as url:
            with self.assertRaises(SystemExit):
                BENCH_PREFIX_KVHIT.post_chat(url, "model", HELLO, 2, 2)

    def test_missing_done_fails_measurement(self):
        events = [
            (0, chunk("partial")),
            (0, chunk(finish_reason="stop")),
            (
                0,
                json.dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 1,
                            "total_tokens": 4,
                        },
                    }
                ),
            ),
        ]
        with sse_server(events) as url:
            with self.assertRaises(SystemExit):
                BENCH_PREFIX_KVHIT.post_chat(url, "model", HELLO, 2, 2)

    def test_hit_wave_runs_requests_concurrently(self):
        usage = json.dumps(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            }
        )
        events = [
            (0.2, chunk("a")),
            (0, chunk(finish_reason="stop")),
            (0, usage),
            (0, "[DONE]"),
        ]
        with sse_server(events) as url:
            start = time.perf_counter()
            result = BENCH_PREFIX_KVHIT.post_hit_wave(url, "model", [HELLO] * 4, 1, 2, 4)
            elapsed = time.perf_counter() - start
        self.assertEqual(result["concurrency"], 4)
        self.assertEqual(result["requests"], 4)
        self.assertEqual(result["summary"]["count"], 4)
        self.assertLess(elapsed, 0.6)

    def test_zero_hit_wave_bodies_are_unique(self):
        bodies = BENCH_PREFIX_KVHIT.make_wave_bodies(0.0, "payload", 4)
        self.assertEqual(len(bodies), 4)
        self.assertEqual(len(set(bodies)), 4)
        self.assertTrue(all(body.endswith(" payload") for body in bodies))

    def test_turn_bodies_share_history_and_differ_in_the_last_turn(self):
        make = BENCH_PREFIX_KVHIT.make_turn_messages
        text = "abcdefghij" * 10
        first, second = make(0.7, text, 4, "A"), make(0.7, text, 4, "B")
        self.assertEqual(first[:-1], second[:-1])
        self.assertEqual([m["role"] for m in first], ["user", "assistant"] * 2 + ["user"])
        self.assertEqual("".join(m["content"] for m in first[:-1]), text[:70])
        self.assertEqual(first[-1]["content"], "A " + text[70:])
        self.assertNotEqual(first[-1], second[-1])
        self.assertEqual(make(0.99, text, 4, "A")[-1]["content"], "A")

    def test_zero_hit_turn_bodies_share_nothing(self):
        first, second = (
            BENCH_PREFIX_KVHIT.make_turn_messages(0.0, "payload", 8, tag) for tag in "AB"
        )
        self.assertEqual(first[0], {"role": "system", "content": "A"})
        self.assertNotEqual(first[0], second[0])

    def test_summary_reports_router_stage_fields(self):
        rows = [
            {
                "ttft_ms": 10.0,
                "stages": json.dumps(
                    {
                        "encode_ms": float(i),
                        "frontend_ms": 2.0 * i,
                        "encode_cached_tokens": i,
                        "n_prompt_tokens": 4,
                    }
                ),
            }
            for i in (1, 2, 3)
        ] + [{"ttft_ms": 12.0, "stages": None}]
        summary = BENCH_PREFIX_KVHIT.summarize_rows(rows)
        self.assertEqual(summary["ttft_ms"]["max"], 12.0)
        self.assertEqual(summary["encode_ms"], {"min": 1.0, "p50": 2.0, "avg": 2.0, "max": 3.0})
        self.assertEqual(summary["frontend_ms"]["max"], 6.0)
        self.assertEqual(summary["encode_reuse"]["avg"], 0.5)
        self.assertNotIn("template_ms", summary)


if __name__ == "__main__":
    unittest.main()
