#!/usr/bin/env python3
"""Chat accuracy check for gRPC prefill/decode.

The gRPC pool serves ``/v1/chat/completions``. ``/v1/completions`` returns 400,
so this is the chat counterpart of ``test_pd_accuracy.py``.
"""

import argparse
import json
import sys

import requests


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def print_error(msg: str):
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.RESET}")


SAMPLE_PROMPTS = [
    "Red Hat is the best company in the world to work for because",
    "The quick brown fox jumps over the lazy dog and",
    "To be or not to be, that is the question:",
    "It was the best of times, it was the worst of times,",
    "Call me Ishmael. Some years ago, never mind how long precisely,",
]


def _generated_text(message: dict) -> str:
    parts = []
    for key in ("content", "reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts)


def _request_body(model: str, prompt: str, max_tokens: int, stream: bool) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }
    # Qwen3 spends a short budget on the thinking trace. Ask for the answer.
    if "qwen3" in model.lower():
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def test_router_health(router_url: str) -> bool:
    try:
        response = requests.get(f"{router_url}/health", timeout=10)
    except requests.RequestException as exc:
        print_error(f"Router health check failed: {exc}")
        return False
    if response.status_code == 200:
        print_success("Router health check passed")
        return True
    print_error(f"Router health returned HTTP {response.status_code}")
    return False


def test_chat_accuracy(
    router_url: str,
    model: str,
    num_requests: int,
    max_tokens: int,
) -> bool:
    print_info(f"Testing chat completions with {num_requests} requests")
    failures = []
    with requests.Session() as session:
        for i in range(num_requests):
            prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
            try:
                response = session.post(
                    f"{router_url}/v1/chat/completions",
                    json=_request_body(model, prompt, max_tokens, stream=False),
                    timeout=120,
                )
            except requests.RequestException as exc:
                failures.append(f"Request {i}: {exc}")
                print_error(f"Request {i} failed: {exc}")
                continue
            if response.status_code != 200:
                failures.append(
                    f"Request {i}: HTTP {response.status_code} {response.text[:300]}"
                )
                print_error(f"Request {i} HTTP {response.status_code}")
                continue
            try:
                data = response.json()
            except json.JSONDecodeError as exc:
                failures.append(f"Request {i}: invalid JSON {exc}")
                continue
            choices = data.get("choices") or []
            if not choices or "message" not in choices[0]:
                failures.append(f"Request {i}: no message")
                print_error(f"Request {i}: no message")
                continue
            finish = choices[0].get("finish_reason")
            if finish not in ("stop", "length"):
                failures.append(f"Request {i}: finish_reason={finish}")
                print_error(f"Request {i}: finish_reason={finish}")
                continue
            text = _generated_text(choices[0]["message"])
            prompt_tokens = (data.get("usage") or {}).get("prompt_tokens") or 0
            if not text:
                failures.append(f"Request {i}: empty generation")
                print_error(f"Request {i}: empty generation")
                continue
            if prompt_tokens < 1:
                failures.append(f"Request {i}: prompt_tokens={prompt_tokens}")
                print_error(f"Request {i}: router reported no prompt tokens")
                continue
            print_success(
                f"Request {i}: {len(text)} chars, prompt_tokens={prompt_tokens}"
            )
    return _report("Chat", num_requests, failures)


def test_chat_streaming(
    router_url: str,
    model: str,
    num_requests: int,
    max_tokens: int,
) -> bool:
    print_info(f"Testing streaming chat with {num_requests} requests")
    failures = []
    with requests.Session() as session:
        for i in range(num_requests):
            prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
            try:
                response = session.post(
                    f"{router_url}/v1/chat/completions",
                    json=_request_body(model, prompt, max_tokens, stream=True),
                    timeout=120,
                    stream=True,
                )
            except requests.RequestException as exc:
                failures.append(f"Stream {i}: {exc}")
                print_error(f"Stream {i} failed: {exc}")
                continue
            if response.status_code != 200:
                failures.append(f"Stream {i}: HTTP {response.status_code}")
                print_error(f"Stream {i} HTTP {response.status_code}")
                continue
            failure = _stream_failure(response.iter_lines())
            if failure:
                failures.append(f"Stream {i}: {failure}")
                print_error(f"Stream {i}: {failure}")
                continue
            print_success(f"Stream {i} finished")
    return _report("Streaming chat", num_requests, failures)


def _stream_failure(lines) -> str | None:
    """None when the stream ends on stop or length after some text.

    An SSE error, ``finish_reason: abort``, or EOF before a normal finish
    fails the request even if earlier frames already contained text.
    """
    parts = []
    finished = False
    for line in lines:
        if not line:
            continue
        line_str = line.decode("utf-8")
        if not line_str.startswith("data: "):
            continue
        data_str = line_str[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError as exc:
            return f"invalid SSE JSON: {exc}"
        if not isinstance(chunk, dict):
            return "SSE payload is not an object"
        if "error" in chunk:
            return f"SSE error: {chunk['error']}"
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        finish = choice.get("finish_reason")
        if finish == "abort":
            return "finish_reason abort"
        if finish not in (None, "stop", "length"):
            return f"finish_reason={finish}"
        if finish in ("stop", "length"):
            finished = True
        delta = choice.get("delta") or {}
        parts.append(_generated_text(delta))
    if not finished:
        return "stream ended without a stop or length finish"
    if not "".join(parts).strip():
        return "empty generation"
    return None


def _report(name: str, total: int, failures: list[str]) -> bool:
    print_info(f"{name}: {total - len(failures)}/{total} passed")
    for failure in failures:
        print_error(f"  - {failure}")
    return not failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--num-streaming-requests", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--skip-streaming", action="store_true")
    args = parser.parse_args()

    print_info(f"Router: {args.router_url}")
    print_info(f"Model:  {args.model}")
    if not test_router_health(args.router_url):
        return 1
    chat_ok = test_chat_accuracy(
        args.router_url, args.model, args.num_requests, args.max_tokens
    )
    stream_ok = True
    if not args.skip_streaming:
        stream_ok = test_chat_streaming(
            args.router_url,
            args.model,
            args.num_streaming_requests,
            args.max_tokens,
        )
    if chat_ok and stream_ok:
        print_success("gRPC P/D chat accuracy passed")
        return 0
    print_error("gRPC P/D chat accuracy failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
