"""TTFT / end-to-end latency for one already-running server.

The prompt is one 896x896 image plus 1600-1900 tokens of text. The image is
the first content part, so the shared text is a suffix. Prefix caching is
off on the server: replaying a request still prefills. Image bytes are a
pure function of the index (drawn here with PIL and sent as a data URL), so
this script commits no image files. A real photograph, when one is wanted,
is a URL the server already allowlists, for example
https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/stop_sign.jpg
Every label uses the same indexes, so stacks see the same requests at each
concurrency. Warmup indexes are disjoint from the measured ones. c=1, c=4,
c=8, and c=16 each get their own 24 images, so a later concurrency does not
hit the encoder cache of an earlier one. The server still admits 8 sequences,
so c=16 is 16 clients against that cap.

Usage: bench_latency.py --url URL --model PATH --label NAME
Writes JSON to $BENCH_OUT (default: stdout only).
"""
import argparse
import base64
import io
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from transformers import AutoTokenizer

SENTENCE = (
    "The road follows the coast past small harbors, salt grass, and low white "
    "houses, and the narrator keeps a careful record of the weather, the "
    "tides, and the names of the towns. "
)


def long_text(tokenizer):
    piece = SENTENCE
    ids = tokenizer.encode(piece, add_special_tokens=False)
    while len(ids) < 1600:
        piece += SENTENCE
        ids = tokenizer.encode(piece, add_special_tokens=False)
    if len(ids) > 1900:
        ids = ids[:1800]
        piece = tokenizer.decode(ids)
        ids = tokenizer.encode(piece, add_special_tokens=False)
    return piece, len(ids)


def png(index):
    color = ((index * 47) % 256, (index * 29) % 256, (index * 13) % 256)
    image = Image.new("RGB", (896, 896), color)
    # A second block keeps two indexes from collapsing to one flat color
    # that some decoders might treat as identical after quantization.
    for x in range(40):
        for y in range(40):
            image.putpixel((x, y), ((index * 13 + x) % 256, y, index % 256))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def payload(model, text, image_uri):
    return {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": text + " Describe the picture in one short sentence."},
            ],
        }],
        "max_tokens": 32,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def one(url, model, text, index):
    body = json.dumps(payload(model, text, png(index))).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    ttft = None
    answer = []
    prompt = None
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                usage = event.get("usage") or {}
                if usage.get("prompt_tokens"):
                    prompt = usage["prompt_tokens"]
                now = time.perf_counter()
                for choice in event.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content") or ""
                    if delta:
                        if ttft is None:
                            ttft = now - started
                        answer.append(delta)
    except urllib.error.HTTPError as error:
        return {"ok": False, "error": error.read().decode(errors="replace")[:240],
                "e2e_ms": (time.perf_counter() - started) * 1000}
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "error": repr(error)[:240],
                "e2e_ms": (time.perf_counter() - started) * 1000}
    return {
        "ok": True,
        "ttft_ms": None if ttft is None else ttft * 1000,
        "e2e_ms": (time.perf_counter() - started) * 1000,
        "prompt": prompt,
        "answer": "".join(answer)[:80],
    }


def summary(rows):
    good = [row for row in rows if row["ok"] and row["ttft_ms"] is not None]
    bad = [row for row in rows if not row["ok"] or row["ttft_ms"] is None]

    def pct(values, p):
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1))))
        return round(ordered[index], 1)

    ttft = [row["ttft_ms"] for row in good]
    e2e = [row["e2e_ms"] for row in good]
    prompts = [row["prompt"] for row in good if row["prompt"]]
    return {
        "ok": len(good),
        "failed": len(bad),
        "error": bad[0]["error"] if bad else None,
        "ttft_p50_ms": pct(ttft, 50),
        "ttft_p90_ms": pct(ttft, 90),
        "e2e_p50_ms": pct(e2e, 50),
        "e2e_p90_ms": pct(e2e, 90),
        "ttft_ms": [round(value, 1) for value in ttft],
        "e2e_ms": [round(value, 1) for value in e2e],
        "prompt_tokens": statistics.median(prompts) if prompts else None,
        "sample": good[0]["answer"] if good else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text, text_tokens = long_text(tokenizer)
    print(f"[{args.label}] text_tokens={text_tokens}", flush=True)

    # CUDA-graph capture happens on the first batches. Two serial requests,
    # then one concurrency-8 burst and one concurrency-16 burst. The engine
    # admits 8 sequences, so the second burst queues. Measured indexes are
    # fixed and disjoint across c=1, c=4, c=8, and c=16.
    measured = {
        "c1": list(range(100, 124)),
        "c4": list(range(200, 224)),
        "c8": list(range(300, 324)),
        "c16": list(range(400, 424)),
    }
    for index in (1, 2):
        warm = one(args.url, args.model, text, index)
        print(f"[{args.label}] warmup {index}: {warm.get('ttft_ms') and round(warm['ttft_ms'])}"
              f" ms ok={warm['ok']} prompt={warm.get('prompt')} {warm.get('error', '')}", flush=True)
    with ThreadPoolExecutor(8) as pool:
        graph = list(pool.map(lambda i: one(args.url, args.model, text, 20 + i), range(8)))
    print(f"[{args.label}] graph warmup c=8 {summary(graph)}", flush=True)
    with ThreadPoolExecutor(16) as pool:
        burst = list(pool.map(lambda i: one(args.url, args.model, text, 40 + i), range(16)))
    print(f"[{args.label}] warmup c=16 {summary(burst)}", flush=True)

    report = {
        "label": args.label,
        "text_tokens": text_tokens,
        "image_indexes": measured,
        "runs": {},
    }
    levels = (("c1", 1, measured["c1"]), ("c4", 4, measured["c4"]),
              ("c8", 8, measured["c8"]), ("c16", 16, measured["c16"]))
    for name, width, indexes in levels:
        started = time.perf_counter()
        if width == 1:
            rows = [one(args.url, args.model, text, index) for index in indexes]
        else:
            with ThreadPoolExecutor(width) as pool:
                rows = list(pool.map(lambda index: one(args.url, args.model, text, index), indexes))
        wall = time.perf_counter() - started
        report["runs"][name] = summary(rows)
        report["runs"][name]["wall_s"] = round(wall, 2)
        report["runs"][name]["req_per_s"] = round(report["runs"][name]["ok"] / wall, 2)
        print(f"[{args.label}] {name} {report['runs'][name]}", flush=True)

    path = os.environ.get("BENCH_OUT")
    if path:
        with open(path, "w") as handle:
            json.dump(report, handle, indent=1)
        print("wrote", path, flush=True)
    failed = sum(report["runs"][name]["failed"] for name in ("c1", "c4", "c8", "c16"))
    short = report["runs"]["c1"]["prompt_tokens"]
    if short is not None and short < 1000:
        print(f"[{args.label}] prompt is under 1000 tokens", flush=True)
        failed += 1
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
