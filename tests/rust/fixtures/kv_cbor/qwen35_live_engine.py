import json
import os
import socket
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
    probe_dir = Path(os.environ["HYBRID_PROBE_DIR"])
    ports = []
    while len(ports) < 2:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            if port not in ports:
                ports.append(port)
    pub, replay = ports
    llm = LLM(
        model="Qwen/Qwen3.5-4B",
        max_model_len=2048,
        max_num_seqs=1,
        gpu_memory_utilization=0.55,
        enable_prefix_caching=True,
        prefix_caching_hash_algo="sha256_cbor",
        enforce_eager=True,
        kv_events_config={
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "endpoint": f"tcp://*:{pub}",
            "replay_endpoint": f"tcp://*:{replay}",
            "topic": "hybrid-probe-live",
        },
    )
    (probe_dir / "endpoints.json").write_text(json.dumps({"pub": pub, "replay": replay}))
    (probe_dir / "ready").write_text("ready")
    deadline = time.monotonic() + 180
    while not (probe_dir / "go").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("subscriber did not start")
        time.sleep(0.05)
    prompt = {"prompt_token_ids": [100 + i % 100 for i in range(1024)]}
    results = []
    for _ in range(2):
        out = llm.generate(
            [prompt], SamplingParams(temperature=0, max_tokens=4), use_tqdm=False
        )[0]
        results.append(out.num_cached_tokens)
    print("CACHED_TOKENS " + json.dumps(results), flush=True)
    if results != [0, 528]:
        raise AssertionError(f"unexpected engine cache behavior: {results}")
    deadline = time.monotonic() + 180
    while not (probe_dir / "done").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("router probe did not finish")
        time.sleep(0.05)


if __name__ == "__main__":
    main()
