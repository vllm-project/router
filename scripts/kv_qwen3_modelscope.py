#!/usr/bin/env python3
"""Download and verify one fixed public ModelScope snapshot using only stdlib.

This records ModelScope provenance. It does not assert equivalence to any
Hugging Face commit. All model requests originate from modelscope.cn.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


MODEL = "Qwen/Qwen3-0.6B"
REVISION = "09b42cad3d112e832108974449ccb5e8e0f5b5d1"
API_BASE = "https://modelscope.cn/api/v1/models/" + MODEL
EXPECTED_TOKENIZERS = {
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
}
REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "LICENSE",
    "model.safetensors",
)
RETRIES = 3
CHUNK_BYTES = 8 * 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def check_source_url(url):
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    allowed = any(
        host == suffix or host.endswith("." + suffix)
        for suffix in ("modelscope.cn", "modelscope.ai", "aliyuncs.com")
    )
    if parsed.scheme != "https" or not allowed:
        raise ValueError("refusing redirect outside HTTPS ModelScope/Alibaba CDN: " + host)


class ModelScopeRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        check_source_url(new_url)
        return super().redirect_request(request, response, code, message, headers, new_url)


OPENER = build_opener(ModelScopeRedirects())


def open_url(url):
    check_source_url(url)
    request = Request(url, headers={"User-Agent": "cmb-upstream-modelscope-verifier/1"})
    return OPENER.open(request, timeout=60)


def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowed-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    allowed_root = args.allowed_root.resolve(strict=True)
    output = args.output.resolve()
    if not allowed_root.is_dir() or output == allowed_root or not output.is_relative_to(allowed_root):
        parser.error("--output must be a fresh child directory inside --allowed-root")
    if output.exists() or args.output.is_symlink():
        parser.error("--output must not already exist")
    output.mkdir(parents=True, exist_ok=False)

    manifest_path = output / "download-manifest.json"
    log_path = output / "download-events.jsonl"
    listing_url = API_BASE + "/repo/files?" + urlencode({"Revision": REVISION, "Recursive": "true"})
    manifest = {
        "status": "started",
        "started_at": now(),
        "finished_at": None,
        "model": MODEL,
        "source": "ModelScope",
        "source_commit": REVISION,
        "source_listing_url": listing_url,
        "provenance_claim": "Fixed ModelScope commit; no unverified Hugging Face weight equivalence claim",
        "python_version": sys.version,
        "downloader": "urllib stdlib; maximum three attempts per file",
        "allowed_root": str(allowed_root),
        "output": str(output),
        "required_files": list(REQUIRED_FILES),
        "expected_tokenizer_sha256": EXPECTED_TOKENIZERS,
        "files": [],
    }
    write_json(manifest_path, manifest)

    def log(status, **fields):
        record = {"time": now(), "status": status, **fields}
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def download_file(name, metadata):
        expected_hash = metadata["Sha256"].lower()
        expected_size = metadata["Size"]
        source_url = API_BASE + "/repo?" + urlencode({"Revision": REVISION, "FilePath": name})
        target = output / name
        partial = output / (name + ".part")
        for attempt in range(1, RETRIES + 1):
            started_at = now()
            log("download-started", file=name, attempt=attempt, expected_size=expected_size)
            try:
                digest = hashlib.sha256()
                size = 0
                next_progress = 128 * 1024 * 1024
                with open_url(source_url) as response, partial.open("xb") as destination:
                    if response.status != 200:
                        raise ValueError("unexpected HTTP status " + str(response.status))
                    final_host = urlparse(response.geturl()).hostname
                    while chunk := response.read(CHUNK_BYTES):
                        size += len(chunk)
                        if size > expected_size:
                            raise ValueError("download exceeds API file size")
                        digest.update(chunk)
                        destination.write(chunk)
                        if size >= next_progress:
                            log("download-progress", file=name, bytes=size, expected_size=expected_size)
                            next_progress += 128 * 1024 * 1024
                    destination.flush()
                    os.fsync(destination.fileno())
                actual_hash = digest.hexdigest()
                if size != expected_size or actual_hash != expected_hash:
                    raise ValueError("downloaded size/SHA256 does not match immutable ModelScope API metadata")
                if name in EXPECTED_TOKENIZERS and actual_hash != EXPECTED_TOKENIZERS[name]:
                    raise ValueError("downloaded tokenizer fails the separately pinned SHA256")
                os.replace(partial, target)
                result = {
                    "path": name,
                    "status": "verified",
                    "started_at": started_at,
                    "finished_at": now(),
                    "source_url": source_url,
                    "source_commit": REVISION,
                    "file_last_changed_commit": metadata.get("Revision"),
                    "final_host": final_host,
                    "bytes": size,
                    "sha256": actual_hash,
                    "api_sha256": expected_hash,
                    "attempt": attempt,
                }
                log("file-verified", file=name, bytes=size, sha256=actual_hash)
                return result
            except Exception as error:
                partial.unlink(missing_ok=True)
                log("download-attempt-failed", file=name, attempt=attempt, error=str(error))
                if attempt == RETRIES:
                    raise
                time.sleep(attempt * 2)

    try:
        log("listing-started", source_commit=REVISION)
        listing_bytes = None
        for attempt in range(1, RETRIES + 1):
            try:
                with open_url(listing_url) as response:
                    listing_bytes = response.read(1024 * 1024 + 1)
                if len(listing_bytes) > 1024 * 1024:
                    raise ValueError("unexpectedly large ModelScope file listing")
                listing = json.loads(listing_bytes)
                if listing.get("Success") is not True or listing.get("Code") != 200:
                    raise ValueError("ModelScope rejected the fixed revision: " + str(listing.get("Message")))
                break
            except Exception as error:
                log("listing-attempt-failed", attempt=attempt, error=str(error))
                if attempt == RETRIES:
                    raise
                time.sleep(attempt * 2)
        (output / "modelscope-api-files.json").write_bytes(listing_bytes)
        manifest["source_listing_sha256"] = hashlib.sha256(listing_bytes).hexdigest()
        files = {}
        for entry in listing["Data"]["Files"]:
            name = entry.get("Path")
            if name not in REQUIRED_FILES:
                continue
            if name in files:
                raise ValueError("duplicate required path in ModelScope API: " + name)
            if entry.get("Type") != "blob" or not isinstance(entry.get("Size"), int) or entry["Size"] <= 0:
                raise ValueError("invalid file metadata: " + name)
            if not re.fullmatch(r"[0-9a-fA-F]{64}", entry.get("Sha256", "")):
                raise ValueError("missing or malformed ModelScope SHA256: " + name)
            if name in EXPECTED_TOKENIZERS and entry["Sha256"].lower() != EXPECTED_TOKENIZERS[name]:
                raise ValueError("ModelScope tokenizer metadata differs from the pinned contract: " + name)
            files[name] = entry
        if set(files) != set(REQUIRED_FILES):
            raise ValueError("ModelScope snapshot missing required files: " + str(sorted(set(REQUIRED_FILES) - set(files))))
        write_json(manifest_path, manifest)
        for name in REQUIRED_FILES:
            manifest["files"].append(download_file(name, files[name]))
            write_json(manifest_path, manifest)
        manifest["status"] = "verified"
        manifest["finished_at"] = now()
        write_json(manifest_path, manifest)
        log("snapshot-verified", source_commit=REVISION, file_count=len(manifest["files"]))
        return 0
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["finished_at"] = now()
        manifest["error"] = type(error).__name__ + ": " + str(error)
        write_json(manifest_path, manifest)
        log("snapshot-failed", error=manifest["error"])
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    sys.exit(main())
