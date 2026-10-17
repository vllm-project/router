#!/usr/bin/env python3
"""Expose discovery for a single native HTTP engine during device acceptance.

For the pinned Python vLLM 0.30 launcher layout, not a production plugin. The
two discovery routes read the live engine configuration; generation and event
publishing remain native. No cache residency or event payload is synthesized.
"""

import dataclasses
import sys
import uuid


def main():
    from fastapi import Request
    from vllm.entrypoints.launchers import app as app_module
    from vllm.entrypoints.launchers.api_server import entry

    original = app_module.build_app
    instance = uuid.uuid4().hex

    def build_app(*args, **kwargs):
        app = original(*args, **kwargs)
        paths = {getattr(route, "path", None) for route in app.routes}
        if {"/get_server_info", "/kv_event_sources"} & paths:
            raise RuntimeError("native discovery already exists; use it directly")

        def config(request):
            value = request.app.state.vllm_config
            parallel = value.parallel_config
            if parallel.data_parallel_size != 1:
                raise RuntimeError("this acceptance adapter only supports DP=1")
            if value.lora_config is not None or value.kv_transfer_config is not None:
                raise RuntimeError("LoRA and external KV are outside acceptance")
            hf = value.model_config.hf_text_config
            if hf.model_type != "qwen3" or getattr(hf, "use_sliding_window", False):
                raise RuntimeError("this acceptance recipe requires dense Qwen3")
            return value

        @app.get("/get_server_info")
        async def server_info(request: Request):
            value = config(request)
            names = value.model_config.served_model_name
            model = names[0] if isinstance(names, list) else names
            return {
                "instance_id": instance,
                "model_id": model,
                "model_path": value.model_config.model,
                "kv_block_size": value.cache_config.block_size,
            }

        @app.get("/kv_event_sources")
        async def event_sources(request: Request):
            value = config(request).kv_events_config
            if value is None or not value.enable_kv_cache_events:
                raise RuntimeError("native KV publishing must be enabled")
            return {"0": dataclasses.asdict(value)}

        return app

    app_module.build_app = build_app
    if hasattr(entry, "build_app"):
        entry.build_app = build_app
    sys.argv = ["vllm", *sys.argv[1:]]
    from vllm.entrypoints.cli.main import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
