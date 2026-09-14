try:
    from . import vllm_engine_pb2, vllm_engine_pb2_grpc
except ImportError as exc:
    raise ImportError(
        "gRPC stubs are missing. They are generated from "
        "proto/vllm_engine.proto during `pip install -e .` / wheel build. "
        "Re-install the package so setup.py can run grpcio-tools."
    ) from exc

__all__ = ["vllm_engine_pb2", "vllm_engine_pb2_grpc"]
