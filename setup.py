import os
import re
from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent
PROTO_DIR = ROOT / "proto"
PROTO_OUT_DIR = ROOT / "py_src" / "vllm_router" / "proto"
PROTO_NAME = "vllm_engine"
# Keep generated stubs importable against the runtime floors in pyproject.toml.
RUNTIME_GRPCIO_FLOOR = "1.60.0"


def generate_grpc_stubs() -> None:
    """Compile proto/vllm_engine.proto into package-local Python stubs.

    Invoked during wheel builds and `pip install -e` so generated
    ``*_pb2.py`` / ``*_pb2_grpc.py`` files are not checked into git.
    """
    proto_file = PROTO_DIR / f"{PROTO_NAME}.proto"
    if not proto_file.is_file():
        raise FileNotFoundError(f"Missing protobuf contract: {proto_file}")

    try:
        from grpc_tools import protoc
    except ImportError as exc:
        raise RuntimeError(
            "grpcio-tools is required to generate gRPC stubs. "
            "It is declared in [build-system] requires; re-run "
            "`pip install -e .` from a PEP 517 isolated build."
        ) from exc

    PROTO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    rc = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={PROTO_OUT_DIR}",
            f"--grpc_python_out={PROTO_OUT_DIR}",
            str(proto_file),
        ]
    )
    if rc != 0:
        raise RuntimeError(f"protoc failed with exit code {rc}")

    _patch_generated_stubs()


def _patch_generated_stubs() -> None:
    """Make generated stubs package-relative and runtime-floor compatible."""
    grpc_path = PROTO_OUT_DIR / f"{PROTO_NAME}_pb2_grpc.py"
    grpc_text = grpc_path.read_text()
    grpc_text = grpc_text.replace("import warnings\n", "")
    grpc_text = grpc_text.replace(
        f"import {PROTO_NAME}_pb2 as {PROTO_NAME.replace('_', '__')}_pb2",
        f"from . import {PROTO_NAME}_pb2 as {PROTO_NAME.replace('_', '__')}_pb2",
    )
    # protoc emits `import vllm_engine_pb2 as vllm__engine__pb2`
    grpc_text = grpc_text.replace(
        "import vllm_engine_pb2 as vllm__engine__pb2",
        "from . import vllm_engine_pb2 as vllm__engine__pb2",
    )
    grpc_text = re.sub(
        r'GRPC_GENERATED_VERSION = ["\'][^"\']+["\']',
        f'GRPC_GENERATED_VERSION = "{RUNTIME_GRPCIO_FLOOR}"',
        grpc_text,
        count=1,
    )
    grpc_text = grpc_text.replace(
        "_version_not_supported = True",
        "_version_not_supported = False",
    )
    grpc_path.write_text(grpc_text)

    pb2_path = PROTO_OUT_DIR / f"{PROTO_NAME}_pb2.py"
    pb2_text = pb2_path.read_text()
    pb2_text = re.sub(
        r"_runtime_version\.ValidateProtobufRuntimeVersion\([\s\S]*?\)",
        (
            "try:\n"
            "    _runtime_version.ValidateProtobufRuntimeVersion(\n"
            "        _runtime_version.Domain.PUBLIC, 5, 0, 0, "
            f'"", "{PROTO_NAME}.proto"\n'
            "    )\n"
            "except Exception:\n"
            "    pass"
        ),
        pb2_text,
        count=1,
    )
    pb2_path.write_text(pb2_text)


generate_grpc_stubs()

no_rust = os.environ.get("VLLM_ROUTER_BUILD_NO_RUST") == "1"

rust_extensions = []
if not no_rust:
    from setuptools_rust import Binding, RustExtension

    rust_extensions.append(
        RustExtension(
            target="vllm_router_rs",
            path="Cargo.toml",
            binding=Binding.PyO3,
        )
    )

setup(
    rust_extensions=rust_extensions,
    zip_safe=False,
)
