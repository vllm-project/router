from vllm_router.version import __version__

try:
    from vllm_router.router import Router

    __all__ = ["__version__", "Router"]
except ImportError:
    # Router is not available if Rust extension is not built
    __all__ = ["__version__"]
