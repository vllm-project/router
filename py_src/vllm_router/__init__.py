from vllm_router.version import __version__

try:
    from vllm_router.router import Router
except ImportError:
    Router = None

__all__ = ["__version__"]
if Router is not None:
    __all__.append("Router")
