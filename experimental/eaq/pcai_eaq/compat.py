"""Fail at plugin install when the vLLM or vllm-exl3 internals this plugin patches have changed.

The plugin replaces private methods. A changed signature would otherwise fail much later,
during model loading, far from the cause.
"""
import inspect
import os

QUALIFIED_VLLM = "0.32."
ALLOW_UNTESTED = "PCAI_EAQ_ALLOW_UNTESTED_VLLM"


def check_vllm():
    import vllm

    version = getattr(vllm, "__version__", "")
    if not version.startswith(QUALIFIED_VLLM) and os.environ.get(ALLOW_UNTESTED) != "1":
        raise RuntimeError(
            f"pcai_eaq patches vLLM internals qualified on {QUALIFIED_VLLM}x, not {version!r}; "
            f"set {ALLOW_UNTESTED}=1 to try an untested version")


def require(owner, name, params=None):
    """owner.name must exist and, if given, take exactly these parameters."""
    label = f"{getattr(owner, '__module__', '?')}.{getattr(owner, '__qualname__', owner)}.{name}"
    if not hasattr(owner, name):
        raise RuntimeError(f"pcai_eaq: {label} no longer exists; the plugin needs updating")
    if params is not None:
        got = list(inspect.signature(getattr(owner, name)).parameters)
        if got != list(params):
            raise RuntimeError(f"pcai_eaq: {label} now takes {got}, expected {list(params)}; "
                               "the plugin needs updating")
