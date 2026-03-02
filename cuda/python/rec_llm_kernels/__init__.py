try:
    from . import _C  # type: ignore
except Exception:  # pragma: no cover
    _C = None  # type: ignore
