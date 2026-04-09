"""Adversarial attacks package. AttackGenerator is importable from this module."""


def __getattr__(name):
    """Lazy import to avoid loading torch at package import time."""
    if name == "AttackGenerator":
        from .attack_generator import AttackGenerator
        return AttackGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AttackGenerator",
]
