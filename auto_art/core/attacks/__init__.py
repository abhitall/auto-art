# This file makes the 'attacks' directory a Python package.

from .attack_generator import AttackGenerator

# Exposing specific wrappers directly if desired, e.g.:
# from .evasion.auto_attack import AutoAttackWrapper
# from .poisoning.backdoor_attack import BackdoorAttackWrapper
# etc.
# For now, just AttackGenerator and let users import wrappers from submodules.

__all__ = [
    "AttackGenerator",
]
