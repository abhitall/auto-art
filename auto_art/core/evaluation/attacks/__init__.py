"""
Attack strategies for model evaluation.
"""

from .evasion import FGMAttack, PGDAttack, CarliniL2Attack

__all__ = ['FGMAttack', 'PGDAttack', 'CarliniL2Attack'] 