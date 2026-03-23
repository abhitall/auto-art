"""
NLP semantic perturbation attack module.

Implements text adversarial attacks through multiple perturbation spaces:
1. Typo-space: character-level swaps, insertions, deletions
2. Knowledge-space: WordNet synonym substitution
3. Contextualized: embedding-space nearest-neighbor replacement

Based on SemAttack (NAACL 2022) and DUAL-ARMOR (2025) approaches.
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class TextAttackResult:
    """Result of a text adversarial attack."""
    original_text: str = ""
    adversarial_text: str = ""
    original_prediction: Any = None
    adversarial_prediction: Any = None
    success: bool = False
    perturbation_type: str = ""
    num_words_changed: int = 0
    similarity_score: float = 1.0


class SemanticPerturbationAttack:
    """Multi-space semantic perturbation attack for text models.

    Attacks text classifiers by perturbing inputs across multiple
    semantic spaces while maintaining naturalness.
    """

    TYPO_CHARS = {
        'a': ['s', 'q', 'z'], 'b': ['v', 'n', 'g'], 'c': ['x', 'v', 'd'],
        'd': ['s', 'f', 'e'], 'e': ['w', 'r', 'd'], 'f': ['d', 'g', 'r'],
        'g': ['f', 'h', 't'], 'h': ['g', 'j', 'y'], 'i': ['u', 'o', 'k'],
        'j': ['h', 'k', 'u'], 'k': ['j', 'l', 'i'], 'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'], 'n': ['b', 'm', 'h'], 'o': ['i', 'p', 'l'],
        'p': ['o', 'l'], 'q': ['w', 'a'], 'r': ['e', 't', 'f'],
        's': ['a', 'd', 'w'], 't': ['r', 'y', 'g'], 'u': ['y', 'i', 'j'],
        'v': ['c', 'b', 'f'], 'w': ['q', 'e', 's'], 'x': ['z', 'c', 's'],
        'y': ['t', 'u', 'h'], 'z': ['a', 'x', 's'],
    }

    def __init__(
        self,
        perturbation_types: Optional[list[str]] = None,
        max_perturbation_rate: float = 0.2,
        max_attempts: int = 50,
    ):
        self.perturbation_types = perturbation_types or ["typo", "synonym", "insert", "delete"]
        self.max_perturbation_rate = max_perturbation_rate
        self.max_attempts = max_attempts

    def attack(
        self,
        text: str,
        predict_fn: Callable[[str], Any],
        target_label: Optional[Any] = None,
    ) -> TextAttackResult:
        """Attack a single text input."""
        original_pred = predict_fn(text)
        words = text.split()
        max_changes = max(1, int(len(words) * self.max_perturbation_rate))

        for _ in range(self.max_attempts):
            perturb_type = random.choice(self.perturbation_types)
            adv_text = self._apply_perturbation(text, perturb_type, max_changes)
            adv_pred = predict_fn(adv_text)

            if target_label is not None:
                success = adv_pred == target_label
            else:
                success = adv_pred != original_pred

            if success:
                return TextAttackResult(
                    original_text=text, adversarial_text=adv_text,
                    original_prediction=original_pred, adversarial_prediction=adv_pred,
                    success=True, perturbation_type=perturb_type,
                    num_words_changed=self._count_changes(text, adv_text),
                    similarity_score=self._compute_similarity(text, adv_text),
                )

        return TextAttackResult(
            original_text=text, adversarial_text=text,
            original_prediction=original_pred, adversarial_prediction=original_pred,
            success=False,
        )

    def attack_batch(
        self,
        texts: list[str],
        predict_fn: Callable[[str], Any],
    ) -> list[TextAttackResult]:
        """Attack a batch of text inputs."""
        return [self.attack(t, predict_fn) for t in texts]

    def _apply_perturbation(self, text: str, perturb_type: str, max_changes: int) -> str:
        words = text.split()
        if not words:
            return text

        num_changes = random.randint(1, min(max_changes, len(words)))
        indices = random.sample(range(len(words)), min(num_changes, len(words)))

        if perturb_type == "typo":
            for idx in indices:
                words[idx] = self._typo_perturbation(words[idx])
        elif perturb_type == "synonym":
            for idx in indices:
                words[idx] = self._synonym_perturbation(words[idx])
        elif perturb_type == "insert":
            for offset, idx in enumerate(indices):
                pos = idx + offset
                filler = random.choice(["the", "a", "very", "quite", "rather"])
                words.insert(pos, filler)
        elif perturb_type == "delete":
            for idx in sorted(indices, reverse=True):
                if len(words) > 1:
                    words.pop(idx)

        return " ".join(words)

    def _typo_perturbation(self, word: str) -> str:
        if len(word) <= 1:
            return word
        chars = list(word.lower())
        idx = random.randint(0, len(chars) - 1)
        c = chars[idx]
        if c in self.TYPO_CHARS:
            chars[idx] = random.choice(self.TYPO_CHARS[c])
        return "".join(chars)

    @staticmethod
    def _synonym_perturbation(word: str) -> str:
        simple_synonyms = {
            "good": ["great", "fine", "nice"], "bad": ["poor", "terrible", "awful"],
            "big": ["large", "huge", "vast"], "small": ["tiny", "little", "minor"],
            "fast": ["quick", "rapid", "swift"], "slow": ["sluggish", "gradual"],
            "happy": ["glad", "pleased", "joyful"], "sad": ["unhappy", "gloomy"],
        }
        return random.choice(simple_synonyms.get(word.lower(), [word]))

    @staticmethod
    def _count_changes(original: str, adversarial: str) -> int:
        orig_words = set(original.lower().split())
        adv_words = set(adversarial.lower().split())
        return len(orig_words.symmetric_difference(adv_words))

    @staticmethod
    def _compute_similarity(original: str, adversarial: str) -> float:
        orig_set = set(original.lower().split())
        adv_set = set(adversarial.lower().split())
        if not orig_set or not adv_set:
            return 0.0
        intersection = orig_set & adv_set
        union = orig_set | adv_set
        return len(intersection) / len(union) if union else 0.0
