"""
Warm-start and model-diff evaluation.

Enables incremental evaluations that skip unchanged components,
and differential analysis comparing robustness across model versions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import hashlib
import json
import logging
import time

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = ".auto_art/cache"


@dataclass
class ModelDiffReport:
    """Differential analysis between two evaluation reports."""
    model_a: str
    model_b: str
    improved_attacks: List[str] = field(default_factory=list)
    regressed_attacks: List[str] = field(default_factory=list)
    unchanged_attacks: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "improved_attacks": self.improved_attacks,
            "regressed_attacks": self.regressed_attacks,
            "unchanged_attacks": self.unchanged_attacks,
            "summary": self.summary,
        }


class EvaluationCache:
    """Persists evaluation results to disk as JSON, keyed by model hash."""

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model_hash: str, attack_results: Dict[str, Any]) -> None:
        """Save evaluation results for a given model hash."""
        payload = {
            "model_hash": model_hash,
            "timestamp": time.time(),
            "results": attack_results,
        }
        path = self._path_for(model_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.debug("Cached results for %s at %s", model_hash[:12], path)

    def load(self, model_hash: str) -> Optional[Dict[str, Any]]:
        """Load cached results for a model hash, or None if absent."""
        path = self._path_for(model_hash)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("results")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load cache for %s: %s", model_hash[:12], e)
            return None

    def get_model_hash(self, model_path: str) -> str:
        """Compute SHA-256 of model file contents."""
        h = hashlib.sha256()
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def is_stale(self, model_hash: str, max_age_hours: float = 24.0) -> bool:
        """Check whether cached results are older than *max_age_hours*."""
        path = self._path_for(model_hash)
        if not path.exists():
            return True
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ts = data.get("timestamp", 0.0)
            age_hours = (time.time() - ts) / 3600.0
            return age_hours > max_age_hours
        except (json.JSONDecodeError, OSError):
            return True

    def _path_for(self, model_hash: str) -> Path:
        return self._cache_dir / f"{model_hash}.json"


class WarmStartEvaluator:
    """Wraps the orchestrator to reuse cached results for unchanged models.

    When a model hash matches a cached result that is still fresh, attack
    phases are skipped and the cached data is injected into the report.
    """

    def __init__(self, cache: Optional[EvaluationCache] = None):
        self._cache = cache or EvaluationCache()

    def run(
        self,
        orchestrator: Any,
        agent: Any = None,
        force_rerun: bool = False,
    ) -> Any:
        """Run the orchestrator with warm-start caching.

        Args:
            orchestrator: An ``Orchestrator`` instance.
            agent: Optional agent forwarded to ``orchestrator.run(agent)``.
            force_rerun: Skip cache entirely and rerun everything.

        Returns:
            OrchestratorReport from the orchestrator.
        """
        model_path = getattr(orchestrator, "config", None)
        if model_path is not None:
            model_path = getattr(model_path, "target", {}).get("model_path")

        model_hash: Optional[str] = None
        cached: Optional[Dict[str, Any]] = None

        if model_path and not force_rerun:
            try:
                model_hash = self._cache.get_model_hash(model_path)
                if not self._cache.is_stale(model_hash):
                    cached = self._cache.load(model_hash)
                    if cached is not None:
                        logger.info(
                            "Cache hit for model %s (hash %s…)",
                            model_path, model_hash[:12],
                        )
            except FileNotFoundError:
                logger.debug("Model file not found for hashing; running full eval.")

        if cached is not None:
            report = self._build_cached_report(orchestrator, cached)
        else:
            report = orchestrator.run(agent)
            if model_hash is not None:
                self._save_report(model_hash, report)

        return report

    def _save_report(self, model_hash: str, report: Any) -> None:
        try:
            phases = getattr(report, "phases", [])
            results_map: Dict[str, Any] = {}
            for phase in phases:
                for r in phase.get("results", []):
                    key = r.get("attack", r.get("defence", "unknown"))
                    results_map[key] = r
            self._cache.save(model_hash, {
                "phases": phases,
                "gate_results": getattr(report, "gate_results", {}),
                "passed": getattr(report, "passed", True),
                "summary": getattr(report, "summary", {}),
            })
        except Exception as e:
            logger.warning("Failed to cache report: %s", e)

    @staticmethod
    def _build_cached_report(orchestrator: Any, cached: Dict[str, Any]) -> Any:
        from auto_art.core.orchestrator import OrchestratorReport
        report = OrchestratorReport(
            timestamp=time.time(),
            target=getattr(orchestrator.config, "target", {}),
            phases=cached.get("phases", []),
            gate_results=cached.get("gate_results", {}),
            passed=cached.get("passed", True),
            summary=cached.get("summary", {}),
        )
        return report


class ModelDiffAnalyzer:
    """Compares evaluation results across model versions."""

    TOLERANCE = 1e-6

    def compare(self, report_a: Any, report_b: Any) -> ModelDiffReport:
        """Compare two OrchestratorReports and categorize attack changes.

        An attack is *improved* if its success rate decreased from A to B,
        *regressed* if it increased, and *unchanged* otherwise.
        """
        rates_a = self._extract_attack_rates(report_a)
        rates_b = self._extract_attack_rates(report_b)

        model_a = self._model_label(report_a, "A")
        model_b = self._model_label(report_b, "B")

        all_attacks = sorted(set(rates_a) | set(rates_b))
        improved: List[str] = []
        regressed: List[str] = []
        unchanged: List[str] = []

        for atk in all_attacks:
            sr_a = rates_a.get(atk, 0.0)
            sr_b = rates_b.get(atk, 0.0)
            delta = sr_b - sr_a
            if delta < -self.TOLERANCE:
                improved.append(atk)
            elif delta > self.TOLERANCE:
                regressed.append(atk)
            else:
                unchanged.append(atk)

        summary_parts = [
            f"{len(improved)} improved",
            f"{len(regressed)} regressed",
            f"{len(unchanged)} unchanged",
        ]
        summary = f"Comparing {model_a} → {model_b}: " + ", ".join(summary_parts)

        return ModelDiffReport(
            model_a=model_a,
            model_b=model_b,
            improved_attacks=improved,
            regressed_attacks=regressed,
            unchanged_attacks=unchanged,
            summary=summary,
        )

    @staticmethod
    def _extract_attack_rates(report: Any) -> Dict[str, float]:
        rates: Dict[str, float] = {}
        phases = getattr(report, "phases", [])
        for phase in phases:
            for result in phase.get("results", []):
                name = result.get("attack", result.get("defence"))
                if name is None:
                    continue
                sr = result.get("success_rate")
                if sr is not None:
                    rates[name] = float(sr)
        return rates

    @staticmethod
    def _model_label(report: Any, fallback: str) -> str:
        target = getattr(report, "target", {})
        return target.get("model_path", target.get("model_name", fallback))
