"""
RobustBench leaderboard integration.

Compares model robustness against the RobustBench leaderboard,
tracking model rank and accuracy under attack over time.

Reference: Croce et al., 2021 - "RobustBench: a standardized
adversarial robustness benchmark"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

ROBUSTBENCH_API_BASE = (
    "https://raw.githubusercontent.com/RobustBench/robustbench/"
    "master/model_info"
)


class RobustBenchDataset(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    IMAGENET = "imagenet"


class RobustBenchThreatModel(Enum):
    LINF = "Linf"
    L2 = "L2"
    CORRUPTIONS = "corruptions"


@dataclass
class LeaderboardEntry:
    """Single row from the RobustBench leaderboard."""
    model_name: str
    clean_accuracy: float
    robust_accuracy: float
    dataset: str
    threat_model: str
    epsilon: float = 0.0
    paper_url: str = ""


@dataclass
class RobustBenchComparison:
    """Result of comparing a model against the leaderboard."""
    model_name: str
    rank: int
    total_models: int
    percentile: float
    clean_acc: float
    robust_acc: float
    nearest_above: Optional[LeaderboardEntry] = None
    nearest_below: Optional[LeaderboardEntry] = None

    def summary(self) -> str:
        pct = self.percentile
        return (
            f"{self.model_name}: rank {self.rank}/{self.total_models} "
            f"(top {100 - pct:.1f}%), robust_acc={self.robust_acc:.2f}%"
        )


# Public data from robustbench.github.io — CIFAR-10, Linf, eps=8/255.
# Robust accuracy evaluated with AutoAttack.
CACHED_LEADERBOARD: Dict[Tuple[str, str], List[Dict[str, Any]]] = {
    ("cifar10", "Linf"): [
        {
            "model_name": "Wang2023Better_WRN-70-16",
            "clean_accuracy": 93.25,
            "robust_accuracy": 70.69,
            "epsilon": 8 / 255,
            "paper_url": "https://arxiv.org/abs/2302.04638",
        },
        {
            "model_name": "Rebuffi2021Fixing_70_16_cutmix_extra",
            "clean_accuracy": 92.23,
            "robust_accuracy": 66.56,
            "epsilon": 8 / 255,
            "paper_url": "https://arxiv.org/abs/2103.01946",
        },
        {
            "model_name": "Gowal2021Improving_70_16_ddpm_100m",
            "clean_accuracy": 88.74,
            "robust_accuracy": 66.10,
            "epsilon": 8 / 255,
            "paper_url": "https://arxiv.org/abs/2103.01946",
        },
        {
            "model_name": "Rebuffi2021Fixing_28_10_cutmix_ddpm",
            "clean_accuracy": 87.33,
            "robust_accuracy": 64.58,
            "epsilon": 8 / 255,
            "paper_url": "https://arxiv.org/abs/2103.01946",
        },
        {
            "model_name": "Gowal2021Improving_28_10_ddpm_100m",
            "clean_accuracy": 87.50,
            "robust_accuracy": 63.38,
            "epsilon": 8 / 255,
            "paper_url": "https://arxiv.org/abs/2103.01946",
        },
        {
            "model_name": "Pang2022Robustness_WRN70_16",
            "clean_accuracy": 93.27,
            "robust_accuracy": 71.07,
            "epsilon": 8 / 255,
            "paper_url": "https://arxiv.org/abs/2210.06284",
        },
        {
            "model_name": "Xu2023Exploring_WRN-28-10",
            "clean_accuracy": 93.69,
            "robust_accuracy": 63.89,
            "epsilon": 8 / 255,
            "paper_url": "https://arxiv.org/abs/2302.14862",
        },
        {
            "model_name": "Gowal2020Uncovering_70_16_extra",
            "clean_accuracy": 91.10,
            "robust_accuracy": 65.87,
            "epsilon": 8 / 255,
            "paper_url": "https://arxiv.org/abs/2010.03593",
        },
        {
            "model_name": "Sehwag2021Proxy_ResNest152",
            "clean_accuracy": 87.30,
            "robust_accuracy": 60.27,
            "epsilon": 8 / 255,
            "paper_url": "https://arxiv.org/abs/2104.09425",
        },
        {
            "model_name": "Wong2020Fast",
            "clean_accuracy": 83.34,
            "robust_accuracy": 43.21,
            "epsilon": 8 / 255,
            "paper_url": "https://arxiv.org/abs/2001.03994",
        },
    ],
}


class RobustBenchClient:
    """Fetches and queries the RobustBench leaderboard."""

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    def get_leaderboard(
        self,
        dataset: RobustBenchDataset,
        threat_model: RobustBenchThreatModel,
    ) -> List[LeaderboardEntry]:
        """Fetch leaderboard entries, falling back to cached data on failure."""
        ds = dataset.value if hasattr(dataset, 'value') else str(dataset)
        tm = threat_model.value if hasattr(threat_model, 'value') else str(threat_model)

        entries = self._try_fetch_remote(ds, tm)
        if entries:
            return entries

        logger.info("Using cached leaderboard data for %s/%s", ds, tm)
        return self._load_cached(ds, tm)

    def compare_model(
        self,
        model_name: str,
        clean_acc: float,
        robust_acc: float,
        dataset: RobustBenchDataset,
        threat_model: RobustBenchThreatModel,
    ) -> RobustBenchComparison:
        """Compare a model's accuracy against the leaderboard."""
        leaderboard = self.get_leaderboard(dataset, threat_model)
        sorted_lb = sorted(leaderboard, key=lambda e: e.robust_accuracy, reverse=True)

        rank = 1
        nearest_above: Optional[LeaderboardEntry] = None
        nearest_below: Optional[LeaderboardEntry] = None

        for entry in sorted_lb:
            if entry.robust_accuracy > robust_acc:
                rank += 1
                nearest_above = entry
            elif nearest_below is None:
                nearest_below = entry

        total = len(sorted_lb) + 1
        percentile = ((total - rank) / total) * 100.0

        return RobustBenchComparison(
            model_name=model_name,
            rank=rank,
            total_models=total,
            percentile=percentile,
            clean_acc=clean_acc,
            robust_acc=robust_acc,
            nearest_above=nearest_above,
            nearest_below=nearest_below,
        )

    def _try_fetch_remote(self, dataset: str, threat_model: str) -> List[LeaderboardEntry]:
        url = f"{ROBUSTBENCH_API_BASE}/{dataset}/{threat_model}.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "auto-art/0.2"})
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
            return self._parse_remote(data, dataset, threat_model)
        except (urllib.error.URLError, json.JSONDecodeError, Exception) as e:
            logger.debug("Remote fetch failed for %s: %s", url, e)
            return []

    @staticmethod
    def _parse_remote(
        data: Dict[str, Any],
        dataset: str,
        threat_model: str,
    ) -> List[LeaderboardEntry]:
        entries: List[LeaderboardEntry] = []
        for model_name, info in data.items():
            if not isinstance(info, dict):
                continue
            entries.append(LeaderboardEntry(
                model_name=model_name,
                clean_accuracy=float(info.get("clean_acc", 0.0)) * 100,
                robust_accuracy=float(info.get("autoattack_acc", info.get("robust_acc", 0.0))) * 100,
                dataset=dataset,
                threat_model=threat_model,
                epsilon=float(info.get("eps", 0.0)),
                paper_url=info.get("paper_url", ""),
            ))
        return sorted(entries, key=lambda e: e.robust_accuracy, reverse=True)

    @staticmethod
    def _load_cached(dataset: str, threat_model: str) -> List[LeaderboardEntry]:
        key = (dataset, threat_model)
        raw = CACHED_LEADERBOARD.get(key, [])
        entries = [
            LeaderboardEntry(
                model_name=r["model_name"],
                clean_accuracy=r["clean_accuracy"],
                robust_accuracy=r["robust_accuracy"],
                dataset=dataset,
                threat_model=threat_model,
                epsilon=r.get("epsilon", 0.0),
                paper_url=r.get("paper_url", ""),
            )
            for r in raw
        ]
        return sorted(entries, key=lambda e: e.robust_accuracy, reverse=True)
