"""Unit tests for auto_art.core.robustbench."""
from auto_art.core.robustbench import (
    CACHED_LEADERBOARD,
    RobustBenchClient,
    RobustBenchDataset,
    RobustBenchThreatModel,
)


def test_cached_leaderboard():
    assert len(CACHED_LEADERBOARD) >= 1
    key = ("cifar10", "Linf")
    assert key in CACHED_LEADERBOARD
    assert len(CACHED_LEADERBOARD[key]) >= 3


def test_client_get_leaderboard_cached():
    client = RobustBenchClient(timeout=0.01)
    entries = client.get_leaderboard(RobustBenchDataset.CIFAR10, RobustBenchThreatModel.LINF)
    assert len(entries) >= 3
    assert all(e.robust_accuracy > 0 for e in entries)


def test_client_compare_model():
    client = RobustBenchClient(timeout=0.01)
    comp = client.compare_model(
        model_name="TestModel",
        clean_acc=90.0,
        robust_acc=50.0,
        dataset=RobustBenchDataset.CIFAR10,
        threat_model=RobustBenchThreatModel.LINF,
    )
    assert 1 <= comp.rank <= comp.total_models
    assert 0.0 <= comp.percentile <= 100.0
    assert comp.robust_acc == 50.0
