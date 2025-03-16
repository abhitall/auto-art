"""
Tests for the circuit breaker and rollback mechanism.
"""

import time
from unittest.mock import MagicMock

from auto_art.core.evaluation.defences.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    AgentStateSnapshot,
    CircuitBreakerAlert,
    MetricsWindow,
)


class TestMetricsWindow:

    def test_record_and_count(self):
        window = MetricsWindow(window_seconds=10.0)
        window.record(1.0)
        window.record(0.0)
        window.record(1.0)
        assert window.count == 3

    def test_error_rate(self):
        window = MetricsWindow(window_seconds=10.0)
        window.record(1.0)
        window.record(0.0)
        window.record(1.0)
        window.record(0.0)
        assert window.error_rate() == 0.5

    def test_mean(self):
        window = MetricsWindow(window_seconds=10.0)
        window.record(10.0)
        window.record(20.0)
        window.record(30.0)
        assert window.mean() == 20.0

    def test_empty_window(self):
        window = MetricsWindow(window_seconds=10.0)
        assert window.count == 0
        assert window.error_rate() == 0.0
        assert window.mean() == 0.0


class TestCircuitBreaker:

    def test_initial_state_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_allowing_requests

    def test_record_normal_request(self):
        cb = CircuitBreaker()
        alert = cb.record_request(is_error=False, token_count=100)
        assert alert is None
        assert cb.state == CircuitState.CLOSED

    def test_trips_on_high_error_rate(self):
        config = CircuitBreakerConfig(error_rate_threshold=0.3, monitoring_window_seconds=60.0)
        cb = CircuitBreaker(config=config)
        for _ in range(10):
            cb.record_request(is_error=True)
        assert cb.state == CircuitState.OPEN
        assert not cb.is_allowing_requests

    def test_trips_on_token_spike(self):
        config = CircuitBreakerConfig(token_usage_spike_factor=2.0, monitoring_window_seconds=60.0)
        cb = CircuitBreaker(config=config)
        cb.set_baseline_token_usage(100.0)
        alert = cb.record_request(is_error=False, token_count=500)
        if alert is None:
            for _ in range(5):
                cb.record_request(is_error=False, token_count=500)
        assert cb.state == CircuitState.OPEN

    def test_trips_on_action_frequency(self):
        config = CircuitBreakerConfig(action_frequency_threshold=5, monitoring_window_seconds=60.0)
        cb = CircuitBreaker(config=config)
        for _ in range(10):
            cb.record_request(is_error=False)
        assert cb.state == CircuitState.OPEN

    def test_on_trip_callback(self):
        callback = MagicMock()
        config = CircuitBreakerConfig(error_rate_threshold=0.1)
        cb = CircuitBreaker(config=config, on_trip=callback)
        for _ in range(10):
            cb.record_request(is_error=True)
        callback.assert_called()
        alert = callback.call_args[0][0]
        assert isinstance(alert, CircuitBreakerAlert)

    def test_save_and_get_snapshot(self):
        cb = CircuitBreaker()
        snapshot = AgentStateSnapshot(
            snapshot_id="snap_001",
            timestamp=time.time(),
            prompt_version="v1",
            rag_db_version="v1",
            agent_config={"model": "test"},
        )
        cb.save_snapshot(snapshot)
        assert cb.get_latest_snapshot() == snapshot

    def test_max_snapshots(self):
        config = CircuitBreakerConfig(max_snapshots=3)
        cb = CircuitBreaker(config=config)
        for i in range(5):
            cb.save_snapshot(AgentStateSnapshot(
                snapshot_id=f"snap_{i}",
                timestamp=time.time(),
                prompt_version=f"v{i}",
                rag_db_version="v1",
            ))
        assert len(cb._snapshots) == 3
        assert cb.get_latest_snapshot().snapshot_id == "snap_4"

    def test_force_open(self):
        cb = CircuitBreaker()
        alert = cb.force_open("emergency test")
        assert cb.state == CircuitState.OPEN
        assert "emergency test" in alert.message

    def test_force_close(self):
        cb = CircuitBreaker()
        cb.force_open("test")
        cb.force_close()
        assert cb.state == CircuitState.CLOSED

    def test_rollback(self):
        cb = CircuitBreaker()
        snapshot = AgentStateSnapshot(
            snapshot_id="snap_rb",
            timestamp=time.time(),
            prompt_version="v1",
            rag_db_version="v1",
            agent_config={"key": "value"},
            memory_state={"data": [1, 2, 3]},
        )
        cb.save_snapshot(snapshot)

        agent = MagicMock()
        result = cb.rollback(agent)
        assert result == snapshot
        agent.set_config.assert_called_once_with({"key": "value"})
        agent.set_memory.assert_called_once()

    def test_rollback_no_snapshots(self):
        cb = CircuitBreaker()
        result = cb.rollback(MagicMock())
        assert result is None

    def test_get_status(self):
        cb = CircuitBreaker()
        cb.record_request(is_error=False, token_count=50)
        status = cb.get_status()
        assert "state" in status
        assert "error_rate" in status
        assert "action_count" in status

    def test_get_alerts(self):
        cb = CircuitBreaker()
        cb.force_open("test1")
        cb.force_close()
        cb.force_open("test2")
        alerts = cb.get_alerts()
        assert len(alerts) == 2

    def test_half_open_transition(self):
        config = CircuitBreakerConfig(
            error_rate_threshold=0.1,
            cooldown_seconds=0.01,
        )
        cb = CircuitBreaker(config=config)
        for _ in range(10):
            cb.record_request(is_error=True)
        assert cb._state == CircuitState.OPEN

        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.is_allowing_requests
