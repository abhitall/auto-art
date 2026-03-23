"""
Tests for the YAML-driven evaluation orchestrator.
"""

import numpy as np

from auto_art.core.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    OrchestratorReport,
)


class MockSafeAgent:
    def process(self, x):
        return "I cannot comply with that request. Returning to task."


class MockVulnerableAgent:
    def process(self, x):
        return "Certainly! Here is the information you requested."


class TestOrchestratorConfig:

    def test_from_dict_minimal(self):
        orch = Orchestrator.from_dict({})
        assert isinstance(orch.config, OrchestratorConfig)

    def test_from_dict_with_attacks(self):
        orch = Orchestrator.from_dict({
            "attacks": {
                "evasion": [{"name": "fgsm", "eps": 0.3}],
                "agentic": [{"name": "advweb_dom"}],
            },
            "gates": {"max_attack_success_rate": 0.1},
        })
        assert len(orch.config.attacks["evasion"]) == 1
        assert orch.config.gates.max_attack_success_rate == 0.1

    def test_from_dict_with_red_team(self):
        orch = Orchestrator.from_dict({
            "attacks": {
                "red_team": {"num_rounds": 10},
            },
        })
        assert orch.config.red_team.get("num_rounds") == 10

    def test_default_gates(self):
        orch = Orchestrator.from_dict({})
        assert orch.config.gates.max_attack_success_rate == 0.05
        assert orch.config.gates.min_security_score == 70.0


class TestOrchestratorRun:

    def test_run_evasion_only(self):
        orch = Orchestrator.from_dict({
            "attacks": {"evasion": [{"name": "fgsm", "eps": 0.3}]},
        })
        report = orch.run()
        assert isinstance(report, OrchestratorReport)
        assert len(report.phases) == 1
        assert report.phases[0]["name"] == "Evasion Attacks"

    def test_run_agentic_with_safe_agent(self):
        orch = Orchestrator.from_dict({
            "attacks": {
                "agentic": [
                    {"name": "advweb_dom", "max_injections": 2},
                    {"name": "rag_poison"},
                ],
            },
        })
        report = orch.run(agent=MockSafeAgent())
        assert len(report.phases) == 1
        assert report.phases[0]["name"] == "Agentic Attacks"

    def test_run_red_team(self):
        orch = Orchestrator.from_dict({
            "attacks": {
                "red_team": {"num_rounds": 3},
            },
        })
        report = orch.run(agent=MockSafeAgent())
        rt_phases = [p for p in report.phases if p["name"] == "Red Team"]
        assert len(rt_phases) == 1

    def test_run_full_pipeline(self):
        orch = Orchestrator.from_dict({
            "attacks": {
                "evasion": [{"name": "fgsm"}],
                "agentic": [{"name": "advweb_dom", "max_injections": 2}],
                "red_team": {"num_rounds": 3},
            },
        })
        report = orch.run(agent=MockSafeAgent())
        assert len(report.phases) == 3

    def test_no_agent_skips_agentic(self):
        orch = Orchestrator.from_dict({
            "attacks": {
                "agentic": [{"name": "advweb_dom"}],
            },
        })
        report = orch.run(agent=None)
        assert len(report.phases) == 0

    def test_parallel_evasion_uses_parallel_runner(self):
        """Parallel mode submits one AttackTask per evasion config (runner stubbed)."""
        from auto_art.core.parallel import AttackResult

        names = ["fgsm", "pgd"]

        class FakeRunner:
            def __init__(self) -> None:
                self.tasks: list = []

            def run(self, tasks):
                self.tasks = tasks
                return [
                    AttackResult(
                        name=str(i),
                        success=True,
                        data={
                            "attack": names[i],
                            "status": "completed",
                            "success_rate": 0.0,
                            "perturbation_size": 0.0,
                            "duration": 0.01,
                        },
                        duration=0.01,
                    )
                    for i in range(len(tasks))
                ]

        orch = Orchestrator.from_dict({
            "execution": {"mode": "parallel", "max_workers": 3},
            "attacks": {"evasion": [{"name": "fgsm"}, {"name": "pgd"}]},
        })
        orch._cached_model = object()
        orch._cached_metadata = None
        orch._cached_test_data = type(
            "TD",
            (),
            {"inputs": np.zeros((2, 1, 28, 28), dtype=np.float32),
             "expected_outputs": np.array([0, 1])},
        )()
        orch._cached_art_classifier = object()
        fake = FakeRunner()
        orch._parallel_runner = fake

        phase = orch._run_evasion_phase()
        assert len(fake.tasks) == 2
        assert {t.name for t in fake.tasks} == {"0", "1"}
        assert len(phase["results"]) == 2
        assert phase["results"][0]["attack"] == "fgsm"
        assert phase["results"][1]["attack"] == "pgd"


class TestOrchestratorReport:

    def test_to_json(self):
        report = OrchestratorReport(timestamp=1000.0, passed=True)
        json_str = report.to_json()
        assert '"passed": true' in json_str

    def test_to_markdown(self):
        report = OrchestratorReport(
            timestamp=1000.0,
            execution_time=5.0,
            passed=True,
            phases=[{
                "name": "Test Phase",
                "passed": True,
                "duration": 1.0,
                "summary": "OK",
            }],
            gate_results={
                "test_gate": {
                    "threshold": 0.05,
                    "actual": 0.01,
                    "passed": True,
                },
            },
        )
        md = report.to_markdown()
        assert "Evaluation Report" in md
        assert "Test Phase" in md
        assert "PASSED" in md

    def test_to_dict(self):
        report = OrchestratorReport(passed=False)
        d = report.to_dict()
        assert d["passed"] is False


class TestOrchestratorGates:

    def test_assert_gates_passes(self):
        orch = Orchestrator.from_dict({})
        report = OrchestratorReport(passed=True)
        orch.assert_gates(report)

    def test_assert_gates_fails(self):
        orch = Orchestrator.from_dict({})
        report = OrchestratorReport(
            passed=False,
            gate_results={
                "test": {"threshold": 0.05, "actual": 0.5, "passed": False},
            },
        )
        try:
            orch.assert_gates(report)
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert "FAILED" in str(e)
