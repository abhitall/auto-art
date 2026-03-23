"""
Compliance mapping engine for adversarial robustness evaluation.

Maps auto-art evaluation results to regulatory frameworks:
- NIST AI RMF (AI 600-1 GenAI profile, AI 800-1 dual-use, IR 8596 Cyber AI)
- OWASP LLM Top 10 (2025)
- EU AI Act risk classification
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ComplianceResult:
    """Single compliance check result."""
    framework: str
    requirement_id: str
    requirement_name: str
    status: str  # "pass", "fail", "warning", "not_applicable"
    evidence: str = ""
    remediation: str = ""
    severity: str = "medium"


@dataclass
class ComplianceReport:
    """Full compliance report across all frameworks."""
    results: list[ComplianceResult] = field(default_factory=list)
    overall_score: float = 0.0
    frameworks_assessed: list[str] = field(default_factory=list)
    summary: dict[str, dict[str, int]] = field(default_factory=dict)


class NISTAIRMFMapper:
    """Maps evaluation results to NIST AI RMF functions and categories."""

    REQUIREMENTS = [
        {"id": "MEASURE-2.6", "name": "Adversarial robustness evaluation",
         "function": "MEASURE", "metric": "evasion_tested"},
        {"id": "MEASURE-2.7", "name": "Privacy risk assessment",
         "function": "MEASURE", "metric": "inference_tested"},
        {"id": "MEASURE-2.8", "name": "Data integrity verification",
         "function": "MEASURE", "metric": "poisoning_tested"},
        {"id": "MANAGE-2.2", "name": "Continuous monitoring",
         "function": "MANAGE", "metric": "monitoring_enabled"},
        {"id": "MAP-1.5", "name": "Threat model documentation",
         "function": "MAP", "metric": "threat_model_documented"},
        {"id": "GOVERN-1.4", "name": "Risk tolerance thresholds",
         "function": "GOVERN", "metric": "gates_configured"},
    ]

    def assess(self, evaluation_data: dict[str, Any]) -> list[ComplianceResult]:
        results = []
        for req in self.REQUIREMENTS:
            metric_key = req["metric"]
            status = "pass" if evaluation_data.get(metric_key, False) else "fail"
            results.append(ComplianceResult(
                framework="NIST AI RMF",
                requirement_id=req["id"],
                requirement_name=req["name"],
                status=status,
                evidence=f"Metric '{metric_key}' = {evaluation_data.get(metric_key, 'N/A')}",
            ))
        return results


class OWASPLLMMapper:
    """Maps to OWASP LLM Top 10 (2025)."""

    RISKS = [
        {"id": "LLM01", "name": "Prompt Injection", "phase": "red_team"},
        {"id": "LLM02", "name": "Insecure Output Handling", "phase": "agentic"},
        {"id": "LLM03", "name": "Training Data Poisoning", "phase": "poisoning"},
        {"id": "LLM04", "name": "Model Denial of Service", "phase": "evasion"},
        {"id": "LLM05", "name": "Supply Chain Vulnerabilities", "phase": "supply_chain"},
        {"id": "LLM06", "name": "Sensitive Information Disclosure", "phase": "inference"},
        {"id": "LLM07", "name": "Insecure Plugin Design", "phase": "agentic"},
        {"id": "LLM08", "name": "Excessive Agency", "phase": "agentic"},
        {"id": "LLM09", "name": "Overreliance", "phase": "red_team"},
        {"id": "LLM10", "name": "Model Theft", "phase": "extraction"},
    ]

    def assess(self, phases_run: list[str], phase_results: dict[str, Any]) -> list[ComplianceResult]:
        results = []
        for risk in self.RISKS:
            tested = risk["phase"] in phases_run
            passed = True
            if tested:
                phase_data = phase_results.get(risk["phase"], {})
                passed = phase_data.get("passed", True)
            results.append(ComplianceResult(
                framework="OWASP LLM Top 10",
                requirement_id=risk["id"],
                requirement_name=risk["name"],
                status="pass" if (tested and passed) else ("fail" if tested else "not_applicable"),
                evidence=f"Phase '{risk['phase']}' {'tested' if tested else 'not tested'}",
            ))
        return results


class EUAIActMapper:
    """Maps to EU AI Act risk classification requirements."""

    def classify_risk(self, model_domain: str = "general") -> str:
        high_risk_domains = {"healthcare", "finance", "law_enforcement",
                             "education", "employment", "critical_infrastructure"}
        if model_domain in high_risk_domains:
            return "high"
        return "limited"

    def assess(self, risk_level: str, evaluation_data: dict[str, Any]) -> list[ComplianceResult]:
        results = []
        if risk_level == "high":
            results.append(ComplianceResult(
                framework="EU AI Act", requirement_id="Art.15",
                requirement_name="Accuracy, robustness, and cybersecurity",
                status="pass" if evaluation_data.get("robustness_tested", False) else "fail",
                severity="high",
                remediation="High-risk AI must demonstrate robustness against adversarial attacks.",
            ))
            results.append(ComplianceResult(
                framework="EU AI Act", requirement_id="Art.9",
                requirement_name="Risk management system",
                status="pass" if evaluation_data.get("gates_configured", False) else "fail",
                severity="high",
            ))
        return results


class ComplianceEngine:
    """Unified compliance assessment engine."""

    def __init__(self):
        self.nist_mapper = NISTAIRMFMapper()
        self.owasp_mapper = OWASPLLMMapper()
        self.eu_mapper = EUAIActMapper()

    def assess_all(
        self,
        evaluation_data: dict[str, Any],
        phases_run: Optional[list[str]] = None,
        phase_results: Optional[dict[str, Any]] = None,
        model_domain: str = "general",
    ) -> ComplianceReport:
        """Run compliance assessment across all frameworks."""
        report = ComplianceReport()
        report.frameworks_assessed = ["NIST AI RMF", "OWASP LLM Top 10", "EU AI Act"]

        report.results.extend(self.nist_mapper.assess(evaluation_data))
        if phases_run and phase_results:
            report.results.extend(self.owasp_mapper.assess(phases_run, phase_results))

        risk_level = self.eu_mapper.classify_risk(model_domain)
        report.results.extend(self.eu_mapper.assess(risk_level, evaluation_data))

        for fw in report.frameworks_assessed:
            fw_results = [r for r in report.results if r.framework == fw]
            report.summary[fw] = {
                "pass": sum(1 for r in fw_results if r.status == "pass"),
                "fail": sum(1 for r in fw_results if r.status == "fail"),
                "warning": sum(1 for r in fw_results if r.status == "warning"),
                "not_applicable": sum(1 for r in fw_results if r.status == "not_applicable"),
            }

        total = sum(1 for r in report.results if r.status != "not_applicable")
        passed = sum(1 for r in report.results if r.status == "pass")
        report.overall_score = (passed / total * 100) if total > 0 else 0.0

        return report
