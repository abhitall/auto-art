"""
OWASP LLM Top 10 (2025) threat coverage mapping.

Maps auto-art's attack and defence capabilities to the OWASP Top 10 for
LLM Applications 2025, enabling systematic threat coverage tracking and
compliance reporting.

Reference: https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/
"""

from typing import Any, Dict, List
from dataclasses import dataclass, field


@dataclass
class OwaspCategory:
    """A single OWASP LLM Top 10 category."""
    id: str
    name: str
    description: str
    attacks: List[str] = field(default_factory=list)
    defences: List[str] = field(default_factory=list)
    coverage: str = "none"


OWASP_LLM_TOP_10_2025: List[OwaspCategory] = [
    OwaspCategory(
        id="LLM01",
        name="Prompt Injection",
        description=(
            "Crafted inputs that manipulate a LLM's behavior, causing "
            "unintended actions, bypassing filters, or direct/indirect "
            "injection of adversarial instructions."
        ),
        attacks=[
            "InContextInjectionAttack (authority_escalation)",
            "InContextInjectionAttack (contradiction)",
            "InContextInjectionAttack (context_dilution)",
            "AdvWebDOMAttack (invisible DOM injection)",
            "RedTeamLLM (prompt_injection category)",
            "RedTeamLLM (authority_escalation category)",
            "RedTeamLLM (jailbreak category)",
        ],
        defences=[
            "InputSanitizer (SemanticNormalizer)",
            "InputRail (prompt_injection patterns)",
            "InContextDefence (attack exemplars)",
            "GuardrailPipeline",
        ],
        coverage="full",
    ),
    OwaspCategory(
        id="LLM02",
        name="Sensitive Information Disclosure",
        description=(
            "The LLM inadvertently reveals confidential data in its "
            "responses (PII, proprietary data, system prompts)."
        ),
        attacks=[
            "RedTeamLLM (data_extraction category)",
            "MembershipInferenceBlackBox",
            "AttributeInferenceBlackBox",
            "ModelInversion (MIFace)",
        ],
        defences=[
            "InputRail (data_exfiltration patterns)",
            "HighConfidenceDefence (output filtering)",
            "ClassLabelsDefence (strip confidence)",
            "ReverseSigmoidDefence",
        ],
        coverage="full",
    ),
    OwaspCategory(
        id="LLM03",
        name="Supply Chain Vulnerabilities",
        description=(
            "Risks from third-party components, poisoned training data, "
            "or compromised model artifacts."
        ),
        attacks=[
            "BackdoorAttack",
            "CleanLabelAttack",
            "FeatureCollisionAttack",
            "GradientMatchingAttack",
            "AgentPoisonRAGAttack",
        ],
        defences=[
            "ActivationDefenceWrapper (clustering-based detection)",
            "SpectralSignatureDefenceWrapper",
            "RAGPoisoningDetector",
        ],
        coverage="full",
    ),
    OwaspCategory(
        id="LLM04",
        name="Data and Model Poisoning",
        description=(
            "Manipulation of training data or fine-tuning to introduce "
            "vulnerabilities, biases, or backdoors."
        ),
        attacks=[
            "BackdoorAttack",
            "CleanLabelAttack",
            "FeatureCollisionAttack",
            "GradientMatchingAttack",
            "AgentPoisonRAGAttack (RAG store poisoning)",
        ],
        defences=[
            "ActivationDefenceWrapper",
            "SpectralSignatureDefenceWrapper",
            "RAGPoisoningDetector (cosine-similarity anomaly)",
        ],
        coverage="full",
    ),
    OwaspCategory(
        id="LLM05",
        name="Improper Output Handling",
        description=(
            "Failure to validate, sanitize, or handle LLM outputs "
            "before passing them to downstream systems."
        ),
        attacks=[
            "AdvWebDOMAttack (agent executes injected actions)",
            "RedTeamLLM (goal_hijacking category)",
        ],
        defences=[
            "ExecutionRail (policy enforcement)",
            "GuardrailPipeline (action validation)",
            "CircuitBreaker (anomaly-triggered rollback)",
        ],
        coverage="partial",
    ),
    OwaspCategory(
        id="LLM06",
        name="Excessive Agency",
        description=(
            "Granting LLM-based systems excessive permissions, "
            "capabilities, or autonomy without proper controls."
        ),
        attacks=[
            "RedTeamLLM (goal_hijacking category)",
            "InContextInjectionAttack (authority_escalation)",
        ],
        defences=[
            "ExecutionRail (allowed_tools, denied_tools)",
            "ExecutionRail (max_actions_per_turn)",
            "ExecutionRail (require_confirmation_for)",
            "CircuitBreaker (action frequency monitoring)",
        ],
        coverage="full",
    ),
    OwaspCategory(
        id="LLM07",
        name="System Prompt Leakage",
        description=(
            "Adversarial extraction of the system prompt or internal "
            "instructions through crafted queries."
        ),
        attacks=[
            "RedTeamLLM (data_extraction category)",
            "InContextInjectionAttack (authority_escalation)",
        ],
        defences=[
            "InputRail (data_exfiltration patterns)",
            "InContextDefence (system prompt protection exemplars)",
            "SemanticNormalizer (prompt injection filtering)",
        ],
        coverage="full",
    ),
    OwaspCategory(
        id="LLM08",
        name="Vector and Embedding Weaknesses",
        description=(
            "Vulnerabilities in vector/embedding storage and retrieval, "
            "including RAG poisoning and embedding manipulation."
        ),
        attacks=[
            "AgentPoisonRAGAttack (semantic/syntactic/hybrid triggers)",
        ],
        defences=[
            "RAGPoisoningDetector (centroid distance + z-score)",
            "RAGPoisoningDetector (trigger pattern detection)",
        ],
        coverage="full",
    ),
    OwaspCategory(
        id="LLM09",
        name="Misinformation",
        description=(
            "LLM generates false or misleading information that appears "
            "authoritative (hallucination, fabrication)."
        ),
        attacks=[
            "RedTeamLLM (context_poisoning category)",
            "AgentPoisonRAGAttack (fabricated responses)",
        ],
        defences=[
            "AgentTracer (silent failure detection)",
            "AgentTracer (expected_properties checks)",
            "RAGPoisoningDetector",
        ],
        coverage="partial",
    ),
    OwaspCategory(
        id="LLM10",
        name="Unbounded Consumption",
        description=(
            "LLM performs excessive resource consumption through "
            "infinite loops, denial-of-service, or token exhaustion."
        ),
        attacks=[
            "InContextInjectionAttack (context_dilution)",
            "UniversalAdversarialPatchAttack (state manipulation)",
        ],
        defences=[
            "CircuitBreaker (token_usage_spike_factor)",
            "CircuitBreaker (action_frequency_threshold)",
            "CircuitBreaker (error_rate_threshold)",
            "ExecutionRail (max_actions_per_turn)",
            "AgentTracer (infinite loop detection)",
        ],
        coverage="full",
    ),
]


def get_coverage_report() -> Dict[str, Any]:
    """Generate an OWASP LLM Top 10 coverage report."""
    categories = []
    full_count = 0
    partial_count = 0

    for cat in OWASP_LLM_TOP_10_2025:
        entry = {
            "id": cat.id,
            "name": cat.name,
            "coverage": cat.coverage,
            "attack_count": len(cat.attacks),
            "defence_count": len(cat.defences),
            "attacks": cat.attacks,
            "defences": cat.defences,
        }
        categories.append(entry)
        if cat.coverage == "full":
            full_count += 1
        elif cat.coverage == "partial":
            partial_count += 1

    return {
        "standard": "OWASP LLM Top 10 (2025)",
        "total_categories": len(OWASP_LLM_TOP_10_2025),
        "full_coverage": full_count,
        "partial_coverage": partial_count,
        "no_coverage": len(OWASP_LLM_TOP_10_2025) - full_count - partial_count,
        "coverage_percentage": (full_count + partial_count * 0.5) / len(OWASP_LLM_TOP_10_2025) * 100,
        "categories": categories,
    }


def get_coverage_markdown() -> str:
    """Generate a Markdown-formatted OWASP coverage report."""
    report = get_coverage_report()
    lines = [
        "# OWASP LLM Top 10 (2025) Coverage Report",
        "",
        f"**Coverage:** {report['coverage_percentage']:.0f}% "
        f"({report['full_coverage']} full, {report['partial_coverage']} partial, "
        f"{report['no_coverage']} none)",
        "",
        "| ID | Category | Coverage | Attacks | Defences |",
        "|------|----------|----------|---------|----------|",
    ]

    for cat in report["categories"]:
        icon = {"full": "FULL", "partial": "PARTIAL", "none": "NONE"}[cat["coverage"]]
        lines.append(
            f"| {cat['id']} | {cat['name']} | {icon} | "
            f"{cat['attack_count']} | {cat['defence_count']} |"
        )

    lines.append("")
    lines.append("## Detailed Mapping")

    for cat in report["categories"]:
        lines.append(f"\n### {cat['id']}: {cat['name']}")
        if cat["attacks"]:
            lines.append("\n**Attacks:**")
            for a in cat["attacks"]:
                lines.append(f"- {a}")
        if cat["defences"]:
            lines.append("\n**Defences:**")
            for d in cat["defences"]:
                lines.append(f"- {d}")

    return "\n".join(lines)
