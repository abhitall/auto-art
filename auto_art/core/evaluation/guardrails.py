"""
Guardrail pipeline for autonomous agent decision runtime.

Implements the "Decision Runtime" guardrail layer described in the Antigravity
security architecture (Phase 2). The core agent never directly interfaces with
user inputs or executes tool calls without mediation.

Architecture:
    User Input -> InputRail -> Agent Core -> ExecutionRail -> Tool Execution

Components:
    - InputRail: Intercepts and validates incoming prompts / environment state
    - ExecutionRail: Validates proposed agent actions against security policies
    - GuardrailPipeline: Chains rails into a unified mediation layer
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
import time


class RailDecision(Enum):
    """Decision outcome from a guardrail."""
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"
    ESCALATE = "escalate"


@dataclass
class RailResult:
    """Result from a guardrail evaluation."""
    decision: RailDecision
    original_input: Any
    processed_input: Any
    reason: str = ""
    risk_score: float = 0.0
    violations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


@dataclass
class PolicyRule:
    """A single security policy rule for the ExecutionRail."""
    rule_id: str
    description: str
    check_fn: Callable[[Dict[str, Any]], bool]
    severity: str = "high"
    category: str = "general"


class BaseRail(ABC):
    """Base class for guardrails."""

    def __init__(self, rail_name: str = "UnnamedRail"):
        self.rail_name = rail_name
        self.logger = logging.getLogger(f"auto_art.guardrail.{rail_name}")
        self._enabled = True

    @abstractmethod
    def evaluate(self, input_data: Any, **kwargs) -> RailResult:
        """Evaluate the input against this rail's policies."""
        pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False


class InputRail(BaseRail):
    """Intercepts and validates incoming prompts and environment states.

    Acts as the first line of defense, using pattern matching and heuristic
    analysis to detect adversarial payloads before they reach the agent's
    reasoning core. Analogous to LlamaGuard / ShieldGemma.
    """

    THREAT_PATTERNS = {
        "invisible_dom": {
            "patterns": [
                r'<[^>]+style=["\'][^"\']*(?:display\s*:\s*none|visibility\s*:\s*'
                r'hidden|width\s*:\s*0|height\s*:\s*0)[^"\']*["\']',
                r'<[^>]+aria-(?:label|hidden)[^>]*(?:override|execute|system|admin)[^>]*>',
            ],
            "severity": "critical",
            "description": "Invisible DOM element with adversarial content",
        },
        "prompt_injection": {
            "patterns": [
                r'(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+instructions',
                r'(?:system|admin|developer)\s*(?:override|mode|prompt)',
                r'(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)',
                r'\[\[?\s*(?:INST|SYS|SYSTEM)\s*\]?\]',
            ],
            "severity": "critical",
            "description": "Prompt injection attempt detected",
        },
        "authority_escalation": {
            "patterns": [
                r'(?:admin|root|superuser)\s+(?:access|privileges?|override)',
                r'(?:auth(?:orization)?\s+code|verification\s+token)\s*:\s*\S+',
                r'as\s+(?:the|your)\s+(?:administrator|developer|creator)',
            ],
            "severity": "high",
            "description": "Authority escalation attempt",
        },
        "data_exfiltration": {
            "patterns": [
                r'(?:show|reveal|display|output|print)\s+(?:your|the)\s+'
                r'(?:system\s+)?(?:prompt|instructions|configuration)',
                r'(?:what\s+(?:are|were)\s+your\s+(?:initial|original|system)\s+(?:instructions|prompt))',
                r'repeat\s+(?:back|everything)\s+(?:above|before\s+this)',
            ],
            "severity": "high",
            "description": "Data exfiltration attempt",
        },
    }

    def __init__(
        self,
        custom_patterns: Optional[Dict[str, Dict[str, Any]]] = None,
        block_threshold: float = 0.6,
        classifier_fn: Optional[Callable[[str], Tuple[bool, float]]] = None,
    ):
        super().__init__(rail_name="InputRail")
        self.patterns = dict(self.THREAT_PATTERNS)
        if custom_patterns:
            self.patterns.update(custom_patterns)
        self.block_threshold = block_threshold
        self.classifier_fn = classifier_fn

    def evaluate(self, input_data: Any, **kwargs) -> RailResult:
        start_time = time.time()

        if not self._enabled:
            return RailResult(
                decision=RailDecision.ALLOW,
                original_input=input_data,
                processed_input=input_data,
                reason="Rail disabled",
                processing_time=time.time() - start_time,
            )

        text = self._extract_text(input_data)
        violations: List[Dict[str, Any]] = []
        risk_score = 0.0

        for threat_name, threat_config in self.patterns.items():
            for pattern in threat_config["patterns"]:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                if matches:
                    severity = threat_config["severity"]
                    weight = 0.5 if severity == "critical" else 0.3 if severity == "high" else 0.15
                    risk_score += weight * len(matches)
                    violations.append({
                        "threat": threat_name,
                        "severity": severity,
                        "description": threat_config["description"],
                        "match_count": len(matches),
                        "samples": [m[:80] if isinstance(m, str) else str(m)[:80] for m in matches[:2]],
                    })

        if self.classifier_fn and isinstance(text, str):
            is_unsafe, cls_confidence = self.classifier_fn(text)
            if is_unsafe:
                risk_score += cls_confidence
                violations.append({
                    "threat": "classifier_detection",
                    "severity": "critical",
                    "description": "External classifier flagged input as unsafe",
                    "confidence": cls_confidence,
                })

        risk_score = min(1.0, risk_score)

        if risk_score >= self.block_threshold:
            decision = RailDecision.BLOCK
            reason = f"Input blocked: risk score {risk_score:.2f} >= threshold {self.block_threshold:.2f}"
        elif violations:
            decision = RailDecision.ESCALATE
            reason = f"Input flagged for review: {len(violations)} violations detected"
        else:
            decision = RailDecision.ALLOW
            reason = "Input passed all checks"

        return RailResult(
            decision=decision,
            original_input=input_data,
            processed_input=input_data,
            reason=reason,
            risk_score=risk_score,
            violations=violations,
            processing_time=time.time() - start_time,
        )

    @staticmethod
    def _extract_text(input_data: Any) -> str:
        if isinstance(input_data, str):
            return input_data
        if isinstance(input_data, dict):
            parts = []
            for key in ["prompt", "query", "input", "message", "content", "text", "html"]:
                if key in input_data and isinstance(input_data[key], str):
                    parts.append(input_data[key])
            return " ".join(parts)
        return str(input_data)


class ExecutionRail(BaseRail):
    """Validates proposed agent actions against programmatic security policies.

    Before the agent executes a tool call (e.g., API call, file write, database
    query), the ExecutionRail checks the proposed action against a set of
    configurable policy rules implementing the Principle of Least Privilege.
    """

    def __init__(
        self,
        policies: Optional[List[PolicyRule]] = None,
        allowed_tools: Optional[Set[str]] = None,
        denied_tools: Optional[Set[str]] = None,
        max_actions_per_turn: int = 20,
        require_confirmation_for: Optional[Set[str]] = None,
    ):
        super().__init__(rail_name="ExecutionRail")
        self.policies = policies or []
        self.allowed_tools = allowed_tools
        self.denied_tools = denied_tools or set()
        self.max_actions_per_turn = max_actions_per_turn
        self.require_confirmation_for = require_confirmation_for or {
            "delete", "drop", "truncate", "execute_sql", "send_payment",
            "modify_permissions", "create_user", "rm", "shutdown",
        }
        self._action_count = 0

    def evaluate(self, input_data: Any, **kwargs) -> RailResult:
        """Evaluate a proposed agent action.

        Args:
            input_data: Dict with keys 'tool_name', 'parameters', and
                        optionally 'reasoning'.
        """
        start_time = time.time()

        if not self._enabled:
            return RailResult(
                decision=RailDecision.ALLOW,
                original_input=input_data,
                processed_input=input_data,
                reason="Rail disabled",
                processing_time=time.time() - start_time,
            )

        if not isinstance(input_data, dict):
            return RailResult(
                decision=RailDecision.BLOCK,
                original_input=input_data,
                processed_input=input_data,
                reason="Action must be a dictionary with 'tool_name' and 'parameters'",
                risk_score=1.0,
                processing_time=time.time() - start_time,
            )

        tool_name = input_data.get("tool_name", "unknown")
        violations: List[Dict[str, Any]] = []

        self._action_count += 1
        if self._action_count > self.max_actions_per_turn:
            violations.append({
                "rule": "max_actions_exceeded",
                "severity": "high",
                "description": f"Action count {self._action_count} exceeds limit {self.max_actions_per_turn}",
            })

        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            violations.append({
                "rule": "tool_not_allowed",
                "severity": "critical",
                "description": f"Tool '{tool_name}' not in allowed tools list",
            })

        if tool_name in self.denied_tools:
            violations.append({
                "rule": "tool_denied",
                "severity": "critical",
                "description": f"Tool '{tool_name}' is explicitly denied",
            })

        for policy in self.policies:
            try:
                if not policy.check_fn(input_data):
                    violations.append({
                        "rule": policy.rule_id,
                        "severity": policy.severity,
                        "description": policy.description,
                        "category": policy.category,
                    })
            except Exception as e:
                self.logger.error(f"Policy rule {policy.rule_id} check failed: {e}")

        has_critical = any(v["severity"] == "critical" for v in violations)
        needs_confirmation = tool_name.lower() in self.require_confirmation_for

        if has_critical:
            decision = RailDecision.BLOCK
            reason = f"Action blocked: {len(violations)} policy violations (critical)"
        elif needs_confirmation:
            decision = RailDecision.ESCALATE
            reason = f"Action requires confirmation: '{tool_name}' is a sensitive operation"
        elif violations:
            decision = RailDecision.ESCALATE
            reason = f"Action flagged: {len(violations)} policy violations"
        else:
            decision = RailDecision.ALLOW
            reason = "Action permitted"

        risk_score = min(1.0, sum(
            0.5 if v["severity"] == "critical" else 0.3 if v["severity"] == "high" else 0.1
            for v in violations
        ))

        return RailResult(
            decision=decision,
            original_input=input_data,
            processed_input=input_data,
            reason=reason,
            risk_score=risk_score,
            violations=violations,
            metadata={"tool_name": tool_name, "action_count": self._action_count},
            processing_time=time.time() - start_time,
        )

    def reset_action_count(self) -> None:
        """Reset the per-turn action counter."""
        self._action_count = 0

    def add_policy(self, rule: PolicyRule) -> None:
        """Add a policy rule to the execution rail."""
        self.policies.append(rule)


class GuardrailPipeline:
    """Unified guardrail pipeline chaining InputRail and ExecutionRail.

    Provides a single interface for the agent's decision runtime to
    validate both inputs and proposed actions.
    """

    def __init__(
        self,
        input_rail: Optional[InputRail] = None,
        execution_rail: Optional[ExecutionRail] = None,
    ):
        self.logger = logging.getLogger("auto_art.guardrail_pipeline")
        self.input_rail = input_rail or InputRail()
        self.execution_rail = execution_rail or ExecutionRail()
        self._audit_log: List[Dict[str, Any]] = []

    def validate_input(self, input_data: Any, **kwargs) -> RailResult:
        """Validate an incoming input through the InputRail."""
        result = self.input_rail.evaluate(input_data, **kwargs)
        self._log_audit("input", result)
        return result

    def validate_action(self, action: Dict[str, Any], **kwargs) -> RailResult:
        """Validate a proposed agent action through the ExecutionRail."""
        result = self.execution_rail.evaluate(action, **kwargs)
        self._log_audit("action", result)
        return result

    def process_with_guardrails(
        self,
        input_data: Any,
        agent_fn: Callable[[Any], Any],
        action_validator: Optional[Callable[[Any], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Full guardrail-mediated processing pipeline.

        Args:
            input_data: Raw input to process.
            agent_fn: The agent's core processing function.
            action_validator: Optional function to extract action dict from
                              agent output for ExecutionRail validation.

        Returns:
            Dict with 'output', 'input_rail_result', and optionally
            'execution_rail_result'.
        """
        input_result = self.validate_input(input_data)

        if input_result.decision == RailDecision.BLOCK:
            return {
                "output": None,
                "blocked": True,
                "input_rail_result": input_result,
                "reason": input_result.reason,
            }

        agent_output = agent_fn(input_result.processed_input)

        execution_result = None
        if action_validator is not None:
            action = action_validator(agent_output)
            if action:
                execution_result = self.validate_action(action)
                if execution_result.decision == RailDecision.BLOCK:
                    return {
                        "output": None,
                        "blocked": True,
                        "input_rail_result": input_result,
                        "execution_rail_result": execution_result,
                        "reason": execution_result.reason,
                    }

        return {
            "output": agent_output,
            "blocked": False,
            "input_rail_result": input_result,
            "execution_rail_result": execution_result,
        }

    def _log_audit(self, rail_type: str, result: RailResult) -> None:
        entry = {
            "timestamp": time.time(),
            "rail_type": rail_type,
            "decision": result.decision.value,
            "risk_score": result.risk_score,
            "reason": result.reason,
            "violations_count": len(result.violations),
        }
        self._audit_log.append(entry)
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        return self._audit_log[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics from the audit log."""
        if not self._audit_log:
            return {"total": 0}

        total = len(self._audit_log)
        blocked = sum(1 for e in self._audit_log if e["decision"] == "block")
        escalated = sum(1 for e in self._audit_log if e["decision"] == "escalate")

        return {
            "total": total,
            "blocked": blocked,
            "escalated": escalated,
            "allowed": total - blocked - escalated,
            "block_rate": blocked / total if total > 0 else 0.0,
            "avg_risk_score": sum(e["risk_score"] for e in self._audit_log) / total,
        }
