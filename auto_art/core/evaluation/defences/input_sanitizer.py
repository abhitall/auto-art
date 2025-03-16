"""
Input sanitization pipeline for autonomous AI agents.

Implements a multi-layer sanitization pipeline that strips hidden HTML metadata,
applies visual denoisers, and normalizes inputs to neutralize AdvWeb-style
adversarial payloads before they reach the agent's decision-making core.

Architecture:
    Raw Input -> DOMSanitizer -> VisualDenoiser -> SemanticNormalizer -> Clean Input
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import re
import numpy as np


@dataclass
class SanitizationResult:
    """Result of input sanitization."""
    original_input: Any
    sanitized_input: Any
    threats_detected: List[Dict[str, Any]] = field(default_factory=list)
    sanitization_steps: List[str] = field(default_factory=list)
    was_modified: bool = False
    risk_score: float = 0.0


class SanitizationLayer(ABC):
    """Base class for individual sanitization layers."""

    def __init__(self, layer_name: str = "UnnamedLayer"):
        self.layer_name = layer_name
        self.logger = logging.getLogger(f"auto_art.sanitizer.{layer_name}")

    @abstractmethod
    def sanitize(self, input_data: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        """Apply sanitization to the input.

        Returns:
            Tuple of (sanitized_data, list of detected threats).
        """
        pass


class DOMSanitizer(SanitizationLayer):
    """Strips hidden and adversarial HTML/DOM elements from web content.

    Targets AdvWeb-style invisible injection vectors including:
    - Hidden aria-label adversarial text
    - Zero-pixel images with adversarial alt text
    - Hidden form fields with instruction injection
    - Invisible overlay divs
    - Suspicious meta tags
    """

    DANGEROUS_PATTERNS = [
        (r'<[^>]*aria-label=["\'][^"\']*(?:ignore|override|execute|system|'
         r'admin|directive)[^"\']*["\'][^>]*>',
         "adversarial_aria_label"),
        (r'<[^>]*style=["\'][^"\']*(?:display\s*:\s*none|visibility\s*:\s*'
         r'hidden|width\s*:\s*0|height\s*:\s*0|clip\s*:\s*rect\(0)'
         r'[^"\']*["\'][^>]*>.*?</[^>]+>',
         "hidden_element"),
        (r'<input[^>]*type=["\']hidden["\'][^>]*(?:instruction|directive|'
         r'override|command)[^>]*/?\s*>',
         "hidden_input_injection"),
        (r'<meta[^>]*name=["\'](?:agent-instruction|system-override|'
         r'ai-directive)["\'][^>]*/?\s*>',
         "adversarial_meta_tag"),
        (r'<img[^>]*style=["\'][^"\']*display\s*:\s*none[^"\']*["\']'
         r'[^>]*/?\s*>',
         "hidden_image"),
        (r'<[^>]*style=["\'][^"\']*(?:opacity\s*:\s*0(?:\.0+)?|'
         r'font-size\s*:\s*0|color\s*:\s*transparent)[^"\']*["\'][^>]*>',
         "near_invisible_element"),
        (r'<div[^>]*class=["\'][^"\']*(?:popup-overlay|system-notification|'
         r'agent-override)[^"\']*["\'][^>]*>',
         "suspicious_overlay"),
    ]

    ALLOWLISTED_TAGS = {
        "p", "h1", "h2", "h3", "h4", "h5", "h6", "a", "ul", "ol", "li",
        "table", "tr", "td", "th", "thead", "tbody", "strong", "em", "br",
        "span", "div", "img", "form", "input", "button", "label", "select",
        "option", "textarea",
    }

    def __init__(
        self,
        strip_hidden: bool = True,
        strip_scripts: bool = True,
        strip_styles: bool = False,
        custom_patterns: Optional[List[Tuple[str, str]]] = None,
    ):
        super().__init__(layer_name="DOMSanitizer")
        self.strip_hidden = strip_hidden
        self.strip_scripts = strip_scripts
        self.strip_styles = strip_styles
        self.patterns = self.DANGEROUS_PATTERNS.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)

    def sanitize(self, input_data: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        threats: List[Dict[str, Any]] = []

        if isinstance(input_data, dict):
            result = input_data.copy()
            if "html" in result:
                result["html"], html_threats = self._sanitize_html(result["html"])
                threats.extend(html_threats)
            return result, threats

        if isinstance(input_data, str):
            return self._sanitize_html(input_data)

        return input_data, threats

    def _sanitize_html(self, html: str) -> Tuple[str, List[Dict[str, Any]]]:
        threats: List[Dict[str, Any]] = []
        sanitized = html

        if self.strip_scripts:
            script_pattern = r'<script[^>]*>.*?</script>'
            matches = re.findall(script_pattern, sanitized, re.DOTALL | re.IGNORECASE)
            if matches:
                threats.append({
                    "type": "script_tag",
                    "count": len(matches),
                    "severity": "high",
                })
                sanitized = re.sub(script_pattern, "", sanitized, flags=re.DOTALL | re.IGNORECASE)

        if self.strip_styles:
            style_pattern = r'<style[^>]*>.*?</style>'
            matches = re.findall(style_pattern, sanitized, re.DOTALL | re.IGNORECASE)
            if matches:
                sanitized = re.sub(style_pattern, "", sanitized, flags=re.DOTALL | re.IGNORECASE)

        for pattern, threat_type in self.patterns:
            matches = re.findall(pattern, sanitized, re.DOTALL | re.IGNORECASE)
            if matches:
                threats.append({
                    "type": threat_type,
                    "count": len(matches),
                    "severity": "critical" if "adversarial" in threat_type else "high",
                    "samples": [m[:100] if isinstance(m, str) else str(m)[:100] for m in matches[:3]],
                })
                sanitized = re.sub(pattern, "", sanitized, flags=re.DOTALL | re.IGNORECASE)

        return sanitized, threats


class VisualDenoiser(SanitizationLayer):
    """Applies denoising to visual inputs to neutralize adversarial perturbations.

    Uses spatial smoothing, JPEG compression artifacts removal, and median
    filtering to strip imperceptible adversarial noise from image frames.
    """

    def __init__(
        self,
        spatial_smoothing_window: int = 3,
        bit_depth_reduction: int = 4,
        jpeg_quality: int = 75,
    ):
        super().__init__(layer_name="VisualDenoiser")
        self.spatial_smoothing_window = spatial_smoothing_window
        self.bit_depth_reduction = bit_depth_reduction
        self.jpeg_quality = jpeg_quality

    def sanitize(self, input_data: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        threats: List[Dict[str, Any]] = []

        if not isinstance(input_data, np.ndarray):
            return input_data, threats

        if input_data.ndim < 2:
            return input_data, threats

        denoised = input_data.copy().astype(np.float32)

        perturbation_score = self._estimate_perturbation(denoised)
        if perturbation_score > 0.1:
            threats.append({
                "type": "high_frequency_perturbation",
                "severity": "medium",
                "perturbation_score": float(perturbation_score),
            })

        denoised = self._spatial_smoothing(denoised)
        denoised = self._bit_depth_reduce(denoised)

        return denoised.astype(input_data.dtype), threats

    def _spatial_smoothing(self, image: np.ndarray) -> np.ndarray:
        """Apply uniform spatial smoothing via convolution."""
        w = self.spatial_smoothing_window
        if w <= 1:
            return image
        kernel = np.ones((w, w)) / (w * w)
        if image.ndim == 3:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = self._convolve2d(image[:, :, c], kernel)
            return result
        return self._convolve2d(image, kernel)

    @staticmethod
    def _convolve2d(image_2d: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution (no scipy dependency)."""
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(image_2d, ((ph, ph), (pw, pw)), mode="edge")
        out = np.zeros_like(image_2d)
        for i in range(image_2d.shape[0]):
            for j in range(image_2d.shape[1]):
                out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
        return out

    def _bit_depth_reduce(self, image: np.ndarray) -> np.ndarray:
        """Reduce bit depth to strip low-magnitude adversarial noise."""
        levels = 2 ** self.bit_depth_reduction
        if image.max() > 1.0:
            return np.round(image / (256 / levels)) * (256 / levels)
        return np.round(image * levels) / levels

    @staticmethod
    def _estimate_perturbation(image: np.ndarray) -> float:
        """Estimate the high-frequency perturbation level in an image."""
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        dx = np.diff(gray, axis=1)
        dy = np.diff(gray, axis=0)
        hf_energy = np.mean(np.abs(dx)) + np.mean(np.abs(dy))
        return float(hf_energy)


class SemanticNormalizer(SanitizationLayer):
    """Normalizes text inputs by detecting and neutralizing prompt injections.

    Strips known prompt injection patterns, authority escalation phrases,
    and instruction override attempts from text-based inputs.
    """

    INJECTION_PATTERNS = [
        r"(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|directives|rules)",
        r"(?:system|admin|developer)\s*(?:override|mode|prompt|access)",
        r"(?:new|updated|revised)\s+(?:primary\s+)?(?:directive|instruction|objective)\s*:",
        r"EXECUTE_OVERRIDE_\w+",
        r"PRIORITY_DIRECTIVE_\w+",
        r"SYS_UPDATE_\w+",
        r"(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(?:a\s+)?(?:different|new|unrestricted)",
        r"(?:restrictions?\s+(?:are\s+)?(?:lifted|removed|disabled))",
    ]

    def __init__(
        self,
        custom_patterns: Optional[List[str]] = None,
        replacement: str = "[FILTERED]",
        max_input_length: int = 10000,
    ):
        super().__init__(layer_name="SemanticNormalizer")
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        if custom_patterns:
            self.patterns.extend([re.compile(p, re.IGNORECASE) for p in custom_patterns])
        self.replacement = replacement
        self.max_input_length = max_input_length

    def sanitize(self, input_data: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        threats: List[Dict[str, Any]] = []

        if isinstance(input_data, str):
            return self._sanitize_text(input_data, threats)

        if isinstance(input_data, dict):
            result = input_data.copy()
            for key in ["prompt", "query", "input", "message", "content", "text"]:
                if key in result and isinstance(result[key], str):
                    result[key], _ = self._sanitize_text(result[key], threats)
            return result, threats

        return input_data, threats

    def _sanitize_text(
        self, text: str, threats: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if len(text) > self.max_input_length:
            threats.append({
                "type": "context_overflow",
                "severity": "medium",
                "original_length": len(text),
                "max_length": self.max_input_length,
            })
            text = text[:self.max_input_length]

        sanitized = text
        for pattern in self.patterns:
            matches = pattern.findall(sanitized)
            if matches:
                threats.append({
                    "type": "prompt_injection",
                    "severity": "critical",
                    "pattern": pattern.pattern[:80],
                    "count": len(matches),
                    "samples": [m[:50] if isinstance(m, str) else str(m)[:50] for m in matches[:3]],
                })
                sanitized = pattern.sub(self.replacement, sanitized)

        return sanitized, threats


class InputSanitizer:
    """Multi-layer input sanitization pipeline for autonomous agents.

    Chains DOMSanitizer, VisualDenoiser, and SemanticNormalizer into a
    unified pipeline. Each input is routed to the appropriate sanitization
    layers based on its type.
    """

    def __init__(
        self,
        enable_dom: bool = True,
        enable_visual: bool = True,
        enable_semantic: bool = True,
        risk_threshold: float = 0.5,
        custom_layers: Optional[List[SanitizationLayer]] = None,
    ):
        self.logger = logging.getLogger("auto_art.input_sanitizer")
        self.risk_threshold = risk_threshold

        self.layers: List[SanitizationLayer] = []
        if enable_dom:
            self.layers.append(DOMSanitizer())
        if enable_visual:
            self.layers.append(VisualDenoiser())
        if enable_semantic:
            self.layers.append(SemanticNormalizer())
        if custom_layers:
            self.layers.extend(custom_layers)

    def sanitize(self, input_data: Any) -> SanitizationResult:
        """Run the full sanitization pipeline on the input.

        Args:
            input_data: Raw input to sanitize (HTML string, image array,
                        text prompt, or dict containing multiple modalities).

        Returns:
            SanitizationResult with sanitized input and threat report.
        """
        current = input_data
        all_threats: List[Dict[str, Any]] = []
        steps: List[str] = []
        modified = False

        for layer in self.layers:
            try:
                sanitized, threats = layer.sanitize(current)
                if threats:
                    all_threats.extend(threats)
                    modified = True
                if sanitized is not current:
                    modified = True
                    current = sanitized
                steps.append(f"{layer.layer_name}: {len(threats)} threats")
            except Exception as e:
                self.logger.error(f"Sanitization layer {layer.layer_name} failed: {e}")
                steps.append(f"{layer.layer_name}: ERROR - {e}")

        risk_score = self._calculate_risk_score(all_threats)

        return SanitizationResult(
            original_input=input_data,
            sanitized_input=current,
            threats_detected=all_threats,
            sanitization_steps=steps,
            was_modified=modified,
            risk_score=risk_score,
        )

    def should_block(self, result: SanitizationResult) -> bool:
        """Determine whether the input should be blocked based on risk score."""
        return result.risk_score >= self.risk_threshold

    @staticmethod
    def _calculate_risk_score(threats: List[Dict[str, Any]]) -> float:
        """Calculate an aggregate risk score from detected threats."""
        if not threats:
            return 0.0

        severity_weights = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1,
        }
        total = 0.0
        for threat in threats:
            severity = threat.get("severity", "low")
            count = threat.get("count", 1)
            weight = severity_weights.get(severity, 0.1)
            total += weight * count

        return min(1.0, total / max(len(threats), 1))
