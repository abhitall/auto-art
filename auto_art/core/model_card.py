"""
Model Card generator with quantified robustness metrics.

Generates model cards following the Model Cards for Model Reporting
standard (Mitchell et al. 2019), enhanced with:
- Quantified adversarial robustness metrics
- Compliance mapping results
- Attack surface coverage
- Defense recommendations

Also supports SBOM/ML-BOM generation in CycloneDX format.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelCardGenerator:
    """Generate model cards with robustness metrics and compliance mappings."""

    def generate(
        self,
        evaluation_results: Dict[str, Any],
        model_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a model card in Markdown format.

        Args:
            evaluation_results: Results from Auto-ART evaluation
            model_info: Optional model metadata (name, version, description)

        Returns:
            Markdown-formatted model card string
        """
        info = model_info or {}
        model_name = info.get("name", evaluation_results.get("model_path", "Unknown Model"))
        model_version = info.get("version", "1.0")
        framework = evaluation_results.get("framework", "Unknown")

        sections = [
            f"# Model Card: {model_name}\n",
            f"**Version:** {model_version}  ",
            f"**Framework:** {framework}  ",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
            f"**Generator:** Auto-ART v0.4.0\n",
            "---\n",
        ]

        # Model Details
        sections.append("## Model Details\n")
        sections.append(f"- **Architecture:** {info.get('architecture', 'See model documentation')}")
        sections.append(f"- **Task:** {evaluation_results.get('model_type', 'classification')}")
        sections.append(f"- **Framework:** {framework}")
        if info.get("description"):
            sections.append(f"- **Description:** {info['description']}")
        sections.append("")

        # Robustness Evaluation Results
        sections.append("## Adversarial Robustness Evaluation\n")
        summary = evaluation_results.get("summary", {})
        if summary:
            sections.append("| Metric | Value |")
            sections.append("|--------|-------|")
            for key, value in summary.items():
                if isinstance(value, float):
                    sections.append(f"| {key} | {value:.4f} |")
                else:
                    sections.append(f"| {key} | {value} |")
            sections.append("")

        # Attack Results
        attacks = evaluation_results.get("attack_results", evaluation_results.get("attacks", {}))
        if attacks:
            sections.append("### Attack Results\n")
            sections.append("| Attack | Category | Success Rate | Norm | Status |")
            sections.append("|--------|----------|-------------|------|--------|")

            attack_list = attacks if isinstance(attacks, list) else [
                {"name": k, **v} for k, v in attacks.items()
            ]
            for atk in attack_list:
                name = atk.get("name", atk.get("attack_name", "Unknown"))
                cat = atk.get("category", "evasion")
                sr = atk.get("success_rate", 0)
                norm = atk.get("norm", "N/A")
                status = "PASS" if sr < 0.05 else "FAIL"
                sections.append(f"| {name} | {cat} | {sr:.2%} | {norm} | {status} |")
            sections.append("")

        # Compliance
        compliance = evaluation_results.get("compliance", {})
        if compliance:
            sections.append("## Compliance Assessment\n")
            if isinstance(compliance, dict):
                for framework_name, fw_data in compliance.items():
                    if isinstance(fw_data, dict):
                        passed = fw_data.get("pass", 0)
                        failed = fw_data.get("fail", 0)
                        total = passed + failed
                        sections.append(f"- **{framework_name}:** {passed}/{total} requirements met")
            sections.append("")

        # Limitations and Risks
        sections.append("## Limitations and Risks\n")
        sections.append("- Robustness evaluation covers tested attack types only")
        sections.append("- Results are specific to the evaluation configuration and test data")
        sections.append("- New attack techniques may expose additional vulnerabilities")
        sections.append("- Certified robustness bounds are valid only for the specified epsilon")
        sections.append("")

        # Recommendations
        sections.append("## Recommendations\n")
        if summary.get("overall_attack_success_rate", 0) > 0.1:
            sections.append("- **HIGH PRIORITY:** Model shows significant vulnerability to adversarial attacks")
            sections.append("- Consider adversarial training (TRADES, AWP) before deployment")
        else:
            sections.append("- Model demonstrates acceptable robustness for the tested attacks")
            sections.append("- Continue monitoring with expanded attack coverage")
        sections.append("")

        return "\n".join(sections)


class SBOMGenerator:
    """Generate Software Bill of Materials in CycloneDX ML-BOM format.

    Creates a machine-readable inventory of the ML model's components,
    dependencies, and security metadata for supply chain transparency.
    """

    def generate(
        self,
        model_info: Dict[str, Any],
        evaluation_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate CycloneDX ML-BOM.

        Args:
            model_info: Model metadata (path, framework, architecture, etc.)
            evaluation_results: Optional evaluation results to include

        Returns:
            CycloneDX-formatted SBOM dictionary
        """
        now = datetime.now(timezone.utc).isoformat()

        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "serialNumber": f"urn:uuid:{self._generate_uuid()}",
            "version": 1,
            "metadata": {
                "timestamp": now,
                "tools": [{
                    "vendor": "Auto-ART",
                    "name": "auto-art",
                    "version": "0.4.0",
                }],
                "component": {
                    "type": "machine-learning-model",
                    "name": model_info.get("name", "unknown-model"),
                    "version": model_info.get("version", "1.0"),
                },
            },
            "components": [],
        }

        # Model component
        model_component = {
            "type": "machine-learning-model",
            "name": model_info.get("name", "unknown-model"),
            "version": model_info.get("version", "1.0"),
            "description": model_info.get("description", ""),
            "properties": [
                {"name": "ml:framework", "value": model_info.get("framework", "unknown")},
                {"name": "ml:task", "value": model_info.get("task", "classification")},
            ],
        }

        if model_info.get("hash"):
            model_component["hashes"] = [
                {"alg": "SHA-256", "content": model_info["hash"]}
            ]

        sbom["components"].append(model_component)

        # Framework dependencies
        framework = model_info.get("framework", "").lower()
        framework_deps = {
            "pytorch": {"name": "torch", "version": ">=2.0.0"},
            "tensorflow": {"name": "tensorflow", "version": ">=2.14.0"},
            "sklearn": {"name": "scikit-learn", "version": ">=1.2.0"},
        }
        if framework in framework_deps:
            dep = framework_deps[framework]
            sbom["components"].append({
                "type": "library",
                "name": dep["name"],
                "version": dep["version"],
            })

        # Robustness attestation
        if evaluation_results:
            vuln_data = []
            attacks = evaluation_results.get("attack_results", {})
            attack_list = attacks if isinstance(attacks, list) else [
                {"name": k, **v} for k, v in attacks.items()
            ]
            for atk in attack_list:
                sr = atk.get("success_rate", 0)
                if sr > 0.05:
                    vuln_data.append({
                        "id": f"AUTO-ART-{atk.get('name', 'unknown')}",
                        "description": f"Vulnerable to {atk.get('name', 'unknown')} "
                                       f"(success rate: {sr:.2%})",
                        "ratings": [{
                            "score": min(10.0, sr * 10),
                            "severity": "high" if sr > 0.3 else "medium",
                            "method": "auto-art-evaluation",
                        }],
                    })

            if vuln_data:
                sbom["vulnerabilities"] = vuln_data

        return sbom

    @staticmethod
    def _generate_uuid() -> str:
        import uuid
        return str(uuid.uuid4())


class PDFReportGenerator:
    """Generate PDF compliance reports.

    Uses basic HTML-to-text conversion for environments without
    PDF libraries. For full PDF support, install reportlab or weasyprint.
    """

    def generate(self, evaluation_results: Dict[str, Any], output_path: str) -> None:
        """Generate a compliance evidence package as text (PDF stub)."""
        card_gen = ModelCardGenerator()
        content = card_gen.generate(evaluation_results)

        # Try to generate actual PDF
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas

            c = canvas.Canvas(output_path, pagesize=letter)
            width, height = letter
            y = height - 50

            for line in content.split("\n"):
                if y < 50:
                    c.showPage()
                    y = height - 50

                line = line.strip()
                if line.startswith("# "):
                    c.setFont("Helvetica-Bold", 16)
                    line = line[2:]
                elif line.startswith("## "):
                    c.setFont("Helvetica-Bold", 12)
                    line = line[3:]
                elif line.startswith("### "):
                    c.setFont("Helvetica-Bold", 10)
                    line = line[4:]
                else:
                    c.setFont("Helvetica", 10)

                c.drawString(50, y, line[:100])
                y -= 15

            c.save()
            logger.info(f"PDF report generated: {output_path}")

        except ImportError:
            # Fallback: save as text with .pdf extension note
            text_path = output_path.replace(".pdf", ".txt")
            with open(text_path, "w") as f:
                f.write(content)
                f.write("\n\n---\nNote: Install reportlab for PDF output. "
                        "This is the text version of the compliance report.\n")
            logger.warning(f"reportlab not installed, saved text report to: {text_path}")
