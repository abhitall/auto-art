"""
Auto-ART CLI entry point.

Provides subcommands:
  auto-art evaluate  — Run adversarial robustness evaluation
  auto-art scan      — Quick vulnerability scan
  auto-art report    — Generate reports from evaluation results
  auto-art certify   — Run formal verification / certification
  auto-art config    — Manage configuration
  auto-art attacks   — List / search available attacks
  auto-art defenses  — List / search available defenses
  auto-art plugins   — Manage plugins

Design: Click-based CLI with rich console output for progress
visualization. Inspired by Giskard `giskard scan`, ModelScan `modelscan scan`,
and Counterfit interactive CLI patterns.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


@click.group()
@click.version_option(package_name="auto-art")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """Auto-ART: Automated Adversarial Robustness Testing Framework.

    Comprehensive adversarial robustness evaluation for ML models
    and autonomous AI agents.
    """
    _setup_logging(verbose)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output directory for results")
@click.option("--format", "output_format", type=click.Choice(
    ["json", "markdown", "sarif", "html", "all"]), default="json",
    help="Report output format")
@click.option("--preset", type=str, default=None,
              help="Attack preset (quick_scan, standard, comprehensive)")
@click.option("--max-workers", type=int, default=None,
              help="Maximum parallel workers")
@click.option("--timeout", type=int, default=3600,
              help="Global timeout in seconds")
@click.option("--dry-run", is_flag=True, help="Show plan without executing")
def evaluate(
    config_path: str,
    output: Optional[str],
    output_format: str,
    preset: Optional[str],
    max_workers: Optional[int],
    timeout: int,
    dry_run: bool,
) -> None:
    """Run adversarial robustness evaluation from a YAML/JSON config.

    Example:
        auto-art evaluate configs/eval_pytorch.yaml -o results/ --format sarif
    """
    import yaml

    click.echo(f"Loading evaluation config: {config_path}")

    config_file = Path(config_path)
    if config_file.suffix in (".yaml", ".yml"):
        with open(config_file) as f:
            config = yaml.safe_load(f)
    else:
        with open(config_file) as f:
            config = json.load(f)

    if output:
        config["output_dir"] = output
    if output_format != "json":
        config["report_format"] = output_format
    if preset:
        config["attack_preset"] = preset
    if max_workers:
        config["max_workers"] = max_workers

    if dry_run:
        click.echo("\n--- Dry Run: Evaluation Plan ---")
        click.echo(json.dumps(config, indent=2, default=str))
        return

    click.echo("Starting evaluation...")
    start_time = time.time()

    try:
        from auto_art.core.orchestrator import EvaluationOrchestrator

        orchestrator = EvaluationOrchestrator(config)
        results = orchestrator.run()

        elapsed = time.time() - start_time
        click.echo(f"\nEvaluation complete in {elapsed:.1f}s")

        if isinstance(results, dict):
            summary = results.get("summary", {})
            attacks_run = summary.get("attacks_executed", "N/A")
            success_rate = summary.get("overall_attack_success_rate", "N/A")
            click.echo(f"  Attacks executed: {attacks_run}")
            click.echo(f"  Overall attack success rate: {success_rate}")

        if output:
            click.echo(f"  Results saved to: {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Evaluation failed")
        sys.exit(1)


# ---------------------------------------------------------------------------
# scan (quick vulnerability scan)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--framework", type=click.Choice(
    ["pytorch", "tensorflow", "keras", "sklearn", "xgboost",
     "lightgbm", "catboost", "onnx"]),
    default=None, help="ML framework (auto-detected if not specified)")
@click.option("--preset", type=click.Choice(
    ["quick_scan", "standard", "comprehensive"]),
    default="quick_scan", help="Attack preset to use")
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--format", "output_format", type=click.Choice(
    ["json", "sarif", "markdown"]), default="json")
@click.option("--samples", type=int, default=100,
              help="Number of test samples")
def scan(
    model_path: str,
    framework: Optional[str],
    preset: str,
    output: Optional[str],
    output_format: str,
    samples: int,
) -> None:
    """Quick adversarial vulnerability scan on a model file.

    Example:
        auto-art scan model.pt --preset standard --format sarif
    """
    click.echo(f"Scanning model: {model_path}")
    click.echo(f"  Preset: {preset}")
    click.echo(f"  Samples: {samples}")

    start_time = time.time()

    try:
        from auto_art.core.registry import get_attack_registry

        registry = get_attack_registry()
        attack_names = registry.get_preset(preset)
        click.echo(f"  Attacks: {', '.join(attack_names)}")

        config = {
            "model_path": model_path,
            "framework": framework,
            "num_samples": samples,
            "attacks": [{"attack_type": name} for name in attack_names],
            "report_format": output_format,
        }
        if output:
            config["output_dir"] = output

        from auto_art.core.orchestrator import EvaluationOrchestrator

        orchestrator = EvaluationOrchestrator(config)
        results = orchestrator.run()

        elapsed = time.time() - start_time
        click.echo(f"\nScan complete in {elapsed:.1f}s")

        if output:
            click.echo(f"Results saved to: {output}")
        else:
            click.echo(json.dumps(results, indent=2, default=str))

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Scan failed")
        sys.exit(1)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.option("--format", "output_format", type=click.Choice(
    ["json", "markdown", "sarif", "html", "pdf"]), default="html")
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--compliance", multiple=True,
              type=click.Choice(["nist", "owasp", "eu_ai_act", "iso42001",
                                 "mitre_atlas", "etsi"]),
              help="Include compliance mapping")
@click.option("--model-card", is_flag=True,
              help="Generate model card with robustness metrics")
def report(
    results_path: str,
    output_format: str,
    output: Optional[str],
    compliance: tuple,
    model_card: bool,
) -> None:
    """Generate reports from evaluation results.

    Example:
        auto-art report results.json --format html --compliance nist owasp
    """
    click.echo(f"Generating {output_format} report from: {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    if compliance:
        click.echo(f"  Compliance frameworks: {', '.join(compliance)}")
        results["compliance_frameworks"] = list(compliance)

    try:
        from auto_art.core.dashboard import DashboardGenerator
        from auto_art.core.compliance import ComplianceEngine

        if compliance:
            import dataclasses
            engine = ComplianceEngine()
            compliance_results = engine.assess(results, frameworks=list(compliance))
            results["compliance"] = dataclasses.asdict(compliance_results)

        if output_format == "html":
            generator = DashboardGenerator()
            html_content = generator.generate(results)
            out_path = output or "report.html"
            Path(out_path).write_text(html_content)
            click.echo(f"HTML report saved to: {out_path}")
        elif output_format == "pdf":
            click.echo("Generating PDF compliance report...")
            from auto_art.core.model_card import PDFReportGenerator
            generator = PDFReportGenerator()
            out_path = output or "report.pdf"
            generator.generate(results, out_path)
            click.echo(f"PDF report saved to: {out_path}")
        else:
            out_path = output or f"report.{output_format}"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            click.echo(f"Report saved to: {out_path}")

        if model_card:
            from auto_art.core.model_card import ModelCardGenerator
            card_gen = ModelCardGenerator()
            card = card_gen.generate(results)
            card_path = output.replace(f".{output_format}", "_model_card.md") if output else "model_card.md"
            Path(card_path).write_text(card)
            click.echo(f"Model card saved to: {card_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Report generation failed")
        sys.exit(1)


# ---------------------------------------------------------------------------
# certify
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--method", type=click.Choice(
    ["randomized_smoothing", "alpha_crown", "great_score", "ibp"]),
    default="randomized_smoothing", help="Certification method")
@click.option("--epsilon", type=float, default=0.3,
              help="Certification radius")
@click.option("--samples", type=int, default=1000,
              help="Number of samples for statistical certification")
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--confidence", type=float, default=0.99,
              help="Confidence level for statistical certification")
def certify(
    model_path: str,
    method: str,
    epsilon: float,
    samples: int,
    output: Optional[str],
    confidence: float,
) -> None:
    """Run formal verification / robustness certification.

    Example:
        auto-art certify model.pt --method randomized_smoothing --epsilon 0.5
    """
    click.echo(f"Certifying model: {model_path}")
    click.echo(f"  Method: {method}")
    click.echo(f"  Epsilon: {epsilon}")
    click.echo(f"  Confidence: {confidence}")

    start_time = time.time()

    try:
        from auto_art.core.evaluation.metrics.certification import CertificationEvaluator

        evaluator = CertificationEvaluator()
        results = evaluator.certify(
            model_path=model_path,
            method=method,
            epsilon=epsilon,
            num_samples=samples,
            confidence=confidence,
        )

        elapsed = time.time() - start_time
        click.echo(f"\nCertification complete in {elapsed:.1f}s")

        certified_pct = results.get("certified_accuracy", 0) * 100
        click.echo(f"  Certified accuracy at eps={epsilon}: {certified_pct:.1f}%")

        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            click.echo(f"Results saved to: {output}")
        else:
            click.echo(json.dumps(results, indent=2, default=str))

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Certification failed")
        sys.exit(1)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("action", type=click.Choice(["show", "init", "validate"]))
@click.option("--path", type=click.Path(), default=None,
              help="Config file path")
def config(action: str, path: Optional[str]) -> None:
    """Manage Auto-ART configuration.

    Example:
        auto-art config show
        auto-art config init --path ./auto_art_config.yaml
        auto-art config validate --path ./eval_config.yaml
    """
    from auto_art.config.manager import ConfigManager

    if action == "show":
        mgr = ConfigManager()
        click.echo(json.dumps(mgr.get_all(), indent=2, default=str))

    elif action == "init":
        out_path = path or "auto_art_config.yaml"
        template = {
            "version": "1.0",
            "model": {
                "path": "path/to/model",
                "framework": "pytorch",
            },
            "evaluation": {
                "num_samples": 100,
                "attack_preset": "standard",
                "timeout": 3600,
            },
            "reporting": {
                "format": "sarif",
                "compliance": ["nist", "owasp"],
            },
        }
        import yaml
        Path(out_path).write_text(yaml.dump(template, default_flow_style=False))
        click.echo(f"Config template created: {out_path}")

    elif action == "validate":
        if not path:
            click.echo("Error: --path is required for validation", err=True)
            sys.exit(1)
        try:
            import yaml
            with open(path) as f:
                cfg = yaml.safe_load(f)
            click.echo(f"Config valid: {path}")
            click.echo(f"  Keys: {', '.join(cfg.keys())}")
        except Exception as e:
            click.echo(f"Config invalid: {e}", err=True)
            sys.exit(1)


# ---------------------------------------------------------------------------
# attacks (list/search)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--category", type=click.Choice(
    ["evasion", "poisoning", "extraction", "inference", "llm", "nlp",
     "audio", "agentic"]), default=None)
@click.option("--threat-model", type=click.Choice(
    ["white_box", "black_box", "grey_box"]), default=None)
@click.option("--search", type=str, default=None, help="Search query")
@click.option("--preset", type=str, default=None, help="Show preset attacks")
@click.option("--json-output", is_flag=True, help="Output as JSON")
def attacks(
    category: Optional[str],
    threat_model: Optional[str],
    search: Optional[str],
    preset: Optional[str],
    json_output: bool,
) -> None:
    """List and search available attacks.

    Example:
        auto-art attacks --category evasion --threat-model black_box
        auto-art attacks --search gradient
        auto-art attacks --preset quick_scan
    """
    from auto_art.core.registry import (
        AttackCategory as AC,
        ThreatModel as TM,
        get_attack_registry,
    )

    registry = get_attack_registry()

    if preset:
        names = registry.get_preset(preset)
    elif search:
        names = registry.search(search)
    else:
        cat = AC(category) if category else None
        tm = TM(threat_model) if threat_model else None
        names = registry.filter(category=cat, threat_model=tm)

    if json_output:
        entries = []
        for name in names:
            meta = registry.get_metadata(name)
            entries.append({
                "name": meta.name,
                "display_name": meta.display_name,
                "category": meta.category.value,
                "threat_model": meta.threat_model.value,
                "norm": meta.norm_type.value,
                "cost": meta.cost_estimate.name,
                "gpu": meta.requires_gpu,
                "description": meta.description,
            })
        click.echo(json.dumps(entries, indent=2))
    else:
        click.echo(f"Found {len(names)} attacks:\n")
        for name in names:
            meta = registry.get_metadata(name)
            gpu = " [GPU]" if meta.requires_gpu else ""
            click.echo(
                f"  {meta.display_name:<35} "
                f"{meta.category.value:<12} "
                f"{meta.threat_model.value:<12} "
                f"{meta.norm_type.value:<10} "
                f"{meta.cost_estimate.name:<10}"
                f"{gpu}"
            )


# ---------------------------------------------------------------------------
# defenses (list/search)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--type", "defense_type", type=click.Choice(
    ["preprocessor", "postprocessor", "trainer", "detector",
     "certified", "input_sanitizer", "guardrail"]), default=None)
@click.option("--compatible-with", type=str, default=None,
              help="Find defenses effective against this attack")
@click.option("--search", type=str, default=None)
@click.option("--json-output", is_flag=True)
def defenses(
    defense_type: Optional[str],
    compatible_with: Optional[str],
    search: Optional[str],
    json_output: bool,
) -> None:
    """List and search available defenses.

    Example:
        auto-art defenses --type trainer
        auto-art defenses --compatible-with pgd
    """
    from auto_art.core.registry import DefenseType as DT, get_defense_registry

    registry = get_defense_registry()

    if search:
        names = registry.search(search)
    else:
        dt = DT(defense_type) if defense_type else None
        names = registry.filter(defense_type=dt, compatible_with=compatible_with)

    if json_output:
        entries = []
        for name in names:
            meta = registry.get_metadata(name)
            entries.append({
                "name": meta.name,
                "display_name": meta.display_name,
                "type": meta.defense_type.value,
                "cost": meta.cost_estimate.name,
                "requires_training": meta.requires_training,
                "description": meta.description,
            })
        click.echo(json.dumps(entries, indent=2))
    else:
        click.echo(f"Found {len(names)} defenses:\n")
        for name in names:
            meta = registry.get_metadata(name)
            train = " [Train]" if meta.requires_training else ""
            click.echo(
                f"  {meta.display_name:<40} "
                f"{meta.defense_type.value:<18} "
                f"{meta.cost_estimate.name:<10}"
                f"{train}"
            )


# ---------------------------------------------------------------------------
# plugins
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("action", type=click.Choice(["list", "discover"]))
def plugins(action: str) -> None:
    """Manage Auto-ART plugins.

    Example:
        auto-art plugins discover
        auto-art plugins list
    """
    from auto_art.core.plugins import get_plugin_manager

    pm = get_plugin_manager()

    if action == "discover":
        discovered = pm.discover_plugins()
        total = sum(len(v) for v in discovered.values())
        click.echo(f"Discovered {total} plugins:")
        for group, names in discovered.items():
            if names:
                click.echo(f"\n  {group}:")
                for name in names:
                    click.echo(f"    - {name}")
        if total == 0:
            click.echo("  No plugins found. Install third-party packages with auto_art entry points.")

    elif action == "list":
        loaded = pm.loaded_plugin_names
        if loaded:
            click.echo(f"Loaded plugins ({len(loaded)}):")
            for name in loaded:
                click.echo(f"  - {name}")
        else:
            click.echo("No plugins loaded. Run 'auto-art plugins discover' first.")


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
