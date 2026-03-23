"""
HTML dashboard for adversarial robustness evaluation reports.

Generates standalone HTML reports with charts, tables, and
interactive elements for reviewing evaluation results.
"""

from typing import Any, Dict, List
from pathlib import Path
import html
import time
import logging

logger = logging.getLogger(__name__)


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _bar_svg(value: float, max_val: float = 1.0, width: int = 120, height: int = 18,
             color: str = "#3b82f6") -> str:
    """Inline SVG bar proportional to value/max_val."""
    pct = min(max(value / max_val, 0), 1.0) if max_val else 0
    fill_w = int(pct * width)
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{width}" height="{height}" rx="3" fill="#e5e7eb"/>'
        f'<rect width="{fill_w}" height="{height}" rx="3" fill="{color}"/>'
        f'</svg>'
    )


def _status_badge(passed: bool) -> str:
    if passed:
        return '<span style="background:#16a34a;color:#fff;padding:2px 10px;border-radius:12px;font-weight:600;font-size:13px;">PASSED</span>'
    return '<span style="background:#dc2626;color:#fff;padding:2px 10px;border-radius:12px;font-weight:600;font-size:13px;">FAILED</span>'


_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
     background:#f8fafc;color:#1e293b;line-height:1.6;padding:24px}
.container{max-width:960px;margin:0 auto}
h1{font-size:1.6rem;margin-bottom:4px}
h2{font-size:1.15rem;margin:28px 0 10px;color:#334155;border-bottom:2px solid #e2e8f0;padding-bottom:4px}
.meta{color:#64748b;font-size:.85rem;margin-bottom:18px}
.cards{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px 20px;flex:1;min-width:140px;
      box-shadow:0 1px 3px rgba(0,0,0,.06)}
.card .label{font-size:.75rem;text-transform:uppercase;letter-spacing:.5px;color:#64748b;margin-bottom:2px}
.card .value{font-size:1.5rem;font-weight:700}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e2e8f0;border-radius:8px;
      overflow:hidden;margin-bottom:18px}
th{background:#f1f5f9;text-align:left;padding:8px 12px;font-size:.8rem;text-transform:uppercase;
   letter-spacing:.4px;color:#475569;border-bottom:2px solid #e2e8f0}
td{padding:8px 12px;border-bottom:1px solid #f1f5f9;font-size:.88rem}
tr:last-child td{border-bottom:none}
.pass{color:#16a34a;font-weight:600}
.fail{color:#dc2626;font-weight:600}
.warn{color:#ca8a04;font-weight:600}
"""


class DashboardGenerator:
    """Generates a standalone HTML report from an OrchestratorReport."""

    def generate(self, report: Any) -> str:
        """Build and return a complete HTML string.

        Args:
            report: An OrchestratorReport (or any object with the same fields:
                    timestamp, execution_time, passed, phases, gate_results, summary).
        """
        ts = getattr(report, "timestamp", 0.0)
        exec_time = getattr(report, "execution_time", 0.0)
        passed = getattr(report, "passed", True)
        phases: List[Dict[str, Any]] = getattr(report, "phases", [])
        gate_results: Dict[str, Any] = getattr(report, "gate_results", {})
        summary: Dict[str, Any] = getattr(report, "summary", {})

        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "N/A"

        total_phases = summary.get("total_phases", len(phases))
        passed_phases = summary.get("passed_phases", sum(1 for p in phases if p.get("passed", True)))
        failed_phases = total_phases - passed_phases

        parts: List[str] = []
        parts.append(self._header(ts_str, passed, exec_time))
        parts.append(self._summary_cards(total_phases, passed_phases, failed_phases, exec_time))
        if phases:
            parts.append(self._phase_table(phases))
        if gate_results:
            parts.append(self._gate_table(gate_results))
        parts.append(self._attack_details(phases))

        body = "\n".join(parts)
        return self._wrap_html(body)

    def save(self, report: Any, output_path: str) -> None:
        """Generate HTML and write to disk."""
        html_content = self.generate(report)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html_content, encoding="utf-8")
        logger.info("Dashboard saved to %s", path)

    # ------------------------------------------------------------------ parts

    @staticmethod
    def _header(ts_str: str, passed: bool, exec_time: float) -> str:
        badge = _status_badge(passed)
        return (
            f'<h1>Auto-ART Evaluation Dashboard {badge}</h1>'
            f'<p class="meta">Generated {_esc(ts_str)} &middot; '
            f'Execution time: {exec_time:.2f}s</p>'
        )

    @staticmethod
    def _summary_cards(total: int, passed: int, failed: int, exec_time: float) -> str:
        return (
            '<div class="cards">'
            f'<div class="card"><div class="label">Total Phases</div><div class="value">{total}</div></div>'
            f'<div class="card"><div class="label">Passed</div><div class="value" style="color:#16a34a">{passed}</div></div>'
            f'<div class="card"><div class="label">Failed</div><div class="value" style="color:#dc2626">{failed}</div></div>'
            f'<div class="card"><div class="label">Exec Time</div><div class="value">{exec_time:.1f}s</div></div>'
            '</div>'
        )

    @staticmethod
    def _phase_table(phases: List[Dict[str, Any]]) -> str:
        rows: List[str] = []
        for p in phases:
            name = _esc(p.get("name", "—"))
            ok = p.get("passed", True)
            cls = "pass" if ok else "fail"
            status = "PASS" if ok else "FAIL"
            dur = f'{p.get("duration", 0):.2f}s'
            detail = _esc(str(p.get("summary", "")))
            rows.append(
                f'<tr><td>{name}</td><td class="{cls}">{status}</td>'
                f'<td>{dur}</td><td>{detail}</td></tr>'
            )
        return (
            '<h2>Evaluation Phases</h2>'
            '<table><thead><tr><th>Phase</th><th>Status</th><th>Duration</th>'
            '<th>Details</th></tr></thead><tbody>'
            + "\n".join(rows)
            + '</tbody></table>'
        )

    @staticmethod
    def _gate_table(gate_results: Dict[str, Any]) -> str:
        rows: List[str] = []
        for gate_name, gate_data in gate_results.items():
            name = _esc(gate_name)
            threshold = gate_data.get("threshold", "N/A")
            actual = gate_data.get("actual", "N/A")
            ok = gate_data.get("passed", True)

            th_str = f"{threshold:.4f}" if isinstance(threshold, float) else str(threshold)
            ac_str = f"{actual:.4f}" if isinstance(actual, float) else str(actual)

            cls = "pass" if ok else "fail"
            status = "PASS" if ok else "FAIL"

            bar_color = "#16a34a" if ok else "#dc2626"
            max_v = max(float(threshold), float(actual), 0.001) if isinstance(threshold, (int, float)) and isinstance(actual, (int, float)) else 1.0
            bar_th = _bar_svg(float(threshold) if isinstance(threshold, (int, float)) else 0, max_v, color="#94a3b8")
            bar_ac = _bar_svg(float(actual) if isinstance(actual, (int, float)) else 0, max_v, color=bar_color)

            rows.append(
                f'<tr><td>{name}</td><td>{_esc(th_str)} {bar_th}</td>'
                f'<td>{_esc(ac_str)} {bar_ac}</td><td class="{cls}">{status}</td></tr>'
            )
        return (
            '<h2>Gate Results</h2>'
            '<table><thead><tr><th>Gate</th><th>Threshold</th>'
            '<th>Actual</th><th>Result</th></tr></thead><tbody>'
            + "\n".join(rows)
            + '</tbody></table>'
        )

    @staticmethod
    def _attack_details(phases: List[Dict[str, Any]]) -> str:
        sections: List[str] = []
        for phase in phases:
            results = phase.get("results", [])
            if not results:
                continue
            phase_name = _esc(phase.get("name", "Unknown"))
            rows: List[str] = []
            for r in results:
                atk = _esc(r.get("attack", r.get("defence", "—")))
                status = _esc(r.get("status", "—"))
                sr = r.get("success_rate")
                sr_str = f"{sr:.2%}" if sr is not None else "—"
                bar = _bar_svg(sr, 1.0, color="#ef4444") if sr is not None else ""
                err = _esc(r.get("error", ""))
                dur = r.get("duration")
                dur_str = f"{dur:.2f}s" if dur is not None else "—"
                rows.append(
                    f'<tr><td>{atk}</td><td>{status}</td>'
                    f'<td>{sr_str} {bar}</td><td>{dur_str}</td>'
                    f'<td style="color:#94a3b8;font-size:.8rem">{err}</td></tr>'
                )
            sections.append(
                f'<h2>{phase_name} — Details</h2>'
                '<table><thead><tr><th>Attack/Defence</th><th>Status</th>'
                '<th>Success Rate</th><th>Duration</th><th>Error</th></tr></thead><tbody>'
                + "\n".join(rows)
                + '</tbody></table>'
            )
        return "\n".join(sections)

    @staticmethod
    def _wrap_html(body: str) -> str:
        return (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            '<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
            '<title>Auto-ART Dashboard</title>\n'
            f'<style>{_CSS}</style>\n'
            '</head>\n<body>\n'
            f'<div class="container">\n{body}\n</div>\n'
            '</body>\n</html>'
        )
