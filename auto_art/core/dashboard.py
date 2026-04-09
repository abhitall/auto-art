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
.container{max-width:1100px;margin:0 auto}
h1{font-size:1.6rem;margin-bottom:4px}
h2{font-size:1.15rem;margin:28px 0 10px;color:#334155;border-bottom:2px solid #e2e8f0;padding-bottom:4px}
.meta{color:#64748b;font-size:.85rem;margin-bottom:18px}
.cards{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px 20px;flex:1;min-width:140px;
      box-shadow:0 1px 3px rgba(0,0,0,.06);cursor:pointer;transition:box-shadow .2s}
.card:hover{box-shadow:0 4px 12px rgba(0,0,0,.1)}
.card .label{font-size:.75rem;text-transform:uppercase;letter-spacing:.5px;color:#64748b;margin-bottom:2px}
.card .value{font-size:1.5rem;font-weight:700}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e2e8f0;border-radius:8px;
      overflow:hidden;margin-bottom:18px}
th{background:#f1f5f9;text-align:left;padding:8px 12px;font-size:.8rem;text-transform:uppercase;
   letter-spacing:.4px;color:#475569;border-bottom:2px solid #e2e8f0;cursor:pointer;user-select:none}
th:hover{background:#e2e8f0}
th .sort-arrow{font-size:.65rem;margin-left:4px;opacity:.4}
th.sorted .sort-arrow{opacity:1}
td{padding:8px 12px;border-bottom:1px solid #f1f5f9;font-size:.88rem}
tr:last-child td{border-bottom:none}
tr.clickable-row{cursor:pointer;transition:background .15s}
tr.clickable-row:hover{background:#f0f9ff}
.pass{color:#16a34a;font-weight:600}
.fail{color:#dc2626;font-weight:600}
.warn{color:#ca8a04;font-weight:600}
.collapsible{cursor:pointer;user-select:none}
.collapsible::before{content:"\\25BC ";font-size:.7rem;display:inline-block;transition:transform .2s;margin-right:4px}
.collapsible.collapsed::before{transform:rotate(-90deg)}
.collapsible-content{overflow:hidden;transition:max-height .3s ease;max-height:2000px}
.collapsible-content.hidden{max-height:0}
.detail-panel{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;
              margin:8px 0 18px 0;display:none;animation:fadeIn .2s ease}
.detail-panel.visible{display:block}
@keyframes fadeIn{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:none}}
.filter-bar{display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap;align-items:center}
.filter-bar select,.filter-bar input{padding:6px 10px;border:1px solid #cbd5e1;border-radius:6px;
                                       font-size:.85rem;background:#fff}
.filter-bar input{width:200px}
.filter-bar label{font-size:.8rem;color:#64748b;text-transform:uppercase;letter-spacing:.3px}
.chart-container{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:20px;
                  margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.bar-chart{display:flex;align-items:flex-end;gap:6px;height:180px;padding-top:20px}
.bar-chart .bar{flex:1;min-width:20px;max-width:60px;border-radius:4px 4px 0 0;position:relative;
                transition:height .3s ease;cursor:pointer}
.bar-chart .bar:hover{opacity:.85}
.bar-chart .bar .bar-label{position:absolute;bottom:-22px;left:50%;transform:translateX(-50%);
                            font-size:.65rem;color:#64748b;white-space:nowrap;max-width:60px;
                            overflow:hidden;text-overflow:ellipsis}
.bar-chart .bar .bar-value{position:absolute;top:-18px;left:50%;transform:translateX(-50%);
                            font-size:.7rem;font-weight:600;color:#334155}
.tooltip{position:absolute;background:#1e293b;color:#fff;padding:8px 12px;border-radius:6px;
         font-size:.8rem;pointer-events:none;z-index:100;display:none;max-width:280px}
.tab-bar{display:flex;gap:4px;margin-bottom:14px;border-bottom:2px solid #e2e8f0}
.tab-bar button{padding:8px 16px;border:none;background:none;color:#64748b;font-size:.85rem;
                cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-2px;transition:all .2s}
.tab-bar button.active{color:#2563eb;border-bottom-color:#2563eb;font-weight:600}
.tab-bar button:hover{color:#334155}
.tab-content{display:none}
.tab-content.active{display:block}
"""

_JS = """
<script>
// Collapsible sections
document.querySelectorAll('.collapsible').forEach(el => {
  el.addEventListener('click', () => {
    el.classList.toggle('collapsed');
    const content = el.nextElementSibling;
    if (content && content.classList.contains('collapsible-content')) {
      content.classList.toggle('hidden');
    }
  });
});

// Table sorting
document.querySelectorAll('table[data-sortable]').forEach(table => {
  const headers = table.querySelectorAll('th[data-sort]');
  headers.forEach((th, colIdx) => {
    th.addEventListener('click', () => {
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      const isNum = th.dataset.sort === 'number';
      const asc = !th.classList.contains('sorted-asc');
      headers.forEach(h => h.classList.remove('sorted', 'sorted-asc', 'sorted-desc'));
      th.classList.add('sorted', asc ? 'sorted-asc' : 'sorted-desc');
      rows.sort((a, b) => {
        let va = a.cells[colIdx]?.textContent.trim() || '';
        let vb = b.cells[colIdx]?.textContent.trim() || '';
        if (isNum) { va = parseFloat(va) || 0; vb = parseFloat(vb) || 0; }
        if (va < vb) return asc ? -1 : 1;
        if (va > vb) return asc ? 1 : -1;
        return 0;
      });
      rows.forEach(r => tbody.appendChild(r));
    });
  });
});

// Row drill-down
document.querySelectorAll('tr.clickable-row').forEach(row => {
  row.addEventListener('click', () => {
    const panel = row.nextElementSibling;
    if (panel && panel.classList.contains('detail-row')) {
      panel.style.display = panel.style.display === 'none' ? 'table-row' : 'none';
    }
  });
});

// Filter bar for attack tables
document.querySelectorAll('.filter-bar').forEach(bar => {
  const table = bar.nextElementSibling;
  if (!table || table.tagName !== 'TABLE') return;
  const tbody = table.querySelector('tbody');
  if (!tbody) return;

  const filterInput = bar.querySelector('input[data-filter]');
  const filterSelect = bar.querySelector('select[data-filter]');

  function applyFilters() {
    const text = (filterInput?.value || '').toLowerCase();
    const category = filterSelect?.value || '';
    tbody.querySelectorAll('tr:not(.detail-row)').forEach(row => {
      const rowText = row.textContent.toLowerCase();
      const matchText = !text || rowText.includes(text);
      const matchCat = !category || rowText.includes(category.toLowerCase());
      row.style.display = (matchText && matchCat) ? '' : 'none';
      const detail = row.nextElementSibling;
      if (detail?.classList.contains('detail-row')) {
        detail.style.display = 'none';
      }
    });
  }
  if (filterInput) filterInput.addEventListener('input', applyFilters);
  if (filterSelect) filterSelect.addEventListener('change', applyFilters);
});

// Tab switching
document.querySelectorAll('.tab-bar button').forEach(btn => {
  btn.addEventListener('click', () => {
    const tabGroup = btn.closest('.tab-group');
    tabGroup.querySelectorAll('.tab-bar button').forEach(b => b.classList.remove('active'));
    tabGroup.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    const target = tabGroup.querySelector('#' + btn.dataset.tab);
    if (target) target.classList.add('active');
  });
});
</script>
"""


class DashboardGenerator:
    """Generates a standalone HTML report from an OrchestratorReport."""

    @staticmethod
    def _get_field(report: Any, key: str, default: Any) -> Any:
        """Extract a field from a report object or dict."""
        if isinstance(report, dict):
            return report.get(key, default)
        return getattr(report, key, default)

    def generate(self, report: Any) -> str:
        """Build and return a complete HTML string.

        Args:
            report: An OrchestratorReport (or any object/dict with the same fields:
                    timestamp, execution_time, passed, phases, gate_results, summary).
        """
        ts = self._get_field(report, "timestamp", 0.0)
        exec_time = self._get_field(report, "execution_time", 0.0)
        passed = self._get_field(report, "passed", True)
        phases: List[Dict[str, Any]] = self._get_field(report, "phases", [])
        gate_results: Dict[str, Any] = self._get_field(report, "gate_results", {})
        summary: Dict[str, Any] = self._get_field(report, "summary", {})

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
            n_attacks = len(p.get("results", []))
            rows.append(
                f'<tr class="clickable-row"><td>{name}</td><td class="{cls}">{status}</td>'
                f'<td>{dur}</td><td>{n_attacks} attacks</td><td>{detail}</td></tr>'
            )
            # Drill-down row (hidden by default)
            if p.get("results"):
                detail_html = DashboardGenerator._phase_detail_panel(p)
                rows.append(
                    f'<tr class="detail-row" style="display:none">'
                    f'<td colspan="5">{detail_html}</td></tr>'
                )
        return (
            '<h2 class="collapsible">Evaluation Phases</h2>'
            '<div class="collapsible-content">'
            '<table data-sortable><thead><tr>'
            '<th data-sort="text">Phase <span class="sort-arrow">&#9650;</span></th>'
            '<th data-sort="text">Status <span class="sort-arrow">&#9650;</span></th>'
            '<th data-sort="number">Duration <span class="sort-arrow">&#9650;</span></th>'
            '<th data-sort="number">Attacks <span class="sort-arrow">&#9650;</span></th>'
            '<th>Details</th></tr></thead><tbody>'
            + "\n".join(rows)
            + '</tbody></table></div>'
        )

    @staticmethod
    def _phase_detail_panel(phase: Dict[str, Any]) -> str:
        """Render inline detail panel for a phase drill-down."""
        results = phase.get("results", [])
        if not results:
            return ""
        rows: List[str] = []
        for r in results:
            atk = _esc(r.get("attack", r.get("defence", "—")))
            sr = r.get("success_rate")
            sr_str = f"{sr:.2%}" if sr is not None else "—"
            bar = _bar_svg(sr, 1.0, color="#ef4444") if sr is not None else ""
            dur = r.get("duration")
            dur_str = f"{dur:.2f}s" if dur is not None else "—"
            rows.append(f'<tr><td>{atk}</td><td>{sr_str} {bar}</td><td>{dur_str}</td></tr>')
        return (
            '<div class="detail-panel visible">'
            '<table><thead><tr><th>Attack</th><th>Success Rate</th><th>Duration</th></tr></thead><tbody>'
            + "\n".join(rows)
            + '</tbody></table></div>'
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
        # Collect all attack results for the bar chart
        all_attacks: List[Dict[str, Any]] = []
        categories: set = set()
        sections: List[str] = []

        for phase in phases:
            results = phase.get("results", [])
            for r in results:
                r["_phase"] = phase.get("name", "unknown")
                all_attacks.append(r)
                categories.add(phase.get("name", "unknown"))

        # Bar chart of success rates
        if all_attacks:
            sections.append(DashboardGenerator._success_rate_chart(all_attacks))

        # Tabbed view per phase
        if len(categories) > 1:
            sections.append('<div class="tab-group"><div class="tab-bar">')
            for i, phase in enumerate(phases):
                results = phase.get("results", [])
                if not results:
                    continue
                pname = _esc(phase.get("name", "Unknown"))
                active = " active" if i == 0 else ""
                sections.append(
                    f'<button class="{active}" data-tab="tab-{pname}">{pname}</button>'
                )
            sections.append('</div>')

        for i, phase in enumerate(phases):
            results = phase.get("results", [])
            if not results:
                continue
            phase_name = _esc(phase.get("name", "Unknown"))
            active = " active" if i == 0 else ""

            # Filter bar
            filter_bar = (
                f'<div class="filter-bar">'
                f'<label>Search</label>'
                f'<input type="text" data-filter="text" placeholder="Filter attacks...">'
                f'</div>'
            )

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
                    f'<tr class="clickable-row"><td>{atk}</td><td>{status}</td>'
                    f'<td>{sr_str} {bar}</td><td>{dur_str}</td>'
                    f'<td style="color:#94a3b8;font-size:.8rem">{err}</td></tr>'
                )
                # Detail row for drill-down
                if sr is not None:
                    detail = (
                        f'<div class="detail-panel visible">'
                        f'<strong>{atk}</strong><br>'
                        f'Success Rate: {sr_str}<br>'
                        f'Duration: {dur_str}<br>'
                        f'Phase: {phase_name}<br>'
                    )
                    if err:
                        detail += f'Error: {err}<br>'
                    detail += '</div>'
                    rows.append(
                        f'<tr class="detail-row" style="display:none"><td colspan="5">{detail}</td></tr>'
                    )

            tab_content = (
                f'<div id="tab-{phase_name}" class="tab-content{active}">'
                f'<h2 class="collapsible">{phase_name} — Details ({len(results)} attacks)</h2>'
                f'<div class="collapsible-content">'
                f'{filter_bar}'
                f'<table data-sortable><thead><tr>'
                f'<th data-sort="text">Attack <span class="sort-arrow">&#9650;</span></th>'
                f'<th data-sort="text">Status <span class="sort-arrow">&#9650;</span></th>'
                f'<th data-sort="number">Success Rate <span class="sort-arrow">&#9650;</span></th>'
                f'<th data-sort="number">Duration <span class="sort-arrow">&#9650;</span></th>'
                f'<th>Error</th></tr></thead><tbody>'
                + "\n".join(rows)
                + '</tbody></table></div></div>'
            )
            sections.append(tab_content)

        if len(categories) > 1:
            sections.append('</div>')  # close tab-group

        return "\n".join(sections)

    @staticmethod
    def _success_rate_chart(attacks: List[Dict[str, Any]]) -> str:
        """Render an inline bar chart of attack success rates."""
        bars: List[str] = []
        for r in attacks:
            sr = r.get("success_rate")
            if sr is None:
                continue
            name = r.get("attack", r.get("defence", "?"))[:12]
            height = max(4, int(sr * 160))
            color = "#16a34a" if sr < 0.05 else "#ca8a04" if sr < 0.15 else "#dc2626"
            bars.append(
                f'<div class="bar" style="height:{height}px;background:{color}" '
                f'title="{_esc(r.get("attack",""))}: {sr:.2%}">'
                f'<span class="bar-value">{sr:.0%}</span>'
                f'<span class="bar-label">{_esc(name)}</span>'
                f'</div>'
            )
        if not bars:
            return ""
        return (
            '<div class="chart-container">'
            '<h2>Attack Success Rates</h2>'
            '<div class="bar-chart">'
            + "".join(bars)
            + '</div></div>'
        )

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
            f'{_JS}\n'
            '</body>\n</html>'
        )
