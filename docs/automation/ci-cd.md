# CI/CD Integration

Auto-ART's GitHub Actions pipeline runs a 5-phase evaluation on every push/PR.

## Pipeline Phases

```
Build → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
         Static    Defence    Dynamic    Full      Deploy
         Attacks   Tests      Agentic    Suite     Gate
```

| Phase | Tests | Failure Behavior |
|-------|-------|-----------------|
| **1. Static Attack Tests** | Evasion, extraction, inference, poisoning wrappers | Continue on error (legacy tests) |
| **2. Agentic Defence Tests** | Sanitizer, RAG detector, in-context, circuit breaker, preprocessor, postprocessor, trainer, detector | **Fails build** |
| **3. Dynamic Agentic Tests** | Agentic attacks, red teaming, guardrails, telemetry, black-box, certification | **Fails build** |
| **4. Full Test Suite** | All tests combined | **Fails build** |
| **5. Deployment Gate** | Automated resilience check -- runs DOM and RAG attacks against a test agent | **Fails build if > 5% success rate** |

## Adding to Your Project

### Using the Built-In Workflow

The workflow at `.github/workflows/auto_art_ci.yml` runs automatically on push/PR to `main`, `master`, or `develop`.

### Using the Orchestrator in CI

For custom pipelines, use the YAML orchestrator:

```yaml
# .github/workflows/security-eval.yml
name: Security Evaluation
on: [push]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    - run: pip install . pyyaml
    - name: Run Evaluation
      run: |
        python -c "
        from auto_art.core.orchestrator import Orchestrator
        orch = Orchestrator.from_yaml('eval_config.yaml')
        report = orch.run()
        with open('report.md', 'w') as f:
            f.write(report.to_markdown())
        orch.assert_gates(report)
        "
    - uses: actions/upload-artifact@v4
      with:
        name: security-report
        path: report.md
```

## Deployment Gate Details

The deployment gate runs real attacks against a test agent:

```python
from auto_art.core.evaluation.attacks.agentic import AdvWebDOMAttack, AgentPoisonRAGAttack

# DOM injection: must be < 5% success rate
dom = AdvWebDOMAttack(max_injections=3)
dom_result = dom.execute_agentic(agent=test_agent, environment=test_env)
assert dom_result.success_rate < 0.05

# RAG poisoning: must be < 5% success rate
rag = AgentPoisonRAGAttack(num_poison_entries=3)
rag_result = rag.execute_agentic(agent=test_agent, environment=test_env)
assert rag_result.success_rate < 0.05
```

If either assertion fails, the pipeline blocks deployment and reports the exact attack that succeeded.

## SARIF Integration

You can emit **SARIF 2.1.0** from the orchestrator and upload it to **GitHub Code Scanning** (GitLab SAST and Azure DevOps SARIF ingestion follow the same pattern: write the file in CI, attach it as an artifact, and register it with your platform’s scanner). The report’s `to_sarif()` maps phase results to SARIF rules and results so findings show up alongside other security tools.

1. Run the orchestrated evaluation and write `results.sarif` to the workspace.
2. Upload the file as a workflow artifact (optional but useful for debugging).
3. Call `github/codeql-action/upload-sarif` with `security-events: write` on the job.

Example GitHub Actions snippet:

```yaml
- name: Run Security Evaluation
  run: |
    python -c "
    from auto_art.core.orchestrator import Orchestrator
    orch = Orchestrator.from_yaml('eval_config.yaml')
    report = orch.run()
    with open('results.sarif', 'w') as f:
        f.write(report.to_sarif())
    orch.assert_gates(report)
    "
- uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: results.sarif
```

Ensure the job sets `permissions: security-events: write` (and for private repos, typically `contents: read` and `actions: read`) so the upload step can publish results. See GitHub’s [upload SARIF](https://docs.github.com/en/code-security/code-scanning/integrating-with-code-scanning/uploading-a-sarif-file-to-github) documentation for `category`, matrix uploads, and compatibility notes.
