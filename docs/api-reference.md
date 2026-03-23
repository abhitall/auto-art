# API reference

Public entry points for orchestration, attacks, evaluation, metrics, configuration factories, adaptive selection, parallelism, benchmarks, dashboards, and warm-start caching.

---

## `auto_art.core.orchestrator`

### `GateConfig` (dataclass)

Pass/fail and warn thresholds for CI gating.

**Fields (defaults)**

| Field | Default | Role |
|--------|---------|------|
| `max_attack_success_rate` | `0.05` | Fail if evasion/agentic max success exceeds this |
| `warn_attack_success_rate` | `0.02` | Warning band (lower is better) |
| `min_security_score` | `70.0` | Fail if defence security score below this |
| `warn_security_score` | `80.0` | Warning band (higher is better) |
| `max_jailbreak_rate` | `0.05` | Red-team jailbreak rate fail threshold |
| `warn_jailbreak_rate` | `0.02` | Warning band |
| `max_membership_leakage` | `0.8` | Inference membership leakage fail |
| `warn_membership_leakage` | `0.6` | Warning band |
| `max_poisoning_detection_rate` | `0.15` | Poisoning phase fail |
| `warn_poisoning_detection_rate` | `0.08` | Warning band |
| `attack_budget_seconds` | `3600.0` | Documented budget (orchestrator phases use evaluation settings primarily) |

### `OrchestratorConfig` (dataclass)

| Field | Default | Description |
|--------|---------|-------------|
| `target` | `{}` | e.g. `model_path`, `framework` |
| `evaluation` | `{}` | e.g. `num_samples`, `batch_size` |
| `attacks` | `{}` | Keys: `evasion`, `poisoning`, `extraction`, `inference`, `agentic`; `red_team` is extracted from YAML under `attacks` |
| `defences` | `{}` | Categories → list of `{name, ...}` |
| `gates` | `GateConfig()` | Thresholds |
| `red_team` | `{}` | Red-team options (also from `attacks.red_team` in YAML) |

### `OrchestratorReport` (dataclass)

| Field | Description |
|--------|-------------|
| `timestamp`, `execution_time`, `target`, `phases`, `gate_results`, `passed`, `summary` | Structured run output |

**Methods**

- `to_dict()` — JSON-serializable dict  
- `to_json(indent=2)` — JSON string  
- `to_markdown()` — Markdown report  
- `to_sarif()` — SARIF 2.1.0 for CI scanners  

### `Orchestrator`

```python
Orchestrator(config: OrchestratorConfig)
```

**Class methods**

- `from_yaml(yaml_path: str)` — Parse YAML; requires PyYAML  
- `from_dict(config_dict: dict)` — Parse dict  

**Methods**

- `run(agent=None)` — Run configured phases; `agent` required for agentic + red-team  
- `assert_gates(report)` — Raises `AssertionError` if any gate failed  

---

## `auto_art.core.attacks.attack_generator`

### `AttackGenerator`

```python
AttackGenerator()
```

**Attributes**

- `supported_attacks` — `dict` mapping category (`classification`, `poisoning`, …) to allowed `attack_type` strings  

**Methods**

- `create_attack(model, metadata, config)` — Build ART attack or wrapper from `AttackConfig`  
- `apply_attack(attack_instance, test_inputs, test_labels=None, batch_size_override=None)` — `generate()` for evasion-style attacks; raises for extraction/poisoning/inference wrappers  

---

## `auto_art.core.evaluation.art_evaluator`

### `ARTEvaluator`

```python
ARTEvaluator(
    model_obj: Any,
    config: EvaluationConfig,
    observers: Optional[List[EvaluationObserver]] = None,
    model_metadata: Optional[ModelMetadata] = None,
)
```

**Methods**

- `notify_observers(event_type, data)`  
- `evaluate_model(test_data, test_labels, attacks=None, defences=None)` → `EvaluationResult`  
- `generate_report(result)` — Text report  
- `add_observer` / `remove_observer`  
- `evaluate_robustness_from_path(model_path, framework, num_samples=100)` — Load model, default attack suite  
- `calculate_metrics(...)` — Intended bridge for metrics (uses internal calculator when available)  

**Property**

- `art_estimator` — Lazy `ClassifierFactory` wrapper  

**Related**

- `EvasionAttackStrategy(attack_class, params, attack_name)` — `execute(classifier, data, labels)` → name, adv examples, success rate  
- `EvaluationObserver(logger)` — Logs `update(event_type, data)`  

---

## `auto_art.core.evaluation.metrics.calculator`

### `MetricsCalculator`

```python
MetricsCalculator(cache_size: int = 128)
```

**Methods**

- `calculate_basic_metrics(classifier, data, labels)` — `accuracy`, `average_confidence`  
- `calculate_robustness_metrics(classifier, data, labels, num_samples=5)` — empirical robustness, loss sensitivity, CLEVER, optional tree verification  
- `calculate_security_score(base_accuracy, attack_results, robustness_metrics)` — Weighted 0–100 score  
- `calculate_great_score(...)` — Optional GREAT score (ART version dependent)  
- `calculate_wasserstein_distance(data_batch_1, data_batch_2)` — Optional; needs SciPy/ART support  

---

## `auto_art.core.evaluation.config.evaluation_config`

### `ModelType` (enum)

`CLASSIFICATION`, `OBJECT_DETECTION`, `GENERATION`, `REGRESSION`, `LLM`

### `Framework` (enum)

`PYTORCH`, `TENSORFLOW`, `KERAS`, `SKLEARN`, `XGBOOST`, `LIGHTGBM`, `CATBOOST`, `GPY`, `TRANSFORMERS`

### `EvaluationConfig` (frozen dataclass)

Required: `model_type`, `framework`. See [configuration.md](configuration.md) for full defaults.

### `EvaluationBuilder`

```python
EvaluationBuilder()
```

Fluent setters: `with_model_type`, `with_framework`, `with_batch_size`, `with_num_workers`, `with_cache`, `with_timeout`, `with_metrics_to_calculate`, `with_device_preference`, `with_input_shape`, `with_nb_classes`, `with_loss_function`, `with_num_samples_for_adv_metrics`, `with_metrics_overall_list`, `with_attack_params` → `build()` returns `EvaluationConfig`.

### `EvaluationResult`

`success`, `metrics_data`, `errors`, `execution_time`

---

## `auto_art.core.evaluation.factories.classifier_factory`

### `ClassifierFactory`

Static methods:

- `create_classifier(model, model_type, framework, device_type=None, **kwargs)` — ART estimator; kwargs include `input_shape`, `nb_classes`, `loss_fn`, etc.  
- `_determine_actual_device(requested_device)` — Resolves `cpu`/`gpu` using `ConfigManager` and backends  

---

## `auto_art.core.evaluation.defences.base`

### `DefenceStrategy` (ABC)

```python
DefenceStrategy(defence_name: str = "UnnamedDefence")
```

- `name` — Identifier  
- `apply(art_estimator, **kwargs)` → defended estimator  
- `get_params()` → dict  
- `set_params(**params)` — Default no-op  

---

## `auto_art.core.adaptive`

### `AttackMemory`

```python
AttackMemory(memory_path: Optional[str] = None)  # default ~/.auto_art/attack_memory.json
```

- `record_result(attack_name, success_rate, cost_seconds, model_arch="unknown")`  
- `get_priority_score(attack_name, model_arch="unknown")`  
- `save()` — Persist JSON  

### `AdaptiveAttackSelector`

```python
AdaptiveAttackSelector(
    memory: Optional[AttackMemory] = None,
    budget_seconds: float = 3600.0,
    success_threshold: float = 0.5,
    escalate_if_below: float = 0.1,
)
```

- `select_attacks(model_arch="unknown", available_attacks=None, max_attacks=10)` — Ordered attack dicts with `name`, `tier`, `priority`, `estimated_cost`  
- `should_escalate(current_results)` — `True` if max success rate &lt; `escalate_if_below`  
- `update_memory(...)` — Record + `save()`  

---

## `auto_art.core.parallel`

### `AttackTask` / `AttackResult` (dataclasses)

`AttackTask`: `name`, `callable`, `args`, `kwargs`, `use_gpu`, `timeout`

### `ParallelAttackRunner`

```python
ParallelAttackRunner(
    max_cpu_workers: int = 4,
    max_gpu_workers: int = 2,
    default_timeout: float = 300.0,
)
```

- `run(tasks: List[AttackTask])` → `List[AttackResult]` — GPU tasks use `ThreadPoolExecutor`, CPU tasks `ProcessPoolExecutor`  

---

## `auto_art.core.harmbench`

### `HarmBenchLoader`

```python
HarmBenchLoader(behaviors_path: Optional[str] = None)
```

- `load_behaviors(categories=None, max_per_category=None)` → `List[HarmBenchBehavior]`  

### `HarmBenchEvaluator`

```python
HarmBenchEvaluator(default_judge: Optional[Callable[[str, str], bool]] = None)
```

- `evaluate(agent, behaviors, judge_fn=None)` → `HarmBenchReport` (`asr`, `per_category_asr`, `behaviors_tested`, …)  

---

## `auto_art.core.robustbench`

### `RobustBenchClient`

```python
RobustBenchClient(timeout: float = 10.0)
```

- `get_leaderboard(dataset, threat_model)` → `List[LeaderboardEntry]` — Remote JSON with cache fallback  
- `compare_model(model_name, clean_acc, robust_acc, dataset, threat_model)` → `RobustBenchComparison` (`rank`, `percentile`, …)  

---

## `auto_art.core.dashboard`

### `DashboardGenerator`

- `generate(report)` → HTML `str` (expects orchestrator-like report: `timestamp`, `execution_time`, `passed`, `phases`, `gate_results`, `summary`)  
- `save(report, output_path)` — Write UTF-8 file  

---

## `auto_art.core.warmstart`

### `EvaluationCache`

```python
EvaluationCache(cache_dir: Optional[str] = None)  # default .auto_art/cache
```

- `save(model_hash, attack_results)`  
- `load(model_hash)`  
- `get_model_hash(model_path)` — SHA-256 of file bytes  
- `is_stale(model_hash, max_age_hours=24.0)`  

### `WarmStartEvaluator`

```python
WarmStartEvaluator(cache: Optional[EvaluationCache] = None)
```

- `run(orchestrator, agent=None, force_rerun=False)` — Cache hit injects phases into `OrchestratorReport`  

### `ModelDiffAnalyzer`

- `compare(report_a, report_b)` → `ModelDiffReport` (`improved_attacks`, `regressed_attacks`, …) — Compares `success_rate` per attack across phases  

---

## Import examples

```python
from auto_art.core.orchestrator import Orchestrator, OrchestratorConfig, GateConfig
from auto_art.core.attacks.attack_generator import AttackGenerator
from auto_art.core.evaluation.art_evaluator import ARTEvaluator
from auto_art.core.evaluation.metrics.calculator import MetricsCalculator
from auto_art.core.evaluation.config.evaluation_config import (
    EvaluationConfig, EvaluationBuilder, Framework, ModelType,
)
from auto_art.core.evaluation.factories.classifier_factory import ClassifierFactory
from auto_art.core.evaluation.defences.base import DefenceStrategy
from auto_art.core.adaptive import AdaptiveAttackSelector, AttackMemory
from auto_art.core.parallel import ParallelAttackRunner, AttackTask
from auto_art.core.harmbench import HarmBenchLoader, HarmBenchEvaluator
from auto_art.core.robustbench import RobustBenchClient, RobustBenchDataset, RobustBenchThreatModel
from auto_art.core.dashboard import DashboardGenerator
from auto_art.core.warmstart import EvaluationCache, WarmStartEvaluator, ModelDiffAnalyzer
```
