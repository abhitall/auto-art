# Configuration reference

## `EvaluationConfig` (`auto_art.core.evaluation.config.evaluation_config`)

Immutable dataclass. **Required fields:** `model_type: ModelType`, `framework: Framework`.

| Field | Default | Description |
|--------|---------|-------------|
| `batch_size` | `32` | Default batch size for evaluation / attacks in many flows |
| `num_workers` | `4` | Thread pool workers for parallel attack evaluation in `ARTEvaluator` |
| `use_cache` | `True` | Reserved / builder default |
| `timeout` | `300.0` | Builder default for long operations (seconds) |
| `metrics_to_calculate` | `["empirical_robustness"]` | Drives optional robustness metrics in `evaluate_model` |
| `device_preference` | `None` | Passed to classifier factory (`cpu` / `gpu` / `auto` style) |
| `input_shape` | `None` | `(C, H, W)` etc., no batch dim |
| `nb_classes` | `None` | Output classes for classification |
| `loss_function` | `None` | PyTorch/ART loss instance when needed |
| `num_samples_for_adv_metrics` | `5` | Subset size for CLEVER etc. |
| `metrics` | `["accuracy", "robustness"]` | High-level report categories |
| `attack_params` | See below | Defaults for `evaluate_robustness_from_path` → `AttackConfig` |

**Default `attack_params` dict**

| Key | Default |
|-----|---------|
| `epsilon` | `0.031` |
| `eps_step` | `0.007` |
| `max_iter` | `100` |
| `targeted` | `False` |
| `num_random_init` | `0` |
| `batch_size` | `32` |
| `norm` | `"inf"` |
| `confidence` | `0.0` |
| `learning_rate` | `0.01` |
| `binary_search_steps` | `9` |
| `initial_const` | `0.01` |
| `delta` | `0.01` |
| `step_adapt` | `0.667` |

**Note:** `ARTEvaluator._evaluate_attacks_in_parallel` reads `timeout_per_attack` via `getattr(config, "timeout_per_attack", 300.0)` and `num_workers` from `config.num_workers`. To align timeouts, either set a dynamic attribute `timeout_per_attack` on the config instance or rely on the default `300.0`.

---

## `AttackConfig` (`auto_art.core.interfaces`)

Used by `AttackGenerator` and the orchestrator when building attacks.

| Field | Default | Description |
|--------|---------|-------------|
| `attack_type` | *(required)* | e.g. `fgsm`, `pgd`, `membership_inference_bb` |
| `epsilon` | `0.3` | L∞ / generic perturbation budget |
| `eps_step` | `0.01` | Iterative step |
| `max_iter` | `100` | Iteration cap |
| `targeted` | `False` | Targeted evasion |
| `num_random_init` | `0` | PGD random restarts |
| `batch_size` | `32` | Minibatch |
| `norm` | `"inf"` | Norm for PGD-style attacks |
| `confidence` | `0.0` | C&W–style |
| `learning_rate` | `0.01` | Several attacks |
| `binary_search_steps` | `9` | C&W |
| `initial_const` | `0.01` | C&W |
| `delta` | `0.01` | Boundary-style |
| `step_adapt` | `0.667` | Boundary-style |
| `additional_params` | `{}` | Attack-specific keys (see attack docs) |

Orchestrator YAML maps common keys per attack: `name`, `eps`, `eps_step`, `max_iter`, `targeted`, `batch_size`, `norm`, plus extra keys into `additional_params`.

---

## Orchestrator YAML schema

Top-level keys are merged into `OrchestratorConfig` by `Orchestrator._parse_config`.

### `target`

| Field | Type | Description |
|--------|------|-------------|
| `model_path` | string | **Required** for phases that load a model |
| `framework` | string | Default `"pytorch"` |

### `evaluation`

| Field | Type | Default (in code) | Description |
|--------|------|-------------------|-------------|
| `num_samples` | int | `200` | Test sample count |
| `batch_size` | int | `32` | Batch for attacks / data |

### `attacks`

Each sub-key is optional. Values are lists of attack dicts unless noted.

**`evasion`** — list of:

| Key | Description |
|-----|-------------|
| `name` | Attack type string (e.g. `fgsm`, `pgd`, `square_attack`) |
| `eps`, `eps_step`, `max_iter`, `targeted`, `batch_size`, `norm` | Passed to `AttackConfig` |
| *(other keys)* | Routed to `additional_params` |

**`poisoning`** — list of `{name, ...}`; orchestrator supports `backdoor`, `clean_label`, `gradient_matching` with `poisoning_rate` in params.

**`extraction`** — list of `{name, ...}`; e.g. `copycat_cnn`, `knockoff_nets`.

**`inference`** — list of `{name, ...}`; e.g. `membership_inference_bb`, `attribute_inference_bb` (needs `attack_feature_index` in extras).

**`agentic`** — list of:

| `name` | Role |
|--------|------|
| `advweb_dom` | `stealth_level`, `max_injections`, `target_action`, `environment` |
| `rag_poison` | `trigger_type`, `injection_rate`, `environment` |

**`red_team`** — Either a dict or single-element list merged into `OrchestratorConfig.red_team`:

| Key | Description |
|-----|-------------|
| `num_rounds` | Default `50` |
| `jailbreak_threshold` | Overrides gate default for red-team pass/fail |

### `defences`

Structure: category name → list of defence specs:

```yaml
defences:
  preprocessor:
    - name: spatial_smoothing
      window_size: 3
  postprocessor:
    - name: rounding
      decimals: 4
```

Supported `name` values match `orchestrator._run_defence_phase` (e.g. `spatial_smoothing`, `feature_squeezing`, `jpeg_compression`, `adversarial_training_pgd`, `beyond`, …).

### `gates`

Maps to `GateConfig`. Any omitted key keeps the dataclass default.

```yaml
gates:
  max_attack_success_rate: 0.05
  warn_attack_success_rate: 0.02
  min_security_score: 70.0
  warn_security_score: 80.0
  max_jailbreak_rate: 0.05
  warn_jailbreak_rate: 0.02
  max_membership_leakage: 0.8
  warn_membership_leakage: 0.6
  max_poisoning_detection_rate: 0.15
  warn_poisoning_detection_rate: 0.08
  attack_budget_seconds: 3600.0
```

---

## `GateConfig` and progressive gating

The orchestrator’s `_evaluate_gates` uses **two thresholds** per metric:

1. **Warn threshold** — Stricter “yellow” band; `passed` stays `True` but `level` is `warning`.
2. **Fail threshold** — `passed` is `False` and `level` is `fail`.

**Lower is better** (success rate, jailbreak rate, leakage, poisoning rate):  
- `actual > max_*` → fail  
- `actual > warn_*` (but ≤ max) → pass with `level: warning`  
- else → `level: pass`

**Higher is better** (`min_security_score` / defence score):  
- `actual < min_security_score` → fail  
- `actual < warn_security_score` (but ≥ min) → warning  
- else → pass

Each gate entry in `OrchestratorReport.gate_results` includes `threshold`, `warn_threshold`, `actual`, `passed`, and `level`.

Overall `report.passed` is `True` only if every gate has `passed: True` (warnings still count as passed).

---

## Environment variables and config file locations

### Framework JSON config (`ConfigManager`)

- **Filename:** `auto_art_config.json`
- **Search order:** `./auto_art_config.json`, then `~/.auto_art/auto_art_config.json`
- If missing, defaults come from `FrameworkConfig` (`use_gpu`, `default_device`, `default_batch_size`, `timeout`, `cache_dir`, …)

Relevant fields for classifiers and devices:

- `use_gpu` — When `default_device` is `auto`, GPU is used only if `True` and a backend exposes a GPU.
- `default_device` — `"auto"`, `"cpu"`, or `"gpu"`.

### Other paths (code defaults)

| Component | Path / env |
|-----------|------------|
| `AttackMemory` | `~/.auto_art/attack_memory.json` unless `memory_path` is set |
| `EvaluationCache` | `.auto_art/cache` under cwd unless `cache_dir` is set |
| Orchestrator YAML | Any path passed to `Orchestrator.from_yaml` |

There is **no** required environment variable for core evaluation; optional tooling may add its own (not defined in the core package).

---

## Quick YAML skeleton

```yaml
target:
  model_path: path/to/model.pt
  framework: pytorch

evaluation:
  num_samples: 200
  batch_size: 32

attacks:
  evasion:
    - name: fgsm
      eps: 0.3
  red_team:
    num_rounds: 50
    jailbreak_threshold: 0.05

defences:
  preprocessor:
    - name: spatial_smoothing
      window_size: 3

gates:
  max_attack_success_rate: 0.05
  min_security_score: 70.0
```
