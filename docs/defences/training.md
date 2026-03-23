# Adversarial training defences

YAML `defences.trainer` (and related sections) can reference:

| `name` | Class |
|--------|--------|
| `adversarial_training_pgd` | PGD adversarial training |
| `fast_is_better_than_free` | FBF |
| `oaat` | OAAT |
| `certified_at` / `ibp` | Certified / IBP trainers |
| `trades` / `adversarial_training_trades` | TRADES |
| `awp` / `adversarial_training_awp` | AWP |

Implementations: `auto_art.core.evaluation.defences.trainer*`. See `docs/defences/art-wrappers.md`.
