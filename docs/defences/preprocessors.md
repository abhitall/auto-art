# Advanced preprocessors

Orchestrator `defences.preprocessor` names:

| `name` | Description |
|--------|-------------|
| `label_smoothing` | Label smoothing |
| `thermometer_encoding` | Thermometer encoding |
| `total_variance_minimization` / `tvm` | TV minimization |
| `video_compression` | Video codec purification |
| `mp3_compression` | MP3 compression (audio) |

Plus standard preprocessors: `spatial_smoothing`, `feature_squeezing`, `jpeg_compression`, `gaussian_augmentation`, augmentation family (`cutout`, `mixup`, `cutmix`). Implementations in `preprocessor.py`, `preprocessor_augmentation.py`, `preprocessor_advanced.py`.
