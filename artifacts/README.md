# Artifacts

Runtime data, model weights, caches, and experiment outputs live here.

Layout:

- `datasets/` for rollouts, WebDataset shards, and caption-pair datasets
- `models/` for local model weights such as Qwen3-VL and CLIP fine-tunes
- `world_models/` for saved World Models checkpoints and derived caches
- `experiments/` for fusion reconstruction runs and similar experiment outputs
- `visualizations/` for rendered videos and preview assets
- `cache/` for Hugging Face caches and disposable virtualenvs
- `logs/` for Slurm and local run logs

Set `AGILE_WM_ARTIFACTS_ROOT` to relocate this whole tree outside the repo, for example onto cluster scratch storage.
