# AGILE-WM

Aligning Generative and dIscriminative Latent rEpresenations in World Models (AGILE-WM) is a research codebase for combining World Models-style control with vision-language data generation and representation learning. It supports an end-to-end workflow that starts with environment rollouts, converts trajectories into frame datasets, generates natural-language captions for sampled frames, and fine-tunes CLIP on the resulting image-text pairs.

The project is built for experiments in environments such as CarRacing and Doom, and is designed to make control data easier to inspect, enrich, and reuse. In addition to the original latent-control pipeline based on a VAE, MDN-RNN, and controller, AGILE-WM adds practical tools for large-scale dataset construction, captioning with multimodal language models, and cluster-friendly training workflows.

## What This Project Does

AGILE-WM covers the full data and modeling loop for language-aware world model experiments:

- collect rollout trajectories from trained world-model controllers
- save full episodes and optionally raw image observations
- extract and sample frames from rollouts
- package frames into WebDataset shards for scalable training
- generate frame captions with Qwen3-VL
- fine-tune CLIP on frame-caption pairs
- inspect caption quality and shard contents with lightweight notebooks

## Pipeline Overview

1. Collect rollout episodes from trained controllers in CarRacing or Doom.
2. Save those trajectories as rollout files, optionally with raw RGB observations.
3. Sample frames from the saved rollouts and convert them into WebDataset shards.
4. Run multimodal captioning over frame shards to build paired image-text data.
5. Fine-tune CLIP on the generated frame-caption pairs.
6. Inspect shards, captions, and model outputs with notebooks and analysis utilities.

## Main Components

- `collect_rollouts.py` collects rollout episodes from trained controllers.
- `env.py`, `controller.py`, `vae/`, and `rnn/` contain the world-model environment wrappers, controller, latent encoder, and recurrent dynamics code.
- `extract_frames.py` samples frames from rollout files and writes WebDataset shards with metadata.
- `caption_shard.py` runs Qwen3-VL captioning over a shard of frames and writes image-caption pairs.
- `CLIP_finetune.py` fine-tunes CLIP on captioned frame datasets.
- `run_caption_array.sh` and `run_clip_finetune.sh` provide cluster-oriented entry points for large-scale captioning and training.
- `sample_random_caption_pairs.ipynb` and related notebooks help inspect shard contents and caption quality.

## Why AGILE-WM

Classic world-model pipelines learn compact latent dynamics for control, but they do not directly expose semantic descriptions of what the agent sees. AGILE-WM extends that workflow by attaching language to visual rollouts. That makes it easier to inspect behavior, build reusable image-text datasets from control trajectories, and study whether language supervision improves learned visual representations for downstream tasks.

## Requirements

- Python 3.10
- `uv` for environment management
- TensorFlow for the original world-model components
- PyTorch, Transformers, PEFT, Pillow, and WebDataset for captioning and CLIP fine-tuning
- access to local Qwen3-VL weights for caption generation
- optional access to a cluster environment for large-scale runs

## Example Workflow

### 1. Collect rollouts

```bash
uv run --python 3.10 python collect_rollouts.py \
  --config_path configs/carracing.config \
  --num_episodes 100 \
  --with_obs
```

### 2. Build frame shards from rollouts

```bash
uv run --python 3.10 python extract_frames.py
```

### 3. Caption a shard with Qwen3-VL

```bash
uv run --python 3.10 python caption_shard.py \
  --shard_path webdataset_frames/shard-00000.tar \
  --output_dir outputs \
  --model_dir qwen3-vl-8b-instruct
```

### 4. Fine-tune CLIP on captioned shards

```bash
uv run --python 3.10 python CLIP_finetune.py \
  --data_root outputs
```

### 5. Test fused latent reconstruction

This experiment is separate from rollout collection. It loads a frozen VAE and a fine-tuned CLIP checkpoint, fuses the VAE latent and CLIP image embedding with compact bilinear pooling, then trains only a linear projection back into the VAE decoder latent space.

```bash
uv run --python 3.10 python fusion_reconstruction_experiment.py \
  --config_path configs/carracing.config \
  --data_root webdataset_frames \
  --clip_checkpoint path/to/merged_final \
  --output_dir fusion_reconstruction_runs/carracing_cbp
```

The script now precomputes and caches the frozen VAE latents and CLIP image embeddings once under `output_dir/feature_cache` by default, then trains the projection from those cached arrays. It writes training curves, saved projection weights, the cache metadata, and preview grids that compare the original frame, the frozen VAE reconstruction baseline, and the reconstruction produced from the fused latent.

## Cluster Usage

The repository includes Slurm scripts for large-scale runs on Mila-style cluster setups:

- `run_caption_array.sh` stages shards locally and captions them with Qwen3-VL.
- `run_clip_finetune.sh` fine-tunes CLIP on captioned WebDataset shards.

These scripts assume scratch storage for model caches and outputs, and they are set up to rebuild the locked `uv` environment on the cluster before launching jobs.

## Repository Layout

```text
configs/             environment and experiment configuration files
rollouts/            collected rollout episodes
webdataset_frames/   frame-only WebDataset shards
outputs/             captioned image-text shards and summaries
results/             controller and training outputs
vae/                 VAE implementation
rnn/                 MDN-RNN implementation
logs/                cluster job logs
```

## Acknowledgments

Parts of the world-model pipeline in this repository build on ideas and code structure from the WorldModels codebase by zacwellmer
- https://github.com/zacwellmer/WorldModels

## Status

AGILE-WM is a research-oriented codebase optimized for experimentation rather than polished packaging. It combines older world-model control components with newer data-generation and vision-language training utilities, so the repository is best understood as an experimental pipeline for iterative research.