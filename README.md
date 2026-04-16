# AGILE-WM

Aligning Generative and dIscriminative Latent rEpresenations in World Models (AGILE-WM) is a research codebase for combining World Models-style control with vision-language data generation and representation learning. It supports an end-to-end workflow that starts with environment rollouts, converts trajectories into frame datasets, generates natural-language captions for sampled frames, fine-tunes CLIP on the resulting image-text pairs, and runs reconstruction experiments that decode fused latents back into image space.

The project is built for experiments in environments such as CarRacing and Doom, and is designed to make control data easier to inspect, enrich, and reuse. In addition to the original latent-control pipeline based on a VAE, MDN-RNN, and controller, AGILE-WM adds practical tools for large-scale dataset construction, captioning with multimodal language models, fused-latent reconstruction, and cluster-friendly training workflows.

## What This Project Does

AGILE-WM covers the full data and modeling loop for language-aware world model experiments:

- collect rollout trajectories from trained world-model controllers
- save full episodes and optionally raw image observations
- extract and sample frames from rollouts
- package frames into WebDataset shards for scalable training
- generate frame captions with Qwen3-VL
- fine-tune CLIP on frame-caption pairs
- reconstruct frames from fused VAE+CLIP latents at the current timestep
- train an MDN-RNN directly on fused rollout latents and decode predicted next-step states
- inspect caption quality, shard contents, and reconstruction outputs with lightweight notebooks

## Pipeline Overview

1. Collect rollout episodes from trained controllers in CarRacing or Doom.
2. Save those trajectories as rollout files, optionally with raw RGB observations.
3. Sample frames from the saved rollouts and convert them into WebDataset shards.
4. Run multimodal captioning over frame shards to build paired image-text data.
5. Fine-tune CLIP on the generated frame-caption pairs.
6. Train a fusion-and-projection model that reconstructs the current frame from the fused latent `z_t`.
7. Encode rollout sequences into fused latents, train an MDN-RNN on those sequences, and predict fused `z_{t+1}` states conditioned on actions.
8. Decode both the current fused state and the predicted next fused state back into frames for qualitative comparison.
9. Inspect shards, captions, reconstructions, and model outputs with notebooks and analysis utilities.

## Main Components

- `collect_rollouts.py` collects rollout episodes from trained controllers.
- `env.py`, `controller.py`, `vae/`, and `rnn/` contain the world-model environment wrappers, controller, latent encoder, and recurrent dynamics code.
- `extract_frames.py` samples frames from rollout files and writes WebDataset shards with metadata.
- `caption_shard.py` runs Qwen3-VL captioning over a shard of frames and writes image-caption pairs.
- `CLIP_finetune.py` fine-tunes CLIP on captioned frame datasets.
- `fusion_reconstruction_experiment.py` fuses frozen VAE and CLIP features with compact bilinear pooling and trains a linear projection that reconstructs frames from the fused `z_t` representation.
- `series.py` encodes rollout files into cached VAE, CLIP, and fused latent trajectories aligned with actions for recurrent modeling.
- `rnn_train.py` trains the MDN-RNN directly on cached fused latent sequences.
- `predict_rnn_next_frame.py` reconstructs the current fused state and the RNN-predicted next fused state into side-by-side preview sheets.
- `run_caption_array.sh` and `run_clip_finetune.sh` provide cluster-oriented entry points for large-scale captioning and training.
- `sample_random_caption_pairs.ipynb`, `plot_reconstruction_losses.ipynb`, and related notebooks help inspect shard contents, caption quality, and reconstruction behavior.

## Why AGILE-WM

Classic world-model pipelines learn compact latent dynamics for control, but they do not directly expose semantic descriptions of what the agent sees. AGILE-WM extends that workflow by attaching language to visual rollouts and by adding reconstruction probes over fused latents. That makes it easier to inspect behavior, build reusable image-text datasets from control trajectories, study whether language supervision improves learned visual representations for downstream tasks, and visualize what information is preserved in both current-step and predicted next-step latent states. A further motivation is control robustness: adding semantic information to the latent representation can make the policy and dynamics model less brittle when input frames are noisy or partially corrupted, because they can rely more on higher-level scene content than on exact pixel detail.

## Research Framing

The core research question behind AGILE-WM is whether a single world-model representation can preserve both perceptual detail and semantic abstraction instead of forcing a choice between generative and discriminative encoders.

A motivating intuition from the poster is that brains and models are both predictors, but they need not predict in the same representational style. Some systems are more detail-oriented: they keep internal states closer to the raw sensory manifold and may therefore excel at visual memory, exact copying, and local pattern recognition, while being weaker at verbal abstraction, flexible conceptual reasoning, and more global interpretation. Other systems are more semantic-oriented: they compress sensory data more aggressively into abstract structure, which can help with categorization, abstraction, and relational inference, but may lose fine visual specificity and exact detail.

AGILE-WM is motivated by the idea that a useful world-model latent should avoid committing too early to either extreme. Instead, it should preserve enough perceptual structure to remain decodable and dynamics-friendly while also carrying enough semantic structure to support abstraction, robustness, and control under noisy observations.

The working hypothesis is that these two views are complementary rather than redundant:

- the VAE latent captures low-level visual structure that is useful for reconstruction and compact dynamics modeling
- the CLIP embedding contributes higher-level semantic information about scene layout, road geometry, alignment, and motion-relevant context
- a fused latent should therefore support both visual reconstruction and action-conditioned prediction better than either source alone

The current reconstruction-and-prediction objective can be summarized as:

```text
v_t = VAE_Enc(x_t)
c_t = CLIP_LoRA(x_t)
z_t = CBP(v_t, c_t)
x_hat_t = VAE_Dec(FC(z_t))

(z_hat_{t+1}, h_{t+1}) = RNN(z_t, a_t, h_t)
y_hat_{t+1} = VAE_Dec(FC(z_hat_{t+1}))
```

In other words, AGILE-WM first reconstructs the current frame from the fused state `z_t`, then asks whether an action-conditioned recurrent model can roll that fused representation forward to a meaningful predicted `z_{t+1}` that is still decodable back into image space.

## Method Details

The poster experiments highlight four concrete stages beyond the baseline world-model pipeline:

1. Caption rollout frames with Qwen3-VL-8B-Instruct. Captions are intentionally short and grounded in control-relevant content. In CarRacing this includes road curvature, vehicle alignment, surface type, drifting behavior, and other motion cues that matter for planning.
2. Fine-tune CLIP with LoRA on the generated image-text pairs. The poster setup uses a lightweight adaptation regime with LoRA rank `r = 16`, `alpha = 32`, InfoNCE training, and a cosine decay schedule so the vision-language encoder can adapt to the rollout domain without full-model retraining.
3. Fuse VAE and CLIP latents with Compact Bilinear Pooling. In the current poster configuration the VAE latent is 32-dimensional and the CLIP image embedding is 512-dimensional. A full outer product would produce 16384 dimensions, so the CBP output is compressed to 544 dimensions to keep the fused state compact while still modeling higher-order interactions.
4. Decode and predict in the fused space. A single fully connected projection maps the fused latent back into the VAE decoder latent space for frame reconstruction, while the MDN-RNN models transitions directly in the fused latent space conditioned on actions.

This design is meant to preserve the practical strengths of the original World Models pipeline while adding language-aware structure to the latent representation instead of treating semantics as a separate downstream feature.

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

### 5. Reconstruct frames from fused `z_t`

This experiment is separate from rollout collection. It loads a frozen VAE and a fine-tuned CLIP checkpoint, fuses the VAE latent and CLIP image embedding with compact bilinear pooling, then trains only a linear projection back into the VAE decoder latent space.

```bash
uv run --python 3.10 python fusion_reconstruction_experiment.py \
  --config_path configs/carracing.config \
  --data_root webdataset_frames \
  --clip_checkpoint path/to/merged_final \
  --output_dir fusion_reconstruction_runs/carracing_cbp
```

The script now precomputes and caches the frozen VAE latents and CLIP image embeddings once under `output_dir/feature_cache` by default, then trains the projection from those cached arrays. It writes training curves, saved projection weights, the cache metadata, and preview grids that compare the original frame, the frozen VAE reconstruction baseline, and the reconstruction produced from the fused latent.

### 6. Cache rollout latents for recurrent prediction

This stage re-encodes saved rollout frames with the frozen VAE and fine-tuned CLIP models, applies the saved compact bilinear pooling transform from the reconstruction run, and writes aligned rollout-level arrays for fused latents, actions, rewards, and done flags.

```bash
uv run --python 3.10 python series.py \
  --config_path configs/carracing.config \
  --rollout_dir rollouts \
  --clip_checkpoint clip_finetune/merged_final \
  --reconstruction_dir fusion_reconstruction_runs/carracing_cbp
```

By default this produces a cache under `results/<exp>/<env>/series`. That cache keeps the fused representation consistent with the `z_t` reconstruction run by reusing the saved `cbp_state.npz`.

### 7. Train the MDN-RNN on fused latent sequences

```bash
uv run --python 3.10 python rnn_train.py \
  --config_path configs/carracing.config \
  --series_dir results/WorldModels/CarRacing-v0/series
```

The recurrent model is trained directly on the cached fused latent sequences plus actions. It writes the exported TensorFlow model under `results/<exp>/<env>/tf_rnn` and also saves training checkpoints for downstream visualization.

### 8. Reconstruct predicted `z_{t+1}` frames

This experiment uses the saved RNN, the compact bilinear pooling state, and the learned projection from the `z_t` reconstruction run. For each selected frame it reconstructs the current fused state and the RNN-predicted next fused state, then renders a comparison grid with columns for the ground-truth frame, the reconstruction from `z_t`, and the reconstruction from predicted `z_{t+1}`.

```bash
uv run --python 3.10 python predict_rnn_next_frame.py \
  --config_path configs/carracing.config \
  --series_dir results/WorldModels/CarRacing-v0/series \
  --rnn_dir results/WorldModels/CarRacing-v0/tf_rnn \
  --reconstruction_dir fusion_reconstruction_runs/carracing_cbp \
  --rollout_index 0 \
  --frame_index 0 \
  --num_frames 8
```

The output is saved under `results/<exp>/<env>/rnn_prediction_preview` by default, together with a JSON sidecar that records the source rollout, frame indices, selected projection weights, and action sequence used for the preview.

## Reconstruction Experiments

AGILE-WM currently documents two related reconstruction probes:

- `fusion_reconstruction_experiment.py` evaluates whether the fused representation at the current timestep retains enough information to reconstruct the observed frame. It freezes the VAE and CLIP encoders, learns only a small projection back into the VAE decoder latent space, and reports both the fused reconstruction loss and the frozen-VAE reconstruction baseline.
- `predict_rnn_next_frame.py` evaluates whether the RNN's predicted fused state is visually meaningful. It encodes a rollout frame into the same fused space, predicts the next fused latent with the MDN-RNN, decodes both states with the saved projection plus VAE decoder, and saves GT | `z_t` reconstruction | predicted `z_{t+1}` reconstruction comparison sheets.

Together these experiments separate two questions: whether the fused representation is reconstructive at all, and whether the recurrent dynamics learned in that fused space preserve enough structure to support next-frame visualization.

## Evaluation and Early Findings

The poster results are best interpreted as an initial validation of the representation design rather than a final benchmark. The current evaluation emphasizes three signals:

- CLIP adaptation quality, monitored with training and validation loss together with image-to-text and text-to-image retrieval metrics such as recall@1 and recall@5.
- Reconstruction quality from the fused latent, compared against the frozen-VAE reconstruction baseline to check whether semantic fusion preserves enough information for decoding.
- Qualitative roll-forward behavior, visualized with comparison sheets that place the ground-truth frame, the reconstruction from `z_t`, and the reconstruction from predicted `z_{t+1}` side by side.

The preliminary takeaway from the poster is that the fused latent remains informative enough that a single linear projection can recover recognizable frames from it. That is encouraging evidence that perceptual structure from the VAE and semantic structure from CLIP can coexist in one compact latent without immediately destroying decodability.

The next-step prediction visuals are especially useful because they probe more than static reconstruction quality: they test whether the fused state also supports dynamics learning. If the predicted `z_{t+1}` stays decodable under action-conditioned rollout, that suggests the representation is not only semantically enriched but also operationally usable inside the world model.

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
clip_finetune/       saved LoRA adapters and merged CLIP checkpoints
fusion_reconstruction_runs/ saved reconstruction runs, projections, caches, and preview grids
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