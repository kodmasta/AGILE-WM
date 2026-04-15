"""
Load a saved MDN-RNN, pick a frame/action pair from the cached rollout feature
store, predict the next CBP-fused latent state, and reconstruct the current
and predicted next frames using the saved projection layer and VAE decoder.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence, Tuple

import numpy as np
import tensorflow as tf
import torch
from PIL import Image, ImageDraw, ImageFont

from fusion_reconstruction_experiment import (
    _load_saved_model_weights,
    build_model_args,
    configure_tensorflow_memory_growth,
    ensure_supported_cuda_device,
    extract_clip_features,
    extract_vae_latents,
    find_vae_checkpoint_dir,
    load_clip_model,
    load_config,
    resolve_path,
    resolve_clip_checkpoint_dir,
)
from rnn.rnn import MDNRNN
from series import load_cbp, load_rollout
from vae.vae import CVAE


SCRIPT_DIR = Path(__file__).resolve().parent


def default_results_dir(exp_name: str, env_name: str, leaf: str) -> Path:
    return SCRIPT_DIR / "results" / exp_name / env_name / leaf


def fallback_results_dir(exp_name: str, env_name: str, leaf: str) -> Path:
    if "-" in env_name:
        base_env = env_name.split("-", 1)[0]
        candidate = SCRIPT_DIR / "results" / exp_name / base_env / leaf
        if candidate.exists():
            return candidate
    return default_results_dir(exp_name, env_name, leaf)


def resolve_existing_dir(path_like: str | None, exp_name: str, env_name: str, leaf: str) -> Path:
    explicit = resolve_path(path_like)
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Required directory not found: {explicit}")
        return explicit

    candidate = default_results_dir(exp_name, env_name, leaf)
    if candidate.exists():
        return candidate

    fallback = fallback_results_dir(exp_name, env_name, leaf)
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not find {leaf} under results/{exp_name}/{env_name} or its version-stripped fallback."
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_series_metadata(series_dir: Path) -> dict:
    metadata = load_json(series_dir / "metadata.json")
    if metadata.get("fused_representation_type") != "cbp_fused":
        raise ValueError(
            f"Series cache in {series_dir} uses fused_representation_type={metadata.get('fused_representation_type')!r}. "
            "Re-run series.py to build a CBP-fused cache before using this script."
        )
    return metadata


def build_rnn_args(config: dict, metadata: dict) -> SimpleNamespace:
    fused_dim = int(metadata["fused_feature_dim"])
    return SimpleNamespace(
        rnn_learning_rate=float(config.get("rnn_learning_rate", 1e-3)),
        rnn_grad_clip=float(config.get("rnn_grad_clip", 1.0)),
        rnn_size=int(config.get("rnn_size", 256)),
        rnn_num_mixture=int(config.get("rnn_num_mixture", 5)),
        rnn_r_pred=int(config.get("rnn_r_pred", 0)),
        rnn_d_pred=int(config.get("rnn_d_pred", 0)),
        rnn_d_true_weight=float(config.get("rnn_d_true_weight", 1.0)),
        rnn_temperature=float(config.get("rnn_temperature", 1.0)),
        rnn_batch_size=1,
        rnn_max_seq_len=1,
        z_size=fused_dim,
        a_width=int(config.get("a_width", 3)),
        rnn_input_seq_width=fused_dim + int(config.get("a_width", 3)),
    )


def load_rnn(rnn_dir: Path, config: dict, metadata: dict) -> Tuple[MDNRNN, SimpleNamespace]:
    args = build_rnn_args(config, metadata)
    rnn = MDNRNN(args=args)
    checkpoint_dir = rnn_dir.parent / f"{rnn_dir.name}_training_checkpoints"
    latest_checkpoint = tf.train.latest_checkpoint(str(checkpoint_dir))
    if latest_checkpoint is None:
        raise FileNotFoundError(
            f"No training checkpoint found in {checkpoint_dir}. Re-run rnn_train.py with the current code so it writes a final checkpoint."
        )

    checkpoint = tf.train.Checkpoint(rnn=rnn)
    checkpoint.restore(latest_checkpoint).expect_partial()
    return rnn, args


def load_vae(model_args: SimpleNamespace, vae_checkpoint: Path) -> CVAE:
    vae = CVAE(model_args)
    _load_saved_model_weights(vae, vae_checkpoint)
    vae.trainable = False
    return vae


def load_projection(reconstruction_dir: Path, fusion_dim: int, z_size: int, use_best: bool) -> tf.keras.Sequential:
    projection = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(fusion_dim,)),
            tf.keras.layers.Dense(z_size, name="fusion_projection"),
        ]
    )
    projection(tf.zeros((1, fusion_dim), dtype=tf.float32))
    weights_name = "best_projection.weights.h5" if use_best else "last_projection.weights.h5"
    projection.load_weights(reconstruction_dir / weights_name)
    return projection


def resolve_rollout_path(metadata: dict, rollout_index: int) -> Path:
    rollout_dir = Path(metadata["rollout_dir"])
    rollout_files = metadata.get("rollout_files") or []
    if rollout_files:
        rollout_path = rollout_dir / rollout_files[rollout_index]
        if not rollout_path.exists():
            raise FileNotFoundError(f"Rollout file not found: {rollout_path}")
        return rollout_path

    rollout_glob = str(metadata.get("rollout_glob", "episode_*.npz"))
    candidates = sorted(rollout_dir.glob(rollout_glob))
    if not 0 <= rollout_index < len(candidates):
        raise IndexError(f"rollout_index={rollout_index} is outside [0, {len(candidates) - 1}]")
    return candidates[rollout_index]


def load_rollout_sample(metadata: dict, rollout_index: int) -> Tuple[Path, np.ndarray, np.ndarray]:
    rollout_path = resolve_rollout_path(metadata, rollout_index)
    obs, actions, _, _ = load_rollout(
        rollout_path=rollout_path,
        source_frames_per_rollout=int(metadata["source_frames_per_rollout"]),
        discard_start_frames=int(metadata.get("discard_start_frames", 0)),
        cached_frames_per_rollout=int(metadata["frames_per_rollout"]),
    )
    return rollout_path, obs, actions


def load_clip_encoder(
    metadata: dict,
    explicit_clip_checkpoint: str | None,
    hf_cache_dir: str | None,
    local_files_only: bool,
    clip_device_name: str,
):
    clip_checkpoint = resolve_path(explicit_clip_checkpoint)
    if clip_checkpoint is None:
        clip_checkpoint = resolve_path(metadata.get("clip_checkpoint"))
    clip_checkpoint = resolve_clip_checkpoint_dir(clip_checkpoint)

    clip_device = torch.device(
        clip_device_name if clip_device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ensure_supported_cuda_device(clip_device)
    clip_model, clip_processor = load_clip_model(
        checkpoint_dir=clip_checkpoint,
        cache_dir=resolve_path(hf_cache_dir),
        local_files_only=local_files_only,
    )
    clip_model.to(clip_device)
    return clip_model, clip_processor, clip_device, clip_checkpoint


def encode_frame_to_fused_latent(
    frame: np.ndarray,
    vae: CVAE,
    clip_model,
    clip_processor,
    clip_device: torch.device,
    cbp,
    clip_feature_source: str,
    vae_latent_source: str,
    normalize_clip_features: bool,
) -> np.ndarray:
    batch_images = frame[None, ...].astype(np.float32) / 255.0
    vae_latent = extract_vae_latents(vae, batch_images, vae_latent_source)
    clip_feature = extract_clip_features(
        clip_model,
        clip_processor,
        batch_images,
        clip_device,
        clip_feature_source,
        normalize_clip_features,
    )
    fused_latent = cbp(
        tf.convert_to_tensor(vae_latent, dtype=tf.float32),
        tf.convert_to_tensor(clip_feature, dtype=tf.float32),
    )
    return fused_latent.numpy()[0].astype(np.float32)


def predict_next_latent(rnn: MDNRNN, z_sequence: np.ndarray, action_sequence: np.ndarray) -> np.ndarray:
    inputs = np.concatenate([z_sequence, action_sequence], axis=-1).astype(np.float32)[None, :, :]
    outputs = rnn(tf.convert_to_tensor(inputs, dtype=tf.float32), training=False)
    mdn = outputs["MDN"]
    mdn = tf.reshape(mdn, [1, z_sequence.shape[0], rnn.args.z_size, 3 * rnn.args.rnn_num_mixture])
    mdn_last = mdn[:, -1, :, :]
    mu, _, logpi = tf.split(mdn_last, num_or_size_splits=3, axis=-1)
    weights = tf.nn.softmax(logpi, axis=-1)
    expected_next = tf.reduce_sum(weights * mu, axis=-1)
    return expected_next.numpy()[0].astype(np.float32)


def decode_fused_latent(fused_latent: np.ndarray, projection: tf.keras.Sequential, vae: CVAE) -> np.ndarray:
    projected_z = projection(tf.convert_to_tensor(fused_latent[None, :], dtype=tf.float32), training=False)
    reconstruction = vae.decode(projected_z).numpy()[0]
    return np.clip(reconstruction * 255.0, 0.0, 255.0).astype(np.uint8)


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def make_comparison_sheet(
    frame_indices: Sequence[int],
    original_frames: Sequence[np.ndarray],
    current_frames: Sequence[np.ndarray],
    predicted_frames: Sequence[np.ndarray],
    output_path: Path,
) -> None:
    if (
        len(frame_indices) != len(original_frames)
        or len(original_frames) != len(current_frames)
        or len(current_frames) != len(predicted_frames)
        or not original_frames
        or not current_frames
        or not predicted_frames
    ):
        raise ValueError("original_frames, current_frames, and predicted_frames must be non-empty and aligned")

    font = ImageFont.load_default()
    tile_width = 64
    tile_height = 64
    title_height = 0
    header_height = 24
    column_labels = ("GT", "t", "t+1")

    grid = Image.new(
        "RGB",
        (tile_width * 3, title_height + header_height + tile_height * len(frame_indices)),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(grid)

    for column, label in enumerate(column_labels):
        label_width, label_height = measure_text(draw, label, font)
        label_x = column * tile_width + (tile_width - label_width) // 2
        label_y = title_height + (header_height - label_height) // 2
        draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

    for row, (original_frame, current_frame, predicted_frame) in enumerate(
        zip(original_frames, current_frames, predicted_frames)
    ):
        y_offset = title_height + header_height + row * tile_height
        grid.paste(Image.fromarray(original_frame, mode="RGB"), (0, y_offset))
        grid.paste(Image.fromarray(current_frame, mode="RGB"), (tile_width, y_offset))
        grid.paste(Image.fromarray(predicted_frame, mode="RGB"), (tile_width * 2, y_offset))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def choose_output_path(args: argparse.Namespace, exp_name: str, env_name: str) -> Path:
    explicit = resolve_path(args.output_path)
    if explicit is not None:
        return explicit
    end_frame_index = args.frame_index + args.num_frames - 1
    return SCRIPT_DIR / "results" / exp_name / env_name / "rnn_prediction_preview" / (
        f"rollout_{args.rollout_index:04d}_frames_{args.frame_index:04d}_{end_frame_index:04d}.png"
    )


def validate_indices(
    metadata: dict,
    obs: np.ndarray,
    actions: np.ndarray,
    rollout_index: int,
    frame_index: int,
    num_frames: int,
) -> None:
    num_rollouts = int(metadata["num_rollouts"])
    frames_per_rollout = int(actions.shape[0])
    if not 0 <= rollout_index < num_rollouts:
        raise IndexError(f"rollout_index={rollout_index} is outside [0, {num_rollouts - 1}]")
    if not 0 <= frame_index < frames_per_rollout:
        raise IndexError(f"frame_index={frame_index} is outside [0, {frames_per_rollout - 1}]")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if frame_index + num_frames > frames_per_rollout:
        raise IndexError(
            f"Requested frames [{frame_index}, {frame_index + num_frames - 1}] but rollout only has frame indices [0, {frames_per_rollout - 1}]"
        )
    if obs.shape[0] != frames_per_rollout + 1:
        raise ValueError(
            f"Expected obs to contain frames_per_rollout + 1 frames, got obs.shape[0]={obs.shape[0]} and actions.shape[0]={frames_per_rollout}."
        )


def main(args: argparse.Namespace) -> None:
    configure_tensorflow_memory_growth()

    config = load_config(Path(args.config_path)) if Path(args.config_path).exists() else {}
    exp_name = args.exp_name or config.get("exp_name", "WorldModels")
    env_name = args.env_name or config.get("env_name", "CarRacing-v0")

    series_dir = resolve_existing_dir(args.series_dir, exp_name, env_name, "series")
    rnn_dir = resolve_existing_dir(args.rnn_dir, exp_name, env_name, "tf_rnn")
    reconstruction_dir = resolve_path(args.reconstruction_dir)
    if reconstruction_dir is None or not reconstruction_dir.exists():
        raise FileNotFoundError(f"Reconstruction directory not found: {args.reconstruction_dir}")

    metadata = load_series_metadata(series_dir)
    rollout_path, obs, actions = load_rollout_sample(metadata, args.rollout_index)
    validate_indices(metadata, obs, actions, args.rollout_index, args.frame_index, args.num_frames)

    model_args = build_model_args(config, args)
    vae_checkpoint = find_vae_checkpoint_dir(model_args, resolve_path(args.vae_checkpoint))
    vae = load_vae(model_args, vae_checkpoint)

    rnn, rnn_args = load_rnn(rnn_dir, config, metadata)
    reconstruction_config = load_json(reconstruction_dir / "run_config.json")
    cbp = load_cbp(reconstruction_dir / "cbp_state.npz")
    projection = load_projection(
        reconstruction_dir=reconstruction_dir,
        fusion_dim=int(reconstruction_config["fusion_dim"]),
        z_size=int(reconstruction_config["z_size"]),
        use_best=not args.use_last_projection,
    )

    clip_model, clip_processor, clip_device, clip_checkpoint = load_clip_encoder(
        metadata=metadata,
        explicit_clip_checkpoint=args.clip_checkpoint,
        hf_cache_dir=args.hf_cache_dir,
        local_files_only=args.local_files_only,
        clip_device_name=args.clip_device,
    )

    clip_feature_source = str(metadata.get("clip_feature_source", reconstruction_config.get("clip_feature_source", "image_features")))
    vae_latent_source = str(metadata.get("vae_latent_source", reconstruction_config.get("vae_latent_source", "mean")))
    normalize_clip_features = bool(metadata.get("normalize_clip_features", False))

    frame_indices = []
    original_frames = []
    current_frames = []
    predicted_frames = []
    actions_summary = []

    for offset in range(args.num_frames):
        frame_index = args.frame_index + offset
        current_latent = encode_frame_to_fused_latent(
            frame=obs[frame_index],
            vae=vae,
            clip_model=clip_model,
            clip_processor=clip_processor,
            clip_device=clip_device,
            cbp=cbp,
            clip_feature_source=clip_feature_source,
            vae_latent_source=vae_latent_source,
            normalize_clip_features=normalize_clip_features,
        )
        current_action = np.asarray(actions[frame_index:frame_index + 1], dtype=np.float32)
        predicted_next_latent = predict_next_latent(rnn, current_latent[None, :], current_action)

        frame_indices.append(int(frame_index))
        original_frames.append(np.asarray(obs[frame_index], dtype=np.uint8))
        current_frames.append(decode_fused_latent(current_latent, projection, vae))
        predicted_frames.append(decode_fused_latent(predicted_next_latent, projection, vae))
        actions_summary.append(current_action[0].tolist())

    output_path = choose_output_path(args, exp_name, env_name)
    projection_label = "last_projection.weights.h5" if args.use_last_projection else "best_projection.weights.h5"
    make_comparison_sheet(frame_indices, original_frames, current_frames, predicted_frames, output_path)

    summary = {
        "series_dir": str(series_dir),
        "rnn_dir": str(rnn_dir),
        "reconstruction_dir": str(reconstruction_dir),
        "rollout_path": str(rollout_path),
        "rollout_index": int(args.rollout_index),
        "start_frame_index": int(args.frame_index),
        "end_frame_index": int(args.frame_index + args.num_frames - 1),
        "num_frames": int(args.num_frames),
        "source_frame_indices": [
            int(metadata.get("discard_start_frames", 0) + frame_index) for frame_index in frame_indices
        ],
        "actions": actions_summary,
        "vae_dim": int(model_args.z_size),
        "fused_dim": int(rnn_args.z_size),
        "fused_representation_type": str(metadata["fused_representation_type"]),
        "clip_checkpoint": str(clip_checkpoint),
        "projection_weights": projection_label,
        "preview_title": None,
        "preview_columns": ["GT", "t", "t+1"],
        "output_path": str(output_path),
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_path)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config_path", default="configs/carracing.config")
    known_args, _ = config_parser.parse_known_args()

    config = {}
    config_path = Path(known_args.config_path)
    if config_path.exists():
        config = load_config(config_path)

    parser = argparse.ArgumentParser(
        description="Render GT, fused-latent reconstruction, and RNN-predicted reconstruction from saved models."
    )
    parser.add_argument("--config_path", default=str(config_path))
    parser.add_argument("--series_dir", default=None, help="Defaults to results/<exp>/<env>/series")
    parser.add_argument("--rnn_dir", default=None, help="Defaults to results/<exp>/<env>/tf_rnn")
    parser.add_argument("--reconstruction_dir", default="fusion_reconstruction_runs/shard0_ep5_cbp544")
    parser.add_argument("--vae_checkpoint", default=None)
    parser.add_argument("--clip_checkpoint", default=None)
    parser.add_argument("--env_name", default=config.get("env_name", "CarRacing-v0"))
    parser.add_argument("--exp_name", default=config.get("exp_name", "WorldModels"))
    parser.add_argument("--z_size", type=int, default=int(config.get("z_size", 32)))
    parser.add_argument("--rollout_index", type=int, default=0)
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--clip_device", default="auto", help="auto, cpu, or cuda[:index].")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--use_last_projection", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())