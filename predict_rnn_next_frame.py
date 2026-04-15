"""
Load a saved MDN-RNN, pick a frame/action pair from the cached rollout feature
store, predict the next CBP-fused latent state, and reconstruct the current
and predicted next frames using the saved projection layer and VAE decoder.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from fusion_reconstruction_experiment import (
    _load_saved_model_weights,
    build_model_args,
    configure_tensorflow_memory_growth,
    find_vae_checkpoint_dir,
    load_config,
    resolve_path,
)
from rnn.rnn import MDNRNN
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


def load_series_store(series_dir: Path) -> dict:
    metadata = load_json(series_dir / "metadata.json")
    if metadata.get("fused_representation_type") != "cbp_fused":
        raise ValueError(
            f"Series cache in {series_dir} uses fused_representation_type={metadata.get('fused_representation_type')!r}. "
            "Re-run series.py to build a CBP-fused cache before using this script."
        )
    return {
        "metadata": metadata,
        "fused_latents": np.load(series_dir / "fused_latents.npy", mmap_mode="r"),
        "actions": np.load(series_dir / "actions.npy", mmap_mode="r"),
    }


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


def make_two_frame_image(frame0: np.ndarray, frame1: np.ndarray, output_path: Path) -> None:
    title_height = 32
    gap = 12
    panel_width = frame0.shape[1]
    panel_height = frame0.shape[0]
    canvas = Image.new("RGB", (panel_width * 2 + gap * 3, panel_height + title_height + gap * 2), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((gap, 8), "frame 0", fill=(0, 0, 0), font=font)
    draw.text((panel_width + gap * 2, 8), "frame 1", fill=(0, 0, 0), font=font)

    canvas.paste(Image.fromarray(frame0, mode="RGB"), (gap, title_height + gap))
    canvas.paste(Image.fromarray(frame1, mode="RGB"), (panel_width + gap * 2, title_height + gap))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def choose_output_path(args: argparse.Namespace, exp_name: str, env_name: str) -> Path:
    explicit = resolve_path(args.output_path)
    if explicit is not None:
        return explicit
    return SCRIPT_DIR / "results" / exp_name / env_name / "rnn_prediction_preview" / (
        f"rollout_{args.rollout_index:04d}_frame_{args.frame_index:04d}.png"
    )


def validate_indices(store: dict, rollout_index: int, frame_index: int) -> None:
    num_rollouts = int(store["fused_latents"].shape[0])
    frames_per_rollout = int(store["actions"].shape[1])
    if not 0 <= rollout_index < num_rollouts:
        raise IndexError(f"rollout_index={rollout_index} is outside [0, {num_rollouts - 1}]")
    if not 0 <= frame_index < frames_per_rollout:
        raise IndexError(f"frame_index={frame_index} is outside [0, {frames_per_rollout - 1}]")


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

    store = load_series_store(series_dir)
    validate_indices(store, args.rollout_index, args.frame_index)
    metadata = store["metadata"]

    model_args = build_model_args(config, args)
    vae_checkpoint = find_vae_checkpoint_dir(model_args, resolve_path(args.vae_checkpoint))
    vae = load_vae(model_args, vae_checkpoint)

    rnn, rnn_args = load_rnn(rnn_dir, config, metadata)
    reconstruction_config = load_json(reconstruction_dir / "run_config.json")
    projection = load_projection(
        reconstruction_dir=reconstruction_dir,
        fusion_dim=int(reconstruction_config["fusion_dim"]),
        z_size=int(reconstruction_config["z_size"]),
        use_best=not args.use_last_projection,
    )

    chunk_size = int(metadata["chunk_size"])
    chunk_start = (args.frame_index // chunk_size) * chunk_size
    prefix_end = args.frame_index + 1

    z_prefix = np.asarray(store["fused_latents"][args.rollout_index, chunk_start:prefix_end], dtype=np.float32)
    action_prefix = np.asarray(store["actions"][args.rollout_index, chunk_start:prefix_end], dtype=np.float32)
    current_latent = np.asarray(store["fused_latents"][args.rollout_index, args.frame_index], dtype=np.float32)
    predicted_next_latent = predict_next_latent(rnn, z_prefix, action_prefix)

    current_frame = decode_fused_latent(current_latent, projection, vae)
    predicted_frame = decode_fused_latent(predicted_next_latent, projection, vae)

    output_path = choose_output_path(args, exp_name, env_name)
    make_two_frame_image(current_frame, predicted_frame, output_path)

    summary = {
        "series_dir": str(series_dir),
        "rnn_dir": str(rnn_dir),
        "reconstruction_dir": str(reconstruction_dir),
        "rollout_index": int(args.rollout_index),
        "frame_index": int(args.frame_index),
        "source_frame_index": int(metadata.get("discard_start_frames", 0) + args.frame_index),
        "chunk_start": int(chunk_start),
        "action": action_prefix[-1].tolist(),
        "vae_dim": int(model_args.z_size),
        "fused_dim": int(rnn_args.z_size),
        "fused_representation_type": str(metadata["fused_representation_type"]),
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
        description="Predict the next latent state from a saved RNN and reconstruct frame 0 and predicted frame 1."
    )
    parser.add_argument("--config_path", default=str(config_path))
    parser.add_argument("--series_dir", default=None, help="Defaults to results/<exp>/<env>/series")
    parser.add_argument("--rnn_dir", default=None, help="Defaults to results/<exp>/<env>/tf_rnn")
    parser.add_argument("--reconstruction_dir", default="fusion_reconstruction_runs/shard0_ep5_cbp544")
    parser.add_argument("--vae_checkpoint", default=None)
    parser.add_argument("--env_name", default=config.get("env_name", "CarRacing-v0"))
    parser.add_argument("--exp_name", default=config.get("exp_name", "WorldModels"))
    parser.add_argument("--z_size", type=int, default=int(config.get("z_size", 32)))
    parser.add_argument("--rollout_index", type=int, default=0)
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--use_last_projection", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())