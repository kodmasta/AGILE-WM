"""
Precompute frozen VAE and CLIP features for rollout frames.

This replaces the old record-to-series pipeline. It reads rollout files from
artifacts/datasets/rollouts/episode_*.npz, encodes every frame with the trained VAE and CLIP
models, and stores per-rollout feature arrays that rnn_train.py can reshape
into fixed chunk-size sequences.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import torch

from agile_wm.paths import (
    DEFAULT_RECONSTRUCTION_RUN,
    default_reconstruction_dir,
    default_rollouts_dir,
    default_world_model_leaf,
)
from fusion_reconstruction_experiment import (
    CompactBilinearPooling,
    _load_saved_model_weights,
    build_model_args,
    configure_tensorflow_memory_growth,
    ensure_supported_cuda_device,
    extract_clip_features,
    extract_vae_latents,
    find_vae_checkpoint_dir,
    load_clip_model,
    load_config,
    resolve_clip_checkpoint_dir,
    resolve_path,
)
from vae.vae import CVAE


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROLLOUT_GLOB = "episode_*.npz"


def default_output_dir(exp_name: str, env_name: str) -> Path:
    return default_world_model_leaf(exp_name, env_name, "series")


def list_rollout_files(rollout_dir: Path, rollout_glob: str, num_rollouts: int) -> List[Path]:
    rollout_files = sorted(rollout_dir.glob(rollout_glob))
    if len(rollout_files) < num_rollouts:
        raise ValueError(
            f"Requested {num_rollouts} rollouts from {rollout_dir}, but only found {len(rollout_files)} "
            f"matching '{rollout_glob}'."
        )
    return rollout_files[:num_rollouts]


def load_rollout(
    rollout_path: Path,
    source_frames_per_rollout: int,
    discard_start_frames: int,
    cached_frames_per_rollout: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(rollout_path) as data:
        required = {"obs", "actions", "rewards", "terminated", "truncated"}
        missing = required.difference(data.files)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise KeyError(f"Rollout {rollout_path} is missing keys: {missing_list}")

        obs = np.asarray(data["obs"], dtype=np.uint8)
        actions = np.asarray(data["actions"], dtype=np.float32)
        rewards = np.asarray(data["rewards"], dtype=np.float32)
        terminated = np.asarray(data["terminated"], dtype=np.bool_)
        truncated = np.asarray(data["truncated"], dtype=np.bool_)

    required_obs = discard_start_frames + cached_frames_per_rollout + 1
    if obs.shape[0] < required_obs:
        raise ValueError(
            f"Rollout {rollout_path} has {obs.shape[0]} observations, expected at least {required_obs}."
        )
    required_actions = discard_start_frames + cached_frames_per_rollout
    if actions.shape[0] < required_actions:
        raise ValueError(
            f"Rollout {rollout_path} has {actions.shape[0]} actions, expected at least {required_actions}."
        )

    start = int(discard_start_frames)
    action_end = start + int(cached_frames_per_rollout)
    obs_end = action_end + 1
    done = np.logical_or(terminated[start:action_end], truncated[start:action_end])
    return (
        obs[start:obs_end],
        actions[start:action_end],
        rewards[start:action_end],
        done,
    )


def encode_rollout_frames(
    *,
    obs: np.ndarray,
    vae: CVAE,
    clip_model,
    clip_processor,
    clip_device: torch.device,
    frame_batch_size: int,
    clip_feature_source: str,
    vae_latent_source: str,
    normalize_clip_features: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    vae_features = []
    clip_features = []

    for start in range(0, obs.shape[0], frame_batch_size):
        end = min(start + frame_batch_size, obs.shape[0])
        batch_images = obs[start:end].astype(np.float32) / 255.0
        batch_vae = extract_vae_latents(vae, batch_images, vae_latent_source).numpy().astype(np.float16)
        batch_clip = extract_clip_features(
            clip_model,
            clip_processor,
            batch_images,
            clip_device,
            clip_feature_source,
            normalize_clip_features,
        ).astype(np.float16)
        vae_features.append(batch_vae)
        clip_features.append(batch_clip)

    return np.concatenate(vae_features, axis=0), np.concatenate(clip_features, axis=0)


def create_memmaps(
    output_dir: Path,
    num_rollouts: int,
    frames_per_rollout: int,
    vae_dim: int,
    clip_dim: int,
    fused_dim: int,
    action_dim: int,
):
    return {
        "vae_latents": np.lib.format.open_memmap(
            str(output_dir / "vae_latents.npy"),
            mode="w+",
            dtype=np.float16,
            shape=(num_rollouts, frames_per_rollout + 1, vae_dim),
        ),
        "clip_features": np.lib.format.open_memmap(
            str(output_dir / "clip_features.npy"),
            mode="w+",
            dtype=np.float16,
            shape=(num_rollouts, frames_per_rollout + 1, clip_dim),
        ),
        "fused_latents": np.lib.format.open_memmap(
            str(output_dir / "fused_latents.npy"),
            mode="w+",
            dtype=np.float16,
            shape=(num_rollouts, frames_per_rollout + 1, fused_dim),
        ),
        "actions": np.lib.format.open_memmap(
            str(output_dir / "actions.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(num_rollouts, frames_per_rollout, action_dim),
        ),
        "rewards": np.lib.format.open_memmap(
            str(output_dir / "rewards.npy"),
            mode="w+",
            dtype=np.float16,
            shape=(num_rollouts, frames_per_rollout, 1),
        ),
        "done": np.lib.format.open_memmap(
            str(output_dir / "done.npy"),
            mode="w+",
            dtype=np.float16,
            shape=(num_rollouts, frames_per_rollout, 1),
        ),
    }


def load_cbp(cbp_path: Path) -> CompactBilinearPooling:
    state = np.load(cbp_path)
    output_dim = int(state["output_dim"][0])
    normalize = bool(state["normalize"][0])
    hash_a = state["hash_a"].astype(np.int32)
    hash_b = state["hash_b"].astype(np.int32)
    sign_a = state["sign_a"].astype(np.float32)
    sign_b = state["sign_b"].astype(np.float32)

    cbp = CompactBilinearPooling(
        input_dim_a=int(hash_a.shape[0]),
        input_dim_b=int(hash_b.shape[0]),
        output_dim=output_dim,
        seed=0,
        normalize=normalize,
    )

    sketch_a = np.zeros((hash_a.shape[0], output_dim), dtype=np.float32)
    sketch_a[np.arange(hash_a.shape[0]), hash_a] = sign_a
    sketch_b = np.zeros((hash_b.shape[0], output_dim), dtype=np.float32)
    sketch_b[np.arange(hash_b.shape[0]), hash_b] = sign_b

    cbp.hash_a = hash_a
    cbp.hash_b = hash_b
    cbp.sign_a = sign_a
    cbp.sign_b = sign_b
    cbp.sketch_a = tf.constant(sketch_a)
    cbp.sketch_b = tf.constant(sketch_b)
    return cbp


def build_metadata(
    args: argparse.Namespace,
    model_args,
    rollout_dir: Path,
    rollout_files: List[Path],
    vae_dim: int,
    clip_dim: int,
    fused_dim: int,
) -> dict:
    return {
        "config_path": str(Path(args.config_path).resolve()),
        "rollout_dir": str(rollout_dir.resolve()),
        "rollout_glob": args.rollout_glob,
        "num_rollouts": int(len(rollout_files)),
        "frames_per_rollout": int(args.frames_per_rollout),
        "source_frames_per_rollout": int(args.source_frames_per_rollout),
        "discard_start_frames": int(args.discard_start_frames),
        "trimmed_tail_frames": int(args.trimmed_tail_frames),
        "chunk_size": int(args.chunk_size),
        "chunks_per_rollout": int(args.frames_per_rollout // args.chunk_size),
        "total_chunks": int(len(rollout_files) * (args.frames_per_rollout // args.chunk_size)),
        "total_sequences": int(len(rollout_files) * (args.frames_per_rollout // args.chunk_size)),
        "exp_name": model_args.exp_name,
        "env_name": model_args.env_name,
        "z_size": int(model_args.z_size),
        "vae_feature_dim": int(vae_dim),
        "clip_feature_dim": int(clip_dim),
        "fused_feature_dim": int(fused_dim),
        "fused_representation_type": "cbp_fused",
        "clip_checkpoint": str(args.clip_checkpoint.resolve()),
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "reconstruction_dir": str(args.reconstruction_dir.resolve()),
        "clip_feature_source": args.clip_feature_source,
        "vae_latent_source": args.vae_latent_source,
        "normalize_clip_features": bool(args.normalize_clip_features),
        "frame_batch_size": int(args.frame_batch_size),
        "rollout_files": [path.name for path in rollout_files],
    }


def preprocess_rollouts(args: argparse.Namespace) -> None:
    args.source_frames_per_rollout = int(args.frames_per_rollout)
    usable_frames = args.source_frames_per_rollout - int(args.discard_start_frames)
    if usable_frames <= 0:
        raise ValueError(
            f"discard_start_frames={args.discard_start_frames} leaves no usable frames out of source_frames_per_rollout={args.source_frames_per_rollout}."
        )

    args.frames_per_rollout = (usable_frames // args.chunk_size) * args.chunk_size
    args.trimmed_tail_frames = usable_frames - int(args.frames_per_rollout)
    if args.frames_per_rollout <= 0:
        raise ValueError(
            f"No full chunks remain after discarding {args.discard_start_frames} start frames with chunk_size={args.chunk_size}."
        )
    if args.trimmed_tail_frames > 0:
        log.warning(
            "Discarding the first %d frames and trimming the last %d frames of each rollout to keep %d frames divisible by chunk_size=%d.",
            args.discard_start_frames,
            args.trimmed_tail_frames,
            args.frames_per_rollout,
            args.chunk_size,
        )

    configure_tensorflow_memory_growth()

    config = load_config(Path(args.config_path)) if Path(args.config_path).exists() else {}
    model_args = build_model_args(config, args)

    rollout_dir = resolve_path(args.rollout_dir)
    if rollout_dir is None or not rollout_dir.exists():
        raise FileNotFoundError(f"Rollout directory not found: {args.rollout_dir}")

    output_dir = resolve_path(args.output_dir) if args.output_dir else default_output_dir(model_args.exp_name, model_args.env_name)
    if output_dir is None:
        raise ValueError("Could not resolve output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    args.clip_checkpoint = resolve_clip_checkpoint_dir(resolve_path(args.clip_checkpoint))
    args.vae_checkpoint = find_vae_checkpoint_dir(model_args, resolve_path(args.vae_checkpoint))
    args.reconstruction_dir = resolve_path(args.reconstruction_dir)
    if args.reconstruction_dir is None or not args.reconstruction_dir.exists():
        raise FileNotFoundError(f"Reconstruction directory not found: {args.reconstruction_dir}")
    cbp_state_path = args.reconstruction_dir / "cbp_state.npz"
    if not cbp_state_path.exists():
        raise FileNotFoundError(f"CBP state not found: {cbp_state_path}")
    cbp = load_cbp(cbp_state_path)

    rollout_files = list_rollout_files(rollout_dir, args.rollout_glob, args.num_rollouts)

    vae = CVAE(model_args)
    _load_saved_model_weights(vae, args.vae_checkpoint)
    vae.trainable = False

    clip_device = torch.device(args.clip_device if args.clip_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    ensure_supported_cuda_device(clip_device)
    clip_model, clip_processor = load_clip_model(
        checkpoint_dir=args.clip_checkpoint,
        cache_dir=resolve_path(args.hf_cache_dir),
        local_files_only=args.local_files_only,
    )
    clip_model.to(clip_device)

    caches = None
    for rollout_index, rollout_path in enumerate(rollout_files):
        obs, actions, rewards, done = load_rollout(
            rollout_path,
            source_frames_per_rollout=args.source_frames_per_rollout,
            discard_start_frames=args.discard_start_frames,
            cached_frames_per_rollout=args.frames_per_rollout,
        )
        rollout_vae, rollout_clip = encode_rollout_frames(
            obs=obs,
            vae=vae,
            clip_model=clip_model,
            clip_processor=clip_processor,
            clip_device=clip_device,
            frame_batch_size=args.frame_batch_size,
            clip_feature_source=args.clip_feature_source,
            vae_latent_source=args.vae_latent_source,
            normalize_clip_features=args.normalize_clip_features,
        )

        if caches is None:
            caches = create_memmaps(
                output_dir=output_dir,
                num_rollouts=len(rollout_files),
                frames_per_rollout=args.frames_per_rollout,
                vae_dim=int(rollout_vae.shape[-1]),
                clip_dim=int(rollout_clip.shape[-1]),
                fused_dim=int(cbp.output_dim),
                action_dim=int(actions.shape[-1]),
            )

        caches["vae_latents"][rollout_index] = rollout_vae
        caches["clip_features"][rollout_index] = rollout_clip
        rollout_fused = cbp(
            tf.convert_to_tensor(rollout_vae.astype(np.float32), dtype=tf.float32),
            tf.convert_to_tensor(rollout_clip.astype(np.float32), dtype=tf.float32),
        ).numpy().astype(np.float16)
        caches["fused_latents"][rollout_index] = rollout_fused
        caches["actions"][rollout_index] = actions
        caches["rewards"][rollout_index, :, 0] = rewards.astype(np.float16)
        caches["done"][rollout_index, :, 0] = done.astype(np.float16)

        if rollout_index == 0 or (rollout_index + 1) % 50 == 0 or rollout_index + 1 == len(rollout_files):
            log.info("Encoded rollouts: %d/%d", rollout_index + 1, len(rollout_files))

    if caches is None:
        raise RuntimeError("No rollout features were written.")

    for cache in caches.values():
        cache.flush()

    metadata = build_metadata(
        args=args,
        model_args=model_args,
        rollout_dir=rollout_dir,
        rollout_files=rollout_files,
        vae_dim=int(caches["vae_latents"].shape[-1]),
        clip_dim=int(caches["clip_features"].shape[-1]),
        fused_dim=int(caches["fused_latents"].shape[-1]),
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info(
        "Prepared %d rollouts x %d chunks = %d total sequences",
        metadata["num_rollouts"],
        metadata["chunks_per_rollout"],
        metadata["total_chunks"],
    )
    log.info("Saved rollout feature cache to %s", output_dir)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config_path", default="configs/carracing.config")
    known_args, _ = config_parser.parse_known_args()

    config = {}
    config_path = Path(known_args.config_path)
    if config_path.exists():
        config = load_config(config_path)

    parser = argparse.ArgumentParser(
        description="Encode rollout frames with frozen VAE and CLIP models and cache the features for RNN training."
    )
    parser.add_argument("--config_path", default=str(config_path))
    parser.add_argument("--rollout_dir", default=str(default_rollouts_dir()))
    parser.add_argument("--rollout_glob", default=DEFAULT_ROLLOUT_GLOB)
    parser.add_argument("--output_dir", default=None, help="Defaults to artifacts/world_models/<exp>/<env>/series")
    parser.add_argument("--clip_checkpoint", default=None, help="Path to merged_final or lora_final from CLIP fine-tuning.")
    parser.add_argument("--vae_checkpoint", default=None, help="Optional explicit path to the tf_vae SavedModel directory.")
    parser.add_argument(
        "--reconstruction_dir",
        default=str(default_reconstruction_dir(DEFAULT_RECONSTRUCTION_RUN)),
        help="Directory containing cbp_state.npz from the reconstruction run that defines the fused representation.",
    )
    parser.add_argument("--env_name", default=config.get("env_name", "CarRacing-v0"))
    parser.add_argument("--exp_name", default=config.get("exp_name", "WorldModels"))
    parser.add_argument("--z_size", type=int, default=int(config.get("z_size", 32)))
    parser.add_argument("--num_rollouts", type=int, default=6000)
    parser.add_argument("--frames_per_rollout", type=int, default=int(config.get("max_frames", 1000)))
    parser.add_argument("--discard_start_frames", type=int, default=50)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--frame_batch_size", type=int, default=128)
    parser.add_argument("--clip_device", default="auto", help="auto, cpu, or cuda[:index].")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--clip_feature_source",
        choices=["image_features", "pooler_output"],
        default="image_features",
    )
    parser.add_argument(
        "--vae_latent_source",
        choices=["mean", "sample"],
        default="mean",
    )
    parser.add_argument("--normalize_clip_features", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    preprocess_rollouts(parse_args())
