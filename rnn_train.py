"""
Train the MDN-RNN on chunked sequences of CBP-fused frame representations.

Each cached latent is already the 544-d fused representation used by the
reconstruction stack, so the RNN is trained directly in that space.
"""

import argparse
import json
import logging
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from rnn.rnn import MDNRNN


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    config = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.split("#", 1)[0].strip()
            if key:
                config[key] = value
    return config


def resolve_path(path_like: Optional[str]) -> Optional[Path]:
    if path_like is None:
        return None
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def configure_tensorflow_memory_growth() -> None:
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as exc:
        log.warning("Could not enable TensorFlow memory growth: %s", exc)


@dataclass
class FeatureStore:
    metadata: dict
    fused_latents: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    done: np.ndarray

    @property
    def num_rollouts(self) -> int:
        return int(self.fused_latents.shape[0])

    @property
    def frames_per_rollout(self) -> int:
        return int(self.actions.shape[1])

    @property
    def chunk_size(self) -> int:
        return int(self.metadata["chunk_size"])

    @property
    def chunks_per_rollout(self) -> int:
        return int(self.metadata["chunks_per_rollout"])

    @property
    def vae_dim(self) -> int:
        return int(self.metadata.get("vae_feature_dim", 0))

    @property
    def clip_dim(self) -> int:
        return int(self.metadata.get("clip_feature_dim", 0))

    @property
    def fused_dim(self) -> int:
        return int(self.fused_latents.shape[-1])

    @property
    def total_chunks(self) -> int:
        return int(self.num_rollouts * self.chunks_per_rollout)


def default_series_dir(exp_name: str, env_name: str) -> Path:
    return SCRIPT_DIR / "results" / exp_name / env_name / "series"


def default_model_dir(exp_name: str, env_name: str) -> Path:
    return SCRIPT_DIR / "results" / exp_name / env_name / "tf_rnn"


def default_initial_z_dir(exp_name: str, env_name: str) -> Path:
    return SCRIPT_DIR / "results" / exp_name / env_name / "tf_initial_z"


def load_feature_store(series_dir: Path) -> FeatureStore:
    metadata_path = series_dir / "metadata.json"
    fused_latents_path = series_dir / "fused_latents.npy"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Series cache in {series_dir} is missing: metadata")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    representation_type = metadata.get("fused_representation_type")
    if representation_type != "cbp_fused":
        raise ValueError(
            f"Series cache in {series_dir} uses fused_representation_type={representation_type!r}. "
            "Re-run series.py to build a CBP-fused cache for the current RNN pipeline."
        )

    if fused_latents_path.exists():
        required_paths = {
            "metadata": metadata_path,
            "fused_latents": fused_latents_path,
            "actions": series_dir / "actions.npy",
            "rewards": series_dir / "rewards.npy",
            "done": series_dir / "done.npy",
        }
        missing = [name for name, path in required_paths.items() if not path.exists()]
        if missing:
            missing_list = ", ".join(missing)
            raise FileNotFoundError(f"Series cache in {series_dir} is missing: {missing_list}")

        return FeatureStore(
            metadata=metadata,
            fused_latents=np.load(required_paths["fused_latents"], mmap_mode="r"),
            actions=np.load(required_paths["actions"], mmap_mode="r"),
            rewards=np.load(required_paths["rewards"], mmap_mode="r"),
            done=np.load(required_paths["done"], mmap_mode="r"),
        )

    raise FileNotFoundError(
        f"Series cache in {series_dir} is missing fused_latents.npy. Re-run series.py to build the optimized fused-latent cache."
    )


def split_chunk_indices(total_chunks: int, val_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(total_chunks, dtype=np.int32)
    if val_split <= 0.0:
        return indices, np.array([], dtype=np.int32)
    if total_chunks < 2:
        raise ValueError("Need at least two chunk sequences to create a validation split")

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_count = max(1, int(round(total_chunks * val_split)))
    val_count = min(val_count, total_chunks - 1)
    val_indices = np.sort(indices[:val_count])
    train_indices = np.sort(indices[val_count:])
    return train_indices, val_indices


def sample_chunk_batch(
    store: FeatureStore,
    chunk_indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    selected_chunks = rng.choice(chunk_indices, size=batch_size, replace=True)
    batch_rollouts = selected_chunks // store.chunks_per_rollout
    batch_chunks = selected_chunks % store.chunks_per_rollout

    time_offsets = np.arange(store.chunk_size, dtype=np.int32)[None, :]
    current_steps = batch_chunks[:, None] * store.chunk_size + time_offsets
    next_steps = current_steps + 1
    rollout_rows = batch_rollouts[:, None]

    current_z = store.fused_latents[rollout_rows, current_steps].astype(np.float32)
    next_z = store.fused_latents[rollout_rows, next_steps].astype(np.float32)
    actions = store.actions[rollout_rows, current_steps].astype(np.float32)
    rewards = store.rewards[rollout_rows, current_steps].astype(np.float32)
    done = store.done[rollout_rows, current_steps].astype(np.float32)

    return {
        "current_z": current_z,
        "next_z": next_z,
        "actions": actions,
        "rewards": rewards,
        "done": done,
    }


def build_targets(args: argparse.Namespace, next_z: tf.Tensor, rewards: tf.Tensor, done: tf.Tensor) -> Dict[str, tf.Tensor]:
    targets = {"MDN": tf.concat([next_z, 1.0 - done], axis=2)}

    prev_alive = tf.concat(
        [
            tf.ones((args.rnn_batch_size, 1, 1), dtype=tf.float32),
            1.0 - done[:, :-1, :],
        ],
        axis=1,
    )
    if args.rnn_r_pred == 1:
        targets["r"] = tf.concat([rewards, prev_alive], axis=2)
    if args.rnn_d_pred == 1:
        targets["d"] = tf.concat([done, prev_alive], axis=2)
    return targets


def compute_losses(
    args: argparse.Namespace,
    rnn: MDNRNN,
    current_z: tf.Tensor,
    next_z: tf.Tensor,
    actions: tf.Tensor,
    rewards: tf.Tensor,
    done: tf.Tensor,
    training: bool,
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], tf.Tensor]:
    inputs = tf.concat([current_z, actions], axis=2)
    predictions = rnn(inputs, training=training)
    targets = build_targets(args, next_z, rewards, done)

    losses = {}
    total_loss = tf.constant(0.0, dtype=tf.float32)
    for loss_name, loss_fn in rnn.loss_fn.items():
        loss_value = tf.cast(loss_fn(targets[loss_name], predictions[loss_name]), tf.float32)
        losses[loss_name] = loss_value
        total_loss += loss_value

    return total_loss, losses, current_z[:, 0, :]


def batch_to_tensors(batch: Dict[str, np.ndarray]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    return (
        tf.convert_to_tensor(batch["current_z"], dtype=tf.float32),
        tf.convert_to_tensor(batch["next_z"], dtype=tf.float32),
        tf.convert_to_tensor(batch["actions"], dtype=tf.float32),
        tf.convert_to_tensor(batch["rewards"], dtype=tf.float32),
        tf.convert_to_tensor(batch["done"], dtype=tf.float32),
    )


def evaluate(
    args: argparse.Namespace,
    store: FeatureStore,
    chunk_indices: np.ndarray,
    rnn: MDNRNN,
    eval_step_fn,
    num_batches: int,
    seed: int,
) -> Optional[Dict[str, float]]:
    if chunk_indices.size == 0 or num_batches <= 0:
        return None

    rng = np.random.default_rng(seed)
    accumulated: Dict[str, float] = {"total": 0.0}
    for _ in range(num_batches):
        batch = sample_chunk_batch(store, chunk_indices, args.rnn_batch_size, rng)
        current_z, next_z, actions, rewards, done = batch_to_tensors(batch)
        total_loss, losses, _ = eval_step_fn(current_z, next_z, actions, rewards, done)
        accumulated["total"] += float(total_loss.numpy())
        for loss_name, loss_value in losses.items():
            accumulated[loss_name] = accumulated.get(loss_name, 0.0) + float(loss_value.numpy())

    for key in list(accumulated.keys()):
        accumulated[key] /= float(num_batches)
    return accumulated


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_scalar_summaries(
    summary_writer: Optional[tf.summary.SummaryWriter],
    step: int,
    metrics: Dict[str, float],
    prefix: str,
) -> Optional[tf.summary.SummaryWriter]:
    if summary_writer is None:
        return None

    try:
        with summary_writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(f"{prefix}/{name}", value, step=step)
    except Exception as exc:
        log.warning("Disabling TensorBoard logging because scalar summaries are unavailable: %s", exc)
        return None

    return summary_writer


def train(args: argparse.Namespace) -> None:
    configure_tensorflow_memory_growth()
    np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

    series_dir = resolve_path(args.series_dir) if args.series_dir else default_series_dir(args.exp_name, args.env_name)
    if series_dir is None:
        raise ValueError("Could not resolve series directory")
    store = load_feature_store(series_dir)

    cached_vae_z_dim = int(store.metadata["z_size"])
    fused_feature_dim = int(store.metadata.get("fused_feature_dim", store.fused_dim))
    args.z_size = fused_feature_dim

    args.rnn_max_seq_len = store.chunk_size
    args.rnn_input_seq_width = args.z_size + args.a_width

    log.info(
        "Training the RNN directly on CBP-fused latent codes with dim %d; the VAE decoder still consumes the projection output of dim %d.",
        fused_feature_dim,
        cached_vae_z_dim,
    )

    model_dir = resolve_path(args.output_dir) if args.output_dir else default_model_dir(args.exp_name, args.env_name)
    if model_dir is None:
        raise ValueError("Could not resolve model output directory")

    tensorboard_dir = model_dir.parent / f"{model_dir.name}_tensorboard"
    summary_writer: Optional[tf.summary.SummaryWriter] = tf.summary.create_file_writer(str(tensorboard_dir))

    rnn = MDNRNN(args=args)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.rnn_learning_rate,
        clipvalue=args.rnn_grad_clip,
    )
    rnn.optimizer = optimizer

    @tf.function(reduce_retracing=True)
    def train_step(current_z, next_z, actions, rewards, done):
        with tf.GradientTape() as tape:
            total_loss, losses, initial_z = compute_losses(
                args,
                rnn,
                current_z,
                next_z,
                actions,
                rewards,
                done,
                training=True,
            )
        gradients = tape.gradient(total_loss, rnn.trainable_variables)
        optimizer.apply_gradients(zip(gradients, rnn.trainable_variables))
        return total_loss, losses, initial_z

    @tf.function(reduce_retracing=True)
    def eval_step(current_z, next_z, actions, rewards, done):
        return compute_losses(
            args,
            rnn,
            current_z,
            next_z,
            actions,
            rewards,
            done,
            training=False,
        )

    checkpoint = tf.train.Checkpoint(rnn=rnn, optimizer=optimizer)
    checkpoint_dir = model_dir.parent / f"{model_dir.name}_training_checkpoints"
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir), max_to_keep=3)

    train_chunk_indices, val_chunk_indices = split_chunk_indices(store.total_chunks, args.val_split, args.seed)

    run_config = {
        **vars(args),
        "series_dir": str(series_dir),
        "series_metadata": store.metadata,
        "cached_vae_z_dim": cached_vae_z_dim,
        "rnn_latent_dim": int(fused_feature_dim),
        "fused_feature_dim": fused_feature_dim,
        "fused_representation_type": str(store.metadata["fused_representation_type"]),
        "total_chunks": int(store.total_chunks),
        "train_chunks": int(train_chunk_indices.size),
        "val_chunks": int(val_chunk_indices.size),
        "rnn_input_seq_width": int(args.rnn_input_seq_width),
    }

    log.info(
        "Loaded %d rollouts -> %d total sequences (%d frames / %d per chunk, latent_dim=%d, input_width=%d)",
        store.num_rollouts,
        store.total_chunks,
        store.frames_per_rollout,
        store.chunk_size,
        fused_feature_dim,
        args.rnn_input_seq_width,
    )

    rng = np.random.default_rng(args.seed)
    best_val_total = math.inf
    best_step: Optional[int] = None
    last_metrics: Dict[str, float] = {}

    start_time = time.time()
    for step in range(args.rnn_num_steps):
        current_learning_rate = (
            (args.rnn_learning_rate - args.rnn_min_learning_rate) * (args.rnn_decay_rate ** step)
            + args.rnn_min_learning_rate
        )
        optimizer.learning_rate.assign(current_learning_rate)

        batch = sample_chunk_batch(store, train_chunk_indices, args.rnn_batch_size, rng)
        current_z, next_z, actions, rewards, done = batch_to_tensors(batch)
        total_loss, losses, initial_z = train_step(current_z, next_z, actions, rewards, done)

        train_metrics = {"total": float(total_loss.numpy())}
        for loss_name, loss_value in losses.items():
            train_metrics[loss_name] = float(loss_value.numpy())
        last_metrics = dict(train_metrics)

        summary_writer = write_scalar_summaries(
            summary_writer,
            step,
            {
                "total": train_metrics["total"],
                "learning_rate": current_learning_rate,
                **{loss_name: float(loss_value.numpy()) for loss_name, loss_value in losses.items()},
                "initial_z_norm": float(tf.reduce_mean(tf.norm(initial_z, axis=1)).numpy()),
            },
            "train",
        )

        if step > 0 and step % args.log_every == 0:
            elapsed = time.time() - start_time
            start_time = time.time()
            message = f"step: {step}, train_time_taken: {elapsed:.4f}, lr: {current_learning_rate:.6f}"
            for loss_name, loss_value in train_metrics.items():
                message += f", train_{loss_name}: {loss_value:.4f}"

            val_metrics = evaluate(
                args=args,
                store=store,
                chunk_indices=val_chunk_indices,
                rnn=rnn,
                eval_step_fn=eval_step,
                num_batches=args.val_batches,
                seed=args.seed + step,
            )
            if val_metrics is not None:
                for metric_name, metric_value in val_metrics.items():
                    message += f", val_{metric_name}: {metric_value:.4f}"
                summary_writer = write_scalar_summaries(summary_writer, step, val_metrics, "val")

                if val_metrics["total"] < best_val_total:
                    best_val_total = val_metrics["total"]
                    best_step = step

            log.info(message)

        if step > 0 and step % args.save_every == 0:
            checkpoint_manager.save(checkpoint_number=step)

    final_checkpoint_path = checkpoint_manager.save(checkpoint_number=args.rnn_num_steps)
    log.info("Saved final training checkpoint to %s", final_checkpoint_path)

    reset_directory(model_dir)
    tf.saved_model.save(rnn, str(model_dir))
    (model_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    log.warning(
        "Skipping tf_initial_z export because the RNN is trained on CBP-fused latents (dim %d) rather than VAE latents (dim %d).",
        fused_feature_dim,
        cached_vae_z_dim,
    )

    training_summary = {
        "final_step": int(args.rnn_num_steps - 1),
        "best_val_total": None if best_step is None else float(best_val_total),
        "best_step": None if best_step is None else int(best_step),
        "last_train_metrics": last_metrics,
        "series_dir": str(series_dir),
        "cached_vae_z_dim": int(cached_vae_z_dim),
        "rnn_latent_dim": int(fused_feature_dim),
        "fused_feature_dim": int(fused_feature_dim),
        "fused_representation_type": str(store.metadata["fused_representation_type"]),
        "total_chunks": int(store.total_chunks),
        "projection_output_dim": int(args.z_size),
        "latent_source": f"CBP(vae, clip) -> RNN({fused_feature_dim})",
    }
    (model_dir / "training_summary.json").write_text(json.dumps(training_summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config_path", default="configs/carracing.config")
    known_args, _ = config_parser.parse_known_args()

    config = {}
    config_path = Path(known_args.config_path)
    if config_path.exists():
        config = load_config(config_path)

    parser = argparse.ArgumentParser(
        description="Train the MDN-RNN on chunked CBP-fused rollout features."
    )
    parser.add_argument("--config_path", default=str(config_path))
    parser.add_argument("--series_dir", default=None, help="Defaults to results/<exp>/<env>/series")
    parser.add_argument("--output_dir", default=None, help="Defaults to results/<exp>/<env>/tf_rnn")
    parser.add_argument("--initial_z_dir", default=None, help="Defaults to results/<exp>/<env>/tf_initial_z")
    parser.add_argument("--env_name", default=config.get("env_name", "CarRacing-v0"))
    parser.add_argument("--exp_name", default=config.get("exp_name", "WorldModels"))
    parser.add_argument(
        "--z_size",
        type=int,
        default=int(config.get("z_size", 32)),
        help="Legacy VAE latent size from config. The trained RNN latent size is derived from the CBP fused feature dimension.",
    )
    parser.add_argument("--a_width", type=int, default=int(config.get("a_width", 3)))
    parser.add_argument("--state_space", type=int, default=int(config.get("state_space", 1)))
    parser.add_argument("--rnn_size", type=int, default=int(config.get("rnn_size", 256)))
    parser.add_argument("--rnn_num_steps", type=int, default=int(config.get("rnn_num_steps", 4000)))
    parser.add_argument("--rnn_learning_rate", type=float, default=float(config.get("rnn_learning_rate", 1e-3)))
    parser.add_argument("--rnn_min_learning_rate", type=float, default=float(config.get("rnn_min_learning_rate", 1e-5)))
    parser.add_argument("--rnn_decay_rate", type=float, default=float(config.get("rnn_decay_rate", 1.0)))
    parser.add_argument("--rnn_grad_clip", type=float, default=float(config.get("rnn_grad_clip", 1.0)))
    parser.add_argument("--rnn_num_mixture", type=int, default=int(config.get("rnn_num_mixture", 5)))
    parser.add_argument("--rnn_r_pred", type=int, default=int(config.get("rnn_r_pred", 0)))
    parser.add_argument("--rnn_d_pred", type=int, default=int(config.get("rnn_d_pred", 0)))
    parser.add_argument("--rnn_batch_size", type=int, default=512)
    parser.add_argument("--rnn_d_true_weight", type=float, default=float(config.get("rnn_d_true_weight", 1.0)))
    parser.add_argument("--rnn_temperature", type=float, default=float(config.get("rnn_temperature", 1.0)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--val_batches", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--initial_z_count", type=int, default=1000)
    args = parser.parse_args()
    args.rnn_max_seq_len = None
    args.rnn_input_seq_width = None
    return args


if __name__ == "__main__":
    train(parse_args())