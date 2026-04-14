import argparse
import csv
import json
import logging
import math
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

from vae.vae import CVAE


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_SHARD_GLOB = "shard-*-caption-*.tar"


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


def ensure_supported_cuda_device(device: torch.device) -> None:
    if device.type != "cuda":
        return

    device_cc = torch.cuda.get_device_capability(device)
    device_arch = f"sm_{device_cc[0]}{device_cc[1]}"
    supported_arches = set(torch.cuda.get_arch_list())

    if supported_arches and device_arch not in supported_arches:
        supported = ", ".join(sorted(supported_arches))
        raise RuntimeError(
            "Installed PyTorch build is not compatible with the active GPU. "
            f"Found {torch.cuda.get_device_name(device)} ({device_arch}), but this build only supports: {supported}."
        )


def _load_saved_model_weights(model: tf.keras.Model, path: Path) -> None:
    saved = tf.saved_model.load(str(path))
    model.set_weights([var.numpy() for var in saved.variables])


def normalize_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.asarray(image).astype(np.uint8)
    return image


def resize_rgb(image: np.ndarray, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    pil_image = Image.fromarray(normalize_uint8_image(image), mode="RGB")
    if pil_image.size != size:
        pil_image = pil_image.resize(size, resample=Image.BILINEAR)
    return np.asarray(pil_image, dtype=np.uint8)


class FrameDataset:
    def __init__(
        self,
        data_root: Path,
        shard_glob: str,
        max_samples: Optional[int],
        seed: int,
    ):
        self.data_root = data_root
        self.shard_glob = shard_glob
        self.samples: List[Tuple[str, str, str]] = []

        if data_root.is_file() and data_root.suffix == ".tar":
            self._index_tar(data_root)
        elif data_root.is_dir():
            shard_paths = sorted(data_root.glob(shard_glob))
            if shard_paths:
                for shard_path in shard_paths:
                    self._index_tar(shard_path)
            else:
                self._index_image_tree(data_root)
        else:
            raise FileNotFoundError(f"Data source not found: {data_root}")

        if not self.samples:
            raise RuntimeError(
                f"No images found in {data_root}. Expected caption shards matching '{shard_glob}' or image files."
            )

        if max_samples is not None and len(self.samples) > max_samples:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(len(self.samples), size=max_samples, replace=False)
            self.samples = [self.samples[int(index)] for index in np.sort(chosen)]

        log.info("Indexed %d frames from %s", len(self.samples), data_root)

    def _index_tar(self, tar_path: Path) -> None:
        with tarfile.open(tar_path, "r") as handle:
            for member in handle.getmembers():
                if not member.isfile() or not member.name.lower().endswith(".png"):
                    continue
                sample_key = Path(member.name).stem
                self.samples.append(("tar", str(tar_path), member.name if sample_key == member.name else member.name))

    def _index_image_tree(self, root: Path) -> None:
        for image_path in sorted(root.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            self.samples.append(("file", str(image_path), image_path.stem))

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, index: int) -> Tuple[np.ndarray, str]:
        source_kind, source_path, member_name = self.samples[index]

        if source_kind == "tar":
            with tarfile.open(source_path, "r") as handle:
                extracted = handle.extractfile(member_name)
                if extracted is None:
                    raise FileNotFoundError(f"Could not extract {member_name} from {source_path}")
                with Image.open(extracted) as image:
                    rgb = image.convert("RGB")
                    array = np.asarray(rgb, dtype=np.uint8)
            sample_key = Path(member_name).stem
        else:
            with Image.open(source_path) as image:
                rgb = image.convert("RGB")
                array = np.asarray(rgb, dtype=np.uint8)
            sample_key = Path(source_path).stem

        return resize_rgb(array), sample_key


class CompactBilinearPooling:
    def __init__(self, input_dim_a: int, input_dim_b: int, output_dim: int, seed: int, normalize: bool):
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b
        self.output_dim = output_dim
        self.normalize = normalize

        rng = np.random.default_rng(seed)
        self.hash_a, self.sign_a, self.sketch_a = self._build_sketch(input_dim_a, output_dim, rng)
        self.hash_b, self.sign_b, self.sketch_b = self._build_sketch(input_dim_b, output_dim, rng)

    @staticmethod
    def _build_sketch(input_dim: int, output_dim: int, rng: np.random.Generator):
        hash_indices = rng.integers(0, output_dim, size=input_dim, endpoint=False)
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=input_dim)
        sketch = np.zeros((input_dim, output_dim), dtype=np.float32)
        sketch[np.arange(input_dim), hash_indices] = signs
        return hash_indices.astype(np.int32), signs.astype(np.float32), tf.constant(sketch)

    def __call__(self, x_a: tf.Tensor, x_b: tf.Tensor) -> tf.Tensor:
        sketch_a = tf.matmul(x_a, self.sketch_a)
        sketch_b = tf.matmul(x_b, self.sketch_b)

        fft_a = tf.signal.fft(tf.cast(sketch_a, tf.complex64))
        fft_b = tf.signal.fft(tf.cast(sketch_b, tf.complex64))
        fused = tf.math.real(tf.signal.ifft(fft_a * fft_b))

        if self.normalize:
            fused = tf.sign(fused) * tf.sqrt(tf.abs(fused) + 1e-8)
            fused = tf.math.l2_normalize(fused, axis=-1)

        return fused

    def save(self, path: Path) -> None:
        np.savez(
            path,
            hash_a=self.hash_a,
            sign_a=self.sign_a,
            hash_b=self.hash_b,
            sign_b=self.sign_b,
            output_dim=np.asarray([self.output_dim], dtype=np.int32),
            normalize=np.asarray([int(self.normalize)], dtype=np.int32),
        )


def build_model_args(config: dict, overrides: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        env_name=overrides.env_name or config.get("env_name", "CarRacing-v0"),
        exp_name=overrides.exp_name or config.get("exp_name", "WorldModels"),
        z_size=int(overrides.z_size or config.get("z_size", 32)),
        vae_learning_rate=float(config.get("vae_learning_rate", 1e-4)),
        vae_kl_tolerance=float(config.get("vae_kl_tolerance", 0.5)),
    )


def find_vae_checkpoint_dir(model_args: SimpleNamespace, explicit_path: Optional[Path]) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"VAE checkpoint not found: {explicit_path}")
        return explicit_path

    base = SCRIPT_DIR / "results" / model_args.exp_name / model_args.env_name / "tf_vae"
    if base.exists():
        return base

    if model_args.env_name.startswith("CarRacing"):
        fallback = SCRIPT_DIR / "results" / model_args.exp_name / "CarRacing" / "tf_vae"
        if fallback.exists():
            return fallback
    if model_args.env_name.startswith("DoomTakeCover"):
        fallback = SCRIPT_DIR / "results" / model_args.exp_name / "DoomTakeCover" / "tf_vae"
        if fallback.exists():
            return fallback

    raise FileNotFoundError(
        f"Could not find VAE weights under results/{model_args.exp_name}/{model_args.env_name}/tf_vae"
    )


def default_clip_checkpoint_candidates() -> List[Path]:
    return [
        SCRIPT_DIR / "clip_finetune" / "merged_final",
        SCRIPT_DIR / "clip_finetune" / "lora_final",
        SCRIPT_DIR / "merged_final",
        SCRIPT_DIR / "lora_final",
    ]


def resolve_clip_checkpoint_dir(explicit_path: Optional[Path]) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"CLIP checkpoint not found: {explicit_path}")
        return explicit_path

    for candidate in default_clip_checkpoint_candidates():
        if candidate.exists():
            return candidate

    checked = "\n".join(str(path) for path in default_clip_checkpoint_candidates())
    raise FileNotFoundError(
        "Could not infer a fine-tuned CLIP checkpoint. Pass --clip_checkpoint explicitly.\n"
        f"Checked:\n{checked}"
    )


def load_clip_model(
    checkpoint_dir: Path,
    cache_dir: Optional[Path],
    local_files_only: bool,
) -> Tuple[CLIPModel, CLIPProcessor]:
    training_config_path = checkpoint_dir / "training_config.json"
    training_config = {}
    if training_config_path.exists():
        training_config = json.loads(training_config_path.read_text(encoding="utf-8"))

    load_kwargs = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        load_kwargs["cache_dir"] = str(cache_dir)
    if local_files_only:
        load_kwargs["local_files_only"] = True

    is_adapter = (checkpoint_dir / "adapter_config.json").exists()
    if is_adapter:
        if PeftModel is None:
            raise ImportError("peft is required to load adapter-only CLIP checkpoints")
        base_source = training_config.get("clip_model", "openai/clip-vit-base-patch32")
        base_model = CLIPModel.from_pretrained(base_source, **load_kwargs)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    else:
        model = CLIPModel.from_pretrained(checkpoint_dir, **load_kwargs)

    processor = CLIPProcessor.from_pretrained(checkpoint_dir, **load_kwargs)
    model.eval()
    return model, processor


def split_indices(count: int, val_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if count < 2:
        raise ValueError("Need at least 2 samples to create train/validation splits")

    indices = np.arange(count)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    val_count = max(1, int(round(count * val_split)))
    val_count = min(val_count, count - 1)
    val_indices = np.sort(indices[:val_count])
    train_indices = np.sort(indices[val_count:])
    return train_indices, val_indices


def iter_batches(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int, epoch: int):
    ordered = np.array(indices, copy=True)
    if shuffle:
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(ordered)

    for start in range(0, len(ordered), batch_size):
        yield ordered[start:start + batch_size]


def load_batch_images(dataset: FrameDataset, batch_indices: Sequence[int]) -> Tuple[np.ndarray, List[str]]:
    images = []
    keys = []
    for index in batch_indices:
        image, key = dataset.load_image(int(index))
        images.append(image)
        keys.append(key)
    batch = np.stack(images, axis=0).astype(np.float32) / 255.0
    return batch, keys


def to_pil_images(batch_images: np.ndarray) -> List[Image.Image]:
    uint8_images = np.clip(batch_images * 255.0, 0.0, 255.0).astype(np.uint8)
    return [Image.fromarray(image, mode="RGB") for image in uint8_images]


def extract_clip_features(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    batch_images: np.ndarray,
    device: torch.device,
    feature_source: str,
    normalize_features: bool,
) -> np.ndarray:
    pil_images = to_pil_images(batch_images)
    encoded = clip_processor(images=pil_images, return_tensors="pt")
    pixel_values = encoded["pixel_values"].to(device)

    with torch.no_grad():
        if feature_source == "image_features":
            features = clip_model.get_image_features(pixel_values=pixel_values)
        else:
            outputs = clip_model.vision_model(pixel_values=pixel_values)
            features = outputs.pooler_output
        if normalize_features:
            features = torch.nn.functional.normalize(features, dim=-1)

    return features.detach().cpu().float().numpy()


def extract_vae_latents(vae: CVAE, batch_images: np.ndarray, latent_source: str) -> tf.Tensor:
    inputs = tf.convert_to_tensor(batch_images, dtype=tf.float32)
    if latent_source == "sample":
        return tf.cast(vae.encode(inputs), tf.float32)
    mean, _ = vae.encode_mu_logvar(inputs)
    return tf.cast(mean, tf.float32)


def reconstruction_loss(targets: tf.Tensor, predictions: tf.Tensor) -> tf.Tensor:
    per_example = tf.reduce_mean(tf.math.squared_difference(targets, predictions), axis=(1, 2, 3))
    return tf.reduce_mean(per_example)


def save_history_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_preview_grid(
    originals: np.ndarray,
    baseline_recon: np.ndarray,
    fused_recon: np.ndarray,
    sample_keys: Sequence[str],
    output_path: Path,
) -> None:
    originals_uint8 = np.clip(originals * 255.0, 0.0, 255.0).astype(np.uint8)
    baseline_uint8 = np.clip(baseline_recon * 255.0, 0.0, 255.0).astype(np.uint8)
    fused_uint8 = np.clip(fused_recon * 255.0, 0.0, 255.0).astype(np.uint8)

    tile_width = 64
    tile_height = 64
    rows = len(sample_keys)
    grid = Image.new("RGB", (tile_width * 3, tile_height * rows), color=(255, 255, 255))

    for row in range(rows):
        grid.paste(Image.fromarray(originals_uint8[row], mode="RGB"), (0, row * tile_height))
        grid.paste(Image.fromarray(baseline_uint8[row], mode="RGB"), (tile_width, row * tile_height))
        grid.paste(Image.fromarray(fused_uint8[row], mode="RGB"), (tile_width * 2, row * tile_height))

    grid.save(output_path)

    labels = {
        "columns": ["original", "vae_reconstruction", "fused_reconstruction"],
        "sample_keys": list(sample_keys),
    }
    output_path.with_suffix(".json").write_text(json.dumps(labels, indent=2), encoding="utf-8")


def evaluate(
    dataset: FrameDataset,
    indices: np.ndarray,
    batch_size: int,
    vae: CVAE,
    projection: tf.keras.Sequential,
    cbp: CompactBilinearPooling,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    clip_device: torch.device,
    feature_source: str,
    normalize_clip_features: bool,
    latent_source: str,
    num_preview: int,
) -> dict:
    fused_losses = []
    baseline_losses = []
    preview = None

    for batch_indices in iter_batches(indices, batch_size=batch_size, shuffle=False, seed=0, epoch=0):
        batch_images, keys = load_batch_images(dataset, batch_indices)
        target_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
        vae_latents = extract_vae_latents(vae, batch_images, latent_source)
        clip_features = extract_clip_features(
            clip_model,
            clip_processor,
            batch_images,
            clip_device,
            feature_source,
            normalize_clip_features,
        )
        fused_latents = cbp(vae_latents, tf.convert_to_tensor(clip_features, dtype=tf.float32))

        baseline_recon = vae.decode(vae_latents)
        fused_recon = vae.decode(projection(fused_latents, training=False))

        baseline_losses.append(float(reconstruction_loss(target_images, baseline_recon).numpy()))
        fused_losses.append(float(reconstruction_loss(target_images, fused_recon).numpy()))

        if preview is None:
            take = min(num_preview, batch_images.shape[0])
            preview = {
                "keys": keys[:take],
                "images": batch_images[:take],
                "baseline": baseline_recon.numpy()[:take],
                "fused": fused_recon.numpy()[:take],
            }

    return {
        "baseline_recon_loss": float(np.mean(baseline_losses)),
        "fused_recon_loss": float(np.mean(fused_losses)),
        "preview": preview,
    }


def train(args: argparse.Namespace) -> None:
    configure_tensorflow_memory_growth()

    config = load_config(Path(args.config_path)) if Path(args.config_path).exists() else {}
    model_args = build_model_args(config, args)

    data_root = resolve_path(args.data_root)
    if data_root is None:
        raise ValueError("--data_root is required")
    clip_checkpoint = resolve_clip_checkpoint_dir(resolve_path(args.clip_checkpoint))
    vae_checkpoint = find_vae_checkpoint_dir(model_args, resolve_path(args.vae_checkpoint))

    output_dir = resolve_path(args.output_dir)
    if output_dir is None:
        raise ValueError("--output_dir is required")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FrameDataset(
        data_root=data_root,
        shard_glob=args.shard_glob,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    train_indices, val_indices = split_indices(len(dataset), args.val_split, args.seed)

    vae = CVAE(model_args)
    _load_saved_model_weights(vae, vae_checkpoint)
    vae.trainable = False
    vae.inference_net_base.trainable = False
    vae.mu_net.trainable = False
    vae.logvar_net.trainable = False
    vae.generative_net.trainable = False

    clip_device = torch.device(args.clip_device if args.clip_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    ensure_supported_cuda_device(clip_device)
    clip_model, clip_processor = load_clip_model(
        checkpoint_dir=clip_checkpoint,
        cache_dir=resolve_path(args.hf_cache_dir),
        local_files_only=args.local_files_only,
    )
    clip_model.to(clip_device)

    sample_images, _ = load_batch_images(dataset, train_indices[: min(args.batch_size, len(train_indices))])
    sample_vae_latents = extract_vae_latents(vae, sample_images, args.vae_latent_source)
    sample_clip_features = extract_clip_features(
        clip_model,
        clip_processor,
        sample_images,
        clip_device,
        args.clip_feature_source,
        args.normalize_clip_features,
    )

    cbp = CompactBilinearPooling(
        input_dim_a=int(sample_vae_latents.shape[-1]),
        input_dim_b=int(sample_clip_features.shape[-1]),
        output_dim=args.fusion_dim,
        seed=args.seed,
        normalize=args.normalize_fused_features,
    )
    projection = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(args.fusion_dim,)),
            tf.keras.layers.Dense(model_args.z_size, name="fusion_projection"),
        ]
    )
    projection(tf.zeros((1, args.fusion_dim), dtype=tf.float32))
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    run_config = {
        "config_path": str(Path(args.config_path).resolve()),
        "data_root": str(data_root),
        "clip_checkpoint": str(clip_checkpoint),
        "vae_checkpoint": str(vae_checkpoint),
        "output_dir": str(output_dir),
        "train_size": int(len(train_indices)),
        "val_size": int(len(val_indices)),
        "z_size": int(model_args.z_size),
        "clip_feature_dim": int(sample_clip_features.shape[-1]),
        "fusion_dim": int(args.fusion_dim),
        "clip_feature_source": args.clip_feature_source,
        "vae_latent_source": args.vae_latent_source,
        "normalize_clip_features": bool(args.normalize_clip_features),
        "normalize_fused_features": bool(args.normalize_fused_features),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "seed": int(args.seed),
        "max_samples": None if args.max_samples is None else int(args.max_samples),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    cbp.save(output_dir / "cbp_state.npz")

    history_rows: List[dict] = []
    best_val_loss = math.inf

    for epoch in range(args.epochs):
        train_losses = []

        for batch_indices in iter_batches(train_indices, args.batch_size, shuffle=True, seed=args.seed, epoch=epoch):
            batch_images, _ = load_batch_images(dataset, batch_indices)
            target_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
            vae_latents = extract_vae_latents(vae, batch_images, args.vae_latent_source)
            clip_features = extract_clip_features(
                clip_model,
                clip_processor,
                batch_images,
                clip_device,
                args.clip_feature_source,
                args.normalize_clip_features,
            )
            fused_latents = cbp(vae_latents, tf.convert_to_tensor(clip_features, dtype=tf.float32))

            with tf.GradientTape() as tape:
                projected_z = projection(fused_latents, training=True)
                reconstructions = vae.decode(projected_z)
                loss = reconstruction_loss(target_images, reconstructions)

            gradients = tape.gradient(loss, projection.trainable_variables)
            optimizer.apply_gradients(zip(gradients, projection.trainable_variables))
            train_losses.append(float(loss.numpy()))

        eval_metrics = evaluate(
            dataset=dataset,
            indices=val_indices,
            batch_size=args.batch_size,
            vae=vae,
            projection=projection,
            cbp=cbp,
            clip_model=clip_model,
            clip_processor=clip_processor,
            clip_device=clip_device,
            feature_source=args.clip_feature_source,
            normalize_clip_features=args.normalize_clip_features,
            latent_source=args.vae_latent_source,
            num_preview=args.num_preview,
        )

        row = {
            "epoch": epoch,
            "train_recon_loss": float(np.mean(train_losses)),
            "val_fused_recon_loss": float(eval_metrics["fused_recon_loss"]),
            "val_vae_recon_loss": float(eval_metrics["baseline_recon_loss"]),
        }
        history_rows.append(row)
        save_history_csv(output_dir / "history.csv", history_rows)

        log.info(
            "Epoch %d/%d train=%.6f val_fused=%.6f val_vae=%.6f",
            epoch + 1,
            args.epochs,
            row["train_recon_loss"],
            row["val_fused_recon_loss"],
            row["val_vae_recon_loss"],
        )

        preview = eval_metrics["preview"]
        if preview is not None and ((epoch + 1) % args.save_preview_every == 0 or epoch == args.epochs - 1):
            make_preview_grid(
                originals=preview["images"],
                baseline_recon=preview["baseline"],
                fused_recon=preview["fused"],
                sample_keys=preview["keys"],
                output_path=output_dir / f"preview_epoch_{epoch:04d}.png",
            )

        if row["val_fused_recon_loss"] < best_val_loss:
            best_val_loss = row["val_fused_recon_loss"]
            projection.save_weights(output_dir / "best_projection.weights.h5")
            preview = eval_metrics["preview"]
            if preview is not None:
                make_preview_grid(
                    originals=preview["images"],
                    baseline_recon=preview["baseline"],
                    fused_recon=preview["fused"],
                    sample_keys=preview["keys"],
                    output_path=output_dir / "best_preview.png",
                )

    projection.save_weights(output_dir / "last_projection.weights.h5")
    summary = {
        "best_val_fused_recon_loss": float(best_val_loss),
        "last_train_recon_loss": history_rows[-1]["train_recon_loss"],
        "last_val_fused_recon_loss": history_rows[-1]["val_fused_recon_loss"],
        "last_val_vae_recon_loss": history_rows[-1]["val_vae_recon_loss"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("Finished. Best fused validation reconstruction loss: %.6f", best_val_loss)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config_path", default="configs/carracing.config")
    known_args, _ = config_parser.parse_known_args()

    config = {}
    config_path = Path(known_args.config_path)
    if config_path.exists():
        config = load_config(config_path)

    parser = argparse.ArgumentParser(
        description="Fuse frozen VAE and CLIP latents with compact bilinear pooling and train a linear projection for frame reconstruction."
    )
    parser.add_argument("--config_path", default=str(config_path))
    parser.add_argument("--data_root", default="outputs", help="Caption shard directory, a specific .tar shard, or an image directory.")
    parser.add_argument("--shard_glob", default=DEFAULT_SHARD_GLOB)
    parser.add_argument("--clip_checkpoint", default=None, help="Path to merged_final or lora_final from CLIP fine-tuning.")
    parser.add_argument("--vae_checkpoint", default=None, help="Optional explicit path to the tf_vae SavedModel directory.")
    parser.add_argument("--output_dir", default="fusion_reconstruction_runs/default")
    parser.add_argument("--env_name", default=config.get("env_name", "CarRacing-v0"))
    parser.add_argument("--exp_name", default=config.get("exp_name", "WorldModels"))
    parser.add_argument("--z_size", type=int, default=int(config.get("z_size", 32)))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--fusion_dim", type=int, default=4096)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip_device", default="auto", help="auto, cpu, or cuda[:index].")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--clip_feature_source",
        choices=["image_features", "pooler_output"],
        default="image_features",
        help="Which CLIP image representation to fuse with the VAE latent.",
    )
    parser.add_argument(
        "--vae_latent_source",
        choices=["mean", "sample"],
        default="mean",
        help="Use the VAE encoder mean or a sampled latent code.",
    )
    parser.add_argument("--normalize_clip_features", action="store_true")
    parser.add_argument("--normalize_fused_features", action="store_true")
    parser.add_argument("--save_preview_every", type=int, default=1)
    parser.add_argument("--num_preview", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())