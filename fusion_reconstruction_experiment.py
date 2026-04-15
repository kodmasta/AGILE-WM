import argparse
import csv
import hashlib
import json
import logging
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import torch
from PIL import Image, ImageDraw, ImageFont
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
DEFAULT_SHARD_GLOB = "shard-*.tar"


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
                f"No images found in {data_root}. Expected WebDataset shards matching '{shard_glob}' or image files."
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


@dataclass
class CachedSplitData:
    name: str
    indices: np.ndarray
    images: np.ndarray
    vae_latents: np.ndarray
    clip_features: np.ndarray
    keys: List[str]

    @property
    def size(self) -> int:
        return int(self.indices.shape[0])


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
        Path.home() / "clip_finetune" / "merged_final",
        Path.home() / "clip_finetune" / "lora_final",
        SCRIPT_DIR / "merged_final",
        SCRIPT_DIR / "lora_final",
    ]


def fallback_home_clip_checkpoint(explicit_path: Path) -> Optional[Path]:
    try:
        relative_to_repo = explicit_path.relative_to(SCRIPT_DIR)
    except ValueError:
        return None

    parts = relative_to_repo.parts
    if not parts or parts[0] != "clip_finetune":
        return None

    home_candidate = Path.home() / relative_to_repo
    if home_candidate.exists():
        return home_candidate
    return None


def resolve_clip_checkpoint_dir(explicit_path: Optional[Path]) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            home_fallback = fallback_home_clip_checkpoint(explicit_path)
            if home_fallback is not None:
                return home_fallback
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

    def as_feature_tensor(output, source: str) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            pooled = output.pooler_output
            if (
                source == "image_features"
                and hasattr(clip_model, "visual_projection")
                and getattr(clip_model.visual_projection, "in_features", None) == pooled.shape[-1]
            ):
                return clip_model.visual_projection(pooled)
            return pooled
        raise TypeError(f"Unsupported CLIP feature output type: {type(output)!r}")

    with torch.no_grad():
        if feature_source == "image_features":
            features = as_feature_tensor(clip_model.get_image_features(pixel_values=pixel_values), feature_source)
        else:
            outputs = clip_model.vision_model(pixel_values=pixel_values)
            features = as_feature_tensor(outputs, feature_source)
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


def cache_paths(cache_dir: Path, split_name: str) -> dict:
    prefix = cache_dir / split_name
    return {
        "images": prefix.with_name(f"{split_name}_images.npy"),
        "vae_latents": prefix.with_name(f"{split_name}_vae_latents.npy"),
        "clip_features": prefix.with_name(f"{split_name}_clip_features.npy"),
        "indices": prefix.with_name(f"{split_name}_indices.npy"),
        "keys": prefix.with_name(f"{split_name}_keys.json"),
        "meta": prefix.with_name(f"{split_name}_cache_meta.json"),
    }


def indices_digest(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def cache_metadata_matches(existing_meta: dict, expected_meta: dict) -> bool:
    return all(existing_meta.get(key) == value for key, value in expected_meta.items())


def load_cached_split(cache_dir: Path, split_name: str) -> CachedSplitData:
    paths = cache_paths(cache_dir, split_name)
    images = np.load(paths["images"], mmap_mode="r")
    vae_latents = np.load(paths["vae_latents"], mmap_mode="r")
    clip_features = np.load(paths["clip_features"], mmap_mode="r")
    indices = np.load(paths["indices"])
    keys = json.loads(paths["keys"].read_text(encoding="utf-8"))
    return CachedSplitData(
        name=split_name,
        indices=indices,
        images=images,
        vae_latents=vae_latents,
        clip_features=clip_features,
        keys=keys,
    )


def prepare_cached_split(
    *,
    dataset: FrameDataset,
    indices: np.ndarray,
    split_name: str,
    cache_dir: Path,
    cache_metadata: dict,
    rebuild_cache: bool,
    batch_size: int,
    vae: CVAE,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    clip_device: torch.device,
    feature_source: str,
    normalize_clip_features: bool,
    latent_source: str,
) -> CachedSplitData:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir, split_name)
    required_paths = [
        paths["images"],
        paths["vae_latents"],
        paths["clip_features"],
        paths["indices"],
        paths["keys"],
        paths["meta"],
    ]

    if not rebuild_cache and all(path.exists() for path in required_paths):
        try:
            existing_meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
            if cache_metadata_matches(existing_meta, cache_metadata):
                log.info("Reusing %s cache from %s", split_name, cache_dir)
                return load_cached_split(cache_dir, split_name)
            log.info("Rebuilding %s cache because the cache configuration changed", split_name)
        except Exception as exc:
            log.info("Rebuilding %s cache because the existing cache could not be read: %s", split_name, exc)

    log.info("Building %s cache with %d samples in %s", split_name, len(indices), cache_dir)

    image_cache = None
    vae_cache = None
    clip_cache = None
    keys: List[str] = []
    write_offset = 0
    image_shape = None
    vae_dim = None
    clip_dim = None

    for batch_number, batch_indices in enumerate(
        iter_batches(indices, batch_size=batch_size, shuffle=False, seed=0, epoch=0),
        start=1,
    ):
        batch_images, batch_keys = load_batch_images(dataset, batch_indices)
        batch_vae_latents = extract_vae_latents(vae, batch_images, latent_source).numpy().astype(np.float32)
        batch_clip_features = extract_clip_features(
            clip_model,
            clip_processor,
            batch_images,
            clip_device,
            feature_source,
            normalize_clip_features,
        ).astype(np.float32)

        if image_cache is None:
            image_shape = batch_images.shape[1:]
            vae_dim = int(batch_vae_latents.shape[-1])
            clip_dim = int(batch_clip_features.shape[-1])
            image_cache = np.lib.format.open_memmap(
                str(paths["images"]),
                mode="w+",
                dtype=np.uint8,
                shape=(len(indices),) + image_shape,
            )
            vae_cache = np.lib.format.open_memmap(
                str(paths["vae_latents"]),
                mode="w+",
                dtype=np.float32,
                shape=(len(indices), vae_dim),
            )
            clip_cache = np.lib.format.open_memmap(
                str(paths["clip_features"]),
                mode="w+",
                dtype=np.float32,
                shape=(len(indices), clip_dim),
            )

        batch_size_actual = batch_images.shape[0]
        batch_end = write_offset + batch_size_actual
        image_cache[write_offset:batch_end] = np.clip(batch_images * 255.0, 0.0, 255.0).astype(np.uint8)
        vae_cache[write_offset:batch_end] = batch_vae_latents
        clip_cache[write_offset:batch_end] = batch_clip_features
        keys.extend(batch_keys)
        write_offset = batch_end

        if batch_number == 1 or write_offset == len(indices) or batch_number % 100 == 0:
            log.info("Cached %s samples: %d/%d", split_name, write_offset, len(indices))

    if image_cache is None or vae_cache is None or clip_cache is None:
        raise RuntimeError(f"Could not build {split_name} cache because no samples were processed")

    np.save(paths["indices"], np.asarray(indices, dtype=np.int64))
    paths["keys"].write_text(json.dumps(keys), encoding="utf-8")
    completed_meta = dict(cache_metadata)
    completed_meta.update(
        {
            "image_shape": list(image_shape),
            "vae_dim": vae_dim,
            "clip_dim": clip_dim,
            "cached_samples": int(write_offset),
        }
    )
    paths["meta"].write_text(json.dumps(completed_meta, indent=2, sort_keys=True), encoding="utf-8")

    del image_cache
    del vae_cache
    del clip_cache

    return load_cached_split(cache_dir, split_name)


def cached_images_to_float(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) / 255.0


def save_history_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def filter_preview_rows(preview: Optional[dict], preview_rows: Optional[Sequence[int]]) -> Optional[dict]:
    if preview is None or preview_rows is None:
        return preview

    max_rows = len(preview["keys"])
    selected_rows = [row for row in preview_rows if 0 <= row < max_rows]
    if not selected_rows:
        raise ValueError(
            f"None of the requested --preview_rows values are valid for a preview with {max_rows} row(s)."
        )

    return {
        "keys": [preview["keys"][row] for row in selected_rows],
        "images": preview["images"][selected_rows],
        "baseline": preview["baseline"][selected_rows],
        "fused": preview["fused"][selected_rows],
    }


def make_preview_grid(
    originals: np.ndarray,
    baseline_recon: np.ndarray,
    fused_recon: np.ndarray,
    sample_keys: Sequence[str],
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    originals_uint8 = np.clip(originals * 255.0, 0.0, 255.0).astype(np.uint8)
    baseline_uint8 = np.clip(baseline_recon * 255.0, 0.0, 255.0).astype(np.uint8)
    fused_uint8 = np.clip(fused_recon * 255.0, 0.0, 255.0).astype(np.uint8)

    tile_width = 64
    tile_height = 64
    title_height = 22 if title else 0
    header_height = 24
    rows = len(sample_keys)
    grid = Image.new(
        "RGB",
        (tile_width * 3, title_height + header_height + tile_height * rows),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()
    column_labels = ["GT", "VAE-only", "VAE+CLIP"]

    if title:
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (tile_width * 3 - text_width) // 2
        y = (title_height - text_height) // 2
        draw.text((x, y), title, fill=(0, 0, 0), font=font)

    for column, label in enumerate(column_labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = column * tile_width + (tile_width - text_width) // 2
        y = title_height + (header_height - text_height) // 2
        draw.text((x, y), label, fill=(0, 0, 0), font=font)

    for row in range(rows):
        y_offset = title_height + header_height + row * tile_height
        grid.paste(Image.fromarray(originals_uint8[row], mode="RGB"), (0, y_offset))
        grid.paste(Image.fromarray(baseline_uint8[row], mode="RGB"), (tile_width, y_offset))
        grid.paste(Image.fromarray(fused_uint8[row], mode="RGB"), (tile_width * 2, y_offset))

    grid.save(output_path)

    labels = {
        "title": title,
        "columns": ["GT", "VAE-only", "VAE+CLIP"],
        "sample_keys": list(sample_keys),
    }
    output_path.with_suffix(".json").write_text(json.dumps(labels, indent=2), encoding="utf-8")


def evaluate(
    cached_split: CachedSplitData,
    batch_size: int,
    vae: CVAE,
    projection: tf.keras.Sequential,
    cbp: CompactBilinearPooling,
    num_preview: int,
) -> dict:
    fused_losses = []
    baseline_losses = []
    preview = None

    cached_indices = np.arange(cached_split.size)
    for batch_positions in iter_batches(cached_indices, batch_size=batch_size, shuffle=False, seed=0, epoch=0):
        batch_images = cached_images_to_float(cached_split.images[batch_positions])
        target_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
        vae_latents = tf.convert_to_tensor(cached_split.vae_latents[batch_positions], dtype=tf.float32)
        clip_features = tf.convert_to_tensor(cached_split.clip_features[batch_positions], dtype=tf.float32)
        fused_latents = cbp(vae_latents, clip_features)

        baseline_recon = vae.decode(vae_latents)
        fused_recon = vae.decode(projection(fused_latents, training=False))

        baseline_losses.append(float(reconstruction_loss(target_images, baseline_recon).numpy()))
        fused_losses.append(float(reconstruction_loss(target_images, fused_recon).numpy()))

        if preview is None:
            take = min(num_preview, batch_images.shape[0])
            preview = {
                "keys": [cached_split.keys[int(pos)] for pos in batch_positions[:take]],
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
    cache_dir = resolve_path(args.cache_dir) if args.cache_dir else (output_dir / "feature_cache")
    cache_batch_size = args.cache_batch_size or args.batch_size

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

    common_cache_metadata = {
        "data_root": str(data_root),
        "shard_glob": args.shard_glob,
        "dataset_size": len(dataset),
        "max_samples": None if args.max_samples is None else int(args.max_samples),
        "seed": int(args.seed),
        "val_split": float(args.val_split),
        "clip_checkpoint": str(clip_checkpoint.resolve()),
        "vae_checkpoint": str(vae_checkpoint.resolve()),
        "clip_feature_source": args.clip_feature_source,
        "vae_latent_source": args.vae_latent_source,
        "normalize_clip_features": bool(args.normalize_clip_features),
    }
    train_cache = prepare_cached_split(
        dataset=dataset,
        indices=train_indices,
        split_name="train",
        cache_dir=cache_dir,
        cache_metadata={
            **common_cache_metadata,
            "split_name": "train",
            "indices_digest": indices_digest(train_indices),
        },
        rebuild_cache=args.rebuild_cache,
        batch_size=cache_batch_size,
        vae=vae,
        clip_model=clip_model,
        clip_processor=clip_processor,
        clip_device=clip_device,
        feature_source=args.clip_feature_source,
        normalize_clip_features=args.normalize_clip_features,
        latent_source=args.vae_latent_source,
    )
    val_cache = prepare_cached_split(
        dataset=dataset,
        indices=val_indices,
        split_name="val",
        cache_dir=cache_dir,
        cache_metadata={
            **common_cache_metadata,
            "split_name": "val",
            "indices_digest": indices_digest(val_indices),
        },
        rebuild_cache=args.rebuild_cache,
        batch_size=cache_batch_size,
        vae=vae,
        clip_model=clip_model,
        clip_processor=clip_processor,
        clip_device=clip_device,
        feature_source=args.clip_feature_source,
        normalize_clip_features=args.normalize_clip_features,
        latent_source=args.vae_latent_source,
    )
    clip_model.to("cpu")
    del clip_model
    del clip_processor
    if clip_device.type == "cuda":
        torch.cuda.empty_cache()

    cbp = CompactBilinearPooling(
        input_dim_a=int(train_cache.vae_latents.shape[-1]),
        input_dim_b=int(train_cache.clip_features.shape[-1]),
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
        "cache_dir": str(cache_dir),
        "cache_batch_size": int(cache_batch_size),
        "train_size": int(train_cache.size),
        "val_size": int(val_cache.size),
        "z_size": int(model_args.z_size),
        "clip_feature_dim": int(train_cache.clip_features.shape[-1]),
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
        "rebuild_cache": bool(args.rebuild_cache),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    cbp.save(output_dir / "cbp_state.npz")

    history_rows: List[dict] = []
    best_val_loss = math.inf
    best_epoch: Optional[int] = None
    train_positions = np.arange(train_cache.size)

    for epoch in range(args.epochs):
        train_losses = []

        for batch_positions in iter_batches(train_positions, args.batch_size, shuffle=True, seed=args.seed, epoch=epoch):
            batch_images = cached_images_to_float(train_cache.images[batch_positions])
            target_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
            vae_latents = tf.convert_to_tensor(train_cache.vae_latents[batch_positions], dtype=tf.float32)
            clip_features = tf.convert_to_tensor(train_cache.clip_features[batch_positions], dtype=tf.float32)
            fused_latents = cbp(vae_latents, clip_features)

            with tf.GradientTape() as tape:
                projected_z = projection(fused_latents, training=True)
                reconstructions = vae.decode(projected_z)
                loss = reconstruction_loss(target_images, reconstructions)

            gradients = tape.gradient(loss, projection.trainable_variables)
            optimizer.apply_gradients(zip(gradients, projection.trainable_variables))
            train_losses.append(float(loss.numpy()))

        eval_metrics = evaluate(
            cached_split=val_cache,
            batch_size=args.batch_size,
            vae=vae,
            projection=projection,
            cbp=cbp,
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
        preview = filter_preview_rows(preview, args.preview_rows)
        if preview is not None and ((epoch + 1) % args.save_preview_every == 0 or epoch == args.epochs - 1):
            make_preview_grid(
                originals=preview["images"],
                baseline_recon=preview["baseline"],
                fused_recon=preview["fused"],
                sample_keys=preview["keys"],
                output_path=output_dir / f"preview_epoch_{epoch:04d}.png",
                title=f"Epoch {epoch}",
            )

        if row["val_fused_recon_loss"] < best_val_loss:
            best_val_loss = row["val_fused_recon_loss"]
            best_epoch = epoch
            projection.save_weights(output_dir / "best_projection.weights.h5")
            preview = filter_preview_rows(eval_metrics["preview"], args.preview_rows)
            if preview is not None:
                make_preview_grid(
                    originals=preview["images"],
                    baseline_recon=preview["baseline"],
                    fused_recon=preview["fused"],
                    sample_keys=preview["keys"],
                    output_path=output_dir / "best_preview.png",
                    title=f"Best epoch {epoch}",
                )

    projection.save_weights(output_dir / "last_projection.weights.h5")
    summary = {
        "best_epoch": None if best_epoch is None else int(best_epoch),
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
    parser.add_argument("--data_root", default="webdataset_frames", help="WebDataset shard directory, a specific .tar shard, or an image directory.")
    parser.add_argument("--shard_glob", default=DEFAULT_SHARD_GLOB)
    parser.add_argument("--clip_checkpoint", default=None, help="Path to merged_final or lora_final from CLIP fine-tuning.")
    parser.add_argument("--vae_checkpoint", default=None, help="Optional explicit path to the tf_vae SavedModel directory.")
    parser.add_argument("--output_dir", default="fusion_reconstruction_runs/default")
    parser.add_argument("--cache_dir", default=None, help="Directory for cached frozen features. Defaults to <output_dir>/feature_cache.")
    parser.add_argument("--env_name", default=config.get("env_name", "CarRacing-v0"))
    parser.add_argument("--exp_name", default=config.get("exp_name", "WorldModels"))
    parser.add_argument("--z_size", type=int, default=int(config.get("z_size", 32)))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cache_batch_size", type=int, default=None, help="Batch size used while building the frozen-feature cache.")
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
    parser.add_argument("--rebuild_cache", action="store_true", help="Ignore any existing cached frozen features and rebuild them.")
    parser.add_argument("--normalize_clip_features", action="store_true")
    parser.add_argument("--normalize_fused_features", action="store_true")
    parser.add_argument("--save_preview_every", type=int, default=1)
    parser.add_argument("--num_preview", type=int, default=4)
    parser.add_argument(
        "--preview_rows",
        type=str,
        default=None,
        help="Comma-separated 0-based preview row indices to keep in the saved preview image, for example '0,2,3'.",
    )
    args = parser.parse_args()
    if args.preview_rows is not None:
        args.preview_rows = [int(part.strip()) for part in args.preview_rows.split(",") if part.strip()]
    return args


if __name__ == "__main__":
    train(parse_args())