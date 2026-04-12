import csv
import json
import logging
import math
import argparse
import os
import time
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    raise ImportError("pip install transformers")

try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError:
    raise ImportError("pip install peft")


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
LATEST_CHECKPOINT_POINTER = "checkpoint_latest.txt"
DEFAULT_CLUSTER_DATASET = Path("/network/scratch/h/hengh/my_dataset")
DEFAULT_SHARD_GLOB = "shard-*-caption-*.tar"


def resolve_path(path_like: Optional[str]) -> Optional[Path]:
    if path_like is None:
        return None
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def cluster_scratch_root() -> Path:
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch).expanduser()
    return DEFAULT_CLUSTER_DATASET.parent


def default_data_root() -> Path:
    candidates = [
        (cluster_scratch_root() / "my_dataset" / "frame-caption-pairs").resolve(),
        (cluster_scratch_root() / "my_dataset" / "outputs").resolve(),
        (cluster_scratch_root() / "my_dataset").resolve(),
        (SCRIPT_DIR / "outputs").resolve(),
    ]
    for candidate in candidates:
        resolved = resolve_data_root(candidate, DEFAULT_SHARD_GLOB)
        if directory_has_shards(resolved, DEFAULT_SHARD_GLOB):
            return resolved
    return candidates[0]


def default_output_dir() -> Path:
    return (cluster_scratch_root() / "AGILE-WM" / "clip_finetune").resolve()


def default_hf_cache_dir() -> Optional[Path]:
    return (cluster_scratch_root() / "hf_cache").resolve()


def resolve_model_source(model_name_or_path: str):
    model_path = Path(model_name_or_path).expanduser()
    if model_path.exists():
        return model_path
    return model_name_or_path


def build_hf_load_kwargs(cache_dir: Optional[Path], local_files_only: bool) -> dict:
    load_kwargs = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        load_kwargs["cache_dir"] = str(cache_dir)
    if local_files_only:
        load_kwargs["local_files_only"] = True
    return load_kwargs


def load_clip_components(model_name_or_path: str, cache_dir: Optional[Path], local_files_only: bool):
    model_source = resolve_model_source(model_name_or_path)
    load_kwargs = build_hf_load_kwargs(cache_dir, local_files_only)
    processor = CLIPProcessor.from_pretrained(model_source, **load_kwargs)
    base = CLIPModel.from_pretrained(model_source, **load_kwargs)
    return processor, base


def directory_has_shards(data_root: Path, shard_glob: str) -> bool:
    return data_root.is_dir() and any(data_root.glob(shard_glob))


def resolve_data_root(data_root: Path, shard_glob: str) -> Path:
    if directory_has_shards(data_root, shard_glob):
        return data_root

    nested_outputs = data_root / "outputs"
    if directory_has_shards(nested_outputs, shard_glob):
        return nested_outputs

    return data_root


def discover_shard_directories(primary_root: Path, shard_glob: str) -> List[Path]:
    candidates = [
        primary_root,
        primary_root / "outputs",
        primary_root.parent if primary_root.parent != primary_root else None,
        SCRIPT_DIR / "outputs",
        (cluster_scratch_root() / "my_dataset" / "frame-caption-pairs").resolve(),
        (cluster_scratch_root() / "my_dataset" / "outputs").resolve(),
        (cluster_scratch_root() / "my_dataset").resolve(),
    ]

    seen = set()
    matches = []
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if directory_has_shards(candidate, shard_glob):
            matches.append(candidate)
    return matches


def stage_shards_to_local(data_root: Path, shard_glob: str, local_data_dir: Path) -> Path:
    shard_paths = sorted(data_root.glob(shard_glob))
    if not shard_paths:
        return data_root

    local_data_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for shard_path in shard_paths:
        dst = local_data_dir / shard_path.name
        if dst.exists() and dst.stat().st_size == shard_path.stat().st_size:
            continue
        shutil.copy2(shard_path, dst)
        copied += 1

    log.info(
        "Using %d shards staged in %s (%d copied this run)",
        len(shard_paths), local_data_dir, copied,
    )
    return local_data_dir


def remove_path(path: Path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def update_latest_checkpoint_pointer(output_dir: Path, checkpoint_dir: Path):
    pointer_path = output_dir / LATEST_CHECKPOINT_POINTER
    pointer_path.write_text(f"{checkpoint_dir.name}\n", encoding="utf-8")

    latest = output_dir / "checkpoint_latest"
    if latest.exists() or latest.is_symlink():
        remove_path(latest)

    try:
        latest.symlink_to(checkpoint_dir.name)
    except OSError as exc:
        log.info(
            "Could not create checkpoint_latest symlink (%s); using %s instead",
            exc,
            pointer_path,
        )


def resolve_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    latest = output_dir / "checkpoint_latest"
    if latest.exists():
        return latest.resolve()

    pointer_path = output_dir / LATEST_CHECKPOINT_POINTER
    if not pointer_path.exists():
        return None

    checkpoint_name = pointer_path.read_text(encoding="utf-8").strip()
    if not checkpoint_name:
        return None

    checkpoint_dir = output_dir / checkpoint_name
    if checkpoint_dir.exists():
        return checkpoint_dir.resolve()

    raise FileNotFoundError(
        f"Latest checkpoint pointer refers to missing directory: {checkpoint_dir}"
    )


def build_training_config(args) -> dict:
    training_config = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            training_config[key] = str(value)
        else:
            training_config[key] = value
    return training_config


# ──────────────────────────────────────────────────────────────────────────────
# LoRA target module options
# ──────────────────────────────────────────────────────────────────────────────
# peft matches target_modules by substring of the full param name, so
# ["q_proj","v_proj"] hits both vision_model and text_model transformer blocks.
LORA_TARGET_MODULES = {
    "vision_and_text": ["q_proj", "v_proj"],
    "vision_only":     ["vision_model.encoder.layers"],   
    "text_only":       ["text_model.encoder.layers"],     
}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset  –  CarRacing tar shard format
# ──────────────────────────────────────────────────────────────────────────────
class CarRacingCLIPDataset(Dataset):
    """
    Loads image-caption pairs from your tar shards.

    Confirmed shard format (from shard-00000-caption-00000.tar):
    ─────────────────────────────────────────────────────────────
    Each tar holds 3000 members = 1000 triplets:
        episode_XXXXXX_frame_XXXXXX.png   ← the frame image
        episode_XXXXXX_frame_XXXXXX.txt   ← raw caption text  (ignored)
        episode_XXXXXX_frame_XXXXXX.json  ← metadata + caption

    JSON structure:
        {
          "sample_key":    "episode_000000_frame_000091",
          "episode_index": 0,
          "frame_index":   91,
          "rollout_file":  "episode_000000.npz",
          "image_name":    "episode_000000_frame_000091.png",
          "caption":       "road shape=gentle left, car position=center, ..."
        }

    Shard naming convention:
        shard-00000-caption-00000.tar
        shard-00001-caption-00000.tar
        ...

    Loading strategy (index at startup, stream at runtime)
    ──────────────────────────────────────────────────────
    __init__ opens every shard once, reads only the JSON members to build
    an in-memory index list of (tar_path, png_name, caption) tuples.
    PNG bytes are NOT read until __getitem__ is called — so RAM stays flat
    regardless of dataset size, and startup is fast.
    """

    # Shard glob 
    DEFAULT_GLOB = DEFAULT_SHARD_GLOB

    def __init__(
        self,
        data_root: str,
        processor: CLIPProcessor,
        shard_glob: str = DEFAULT_GLOB,
        caption_key: str = "caption",
    ):
        import tarfile as _tarfile
        super().__init__()
        self.processor   = processor
        self.caption_key = caption_key
        # Each entry: (tar_path_str, png_member_name, caption_str)
        self.index: List[Tuple[str, str, str]] = []

        data_root   = resolve_data_root(Path(data_root), shard_glob)
        shard_paths = sorted(data_root.glob(shard_glob))
        if not shard_paths:
            discovered_roots = discover_shard_directories(data_root, shard_glob)
            message = (
                f"No shards found matching '{shard_glob}' in {data_root}\n"
                f"Expected files like: shard-00000-caption-00000.tar"
            )
            if not data_root.exists():
                message += "\nConfigured data root does not exist."
            if discovered_roots:
                discovered = "\n".join(f"- {path}" for path in discovered_roots)
                message += f"\nAvailable shard directories:\n{discovered}"
            raise FileNotFoundError(
                message
            )

        log.info("Indexing %d shards in %s …", len(shard_paths), data_root)
        skipped = 0

        for tar_path in shard_paths:
            try:
                with _tarfile.open(tar_path, "r") as tf:
                    members   = tf.getnames()
                    # Fast lookup set of available PNG names in this shard
                    png_names = {m for m in members if m.endswith(".png")}

                    for member_name in members:
                        if not member_name.endswith(".json"):
                            continue

                        # Derive PNG name: strip .json → add .png
                        # e.g. "episode_000000_frame_000091.json"
                        #   →  "episode_000000_frame_000091.png"
                        stem     = member_name[:-5]          # remove ".json"
                        png_name = stem + ".png"

                        if png_name not in png_names:
                            skipped += 1
                            continue

                        # Parse JSON to extract caption (cheap — text only)
                        try:
                            fh      = tf.extractfile(member_name)
                            meta    = json.load(fh)
                            fh.close()
                        except Exception as e:
                            log.warning("Bad JSON %s::%s — %s", tar_path.name, member_name, e)
                            skipped += 1
                            continue

                        caption = meta.get(caption_key, "").strip()
                        if not caption:
                            log.warning("Empty caption in %s::%s", tar_path.name, member_name)
                            skipped += 1
                            continue

                        self.index.append((str(tar_path), png_name, caption))

            except Exception as e:
                log.warning("Cannot open shard %s — %s", tar_path.name, e)

        if not self.index:
            raise RuntimeError(
                f"No valid pairs found in {data_root}.\n"
                f"Shard glob used: '{shard_glob}'\n"
                f"Caption key used: '{caption_key}'\n"
                "Check that your shards follow the naming shard-XXXXX-caption-XXXXX.tar"
            )

        log.info(
            "Dataset ready — %d pairs from %d shards  (%d skipped)",
            len(self.index), len(shard_paths), skipped,
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        import tarfile as _tarfile
        tar_path, png_name, caption = self.index[idx]

        # Re-open the shard and extract just this one PNG
        # (tarfile seeks directly; no full extraction to disk)
        try:
            with _tarfile.open(tar_path, "r") as tf:
                fh    = tf.extractfile(png_name)
                image = Image.open(fh).convert("RGB")
                image.load()   # materialise pixels before the file handle closes
                fh.close()
        except Exception as e:
            log.warning("Cannot load %s::%s — %s", tar_path, png_name, e)
            image = Image.new("RGB", (224, 224))

        # NOTE: CarRacing frames are 64×64. CLIPProcessor automatically
        # resizes to 224×224 using bicubic interpolation — no manual resize needed.
        enc = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        return {
            "pixel_values":   enc["pixel_values"].squeeze(0),   # (3, H, W)
            "input_ids":      enc["input_ids"].squeeze(0),       # (77,)
            "attention_mask": enc["attention_mask"].squeeze(0),  # (77,)
        }


# ──────────────────────────────────────────────────────────────────────────────
# Loss  –  symmetric InfoNCE with learnable temperature
# ──────────────────────────────────────────────────────────────────────────────
class CLIPContrastiveLoss(nn.Module):
    def __init__(self, init_temperature: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / init_temperature)
        )

    def forward(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        image_features = F.normalize(image_features, dim=-1)
        text_features  = F.normalize(text_features,  dim=-1)

        scale    = self.logit_scale.exp().clamp(max=100.0)
        logits_i = scale * image_features @ text_features.T   # [B, B]
        logits_t = logits_i.T
        labels   = torch.arange(len(image_features), device=image_features.device)

        return (F.cross_entropy(logits_i, labels) + F.cross_entropy(logits_t, labels)) / 2.0


# ──────────────────────────────────────────────────────────────────────────────
# LR scheduler  –  linear warmup → cosine decay
# ──────────────────────────────────────────────────────────────────────────────
def cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────────────────────
def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model,          # PeftModel (LoRA)
    processor,
    optimizer,
    scheduler,
    scaler,
    loss_fn,
    best_val_loss: float,
    is_best: bool,
    training_config: dict,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # LoRA adapter weights are tiny – only the delta parameters
    adapter_dir = output_dir / f"lora_epoch_{epoch:04d}"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    (adapter_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Training state alongside adapter
    torch.save(
        {
            "epoch":         epoch,
            "best_val_loss": best_val_loss,
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "scaler":        scaler.state_dict(),
            "loss_fn":       loss_fn.state_dict(),
        },
        adapter_dir / "train_state.pt",
    )

    update_latest_checkpoint_pointer(output_dir, adapter_dir)

    if is_best:
        best_dir = output_dir / "checkpoint_best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(adapter_dir, best_dir)
        log.info("New best checkpoint  →  %s", best_dir)

    log.info("Checkpoint saved  →  %s", adapter_dir)

    # Keep only the last 3 epoch adapter dirs
    old = sorted(output_dir.glob("lora_epoch_*"))[:-3]
    for d in old:
        shutil.rmtree(d, ignore_errors=True)


def load_checkpoint(
    checkpoint_dir: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    loss_fn,
    device,
):
    """Load optimizer / scheduler / scaler / loss_fn state from a checkpoint dir."""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_dir}")

    log.info("Resuming training state from  %s", checkpoint_dir.resolve())
    state = torch.load(checkpoint_dir / "train_state.pt", map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    loss_fn.load_state_dict(state["loss_fn"])
    return state["epoch"], state["best_val_loss"]
    # NOTE: LoRA weights are loaded separately via PeftModel.from_pretrained
    #       before this function is called (see main()).


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval metrics
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def retrieval_metrics(img_feats: torch.Tensor, txt_feats: torch.Tensor):
    img_feats = F.normalize(img_feats.float(), dim=-1)
    txt_feats = F.normalize(txt_feats.float(), dim=-1)
    sim    = img_feats @ txt_feats.T
    n      = sim.size(0)
    labels = torch.arange(n, device=sim.device)
    out    = {}
    for name, s in [("i2t", sim), ("t2i", sim.T)]:
        for k in (1, 5):
            _, top = s.topk(min(k, n), dim=-1)
            acc    = top.eq(labels.unsqueeze(1)).any(1).float().mean().item() * 100
            out[f"{name}_top{k}"] = acc
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Train one epoch
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    loss_fn,
    device,
    grad_accum: int,
    epoch: int,
    writer: SummaryWriter,
    global_step: list,
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        pv  = batch["pixel_values"].to(device, non_blocking=True)
        ids = batch["input_ids"].to(device, non_blocking=True)
        msk = batch["attention_mask"].to(device, non_blocking=True)

        with autocast():
            out  = model(pixel_values=pv, input_ids=ids, attention_mask=msk, return_loss=False)
            loss = loss_fn(out.image_embeds, out.text_embeds) / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(loss_fn.parameters()),
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            gs = global_step[0]
            writer.add_scalar("train/loss",        loss.item() * grad_accum, gs)
            writer.add_scalar("train/lr",          scheduler.get_last_lr()[0], gs)
            writer.add_scalar("train/temperature", loss_fn.logit_scale.exp().item(), gs)
            global_step[0] += 1

        total_loss += loss.item() * grad_accum

        if step % 50 == 0:
            log.info(
                "Epoch %d  step %4d/%d  loss=%.4f  lr=%.2e  temp=%.3f",
                epoch, step, len(loader),
                loss.item() * grad_accum,
                scheduler.get_last_lr()[0],
                loss_fn.logit_scale.exp().item(),
            )

    return total_loss / len(loader)


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, loss_fn, device, writer, epoch):
    model.eval()
    total_loss    = 0.0
    all_img_feats = []
    all_txt_feats = []

    for batch in loader:
        pv  = batch["pixel_values"].to(device, non_blocking=True)
        ids = batch["input_ids"].to(device, non_blocking=True)
        msk = batch["attention_mask"].to(device, non_blocking=True)

        with autocast():
            out  = model(pixel_values=pv, input_ids=ids, attention_mask=msk, return_loss=False)
            loss = loss_fn(out.image_embeds, out.text_embeds)

        total_loss += loss.item()
        all_img_feats.append(out.image_embeds.float())
        all_txt_feats.append(out.text_embeds.float())

    val_loss = total_loss / len(loader)
    metrics  = retrieval_metrics(torch.cat(all_img_feats), torch.cat(all_txt_feats))

    writer.add_scalar("val/loss", val_loss, epoch)
    for k, v in metrics.items():
        writer.add_scalar(f"val/{k}", v, epoch)

    log.info(
        "Val epoch=%d  loss=%.4f  i2t@1=%.1f%%  i2t@5=%.1f%%  "
        "t2i@1=%.1f%%  t2i@5=%.1f%%",
        epoch, val_loss,
        metrics.get("i2t_top1", 0), metrics.get("i2t_top5", 0),
        metrics.get("t2i_top1", 0), metrics.get("t2i_top5", 0),
    )
    return val_loss, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tune CLIP on CarRacing data")

    # Data  (tar shard format)
    p.add_argument("--data_root",   default=None,
                   help="Folder containing caption shard .tar files. Defaults to the first available shard directory among Mila scratch frame-caption-pairs, Mila scratch outputs, a nested outputs/ folder, or this repo's outputs/.")
    p.add_argument("--shard_glob",  default=DEFAULT_SHARD_GLOB,
                   help="Glob to find shards, e.g. 'shard_*.tar'")
    p.add_argument("--caption_key", default="caption",
                   help="JSON key for the caption, e.g. 'caption' or 'text'")
    p.add_argument("--stage_shards_to_local", action="store_true",
                   help="Copy matching shards from scratch to local job disk before training. On Mila this is typically $SLURM_TMPDIR.")
    p.add_argument("--local_data_dir", default=None,
                   help="Where to copy shards when --stage_shards_to_local is used. Defaults to $SLURM_TMPDIR/clip_caption_shards.")
    p.add_argument("--val_split",   type=float, default=0.1)

    # Model
    p.add_argument("--clip_model", default="openai/clip-vit-base-patch32",
                   help="Hugging Face model ID or a local model directory")
    p.add_argument("--hf_cache_dir", default=None,
                   help="Cache directory for Hugging Face downloads. Defaults to $SCRATCH/hf_cache.")
    p.add_argument("--local_files_only", action="store_true",
                   help="Load the CLIP model only from local files or cache. Use this on Mila compute nodes without internet.")

    # LoRA
    p.add_argument("--lora_r",       type=int,   default=16,
                   help="LoRA rank (higher = more capacity, more params)")
    p.add_argument("--lora_alpha",   type=int,   default=32,
                   help="LoRA alpha (effective scaling = alpha/r)")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target",  default="vision_and_text",
                   choices=list(LORA_TARGET_MODULES.keys()),
                   help="Which encoders to apply LoRA to")

    # Training
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=2e-4,
                   help="AdamW LR (LoRA can use higher LR than full fine-tune)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int,   default=100)
    p.add_argument("--grad_accum",   type=int,   default=1)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)

    # Checkpointing
    p.add_argument("--output_dir", default=None,
                   help="Directory for checkpoints and exports. Defaults to $SCRATCH/AGILE-WM/clip_finetune.")
    p.add_argument("--resume",     action="store_true",
                   help="Resume training from checkpoint_latest/")
    p.add_argument("--save_every", type=int, default=1)

    args = p.parse_args()
    if args.data_root:
        args.data_root = resolve_data_root(resolve_path(args.data_root), args.shard_glob)
    else:
        args.data_root = default_data_root()
    args.local_data_dir = resolve_path(args.local_data_dir)
    args.hf_cache_dir = resolve_path(args.hf_cache_dir) if args.hf_cache_dir else default_hf_cache_dir()
    args.output_dir = resolve_path(args.output_dir) if args.output_dir else default_output_dir()
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Device: %s", device)

    latest_ckpt = resolve_latest_checkpoint(output_dir) if args.resume else None
    if args.resume and latest_ckpt is not None:
        saved_config_path = latest_ckpt / "training_config.json"
        if saved_config_path.exists():
            try:
                saved_config = json.loads(saved_config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                saved_config = {}
            saved_clip_model = saved_config.get("clip_model")
            if saved_clip_model and args.clip_model == "openai/clip-vit-base-patch32":
                args.clip_model = saved_clip_model
            if args.hf_cache_dir is None and saved_config.get("hf_cache_dir"):
                args.hf_cache_dir = resolve_path(saved_config["hf_cache_dir"])
            if not args.local_files_only and saved_config.get("local_files_only"):
                args.local_files_only = True

    if args.stage_shards_to_local or args.local_data_dir is not None:
        local_data_dir = args.local_data_dir
        if local_data_dir is None:
            slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
            if not slurm_tmpdir:
                raise ValueError(
                    "--stage_shards_to_local requires $SLURM_TMPDIR or an explicit --local_data_dir"
                )
            local_data_dir = Path(slurm_tmpdir).expanduser() / "clip_caption_shards"
        args.data_root = stage_shards_to_local(Path(args.data_root), args.shard_glob, local_data_dir)

    log.info("Data root: %s", args.data_root)
    log.info("Output dir: %s", output_dir)
    if args.hf_cache_dir is not None:
        log.info("HF cache dir: %s", args.hf_cache_dir)

    training_config = build_training_config(args)
    (output_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # ── 1. Load CLIP base + attach (or reload) LoRA ───────────────────────────
    log.info("Loading CLIP: %s", args.clip_model)
    processor, base = load_clip_components(args.clip_model, args.hf_cache_dir, args.local_files_only)

    if args.resume and latest_ckpt is not None:
        log.info("Reloading LoRA adapters from %s", latest_ckpt)
        model = PeftModel.from_pretrained(base, str(latest_ckpt))
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=LORA_TARGET_MODULES[args.lora_target],
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        model = get_peft_model(base, lora_cfg)

    model.to(device)
    model.print_trainable_parameters()

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    dataset = CarRacingCLIPDataset(
        data_root=args.data_root,
        processor=processor,
        caption_key=args.caption_key,
        shard_glob=args.shard_glob,
    )
    n_val   = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    log.info("Train: %d  |  Val: %d", n_train, n_val)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── 3. Optimizer  (LoRA params + learnable temperature only) ─────────────
    loss_fn   = CLIPContrastiveLoss().to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    # peft.parameters() already yields only the LoRA delta weights

    optimizer = AdamW(
        trainable + list(loss_fn.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    # ── 4. Scheduler ──────────────────────────────────────────────────────────
    total_steps = len(train_loader) * args.epochs // max(1, args.grad_accum)
    scheduler   = cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    # ── 5. AMP ────────────────────────────────────────────────────────────────
    scaler = GradScaler(enabled=device.type == "cuda")

    # ── 6. TensorBoard ────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(output_dir / "runs"))

    # ── 7. Restore optimizer / scheduler / scaler if resuming ─────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    global_step   = [0]

    if args.resume and latest_ckpt is not None:
        start_epoch, best_val_loss = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler, scaler, loss_fn, device
        )
        start_epoch += 1
    elif args.resume:
        log.warning("--resume set but no checkpoint found – starting fresh")

    # ── 8. Training loop ──────────────────────────────────────────────────────
    log.info(
        "Starting LoRA fine-tuning  |  epochs=%d  batch=%d  lr=%.2e  "
        "lora_r=%d  lora_alpha=%d  target=%s  device=%s",
        args.epochs, args.batch_size, args.lr,
        args.lora_r, args.lora_alpha, args.lora_target, device,
    )

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            loss_fn, device, args.grad_accum, epoch, writer, global_step,
        )
        val_loss, _ = validate(model, val_loader, loss_fn, device, writer, epoch)

        log.info(
            "Epoch %d  train=%.4f  val=%.4f  time=%.0fs",
            epoch, train_loss, val_loss, time.time() - t0,
        )

        is_best       = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)

        if (epoch + 1) % args.save_every == 0 or is_best:
            save_checkpoint(
                output_dir, epoch, model, processor, optimizer,
                scheduler, scaler, loss_fn, best_val_loss, is_best, training_config,
            )

    writer.close()
    log.info("Training complete. Best val loss: %.4f", best_val_loss)

    # ── 9. Export ─────────────────────────────────────────────────────────────
    # 9a. Save raw LoRA adapters (small files, requires peft at inference)
    final_adapter_dir = output_dir / "lora_final"
    model.save_pretrained(final_adapter_dir)
    processor.save_pretrained(final_adapter_dir)
    (final_adapter_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    log.info("LoRA adapters saved  →  %s", final_adapter_dir)

    # 9b. Merge LoRA into base weights (single HF model, no peft needed)
    try:
        merged     = model.merge_and_unload()
        merged_dir = output_dir / "merged_final"
        merged.save_pretrained(merged_dir)
        processor.save_pretrained(merged_dir)
        (merged_dir / "training_config.json").write_text(
            json.dumps(training_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        log.info(
            "Merged model saved  →  %s  (use this for inference, peft not required)",
            merged_dir,
        )
    except Exception as e:
        log.warning("merge_and_unload failed: %s  –  use adapter weights instead", e)


# ──────────────────────────────────────────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────────────────────────────────────────
def load_for_inference(
    checkpoint_dir: str,
    use_merged: bool = True,
    cache_dir: Optional[str] = None,
    local_files_only: Optional[bool] = None,
):
    """
    Load the fine-tuned model for inference.

    use_merged=True  →  loads merged weights (no peft dependency needed)
    use_merged=False →  loads base + LoRA adapters (smaller files on disk)

    Example usage:
        model, processor = load_for_inference("./checkpoints/merged_final")
        enc = processor(text=["a car on a racing track"], images=img, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        print(out.logits_per_image)
    """
    checkpoint_path = Path(checkpoint_dir).expanduser()
    training_config_path = checkpoint_path / "training_config.json"
    training_config = {}
    if training_config_path.exists():
        try:
            training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            training_config = {}

    base_model_source = training_config.get("clip_model", "openai/clip-vit-base-patch32")
    resolved_cache_dir = resolve_path(cache_dir) if cache_dir else None
    if resolved_cache_dir is None and training_config.get("hf_cache_dir"):
        resolved_cache_dir = resolve_path(training_config["hf_cache_dir"])
    if local_files_only is None:
        local_files_only = bool(training_config.get("local_files_only", False))

    if use_merged:
        model     = CLIPModel.from_pretrained(checkpoint_path)
        processor = CLIPProcessor.from_pretrained(checkpoint_path)
    else:
        _, base   = load_clip_components(base_model_source, resolved_cache_dir, local_files_only)
        model     = PeftModel.from_pretrained(base, checkpoint_path)
        processor = CLIPProcessor.from_pretrained(checkpoint_path)
    model.eval()
    return model, processor


if __name__ == "__main__":
    main()