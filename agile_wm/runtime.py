from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent

_LEGACY_ROOT_MAP = {
    ".venvs": ("cache", "venvs"),
    "clip_finetune": ("models", "clip_finetune"),
    "fusion_reconstruction_runs": ("experiments", "fusion_reconstruction"),
    "hf_cache": ("cache", "hf_cache"),
    "hf_home": ("cache", "hf_home"),
    "logs": ("logs",),
    "outputs": ("datasets", "frame_caption_pairs"),
    "qwen3-vl-8b-instruct": ("models", "qwen3-vl-8b-instruct"),
    "results": ("world_models",),
    "rollout_videos": ("visualizations", "rollout_videos"),
    "rollouts": ("datasets", "rollouts"),
    "webdataset_frames": ("datasets", "webdataset_frames"),
}


def artifacts_root() -> Path:
    configured = os.environ.get("AGILE_WM_ARTIFACTS_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (REPO_ROOT / "artifacts").resolve()


def resolve_repo_path(path_like: Optional[str | Path]) -> Optional[Path]:
    if path_like is None:
        return None

    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path

    resolved = (REPO_ROOT / path).resolve()
    if resolved.exists():
        return resolved

    if not path.parts:
        return resolved

    mapped_root = _LEGACY_ROOT_MAP.get(path.parts[0])
    if mapped_root is None:
        return resolved

    remapped = artifacts_root().joinpath(*mapped_root, *path.parts[1:]).resolve()
    if remapped.exists():
        return remapped

    return remapped


def ensure_supported_cuda_device(device: torch.device) -> None:
    if device.type != "cuda":
        return

    device_cc = torch.cuda.get_device_capability(device)
    device_arch = f"sm_{device_cc[0]}{device_cc[1]}"
    supported_arches = set(torch.cuda.get_arch_list())

    def build_supports_device() -> bool:
        if not supported_arches:
            return True
        if device_arch in supported_arches:
            return True

        supported_ccs: list[tuple[int, int]] = []
        for arch in supported_arches:
            if not arch.startswith("sm_"):
                continue
            suffix = arch.removeprefix("sm_")
            if not suffix.isdigit():
                continue
            supported_ccs.append((int(suffix[:-1]), int(suffix[-1])))

        return any(
            major == device_cc[0] and minor <= device_cc[1]
            for major, minor in supported_ccs
        )

    if not build_supports_device():
        supported = ", ".join(sorted(supported_arches))
        raise RuntimeError(
            "Installed PyTorch build is not compatible with the active GPU. "
            f"Found {torch.cuda.get_device_name(device)} ({device_arch}), but this build only supports: {supported}."
        )
