from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .runtime import artifacts_root


DEFAULT_RECONSTRUCTION_RUN = "shard0_ep5_cbp544"


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        if expanded in seen:
            continue
        seen.add(expanded)
        ordered.append(expanded)
    return ordered


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def artifacts_path(*parts: str) -> Path:
    return artifacts_root().joinpath(*parts)


def default_rollouts_dir() -> Path:
    return artifacts_path("datasets", "rollouts")


def rollout_dir_candidates() -> list[Path]:
    return [default_rollouts_dir()]


def default_frame_shards_dir() -> Path:
    return artifacts_path("datasets", "webdataset_frames")


def frame_shard_dir_candidates() -> list[Path]:
    return [default_frame_shards_dir()]


def default_caption_pairs_dir(*, temp: bool = False) -> Path:
    leaf = "frame_caption_pairs_temp" if temp else "frame_caption_pairs"
    return artifacts_path("datasets", leaf)


def caption_pair_dir_candidates(*, include_temp: bool = True) -> list[Path]:
    candidates = []
    if include_temp:
        candidates.append(default_caption_pairs_dir(temp=True))
    candidates.append(default_caption_pairs_dir(temp=False))
    return _unique_paths(candidates)


def default_clip_finetune_dir() -> Path:
    return artifacts_path("models", "clip_finetune")


def clip_finetune_dir_candidates() -> list[Path]:
    return [default_clip_finetune_dir()]


def clip_checkpoint_candidates() -> list[Path]:
    candidates: list[Path] = []
    for root in clip_finetune_dir_candidates():
        candidates.extend([root / "merged_final", root / "lora_final"])
    return _unique_paths(candidates)


def default_qwen_model_dir(model_name: str = "qwen3-vl-8b-instruct") -> Path:
    return artifacts_path("models", model_name)


def qwen_model_dir_candidates(model_name: str = "qwen3-vl-8b-instruct") -> list[Path]:
    return [default_qwen_model_dir(model_name)]


def default_hf_cache_dir() -> Path:
    return artifacts_path("cache", "hf_cache")


def default_hf_home_dir() -> Path:
    return artifacts_path("cache", "hf_home")


def default_logs_dir() -> Path:
    return artifacts_path("logs")


def default_rollout_videos_dir() -> Path:
    return artifacts_path("visualizations", "rollout_videos")


def default_fusion_reconstruction_root() -> Path:
    return artifacts_path("experiments", "fusion_reconstruction")


def default_fusion_reconstruction_dir(run_name: str) -> Path:
    return default_fusion_reconstruction_root() / run_name


def default_reconstruction_dir(run_name: str) -> Path:
    return default_fusion_reconstruction_dir(run_name)


def reconstruction_run_dir_candidates(run_name: str) -> list[Path]:
    return [default_fusion_reconstruction_dir(run_name)]


def default_world_model_leaf(exp_name: str, env_name: str, leaf: str) -> Path:
    return artifacts_path("world_models", exp_name, env_name, leaf)


def world_model_leaf_candidates(exp_name: str, env_name: str, leaf: str) -> list[Path]:
    candidates = [default_world_model_leaf(exp_name, env_name, leaf)]
    if "-" in env_name:
        base_env = env_name.split("-", 1)[0]
        candidates.append(default_world_model_leaf(exp_name, base_env, leaf))
    return _unique_paths(candidates)


def default_controller_checkpoint_path(exp_name: str, env_name: str) -> Path:
    candidates: list[Path] = []
    names = [f"{env_name}.cma.16.64.best.json"]
    if "-" in env_name:
        base_env = env_name.split("-", 1)[0]
        names.extend([f"{base_env}.cma.16.64.best.json", f"{base_env.lower()}.cma.16.64.best.json"])

    for log_dir in world_model_leaf_candidates(exp_name, env_name, "log"):
        for name in names:
            candidates.append(log_dir / name)

    return first_existing(candidates) or candidates[0]
