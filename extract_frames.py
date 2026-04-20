import argparse
import concurrent.futures
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from agile_wm.paths import default_frame_shards_dir, default_rollouts_dir
from agile_wm.runtime import resolve_repo_path


def parse_episode_index(rollout_file: Path) -> int:
    stem = rollout_file.stem
    if not stem.startswith("episode_"):
        raise ValueError(f"Unexpected rollout filename: {rollout_file.name}")
    return int(stem.split("_", 1)[1])


def build_sample_key(episode_index: int, frame_index: int) -> str:
    return f"episode_{episode_index:06d}_frame_{frame_index:06d}"


def per_rollout_seed(base_seed: int | None, episode_index: int) -> int:
    if base_seed is None:
        entropy = np.random.SeedSequence().entropy
        return int(np.random.SeedSequence([int(entropy), episode_index]).generate_state(1, dtype=np.uint64)[0])
    return int(np.random.SeedSequence([base_seed, episode_index]).generate_state(1, dtype=np.uint64)[0])


def sample_rollout_frames(
    rollout_path: Path,
    num_frames: int,
    rng_seed: int,
    min_frame_index: int,
):
    with np.load(rollout_path) as data:
        if "obs" not in data:
            raise KeyError(f"Missing 'obs' in {rollout_path}")
        obs = data["obs"]

        if obs.dtype != np.uint8 or obs.ndim != 4:
            raise ValueError(
                f"Expected raw image observations shaped [T, H, W, C] uint8 in {rollout_path}, got {obs.shape} {obs.dtype}"
            )

        frame_count = len(obs)
        if min_frame_index < 0:
            raise ValueError(f"min_frame_index must be non-negative, got {min_frame_index}")
        if min_frame_index >= frame_count:
            raise ValueError(
                f"min_frame_index={min_frame_index} is out of range for {rollout_path} with {frame_count} frames"
            )

        available_indices = np.arange(min_frame_index, frame_count)
        frames_to_extract = min(num_frames, len(available_indices))
        rng = np.random.default_rng(rng_seed)
        frame_indices = np.sort(rng.choice(available_indices, size=frames_to_extract, replace=False))
        frames = obs[frame_indices]
        seed = int(data["seed"][0]) if "seed" in data else None

    return frames, frame_indices, seed, frame_count


def encode_png(frame: np.ndarray) -> bytes:
    image = Image.fromarray(frame)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def add_bytes_to_tar(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def load_existing_manifest(manifest_path: Path) -> dict[int, set[int]]:
    existing_frames_by_episode: dict[int, set[int]] = {}
    if not manifest_path.exists():
        return existing_frames_by_episode

    with manifest_path.open("r", encoding="utf-8") as manifest:
        for line in manifest:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            episode_index = record.get("episode_index")
            frame_index = record.get("frame_index")
            if episode_index is None or frame_index is None:
                continue
            existing_frames_by_episode.setdefault(int(episode_index), set()).add(int(frame_index))
    return existing_frames_by_episode


def get_next_shard_index(output_dir: Path) -> int:
    shard_indices = []
    for shard_path in output_dir.glob("shard-*.tar"):
        try:
            shard_indices.append(int(shard_path.stem.split("-")[1]))
        except (IndexError, ValueError):
            continue
    return max(shard_indices, default=-1) + 1


def process_rollout(task: tuple[str, int, int | None, int, int, set[int]]):
    rollout_path_str, num_frames, base_seed, episode_index, min_frame_index, existing_frame_indices = task
    rollout_path = Path(rollout_path_str)
    rollout_rng_seed = per_rollout_seed(base_seed, episode_index)
    frames, frame_indices, rollout_seed, frame_count = sample_rollout_frames(
        rollout_path, num_frames, rollout_rng_seed, min_frame_index
    )

    samples = []
    skipped_existing = 0
    for frame, frame_index in zip(frames, frame_indices):
        frame_index = int(frame_index)
        sample_key = build_sample_key(episode_index, frame_index)
        if frame_index in existing_frame_indices:
            skipped_existing += 1
            continue

        metadata = {
            "sample_key": sample_key,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "rollout_file": rollout_path.name,
            "rollout_seed": rollout_seed,
            "frame_count": frame_count,
            "image_name": f"{sample_key}.png",
        }
        samples.append(
            {
                "sample_key": sample_key,
                "png": encode_png(frame),
                "json": json.dumps(metadata, separators=(",", ":")).encode("utf-8"),
                "metadata": metadata,
            }
        )

    return {
        "episode_index": episode_index,
        "rollout_file": rollout_path.name,
        "sampled": len(frame_indices),
        "written": len(samples),
        "skipped_existing": skipped_existing,
        "samples": samples,
    }


def build_webdataset(
    rollouts_dir: Path,
    output_dir: Path,
    num_frames: int,
    min_frame_index: int,
    seed: int | None,
    shard_size: int,
    limit_rollouts: int | None,
    num_workers: int,
    resume: bool,
) -> None:
    rollout_files = sorted(rollouts_dir.glob("episode_*.npz"))
    if limit_rollouts is not None:
        rollout_files = rollout_files[:limit_rollouts]

    if not rollout_files:
        raise FileNotFoundError(f"No rollout files found in {rollouts_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    if resume and seed is None:
        raise ValueError("--resume requires --seed so frame sampling is reproducible")

    existing_frames_by_episode = load_existing_manifest(manifest_path) if resume else {}
    manifest_mode = "a" if resume and manifest_path.exists() else "w"
    shard_index = get_next_shard_index(output_dir) if resume else 0
    shard_sample_count = 0
    total_samples = sum(len(frame_indices) for frame_indices in existing_frames_by_episode.values())
    new_samples = 0
    skipped_existing = 0
    current_tar = None

    def open_next_shard(next_index: int) -> tarfile.TarFile:
        shard_path = output_dir / f"shard-{next_index:05d}.tar"
        return tarfile.open(shard_path, mode="w")

    tasks = [
        (
            str(rollout_file),
            num_frames,
            seed,
            parse_episode_index(rollout_file),
            min_frame_index,
            existing_frames_by_episode.get(parse_episode_index(rollout_file), set()),
        )
        for rollout_file in rollout_files
    ]

    with manifest_path.open(manifest_mode, encoding="utf-8") as manifest:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for rollout_idx, result in enumerate(executor.map(process_rollout, tasks), start=1):
                for sample in result["samples"]:
                    if current_tar is None or shard_sample_count >= shard_size:
                        if current_tar is not None:
                            current_tar.close()
                        current_tar = open_next_shard(shard_index)
                        shard_index += 1
                        shard_sample_count = 0

                    sample_key = sample["sample_key"]
                    add_bytes_to_tar(current_tar, f"{sample_key}.png", sample["png"])
                    add_bytes_to_tar(
                        current_tar,
                        f"{sample_key}.json",
                        sample["json"],
                    )
                    manifest.write(json.dumps(sample["metadata"]) + "\n")
                    existing_frames_by_episode.setdefault(
                        sample["metadata"]["episode_index"], set()
                    ).add(sample["metadata"]["frame_index"])

                    shard_sample_count += 1
                    total_samples += 1
                    new_samples += 1

                skipped_existing += result["skipped_existing"]
                print(
                    f"processed={rollout_idx}/{len(rollout_files)} "
                    f"episode={result['episode_index']} sampled={result['sampled']} "
                    f"written={result['written']} skipped_existing={result['skipped_existing']} "
                    f"total_samples={total_samples}"
                )

    if current_tar is not None:
        current_tar.close()

    summary = {
        "rollouts_dir": str(rollouts_dir),
        "num_rollouts": len(rollout_files),
        "frames_per_rollout": num_frames,
        "min_frame_index": min_frame_index,
        "total_samples": total_samples,
        "new_samples": new_samples,
        "skipped_existing": skipped_existing,
        "shard_size": shard_size,
        "num_shards": get_next_shard_index(output_dir),
        "seed": seed,
        "num_workers": num_workers,
        "resume": resume,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Wrote {new_samples} new samples, skipped {skipped_existing} existing samples, "
        f"total dataset size is {total_samples} samples in {get_next_shard_index(output_dir)} shard(s) at {output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description="Build a WebDataset from rollout frames.")
    parser.add_argument(
        "--rollouts_dir",
        type=str,
        default=str(default_rollouts_dir()),
        help="Directory containing rollout .npz files",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=20,
        help="Number of random frames to sample per rollout",
    )
    parser.add_argument(
        "--min_frame_index",
        type=int,
        default=50,
        help="Minimum frame index eligible for sampling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(default_frame_shards_dir()),
        help="Directory where WebDataset shards and manifests will be written",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=1000,
        help="Maximum number of samples per tar shard",
    )
    parser.add_argument(
        "--limit_rollouts",
        type=int,
        default=None,
        help="Optional limit for number of rollouts to process",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes used to load and encode rollouts",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append new shards and skip samples already listed in manifest.jsonl",
    )

    args = parser.parse_args()
    build_webdataset(
        rollouts_dir=resolve_repo_path(args.rollouts_dir),
        output_dir=resolve_repo_path(args.output_dir),
        num_frames=args.num_frames,
        min_frame_index=args.min_frame_index,
        seed=args.seed,
        shard_size=args.shard_size,
        limit_rollouts=args.limit_rollouts,
        num_workers=args.num_workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
