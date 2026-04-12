import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def save_frames(rollout_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(rollout_path) as data:
        obs = data["obs"]
        assert obs.dtype == np.uint8 and obs.ndim == 4, "Expected [T, H, W, C] uint8"
        for idx, frame in enumerate(obs):
            img = Image.fromarray(frame)
            img.save(output_dir / f"frame_{idx:05d}.png")
    print(f"Saved {len(obs)} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save all frames from a rollout .npz file to a folder.")
    parser.add_argument("rollout", type=str, help="Path to rollout .npz file")
    parser.add_argument("output_dir", type=str, help="Directory to save frames")
    args = parser.parse_args()
    save_frames(Path(args.rollout), Path(args.output_dir))