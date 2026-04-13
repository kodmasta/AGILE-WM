import argparse
import io
import json
import tarfile
import time
from pathlib import Path
from typing import Union

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "qwen3-vl-8b-instruct"
MAX_NEW_TOKENS = 70

PROMPT = """
Describe this 64x64 top-down racing game frame in one sentence.

Focus on:
- the road shape and direction
- where the car is on the road
- whether the car looks aligned with the road or turning
- whether it is on asphalt, grass, or partly offroad
- any obvious motion cues like drifting, recovering, entering a turn, or skid marks

Requirements:
- Output only one concise caption
- Use natural language, not labels
- Mention the most important driving situation first
- Prefer concrete visual descriptions over generic ones
- If uncertain, describe the closest visible situation

Example style:
car near the left edge of a sharp left turn, slightly misaligned, with skid marks as it enters the corner
car centered on a straight asphalt section, aligned with the road and driving steadily
car drifting off the right side of a gentle right curve, partly on grass and recovering
""".strip()

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption a WebDataset shard and write frame-caption pairs as output shards."
    )
    parser.add_argument(
        "--shard_path",
        type=Path,
        required=True,
        help="Path to the input WebDataset tar shard containing .png entries.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where caption shards will be written.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing the local Qwen model files.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=1000,
        help="Maximum number of frame-caption pairs per output tar shard.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum number of tokens generated for each caption.",
    )
    return parser.parse_args()


def ensure_supported_cuda_device() -> None:
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
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
        if device_cc[0] == 7:
            recommendation = (
                "Install a CUDA 12.4 PyTorch stack for sm_7x GPUs, for example "
                "torch==2.6.0, torchvision==0.21.0, and torchaudio==2.6.0, "
                "or use run_caption_array.sh to install a compatible build automatically."
            )
        else:
            recommendation = (
                "Install a PyTorch build that supports this GPU, or use run_caption_array.sh "
                "to install a compatible build automatically."
            )
        raise RuntimeError(
            "Installed PyTorch build is not compatible with the active GPU. "
            f"Found {torch.cuda.get_device_name(device)} ({device_arch}), but this build only supports: {supported}. "
            f"{recommendation}"
        )


def choose_model_dtype() -> Union[str, torch.dtype]:
    if not torch.cuda.is_available():
        return "auto"

    if torch.cuda.is_bf16_supported():
        log("Using bfloat16 model weights on CUDA")
        return torch.bfloat16

    log("Using float16 model weights on CUDA")
    return torch.float16


def load_model_and_processor(model_dir: Path):
    log(f"Loading model from {model_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Local model directory does not exist: {model_dir}")

    ensure_supported_cuda_device()
    model_dtype = choose_model_dtype()

    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=model_dtype,
        device_map="auto",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    log(f"Model and processor loaded in {time.time() - t0:.2f}s")
    return model, processor


def add_bytes_to_tar(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


class OutputShardWriter:
    def __init__(self, output_dir: Path, output_prefix: str, shard_size: int):
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.shard_size = shard_size
        self.shard_index = 0
        self.sample_count_in_shard = 0
        self.total_samples = 0
        self.current_tar: tarfile.TarFile | None = None
        self.output_paths: list[Path] = []

    def _open_next_shard(self) -> None:
        if self.current_tar is not None:
            self.current_tar.close()

        output_path = self.output_dir / f"{self.output_prefix}-{self.shard_index:05d}.tar"
        if output_path.exists():
            raise FileExistsError(
                f"Output shard already exists: {output_path}. Remove it or choose a different output directory."
            )

        self.current_tar = tarfile.open(output_path, mode="w")
        self.output_paths.append(output_path)
        self.shard_index += 1
        self.sample_count_in_shard = 0
        log(f"Opened output shard {output_path}")

    def write(self, sample_key: str, png_bytes: bytes, caption: str, metadata: dict) -> None:
        if self.current_tar is None or self.sample_count_in_shard >= self.shard_size:
            self._open_next_shard()

        txt_bytes = caption.encode("utf-8")
        json_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")

        add_bytes_to_tar(self.current_tar, f"{sample_key}.png", png_bytes)
        add_bytes_to_tar(self.current_tar, f"{sample_key}.txt", txt_bytes)
        add_bytes_to_tar(self.current_tar, f"{sample_key}.json", json_bytes)

        self.sample_count_in_shard += 1
        self.total_samples += 1

    def close(self) -> None:
        if self.current_tar is not None:
            self.current_tar.close()
            self.current_tar = None


def caption_one_image(model, processor, img: Image.Image, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def load_metadata(tar: tarfile.TarFile, member_name: str) -> dict:
    member = tar.getmember(member_name)
    extracted = tar.extractfile(member)
    if extracted is None:
        raise ValueError(f"Could not extract {member_name}")
    return json.loads(extracted.read().decode("utf-8"))


def process_shard(
    shard_path: Path,
    output_dir: Path,
    model_dir: Path,
    shard_size: int,
    max_new_tokens: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = f"{shard_path.stem}-caption"
    summary_path = output_dir / f"{output_prefix}-summary.json"
    if summary_path.exists():
        raise FileExistsError(
            f"Summary already exists: {summary_path}. Remove previous outputs before re-running this shard."
        )

    model, processor = load_model_and_processor(model_dir)
    writer = OutputShardWriter(output_dir=output_dir, output_prefix=output_prefix, shard_size=shard_size)

    try:
        log(f"Reading input shard {shard_path}")
        with tarfile.open(shard_path, "r") as tar:
            members = tar.getmembers()
            png_members = sorted((member for member in members if member.name.endswith(".png")), key=lambda member: member.name)
            member_names = {member.name for member in members}
            log(f"Found {len(png_members)} PNG samples in {shard_path.name}")

            for index, png_member in enumerate(png_members, start=1):
                sample_key = Path(png_member.name).stem
                extracted = tar.extractfile(png_member)
                if extracted is None:
                    raise ValueError(f"Could not extract {png_member.name}")

                png_bytes = extracted.read()
                image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                caption = caption_one_image(
                    model=model,
                    processor=processor,
                    img=image,
                    prompt=PROMPT,
                    max_new_tokens=max_new_tokens,
                )

                metadata_name = f"{sample_key}.json"
                metadata = {}
                if metadata_name in member_names:
                    metadata = load_metadata(tar, metadata_name)

                metadata.update(
                    {
                        "sample_key": sample_key,
                        "source_shard": shard_path.name,
                        "image_name": f"{sample_key}.png",
                        "text_name": f"{sample_key}.txt",
                        "caption": caption,
                    }
                )
                writer.write(sample_key=sample_key, png_bytes=png_bytes, caption=caption, metadata=metadata)

                if index % 25 == 0 or index == len(png_members):
                    log(f"Processed {index}/{len(png_members)} samples")

        summary = {
            "source_shard": str(shard_path),
            "output_dir": str(output_dir),
            "output_shards": [str(path) for path in writer.output_paths],
            "samples_written": writer.total_samples,
            "shard_size": shard_size,
            "max_new_tokens": max_new_tokens,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(f"Wrote {writer.total_samples} frame-caption pairs")
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    log("Script started")
    log(f"torch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        capability = torch.cuda.get_device_capability(device)
        log(f"CUDA device: {torch.cuda.get_device_name(device)} (sm_{capability[0]}{capability[1]})")
    process_shard(
        shard_path=args.shard_path,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        shard_size=args.shard_size,
        max_new_tokens=args.max_new_tokens,
    )
    log("Script finished successfully")


if __name__ == "__main__":
    main()