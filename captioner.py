import tarfile
import io
import time
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ----------------------------
# Settings
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "qwen3-vl-8b-instruct"

TAR_PATH = "webdataset_frames/shard-00000.tar"
OUTPUT_PATH = "captions_selected_frames.txt"

FRAME_INDICES = [0]

MAX_NEW_TOKENS = 40

PROMPT = """
You are labeling a 64x64 top-down racing game frame.
Choose exactly one value for each field from the allowed options below.

road shape: straight, gentle left, gentle right, sharp left, sharp right
car position: center, left, right, far left, far right
surface: asphalt, grass, mixed
heading alignment: aligned, slightly left, slightly right, misaligned
action steer: left, right, neutral
skid marks visible: yes, no
transition: stable, moving left, moving right, recovering, drifting offroad, entering turn, exiting turn

Output format:
road shape=<value>, car position=<value>, surface=<value>, heading alignment=<value>, action steer=<value>, skid marks visible=<value>, transition=<value>

Rules:
- Analyze the road geometry for road shape.
- Output exactly one value per field.
- Use only the allowed labels.
- Do not explain.
- Do not add any extra words.
- If uncertain, choose the closest label.
""".strip()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model_and_processor():
    log("About to load model")
    log(f"Loading from local folder: {MODEL_DIR}")
    t0 = time.time()

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Local model directory does not exist: {MODEL_DIR}"
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )

    t1 = time.time()
    log(f"Model loaded in {t1 - t0:.2f}s")

    log("About to load processor")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    t1 = time.time()
    log(f"Processor loaded in {t1 - t0:.2f}s")

    return model, processor


def extract_selected_images_from_tar(tar_path, frame_indices):
    log(f"About to open tar file: {tar_path}")
    t0 = time.time()

    with tarfile.open(tar_path, "r") as tar:
        t1 = time.time()
        log(f"Tar opened in {t1 - t0:.2f}s")

        log("About to read tar members")
        t0 = time.time()
        members = tar.getmembers()
        t1 = time.time()
        log(f"Read {len(members)} total tar members in {t1 - t0:.2f}s")

        log("About to filter PNG members")
        t0 = time.time()
        png_members = [m for m in members if m.name.endswith(".png")]
        t1 = time.time()
        log(f"Found {len(png_members)} PNG files in {t1 - t0:.2f}s")

        if not png_members:
            raise ValueError("No .png files found in the tar archive.")

        log("About to sort PNG members")
        t0 = time.time()
        png_members = sorted(png_members, key=lambda m: m.name)
        t1 = time.time()
        log(f"Sorted PNG members in {t1 - t0:.2f}s")

        max_idx = max(frame_indices)
        if max_idx >= len(png_members):
            raise IndexError(
                f"Requested frame index {max_idx}, but only {len(png_members)} PNG files exist."
            )

        selected = []
        for idx in frame_indices:
            member = png_members[idx]
            log(f"Selected PNG index {idx} -> {member.name}")

            log(f"About to extract {member.name}")
            t0 = time.time()
            extracted = tar.extractfile(member)
            t1 = time.time()
            log(f"tar.extractfile finished in {t1 - t0:.2f}s")

            if extracted is None:
                raise ValueError(f"Could not extract {member.name}")

            log(f"About to read bytes for {member.name}")
            t0 = time.time()
            img_bytes = extracted.read()
            t1 = time.time()
            log(f"Read {len(img_bytes)} bytes in {t1 - t0:.2f}s")

            log(f"About to decode {member.name} with PIL")
            t0 = time.time()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            t1 = time.time()
            log(f"Decoded in {t1 - t0:.2f}s; size={img.size}, mode={img.mode}")

            selected.append((member.name, img))

    return selected


def caption_one_image(model, processor, img, prompt, max_new_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    log("About to apply chat template")
    t0 = time.time()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    t1 = time.time()
    log(f"Chat template applied in {t1 - t0:.2f}s")

    log("About to move inputs to model device")
    t0 = time.time()
    inputs = inputs.to(model.device)
    t1 = time.time()
    log(f"Inputs moved in {t1 - t0:.2f}s")

    log("About to generate")
    t0 = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    t1 = time.time()
    log(f"Generation finished in {t1 - t0:.2f}s")

    log("About to trim prompt tokens")
    t0 = time.time()
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    t1 = time.time()
    log(f"Trimmed in {t1 - t0:.2f}s")

    log("About to decode output")
    t0 = time.time()
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    t1 = time.time()
    log(f"Decoded in {t1 - t0:.2f}s")

    caption = output_text[0].strip()
    return caption


def main():
    try:
        log("Script started")
        log(f"torch version: {torch.__version__}")
        log(f"CUDA available: {torch.cuda.is_available()}")

        model, processor = load_model_and_processor()
        selected_images = extract_selected_images_from_tar(TAR_PATH, FRAME_INDICES)

        results = []

        for i, (name, img) in enumerate(selected_images, start=1):
            log(f"Processing image {i}/{len(selected_images)}: {name}")
            caption = caption_one_image(
                model=model,
                processor=processor,
                img=img,
                prompt=PROMPT,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            log(f"Caption for {name}: {caption}")
            results.append((name, caption))

        log(f"About to save output to {OUTPUT_PATH}")
        t0 = time.time()
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for name, caption in results:
                f.write(f"{name}\t{caption}\n")
        t1 = time.time()
        log(f"Saved output in {t1 - t0:.2f}s")

        log("Script finished successfully")

    except Exception as e:
        log(f"ERROR: {repr(e)}")
        raise


if __name__ == "__main__":
    main()