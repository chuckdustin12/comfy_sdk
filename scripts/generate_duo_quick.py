from __future__ import annotations

import argparse
import os
import sys
import urllib.error
from pathlib import Path

from comfy_sdk import ComfyClient, generate_from_workflow


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return int(value)


def _env_optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return float(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return float(value)


BASE_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8000")
DEFAULT_OUTPUT_DIR = Path(os.environ.get("DUO_OUTPUT_DIR", "generated/duo_quick"))
DEFAULT_OUTPUT_PREFIX = os.environ.get("DUO_OUTPUT_PREFIX", "duo_quick")
DEFAULT_CKPT_NAME = os.environ.get("CKPT_NAME", "illustriousMixedCGI_v20.safetensors")

DEFAULT_AMBER_LORA = os.environ.get("AMBER_LORA", "AMBER8-000005.safetensors")
DEFAULT_CLIN_LORA = os.environ.get("CLIN_LORA", "CLIN8-000003.safetensors")
DEFAULT_AMBER_TOKEN = os.environ.get("AMBER_TOKEN", "AMBER6")
DEFAULT_CLIN_TOKEN = os.environ.get("CLIN_TOKEN", "CLIN6")

DEFAULT_AMBER_STRENGTH = _env_float("AMBER_LORA_STRENGTH", 1.0)
DEFAULT_CLIN_STRENGTH = _env_float("CLIN_LORA_STRENGTH", 1.0)

DEFAULT_POSITIVE = (
    "DUO portrait, two adult women, not twins, different faces, "
    "cinematic realistic photography, soft warm lighting, detailed skin texture, "
    "sharp focus, shallow depth of field, eye contact with camera, upper body, close-up, "
    "(left side: 1.2), "
    "(AMBER6:1.3), young adult woman, blue eyes, natural expression, "
    "(right side: 1.2), "
    "(CLIN7:1.3), woman with freckles, hazel eyes, distinct facial structure, "
    "cheek to cheek composition"
)
DEFAULT_NEGATIVE = (
    "lowres, blurry, pixelated, jpeg artifacts, bad anatomy, bad proportions, "
    "out of frame, cropped face, missing limbs, extra limbs, underage, teen, child, "
    "twins, identical faces, duplicate face, same face"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a quick duo image using the workflow.")
    parser.add_argument("--positive", default=os.environ.get("POSITIVE_PROMPT", DEFAULT_POSITIVE))
    parser.add_argument("--negative", default=os.environ.get("NEGATIVE_PROMPT", DEFAULT_NEGATIVE))
    parser.add_argument("--ckpt-name", default=DEFAULT_CKPT_NAME)
    parser.add_argument("--amber-lora", default=DEFAULT_AMBER_LORA)
    parser.add_argument("--clin-lora", default=DEFAULT_CLIN_LORA)
    parser.add_argument("--amber-strength", type=float, default=DEFAULT_AMBER_STRENGTH)
    parser.add_argument("--clin-strength", type=float, default=DEFAULT_CLIN_STRENGTH)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--seed", type=int, default=_env_optional_int("SEED"))
    parser.add_argument("--steps", type=int, default=_env_optional_int("STEPS"))
    parser.add_argument("--cfg", type=float, default=_env_optional_float("CFG"))
    parser.add_argument("--sampler-name", default=os.environ.get("SAMPLER_NAME"))
    parser.add_argument("--scheduler", default=os.environ.get("SCHEDULER"))
    parser.add_argument("--denoise", type=float, default=_env_optional_float("DENOISE"))
    parser.add_argument("--width", type=int, default=_env_optional_int("WIDTH"))
    parser.add_argument("--height", type=int, default=_env_optional_int("HEIGHT"))
    parser.add_argument("--batch-size", type=int, default=_env_optional_int("BATCH_SIZE"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = ComfyClient(base_url=BASE_URL)

    loras = [
        {
            "lora_name": args.amber_lora,
            "strength_model": args.amber_strength,
            "strength_clip": args.amber_strength,
        },
        {
            "lora_name": args.clin_lora,
            "strength_model": args.clin_strength,
            "strength_clip": args.clin_strength,
        },
    ]

    try:
        result = generate_from_workflow(
            client=client,
            positive=args.positive,
            negative=args.negative,
            ckpt_name=args.ckpt_name,
            loras=loras,
            seed=args.seed,
            steps=args.steps,
            cfg=args.cfg,
            sampler_name=args.sampler_name,
            scheduler=args.scheduler,
            denoise=args.denoise,
            width=args.width,
            height=args.height,
            batch_size=args.batch_size,
            output_prefix=args.output_prefix,
        )
    except urllib.error.URLError as exc:
        print(
            f"ComfyUI is not reachable at {BASE_URL}. "
            "Start the server or set COMFY_URL.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    images = result.get("images", [])
    if not images:
        raise RuntimeError("No images returned")

    saved = []
    for image in images:
        data = client.download_image(
            image["filename"],
            subfolder=image.get("subfolder", ""),
            image_type=image.get("type", "output"),
        )
        target = output_dir / image["filename"]
        target.write_bytes(data)
        saved.append(str(target))

    print(f"Saved {len(saved)} image(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
