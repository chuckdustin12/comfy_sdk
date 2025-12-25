from __future__ import annotations

import os
import sys
import urllib.error
from pathlib import Path

from comfy_sdk import ComfyClient, generate_from_workflow


BASE_URL = os.environ.get("COMFY_URL", "http://localhost:8000/")
OUTPUT_DIR = Path("generated/amber_caitlin")
CKPT_NAME = "illustriousMixedCGI_v20.safetensors"

LORAS = [
    {
        "lora_name": os.environ.get("AMBER_LORA", "AMBER8-000005.safetensors"),
        "strength_model": 1,
        "strength_clip": 1,
    },
    {
        "lora_name": os.environ.get("CAITLIN_LORA", "CLIN6.safetensors"),
        "strength_model": 1.0,
        "strength_clip": 1.0,
    },
]

POSITIVE_PROMPT = (
  """two adult women, different faces, not twins,(
  AMBER6 woman: auburn bob hair, blue eyes, small breasts,
  subtle chest tattoo, soft makeup
),
(
  CLIN6 woman: brown wavy hair, blue eyes, freckles,
  small breasts, slim face, soft makeup
),
close-up portrait, cheek to cheek, upper body,
soft warm lighting, sharp focus, eye contact,
intimate realistic portrait""")

NEGATIVE_PROMPT = (
    "lowres, blurry, pixelated, jpeg artifacts, bad anatomy, bad proportions, "
    "out of frame, cropped face, missing limbs, extra limbs, underage, teen, child, "
    "twins, identical faces, duplicate face, same face"
)

SEED = 2784119


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = ComfyClient(base_url=BASE_URL)

    try:
        result = generate_from_workflow(
            client=client,
            positive=POSITIVE_PROMPT,
            negative=NEGATIVE_PROMPT,
            ckpt_name=CKPT_NAME,
            loras=LORAS,
            seed=SEED,
            steps=50,
            cfg=3.5,
            sampler_name="dpmpp_2m_sde",
            scheduler="karras",
            denoise=1.0,
            output_prefix="amber_caitlin_duo",
        )
    except urllib.error.URLError as exc:
        print(
            f"ComfyUI is not reachable at {BASE_URL}. Start the server or set COMFY_URL.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    images = result.get("images", [])
    if not images:
        raise RuntimeError("No images returned")

    for image in images:
        data = client.download_image(
            image["filename"],
            subfolder=image.get("subfolder", ""),
            image_type=image.get("type", "output"),
        )
        target = OUTPUT_DIR / image["filename"]
        target.write_bytes(data)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
