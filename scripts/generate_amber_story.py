from __future__ import annotations

import json
import os
import sys
import urllib.error
from pathlib import Path

from comfy_sdk import ComfyClient, generate_from_workflow


BASE_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
OUTPUT_DIR = Path("generated/amber_story")

NEGATIVE_PROMPT = (
    "lowres, blurry, pixelated, jpeg artifacts, bad anatomy, bad proportions, "
    "out of frame, cropped face, missing limbs, extra limbs, "
    "underage, teen, child, multiple men, gangbang, large breasts, big breasts, huge breasts"
)

SEED_BASE = 24189573

SCENES = [
    {
        "name": "scene_01_tease",
        "positive": (
            "AMBER6, adult woman, close-up face, blue eyes, auburn bob hair, small breasts, "
            "sheer robe open, topless, nipples visible, seated on bed, "
            "soft warm light, eye contact, teasing smile"
        ),
    },
    {
        "name": "scene_02_undress",
        "positive": (
            "AMBER6, adult woman, nude, close-up face, small breasts, flushed cheeks, "
            "hand on breast, lying on bed, legs slightly parted, "
            "soft daylight, intimate portrait"
        ),
    },
    {
        "name": "scene_03_oral",
        "positive": (
            "AMBER6, adult woman, nude, close-up face, small breasts, kneeling on bed, "
            "oral sex with male partner, penis near lips, hand on shaft, "
            "eye contact, warm bedroom light"
        ),
    },
    {
        "name": "scene_04_sex",
        "positive": (
            "AMBER6, adult woman, nude, close-up face, small breasts, riding male partner, "
            "vaginal sex, hair slightly messy, moaning expression, "
            "tight framing, soft warm light"
        ),
    },
    {
        "name": "scene_05_afterglow",
        "positive": (
            "AMBER6, adult woman, nude, close-up face, small breasts, lying on bed, "
            "relaxed smile, soft sweat on skin, gentle eye contact, "
            "warm ambient light, post-sex glow"
        ),
    },
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = ComfyClient(base_url=BASE_URL)

    manifest = []
    for index, scene in enumerate(SCENES, start=1):
        try:
            result = generate_from_workflow(
                client=client,
                positive=scene["positive"],
                negative=NEGATIVE_PROMPT,
                seed=SEED_BASE + index,
                steps=30,
                cfg=3.5,
                sampler_name="dpmpp_2m_sde",
                scheduler="karras",
                denoise=1.0,
                output_prefix=f"amber_story_{index:02d}",
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
            raise RuntimeError(f"No images returned for {scene['name']}")

        saved = []
        for image in images:
            data = client.download_image(
                image["filename"],
                subfolder=image.get("subfolder", ""),
                image_type=image.get("type", "output"),
            )
            target = OUTPUT_DIR / image["filename"]
            target.write_bytes(data)
            saved.append(str(target))

        manifest.append(
            {
                "scene": scene["name"],
                "prompt_id": result["prompt_id"],
                "positive": scene["positive"],
                "negative": NEGATIVE_PROMPT,
                "images": saved,
            }
        )

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
