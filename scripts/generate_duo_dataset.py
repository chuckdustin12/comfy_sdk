from __future__ import annotations

import json
import os
import sys
import urllib.error
from pathlib import Path

from comfy_sdk import ComfyClient, generate_from_workflow


BASE_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8000")
OUTPUT_DIR = Path("DUO/10_DUO")

LORAS = [
    {"lora_name": "AMBER8-000005.safetensors", "strength_model": 0.7, "strength_clip": 0.7},
    {"lora_name": "CLIN6.safetensors", "strength_model": 1.0, "strength_clip": 1.0},
]

NEGATIVE_PROMPT = (
    "male, man, penis, multiple people, solo, "
    "underage, teen, child, "
    "extra limbs, extra fingers, bad anatomy, bad proportions, "
    "duplicate face, identical faces, cropped face, out of frame, lowres, blurry"
)

BASE_PROMPT = (
    "DUO, AMBER6, CLIN6, two adult women, close-up faces, faces visible, "
    "eye contact, detailed skin, soft warm light, shallow depth of field"
)

EXPLICIT_SCENES = [
    "nude, kissing, tongues, hands on breasts, lying on bed",
    "nude, mutual oral sex, 69 position, bedroom",
    "nude, cunnilingus, one woman on back, other between legs",
    "nude, fingering, one woman on knees, other standing, bedroom",
    "nude, tribbing, hips pressed together, bed",
    "nude, scissoring, legs intertwined, sheets",
    "nude, one woman licking nipples, other moaning, close embrace",
    "nude, oral sex, sitting on edge of bed, partner kneeling",
    "nude, one woman straddling the other, grinding, bed",
    "nude, breast suckling, close embrace, soft light",
    "nude, one woman on stomach, other behind, hand between thighs",
    "nude, shower scene, wet skin, kissing, steam",
    "nude, bathtub, legs entwined, kissing, warm steam",
    "nude, couch, cunnilingus, partner between thighs",
    "nude, mirror reflection, kissing, close faces",
    "nude, playful pinning, kissing neck, hands on hips",
    "nude, bed, one woman on top, breast kissing",
    "nude, bedside, one woman sitting, other kneeling, oral sex",
    "nude, standing, one woman lifting the other, kissing",
    "nude, floor, making out, fingers between thighs",
    "nude, chest-to-chest, tongues, hands on ass",
    "nude, candlelight, fingering, eye contact",
    "nude, bedroom, intimate close-up, biting lip",
    "nude, bed, both women on all fours, kissing, hands between thighs",
    "nude, one woman sitting on lap, grinding, kissing",
    "nude, close-up faces, orgasmic expression, hands on breasts",
    "nude, kissing while one woman fingers the other",
    "nude, soft sheets, mutual touching, close-up faces",
]

MIXED_SCENES = [
    "lingerie, kissing, hands on waist, bedroom",
    "topless, panties, cuddling on bed, soft light",
    "sheer robe, nipples visible, hugging, warm light",
    "bra and panties, mirror portrait, close-up faces",
    "bikini tops, wet hair, shower kiss, soft steam",
    "see-through tank top, kissing, soft bedroom light",
    "lacy underwear, sitting on couch, cuddling",
    "topless, short shorts, playful pose, bedroom",
    "silk slip dresses, kissing, soft ambient light",
    "stockings and garter belts, close embrace, teasing smiles",
    "towels, post-shower, kissing, wet skin",
    "strapless tops, close-up portrait, teasing smiles",
]

SEED_BASE = 9204371


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = ComfyClient(base_url=BASE_URL)

    scenes = [("explicit", s) for s in EXPLICIT_SCENES] + [("mixed", s) for s in MIXED_SCENES]
    if len(scenes) != 40:
        raise RuntimeError(f"Expected 40 scenes, got {len(scenes)}")

    manifest = []
    for index, (scene_type, scene) in enumerate(scenes, start=1):
        positive = f"{BASE_PROMPT}, {scene}"

        try:
            result = generate_from_workflow(
                client=client,
                positive=positive,
                negative=NEGATIVE_PROMPT,
                loras=LORAS,
                seed=SEED_BASE + index,
                steps=30,
                cfg=3.5,
                sampler_name="dpmpp_2m_sde",
                scheduler="karras",
                denoise=1.0,
                output_prefix=f"duo_gen_{index:03d}",
            )
        except urllib.error.URLError as exc:
            print(
                f"ComfyUI is not reachable at {BASE_URL}. Start the server or set COMFY_URL.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

        images = result.get("images", [])
        if not images:
            raise RuntimeError(f"No images returned for scene {index}")

        saved = []
        for image in images:
            data = client.download_image(
                image["filename"],
                subfolder=image.get("subfolder", ""),
                image_type=image.get("type", "output"),
            )
            target = OUTPUT_DIR / image["filename"]
            target.write_bytes(data)

            caption_path = target.with_suffix(".txt")
            caption_path.write_text(positive, encoding="utf-8")
            saved.append(str(target))

        manifest.append(
            {
                "scene_index": index,
                "scene_type": scene_type,
                "prompt_id": result["prompt_id"],
                "positive": positive,
                "negative": NEGATIVE_PROMPT,
                "images": saved,
            }
        )

    (OUTPUT_DIR / "duo_gen_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
