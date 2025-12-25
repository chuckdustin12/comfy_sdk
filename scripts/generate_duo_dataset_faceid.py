from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.error
from pathlib import Path

from comfy_sdk import ComfyClient, build_prompt_from_workflow
from comfy_sdk.workflow import find_nodes_by_type


BASE_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8000")
OUTPUT_DIR = Path(os.environ.get("DUO_OUTPUT_DIR", "DUO/10_DUO"))
OUTPUT_PREFIX = os.environ.get("DUO_OUTPUT_PREFIX", "duo_faceid")
COMFY_INPUT_DIR = Path("C:/Users/chuck/ComfyUI/input")
CKPT_NAME = "illustriousMixedCGI_v20.safetensors"

AMBER_REF = Path(os.environ.get("AMBER_REF", "DUO/10_DUO/12.jpg"))
CAITLIN_REF = Path(os.environ.get("CAITLIN_REF", "DUO/10_DUO/CM32.jpg"))
AMBER_REF_NAME = "amber_ref.jpg"
CAITLIN_REF_NAME = "caitlin_ref.jpg"

AMBER_LORA = os.environ.get("AMBER_LORA", "AMBER8-000005.safetensors")
CLIN_LORA = os.environ.get("CLIN_LORA", "CLIN8-000003.safetensors")
AMBER_TOKEN = os.environ.get("AMBER_TOKEN", "AMBER6, young adult, blue eyes")
CLIN_TOKEN = os.environ.get("CLIN_TOKEN", "CLIN6, woman, green eyes, freckles")

LORAS = [
    {"lora_name": AMBER_LORA, "strength_model": 1, "strength_clip": 1},
    {"lora_name": CLIN_LORA, "strength_model": 1.1, "strength_clip": 1.1},
]

FACEID_PRESET = os.environ.get("FACEID_PRESET", "FACEID PLUS V2")
FACEID_PROVIDER = os.environ.get("FACEID_PROVIDER", "CUDA")
FACEID_LORA_STRENGTH = float(os.environ.get("FACEID_LORA_STRENGTH", "1"))
FACEID_WEIGHT = float(os.environ.get("FACEID_WEIGHT", "1.0"))
FACEID_WEIGHT_TYPE = os.environ.get("FACEID_WEIGHT_TYPE", "composition")

NEGATIVE_PROMPT = (
    "twins, identical faces, same face, face blending, averaged face, "
    "morphed features, duplicated identity, cloned face, "
    "lowres, blurry, jpeg artifacts, bad anatomy, bad proportions, "
    "out of frame, cropped face, extra limbs, missing limbs, "
    "underage, teen, child"
)

BASE_PROMPT = (
    "2girl, not twins, different faces, "
    "cinematic realistic photography, soft warm lighting, detailed skin texture, "
    "sharp focus, shallow depth of field, eye contact with camera, upper body, close-up, "
    "(left side: 1.2), "
    "(AMBER6:1.3), young adult woman, blue eyes, natural expression, "
    "(right side: 1.2), "
    "(CLIN7:1.3), woman with freckles, hazel eyes, distinct facial structure, "
    "cheek to cheek composition"
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

SEED_BASE = 4389017
DUO_COUNT = int(os.environ.get("DUO_COUNT", "40"))


def _next_id(prompt: dict[str, dict[str, object]]) -> str:
    ids = [int(node_id) for node_id in prompt.keys() if node_id.isdigit()]
    return str((max(ids) if ids else 0) + 1)


def _get_size(prompt: dict[str, dict[str, object]]) -> tuple[int, int]:
    for node_id in find_nodes_by_type(prompt, "EmptyLatentImage"):
        inputs = prompt[node_id].get("inputs", {})
        width = int(inputs.get("width", 1024))
        height = int(inputs.get("height", 1024))
        return width, height
    return 1024, 1024


def _insert_faceid_duo(
    prompt: dict[str, dict[str, object]],
    *,
    amber_ref: str,
    caitlin_ref: str,
) -> None:
    width, height = _get_size(prompt)

    ksampler_id = next(iter(find_nodes_by_type(prompt, "KSampler")), None)
    if not ksampler_id:
        raise RuntimeError("No KSampler node found")

    model_link = prompt[ksampler_id]["inputs"].get("model")
    if not (isinstance(model_link, list) and model_link):
        raise RuntimeError("KSampler model link is missing")

    face_loader_id = _next_id(prompt)
    prompt[face_loader_id] = {
        "class_type": "IPAdapterUnifiedLoaderFaceID",
        "inputs": {
            "model": model_link,
            "preset": FACEID_PRESET,
            "lora_strength": FACEID_LORA_STRENGTH,
            "provider": FACEID_PROVIDER,
        },
    }

    amber_img_id = _next_id(prompt)
    prompt[amber_img_id] = {"class_type": "LoadImage", "inputs": {"image": amber_ref}}

    caitlin_img_id = _next_id(prompt)
    prompt[caitlin_img_id] = {"class_type": "LoadImage", "inputs": {"image": caitlin_ref}}

    left_mask_id = _next_id(prompt)
    prompt[left_mask_id] = {
        "class_type": "CreateShapeMask",
        "inputs": {
            "shape": "square",
            "frames": 1,
            "location_x": width // 4,
            "location_y": height // 2,
            "grow": 0,
            "frame_width": width,
            "frame_height": height,
            "shape_width": width // 2,
            "shape_height": height,
        },
    }

    right_mask_id = _next_id(prompt)
    prompt[right_mask_id] = {
        "class_type": "CreateShapeMask",
        "inputs": {
            "shape": "square",
            "frames": 1,
            "location_x": (width * 3) // 4,
            "location_y": height // 2,
            "grow": 0,
            "frame_width": width,
            "frame_height": height,
            "shape_width": width // 2,
            "shape_height": height,
        },
    }

    amber_region_id = _next_id(prompt)
    prompt[amber_region_id] = {
        "class_type": "IPAdapterRegionalConditioning",
        "inputs": {
            "image": [amber_img_id, 0],
            "image_weight": FACEID_WEIGHT,
            "prompt_weight": 1.0,
            "weight_type": FACEID_WEIGHT_TYPE,
            "start_at": 0.0,
            "end_at": 1.0,
            "mask": [left_mask_id, 0],
        },
    }

    caitlin_region_id = _next_id(prompt)
    prompt[caitlin_region_id] = {
        "class_type": "IPAdapterRegionalConditioning",
        "inputs": {
            "image": [caitlin_img_id, 0],
            "image_weight": FACEID_WEIGHT,
            "prompt_weight": 1.0,
            "weight_type": FACEID_WEIGHT_TYPE,
            "start_at": 0.0,
            "end_at": 1.0,
            "mask": [right_mask_id, 0],
        },
    }

    combine_id = _next_id(prompt)
    prompt[combine_id] = {
        "class_type": "IPAdapterCombineParams",
        "inputs": {
            "params_1": [amber_region_id, 0],
            "params_2": [caitlin_region_id, 0],
        },
    }

    apply_id = _next_id(prompt)
    prompt[apply_id] = {
        "class_type": "IPAdapterFromParams",
        "inputs": {
            "model": [face_loader_id, 0],
            "ipadapter": [face_loader_id, 1],
            "ipadapter_params": [combine_id, 0],
            "combine_embeds": "concat",
            "embeds_scaling": "K+V",
        },
    }

    prompt[ksampler_id]["inputs"]["model"] = [apply_id, 0]


def _prepare_ref_image(src: Path, name: str) -> str:
    if not src.exists():
        raise FileNotFoundError(f"Reference image not found: {src}")
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = COMFY_INPUT_DIR / name
    if not target.exists():
        shutil.copyfile(src, target)
    return target.name


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = ComfyClient(base_url=BASE_URL)

    amber_ref_name = _prepare_ref_image(AMBER_REF, AMBER_REF_NAME)
    caitlin_ref_name = _prepare_ref_image(CAITLIN_REF, CAITLIN_REF_NAME)

    scenes = [("explicit", s) for s in EXPLICIT_SCENES] + [("mixed", s) for s in MIXED_SCENES]
    if len(scenes) != 40:
        raise RuntimeError(f"Expected 40 scenes, got {len(scenes)}")
    if DUO_COUNT < 1 or DUO_COUNT > len(scenes):
        raise ValueError(f"DUO_COUNT must be between 1 and {len(scenes)}")
    scenes = scenes[:DUO_COUNT]

    manifest = []
    for index, (scene_type, scene) in enumerate(scenes, start=1):
        positive = f"{BASE_PROMPT}, {scene}"
        prompt = build_prompt_from_workflow(
            positive=positive,
            negative=NEGATIVE_PROMPT,
            ckpt_name=CKPT_NAME,
            loras=LORAS,
            seed=SEED_BASE + index,
            steps=30,
            cfg=3.5,
            sampler_name="dpmpp_2m_sde",
            scheduler="karras",
            denoise=1.0,
            output_prefix=f"{OUTPUT_PREFIX}_{index:03d}",
        )

        _insert_faceid_duo(
            prompt,
            amber_ref=amber_ref_name,
            caitlin_ref=caitlin_ref_name,
        )

        try:
            prompt_id = client.queue_prompt(prompt)
        except urllib.error.URLError as exc:
            print(
                f"ComfyUI is not reachable at {BASE_URL}. Start the server or set COMFY_URL.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

        history = client.wait_for_prompt(prompt_id, poll_interval=1.0, timeout=600.0)
        images = client.extract_images(history, prompt_id)
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
                "prompt_id": prompt_id,
                "positive": positive,
                "negative": NEGATIVE_PROMPT,
                "images": saved,
            }
        )

    (OUTPUT_DIR / "duo_faceid_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
