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
OUTPUT_DIR = Path(os.environ.get("RAPUNZEL_OUTPUT_DIR", "generated/rapunzel_sfw"))
OUTPUT_PREFIX = os.environ.get("RAPUNZEL_OUTPUT_PREFIX", "rapunzel_sfw")
CKPT_NAME = os.environ.get("CKPT_NAME", "illustriousMixedCGI_v20.safetensors")

RAPUNZEL_LORA = os.environ.get("RAPUNZEL_LORA", "TANGLEDFUN2.safetensors")
RAPUNZEL_LORA_STRENGTH = float(os.environ.get("RAPUNZEL_LORA_STRENGTH", "0.9"))
STYLE_LORA = os.environ.get("STYLE_LORA")
STYLE_LORA_STRENGTH = float(os.environ.get("STYLE_LORA_STRENGTH", "0.6"))

REF_IMAGE = Path(os.environ.get("RAPUNZEL_REF", "example_images/SFW/rapunzel/0.png"))
REF_NAME = "rapunzel_ref.png"
COMFY_INPUT_DIR = Path("C:/Users/chuck/ComfyUI/input")

FACEID_PRESET = os.environ.get("FACEID_PRESET", "FACEID PLUS V2")
FACEID_PROVIDER = os.environ.get("FACEID_PROVIDER", "CUDA")
FACEID_LORA_STRENGTH = float(os.environ.get("FACEID_LORA_STRENGTH", "0.9"))
FACEID_WEIGHT = float(os.environ.get("FACEID_WEIGHT", "1.0"))
FACEID_WEIGHT_TYPE = os.environ.get("FACEID_WEIGHT_TYPE", "composition")

BASE_PROMPT = (
    "rapunzel, tangled, adult woman, long blonde hair, green eyes, "
    "purple dress, pink lace, soft warm lighting, 3d animated film style, "
    "movie-accurate, detailed face, gentle expression"
)

SCENES = [
    "close-up portrait, soft window light, subtle smile",
    "mid-shot in her tower, warm wooden interior, hands clasped",
    "outdoor forest path, golden sunlight, holding frying pan",
    "lantern festival night, glowing lanterns, dreamy bokeh",
    "painting on the wall, colorful mural, playful smile",
    "sitting by a window, breeze in hair, sunlit dust motes",
    "braided hair with flowers, garden scene, soft pastel light",
    "full body in purple dress, standing in a tower doorway",
]

NEGATIVE_PROMPT = (
    "nsfw, nude, naked, lingerie, underwear, bra, panties, bikini, "
    "cleavage, see-through, exposed midriff, nipples, fetish, explicit, "
    "underage, child, teen, lowres, blurry, jpeg artifacts, bad anatomy, "
    "bad proportions, extra limbs, cropped face, out of frame"
)

SEED_BASE = int(os.environ.get("SEED_BASE", "2389117"))


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


def _insert_faceid_single(prompt: dict[str, dict[str, object]], *, ref_image: str) -> None:
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

    ref_img_id = _next_id(prompt)
    prompt[ref_img_id] = {"class_type": "LoadImage", "inputs": {"image": ref_image}}

    mask_id = _next_id(prompt)
    prompt[mask_id] = {
        "class_type": "CreateShapeMask",
        "inputs": {
            "shape": "square",
            "frames": 1,
            "location_x": width // 2,
            "location_y": height // 2,
            "grow": 0,
            "frame_width": width,
            "frame_height": height,
            "shape_width": width,
            "shape_height": height,
        },
    }

    region_id = _next_id(prompt)
    prompt[region_id] = {
        "class_type": "IPAdapterRegionalConditioning",
        "inputs": {
            "image": [ref_img_id, 0],
            "image_weight": FACEID_WEIGHT,
            "prompt_weight": 1.0,
            "weight_type": FACEID_WEIGHT_TYPE,
            "start_at": 0.0,
            "end_at": 1.0,
            "mask": [mask_id, 0],
        },
    }

    apply_id = _next_id(prompt)
    prompt[apply_id] = {
        "class_type": "IPAdapterFromParams",
        "inputs": {
            "model": [face_loader_id, 0],
            "ipadapter": [face_loader_id, 1],
            "ipadapter_params": [region_id, 0],
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


def _build_loras() -> list[dict[str, object]]:
    loras = [
        {
            "lora_name": RAPUNZEL_LORA,
            "strength_model": RAPUNZEL_LORA_STRENGTH,
            "strength_clip": RAPUNZEL_LORA_STRENGTH,
        }
    ]
    if STYLE_LORA:
        loras.append(
            {
                "lora_name": STYLE_LORA,
                "strength_model": STYLE_LORA_STRENGTH,
                "strength_clip": STYLE_LORA_STRENGTH,
            }
        )
    return loras


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = ComfyClient(base_url=BASE_URL)

    ref_name = _prepare_ref_image(REF_IMAGE, REF_NAME)
    loras = _build_loras()

    manifest = []
    for index, scene in enumerate(SCENES, start=1):
        positive = f"{BASE_PROMPT}, {scene}"
        prompt = build_prompt_from_workflow(
            positive=positive,
            negative=NEGATIVE_PROMPT,
            ckpt_name=CKPT_NAME,
            loras=loras,
            seed=SEED_BASE + index,
            steps=30,
            cfg=3.5,
            sampler_name="dpmpp_2m_sde",
            scheduler="karras",
            denoise=1.0,
            width=1024,
            height=1024,
            output_prefix=f"{OUTPUT_PREFIX}_{index:03d}",
        )

        _insert_faceid_single(prompt, ref_image=ref_name)

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
                "prompt_id": prompt_id,
                "positive": positive,
                "negative": NEGATIVE_PROMPT,
                "images": saved,
            }
        )

    (OUTPUT_DIR / "rapunzel_sfw_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
