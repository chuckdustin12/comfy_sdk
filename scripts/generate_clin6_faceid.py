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
OUTPUT_DIR = Path(os.environ.get("CLIN6_OUTPUT_DIR", "generated/clin6_hq"))
OUTPUT_PREFIX = os.environ.get("CLIN6_OUTPUT_PREFIX", "clin6_hq")
CKPT_NAME = os.environ.get("CKPT_NAME", "illustriousMixedCGI_v20.safetensors")

CLIN6_LORA = os.environ.get("CLIN6_LORA", "CLIN8-000009.safetensors")
CLIN6_LORA_STRENGTH = float(os.environ.get("CLIN6_LORA_STRENGTH", "1.0"))
STYLE_LORA = os.environ.get("STYLE_LORA")
STYLE_LORA_STRENGTH = float(os.environ.get("STYLE_LORA_STRENGTH", "1"))

DEFAULT_REFS = [
    Path("example_images/SFW/caitlin/ComfyUI_02289_.png"),
    Path("example_images/SFW/caitlin/ComfyUI_02286_.png"),
    Path("example_images/SFW/caitlin/ComfyUI_02291_.png"),
    Path("example_images/SFW/caitlin/ComfyUI_02306_.png"),
    Path("example_images/SFW/caitlin/ComfyUI_02283_.png"),
    Path("example_images/SFW/caitlin/ComfyUI_01752_.png"),
]
REFS_ENV = os.environ.get("CLIN6_REFS", "")
REF_ENV_SINGLE = os.environ.get("CLIN6_REF", "")
if REFS_ENV:
    REF_IMAGES = [Path(p.strip()) for p in REFS_ENV.split(",") if p.strip()]
elif REF_ENV_SINGLE:
    REF_IMAGES = [Path(REF_ENV_SINGLE.strip())]
else:
    REF_IMAGES = DEFAULT_REFS
COMFY_INPUT_DIR = Path("C:/Users/chuck/ComfyUI/input")

FACEID_PRESET = os.environ.get("FACEID_PRESET", "FACEID PLUS V2")
FACEID_PROVIDER = os.environ.get("FACEID_PROVIDER", "CUDA")
FACEID_LORA_STRENGTH = float(os.environ.get("FACEID_LORA_STRENGTH", "1"))
FACEID_WEIGHT = float(os.environ.get("FACEID_WEIGHT", "1.0"))
FACEID_WEIGHT_TYPE = os.environ.get("FACEID_WEIGHT_TYPE", "composition")

BASE_PROMPT = os.environ.get("CLIN_BASE_PROMPT", "CLIN6, adult woman, realistic photo")

SCENES = [
    # ---------- Portrait / identity anchors ----------
    "tight headshot, eye-level, 85mm portrait look, sharp focus on eyes, detailed skin texture, neutral background",
    "shoulders-up portrait, 50mm look, soft but crisp studio key light, catchlights visible, natural makeup",
    "three-quarter portrait, slight head tilt, window light from left, gentle shadows, clean background",

    # ---------- Mid / half body ----------
    "half body, standing, hands relaxed at sides, indoor apartment, natural daylight, realistic color",
    "half body, seated on chair, studio rim light + soft fill, dark seamless background, cinematic contrast",
    "half body, candid laugh, outdoor shade, even skin tones, urban background",

    # ---------- Full body near (face still readable) ----------
    "full body, 35mm environmental portrait look, subject centered, face clearly visible, outdoor park path, midday sun with soft shadow",
    "full body, walking toward camera, slight motion in hair, street scene, overcast daylight, natural colors",
    "full body, confident stance, low angle slightly, studio hard key light, crisp edges, high contrast",

    # ---------- Full body far (the hard mode) ----------
    "full body far shot, wide environment visible, 24mm wide lens feel, subject still facing camera, face visible, city plaza background",
    "full body far shot, telephoto compression, subject standing still, background softly separated, face visible, natural light",

    # ---------- Lighting extremes ----------
    "low light, single practical lamp, warm shadows, moody atmosphere, clean facial features",
    "backlit sunset, golden hour rim light, lens flare minimal, face still visible, warm tones",
    "night street lighting, mixed color temperature, realistic grain, face visible, sharp focus",

    # ---------- Wardrobe variety ----------
    "casual outfit, plain t-shirt and jeans, natural daylight, neutral color palette",
    "business casual, blazer, indoor office hallway, fluorescent overhead mixed with window light",
    "summer dress, outdoor garden, bright daylight, natural shadows",

    # ---------- Hair / styling variety ----------
    "hair tucked behind one ear, clean jawline visible, studio softbox, crisp detail",
    "hair loose and slightly windswept, outdoor daylight, face unobstructed, sharp focus",
    "hair pulled back, minimal makeup, neutral expression, sharp eye detail",

    # ---------- Composition / framing variety ----------
    "subject off-center rule of thirds, negative space background, calm pose, natural daylight",
    "symmetrical framing, centered portrait, studio lighting, high clarity",
    "over-the-shoulder turn toward camera, face visible, soft background, crisp subject",

    # ---------- Micro-detail / texture emphasis ----------
    "freckles visible, natural skin texture, no beauty filter look, sharp focus, realistic pores",
    "subtle specular highlights on cheekbones, clean edges, high detail, realistic photo",
]


NEGATIVE_PROMPT = (
    "lowres, blurry, jpeg artifacts, bad anatomy, bad proportions, "
    "extra limbs, cropped face, out of frame, "
    "duplicate face, identical faces, twins"
)

SEED_BASE = int(os.environ.get("SEED_BASE", "5542111"))


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


def _prepare_ref_images(refs: list[Path]) -> list[str]:
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_names = []
    for index, src in enumerate(refs, start=1):
        if not src.exists():
            raise FileNotFoundError(f"Reference image not found: {src}")
        target = COMFY_INPUT_DIR / f"clin6_ref_{index}{src.suffix}"
        if not target.exists():
            shutil.copyfile(src, target)
        ref_names.append(target.name)
    return ref_names


def _build_loras() -> list[dict[str, object]]:
    loras = [
        {
            "lora_name": CLIN6_LORA,
            "strength_model": CLIN6_LORA_STRENGTH,
            "strength_clip": CLIN6_LORA_STRENGTH,
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

    ref_names = _prepare_ref_images(REF_IMAGES)
    loras = _build_loras()

    manifest = []
    for index, scene in enumerate(SCENES, start=1):
        positive = f"{BASE_PROMPT}, {scene}"
        ref_name = ref_names[(index - 1) % len(ref_names)]
        prompt = build_prompt_from_workflow(
            positive=positive,
            negative=NEGATIVE_PROMPT,
            ckpt_name=CKPT_NAME,
            loras=loras,
            seed=SEED_BASE + index,
            steps=35,
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
                "ref_image": ref_name,
                "positive": positive,
                "negative": NEGATIVE_PROMPT,
                "images": saved,
            }
        )

    (OUTPUT_DIR / "clin6_hq_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
