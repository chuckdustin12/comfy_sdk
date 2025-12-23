from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

Prompt = Dict[str, Dict[str, Any]]


def load_workflow(path: str | Path) -> Dict[str, Any]:
    workflow_path = Path(path)
    with workflow_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def workflow_to_prompt(workflow: Mapping[str, Any]) -> Prompt:
    links_by_id = {link[0]: link for link in workflow.get("links", [])}
    prompt: Prompt = {}

    for node in workflow.get("nodes", []):
        node_id = str(node["id"])
        node_type = node["type"]
        inputs: Dict[str, Any] = {}

        widget_values = list(node.get("widgets_values") or [])
        widget_inputs = [item for item in node.get("inputs", []) if "widget" in item]
        if (
            node_type == "KSampler"
            and len(widget_values) == len(widget_inputs) + 1
            and len(widget_values) >= 2
            and isinstance(widget_values[1], str)
        ):
            # ComfyUI stores the seed "control_after_generate" after the seed value in UI workflows.
            widget_values = [widget_values[0]] + widget_values[2:]

        widget_index = 0
        for input_def in node.get("inputs", []):
            name = input_def.get("name")
            if not name:
                continue

            link_id = input_def.get("link")
            if link_id is not None:
                link = links_by_id.get(link_id)
                if link is None:
                    continue
                inputs[name] = [str(link[1]), link[2]]
                continue

            if "widget" in input_def:
                if widget_index < len(widget_values):
                    inputs[name] = widget_values[widget_index]
                    widget_index += 1

        prompt[node_id] = {"inputs": inputs, "class_type": node_type}

    return prompt


def find_nodes_by_type(prompt: Prompt, class_type: str) -> Iterable[str]:
    return [node_id for node_id, node in prompt.items() if node.get("class_type") == class_type]


def _get_node_id_by_type(prompt: Prompt, class_type: str, index: int) -> str:
    matches = list(find_nodes_by_type(prompt, class_type))
    if not matches:
        raise ValueError(f"No nodes found for class_type={class_type!r}")
    try:
        return matches[index]
    except IndexError as exc:
        raise ValueError(
            f"Requested {class_type!r} index {index} but only {len(matches)} nodes exist"
        ) from exc


def _max_node_id(prompt: Prompt) -> int:
    numeric_ids = [int(node_id) for node_id in prompt.keys() if node_id.isdigit()]
    return max(numeric_ids, default=0)


def set_clip_text(prompt: Prompt, node_id: str, text: str) -> None:
    node = prompt.get(str(node_id))
    if not node:
        raise ValueError(f"Unknown node_id={node_id!r}")
    if node.get("class_type") != "CLIPTextEncode":
        raise ValueError(f"Node {node_id!r} is not a CLIPTextEncode node")
    node.setdefault("inputs", {})["text"] = text


def set_sampler_prompt(prompt: Prompt, text: str, role: str = "positive", sampler_index: int = 0) -> None:
    if role not in {"positive", "negative"}:
        raise ValueError("role must be 'positive' or 'negative'")

    sampler_id = _get_node_id_by_type(prompt, "KSampler", sampler_index)
    sampler_inputs = prompt[sampler_id].get("inputs", {})
    link = sampler_inputs.get(role)

    if not (isinstance(link, list) and link):
        raise ValueError(f"KSampler {sampler_id} does not link a {role} prompt")

    clip_id = str(link[0])
    set_clip_text(prompt, clip_id, text)


def set_positive_prompt(prompt: Prompt, text: str, sampler_index: int = 0) -> None:
    set_sampler_prompt(prompt, text, role="positive", sampler_index=sampler_index)


def set_negative_prompt(prompt: Prompt, text: str, sampler_index: int = 0) -> None:
    set_sampler_prompt(prompt, text, role="negative", sampler_index=sampler_index)


def set_sampler_params(
    prompt: Prompt,
    sampler_index: int = 0,
    *,
    seed: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    sampler_name: str | None = None,
    scheduler: str | None = None,
    denoise: float | None = None,
) -> None:
    sampler_id = _get_node_id_by_type(prompt, "KSampler", sampler_index)
    inputs = prompt[sampler_id].setdefault("inputs", {})

    if seed is not None:
        inputs["seed"] = seed
    if steps is not None:
        inputs["steps"] = steps
    if cfg is not None:
        inputs["cfg"] = cfg
    if sampler_name is not None:
        inputs["sampler_name"] = sampler_name
    if scheduler is not None:
        inputs["scheduler"] = scheduler
    if denoise is not None:
        inputs["denoise"] = denoise


def set_latent_size(
    prompt: Prompt,
    *,
    width: int | None = None,
    height: int | None = None,
    batch_size: int | None = None,
    latent_index: int = 0,
) -> None:
    latent_id = _get_node_id_by_type(prompt, "EmptyLatentImage", latent_index)
    inputs = prompt[latent_id].setdefault("inputs", {})

    if width is not None:
        inputs["width"] = width
    if height is not None:
        inputs["height"] = height
    if batch_size is not None:
        inputs["batch_size"] = batch_size


def set_checkpoint(prompt: Prompt, ckpt_name: str, checkpoint_index: int = 0) -> None:
    ckpt_id = _get_node_id_by_type(prompt, "CheckpointLoaderSimple", checkpoint_index)
    prompt[ckpt_id].setdefault("inputs", {})["ckpt_name"] = ckpt_name


def set_lora(
    prompt: Prompt,
    lora_name: str,
    *,
    strength_model: float | None = None,
    strength_clip: float | None = None,
    lora_index: int = 0,
) -> None:
    lora_id = _get_node_id_by_type(prompt, "LoraLoader", lora_index)
    inputs = prompt[lora_id].setdefault("inputs", {})
    inputs["lora_name"] = lora_name

    if strength_model is not None:
        inputs["strength_model"] = strength_model
    if strength_clip is not None:
        inputs["strength_clip"] = strength_clip


def add_lora_to_chain(
    prompt: Prompt,
    lora_name: str,
    *,
    strength_model: float | None = None,
    strength_clip: float | None = None,
    base_lora_id: str | None = None,
    sampler_index: int = 0,
) -> str:
    sampler_id = _get_node_id_by_type(prompt, "KSampler", sampler_index)
    sampler_inputs = prompt[sampler_id].get("inputs", {})

    if base_lora_id is None:
        model_link = sampler_inputs.get("model")
        if not (isinstance(model_link, list) and model_link):
            raise ValueError(f"KSampler {sampler_id} does not link a model input")
        base_lora_id = str(model_link[0])

    new_id = str(_max_node_id(prompt) + 1)
    lora_inputs: Dict[str, Any] = {
        "model": [str(base_lora_id), 0],
        "clip": [str(base_lora_id), 1],
        "lora_name": lora_name,
    }
    if strength_model is not None:
        lora_inputs["strength_model"] = strength_model
    if strength_clip is not None:
        lora_inputs["strength_clip"] = strength_clip

    prompt[new_id] = {"inputs": lora_inputs, "class_type": "LoraLoader"}

    sampler_inputs["model"] = [new_id, 0]
    for node in prompt.values():
        if node.get("class_type") != "CLIPTextEncode":
            continue
        inputs = node.get("inputs", {})
        clip_link = inputs.get("clip")
        if isinstance(clip_link, list) and clip_link and str(clip_link[0]) == str(base_lora_id):
            inputs["clip"] = [new_id, clip_link[1] if len(clip_link) > 1 else 1]

    return new_id


def set_loras(
    prompt: Prompt,
    loras: Sequence[Mapping[str, Any]],
    *,
    sampler_index: int = 0,
) -> None:
    if not loras:
        return

    sampler_id = _get_node_id_by_type(prompt, "KSampler", sampler_index)
    sampler_inputs = prompt[sampler_id].get("inputs", {})
    model_link = sampler_inputs.get("model")
    if not (isinstance(model_link, list) and model_link):
        raise ValueError(f"KSampler {sampler_id} does not link a model input")

    base_id = str(model_link[0])
    base_node = prompt.get(base_id)

    current_id = base_id
    first = loras[0]
    if base_node and base_node.get("class_type") == "LoraLoader":
        inputs = base_node.setdefault("inputs", {})
        inputs["lora_name"] = first["lora_name"]
        if "strength_model" in first:
            inputs["strength_model"] = first["strength_model"]
        if "strength_clip" in first:
            inputs["strength_clip"] = first["strength_clip"]
    else:
        current_id = add_lora_to_chain(
            prompt,
            first["lora_name"],
            strength_model=first.get("strength_model"),
            strength_clip=first.get("strength_clip"),
            base_lora_id=base_id,
            sampler_index=sampler_index,
        )

    for lora in loras[1:]:
        current_id = add_lora_to_chain(
            prompt,
            lora["lora_name"],
            strength_model=lora.get("strength_model"),
            strength_clip=lora.get("strength_clip"),
            base_lora_id=current_id,
            sampler_index=sampler_index,
        )


def set_output_prefix(prompt: Prompt, prefix: str, save_index: int = 0) -> None:
    save_id = _get_node_id_by_type(prompt, "SaveImage", save_index)
    prompt[save_id].setdefault("inputs", {})["filename_prefix"] = prefix
