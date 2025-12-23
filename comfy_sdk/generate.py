from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .client import ComfyClient
from .workflow import (
    load_workflow,
    set_checkpoint,
    set_latent_size,
    set_lora,
    set_loras,
    set_negative_prompt,
    set_output_prefix,
    set_positive_prompt,
    set_sampler_params,
    workflow_to_prompt,
)

DEFAULT_WORKFLOW_PATH = Path(__file__).resolve().parents[1] / "workflows" / "girls_workflow.json"


def build_prompt_from_workflow(
    workflow_path: str | Path = DEFAULT_WORKFLOW_PATH,
    *,
    positive: str | None = None,
    negative: str | None = None,
    seed: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    sampler_name: str | None = None,
    scheduler: str | None = None,
    denoise: float | None = None,
    width: int | None = None,
    height: int | None = None,
    batch_size: int | None = None,
    ckpt_name: str | None = None,
    lora_name: str | None = None,
    lora_strength_model: float | None = None,
    lora_strength_clip: float | None = None,
    loras: list[dict[str, Any]] | None = None,
    output_prefix: str | None = None,
) -> Dict[str, Any]:
    workflow = load_workflow(workflow_path)
    prompt = workflow_to_prompt(workflow)

    if positive is not None:
        set_positive_prompt(prompt, positive)
    if negative is not None:
        set_negative_prompt(prompt, negative)
    if any(value is not None for value in (seed, steps, cfg, sampler_name, scheduler, denoise)):
        set_sampler_params(
            prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
        )
    if any(value is not None for value in (width, height, batch_size)):
        set_latent_size(prompt, width=width, height=height, batch_size=batch_size)
    if ckpt_name is not None:
        set_checkpoint(prompt, ckpt_name)
    if loras is not None:
        set_loras(prompt, loras)
    elif lora_name is not None:
        set_lora(
            prompt,
            lora_name,
            strength_model=lora_strength_model,
            strength_clip=lora_strength_clip,
        )
    if output_prefix is not None:
        set_output_prefix(prompt, output_prefix)

    return prompt


def generate_from_workflow(
    workflow_path: str | Path = DEFAULT_WORKFLOW_PATH,
    *,
    client: ComfyClient | None = None,
    wait: bool = True,
    poll_interval: float = 1.0,
    timeout: float = 300.0,
    **overrides: Any,
) -> Dict[str, Any]:
    comfy = client or ComfyClient()
    prompt = build_prompt_from_workflow(workflow_path, **overrides)
    prompt_id = comfy.queue_prompt(prompt)

    if not wait:
        return {"prompt_id": prompt_id}

    history = comfy.wait_for_prompt(prompt_id, poll_interval=poll_interval, timeout=timeout)
    images = comfy.extract_images(history, prompt_id)
    return {"prompt_id": prompt_id, "images": images, "history": history}
