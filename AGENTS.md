AGENTS.md
Overview

This agent automates two core tasks using ComfyUI and Python:

Captioning Images: Generating descriptive captions for both SFW (Safe For Work) and NSFW (Not Safe For Work) example images.

Generating New Images: Creating new images (SFW or NSFW) using ComfyUI, driven by prompt-based workflows and optionally enhanced by captions or LoRA/SDXL training inputs.

The agent supports batch processing, flexible prompt construction, and leverages existing image samples for training, fine-tuning, and testing.

1. Captioning Task
Purpose

Automated annotation of large image datasets for LoRA or SDXL training.

Consistent, prompt-friendly descriptions for SFW and NSFW domains.

Image Inputs

SFW Example Images: e.g., portraits, standard poses, business attire, etc.

NSFW Example Images: e.g., explicit scenes, erotica, or adult-themed images.

Captioning Logic

Each image is passed through a Python captioning pipeline.

The agent can use:

Manual captions: Predefined phrases for known images or themes.

Auto-captioning models: BLIP, CLIP, or custom scripts for novel content.

NSFW EXAMPLES:

Using AMBER6, young adult,

AMBER6, young adult, nude, small perky breasts, spread legs, explicit view, seductive blue eyes, blonde hair in pigtails, on bed, wet skin, soft lighting, POV, inviting, lustful expression

AMBER6, young adult, topless, thong pulled aside, fingers in pussy, looking up, drooling, on knees, begging, studio lights, detailed skin texture

AMBER6, young adult, riding cock, fully nude, grabbing her own ass, tight stomach, moaning, flushed cheeks, intense eye contact, perfect tits bouncing, close-up, high detail

Using CLIN6, woman,

CLIN6, woman, nude, thick thighs, curvy hips, large round ass, oily skin, straddling, cum dripping, licking lips, brown hair, confident, explicit sexual pose, detailed lighting

CLIN6, woman, kneeling, hands tied behind back, deepthroating, spit running down chin, tits out, face covered in cum, looking up submissively, POV, hardcore

CLIN6, woman, bent over, getting fucked from behind, breasts swinging, open mouth, moaning, arching her back, close-up of penetration, glistening skin

Using Rapunzel, tangled,

Rapunzel, tangled, nude, very long blonde hair, wrapped around wrists, on all fours, ass up, pussy exposed, looking back with a teasing smile, fantasy setting, glowing skin, high detail

Rapunzel, tangled, tied up with her own hair, breasts squeezed, nipples erect, flushed cheeks, wet between legs, fairy-tale castle background, soft golden lighting

Rapunzel, tangled, sitting naked by the window, hair cascading down, legs spread, fingers in pussy, biting lip, innocent but explicit, magical sparkle, POV from below.




PURPOSE:

Auto-generate scripts and execute generations using ComfyUI.