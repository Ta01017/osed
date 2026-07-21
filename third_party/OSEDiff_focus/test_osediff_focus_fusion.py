"""Focus-fusion inference with complete debug output and B-condition ablations."""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataloaders.focus_fusion_dataset import FocusFusionDataset, FIXED_FUSION_PROMPT, focus_fusion_collate
from osediff_focus_fusion import (FocusFusionGenerator, expand_unet_conv_in, load_focus_checkpoint,
                                  move_scheduler_to_device, normalize_input_mode, read_focus_checkpoint_config,
                                  load_vae_lora_state, load_verified_checkpoint)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--metadata_path", required=True); p.add_argument("--dataset_base_path", required=True)
    p.add_argument("--output_dir", required=True); p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--start_index", type=int, default=0); p.add_argument("--max_samples", type=int)
    p.add_argument("--input_mode", choices=["single", "dual", "quad_rgb", "ab_focus", "ab", "four"], default=None)
    p.add_argument("--condition_ablation", choices=["normal", "refs_equal_a", "refs_zero", "b_equals_a", "b_zero"], default="normal")
    p.add_argument("--run_all_ablations", action="store_true")
    p.add_argument("--vae_encode_mode", choices=["sample", "mode"], default="mode")
    p.add_argument("--prompt_mode", choices=["fixed", "metadata"], default="fixed")
    p.add_argument("--cache_fixed_prompt_embedding", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keep_a_composite", action="store_true"); p.add_argument("--keep_threshold", type=float, default=.5)
    p.add_argument("--keep_soft_width", type=float, default=.1); p.add_argument("--seed", type=int, default=123)
    p.add_argument("--mixed_precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--tiled", action="store_true"); p.add_argument("--latent_tiled_size", type=int, default=96)
    p.add_argument("--latent_tiled_overlap", type=int, default=32)
    p.add_argument("--native_resolution", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict_native_size", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_pixels", type=int, help="Maximum allowed native pixels. Images are rejected rather than resized.")
    return p.parse_args(argv)


def _pil(x, focus=False):
    x = x.detach().float().cpu()[0]
    if not focus: x = x.mul(.5).add(.5)
    a = x.clamp(0, 1).mul(255).byte().numpy()
    if a.shape[0] == 1: return Image.fromarray(a[0], "L")
    return Image.fromarray(a.transpose(1, 2, 0), "RGB")


def soft_keep(mask, threshold, width):
    if width <= 0: return (mask >= threshold).to(mask.dtype)
    return ((mask - (threshold - width / 2)) / width).clamp(0, 1)


def prepare_condition_latents(model, condition_images, vae_encode_mode, generator=None):
    del generator
    with torch.no_grad():
        return model.encode_images(*condition_images, mode=vae_encode_mode)


def ablation_latents_and_focus(input_mode, mode, base_latents, focus_maps):
    latents = [z.clone() for z in base_latents]
    fmaps = [f.clone() for f in focus_maps]
    if mode == "normal":
        return latents, fmaps
    if mode == "refs_equal_a":
        for i in range(1, len(latents)):
            latents[i] = latents[0].clone()
            assert torch.equal(latents[i], latents[0])
        if input_mode == "ab_focus":
            fmaps[1] = fmaps[0].clone()
        return latents, fmaps
    if mode == "refs_zero":
        for i in range(1, len(latents)):
            latents[i] = torch.zeros_like(latents[i])
        if input_mode == "ab_focus":
            fmaps[1] = torch.zeros_like(fmaps[1])
        return latents, fmaps
    raise ValueError(f"unknown condition ablation: {mode}")


def run_ablation_from_latents(model, input_mode, mode, base_latents, focus_maps, prompt_embeds, args):
    latents, ablated_focus = ablation_latents_and_focus(input_mode, mode, base_latents, focus_maps)
    fa = ablated_focus[0] if ablated_focus else None
    fb = ablated_focus[1] if len(ablated_focus) > 1 else None
    with torch.no_grad():
        return model.forward_from_latents(latents, fa, fb, prompt_embeds, args.tiled, args.latent_tiled_size, args.latent_tiled_overlap)[0]


def main(args):
    from diffusers import DDPMScheduler
    from transformers import AutoTokenizer, CLIPTextModel
    from models.autoencoder_kl import AutoencoderKL
    from models.unet_2d_condition import UNet2DConditionModel
    verified = load_verified_checkpoint(args.checkpoint_path)
    ckpt = read_focus_checkpoint_config(verified["model_state"])
    state = ckpt["state"]
    ckpt_mode = ckpt["input_mode"]
    args.input_mode = normalize_input_mode(args.input_mode or ckpt_mode)
    if args.input_mode != ckpt_mode:
        raise RuntimeError(f"input_mode mismatch: checkpoint={ckpt_mode}, requested={args.input_mode}")
    args.condition_ablation = {"b_equals_a": "refs_equal_a", "b_zero": "refs_zero"}.get(args.condition_ablation, args.condition_ablation)
    if args.keep_a_composite and args.input_mode != "ab_focus":
        raise RuntimeError("--keep_a_composite is only valid for ab_focus")
    if args.input_mode == "single":
        if args.run_all_ablations:
            raise RuntimeError("single input_mode supports only condition_ablation=normal; --run_all_ablations would include refs_equal_a/refs_zero")
        if args.condition_ablation in ("refs_equal_a", "refs_zero"):
            raise RuntimeError(f"single input_mode supports only condition_ablation=normal, got {args.condition_ablation}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.mixed_precision]
    if device.type == "cpu": dtype = torch.float32
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cached_prompt = state.get("fixed_prompt_embedding")
    have_cached_prompt = args.prompt_mode == "fixed" and args.cache_fixed_prompt_embedding and cached_prompt is not None and cached_prompt.numel()
    tokenizer = text = None
    if not have_cached_prompt:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        text = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    expand_unet_conv_in(unet, state["generator_in_channels"])
    # Adapters must exist before loading LoRA. Recreate exact target names stored by PEFT keys.
    lora_state = state.get("generator_unet_lora", {})
    rank = ckpt["generator_lora_rank"]
    if lora_state:
        from peft import LoraConfig
        targets = ckpt["generator_lora_targets"] or sorted({k.split(".lora_", 1)[0] for k in lora_state if ".lora_" in k})
        unet.add_adapter(LoraConfig(r=rank, target_modules=targets), adapter_name=ckpt["generator_lora_adapter_name"])
        unet.set_adapter([ckpt["generator_lora_adapter_name"]])
    vae_lora_state = state.get("vae_lora", {})
    if vae_lora_state:
        from peft import LoraConfig
        targets = ckpt["vae_lora_targets"]
        if not targets:
            raise RuntimeError("checkpoint contains VAE LoRA weights but no vae_lora_targets metadata")
        vae.add_adapter(LoraConfig(r=ckpt["vae_lora_rank"], target_modules=targets), adapter_name=ckpt["vae_lora_adapter_name"])
        vae.set_adapter([ckpt["vae_lora_adapter_name"]])
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler"); scheduler.set_timesteps(1, device=device)
    move_scheduler_to_device(scheduler, device)
    model = FocusFusionGenerator(unet, vae, scheduler, args.input_mode)
    load_focus_checkpoint(model, state)
    if vae_lora_state:
        load_vae_lora_state(model, state)
    model.to(device, dtype=dtype).eval()
    if text is not None:
        text.to(device, dtype=dtype).eval()
    dataset = FocusFusionDataset(args.metadata_path, args.dataset_base_path, args.resolution,
        max_samples=args.max_samples, start_index=args.start_index, prompt_mode=args.prompt_mode,
        native_resolution=args.native_resolution, strict_native_size=args.strict_native_size,
        input_mode=args.input_mode, vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1),
        max_pixels=args.max_pixels)

    def embeds(prompts):
        cached = model.fixed_prompt_embedding
        if args.prompt_mode == "fixed" and cached.numel(): return cached.to(device, dtype=dtype).expand(len(prompts), -1, -1)
        if tokenizer is None or text is None:
            raise RuntimeError("fixed prompt embedding is absent; tokenizer/text encoder must be available")
        ids = tokenizer(prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
        return text(ids)[0]
    modes = ["normal", "refs_equal_a", "refs_zero"] if args.run_all_ablations else [args.condition_ablation]
    stats = {"sum_equal": 0.0, "sum_zero": 0.0, "count": 0}
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=focus_fusion_collate):
        conditions = [x.to(device, dtype=dtype) for x in batch["conditions"]]
        focus_maps = [x.to(device, dtype=dtype) for x in batch["focus_maps"]]
        prompt = embeds(batch["prompt"])
        base_latents = prepare_condition_latents(model, conditions, args.vae_encode_mode)
        rendered = {}
        for mode in modes:
            if args.input_mode == "single" and mode != "normal":
                raise RuntimeError(f"{mode} is not applicable to single input_mode")
            _, save_focus_maps = ablation_latents_and_focus(args.input_mode, mode, base_latents, focus_maps)
            fa = save_focus_maps[0] if save_focus_maps else None
            raw = run_ablation_from_latents(model, args.input_mode, mode, base_latents, focus_maps, prompt, args)
            rendered[mode] = raw
            out_root = Path(args.output_dir, mode); out_root.mkdir(parents=True, exist_ok=True)
            final = raw
            if args.keep_a_composite:
                keep_mask = soft_keep(fa, args.keep_threshold, args.keep_soft_width)
                final = keep_mask * conditions[0] + (1 - keep_mask) * raw
            idx = int(batch["metadata_index"].item()); folder = out_root / f"{idx:06d}"; folder.mkdir(exist_ok=True)
            images = {"pred_raw.png": _pil(raw), "final.png": _pil(final), "GT.png": _pil(batch["gt"])}
            for i, cond in enumerate(conditions):
                images[f"{chr(ord('A') + i)}.png"] = _pil(cond)
            for i, fmap in enumerate(save_focus_maps):
                images[f"focus_{i}.png"] = _pil(fmap, True)
            if args.keep_a_composite:
                images["pred_keepa_composite.png"] = _pil(final)
            for name, image in images.items(): image.save(folder / name)
            panel_names = [f"{chr(ord('A') + i)}.png" for i in range(len(conditions))] + ["GT.png"] + [f"focus_{i}.png" for i in range(len(save_focus_maps))] + ["pred_raw.png"]
            if args.keep_a_composite:
                panel_names.append("pred_keepa_composite.png")
            panels = [images[n].convert("RGB") for n in panel_names]
            canvas = Image.new("RGB", (sum(x.width for x in panels), max(x.height for x in panels)))
            x = 0
            for panel in panels: canvas.paste(panel, (x, 0)); x += panel.width
            canvas.save(folder / "comparison_A_B_GT_raw_keepa.png")
        if "normal" in rendered and "refs_equal_a" in rendered:
            stats["sum_equal"] += float((rendered["normal"] - rendered["refs_equal_a"]).abs().mean().item())
        if "normal" in rendered and "refs_zero" in rendered:
            stats["sum_zero"] += float((rendered["normal"] - rendered["refs_zero"]).abs().mean().item())
        stats["count"] += 1
        del rendered, base_latents
    if len(modes) > 1:
        out_stats = {
            "normal_vs_refs_equal_a_mae": stats["sum_equal"] / max(stats["count"], 1),
            "normal_vs_refs_zero_mae": stats["sum_zero"] / max(stats["count"], 1),
            "sample_count": stats["count"],
        }
        print(json.dumps(out_stats, indent=2))
        if min(out_stats["normal_vs_refs_equal_a_mae"], out_stats["normal_vs_refs_zero_mae"]) < 1e-5:
            print("WARNING: ablation difference is near zero; the model may not use B")


if __name__ == "__main__": main(parse_args())
