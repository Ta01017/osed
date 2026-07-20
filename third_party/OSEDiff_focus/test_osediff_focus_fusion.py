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

from dataloaders.focus_fusion_dataset import FocusFusionDataset, FIXED_FUSION_PROMPT
from osediff_focus_fusion import FocusFusionGenerator, expand_unet_conv_in, load_focus_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--metadata_path", required=True); p.add_argument("--dataset_base_path", required=True)
    p.add_argument("--output_dir", required=True); p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--start_index", type=int, default=0); p.add_argument("--max_samples", type=int)
    p.add_argument("--condition_ablation", choices=["normal", "b_equals_a", "b_zero"], default="normal")
    p.add_argument("--run_all_ablations", action="store_true")
    p.add_argument("--vae_encode_mode", choices=["sample", "mode"], default="mode")
    p.add_argument("--prompt_mode", choices=["fixed", "metadata"], default="fixed")
    p.add_argument("--keep_a_composite", action="store_true"); p.add_argument("--keep_threshold", type=float, default=.5)
    p.add_argument("--keep_soft_width", type=float, default=.1); p.add_argument("--seed", type=int, default=123)
    p.add_argument("--mixed_precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--tiled", action="store_true"); p.add_argument("--latent_tiled_size", type=int, default=96)
    p.add_argument("--latent_tiled_overlap", type=int, default=32)
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


def main(args):
    from diffusers import DDPMScheduler
    from transformers import AutoTokenizer, CLIPTextModel
    from models.autoencoder_kl import AutoencoderKL
    from models.unet_2d_condition import UNet2DConditionModel
    state = torch.load(args.checkpoint_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.mixed_precision]
    if device.type == "cpu": dtype = torch.float32
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    expand_unet_conv_in(unet, state["generator_in_channels"])
    # Adapters must exist before loading LoRA. Recreate exact target names stored by PEFT keys.
    lora_state = state.get("generator_unet_lora", {})
    rank = int(state.get("rank_unet") or state.get("args", {}).get("lora_rank_unet", 4))
    if lora_state:
        from peft import LoraConfig
        targets = state.get("generator_unet_lora_targets") or sorted({k.split(".lora_", 1)[0] for k in lora_state if ".lora_" in k})
        unet.add_adapter(LoraConfig(r=rank, target_modules=targets), adapter_name="focus_fusion")
        unet.set_adapter(["focus_fusion"])
    vae_lora_state = state.get("vae_lora", {})
    if vae_lora_state:
        from peft import LoraConfig
        targets = state.get("vae_lora_targets")
        if not targets:
            raise RuntimeError("checkpoint contains VAE LoRA weights but no vae_lora_targets metadata")
        vae.add_adapter(LoraConfig(r=int(state.get("rank_vae") or rank), target_modules=targets), adapter_name="focus_vae_encoder")
        vae.set_adapter(["focus_vae_encoder"])
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler"); scheduler.set_timesteps(1, device=device)
    model = FocusFusionGenerator(unet, vae, scheduler, state["condition_mode"])
    load_focus_checkpoint(model, state)
    if vae_lora_state:
        result = model.vae.load_state_dict(vae_lora_state, strict=False)
        print("VAE missing keys:", result.missing_keys, "unexpected keys:", result.unexpected_keys)
    model.to(device, dtype=dtype).eval(); text.to(device, dtype=dtype).eval()
    dataset = FocusFusionDataset(args.metadata_path, args.dataset_base_path, args.resolution,
        max_samples=args.max_samples, start_index=args.start_index, prompt_mode=args.prompt_mode)

    def embeds(prompts):
        cached = model.fixed_prompt_embedding
        if args.prompt_mode == "fixed" and cached.numel(): return cached.to(device, dtype=dtype).expand(len(prompts), -1, -1)
        ids = tokenizer(prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
        return text(ids)[0]
    modes = ["normal", "b_equals_a", "b_zero"] if args.run_all_ablations else [args.condition_ablation]
    predictions = {m: [] for m in modes}
    for mode in modes:
        out_root = Path(args.output_dir, mode); out_root.mkdir(parents=True, exist_ok=True)
        for batch in DataLoader(dataset, batch_size=1, shuffle=False):
            a, b = batch["a"].to(device, dtype=dtype), batch["b_warp"].to(device, dtype=dtype)
            fa, fb = batch["focus_a"].to(device, dtype=dtype), batch["focus_b_warp"].to(device, dtype=dtype)
            if mode == "b_equals_a": b = a.clone()
            elif mode == "b_zero": b, fb = torch.zeros_like(b), torch.zeros_like(fb)
            with torch.no_grad(): raw, _, _ = model(a, b, fa, fb, embeds(batch["prompt"]), args.vae_encode_mode,
                args.tiled, args.latent_tiled_size, args.latent_tiled_overlap)
            keep_mask = soft_keep(fa, args.keep_threshold, args.keep_soft_width)
            final = keep_mask * a + (1 - keep_mask) * raw if args.keep_a_composite else raw
            idx = int(batch["metadata_index"].item()); folder = out_root / f"{idx:06d}"; folder.mkdir(exist_ok=True)
            images = {"pred_raw.png": _pil(raw), "pred_keepa_composite.png": _pil(keep_mask * a + (1 - keep_mask) * raw), "A.png": _pil(a),
                      "B_warp.png": _pil(b), "GT.png": _pil(batch["gt"]), "focus_A.png": _pil(fa, True),
                      "focus_B_warp.png": _pil(fb, True)}
            for name, image in images.items(): image.save(folder / name)
            panels = [images[n].convert("RGB") for n in ("A.png", "B_warp.png", "GT.png", "pred_raw.png", "pred_keepa_composite.png")]
            canvas = Image.new("RGB", (sum(x.width for x in panels), max(x.height for x in panels)))
            x = 0
            for panel in panels: canvas.paste(panel, (x, 0)); x += panel.width
            canvas.save(folder / "comparison_A_B_GT_raw_keepa.png")
            predictions[mode].append(raw.float().cpu())
    if len(modes) > 1:
        normal = predictions["normal"]
        stats = {f"normal_vs_{m}_mae": float(torch.cat([(x-y).abs().flatten() for x,y in zip(normal, predictions[m])]).mean()) for m in ("b_equals_a", "b_zero")}
        print(json.dumps(stats, indent=2))
        if min(stats.values()) < 1e-5: print("WARNING: ablation difference is near zero; the model may not use B")


if __name__ == "__main__": main(parse_args())
