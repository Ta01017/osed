"""Train the one-step OSEDiff focus-fusion generator."""
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from dataloaders.focus_fusion_dataset import FocusFusionDataset, FIXED_FUSION_PROMPT, focus_fusion_collate
from osediff_focus_fusion import (FocusFusionGenerator, checkpoint_payload, gradient_loss,
                                  laplacian_loss, masked_l1, move_scheduler_to_device,
                                  get_generator_in_channels, normalize_input_mode, vae_scale_factor)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--metadata_path", required=True)
    p.add_argument("--dataset_base_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--input_mode", choices=["single", "dual", "quad_rgb", "ab_focus", "ab", "four"], default="ab_focus")
    p.add_argument("--condition_mode", choices=["ab", "ab_focus", "dual", "single", "quad_rgb", "four"], default=None)
    p.add_argument("--prompt_mode", choices=["fixed", "metadata", "ram"], default="fixed")
    p.add_argument("--cache_fixed_prompt_embedding", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--max_samples", type=int)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--random_crop", action="store_true")
    p.add_argument("--center_crop", action="store_true")
    p.add_argument("--random_flip", action="store_true")
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--max_train_steps", type=int, default=10000)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--validation_steps", type=int, default=500)
    p.add_argument("--validation_max_samples", type=int, default=4)
    p.add_argument("--keep_a_composite", action="store_true")
    p.add_argument("--keep_threshold", type=float, default=.5)
    p.add_argument("--keep_soft_width", type=float, default=.1)
    p.add_argument("--native_resolution", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict_native_size", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_pixels", type=int, help="Maximum allowed native pixels. Images are rejected rather than resized.")
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--lr_scheduler", type=str, default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--lr_num_cycles", type=int, default=1)
    p.add_argument("--lr_power", type=float, default=1.0)
    p.add_argument("--lora_rank_unet", type=int, default=4)
    p.add_argument("--lora_rank", type=int, default=None)
    p.add_argument("--train_conv_in", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train_vae_lora", action="store_true")
    p.add_argument("--use_vsd", type=int, choices=[0, 1], default=0)
    p.add_argument("--lambda_l2", type=float, default=1.0)
    p.add_argument("--lambda_lpips", type=float, default=.1)
    p.add_argument("--lambda_vsd", type=float, default=0.0)
    p.add_argument("--lambda_vsd_lora", type=float, default=1.0)
    p.add_argument("--cfg_vsd", type=float, default=7.5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lambda_keep", type=float, default=.5)
    p.add_argument("--lambda_bref", type=float, default=1.0)
    p.add_argument("--lambda_gradient", type=float, default=.05)
    p.add_argument("--lambda_laplacian", type=float, default=.02)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--resume_from_checkpoint")
    p.add_argument("--ram_path")
    p.add_argument("--ram_ft_path")
    return p.parse_args(argv)


def _add_generator_lora(unet, rank):
    from peft import LoraConfig
    targets = sorted({n.rsplit(".", 1)[0] for n, p in unet.named_parameters()
                      if p.ndim >= 2 and "conv_in" not in n and any(x in n for x in ("to_q", "to_k", "to_v", "to_out.0"))})
    if not targets: raise RuntimeError("no UNet attention modules found for LoRA")
    unet.add_adapter(LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=targets), adapter_name="focus_fusion")
    return targets


def _add_vae_lora(vae, rank):
    from peft import LoraConfig
    patterns = ("conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0")
    targets = []
    for name, param in vae.named_parameters():
        if "bias" in name or "norm" in name or param.ndim < 2:
            continue
        if ("encoder" in name and any(p in name for p in patterns)) or ("quant_conv" in name and "post_quant_conv" not in name):
            targets.append(name.replace(".weight", ""))
    targets = sorted(set(targets))
    if not targets:
        raise RuntimeError("no VAE encoder modules found for LoRA")
    vae.add_adapter(LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=targets), adapter_name="focus_vae_encoder")
    vae.set_adapter(["focus_vae_encoder"])
    return targets


def _groups(model, args, vsd=None):
    groups = []
    lora = [p for n, p in model.unet.named_parameters() if "lora" in n and p.requires_grad]
    groups.append({"name": "generator_unet_lora", "params": lora})
    if args.train_conv_in:
        model.unet.conv_in.requires_grad_(True)
        groups.append({"name": "generator_conv_in", "params": list(model.unet.conv_in.parameters())})
    if args.train_vae_lora:
        groups.append({"name": "vae_lora", "params": [p for n, p in model.vae.named_parameters() if "lora" in n and p.requires_grad]})
    if vsd is not None:
        groups.append({"name": "vsd_lora", "params": [p for n, p in vsd.unet_update.named_parameters() if "lora" in n and p.requires_grad]})
    groups = [g for g in groups if g["params"]]
    ids = {id(p) for g in groups for p in g["params"]}
    if args.train_conv_in:
        assert id(model.unet.conv_in.weight) in ids, "expanded conv_in weight is absent from optimizer"
    else:
        model.unet.conv_in.requires_grad_(False)
        assert id(model.unet.conv_in.weight) not in ids, "conv_in was added while train_conv_in=false"
    total = 0
    for g in groups:
        count = sum(p.numel() for p in g["params"]); train = sum(p.numel() for p in g["params"] if p.requires_grad); total += train
        print(f"optimizer group {g['name']}: parameters={count:,}, requires_grad={train:,}")
    print(f"total trainable parameters: {total:,}")
    return groups


def _pil(x, focus=False):
    x = x.detach().float().cpu()[0]
    if not focus:
        x = x.mul(.5).add(.5)
    a = x.clamp(0, 1).mul(255).byte().numpy()
    if a.shape[0] == 1:
        return Image.fromarray(a[0], "L")
    return Image.fromarray(a.transpose(1, 2, 0), "RGB")


def _soft_keep(mask, threshold, width):
    if width <= 0:
        return (mask >= threshold).to(mask.dtype)
    return ((mask - (threshold - width / 2)) / width).clamp(0, 1)


@torch.no_grad()
def run_validation(model, loader, encode_fn, args, accelerator, step):
    if not accelerator.is_main_process:
        return
    raw_model = accelerator.unwrap_model(model)
    was_training = raw_model.training
    raw_model.eval()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    out_root = Path(args.output_dir, "validation", f"global_step_{step:06d}")
    out_root.mkdir(parents=True, exist_ok=True)
    for batch in loader:
        prompts = batch["prompt"]
        if args.prompt_mode == "fixed" and raw_model.fixed_prompt_embedding.numel():
            emb = raw_model.fixed_prompt_embedding.to(accelerator.device).expand(len(prompts), -1, -1)
        else:
            emb = encode_fn(prompts, accelerator.device)
        conditions = [x.to(accelerator.device) for x in batch["conditions"]]
        gt = batch["gt"].to(accelerator.device)
        focus_maps = [x.to(accelerator.device) for x in batch["focus_maps"]]
        fa = focus_maps[0] if focus_maps else None
        fb = focus_maps[1] if len(focus_maps) > 1 else None
        pred, _, _ = raw_model(conditions, fa, fb, emb, "mode")
        final = pred
        if args.keep_a_composite:
            if args.input_mode != "ab_focus":
                raise RuntimeError("--keep_a_composite is only valid for ab_focus")
            keep = _soft_keep(fa, args.keep_threshold, args.keep_soft_width)
            final = keep * conditions[0] + (1 - keep) * pred
        idx = int(batch["metadata_index"].item())
        folder = out_root / f"{idx:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        images = {"GT.png": _pil(gt), "pred_raw.png": _pil(pred), "final.png": _pil(final)}
        for i, cond in enumerate(conditions):
            images[f"{chr(ord('A') + i)}.png"] = _pil(cond)
        for i, fmap in enumerate(focus_maps):
            images[f"focus_{i}.png"] = _pil(fmap, True)
        if args.keep_a_composite:
            images["pred_keepa_composite.png"] = _pil(final)
        for name, image in images.items():
            image.save(folder / name)
        panel_names = [f"{chr(ord('A') + i)}.png" for i in range(len(conditions))] + ["GT.png"] + [f"focus_{i}.png" for i in range(len(focus_maps))] + ["pred_raw.png"]
        if args.keep_a_composite:
            panel_names.append("pred_keepa_composite.png")
        panels = [images[n].convert("RGB") for n in panel_names]
        canvas = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)))
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0)); x += panel.width
        canvas.save(folder / "comparison.png")
    if was_training:
        raw_model.train()


def main(args):
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import DDPMScheduler
    from diffusers.optimization import get_scheduler
    from transformers import AutoTokenizer, CLIPTextModel
    from models.autoencoder_kl import AutoencoderKL
    from models.unet_2d_condition import UNet2DConditionModel
    from osediff import OSEDiff_reg
    from osediff_focus_fusion import expand_unet_conv_in, load_focus_checkpoint

    if args.lora_rank is None:
        args.lora_rank = args.lora_rank_unet
    args.input_mode = normalize_input_mode(args.condition_mode or args.input_mode)
    if args.input_mode != "ab_focus" and (args.lambda_keep or args.lambda_bref):
        print(f"WARNING: keep-A and B-reference losses are disabled for input_mode={args.input_mode}; no fake masks will be created")
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)
    set_seed(args.seed); Path(args.output_dir, "checkpoints").mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder.to(accelerator.device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    # Required order: base UNet -> expand conv_in -> add adapters.
    expand_unet_conv_in(unet, get_generator_in_channels(args.input_mode))
    unet.requires_grad_(False); vae.requires_grad_(False); text_encoder.requires_grad_(False)
    lora_targets = _add_generator_lora(unet, args.lora_rank_unet)
    vae_lora_targets = _add_vae_lora(vae, args.lora_rank_unet) if args.train_vae_lora else []
    for n, p in unet.named_parameters(): p.requires_grad_("lora" in n)
    for n, p in vae.named_parameters(): p.requires_grad_("lora" in n)
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    scheduler.set_timesteps(1, device=accelerator.device)
    move_scheduler_to_device(scheduler, accelerator.device)
    model = FocusFusionGenerator(unet, vae, scheduler, args.input_mode)
    assert model.unet.config.out_channels == 4
    model.focus_lora_targets = lora_targets
    model.lora_rank_unet = args.lora_rank_unet
    model.focus_vae_lora_targets = vae_lora_targets
    model.lora_rank_vae = args.lora_rank_unet
    model_reg = OSEDiff_reg(args=args, accelerator=accelerator) if args.use_vsd else None
    if model_reg is not None:
        model_reg.set_train()
        assert model_reg.unet_fix.config.in_channels == 4
        assert model_reg.unet_update.config.in_channels == 4

    def encode(prompts, device):
        ids = tokenizer(prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
        with torch.no_grad(): return text_encoder(ids)[0]
    if args.prompt_mode == "fixed" and args.cache_fixed_prompt_embedding:
        model.cache_prompt(encode([FIXED_FUSION_PROMPT], accelerator.device).cpu())
    ram_infer, ram_model, ram_tf = None, None, None
    if args.prompt_mode == "ram":
        from ram.models.ram_lora import ram
        from ram import inference_ram as ram_infer
        from torchvision import transforms
        ram_tf = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ram_model = ram(pretrained=args.ram_path, pretrained_condition=args.ram_ft_path, image_size=384, vit="swin_l")
        ram_model.eval().to(accelerator.device, dtype=torch.float16)

    sf = vae_scale_factor(vae)
    dataset = FocusFusionDataset(args.metadata_path, args.dataset_base_path, args.resolution,
        args.random_crop, args.center_crop, args.random_flip, args.max_samples, args.start_index, args.smoke, args.prompt_mode,
        args.native_resolution, args.strict_native_size, args.input_mode, sf, args.max_pixels)
    val_dataset = FocusFusionDataset(args.metadata_path, args.dataset_base_path, args.resolution,
        False, False, False, args.validation_max_samples, 0, False, args.prompt_mode,
        args.native_resolution, args.strict_native_size, args.input_mode, sf, args.max_pixels)
    loader = DataLoader(dataset, args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, collate_fn=focus_fusion_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=focus_fusion_collate)
    param_groups = _groups(model, args, model_reg)
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)
    start = 0
    pending_lr_state = None
    if args.resume_from_checkpoint:
        state = load_focus_checkpoint(model, args.resume_from_checkpoint)
        if state.get("optimizer"):
            if len(state["optimizer"].get("param_groups", [])) != len(optimizer.param_groups):
                raise RuntimeError("optimizer parameter group count mismatch during resume")
            optimizer.load_state_dict(state["optimizer"])
        pending_lr_state = state.get("lr_scheduler")
        start = state.get("global_step", state.get("training_step", 0))
    if pending_lr_state:
        lr_scheduler.load_state_dict(pending_lr_state)
    if model_reg is None:
        model, text_encoder, optimizer, loader, val_loader, lr_scheduler = accelerator.prepare(model, text_encoder, optimizer, loader, val_loader, lr_scheduler)
    else:
        model, model_reg, text_encoder, optimizer, loader, val_loader, lr_scheduler = accelerator.prepare(model, model_reg, text_encoder, optimizer, loader, val_loader, lr_scheduler)
    iterator = iter(loader)
    for step in range(start + 1, args.max_train_steps + 1):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        with accelerator.accumulate(model):
            prompts = batch["prompt"]
            if args.prompt_mode == "ram":
                x = ram_tf(batch["gt"].mul(0.5).add(0.5)).to(accelerator.device, dtype=torch.float16)
                prompts = [str(x) for x in ram_infer(x, ram_model)]
            raw = accelerator.unwrap_model(model)
            if args.prompt_mode == "fixed" and raw.fixed_prompt_embedding.numel():
                emb = raw.fixed_prompt_embedding.to(accelerator.device).expand(len(prompts), -1, -1)
            else: emb = encode(prompts, accelerator.device)
            conditions = [x.to(accelerator.device) for x in batch["conditions"]]
            focus_maps = [x.to(accelerator.device) for x in batch["focus_maps"]]
            fa = focus_maps[0] if focus_maps else None
            fb = focus_maps[1] if len(focus_maps) > 1 else None
            pred, latent, _ = model(conditions, fa, fb, emb, "sample")
            gt, a = batch["gt"].to(accelerator.device), conditions[0]
            losses = {"l2": F.mse_loss(pred.float(), gt.float()) * args.lambda_l2,
                      "gradient": gradient_loss(pred.float(), gt.float()) * args.lambda_gradient,
                      "laplacian": laplacian_loss(pred.float(), gt.float()) * args.lambda_laplacian}
            if args.input_mode == "ab_focus":
                keep, bref = fa.clamp(0, 1), ((1 - fa) * fb).clamp(0, 1)
                losses["keep"] = masked_l1(pred.float(), a.float(), keep.float()) * args.lambda_keep
                losses["bref"] = masked_l1(pred.float(), gt.float(), bref.float()) * args.lambda_bref
            else:
                losses_disabled = ["keep", "bref"]
            if args.lambda_lpips:
                import lpips
                if not hasattr(main, "lpips_net"): main.lpips_net = lpips.LPIPS(net="vgg").to(accelerator.device).requires_grad_(False)
                losses["lpips"] = main.lpips_net(pred.float(), gt.float()).mean() * args.lambda_lpips
            neg_emb = None
            if model_reg is not None and args.lambda_vsd:
                neg_emb = encode([""] * len(prompts), accelerator.device)
                reg = model_reg.module if hasattr(model_reg, "module") else model_reg
                losses["vsd"] = reg.distribution_matching_loss(latent, emb, neg_emb, args) * args.lambda_vsd
            if model_reg is not None and args.lambda_vsd_lora:
                reg = model_reg.module if hasattr(model_reg, "module") else model_reg
                losses["vsd_lora"] = reg.diff_loss(latent, emb, args) * args.lambda_vsd_lora
            loss = sum(losses.values())
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], args.max_grad_norm)
            optimizer.step(); lr_scheduler.step(); optimizer.zero_grad(set_to_none=True)
        if accelerator.is_main_process and (step == 1 or step % 10 == 0): print(step, {k: round(v.item(), 6) for k, v in losses.items()})
        if args.validation_steps > 0 and step % args.validation_steps == 0:
            run_validation(model, val_loader, encode, args, accelerator, step)
        if accelerator.is_main_process and step % args.checkpointing_steps == 0:
            raw = accelerator.unwrap_model(model)
            raw_reg = accelerator.unwrap_model(model_reg) if model_reg is not None else None
            torch.save(checkpoint_payload(raw, step, args, optimizer, lr_scheduler, raw_reg, accelerator), Path(args.output_dir, "checkpoints", f"focus_fusion_{step}.pt"))


if __name__ == "__main__": main(parse_args())
