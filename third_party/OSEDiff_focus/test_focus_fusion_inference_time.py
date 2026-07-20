"""Benchmark focus-fusion inference.

Default mode runs the real VAE encode -> UNet -> scheduler.step -> VAE decode path.
Use --mock to run a clearly labelled shape-only Conv2d microbenchmark.
"""
import argparse
import time

import torch

from osediff_focus_fusion import (FocusFusionGenerator, expand_unet_conv_in, load_focus_checkpoint,
                                  move_scheduler_to_device, normalize_input_mode, read_focus_checkpoint_config,
                                  load_vae_lora_state)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true")
    p.add_argument("--warmup_iterations", type=int, default=5)
    p.add_argument("--inference_iterations", type=int, default=50)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--input_mode", choices=["single", "dual", "ab_focus", "quad_rgb", "ab", "four"])
    p.add_argument("--tiled", action="store_true")
    p.add_argument("--latent_tiled_size", type=int, default=96)
    p.add_argument("--latent_tiled_overlap", type=int, default=32)
    p.add_argument("--vae_encode_mode", choices=["sample", "mode"], default="mode")
    p.add_argument("--pretrained_model_name_or_path")
    p.add_argument("--checkpoint_path")
    p.add_argument("--mixed_precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    return p.parse_args()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _peak(device):
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def run_mock(args, device):
    channels = 10 if args.input_mode == "ab_focus" else 8
    conv = torch.nn.Conv2d(channels, 4, 3, padding=1).to(device)
    x = torch.randn(1, channels, args.height // 8, args.width // 8, device=device)
    for _ in range(args.warmup_iterations):
        conv(x)
    _sync(device)
    start = time.perf_counter()
    for _ in range(args.inference_iterations):
        conv(x)
    _sync(device)
    print(f"MOCK Conv2d-only time, not OSEDiff inference: {(time.perf_counter() - start) / args.inference_iterations:.6f}s")


def run_real(args, device):
    if not args.pretrained_model_name_or_path or not args.checkpoint_path:
        raise ValueError("real benchmark requires --pretrained_model_name_or_path and --checkpoint_path")
    from diffusers import DDPMScheduler
    from models.autoencoder_kl import AutoencoderKL
    from models.unet_2d_condition import UNet2DConditionModel

    ckpt = read_focus_checkpoint_config(args.checkpoint_path)
    state = ckpt["state"]
    ckpt_mode = ckpt["input_mode"]
    if args.input_mode is not None and normalize_input_mode(args.input_mode) != ckpt_mode:
        raise RuntimeError(f"input_mode mismatch: checkpoint={ckpt_mode}, requested={args.input_mode}")
    args.input_mode = ckpt_mode
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.mixed_precision]
    if device.type == "cpu":
        dtype = torch.float32
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    expand_unet_conv_in(unet, state["generator_in_channels"])
    if state.get("generator_unet_lora"):
        from peft import LoraConfig
        unet.add_adapter(LoraConfig(r=ckpt["generator_lora_rank"], target_modules=ckpt["generator_lora_targets"]),
                         adapter_name=ckpt["generator_lora_adapter_name"])
        unet.set_adapter([ckpt["generator_lora_adapter_name"]])
    if state.get("vae_lora"):
        from peft import LoraConfig
        vae.add_adapter(LoraConfig(r=ckpt["vae_lora_rank"], target_modules=ckpt["vae_lora_targets"]),
                        adapter_name=ckpt["vae_lora_adapter_name"])
        vae.set_adapter([ckpt["vae_lora_adapter_name"]])
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    scheduler.set_timesteps(1, device=device)
    move_scheduler_to_device(scheduler, device)
    model = FocusFusionGenerator(unet, vae, scheduler, args.input_mode)
    load_focus_checkpoint(model, state)
    load_vae_lora_state(model, state)
    model.to(device, dtype=dtype).eval()
    print(f"checkpoint: {args.checkpoint_path}")
    print(f"input_mode: {args.input_mode}, generator_in_channels: {state['generator_in_channels']}, generator_lora_rank: {ckpt['generator_lora_rank']}")
    print(f"train_vae_lora: {ckpt['train_vae_lora']}, use_vsd: {ckpt['use_vsd']}")
    print(f"native_hw: {args.height}x{args.width}, tiled: {args.tiled}, tile_size: {args.latent_tiled_size}, tile_overlap: {args.latent_tiled_overlap}")
    print(f"vae_encode_mode: {args.vae_encode_mode}, warmup: {args.warmup_iterations}, repeat: {args.inference_iterations}")
    prep_start = time.perf_counter()
    ncond = {"single": 1, "dual": 2, "ab_focus": 2, "quad_rgb": 4}[args.input_mode]
    conditions = [torch.randn(1, 3, args.height, args.width, device=device, dtype=dtype).clamp(-1, 1) for _ in range(ncond)]
    focus_maps = []
    if args.input_mode == "ab_focus":
        focus_maps = [torch.rand(1, 1, args.height, args.width, device=device, dtype=dtype) for _ in range(2)]
    _sync(device)
    preprocess_time = time.perf_counter() - prep_start
    prompt = model.fixed_prompt_embedding
    if not prompt.numel():
        prompt = torch.zeros(1, 77, 1024)
    prompt = prompt.to(device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    phase = {"image_load_preprocess": 0.0, "vae_encode": 0.0, "unet_one_step": 0.0, "scheduler_step": 0.0, "vae_decode": 0.0, "end_to_end": 0.0}
    for i in range(args.warmup_iterations + args.inference_iterations):
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            s = time.perf_counter(); latents = model.encode_images(*conditions, mode=args.vae_encode_mode); _sync(device); e = time.perf_counter()
            z_a = latents[0]
            fa = focus_maps[0] if focus_maps else None
            fb = focus_maps[1] if len(focus_maps) > 1 else None
            from osediff_focus_fusion import build_generator_unet_input
            unet_input = build_generator_unet_input(args.input_mode, latents, fa, fb)
            ts = model.timesteps.to(device)
            s2 = time.perf_counter()
            if args.tiled:
                from osediff_focus_fusion import tiled_unet_forward
                pred = tiled_unet_forward(model.unet, unet_input, ts, prompt, 4, args.latent_tiled_size, args.latent_tiled_overlap)
            else:
                pred = model.unet(unet_input, ts, encoder_hidden_states=prompt).sample
            _sync(device); e2 = time.perf_counter()
            s3 = time.perf_counter(); den = model.scheduler.step(pred, ts, z_a, return_dict=True).prev_sample; _sync(device); e3 = time.perf_counter()
            s4 = time.perf_counter(); model.vae.decode(den / model.vae.config.scaling_factor).sample; _sync(device); e4 = time.perf_counter()
        t1 = time.perf_counter()
        if i >= args.warmup_iterations:
            phase["vae_encode"] += e - s
            phase["unet_one_step"] += e2 - s2
            phase["scheduler_step"] += e3 - s3
            phase["vae_decode"] += e4 - s4
            phase["end_to_end"] += t1 - t0
            phase["image_load_preprocess"] += preprocess_time
    for key, value in phase.items():
        print(f"{key}: {value / args.inference_iterations:.6f}s")
    print(f"peak_memory_mb: {_peak(device):.2f}")
    print(f"resolution: {args.height}x{args.width}, input_mode: {args.input_mode}, tiled: {args.tiled}")


if __name__ == "__main__":
    args = parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mock:
        run_mock(args, dev)
    else:
        run_real(args, dev)
