"""One-step OSEDiff generator for native-resolution focus fusion."""
import copy
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloaders.focus_fusion_dataset import FIXED_FUSION_PROMPT

INPUT_MODE_TO_CHANNELS = {
    "single": 4,
    "dual": 8,
    "ab_focus": 10,
    "quad_rgb": 16,
}

INPUT_MODE_ALIASES = {
    "ab": "dual",
    "four": "quad_rgb",
}


def normalize_input_mode(input_mode):
    mode = INPUT_MODE_ALIASES.get(str(input_mode), str(input_mode))
    if mode not in INPUT_MODE_TO_CHANNELS:
        raise ValueError(f"unknown input_mode: {input_mode}")
    return mode


def get_generator_in_channels(input_mode):
    return INPUT_MODE_TO_CHANNELS[normalize_input_mode(input_mode)]


def expand_unet_conv_in(unet, new_in_channels):
    """Expand only ``conv_in``; preserve channel 0:4 and zero new channels."""
    old = unet.conv_in
    if new_in_channels == old.in_channels == 4:
        if hasattr(unet, "register_to_config"):
            unet.register_to_config(in_channels=4)
        return unet
    if new_in_channels < old.in_channels:
        raise ValueError(f"cannot shrink conv_in from {old.in_channels} to {new_in_channels}")
    new = nn.Conv2d(new_in_channels, old.out_channels, old.kernel_size, old.stride,
                    old.padding, old.dilation, old.groups, old.bias is not None,
                    old.padding_mode, device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_channels].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    unet.conv_in = new
    if hasattr(unet, "register_to_config"):
        unet.register_to_config(in_channels=int(new_in_channels))
    elif hasattr(unet, "config"):
        try:
            unet.config.in_channels = int(new_in_channels)
        except Exception:
            cfg = dict(unet.config)
            cfg["in_channels"] = int(new_in_channels)
            unet.config = SimpleNamespace(**cfg)
    return unet


def condition_channels(mode):
    return get_generator_in_channels(mode)


def move_scheduler_to_device(scheduler, device):
    if hasattr(scheduler, "alphas_cumprod"):
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    return scheduler


def masked_l1(pred, target, mask, eps=1e-6):
    mask = mask.clamp(0, 1)
    return ((pred - target).abs() * mask).sum() / (mask.sum() * pred.shape[1] + eps)


def image_gradients(x):
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]


def gradient_loss(pred, target):
    px, py = image_gradients(pred); tx, ty = image_gradients(target)
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


def laplacian_loss(pred, target):
    kernel = pred.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    kernel = kernel.repeat(pred.shape[1], 1, 1, 1)
    return F.l1_loss(F.conv2d(pred, kernel, padding=1, groups=pred.shape[1]),
                     F.conv2d(target, kernel, padding=1, groups=target.shape[1]))


def tiled_unet_forward(unet, unet_input, timestep, prompt_embeds, output_channels=4,
                       tile_size=96, overlap=32):
    """Tile full condition input with Gaussian blending and 4-channel output."""
    if not (0 <= overlap < tile_size):
        raise ValueError(f"tile overlap must satisfy 0 <= overlap < tile_size, got {overlap}, {tile_size}")
    _, _, h, w = unet_input.shape
    if h <= tile_size and w <= tile_size:
        return unet(unet_input, timestep, encoder_hidden_states=prompt_embeds).sample
    stride = max(1, tile_size - overlap)
    ys = list(range(0, h, stride)); xs = list(range(0, w, stride))
    out = unet_input.new_zeros((unet_input.shape[0], output_channels, h, w))
    weights = unet_input.new_zeros((unet_input.shape[0], 1, h, w))

    def gaussian(th, tw):
        yy = torch.linspace(-1, 1, th, device=unet_input.device, dtype=unet_input.dtype)
        xx = torch.linspace(-1, 1, tw, device=unet_input.device, dtype=unet_input.dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        weight = torch.exp(-(gx.square() + gy.square()) / 0.5).clamp_min(1e-3)
        return weight.view(1, 1, th, tw)

    for y in ys:
        for x in xs:
            y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
            tile = unet_input[..., y:y2, x:x2]
            pred = unet(tile, timestep, encoder_hidden_states=prompt_embeds).sample
            weight = gaussian(y2 - y, x2 - x)
            out[..., y:y2, x:x2] += pred * weight
            weights[..., y:y2, x:x2] += weight
    return out / weights.clamp_min(1)


def encode_rgb_conditions(images, vae, encode_mode="sample"):
    if not images:
        raise ValueError("at least one RGB condition image is required")
    latent_dist = vae.encode(torch.cat(images, dim=0)).latent_dist
    latents = latent_dist.sample() if encode_mode == "sample" else latent_dist.mode()
    latents = latents * vae.config.scaling_factor
    return list(latents.chunk(len(images), dim=0))


def build_generator_unet_input(input_mode, latents, focus_a=None, focus_b=None):
    mode = normalize_input_mode(input_mode)
    needed = {"single": 1, "dual": 2, "ab_focus": 2, "quad_rgb": 4}[mode]
    if len(latents) != needed:
        raise ValueError(f"{mode} requires {needed} RGB latents, got {len(latents)}")
    parts = list(latents)
    if mode == "ab_focus":
        if focus_a is None or focus_b is None:
            raise ValueError("ab_focus requires both focus maps")
        size = latents[0].shape[-2:]
        parts += [F.interpolate(focus_a, size=size, mode="bilinear", align_corners=False),
                  F.interpolate(focus_b, size=size, mode="bilinear", align_corners=False)]
    value = torch.cat(parts, 1)
    assert value.shape[1] == get_generator_in_channels(mode)
    return value


def vae_scale_factor(vae):
    return 2 ** (len(vae.config.block_out_channels) - 1)


class FocusFusionGenerator(nn.Module):
    def __init__(self, unet, vae, scheduler, input_mode="ab_focus", timestep=999):
        super().__init__()
        self.unet, self.vae, self.scheduler = unet, vae, scheduler
        self.input_mode = normalize_input_mode(input_mode)
        self.condition_mode = self.input_mode
        wanted = get_generator_in_channels(self.input_mode)
        if unet.conv_in.in_channels != wanted:
            expand_unet_conv_in(unet, wanted)
        assert self.unet.config.in_channels == wanted
        assert self.unet.config.out_channels == 4
        self.register_buffer("timesteps", torch.tensor([timestep], dtype=torch.long), persistent=False)
        self.register_buffer("fixed_prompt_embedding", torch.empty(0), persistent=True)

    def cache_prompt(self, embedding):
        self.fixed_prompt_embedding = embedding.detach().clone()

    def encode_images(self, *images, mode="sample"):
        return encode_rgb_conditions(list(images), self.vae, mode)

    def make_unet_input(self, z_a, z_b, focus_a=None, focus_b=None):
        return build_generator_unet_input(self.input_mode, [z_a, z_b], focus_a, focus_b)

    def forward(self, conditions, focus_a=None, focus_b=None, prompt_embeds=None, vae_encode_mode="sample",
                tiled=False, tile_size=96, tile_overlap=32, latent_overrides=None):
        if isinstance(conditions, torch.Tensor):
            conditions = [conditions]
        a = conditions[0]
        latents = self.encode_images(*conditions, mode=vae_encode_mode)
        if latent_overrides:
            for idx, value in latent_overrides.items():
                latents[idx] = value.to(device=latents[0].device, dtype=latents[0].dtype)
        z_a = latents[0]
        unet_input = build_generator_unet_input(self.input_mode, latents, focus_a, focus_b)
        ts = self.timesteps.to(a.device)
        if hasattr(self.scheduler, "alphas_cumprod") and self.scheduler.alphas_cumprod.device != z_a.device:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(z_a.device)
        if tiled:
            pred = tiled_unet_forward(self.unet, unet_input, ts, prompt_embeds, 4, tile_size, tile_overlap)
        else:
            pred = self.unet(unet_input, ts, encoder_hidden_states=prompt_embeds).sample
        assert pred.shape[1] == z_a.shape[1] == 4
        denoised = self.scheduler.step(pred, ts, z_a, return_dict=True).prev_sample
        assert denoised.shape[1] == 4
        output = self.vae.decode(denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)
        if output.shape[-2:] != a.shape[-2:]:
            raise RuntimeError(
                f"decoded output size mismatch: A={tuple(a.shape[-2:])}, output={tuple(output.shape[-2:])}, "
                f"latent={tuple(z_a.shape[-2:])}, vae_scale_factor={vae_scale_factor(self.vae)}, input_mode={self.input_mode}"
            )
        assert output.shape[1] == 3
        return output, denoised, pred


def checkpoint_payload(model, step, args, optimizer=None, lr_scheduler=None, vsd=None, accelerator=None):
    unet_state = {k: v.detach().cpu() for k, v in model.unet.state_dict().items() if "lora" in k}
    input_mode = getattr(args, "input_mode", getattr(args, "condition_mode", model.condition_mode))
    return {"format_version": 2, "input_mode": input_mode, "condition_mode": model.condition_mode,
            "generator_in_channels": model.unet.conv_in.in_channels,
            "generator_conv_in": copy.deepcopy(model.unet.conv_in.state_dict()),
            "generator_unet_lora": unet_state,
            "generator_unet_lora_targets": getattr(model, "focus_lora_targets", []),
            "rank_unet": getattr(model, "lora_rank_unet", getattr(args, "lora_rank_unet", None)),
            "vae_lora_targets": getattr(model, "focus_vae_lora_targets", []),
            "rank_vae": getattr(model, "lora_rank_vae", getattr(args, "lora_rank_unet", None)),
            "vae_lora": {k: v.detach().cpu() for k, v in model.vae.state_dict().items() if "lora" in k},
            "vsd_unet_lora": {k: v.detach().cpu() for k, v in vsd.unet_update.state_dict().items() if "lora" in k} if vsd is not None else {},
            "vsd_lora_targets": getattr(vsd, "lora_unet_modules_encoder", []) + getattr(vsd, "lora_unet_modules_decoder", []) + getattr(vsd, "lora_unet_others", []) if vsd is not None else [],
            "fixed_prompt": FIXED_FUSION_PROMPT,
            "cache_fixed_prompt_embedding": bool(getattr(args, "cache_fixed_prompt_embedding", True)),
            "fixed_prompt_embedding": model.fixed_prompt_embedding.cpu(),
            "training_step": int(step), "global_step": int(step),
            "gradient_accumulation_steps": int(getattr(args, "gradient_accumulation_steps", 1)),
            "args": vars(args).copy(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None}


def load_focus_checkpoint(model, checkpoint, load_lora=True):
    state = torch.load(checkpoint, map_location="cpu") if isinstance(checkpoint, (str, bytes)) else checkpoint
    if state.get("input_mode", state.get("condition_mode")) != model.condition_mode:
        raise RuntimeError(f"input_mode mismatch: checkpoint={state.get('input_mode', state.get('condition_mode'))} model={model.condition_mode}")
    if model.unet.conv_in.in_channels != state["generator_in_channels"]:
        raise RuntimeError("model conv_in must be expanded before checkpoint loading")
    result = model.unet.conv_in.load_state_dict(state["generator_conv_in"], strict=True)
    print("conv_in missing keys:", result.missing_keys, "unexpected keys:", result.unexpected_keys)
    if result.missing_keys or result.unexpected_keys: raise RuntimeError("conv_in checkpoint load failed")
    if "weight" not in state["generator_conv_in"] or ("bias" in model.unet.conv_in.state_dict() and "bias" not in state["generator_conv_in"]):
        raise RuntimeError("checkpoint did not provide required conv_in weight/bias")
    if load_lora:
        lora_state = state.get("generator_unet_lora", {})
        result = model.unet.load_state_dict(lora_state, strict=False)
        print("UNet missing keys:", result.missing_keys, "unexpected keys:", result.unexpected_keys)
        bad_unexpected = [k for k in result.unexpected_keys if "lora" in k]
        loaded = set(lora_state)
        missing_lora = [k for k in loaded if k not in model.unet.state_dict()]
        if bad_unexpected or missing_lora:
            raise RuntimeError(f"LoRA checkpoint load failed: unexpected={bad_unexpected} missing_lora={missing_lora}")
    if state.get("fixed_prompt_embedding") is not None:
        model.cache_prompt(state["fixed_prompt_embedding"])
    return state
