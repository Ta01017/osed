"""One-step, dual-image OSEDiff model for multi-focus image fusion."""
import copy
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloaders.focus_fusion_dataset import FIXED_FUSION_PROMPT


def expand_unet_conv_in(unet, new_in_channels):
    """Expand only ``conv_in``; preserve channel 0:4 and zero new channels."""
    old = unet.conv_in
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
    if mode not in ("ab", "ab_focus"):
        raise ValueError(f"unknown condition_mode: {mode}")
    return 8 if mode == "ab" else 10


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
    """Tile full 8/10-channel input while accumulating exactly 4 output channels."""
    _, _, h, w = unet_input.shape
    if h <= tile_size and w <= tile_size:
        return unet(unet_input, timestep, encoder_hidden_states=prompt_embeds).sample
    stride = max(1, tile_size - overlap)
    ys = list(range(0, max(h - tile_size, 0) + 1, stride)); xs = list(range(0, max(w - tile_size, 0) + 1, stride))
    if not ys or ys[-1] != h - tile_size: ys.append(max(0, h - tile_size))
    if not xs or xs[-1] != w - tile_size: xs.append(max(0, w - tile_size))
    out = unet_input.new_zeros((unet_input.shape[0], output_channels, h, w))
    weights = unet_input.new_zeros((unet_input.shape[0], 1, h, w))
    for y in ys:
        for x in xs:
            tile = unet_input[..., y:y + tile_size, x:x + tile_size]
            pred = unet(tile, timestep, encoder_hidden_states=prompt_embeds).sample
            out[..., y:y + tile_size, x:x + tile_size] += pred
            weights[..., y:y + tile_size, x:x + tile_size] += 1
    return out / weights.clamp_min(1)


class FocusFusionGenerator(nn.Module):
    def __init__(self, unet, vae, scheduler, condition_mode="ab_focus", timestep=999):
        super().__init__()
        self.unet, self.vae, self.scheduler = unet, vae, scheduler
        self.condition_mode = condition_mode
        wanted = condition_channels(condition_mode)
        if unet.conv_in.in_channels != wanted:
            expand_unet_conv_in(unet, wanted)
        assert self.unet.config.in_channels in (8, 10)
        assert self.unet.config.out_channels == 4
        self.register_buffer("timesteps", torch.tensor([timestep], dtype=torch.long), persistent=False)
        self.register_buffer("fixed_prompt_embedding", torch.empty(0), persistent=True)

    def cache_prompt(self, embedding):
        self.fixed_prompt_embedding = embedding.detach().clone()

    def encode_images(self, a, b, mode="sample"):
        posterior = self.vae.encode(torch.cat([a, b], 0)).latent_dist
        z = posterior.sample() if mode == "sample" else posterior.mode()
        z = z * self.vae.config.scaling_factor
        return z.chunk(2, 0)

    def make_unet_input(self, z_a, z_b, focus_a=None, focus_b=None):
        parts = [z_a, z_b]
        if self.condition_mode == "ab_focus":
            if focus_a is None or focus_b is None: raise ValueError("ab_focus requires both focus maps")
            size = z_a.shape[-2:]
            parts += [F.interpolate(focus_a, size=size, mode="bilinear", align_corners=False),
                      F.interpolate(focus_b, size=size, mode="bilinear", align_corners=False)]
        value = torch.cat(parts, 1)
        assert value.shape[1] == condition_channels(self.condition_mode)
        return value

    def forward(self, a, b, focus_a, focus_b, prompt_embeds, vae_encode_mode="sample",
                tiled=False, tile_size=96, tile_overlap=32, z_b_override=None):
        z_a, z_b = self.encode_images(a, b, vae_encode_mode)
        if z_b_override is not None:
            z_b = z_b_override.to(device=z_a.device, dtype=z_a.dtype)
        unet_input = self.make_unet_input(z_a, z_b, focus_a, focus_b)
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
            output = F.interpolate(output, size=a.shape[-2:], mode="bilinear", align_corners=False).clamp(-1, 1)
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
