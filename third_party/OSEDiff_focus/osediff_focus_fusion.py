"""One-step OSEDiff generator for native-resolution focus fusion."""
import copy
import hashlib
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
    if tile_size <= 0 or not (0 <= overlap < tile_size):
        raise ValueError(f"tile overlap must satisfy 0 <= overlap < tile_size, got {overlap}, {tile_size}")
    _, _, h, w = unet_input.shape
    if h <= tile_size and w <= tile_size:
        return unet(unet_input, timestep, encoder_hidden_states=prompt_embeds).sample
    stride = max(1, tile_size - overlap)
    ys = list(range(0, h, stride)); xs = list(range(0, w, stride))
    original_dtype = unet_input.dtype
    out = torch.zeros((unet_input.shape[0], output_channels, h, w), dtype=torch.float32, device=unet_input.device)
    weights = torch.zeros((unet_input.shape[0], 1, h, w), dtype=torch.float32, device=unet_input.device)

    def gaussian(th, tw):
        yy = torch.linspace(-1, 1, th, device=unet_input.device, dtype=torch.float32)
        xx = torch.linspace(-1, 1, tw, device=unet_input.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        weight = torch.exp(-(gx.square() + gy.square()) / 0.5).clamp_min(1e-3)
        return weight.view(1, 1, th, tw)

    for y in ys:
        for x in xs:
            y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
            tile = unet_input[..., y:y2, x:x2]
            pred = unet(tile, timestep, encoder_hidden_states=prompt_embeds).sample
            weight = gaussian(y2 - y, x2 - x)
            out[..., y:y2, x:x2] += pred.float() * weight
            weights[..., y:y2, x:x2] += weight
    return (out / weights.clamp_min(1e-6)).to(original_dtype)


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
        raise ValueError(
            f"make_unet_input input_mode={mode}: expected latent count={needed}, "
            f"actual latent count={len(latents)}, focus_a_present={focus_a is not None}, "
            f"focus_b_present={focus_b is not None}"
        )
    parts = list(latents)
    if mode == "ab_focus":
        if focus_a is None or focus_b is None:
            raise ValueError(
                f"make_unet_input input_mode={mode}: expected latent count={needed}, "
                f"actual latent count={len(latents)}, focus_a_present={focus_a is not None}, "
                f"focus_b_present={focus_b is not None}"
            )
        size = latents[0].shape[-2:]
        parts += [F.interpolate(focus_a, size=size, mode="bilinear", align_corners=False),
                  F.interpolate(focus_b, size=size, mode="bilinear", align_corners=False)]
    value = torch.cat(parts, 1)
    assert value.shape[1] == get_generator_in_channels(mode)
    return value


def vae_scale_factor(vae):
    return 2 ** (len(vae.config.block_out_channels) - 1)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    if not state:
        return False
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    return True


def checkpoint_complete_path(checkpoint_path):
    return os.fspath(checkpoint_path) + ".complete.json"


def compute_file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(path):
    return {"filename": Path(path).name, "size": os.path.getsize(path), "sha256": compute_file_sha256(path)}


def write_json_atomically(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def prepare_checkpoint_temp_dir(final_dir):
    import uuid

    final_dir = Path(final_dir)
    if final_dir.exists():
        raise FileExistsError(f"checkpoint already exists and will not be overwritten: {final_dir}")
    temp_dir = final_dir.parent / f".{final_dir.name}.tmp-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True)
    return temp_dir


def finalize_checkpoint_directory(temp_dir, final_dir, model_state, trainer_state, optimizer_manifest):
    temp_dir = Path(temp_dir)
    final_dir = Path(final_dir)
    if final_dir.exists():
        raise FileExistsError(f"checkpoint already exists and will not be overwritten: {final_dir}")
    manifest = {
        "complete": True,
        "checkpoint_version": 2,
        "global_step": int(trainer_state["global_step"]),
        "input_mode": model_state["input_mode"],
        "generator_in_channels": int(model_state["generator_in_channels"]),
        "model_state": _file_entry(temp_dir / "model_state.pt"),
        "trainer_state": _file_entry(temp_dir / "trainer_state.json"),
        "optimizer_manifest": _file_entry(temp_dir / "optimizer_manifest.json"),
        "accelerator_state_present": (temp_dir / "accelerator_state").is_dir(),
    }
    if not manifest["accelerator_state_present"] or not any((temp_dir / "accelerator_state").iterdir()):
        raise RuntimeError(f"accelerator_state directory is missing or empty: {temp_dir / 'accelerator_state'}")
    write_json_atomically(temp_dir / "checkpoint_complete.json", manifest)
    os.replace(temp_dir, final_dir)
    return manifest


def write_checkpoint_atomically(checkpoint_dir, model_state, trainer_state, optimizer_manifest, accelerator_state_present=False):
    """Compatibility helper for tests: writes non-Accelerate files into a fresh temp dir."""
    temp_dir = prepare_checkpoint_temp_dir(checkpoint_dir)
    if accelerator_state_present:
        (temp_dir / "accelerator_state").mkdir()
        write_json_atomically(temp_dir / "accelerator_state" / "state.json", {"present": True})
    torch.save(model_state, temp_dir / "model_state.pt")
    write_json_atomically(temp_dir / "trainer_state.json", trainer_state)
    write_json_atomically(temp_dir / "optimizer_manifest.json", optimizer_manifest)
    return finalize_checkpoint_directory(temp_dir, checkpoint_dir, model_state, trainer_state, optimizer_manifest)


def load_verified_checkpoint(checkpoint_path):
    checkpoint_dir = Path(checkpoint_path)
    if checkpoint_dir.is_file():
        raise RuntimeError(f"legacy .pt checkpoint path is not allowed for formal entrypoints: {checkpoint_dir}")
    manifest_path = checkpoint_dir / "checkpoint_complete.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"checkpoint_complete.json missing: {checkpoint_dir}")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("complete") is not True or int(manifest.get("checkpoint_version", 0)) != 2:
        raise RuntimeError(f"invalid checkpoint manifest: {manifest_path}")
    if manifest.get("accelerator_state_present") and not (checkpoint_dir / "accelerator_state").is_dir():
        raise RuntimeError(f"accelerator_state directory missing: {checkpoint_dir / 'accelerator_state'}")
    if manifest.get("accelerator_state_present") and not any((checkpoint_dir / "accelerator_state").iterdir()):
        raise RuntimeError(f"accelerator_state directory is empty: {checkpoint_dir / 'accelerator_state'}")
    verified = {"manifest": manifest, "checkpoint_dir": str(checkpoint_dir)}
    for key in ("model_state", "trainer_state", "optimizer_manifest"):
        entry = manifest[key]
        path = checkpoint_dir / entry["filename"]
        if not path.is_file():
            raise RuntimeError(f"checkpoint file missing: {path}")
        if os.path.getsize(path) != int(entry["size"]):
            raise RuntimeError(f"checkpoint file size mismatch: {path}")
        actual = compute_file_sha256(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"checkpoint sha256 mismatch: {path}")
        if key == "model_state":
            verified[key] = torch.load(path, map_location="cpu")
        else:
            with open(path, "r", encoding="utf-8") as handle:
                verified[key] = json.load(handle)
    if int(manifest["global_step"]) != int(verified["trainer_state"].get("global_step", -1)):
        raise RuntimeError(
            f"checkpoint global_step mismatch: manifest={manifest['global_step']} "
            f"trainer_state={verified['trainer_state'].get('global_step')}"
        )
    required_progress = (
        "global_step", "current_epoch", "completed_epochs", "batches_consumed_in_current_epoch",
        "micro_batches", "optimizer_updates", "scheduler_steps", "sampler_epoch",
        "gradient_accumulation_steps",
    )
    for field in required_progress:
        if field not in verified["trainer_state"]:
            raise RuntimeError(f"[INVALID TRAINER STATE] missing required progress field: {field}")
    trainer_state = verified["trainer_state"]
    global_step = int(trainer_state["global_step"])
    if int(trainer_state["optimizer_updates"]) != global_step:
        raise RuntimeError(
            f"[INVALID TRAINER STATE] optimizer_updates={trainer_state['optimizer_updates']} global_step={global_step}"
        )
    if int(trainer_state["scheduler_steps"]) != global_step:
        raise RuntimeError(
            f"[INVALID TRAINER STATE] scheduler_steps={trainer_state['scheduler_steps']} global_step={global_step}"
        )
    expected_min_micro_batches = global_step * int(trainer_state["gradient_accumulation_steps"])
    if int(trainer_state["micro_batches"]) < expected_min_micro_batches:
        raise RuntimeError(
            "[INVALID TRAINER STATE] "
            f"micro_batches={trainer_state['micro_batches']} expected_at_least={expected_min_micro_batches}"
        )
    forbidden_progress = {
        "global_step", "current_epoch", "completed_epochs", "micro_batches",
        "optimizer_updates", "scheduler_steps", "batches_consumed_in_current_epoch",
        "sampler_epoch", "sampler_seed", "optimizer", "lr_scheduler", "scaler_state", "rng_state",
    }
    present = sorted(forbidden_progress.intersection(verified["model_state"]))
    if present:
        raise RuntimeError(
            "[INVALID MODEL STATE] training progress fields must be stored in trainer_state.json; "
            f"found={present}"
        )
    print(f"checkpoint verified: {checkpoint_dir}")
    return verified


def resolve_prompt_embedding(*, checkpoint_state, checkpoint_config, pretrained_model_name_or_path,
                             prompt_text=None, batch_size=1, device=None, dtype=None):
    cached = checkpoint_state.get("fixed_prompt_embedding")
    if cached is not None and hasattr(cached, "numel") and cached.numel():
        emb = cached.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        return {"embedding": emb, "source": "checkpoint_cache", "tokenizer": None, "text_encoder": None}
    text = prompt_text or checkpoint_state.get("fixed_prompt") or checkpoint_config.get("fixed_prompt")
    if not text:
        raise RuntimeError("fixed prompt embedding is absent and fixed prompt text is missing")
    from transformers import AutoTokenizer, CLIPTextModel
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder.to(device=device, dtype=dtype).eval()
    ids = tokenizer([text] * batch_size, max_length=tokenizer.model_max_length, padding="max_length",
                    truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        emb = text_encoder(ids)[0]
    return {"embedding": emb, "source": "text_encoder", "tokenizer": tokenizer, "text_encoder": text_encoder}


def write_checkpoint_complete_manifest(checkpoint_path, payload):
    import json

    manifest = {
        "complete": True,
        "format_version": int(payload.get("format_version", 0)),
        "global_step": int(payload.get("global_step", payload.get("training_step", 0))),
        "input_mode": payload.get("input_mode"),
        "generator_in_channels": int(payload.get("generator_in_channels", 0)),
    }
    with open(checkpoint_complete_path(checkpoint_path), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def _load_lora_state(module, state, label):
    module_state = module.state_dict()
    missing = [k for k in state if k not in module_state]
    unexpected = [k for k in module_state if "lora" in k and k not in state]
    print(f"{label} LoRA missing keys:", missing, "unexpected keys:", unexpected)
    if missing or unexpected:
        raise RuntimeError(f"{label} LoRA checkpoint load failed")
    with torch.no_grad():
        for key, value in state.items():
            module_state[key].copy_(value.to(device=module_state[key].device, dtype=module_state[key].dtype))


def load_generator_lora_state(model, state):
    load_state = state.get("generator_unet_lora", {})
    if load_state:
        _load_lora_state(model.unet, load_state, "Generator UNet")


def load_vae_lora_state(model, state):
    load_state = state.get("vae_lora", {})
    if state.get("train_vae_lora") and not load_state:
        raise RuntimeError("checkpoint train_vae_lora=true but VAE LoRA state is missing")
    if load_state:
        _load_lora_state(model.vae, load_state, "VAE")


def load_vsd_lora_state(vsd, state):
    load_state = state.get("vsd_unet_lora", {})
    if state.get("use_vsd") and not load_state:
        raise RuntimeError("checkpoint use_vsd=true but VSD LoRA state is missing")
    if load_state:
        if vsd is None:
            raise RuntimeError("checkpoint contains VSD LoRA but VSD module was not created")
        _load_lora_state(vsd.unet_update, load_state, "VSD update UNet")


def read_focus_checkpoint_config(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu") if isinstance(checkpoint_path, (str, bytes)) else checkpoint_path
    args = state.get("args", {})
    input_mode = normalize_input_mode(state.get("input_mode", state.get("condition_mode", args.get("input_mode", "ab_focus"))))
    return {
        "state": state,
        "input_mode": input_mode,
        "generator_in_channels": int(state["generator_in_channels"]),
        "generator_lora_rank": int(state.get("rank_unet") or args.get("lora_rank_unet", 8)),
        "generator_lora_adapter_name": state.get("generator_lora_adapter_name", "focus_fusion"),
        "generator_lora_targets": state.get("generator_unet_lora_targets", []),
        "train_conv_in": bool(state.get("train_conv_in", args.get("train_conv_in", True))),
        "train_vae_lora": bool(state.get("train_vae_lora", args.get("train_vae_lora", bool(state.get("vae_lora"))))),
        "vae_lora_rank": int(state.get("rank_vae") or args.get("lora_rank_vae", args.get("lora_rank_unet", 4))),
        "vae_lora_adapter_name": state.get("vae_lora_adapter_name", "focus_vae_encoder"),
        "vae_lora_targets": state.get("vae_lora_targets", []),
        "use_vsd": bool(state.get("use_vsd", args.get("use_vsd", bool(state.get("vsd_unet_lora"))))),
        "vsd_lora_rank": int(state.get("rank_vsd") or args.get("lora_rank_vsd", args.get("lora_rank", 8))),
        "vsd_lora_adapter_name": state.get("vsd_lora_adapter_name", "default_others"),
        "vsd_lora_targets": state.get("vsd_lora_targets", []),
        "prompt_mode": state.get("prompt_mode", args.get("prompt_mode", "fixed")),
        "fixed_prompt": state.get("fixed_prompt", FIXED_FUSION_PROMPT),
        "global_step": int(state.get("global_step", state.get("training_step", 0))),
    }


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

    def make_unet_input(self, latents, focus_a=None, focus_b=None):
        return build_generator_unet_input(self.input_mode, list(latents), focus_a, focus_b)

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
        unet_input = self.make_unet_input(latents, focus_a, focus_b)
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

    def forward_from_latents(self, latents, focus_a=None, focus_b=None, prompt_embeds=None, *,
                             expected_output_hw, tiled=False, tile_size=96, tile_overlap=32):
        pred = self.predict_noise_from_latents(latents, focus_a, focus_b, prompt_embeds, tiled, tile_size, tile_overlap)
        denoised = self.scheduler_step_from_prediction(pred, latents[0])
        output = self.decode_latents(denoised, expected_output_hw=expected_output_hw)
        return output, denoised, pred

    def predict_noise_from_latents(self, latents, focus_a=None, focus_b=None, prompt_embeds=None,
                                   tiled=False, tile_size=96, tile_overlap=32):
        z_a = latents[0]
        unet_input = self.make_unet_input(latents, focus_a, focus_b)
        ts = self.timesteps.to(z_a.device)
        if hasattr(self.scheduler, "alphas_cumprod") and self.scheduler.alphas_cumprod.device != z_a.device:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(z_a.device)
        if tiled:
            return tiled_unet_forward(self.unet, unet_input, ts, prompt_embeds, 4, tile_size, tile_overlap)
        return self.unet(unet_input, ts, encoder_hidden_states=prompt_embeds).sample

    def scheduler_step_from_prediction(self, model_pred, z_a):
        ts = self.timesteps.to(z_a.device)
        return self.scheduler.step(model_pred, ts, z_a, return_dict=True).prev_sample

    def decode_latents(self, latents, *, expected_output_hw):
        output = self.vae.decode(latents / self.vae.config.scaling_factor).sample.clamp(-1, 1)
        if output.shape[1] != 3:
            raise RuntimeError(f"decoded output channel mismatch: expected=3, actual={output.shape[1]}")
        if tuple(output.shape[-2:]) != tuple(expected_output_hw):
            raise RuntimeError(
                "decoded output size mismatch: "
                f"expected={tuple(expected_output_hw)}, actual={tuple(output.shape[-2:])}, "
                f"latent={tuple(latents.shape[-2:])}, vae_scale_factor={vae_scale_factor(self.vae)}, input_mode={self.input_mode}"
            )
        return output


def checkpoint_payload(model, step, args, optimizer=None, lr_scheduler=None, vsd=None, accelerator=None,
                       optimizer_group_manifest=None, completed_epochs=0, batch_position=0,
                       dataloader_position=0, sampler_epoch=None):
    unet_state = {k: v.detach().cpu() for k, v in model.unet.state_dict().items() if "lora" in k}
    input_mode = getattr(args, "input_mode", getattr(args, "condition_mode", model.condition_mode))
    return {"format_version": 2, "input_mode": normalize_input_mode(input_mode), "condition_mode": model.condition_mode,
            "generator_in_channels": model.unet.conv_in.in_channels,
            "generator_conv_in": copy.deepcopy(model.unet.conv_in.state_dict()),
            "generator_unet_lora": unet_state,
            "generator_lora_adapter_name": getattr(model, "generator_lora_adapter_name", "focus_fusion"),
            "generator_unet_lora_targets": getattr(model, "focus_lora_targets", []),
            "rank_unet": getattr(model, "generator_lora_rank", getattr(model, "lora_rank_unet", getattr(args, "lora_rank_unet", None))),
            "vae_lora_targets": getattr(model, "focus_vae_lora_targets", []),
            "vae_lora_adapter_name": getattr(model, "vae_lora_adapter_name", "focus_vae_encoder"),
            "rank_vae": getattr(model, "vae_lora_rank", getattr(args, "lora_rank_vae", None)),
            "vae_lora": {k: v.detach().cpu() for k, v in model.vae.state_dict().items() if "lora" in k},
            "vsd_unet_lora": {k: v.detach().cpu() for k, v in vsd.unet_update.state_dict().items() if "lora" in k} if vsd is not None else {},
            "vsd_lora_targets": getattr(vsd, "lora_unet_modules_encoder", []) + getattr(vsd, "lora_unet_modules_decoder", []) + getattr(vsd, "lora_unet_others", []) if vsd is not None else [],
            "vsd_lora_adapter_name": getattr(vsd, "vsd_lora_adapter_name", "default_others") if vsd is not None else None,
            "rank_vsd": getattr(model, "vsd_lora_rank", getattr(args, "lora_rank_vsd", None)),
            "train_conv_in": bool(getattr(args, "train_conv_in", True)),
            "train_vae_lora": bool(getattr(args, "train_vae_lora", False)),
            "use_vsd": bool(getattr(args, "use_vsd", False)),
            "prompt_mode": getattr(args, "prompt_mode", "fixed"),
            "fixed_prompt": FIXED_FUSION_PROMPT,
            "cache_fixed_prompt_embedding": bool(getattr(args, "cache_fixed_prompt_embedding", True)),
            "fixed_prompt_embedding": model.fixed_prompt_embedding.cpu(),
            "gradient_accumulation_steps": int(getattr(args, "gradient_accumulation_steps", 1)),
            "world_size": int(getattr(accelerator, "num_processes", 1)) if accelerator is not None else 1,
            "train_batch_size": int(getattr(args, "train_batch_size", 1)),
            "optimizer_group_manifest": optimizer_group_manifest or [],
            "args": vars(args).copy(),
            }


def load_focus_checkpoint(model, checkpoint, load_lora=True):
    state = torch.load(checkpoint, map_location="cpu") if isinstance(checkpoint, (str, bytes)) else checkpoint
    if isinstance(checkpoint, (str, bytes)) and not os.path.exists(checkpoint_complete_path(checkpoint)):
        raise RuntimeError(f"checkpoint is missing complete manifest: {checkpoint_complete_path(checkpoint)}")
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
        if "generator_unet_lora" not in state or not isinstance(state["generator_unet_lora"], dict) or not state["generator_unet_lora"]:
            raise RuntimeError("Generator LoRA checkpoint state is required and must be a non-empty dict")
        load_generator_lora_state(model, state)
    if state.get("fixed_prompt_embedding") is not None:
        model.cache_prompt(state["fixed_prompt_embedding"])
    return state
