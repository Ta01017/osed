"""Train the one-step OSEDiff focus-fusion generator."""
import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataloaders.focus_fusion_dataset import FocusFusionDataset, FIXED_FUSION_PROMPT, focus_fusion_collate
from osediff_focus_fusion import (FocusFusionGenerator, checkpoint_payload, gradient_loss,
                                  laplacian_loss, masked_l1, move_scheduler_to_device,
                                  get_generator_in_channels, normalize_input_mode, vae_scale_factor,
                                  load_vae_lora_state, load_vsd_lora_state,
                                  read_focus_checkpoint_config, load_focus_checkpoint,
                                  capture_rng_state, restore_rng_state,
                                  prepare_checkpoint_temp_dir, finalize_checkpoint_directory,
                                  write_json_atomically, load_verified_checkpoint)


def get_parser_argument_dests(parser):
    return {a.dest for a in parser._actions if a.dest != "help"}


def assert_cli_argument_classification(parser):
    groups = [RESUME_CONFIG_FIELDS, RESUME_OVERRIDE_FIELDS, NON_TRAINING_FIELDS, DEPRECATED_ALIAS_FIELDS, RUNTIME_DERIVED_FIELDS]
    seen = {}
    duplicates = set()
    for group in groups:
        for item in group:
            if item in seen:
                duplicates.add(item)
            seen[item] = True
    if duplicates:
        raise RuntimeError("duplicate CLI argument classification: " + ", ".join(sorted(duplicates)))
    unclassified = get_parser_argument_dests(parser) - set(seen)
    if unclassified:
        raise RuntimeError("unclassified CLI arguments: " + ", ".join(sorted(unclassified)))


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
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--validation_max_samples", type=int, default=4)
    p.add_argument("--keep_a_composite", action="store_true")
    p.add_argument("--keep_threshold", type=float, default=.5)
    p.add_argument("--keep_soft_width", type=float, default=.1)
    p.add_argument("--native_resolution", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict_native_size", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_pixels", type=int, help="Maximum allowed native pixels. Images are rejected rather than resized.")
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--lr_scheduler", type=str, default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--lr_num_cycles", type=int, default=1)
    p.add_argument("--lr_power", type=float, default=1.0)
    p.add_argument("--lora_rank_unet", type=int, default=8)
    p.add_argument("--lora_rank_vae", type=int, default=4)
    p.add_argument("--lora_rank_vsd", type=int, default=8)
    p.add_argument("--lora_rank", type=int, default=None, help="Deprecated. Use --lora_rank_unet/vae/vsd.")
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
    p.add_argument("--sync_with_dataloader", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--resume_from_checkpoint")
    p.add_argument("--ram_path")
    p.add_argument("--ram_ft_path")
    assert_cli_argument_classification(p)
    args = p.parse_args(argv)
    if args.lora_rank is not None:
        raise ValueError("--lora_rank is deprecated and no longer accepted. Use --lora_rank_unet, --lora_rank_vae and --lora_rank_vsd explicitly.")
    return args


def _add_generator_lora(unet, rank, targets=None, adapter_name="focus_fusion"):
    from peft import LoraConfig
    targets = targets or sorted({n.rsplit(".", 1)[0] for n, p in unet.named_parameters()
                      if p.ndim >= 2 and "conv_in" not in n and any(x in n for x in ("to_q", "to_k", "to_v", "to_out.0"))})
    if not targets: raise RuntimeError("no UNet attention modules found for LoRA")
    unet.add_adapter(LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=targets), adapter_name=adapter_name)
    unet.set_adapter([adapter_name])
    return targets


def _add_vae_lora(vae, rank, targets=None, adapter_name="focus_vae_encoder"):
    from peft import LoraConfig
    patterns = ("conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0")
    if targets is None:
        targets = []
        for name, param in vae.named_parameters():
            if "bias" in name or "norm" in name or param.ndim < 2:
                continue
            if ("encoder" in name and any(p in name for p in patterns)) or ("quant_conv" in name and "post_quant_conv" not in name):
                targets.append(name.replace(".weight", ""))
        targets = sorted(set(targets))
    if not targets:
        raise RuntimeError("no VAE encoder modules found for LoRA")
    vae.add_adapter(LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=targets), adapter_name=adapter_name)
    vae.set_adapter([adapter_name])
    return targets


def _named_params(module, predicate, prefix=""):
    return [(prefix + n, p) for n, p in module.named_parameters() if predicate(n, p)]


def optimizer_group_manifest(param_groups):
    seen = set()
    manifest = []
    for group in param_groups:
        names = group.get("parameter_names", [])
        shapes = group.get("parameter_shapes", {})
        for p in group["params"]:
            if id(p) in seen:
                raise RuntimeError("same Parameter appears in multiple optimizer groups")
            seen.add(id(p))
            if not p.requires_grad:
                raise RuntimeError(f"frozen parameter in optimizer group {group['name']}")
        manifest.append({
            "name": group["name"],
            "group_name": group["name"],
            "parameter_names": list(names),
            "parameter_shapes": {k: list(v) for k, v in shapes.items()},
            "num_tensors": len(group["params"]),
            "num_parameters": int(sum(p.numel() for p in group["params"])),
            "tensor_count": len(group["params"]),
            "total_numel": int(sum(p.numel() for p in group["params"])),
            "lr": group.get("lr"),
            "weight_decay": group.get("weight_decay"),
        })
    return manifest


def validate_optimizer_manifest(saved, current):
    if saved is None:
        raise RuntimeError("checkpoint optimizer_group_manifest is missing")
    if len(saved) != len(current):
        raise RuntimeError(f"optimizer group count mismatch: checkpoint={len(saved)} current={len(current)}")
    for old, new in zip(saved, current):
        for key in ("name", "num_tensors", "num_parameters", "lr", "weight_decay"):
            if old.get(key) != new.get(key):
                raise RuntimeError(f"optimizer group manifest mismatch for {new.get('name')}: {key} checkpoint={old.get(key)} current={new.get(key)}")
        if old.get("parameter_names", []) != new.get("parameter_names", []):
            raise RuntimeError(
                "[OPTIMIZER MANIFEST MISMATCH]\n"
                f"group={new['name']}\n"
                f"checkpoint_names={old.get('parameter_names', [])}\n"
                f"current_names={new.get('parameter_names', [])}"
            )
        if old.get("parameter_shapes", {}) != new.get("parameter_shapes", {}):
            raise RuntimeError(f"optimizer group parameter shapes mismatch for {new['name']}")


RESUME_ALLOWED_CLI_OVERRIDES = {"max_train_steps", "validation_steps", "checkpointing_steps", "output_dir"}


@dataclass
class TrainingProgress:
    global_step: int = 0
    current_epoch: int = 0
    completed_epochs: int = 0
    batches_consumed_in_current_epoch: int = 0
    micro_batches: int = 0
    optimizer_updates: int = 0
    scheduler_steps: int = 0
    sampler_epoch: int = 0

    @classmethod
    def from_trainer_state(cls, trainer_state):
        if not trainer_state:
            raise ValueError("[INVALID TRAINER STATE] missing trainer_state")
        required = ("global_step", "current_epoch", "completed_epochs", "batches_consumed_in_current_epoch",
                    "micro_batches", "optimizer_updates", "scheduler_steps", "sampler_epoch")
        for field in required:
            if field not in trainer_state:
                raise ValueError(f"[INVALID TRAINER STATE] missing required progress field: {field}")
        return cls(
            global_step=int(trainer_state["global_step"]),
            current_epoch=int(trainer_state.get("current_epoch", 0)),
            completed_epochs=int(trainer_state.get("completed_epochs", 0)),
            batches_consumed_in_current_epoch=int(trainer_state["batches_consumed_in_current_epoch"]),
            micro_batches=int(trainer_state.get("micro_batches", 0)),
            optimizer_updates=int(trainer_state.get("optimizer_updates", trainer_state["global_step"])),
            scheduler_steps=int(trainer_state.get("scheduler_steps", trainer_state["global_step"])),
            sampler_epoch=int(trainer_state.get("sampler_epoch", trainer_state.get("current_epoch", 0))),
        )

    def to_trainer_state_fields(self):
        return {
            "global_step": int(self.global_step),
            "current_epoch": int(self.current_epoch),
            "completed_epochs": int(self.completed_epochs),
            "batches_consumed_in_current_epoch": int(self.batches_consumed_in_current_epoch),
            "micro_batches": int(self.micro_batches),
            "optimizer_updates": int(self.optimizer_updates),
            "scheduler_steps": int(self.scheduler_steps),
            "sampler_epoch": int(self.sampler_epoch),
        }


def validate_training_progress(progress: TrainingProgress, *, gradient_accumulation_steps, dataloader_length=None):
    fields = ("global_step", "current_epoch", "completed_epochs", "sampler_epoch",
              "batches_consumed_in_current_epoch", "micro_batches", "optimizer_updates", "scheduler_steps")
    for field in fields:
        if getattr(progress, field) < 0:
            raise ValueError(f"[INVALID TRAINING PROGRESS] {field} must be >= 0, got {getattr(progress, field)}")
    if progress.optimizer_updates != progress.global_step:
        raise ValueError(f"[INVALID TRAINING PROGRESS] optimizer_updates={progress.optimizer_updates} global_step={progress.global_step}")
    if progress.scheduler_steps != progress.global_step:
        raise ValueError(f"[INVALID TRAINING PROGRESS] scheduler_steps={progress.scheduler_steps} global_step={progress.global_step}")
    if progress.micro_batches < progress.optimizer_updates:
        raise ValueError(f"[INVALID TRAINING PROGRESS] micro_batches={progress.micro_batches} optimizer_updates={progress.optimizer_updates}")
    if progress.completed_epochs > progress.current_epoch:
        raise ValueError(f"[INVALID TRAINING PROGRESS] completed_epochs={progress.completed_epochs} current_epoch={progress.current_epoch}")
    if progress.sampler_epoch != progress.current_epoch:
        raise ValueError(f"[INVALID TRAINING PROGRESS] sampler_epoch={progress.sampler_epoch} current_epoch={progress.current_epoch}")
    if dataloader_length is not None and progress.batches_consumed_in_current_epoch > dataloader_length:
        raise ValueError(f"[INVALID TRAINING PROGRESS] batches_consumed_in_current_epoch={progress.batches_consumed_in_current_epoch} dataloader_length={dataloader_length}")


def validate_resume_config(args, resume_cfg):
    """Validate structural resume fields and copy checkpoint-owned structure onto args."""
    if not resume_cfg:
        return args
    print("[RESUME] allowed CLI overrides:", sorted(RESUME_ALLOWED_CLI_OVERRIDES))
    structural = {
        "input_mode": normalize_input_mode(getattr(args, "condition_mode", None) or args.input_mode),
        "generator_lora_rank": args.lora_rank_unet,
        "vae_lora_rank": args.lora_rank_vae,
        "vsd_lora_rank": args.lora_rank_vsd,
        "train_conv_in": args.train_conv_in,
        "train_vae_lora": args.train_vae_lora,
        "use_vsd": bool(args.use_vsd),
        "prompt_mode": args.prompt_mode,
    }
    for name, cli_value in structural.items():
        ckpt_value = resume_cfg[name]
        if cli_value != ckpt_value:
                _resume_mismatch(name, ckpt_value, cli_value)
    args.input_mode = resume_cfg["input_mode"]
    args.lora_rank_unet = resume_cfg["generator_lora_rank"]
    args.lora_rank_vae = resume_cfg["vae_lora_rank"]
    args.lora_rank_vsd = resume_cfg["vsd_lora_rank"]
    args.train_conv_in = resume_cfg["train_conv_in"]
    args.train_vae_lora = resume_cfg["train_vae_lora"]
    args.use_vsd = int(resume_cfg["use_vsd"])
    args.prompt_mode = resume_cfg["prompt_mode"]
    return args


def log_accelerator_resume_success(*, trainer_state):
    """Log resume success; training progress is owned by TrainingProgress.from_trainer_state."""
    if not trainer_state:
        return
    print("[RESUME] optimizer restored by Accelerate")
    print("[RESUME] scheduler restored by Accelerate")
    print("[RESUME] scaler restored by Accelerate")
    print("[RESUME] rank RNG restored by Accelerate")
    print(f"[RESUME] checkpoint global_step {int(trainer_state['global_step'])}")
    print(f"[RESUME] completed_epochs {int(trainer_state['completed_epochs'])}")
    print(f"[RESUME] batches consumed in current epoch {int(trainer_state['batches_consumed_in_current_epoch'])}")
    print("[RESUME] sampler position restored")
    print("[RESUME] trainer state restored")


def build_train_sampler(dataset, accelerator, args, epoch=0):
    deterministic_shuffle = True
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=deterministic_shuffle,
        seed=args.seed,
        drop_last=False,
    )
    sampler.set_epoch(epoch)
    return sampler


def _resume_mismatch(field, checkpoint_value, current_value):
    raise ValueError(
        "[RESUME CONFIG MISMATCH] "
        f"field={field}, checkpoint={checkpoint_value!r}, command_line={current_value!r}"
    )


def normalize_path_for_resume(path):
    return None if path is None else str(Path(path).expanduser().resolve())


def normalize_model_identifier(value):
    if value is None:
        return None
    candidate = Path(value).expanduser()
    return str(candidate.resolve()) if candidate.exists() else str(value).strip()


RESUME_CONFIG_FIELDS = {
    "pretrained_model_name_or_path", "input_mode", "generator_in_channels", "lora_rank_unet",
    "lora_rank_vae", "lora_rank_vsd", "train_conv_in", "train_vae_lora", "use_vsd",
    "prompt_mode", "fixed_prompt", "cache_fixed_prompt_embedding", "generator_adapter_name",
    "vae_adapter_name", "vsd_adapter_name", "generator_target_modules", "vae_target_modules",
    "vsd_target_modules", "vae_scale_factor", "metadata_path", "dataset_base_path",
    "start_index", "max_samples", "dataset_length", "train_batch_size",
    "gradient_accumulation_steps", "sync_with_dataloader", "world_size", "seed", "sampler_seed", "drop_last",
    "random_flip", "mixed_precision", "learning_rate", "weight_decay", "lr_scheduler",
    "lr_warmup_steps", "lr_num_cycles", "lr_power", "max_grad_norm", "cfg_vsd",
    "keep_threshold", "keep_soft_width", "max_pixels", "native_resolution", "strict_native_size", "lambda_l2",
    "lambda_lpips", "lambda_vsd", "lambda_vsd_lora", "lambda_keep", "lambda_bref",
    "lambda_gradient", "lambda_laplacian",
}
RESUME_OVERRIDE_FIELDS = {"max_train_steps", "checkpointing_steps", "validation_steps", "validation_max_samples", "logging_steps", "output_dir"}
RESUME_INVARIANT_FIELDS = RESUME_CONFIG_FIELDS
NON_TRAINING_FIELDS = {
    "resume_from_checkpoint", "ram_path", "ram_ft_path", "smoke", "resolution",
    "random_crop", "center_crop", "dataloader_num_workers", "keep_a_composite",
}
DEPRECATED_ALIAS_FIELDS = {"condition_mode", "lora_rank"}
RUNTIME_DERIVED_FIELDS = set()


def build_resume_config_snapshot(args, *, model, accelerator, dataset_length):
    return {
        "pretrained_model_name_or_path": normalize_model_identifier(args.pretrained_model_name_or_path),
        "input_mode": normalize_input_mode(args.input_mode),
        "generator_in_channels": int(model.unet.conv_in.in_channels),
        "lora_rank_unet": int(args.lora_rank_unet),
        "lora_rank_vae": int(args.lora_rank_vae),
        "lora_rank_vsd": int(args.lora_rank_vsd),
        "train_conv_in": bool(args.train_conv_in),
        "train_vae_lora": bool(args.train_vae_lora),
        "use_vsd": bool(args.use_vsd),
        "prompt_mode": args.prompt_mode,
        "fixed_prompt": FIXED_FUSION_PROMPT,
        "cache_fixed_prompt_embedding": bool(args.cache_fixed_prompt_embedding),
        "generator_adapter_name": getattr(model, "generator_lora_adapter_name", "focus_fusion"),
        "vae_adapter_name": getattr(model, "vae_lora_adapter_name", "focus_vae_encoder"),
        "vsd_adapter_name": getattr(model, "vsd_lora_adapter_name", None),
        "generator_target_modules": list(getattr(model, "focus_lora_targets", [])),
        "vae_target_modules": list(getattr(model, "focus_vae_lora_targets", [])),
        "vsd_target_modules": list(getattr(model, "focus_vsd_lora_targets", [])),
        "vae_scale_factor": int(vae_scale_factor(model.vae)),
        "metadata_path": normalize_path_for_resume(args.metadata_path),
        "dataset_base_path": normalize_path_for_resume(args.dataset_base_path),
        "start_index": int(args.start_index),
        "max_samples": args.max_samples,
        "dataset_length": int(dataset_length),
        "train_batch_size": int(args.train_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "sync_with_dataloader": bool(args.sync_with_dataloader),
        "world_size": int(accelerator.num_processes),
        "seed": int(args.seed),
        "sampler_seed": int(args.seed),
        "drop_last": False,
        "random_flip": bool(args.random_flip),
        "mixed_precision": args.mixed_precision,
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(getattr(args, "weight_decay", 0.01)),
        "lr_scheduler": args.lr_scheduler,
        "lr_warmup_steps": int(args.lr_warmup_steps),
        "lr_num_cycles": int(args.lr_num_cycles),
        "lr_power": float(args.lr_power),
        "max_grad_norm": float(args.max_grad_norm),
        "cfg_vsd": float(args.cfg_vsd),
        "keep_threshold": float(args.keep_threshold),
        "keep_soft_width": float(args.keep_soft_width),
        "max_pixels": args.max_pixels,
        "native_resolution": bool(args.native_resolution),
        "strict_native_size": bool(args.strict_native_size),
        "lambda_l2": float(args.lambda_l2),
        "lambda_lpips": float(args.lambda_lpips),
        "lambda_vsd": float(args.lambda_vsd),
        "lambda_vsd_lora": float(args.lambda_vsd_lora),
        "lambda_keep": float(args.lambda_keep),
        "lambda_bref": float(args.lambda_bref),
        "lambda_gradient": float(args.lambda_gradient),
        "lambda_laplacian": float(args.lambda_laplacian),
        "max_train_steps": int(args.max_train_steps),
        "checkpointing_steps": int(args.checkpointing_steps),
        "validation_steps": int(args.validation_steps),
        "validation_max_samples": int(args.validation_max_samples),
        "logging_steps": int(args.logging_steps),
        "output_dir": normalize_path_for_resume(args.output_dir),
    }


def validate_resume_configuration(*, saved_config, current_config, allowed_overrides=None):
    allowed = RESUME_OVERRIDE_FIELDS if allowed_overrides is None else set(allowed_overrides)
    problems = []
    for field in sorted(RESUME_CONFIG_FIELDS):
        if field not in saved_config:
            problems.append((field, "<missing>", current_config.get(field, "<missing>")))
            continue
        if field not in current_config:
            problems.append((field, saved_config.get(field), "<missing>"))
            continue
        if field in allowed:
            continue
        if saved_config[field] != current_config[field]:
            problems.append((field, saved_config[field], current_config[field]))
    if problems:
        detail = "\n".join(f"* field={f}\n  checkpoint={a!r}\n  command_line={b!r}" for f, a, b in problems)
        raise ValueError("[RESUME CONFIG MISMATCH]\n" + detail)
    print("[RESUME] resume config validated")


def validate_sampler_resume_state(trainer_state, *, dataset_length, world_size, train_batch_size,
                                  gradient_accumulation_steps, sampler_seed, drop_last,
                                  sync_with_dataloader=True, dataloader_length=None):
    if not trainer_state:
        return
    version = int(trainer_state.get("trainer_state_version", 3))
    if version not in (3, 4):
        _resume_mismatch("trainer_state_version", trainer_state.get("trainer_state_version"), "3 or 4")
    checks = {
        "dataset_length": dataset_length,
        "world_size": world_size,
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "sampler_seed": sampler_seed,
        "drop_last": drop_last,
    }
    if version >= 4 or "sync_with_dataloader" in trainer_state:
        checks["sync_with_dataloader"] = sync_with_dataloader
    for field, current in checks.items():
        if trainer_state.get(field) != current:
            _resume_mismatch(field, trainer_state.get(field), current)
    sampler_epoch = int(trainer_state.get("sampler_epoch", -1))
    batch_pos = int(trainer_state.get("batches_consumed_in_current_epoch", -1))
    if sampler_epoch < 0:
        _resume_mismatch("sampler_epoch", sampler_epoch, ">=0")
    if int(trainer_state.get("current_epoch", sampler_epoch)) != sampler_epoch:
        _resume_mismatch("current_epoch", trainer_state.get("current_epoch"), sampler_epoch)
    if batch_pos < 0:
        _resume_mismatch("batches_consumed_in_current_epoch", batch_pos, ">=0")
    if dataloader_length is not None and batch_pos > dataloader_length:
        _resume_mismatch("batches_consumed_in_current_epoch", batch_pos, f"<= {dataloader_length}")
    if version >= 4 and dataloader_length is not None and batch_pos == dataloader_length:
        raise ValueError("[INVALID TRAINER STATE] version 4 checkpoints must store normalized epoch state")
    global_step = int(trainer_state.get("global_step", -1))
    if global_step < 0:
        _resume_mismatch("global_step", global_step, ">=0")
    if int(trainer_state.get("optimizer_updates", global_step)) != global_step:
        _resume_mismatch("optimizer_updates", trainer_state.get("optimizer_updates"), global_step)
    if int(trainer_state.get("scheduler_steps", global_step)) != global_step:
        _resume_mismatch("scheduler_steps", trainer_state.get("scheduler_steps"), global_step)
    if int(trainer_state.get("micro_batches", global_step)) < global_step:
        _resume_mismatch("micro_batches", trainer_state.get("micro_batches"), f">= {global_step}")
    if int(trainer_state.get("completed_epochs", 0)) > int(trainer_state.get("current_epoch", 0)):
        _resume_mismatch("completed_epochs", trainer_state.get("completed_epochs"), f"<= {trainer_state.get('current_epoch')}")
    print("[RESUME] sampler state validated")


def broadcast_checkpoint_temp_dir(accelerator, temp_dir_on_main, final_dir):
    final_dir = Path(final_dir)
    if accelerator.num_processes == 1:
        temp_dir = Path(temp_dir_on_main)
    else:
        import torch.distributed as dist
        payload = [str(temp_dir_on_main) if accelerator.is_main_process else None]
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("distributed temp_dir broadcast requires initialized torch.distributed")
        dist.broadcast_object_list(payload, src=0)
        if not payload[0]:
            raise RuntimeError("checkpoint temp_dir broadcast returned an empty path")
        temp_dir = Path(payload[0])
    expected_prefix = f".{final_dir.name}.tmp-"
    if not temp_dir.name.startswith(expected_prefix):
        raise RuntimeError(f"broadcast temp_dir name mismatch: expected prefix={expected_prefix}, actual={temp_dir.name}")
    if temp_dir.parent.resolve() != final_dir.parent.resolve():
        raise RuntimeError(f"broadcast temp_dir parent mismatch: expected={final_dir.parent}, actual={temp_dir.parent}")
    if not temp_dir.is_dir():
        raise RuntimeError(f"broadcast temp_dir does not exist: {temp_dir}")
    return temp_dir


def run_training_loop(*, accelerator, model, train_dataloader, train_sampler, optimizer, lr_scheduler,
                      progress: TrainingProgress, gradient_accumulation_steps: int,
                      max_train_steps, checkpointing_steps=0,
                      validation_steps=0, compute_loss_fn=None, checkpoint_fn=None,
                      validation_fn=None, logging_fn=None, event_callback=None, logging_steps=10,
                      max_grad_norm=None):
    """Shared optimizer-update loop used by formal training tests and main-like code."""
    optimizer.zero_grad(set_to_none=True)
    validate_training_progress(progress, gradient_accumulation_steps=gradient_accumulation_steps, dataloader_length=len(train_dataloader) if hasattr(train_dataloader, "__len__") else None)
    while progress.global_step < max_train_steps:
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(progress.sampler_epoch)
        epoch_exhausted = True
        for batch_index, batch in enumerate(train_dataloader):
            if batch_index < progress.batches_consumed_in_current_epoch:
                continue
            with accelerator.accumulate(model):
                result = compute_loss_fn(model, batch) if compute_loss_fn else model(batch)
                if isinstance(result, tuple):
                    loss, losses = result
                else:
                    loss, losses = result, {}
                before_step = progress.global_step
                progress.micro_batches += 1
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if max_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    progress.optimizer_updates += 1
                    progress.global_step += 1
                    lr_scheduler.step()
                    progress.scheduler_steps += 1
                    if event_callback:
                        event_callback({
                            "event": "optimizer_update",
                            "global_step": progress.global_step,
                            "micro_batches": progress.micro_batches,
                        })
                if event_callback:
                    event_callback({
                        "event": "micro_batch",
                        "epoch": progress.current_epoch,
                        "batch_index": batch_index,
                        "sync_gradients": bool(accelerator.sync_gradients),
                        "global_step_before": before_step,
                        "global_step_after": progress.global_step,
                    })
            progress.batches_consumed_in_current_epoch = batch_index + 1
            if accelerator.sync_gradients:
                if logging_fn and (progress.global_step == 1 or (logging_steps and progress.global_step % logging_steps == 0)):
                    logging_fn(progress, losses)
                if validation_steps and progress.global_step % validation_steps == 0 and validation_fn:
                    validation_fn(progress)
                if checkpointing_steps and progress.global_step % checkpointing_steps == 0 and checkpoint_fn:
                    checkpoint_fn(progress, losses)
            if progress.global_step >= max_train_steps:
                if hasattr(train_dataloader, "__len__") and progress.batches_consumed_in_current_epoch == len(train_dataloader):
                    progress.completed_epochs += 1
                    progress.current_epoch += 1
                    progress.sampler_epoch = progress.current_epoch
                    progress.batches_consumed_in_current_epoch = 0
                epoch_exhausted = False
                break
        if epoch_exhausted:
            progress.completed_epochs += 1
            progress.current_epoch += 1
            progress.sampler_epoch = progress.current_epoch
            progress.batches_consumed_in_current_epoch = 0
    validate_training_progress(progress, gradient_accumulation_steps=gradient_accumulation_steps, dataloader_length=len(train_dataloader) if hasattr(train_dataloader, "__len__") else None)
    return progress


def _groups(model, args, vsd=None):
    groups = []
    named = sorted(_named_params(model.unet, lambda n, p: "lora" in n and p.requires_grad, "generator_unet."), key=lambda x: x[0])
    groups.append({"name": "generator_unet_lora", "params": [p for _, p in named], "parameter_names": [n for n, _ in named]})
    if args.train_conv_in:
        model.unet.conv_in.requires_grad_(True)
        named = sorted(_named_params(model.unet.conv_in, lambda n, p: p.requires_grad, "generator_conv_in."), key=lambda x: x[0])
        groups.append({"name": "generator_conv_in", "params": [p for _, p in named], "parameter_names": [n for n, _ in named]})
    if args.train_vae_lora:
        named = sorted(_named_params(model.vae, lambda n, p: "lora" in n and p.requires_grad, "vae."), key=lambda x: x[0])
        groups.append({"name": "vae_lora", "params": [p for _, p in named], "parameter_names": [n for n, _ in named]})
    if vsd is not None:
        named = sorted(_named_params(vsd.unet_update, lambda n, p: "lora" in n and p.requires_grad, "vsd_update_unet."), key=lambda x: x[0])
        groups.append({"name": "vsd_update_lora", "params": [p for _, p in named], "parameter_names": [n for n, _ in named]})
    groups = [g for g in groups if g["params"]]
    for g in groups:
        g["parameter_shapes"] = {name: tuple(p.shape) for name, p in zip(g["parameter_names"], g["params"])}
    seen = set()
    for g in groups:
        for p in g["params"]:
            if id(p) in seen:
                raise RuntimeError("same Parameter appears in multiple optimizer groups")
            seen.add(id(p))
    ids = {id(p) for g in groups for p in g["params"]}
    if args.train_conv_in:
        assert id(model.unet.conv_in.weight) in ids, "expanded conv_in weight is absent from optimizer"
    else:
        model.unet.conv_in.requires_grad_(False)
        assert id(model.unet.conv_in.weight) not in ids, "conv_in was added while train_conv_in=false"
    total = 0
    for g in groups:
        count = sum(p.numel() for p in g["params"]); train = sum(p.numel() for p in g["params"] if p.requires_grad); total += train
        print(f"optimizer group {g['name']}: tensors={len(g['params'])}, parameters={count:,}, trainable={train:,}")
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


def simulate_optimizer_update_schedule(num_micro_batches, gradient_accumulation_steps, max_train_steps,
                                       initial_global_step=0, checkpointing_steps=0, validation_steps=0):
    global_step = int(initial_global_step)
    optimizer_steps = scheduler_steps = consumed = 0
    checkpoints, validations = [], []
    for _ in range(num_micro_batches):
        if global_step >= max_train_steps:
            break
        consumed += 1
        if consumed % gradient_accumulation_steps == 0:
            optimizer_steps += 1
            scheduler_steps += 1
            global_step += 1
            if checkpointing_steps and global_step % checkpointing_steps == 0:
                checkpoints.append(global_step)
            if validation_steps and global_step % validation_steps == 0:
                validations.append(global_step)
    return {"global_step": global_step, "micro_batches": consumed, "optimizer_steps": optimizer_steps,
            "scheduler_steps": scheduler_steps, "checkpoints": checkpoints, "validations": validations}


@torch.no_grad()
def run_validation(model, loader, encode_fn, args, accelerator, step):
    if not accelerator.is_main_process:
        return
    raw_model = accelerator.unwrap_model(model)
    was_training = raw_model.training
    rng_state = capture_rng_state()
    try:
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
                canvas.paste(panel, (x, 0))
                x += panel.width
            canvas.save(folder / "comparison.png")
            pred_saved = Image.open(folder / "pred_raw.png")
            a_saved = Image.open(folder / "A.png")
            if pred_saved.size != a_saved.size:
                raise RuntimeError(f"validation saved size mismatch: pred_raw={pred_saved.size}, A={a_saved.size}")
    finally:
        restore_rng_state(rng_state)
        if was_training:
            raw_model.train()


def main(args):
    from accelerate import Accelerator
    from accelerate.utils import GradientAccumulationPlugin, set_seed
    from diffusers import DDPMScheduler
    from diffusers.optimization import get_scheduler
    from transformers import AutoTokenizer, CLIPTextModel
    from models.autoencoder_kl import AutoencoderKL
    from models.unet_2d_condition import UNet2DConditionModel
    from osediff import OSEDiff_reg
    from osediff_focus_fusion import expand_unet_conv_in

    args.input_mode = normalize_input_mode(args.condition_mode or args.input_mode)
    verified_resume = load_verified_checkpoint(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    trainer_state = verified_resume["trainer_state"] if verified_resume else None
    if verified_resume and verified_resume["manifest"]["global_step"] != trainer_state["global_step"]:
        raise ValueError("[RESUME CONFIG MISMATCH]\n* field=global_step\n  checkpoint_manifest="
                         f"{verified_resume['manifest']['global_step']!r}\n  trainer_state={trainer_state['global_step']!r}")
    resume_cfg = read_focus_checkpoint_config(verified_resume["model_state"]) if verified_resume else None
    validate_resume_config(args, resume_cfg)
    if args.input_mode != "ab_focus" and (args.lambda_keep or args.lambda_bref):
        print(f"WARNING: keep-A and B-reference losses are disabled for input_mode={args.input_mode}; no fake masks will be created")
    accumulation_plugin = GradientAccumulationPlugin(
        num_steps=args.gradient_accumulation_steps,
        sync_with_dataloader=args.sync_with_dataloader,
    )
    accelerator = Accelerator(gradient_accumulation_plugin=accumulation_plugin, mixed_precision=args.mixed_precision)
    set_seed(args.seed); Path(args.output_dir, "checkpoints").mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder.to(accelerator.device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    # Required order: base UNet -> expand conv_in -> add adapters.
    expand_unet_conv_in(unet, get_generator_in_channels(args.input_mode))
    unet.requires_grad_(False); vae.requires_grad_(False); text_encoder.requires_grad_(False)
    lora_targets = _add_generator_lora(
        unet, args.lora_rank_unet,
        targets=resume_cfg["generator_lora_targets"] if resume_cfg else None,
        adapter_name=resume_cfg["generator_lora_adapter_name"] if resume_cfg else "focus_fusion",
    )
    vae_lora_targets = _add_vae_lora(
        vae,
        resume_cfg["vae_lora_rank"] if resume_cfg else args.lora_rank_vae,
        targets=resume_cfg["vae_lora_targets"] if resume_cfg else None,
        adapter_name=resume_cfg["vae_lora_adapter_name"] if resume_cfg else "focus_vae_encoder",
    ) if args.train_vae_lora else []
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
    model.generator_lora_rank = args.lora_rank_unet
    model.vae_lora_rank = args.lora_rank_vae if args.train_vae_lora else None
    model.vsd_lora_rank = args.lora_rank_vsd if args.use_vsd else None
    model.lora_rank_vae = model.vae_lora_rank
    model.generator_lora_adapter_name = resume_cfg["generator_lora_adapter_name"] if resume_cfg else "focus_fusion"
    model.vae_lora_adapter_name = resume_cfg["vae_lora_adapter_name"] if resume_cfg else "focus_vae_encoder"
    if args.use_vsd:
        import copy
        vsd_args = copy.deepcopy(args)
        vsd_args.lora_rank = args.lora_rank_vsd
        model_reg = OSEDiff_reg(args=vsd_args, accelerator=accelerator)
    else:
        model_reg = None
    if model_reg is not None:
        model_reg.set_train()
        assert model_reg.unet_fix.config.in_channels == 4
        assert model_reg.unet_update.config.in_channels == 4
    resume_state = None
    if resume_cfg:
        resume_state = load_focus_checkpoint(model, verified_resume["model_state"])
        load_vae_lora_state(model, resume_state)
        if model_reg is not None:
            load_vsd_lora_state(model_reg, resume_state)

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
    initial_sampler_epoch = int(trainer_state.get("sampler_epoch", trainer_state.get("current_epoch", 0))) if trainer_state else 0
    train_sampler = build_train_sampler(dataset, accelerator, args, initial_sampler_epoch)
    loader = DataLoader(dataset, args.train_batch_size, shuffle=False, sampler=train_sampler,
                        num_workers=args.dataloader_num_workers, collate_fn=focus_fusion_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=focus_fusion_collate)
    param_groups = _groups(model, args, model_reg)
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)
    current_manifest = optimizer_group_manifest(optimizer.param_groups)
    for group in optimizer.param_groups:
        print(
            f"optimizer group {group['name']}: tensors={len(group['params'])}, "
            f"lr={group['lr']}, weight_decay={group['weight_decay']}"
        )
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)
    current_resume_config = build_resume_config_snapshot(args, model=model, accelerator=accelerator, dataset_length=len(dataset))
    if trainer_state:
        validate_resume_configuration(saved_config=trainer_state["resume_config"], current_config=current_resume_config)
        validate_sampler_resume_state(
            trainer_state,
            dataset_length=len(dataset),
            dataloader_length=len(loader),
            world_size=accelerator.num_processes,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            sync_with_dataloader=args.sync_with_dataloader,
            sampler_seed=args.seed,
            drop_last=False,
        )
        validate_optimizer_manifest(verified_resume["optimizer_manifest"], current_manifest)
        print("[RESUME] optimizer manifest validated")
    if model_reg is None:
        model, text_encoder, optimizer, loader, lr_scheduler = accelerator.prepare(model, text_encoder, optimizer, loader, lr_scheduler)
    else:
        model, model_reg, text_encoder, optimizer, loader, lr_scheduler = accelerator.prepare(model, model_reg, text_encoder, optimizer, loader, lr_scheduler)
    if verified_resume:
        accelerator.load_state(str(Path(verified_resume["checkpoint_dir"], "accelerator_state")))
        print("[RESUME] accelerator state restored")
    log_accelerator_resume_success(trainer_state=trainer_state)
    progress = TrainingProgress.from_trainer_state(trainer_state) if trainer_state else TrainingProgress()
    validate_training_progress(progress, gradient_accumulation_steps=args.gradient_accumulation_steps, dataloader_length=len(loader))

    def compute_loss_callback(train_model, batch):
        prompts = batch["prompt"]
        if args.prompt_mode == "ram":
            x = ram_tf(batch["gt"].mul(0.5).add(0.5)).to(accelerator.device, dtype=torch.float16)
            prompts = [str(x) for x in ram_infer(x, ram_model)]
        raw = accelerator.unwrap_model(train_model)
        if args.prompt_mode == "fixed" and raw.fixed_prompt_embedding.numel():
            emb = raw.fixed_prompt_embedding.to(accelerator.device).expand(len(prompts), -1, -1)
        else:
            emb = encode(prompts, accelerator.device)
        conditions = [x.to(accelerator.device) for x in batch["conditions"]]
        focus_maps = [x.to(accelerator.device) for x in batch["focus_maps"]]
        fa = focus_maps[0] if focus_maps else None
        fb = focus_maps[1] if len(focus_maps) > 1 else None
        pred, latent, _ = train_model(conditions, fa, fb, emb, "sample")
        gt, a = batch["gt"].to(accelerator.device), conditions[0]
        losses = {"l2": F.mse_loss(pred.float(), gt.float()) * args.lambda_l2,
                  "gradient": gradient_loss(pred.float(), gt.float()) * args.lambda_gradient,
                  "laplacian": laplacian_loss(pred.float(), gt.float()) * args.lambda_laplacian}
        if args.input_mode == "ab_focus":
            keep, bref = fa.clamp(0, 1), ((1 - fa) * fb).clamp(0, 1)
            losses["keep"] = masked_l1(pred.float(), a.float(), keep.float()) * args.lambda_keep
            losses["bref"] = masked_l1(pred.float(), gt.float(), bref.float()) * args.lambda_bref
        if args.lambda_lpips:
            import lpips
            if not hasattr(main, "lpips_net"):
                main.lpips_net = lpips.LPIPS(net="vgg").to(accelerator.device).requires_grad_(False)
            losses["lpips"] = main.lpips_net(pred.float(), gt.float()).mean() * args.lambda_lpips
        if model_reg is not None and args.lambda_vsd:
            neg_emb = encode([""] * len(prompts), accelerator.device)
            reg = model_reg.module if hasattr(model_reg, "module") else model_reg
            losses["vsd"] = reg.distribution_matching_loss(latent, emb, neg_emb, args) * args.lambda_vsd
        if model_reg is not None and args.lambda_vsd_lora:
            reg = model_reg.module if hasattr(model_reg, "module") else model_reg
            losses["vsd_lora"] = reg.diff_loss(latent, emb, args) * args.lambda_vsd_lora
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            print("non-finite loss components:", {k: float(v.detach().float().cpu()) for k, v in losses.items()})
            print("metadata_index:", batch["metadata_index"])
            raise RuntimeError("NaN/Inf loss detected")
        return loss, losses

    def logging_callback(progress, losses):
        if accelerator.is_main_process:
            log_losses = {k: round(v.item(), 6) for k, v in losses.items()}
            if args.input_mode != "ab_focus":
                log_losses["keep"] = "disabled"; log_losses["bref"] = "disabled"
            print(progress.global_step, log_losses)

    def validation_callback(progress):
        accelerator.wait_for_everyone()
        run_validation(model, val_loader, encode, args, accelerator, progress.global_step)
        accelerator.wait_for_everyone()

    def checkpoint_callback(progress, losses):
        del losses
        accelerator.wait_for_everyone()
        checkpoint_dir = Path(args.output_dir, "checkpoints", f"checkpoint-{progress.global_step:08d}")
        temp_on_main = prepare_checkpoint_temp_dir(checkpoint_dir) if accelerator.is_main_process else None
        temp_dir = broadcast_checkpoint_temp_dir(accelerator, temp_on_main, checkpoint_dir)
        accelerator.wait_for_everyone()
        accelerator_state_dir = temp_dir / "accelerator_state"
        accelerator_state_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(str(accelerator_state_dir))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            validate_training_progress(progress, gradient_accumulation_steps=args.gradient_accumulation_steps, dataloader_length=len(loader))
            raw = accelerator.unwrap_model(model)
            raw_reg = accelerator.unwrap_model(model_reg) if model_reg is not None else None
            payload = checkpoint_payload(
                raw, progress.global_step, args, None, None, raw_reg, accelerator,
                optimizer_group_manifest=current_manifest,
                completed_epochs=progress.current_epoch,
                batch_position=progress.batches_consumed_in_current_epoch,
                dataloader_position=progress.batches_consumed_in_current_epoch,
                sampler_epoch=progress.current_epoch,
            )
            trainer_state = {
                "trainer_state_version": 4,
                **progress.to_trainer_state_fields(),
                "sampler_seed": args.seed,
                "dataset_length": len(dataset),
                "dataloader_length": len(loader),
                "world_size": accelerator.num_processes,
                "process_count": accelerator.num_processes,
                "train_batch_size": args.train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "sync_with_dataloader": args.sync_with_dataloader,
                "drop_last": False,
                "resume_config": build_resume_config_snapshot(args, model=raw, accelerator=accelerator, dataset_length=len(dataset)),
            }
            torch.save(payload, temp_dir / "model_state.pt")
            write_json_atomically(temp_dir / "trainer_state.json", trainer_state)
            write_json_atomically(temp_dir / "optimizer_manifest.json", current_manifest)
            finalize_checkpoint_directory(temp_dir, checkpoint_dir, payload, trainer_state, current_manifest)
        accelerator.wait_for_everyone()

    run_training_loop(
        accelerator=accelerator,
        model=model,
        train_dataloader=loader,
        train_sampler=train_sampler,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_train_steps=args.max_train_steps,
        progress=progress,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        compute_loss_fn=compute_loss_callback,
        checkpoint_fn=checkpoint_callback,
        validation_fn=validation_callback,
        logging_fn=logging_callback,
        checkpointing_steps=args.checkpointing_steps,
        validation_steps=args.validation_steps,
        logging_steps=args.logging_steps,
        max_grad_norm=args.max_grad_norm,
    )


if __name__ == "__main__": main(parse_args())
