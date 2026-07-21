from types import SimpleNamespace

import pytest
import torch

from osediff_focus_fusion import capture_rng_state
from train_osediff_focus_fusion import (
    resume_training_from_checkpoint,
    validate_resume_config,
)


def _args(**kw):
    base = dict(
        input_mode="dual",
        condition_mode=None,
        lora_rank_unet=4,
        train_conv_in=True,
        train_vae_lora=False,
        use_vsd=0,
        prompt_mode="fixed",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _cfg(**kw):
    base = dict(
        input_mode="dual",
        generator_lora_rank=4,
        train_conv_in=True,
        train_vae_lora=False,
        use_vsd=False,
        prompt_mode="fixed",
    )
    base.update(kw)
    return base


def test_validate_resume_config_accepts_matching_structure():
    args = validate_resume_config(_args(), _cfg())
    assert args.input_mode == "dual"
    assert args.use_vsd == 0


@pytest.mark.parametrize(
    ("field", "args_kw", "cfg_kw"),
    [
        ("use_vsd", {"use_vsd": 0}, {"use_vsd": True}),
        ("train_vae_lora", {"train_vae_lora": False}, {"train_vae_lora": True}),
        ("generator_lora_rank", {"lora_rank_unet": 4}, {"generator_lora_rank": 8}),
        ("input_mode", {"input_mode": "single"}, {"input_mode": "dual"}),
        ("prompt_mode", {"prompt_mode": "metadata"}, {"prompt_mode": "fixed"}),
    ],
)
def test_validate_resume_config_rejects_structural_conflicts(field, args_kw, cfg_kw):
    with pytest.raises(RuntimeError, match=field):
        validate_resume_config(_args(**args_kw), _cfg(**cfg_kw))


class DummyAccelerator:
    scaler = None


def test_resume_training_from_checkpoint_restores_progress_and_rng():
    parameter = torch.nn.Parameter(torch.ones(2, requires_grad=True))
    optimizer = torch.optim.SGD([{"name": "generator_unet_lora", "params": [parameter], "parameter_names": ["p"]}], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    manifest = [{"name": "generator_unet_lora", "parameter_names": ["p"], "parameter_shapes": {"p": [2]}, "num_tensors": 1, "num_parameters": 2}]
    torch.manual_seed(11)
    rng = capture_rng_state()
    expected_next = torch.rand(2)
    state = {
        "optimizer_group_manifest": manifest,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "rng_state": rng,
        "global_step": 3,
        "completed_epochs": 1,
        "micro_steps_in_current_epoch": 5,
    }
    torch.manual_seed(999)
    progress = resume_training_from_checkpoint(
        resume_state=state,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        accelerator=DummyAccelerator(),
        optimizer_manifest=manifest,
    )
    assert progress == {"global_step": 3, "completed_epochs": 1, "micro_steps_in_current_epoch": 5}
    torch.testing.assert_close(torch.rand(2), expected_next)


def test_resume_training_from_checkpoint_rejects_optimizer_manifest_conflict():
    parameter = torch.nn.Parameter(torch.ones(2, requires_grad=True))
    optimizer = torch.optim.SGD([{"name": "generator_unet_lora", "params": [parameter], "parameter_names": ["p"]}], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    saved = [{"name": "generator_conv_in", "parameter_names": ["p"], "parameter_shapes": {"p": [2]}, "num_tensors": 1, "num_parameters": 2}]
    current = [{"name": "generator_unet_lora", "parameter_names": ["p"], "parameter_shapes": {"p": [2]}, "num_tensors": 1, "num_parameters": 2}]
    with pytest.raises(RuntimeError, match="name"):
        resume_training_from_checkpoint(
            resume_state={"optimizer_group_manifest": saved},
            optimizer=optimizer,
            lr_scheduler=scheduler,
            accelerator=DummyAccelerator(),
            optimizer_manifest=current,
        )
