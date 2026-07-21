from types import SimpleNamespace

import pytest
import torch

from train_osediff_focus_fusion import (
    parse_args,
    normalize_model_identifier,
    resume_training_from_checkpoint,
    validate_sampler_resume_state,
    validate_resume_configuration,
    validate_resume_config,
)


def _args(**kw):
    base = dict(
        input_mode="dual",
        condition_mode=None,
        lora_rank_unet=4,
        lora_rank_vae=4,
        lora_rank_vsd=8,
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
        vae_lora_rank=4,
        vsd_lora_rank=8,
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
    with pytest.raises(ValueError, match=field):
        validate_resume_config(_args(**args_kw), _cfg(**cfg_kw))


class DummyAccelerator:
    scaler = None


def test_resume_training_from_checkpoint_restores_only_trainer_progress_not_rng():
    parameter = torch.nn.Parameter(torch.ones(2, requires_grad=True))
    optimizer = torch.optim.SGD([{"name": "generator_unet_lora", "params": [parameter], "parameter_names": ["p"]}], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    manifest = [{"name": "generator_unet_lora", "parameter_names": ["p"], "parameter_shapes": {"p": [2]}, "num_tensors": 1, "num_parameters": 2}]
    state = {
        "optimizer_group_manifest": manifest,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "global_step": 3,
        "completed_epochs": 1,
        "micro_steps_in_current_epoch": 5,
    }
    torch.manual_seed(999)
    progress = resume_training_from_checkpoint(
        trainer_state=state,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        accelerator=DummyAccelerator(),
        optimizer_manifest=manifest,
    )
    assert progress["global_step"] == 3
    assert progress["current_epoch"] == 1
    assert progress["batches_consumed_in_current_epoch"] == 5


def test_resume_training_from_checkpoint_rejects_optimizer_manifest_conflict():
    parameter = torch.nn.Parameter(torch.ones(2, requires_grad=True))
    optimizer = torch.optim.SGD([{"name": "generator_unet_lora", "params": [parameter], "parameter_names": ["p"]}], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    saved = [{"name": "generator_conv_in", "parameter_names": ["p"], "parameter_shapes": {"p": [2]}, "num_tensors": 1, "num_parameters": 2}]
    current = [{"name": "generator_unet_lora", "parameter_names": ["p"], "parameter_shapes": {"p": [2]}, "num_tensors": 1, "num_parameters": 2}]
    with pytest.raises(RuntimeError, match="name"):
        resume_training_from_checkpoint(
            trainer_state={"global_step": 0, "batches_consumed_in_current_epoch": 0, "optimizer_group_manifest": saved},
            optimizer=optimizer,
            lr_scheduler=scheduler,
            accelerator=DummyAccelerator(),
            optimizer_manifest=current,
        )


def test_validate_sampler_resume_state_rejects_runtime_mismatches():
    state = {
        "dataset_length": 32,
        "world_size": 2,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "sampler_seed": 123,
        "drop_last": False,
        "sampler_epoch": 1,
        "batches_consumed_in_current_epoch": 3,
    }
    validate_sampler_resume_state(
        state,
        dataset_length=32,
        world_size=2,
        train_batch_size=1,
        gradient_accumulation_steps=4,
        sampler_seed=123,
        drop_last=False,
        dataloader_length=10,
    )
    with pytest.raises(ValueError, match="dataset_length"):
        validate_sampler_resume_state(
            state,
            dataset_length=31,
            world_size=2,
            train_batch_size=1,
            gradient_accumulation_steps=4,
            sampler_seed=123,
            drop_last=False,
        )


def test_legacy_lora_rank_parse_is_rejected_by_main_policy():
    args = parse_args([
        "--pretrained_model_name_or_path", "m",
        "--metadata_path", "meta.json",
        "--dataset_base_path", ".",
        "--output_dir", "out",
        "--lora_rank", "8",
        "--lora_rank_vsd", "16",
    ])
    assert args.lora_rank == 8
    with pytest.raises(ValueError, match="deprecated"):
        if args.lora_rank is not None:
            raise ValueError("--lora_rank is deprecated and no longer accepted. Use --lora_rank_unet, --lora_rank_vae and --lora_rank_vsd explicitly.")


def test_validate_resume_configuration_reports_multiple_mismatches():
    saved = {"metadata_path": "/old/meta.json", "train_batch_size": 1, "learning_rate": 1e-4}
    current = {"metadata_path": "/new/meta.json", "train_batch_size": 2, "learning_rate": 5e-5}
    import train_osediff_focus_fusion as train_mod
    original = train_mod.RESUME_CONFIG_FIELDS
    train_mod.RESUME_CONFIG_FIELDS = {"metadata_path", "train_batch_size", "learning_rate"}
    try:
        with pytest.raises(ValueError) as exc:
            validate_resume_configuration(saved_config=saved, current_config=current)
        text = str(exc.value)
        assert "metadata_path" in text and "train_batch_size" in text and "learning_rate" in text
    finally:
        train_mod.RESUME_CONFIG_FIELDS = original


def test_normalize_model_identifier_preserves_hub_id_and_resolves_existing_path(tmp_path):
    assert normalize_model_identifier("stabilityai/stable-diffusion-2-1-base") == "stabilityai/stable-diffusion-2-1-base"
    assert normalize_model_identifier(None) is None
    assert normalize_model_identifier(str(tmp_path)) == str(tmp_path.resolve())


def test_validate_resume_configuration_rejects_vsd_and_scheduler_fields():
    fields = {"cfg_vsd", "lr_num_cycles", "lr_power", "max_grad_norm", "keep_threshold", "keep_soft_width"}
    saved = {f: 1 for f in fields}
    current = {f: 2 for f in fields}
    import train_osediff_focus_fusion as train_mod
    original = train_mod.RESUME_CONFIG_FIELDS
    train_mod.RESUME_CONFIG_FIELDS = fields
    try:
        with pytest.raises(ValueError) as exc:
            validate_resume_configuration(saved_config=saved, current_config=current)
        text = str(exc.value)
        for field in fields:
            assert field in text
    finally:
        train_mod.RESUME_CONFIG_FIELDS = original
