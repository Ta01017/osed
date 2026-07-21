from types import SimpleNamespace

import pytest
import torch

from train_osediff_focus_fusion import (
    assert_cli_argument_classification,
    get_parser_argument_dests,
    parse_args,
    normalize_model_identifier,
    TrainingProgress,
    log_accelerator_resume_success,
    run_from_args,
    validate_sampler_resume_state,
    validate_resume_configuration,
    validate_resume_config,
)
from torch.utils.data import DataLoader, TensorDataset

from test_focus_fusion_training_steps import TailSyncAccelerator, Tiny, CountingScheduler


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
        sync_with_dataloader=True,
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


def test_training_progress_from_trainer_state_is_unique_resume_source():
    state = {
        "global_step": 3,
        "current_epoch": 1,
        "completed_epochs": 1,
        "batches_consumed_in_current_epoch": 5,
        "micro_batches": 12,
        "optimizer_updates": 3,
        "scheduler_steps": 3,
        "sampler_epoch": 1,
    }
    log_accelerator_resume_success(trainer_state=state)
    progress = TrainingProgress.from_trainer_state(state)
    assert progress.global_step == 3
    assert progress.micro_batches == 12


def test_training_progress_missing_field_errors():
    with pytest.raises(ValueError, match="optimizer_updates"):
        TrainingProgress.from_trainer_state({
            "global_step": 1, "current_epoch": 0, "completed_epochs": 0,
            "batches_consumed_in_current_epoch": 4, "micro_batches": 4,
            "scheduler_steps": 1, "sampler_epoch": 0,
        })


def test_validate_sampler_resume_state_rejects_runtime_mismatches():
    state = {
        "dataset_length": 32,
        "world_size": 2,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "sync_with_dataloader": True,
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
            sync_with_dataloader=True,
            sampler_seed=123,
            drop_last=False,
        )


def test_legacy_lora_rank_parse_is_rejected_by_main_policy():
    with pytest.raises(ValueError, match="deprecated"):
        parse_args([
            "--pretrained_model_name_or_path", "m",
            "--metadata_path", "meta.json",
            "--dataset_base_path", ".",
            "--output_dir", "out",
            "--lora_rank", "8",
            "--lora_rank_vsd", "16",
        ])


def test_parse_args_classifies_all_current_cli_fields():
    args = parse_args([
        "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-2-1-base",
        "--metadata_path", "meta.json",
        "--dataset_base_path", ".",
        "--output_dir", "out",
        "--condition_mode", "ab_focus",
        "--keep_a_composite",
        "--keep_threshold", "0.4",
        "--keep_soft_width", "0.2",
        "--lora_rank_unet", "4",
        "--lora_rank_vae", "2",
        "--lora_rank_vsd", "6",
        "--lr_num_cycles", "2",
        "--lr_power", "0.5",
        "--max_grad_norm", "0.7",
        "--cfg_vsd", "5.5",
        "--sync_with_dataloader",
    ])
    assert args.condition_mode == "ab_focus"
    assert args.keep_a_composite is True
    assert args.lora_rank_unet == 4
    assert args.sync_with_dataloader is True


def test_parser_classification_helpers_reject_unclassified_dest():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--surprise_training_flag")
    assert get_parser_argument_dests(parser) == {"surprise_training_flag"}
    with pytest.raises(RuntimeError, match="unclassified CLI arguments"):
        assert_cli_argument_classification(parser)


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


def test_resume_configuration_tracks_sync_with_dataloader():
    saved = {"sync_with_dataloader": True, "gradient_accumulation_steps": 4}
    current = {"sync_with_dataloader": False, "gradient_accumulation_steps": 4}
    import train_osediff_focus_fusion as train_mod
    original = train_mod.RESUME_CONFIG_FIELDS
    train_mod.RESUME_CONFIG_FIELDS = {"sync_with_dataloader", "gradient_accumulation_steps"}
    try:
        with pytest.raises(ValueError, match="sync_with_dataloader"):
            validate_resume_configuration(saved_config=saved, current_config=current)
    finally:
        train_mod.RESUME_CONFIG_FIELDS = original


def test_validate_sampler_resume_state_accepts_partial_accumulation():
    state = {
        "trainer_state_version": 4,
        "dataset_length": 6,
        "world_size": 1,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "sync_with_dataloader": True,
        "sampler_seed": 123,
        "drop_last": False,
        "current_epoch": 1,
        "completed_epochs": 1,
        "sampler_epoch": 1,
        "batches_consumed_in_current_epoch": 0,
        "global_step": 2,
        "optimizer_updates": 2,
        "scheduler_steps": 2,
        "micro_batches": 6,
    }
    validate_sampler_resume_state(
        state,
        dataset_length=6,
        world_size=1,
        train_batch_size=1,
        gradient_accumulation_steps=4,
        sync_with_dataloader=True,
        sampler_seed=123,
        drop_last=False,
        dataloader_length=6,
    )
    state["micro_batches"] = 1
    with pytest.raises(ValueError, match="micro_batches"):
        validate_sampler_resume_state(
            state,
            dataset_length=6,
            world_size=1,
            train_batch_size=1,
            gradient_accumulation_steps=4,
            sync_with_dataloader=True,
            sampler_seed=123,
            drop_last=False,
            dataloader_length=6,
        )


def test_partial_state_enters_formal_run_from_args_and_updates():
    loader = DataLoader(TensorDataset(torch.ones(6, 1)), batch_size=1)
    model = Tiny()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = CountingScheduler()
    args = SimpleNamespace(
        gradient_accumulation_steps=4,
        max_train_steps=3,
        checkpointing_steps=0,
        validation_steps=0,
        logging_steps=10,
        max_grad_norm=None,
    )
    progress = run_from_args(args, dependencies={
        "accelerator": TailSyncAccelerator(4, len(loader)),
        "model": model,
        "train_dataloader": loader,
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "compute_loss_fn": lambda m, b: m(b),
        "trainer_state": {
            "global_step": 2,
            "current_epoch": 1,
            "completed_epochs": 1,
            "batches_consumed_in_current_epoch": 0,
            "micro_batches": 6,
            "optimizer_updates": 2,
            "scheduler_steps": 2,
            "sampler_epoch": 1,
        },
    })
    assert progress.global_step == 3
    assert progress.optimizer_updates == 3
