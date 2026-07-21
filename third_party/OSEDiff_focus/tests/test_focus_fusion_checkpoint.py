from types import SimpleNamespace

import pytest
import torch

from osediff_focus_fusion import (
    capture_rng_state,
    checkpoint_payload,
    checkpoint_complete_path,
    load_verified_checkpoint,
    prepare_checkpoint_temp_dir,
    finalize_checkpoint_directory,
    expand_unet_conv_in,
    load_focus_checkpoint,
    load_vae_lora_state,
    load_vsd_lora_state,
    read_focus_checkpoint_config,
    restore_rng_state,
    write_checkpoint_complete_manifest,
    write_json_atomically,
)
from train_osediff_focus_fusion import broadcast_checkpoint_temp_dir
from train_osediff_focus_fusion import validate_optimizer_manifest


class LoRAModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.zeros(2, 2))


class TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.config = SimpleNamespace(in_channels=4, out_channels=4)
        self.adapter = LoRAModule()

    def register_to_config(self, **kw):
        for k, v in kw.items():
            setattr(self.config, k, v)


class TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=1.0, block_out_channels=[1, 2, 3, 4])
        self.adapter = LoRAModule()


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = expand_unet_conv_in(TinyUNet(), 8)
        self.vae = TinyVAE()
        self.condition_mode = "dual"
        self.input_mode = "dual"
        self.focus_lora_targets = ["adapter"]
        self.focus_vae_lora_targets = ["adapter"]
        self.lora_rank_unet = 2
        self.lora_rank_vae = 2
        self.register_buffer("fixed_prompt_embedding", torch.ones(1, 2, 3))

    def cache_prompt(self, emb):
        self.fixed_prompt_embedding = emb.clone()


class TinyVSD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet_update = TinyUNet()
        self.lora_unet_modules_encoder = ["adapter"]
        self.lora_unet_modules_decoder = []
        self.lora_unet_others = []


def _args(**kw):
    base = dict(input_mode="dual", train_conv_in=True, train_vae_lora=True, use_vsd=True,
                lora_rank_unet=2, lora_rank=2, prompt_mode="fixed", cache_fixed_prompt_embedding=True,
                gradient_accumulation_steps=1)
    base.update(kw)
    return SimpleNamespace(**base)


def _trainer_state(step=1, accumulation=1, **kw):
    base = {
        "trainer_state_version": 4,
        "global_step": step,
        "current_epoch": 0,
        "completed_epochs": 0,
        "batches_consumed_in_current_epoch": step * accumulation,
        "micro_batches": step * accumulation,
        "optimizer_updates": step,
        "scheduler_steps": step,
        "sampler_epoch": 0,
        "gradient_accumulation_steps": accumulation,
        "sync_with_dataloader": True,
        "dataloader_length": 32,
    }
    base.update(kw)
    return base


def test_checkpoint_round_trip_generator_vae_vsd_lora():
    model = TinyModel()
    vsd = TinyVSD()
    with torch.no_grad():
        model.unet.adapter.lora_weight.fill_(3)
        model.vae.adapter.lora_weight.fill_(4)
        vsd.unet_update.adapter.lora_weight.fill_(5)
    state = checkpoint_payload(model, 7, _args(), vsd=vsd)
    cfg = read_focus_checkpoint_config(state)
    assert cfg["input_mode"] == "dual"
    assert "global_step" not in state

    restored = TinyModel()
    load_focus_checkpoint(restored, state)
    load_vae_lora_state(restored, state)
    restored_vsd = TinyVSD()
    load_vsd_lora_state(restored_vsd, state)
    torch.testing.assert_close(restored.unet.adapter.lora_weight, torch.full((2, 2), 3.0))
    torch.testing.assert_close(restored.vae.adapter.lora_weight, torch.full((2, 2), 4.0))
    torch.testing.assert_close(restored_vsd.unet_update.adapter.lora_weight, torch.full((2, 2), 5.0))
    torch.testing.assert_close(restored.fixed_prompt_embedding, torch.ones(1, 2, 3))


def test_checkpoint_input_mode_and_channels_errors():
    state = checkpoint_payload(TinyModel(), 1, _args())
    wrong = TinyModel()
    wrong.condition_mode = wrong.input_mode = "single"
    with pytest.raises(RuntimeError, match="input_mode mismatch"):
        load_focus_checkpoint(wrong, state)
    wrong2 = TinyModel()
    wrong2.unet = TinyUNet()
    with pytest.raises(RuntimeError, match="conv_in"):
        load_focus_checkpoint(wrong2, state)


def test_missing_conv_weight_and_missing_lora_states_error():
    state = checkpoint_payload(TinyModel(), 1, _args())
    del state["generator_conv_in"]["weight"]
    with pytest.raises(RuntimeError):
        load_focus_checkpoint(TinyModel(), state)
    state2 = checkpoint_payload(TinyModel(), 1, _args())
    state2["vae_lora"] = {}
    with pytest.raises(RuntimeError, match="VAE LoRA state is missing"):
        load_vae_lora_state(TinyModel(), state2)
    state3 = checkpoint_payload(TinyModel(), 1, _args(), vsd=TinyVSD())
    state3["vsd_unet_lora"] = {}
    with pytest.raises(RuntimeError, match="VSD LoRA state is missing"):
        load_vsd_lora_state(TinyVSD(), state3)


def test_rng_round_trip_restores_torch_sequence():
    torch.manual_seed(123)
    before = torch.rand(3)
    state = capture_rng_state()
    expected = torch.rand(3)
    torch.manual_seed(999)
    assert restore_rng_state(state)
    actual = torch.rand(3)
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(before, expected)


def test_checkpoint_complete_manifest_and_strict_file_load(tmp_path):
    model = TinyModel()
    payload = checkpoint_payload(model, 9, _args())
    checkpoint = tmp_path / "focus_fusion_9.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="complete manifest"):
        load_focus_checkpoint(TinyModel(), str(checkpoint))
    manifest = write_checkpoint_complete_manifest(checkpoint, payload)
    assert manifest["complete"] is True
    assert manifest["global_step"] == 9
    assert checkpoint_complete_path(checkpoint).endswith(".complete.json")
    restored = TinyModel()
    load_focus_checkpoint(restored, str(checkpoint))


def test_checkpoint_directory_atomic_finalize_preserves_accelerator_state(tmp_path):
    final_dir = tmp_path / "checkpoint-00000001"
    temp_dir = prepare_checkpoint_temp_dir(final_dir)
    (temp_dir / "accelerator_state").mkdir()
    write_json_atomically(temp_dir / "accelerator_state" / "rank0.json", {"ok": True})
    payload = checkpoint_payload(TinyModel(), 1, _args())
    trainer = _trainer_state(1)
    optim = [{"name": "generator_unet_lora", "parameter_names": ["p"], "parameter_shapes": {"p": [2]}, "num_tensors": 1, "num_parameters": 2}]
    torch.save(payload, temp_dir / "model_state.pt")
    write_json_atomically(temp_dir / "trainer_state.json", trainer)
    write_json_atomically(temp_dir / "optimizer_manifest.json", optim)
    manifest = finalize_checkpoint_directory(temp_dir, final_dir, payload, trainer, optim)
    assert manifest["accelerator_state_present"] is True
    assert final_dir.is_dir()
    assert not temp_dir.exists()
    assert (final_dir / "accelerator_state" / "rank0.json").is_file()
    verified = load_verified_checkpoint(final_dir)
    assert verified["trainer_state"]["global_step"] == 1
    with pytest.raises(FileExistsError):
        prepare_checkpoint_temp_dir(final_dir)


def test_verified_checkpoint_rejects_missing_accelerator_state(tmp_path):
    final_dir = tmp_path / "checkpoint-00000001"
    temp_dir = prepare_checkpoint_temp_dir(final_dir)
    (temp_dir / "accelerator_state").mkdir()
    write_json_atomically(temp_dir / "accelerator_state" / "rank0.json", {"ok": True})
    payload = checkpoint_payload(TinyModel(), 1, _args())
    trainer = _trainer_state(1)
    optim = []
    torch.save(payload, temp_dir / "model_state.pt")
    write_json_atomically(temp_dir / "trainer_state.json", trainer)
    write_json_atomically(temp_dir / "optimizer_manifest.json", optim)
    finalize_checkpoint_directory(temp_dir, final_dir, payload, trainer, optim)
    for child in (final_dir / "accelerator_state").iterdir():
        child.unlink()
    with pytest.raises(RuntimeError, match="accelerator_state"):
        load_verified_checkpoint(final_dir)


def test_verified_checkpoint_rejects_progress_fields_in_model_state(tmp_path):
    final_dir = tmp_path / "checkpoint-00000001"
    temp_dir = prepare_checkpoint_temp_dir(final_dir)
    (temp_dir / "accelerator_state").mkdir()
    write_json_atomically(temp_dir / "accelerator_state" / "rank0.json", {"ok": True})
    payload = checkpoint_payload(TinyModel(), 1, _args())
    payload["global_step"] = 999
    trainer = _trainer_state(1)
    optim = []
    torch.save(payload, temp_dir / "model_state.pt")
    write_json_atomically(temp_dir / "trainer_state.json", trainer)
    write_json_atomically(temp_dir / "optimizer_manifest.json", optim)
    finalize_checkpoint_directory(temp_dir, final_dir, payload, trainer, optim)
    with pytest.raises(RuntimeError, match="INVALID MODEL STATE"):
        load_verified_checkpoint(final_dir)


def test_verified_checkpoint_rejects_corrupt_trainer_progress(tmp_path):
    final_dir = tmp_path / "checkpoint-00000001"
    temp_dir = prepare_checkpoint_temp_dir(final_dir)
    (temp_dir / "accelerator_state").mkdir()
    write_json_atomically(temp_dir / "accelerator_state" / "rank0.json", {"ok": True})
    payload = checkpoint_payload(TinyModel(), 1, _args())
    trainer = _trainer_state(2, accumulation=4, micro_batches=1, optimizer_updates=2, scheduler_steps=2)
    optim = []
    torch.save(payload, temp_dir / "model_state.pt")
    write_json_atomically(temp_dir / "trainer_state.json", trainer)
    write_json_atomically(temp_dir / "optimizer_manifest.json", optim)
    finalize_checkpoint_directory(temp_dir, final_dir, payload, trainer, optim)
    with pytest.raises(RuntimeError, match="INVALID TRAINER STATE"):
        load_verified_checkpoint(final_dir)


def test_verified_checkpoint_accepts_partial_epoch_accumulation(tmp_path):
    final_dir = tmp_path / "checkpoint-00000002"
    temp_dir = prepare_checkpoint_temp_dir(final_dir)
    (temp_dir / "accelerator_state").mkdir()
    write_json_atomically(temp_dir / "accelerator_state" / "rank0.json", {"ok": True})
    payload = checkpoint_payload(TinyModel(), 2, _args())
    trainer = _trainer_state(
        2,
        accumulation=4,
        micro_batches=6,
        batches_consumed_in_current_epoch=0,
        current_epoch=1,
        completed_epochs=1,
        sampler_epoch=1,
        dataloader_length=6,
    )
    optim = []
    torch.save(payload, temp_dir / "model_state.pt")
    write_json_atomically(temp_dir / "trainer_state.json", trainer)
    write_json_atomically(temp_dir / "optimizer_manifest.json", optim)
    finalize_checkpoint_directory(temp_dir, final_dir, payload, trainer, optim)
    verified = load_verified_checkpoint(final_dir)
    assert verified["trainer_state"]["micro_batches"] == 6
    assert verified["trainer_state"]["global_step"] == 2


def test_verified_checkpoint_rejects_version4_unnormalized_epoch_state(tmp_path):
    final_dir = tmp_path / "checkpoint-00000002"
    temp_dir = prepare_checkpoint_temp_dir(final_dir)
    (temp_dir / "accelerator_state").mkdir()
    write_json_atomically(temp_dir / "accelerator_state" / "rank0.json", {"ok": True})
    payload = checkpoint_payload(TinyModel(), 2, _args())
    trainer = _trainer_state(
        2,
        accumulation=4,
        micro_batches=6,
        batches_consumed_in_current_epoch=6,
        dataloader_length=6,
    )
    optim = []
    torch.save(payload, temp_dir / "model_state.pt")
    write_json_atomically(temp_dir / "trainer_state.json", trainer)
    write_json_atomically(temp_dir / "optimizer_manifest.json", optim)
    finalize_checkpoint_directory(temp_dir, final_dir, payload, trainer, optim)
    with pytest.raises(RuntimeError, match="version 4"):
        load_verified_checkpoint(final_dir)


class SingleAccelerator:
    num_processes = 1
    is_main_process = True


def test_broadcast_checkpoint_temp_dir_single_rank_exact_path(tmp_path):
    final_dir = tmp_path / "checkpoint-00000007"
    temp_dir = prepare_checkpoint_temp_dir(final_dir)
    got = broadcast_checkpoint_temp_dir(SingleAccelerator(), temp_dir, final_dir)
    assert got == temp_dir


def test_optimizer_manifest_rejects_same_names_different_order():
    saved = [{"name": "generator_unet_lora", "parameter_names": ["a", "b"], "parameter_shapes": {"a": [1], "b": [2]}, "num_tensors": 2, "num_parameters": 3, "lr": 1e-4, "weight_decay": 0.01}]
    current = [{"name": "generator_unet_lora", "parameter_names": ["b", "a"], "parameter_shapes": {"a": [1], "b": [2]}, "num_tensors": 2, "num_parameters": 3, "lr": 1e-4, "weight_decay": 0.01}]
    with pytest.raises(RuntimeError, match="OPTIMIZER MANIFEST MISMATCH"):
        validate_optimizer_manifest(saved, current)


def test_optimizer_manifest_rejects_lr_and_weight_decay_changes():
    saved = [{"name": "generator_unet_lora", "parameter_names": ["a"], "parameter_shapes": {"a": [1]}, "num_tensors": 1, "num_parameters": 1, "lr": 1e-4, "weight_decay": 0.01}]
    changed_lr = [{"name": "generator_unet_lora", "parameter_names": ["a"], "parameter_shapes": {"a": [1]}, "num_tensors": 1, "num_parameters": 1, "lr": 2e-4, "weight_decay": 0.01}]
    with pytest.raises(RuntimeError, match="lr"):
        validate_optimizer_manifest(saved, changed_lr)
    changed_wd = [{"name": "generator_unet_lora", "parameter_names": ["a"], "parameter_shapes": {"a": [1]}, "num_tensors": 1, "num_parameters": 1, "lr": 1e-4, "weight_decay": 0.02}]
    with pytest.raises(RuntimeError, match="weight_decay"):
        validate_optimizer_manifest(saved, changed_wd)
