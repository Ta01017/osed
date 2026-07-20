from types import SimpleNamespace

import pytest
import torch

from osediff_focus_fusion import (
    checkpoint_payload,
    expand_unet_conv_in,
    load_focus_checkpoint,
    load_vae_lora_state,
    load_vsd_lora_state,
    read_focus_checkpoint_config,
)


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


def test_checkpoint_round_trip_generator_vae_vsd_lora():
    model = TinyModel()
    vsd = TinyVSD()
    with torch.no_grad():
        model.unet.adapter.lora_weight.fill_(3)
        model.vae.adapter.lora_weight.fill_(4)
        vsd.unet_update.adapter.lora_weight.fill_(5)
    state = checkpoint_payload(model, 7, _args(), vsd=vsd)
    cfg = read_focus_checkpoint_config(state)
    assert cfg["input_mode"] == "dual" and cfg["global_step"] == 7

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
