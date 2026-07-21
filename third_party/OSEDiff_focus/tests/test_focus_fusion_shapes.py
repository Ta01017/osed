from types import SimpleNamespace

import torch
import pytest

from osediff_focus_fusion import (
    FocusFusionGenerator,
    INPUT_MODE_TO_CHANNELS,
    build_generator_unet_input,
    encode_rgb_conditions,
    expand_unet_conv_in,
    tiled_unet_forward,
    normalize_input_mode,
)


class DummyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(4, 8, 3, padding=1)
        self.conv_out = torch.nn.Conv2d(8, 4, 1)
        self.config = SimpleNamespace(in_channels=4, out_channels=4)

    def register_to_config(self, **kw):
        for k, v in kw.items():
            setattr(self.config, k, v)

    def forward(self, x, timestep, encoder_hidden_states=None):
        return SimpleNamespace(sample=self.conv_out(torch.relu(self.conv_in(x))))


class DummyVAE:
    config = SimpleNamespace(scaling_factor=1.0, block_out_channels=[1, 2, 3, 4])

    def encode(self, x):
        class Dist:
            def __init__(self, y):
                self.y = y[:, :1].repeat(1, 4, 1, 1)

            def sample(self):
                return self.y

            def mode(self):
                return self.y

        return SimpleNamespace(latent_dist=Dist(x))

    def decode(self, x):
        return SimpleNamespace(sample=x[:, :3])


def test_input_mode_channels():
    assert INPUT_MODE_TO_CHANNELS == {"single": 4, "dual": 8, "ab_focus": 10, "quad_rgb": 16}
    assert normalize_input_mode("ab") == "dual"
    assert normalize_input_mode("four") == "quad_rgb"


def test_conv_in_expand_rules():
    base = DummyUNet()
    old_layer = base.conv_in
    same = expand_unet_conv_in(base, 4)
    assert same.conv_in is old_layer
    for channels in (8, 10, 16):
        u = DummyUNet()
        weight = u.conv_in.weight.detach().clone()
        bias = u.conv_in.bias.detach().clone()
        expand_unet_conv_in(u, channels)
        assert u.conv_in.in_channels == u.config.in_channels == channels
        torch.testing.assert_close(u.conv_in.weight[:, :4], weight, rtol=0, atol=0)
        assert torch.count_nonzero(u.conv_in.weight[:, 4:]) == 0
        torch.testing.assert_close(u.conv_in.bias, bias, rtol=0, atol=0)


def test_build_inputs_all_modes_and_scheduler_sample_is_four():
    z = [torch.randn(1, 4, 8, 12) + i for i in range(4)]
    assert build_generator_unet_input("single", z[:1]).shape == (1, 4, 8, 12)
    assert build_generator_unet_input("dual", z[:2]).shape == (1, 8, 8, 12)
    fa, fb = torch.rand(1, 1, 64, 96), torch.rand(1, 1, 64, 96)
    assert build_generator_unet_input("ab_focus", z[:2], fa, fb).shape == (1, 10, 8, 12)
    assert build_generator_unet_input("quad_rgb", z).shape == (1, 16, 8, 12)
    pred = torch.randn_like(z[0])
    class Scheduler:
        def step(self, model_pred, timestep, sample, return_dict=True):
            assert model_pred.shape[1] == sample.shape[1] == 4
            return SimpleNamespace(prev_sample=sample - model_pred)
    assert Scheduler().step(pred, None, z[0]).prev_sample.shape[1] == 4


@pytest.mark.parametrize(
    ("mode", "count", "has_focus", "channels"),
    [("single", 1, False, 4), ("dual", 2, False, 8), ("ab_focus", 2, True, 10), ("quad_rgb", 4, False, 16)],
)
def test_make_unet_input_public_api_all_modes(mode, count, has_focus, channels):
    unet = expand_unet_conv_in(DummyUNet(), channels)
    model = FocusFusionGenerator(unet, DummyVAE(), SimpleNamespace(), mode)
    latents = [torch.randn(1, 4, 8, 8) for _ in range(count)]
    focus_a = torch.rand(1, 1, 64, 64) if has_focus else None
    focus_b = torch.rand(1, 1, 64, 64) if has_focus else None
    assert model.make_unet_input(latents, focus_a, focus_b).shape == (1, channels, 8, 8)


def test_make_unet_input_error_reports_mode_count_and_focus():
    unet = expand_unet_conv_in(DummyUNet(), 10)
    model = FocusFusionGenerator(unet, DummyVAE(), SimpleNamespace(), "ab_focus")
    with pytest.raises(ValueError, match="input_mode=ab_focus.*expected latent count=2.*actual latent count=1.*focus_a_present=False"):
        model.make_unet_input([torch.randn(1, 4, 8, 8)])


def test_multi_image_vae_order():
    images = [torch.full((1, 3, 4, 4), float(i)) for i in range(1, 5)]
    latents = encode_rgb_conditions(images, DummyVAE(), "mode")
    assert [float(x[0, 0, 0, 0]) for x in latents] == [1.0, 2.0, 3.0, 4.0]


def test_tiled_gaussian_shape_all_channel_counts():
    for channels in (4, 8, 10, 16):
        u = expand_unet_conv_in(DummyUNet(), channels)
        x = torch.randn(1, channels, 19, 23)
        y = tiled_unet_forward(u, x, torch.tensor([999]), torch.empty(1, 1, 1), 4, 8, 3)
        assert y.shape == (1, 4, 19, 23)


class ConstantUNet(torch.nn.Module):
    config = SimpleNamespace(out_channels=4)

    def __init__(self):
        super().__init__()
        self.tile_shapes = []

    def forward(self, x, timestep, encoder_hidden_states=None):
        self.tile_shapes.append(tuple(x.shape[-2:]))
        return SimpleNamespace(sample=torch.ones(x.shape[0], 4, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype))


@pytest.mark.parametrize("overlap", [0, 8])
def test_tiled_constant_prediction_preserved(overlap):
    u = ConstantUNet()
    x = torch.randn(1, 10, 65, 91)
    y = tiled_unet_forward(u, x, torch.tensor([999]), torch.empty(1, 1, 1), 4, 32, overlap)
    torch.testing.assert_close(y, torch.ones_like(y), atol=1e-5, rtol=1e-5)
    assert (1, 27) in u.tile_shapes or (17, 32) in u.tile_shapes


@pytest.mark.parametrize("bad_overlap", [-1, 32, 33])
def test_tiled_invalid_overlap_errors(bad_overlap):
    with pytest.raises(ValueError):
        tiled_unet_forward(ConstantUNet(), torch.randn(1, 4, 65, 91), torch.tensor([999]), torch.empty(1, 1, 1), 4, 32, bad_overlap)


def test_vsd_unet_stays_four_channel():
    u = DummyUNet()
    assert u.config.in_channels == 4


def test_decode_latents_requires_expected_output_hw_and_checks_size():
    model = FocusFusionGenerator(DummyUNet(), DummyVAE(), SimpleNamespace(), "single")
    latents = torch.randn(1, 4, 8, 8)
    assert model.decode_latents(latents, expected_output_hw=(8, 8)).shape == (1, 3, 8, 8)
    with pytest.raises(RuntimeError, match="decoded output size mismatch"):
        model.decode_latents(latents, expected_output_hw=(64, 64))
    with pytest.raises(TypeError):
        model.decode_latents(latents)


def test_forward_from_latents_requires_expected_output_hw():
    model = FocusFusionGenerator(DummyUNet(), DummyVAE(), SimpleNamespace(step=lambda *a, **k: SimpleNamespace(prev_sample=a[2])), "single")
    z = [torch.randn(1, 4, 8, 8)]
    with pytest.raises(TypeError):
        model.forward_from_latents(z, prompt_embeds=torch.empty(1, 1, 1))
