# OSEDiff one-step focus fusion

This extension adds a dual-image generator without modifying the official OSEDiff files. JSON fields are interpreted strictly as `image=GT` and `edit_image=[A, B_warp, focus_A, focus_B_warp]`. RGB tensors use `[-1,1]`; one-channel focus tensors use `[0,1]`. One shared resize/crop/flip decision is applied to all five images.

## Architecture

`ab` concatenates `[z_A,z_B]` (8 channels). `ab_focus` additionally concatenates bilinearly resized focus maps (10 channels). Only the generator `conv_in` is expanded. Its original four channel weights and bias are copied exactly and additional weights start at zero. The UNet always predicts four channels, `scheduler.step` always receives four-channel `z_A`, and the VAE decodes a four-channel denoised latent to RGB. Tiled inference tiles the complete condition tensor but allocates four-channel output accumulation.

The fixed prompt is cached in the checkpoint as a model buffer. Metadata prompts are supported. Fixed mode does not load RAM/DAPE. `vae_encode_mode=mode` is the deterministic inference default.

## Commands

Tiny-16:

```bash
PRETRAINED_MODEL=/path/to/sd21 bash scripts/0720_train_focus_fusion_tiny16.sh
```

Four-GPU full training:

```bash
PRETRAINED_MODEL=/path/to/sd21 GPUS=0,1,2,3 bash scripts/0720_train_focus_fusion_full.sh
```

Inference and all B ablations:

```bash
CHECKPOINT=/path/to/focus_fusion_2000.pt PRETRAINED_MODEL=/path/to/sd21 bash scripts/0720_infer_focus_fusion.sh
```

Each inference sample saves raw prediction, optional keep-A composite, all five inputs/GT, and a horizontal comparison. Ablations are isolated under `normal`, `b_equals_a`, and `b_zero`; MAE statistics warn if B appears unused.

## Checkpoint format

The `.pt` dictionary contains `format_version`, `condition_mode`, `generator_in_channels`, strict full `generator_conv_in` state, generator UNet LoRA, optional VAE LoRA, fixed prompt and cached embedding, `training_step`, all CLI args, optimizer state, and scheduler state. Restore order is base SD2.1 UNet, conv expansion, LoRA adapter creation, strict conv load, LoRA load, then optimizer/scheduler state. A conv mismatch is fatal and is never hidden by `strict=False`.

## Losses

Training combines L2, LPIPS, keep-A masked L1, B-reference masked GT L1, gradient, and Laplacian losses. Masked terms divide by effective mask pixels and channels. The masks are `keep=focus_A` and `bref=(1-focus_A)*focus_B_warp`.
