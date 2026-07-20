import json

import numpy as np
import pytest
from PIL import Image

from dataloaders.focus_fusion_dataset import FocusFusionDataset, focus_fusion_collate


def _img(root, name, size=(768, 512), mode="RGB", value=32):
    shape = (size[1], size[0], 3) if mode == "RGB" else (size[1], size[0])
    arr = np.full(shape, value % 256, np.uint8)
    Image.fromarray(arr, mode).save(root / name)
    return name


def _meta(root, records):
    p = root / "metadata.json"
    p.write_text(json.dumps(records), encoding="utf-8")
    return p


def _record(root, ncond, explicit=False, focus=False, size=(768, 512)):
    gt = _img(root, "gt.png", size, value=1)
    cond = [_img(root, f"c{i}.png", size, value=10 + i) for i in range(ncond)]
    rec = {"image": gt, "prompt": "p"}
    if explicit:
        rec["condition_images"] = cond
    else:
        rec["edit_image"] = list(cond)
    if focus:
        fmaps = [_img(root, f"f{i}.png", size, "L", 100 + i) for i in range(2)]
        if explicit:
            rec["focus_maps"] = fmaps
        else:
            rec["edit_image"] += fmaps
    return rec


@pytest.mark.parametrize("mode,ncond", [("single", 1), ("dual", 2), ("quad_rgb", 4)])
@pytest.mark.parametrize("explicit", [False, True])
def test_rgb_modes_fields_and_native_size(tmp_path, mode, ncond, explicit):
    meta = _meta(tmp_path, [_record(tmp_path, ncond, explicit=explicit, size=(768, 512))])
    ds = FocusFusionDataset(meta, tmp_path, input_mode=mode, vae_scale_factor=8)
    item = ds[0]
    assert len(item["conditions"]) == ncond
    assert item["gt"].shape[-2:] == (512, 768)
    assert item["conditions"][0].shape[-2:] == (512, 768)


@pytest.mark.parametrize("explicit", [False, True])
def test_ab_focus_fields_and_single_channel_focus(tmp_path, explicit):
    meta = _meta(tmp_path, [_record(tmp_path, 2, explicit=explicit, focus=True, size=(960, 640))])
    item = FocusFusionDataset(meta, tmp_path, input_mode="ab_focus", vae_scale_factor=8)[0]
    assert len(item["conditions"]) == 2 and len(item["focus_maps"]) == 2
    assert item["focus_maps"][0].shape == (1, 640, 960)


def test_absolute_paths(tmp_path):
    rec = _record(tmp_path, 1, explicit=True)
    rec["image"] = str((tmp_path / rec["image"]).resolve())
    rec["condition_images"] = [str((tmp_path / p).resolve()) for p in rec["condition_images"]]
    item = FocusFusionDataset(_meta(tmp_path, [rec]), tmp_path, input_mode="single", vae_scale_factor=8)[0]
    assert item["gt"].shape[-2:] == (512, 768)


def test_size_mismatch_reports_all_sizes(tmp_path):
    rec = _record(tmp_path, 2, explicit=True)
    rec["condition_images"][1] = _img(tmp_path, "bad.png", (640, 512), value=9)
    with pytest.raises(ValueError, match="size mismatch"):
        FocusFusionDataset(_meta(tmp_path, [rec]), tmp_path, input_mode="dual", vae_scale_factor=8)[0]


def test_vae_scale_factor_error_and_max_pixels(tmp_path):
    rec = _record(tmp_path, 1, explicit=True, size=(770, 512))
    with pytest.raises(ValueError, match="Native-resolution mode forbids"):
        FocusFusionDataset(_meta(tmp_path, [rec]), tmp_path, input_mode="single", vae_scale_factor=8)[0]
    rec2 = _record(tmp_path, 1, explicit=True, size=(768, 512))
    with pytest.raises(ValueError, match="rejected rather than resized"):
        FocusFusionDataset(_meta(tmp_path, [rec2]), tmp_path, input_mode="single", vae_scale_factor=8, max_pixels=10)[0]


def test_mixed_batch_sizes_error(tmp_path):
    rec1 = _record(tmp_path, 1, explicit=True, size=(768, 512))
    rec2 = _record(tmp_path, 1, explicit=True, size=(960, 640))
    ds = FocusFusionDataset(_meta(tmp_path, [rec1, rec2]), tmp_path, input_mode="single", vae_scale_factor=8)
    with pytest.raises(ValueError, match="padding is forbidden"):
        focus_fusion_collate([ds[0], ds[1]])


def test_explicit_conflict_errors(tmp_path):
    rec = _record(tmp_path, 2, explicit=False)
    rec["condition_images"] = [rec["edit_image"][0], "different.png"]
    _img(tmp_path, "different.png")
    with pytest.raises(ValueError, match="conflicts"):
        FocusFusionDataset(_meta(tmp_path, [rec]), tmp_path, input_mode="dual")[0]
