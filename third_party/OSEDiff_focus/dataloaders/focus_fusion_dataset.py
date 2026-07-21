"""Native-resolution JSON dataset for OSEDiff focus fusion."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


FIXED_FUSION_PROMPT = (
    "Fuse the focus-bracketed inputs into one all-in-focus image. Preserve the "
    "structure, viewpoint, color, and sharp details of Image A. Use reference "
    "images only to restore regions that are blurred in Image A."
)

INPUT_MODE_ALIASES = {"ab": "dual", "four": "quad_rgb"}
INPUT_MODE_COUNTS = {"single": 1, "dual": 2, "quad_rgb": 4, "ab_focus": 2}


def normalize_input_mode(input_mode):
    mode = INPUT_MODE_ALIASES.get(str(input_mode), str(input_mode))
    if mode not in INPUT_MODE_COUNTS:
        raise ValueError(f"unknown input_mode: {input_mode}")
    return mode


def _metadata_records(path):
    with open(path, "r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if isinstance(obj, dict):
        for key in ("data", "items", "annotations", "metadata"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break
    if not isinstance(obj, list):
        raise ValueError(f"metadata must be a JSON list (or contain one): {path}")
    return obj


def parse_metadata_inputs(item, input_mode, metadata_index):
    mode = normalize_input_mode(input_mode)

    def parse_explicit():
        conditions = item.get("condition_images")
        focus = item.get("focus_maps", [])
        if conditions is None:
            return None
        if not isinstance(conditions, list) or len(conditions) != INPUT_MODE_COUNTS[mode]:
            raise ValueError(f"metadata index {metadata_index}, input_mode {mode}: condition_images has wrong length")
        if mode == "ab_focus":
            if not isinstance(focus, list) or len(focus) != 2:
                raise ValueError(f"metadata index {metadata_index}, input_mode {mode}: focus_maps has wrong length")
        elif focus:
            raise ValueError(f"metadata index {metadata_index}, input_mode {mode}: focus_maps is only valid for ab_focus")
        return list(conditions), list(focus)

    def parse_edit():
        edits = item.get("edit_image")
        if edits is None:
            return None
        if not isinstance(edits, list):
            raise ValueError(f"metadata index {metadata_index}, input_mode {mode}: edit_image must be a list")
        expected = 4 if mode in ("quad_rgb", "ab_focus") else INPUT_MODE_COUNTS[mode]
        if len(edits) != expected:
            raise ValueError(f"metadata index {metadata_index}, input_mode {mode}: edit_image has wrong length {len(edits)}, expected {expected}")
        if mode == "ab_focus":
            return list(edits[:2]), list(edits[2:4])
        return list(edits[:INPUT_MODE_COUNTS[mode]]), []

    explicit = parse_explicit()
    legacy = parse_edit()
    if explicit and legacy:
        def normalized_pair(pair):
            return tuple(tuple(str(Path(x)) for x in values) for values in pair)
        if normalized_pair(explicit) != normalized_pair(legacy):
            raise ValueError(
                f"metadata index {metadata_index}, input_mode {mode}: condition_images/focus_maps conflict with edit_image; "
                f"condition_images={item.get('condition_images')}, focus_maps={item.get('focus_maps')}, edit_image={item.get('edit_image')}"
            )
    result = explicit or legacy
    if result is None:
        raise ValueError(f"metadata index {metadata_index}, input_mode {mode}: missing condition_images or edit_image")
    return result


class FocusFusionDataset(Dataset):
    """Load GT, RGB conditions, and optional focus maps without resizing."""

    def __init__(self, metadata_path, dataset_base_path=None, resolution=None,
                 random_crop=False, center_crop=False, random_flip=False,
                 max_samples=None, start_index=0, smoke=False, prompt_mode="fixed",
                 native_resolution=True, strict_native_size=True, input_mode="ab_focus",
                 vae_scale_factor=None, max_pixels=None):
        self.input_mode = normalize_input_mode(input_mode)
        if native_resolution and (random_crop or center_crop):
            raise ValueError("native_resolution forbids random_crop and center_crop")
        self.metadata_path = str(metadata_path)
        self.base = Path(dataset_base_path or Path(metadata_path).parent)
        self.resolution = resolution
        self.random_crop = bool(random_crop)
        self.center_crop = bool(center_crop)
        self.random_flip = bool(random_flip)
        self.prompt_mode = prompt_mode
        self.native_resolution = bool(native_resolution)
        self.strict_native_size = bool(strict_native_size)
        self.vae_scale_factor = vae_scale_factor
        self.max_pixels = max_pixels
        records = _metadata_records(metadata_path)
        start = int(start_index)
        limit = 16 if smoke else max_samples
        self.records = records[start:None if limit is None else start + int(limit)]
        self.source_indices = list(range(start, start + len(self.records)))

    def __len__(self):
        return len(self.records)

    def _path(self, value, index, field):
        raw = str(value)
        path = Path(raw)
        full = path if path.is_absolute() else self.base / path
        full = full.resolve()
        if not full.is_file():
            raise FileNotFoundError(
                f"metadata index {index}, input_mode {self.input_mode}, field {field}: "
                f"raw path={raw}, resolved path={full}"
            )
        return full

    def _conditions_and_focus(self, rec, index):
        return parse_metadata_inputs(rec, self.input_mode, index)

    def _check_size(self, index, named_images, a_path):
        sizes = {name: image.size for name, image, _ in named_images}
        if len(set(sizes.values())) != 1:
            detail = ", ".join(f"{name}={path} size={size[1]}x{size[0]}" for name, image, path in named_images for size in [image.size])
            raise ValueError(f"metadata index {index}, input_mode {self.input_mode}: size mismatch: {detail}")
        w, h = next(iter(sizes.values()))
        if self.max_pixels is not None and h * w > self.max_pixels:
            raise ValueError(f"metadata index {index}, input_mode {self.input_mode}: {h}x{w} exceeds max_pixels={self.max_pixels}; images are rejected rather than resized")
        if self.native_resolution and self.strict_native_size and self.vae_scale_factor:
            if h % self.vae_scale_factor or w % self.vae_scale_factor:
                raise ValueError(
                    "Native-resolution mode forbids resize, crop, or padding. "
                    f"H={h}, W={w}, vae_scale_factor={self.vae_scale_factor}, metadata index {index}, "
                    f"A path={a_path}, input_mode={self.input_mode}"
                )
        return h, w

    def __getitem__(self, item):
        rec, index = self.records[item], self.source_indices[item]
        cond_values, focus_values = self._conditions_and_focus(rec, index)
        gt_path = self._path(rec["image"], index, "GT[0]")
        cond_paths = [self._path(x, index, f"condition[{i}]") for i, x in enumerate(cond_values)]
        focus_paths = [self._path(x, index, f"focus[{i}]") for i, x in enumerate(focus_values)]
        gt = Image.open(gt_path).convert("RGB")
        cond_images = [Image.open(p).convert("RGB") for p in cond_paths]
        focus_images = [Image.open(p).convert("L") for p in focus_paths]
        named = [("GT", gt, gt_path)] + [(chr(ord("A") + i), im, path) for i, (im, path) in enumerate(zip(cond_images, cond_paths))]
        named += [(f"focus_{i}", im, path) for i, (im, path) in enumerate(zip(focus_images, focus_paths))]
        h, w = self._check_size(index, named, cond_paths[0])
        do_flip = self.random_flip and random.random() < 0.5
        all_images = [gt] + cond_images + focus_images
        arrays = []
        for im in all_images:
            if do_flip:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            arrays.append(np.asarray(im, dtype=np.float32).copy())

        def rgb(x):
            return torch.from_numpy(x).permute(2, 0, 1).div_(127.5).sub_(1.0)

        def focus(x):
            return torch.from_numpy(x).unsqueeze(0).div_(255.0)

        out = {
            "gt": rgb(arrays[0]),
            "conditions": [rgb(x) for x in arrays[1:1 + len(cond_images)]],
            "focus_maps": [focus(x) for x in arrays[1 + len(cond_images):]],
            "prompt": FIXED_FUSION_PROMPT if self.prompt_mode == "fixed" else str(rec.get("prompt", "")),
            "metadata_index": index,
            "input_mode": self.input_mode,
            "size": (h, w),
            "paths": {"gt": str(gt_path), "conditions": [str(p) for p in cond_paths], "focus_maps": [str(p) for p in focus_paths]},
        }
        out["a"] = out["conditions"][0]
        if len(out["conditions"]) > 1:
            out["b_warp"] = out["conditions"][1]
        if len(out["focus_maps"]) == 2:
            out["focus_a"], out["focus_b_warp"] = out["focus_maps"]
        return out


def focus_fusion_collate(batch):
    sizes = [item["size"] for item in batch]
    if len(set(sizes)) != 1:
        detail = ", ".join(f"index={item['metadata_index']} size={item['size']}" for item in batch)
        raise ValueError(f"native-resolution batch has mixed sizes and padding is forbidden: {detail}")
    out = {k: batch[0][k] for k in ("input_mode",) if k in batch[0]}
    out["prompt"] = [item["prompt"] for item in batch]
    out["metadata_index"] = torch.tensor([item["metadata_index"] for item in batch], dtype=torch.long)
    out["gt"] = torch.stack([item["gt"] for item in batch])
    ncond = len(batch[0]["conditions"])
    out["conditions"] = [torch.stack([item["conditions"][i] for item in batch]) for i in range(ncond)]
    nfocus = len(batch[0]["focus_maps"])
    out["focus_maps"] = [torch.stack([item["focus_maps"][i] for item in batch]) for i in range(nfocus)]
    out["paths"] = [item["paths"] for item in batch]
    out["size"] = sizes[0]
    out["a"] = out["conditions"][0]
    if ncond > 1:
        out["b_warp"] = out["conditions"][1]
    if nfocus == 2:
        out["focus_a"], out["focus_b_warp"] = out["focus_maps"]
    return out
