"""JSON dataset for aligned focus-bracket fusion training and inference."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


FIXED_FUSION_PROMPT = (
    "Fuse the two focus-bracketed images into one all-in-focus image. Preserve the "
    "structure, viewpoint, color, and sharp details of Image A. Use Image B only "
    "to restore regions that are blurred in Image A."
)


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


class FocusFusionDataset(Dataset):
    """Loads GT/A/B/focus-A/focus-B and applies one shared geometric transform."""

    def __init__(self, metadata_path, dataset_base_path=None, resolution=512,
                 random_crop=False, center_crop=False, random_flip=False,
                 max_samples=None, start_index=0, smoke=False, prompt_mode="fixed"):
        if random_crop and center_crop:
            raise ValueError("random_crop and center_crop are mutually exclusive")
        self.metadata_path = str(metadata_path)
        self.base = Path(dataset_base_path or Path(metadata_path).parent)
        self.resolution = int(resolution)
        self.random_crop = bool(random_crop)
        self.center_crop = bool(center_crop)
        self.random_flip = bool(random_flip)
        self.prompt_mode = prompt_mode
        records = _metadata_records(metadata_path)
        start = int(start_index)
        limit = 16 if smoke else max_samples
        self.records = records[start:None if limit is None else start + int(limit)]
        self.source_indices = list(range(start, start + len(self.records)))

    def __len__(self):
        return len(self.records)

    def _path(self, value, index):
        path = Path(value)
        if not path.is_absolute():
            path = self.base / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"metadata index {index}: missing path: {path}")
        return path

    def __getitem__(self, item):
        rec, index = self.records[item], self.source_indices[item]
        edits = rec.get("edit_image")
        if not isinstance(edits, list) or len(edits) < 4:
            raise ValueError(f"metadata index {index}: edit_image must contain A, B_warp, focus_A, focus_B_warp")
        paths = [self._path(rec["image"], index)] + [self._path(x, index) for x in edits[:4]]
        images = [Image.open(paths[i]).convert("RGB" if i < 3 else "L") for i in range(5)]
        w, h = images[0].size
        if any(im.size != (w, h) for im in images[1:]):
            sizes = [im.size for im in images]
            raise ValueError(f"metadata index {index}: aligned inputs have different sizes: {sizes}")

        # Resize the short side, then share exactly one crop/flip decision.
        scale = max(self.resolution / w, self.resolution / h)
        rw, rh = max(self.resolution, round(w * scale)), max(self.resolution, round(h * scale))
        resized = []
        for i, im in enumerate(images):
            resized.append(im.resize((rw, rh), Image.Resampling.BILINEAR))
        if self.random_crop:
            left = random.randint(0, rw - self.resolution)
            top = random.randint(0, rh - self.resolution)
        else:  # center crop is also the deterministic default
            left, top = (rw - self.resolution) // 2, (rh - self.resolution) // 2
        box = (left, top, left + self.resolution, top + self.resolution)
        do_flip = self.random_flip and random.random() < 0.5
        arrays = []
        for im in resized:
            im = im.crop(box)
            if do_flip:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            arrays.append(np.asarray(im, dtype=np.float32).copy())

        def rgb(x):
            return torch.from_numpy(x).permute(2, 0, 1).div_(127.5).sub_(1.0)
        def focus(x):
            return torch.from_numpy(x).unsqueeze(0).div_(255.0)
        prompt = FIXED_FUSION_PROMPT if self.prompt_mode == "fixed" else str(rec.get("prompt", ""))
        return {
            "gt": rgb(arrays[0]), "a": rgb(arrays[1]), "b_warp": rgb(arrays[2]),
            "focus_a": focus(arrays[3]), "focus_b_warp": focus(arrays[4]),
            "prompt": prompt, "metadata_index": index,
            "paths": {"gt": str(paths[0]), "a": str(paths[1]), "b_warp": str(paths[2]),
                      "focus_a": str(paths[3]), "focus_b_warp": str(paths[4])},
        }

