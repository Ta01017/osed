#!/usr/bin/env bash
set -euo pipefail
GPU="${GPU:-0}"; INPUT_MODE="${INPUT_MODE:-ab_focus}"; MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
LORA_RANK_UNET="${LORA_RANK_UNET:-8}"; LORA_RANK_VAE="${LORA_RANK_VAE:-4}"; LORA_RANK_VSD="${LORA_RANK_VSD:-8}"
METADATA="${METADATA:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/focus_merged_6000_dedup_0710_v3/train/metadata_with_homography_warped_focus_ckptA.json}"
DATASET_BASE="${DATASET_BASE:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/focus_merged_6000_dedup_0710_v3/train}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-stabilityai/stable-diffusion-2-1-base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/focus_fusion_smoke}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"; OUTPUT_DIR="${OUTPUT_ROOT}/osediff_focus_${INPUT_MODE}_native_smoke1_${TIMESTAMP}"
[[ -e "$OUTPUT_DIR" ]] && { echo "output exists: $OUTPUT_DIR" >&2; exit 1; }
LOG="$OUTPUT_DIR/smoke_1step.log"
mkdir -p "$OUTPUT_DIR"
set +e
CUDA_VISIBLE_DEVICES="$GPU" accelerate launch --num_processes 1 train_osediff_focus_fusion.py \
 --pretrained_model_name_or_path "$PRETRAINED_MODEL" --metadata_path "$METADATA" --dataset_base_path "$DATASET_BASE" --output_dir "$OUTPUT_DIR" \
 --input_mode "$INPUT_MODE" --prompt_mode fixed --use_vsd 0 --train_batch_size 1 --gradient_accumulation_steps 1 --sync_with_dataloader \
 --lora_rank_unet "$LORA_RANK_UNET" --lora_rank_vae "$LORA_RANK_VAE" --lora_rank_vsd "$LORA_RANK_VSD" \
 --max_samples 16 --max_train_steps 1 --checkpointing_steps 1 --validation_steps 1 --validation_max_samples 1 \
 --native_resolution --strict_native_size --mixed_precision "$MIXED_PRECISION" 2>&1 | tee "$LOG"
code=${PIPESTATUS[0]}
set -e
[[ "$code" == "0" ]] || { echo "[SMOKE FAIL] training command failed: $code" >&2; exit "$code"; }
python - "$OUTPUT_DIR" "$INPUT_MODE" "$LOG" <<'PY'
import json, re, sys
from pathlib import Path
from PIL import Image
from osediff_focus_fusion import compute_file_sha256

out = Path(sys.argv[1]); mode = sys.argv[2]; log = Path(sys.argv[3])
text = log.read_text(errors="replace")
assert "Traceback" not in text
assert not re.search(r"(nan|inf) loss", text, re.I)
ckpt = out / "checkpoints" / "checkpoint-00000001"
manifest = ckpt / "checkpoint_complete.json"
assert ckpt.is_dir(), ckpt
assert (ckpt / "accelerator_state").is_dir() and any((ckpt / "accelerator_state").iterdir())
assert manifest.is_file(), manifest
m = json.loads(manifest.read_text())
assert m["complete"] is True and m["global_step"] == 1
assert m["input_mode"] == mode
for key in ("model_state", "trainer_state", "optimizer_manifest"):
    p = ckpt / m[key]["filename"]
    assert p.stat().st_size == m[key]["size"]
    assert compute_file_sha256(p) == m[key]["sha256"]
trainer = json.loads((ckpt / "trainer_state.json").read_text())
assert trainer["global_step"] == 1
assert trainer["sync_with_dataloader"] is True
assert m["generator_in_channels"] in (4, 8, 10, 16)
val = out / "validation" / "global_step_000001"
assert val.is_dir(), val
sample = next(val.iterdir())
pred = Image.open(sample / "pred_raw.png")
a = Image.open(sample / "A.png")
assert pred.size == a.size
print("[SMOKE PASS]")
PY
