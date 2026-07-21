#!/usr/bin/env bash
set -euo pipefail
GPU="${GPU:-0}"; INPUT_MODE="${INPUT_MODE:-ab_focus}"; MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
TRAIN_VAE_LORA="${TRAIN_VAE_LORA:-0}"; USE_VSD="${USE_VSD:-0}"
LORA_RANK_UNET="${LORA_RANK_UNET:-8}"; LORA_RANK_VAE="${LORA_RANK_VAE:-4}"; LORA_RANK_VSD="${LORA_RANK_VSD:-8}"
METADATA="${METADATA:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/focus_merged_6000_dedup_0710_v3/train/metadata_with_homography_warped_focus_ckptA.json}"
DATASET_BASE="${DATASET_BASE:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/focus_merged_6000_dedup_0710_v3/train}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-stabilityai/stable-diffusion-2-1-base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/focus_fusion_smoke_resume}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"; OUTPUT_DIR="${OUTPUT_ROOT}/osediff_focus_${INPUT_MODE}_native_resume_${TIMESTAMP}"
[[ -e "$OUTPUT_DIR" ]] && { echo "output exists: $OUTPUT_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
VAE_ARGS=(); [[ "$TRAIN_VAE_LORA" == "1" ]] && VAE_ARGS+=(--train_vae_lora)
LOG1="$OUTPUT_DIR/resume_step1.log"; LOG2="$OUTPUT_DIR/resume_step2.log"
set +e
CUDA_VISIBLE_DEVICES="$GPU" accelerate launch --num_processes 1 train_osediff_focus_fusion.py \
 --pretrained_model_name_or_path "$PRETRAINED_MODEL" --metadata_path "$METADATA" --dataset_base_path "$DATASET_BASE" --output_dir "$OUTPUT_DIR" \
 --input_mode "$INPUT_MODE" --prompt_mode fixed --use_vsd "$USE_VSD" "${VAE_ARGS[@]}" --train_batch_size 1 --gradient_accumulation_steps 1 \
 --lora_rank_unet "$LORA_RANK_UNET" --lora_rank_vae "$LORA_RANK_VAE" --lora_rank_vsd "$LORA_RANK_VSD" \
 --max_samples 16 --max_train_steps 1 --checkpointing_steps 1 --validation_steps 1 --validation_max_samples 1 \
 --native_resolution --strict_native_size --mixed_precision "$MIXED_PRECISION" 2>&1 | tee "$LOG1"
code=${PIPESTATUS[0]}; set -e; [[ "$code" == "0" ]] || { echo "[SMOKE FAIL] first training command failed: $code" >&2; exit "$code"; }
CHECKPOINT="$OUTPUT_DIR/checkpoints/checkpoint-00000001"
python - "$CHECKPOINT" "$OUTPUT_DIR/checksum_step1.json" <<'PY'
import json, sys, torch
from pathlib import Path
from osediff_focus_fusion import compute_file_sha256
ckpt = Path(sys.argv[1]); out = Path(sys.argv[2])
m = json.loads((ckpt/"checkpoint_complete.json").read_text())
assert (ckpt/"accelerator_state").is_dir() and any((ckpt/"accelerator_state").iterdir())
for key in ("model_state","trainer_state","optimizer_manifest"):
    p = ckpt / m[key]["filename"]
    assert compute_file_sha256(p) == m[key]["sha256"]
state = torch.load(ckpt/"model_state.pt", map_location="cpu")
checks = {}
for name in ("generator_unet_lora","vae_lora","vsd_unet_lora"):
    tensors = state.get(name, {})
    checks[name] = sum(float(v.float().sum()) for v in tensors.values()) if tensors else None
out.write_text(json.dumps(checks, sort_keys=True))
print(json.dumps(checks, sort_keys=True))
PY
set +e
CUDA_VISIBLE_DEVICES="$GPU" accelerate launch --num_processes 1 train_osediff_focus_fusion.py \
 --pretrained_model_name_or_path "$PRETRAINED_MODEL" --metadata_path "$METADATA" --dataset_base_path "$DATASET_BASE" --output_dir "$OUTPUT_DIR" \
 --input_mode "$INPUT_MODE" --prompt_mode fixed --use_vsd "$USE_VSD" "${VAE_ARGS[@]}" --train_batch_size 1 --gradient_accumulation_steps 1 \
 --lora_rank_unet "$LORA_RANK_UNET" --lora_rank_vae "$LORA_RANK_VAE" --lora_rank_vsd "$LORA_RANK_VSD" \
 --max_samples 16 --max_train_steps 2 --checkpointing_steps 1 --validation_steps 1 --validation_max_samples 1 \
 --resume_from_checkpoint "$CHECKPOINT" --native_resolution --strict_native_size --mixed_precision "$MIXED_PRECISION" 2>&1 | tee "$LOG2"
code=${PIPESTATUS[0]}; set -e; [[ "$code" == "0" ]] || { echo "[SMOKE FAIL] resume command failed: $code" >&2; exit "$code"; }
python - "$OUTPUT_DIR" "$LOG2" <<'PY'
import json, sys, re
from pathlib import Path
from osediff_focus_fusion import compute_file_sha256
out = Path(sys.argv[1]); log = Path(sys.argv[2])
text = log.read_text(errors="replace")
for needle in ("checkpoint verified", "accelerator state restored", "sampler position restored"):
    assert needle in text, needle
assert "checkpoint global_step 1" in text
ckpt2 = out/"checkpoints"/"checkpoint-00000002"
m = json.loads((ckpt2/"checkpoint_complete.json").read_text())
assert (ckpt2/"accelerator_state").is_dir() and any((ckpt2/"accelerator_state").iterdir())
assert m["global_step"] == 2
for key in ("model_state","trainer_state","optimizer_manifest"):
    p = ckpt2 / m[key]["filename"]
    assert compute_file_sha256(p) == m[key]["sha256"]
trainer = json.loads((ckpt2/"trainer_state.json").read_text())
assert trainer["global_step"] == 2
print("[SMOKE PASS]")
PY
