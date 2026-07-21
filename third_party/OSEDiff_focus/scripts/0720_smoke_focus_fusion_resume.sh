#!/usr/bin/env bash
set -euo pipefail
GPU="${GPU:-0}"; INPUT_MODE="${INPUT_MODE:-ab_focus}"; MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
TRAIN_VAE_LORA="${TRAIN_VAE_LORA:-0}"; USE_VSD="${USE_VSD:-0}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
RUN_PARTIAL_EPOCH_TEST="${RUN_PARTIAL_EPOCH_TEST:-0}"
LORA_RANK_UNET="${LORA_RANK_UNET:-8}"; LORA_RANK_VAE="${LORA_RANK_VAE:-4}"; LORA_RANK_VSD="${LORA_RANK_VSD:-8}"
METADATA="${METADATA:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/focus_merged_6000_dedup_0710_v3/train/metadata_with_homography_warped_focus_ckptA.json}"
DATASET_BASE="${DATASET_BASE:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/focus_merged_6000_dedup_0710_v3/train}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-stabilityai/stable-diffusion-2-1-base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/focus_fusion_smoke_resume}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"; OUTPUT_DIR="${OUTPUT_ROOT}/osediff_focus_${INPUT_MODE}_native_resume_${TIMESTAMP}"
[[ -e "$OUTPUT_DIR" ]] && { echo "output exists: $OUTPUT_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
VAE_ARGS=(); [[ "$TRAIN_VAE_LORA" == "1" ]] && VAE_ARGS+=(--train_vae_lora)
LOG1="$OUTPUT_DIR/resume_step1.log"; LOG2="$OUTPUT_DIR/resume_step2.log"; LOG3="$OUTPUT_DIR/resume_step3.log"
set +e
CUDA_VISIBLE_DEVICES="$GPU" accelerate launch --num_processes 1 train_osediff_focus_fusion.py \
 --pretrained_model_name_or_path "$PRETRAINED_MODEL" --metadata_path "$METADATA" --dataset_base_path "$DATASET_BASE" --output_dir "$OUTPUT_DIR" \
 --input_mode "$INPUT_MODE" --prompt_mode fixed --use_vsd "$USE_VSD" "${VAE_ARGS[@]}" --train_batch_size 1 --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" --sync_with_dataloader \
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
trainer = json.loads((ckpt/"trainer_state.json").read_text())
assert trainer["global_step"] == trainer["optimizer_updates"] == trainer["scheduler_steps"] == 1
assert trainer["micro_batches"] == int(1 * int(trainer["gradient_accumulation_steps"]))
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
 --input_mode "$INPUT_MODE" --prompt_mode fixed --use_vsd "$USE_VSD" "${VAE_ARGS[@]}" --train_batch_size 1 --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" --sync_with_dataloader \
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
assert trainer["global_step"] == trainer["optimizer_updates"] == trainer["scheduler_steps"] == 2
assert trainer["micro_batches"] == int(2 * int(trainer["gradient_accumulation_steps"]))
PY
CHECKPOINT="$OUTPUT_DIR/checkpoints/checkpoint-00000002"
set +e
CUDA_VISIBLE_DEVICES="$GPU" accelerate launch --num_processes 1 train_osediff_focus_fusion.py \
 --pretrained_model_name_or_path "$PRETRAINED_MODEL" --metadata_path "$METADATA" --dataset_base_path "$DATASET_BASE" --output_dir "$OUTPUT_DIR" \
 --input_mode "$INPUT_MODE" --prompt_mode fixed --use_vsd "$USE_VSD" "${VAE_ARGS[@]}" --train_batch_size 1 --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" --sync_with_dataloader \
 --lora_rank_unet "$LORA_RANK_UNET" --lora_rank_vae "$LORA_RANK_VAE" --lora_rank_vsd "$LORA_RANK_VSD" \
 --max_samples 16 --max_train_steps 3 --checkpointing_steps 1 --validation_steps 1 --validation_max_samples 1 \
 --resume_from_checkpoint "$CHECKPOINT" --native_resolution --strict_native_size --mixed_precision "$MIXED_PRECISION" 2>&1 | tee "$LOG3"
code=${PIPESTATUS[0]}; set -e; [[ "$code" == "0" ]] || { echo "[SMOKE FAIL] second resume command failed: $code" >&2; exit "$code"; }
python - "$OUTPUT_DIR" "$LOG2" "$LOG3" <<'PY'
import json, sys
from pathlib import Path
from osediff_focus_fusion import compute_file_sha256
out = Path(sys.argv[1])
for log_path in map(Path, sys.argv[2:]):
    text = log_path.read_text(errors="replace")
    for needle in ("checkpoint verified", "resume config validated", "sampler state validated", "optimizer manifest validated", "accelerator state restored", "trainer state restored"):
        assert needle in text, (log_path, needle)
states = []
assert (out/"checkpoints"/"checkpoint-00000003").is_dir()
for step in (1, 2, 3):
    ckpt = out/"checkpoints"/f"checkpoint-{step:08d}"
    m = json.loads((ckpt/"checkpoint_complete.json").read_text())
    assert m["global_step"] == step
    assert (ckpt/"accelerator_state").is_dir() and any((ckpt/"accelerator_state").iterdir())
    for key in ("model_state","trainer_state","optimizer_manifest"):
        p = ckpt / m[key]["filename"]
        assert compute_file_sha256(p) == m[key]["sha256"]
    trainer = json.loads((ckpt/"trainer_state.json").read_text())
    assert trainer["global_step"] == trainer["optimizer_updates"] == trainer["scheduler_steps"] == step
    assert trainer["sync_with_dataloader"] is True
    assert trainer["micro_batches"] == step * int(trainer["gradient_accumulation_steps"])
    states.append(trainer)
assert states[0]["micro_batches"] < states[1]["micro_batches"] < states[2]["micro_batches"]
print("[RESUME SMOKE PASS]")
PY

if [[ "$RUN_PARTIAL_EPOCH_TEST" == "1" ]]; then
  PARTIAL_DIR="${OUTPUT_DIR}_partial_epoch"
  PARTIAL_LOG="$PARTIAL_DIR/partial_epoch.log"
  mkdir -p "$PARTIAL_DIR"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" accelerate launch --num_processes 1 train_osediff_focus_fusion.py \
   --pretrained_model_name_or_path "$PRETRAINED_MODEL" --metadata_path "$METADATA" --dataset_base_path "$DATASET_BASE" --output_dir "$PARTIAL_DIR" \
   --input_mode "$INPUT_MODE" --prompt_mode fixed --use_vsd "$USE_VSD" "${VAE_ARGS[@]}" --train_batch_size 1 --gradient_accumulation_steps 4 --sync_with_dataloader \
   --lora_rank_unet "$LORA_RANK_UNET" --lora_rank_vae "$LORA_RANK_VAE" --lora_rank_vsd "$LORA_RANK_VSD" \
   --max_samples 6 --max_train_steps 2 --checkpointing_steps 1 --validation_steps 0 --validation_max_samples 1 \
   --native_resolution --strict_native_size --mixed_precision "$MIXED_PRECISION" 2>&1 | tee "$PARTIAL_LOG"
  code=${PIPESTATUS[0]}; set -e; [[ "$code" == "0" ]] || { echo "[SMOKE FAIL] partial epoch command failed: $code" >&2; exit "$code"; }
  PARTIAL_CHECKPOINT="$PARTIAL_DIR/checkpoints/checkpoint-00000002"
  python - "$PARTIAL_CHECKPOINT" <<'PY'
import json, sys
from pathlib import Path
from osediff_focus_fusion import load_verified_checkpoint
ckpt = Path(sys.argv[1])
state = load_verified_checkpoint(ckpt)
trainer = state["trainer_state"]
assert trainer["global_step"] == trainer["optimizer_updates"] == trainer["scheduler_steps"]
assert trainer["global_step"] == 2
assert trainer["micro_batches"] == 6
assert trainer["gradient_accumulation_steps"] == 4
assert trainer["sync_with_dataloader"] is True
assert trainer["current_epoch"] == trainer["sampler_epoch"] == trainer["completed_epochs"] == 1
assert trainer["batches_consumed_in_current_epoch"] == 0
print(json.dumps(trainer, sort_keys=True))
PY
  PARTIAL_RESUME_LOG="$PARTIAL_DIR/partial_epoch_resume.log"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" accelerate launch --num_processes 1 train_osediff_focus_fusion.py \
   --pretrained_model_name_or_path "$PRETRAINED_MODEL" --metadata_path "$METADATA" --dataset_base_path "$DATASET_BASE" --output_dir "$PARTIAL_DIR" \
   --input_mode "$INPUT_MODE" --prompt_mode fixed --use_vsd "$USE_VSD" "${VAE_ARGS[@]}" --train_batch_size 1 --gradient_accumulation_steps 4 --sync_with_dataloader \
   --lora_rank_unet "$LORA_RANK_UNET" --lora_rank_vae "$LORA_RANK_VAE" --lora_rank_vsd "$LORA_RANK_VSD" \
   --max_samples 6 --max_train_steps 3 --checkpointing_steps 1 --validation_steps 0 --validation_max_samples 1 \
   --resume_from_checkpoint "$PARTIAL_CHECKPOINT" --native_resolution --strict_native_size --mixed_precision "$MIXED_PRECISION" 2>&1 | tee "$PARTIAL_RESUME_LOG"
  code=${PIPESTATUS[0]}; set -e; [[ "$code" == "0" ]] || { echo "[SMOKE FAIL] partial epoch resume failed: $code" >&2; exit "$code"; }
  python - "$PARTIAL_DIR" <<'PY'
import json, sys
from pathlib import Path
from osediff_focus_fusion import load_verified_checkpoint
out = Path(sys.argv[1])
step2 = load_verified_checkpoint(out / "checkpoints" / "checkpoint-00000002")["trainer_state"]
step3 = load_verified_checkpoint(out / "checkpoints" / "checkpoint-00000003")["trainer_state"]
assert step3["global_step"] == step3["optimizer_updates"] == step3["scheduler_steps"] == 3
assert step3["micro_batches"] > step2["micro_batches"]
print("[PARTIAL EPOCH SMOKE PASS]")
PY
fi

mutate_checkpoint() {
  local src="$1" dst="$2" expr="$3"
  rm -rf "$dst"
  cp -a "$src" "$dst"
  python - "$dst" "$expr" <<'PY'
import json, sys, os, hashlib
from pathlib import Path
ckpt = Path(sys.argv[1]); expr = sys.argv[2]
trainer_path = ckpt / "trainer_state.json"
manifest_path = ckpt / "checkpoint_complete.json"
trainer = json.loads(trainer_path.read_text())
manifest = json.loads(manifest_path.read_text())
ns = {"trainer": trainer, "manifest": manifest}
exec(expr, ns, ns)
trainer_path.write_text(json.dumps(trainer, sort_keys=True), encoding="utf-8")
entry = manifest["trainer_state"]
data = trainer_path.read_bytes()
entry["size"] = len(data)
entry["sha256"] = hashlib.sha256(data).hexdigest()
manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
PY
}

mutate_optimizer_manifest() {
  local src="$1" dst="$2"
  rm -rf "$dst"
  cp -a "$src" "$dst"
  python - "$dst" <<'PY'
import json, sys, hashlib
from pathlib import Path
ckpt = Path(sys.argv[1])
manifest_path = ckpt / "checkpoint_complete.json"
optim_path = ckpt / "optimizer_manifest.json"
manifest = json.loads(manifest_path.read_text())
optim = json.loads(optim_path.read_text())
for group in optim:
    names = group.get("parameter_names", [])
    if len(names) >= 2:
        names[0], names[1] = names[1], names[0]
        break
else:
    optim[0].setdefault("parameter_names", ["a", "b"])
    optim[0]["parameter_names"][0], optim[0]["parameter_names"][1] = "b", "a"
optim_path.write_text(json.dumps(optim, sort_keys=True), encoding="utf-8")
data = optim_path.read_bytes()
manifest["optimizer_manifest"]["size"] = len(data)
manifest["optimizer_manifest"]["sha256"] = hashlib.sha256(data).hexdigest()
manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
PY
}

expect_resume_fail() {
  local bad_ckpt="$1" log="$2" needle="$3"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" accelerate launch --num_processes 1 train_osediff_focus_fusion.py \
   --pretrained_model_name_or_path "$PRETRAINED_MODEL" --metadata_path "$METADATA" --dataset_base_path "$DATASET_BASE" --output_dir "$OUTPUT_DIR" \
   --input_mode "$INPUT_MODE" --prompt_mode fixed --use_vsd "$USE_VSD" "${VAE_ARGS[@]}" --train_batch_size 1 --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" --sync_with_dataloader \
   --lora_rank_unet "$LORA_RANK_UNET" --lora_rank_vae "$LORA_RANK_VAE" --lora_rank_vsd "$LORA_RANK_VSD" \
   --max_samples 16 --max_train_steps 3 --checkpointing_steps 1 --validation_steps 0 --validation_max_samples 1 \
   --resume_from_checkpoint "$bad_ckpt" --native_resolution --strict_native_size --mixed_precision "$MIXED_PRECISION" >"$log" 2>&1
  code=$?; set -e
  [[ "$code" != "0" ]] || { echo "[NEGATIVE SMOKE FAIL] corrupt checkpoint resumed successfully: $bad_ckpt" >&2; exit 1; }
  grep -Fq "$needle" "$log" || { echo "[NEGATIVE SMOKE FAIL] missing expected error '$needle' in $log" >&2; exit 1; }
}

BAD_ROOT="$OUTPUT_DIR/bad_checkpoints"; mkdir -p "$BAD_ROOT"
mutate_checkpoint "$OUTPUT_DIR/checkpoints/checkpoint-00000002" "$BAD_ROOT/bad_optimizer_updates" 'trainer["optimizer_updates"]=1'
expect_resume_fail "$BAD_ROOT/bad_optimizer_updates" "$BAD_ROOT/bad_optimizer_updates.log" "optimizer_updates"
mutate_checkpoint "$OUTPUT_DIR/checkpoints/checkpoint-00000002" "$BAD_ROOT/bad_scheduler_steps" 'trainer["scheduler_steps"]=1'
expect_resume_fail "$BAD_ROOT/bad_scheduler_steps" "$BAD_ROOT/bad_scheduler_steps.log" "scheduler_steps"
mutate_checkpoint "$OUTPUT_DIR/checkpoints/checkpoint-00000002" "$BAD_ROOT/bad_micro_batches" 'trainer["micro_batches"]=1'
expect_resume_fail "$BAD_ROOT/bad_micro_batches" "$BAD_ROOT/bad_micro_batches.log" "micro_batches"
mutate_checkpoint "$OUTPUT_DIR/checkpoints/checkpoint-00000002" "$BAD_ROOT/bad_manifest_step" 'manifest["global_step"]=3'
expect_resume_fail "$BAD_ROOT/bad_manifest_step" "$BAD_ROOT/bad_manifest_step.log" "global_step mismatch"
mutate_checkpoint "$OUTPUT_DIR/checkpoints/checkpoint-00000002" "$BAD_ROOT/bad_resume_config" 'trainer["resume_config"]["learning_rate"]=999.0'
expect_resume_fail "$BAD_ROOT/bad_resume_config" "$BAD_ROOT/bad_resume_config.log" "[RESUME CONFIG MISMATCH]"
mutate_optimizer_manifest "$OUTPUT_DIR/checkpoints/checkpoint-00000002" "$BAD_ROOT/bad_optimizer_order"
expect_resume_fail "$BAD_ROOT/bad_optimizer_order" "$BAD_ROOT/bad_optimizer_order.log" "[OPTIMIZER MANIFEST MISMATCH]"
echo "[NEGATIVE RESUME SMOKE PASS]"
