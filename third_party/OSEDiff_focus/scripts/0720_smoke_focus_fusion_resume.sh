#!/usr/bin/env bash
set -euo pipefail
GPU="${GPU:-0}"; INPUT_MODE="${INPUT_MODE:-ab_focus}"; MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
METADATA="${METADATA:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/focus_merged_6000_dedup_0710_v3/train/metadata_with_homography_warped_focus_ckptA.json}"
DATASET_BASE="${DATASET_BASE:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/focus_merged_6000_dedup_0710_v3/train}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-stabilityai/stable-diffusion-2-1-base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/focus_fusion_smoke_resume}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"; OUTPUT_DIR="${OUTPUT_ROOT}/osediff_focus_${INPUT_MODE}_native_resume_${TIMESTAMP}"
[[ -e "$OUTPUT_DIR" ]] && { echo "output exists: $OUTPUT_DIR" >&2; exit 1; }
CUDA_VISIBLE_DEVICES="$GPU" accelerate launch --num_processes 1 train_osediff_focus_fusion.py \
 --pretrained_model_name_or_path "$PRETRAINED_MODEL" --metadata_path "$METADATA" --dataset_base_path "$DATASET_BASE" --output_dir "$OUTPUT_DIR" \
 --input_mode "$INPUT_MODE" --prompt_mode fixed --use_vsd "${USE_VSD:-0}" --train_batch_size 1 --gradient_accumulation_steps 1 \
 --max_samples 16 --max_train_steps 1 --checkpointing_steps 1 --validation_steps 1 --validation_max_samples 1 \
 --native_resolution --strict_native_size --mixed_precision "$MIXED_PRECISION"
CHECKPOINT="$OUTPUT_DIR/checkpoints/focus_fusion_1.pt"
CUDA_VISIBLE_DEVICES="$GPU" accelerate launch --num_processes 1 train_osediff_focus_fusion.py \
 --pretrained_model_name_or_path "$PRETRAINED_MODEL" --metadata_path "$METADATA" --dataset_base_path "$DATASET_BASE" --output_dir "$OUTPUT_DIR" \
 --input_mode "$INPUT_MODE" --prompt_mode fixed --use_vsd "${USE_VSD:-0}" --train_batch_size 1 --gradient_accumulation_steps 1 \
 --max_samples 16 --max_train_steps 2 --checkpointing_steps 1 --validation_steps 1 --validation_max_samples 1 \
 --resume_from_checkpoint "$CHECKPOINT" --native_resolution --strict_native_size --mixed_precision "$MIXED_PRECISION"
