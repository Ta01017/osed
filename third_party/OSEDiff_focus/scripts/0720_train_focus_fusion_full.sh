#!/usr/bin/env bash
set -euo pipefail
GPUS="${GPUS:-0,1,2,3}"; NUM_PROCESSES="${NUM_PROCESSES:-4}"; MAX_STEPS="${MAX_STEPS:-10000}"; MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
METADATA="${METADATA:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/focus_merged_6000_dedup_0710_v3/train/metadata_with_homography_warped_focus_ckptA.json}"
DATASET_BASE="${DATASET_BASE:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/focus_merged_6000_dedup_0710_v3/train}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-stabilityai/stable-diffusion-2-1-base}"; OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/focus_fusion_full}"
OUTPUT_DIR="${OUTPUT_ROOT}/$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --num_processes "$NUM_PROCESSES" train_osediff_focus_fusion.py \
 --pretrained_model_name_or_path "$PRETRAINED_MODEL" --metadata_path "$METADATA" --dataset_base_path "$DATASET_BASE" --output_dir "$OUTPUT_DIR" \
 --resolution 512 --condition_mode ab_focus --prompt_mode fixed --use_vsd "${USE_VSD:-0}" --train_batch_size "${BATCH_SIZE:-1}" \
 --max_train_steps "$MAX_STEPS" --checkpointing_steps "${CHECKPOINT_STEPS:-500}" --validation_steps "${VALIDATION_STEPS:-500}" --mixed_precision "$MIXED_PRECISION"

