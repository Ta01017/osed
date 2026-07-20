#!/usr/bin/env bash
set -euo pipefail
GPUS="${GPUS:-0,1,2,3}"; NUM_PROCESSES="${NUM_PROCESSES:-4}"; INPUT_MODE="${INPUT_MODE:-ab_focus}"
MAX_STEPS="${MAX_STEPS:-10000}"; MIXED_PRECISION="${MIXED_PRECISION:-fp16}"; TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"; NATIVE_RESOLUTION="${NATIVE_RESOLUTION:-1}"; STRICT_NATIVE_SIZE="${STRICT_NATIVE_SIZE:-1}"; MAX_PIXELS="${MAX_PIXELS:-}"
TRAIN_CONV_IN="${TRAIN_CONV_IN:-1}"; CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}"; VALIDATION_STEPS="${VALIDATION_STEPS:-500}"; VALIDATION_MAX_SAMPLES="${VALIDATION_MAX_SAMPLES:-4}"
METADATA="${METADATA:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/focus_merged_6000_dedup_0710_v3/train/metadata_with_homography_warped_focus_ckptA.json}"
DATASET_BASE="${DATASET_BASE:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/focus_merged_6000_dedup_0710_v3/train}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-stabilityai/stable-diffusion-2-1-base}"; OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/focus_fusion_full}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"; OUTPUT_DIR="${OUTPUT_ROOT}/osediff_focus_${INPUT_MODE}_native_${TIMESTAMP}"
[[ -e "$OUTPUT_DIR" ]] && { echo "output exists: $OUTPUT_DIR" >&2; exit 1; }
NATIVE_ARGS=(); [[ "$NATIVE_RESOLUTION" == "1" ]] && NATIVE_ARGS+=(--native_resolution) || NATIVE_ARGS+=(--no-native_resolution)
[[ "$STRICT_NATIVE_SIZE" == "1" ]] && NATIVE_ARGS+=(--strict_native_size) || NATIVE_ARGS+=(--no-strict_native_size)
MAX_PIXEL_ARGS=(); [[ -n "$MAX_PIXELS" ]] && MAX_PIXEL_ARGS+=(--max_pixels "$MAX_PIXELS")
TRAIN_VAE_ARGS=(); [[ "${TRAIN_VAE_LORA:-0}" == "1" ]] && TRAIN_VAE_ARGS+=(--train_vae_lora)
TRAIN_CONV_ARGS=(); [[ "$TRAIN_CONV_IN" == "1" ]] && TRAIN_CONV_ARGS+=(--train_conv_in) || TRAIN_CONV_ARGS+=(--no-train_conv_in)
CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --num_processes "$NUM_PROCESSES" train_osediff_focus_fusion.py \
 --pretrained_model_name_or_path "$PRETRAINED_MODEL" --metadata_path "$METADATA" --dataset_base_path "$DATASET_BASE" --output_dir "$OUTPUT_DIR" \
 --input_mode "$INPUT_MODE" --prompt_mode fixed --use_vsd "${USE_VSD:-0}" "${TRAIN_VAE_ARGS[@]}" "${TRAIN_CONV_ARGS[@]}" --train_batch_size "$TRAIN_BATCH_SIZE" \
 --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" --max_train_steps "$MAX_STEPS" --checkpointing_steps "$CHECKPOINTING_STEPS" \
 --validation_steps "$VALIDATION_STEPS" --validation_max_samples "$VALIDATION_MAX_SAMPLES" --mixed_precision "$MIXED_PRECISION" "${NATIVE_ARGS[@]}" "${MAX_PIXEL_ARGS[@]}"
