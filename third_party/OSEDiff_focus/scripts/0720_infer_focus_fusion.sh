#!/usr/bin/env bash
set -euo pipefail
: "${CHECKPOINT:?set CHECKPOINT to a focus-fusion .pt checkpoint}"
GPU="${GPU:-0}"; INPUT_MODE="${INPUT_MODE:-ab_focus}"; VAE_ENCODE_MODE="${VAE_ENCODE_MODE:-mode}"
NATIVE_RESOLUTION="${NATIVE_RESOLUTION:-1}"; STRICT_NATIVE_SIZE="${STRICT_NATIVE_SIZE:-1}"; MAX_PIXELS="${MAX_PIXELS:-}"
METADATA="${METADATA:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/focus_merged_6000_dedup_0710_v3/train/metadata_with_homography_warped_focus_ckptA.json}"
DATASET_BASE="${DATASET_BASE:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/focus_merged_6000_dedup_0710_v3/train}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-stabilityai/stable-diffusion-2-1-base}"; OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/focus_fusion_infer}"
KEEP_ARGS=(); if [[ "${KEEP_A_COMPOSITE:-0}" == "1" ]]; then KEEP_ARGS+=(--keep_a_composite); fi
NATIVE_ARGS=(); [[ "$NATIVE_RESOLUTION" == "1" ]] && NATIVE_ARGS+=(--native_resolution) || NATIVE_ARGS+=(--no-native_resolution)
[[ "$STRICT_NATIVE_SIZE" == "1" ]] && NATIVE_ARGS+=(--strict_native_size) || NATIVE_ARGS+=(--no-strict_native_size)
MAX_PIXEL_ARGS=(); [[ -n "$MAX_PIXELS" ]] && MAX_PIXEL_ARGS+=(--max_pixels "$MAX_PIXELS")
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"; OUTPUT_DIR="${OUTPUT_ROOT}/osediff_focus_${INPUT_MODE}_native_infer_${TIMESTAMP}"
[[ -e "$OUTPUT_DIR" ]] && { echo "output exists: $OUTPUT_DIR" >&2; exit 1; }
CUDA_VISIBLE_DEVICES="$GPU" python test_osediff_focus_fusion.py --pretrained_model_name_or_path "$PRETRAINED_MODEL" --checkpoint_path "$CHECKPOINT" \
 --metadata_path "$METADATA" --dataset_base_path "$DATASET_BASE" --output_dir "$OUTPUT_DIR" \
 --input_mode "$INPUT_MODE" --vae_encode_mode "$VAE_ENCODE_MODE" --start_index "${START_INDEX:-0}" --max_samples "${MAX_SAMPLES:-16}" \
 --seed "${SEED:-123}" --run_all_ablations "${KEEP_ARGS[@]}" "${NATIVE_ARGS[@]}" "${MAX_PIXEL_ARGS[@]}"
