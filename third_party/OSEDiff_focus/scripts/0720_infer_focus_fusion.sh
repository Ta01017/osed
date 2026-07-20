#!/usr/bin/env bash
set -euo pipefail
: "${CHECKPOINT:?set CHECKPOINT to a focus-fusion .pt checkpoint}"
GPU="${GPU:-0}"; METADATA="${METADATA:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/focus_merged_6000_dedup_0710_v3/train/metadata_with_homography_warped_focus_ckptA.json}"
DATASET_BASE="${DATASET_BASE:-/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880/dataset/focus_merged_6000_dedup_0710_v3/train}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-stabilityai/stable-diffusion-2-1-base}"; OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/focus_fusion_infer}"
KEEP_ARGS=(); if [[ "${KEEP_A_COMPOSITE:-0}" == "1" ]]; then KEEP_ARGS+=(--keep_a_composite); fi
CUDA_VISIBLE_DEVICES="$GPU" python test_osediff_focus_fusion.py --pretrained_model_name_or_path "$PRETRAINED_MODEL" --checkpoint_path "$CHECKPOINT" \
 --metadata_path "$METADATA" --dataset_base_path "$DATASET_BASE" --output_dir "${OUTPUT_ROOT}/$(date +%Y%m%d_%H%M%S)" \
 --start_index "${START_INDEX:-0}" --max_samples "${MAX_SAMPLES:-16}" --seed "${SEED:-123}" --run_all_ablations "${KEEP_ARGS[@]}"
