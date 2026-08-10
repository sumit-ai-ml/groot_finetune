#!/usr/bin/env bash
# Phase 4 fine-tune. Run under tmux so a dropped connection cannot kill it:
#   tmux new -d -s ft 'bash scripts/train.sh'
#
# MAX_STEPS and GLOBAL_BATCH_SIZE are overridable, because the wall-clock gate
# at step 100 may tell us to halve the batch or cut the step count.
set -euo pipefail

source /home/sushi/Documents/groot_finetune/scripts/env.sh

MAX_STEPS="${MAX_STEPS:-3000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-$GROOT_PROJECT/checkpoints/so101_ft}"
LOG="${LOG:-$GROOT_PROJECT/runs/phase4/train.log}"

# --save_total_limit 10 is load-bearing: it defaults to 5, and with six
# checkpoints (500..3000) the Trainer prunes oldest-first, silently deleting
# checkpoint-500 -- one of the four this study needs.
#
# --save_only_model halves each checkpoint from 24G to 12G by dropping optimizer
# state. We never resume from these; they are only ever loaded for evaluation
# and quantization.
CUDA_VISIBLE_DEVICES=0 exec python gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.7-3B \
    --dataset_path ./demo_data/so101-table-cleanup-train \
    --embodiment_tag NEW_EMBODIMENT \
    --modality_config_path examples/SO100/so100_config.py \
    --num_gpus 1 \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS" \
    --global_batch_size "$GLOBAL_BATCH_SIZE" \
    --dataloader_num_workers 8 \
    --save_steps 500 \
    --save_total_limit 10 \
    --save_only_model \
    > "$LOG" 2>&1
