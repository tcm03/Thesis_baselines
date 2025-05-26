#!/bin/bash
PATH_TO_FOLDERS="/media02/nthuy/SnapUGC/SnapUGC_0"
TRAIN_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_train_engcaption_cls.json"
EVAL_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_val_engcaption_cls.json"

OUTPUT_DIR="./checkpoints/longvu_llama_snapugc0_txtclsonly0"

CKPT_NAME="longvu_llama_snapugc0_txtclsonly0"
# PREV_STAGE_CHECKPOINT="./checkpoints/longvu_llama_snapugc0_txtcls0/longvu_llama_snapugc0_txtcls0-epoch0-step379.pt"
MODEL_PATH="./checkpoints/longvu_llama3_2"
VERSION="llama3"

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=2 --master_port=29503 models/train.py \
  --output_dir $OUTPUT_DIR \
  --input_model_filename $MODEL_PATH \
  --output_model_filename $OUTPUT_DIR \
  --checkpoint_fname $CKPT_NAME \
  --image_folders $PATH_TO_FOLDERS \
  --train_paths $TRAIN_PATHS \
  --eval_paths $EVAL_PATHS \
  --train_log "train_log_txtcls_2.json" \
  --train_perf_log "train_perf_txtcls_2.json" \
  --eval_perf_log "eval_perf_txtcls_2.json" \
  --model_max_length 8192 \
  --fp16 True \
  --bf16 False \
  --tf32 False \
  --log_on_each_node False \
  --logging_dir /tmp/llava/test/ \
  --report_to "tensorboard" \
  --save_total_limit 1 \
  --version $VERSION \
  --mm_vision_select_layer "-2" \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --lazy_preprocess True \
  --tune_mm_mlp_adapter True \
  --tune_lm_head False \
  --tune_cls_head True \
  --cls_only False \
  --tune_embed_tokens False \
  --freeze_mm_mlp_adapter False \
  --freeze_backbone True \
  --gradient_checkpointing True \
  --mm_projector_type sva \
  --image_token_len 144 \
  --query_num_list "[144]" \
  --lowres_token 8 \
  --video_fps 1 \
  --highres True \
  --drop_threshold 0.8 \
  --eval_strategy "steps" \
  --eval_steps 2 \
  --save_strategy "steps" \
  --save_steps 200 \
  --logging_steps 2 \
  --num_train_epochs 2 \
  --warmup_ratio 0.03 \
  --learning_rate 5e-6 \
  --weight_decay 0. \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  # --resume_from_checkpoint $PREV_STAGE_CHECKPOINT
