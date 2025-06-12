#!/bin/bash
PATH_TO_FOLDERS="/media02/nthuy/SnapUGC/SnapUGC_0"
TRAIN_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_train_engcaption_image.json"
EVAL_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_val_engcaption_image.json"

OUTPUT_DIR="./checkpoints/longvu_llama_snapugc0_txt2txt_resume_again"

CKPT_NAME="longvu_llama_snapugc0_txt2txt_resume_again"
PREV_STAGE_CHECKPOINT=""
MODEL_PATH="./checkpoints/longvu_llama3_2"
VERSION="llama3"

torchrun --nproc_per_node=2 --master_port=29505 models/train.py \
  --output_dir $OUTPUT_DIR \
  --input_model_filename $MODEL_PATH \
  --output_model_filename $OUTPUT_DIR \
  --checkpoint_fname $CKPT_NAME \
  --image_folders $PATH_TO_FOLDERS \
  --train_paths $TRAIN_PATHS \
  --eval_paths $EVAL_PATHS \
  --train_log "train_log.json" \
  --train_perf_log "train_perf.json" \
  --eval_perf_log "eval_perf.json" \
  --eval_log "eval_log.json" \
  --model_max_length 8192 \
  --fp16 False \
  --bf16 True \
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
  --tune_lm_head True \
  --cls_only False \
  --tune_embed_tokens False \
  --freeze_mm_mlp_adapter False \
  --freeze_backbone True \
  --gradient_checkpointing True \
  --generation_eval False \
  --mm_projector_type sva \
  --image_token_len 144 \
  --query_num_list "[144]" \
  --lowres_token 8 \
  --video_fps 1 \
  --highres True \
  --drop_threshold 0.8 \
  --eval_strategy "steps" \
  --eval_steps 76 \
  --save_strategy "epoch" \
  --save_steps 189 \
  --logging_steps 10 \
  --num_train_epochs 1 \
  --warmup_ratio 0.03 \
  --learning_rate 1e-5 \
  --weight_decay 0. \
  --cls_loss_weight 0.5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 32 \
  # --resume_from_checkpoint $PREV_STAGE_CHECKPOINT
