#!/bin/bash
# PATH_TO_FOLDERS="/media02/nthuy/SnapUGC/SnapUGC_0"
PATH_TO_FOLDERS="/media02/nthuy/EnTube/2021"

# we just care about the test paths, but the current test code use --eval_paths for convenience
# TEST_PATHS="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_nano_test_engcaption_cls.json"
TEST_PATHS="/media02/nthuy/EnTube/entube_2021.json"

OUTPUT_DIR="./checkpoints/entube_2021"

CKPT_NAME="nothing"
PREV_STAGE_CHECKPOINT="./checkpoints/final_clsonly_0/final_clsonly_0-epoch1-final.pt"
MODEL_PATH="./checkpoints/longvu_llama3_2"
VERSION="llama3"

torchrun --nproc_per_node=2 --master_port=29503 models/test_2.py \
  --output_dir $OUTPUT_DIR \
  --input_model_filename $MODEL_PATH \
  --output_model_filename $OUTPUT_DIR \
  --checkpoint_fname $CKPT_NAME \
  --image_folders $PATH_TO_FOLDERS \
  --train_paths $TEST_PATHS \
  --eval_paths $TEST_PATHS \
  --test_paths $TEST_PATHS \
  --eval_perf_log "test_perf_txtcls.json" \
  --eval_log "test_log_txtcls.json" \
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
  --tune_cls_head True \
  --cls_only True \
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
  --save_strategy "steps" \
  --save_steps 189 \
  --logging_steps 10 \
  --num_train_epochs 2 \
  --warmup_ratio 0.03 \
  --learning_rate 3e-5 \
  --weight_decay 0. \
  --cls_loss_weight 0.5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --resume_from_checkpoint $PREV_STAGE_CHECKPOINT
