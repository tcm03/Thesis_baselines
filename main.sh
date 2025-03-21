DATA_PATH="/raid/nthuy/SnapUGC"
TRAIN_PATH="/raid/nthuy/SnapUGC/snapugc_30s_train_short.json"
EVAL_PATH="/raid/nthuy/SnapUGC/snapugc_30s_test_short.json"
MODEL_PATH="./checkpoints/longvu_llama3_2/pytorch_model.bin"

python main.py \
--data_path $DATA_PATH \
--train_path $TRAIN_PATH \
--eval_path $EVAL_PATH \
--model_path $MODEL_PATH \
--output_file "test.safetensors" \
--config_file "config_llama.json" \
--batch_size 4 \
