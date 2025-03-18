DATA_PATH="/raid/nthuy/SnapUGC"
JSON_PATH="/raid/nthuy/SnapUGC/snapugc_60s_4eval_train.json"

python main.py \
--data_path $DATA_PATH \
--json_path $JSON_PATH \
--output_file "test.safetensors" \
--config_file "config_llama.json" \
--batch_size 4 \
