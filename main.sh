DATA_PATH="/raid/nthuy/SnapUGC/train /raid/nthuy/SnapUGC/test"

python preprocessing/main.py \
--data $DATA_PATH \
--output_file "test.safetensors" \
--config_file "preprocessing/config_llama.json" \
--batch_size 4 \
