torchrun --nproc_per_node=4 --master_port=29503 models/test.py \
--model_path ./checkpoints/test_w0.25_txtcls \
--data_path ./data/test.json \
--model_max_length 8192 \
--version llama3 \
# --cls_only