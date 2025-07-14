#! /bin/bash

JSON_PATH="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_test_gemini_rawcap.json"
MODEL_NAME="gemini-2.0-flash-lite"
OUTPUT_DIR="baselines/gemini/gem2.0_flashlite_test_gemrawcap.json"

python baselines/gemini/gemini_desc.py \
--json_path $JSON_PATH \
--model $MODEL_NAME \
--output_dir $OUTPUT_DIR \
--logging_steps 10