python3 tools/convert_mp.py \
    --input_path results/qwen3/train/sft/Qwen3-8B/teacher_model \
    --source_mp_size 1 \
    --target_mp_size 2 \
    --model_type qwen2 # choose from opt and llama