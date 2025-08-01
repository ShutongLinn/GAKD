#!bash

lora_paths=(
    "./checkpoints/Qwen3-1.7B"
)

for path in "${lora_paths[@]}"; do
    echo "Processing $path ..."
    python ./merge_lora.py --lora_model "$path" --output "$path"_merge
done