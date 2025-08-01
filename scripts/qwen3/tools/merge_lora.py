from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

def merge_lora_to_base_model(lora_model_path, output_path):
    peft_config = PeftConfig.from_pretrained(lora_model_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    lora_model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    merged_model = lora_model.merge_and_unload()
    
    merged_model.save_pretrained(output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"merge sucessfully, saved in {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a LoRA model into the base model.")
    parser.add_argument("--lora_model", type=str, required=True, help="Path to the LoRA model directory.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the merged model.")

    args = parser.parse_args()

    merge_lora_to_base_model(args.lora_model, args.output)