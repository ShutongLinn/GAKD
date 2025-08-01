import json
import os

base_path_str = os.getenv("BASE_PATH_LIST")
base_path_list = base_path_str.split(":")

for base_path in base_path_list:
    input_path = base_path + "/" + "valid.jsonl"

    output_path = base_path + "/" + "qwen_valid.jsonl"

    template = (
        "<|im_start|>Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{inputs}\n\n### Response:\n<|im_end|><|im_start|>Assistant:\n<think>\n\n</think>\n\n"
    )

    template_no_input = (
        "<|im_start|>Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n<|im_end|><|im_start|>Assistant:\n<think>\n\n</think>\n\n"
    )
    
    with open(input_path, "r", encoding="utf-8") as infile, \
        open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            instruction = data.get("instruction", "")
            inputs = data.get("input", "")
            
            if inputs == "":
                data["prompt"] = template_no_input.format(instruction=instruction)
            else:
                data["prompt"] = template.format(instruction=instruction, inputs=inputs)
            
            if isinstance(data.get('output'), list) and len(data['output']) > 0:
                # if len(data['output']) > 1:
                #     print("list")
                data['output'] = data['output'][0]

            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"write sucessfully: {output_path}")
