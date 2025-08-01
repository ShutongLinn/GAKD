# GAKD: Generative Adversarial Knowledge Distillation

GAKD (Generative Adversarial Knowledge Distillation) is a framework that leverages adversarial training to enhance knowledge distillation from large teacher models to smaller student models. This repository provides code, data processing scripts, evaluation pipelines, and training configurations for reproducing the GAKD method and its baselines.

## 1 Environment

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 2 Data
### 2.1 Resources
The instruction-response datasets used for training and evaluation before processing can be downloaded from the following Hugging Face links. Please place them under `GAKD/data/`: [dolly](https://huggingface.co/datasets/MiniLLM/dolly), [self-inst](https://huggingface.co/datasets/MiniLLM/self-inst), [sinst](https://huggingface.co/datasets/MiniLLM/sinst), and [uinst](https://huggingface.co/datasets/MiniLLM/uinst)


### 2.2 Data Processing for training
After downloading the Dolly dataset, run the following script to tokenize and encode the training and validation splits. The processed data will be saved to `GAKD/processed_data/dolly/full`.

```bash
bash scripts/qwen3/tools/process_data_dolly.sh /PATH_TO/GAKD
```
### 2.2 Data Processing for evaluation
To align with the Qwen3 model's input format, evaluation datasets must be formatted using the Qwen3 chat template. Run the script below to process all evaluation sets:

```bash
bash scripts/qwen3/tools/process_data_eval.sh /PATH_TO/GAKD
```

## 3 Models
### 3.1 Base Pre-trained Models
For fine-tuning or baseline experiments, you need to download the base model(Qwen3) checkpoints from huggingface and place them in `GAKD/checkpoints/`. For example, for Qwen3-0.6B, you can download the model from this [link](https://huggingface.co/Qwen/Qwen3-0.6B) and put them in `checkpoints/Qwen3-0.6B`.

## 4 Run Evaluation
You can run evaluation on benchmarks DollEval, SelfInst, S-IN, UnNI, respectively. 

```bash
bash scripts/qwen3/eval/eval_all_dolly.sh /PATH_TO/GAKD
bash scripts/qwen3/eval/eval_all_self_inst.sh /PATH_TO/GAKD
bash scripts/qwen3/eval/eval_all_sinst.sh /PATH_TO/GAKD
bash scripts/qwen3/eval/eval_all_uinst.sh /PATH_TO/GAKD
```
Or run all evaluations at once:

```bash
bash scripts/qwen3/eval/run_all.sh /PATH_TO/GAKD
```

Evaluation results(e.g., BLEU, ROUGE) will be saved in the respective model's directory.

## 5 Train
We provide modular scripts to train both baseline methods and the proposed GAKD framework.
### 5.1 Baselines
Final checkpoints are selected based on Rouge-L scores on the validation set.
#### Fine-tune the teacher models
```bash
bash scripts/qwen3/sft/sft_8B.sh /PATH_TO/GAKD
```
#### SFT Baselines
Train student models via standard SFT:
```bash
bash scripts/qwen3/sft/sft_0.6B.sh /PATH_TO/GAKD
bash scripts/qwen3/sft/sft_1.7B.sh /PATH_TO/GAKD
bash scripts/qwen3/sft/sft_4B.sh /PATH_TO/GAKD
```

#### FKLD Baselines
Knowledge distillation using forward KL divergence:
```bash
bash scripts/qwen3/fkld/fkld_base_0.6B.sh /PATH_TO/GAKD
bash scripts/qwen3/fkld/fkld_base_1.7B.sh /PATH_TO/GAKD
bash scripts/qwen3/fkld/fkld_base_4B.sh /PATH_TO/GAKD
```

#### RKLD Baselines
Knowledge distillation using reverse KL divergence:
```bash
bash scripts/qwen3/rkld/rkld_base_0.6B.sh /PATH_TO/GAKD
bash scripts/qwen3/rkld/rkld_base_1.7B.sh /PATH_TO/GAKD
bash scripts/qwen3/rkld/rkld_base_4B.sh /PATH_TO/GAKD
```

### 5.2 GAKD
#### Initial Checkpoints
First, train initial student models using SFT. These serve as the starting point for adversarial distillation. The final checkpoints are selected by the **validation loss**.
```bash
bash scripts/qwen3/sft/sft_0.6B.sh /PATH_TO/GAKD
bash scripts/qwen3/sft/sft_1.7B.sh /PATH_TO/GAKD
bash scripts/qwen3/sft/sft_4B.sh /PATH_TO/GAKD
```

#### Train
The final checkpoints are selected by the Rouge-L scores.
```bash
bash scripts/qwen3/gakd_rkl/rgan_rkld_0.6B.sh /PATH_TO/GAKD
bash scripts/qwen3/gakd_rkl/rgan_rkld_1.7B.sh /PATH_TO/GAKD
bash scripts/qwen3/gakd_rkl/rgan_rkld_4B.sh /PATH_TO/GAKD
```

### 5.3 Others
You can run our GAKD flow in other GAN variants
+ To run our GAKD flow using standard GAN
```bash
bash scripts/qwen3/gakd_rkl/gan_rkl_0.6B.sh /PATH_TO/GAKD
bash scripts/qwen3/gakd_rkl/gan_rkl_1.7B.sh /PATH_TO/GAKD
bash scripts/qwen3/gakd_rkl/gan_rkl_4B.sh /PATH_TO/GAKD
```

+ To run our GAKD flow using WGAN-GP
```bash
bash scripts/qwen3/gakd_rkl/wgan_rkl_0.6B.sh /PATH_TO/GAKD
bash scripts/qwen3/gakd_rkl/wgan_rkl_1.7B.sh /PATH_TO/GAKD
bash scripts/qwen3/gakd_rkl/wgan_rkl_4B.sh /PATH_TO/GAKD
```
