#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=29878
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/GAKD"}
CKPT_NAME="Qwen3-4B_init"
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}"

DISCRIM="others" #type: SimpleMLPDiscriminator/MLPEmbeddingLLMDiscriminator/WGANLLMDiscriminator/others
DISCRIM_NAME="Qwen3-0.6B"
DISCRIM_PATH="${BASE_PATH}/checkpoints/${DISCRIM_NAME}"

TEACHER_CKPT_NAME="Qwen3-8B"
TEACHER_CKPT="${BASE_PATH}/checkpoints/${TEACHER_CKPT_NAME}"
TEACHER_LORA="${BASE_PATH}/checkpoints/Qwen3-8B-sft-teacher"

# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/qwen3/"
MP_SIZE=1
# hp
BATCH_SIZE=2
# LR=5e-5
LR=1e-5
# LR=5e-6
GRAD_ACC=16
EVAL_BATCH_SIZE=64
# length
MAX_LENGTH=512

KD_RETIO=0.5
DS_SCALE=15
G_SCALE=0.1

SAVE_PATH="${BASE_PATH}/results/rgan-fkld/${CKPT_NAME}_lr_${LR}"
# seed
SEED=42


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --discrim-path ${DISCRIM_PATH}"
OPTS+=" --discrim ${DISCRIM}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type qwen3"
OPTS+=" --peft lora"
OPTS+=" --peft-lora-r 16"
OPTS+=" --teacher-peft-path ${TEACHER_LORA}"
# OPTS+=" --peft-path ${STUDENT_LORA}"
# OPTS+=" --gradient-checkpointing"
# OPTS+=" --model-parallel"
# OPTS+=" --model-parallel-size ${MP_SIZE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 1000"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --generator_scale ${G_SCALE}"
OPTS+=" --ds_scale ${DS_SCALE}"
OPTS+=" --critic_scale 1000"
OPTS+=" --critic_it 1"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 10"
OPTS+=" --kd-ratio ${KD_RETIO}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2_bf16.json"
# type
OPTS+=" --type rgan_fkld"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
# export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune_rgan_fkld.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
