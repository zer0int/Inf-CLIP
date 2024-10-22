# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=${3:-0}

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=262144
LOCAL_BATCH_SIZE=512
ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]
EPOCHS=8
TRAIN_NUM_SAMPLES=280321756
WARMUP_STEPS=$[$TRAIN_NUM_SAMPLES/(2*$GLOBAL_BATCH_SIZE)]
echo "ACCUMULATION_STEPS: $ACCUMULATION_STEPS"

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=lit_laion400m
RUN_NAME=lit-b-16_bs256k_e8
DATA_DIR=datasets
OUTP_DIR=work_dirs


torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    -m inf_clip.train.main \
    --model LiT-B-16 \
    --train-data ${DATA_DIR}'/laion400m/{00000..41407}.tar' \
    --train-num-samples $TRAIN_NUM_SAMPLES \
    --aug-cfg scale='(0.08, 1.0)'\
    --dataset-type webdataset \
    --imagenet-val ${DATA_DIR}/imagenet-1k/val \
    --epochs $EPOCHS \
    --warmup $WARMUP_STEPS \
    --batch-size $LOCAL_BATCH_SIZE \
    --accum-freq $ACCUMULATION_STEPS \
    --optim adafactor \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.95 \
    --eps 1.0e-8 \
    --wd 1e-4 \
    --grad-clip-norm 1.0 \
    --workers 16 \
    --precision amp \
    --infloss \
    --log-every-n-steps 1 \
    --logs $OUTP_DIR/$WANDB_PROJECT \
    --name $RUN_NAME \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --resume latest \
