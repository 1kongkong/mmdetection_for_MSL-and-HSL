#!/usr/bin/env bash
cd /home/bisifu/bsf/code/mmdetection3d/
# CONFIG=configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_s3dis-seg.py
# CONFIG=configs/kpfcnn/kpfcnn_2xb16-cosine-80e_s3dis-seg.py
# CONFIG=configs/kpfcnn/kpfcnn_2xb16-cosine-80e_titan-seg.py
CONFIG=configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_titan-seg.py
# CONFIG=configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_hsl-seg.py
# CONFIG=/home/bisifu/bsf/code/mmdetection3d/configs/paconv/paconv_ssg-cuda_8xb8-cosine-80e_titan-seg.py
GPUS=1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.2"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
