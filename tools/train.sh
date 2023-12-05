#!/usr/bin/env bash
cd /home/bisifu/bsf/code/mmdetection3d/

# echo "sleep 3h!"
# sleep 3h

# CONFIG=configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_s3dis-seg.py
# CONFIG=configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_titan-seg.py
# CONFIG=configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_hsl-seg.py

# CONFIG=configs/kpfcnn/kpfcnn_1xb6-cosine-80e_s3dis-seg.py
# CONFIG=configs/kpfcnn/kpfcnn_1xb6-cosine-200e_vs_titan-seg.py
# CONFIG=configs/kpfcnn/kpfcnn_1xb6-cosine-200e_titan-seg.py

CONFIG=configs/dual_kpfcnn/dual_kpfcnn_1xb6-cosine-200e_titan-seg.py

# CONFIG=configs/paconv/paconv_ssg-cuda_8xb8-cosine-80e_titan-seg.py

python tools/train.py \
    $CONFIG \
    # --amp \
    # --auto-scale-lr 

sleep 2s

CONFIG=configs/kpfcnn/kpfcnn_1xb6-cosine-200e_titan-seg.py
python tools/train.py \
    $CONFIG \
    # --amp \
    # --auto-scale-lr 

sleep 2s

CONFIG=configs/kpfcnn/kpfcnn_1xb6-cosine-200e_vs_titan-seg.py
python tools/train.py \
    $CONFIG \
    # --amp \
    # --auto-scale-lr 