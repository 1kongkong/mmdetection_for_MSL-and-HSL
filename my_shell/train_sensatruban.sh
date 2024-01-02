#!/usr/bin/env bash

curdir=$(cd $(dirname $0); pwd)
workdir=$(cd "$curdir/.."; pwd)
echo "$workdir"
export PYTHONPATH=$PYTHONPATH:$workdir
cd $workdir

# echo "sleep 3h!"
# sleep 3h

# CONFIG=configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_s3dis-seg.py
# CONFIG=configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_sensaturban-seg.py
# CONFIG=configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_hsl-seg.py

pointnet2=configs/pointnet2/pointnet2_msg_1xb6-cosine-100e_sensaturban-seg.py
dgcnn=configs/dgcnn/dgcnn_1xb6-cosine-100e_sensaturban-seg.py
kpfcnn=configs/kpfcnn/kpfcnn_1xb6-cosine-100e_vs_sensaturban-seg.py
kpfcnn_ad=configs/kpfcnn/kpfcnn_1xb6-cosine-100e_sensaturban-seg.py
dual_kpfcnn=configs/dual_kpfcnn/dual_kpfcnn_1xb6-cosine-100e_sensaturban-seg.py
dual_kpfcnn_gr=configs/dual_kpfcnn/dual_kpfcnn_1xb6-cosine-100e_gr_sensaturban-seg.py
paconv=configs/paconv/paconv_ssg-cuda_8xb8-cosine-100e_sensaturban-seg.py

# CONFIG=configs/paconv/paconv_ssg-cuda_8xb8-cosine-80e_sensaturban-seg.py

GPU_num=$(nvidia-smi -L | wc -l)
GPU_used=$(expr $GPU_num - 3)
for CONFIG in {${dual_kpfcnn_gr},}
do 
    CUDA_VISIBLE_DEVICES=$GPU_used \
    python tools/train.py \
        $CONFIG \
        # --amp \
        # --auto-scale-lr 
    sleep 2s
done

# sleep 2s

# CONFIG=configs/kpfcnn/kpfcnn_1xb6-cosine-200e_sensaturban-seg.py
# python tools/train.py \
#     $CONFIG \
#     # --resume /home/bisifu/bsf/code/mmdetection3d/work_dirs/kpfcnn_1xb6-cosine-200e_sensaturban-seg/epoch_20.pth \
#     # --amp \
#     # --auto-scale-lr 

# sleep 2s

# CONFIG=configs/kpfcnn/kpfcnn_1xb6-cosine-200e_vs_sensaturban-seg.py
# python tools/train.py \
#     $CONFIG \
#     # --amp \
#     # --auto-scale-lr 