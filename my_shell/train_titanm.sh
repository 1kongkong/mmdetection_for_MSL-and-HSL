#!/usr/bin/env bash

curdir=$(cd $(dirname $0); pwd)
workdir=$(cd "$curdir/.."; pwd)
echo "$workdir"
export PYTHONPATH=$PYTHONPATH:$workdir
cd $workdir

# echo "sleep 4h!"
# sleep 4h

kpcross_kpfcnn_gr_vca=configs/dual_kpfcnn/kpcross_dual_kpfcnn_1xb6-cosine-100e_gr_titan-seg_vca.py
cross_kpfcnn_gr_vca=configs/dual_kpfcnn/cross_dual_kpfcnn_1xb6-cosine-100e_gr_titan-seg_vca.py
idw_kpfcnn_gr_vca=configs/dual_kpfcnn/idw_dual_kpfcnn_1xb6-cosine-100e_gr_titan-seg_vca.py
nn_kpfcnn_gr_vca=configs/dual_kpfcnn/nn_dual_kpfcnn_1xb6-cosine-100e_gr_titan-seg_vca.py
cross2_kpfcnn_gr_vca=configs/dual_kpfcnn/cross2_dual_kpfcnn_1xb6-cosine-100e_gr_titan-seg_vca.py

GPU_num=$(nvidia-smi -L | wc -l)
GPU_used=$(expr $GPU_num - 2)
for CONFIG in {${cross2_kpfcnn_gr_vca},}
do 
    CUDA_VISIBLE_DEVICES=$GPU_used \
    python tools/train.py \
        $CONFIG \
        # --amp \
        # --auto-scale-lr 
    sleep 2s
done

# sleep 2s

# CONFIG=configs/kpfcnn/kpfcnn_1xb6-cosine-200e_titan-seg.py
# python tools/train.py \
#     $CONFIG \
#     # --resume /home/bisifu/bsf/code/mmdetection3d/work_dirs/kpfcnn_1xb6-cosine-200e_titan-seg/epoch_20.pth \
#     # --amp \
#     # --auto-scale-lr 

# sleep 2s

# CONFIG=configs/kpfcnn/kpfcnn_1xb6-cosine-200e_vs_titan-seg.py
# python tools/train.py \
#     $CONFIG \
#     # --amp \
#     # --auto-scale-lr 