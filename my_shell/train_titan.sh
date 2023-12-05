#!/usr/bin/env bash
cd /home/bisifu/bsf/code/mmdetection3d/

# echo "sleep 3h!"
# sleep 3h

# CONFIG=configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_s3dis-seg.py
# CONFIG=configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_titan-seg.py
# CONFIG=configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_hsl-seg.py

# CONFIG=configs/kpfcnn/kpfcnn_1xb6-cosine-80e_s3dis-seg.py
pointnet2=configs/pointnet2/pointnet2_msg_1xb6-cosine-100e_titan-seg.py
dgcnn=configs/dgcnn/dgcnn_1xb6-cosine-100e_titan-seg.py
kpfcnn=configs/kpfcnn/kpfcnn_1xb6-cosine-100e_vs_titan-seg.py
kpfcnn_ad=configs/kpfcnn/kpfcnn_1xb6-cosine-100e_titan-seg.py
dual_kpfcnn=configs/dual_kpfcnn/dual_kpfcnn_1xb6-cosine-100e_titan-seg.py
paconv=configs/paconv/paconv_ssg-cuda_8xb8-cosine-100e_titan-seg.py
point_transformer=configs/point_transformer/point_transformer_1xb8-cosine-100e_titan-seg.py
point_transformer_=configs/point_transformer/point_transformer_1xb8-cosine-150e_titan-seg.py

# CONFIG=configs/paconv/paconv_ssg-cuda_8xb8-cosine-80e_titan-seg.py


for CONFIG in {${point_transformer_},}
do 
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