cd /home/bisifu/bsf/code/mmdetection3d

# dataset_name=hsl
# root_path=/home/bisifu/bsf/code/mmdetection3d/data/mmdetection3d_data/HSL
# out_dir=/home/bisifu/bsf/code/mmdetection3d/data/mmdetection3d_data/HSL
# extra_tag=hsl

# python -u tools/create_data.py $dataset_name \
#         --root-path $root_path \
#         --out-dir $out_dir \
#         --extra-tag $extra_tag

# dataset_name=titan
# root_path=/home/bisifu/bsf/code/mmdetection3d/data/Titan
# out_dir=/home/bisifu/bsf/code/mmdetection3d/data/Titan
# extra_tag=titan

# python -u tools/create_data.py $dataset_name \
#         --root-path $root_path \
#         --out-dir $out_dir \
#         --extra-tag $extra_tag

dataset_name=SensatUrban
root_path=/home/bisifu/bsf/code/mmdetection3d/data/SensatUrban
out_dir=/home/bisifu/bsf/code/mmdetection3d/data/SensatUrban
extra_tag=sensaturban

python -u tools/create_data.py $dataset_name \
        --root-path $root_path \
        --out-dir $out_dir \
        --extra-tag $extra_tag