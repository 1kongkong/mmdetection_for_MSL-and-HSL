cd /home/bisifu/bsf/code/mmdetection3d

pcd_path=data/mmdetection3d_data/HSL/points
# config=configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_hsl-seg.py
config=configs/kpfcnn/kpfcnn_2xb16-cosine-80e_hsl-seg.py
# work_dirs=work_dirs/pointnet2_msg_2xb16-cosine-80e_hsl-seg/
work_dirs=work_dirs/kpfcnn_2xb16-cosine-80e_hsl-seg/
check_point=${work_dirs}epoch_75.pth

# python tools/test.py ${config} ${check_point} --show --show-dir ${work_dirs} --task lidar_seg

# pcd=${pcd_path}/area_1.bin
# save_file=${work_dirs}area_1
# python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${save_file} --snapshot

for i in {1..12}
do 
    pcd=${pcd_path}/area_${i}.bin
    save_file=${work_dirs}area_${i}
    python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${save_file} --snapshot
done

# pcd=${pcd_path}/area_7.bin
# python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${work_dirs} --snapshot

# pcd=${pcd_path}/area_12.bin
# python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${work_dirs} --snapshot

# pcd=${pcd_path}/area_14.bin
# python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${work_dirs} --snapshot

# pcd=${pcd_path}/area_18.bin
# python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${work_dirs} --snapshot

# pcd=${pcd_path}/area_21.bin
# python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${work_dirs} --snapshot

# pcd=${pcd_path}/area_24.bin
# python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${work_dirs} --snapshot

# pcd=${pcd_path}/area_27.bin
# python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${work_dirs} --snapshot