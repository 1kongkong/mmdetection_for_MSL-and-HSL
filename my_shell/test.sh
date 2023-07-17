cd /home/bisifu/bsf/code/mmdetection3d

pcd_path=data/Titan/points
config=configs/kpfcnn/kpfcnn_2xb16-cosine-80e_titan-seg.py
work_dirs=work_dirs/kpfcnn_2xb16-cosine-80e_titan-seg/
check_point=${work_dirs}epoch_32.pth

python tools/test.py ${config} ${check_point} --show --show-dir ${work_dirs} --task lidar_seg

# pcd=${pcd_path}/area_2.bin
# save_file=${work_dirs}area_2
# python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${save_file} --snapshot

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