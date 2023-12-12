curdir=$(cd $(dirname $0); pwd)
workdir=$(cd "$curdir/.."; pwd)
echo "$workdir"
export PYTHONPATH=$PYTHONPATH:$workdir
cd $workdir

pcd_path=data/Titan/points
# work_dirs/dgcnn_1xb6-cosine-100e_titan-seg/20231121_092442/dgcnn_1xb6-cosine-100e_titan-seg/best_miou_epoch_54.pth
model_type=dgcnn_1xb6-cosine-100e_titan-seg
train_date=20231121_092442
model_pth=best_miou_epoch_54

work_dirs=work_dirs/${model_type}/${train_date}
config=work_dirs/${model_type}/${model_type}.py
check_point=${work_dirs}/${model_type}/${model_pth}.pth

# python tools/test.py ${config} ${check_point} --show --show-dir ${work_dirs} --task lidar_seg

for i in {2,7,12,14,18,21,24,27}
do 
    pcd=${pcd_path}/area_${i}.bin
    save_file=${work_dirs}/area_${i}
    python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${save_file} --snapshot
done



# pcd=${pcd_path}/area_2.bin
# save_file=${work_dirs}/area_2
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