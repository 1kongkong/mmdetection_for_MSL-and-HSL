curdir=$(cd $(dirname $0); pwd)
workdir=$(cd "$curdir/.."; pwd)
echo "$workdir"
export PYTHONPATH=$PYTHONPATH:$workdir
cd $workdir

pcd_path=data/SensatUrban/test
# work_dirs/dual_kpfcnn_1xb6-cosine-100e_sensaturban-seg/20231203_103749/dual_kpfcnn_1xb6-cosine-100e_sensaturban-seg/best_miou_epoch_54.pth
model_type=dual_kpfcnn_1xb6-cosine-100e_sensaturban-seg
train_date=20231203_103749
model_pth=best_miou_epoch_54

work_dirs=work_dirs/${model_type}/${train_date}
config=work_dirs/${model_type}/${model_type}.py
check_point=${work_dirs}/${model_type}/${model_pth}.pth

# python tools/test.py ${config} ${check_point} --show --show-dir ${work_dirs} --task lidar_seg

for i in {birmingham_block_2,birmingham_block_8,cambridge_block_15,cambridge_block_16,cambridge_block_22,cambridge_block_27}
do 
    pcd=${pcd_path}/${i}.bin
    save_file=${work_dirs}/${i}
    python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${save_file} --snapshot
done



# pcd=${pcd_path}/area_2.bin
# save_file=${work_dirs}/area_2
# python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${save_file} --snapshot
