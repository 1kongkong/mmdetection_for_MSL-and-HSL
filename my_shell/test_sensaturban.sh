curdir=$(cd $(dirname $0); pwd)
workdir=$(cd "$curdir/.."; pwd)
echo "$workdir"
export PYTHONPATH=$PYTHONPATH:$workdir
cd $workdir

pcd_path=data/SensatUrban/test
# work_dirs/dual_kpfcnn_1xb6-cosine-100e_gr_sensaturban-seg_vca/20240104_172739/dual_kpfcnn_1xb6-cosine-100e_gr_sensaturban-seg_vca/best_acc_epoch_100.pth
model_type=dual_kpfcnn_1xb6-cosine-100e_gr_sensaturban-seg_vca
train_date=20240104_172739
model_pth=best_acc_epoch_100

work_dirs=work_dirs/${model_type}/${train_date}
config=work_dirs/${model_type}/${train_date}/vis_data/config.py
check_point=${work_dirs}/${model_type}/${model_pth}.pth


# for i in {cambridge_block_22,cambridge_block_27}
# do 
#     pcd=${pcd_path}/${i}.bin
#     save_file=${work_dirs}/${i}
#     CUDA_VISIBLE_DEVICES=1 \
#     python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${save_file} --snapshot
# done

python my_tools/offline_test.py --path ${work_dirs} --dataset sensaturban
python my_tools/draw_seg_mask.py --path ${work_dirs} --dataset sensaturban

# pcd=${pcd_path}/area_2.bin
# save_file=${work_dirs}/area_2
# python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${save_file} --snapshot
