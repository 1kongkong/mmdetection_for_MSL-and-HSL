curdir=$(cd $(dirname $0); pwd)
workdir=$(cd "$curdir/.."; pwd)
echo "$workdir"
export PYTHONPATH=$PYTHONPATH:$workdir
cd $workdir

pcd_path=data/Titan_M/points
# work_dirs/cross_dual_kpfcnn_1xb6-cosine-100e_gr_titan-seg_vca/20240123_181303/cross_dual_kpfcnn_1xb6-cosine-100e_gr_titan-seg_vca/best_acc_epoch_94.pth
model_type=cross_dual_kpfcnn_1xb6-cosine-100e_gr_titan-seg_vca
train_date=20240123_181303
model_pth=best_acc_epoch_94

work_dirs=work_dirs/${model_type}/${train_date}
config=work_dirs/${model_type}/${train_date}/vis_data/config.py
check_point=${work_dirs}/${model_type}/${model_pth}.pth

# python tools/test.py ${config} ${check_point} --show --show-dir ${work_dirs} --task lidar_seg

for i in {2,7,12,14,18,21,24,27}
do 
    pcd=${pcd_path}/area_${i}.bin
    save_file=${work_dirs}/area_${i}
    CUDA_VISIBLE_DEVICES=0 \
    python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${save_file} --snapshot
done

python my_tools/offline_test.py --path ${work_dirs} --dataset titanm
python my_tools/draw_seg_mask.py --path ${work_dirs} --dataset titan