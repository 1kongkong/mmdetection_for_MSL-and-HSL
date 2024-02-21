curdir=$(cd $(dirname $0); pwd)
workdir=$(cd "$curdir/.."; pwd)
echo "$workdir"
export PYTHONPATH=$PYTHONPATH:$workdir
cd $workdir

pcd_path=data/Titan/points
# work_dirs/dual_kpfcnn_1xb6-cosine-100e_gr_titan-seg_vca/20240105_101628/dual_kpfcnn_1xb6-cosine-100e_gr_titan-seg_vca/best_miou_epoch_36.pth
model_type=dual_kpfcnn_1xb6-cosine-100e_gr_titan-seg_vca
train_date=20240105_101628
model_pth=best_miou_epoch_36

work_dirs=work_dirs/${model_type}/${train_date}
config=work_dirs/${model_type}/${train_date}/vis_data/config.py
check_point=${work_dirs}/${model_type}/${model_pth}.pth


for i in {2,7,12,14,18,21,24,27}
do 
    pcd=${pcd_path}/area_${i}.bin
    save_file=${work_dirs}/area_${i}
    CUDA_VISIBLE_DEVICES=1 \
    python demo/pcd_seg_demo.py ${pcd} ${config} ${check_point} --out-dir ${save_file} --snapshot
done

python my_tools/offline_test.py --path ${work_dirs} --dataset titan
python my_tools/draw_seg_mask.py --path ${work_dirs} --dataset titan