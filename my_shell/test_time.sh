# echo "Hello there!"
# sleep 1m
# echo "Oops! I fell asleep for a couple seconds!"
dual_kpfcnn=configs/dual_kpfcnn/dual_kpfcnn_1xb6-cosine-200e_titan-seg.py
kpfcnn=configs/kpfcnn/kpfcnn_1xb6-cosine-200e_titan-seg.py
for config in {${dual_kpfcnn},${kpfcnn}}
do 
    echo $config
done