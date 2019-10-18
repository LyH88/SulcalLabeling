root=/home/lyui/mount/nfs/Parcellation/Sulcus 

fold=$1

export CUDA_VISIBLE_DEVICES=0

python train_sulc.py \
--batch-size 4 \
--test-batch-size 10 \
--epochs 100 \
--data_folder ${root}/aug_bin/ \
--max_level 5 \
--min_level 0 \
--feat 64 \
--log_dir ${root}/model/cv/$fold/ico5_64_cross \
--log-interval 16 \
--fold $fold \
--decay \
--in_ch 3 \
