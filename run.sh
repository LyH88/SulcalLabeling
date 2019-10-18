root=/home/lyui/mount/nfs/Parcellation/ZALD_TTS 

export CUDA_VISIBLE_DEVICES=0

python train.py \
--batch-size 7 \
--test-batch-size 14 \
--epochs 100 \
--data_folder ${root}/aug/ \
--max_level 6 \
--min_level 0 \
--feat 16 \
--log_dir ${root}/model/8k/lh \
--log-interval 280 \
--decay \
--in_ch 3 \
