
ts=`date +%Y%m%d%-H%M`
dataset=reddit

#batch_sizeはcustom embでは30
CUDA_VISIBLE_DEVICES=6,7 ${PBIN_DIR}/python run.py \
    --task ${dataset} \
    --batch_size 30 \
    --eval_steps 40000 \
    --emb_len 200 \
    --max_utterances 29 \
    --learning_rate 5e-4\
    --max_words 50 \
    --n_gpu 2 \
    --epochs 100 \
    --n_layer 3 \
    --max_hop 2 \
    --score_file_path score_file.txt \
    --model_file_name ${dataset}_impchat.pt \
    --is_training True