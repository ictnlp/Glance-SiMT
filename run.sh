export CUDA_VISIBLE_DEVICES=0,1,2,3

TGT_FILE=L9_K8
MODELFILE=/data/guoshoutao/HMT_glancing_futture/checkpoints/${TGT_FILE}
DATAFILE=/data/guoshoutao/wmt15_de_en_bpe32k

FIRST_READ=9
CANDS_PER_TOKEN=8

python train.py --ddp-backend=no_c10d ${DATAFILE} --arch transformer --share-all-embeddings \
 --optimizer adam \
 --adam-betas '(0.9, 0.98)' \
 --clip-norm 0.0 \
 --lr 5e-4 \
 --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 \
 --warmup-updates 4000 \
 --dropout 0.3 \
 --encoder-attention-heads 8 \
 --decoder-attention-heads 8 \
 --criterion label_smoothed_cross_entropy \
 --label-smoothing 0.1 \
 --left-pad-source False \
 --save-dir ${MODELFILE} \
 --first-read ${FIRST_READ} \
 --cands-per-token ${CANDS_PER_TOKEN} \
 --max-tokens 4096 --update-freq 2 \
 --max-target-positions 200 \
 --curriculum-update 160000 \
 --min-future 0.05 \
 --skip-invalid-size-inputs-valid-test \
 --fp16 \
 --save-interval-updates 5000 \
 --keep-interval-updates 300 \
 --log-interval 10 > train_log/${TGT_FILE} 2>&1 &