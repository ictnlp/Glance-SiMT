# Glance-SiMT
Source code for the paper ["Glancing Future for Simultaneous Machine Translation"](https://arxiv.org/abs/2309.06179)

Our method is implemented based on the open-source toolkit [Fairseq](https://github.com/facebookresearch/fairseq) .

## Requirements and Installation

* Python version = 3.6

* PyTorch version = 1.7

* Install fairseq:

```
git clone https://github.com/ictnlp/Glance-SiMT.git
cd Glance-SiMT
pip install --editable ./
```

## Quick Start

### Data Pre-processing

We use the data of IWSLT15 English-Vietnamese (download [here](https://nlp.stanford.edu/projects/nmt/)), WMT15 German-English (download [here](www.statmt.org/wmt15)).

For WMT15 German-English, we tokenize the corpus via [mosesdecoder/scripts/tokenizer/normalize-punctuation.perl](https://github.com/moses-smt/mosesdecoder) and apply BPE with 32K merge operations via [subword_nmt/apply_bpe.py](https://github.com/rsennrich/subword-nmt).

Then, we process the data into the fairseq format, adding ```--joined-dictionary``` for WMT15 German-English:

```
SRC=source_language
TGT=target_language
TRAIN_DATA=path_to_training_data
VALID_DATA=path_to_valid_data
TEST_DATA=path_to_test_data
DATA=path_to_processed_data

# add --joined-dictionary for WMT15 German-English
fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT} \
    --trainpref ${TRAIN_DATA} --validpref ${VALID_DATA} \
    --testpref ${TEST_DATA}\
    --destdir ${DATA}
```

### Glancing Future Training

Conduct the glancing future training of HMT on WMT15 German-English task.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

TGT_FILE=dit_to_save_training_log
MODELFILE=dir_to_save_model
DATAFILE=dir_to_data

FIRST_READ=1
CANDS_PER_TOKEN=4

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
```

### Inference
Evaluate the model with the following command:

```
export CUDA_VISIBLE_DEVICES=0

MODELFILE=dir_to_save_model
DATAFILE=dir_to_data
REFERENCE=path_to_reference

python scripts/average_checkpoints.py --inputs ${MODELFILE} --num-update-checkpoints 5 --output ${MODELFILE}/average-model.pt 

python generate.py ${DATAFILE} --path ${MODELFILE}/average-model.pt --batch-size 200 --beam 1 --left-pad-source False --fp16  --remove-bpe > pred.out

grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
multi-bleu.perl -lc ${REFERENCE} < pred.translation

```


## Citation
```
@misc{guo2023glancing,
      title={Glancing Future for Simultaneous Machine Translation}, 
      author={Shoutao Guo and Shaolei Zhang and Yang Feng},
      year={2023},
      eprint={2309.06179},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
