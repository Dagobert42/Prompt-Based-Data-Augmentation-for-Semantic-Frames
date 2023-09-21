#!/bin/bash

basemodel=microsoft/deberta-v3-base/
batchsize=4
checkpoints="
wnut_17-finetuned
"

for checkpoint in $checkpoints; do
    python ../evaluation.py \
        --model ../models/$basemodel/$checkpoint/ --is_local \
        --save_to ../logs/$basemodel/$checkpoint/ \
        --data_file ../data/wnut_17/original1_splits.pt \
        --batch_size $batchsize
done
