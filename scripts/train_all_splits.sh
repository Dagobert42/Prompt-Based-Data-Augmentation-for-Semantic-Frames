#!/bin/bash

# google/mobilebert-uncased
# distilbert-base-uncased
# bert-base-uncased
# xlnet/xlnet-base-cased
# microsoft/deberta-v3-base
checkpoints="
dslim/distilbert-NER
"
dataset=wnut_17
batchsize=8
gpuids=0,1


for checkpoint in $checkpoints; do
    python ../train.py \
        --model $checkpoint \
        --save_to ../models/$checkpoint/$dataset-finetuned/ \
        --data_file ../data/$dataset/original_splits.pt \
        --gpu_ids $gpuids \
        --batch_size $batchsize \
        --max_input_length 64
    
    # python ../train.py \
    #     --model $checkpoint \
    #     --save_to ../models/$checkpoint/$dataset-augmentations-only/ \
    #     --data_file ../data/$dataset/augmented_splits.pt \
    #     --gpu_ids $gpuids \
    #     --batch_size $batchsize \
    #     --max_input_length 128

    # python ../train.py \
    #     --model ../models/$checkpoint/$dataset-augmentations-only/ --is_local \
    #     --save_to ../models/$checkpoint/$dataset-augmented/ \
    #     --data_file ../data/$dataset/original_splits.pt \
    #     --gpu_ids $gpuids \
    #     --batch_size $batchsize \
    #     --max_input_length 128
done