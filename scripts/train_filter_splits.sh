#!/bin/bash

checkpoint=distilbert-base-uncased
batchsize=4
gpuids=0,1

python train.py \
    --model $checkpoint \
    --save_to ./models/$checkpoint/7193-filtered-augmentations-only/ \
    --data_file ./data/synthesis/7193-filtered_splits.pt \
    --gpu_ids $gpuids \
    --batch_size $batchsize

python train.py \
    --model ./models/$checkpoint/7193-filtered-augmentations-only/ --is_local \
    --save_to ./models/$checkpoint/7193-filtered-augmented/ \
    --data_file ./data/synthesis/original_splits.pt \
    --gpu_ids $gpuids \
    --batch_size $batchsize

python train.py \
    --model $checkpoint \
    --save_to ./models/$checkpoint/8674-filtered-augmentations-only/ \
    --data_file ./data/synthesis/8674-filtered_splits.pt \
    --gpu_ids $gpuids \
    --batch_size $batchsize

python train.py \
    --model ./models/$checkpoint/8674-filtered-augmentations-only/ --is_local \
    --save_to ./models/$checkpoint/8674-filtered-augmented/ \
    --data_file ./data/synthesis/original_splits.pt \
    --gpu_ids $gpuids \
    --batch_size $batchsize
