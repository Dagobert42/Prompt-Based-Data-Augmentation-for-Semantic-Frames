#!/bin/bash

checkpoint=distilbert-base-uncased
batchsize=4
gpuids=0,1

python train_filter.py \
    --model $checkpoint \
    --save_to ./models/filter/$checkpoint/ \
    --data_file ./data/synthesis/original_splits.pt \
    --counter_examples ./data/synthesis/filter/counter_examples.pt \
    --gpu_ids $gpuids \
    --batch_size $batchsize

python run_filter.py \
    --model ./models/filter/$checkpoint/checkpoint-1136/ \
    --out_folder ./data/synthesis/ \
    --augmentations_file ./data/synthesis/augmented_splits.pt \
    --thresholds 0.5,0.75,0.9,0.95 \
    --gpu_ids $gpuids