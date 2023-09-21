#!/bin/bash

dataset=biored
gpuids=0

python ../augment.py \
    --data_file ../data/$dataset/original_splits.pt \
    --entity_descriptions_file ../data/$dataset/entity_descriptions.json \
    --save_to ../data/$dataset/augmentations/ \
    --gpu_ids $gpuids \
    --max_calls 100 \
    --rounds 5
