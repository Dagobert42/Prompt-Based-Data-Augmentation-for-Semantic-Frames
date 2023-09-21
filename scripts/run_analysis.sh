#!/bin/bash

dataset=wnut_17
gpuids=0


python ../analysis.py \
    --save_to ../data/$dataset/ \
    --data_file ../data/$dataset/original_splits.pt \
    --gpu_ids $gpuids