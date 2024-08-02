#!/bin/bash

device=0            # CUDA device
seed=0              # random seed
model="deit_tiny"   # model flavor
model="fastvit_sa12"
mode="e2e"          # mode from main.py
# mode="fq_vit"
# mode="fp_no_quant"

# specify # of bits for weights & activations ex: 3,4,8
weight=8
w=uint${weight}
a=uint8

# output folder for checkpoint saving & script outputs
output_folder="output/${model}_${weight}W8A_s${seed}"
out_file=$output_folder/logs.txt
mkdir -p $output_folder

date
CUDA_VISIBLE_DEVICES=$device python3 main.py \
    $model ~/data/ImageNet --ptf \
    --mode ${mode} \
    --img_size 256 \
    --seed ${seed} \
    --w_bit_type ${w} \
    --a_bit_type ${a} \
    --quant-method omse \
    --bias-corr \
    --save_folder $output_folder \
    --val-batchsize 8 \
    2>&1 | tee -a $out_file # append to output folder

date
