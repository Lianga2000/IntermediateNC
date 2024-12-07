#!/bin/bash

DIRS=($(ls -d -v ./results_new/*))
dir=${DIRS[-1]}
echo ${dir}

python train.py input${SLURM_PROCID} > ${dir}/print_${SLURM_PROCID}.txt 2>&1