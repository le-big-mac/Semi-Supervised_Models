#!/bin/bash

declare -a IMPUTATION=("drop_samples" "drop_genes" "mean_value" "zero")

for I in ${IMPUTATION[@]}; do
    sbatch imputation_script.sh $I
done