#!/bin/bash

declare -a MODELS=("simple" "sdae" "ladder")
declare -a LABELLED=("100" "500" "1000" "100000")
declare -a FOLD=("0" "1" "2" "3" "4")

for M in ${MODELS[@]}; do
    for L in ${LABELLED[@]}; do
        for F in ${FOLD[@]}; do
            sbatch tcga_standard_script.sh $M $L $F
            echo $M $L $F
        done
    done
done