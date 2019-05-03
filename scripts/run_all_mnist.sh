#!/bin/bash

declare -a MODELS=("simple" "m1" "sdae" "m2" "ladder")
declare -a LABELLED=("100" "1000" "3000" "100000")

for M in ${MODELS[@]}; do
    for L in ${LABELLED[@]}; do
        sbatch mnist_script.sh $M $L
        echo $M $L
    done
done