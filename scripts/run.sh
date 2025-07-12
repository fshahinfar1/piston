#!/bin/bash

curdir=$(dirname $0)
VLLM_VENV=$(realpath $curdir/../../vllm/.venv)
source $VLLM_VENV/bin/activate

timestamp_begin=$(date +%s)
taskset -c 0 python $curdir/../src/main.py
timestamp_end=$(date +%s)
duration=$((timestamp_end - timestamp_begin))
min=$((duration / 60))
sec=$((duration - ( min * 60 ) ))
echo "$min:$sec"
