#! /bin/bash

NUM_REQ=64
NUM_ITER=32

echo 'Baseline'
python ./main.py --batch 8 --num-request $NUM_REQ --iters $NUM_ITER --num-stages 1 --pipeline simple

echo 'With swapping'
python ./main.py --batch 32 --num-request $NUM_REQ --iters $NUM_ITER --num-stages 1 --pipeline swapping 