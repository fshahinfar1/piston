#!/bin/bash

BATCH_SIZE=1
NUM_ITER=32
NUM_REQ=64
LOG_FILE=/tmp/exec_log.txt
iters=( 1 2 4 8 16 32 64 128 256 512 1024 )

do_exp() {
    # echo Iterations $N | tee -a $LOG_FILE
    logs=$(python ./main.py \
        --batch $BATCH_SIZE \
        --num-request $NUM_REQ \
        --iters $NUM_ITER \
        --num-stages 1 \
        --pipeline simple 2>&1 | tee -a $LOG_FILE)

    res=$(echo "$logs" | grep '^Per layer in stage' -A 1 | tail -n 1)
    req_size=$(echo "$logs" | grep '^Req' | tail -n 1 | awk '{printf "%.2f MB", $4 / 1024 / 1024 }')
    echo "B: $BATCH_SIZE I: $NUM_ITER  KV Size: $req_size  ::  $res"
}


nvidia-smi | tee -a $LOG_FILE

echo ------------------------- | tee -a $LOG_FILE
for BATCH_SIZE in ${iters[@]}; do
    NUM_REQ=$((BATCH_SIZE * 2))
    for NUM_ITER in ${iters[@]}; do
        do_exp
    done
done
