#!/bin/bash
TASK_NAME=tensor_parallel_test
TASK_SHARE_DISK=$WORK/$USER/$TASK_NAME/
VLLM_IP_FILE=$TASK_SHARE_DISK/vllm_ip.txt
VLLM_LOG=$TASK_SHARE_DISK/vllm_out.txt

initialize() {
    if [ ! -d $TASK_SHARE_DISK ]; then
        mkdir -p $TASK_SHARE_DISK || exit 1
    fi
    echo 0 > $TASK_SHARE_DISK/terminate.txt
    echo "NOT_SET" > $TASK_SHARE_DISK/head_host.txt
    echo "NOT_SET" > $VLLM_IP_FILE
    if [ -f $VLLM_LOG ]; then
        # make sure we don't read start up status of previous run
        rm $VLLM_LOG
    fi
}

main() {
    initialize
    srun -N 2 --ntasks-per-node 1 ./launch_script.sh
}

main