#!/bin/bash

TASK_NAME=tensor_parallel_test
TASK_SHARE_DISK=$WORK/$USER/$TASK_NAME/

initialize() {
    if [ ! -d $TASK_SHARE_DISK ]; then
        mkdir -p $TASK_SHARE_DISK || exit 1
    fi
    echo 0 > $TASK_SHARE_DISK/terminate.txt
    echo "NOT_SET" > $TASK_SHARE_DISK/head_host.txt
}

on_interrupt() {
    echo 1 > $TASK_SHARE_DISK/terminate.txt
    pkill -INT launch_script.sh
    exit 0
}

trap "on_interrupt" SIGINT SIGHUP

initialize

# Request resources
# sbatch \
salloc \
    --job-name=tensor_para_test \
    --account=euhpc_d17_077 \
    --partition=boost_usr_prod \
    --nodes=2 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --gres=gpu:1 \
    --mem=40G \
    --time=04:00:00 \
 

# NOTE: run following script in the interactive session for the experiment
# srun -N 2 --ntasks-per-node 1 ./launch_script.sh
# ---