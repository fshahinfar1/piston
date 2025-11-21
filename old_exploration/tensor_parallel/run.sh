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
    echo "NOT_SET" > $TASK_SHARE_DISK/node0.txt
    echo "NOT_SET" > $TASK_SHARE_DISK/node1.txt
    echo "NOT_SET" > $TASK_SHARE_DISK/node2.txt

    echo "NOT_SET" > $TASK_SHARE_DISK/node0_status.txt
    echo "NOT_SET" > $TASK_SHARE_DISK/node1_status.txt

    if [ -f $VLLM_LOG ]; then
        # make sure we don't read start up status of previous run
        rm $VLLM_LOG
    fi
}

check_with_different_sin() {
    mkdir -p ./results/
    for sin in 64 128 512 1024; do
        out=./results/tensor_$sin.txt
        if [ ! -f $out ]; then
            echo Running tensor 
            srun -N 3 --ntasks-per-node 1 ./launch_script.sh $TASK_NAME \
                --mode tensor \
                --s-in $sin --s-out 64 | tee $out
        fi

        B=./benchmark.json
        rm -f $B || true

        out=./results/parallel_$sin.txt
        if [ ! -f $out ]; then
            echo Running parallel
            srun -N 3 --ntasks-per-node 1 ./launch_script.sh $TASK_NAME \
                --mode pipeline \
                --s-in $sin --s-out 64 | tee $out
        fi

        rm -f $B || true

    done
}

main() {
    initialize

    B=./benchmark.json
    rm -f $B || true
    srun -N 3 --ntasks-per-node 1 ./launch_script.sh $TASK_NAME \
        --mode tensor \
        --s-in 64 --s-out 64

    # check_with_different_sin
}

main