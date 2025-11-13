#!/bin/bash

<< EOF
Run a script on two nodes of Leonardo booster cluster
EOF

TASK_NAME=throughput_test
TASK_SHARE_DISK=$WORK/$USER/$TASK_NAME/

initialize() {
    if [ ! -d $TASK_SHARE_DISK ]; then
        mkdir -p $TASK_SHARE_DISK
    fi
    rm $TASK_SHARE_DISK/*
    echo 0 > $TASK_SHARE_DISK/terminate.txt
}

main() {
    initialize
    srun -N 2 --ntasks-per-node 1 ./experiment.sh $TASK_NAME
}

main