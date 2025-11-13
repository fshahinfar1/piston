#!/bin/bash
NID=$SLURM_NODEID
TASK_NAME=tensor_parallel_test
TASK_SHARE_DISK=$WORK/$USER/$TASK_NAME/
# MODEL_PATH=$WORK/$USER/models/phi3.5/models--microsoft--Phi-3.5-mini-instruct
MODEL_PATH=$WORK/$USER/dequantized/phi35

if [ ! -d $TASK_SHARE_DISK ]; then
    if [ $NID -eq 0 ]; then
        mkdir -p $TASK_SHARE_DISK || exit 1
    else
        # wait until the task shared dir is created
        sleep 3
        if [ ! -d $TASK_SHARE_DISK ]; then
            echo "Timeout: the head did not created the shared task directory"
            exit 1
        fi
    fi
fi


echo "Inside the launch script"
echo "NODEID: $NID"
echo "PROCID: $SLURM_PROCID"
NAME=$(hostname)
echo "Hosetname: $NAME"
echo "NODELIST: $SLURM_NODELIST"
# ip addr
# exit 0

module load python/3.11.7
module load cuda/12.2

# Enable the python environment
source $HOME/my_venv/bin/activate

VLLM_IP_FILE=$TASK_SHARE_DISK/vllm_ip.txt
VLLM_LOG=$TASK_SHARE_DISK/vllm_out.txt

wait_for_vllm() {
    while [ -z "$(grep 'Application startup complete' $VLLM_LOG)" ]; do
        sleep 5
    done
}

generate_traffic() {
    VLLM_IP=$(cat $VLLM_IP_FILE)
    VLLM_URL="http://$VLLM_IP:8000"
    echo $VLLM_URL

    guidellm benchmark \
    --target $VLLM_URL \
    --rate-type sweep \
    --max-seconds 30 \
    --data "prompt_tokens=256,output_tokens=128"
}

node0() {
    echo 0 > $TASK_SHARE_DISK/terminate.txt
    echo "NOT_SET" > $TASK_SHARE_DISK/head_host.txt
    ray start --head --port=6379 &> $TASK_SHARE_DISK/ray_node0.txt

    IP=$(cat $TASK_SHARE_DISK/ray_node0.txt | grep "ray start --address=" | cut -d '=' -f 2 | tr -d "'" | cut -d : -f 1)
    echo $IP > $TASK_SHARE_DISK/head_host.txt

    echo "node0: Waiting for vLLM to start..."
    wait_for_vllm

    echo "node0: Ready to generate traffic"
    generate_traffic

    while  true; do
        sleep 5
        t=$(cat $TASK_SHARE_DISK/terminate.txt)
        if [ $t -ne 0 ]; then
            exit 0
        fi
    done
}

node1() {
    # Get the IP address of experiment interface
    IP=$(ip -j addr show dev enp1s0f0 | jq '.[0].addr_info[].local' | tr -d \" | grep -e '^10.')
    echo $NAME > $TASK_SHARE_DISK/worker_host.txt

    # Write the IP the traffic generator should use
    echo $IP > $VLLM_IP_FILE

    # wait until head is started
    # TODO: can I do something better?
    sleep 2
    iter=0
    HEAD_HOST_NAME=$(cat $TASK_SHARE_DISK/head_host.txt)
    while [ $HEAD_HOST_NAME = "NOT_SET" ]; do
        sleep 10
        HEAD_HOST_NAME=$(cat $TASK_SHARE_DISK/head_host.txt)
        iter=$((iter + 1))
        if [ $iter -gt 3 ]; then
            echo "Failed to get remote ip"
            echo 1 > $TASK_SHARE_DISK/terminate.txt
            exit 1
        fi
    done

    echo node1: Got Ray head IP. Connecting to it ...
    ray start --address="$HEAD_HOST_NAME:6379"

    sleep 2

    echo node1: launching vLLM
    vllm serve $MODEL_PATH --tensor-parallel-size 2  --distributed-executor-backend ray 2>&1 | tee $VLLM_LOG
    # vllm serve $MODEL_PATH --pipeline-parallel-size 2  --distributed-executor-backend ray 2>&1 | tee $VLLM_LOG
}

main() {
    case $NID in
        0) node0 ;;
        1) node1 ;;
        *)
        echo Unexpected task id $NID
        ;;
    esac

    echo 1 > $TASK_SHARE_DISK/terminate.txt
    echo "Done!"
}

main
