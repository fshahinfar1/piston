#!/bin/bash
NID=$SLURM_NODEID
TASK_NAME=tensor_parallel_test
TASK_SHARE_DISK=$WORK/$USER/$TASK_NAME/
# MODEL_PATH=$WORK/$USER/models/phi3.5/models--microsoft--Phi-3.5-mini-instruct
MODEL_PATH=$WORK/$USER/dequantized/phi35

if [ ! -d $TASK_SHARE_DISK ]; then
    echo 'Shared directory not found!'
    exit 1
fi


echo "Inside the launch script"
echo "NODEID: $NID"
echo "PROCID: $SLURM_PROCID"
NAME=$(hostname)
echo "Hosetname: $NAME"
echo "NODELIST: $SLURM_NODELIST"
IFACE=ib0
IP=$(ip -j addr show dev $IFACE | jq '.[0].addr_info[].local' | tr -d \" | grep -e '^10.')
echo "$NAME: $IP"
# ip addr
# exit 0

module load python/3.11.7
module load cuda/12.2

# nvidia-smi topo -m
# exit 0

# Enable the python environment
source $HOME/my_venv/bin/activate

VLLM_IP_FILE=$TASK_SHARE_DISK/vllm_ip.txt
VLLM_LOG=$TASK_SHARE_DISK/vllm_out.txt

# This is needed for vLLM to use  Infiniband interface
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=$IFACE  # or your IB interface name
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=PXB
export NCCL_P2P_LEVEL=PXB 

# Parse cli arguments
shift;
MODE=tensor
S_IN=128
S_OUT=64
while [ $# -gt 0 ]; do
    case $1 in
        --mode)
            MODE=$2
            shift; shift;
            ;;
        --s-in)
            S_IN=$2
            shift; shift;
            ;;
        --s-out)
            S_OUT=$2
            shift; shift;
            ;;
        *)
            echo Unexpected argument $1
            exit 1
            ;;
    esac
done
# ------------------

wait_until_terminate_flag() {
    T=$(cat $TASK_SHARE_DISK/terminate.txt)
    while [ $T -eq 0 ]; do
        sleep 10;
        T=$(cat $TASK_SHARE_DISK/terminate.txt)
    done

}

wait_for_ray_to_be_ready() {
    T=$(cat $TASK_SHARE_DISK/node0_status.txt)
    while [ $T = "NOT_SET" ]; do
        T=$(cat $TASK_SHARE_DISK/node0_status.txt)
    done
}

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
    --max-seconds 10 \
    --data "prompt_tokens=$S_IN,output_tokens=$S_OUT"
}

node0() {
    echo 0 > $TASK_SHARE_DISK/terminate.txt
    echo $IP > $TASK_SHARE_DISK/node0.txt

    # Start a ray cluster
    echo "node0: starting ray ..."
    ray start --head --port=6379 &> $TASK_SHARE_DISK/ray_node0.txt

    HEAD_NODE_IP=$(cat $TASK_SHARE_DISK/ray_node0.txt | grep "ray start --address=" | cut -d '=' -f 2 | tr -d "'" | cut -d : -f 1)
    echo $HEAD_NODE_IP > $TASK_SHARE_DISK/head_node_ip.txt

    echo "Ready" > $TASK_SHARE_DISK/node0_status.txt
    echo "node0: Ray ready. Waiting for other nodes ..."

    # Wait until other nodes join the cluster
    NODE1_STATUS=$(cat $TASK_SHARE_DISK/node1_status.txt)
    while [ "$NODE1_STATUS" = "NOT_SET" ]; do
        NODE1_STATUS=$(cat $TASK_SHARE_DISK/node1_status.txt)
    done


    # vllm serve $MODEL_PATH --tensor-parallel-size 2  --distributed-executor-backend ray 2>&1 | tee $VLLM_LOG
    # vllm serve $MODEL_PATH --pipeline-parallel-size 2  --distributed-executor-backend ray 2>&1 | tee $VLLM_LOG

    if [ $MODE = tensor ]; then
        mode="--tensor-parallel-size 2"
    else
        mode="--pipeline-parallel-size 2"
    fi

    echo "Experiment mode: $MODE   --  $mode"

    echo $HEAD_NODE_IP > $VLLM_IP_FILE
    echo "node0: starting vLLM ..."
    nohup vllm serve $MODEL_PATH \
        $mode \
        --distributed-executor-backend ray \
        --host $HEAD_NODE_IP --port 8000 \
        2>&1 | tee $VLLM_LOG &> /dev/null &
    
    wait_until_terminate_flag

    pkill VLLM:ENGINE
}

node1() {
    echo $IP > $TASK_SHARE_DISK/node1.txt

    echo "node1: waiting for Ray ..."
    wait_for_ray_to_be_ready

    HEAD_NODE_IP=$(cat $TASK_SHARE_DISK/head_node_ip.txt)
    echo "node1: connect to ray at address $HEAD_NODE_IP"

    ray start --address="$HEAD_NODE_IP:6379"
    echo Ready > $TASK_SHARE_DISK/node1_status.txt


    # I must stop the script from closing. Otherwise thing don't work
    wait_until_terminate_flag
}

node2() {
    echo $IP > $TASK_SHARE_DISK/node2.txt

    echo "node2: Waiting for vLLM to start..."
    wait_for_vllm

    echo "node2: Ready to generate traffic"
    generate_traffic

    echo "node2: End of traffic generation"
}

main() {
    case $NID in
        0) node0 ;;
        1) node1 ;;
        2) node2 ;;
        *)
        echo Unexpected task id $NID
        ;;
    esac

    echo 1 > $TASK_SHARE_DISK/terminate.txt
    echo "node$NID: Done!"
}

main $@
