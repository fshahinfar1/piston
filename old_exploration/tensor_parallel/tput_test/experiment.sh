#!/bin/bash
NID=$SLURM_NODEID
TASK_NAME=$1
TASK_SHARE_DISK=$WORK/$USER/$TASK_NAME/
IPERF_BIN=/leonardo/home/userexternal/fshahinf/iperf3/iperf-3.19/src/iperf3

if [ ! -d $TASK_SHARE_DISK ]; then
    echo 'Shared directory does not exists (use run.sh to launch this script on cluster nodes)'
    exit 1
fi


echo "Inside the experiment script"
echo "NODEID: $NID"
echo "PROCID: $SLURM_PROCID"
NAME=$(hostname)
echo "Hosetname: $NAME"
echo "NODELIST: $SLURM_NODELIST"
# Get the IP address of experiment interface
# IFACE=enp1s0f0
IFACE=ib0
IP=$(ip -j addr show dev $IFACE | jq '.[0].addr_info[].local' | tr -d \" | grep -e '^10.')
echo "$NAME: $IP"
# ethtool $IFACE
# ip addr
# exit 0

module load python/3.11.7
module load cuda/12.2

# Enable the python environment
source $HOME/my_venv/bin/activate

node0() {
    echo $NAME > $TASK_SHARE_DISK/node0_host.txt
    echo $IP > $TASK_SHARE_DISK/node0_ip.txt

    $IPERF_BIN -s &> $TASK_SHARE_DISK/server_log.txt &

    while  true; do
        sleep 5
        t=$(cat $TASK_SHARE_DISK/terminate.txt)
        if [ $t -ne 0 ]; then
            exit 0
        fi
    done
}

node1() {
    echo $NAME > $TASK_SHARE_DISK/node1_host.txt
    echo $IP > $TASK_SHARE_DISK/node1_ip.txt

    # wait until head is started
    # TODO: can I do something better?
    sleep 10

    SERVER_IP=$(cat $TASK_SHARE_DISK/node0_ip.txt)
    while [ -z "$SERVER_IP" ]; do
        sleep 5
        SERVER_IP=$(cat $TASK_SHARE_DISK/node0_ip.txt)
    done

    echo "node1: connecting to $SERVER_IP"
    $IPERF_BIN -c $SERVER_IP -P 4
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
