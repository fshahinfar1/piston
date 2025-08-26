#! /bin/bash
set -e
mkdir ./data/
for sz in $(seq 1 63); do
    echo $sz
    python3 ./nvlink_speed.py $sz > ./data/$sz.txt
    sleep 1
done
