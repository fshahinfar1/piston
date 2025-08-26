#!/bin/bash
printf "["
for i in $(seq 1 38); do
	v=$(cat data/$i.txt | grep Mean | grep GB/s | cut -d ' ' -f 2)
	printf "%s," $v
done
printf "]\n"
