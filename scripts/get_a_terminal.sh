#!/bin/sh

# Request resources
salloc \
    --job-name=pipeline_test \
    --account=euhpc_d17_077 \
    --partition=boost_usr_prod \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --gres=gpu:4 \
    --mem=100G \
    --time=02:00:00 \

# ---

# Grab a terminal
# srun --pty /bin/bash

# TODO: these commands must be executed in the new environment
##  source $HOME/my_venv/bin/activate
##  module load cuda/12.2
##  module load python/3.11.7
##  
##  # NOTE: $HOME is very slow < 300MB/s $SCRATCH should be the fastest but is temporary
##  export HF_HOME="$SCRATCH/huggingface"
