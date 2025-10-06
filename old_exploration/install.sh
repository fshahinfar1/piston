#!/bin/bash

# expect uv to be installed

curdir=$(dirname $0)
vllm_path=$($curdir/../../vllm)
git clone https://github.com/vllm-project/vllm.git $vllm_path
cd $vllm_path
uv venv
source .venv/bin/activate
uv sync
uv pip install ./

