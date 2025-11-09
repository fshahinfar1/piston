#! /bin/bash
#SBATCH --job-name ToT_test 
#SBATCH --account=euhpc_d17_077
#SBATCH --partition=boost_usr_prod      # or your assigned partition
#SBATCH --nodes=1                      # one node to ensure GPUs are on same physical machine
#SBATCH --ntasks=1                     # one task
#SBATCH --cpus-per-task=4              # CPU cores allocated; tune as needed
#SBATCH --gres=gpu:1                   # request two GPUs
#SBATCH --mem=16G                      # request RAM, change xx as needed
#SBATCH --time=00:40:00                # e.g. ten minutes of runtime
#SBATCH --output=output/stdout.txt
#SBATCH --error=output/stderr.txt

<< EOF
Description: Run a test using princtons original ToT implementation and use vLLM
instead of OpenAI server
EOF

sBATCH=false
if [ "$sBATCH" = true ]; then

module load cuda/12.2
module load python/3.11.7

# NOTE: $HOME is very slow < 300MB/s $SCRATCH should be the fastest but is temporary
export HF_HOME="$WORK/$USER/huggingface"


# Your program commands here, for example:
nvidia-smi
nvidia-smi topo -m

echo '---------------------------------------------'

fi

OUTDIR=tot_output/
VLLM_LOG=$OUTDIR/vllm.txt
VLLM_PORT=8000

mkdir -p $OUTDIR

clean() {
    pkill VLLM::EngineCor
}

start_vllm_server() {
    # Activate your Python virtual environment
    source $HOME/my_venv/bin/activate
    # MODEL_NAME=$WORK/$USER/dequantized/gpt-oss-20b-bf16
    MODEL_NAME=$WORK/$USER/dequantized/phi35

    # Run vLLM
    nohup vllm serve $MODEL_NAME \
        --served-model-name "gpt-4" \
        --port $VLLM_PORT &> $VLLM_LOG &
    deactivate
}

wait_until_vllm_up() {
    # Wait for vLLM to start
    MAX_WAIT=250 # seconds
    B=$(date +%s)
    while true; do
        sleep 20
        T=$(grep 'Application startup complete' $VLLM_LOG 2> /dev/null)
        if [ -n "$T" ]; then
            # we are ready
            break
        fi

        # check timeout
        N=$(date +%s)
        DELTA=$((N - B))
        if [ $DELTA -gt $MAX_WAIT ]; then
            echo Timeout waiting for vLLM server
            clean
            exit 1
        fi

        # check its running
        # if [ -z "$(pidof VLLM::EngineCor)" ]; then
        #     echo vLLM failed!
        #     exit 1
        # fi
    done
}

run_tot_exp() {
    TOT_DIR=../others/tree-of-thought-llm/
    TOT_EXP=$TOT_DIR/run.py
    # Run the test
    export OPENAI_API_KEY=""
    export OPENAI_API_BASE="http://localhost:8000/v1"
    source $TOT_DIR/venv/bin/activate
    python3 $TOT_EXP --task game24 \
        --task_start_index 900 --task_end_index 910 \
        --method_generate propose --method_evaluate value --n_select_sample 5
}

main() {
    start_vllm_server
    echo Waiting for vLLM ...
    wait_until_vllm_up
    echo vLLM is ready.
    run_tot_exp
    clean
}

main
