#! /bin/bash

#SBATCH --job-name  batch_iter_info
#SBATCH --account=euhpc_d17_077
#SBATCH --partition=boost_usr_prod      # or your assigned partition
#SBATCH --nodes=1                      # one node to ensure GPUs are on same physical machine
#SBATCH --ntasks=1                     # one task
#SBATCH --cpus-per-task=4              # CPU cores allocated; tune as needed
#SBATCH --gres=gpu:2                   # request two GPUs
#SBATCH --mem=128G                     # request RAM, change xx as needed
#SBATCH --time=03:00:00                # e.g. ten minutes of runtime
#SBATCH --output=analysis/stdout.txt
#SBATCH --error=analysis/stderr.txt

LEONARDO=true

if [ $LEONARDO = true ]; then

  module load cuda/12.2
  module load python/3.11.7

  # NOTE: $HOME is very slow < 300MB/s $SCRATCH should be the fastest but is temporary
  export HF_HOME="$WORK/$USER/huggingface"

  # VNEV
  VENV_DIR=$HOME/my_venv
  if [ ! -d $VENV_DIR ]; then
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    python3 -m pip install -U pip
    python3 -m pip install -r ./requirements.txt
    deactivate
  fi
  source $VENV_DIR/bin/activate 

  nvidia-smi
  nvidia-smi topo -m

  export TMPDIR="$WORK/$USER/tmp"
  if [ ! -d $TMPDIR ]; then
    mkdir $TMPDIR
  fi
fi

# MAIN SCRIPT ---------------

BATCH_SIZE=1
NUM_ITER=32
NUM_REQ=64
LOG_FILE=$TMPDIR/exec_log.txt
iters=( 1 2 4 8 16 32 64 128 256 512 1024 )

do_exp() {
    # echo Iterations $N | tee -a $LOG_FILE
    logs=$(python3 -m piston.main \
        --batch $BATCH_SIZE \
        --num-request $NUM_REQ \
        --iters $NUM_ITER \
        --num-stages 1 \
        --pipeline simple 2>&1 | tee -a $LOG_FILE)

    res=$(echo "$logs" | grep '^Per layer in stage' -A 1 | tail -n 1)
    req_size=$(echo "$logs" | grep '^Req' | tail -n 1 | awk '{printf "%.2f MB", $4 / 1024 / 1024 }')
    echo "B: $BATCH_SIZE I: $NUM_ITER  KV Size: $req_size  ::  $res"
}

nvidia-smi | tee -a $LOG_FILE

echo ------------------------- | tee -a $LOG_FILE
for BATCH_SIZE in ${iters[@]}; do
    NUM_REQ=$((BATCH_SIZE * 2))
    for NUM_ITER in ${iters[@]}; do
        do_exp
    done
done
