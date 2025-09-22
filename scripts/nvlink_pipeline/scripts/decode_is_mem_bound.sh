#! /bin/bash

#SBATCH --job-name exp_decode_is_mem_bound 
#SBATCH --account=euhpc_d17_077
#SBATCH --partition=boost_usr_prod      # or your assigned partition
#SBATCH --nodes=1                      # one node to ensure GPUs are on same physical machine
#SBATCH --ntasks=1                     # one task
#SBATCH --cpus-per-task=8              # CPU cores allocated; tune as needed
#SBATCH --gres=gpu:3                   # request two GPUs
#SBATCH --mem=128G                     # request RAM, change xx as needed
#SBATCH --time=03:00:00                # e.g. ten minutes of runtime
#SBATCH --output=out_is_mem_bound/stdout.txt
#SBATCH --error=out_is_mem_bound/stderr.txt

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

OUTDIR=$HOME/res_decode_is_mem_bound/
mkdir -p $OUTDIR/simple
mkdir -p $OUTDIR/swapping

# Stop when there is an error
set -e

NUM_REQ=1024
ITERATION=512
BATCH_SIZES=( 512 256 128 64 32 16 1 )
PIPELINE="simple"

for B in ${BATCH_SIZES[@]}; do
      outfile=$OUTDIR/$PIPELINE/$B.txt

      cmd="python ./main.py \
        --batch $B \
        --num-requests $NUM_REQ \
        --pipeline $PIPELINE \
        --num-stages 1 \
        --iters $ITERATION"

      echo $cmd | tee $outfile
      $cmd | tee -a $outfile
      echo '------------------------------'
done

echo Done
