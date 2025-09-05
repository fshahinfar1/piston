#! /bin/bash
#SBATCH --job-name pipeline_test 
#SBATCH --account=euhpc_d17_077
#SBATCH --partition=boost_usr_prod      # or your assigned partition
#SBATCH --nodes=1                      # one node to ensure GPUs are on same physical machine
#SBATCH --ntasks=1                     # one task
#SBATCH --cpus-per-task=8              # CPU cores allocated; tune as needed
#SBATCH --gres=gpu:3                   # request two GPUs
#SBATCH --mem=32G                      # request RAM, change xx as needed
#SBATCH --time=02:00:00                # e.g. ten minutes of runtime
#SBATCH --output=output/stdout.txt
#SBATCH --error=output/stderr.txt

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

OUTDIR=$HOME/results/
mkdir -p $OUTDIR/simple
mkdir -p $OUTDIR/swapping

for B in $(seq 1 8); do
    for P in  "simple" "swapping" ; do
        python ./main.py --batch $B --num-requests 64 --pipeline $P --iters 1024 | tee $OUTDIR/$P/$B.txt
    done
done

echo Done
