#! /bin/bash
#SBATCH --job-name pipeline_test 
#SBATCH --account=euhpc_d17_077
#SBATCH --partition=boost_usr_prod      # or your assigned partition
#SBATCH --nodes=1                      # one node to ensure GPUs are on same physical machine
#SBATCH --ntasks=1                     # one task
#SBATCH --cpus-per-task=8              # CPU cores allocated; tune as needed
#SBATCH --gres=gpu:3                   # request two GPUs
#SBATCH --mem=16G                      # request RAM, change xx as needed
#SBATCH --time=00:20:00                # e.g. ten minutes of runtime
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

nsys profile \
  --output=nvlink_pipeline \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --gpu-metrics-device=all \
  --force-overwrite=true \
  python ./main.py


echo "----------------------------------"
echo "done"