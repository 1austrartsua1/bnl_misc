#!/bin/bash

#SBATCH --partition debug
#SBATCH --time=30
#SBATCH -A mlg-core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH -o maindebug-mpi.out

echo "running cifar10_ddp..."

export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=7440
export MASTER_ADDR=127.0.0.1

# set up conda 
source /hpcgpfs01/software/anaconda3/2019.03-py3.7/etc/profile.d/conda.sh

# set up python environment
conda activate torchenv



time python cifar10_ddp.py

conda deactivate
 

echo "cifar10_ddp script finished"

exit 0

