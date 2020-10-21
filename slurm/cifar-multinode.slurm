#!/bin/bash

#SBATCH --partition debug
#SBATCH --time=30
#SBATCH -A mlg-core
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH -o ../JobOutputs/maindebug-cifar-multinode.out

echo "running cifar10_ddp on $SLURM_NTASKS processors: $SLURM_NODELIST"

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=7440
export MASTER_ADDR=127.0.0.1

# set up conda
source /hpcgpfs01/software/anaconda3/2019.03-py3.7/etc/profile.d/conda.sh

# set up python environment
conda activate torchenv

srun --nodes=${SLURM_NNODES} python ../ddp/cifar10_ddp_multinode.py

conda deactivate

echo "cifar10_ddp script finished at " `date`

exit 0

