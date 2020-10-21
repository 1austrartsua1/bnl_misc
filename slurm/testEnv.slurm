#!/bin/bash

#SBATCH --partition debug
#SBATCH --time=30
#SBATCH -A mlg-core
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH -o ./multinode/testEnv.out

echo "Testing the SLURM environment variables"

echo SLURM_TASKS_PER_NODE=$SLURM_TASKS_PER_NODE
echo SLURM_NNODES=$SLURM_NNODES

exit 0



