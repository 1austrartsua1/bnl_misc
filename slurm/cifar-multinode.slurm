#!/bin/bash
#SBATCH --partition debug
#SBATCH --time=30
#SBATCH -A mlg-core
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH -o ../JobOutputs/maindebug-cifar-multinode.out

echo "running cifar10_ddp on processors: $SLURM_NODELIST"

echo "defining my environment variables"
export PROCESSES_PER_NODE=2
export COMM_BACKEND=gloo
numEpochs=2


echo "defining environment variables used by DDP"
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=7440
#export MASTER_ADDR=127.0.0.1 # this is the local host address to use if training on a single machine/node
 
export MASTER_ADDR=ichost001 # this is the address of the node with global rank 0.

# set up conda command
source /hpcgpfs01/software/anaconda3/2019.03-py3.7/etc/profile.d/conda.sh


echo "setting up conda environment"
conda activate torchenv

echo "calling srun"

srun --nodes=$SLURM_NNODES python ../ddp/cifar10_ddp_multinode.py --epochs $numEpochs

conda deactivate

echo "cifar10_ddp script finished at " `date`

exit 0

