#!/bin/bash
#SBATCH --partition mypartition
#SBATCH --time=myRunningTime
#SBATCH -A mlg-core
#SBATCH --nodes=NumOfNodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4 #TODO: set to 2*GPUsPerNode or (numWorkers+1)*GPUsPerNode
#SBATCH --gres=gpu:GPUsPerNode
#SBATCH -o myResultsRoot/standardOuts/mydataset-mypartition-myScalingType-myBatchSize.stdout
#SBATCH --mail-user=myEmail
#SBATCH --mail-type END




echo "running cifar10_ddp on processors: $SLURM_NODELIST"

export CUDA_VISIBLE_DEVICES=cudaVisDevs
export MASTER_PORT=7001

export MASTER_ADDR=$SLURMD_NODENAME # this is the address of the GPU/process with global rank 0, which is the GPU device id 0 on node 0.

# set up conda command
source /hpcgpfs01/software/anaconda3/2019.03-py3.7/etc/profile.d/conda.sh

conda activate torchenv
