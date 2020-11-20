#!/bin/bash
#SBATCH --partition mypartition
#SBATCH --time=myRunningTime
#SBATCH -A mlg-core
#SBATCH --nodes=NumOfNodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=myCPUsPerTask
#SBATCH --gres=gpu:GPUsPerNode
#SBATCH -o myResultsRoot/standardOuts/mydataset-mypartition-myScalingType-NumOfNodesxGPUsPerNode.stdout
#SBATCH --mail-user=myEmail
#SBATCH --mail-type BEGIN,END
#SBATCH --job-name=DDP

echo "running ddp on processors: $SLURM_NODELIST"

# set up conda command
source /hpcgpfs01/software/anaconda3/2019.03-py3.7/etc/profile.d/conda.sh

conda activate torchenv
