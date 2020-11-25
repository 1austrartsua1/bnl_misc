#!/bin/bash
#SBATCH -C gpu
#SBATCH --time=myRunningTime
#SBATCH --nodes=NumOfNodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=myCPUsPerTask
#SBATCH --gpus-per-node=GPUsPerNode
#SBATCH -A m1759
#SBATCH --exclusive
#SBATCH -o /global/homes/p/pjohnsto/bnl_misc/ddp_scale_exps/results/standardOuts/mydataset-mypartition-myScalingType-NumOfNodesxGPUsPerNode.stdout
#SBATCH --mail-user=myEmail
#SBATCH --mail-type BEGIN,END
#SBATCH --job-name=DDP
#SBATCH -q special

module load pytorch/v1.5.1-gpu
