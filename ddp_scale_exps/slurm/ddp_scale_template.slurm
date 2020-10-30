#!/bin/bash
#SBATCH --partition mypartition
#SBATCH --time=30
#SBATCH -A mlg-core
#SBATCH --nodes=NumOfNodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4 #TODO: set to 2*GPUsPerNode or (numWorkers+1)*GPUsPerNode 
#SBATCH --gres=gpu:GPUsPerNode
#SBATCH -o ../JobOutputs/DDP_Scaling/cifar-mypartition-myScalingType-NumOfNodes-GPUsPerNode.out
#SBATCH --mail-user=myEmail
#SBATCH --mail-type END




echo "running cifar10_ddp on processors: $SLURM_NODELIST"

export CUDA_VISIBLE_DEVICES=cudaVisDevs 
export MASTER_PORT=myPort 

export MASTER_ADDR=$SLURMD_NODENAME # this is the address of the GPU/process with global rank 0, which is the GPU device id 0 on node 0.

# set up conda command
source /hpcgpfs01/software/anaconda3/2019.03-py3.7/etc/profile.d/conda.sh

conda activate torchenv

srun --nodes=$SLURM_NNODES python ../ddp/cifar10_ddp_multinode.py --epochs myEpochs --processes_per_node GPUsPerNode \
                                                                  --comm_backend mycomm_backend --batch-size myBatchSize\
                                                                  --bucket_cap myBucketCap --scaling-type myScalingType

conda deactivate

echo "cifar10_ddp script finished at " `date`

exit 0

