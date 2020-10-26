#!/bin/bash
#SBATCH --partition volta
#SBATCH --time=30
#SBATCH -A mlg-core
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH -o ../JobOutputs/volta-cifar-multinode.out
#SBATCH --mail-user=patrick.r.johnstone@gmail.com
#SBATCH --mail-type BEGIN,END



echo "running cifar10_ddp on processors: $SLURM_NODELIST"
echo "2 nodes"
echo "2 processes per node"


echo "defining environment variables used by DDP"
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=7440
#export MASTER_ADDR=127.0.0.1 # this is the local host address to use if training on a single machine/node
 
echo "This is node $SLURM_NODEID (should be Node 0!), with node name: $SLURMD_NODENAME"
export MASTER_ADDR=$SLURMD_NODENAME # this is the address of the GPU/process with global rank 0, which is the GPU device id 0 on node 0.

# set up conda command
source /hpcgpfs01/software/anaconda3/2019.03-py3.7/etc/profile.d/conda.sh


echo "setting up conda environment"
conda activate torchenv

echo "calling srun..."
printf "\n\n\n"

srun --nodes=$SLURM_NNODES python ../ddp/cifar10_ddp_multinode.py --epochs 2 --processes_per_node 2 --comm_backend "gloo"

conda deactivate

echo "cifar10_ddp script finished at " `date`

exit 0

