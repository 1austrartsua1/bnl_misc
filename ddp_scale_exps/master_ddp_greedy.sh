#!/bin/bash

# the greedy variant requests the max amount of resources and runs all settings as one job
# the nice variant runs each setting as a different job setting with the correct amount of resources

#inputs
maxGPUs=8
maxGPUsPerNode=8
mycomm_backend=nccl
mypartition=voltadebug
bucket_cap=10
batch_size=1024 # for weak scaling this is batchsize per GPU, for strong scaling this is total batch_size
num_epochs=10
scaling_type=weak
my_email=none # set to None if you don't want email



gpusPerNode=1
totalgpus=1
nodes=1

cat ddp_scale_template_g.slurm > ddp_scale_exp_greedy.slurm

while [ $totalgpus -le $maxGPUs ]; do
    echo "srun --nodes="$nodes" python ../ddp/cifar10_ddp_multinode.py --epochs myEpochs --processes_per_node "$gpusPerNode" --comm_backend mycomm_backend --batch-size myBatchSize --bucket_cap myBucketCap --scaling-type myScalingType"  >> ddp_scale_exp_greedy.slurm
    
    totalgpus=$(expr $totalgpus \* 2)
    
    nodes=$(expr $totalgpus - 1)
    nodes=$(expr $nodes \/ $maxGPUsPerNode)
    nodes=$(expr $nodes + 1)
    
    gpusPerNode=$(expr $totalgpus \/ $nodes)
    
done

maxNodes=$(expr $maxGPUs / $maxGPUsPerNode)

gpuList=0
maxGpuList=$(expr $maxGPUsPerNode - 1)

for i in `seq 1 $maxGpuList`;
do
   gpuList=$gpuList,$i
done

declare -A templateMapping

templateMapping[NumOfNodes]=$maxNodes
templateMapping[mypartition]=$mypartition
templateMapping[mycomm_backend]=$mycomm_backend
templateMapping[GPUsPerNode]=$maxGPUsPerNode
templateMapping[cudaVisDevs]=$gpuList
templateMapping[myBucketCap]=$bucket_cap
templateMapping[myBatchSize]=$batch_size
templateMapping[myEpochs]=$num_epochs
templateMapping[myScalingType]=$scaling_type
templateMapping[myEmail]=$my_email


cat ddp_scale_exp_greedy.slurm > temp0
for key in "${!templateMapping[@]}";
do
    cat temp0 | sed 's/'$key'/'${templateMapping[$key]}'/' > temp1
    cat temp1 > temp0
done
    
cat temp0 > ddp_scale_exp_greedy.slurm
rm temp0
rm temp1







