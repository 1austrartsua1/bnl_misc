#!/bin/bash

# the greedy variant requests the max amount of resources and runs all settings as one job
# the nice variant runs each setting as a different job setting with the correct amount of resources

#inputs
maxGPUs=8
maxGPUsPerNode=8
mycomm_backend=nccl
mypartition=voltadebug
bucket_cap=10
batch_size=512 # for weak scaling this is batchsize per GPU, for strong scaling this is total batch_size
num_epochs=10
scaling_type=weak

#important...
dryrun=yes #if set to yes, does not invoke sbatch, just creates the job scripts which you can then inspect for errors
my_email=none # set to None if you don't want email


nodes=1
gpusPerNode=1
totalgpus=1

declare -A templateMapping


while [ $totalgpus -le $maxGPUs ]; do
    #echo "nodes=$nodes"
    #echo "gpusPerNode=$gx  pusPerNode"
    #echo "totalgpus=$totalgpus"
    
    
    gpuList=0
    maxGpuList=$(expr $gpusPerNode - 1)
    
    for i in `seq 1 $maxGpuList`;
    do
       gpuList=$gpuList,$i
    done
    
    #echo "gpuList=$gpuList"
    #printf "\n\n\n"
    
    my_port=$(expr 2000 + $totalgpus)
    
        
    templateMapping[NumOfNodes]=$nodes
    templateMapping[mypartition]=$mypartition
    templateMapping[mycomm_backend]=$mycomm_backend
    templateMapping[GPUsPerNode]=$gpusPerNode
    templateMapping[cudaVisDevs]=$gpuList
    templateMapping[myBucketCap]=$bucket_cap
    templateMapping[myBatchSize]=$batch_size
    templateMapping[myEpochs]=$num_epochs
    templateMapping[myScalingType]=$scaling_type
    templateMapping[myEmail]=$my_email
    templateMapping[myPort]=$my_port


    #numChanges=${#templateMapping[@]} don't need anymore

    cat ddp_scale_template.slurm > temp0
    for key in "${!templateMapping[@]}";
    do
        cat temp0 | sed 's/'$key'/'${templateMapping[$key]}'/' > temp1
        cat temp1 > temp0
    done
        
    cat temp0 > ddp_scale_exp_${totalgpus}.slurm
    rm temp0
    rm temp1
    
    if [ $dryrun == no ]
    then
       sbatch ddp_scale_exp_${totalgpus}.slurm
    fi
    
    totalgpus=$(expr $totalgpus \* 2)
    
    nodes=$(expr $totalgpus - 1)
    nodes=$(expr $nodes \/ $maxGPUsPerNode)
    nodes=$(expr $nodes + 1)
    
    gpusPerNode=$(expr $totalgpus \/ $nodes)
    
done









