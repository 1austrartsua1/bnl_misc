#!/bin/bash

#inputs
maxGPUs=4
maxGPUsPerNode=2
mycomm_backend=gloo
mypartition=debug


nodes=1
gpusPerNode=1
totalgpus=1



while [ $totalgpus -le $maxGPUs ]; do
    #echo "nodes=$nodes"
    #echo "gpusPerNode=$gpusPerNode"
    #echo "totalgpus=$totalgpus"
    
    
    gpuList=0
    maxGpuList=$(expr $gpusPerNode - 1)
    
    for i in `seq 1 $maxGpuList`;
    do
       gpuList=$gpuList,$i
    done
    
    #echo "gpuList=$gpuList"
    #printf "\n\n\n"
    
    cat ddp_scale_template.slurm | sed 's/NumOfNodes/'$nodes'/;s/GPUsPerNode/'$gpusPerNode'/;s/mycomm_backend/'$mycomm_backend'/;s/mypartition/'$mypartition'/;s/cudaVisDevs/'$gpuList'/' > ddp_scale_exp_${totalgpus}.slurm
    
    sbatch ddp_scale_exp_${totalgpus}.slurm
    
    totalgpus=$(expr $totalgpus \* 2)
    
    nodes=$(expr $totalgpus - 1)
    nodes=$(expr $nodes \/ $maxGPUsPerNode)
    nodes=$(expr $nodes + 1)
    
    gpusPerNode=$(expr $totalgpus \/ $nodes)
    
done









