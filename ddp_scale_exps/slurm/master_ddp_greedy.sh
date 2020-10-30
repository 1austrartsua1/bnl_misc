#!/bin/bash

# the greedy variant requests the max amount of resources and runs all settings as one job
# the nice variant runs each setting as a different job setting with the correct amount of resources

#inputs
maxGPUs=2
maxGPUsPerNode=2
mycomm_backend=gloo
mypartition=debug
bucket_cap=25
batch_size=64 # for weak scaling this is batchsize per GPU, for strong scaling this is total batch_size
num_epochs=2
scaling_type=weak

write_scale_results=1
my_email=none # set to None if you don't want email
data_root='/sdcc/u/pjohnston/bnl_misc/cifar10' # location of the data
src_file='/sdcc/u/pjohnston/bnl_misc/ddp_scale_exps/python_src/cifar10_ddp_multinode.py' # python source file including full path
results_root='/sdcc/u/pjohnston/bnl_misc/ddp_scale_exps/results' # where you want the results



gpusPerNode=1
totalgpus=1
nodes=1

cat ddp_scale_template_g.slurm > ddp_scale_exp_greedy.slurm

while [ $totalgpus -le $maxGPUs ]; do
    echo "srun --nodes="$nodes" --ntasks="$nodes" python "$src_file" --epochs myEpochs --processes_per_node "$gpusPerNode" --comm_backend mycomm_backend --batch-size myBatchSize --bucket_cap myBucketCap --scaling-type myScalingType --write_scaling_results myWriteScaleResults --data-root "$data_root" --results_root "$results_root"/pyOuts/"  >> ddp_scale_exp_greedy.slurm
    
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
templateMapping[myWriteScaleResults]=$write_scale_results
templateMapping[myResultsRoot]=$results_root

cat ddp_scale_exp_greedy.slurm > temp0
for key in "${!templateMapping[@]}";
do
    cat temp0 | sed 's|'$key'|'${templateMapping[$key]}'|' > temp1
    cat temp1 > temp0
done
    
cat temp0 > ddp_scale_exp_greedy.slurm
rm temp0
rm temp1







