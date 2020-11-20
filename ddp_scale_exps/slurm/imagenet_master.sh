#!/bin/bash

#inputs
maxGPUs=32
maxGPUsPerNode=8
mycomm_backend=nccl
mypartition=volta
bucket_cap=25
batch_size=256 # for weak scaling this is batchsize per GPU, for strong scaling this is total batch_size
num_epochs=1
scaling_type=weak
sbatch_running_time=600
model=resnet18
num_workers=4
cpus_per_task=32
early_stop=-1

write_scale_results=1
my_email=patrick.r.johnstone@gmail.com # set to None if you don't want email



data_root='/hpcgpfs01/scratch/pjohnston/Imagenet2012' # location of the data
src_file='/sdcc/u/pjohnston/bnl_misc/ddp_scale_exps/python_src/imagenet_main.py' # python source file including full path
results_root='/sdcc/u/pjohnston/bnl_misc/ddp_scale_exps/results' # where you want the results (python outputs will be in /pyOuts, standard output in /standardOuts



gpusPerNode=1
totalgpus=1
nodes=1

cat template.slurm > torun.sbatch
printf "\n\n" >> torun.sbatch


while [ $totalgpus -le $maxGPUs ]; do

    sruncmd="srun --nodes="$nodes" --ntasks="$nodes" python "$src_file
    sruncmd=$sruncmd" -a "$model
    sruncmd=$sruncmd" --dist-url tcp://\$SLURMD_NODENAME:8008"
    sruncmd=$sruncmd" --dist-backend $mycomm_backend"
    sruncmd=$sruncmd" --multiprocessing-distributed"
    sruncmd=$sruncmd" --world-size $nodes"
    sruncmd=$sruncmd" --epochs $num_epochs"
    sruncmd=$sruncmd" --workers $num_workers"
    sruncmd=$sruncmd" --bucket_cap $bucket_cap"
    sruncmd=$sruncmd" --ngpus_per_node $gpusPerNode"
    sruncmd=$sruncmd" --batch-size $batch_size"



    if [ $scaling_type = strong ]
    then
      sruncmd=$sruncmd" --strong_scaling"
    fi

    sruncmd=$sruncmd" --write_scaling_results $write_scale_results"
    sruncmd=$sruncmd" --results_root "$results_root"/pyOuts/"
    sruncmd=$sruncmd" --early_stop $early_stop"
    sruncmd=$sruncmd" "$data_root

    echo $sruncmd >> torun.sbatch

    printf "\n\n" >> torun.sbatch

    totalgpus=$(expr $totalgpus \* 2)

    nodes=$(expr $totalgpus - 1)
    nodes=$(expr $nodes \/ $maxGPUsPerNode)
    nodes=$(expr $nodes + 1)

    gpusPerNode=$(expr $totalgpus \/ $nodes)

done

declare -A templateMapping

maxNodes=$(expr $maxGPUs / $maxGPUsPerNode)

templateMapping[NumOfNodes]=$maxNodes
templateMapping[mypartition]=$mypartition
templateMapping[GPUsPerNode]=$maxGPUsPerNode
templateMapping[myScalingType]=$scaling_type
templateMapping[myEmail]=$my_email
templateMapping[myResultsRoot]=$results_root
templateMapping[myRunningTime]=$sbatch_running_time
templateMapping[myCPUsPerTask]=$cpus_per_task

templateMapping[mydataset]=imagenet


cat torun.sbatch > temp0
for key in "${!templateMapping[@]}";
do
    cat temp0 | sed 's|'$key'|'${templateMapping[$key]}'|' > temp1
    cat temp1 > temp0
done

cat temp0 > torun.sbatch
rm temp0
rm temp1
