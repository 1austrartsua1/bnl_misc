#!/bin/bash

# the greedy variant requests the max amount of resources and runs all settings as one job
# the nice variant runs each setting as a different job setting with the correct amount of resources

#inputs
cluster=cori
maxGPUs=128
maxGPUsPerNode=8
mycomm_backend=nccl
mypartition=\-
bucket_cap=25
batch_size=128 # for weak scaling this is batchsize per GPU, for strong scaling this is total batch_size
num_epochs=10
scaling_type=weak
sbatch_running_time=120
cpus_per_task=4
learning_rate=0.00001
model=resnet50

exclusive_node=true
qos_flag=true
jobname=ddp_nccl_repeat

random_data=true
random_data_dim=200
random_data_num=2000
random_nlabels=10

write_scale_results=1
my_email=patrick.r.johnstone@gmail.com # set to None if you don't want email


data_root='/global/homes/p/pjohnsto/bnl_misc/datasets/cifar10' # location of the data
src_file='/global/homes/p/pjohnsto/bnl_misc/ddp_scale_exps/python_src/cifar10_ddp_multinode.py' # python source file including full path
results_root='/global/homes/p/pjohnsto/bnl_misc/ddp_scale_exps/results' # where you want the results (python outputs will be in /pyOuts, standard output in /standardOuts




gpusPerNode=1
totalgpus=1
nodes=1

cat ${cluster}_template.slurm > torun.sbatch

if [ $exclusive_node = false ]
then
  cat torun.sbatch | sed 's|#SBATCH --exclusive||' > temp
  cat temp > torun.sbatch
  rm temp
fi

if [ $qos_flag = false ]
then
  cat torun.sbatch | sed 's|#SBATCH -q special||' > temp
  cat temp > torun.sbatch
  rm temp
fi


maxNodes=$(expr $maxGPUs / $maxGPUsPerNode)

gpuList=0
maxGpuList=$(expr $maxGPUsPerNode - 1)

for i in `seq 1 $maxGpuList`;
do
   gpuList=$gpuList,$i
done

printf "\n\n" >> torun.sbatch
echo "export CUDA_VISIBLE_DEVICES=$gpuList" >> torun.sbatch
echo "export MASTER_PORT=7001" >> torun.sbatch
echo "export MASTER_ADDR=\$SLURMD_NODENAME" >> torun.sbatch
printf "\n\n" >> torun.sbatch

while [ $totalgpus -le $maxGPUs ]; do

    sruncmd="srun --nodes="$nodes" --ntasks="$nodes" python "$src_file" --epochs $num_epochs"
    sruncmd=$sruncmd" --processes_per_node "$gpusPerNode" --comm_backend $mycomm_backend"
    sruncmd=$sruncmd" --batch-size $batch_size --bucket_cap $bucket_cap --scaling-type $scaling_type"
    sruncmd=$sruncmd" --write_scaling_results $write_scale_results --data-root "$data_root
    sruncmd=$sruncmd" --results_root "$results_root"/pyOuts/"
    sruncmd=$sruncmd" --lr "$learning_rate
    sruncmd=$sruncmd" --model "$model


    if [ $random_data = true ]
    then
       sruncmd=$sruncmd" --random_data --random_data_dim "$random_data_dim" --random_data_num "$random_data_num
       sruncmd=$sruncmd" --random_nlabels "$random_nlabels
    fi

    echo $sruncmd >> torun.sbatch

    printf "\n\n" >> torun.sbatch

    totalgpus=$(expr $totalgpus \* 2)

    nodes=$(expr $totalgpus - 1)
    nodes=$(expr $nodes \/ $maxGPUsPerNode)
    nodes=$(expr $nodes + 1)

    gpusPerNode=$(expr $totalgpus \/ $nodes)

done



declare -A templateMapping

templateMapping[NumOfNodes]=$maxNodes
templateMapping[mypartition]=$mypartition
templateMapping[GPUsPerNode]=$maxGPUsPerNode
templateMapping[myScalingType]=$scaling_type
templateMapping[myEmail]=$my_email
templateMapping[myResultsRoot]=$results_root
templateMapping[myRunningTime]=$sbatch_running_time
templateMapping[myCPUsPerTask]=$cpus_per_task
templateMapping[myJobName]=$jobname

if [ $random_data = true ]
then
   templateMapping[mydataset]=random
else
   templateMapping[mydataset]=cifar
fi



cat torun.sbatch > temp0
for key in "${!templateMapping[@]}";
do
    cat temp0 | sed 's|'$key'|'${templateMapping[$key]}'|' > temp1
    cat temp1 > temp0
done

cat temp0 > torun.sbatch
rm temp0
rm temp1
