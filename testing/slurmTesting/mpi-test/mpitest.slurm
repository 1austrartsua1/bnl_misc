#!/bin/bash
#SBATCH -p debug
#SBATCH -A mlg-core 
#SBATCH --ntasks-per-node=8
#SBATCH --nodes 2
#SBATCH --time=0:30:00
#SBATCH --export=NONE # This is needed to ensure a proper job environment


mycommand="./mpitest" # note "./" is used to not rely on $PATH
myargs=""             # arguments for $mycommand
infiles=("")          # list of input files
outfiles=("")         # list output files


#Get the number of processors assigned by Slurm
echo "Running on $SLURM_NTASKS processors: $SLURM_NODELIST"

# added by Patrick - compile code 
module load openmpi
mpicc mpitest.c -o mpitest
module purge 

# set up the job and input files, exiting if anything went wrong


# run the command, saving the exit value
echo "Running $mycommand"
srun --ntasks=${SLURM_NTASKS} --mpi=pmi2 $SLURM_SUBMIT_DIR/$mycommand $myargs
# Using srun is the preferred way to launch mpi programs but mpirun will work.



echo "Done " `date`

exit 0
