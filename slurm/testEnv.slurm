#!/bin/bash

#SBATCH --partition debug
#SBATCH --time=30
#SBATCH -A mlg-core
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2

printenv

exit 0



