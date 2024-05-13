#!/bin/bash

#SBATCH --partition=shortq  # partition name 
#SBATCH --job-name=trainnavigation    # job name
#SBATCH --output=output/train_out_%J.out  # output file name
#SBATCH --error=simlab_output/train_err_%J.err   # error file name
#SBATCH --nodes=4           # number of nodes
#SBATCH --ntasks-per-node=1 # number of tasks

module purge                # unload all modules
module load slurm           # load slurm module 
module load anaconda/3-5.3.1 # load anaconda module
source $ANACONDA_HOME/etc/profile.d/conda.sh
module load anaconda/3-5.3.1 # load anaconda module

conda activate navigationEnv

python Train.py        # execute the file 