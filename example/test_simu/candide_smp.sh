#!/bin/bash

##########################
# SMP Script for Candide #
##########################

# Receive email when job finishes or aborts
#PBS -M axel.guinot@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N shapepipe_smp
# Join output and errors in one file
#PBS -j oe
# Request number of cores
#PBS -l nodes=n01:ppn=48
#PBS -l walltime=48:00:00

# Activate conda environment
module load intelpython/3
source activate $HOME/.conda/envs/shapepipe

# Run ShapePipe
cd $HOME/ShapePipe_github/shapepipe
$HOME/.conda/envs/shapepipe/bin/python shapepipe_run.py -c /home/guinot/ShapePipe_github/shapepipe/example/test_simu/config.ini

# Return exit code
exit 0
