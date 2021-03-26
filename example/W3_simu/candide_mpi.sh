#!/bin/bash

##########################
# MPI Script for CANDIDE #
##########################

# Receive email when job finishes or aborts
## #PBS -M axel.guinot.astro@gmail.com
## #PBS -m ea

# Set a name for the job
#PBS -N shapepipe_mpi

# Join output and errors in one file
#PBS -j oe

# Set maximum computing time (e.g. 5min)
#PBS -l walltime=48:00:00

# Request number of cores (e.g. 2 from 2 different machines)
#PBS -l nodes=n01:ppn=20+n02:ppn=20+n03:ppn=20+n04:ppn=20+n05:ppn=20+n06:ppn=10+n07:ppn=10+n13:ppn=10+n14:ppn=10+n15:ppn=10+n16:ppn=20
##  +n02:ppn=1

# Full path to environment
#export SPENV="$HOME/.conda/envs/shapepipe"

# Full path to example config file and input data
#export SPDIR="$HOME/shapepipe"

# Load modules
module load intelpython/3
#module load openmpi/4.1.0

# Activate conda environment
# source activate $SPENV

# Run ShapePipe using full paths to executables
mpiexec --map-by node $HOME/ShapePipe_github/shapepipe/shapepipe_run.py -c $HOME/ShapePipe_github/shapepipe/example/W3_simu/config_make_simu.ini

# Return exit code
exit 0
