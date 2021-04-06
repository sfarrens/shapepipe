#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################

# Receive email when job finishes or aborts
#PBS -M axel.guinot.astro@gmail.com
#PBS -m ea
# Set a name for the job
#PBS -N shapepipe_smp
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=96:00:00
# Request number of cores
#PBS -l nodes=n17:ppn=48

# Full path to environment
#export SPENV="$HOME/.conda/envs/shapepipe"
#export SPDIR="$HOME/shapepipe"

# Activate conda environment
module load intelpython/3
#source activate $SPENV

# Run ShapePipe using full paths to executables
#python $HOME/ShapePipe_github/shapepipe/shapepipe_run.py -c $HOME/ShapePipe_github/shapepipe/example/W3_simu/config_exp_psfex.ini
python $HOME/ShapePipe_github/shapepipe/shapepipe_run.py -c $HOME/ShapePipe_github/shapepipe/example/W3_simu/ngmix_run/config_tile_Ng3.ini

# Return exit code
exit 0
