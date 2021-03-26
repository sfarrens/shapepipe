#!/bin/bash

##########################
# MPI Script for Candide #
##########################

# Receive email when job finishes or aborts
#PBS -M axel.guinot@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N shapepipe_w3_mpi
# Join output and errors in one file
#PBS -j oe
# Queue
## #PBS -q batch
# Request number of cores
#PBS -l nodes=n01:ppn=48+n02:ppn=48+n03:ppn=48+n08:ppn=21+n09:ppn=48
##+n03:ppn=48
## +n02:ppn=46+n06:ppn=30+n07:ppn=30+n08:ppn=46+n09:ppn=46+n10:ppn=22+n11:ppn=22+n12:ppn=22+n13:ppn=30+n14:ppn=30+n15:ppn=30
# n13:ppn=30
#PBS -l walltime=48:00:00
NSLOTS=`cat $PBS_NODEFILE | wc -l`

# Activate conda environment
module load intelpython/3
module load openmpi/4.0.2
source activate $HOME/.conda/envs/shapepipe_github
export OMP_NUM_THREADS=1

# Run ShapePipe
cd $HOME/ShapePipe_github/shapepipe
/softs/openmpi/4.0.2-torque-CentOS7/bin/mpiexec -np $NSLOTS $HOME/.conda/envs/shapepipe_github/bin/python shapepipe_run.py -c /home/guinot/ShapePipe_github/shapepipe/example/test/config_exp.ini

# Return exit code
exit 0
