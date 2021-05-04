#!/bin/bash

export OMP_NUM_THREADS=$nproc
export OMP_STACKSIZE=8000m
export KMP_STACKSIZE=8000m
ulimit -s unlimited

module load intel/2016
source /usr/local/bin/intel-setup-2016.sh  intel64

echo "Number of OMP threads :" $OMP_NUM_THREADS
echo "OMP stack size :" $OMP_STACKSIZE
echo "KMP stack size :" $KMP_STACKSIZE
echo "ulimit :" `ulimit`
echo "Current directory :" $pwd
echo "Executable :" $exec
echo "Job name :" $jobname
echo "Running on master node :" `hostname`
echo "Job ID :" $SLURM_JOB_ID
echo "Start time :" `date`

time $exec $pwd/$jobname.inp >$pwd/$jobname.out

echo "Finish time :" date
echo "DONE"
