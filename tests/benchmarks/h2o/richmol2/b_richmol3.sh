#!/bin/bash

#export exec="python3 /home/yachmena/RICHMOL/richmol/richmol.py"
export exec="python3 /home/yachmena/richmol/bin/richmol"
export jobname=`echo $1 | sed -e 's/\.inp//'`
export pwd=`pwd`

export jobtype="cfel"
export nproc=32 #`nproc --all`
export wclim=6
export nnodes=1

echo "Job type :" $jobtype
echo "Job name :" $jobname
echo "Requested time :" $wclim
echo "Requested number of nodes :" $nnodes
echo "Requested number of cores :" $nproc
echo "sbatch submit..."

#sbatch --partition=$jobtype --nodes=$nnodes --ntasks=$nproc --time=$wclim:00:00 --job-name=$jobname --output=$jobname.o --error=$jobname.e \
#       --workdir=$pwd $pwd/run_richmol3.sh

$pwd/run_richmol3.sh
