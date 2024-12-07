#!/bin/bash
#SBATCH --qos=<qos_name>
#SBATCH -p <partition_name>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --time=80:00:00
#SBATCH --exclude=node021
#SBATCH --output=./output.sh
date;hostname;id;pwd

echo 'activating virtual environment'
source ~/.bashrc
source activate pytorch

chmod u=rwx,g=r,o=r /program.sh
chmod u=rwx,g=r,o=r /train.py
module load <namespace>/gcc/11.1.0

DIRS=($(ls -v -d ./results_new/*))
idx=${DIRS[-1]}
idx=${idx:14}

new_dir=./results_new/$((${idx}+1))
mkdir ${new_dir}

echo 'running script'
srun -n 1 /program.sh
