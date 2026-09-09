#!/bin/bash

#SBATCH --job-name=aba_joint
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=aba_joint_%j.out
#SBATCH --error=aba_joint_%j.err

module load python/3.11
module load gcc
module load opencv
module load mujoco

source ~/avg/ENV/bin/activate
cd ~/avg

python3 run_min_delay_aba.py --shift-type actuator --actuator-index 0
