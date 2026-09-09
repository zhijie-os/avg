#!/bin/bash

#SBATCH --job-name=aba_combined
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=aba_combined_%j.out
#SBATCH --error=aba_combined_%j.err

module load python/3.11
module load gcc
module load opencv
module load mujoco

source ~/avg/ENV/bin/activate
cd ~/avg

python3 run_min_delay_aba.py --shift-type combined --actuator-index 0
