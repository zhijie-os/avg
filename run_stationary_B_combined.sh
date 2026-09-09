#!/bin/bash

#SBATCH --job-name=stat_B_comb
#SBATCH --time=2-00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=stat_B_comb_%j.out
#SBATCH --error=stat_B_comb_%j.err

module load python/3.11
module load gcc
module load opencv
module load mujoco

source ~/avg/ENV/bin/activate
cd ~/avg

python3 run_min_delay_stationary.py --regime B_combined --actuator-index 0
