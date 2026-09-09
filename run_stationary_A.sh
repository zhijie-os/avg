#!/bin/bash

#SBATCH --job-name=stat_A
#SBATCH --time=2-00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=stat_A_%j.out
#SBATCH --error=stat_A_%j.err

module load python/3.11
module load gcc
module load opencv
module load mujoco

source ~/avg/ENV/bin/activate
cd ~/avg

python3 run_min_delay_stationary.py --regime A
