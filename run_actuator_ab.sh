#!/bin/bash

#SBATCH --job-name=actuator_ab
#SBATCH --time=1-00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=actuator_ab_%j.out
#SBATCH --error=actuator_ab_%j.err

module load python/3.11
module load gcc
module load opencv
module load mujoco

source ~/avg/ENV/bin/activate

cd ~/avg

python3 run_actuator_ab.py
