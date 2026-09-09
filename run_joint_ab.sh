#!/bin/bash

#SBATCH --job-name=joint_ab
#SBATCH --time=2-00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=joint_ab_%j.out
#SBATCH --error=joint_ab_%j.err

module load python/3.11
module load gcc
module load opencv
module load mujoco

source ~/avg/ENV/bin/activate

cd ~/avg

python3 run_min_delay_ab.py --shift-type actuator --actuator-index 0
