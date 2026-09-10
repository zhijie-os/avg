#!/bin/bash
#SBATCH --job-name=avg_run
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

module load python/3.11
module load gcc
module load opencv
module load mujoco

source ~/avg/ENV/bin/activate
cd ~/avg

mkdir -p logs
mkdir -p results

SCRIPT=$1
shift

echo "Running: $SCRIPT"
echo "Extra args: $@"

python "$SCRIPT" \
    --results_dir ./results \
    --device cpu \
    "$@"