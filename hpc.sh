#!/bin/bash
#
#SBATCH --job-name=mvtn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#
#SBATCH --mail-type=END
#SBATCH --mail-user=cc6858@nyu.edu

module purge;
module load anaconda3/2020.07;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /home/cc6858/.conda/envs/dense;
export PATH=/home/cc6858/.conda/envs/dense/bin:$PATH;

cd /scratch/$USER/mvtn
./experiments/scripts/train_point.sh