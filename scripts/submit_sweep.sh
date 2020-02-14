#!/bin/bash

#SBATCH -C gpu -N 1 -t 8:00:00 -c 10
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --exclusive
#SBATCH -o logs/%x-%j.out
#SBATCH -J sweep-testing
#SBATCH -A m1759

# source $VENV/reqtest/bin/activate

module load esslurm
module load pytorch/v1.2.0-gpu


echo -e "\nStarting preprocessing\n"

for i in {0..7}; do
    echo "Launching task $i"
    srun -N 1 -n 1 -G 1 wandb agent murnanedaniel/convnet-toy/mpjidtgm &
done
wait
