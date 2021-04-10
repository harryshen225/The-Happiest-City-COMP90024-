#!/bin/bash
#SBATCH --ntasks-per-node=8
#SBATCH --partition=snowy
#SBATCH --nodes=1
#SBATCH --time = 0-1:00:00
module load foss/2019b
module load python/3.7.4
srun -n 8 python3 twitter_senti.py
