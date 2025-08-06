#!/bin/bash
#SBATCH --partition=SP2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20000 
#SBATCH -J optuna_ic2
#SBATCH --time=30:00:00
#SBATCH -o out2.txt
#SBATCH -e err2.txt

python optune_from_scratch.py model=ft_transformer dataset=ic_upstream3
