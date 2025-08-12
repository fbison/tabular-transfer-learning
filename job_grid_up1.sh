#!/bin/bash
#SBATCH --partition=SP2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=20000 
#SBATCH -J ic1_optuna
#SBATCH --time=100:00:00
#SBATCH -o out1.txt
#SBATCH -e err1.txt

python optune_from_scratch.py model=ft_transformer dataset=ic_upstream1
