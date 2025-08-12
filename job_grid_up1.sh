#!/bin/bash
#SBATCH --partition=SP2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=20000 
#SBATCH -J ic1_optuna
#SBATCH --time=100:00:00
#SBATCH -o out1.txt
#SBATCH -e err1.txt

# 1. Carregar o módulo base do Miniconda
module load Miniconda/biopython

# 2. Criar ou ativar um ambiente virtual dedicado
conda create -n my_optuna_env --clone Miniconda/biopython --yes
conda activate my_optuna_env

# 3. Instalar as dependências no seu ambiente virtual
pip install -r requirements.txt

# 4. Executar o seu script Python
python optune_from_scratch.py model=ft_transformer dataset=ic_upstream1

# 5. (Opcional) Desativar o ambiente após a execução
conda deactivate