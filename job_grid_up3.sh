#!/bin/bash
#SBATCH --partition=SP2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=30000 
#SBATCH -J ic3_optuna
#SBATCH --time=192:00:00
#SBATCH -o out3.txt
#SBATCH -e err3.txt

# 1. Carregar o módulo base do Miniconda
module load Miniconda/biopython

# 2. Criar ou ativar um ambiente virtual dedicado
conda create -n my_optuna_env --clone Miniconda/biopython --yes
conda activate my_optuna_env

# 3. Instalar as dependências no seu ambiente virtual
pip install -r requirements.txt

# 4. Executar o seu script Python
python optune_from_scratch.py model=ft_transformer dataset=ic_upstream3_ImputacaoEstatistica_exp_100_1

# 5. (Opcional) Desativar o ambiente após a execução
conda deactivate