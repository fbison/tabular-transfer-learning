#!/bin/bash
#SBATCH --partition=SP2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=20000 
#SBATCH -J ic3DsGaus
#SBATCH --time=192:00:00
#SBATCH -o out3DsGaus.txt
#SBATCH -e err3DsGaus.txt

# 1. Carregar o módulo base do Miniconda
module load Miniconda/biopython

# 2. Criar ou ativar um ambiente virtual dedicado
conda create -n my_optuna_env --clone Miniconda/biopython --yes
conda activate my_optuna_env

# 3. Instalar as dependências no seu ambiente virtual
pip install -r requirements.txt

# 4. Executar o seu script Python
python transfer_learn_net_from_upstream.py preTrained=optuning-ft_transformer-ic_upstream3_Imputation_Gaussian_exp_100_1

# 5. (Opcional) Desativar o ambiente após a execução
conda deactivate