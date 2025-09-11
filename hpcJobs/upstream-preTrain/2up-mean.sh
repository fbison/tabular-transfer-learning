#!/bin/bash
#SBATCH --partition=SP2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20000 
#SBATCH -J ic2UpsMean
#SBATCH --time=192:00:00
#SBATCH -o out2UpsMean.txt
#SBATCH -e err2UpsMean.txt

# 1. Carregar o módulo base do Miniconda
module load Miniconda/biopython

# 2. Criar ou ativar um ambiente virtual dedicado
conda create -n my_optuna_env --clone Miniconda/biopython --yes
conda activate my_optuna_env

# 3. Instalar as dependências no seu ambiente virtual
pip install -r requirements.txt

# 4. Executar o seu script Python
python transfer_learn_net.py model=ft_transformer_pretrain_up2_mean hyp=hyp_pretrain_up2_mean dataset=ic_upstream2_Imputation_Mean_exp_100_1

# 5. (Opcional) Desativar o ambiente após a execução
conda deactivate