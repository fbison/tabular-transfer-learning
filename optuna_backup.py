import os
import json
import optuna
from optuna.trial import FrozenTrial, TrialState
from datetime import datetime, timedelta
import plotly.io as pio
from deep_tabular.utils.optuna_tools import load_frozen_trials, save_graphs

# Caminho para o arquivo com todos os trials
#optuning-ft_transformer-ic_upstream3_Imputation_Mean_exp_100_1
#optuning-ft_transformer-ic_upstream4_Imputation_Gaussian_exp_100_1
MODEL_NAME = "ft_transformer"
INPUT_PATH = r"outputs\from_scratch_optuna\optuning-mlp-ic_upstream2_Imputation_Mean_exp_100_1\all_trials.jsonl"
   
if __name__ == "__main__":

    output_dir = os.path.dirname(INPUT_PATH)

    # Carrega todos os trials do JSONL
    trials = load_frozen_trials(INPUT_PATH, MODEL_NAME)
    n_done = len(trials)
    print(f"Já existem {n_done} trials concluídos.")

    # Cria o study em memória
    study = optuna.create_study(direction="maximize")
    study.add_trials(trials)

    save_graphs(study)

    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

    best_stats = trials[best_trial.number]

    with open(os.path.join(output_dir, "best_config.json"), "w") as fp:
        json.dump(best_trial, fp, indent = 4)
    with open(os.path.join(output_dir, "best_stats.json"), "w") as fp:
        json.dump(best_stats, fp, indent = 4)