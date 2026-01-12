import os
import json
import optuna
import hydra
import sys
import torch
""" optune_from_scratch.py
    Tune neural networks using Optuna
    Developed for Tabular Transfer Learning project
    March 2022
"""

import time
import train_net_for_optuna
import hydra
import optuna
import sys
import deep_tabular as dt
import os
import copy
from omegaconf import DictConfig, OmegaConf
import json
import torch
import multiprocessing
import copy
import deep_tabular as dt
import multiprocessing
from omegaconf import DictConfig, OmegaConf
import train_net_for_optuna
from deep_tabular.utils.optuna_tools import get_parameters, load_frozen_trials, save_graphs
import gc

INPUT_PATH = r"all_trials.jsonl"
STORAGE_PATH = "sqlite:///optuna_study.db"
N_TOTAL_TRIALS = 200
N_JOBS = 20  # número de processos paralelos


SAVE_INTERVAL = 5  # salva a cada 5 trials

def sample_value_with_default(trial, name, distr, min, max, default):
    # chooses suggested or default value with 50/50 chance
    if distr == 'uniform':
        value_suggested = trial.suggest_uniform(name, min, max)
    elif distr == 'loguniform':
        value_suggested = trial.suggest_loguniform(name, min, max)
    value = value_suggested if trial.suggest_categorical(f'optional_{name}', [False, True]) else default
    return value
#
import optuna

def objective(trial, cfg: DictConfig, trial_stats, 
              trial_counter, n_total_trials, 
              loaders, unique_categories, n_numerical, n_classes, run_id, lock
              ):
    gc.collect()
    # Use a lock to safely get and increment the trial counter
    with lock:
        current_trial = trial_counter[0] + 1
        trial_counter[0] = current_trial
    print(f"Running trial {current_trial}/{n_total_trials}")
    # Generate a unique directory name for this trial
    trial_run_id = f"{run_id}_trial_{current_trial}"
    try:
        model_params, training_params =  get_parameters(cfg.model.name, trial) # need to suggest parameters for optuna here, probably writing a function for suggesting parameters is the optimal way
        
        config = copy.deepcopy(cfg) # create config for train_model with suggested parameters
        for par, value in model_params.items():
            config.model[par] = value
        for par, value in training_params.items():
            config.hyp[par] = value

        config.run_id = trial_run_id  # unique directory for this trial

        if cfg.hyp.save_period < 0:
            cfg.hyp.save_period = 1e8
        beginTime = time.time()
        stats = train_net_for_optuna.main(config, loaders, unique_categories, n_numerical, n_classes)
        endTime = time.time()
        time_taken = endTime - beginTime
        with lock:
            trial_stats.append(stats)
            with open("all_trials.jsonl", "a") as f:
                json.dump({
                    "config": OmegaConf.to_container(config, resolve=True),
                    "stats": stats,
                    "time_taken": time_taken,
                    "trial_number": current_trial,
                }, f)
                f.write("\n")

        gc.collect()
        return stats['val_stats']['score']
    except Exception as e:
        print(f"Trial {trial.number} with ID '{trial_run_id}' failed with an error: {e}")
        # Mark the trial as failed and prune it, so the study can continue
        raise optuna.exceptions.TrialPruned()

@hydra.main(config_path="config", config_name="optune_config")
def main(cfg):
    trial_stats = []
    # Use a shared, mutable counter protected by a lock
    manager = multiprocessing.Manager()
    trial_counter = manager.list([0])
    lock = multiprocessing.Lock()

    trial_counter = [0]  # mutable counter for tracking inside objective
    torch.manual_seed(cfg.hyp.seed)
    torch.cuda.manual_seed_all(cfg.hyp.seed)

    ####################################################
    #               Dataset and Network and Optimizer
    loaders, unique_categories, n_numerical, n_classes, data_schema = dt.utils.get_dataloaders(cfg)
    storage_path = "sqlite:///optuna_study.db"
    study = optuna.create_study(
        study_name="my_study",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
        storage=storage_path,
        load_if_exists=True
    )

    trials = load_frozen_trials(INPUT_PATH, cfg.model.name)
    n_done = len(trials)
    print(f"Já existem {n_done} trials concluídos.")
    trial_counter[0] = n_done
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    study.add_trials(trials)
    func = lambda trial: objective(trial, cfg, trial_stats, trial_counter, N_TOTAL_TRIALS,
                                   loaders, unique_categories, n_numerical, n_classes, cfg.run_id, lock)
    if n_done >= N_TOTAL_TRIALS:
        print("Já atingiu ou ultrapassou o limite de trials.")
    else:
        print("Estudo será iniciado ou continuado.")
        study.optimize(func, n_trials=(N_TOTAL_TRIALS-n_done), n_jobs=N_JOBS, show_progress_bar=True)

    best_trial = study.best_trial

    save_graphs(study)

    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))




if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    sys.argv.append(f"+run_id={run_id}")
    main()



