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

def get_parameters(model, trial):
    if model=='ft_transformer':
        valid_pairs = [(64, 8), (128, 4), (128, 8), (256, 4), (256, 8), (320, 8), (384, 8), (384, 12), (512, 8), (512, 16), ]
        chosen = trial.suggest_categorical("embedding_head_pair", valid_pairs)
        d_embedding, n_heads = chosen
        model_params = {
            'd_embedding': d_embedding,
            'n_heads': n_heads,
            'n_layers': trial.suggest_int('n_layers', 2, 10, step=2),
            'd_ffn_factor': trial.suggest_uniform('d_ffn_factor', 2/3, 8/3),
            'attention_dropout': trial.suggest_uniform('attention_dropout', 0.0, 0.5),
            'ffn_dropout' : trial.suggest_uniform('attention_dropout', 0.0, 0.5),
            "activation": trial.suggest_categorical("activation", ["reglu", "gelu", "relu"]),
            }
        training_params = {
            'lr':  trial.suggest_loguniform('lr', 1e-5, 1e-3) ##,
            ##'weight_decay':  trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
            }

    if model=='resnet':
        model_params = {
            'd_embedding':  trial.suggest_int('d_embedding', 32, 512, step=8),
            'd_hidden_factor': trial.suggest_uniform('d_hidden_factor', 1.0, 4.0),
            'n_layers': trial.suggest_int('n_layers', 1, 8,),
            'hidden_dropout': trial.suggest_uniform('residual_dropout', 0.0, 0.5),
            'residual_dropout': sample_value_with_default(trial, 'residual_dropout', 'uniform', 0.0, 0.5, 0.0),
            }
        training_params = {
            'lr':  trial.suggest_loguniform('lr', 1e-5, 1e-3),
            'weight_decay':  sample_value_with_default(trial, 'weight_decay', 'loguniform', 1e-6, 1e-3, 0.0),
            }

    if model=='mlp':
        n_layers = trial.suggest_int('n_layers', 1, 8)
        suggest_dim = lambda name: trial.suggest_int(name, 1, 512)
        d_first = [suggest_dim('d_first')] if n_layers else []
        d_middle = ([suggest_dim('d_middle')] * (n_layers - 2) if n_layers > 2 else [])
        d_last = [suggest_dim('d_last')] if n_layers > 1 else []
        layers = d_first + d_middle + d_last

        model_params = {
            'd_embedding':  trial.suggest_int('d_embedding', 32, 512, step=8),
            'd_layers': layers,
            'dropout': sample_value_with_default(trial, 'dropout', 'uniform', 0.0, 0.5, 0.0),
            }
        training_params = {
            'lr':  trial.suggest_loguniform('lr', 1e-5, 1e-3),
            'weight_decay':  sample_value_with_default(trial, 'weight_decay', 'loguniform', 1e-6, 1e-3, 0.0),
            }

    return model_params, training_params



def objective(trial, cfg: DictConfig, trial_stats, 
              trial_counter, n_total_trials, 
              loaders, unique_categories, n_numerical, n_classes
              ):

    current_trial = trial_counter[0] + 1
    print(f"Running trial {current_trial}/{n_total_trials}")

    model_params, training_params =  get_parameters(cfg.model.name, trial) # need to suggest parameters for optuna here, probably writing a function for suggesting parameters is the optimal way

    config = copy.deepcopy(cfg) # create config for train_model with suggested parameters
    for par, value in model_params.items():
        config.model[par] = value
    for par, value in training_params.items():
        config.hyp[par] = value

    if cfg.hyp.save_period < 0:
        cfg.hyp.save_period = 1e8
    beginTime = time.time()
    stats = train_net_for_optuna.main(config, loaders, unique_categories, n_numerical, n_classes)
    endTime = time.time()
    time_taken = endTime - beginTime
    trial_stats.append(stats)
    trial_counter[0] += 1

    with open("all_trials.jsonl", "a") as f:
        json.dump({
            "config": OmegaConf.to_container(config, resolve=True),
            "stats": stats,
            "time_taken": time_taken,
            "trial_number": current_trial,
        }, f)
        f.write("\n")


    return stats['val_stats']['score']


@hydra.main(config_path="config", config_name="optune_config")
def main(cfg):
    n_optuna_trials = 180
    trial_stats = []
    trial_counter = [0]  # mutable counter for tracking inside objective
    torch.manual_seed(cfg.hyp.seed)
    torch.cuda.manual_seed_all(cfg.hyp.seed)

    ####################################################
    #               Dataset and Network and Optimizer
    loaders, unique_categories, n_numerical, n_classes = dt.utils.get_dataloaders(cfg)
    storage_path = "sqlite:///optuna_study.db"
    study = optuna.create_study(
        study_name="my_study",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
        storage=storage_path,
        load_if_exists=True
    )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    func = lambda trial: objective(trial, cfg, trial_stats, trial_counter, n_optuna_trials,
                                   loaders, unique_categories, n_numerical, n_classes)
    study.optimize(func, n_trials=n_optuna_trials)

    in_memory_study = study  # assume it's still available in scope

    # Create a persistent study
    storage_path = "sqlite:///optuna_study.db"
    persistent_study = optuna.create_study(
        study_name="my_study",
        direction=in_memory_study.direction,
        sampler=in_memory_study.sampler,
        pruner=in_memory_study.pruner,
        storage=storage_path,
        load_if_exists=True
    )

    # Copy each trial
    for trial in in_memory_study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            persistent_study.enqueue_trial(trial.params)

    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

    best_stats = trial_stats[best_trial.number]

    with open(os.path.join("best_stats.json"), "w") as fp:
        json.dump(best_stats, fp, indent = 4)
    with open(os.path.join("best_config.json"), "w") as fp:
        json.dump(best_trial.params, fp, indent = 4)
    
    from optuna.visualization import (
        plot_optimization_history,
        plot_intermediate_values,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
        plot_contour,
        plot_edf,
    )
    plots = {
        "optimization_history.html": plot_optimization_history,
        "intermediate_values.html": plot_intermediate_values,
        "param_importance.html": plot_param_importances,
        "parallel_coordinate.html": plot_parallel_coordinate,
        "slice_plot.html": plot_slice,
        "contour_plot.html": plot_contour,
        "edf_plot.html": plot_edf,
    }
    for filename, plot_func in plots.items():
        try:
            fig = plot_func(study)
            fig.write_html(os.path.join(filename))
        except Exception as e:
            print(f"Could not generate {filename}: {e}")




if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    sys.argv.append(f"+run_id={run_id}")
    main()



