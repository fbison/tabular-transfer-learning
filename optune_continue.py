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
from optuna.trial import FrozenTrial, TrialState
import json
import torch
import multiprocessing
import copy
import deep_tabular as dt
import multiprocessing
from omegaconf import DictConfig, OmegaConf
import train_net_for_optuna
from datetime import datetime, timedelta
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour,
    plot_edf,
)

INPUT_PATH = r"all_trials.jsonl"
STORAGE_PATH = "sqlite:///optuna_study.db"
N_TOTAL_TRIALS = 5


def load_completed_trials():
    """Lê os trials salvos em JSONL."""
    trials = []
    with open(INPUT_PATH, "r") as f:
        for line in f:
            data = json.loads(line)
            trials.append(data)
    return trials


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
        model_params = {
            'd_embedding': trial.suggest_categorical("d_embedding", [64, 128, 256, 320, 384, 512]),
            'n_heads': trial.suggest_categorical("n_heads", [4, 8, 16]),
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
              loaders, unique_categories, n_numerical, n_classes, run_id, lock
              ):

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

        return stats['val_stats']['score']
    except Exception as e:
        print(f"Trial {trial.number} with ID '{trial_run_id}' failed with an error: {e}")
        # Mark the trial as failed and prune it, so the study can continue
        raise optuna.exceptions.TrialPruned()


def infer_distribution(key, value):
    if isinstance(value, bool):
        # treat booleans as categorical
        return optuna.distributions.CategoricalDistribution([True, False])
    elif isinstance(value, int):
        return optuna.distributions.IntDistribution(low=value, high=value)
    elif isinstance(value, float):
        return optuna.distributions.FloatDistribution(low=value, high=value)
    elif isinstance(value, str):
        return optuna.distributions.CategoricalDistribution([value])
    elif value is None:
        return optuna.distributions.CategoricalDistribution([None])
    else:
        raise ValueError(f"Unsupported param type: {key} = {value} ({type(value)})")


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

        # Carrega trials já feitos do JSONL
    existing_params = load_completed_trials()
    n_done = len(existing_params)
    print(f"Já existem {n_done} trials concluídos.")

    # Carrega todos os trials do JSONL
    trials = []
    for data in existing_params:
        params = data["config"]["model"]
        val_score = data["stats"]["val_stats"]["score"]
        trial_number = data.get("trial_number", len(trials)+1)

        # Parâmetros relevantes
        relevant_keys = ["d_embedding", "n_heads","n_layers", "d_ffn_factor", "attention_dropout", "ffn_dropout", "lr"]

        filtered_params = {k: v for k, v in params.items() if k in relevant_keys}
        distributions = {k: infer_distribution(k, v) for k, v in filtered_params.items()}

        frozen = FrozenTrial(
            number=trial_number,
            value=val_score,
            state=TrialState.COMPLETE,
            params=filtered_params,
            distributions = distributions,
            user_attrs=data,
            system_attrs={},
            intermediate_values={},
            datetime_start=datetime.now(),  # <-- required if not WAITING
            datetime_complete=datetime.now() + timedelta(seconds=1),  # optional
            trial_id=trial_number,
        )
        trials.append(frozen)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    study.add_trials(trials)
    func = lambda trial: objective(trial, cfg, trial_stats, trial_counter, n_optuna_trials,
                                   loaders, unique_categories, n_numerical, n_classes, cfg.run_id, lock)
    if n_done >= N_TOTAL_TRIALS:
        print("Já atingiu ou ultrapassou o limite de trials.")
    else:
        study.optimize(func, n_trials=N_TOTAL_TRIALS, n_jobs=20, show_progress_bar=True)

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
            save_path = os.path.join(filename)
            fig.write_html(save_path + ".html")
            fig.write_image((save_path + ".png"), width=1000, height=600)
            print(f"Salvo: {save_path}")
        except Exception as e:
            print(f"Erro ao gerar {filename}: {e}")




if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    sys.argv.append(f"+run_id={run_id}")
    main()



