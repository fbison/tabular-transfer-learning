""" optuna_tools.py
    Utilities for Optuna hyperparameter tuning
    Developed for Tabular Transfer Learning project
    Usp 2025
"""

import optuna
from typing import List
import os
from optuna.trial import FrozenTrial, TrialState
from datetime import datetime, timedelta
import json
from optuna.visualization import (
        plot_optimization_history,
        plot_intermediate_values,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
        plot_contour,
        plot_edf,
    )

def define_search_space(model: str):
    if model == "ft_transformer":
        return {
            "d_embedding": optuna.distributions.CategoricalDistribution([64, 128, 256, 320, 384, 512]),
            "n_heads": optuna.distributions.CategoricalDistribution([4, 8, 16]),
            "n_layers": optuna.distributions.IntDistribution(2, 10, step=2),
            "d_ffn_factor": optuna.distributions.FloatDistribution(2/3, 8/3),
            "attention_dropout": optuna.distributions.FloatDistribution(0.0, 0.5),
            "ffn_dropout": optuna.distributions.FloatDistribution(0.0, 0.5),
            "activation": optuna.distributions.CategoricalDistribution(["reglu", "gelu", "relu"]),
            "lr": optuna.distributions.FloatDistribution(1e-5, 1e-3, log=True),
            # "weight_decay": optuna.distributions.FloatDistribution(1e-6, 1e-3, log=True),
        }

    elif model == "resnet":
        return {
            "d_embedding": optuna.distributions.IntDistribution(32, 512, step=8),
            "d_hidden_factor": optuna.distributions.FloatDistribution(1.0, 4.0),
            "n_layers": optuna.distributions.IntDistribution(1, 8),
            "hidden_dropout": optuna.distributions.FloatDistribution(0.0, 0.5),
            "residual_dropout": optuna.distributions.FloatDistribution(0.0, 0.5),
            "lr": optuna.distributions.FloatDistribution(1e-5, 1e-3, log=True),
            "weight_decay": optuna.distributions.FloatDistribution(1e-6, 1e-3, log=True),
        }

    elif model == "mlp":
        return {
            "d_embedding": optuna.distributions.IntDistribution(64, 512, step=8),
            "n_layers": optuna.distributions.IntDistribution(1, 8),
            "d_first": optuna.distributions.IntDistribution(1, 512),
            "d_middle": optuna.distributions.IntDistribution(1, 512),
            "d_last": optuna.distributions.IntDistribution(1, 512),
            "dropout": optuna.distributions.FloatDistribution(0.0, 0.5),
            "lr": optuna.distributions.FloatDistribution(1e-5, 1e-3, log=True),
            "weight_decay": optuna.distributions.FloatDistribution(1e-6, 1e-3, log=True),
        }

    else:
        raise ValueError(f"Unknown model: {model}")

def get_parameters(model, trial: optuna.trial.Trial):
    search_space = define_search_space(model)
    all_params = {k: trial._suggest(k, dist) for k, dist in search_space.items()}

    if model == "ft_transformer":
        model_keys = ["d_embedding", "n_heads", "n_layers", "d_ffn_factor",
                      "attention_dropout", "ffn_dropout", "activation"]
        training_keys = ["lr"]

    elif model == "resnet":
        model_keys = ["d_embedding", "d_hidden_factor", "n_layers",
                      "hidden_dropout", "residual_dropout"]
        training_keys = ["lr", "weight_decay"]

    elif model == "mlp":
        model_keys = ["d_embedding", "n_layers", "d_first", "d_middle", "d_last", "dropout"]
        training_keys = ["lr", "weight_decay"]

    model_params = {k: all_params[k] for k in model_keys}
    training_params = {k: all_params[k] for k in training_keys}

    # reconstruir d_layers como lista coerente
    if model == "mlp":
        n_layers = model_params["n_layers"]
        layers = []
        if n_layers >= 1:
            layers.append(model_params["d_first"])
        if n_layers > 2:
            layers.extend([model_params["d_middle"]] * (n_layers - 2))
        if n_layers > 1:
            layers.append(model_params["d_last"])
        model_params["d_layers"] = layers
        # limpar chaves auxiliares
        for k in ["d_first", "d_middle", "d_last", "n_layers"]:
            model_params.pop(k)

    return model_params, training_params

def save_graphs(study: optuna.study.Study):
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
        except Exception as e:
            print(f"Could not generate {filename}: {e}")

def load_completed_trials(path: str):
    """Lê os trials salvos em JSONL."""
    trials = []
    if not os.path.exists(path):
        return trials
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            trials.append(data)
    return trials

def load_frozen_trials(path: str, modelName:str)-> List[FrozenTrial]:
    existing_params = load_completed_trials(path)
    n_done = len(existing_params)
    print(f"Já existem {n_done} trials concluídos.")

    # Carrega todos os trials do JSONL
    trials = []
    for data in existing_params:
        params = data["config"]["model"]
        hyp = data["config"]["hyp"]
        val_score = data["stats"]["val_stats"]["score"]
        trial_number = data.get("trial_number", len(trials)+1)

        # Parâmetros relevantes
        relevant_keys = ["d_embedding", "n_heads","n_layers", "d_ffn_factor", "attention_dropout", "ffn_dropout"]

        filtered_params = {k: v for k, v in params.items() if k in relevant_keys}
        filtered_params.update({k: v for k, v in hyp.items() if k in ["lr"]})
        search_space = define_search_space(modelName)
        distributions = {k: search_space[k] for k in filtered_params.keys()}

        
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
    return trials