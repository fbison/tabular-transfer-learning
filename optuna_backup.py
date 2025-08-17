import os
import json
import optuna
from optuna.trial import FrozenTrial, TrialState
from datetime import datetime, timedelta
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour,
    plot_edf,
)
import plotly.io as pio

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

# Caminho para o arquivo com todos os trials
input_path = r"outputs\from_scratch_optuna\optuning-ft_transformer-ic_upstream3\all_trials.jsonl"
output_dir = os.path.dirname(input_path)

# Carrega todos os trials do JSONL
trials = []
with open(input_path, "r") as f:
    for line in f:
        data = json.loads(line)
        params = data["config"]["model"]
        val_score = data["stats"]["val_stats"]["score"]
        trial_number = data.get("trial_number", len(trials)+1)
        if len(trials) < 18:
            trial_number = len(trials) + 1
        else:
            trial_number = data.get("trial_number") + 18

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

# Cria o study em memória
study = optuna.create_study(direction="maximize")
study.add_trials(trials)

# Lista de gráficos possíveis
plots = {
    "optimization_history": plot_optimization_history,
    "param_importance": plot_param_importances,
    "parallel_coordinate": plot_parallel_coordinate,
    "slice_plot": plot_slice,
    "contour_plot": plot_contour,
    "edf_plot": plot_edf,
}

# Gera e salva os gráficos em PNG
for filename, plot_func in plots.items():
    try:
        fig = plot_func(study)
        save_path = os.path.join(output_dir, filename)
        fig.write_html(save_path + ".html")
        fig.write_image((save_path + ".png"), width=1000, height=600)
        print(f"Salvo: {save_path}")
    except Exception as e:
        print(f"Erro ao gerar {filename}: {e}")

best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))

best_stats = trials[best_trial.number]

with open(os.path.join(output_dir, "best_stats.json"), "w") as fp:
    json.dump(best_stats, fp, indent = 4)
with open(os.path.join(output_dir, "best_config.json"), "w") as fp:
    json.dump(best_trial.params, fp, indent = 4)