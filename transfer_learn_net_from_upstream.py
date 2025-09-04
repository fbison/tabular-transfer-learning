import json
import logging
import os
import sys
from collections import OrderedDict
import copy
import multiprocessing as mp

from hydra import experimental, compose, initialize
import hydra
import numpy as np
import torch
from icecream import ic
from omegaconf import DictConfig, OmegaConf
import transfer_learn_net
import deep_tabular as dt

N_JOBS_MAX = 20  # Número máximo de jobs do HPC DA USP
import numpy as np
import torch

def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    else:
        return obj

def run_job(model_cfg, dataset_cfg, hyp_cfg, configName):
    # Copias independentes para cada job
    model_copy = copy.deepcopy(model_cfg)
    dataset_copy = copy.deepcopy(dataset_cfg)
    hyp_copy = copy.deepcopy(hyp_cfg)

    result = None
    try:
        cfgExecution = OmegaConf.create({
            "model": model_copy,
            "dataset": dataset_copy,
            "hyp": hyp_copy,
            "run_id": configName
        })

        # Executa a função principal do transfer_learn_net
        stats = transfer_learn_net.main(cfgExecution)

        # Retorna config + stats em um único objeto
        result = {
            "config": OmegaConf.to_object(cfgExecution, resolve=True),
            "stats": stats
        }

    except Exception as e:
        logging.getLogger().error(f"Erro no job {configName}: {e}")
        result = {
            "config": make_serializable({
                "model": OmegaConf.to_object(model_copy),
                "dataset": OmegaConf.to_object(dataset_copy),
                "hyp": OmegaConf.to_object(hyp_copy),
                "run_id": f"{configName}_ERROR"
            }),
            "error": str(e)
        }

    return result

# ============================
# Hydra main
# ============================
@hydra.main(config_path="config", config_name="transfer_learn_net_from_upstream_config")
def main(cfg: DictConfig):
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("train_net_from_scratch.py main() running.")
    log.info(OmegaConf.to_yaml(cfg))
    config = cfg["preTrained"]
    model = config["model"]
    hyp = config["hyp"]
    upstream_number = config["number"]
    model['model_path'] = config["model_path"]
    meanDatasets = [
        "ic_downstream1_Sample5_Imputation_Mean_exp_100_",
        "ic_downstream1_Sample10_Imputation_Mean_exp_100_",
        "ic_downstream1_Sample20_Imputation_Mean_exp_100_",
        "ic_downstream1_Sample50_Imputation_Mean_exp_100_",
        "ic_downstream1_Sample75_Imputation_Mean_exp_100_"
    ]
    gaussianDatasets = [
        "ic_downstream1_Sample5_Imputation_Gaussian_exp_100_",
        "ic_downstream1_Sample10_Imputation_Gaussian_exp_100_",
        "ic_downstream1_Sample20_Imputation_Gaussian_exp_100_",
        "ic_downstream1_Sample50_Imputation_Gaussian_exp_100_",
        "ic_downstream1_Sample75_Imputation_Gaussian_exp_100_"
    ]

    # Monta todos os jobs a serem executados
    jobs = []
    datasets = []
    if config['imputationMethod'] == 'mean':
        datasets = meanDatasets
    elif config['imputationMethod'] == 'gaussian':
        datasets = gaussianDatasets

    for dataset_name in datasets:
        full_name = f"{dataset_name}{upstream_number}"
        dataset_cfg = {
            "name": full_name,
            "source": "local",
            "task": "regression",
            "normalization": "quantile",
            "normalizer_path": config["normalizer_path"],
            "stage": "downstream",
            "y_policy": "mean_std"
        }
        for mlpHead in [True, False]: 
            for freeze in [True, False]:
                # Cria cópia independente do model para cada job
                model_cfg = copy.deepcopy(model)
                model_cfg["use_mlp_head"] = mlpHead
                model_cfg["freeze_feature_extractor"] = freeze
                configName = f"{dataset_name}_upstream{upstream_number}_mlpHead{mlpHead}_freeze{freeze}"
                # Adiciona à lista de jobs
                jobs.append((model_cfg, dataset_cfg, hyp, configName))

    # ============================
    # Executa os jobs em paralelo de um mesmo upstream
    # ============================
    all_results = []
    with mp.Pool(processes=N_JOBS_MAX) as pool:
        results = pool.starmap(run_job, jobs)
        all_results.append(results)
    
    log.info("Resultados:")
    for result in results:
        log.info(result)

    with open(os.path.join("results.json"), "w") as fp:
        json.dump(results, fp, indent=4)
    log.info("Todos os jobs concluídos!")


# ============================
# Entry point
# ============================
if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    sys.argv.append(f"+run_id={run_id}")  # Hydra override
    main()
