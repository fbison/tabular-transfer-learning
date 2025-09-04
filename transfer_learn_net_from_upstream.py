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

def run_job(model_cfg, dataset_cfg, hyp_cfg, configName):
    # Copias independentes para cada job
    model_copy = copy.deepcopy(model_cfg)
    dataset_copy = copy.deepcopy(dataset_cfg)
    hyp_copy = copy.deepcopy(hyp_cfg)

    stats = None
    try:
        # Inicializa um contexto Hydra para este job
        with initialize(config_path="config", job_name=f"{configName}_job"):
            # Compondo a config base do YAML
            cfgExecution = compose(config_name="transfer_learn_net_config")
            # Sobrescreve os campos específicos do job
            cfgExecution.model = model_copy
            cfgExecution.dataset = dataset_copy
            cfgExecution.hyp = hyp_copy
            OmegaConf.set_struct(cfgExecution, False)
            cfgExecution = OmegaConf.merge(
                cfgExecution,
                DictConfig({"run_id": configName})
            )
            OmegaConf.set_struct(cfgExecution, True)

            # Chama a main do módulo transfer_learn_net
            stats = transfer_learn_net.main(cfgExecution)

    except Exception as e:
        logging.getLogger().error(f"Erro no job {configName}: {e}")
        stats = {"config": configName, "error": str(e)}

    return stats


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
    log.info("Todos os jobs concluídos!")


# ============================
# Entry point
# ============================
if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    sys.argv.append(f"+run_id={run_id}")  # Hydra override
    main()
