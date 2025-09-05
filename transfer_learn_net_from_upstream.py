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
import train_net_from_scratch
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

def run_job(model_cfg, dataset_cfg, hyp_cfg, configName, log, from_scratch=False, results_file="results.jsonl"):
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
        if from_scratch:
            stats = train_net_from_scratch.main(cfgExecution)
        else:
            stats = transfer_learn_net.main(cfgExecution)

        # Retorna config + stats em um único objeto
        result = {
            "config": OmegaConf.to_object(cfgExecution),
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

    log.info(result)
    try:
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(result, ensure_ascii=False) + "\n")
    except Exception as e:
        log.error(f"Erro ao salvar resultado de {configName}: {e}")
    return result

def select_epoch(samples: int, mlpHead: bool, freeze: bool) -> int:
    
    if freeze:
        return 100 if mlpHead else 200  
        # Congelando, precisa de mais épocas para ajustar a cabeça e já reduz o risco de overfitting
        # Se usar mlpHead, são mais parâmetros, então consegue aprender mais rápido
    if samples <= 10:
        return 30 
    elif samples <= 20:
        return 60
    elif samples <= 50:
        return 90
    else:
        return 200

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
    downstreamName= "ic_downstream1"
    sampleSizes = [5, 10, 20, 50, 75]
    seeds = [2, 12, 22, 32, 42, 52, 62, 72, 82, 92]
    # Monta todos os jobs a serem executados
    jobs = []

    for seed in seeds:
        for sample in sampleSizes:
            dataset_name = f"{downstreamName}_Sample{sample}_Imputation_{config['imputationMethod']}_exp_100_{upstream_number}"
            full_name = f"{dataset_name}"
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
                    hyp_cfg = copy.deepcopy(hyp)
                    hyp_cfg["epochs"] = select_epoch(sample, mlpHead, freeze)
                    hyp_cfg["lr"] = 0.00005
                    hyp_cfg["seed"] = seed
                    model_cfg["use_mlp_head"] = mlpHead
                    model_cfg["freeze_feature_extractor"] = freeze
                    configName = f"{dataset_name}_upstream{upstream_number}_mlpHead{mlpHead}_freeze{freeze}_seed{seed}"
                    # Adiciona à lista de jobs
                    jobs.append((model_cfg, dataset_cfg, hyp, log, configName, False))
            model_from_scratch = copy.deepcopy(model)
            hyp_from_scratch = copy.deepcopy(hyp)
            hyp_from_scratch["epochs"] = 200
            model_from_scratch["model_path"] = None
            model_from_scratch["use_mlp_head"] = False
            model_from_scratch["freeze_feature_extractor"] = False
            hyp_cfg["seed"] = seed
            configName = f"{dataset_name}_fromScratch_seed{seed}"
            jobs.append((model_cfg, dataset_cfg, hyp, log, configName, True))

    # ============================
    # Executa os jobs em paralelo de um mesmo upstream
    # ============================
    all_results = []
    with mp.Pool(processes=N_JOBS_MAX) as pool:
        results = pool.starmap(run_job, jobs)
        all_results.append(results)

    log.info("Todos os jobs concluídos!")


# ============================
# Entry point
# ============================
if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    sys.argv.append(f"+run_id={run_id}")  # Hydra override
    main()
