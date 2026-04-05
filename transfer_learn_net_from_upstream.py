from omegaconf import open_dict
import json
import logging
import os
import sys
from collections import OrderedDict
import copy
import multiprocessing as mp
from typing import Dict
import gc
from hydra import experimental, compose, initialize
import hydra
import numpy as np
import torch
from icecream import ic
from omegaconf import DictConfig, OmegaConf
import train_net_from_scratch
import transfer_learn_net
import deep_tabular as dt

N_JOBS_MAX =   1 # Número máximo de jobs do HPC DA USP
import numpy as np

FILE_NAME= "results.jsonl"

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

def omegaconf_to_serializable(oc):
    if OmegaConf.is_config(oc):
        oc = OmegaConf.to_container(oc, resolve=True)
    return make_serializable(oc)
def run_job(model_cfg, dataset_cfg, hyp_cfg, configName, log, from_scratch=False, results_file=FILE_NAME):
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
            "config": omegaconf_to_serializable(cfgExecution),
            "stats": stats
        }

    except Exception as e:
        logging.getLogger().error(f"Erro no job {configName}: {e}")
        result = {
            "config": make_serializable({
                "model": omegaconf_to_serializable(model_copy),
                "dataset": omegaconf_to_serializable(dataset_copy),
                "hyp": omegaconf_to_serializable(hyp_copy),
                "run_id": f"{configName}_ERROR"
            }),
            "error": str(e)
        }

    log.info(result)
    try:
        file = get_jsonl_files()
        if file and len(file) > 0:
            results_file = file[0]  # Usa o primeiro arquivo encontrado
        results_dir = os.path.dirname(results_file)
        if results_dir:  # só cria se não for string vazia
            os.makedirs(results_dir, exist_ok=True)

        with open(results_file, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(result, ensure_ascii=False) + "\n")
    except Exception as e:
        log.error(f"Erro ao salvar resultado de {configName}: {e}")
    gc.collect()
    torch.cuda.empty_cache() 
    return

##For the data levels of 4 and 10 samples, we simply select 30 fine-tuning epochs. For more data of
#20 samples, we select 60 fine-tuning epochs. In the larger data levels of 100 and 200 samples, we
#sample 20% of the data as a validation set to perform early stopping with the flexible end-to-end
#fine-tuned transfer learning setups prone to overfitting. For early stopping, we terminates training
#if no improvement in the validation score is observed for more than 30 epochs. In the less flexible
#transfer learning setups with a frozen feature extractor, 
#feature extractor. Finally, for the deep baselines with the hyperparameters tuned on a small subsample
#of the upstream data, we select the best epoch from the small upstream subsample.
def select_epoch(samples: int, mlpHead: bool, freeze: bool, from_scratch: bool, plateau_stop: bool, fixed_epochs: bool) -> int:
    if plateau_stop:
        return 1000
    if fixed_epochs:
        return 500
    if from_scratch:
        return 200
    if freeze:
        return 100 if mlpHead else 200  
        #we select 100 fine-tuning epochs for the MLP head atop a frozen feature extractor and 200 fine-tuning epochs for the linear head atop a frozen
        # Congelando, precisa de mais épocas para ajustar a cabeça e já reduz o risco de overfitting
        # Se usar mlpHead, são mais parâmetros, então consegue aprender mais rápido
    if samples <= 20:
        return 60
    elif samples <= 50:
        return 90
    else:
        return 200

def selectLearningRate(lr: float, from_scratch: bool):
    if from_scratch:
        return lr
    return lr/2

def selectHeadLearningRate(mlpHead: bool, freeze: bool, base_lr: float, upstream_head_lr: float) -> float:
    if mlpHead or freeze:
        return base_lr
    return upstream_head_lr

def getImpuationMethodSufix(imputationMethod: str, upstream_number: int) -> str:
    if upstream_number < 0 or upstream_number == None:
        return f"{imputationMethod}"
    return f"{imputationMethod}_exp_100_{upstream_number}"

def get_jsonl_files():
    jsonl_files = []
    for root, _, files in os.walk(os.getcwd()):
        for file in files:
            if file.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, file))
    return jsonl_files

def configsAlreadyExecuted() -> Dict[str, bool]:
    executed_configs = {}
    files = get_jsonl_files()
    if not files or len(files) == 0:
        return executed_configs
    with open(files[0], "r", encoding="utf-8") as fp:  # Assuming the first file is the one to check
        for line in fp:
            try:
                result = json.loads(line)
                if "config" in result and "run_id" in result["config"]:
                    executed_configs[result["config"]["run_id"]] = True
            except json.JSONDecodeError:
                continue
    return executed_configs

# ============================
# Hydra main
# ============================
@hydra.main(config_path="config", config_name="transfer_learn_net_from_upstream_config")
def main(cfg: DictConfig):
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("train_net_from_scratch.py main() running.")
    benchmark = False
    log.info(OmegaConf.to_yaml(cfg))
    config = cfg["preTrained"]
    model = config["model"]
    hyp = config["hyp"]
    with open_dict(hyp):
        hyp["save_all_epochs"] = True
    hyp["use_patience"] = False  
    hyp["save_all_epochs"] = True
    hyp["val_period"] = 1
    # Como downstream é muito pequeno, não há dataset de validação, e por isso não se usa paciência
    # Caso seja possível usar paciência, ainda assim, é preciso garantir que o model_best.pth seja salvo 
    # em uma pasta diferente para cada job, de forma a evitar conflitos
    upstream_number = config["number"]
    model['model_path'] = config["model_path"]
    downstreamName= 'japan' if benchmark else "ic_downstream1"
    sampleSizes = [25, 50, 75, 100] if benchmark else [5, 10, 20, 50, 75]
    seeds = [2, 12, 22, 32, 42, 52, 62, 72, 82, 92]
    # Monta todos os jobs a serem executados
    jobs = []
    configs_executed = configsAlreadyExecuted()
    imputationMethod = getImpuationMethodSufix(config["imputationMethod"], upstream_number)


    FIXED_EPOCHS = True

    for seed in seeds:
        for sample in sampleSizes:
            if not benchmark:
                dataset_name = f"{downstreamName}_Sample{sample}_Imputation_{imputationMethod}"
            else:
                dataset_name = f"{downstreamName}_Sample{sample}"
            full_name = f"{dataset_name}"
            dataset_cfg = {
                "name": full_name,
                "source": "local",
                "task": "regression",
                "normalization": "quantile",
                "normalizer_path": config["normalizer_path"],
                "stage": "downstream",
                "y_policy": ""
            }
            for mlpHead in [True, False]: 
                for freeze in [True, False]:
                    is_from_scratch = False
                    # Cria cópia independente do model para cada job
                    model_cfg = copy.deepcopy(model)
                    hyp_cfg = copy.deepcopy(hyp)
                    hyp_cfg["seed"] = seed
                    model_cfg["use_mlp_head"] = mlpHead
                    model_cfg["freeze_feature_extractor"] = freeze
                    hyp_cfg["epochs"] = select_epoch(sample, mlpHead, freeze, is_from_scratch, hyp["plateau_stop"], FIXED_EPOCHS)
                    hyp_cfg["head_lr"] = selectHeadLearningRate(mlpHead, freeze, hyp["lr"], hyp["head_lr"])
                    hyp_cfg["lr"] = selectLearningRate(hyp["lr"], is_from_scratch)
                    configName = f"{dataset_name}_upstream{upstream_number}_mlpHead{mlpHead}_freeze{freeze}_seed{seed}"
                    if configName not in configs_executed:
                        # Adiciona à lista de jobs
                        jobs.append((model_cfg, dataset_cfg, hyp_cfg, configName, log, False))
            is_from_scratch = True

            ## Model from scratch
            model_from_scratch = copy.deepcopy(model)
            model_from_scratch["model_path"] = None
            model_from_scratch["use_mlp_head"] = False
            model_from_scratch["freeze_feature_extractor"] = False

            # HyperParameter configs adapted for training from scratch
            hyp_from_scratch = copy.deepcopy(hyp)
            hyp_from_scratch["seed"] = seed
            hyp_from_scratch["lr"] = selectLearningRate(hyp["lr"], is_from_scratch)
            hyp_from_scratch["epochs"] = select_epoch(
                sample,
                (not is_from_scratch),
                (not is_from_scratch),
                is_from_scratch,
                hyp["plateau_stop"],
                FIXED_EPOCHS
            )
            


            # Dataset without imputation to validate if the imputation is actually helping or not
            dataset_fs = copy.deepcopy(dataset_cfg)
            dataset_fs["name"] = f"{downstreamName}_Sample{sample}"
            dataset_fs["normalizer_path"] = None
            
            
            configName = f"{dataset_fs['name']}_hypParamsFrom-{upstream_number}-{imputationMethod}_fromScratch_seed{seed}"
            if configName not in configs_executed:
                jobs.append((model_from_scratch, dataset_fs, hyp_from_scratch, configName, log, is_from_scratch))
            
            dataset_fs_with_imputation = copy.deepcopy(dataset_cfg)

            configName = f"{configName}_Imputation_{imputationMethod}"
            if configName not in configs_executed:
                jobs.append((model_from_scratch, dataset_fs_with_imputation, hyp_from_scratch, configName, log, is_from_scratch))


    # ============================
    # Executa os jobs em paralelo de um mesmo upstream
    # ============================
    with mp.Pool(processes=N_JOBS_MAX) as pool:
        pool.starmap(run_job, jobs)

    log.info("Todos os jobs concluídos!")


# ============================
# Entry point
# ============================
if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    sys.argv.append(f"+run_id={run_id}")  # Hydra override
    try:
        main()
    except Exception as e:
        print(f"Erro fatal no main(): {e}")