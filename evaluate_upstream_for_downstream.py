import hydra
import torch
import json
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import deep_tabular as dt


SAMPLES = [5, 10, 20, 50, 75]

# 🔥 Mapeamento correto dos nomes
IMPUTATION_DATASET_MAP = {
    "gaussian": "Gaussian",
    "mean": "Mean",
    "pseudo_features": "pseudo_features",
    "Ground_Truth": "Ground_Truth",
}

def getImpuationMethodSufix(imputationMethod: str, upstream_number: int) -> str:
    if upstream_number < 0 or upstream_number == None:
        return f"{imputationMethod}"
    return f"{imputationMethod}_exp_100_{upstream_number}"

def get_up_number_from_upstream_id(upstream_id: str) -> int:
    import re

    match = re.search(r'(\d+)', upstream_id)
    if not match:
        raise ValueError(f"Invalid upstream_id: {upstream_id}")
    
    return int(match.group(1))

def build_upstream_paths(upstream_id: str, imputation: str):
    """
    upstream_id pode vir como:
    - 'up2'
    - 'upstream2'
    """
    
    up_num = get_up_number_from_upstream_id(upstream_id)

    imp_dataset = IMPUTATION_DATASET_MAP[imputation]

    upstream_short = f"up{up_num}"        # model/hyp
    upstream_full = f"upstream{up_num}"   # dataset

    base_name = f"pretrain_{upstream_short}_{imputation}".lower()

    # dataset yaml
    dataset_name = f"ic_{upstream_full}_Imputation_{imp_dataset}_exp_100_1"

    model_cfg = f"../../../config/model/mlp_{base_name}.yaml"
    hyp_cfg = f"../../../config/hyp/hyp_{base_name}.yaml"
    dataset_cfg = f"../../../config/dataset/{dataset_name}.yaml"

    # 🔥 NOVO PATH CORRETO
    model_path = (
        f"outputs/transfer-learning-experiment/"
        f"mlp-{dataset_name}/model_best.pth"
    )

    return model_cfg, hyp_cfg, dataset_cfg, model_path, dataset_name
def load_upstream_model(upstream_id: str, imputation: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg_path, hyp_cfg_path, dataset_cfg_path, model_path, dataset_name = build_upstream_paths(
        upstream_id, imputation
    )

    modelConfig = OmegaConf.load(model_cfg_path)
    modelConfig["model_path"] = model_path

    hypConfig = OmegaConf.load(hyp_cfg_path)
    datasetConfig = OmegaConf.load(dataset_cfg_path)

    cfg = OmegaConf.create({
        "model": modelConfig,
        "dataset": datasetConfig,
        "hyp": hypConfig,
        "run_id": dataset_name
    })

    _, unique_categories, n_numerical, n_classes, data_schema, _ = dt.utils.get_dataloaders(cfg)

    net, _, _, data_schema_loaded, y_normalizer = dt.utils.load_model_from_checkpoint(
        modelConfig,
        n_numerical,
        unique_categories,
        n_classes,
        device,
        data_schema
    )

    return net, data_schema_loaded, y_normalizer, device, cfg

def getImpuationMethodSufix(imputationMethod: str, upstream_number: int) -> str:
    if upstream_number == None:
        return f"{imputationMethod}"
    return f"{imputationMethod}_exp_100_{upstream_number}"

def build_downstream_cfg(
    downstream_name,
    sample,
    imputation,
    normalizer_path,
    up_number=None
):
    imp_dataset = IMPUTATION_DATASET_MAP[imputation]

    dataset_name = f"{downstream_name}_Sample{sample}_Imputation_{getImpuationMethodSufix(imp_dataset, up_number)}"

    return {
        "name": dataset_name,
        "source": "local",
        "task": "regression",
        "normalization": "quantile",
        "normalizer_path": normalizer_path,
        "stage": "downstream",
        "y_policy": ""
    }


def evaluate_upstream_on_downstream(
    upstream_id: str,
    imputation: str,
    downstream_name: str
):
    net, _, _, device, cfg_upstream  = load_upstream_model(
        upstream_id, imputation
    )
    upstream_dataset_cfg = cfg_upstream["dataset"]
    results = []

    normalizer_path = upstream_dataset_cfg["normalizer_path"]
    up_num = get_up_number_from_upstream_id(upstream_id)
    for sample in SAMPLES:

        dataset_cfg = build_downstream_cfg(
            downstream_name,
            sample,
            imputation,
            normalizer_path,
            up_number=up_num
        )

        cfg = OmegaConf.create({
            "dataset": dataset_cfg,
            "hyp": cfg_upstream["hyp"],
            "model": cfg_upstream["model"]
        })

        loaders, *_ = dt.utils.get_dataloaders(cfg)

        test_stats, val_stats, train_stats = dt.evaluate_model(
            net,
            [loaders["test"], loaders["val"], loaders["train"]],
            dataset_cfg["task"],
            device
            #y_info_normalization=y_norm
        )

        results.append({
            "upstream": upstream_id,
            "imputation": imputation,
            "downstream": downstream_name,
            "sample": sample,
            "test_stats": test_stats,
            "val_stats": val_stats,
            "train_stats": train_stats
        })

    return results


def generate_all_pairs(downstream_name: str):
    upstreams = ["up2", "up3", "up4"]
    imputations = ["gaussian", "mean", "pseudo_features", "Ground_Truth"]

    pairs = []
    for up in upstreams:
        for imp in imputations:
            pairs.append((up, imp, downstream_name))

    return pairs


def evaluate_multiple_pairs(pairs):
    all_results = []

    for upstream_id, imputation, downstream in pairs:
        print(f"Running: {upstream_id} | {imputation} | {downstream}")

        res = evaluate_upstream_on_downstream(
            upstream_id,
            imputation,
            downstream
        )

        all_results.extend(res)

    return all_results


@hydra.main(config_path=None)
def main(cfg: DictConfig):

    downstream_name = "ic_downstream1"

    pairs = generate_all_pairs(downstream_name)

    results = evaluate_multiple_pairs(pairs)

    output_path = Path("results.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()