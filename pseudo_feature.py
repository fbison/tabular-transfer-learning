from omegaconf import OmegaConf
import deep_tabular as dt 
import torch
import pandas as pd
import os
import pandas as pd
import torch
import logging
from pathlib import Path

def get_imputation_model(dataset_name_used_to_train_imputation_model):
    dataset_name_used_to_train_imputation_model = dataset_name_used_to_train_imputation_model.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Caminhos dos arquivos
    model_cfg_path = (
        f"config/model/mlp_pf{dataset_name_used_to_train_imputation_model}.yaml"
    )
    hyp_cfg_path = (
        f"config/hyp/hyp_pf{dataset_name_used_to_train_imputation_model}.yaml"
    )
    dataset_cfg_path = (
        f"config/dataset/ic_upstream{dataset_name_used_to_train_imputation_model}.yaml"
    )

    # Leitura dos YAMLs
    modelConfig = OmegaConf.load(model_cfg_path)
    modelConfig["modelPath"] = f'../../../outputs/from_scratch_default/training-mlp-ic_{dataset_name_used_to_train_imputation_model}/model_best.pth'

    hypConfig = OmegaConf.load(hyp_cfg_path)
    datasetConfig = OmegaConf.load(dataset_cfg_path)
    cfgExecution = OmegaConf.create({
            "model": modelConfig,
            "dataset": datasetConfig,
            "hyp": hypConfig,
            "run_id": "configName"
        })
    # Vai pegar o dataset em que o modelo foi treinado, esses dados não serão usados, porém são necessários
    # para montar o data_schema corretamente e fazer a validação
    _, unique_categories, n_numerical, n_classes, data_schema = dt.utils.get_dataloaders(cfgExecution)
    
    net_to_pseudo_features, _, _, data_schema_loaded = dt.utils.load_model_from_checkpoint(modelConfig,
                                                                    n_numerical,
                                                                    unique_categories,
                                                                    n_classes,
                                                                    device,
                                                                    data_schema
                                                                    )
    return net_to_pseudo_features, data_schema_loaded

def impute_pseudo_features_by_file(
    file_path,
    net_to_pseudo_features,
    data_schema_loaded,
    dataset_used_to_train_model,
    device="cpu",
):
    # ---- Load dataset to be imputed ----
    dataset_to_impute = pd.read_csv(file_path)

    # ---- Extract and order input features according to schema ----
    x_features = data_schema_loaded["x"]["features"]

    # Defensive check 
    missing_features = [f for f in x_features if f not in dataset_to_impute.columns]
    if missing_features:
        logging.error(
            f"Dataset to impute is missing required input features: {missing_features}"
        )
        raise ValueError(
            "Cannot impute pseudo-features because required input features are missing."
        )

    # Order columns exactly as used during training
    dataset_x = dataset_to_impute[x_features]

    # Convert to torch tensor (IC = numerical only)
    x_tensor = torch.tensor(
        dataset_x.values,
        dtype=torch.float32,
        device=device,
    )

    # ---- Run model inference ----
    net_to_pseudo_features.eval()
    with torch.no_grad():
        y_pred = net_to_pseudo_features(x_tensor, None)

    # ---- Convert predictions to DataFrame with semantic labels ----
    y_labels = data_schema_loaded["y"]["labels"]

    features_missing_in_dataset = pd.DataFrame(
        y_pred.detach().cpu().numpy(),
        columns=y_labels,
        index=dataset_to_impute.index,
    )

    # ---- Final imputation ----
    dataset_imputed_with_pseudo_features = dataset_to_impute.copy()

    for feature in features_missing_in_dataset.columns:
        if feature not in dataset_imputed_with_pseudo_features.columns:
            dataset_imputed_with_pseudo_features[feature] = features_missing_in_dataset[feature]

    # ---- Persist result ----
    save_imputed_dataset(
        file_path,
        dataset_imputed_with_pseudo_features,
        dataset_used_to_train_model,
    )

def get_number_from_dataset_name(upstream_name):
    if "upstream2" in upstream_name:
        return 2
    elif "upstream3" in upstream_name:
        return 3
    elif "upstream4" in upstream_name:
        return 4
    elif "downstream1" in upstream_name:
        return 1
    else:
        return None
    
def save_imputed_dataset(path, dataset, dataset_used_to_train_model):
    folder_of_file = os.path.dirname(path)
    folder_new_name = f"{folder_of_file}_Imputation_Pf_exp_{get_number_from_dataset_name(dataset_used_to_train_model)}"
    path = path.replace(folder_of_file, folder_new_name)
    if not os.path.exists(folder_new_name):
        os.makedirs(folder_new_name)
    dataset.to_csv(path, index=False)

def get_dataset_folder(dataset_name):
    return f"./data/{dataset_name}/"

def get_x_files_in_dataset(dataset_name):
    x_files = []
    folder_path = get_dataset_folder(dataset_name)
    for filename in os.listdir(folder_path):
        if filename.endswith("_X.csv"):
            x_files.append(os.path.join(folder_path, filename))
    return x_files

def get_y_files_in_dataset(dataset_name):
    y_files = []
    folder_path = get_dataset_folder(dataset_name)
    for filename in os.listdir(folder_path):
        if filename.endswith("_Y.csv"):
            y_files.append(os.path.join(folder_path, filename))
    return y_files

def clean_y_files(features_to_maintain, dataset_name, dataset_used_to_train_model):
    y_files = get_y_files_in_dataset(dataset_name)
    for y_file in y_files:
        dataset_y = pd.read_csv(y_file)
        dataset_y_cleaned = dataset_y[features_to_maintain]
        save_imputed_dataset(y_file, dataset_y_cleaned, dataset_used_to_train_model)

def impute_pseudo_features_by_dataset(dataset_to_impute, dataset_name_used_to_train_imputation_model, net_to_pseudo_features, data_schema_loaded):
    x_files = get_x_files_in_dataset(dataset_to_impute)
    for x_file in x_files:
        impute_pseudo_features_by_file(x_file, net_to_pseudo_features, data_schema_loaded, dataset_name_used_to_train_imputation_model)
    clean_y_files("pic50", dataset_to_impute, dataset_name_used_to_train_imputation_model)

    
    # Salvar dataset

def get_multiple_samples_dataset_name(base_dataset_name, sample_size):
    data_dir = Path("./data")
    prefix = f"{base_dataset_name}_Sample{sample_size}"

    matching_files = [
        p.stem
        for p in data_dir.iterdir()
        if p.is_file()
        and p.stem.startswith(prefix)
        and "Imputation" not in p.stem
    ]

    return matching_files

def impute_pseudo_features(dataset_to_impute, dataset_name_used_to_train_imputation_model):
    net_to_pseudo_features, data_schema_loaded = get_imputation_model(dataset_name_used_to_train_imputation_model)
    sampled_dataset_names = get_multiple_samples_dataset_name(dataset_to_impute, sample_size=None)
    for dataset_name in sampled_dataset_names:
        impute_pseudo_features_by_dataset(
            dataset_name,
            dataset_name_used_to_train_imputation_model,
            net_to_pseudo_features,
            data_schema_loaded
        )

def main():
    impute_pseudo_features("ic_downstream1", "ic_upstream2")
    impute_pseudo_features("ic_downstream1", "ic_upstream3")
    impute_pseudo_features("ic_downstream1", "ic_upstream4")
    impute_pseudo_features("ic_upstream2", "ic_downstream1")
    impute_pseudo_features("ic_upstream3", "ic_downstream1")
    impute_pseudo_features("ic_upstream4", "ic_downstream1")