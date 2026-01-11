import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
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
        f"../../../config/model/mlp_pf_{dataset_name_used_to_train_imputation_model}.yaml"
    )
    hyp_cfg_path = (
        f"../../../config/hyp/hyp_pf_{dataset_name_used_to_train_imputation_model}.yaml"
    )
    dataset_cfg_path = (
        f"../../../config/dataset/{dataset_name_used_to_train_imputation_model}.yaml"
    )

    # Leitura dos YAMLs
    modelConfig = OmegaConf.load(model_cfg_path)
    modelConfig["model_path"] = f'outputs/from_scratch_default/training-mlp-{dataset_name_used_to_train_imputation_model}/model_best.pth'
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

    # ---- Defensive: empty dataset ----
    if dataset_to_impute.empty:
        logging.warning(
            f"Dataset '{file_path}' is empty. "
            "Generating output file with empty pseudo-features."
        )

        dataset_imputed_with_pseudo_features = dataset_to_impute.copy()

        for feature in data_schema_loaded["y"]["labels"]:
            if feature not in dataset_imputed_with_pseudo_features.columns:
                dataset_imputed_with_pseudo_features[feature] = pd.Series(
                    dtype="float32"
                )

        save_imputed_dataset(
            file_path,
            dataset_imputed_with_pseudo_features,
            dataset_used_to_train_model,
        )
        return

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

def sufix_for_imputed_dataset(dataset_used_to_train_model):
    if dataset_used_to_train_model is None:
        return ""
    return f"_exp_100{get_number_from_dataset_name(dataset_used_to_train_model)}"

def save_imputed_dataset(path, dataset, dataset_used_to_train_model, method="pseudo_features"):
    folder_of_file = os.path.dirname(path)
    folder_new_name = f"{folder_of_file}_Imputation_{method}{sufix_for_imputed_dataset(dataset_used_to_train_model)}"
    path = path.replace(folder_of_file, folder_new_name)
    if not os.path.exists(folder_new_name):
        os.makedirs(folder_new_name)
    dataset = dataset.reindex(sorted(dataset.columns), axis=1)
    dataset.to_csv(path, index=False)

def get_dataset_folder(dataset_name):
    return f"../../../data/{dataset_name}/"

def get_x_files_in_dataset(dataset_name):
    x_files = []
    folder_path = get_dataset_folder(dataset_name)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith("_x.csv"):
            x_files.append(os.path.join(folder_path, filename))
    return x_files

def get_y_files_in_dataset(dataset_name):
    y_files = []
    folder_path = get_dataset_folder(dataset_name)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith("_y.csv"):
            y_files.append(os.path.join(folder_path, filename))
    return y_files

def clean_y_files(features_to_maintain, dataset_name, dataset_used_to_train_model, method="pseudo_features"):
    y_files = get_y_files_in_dataset(dataset_name)
    for y_file in y_files:
        dataset_y = pd.read_csv(y_file)
        dataset_y_cleaned = dataset_y[features_to_maintain]
        save_imputed_dataset(y_file, dataset_y_cleaned, dataset_used_to_train_model, method=method)

def impute_pseudo_features_by_dataset(dataset_to_impute, dataset_name_used_to_train_imputation_model, net_to_pseudo_features, data_schema_loaded):
    x_files = get_x_files_in_dataset(dataset_to_impute)
    for x_file in x_files:
        impute_pseudo_features_by_file(x_file, net_to_pseudo_features, data_schema_loaded, dataset_name_used_to_train_imputation_model)
    clean_y_files("pIC50", dataset_to_impute, dataset_name_used_to_train_imputation_model)

    
    # Salvar dataset

def get_multiple_samples_dataset_name(base_dataset_name):
    path = "../../../data"
    prefix = f"{base_dataset_name}_Sample"
    directories = os.listdir(path)
    matching_files = [
        name
        for name in directories
        if name.startswith(prefix)
        and "Imputation" not in name
    ]
    if matching_files == []:
        matching_files = [
            name
            for name in directories
            if name.startswith(base_dataset_name)
            and "Imputation" not in name
        ]
    return matching_files

def impute_pseudo_features(dataset_to_impute, dataset_name_used_to_train_imputation_model):
    net_to_pseudo_features, data_schema_loaded = get_imputation_model(dataset_name_used_to_train_imputation_model)
    sampled_dataset_names = get_multiple_samples_dataset_name(dataset_to_impute)
    for dataset_name in sampled_dataset_names:
        impute_pseudo_features_by_dataset(
            dataset_name,
            dataset_name_used_to_train_imputation_model,
            net_to_pseudo_features,
            data_schema_loaded
        )

def impute_real_values_by_file(file_path):
    x_features =["SpMax_A","VE1_A","AATS8dv","AATS8s","AATS2i","ATSC1dv","ATSC8d","ATSC0p","ATSC0i","MATS1c","MATS2s","MATS3s","MATS6s","MATS7s","MATS8s","GATS4c","GATS1dv","GATS5dv","GATS7dv","GATS6d","GATS7d","GATS2s","GATS3s","GATS2v","GATS3v","GATS1p","GATS6p","GATS3i","GATS6i","GATS8i","BCUTc-1h","BCUTd-1l","BCUTs-1h","RPCG","Xch-5d","Xch-7d","Xc-5d","Xc-5dv","Xc-6dv","AXp-1d","SdssC","SaasC","SaaaC","SssssC","SsNH2","SssNH","SsOH","SssO","SdS","SddssS","MAXaaCH","AETA_beta_s","AETA_eta_L","AETA_eta_F","ETA_epsilon_5","IC1","IC2","CIC2","ZMIC1","PEOE_VSA1","PEOE_VSA2","PEOE_VSA9","SlogP_VSA1","SlogP_VSA2","SlogP_VSA10","EState_VSA1","EState_VSA2","EState_VSA3","EState_VSA6","EState_VSA9","VSA_EState3","VSA_EState7","VSA_EState8","MDEC-33","TopoPSA(NO)","GGI3","GGI5","GGI6","GGI7","GGI8","GGI9","JGI2","JGI5","FPSA3","RPCS","Mor02m","Mor03m","Mor06m","Mor08m","Mor11m","Mor13m","Mor16m","Mor23m","XLOGP3","Silicos-IT_Log_P","ESOL_Log_S","ESOL_Solubility_(mg/ml)","Ali_Log_S","Ali_Solubility_(mg/ml)","Silicos-IT_Solubility_(mg/ml)","AATS1i","AATS3i","AATS4i","AATS5d","AATS6i","AATS7s","AATSC0v","AETA_alpha","AETA_beta_ns_d","AMID_C","ATSC1d","ATSC7d","ATSC8dv","AXp-0d","CIC1","EState_VSA4","EState_VSA5","ETA_dAlpha_B","ETA_dEpsilon_D","GATS1d","GATS1i","GATS1v","GATS2c","GATS2d","GATS2i","GATS3c","GATS3dv","GATS4dv","GATS4i","GATS4s","GATS5d","GATS5i","GATS5p","GATS6dv","GATS6s","GATS7i","GATS7p","GATS8c","GATS8dv","JGI4","MLOGP","MOMI-Z","Mor26m","Mor30m","Mor31m","PEOE_VSA4","PEOE_VSA6","RNCG","RNCS","SMR_VSA1","SMR_VSA3","SMR_VSA4","SMR_VSA9","SaaS","SdO","SlogP_VSA3","SlogP_VSA4","SpDiam_A","SssS","VSA_EState9","Xc-3dv","ZMIC2"]

    path_to_original_dataset = "../../../data/alk-5/cd_moleculas_544_833.csv"

    original_dataset = pd.read_csv(path_to_original_dataset)
    dataset_to_impute = pd.read_csv(file_path)

    # colunas que identificam a molécula
    id_features = [
        c for c in dataset_to_impute.columns
        if c not in x_features
    ]

    for index, row in dataset_to_impute.iterrows():
        condition = pd.Series(True, index=original_dataset.index)

        for feature in id_features:
            condition &= np.isclose(original_dataset[feature], row[feature])

        matching_rows = original_dataset[condition]

        if not matching_rows.empty:
            selected_row = matching_rows.sample(n=1, random_state=42).iloc[0]

            for feature in x_features:
                if feature not in dataset_to_impute.columns or pd.isna(row.get(feature)):
                    dataset_to_impute.at[index, feature] = selected_row[feature]
    save_imputed_dataset(
            file_path,
            dataset_to_impute,
            None,
            "RealValues"
        )
    return dataset_to_impute

def impute_real_values_by_dataset(dataset_to_impute):
    x_files = get_x_files_in_dataset(dataset_to_impute)
    for x_file in x_files:
        impute_real_values_by_file(x_file)
    clean_y_files("pIC50", dataset_to_impute, None, method="RealValues")

def impute_real_values(dataset):
    sampled_dataset_names = get_multiple_samples_dataset_name(dataset)
    for dataset_name in sampled_dataset_names:
        impute_real_values_by_dataset(dataset_name)

@hydra.main()
def main(_: DictConfig):
    impute_pseudo_features("ic_downstream1", "ic_upstream2")
    impute_real_values("ic_downstream1")
    impute_pseudo_features("ic_downstream1", "ic_upstream3")
    impute_pseudo_features("ic_downstream1", "ic_upstream4")
    impute_pseudo_features("ic_upstream2", "ic_downstream1")
    impute_real_values("ic_upstream2")
    impute_pseudo_features("ic_upstream3", "ic_downstream1")
    impute_real_values("ic_upstream3")
    impute_pseudo_features("ic_upstream4", "ic_downstream1")
    impute_real_values("ic_upstream4")

if __name__ == "__main__":
    main()
