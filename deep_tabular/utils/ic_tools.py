import os
import random
import pickle
 
import numpy as np
import pandas as pd
import re
import torch
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_regression
from typing import List
 
 
# ---------------------------------------------------------------------------
# COLUMN DEFINITIONS / DEFINIÇÕES DE COLUNAS
# ---------------------------------------------------------------------------
 
downstream_columns = [
    "Molecule","SMILES","Formula_x","SpDiam_A","AATS5d","AATS7s","AATS8s",
    "AATS1i","AATS2i","AATS3i","AATS4i","AATS6i","ATSC1dv","ATSC8dv","ATSC1d",
    "ATSC7d","AATSC0v","MATS6s","MATS7s","MATS8s","GATS2c","GATS3c","GATS8c",
    "GATS1dv","GATS3dv","GATS4dv","GATS6dv","GATS7dv","GATS8dv","GATS1d",
    "GATS2d","GATS5d","GATS2s","GATS3s","GATS4s","GATS6s","GATS1v","GATS2v",
    "GATS5p","GATS6p","GATS7p","GATS1i","GATS2i","GATS3i","GATS4i","GATS5i",
    "GATS7i","GATS8i","RNCG","RPCG","Xc-3dv","Xc-5dv","Xc-6dv","AXp-0d",
    "SdssC","SsNH2","SdO","SssO","SssS","SaaS","SddssS","MAXaaCH","AETA_alpha",
    "AETA_beta_ns_d","ETA_dAlpha_B","ETA_epsilon_5","ETA_dEpsilon_D","IC1","CIC1",
    "CIC2","ZMIC2","PEOE_VSA1","PEOE_VSA4","PEOE_VSA6","PEOE_VSA9","SMR_VSA1",
    "SMR_VSA3","SMR_VSA4","SMR_VSA9","SlogP_VSA2","SlogP_VSA3","SlogP_VSA4",
    "SlogP_VSA10","EState_VSA4","EState_VSA5","VSA_EState8","VSA_EState9",
    "AMID_C","TopoPSA(NO)","GGI6","GGI7","JGI4","RNCS","Mor02m","Mor03m",
    "Mor13m","Mor26m","Mor30m","Mor31m","MOMI-Z","MLOGP",
    "ESOL_Solubility_(mg/ml)","Ali_Log_S","pIC50",
]
 
upstream2_columns = [
    "Molecule","SMILES","Formula_x","SpMax_A","VE1_A","AATS8dv","AATS8s",
    "AATS2i","ATSC1dv","ATSC8d","ATSC0p","ATSC0i","MATS1c","MATS2s","MATS3s",
    "MATS6s","MATS7s","MATS8s","GATS4c","GATS1dv","GATS5dv","GATS7dv","GATS6d",
    "GATS7d","GATS2s","GATS3s","GATS2v","GATS3v","GATS1p","GATS6p","GATS3i",
    "GATS6i","GATS8i","BCUTc-1h","BCUTd-1l","BCUTs-1h","RPCG","Xch-5d","Xch-7d",
    "Xc-5d","Xc-5dv","Xc-6dv","AXp-1d","SdssC","SaasC","SaaaC","SssssC","SsNH2",
    "SssNH","SsOH","SssO","SdS","SddssS","MAXaaCH","AETA_beta_s","AETA_eta_L",
    "AETA_eta_F","ETA_epsilon_5","IC1","IC2","CIC2","ZMIC1","PEOE_VSA1",
    "PEOE_VSA2","PEOE_VSA9","SlogP_VSA1","SlogP_VSA2","SlogP_VSA10","EState_VSA1",
    "EState_VSA2","EState_VSA3","EState_VSA6","EState_VSA9","VSA_EState3",
    "VSA_EState7","VSA_EState8","MDEC-33","TopoPSA(NO)","GGI3","GGI5","GGI6",
    "GGI7","GGI8","GGI9","JGI2","JGI5","FPSA3","RPCS","Mor02m","Mor03m","Mor06m",
    "Mor08m","Mor11m","Mor13m","Mor16m","Mor23m","XLOGP3","Silicos-IT_Log_P",
    "ESOL_Log_S","ESOL_Solubility_(mg/ml)","Ali_Log_S","Ali_Solubility_(mg/ml)",
    "Silicos-IT_Solubility_(mg/ml)","pIC50",
]
 
upstream3_columns = [
    "Molecule","SMILES","Formula_x","SpDiam_A","AATS3d","AATS2s","AATS3v",
    "AATS1i","AATS2i","AATS3i","AATS4i","ATSC8dv","ATSC1d","ATSC8d","ATSC0i",
    "AATSC0c","MATS1c","MATS2s","MATS7s","GATS1c","GATS2c","GATS5c","GATS7c",
    "GATS8c","GATS2dv","GATS5dv","GATS7dv","GATS8dv","GATS1d","GATS3d","GATS5d",
    "GATS6d","GATS7d","GATS8d","GATS2s","GATS6s","GATS6v","GATS3p","GATS5p",
    "GATS1i","GATS6i","BCUTs-1h","BCUTs-1l","BCUTi-1h","BCUTi-1l","RPCG",
    "Xch-6d","Xc-5d","Xc-3dv","Xc-4dv","AXp-1d","SsCH3","SdCH2","SdsCH",
    "SsssCH","SaaNH","SdsN","SsOH","SssO","SdS","SaaS","SddssS","MINaasC",
    "AETA_alpha","AETA_beta_ns_d","ETA_epsilon_5","ETA_dEpsilon_C","IC2","ZMIC2",
    "PEOE_VSA1","PEOE_VSA2","PEOE_VSA3","PEOE_VSA6","PEOE_VSA7","PEOE_VSA11",
    "SMR_VSA5","SMR_VSA6","SlogP_VSA2","SlogP_VSA3","EState_VSA1","EState_VSA2",
    "EState_VSA3","EState_VSA6","VSA_EState7","AMID_C","TopoPSA(NO)","GGI5",
    "GGI7","GGI8","GGI9","JGI3","TSRW10","TASA","Mor02m","Mor10m","Mor11m",
    "Mor12m","Mor16m","Mor21m","Mor22m","Mor23m","iLOGP","Silicos-IT_Log_P",
    "pIC50",
]
 
upstream4_columns = [
    "Molecule","SMILES","Formula_x","SpMax_A","VE1_A","AATS7d","AATS6s","AATS8s",
    "AATS3p","AATS1i","AATS3i","AATS4i","AATS6i","ATSC1d","AATSC0c","MATS1c",
    "MATS5s","MATS6s","MATS7s","GATS1c","GATS6c","GATS7c","GATS1dv","GATS3dv",
    "GATS5dv","GATS6dv","GATS3d","GATS4d","GATS6d","GATS1s","GATS2s","GATS8s",
    "GATS2v","GATS3v","GATS7v","GATS4p","GATS8i","BCUTc-1l","BCUTd-1l","BCUTs-1h",
    "BCUTs-1l","RNCG","RPCG","Xc-5dv","Xc-6dv","SsCH3","SdCH2","SdsCH","SsssCH",
    "SaaaC","SssssC","SssNH","SsssN","SsOH","SssO","SdS","SsCl","MAXaasC",
    "MINaaCH","ETA_shape_y","AETA_beta_s","AETA_eta_L","ETA_dEpsilon_D","fMF",
    "ZMIC2","PEOE_VSA3","PEOE_VSA4","PEOE_VSA6","PEOE_VSA7","PEOE_VSA8",
    "PEOE_VSA10","SMR_VSA1","SMR_VSA9","SlogP_VSA1","SlogP_VSA2","SlogP_VSA4",
    "SlogP_VSA5","SlogP_VSA10","EState_VSA1","VSA_EState1","VSA_EState8",
    "VSA_EState9","MDEC-22","MDEC-23","TopoPSA(NO)","GGI3","GGI4","GGI6","GGI8",
    "GGI9","GGI10","JGI2","FNSA1","RASA","Mor02m","Mor08m","Mor10m","Mor12m",
    "Mor23m","Mor24m","MOMI-Z","Silicos-IT_Log_P","ESOL_Solubility_(mg/ml)",
    "pIC50",
]
 
 
# ---------------------------------------------------------------------------
# MISSING FEATURES PER UPSTREAM DATASET
# Features presentes em downstream mas ausentes em cada dataset upstream,
# e vice-versa.
# ---------------------------------------------------------------------------
 
missing_features = {
    # Columns exclusive to upstream2 (not in downstream)
    # Colunas exclusivas do upstream2 (não presentes no downstream)
    "up2_only": [
        "AATS8dv","AETA_beta_s","AETA_eta_F","AETA_eta_L","ATSC0i","ATSC0p",
        "ATSC8d","AXp-1d","Ali_Solubility_(mg/ml)","BCUTc-1h","BCUTd-1l",
        "BCUTs-1h","ESOL_Log_S","EState_VSA1","EState_VSA2","EState_VSA3",
        "EState_VSA6","EState_VSA9","FPSA3","GATS1p","GATS3v","GATS4c",
        "GATS5dv","GATS6d","GATS6i","GATS7d","GGI3","GGI5","GGI8","GGI9",
        "IC2","JGI2","JGI5","MATS1c","MATS2s","MATS3s","MDEC-33","Mor06m",
        "Mor08m","Mor11m","Mor16m","Mor23m","PEOE_VSA2","RPCS","SaaaC","SaasC",
        "SdS","Silicos-IT_Log_P","Silicos-IT_Solubility_(mg/ml)","SlogP_VSA1",
        "SpMax_A","SsOH","SssNH","SssssC","VE1_A","VSA_EState3","VSA_EState7",
        "XLOGP3","Xc-5d","Xch-5d","Xch-7d","ZMIC1",
    ],
    # Columns in downstream but missing from upstream2
    # Colunas do downstream ausentes no upstream2
    "up2_missing": [
        "AATS1i","AATS3i","AATS4i","AATS5d","AATS6i","AATS7s","AATSC0v",
        "AETA_alpha","AETA_beta_ns_d","AMID_C","ATSC1d","ATSC7d","ATSC8dv",
        "AXp-0d","CIC1","EState_VSA4","EState_VSA5","ETA_dAlpha_B","ETA_dEpsilon_D",
        "GATS1d","GATS1i","GATS1v","GATS2c","GATS2d","GATS2i","GATS3c","GATS3dv",
        "GATS4dv","GATS4i","GATS4s","GATS5d","GATS5i","GATS5p","GATS6dv","GATS6s",
        "GATS7i","GATS7p","GATS8c","GATS8dv","JGI4","MLOGP","MOMI-Z","Mor26m",
        "Mor30m","Mor31m","PEOE_VSA4","PEOE_VSA6","RNCG","RNCS","SMR_VSA1",
        "SMR_VSA3","SMR_VSA4","SMR_VSA9","SaaS","SdO","SlogP_VSA3","SlogP_VSA4",
        "SpDiam_A","SssS","VSA_EState9","Xc-3dv","ZMIC2",
    ],
    # Columns shared between downstream and upstream2
    # Colunas em comum entre downstream e upstream2
    "up2_common": [
        "AATS2i","AATS8s","ATSC1dv","Ali_Log_S","CIC2","ESOL_Solubility_(mg/ml)",
        "ETA_epsilon_5","Formula_x","GATS1dv","GATS2s","GATS2v","GATS3i","GATS3s",
        "GATS6p","GATS7dv","GATS8i","GGI6","GGI7","IC1","MATS6s","MATS7s","MATS8s",
        "MAXaaCH","Molecule","Mor02m","Mor03m","Mor13m","PEOE_VSA1","PEOE_VSA9",
        "RPCG","SMILES","SddssS","SdssC","SlogP_VSA10","SlogP_VSA2","SsNH2",
        "SssO","TopoPSA(NO)","VSA_EState8","Xc-5dv","Xc-6dv","pIC50",
    ],
 
    # Columns exclusive to upstream3 (not in downstream)
    # Colunas exclusivas do upstream3 (não presentes no downstream)
    "up3_only": [
        "AATS2s","AATS3d","AATS3v","AATSC0c","ATSC0i","ATSC8d","AXp-1d",
        "BCUTi-1h","BCUTi-1l","BCUTs-1h","BCUTs-1l","EState_VSA1","EState_VSA2",
        "EState_VSA3","EState_VSA6","ETA_dEpsilon_C","GATS1c","GATS2dv","GATS3d",
        "GATS3p","GATS5c","GATS5dv","GATS6d","GATS6i","GATS6v","GATS7c","GATS7d",
        "GATS8d","GGI5","GGI8","GGI9","IC2","JGI3","MATS1c","MATS2s","MINaasC",
        "Mor10m","Mor11m","Mor12m","Mor16m","Mor21m","Mor22m","Mor23m","PEOE_VSA11",
        "PEOE_VSA2","PEOE_VSA3","PEOE_VSA7","SMR_VSA5","SMR_VSA6","SaaNH","SdCH2",
        "SdS","SdsCH","SdsN","Silicos-IT_Log_P","SsCH3","SsOH","SsssCH","TASA",
        "TSRW10","VSA_EState7","Xc-4dv","Xc-5d","Xch-6d","iLOGP",
    ],
    # Columns in downstream but missing from upstream3
    # Colunas do downstream ausentes no upstream3
    "up3_missing": [
        "AATS5d","AATS6i","AATS7s","AATS8s","AATSC0v","ATSC1dv","ATSC7d",
        "AXp-0d","Ali_Log_S","CIC1","CIC2","ESOL_Solubility_(mg/ml)","EState_VSA4",
        "EState_VSA5","ETA_dAlpha_B","ETA_dEpsilon_D","GATS1dv","GATS1v","GATS2d",
        "GATS2i","GATS2v","GATS3c","GATS3dv","GATS3i","GATS3s","GATS4dv","GATS4i",
        "GATS4s","GATS5i","GATS6dv","GATS6p","GATS7i","GATS7p","GATS8i","GGI6",
        "IC1","JGI4","MATS6s","MATS8s","MAXaaCH","MLOGP","MOMI-Z","Mor03m",
        "Mor13m","Mor26m","Mor30m","Mor31m","PEOE_VSA4","PEOE_VSA9","RNCG","RNCS",
        "SMR_VSA1","SMR_VSA3","SMR_VSA4","SMR_VSA9","SdO","SdssC","SlogP_VSA10",
        "SlogP_VSA4","SsNH2","SssS","VSA_EState8","VSA_EState9","Xc-5dv","Xc-6dv",
    ],
    # Columns shared between downstream and upstream3
    # Colunas em comum entre downstream e upstream3
    "up3_common": [
        "AATS1i","AATS2i","AATS3i","AATS4i","AETA_alpha","AETA_beta_ns_d",
        "AMID_C","ATSC1d","ATSC8dv","ETA_epsilon_5","Formula_x","GATS1d","GATS1i",
        "GATS2c","GATS2s","GATS5d","GATS5p","GATS6s","GATS7dv","GATS8c","GATS8dv",
        "GGI7","MATS7s","Molecule","Mor02m","PEOE_VSA1","PEOE_VSA6","RPCG","SMILES",
        "SaaS","SddssS","SlogP_VSA2","SlogP_VSA3","SpDiam_A","SssO","TopoPSA(NO)",
        "Xc-3dv","ZMIC2","pIC50",
    ],
 
    # Columns exclusive to upstream4 (not in downstream)
    # Colunas exclusivas do upstream4 (não presentes no downstream)
    "up4_only": [
        "AATS3p","AATS6s","AATS7d","AATSC0c","AETA_beta_s","AETA_eta_L",
        "BCUTc-1l","BCUTd-1l","BCUTs-1h","BCUTs-1l","EState_VSA1","ETA_shape_y",
        "FNSA1","GATS1c","GATS1s","GATS3d","GATS3v","GATS4d","GATS4p","GATS5dv",
        "GATS6c","GATS6d","GATS7c","GATS7v","GATS8s","GGI10","GGI3","GGI4",
        "GGI8","GGI9","JGI2","MATS1c","MATS5s","MAXaasC","MDEC-22","MDEC-23",
        "MINaaCH","Mor08m","Mor10m","Mor12m","Mor23m","Mor24m","PEOE_VSA10",
        "PEOE_VSA3","PEOE_VSA7","PEOE_VSA8","RASA","SaaaC","SdCH2","SdS","SdsCH",
        "Silicos-IT_Log_P","SlogP_VSA1","SlogP_VSA5","SpMax_A","SsCH3","SsCl",
        "SsOH","SssNH","SsssCH","SsssN","SssssC","VE1_A","VSA_EState1","fMF",
    ],
    # Columns in downstream but missing from upstream4
    # Colunas do downstream ausentes no upstream4
    "up4_missing": [
        "AATS2i","AATS5d","AATS7s","AATSC0v","AETA_alpha","AETA_beta_ns_d",
        "AMID_C","ATSC1dv","ATSC7d","ATSC8dv","AXp-0d","Ali_Log_S","CIC1","CIC2",
        "EState_VSA4","EState_VSA5","ETA_dAlpha_B","ETA_epsilon_5","GATS1d",
        "GATS1i","GATS1v","GATS2c","GATS2d","GATS2i","GATS3c","GATS3i","GATS3s",
        "GATS4dv","GATS4i","GATS4s","GATS5d","GATS5i","GATS5p","GATS6p","GATS6s",
        "GATS7dv","GATS7i","GATS7p","GATS8c","GATS8dv","GGI7","IC1","JGI4",
        "MATS8s","MAXaaCH","MLOGP","Mor03m","Mor13m","Mor26m","Mor30m","Mor31m",
        "PEOE_VSA1","PEOE_VSA9","RNCS","SMR_VSA3","SMR_VSA4","SaaS","SdO",
        "SddssS","SdssC","SlogP_VSA3","SpDiam_A","SsNH2","SssS","Xc-3dv",
    ],
    # Columns shared between downstream and upstream4
    # Colunas em comum entre downstream e upstream4
    "up4_common": [
        "AATS1i","AATS3i","AATS4i","AATS6i","AATS8s","ATSC1d",
        "ESOL_Solubility_(mg/ml)","ETA_dEpsilon_D","Formula_x","GATS1dv","GATS2s",
        "GATS2v","GATS3dv","GATS6dv","GATS8i","GGI6","MATS6s","MATS7s","MOMI-Z",
        "Molecule","Mor02m","PEOE_VSA4","PEOE_VSA6","RNCG","RPCG","SMILES",
        "SMR_VSA1","SMR_VSA9","SlogP_VSA10","SlogP_VSA2","SlogP_VSA4","SssO",
        "TopoPSA(NO)","VSA_EState8","VSA_EState9","Xc-5dv","Xc-6dv","ZMIC2",
        "pIC50",
    ],
}
# ---------------------------------------------------------------------------
# CONSTANTS / CONSTANTES
# ---------------------------------------------------------------------------
 
# Columns that cannot be used as numerical features (identifiers/strings)
# Colunas que não podem ser usadas como features numéricas (identificadores/strings)
non_numerical_columns = ["Molecule", "SMILES", "Formula_x"]
 
# Default regression target column / Coluna alvo padrão para regressão
default_target_columns = ["pIC50"]

# ---------------------------------------------------------------------------
# HELPER / COLUMN UTILITIES
# ---------------------------------------------------------------------------
 
def remove_common_strings(downstream_columns, non_numerical_columns, target_columns, target=0):
    """
    Returns elements from `downstream_columns` that are not in
    `non_numerical_columns` or `target_columns`.
 
    Retorna elementos de `downstream_columns` que não estão em
    `non_numerical_columns` ou `target_columns`.
 
    NOTE / NOTA: This function is not called anywhere in this file.
    Verify whether it is used externally before removing it.
    Verifique se é usada externamente antes de removê-la.
    """
    set_downstream   = set(downstream_columns)
    set_non_numerical = set(non_numerical_columns)
    set_target       = {target_columns}  # single string → set with one element
 
    return list(set_downstream - set_non_numerical - set_target)
 
 
def combine_unique_sorted(arr1, arr2, arr3):
    """
    Concatenates three lists, removes duplicates, and returns them sorted.
    Concatena três listas, remove duplicatas e retorna ordenada.
    """
    return sorted(set(arr1 + arr2 + arr3))
 
 
# ---------------------------------------------------------------------------
# TARGET COLUMN RESOLUTION / RESOLUÇÃO DE COLUNAS ALVO
# ---------------------------------------------------------------------------
 
def get_downstram_target_columns_for_pseudo_features(dataset_name):
    """
    Returns the list of missing feature columns for a downstream dataset,
    based on which upstream variant it corresponds to.
    When no specific upstream variant is identified, returns the union of
    all missing feature columns (used for a single model covering all upstreams).
 
    Retorna a lista de colunas de features ausentes para um dataset downstream,
    com base em qual variante upstream ele corresponde.
    Quando nenhuma variante específica é identificada, retorna a união de todas
    as colunas ausentes (usado para um modelo único que cobre todos os upstreams).
    """
    if "_upstream2" in dataset_name:
        return missing_features["up2_missing"]
    elif "_upstream3" in dataset_name:
        return missing_features["up3_missing"]
    elif "_upstream4" in dataset_name:
        return missing_features["up4_missing"]
 
    # No specific upstream variant → combine all missing columns
    # Nenhuma variante upstream específica → combina todas as colunas ausentes
    return combine_unique_sorted(
        missing_features["up2_missing"],
        missing_features["up3_missing"],
        missing_features["up4_missing"],
    )

def get_target_columns_for_multivariant_task(dataset_name):
    """
    Returns the target columns for the multivariant regression task
    (predicting all missing features for a given upstream dataset).
    The default target (pIC50) is intentionally excluded to avoid data leakage.
 
    Retorna as colunas alvo para a tarefa de regressão multivariante
    (predição de todas as features ausentes de um dataset upstream).
    O alvo padrão (pIC50) é intencionalmente excluído para evitar vazamento de dados.
    """
    if "ic_upstream2" in dataset_name:
        return missing_features['up2_only']
    elif 'ic_upstream3' in dataset_name:
        return missing_features['up3_only']
    elif 'ic_upstream4' in dataset_name:
        return missing_features['up4_only']
    elif 'ic_downstream1' in dataset_name:
        return get_downstram_target_columns_for_pseudo_features(dataset_name)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
def get_target_columns_to_save(dataset_name):
    """
    Function to get the target columns for a given dataset
    """
    ## we always save the default target columns, because they are used when the task is regression
    ## and the others are used when the task is pseudo-feature prediction, so we need to save them all (and remove the default target collumns to avoid leaking data)
    return default_target_columns + get_target_columns_for_multivariant_task(dataset_name)
def get_target_columns_to_save(dataset_name):
    """
    Returns all target columns that should be saved for a given dataset:
    the default regression target (pIC50) plus all multivariant targets.
 
    Retorna todas as colunas alvo que devem ser salvas para um dado dataset:
    o alvo de regressão padrão (pIC50) mais todos os alvos multivariantes.
    """
    return default_target_columns + get_target_columns_for_multivariant_task(dataset_name)
 
 
def define_target_for_task(task, dataset_name):
    """
    Returns the appropriate target column(s) based on the task type.
    For multivariant regression, returns multiple target columns.
    For all other tasks, returns the default target column.
 
    Retorna as colunas alvo apropriadas com base no tipo de tarefa.
    Para regressão multivariante, retorna múltiplas colunas alvo.
    Para todas as outras tarefas, retorna a coluna alvo padrão.
    """
    if task.startswith("multiVariantRegression"):
        return get_target_columns_for_multivariant_task(dataset_name)
    return default_target_columns
 
 
# ---------------------------------------------------------------------------
# DATASET SPLITTING / DIVISÃO DE DATASETS
# ---------------------------------------------------------------------------
 
def split_dataset(base_path, dataset_file, target_columns,
                  delimiter=";", header="infer", drop_columns=None, prefix="dataset"):
    """
    Reads a CSV, splits it into train/val/test sets and saves each split to disk.
 
    Special case: files starting with 'japan' do not receive a validation split
    (an empty DataFrame is used instead).
 
    Lê um CSV, divide em treino/val/teste e salva cada split no disco.
 
    Caso especial: arquivos que começam com 'japan' não recebem split de validação
    (um DataFrame vazio é usado no lugar).
    """
    dataset = pd.read_csv(os.path.join(base_path, dataset_file), delimiter=delimiter, header=header)
 
    if drop_columns:
        dataset = dataset.drop(columns=drop_columns)
 
    dataset = dataset.astype(float)
 
    y_full = dataset[target_columns].copy()
    X_full = dataset.drop(columns=target_columns, errors="ignore")
 
    # 80/20 train-test split / Divisão treino-teste 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=1
    )
 
    if dataset_file.startswith("japan"):
        # Japan dataset has no validation split / Dataset japan não tem validação
        X_val = pd.DataFrame(columns=X_train.columns)
        if isinstance(y_train, pd.DataFrame):
            y_val = pd.DataFrame(columns=y_train.columns)
    else:
        # 15% of total ≈ 18.75% of the remaining training data
        # 15% do total ≈ 18,75% dos dados de treino restantes
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1875, random_state=1
        )
 
    splits = {
        "train_X": X_train, "val_X": X_val, "test_X": X_test,
        "train_y": y_train, "val_y": y_val, "test_y": y_test,
    }
    for split_name, df in splits.items():
        df.to_csv(
            os.path.join(base_path, f"{prefix}_{split_name}.csv"),
            index=False,
            header=(header is not None),
        )
 
    return splits
 
 
def split_ic_dataset(dataset_name, dataset_number, target_columns):
    """
    Splits an IC dataset CSV and saves the resulting splits.
    IC files use '|' as delimiter and include a header row.
 
    Divide um CSV de dataset IC e salva os splits resultantes.
    Arquivos IC usam '|' como delimitador e possuem linha de cabeçalho.
    """
    base_path = f"../../../data/{dataset_name}/"
    dataset_file = f"exp_100_{dataset_number}.csv"
    return split_dataset(
        base_path=base_path,
        dataset_file=dataset_file,
        target_columns=target_columns,
        delimiter="|",
        header=0,
        drop_columns=non_numerical_columns,
        prefix="ic",
    )
 
 
def split_cep_dataset(dataset_name):
    """
    Splits a CEP dataset CSV and saves the resulting splits.
    CEP files use ';' as delimiter, have no header and the first column is the target.
 
    Divide um CSV de dataset CEP e salva os splits resultantes.
    Arquivos CEP usam ';' como delimitador, não têm cabeçalho e a primeira coluna é o alvo.
    """
    base_path = f"../../../data/{dataset_name}/"
    dataset_file = f"{dataset_name}.csv"
    return split_dataset(
        base_path=base_path,
        dataset_file=dataset_file,
        target_columns=[0],  # first column is the target / primeira coluna é o alvo
        delimiter=";",
        header=None,
        drop_columns=None,
        prefix="cep",
    )
# ---------------------------------------------------------------------------
# DATASET READING / LEITURA DE DATASETS
# ---------------------------------------------------------------------------
 
def _order_columns_alphabetically(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the DataFrame with columns sorted alphabetically.
    Retorna o DataFrame com as colunas ordenadas alfabeticamente.
    """
    return df.reindex(sorted(df.columns), axis=1)
 
 
def _remove_target_from_features(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """
    Drops any target columns from X that are also present in y.
    Needed for the pseudo-feature task where target columns may appear in X.
 
    Remove de X quaisquer colunas alvo que também estejam presentes em y.
    Necessário para a tarefa de pseudo-features onde colunas alvo podem aparecer em X.
    """
    overlap = [col for col in y.columns if col in X.columns]
    if overlap:
        X = X.drop(columns=overlap)
    return X
 
 
def _remove_target_from_all_splits(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Applies _remove_target_from_features to all three data splits.
    Aplica _remove_target_from_features a todos os três splits.
    """
    X_train = _remove_target_from_features(X_train, y_train)
    X_val   = _remove_target_from_features(X_val,   y_val)
    X_test  = _remove_target_from_features(X_test,  y_test)
    return X_train, X_val, X_test
 
 
def _safe_read_csv(path, **kwargs) -> pd.DataFrame:
    """
    Reads a CSV file, returning an empty DataFrame if the file is empty.
    Lê um arquivo CSV, retornando um DataFrame vazio se o arquivo estiver vazio.
    """
    try:
        df = pd.read_csv(path, **kwargs)
        return df if not df.empty else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
 
 
def read_ic_dataset(dataset_name, target_columns):
    """
    Reads pre-split IC dataset files and returns all six splits.
    Target columns from X are removed to avoid data leakage
    (in the normal case pIC50 is already absent, but this is required
    for the pseudo-feature scenario).
 
    Lê os arquivos pré-divididos do dataset IC e retorna os seis splits.
    Colunas alvo são removidas de X para evitar vazamento de dados
    (no caso normal a pIC50 já está ausente, mas isso é necessário
    para o cenário de pseudo-features).
    """
    base_path = f"../../../data/{dataset_name}/"
    file_paths = {
        "X_train": os.path.join(base_path, "ic_train_X.csv"),
        "X_val":   os.path.join(base_path, "ic_val_X.csv"),
        "X_test":  os.path.join(base_path, "ic_test_X.csv"),
        "y_train": os.path.join(base_path, "ic_train_y.csv"),
        "y_val":   os.path.join(base_path, "ic_val_y.csv"),
        "y_test":  os.path.join(base_path, "ic_test_y.csv"),
    }
 
    for key, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required dataset file not found: {path}")
 
    X_train, X_val, X_test = [
        _order_columns_alphabetically(pd.read_csv(file_paths[k]))
        for k in ["X_train", "X_val", "X_test"]
    ]
 
    y_train_full, y_val_full, y_test_full = [
        pd.read_csv(file_paths[k]) for k in ["y_train", "y_val", "y_test"]
    ]
 
    # Select only the requested target columns
    # Seleciona apenas as colunas alvo solicitadas
    y_train = y_train_full[target_columns]
    y_val   = y_val_full[target_columns]
    y_test  = y_test_full[target_columns]
 
    X_train, X_val, X_test = _remove_target_from_all_splits(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
 
    return X_train, X_val, X_test, y_train, y_val, y_test
 
def read_cep_dataset(dataset_name):
    """
    Reads pre-split CEP dataset files and returns all six splits.
    Lê os arquivos pré-divididos do dataset CEP e retorna os seis splits.
    """
    base_path = f"../../../data/{dataset_name}/"
    file_paths = {
        "X_train": os.path.join(base_path, "cep_train_X.csv"),
        "X_val":   os.path.join(base_path, "cep_val_X.csv"),
        "X_test":  os.path.join(base_path, "cep_test_X.csv"),
        "y_train": os.path.join(base_path, "cep_train_y.csv"),
        "y_val":   os.path.join(base_path, "cep_val_y.csv"),
        "y_test":  os.path.join(base_path, "cep_test_y.csv"),
    }
 
    X_train, X_val, X_test = [_safe_read_csv(file_paths[k], header=None) for k in ["X_train", "X_val", "X_test"]]
    y_train, y_val, y_test = [_safe_read_csv(file_paths[k], header=None) for k in ["y_train", "y_val", "y_test"]]
 
    return X_train, X_val, X_test, y_train, y_val, y_test
# ---------------------------------------------------------------------------
# DATASET LOADING ORCHESTRATION / ORQUESTRAÇÃO DE CARREGAMENTO
# ---------------------------------------------------------------------------
 
def get_datasets(dataset_name, dataset_number=None, target_columnsToSave=None,
                 task="regression", dataset_type="ic"):
    """
    High-level loader: tries to read existing split files; if they are absent,
    generates the splits first, then reads them.
 
    Carregador de alto nível: tenta ler arquivos de split existentes; se ausentes,
    gera os splits e depois os lê.
 
    Args:
        dataset_name (str): Name/path identifier of the dataset.
                            Nome/identificador do dataset.
        dataset_number (int | None): Used for IC file naming (exp_100_<n>.csv).
                                     Usado para nomear arquivos IC (exp_100_<n>.csv).
        target_columnsToSave (list | None): Target columns to persist when splitting.
                                            Colunas alvo a salvar ao dividir.
        task (str): Task type string (e.g. 'regression', 'multiVariantRegression').
                    Tipo de tarefa (ex: 'regression', 'multiVariantRegression').
        dataset_type (str): 'ic' or 'cep'. / 'ic' ou 'cep'.
    """
    if dataset_type == "ic":
        target_columns = define_target_for_task(task, dataset_name)
        try:
            return read_ic_dataset(dataset_name, target_columns)
        except FileNotFoundError:
            split_ic_dataset(dataset_name, dataset_number, target_columnsToSave)
            return read_ic_dataset(dataset_name, target_columns)
 
    elif dataset_type == "cep":
        try:
            return read_cep_dataset(dataset_name)
        except FileNotFoundError:
            split_cep_dataset(dataset_name)
            return read_cep_dataset(dataset_name)
 
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
 
 
# ---------------------------------------------------------------------------
# DATASET STRUCTURE ASSEMBLY / MONTAGEM DA ESTRUTURA DO DATASET
# ---------------------------------------------------------------------------
 
def get_dataset(X_train, X_val, X_test, y_train, y_val, y_test,
                dataset_name, task, dataset_id=None, n_classes=1):
    """
    Assembles the common output structure used by all dataset getters.
 
    Monta a estrutura de saída comum usada por todos os getters de dataset.
 
    Returns / Retorna:
        numerical_data (dict): numpy arrays for each split.
                               Arrays numpy para cada split.
        categorical_data: None (no categorical features in these datasets).
                          None (sem features categóricas nesses datasets).
        targets (dict): numpy arrays of target values.
                        Arrays numpy dos valores alvo.
        info (dict): metadata about the dataset.
                     Metadados sobre o dataset.
        full_cat_data_for_encoder: None (no categorical features).
                                   None (sem features categóricas).
    """
    info = {
        "name": dataset_id if dataset_id is not None else dataset_name,
        "task_type": task,
        "n_num_features": len(X_train.columns),
        "n_cat_features": 0,
        "train_size": X_train.shape[0],
        "val_size":   X_val.shape[0],
        "test_size":  X_test.shape[0],
        "n_classes":  n_classes,
    }
 
    numerical_data = {
        "train": X_train.values.astype("float"),
        "val":   X_val.values.astype("float"),
        "test":  X_test.values.astype("float"),
    }
 
    targets = {
        "train": y_train.values.astype("float"),
        "val":   y_val.values.astype("float"),
        "test":  y_test.values.astype("float"),
    }
 
    return numerical_data, None, targets, info, None
 
def _get_dataset_number(s: str) -> int:
    """
    Extracts the dataset number from strings like:
    'ic_upstream2' → 2
    'ic_downstream10' → 10
    'ic_upstream2_a3' → 2
    """
    if not s:
        raise ValueError("Input string cannot be empty.")

    match = re.search(r'ic_(?:upstream|downstream)(\d+)', s)
    
    if match:
        return int(match.group(1))
    
    print(f"Could not extract dataset number from '{s}'.")
    return None
 
 
# ---------------------------------------------------------------------------
# PUBLIC DATASET GETTERS (signatures must not change)
# GETTERS PÚBLICOS DE DATASETS (assinaturas não devem ser alteradas)
# ---------------------------------------------------------------------------
 
def get_ic_dataset(dataset_name, task, stage):
    """
    Loads an IC dataset for a given task and stage, building a data_schema
    that preserves feature and label semantics for downstream use.
 
    Carrega um dataset IC para uma dada tarefa e estágio, construindo um data_schema
    que preserva a semântica de features e labels para uso posterior.
    """
    print(f"Loading dataset: {dataset_name} for task: {task} at stage: {stage}")
 
    dataset_id     = _get_dataset_number(dataset_name)
    target_columns = get_target_columns_to_save(dataset_name)
 
    X_train, X_val, X_test, y_train, y_val, y_test = get_datasets(
        dataset_name,
        dataset_number=dataset_id,
        target_columnsToSave=target_columns,
        task=task,
        dataset_type="ic",
    )
    print(f"Target columns: {list(y_train.columns)}")
 
    # Preserve column names as schema metadata before converting to arrays.
    # Preserva nomes de colunas como metadados de schema antes de converter para arrays.
    data_schema = {
        "x": {"features": list(X_train.columns)},
        "y": {"labels": list(y_train.columns), "task": task},
    }
 
    x_numerical, x_categorical, y, info, full_cat_data_for_encoder = get_dataset(
        X_train, X_val, X_test, y_train, y_val, y_test,
        dataset_name, task, dataset_id,
        n_classes=len(set(y_train)),
    )
 
    info["data_schema"] = data_schema
 
    return x_numerical, x_categorical, y, info, full_cat_data_for_encoder
 
 
def get_cep_dataset(dataset_name, task, stage):
    """
    Loads a CEP dataset for a given task and stage.
    Carrega um dataset CEP para uma dada tarefa e estágio.
    """
    print(f"Loading dataset: {dataset_name} for task: {task} at stage: {stage}")
 
    X_train, X_val, X_test, y_train, y_val, y_test = get_datasets(
        dataset_name, task=task, dataset_type="cep"
    )
 
    return get_dataset(X_train, X_val, X_test, y_train, y_val, y_test, dataset_name, task)
 
 
def get_synthetic_dataset(n_samples=1000, n_features=10, noise=10.0,
                          val_size=0.2, test_size=0.2, random_state=42):
    """
    Generates a synthetic regression dataset and splits it into train/val/test.
 
    Gera um dataset de regressão sintético e o divide em treino/val/teste.
    """
    print(f"Generating synthetic dataset with {n_samples} samples, "
          f"{n_features} features, noise={noise}")
 
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
 
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
    )
 
    info = {
        "name": "synthetic",
        "task_type": "regression",
        "n_num_features": n_features,
        "n_cat_features": 0,
        "train_size": X_train.shape[0],
        "val_size":   X_val.shape[0],
        "test_size":  X_test.shape[0],
        "n_classes":  1,
    }
 
    numerical_data = {
        "train": X_train.astype("float"),
        "val":   X_val.astype("float"),
        "test":  X_test.astype("float"),
    }
 
    targets = {
        "train": y_train.astype("float"),
        "val":   y_val.astype("float"),
        "test":  y_test.astype("float"),
    }
 
    return numerical_data, None, targets, info, None
 
 
# ---------------------------------------------------------------------------
# TRAINING UTILITIES / UTILITÁRIOS DE TREINO
# ---------------------------------------------------------------------------
 
def train_sample(xTrain, yTrain, sample_size):
    """
    Returns a random subsample of the training set of size `sample_size`.
    If `sample_size` is larger than the available data, the full set is returned.
 
    Retorna uma subamostra aleatória do conjunto de treino com tamanho `sample_size`.
    Se `sample_size` for maior que os dados disponíveis, retorna o conjunto completo.
    """
    if sample_size >= len(xTrain):
        return xTrain, yTrain
    x_sampled, _, y_sampled, _ = train_test_split(
        xTrain, yTrain, train_size=sample_size, random_state=42
    )
    return x_sampled, y_sampled
 
 
# ---------------------------------------------------------------------------
# IMPUTATION UTILITIES / UTILITÁRIOS DE IMPUTAÇÃO
# ---------------------------------------------------------------------------
 
def get_separator(file_name: str) -> str:
    """
    Returns the CSV field separator for a given file based on its name.
    IC experiment files use '|'; all others use ','.
 
    Retorna o separador de campos CSV com base no nome do arquivo.
    Arquivos de experimento IC usam '|'; todos os outros usam ','.
    """
    return "|" if file_name.startswith("exp_100") else ","
 
 
def list_csv_files(path_dir: str) -> List[str]:
    """
    Returns a list of all CSV filenames in the given directory.
    Retorna uma lista com todos os nomes de arquivos CSV no diretório dado.
    """
    return [f for f in os.listdir(path_dir) if f.lower().endswith(".csv")]
 
 
def load_reference_file(path_dir: str, files: List[str]) -> pd.DataFrame:
    """
    Loads the first CSV file whose name ends with 'x.csv' (case-insensitive).
    This file is used to determine the reference set of feature columns.
 
    Carrega o primeiro arquivo CSV cujo nome termina com 'x.csv' (sem distinção de maiúsculas).
    Este arquivo é usado para determinar o conjunto de referência de colunas de features.
    """
    for f in files:
        if f.lower().endswith("x.csv"):
            return pd.read_csv(os.path.join(path_dir, f), sep=get_separator(os.path.basename(f)))
    raise ValueError("No file ending with 'x.csv' found in the directory. "
                     "/ Nenhum arquivo terminando com 'x.csv' encontrado no diretório.")
 
 
def compute_statistics(df: pd.DataFrame, columns: List[str], method: str) -> dict:
    """
    Computes per-column statistics needed for the chosen imputation method:
      - 'gaussian': stores (mean, std) per column.
      - 'mean':     stores mean per column.
    Non-numeric columns are stored as None and are later skipped.
 
    Calcula estatísticas por coluna necessárias para o método de imputação escolhido:
      - 'gaussian': armazena (média, desvio padrão) por coluna.
      - 'mean':     armazena a média por coluna.
    Colunas não numéricas são armazenadas como None e ignoradas posteriormente.
    """
    stats = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if method == "gaussian":
                stats[col] = (df[col].mean(skipna=True), df[col].std(skipna=True))
            elif method == "mean":
                stats[col] = df[col].mean(skipna=True)
        else:
            stats[col] = None
    return stats
 
 
def impute_and_save(path_dir: str, path_src: str, source_name: str,
                    method: str = "mean", seed: int = 42):
    """
    Imputes missing columns into all CSV files in `path_dir` using statistics
    derived from the source file at `path_src`, then saves the augmented files
    to a new directory named after the imputation method and source.
 
    Target files (ending with 'y.csv') are copied unchanged.
    Feature files (ending with 'x.csv') receive the imputed columns, reordered alphabetically.
 
    Supported methods:
      - 'mean':     fills missing columns with the column mean from the source.
      - 'gaussian': fills missing columns with values sampled from N(mean, std).
 
    Imputa colunas ausentes em todos os CSVs de `path_dir` usando estatísticas
    derivadas do arquivo fonte em `path_src`, salvando os arquivos aumentados em
    um novo diretório nomeado com base no método de imputação e na fonte.
 
    Arquivos alvo (terminando em 'y.csv') são copiados sem alteração.
    Arquivos de features (terminando em 'x.csv') recebem as colunas imputadas, reordenadas alfabeticamente.
 
    Métodos suportados:
      - 'mean':     preenche colunas ausentes com a média da coluna na fonte.
      - 'gaussian': preenche colunas ausentes com valores amostrados de N(média, desvio).
    """
    np.random.seed(seed)
 
    df_src = pd.read_csv(path_src, sep=get_separator(os.path.basename(path_src)))
 
    files = list_csv_files(path_dir)
    if not files:
        raise ValueError("No CSV files found in the directory. "
                         "/ Nenhum arquivo CSV encontrado no diretório.")
 
    df_ref = load_reference_file(path_dir, files)
 
    # Identify columns present in the source but absent in the reference feature file
    # Identifica colunas presentes na fonte mas ausentes no arquivo de referência de features
    extra_columns = [c for c in df_src.columns if c not in df_ref.columns]
    if not extra_columns:
        print("No new columns found for imputation. / Nenhuma coluna nova encontrada para imputação.")
        return
 
    stats = compute_statistics(df_src, extra_columns, method)
 
    # Build output directory name / Constrói o nome do diretório de saída
    dir_name = os.path.basename(os.path.normpath(path_dir))
    out_dir  = os.path.join(
        os.path.dirname(path_dir),
        f"{dir_name}_Imputation_{method.capitalize()}_{source_name.split('.')[0]}",
    )
    os.makedirs(out_dir, exist_ok=True)
 
    for f in files:
        file_path = os.path.join(path_dir, f)
        df = pd.read_csv(file_path, sep=get_separator(os.path.basename(f)))
 
        if f.lower().endswith("y.csv"):
            # Target files are saved without modification
            # Arquivos alvo são salvos sem modificação
            df.to_csv(os.path.join(out_dir, f), index=False)
            continue
 
        # Add each extra column with imputed values
        # Adiciona cada coluna extra com valores imputados
        for col in extra_columns:
            if pd.api.types.is_numeric_dtype(df_src[col]):
                if stats[col] is not None and not np.isnan(
                    stats[col] if method == "mean" else stats[col][0]
                ):
                    if method == "gaussian":
                        mean, std = stats[col]
                        df[col] = np.random.normal(loc=mean, scale=std, size=len(df))
                    elif method == "mean":
                        df[col] = stats[col]
                else:
                    df[col] = np.nan
 
        # Sort columns alphabetically before saving
        # Ordena colunas alfabeticamente antes de salvar
        df = df[sorted(df.columns)]
        df.to_csv(os.path.join(out_dir, f), index=False)
 
    print(f"Files generated in / Arquivos gerados em: {out_dir}")