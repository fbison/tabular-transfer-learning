""" mimic_tools.py
    Utilities for splitting MetaMIMIC data into upstream and downstream tasks
    Developed for Tabular-Transfer-Learning project
    March 2022
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import random
import pandas as pd
import torch
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression



downstream_columns = ["Molecule","SMILES","Formula_x","SpDiam_A","AATS5d","AATS7s","AATS8s","AATS1i","AATS2i","AATS3i","AATS4i","AATS6i","ATSC1dv","ATSC8dv","ATSC1d","ATSC7d","AATSC0v","MATS6s","MATS7s","MATS8s","GATS2c","GATS3c","GATS8c","GATS1dv","GATS3dv","GATS4dv","GATS6dv","GATS7dv","GATS8dv","GATS1d","GATS2d","GATS5d","GATS2s","GATS3s","GATS4s","GATS6s","GATS1v","GATS2v","GATS5p","GATS6p","GATS7p","GATS1i","GATS2i","GATS3i","GATS4i","GATS5i","GATS7i","GATS8i","RNCG","RPCG","Xc-3dv","Xc-5dv","Xc-6dv","AXp-0d","SdssC","SsNH2","SdO","SssO","SssS","SaaS","SddssS","MAXaaCH","AETA_alpha","AETA_beta_ns_d","ETA_dAlpha_B","ETA_epsilon_5","ETA_dEpsilon_D","IC1","CIC1","CIC2","ZMIC2","PEOE_VSA1","PEOE_VSA4","PEOE_VSA6","PEOE_VSA9","SMR_VSA1","SMR_VSA3","SMR_VSA4","SMR_VSA9","SlogP_VSA2","SlogP_VSA3","SlogP_VSA4","SlogP_VSA10","EState_VSA4","EState_VSA5","VSA_EState8","VSA_EState9","AMID_C","TopoPSA(NO)","GGI6","GGI7","JGI4","RNCS","Mor02m","Mor03m","Mor13m","Mor26m","Mor30m","Mor31m","MOMI-Z","MLOGP","ESOL_Solubility_(mg/ml)","Ali_Log_S","pIC50"]
upstream2_columns = ["Molecule","SMILES","Formula_x","SpMax_A","VE1_A","AATS8dv","AATS8s","AATS2i","ATSC1dv","ATSC8d","ATSC0p","ATSC0i","MATS1c","MATS2s","MATS3s","MATS6s","MATS7s","MATS8s","GATS4c","GATS1dv","GATS5dv","GATS7dv","GATS6d","GATS7d","GATS2s","GATS3s","GATS2v","GATS3v","GATS1p","GATS6p","GATS3i","GATS6i","GATS8i","BCUTc-1h","BCUTd-1l","BCUTs-1h","RPCG","Xch-5d","Xch-7d","Xc-5d","Xc-5dv","Xc-6dv","AXp-1d","SdssC","SaasC","SaaaC","SssssC","SsNH2","SssNH","SsOH","SssO","SdS","SddssS","MAXaaCH","AETA_beta_s","AETA_eta_L","AETA_eta_F","ETA_epsilon_5","IC1","IC2","CIC2","ZMIC1","PEOE_VSA1","PEOE_VSA2","PEOE_VSA9","SlogP_VSA1","SlogP_VSA2","SlogP_VSA10","EState_VSA1","EState_VSA2","EState_VSA3","EState_VSA6","EState_VSA9","VSA_EState3","VSA_EState7","VSA_EState8","MDEC-33","TopoPSA(NO)","GGI3","GGI5","GGI6","GGI7","GGI8","GGI9","JGI2","JGI5","FPSA3","RPCS","Mor02m","Mor03m","Mor06m","Mor08m","Mor11m","Mor13m","Mor16m","Mor23m","XLOGP3","Silicos-IT_Log_P","ESOL_Log_S","ESOL_Solubility_(mg/ml)","Ali_Log_S","Ali_Solubility_(mg/ml)","Silicos-IT_Solubility_(mg/ml)","pIC50"]
upstream3_columns = ["Molecule","SMILES","Formula_x","SpDiam_A","AATS3d","AATS2s","AATS3v","AATS1i","AATS2i","AATS3i","AATS4i","ATSC8dv","ATSC1d","ATSC8d","ATSC0i","AATSC0c","MATS1c","MATS2s","MATS7s","GATS1c","GATS2c","GATS5c","GATS7c","GATS8c","GATS2dv","GATS5dv","GATS7dv","GATS8dv","GATS1d","GATS3d","GATS5d","GATS6d","GATS7d","GATS8d","GATS2s","GATS6s","GATS6v","GATS3p","GATS5p","GATS1i","GATS6i","BCUTs-1h","BCUTs-1l","BCUTi-1h","BCUTi-1l","RPCG","Xch-6d","Xc-5d","Xc-3dv","Xc-4dv","AXp-1d","SsCH3","SdCH2","SdsCH","SsssCH","SaaNH","SdsN","SsOH","SssO","SdS","SaaS","SddssS","MINaasC","AETA_alpha","AETA_beta_ns_d","ETA_epsilon_5","ETA_dEpsilon_C","IC2","ZMIC2","PEOE_VSA1","PEOE_VSA2","PEOE_VSA3","PEOE_VSA6","PEOE_VSA7","PEOE_VSA11","SMR_VSA5","SMR_VSA6","SlogP_VSA2","SlogP_VSA3","EState_VSA1","EState_VSA2","EState_VSA3","EState_VSA6","VSA_EState7","AMID_C","TopoPSA(NO)","GGI5","GGI7","GGI8","GGI9","JGI3","TSRW10","TASA","Mor02m","Mor10m","Mor11m","Mor12m","Mor16m","Mor21m","Mor22m","Mor23m","iLOGP","Silicos-IT_Log_P","pIC50"]
upstream4_columns = ["Molecule","SMILES","Formula_x","SpMax_A","VE1_A","AATS7d","AATS6s","AATS8s","AATS3p","AATS1i","AATS3i","AATS4i","AATS6i","ATSC1d","AATSC0c","MATS1c","MATS5s","MATS6s","MATS7s","GATS1c","GATS6c","GATS7c","GATS1dv","GATS3dv","GATS5dv","GATS6dv","GATS3d","GATS4d","GATS6d","GATS1s","GATS2s","GATS8s","GATS2v","GATS3v","GATS7v","GATS4p","GATS8i","BCUTc-1l","BCUTd-1l","BCUTs-1h","BCUTs-1l","RNCG","RPCG","Xc-5dv","Xc-6dv","SsCH3","SdCH2","SdsCH","SsssCH","SaaaC","SssssC","SssNH","SsssN","SsOH","SssO","SdS","SsCl","MAXaasC","MINaaCH","ETA_shape_y","AETA_beta_s","AETA_eta_L","ETA_dEpsilon_D","fMF","ZMIC2","PEOE_VSA3","PEOE_VSA4","PEOE_VSA6","PEOE_VSA7","PEOE_VSA8","PEOE_VSA10","SMR_VSA1","SMR_VSA9","SlogP_VSA1","SlogP_VSA2","SlogP_VSA4","SlogP_VSA5","SlogP_VSA10","EState_VSA1","VSA_EState1","VSA_EState8","VSA_EState9","MDEC-22","MDEC-23","TopoPSA(NO)","GGI3","GGI4","GGI6","GGI8","GGI9","GGI10","JGI2","FNSA1","RASA","Mor02m","Mor08m","Mor10m","Mor12m","Mor23m","Mor24m","MOMI-Z","Silicos-IT_Log_P","ESOL_Solubility_(mg/ml)","pIC50"]


missing_features = {
    'up2_only': ['AATS8dv', 'AETA_beta_s', 'AETA_eta_F', 'AETA_eta_L', 'ATSC0i', 'ATSC0p', 'ATSC8d', 'AXp-1d', 'Ali_Solubility_(mg/ml)', 'BCUTc-1h', 'BCUTd-1l', 'BCUTs-1h', 'ESOL_Log_S', 'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA6', 'EState_VSA9', 'FPSA3', 'GATS1p', 'GATS3v', 'GATS4c', 'GATS5dv', 'GATS6d', 'GATS6i', 'GATS7d', 'GGI3', 'GGI5', 'GGI8', 'GGI9', 'IC2', 'JGI2', 'JGI5', 'MATS1c', 'MATS2s', 'MATS3s', 'MDEC-33', 'Mor06m', 'Mor08m', 'Mor11m', 'Mor16m', 'Mor23m', 'PEOE_VSA2', 'RPCS', 'SaaaC', 'SaasC', 'SdS', 'Silicos-IT_Log_P', 'Silicos-IT_Solubility_(mg/ml)', 'SlogP_VSA1', 'SpMax_A', 'SsOH', 'SssNH', 'SssssC', 'VE1_A', 'VSA_EState3', 'VSA_EState7', 'XLOGP3', 'Xc-5d', 'Xch-5d', 'Xch-7d', 'ZMIC1'],
    'up2_missing': ['AATS1i', 'AATS3i', 'AATS4i', 'AATS5d', 'AATS6i', 'AATS7s', 'AATSC0v', 'AETA_alpha', 'AETA_beta_ns_d', 'AMID_C', 'ATSC1d', 'ATSC7d', 'ATSC8dv', 'AXp-0d', 'CIC1', 'EState_VSA4', 'EState_VSA5', 'ETA_dAlpha_B', 'ETA_dEpsilon_D', 'GATS1d', 'GATS1i', 'GATS1v', 'GATS2c', 'GATS2d', 'GATS2i', 'GATS3c', 'GATS3dv', 'GATS4dv', 'GATS4i', 'GATS4s', 'GATS5d', 'GATS5i', 'GATS5p', 'GATS6dv', 'GATS6s', 'GATS7i', 'GATS7p', 'GATS8c', 'GATS8dv', 'JGI4', 'MLOGP', 'MOMI-Z', 'Mor26m', 'Mor30m', 'Mor31m', 'PEOE_VSA4', 'PEOE_VSA6', 'RNCG', 'RNCS', 'SMR_VSA1', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA9', 'SaaS', 'SdO', 'SlogP_VSA3', 'SlogP_VSA4', 'SpDiam_A', 'SssS', 'VSA_EState9', 'Xc-3dv', 'ZMIC2'],
    'up2_common': ['AATS2i', 'AATS8s', 'ATSC1dv', 'Ali_Log_S', 'CIC2', 'ESOL_Solubility_(mg/ml)', 'ETA_epsilon_5', 'Formula_x', 'GATS1dv', 'GATS2s', 'GATS2v', 'GATS3i', 'GATS3s', 'GATS6p', 'GATS7dv', 'GATS8i', 'GGI6', 'GGI7', 'IC1', 'MATS6s', 'MATS7s', 'MATS8s', 'MAXaaCH', 'Molecule', 'Mor02m', 'Mor03m', 'Mor13m', 'PEOE_VSA1', 'PEOE_VSA9', 'RPCG', 'SMILES', 'SddssS', 'SdssC', 'SlogP_VSA10', 'SlogP_VSA2', 'SsNH2', 'SssO', 'TopoPSA(NO)', 'VSA_EState8', 'Xc-5dv', 'Xc-6dv', 'pIC50'], 
    
    'up3_only': ['AATS2s', 'AATS3d', 'AATS3v', 'AATSC0c', 'ATSC0i', 'ATSC8d', 'AXp-1d', 'BCUTi-1h', 'BCUTi-1l', 'BCUTs-1h', 'BCUTs-1l', 'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA6', 'ETA_dEpsilon_C', 'GATS1c', 'GATS2dv', 'GATS3d', 'GATS3p', 'GATS5c', 'GATS5dv', 'GATS6d', 'GATS6i', 'GATS6v', 'GATS7c', 'GATS7d', 'GATS8d', 'GGI5', 'GGI8', 'GGI9', 'IC2', 'JGI3', 'MATS1c', 'MATS2s', 'MINaasC', 'Mor10m', 'Mor11m', 'Mor12m', 'Mor16m', 'Mor21m', 'Mor22m', 'Mor23m', 'PEOE_VSA11', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA7', 'SMR_VSA5', 'SMR_VSA6', 'SaaNH', 'SdCH2', 'SdS', 'SdsCH', 'SdsN', 'Silicos-IT_Log_P', 'SsCH3', 'SsOH', 'SsssCH', 'TASA', 'TSRW10', 'VSA_EState7', 'Xc-4dv', 'Xc-5d', 'Xch-6d', 'iLOGP'],
    'up3_missing': ['AATS5d', 'AATS6i', 'AATS7s', 'AATS8s', 'AATSC0v', 'ATSC1dv', 'ATSC7d', 'AXp-0d', 'Ali_Log_S', 'CIC1', 'CIC2', 'ESOL_Solubility_(mg/ml)', 'EState_VSA4', 'EState_VSA5', 'ETA_dAlpha_B', 'ETA_dEpsilon_D', 'GATS1dv', 'GATS1v', 'GATS2d', 'GATS2i', 'GATS2v', 'GATS3c', 'GATS3dv', 'GATS3i', 'GATS3s', 'GATS4dv', 'GATS4i', 'GATS4s', 'GATS5i', 'GATS6dv', 'GATS6p', 'GATS7i', 'GATS7p', 'GATS8i', 'GGI6', 'IC1', 'JGI4', 'MATS6s', 'MATS8s', 'MAXaaCH', 'MLOGP', 'MOMI-Z', 'Mor03m', 'Mor13m', 'Mor26m', 'Mor30m', 'Mor31m', 'PEOE_VSA4', 'PEOE_VSA9', 'RNCG', 'RNCS', 'SMR_VSA1', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA9', 'SdO', 'SdssC', 'SlogP_VSA10', 'SlogP_VSA4', 'SsNH2', 'SssS', 'VSA_EState8', 'VSA_EState9', 'Xc-5dv', 'Xc-6dv'],
    'up3_common': ['AATS1i', 'AATS2i', 'AATS3i', 'AATS4i', 'AETA_alpha', 'AETA_beta_ns_d', 'AMID_C', 'ATSC1d', 'ATSC8dv', 'ETA_epsilon_5', 'Formula_x', 'GATS1d', 'GATS1i', 'GATS2c', 'GATS2s', 'GATS5d', 'GATS5p', 'GATS6s', 'GATS7dv', 'GATS8c', 'GATS8dv', 'GGI7', 'MATS7s', 'Molecule', 'Mor02m', 'PEOE_VSA1', 'PEOE_VSA6', 'RPCG', 'SMILES', 'SaaS', 'SddssS', 'SlogP_VSA2', 'SlogP_VSA3', 'SpDiam_A', 'SssO', 'TopoPSA(NO)', 'Xc-3dv', 'ZMIC2', 'pIC50'], 
    
    'up4_only': ['AATS3p', 'AATS6s', 'AATS7d', 'AATSC0c', 'AETA_beta_s', 'AETA_eta_L', 'BCUTc-1l', 'BCUTd-1l', 'BCUTs-1h', 'BCUTs-1l', 'EState_VSA1', 'ETA_shape_y', 'FNSA1', 'GATS1c', 'GATS1s', 'GATS3d', 'GATS3v', 'GATS4d', 'GATS4p', 'GATS5dv', 'GATS6c', 'GATS6d', 'GATS7c', 'GATS7v', 'GATS8s', 'GGI10', 'GGI3', 'GGI4', 'GGI8', 'GGI9', 'JGI2', 'MATS1c', 'MATS5s', 'MAXaasC', 'MDEC-22', 'MDEC-23', 'MINaaCH', 'Mor08m', 'Mor10m', 'Mor12m', 'Mor23m', 'Mor24m', 'PEOE_VSA10', 'PEOE_VSA3', 'PEOE_VSA7', 'PEOE_VSA8', 'RASA', 'SaaaC', 'SdCH2', 'SdS', 'SdsCH', 'Silicos-IT_Log_P', 'SlogP_VSA1', 'SlogP_VSA5', 'SpMax_A', 'SsCH3', 'SsCl', 'SsOH', 'SssNH', 'SsssCH', 'SsssN', 'SssssC', 'VE1_A', 'VSA_EState1', 'fMF'],
    'up4_missing': ['AATS2i', 'AATS5d', 'AATS7s', 'AATSC0v', 'AETA_alpha', 'AETA_beta_ns_d', 'AMID_C', 'ATSC1dv', 'ATSC7d', 'ATSC8dv', 'AXp-0d', 'Ali_Log_S', 'CIC1', 'CIC2', 'EState_VSA4', 'EState_VSA5', 'ETA_dAlpha_B', 'ETA_epsilon_5', 'GATS1d', 'GATS1i', 'GATS1v', 'GATS2c', 'GATS2d', 'GATS2i', 'GATS3c', 'GATS3i', 'GATS3s', 'GATS4dv', 'GATS4i', 'GATS4s', 'GATS5d', 'GATS5i', 'GATS5p', 'GATS6p', 'GATS6s', 'GATS7dv', 'GATS7i', 'GATS7p', 'GATS8c', 'GATS8dv', 'GGI7', 'IC1', 'JGI4', 'MATS8s', 'MAXaaCH', 'MLOGP', 'Mor03m', 'Mor13m', 'Mor26m', 'Mor30m', 'Mor31m', 'PEOE_VSA1', 'PEOE_VSA9', 'RNCS', 'SMR_VSA3', 'SMR_VSA4', 'SaaS', 'SdO', 'SddssS', 'SdssC', 'SlogP_VSA3', 'SpDiam_A', 'SsNH2', 'SssS', 'Xc-3dv'],
    'up4_common': ['AATS1i', 'AATS3i', 'AATS4i', 'AATS6i', 'AATS8s', 'ATSC1d', 'ESOL_Solubility_(mg/ml)', 'ETA_dEpsilon_D', 'Formula_x', 'GATS1dv', 'GATS2s', 'GATS2v', 'GATS3dv', 'GATS6dv', 'GATS8i', 'GGI6', 'MATS6s', 'MATS7s', 'MOMI-Z', 'Molecule', 'Mor02m', 'PEOE_VSA4', 'PEOE_VSA6', 'RNCG', 'RPCG', 'SMILES', 'SMR_VSA1', 'SMR_VSA9', 'SlogP_VSA10', 'SlogP_VSA2', 'SlogP_VSA4', 'SssO', 'TopoPSA(NO)', 'VSA_EState8', 'VSA_EState9', 'Xc-5dv', 'Xc-6dv', 'ZMIC2', 'pIC50']}


non_numerical_columns = ['Molecule', 'SMILES', 'Formula_x']
default_target_columns = ["pIC50"]

def remove_common_strings(downstream_columns, non_numerical_columns, target_columns, target=0):
    """
    Removes strings from the first list that are present in the other two lists.

    Args:
        downstream_columns: The initial list of strings.
        non_numerical_columns: A list of strings to be removed from the first list.
        target_columns: Another list of strings to be removed from the first list.

    Returns:
        A new list containing strings from downstream_columns that are not in
        non_numerical_columns or target_columns.
    """
    # Convert lists to sets for efficient set operations
    set_downstream = set(downstream_columns)
    set_non_numerical = set(non_numerical_columns)
    set_target = set([target_columns])

    # Perform the set difference operation
    # (set_downstream - set_non_numerical) gets elements in downstream_columns but not in non_numerical_columns
    # Then, subtract set_target from that result
    result_set = set_downstream - set_non_numerical - set_target

    # Convert the resulting set back to a list (if a list is required for the return)
    return list(result_set)

def get_x_features(dataset_name, target_columns, target=0):
    """
    Function to get the features for a given dataset
    """
    
    if dataset_name == 'ic_upstream2':
        return remove_common_strings(upstream2_columns, non_numerical_columns, target_columns[target])
    elif dataset_name == 'ic_upstream3':
        return remove_common_strings(upstream3_columns, non_numerical_columns, target_columns[target])
    elif dataset_name == 'ic_upstream4':
        return remove_common_strings(upstream4_columns, non_numerical_columns, target_columns[target])
    elif dataset_name == 'ic_downstream1':
        return remove_common_strings(downstream_columns, non_numerical_columns, target_columns[target])
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

def combine_unique_sorted(arr1, arr2, arr3):
    combined = arr1 + arr2 + arr3  # concatena os arrays
    return sorted(set(combined))  # remove repetições e ordena

def get_target_columns(dataset_name):
    """
    Function to get the target columns for a given dataset
    """
    
    if dataset_name == 'ic_upstream2':
        return default_target_columns + missing_features['up2_only']
    elif dataset_name == 'ic_upstream3':
        return default_target_columns + missing_features['up3_only']
    elif dataset_name == 'ic_upstream4':
        return default_target_columns + missing_features['up4_only']
    elif dataset_name == 'ic_downstream1':
        return default_target_columns + combine_unique_sorted(missing_features['up2_missing'], missing_features['up3_missing'], missing_features['up4_missing'])
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

def split_ic_dataset(dataset_name, dataset_number, target_columns):
    base_path = f'../../../data/{dataset_name}/'
    dataset = pd.read_csv(f'{base_path}exp_100_{dataset_number}.csv', delimiter = '|')
    dataset = dataset.drop(columns = non_numerical_columns)

    dataset = dataset.astype(float)

    y_full = dataset[target_columns].copy()


    dataset.drop(columns = default_target_columns, inplace = True) #Não retira tudo, pois as outras targets collumns serão usadas como X em outras tasks
    
    X_full = dataset.copy()


    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1875, random_state=1) # 0.1875 x 0.8 = 0.15


    X_train.to_csv(f'{base_path}ic_train_X.csv', index = False)
    X_val.to_csv(f'{base_path}ic_val_X.csv', index = False)
    X_test.to_csv(f'{base_path}ic_test_X.csv', index = False)

    y_train.to_csv(f'{base_path}ic_train_y.csv', index = False)
    y_val.to_csv(f'{base_path}ic_val_y.csv', index = False)
    y_test.to_csv(f'{base_path}ic_test_y.csv', index = False)
    return

def read_ic_dataset(dataset_name, target_colums, target, stage='pretrain'):
    """
    Function to read the IC dataset.
    Raises FileNotFoundError if any required CSV file is not found.
    """
    base_path = f'../../../data/{dataset_name}/'
    
    # Define file paths
    file_paths = {
        'X_train': os.path.join(base_path, 'ic_train_X.csv'),
        'X_val': os.path.join(base_path, 'ic_val_X.csv'),
        'X_test': os.path.join(base_path, 'ic_test_X.csv'),
        'y_train': os.path.join(base_path, 'ic_train_y.csv'),
        'y_val': os.path.join(base_path, 'ic_val_y.csv'),
        'y_test': os.path.join(base_path, 'ic_test_y.csv')
    }

    # Check if all files exist before attempting to read
    for key, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required dataset file not found: {path}")

    # Read the CSV files
    X_train = pd.read_csv(file_paths['X_train'])
    X_val = pd.read_csv(file_paths['X_val'])
    X_test = pd.read_csv(file_paths['X_test'])
    
    y_train_full = pd.read_csv(file_paths['y_train']).drop(columns = (target_colums[target] - default_target_columns[0] if target_colums[target] not in default_target_columns else []))
    y_val_full = pd.read_csv(file_paths['y_val']).drop(columns = (target_colums[target] - default_target_columns[0] if target_colums[target] not in default_target_columns else []))
    y_test_full = pd.read_csv(file_paths['y_test']).drop(columns = (target_colums[target] - default_target_columns[0] if target_colums[target] not in default_target_columns else []))

    y_train = y_train_full[[target_colums[target]]]
    y_val = y_val_full[[target_colums[target]]]
    y_test = y_test_full[[target_colums[target]]]

    if stage == 'ic_downstream1':
        # Merge validation set into train, keep the dummy validation set for the code not to fail
        y_train = pd.concat([y_train, y_val], ignore_index=True)
        X_train = pd.concat([X_train, X_val], ignore_index=True)

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_datasets(dataset_name, dataset_number, target_columns, target=0, stage='pretrain'):
    """
    Function to get the datasets for a given dataset name and number.
    """

    try:
        return read_ic_dataset(dataset_name, target_columns, target, stage)
    except FileNotFoundError:
        split_ic_dataset(dataset_name, dataset_number, target_columns)
        return read_ic_dataset(dataset_name, target_columns, target, stage)
    
def get_last_char_as_int(s: str) -> int:
    if not s:
        raise ValueError("Input string cannot be empty.")

    last_char = s[-1]
    
    try:
        return int(last_char)
    except ValueError:
        raise ValueError(f"Last character '{last_char}' cannot be converted to an integer.")

def get_ic_dataset(dataset_name, task, stage):
    print(f"Loading dataset: {dataset_name} for task: {task} at stage: {stage}")
    dataset_id = get_last_char_as_int(dataset_name)
    target_columns = get_target_columns(dataset_name)
    print(f"Target columns: {len(target_columns)}")
    X_features = get_x_features(dataset_name, target_columns)
    X_train, X_val, X_test, y_train, y_val, y_test = get_datasets(dataset_name, dataset_id, target_columns = target_columns, stage=stage)


    info = {"name": dataset_id,
            "task_type": task,
            "n_num_features": len(X_features),
            "n_cat_features": 0, # Ic is a regreession task, so no categorical features
            "train_size": X_train.shape[0],
            "val_size": X_val.shape[0],
            "test_size": X_test.shape[0]}
    
    
    info["n_classes"] = 1
    numerical_data = {"train": X_train.values.astype('float'), "val": X_val.values.astype('float'), "test": X_test.values.astype('float')}
    categorical_data = None
    full_cat_data_for_encoder = None

    targets = {"train": y_train.values.astype('float'), "val": y_val.values.astype('float'), "test": y_test.values.astype('float')}        

    return numerical_data, categorical_data, targets, info, full_cat_data_for_encoder


def get_synthetic_dataset(n_samples=1000, n_features=10, noise=10.0, val_size=0.2, test_size=0.2, random_state=42):
    print(f"Generating synthetic dataset with {n_samples} samples, {n_features} features, noise={noise}")

    # Gerar os dados
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )

    # Dividir entre treino, validação e teste
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_size / (1 - test_size), random_state=random_state)

    # Informações sobre o dataset
    info = {
        "name": "synthetic",
        "task_type": "regression",
        "n_num_features": n_features,
        "n_cat_features": 0,
        "train_size": X_train.shape[0],
        "val_size": X_val.shape[0],
        "test_size": X_test.shape[0],
        "n_classes": 1
    }

    numerical_data = {
        "train": X_train.astype('float'),
        "val": X_val.astype('float'),
        "test": X_test.astype('float')
    }

    categorical_data = None
    full_cat_data_for_encoder = None

    targets = {
        "train": y_train.astype('float'),
        "val": y_val.astype('float'),
        "test": y_test.astype('float')
    }

    return numerical_data, categorical_data, targets, info, full_cat_data_for_encoder
