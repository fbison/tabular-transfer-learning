""" testing.py
    Utilities for testing models
    Developed for Tabular-Transfer-Learning project
    March 2022
    Some functionality adopted from https://github.com/Yura52/rtdl
"""

import torch
from sklearn.metrics import accuracy_score, mean_squared_error, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
import numpy as np

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114

def _predict_from_outputs(outputs, task):
    if task == "multiclass":
        return torch.argmax(outputs, dim=1)
    elif task in {"binclass", "regression", "multiVariantRegression"}:
        return outputs
    else:
        raise ValueError(f"Unknown task: {task}")


def _compute_scores(targets_all, predictions_all, task):
    # Maintain in float, because json has problems with float32
    if task == "multiclass":
        accuracy = float(accuracy_score(targets_all, predictions_all))
        balanced_accuracy = float(balanced_accuracy_score(targets_all, predictions_all))
        balanced_accuracy_adjusted = float(
            balanced_accuracy_score(targets_all, predictions_all, adjusted=True)
        )
        return {
            "score": accuracy,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "balanced_accuracy_adjusted": balanced_accuracy_adjusted,
        }

    elif task in {"regression", "multiVariantRegression"}:
        if targets_all.shape[0] <= 1:
            rmse = 0.0
        else:
            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        targets_all,
                        predictions_all,
                        multioutput="uniform_average"
                    )
                )
            )
        return {
            "score": -rmse,
            "rmse": rmse,
        }

    elif task == "binclass":
        roc_auc = float(roc_auc_score(targets_all, predictions_all))
        return {
            "score": roc_auc,
            "roc_auc": roc_auc,
        }


def evaluate_model(net, loaders, task, device):
    scores = []
    for loader in loaders: #TODO: validar loader vazio
        score = test_default(net, loader, task, device)
        scores.append(score)
    return scores


def test_default(net, testloader, task, device):
    net.eval()
    targets_all = []
    predictions_all = []

    with torch.no_grad():
        for inputs_num, inputs_cat, targets in testloader:
            inputs_num = inputs_num.to(device).float()
            inputs_cat = inputs_cat.to(device)
            targets = targets.to(device)

            inputs_num = inputs_num if inputs_num.nelement() != 0 else None
            inputs_cat = inputs_cat if inputs_cat.nelement() != 0 else None

            outputs = net(inputs_num, inputs_cat)
            predicted = _predict_from_outputs(outputs, task)

            targets_all.append(targets.cpu().numpy())
            predictions_all.append(predicted.cpu().numpy())

    if len(targets_all) == 0:
        # loader vazio → nenhuma amostra
        return {
            "score": -0,
        } 

    targets_all = np.concatenate(targets_all, axis=0)
    predictions_all = np.concatenate(predictions_all, axis=0)

    return _compute_scores(targets_all, predictions_all, task)



def evaluate_backbone(embedders, backbone, heads, loaders, tasks, device):
    scores = {}
    for k in loaders:
        scores[k] = evaluate_backbone_one_dataset(
            embedders[k],
            backbone,
            heads[k],
            loaders[k],
            tasks[k],
            device,
        )
    return scores

def evaluate_backbone_one_dataset(embedder, backbone, head, testloader, task, device):
    embedder.eval()
    backbone.eval()
    head.eval()

    targets_all = []
    predictions_all = []

    with torch.no_grad():
        for inputs_num, inputs_cat, targets in testloader:
            inputs_num = inputs_num.to(device).float()
            inputs_cat = inputs_cat.to(device)
            targets = targets.to(device)

            inputs_num = inputs_num if inputs_num.nelement() != 0 else None
            inputs_cat = inputs_cat if inputs_cat.nelement() != 0 else None

            embedding = embedder(inputs_num, inputs_cat)
            features = backbone(embedding)
            outputs = head(features)

            predicted = _predict_from_outputs(outputs, task)

            targets_all.append(targets.cpu().numpy())
            predictions_all.append(predicted.cpu().numpy())

    targets_all = np.concatenate(targets_all, axis=0)
    predictions_all = np.concatenate(predictions_all, axis=0)

    return _compute_scores(targets_all, predictions_all, task)
