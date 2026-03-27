# # train_utils.py

# import os
# import sys
# import json
# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sn

# from tqdm import tqdm
# from sklearn.metrics import (
#     cohen_kappa_score,
#     confusion_matrix,
#     roc_auc_score
# )


# # ===============================
# # METRICAS
# # ===============================

# def compute_metrics(y_true, y_pred, y_prob):
#     """
#     Calcula Sensitivity, Specificity e AUC multiclass
#     """
#     cm = confusion_matrix(y_true, y_pred)

#     sensitivities = []
#     specificities = []

#     for i in range(len(cm)):
#         tp = cm[i, i]
#         fn = cm[i, :].sum() - tp
#         fp = cm[:, i].sum() - tp
#         tn = cm.sum() - (tp + fn + fp)

#         sens = tp / (tp + fn + 1e-8)
#         spec = tn / (tn + fp + 1e-8)

#         sensitivities.append(sens)
#         specificities.append(spec)

#     sensitivity = np.mean(sensitivities)
#     specificity = np.mean(specificities)

#     y_prob_arr = np.array(y_prob)  # shape = [num_samples, num_classes]
#     try:
#         auc = roc_auc_score(y_true, y_prob_arr, multi_class="ovr")
#     except:
#         auc = 0

#     return sensitivity, specificity, auc


# # ===============================
# # TREINO
# # ===============================

# def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
#     """
#     Treina uma época e retorna métricas detalhadas
#     """
#     model.train()
#     loss_function = nn.CrossEntropyLoss()

#     accu_loss = torch.zeros(1).to(device)
#     accu_num = torch.zeros(1).to(device)
#     sample_num = 0

#     y_true_list = []
#     y_pred_list = []
#     y_prob_list = []

#     data_loader = tqdm(data_loader, file=sys.stdout)

#     for step, data in enumerate(data_loader):
#         images, labels = data
#         images = images.to(device)
#         labels = labels.to(device)
#         sample_num += images.shape[0]

#         pred = model(images)
#         prob = F.softmax(pred, dim=1).detach().cpu().numpy()
#         pred_classes = torch.max(pred, dim=1)[1]

#         accu_num += torch.eq(pred_classes, labels).sum()
#         loss = loss_function(pred, labels)
#         loss.backward()
#         accu_loss += loss.detach()

#         y_pred_list.extend(pred_classes.cpu().numpy())
#         y_true_list.extend(labels.cpu().numpy())
#         y_prob_list.extend(prob)

#         data_loader.desc = "[train epoch {}] loss: {:.4f}, acc: {:.4f}, lr: {:.7f}".format(
#             epoch,
#             accu_loss.item() / (step + 1),
#             accu_num.item() / sample_num,
#             optimizer.param_groups[0]["lr"]
#         )

#         if not torch.isfinite(loss):
#             print('WARNING: non-finite loss, ending training ', loss)
#             sys.exit(1)

#         optimizer.step()
#         optimizer.zero_grad()
#         lr_scheduler.step()

#     train_kappa = cohen_kappa_score(y_true_list, y_pred_list)
#     sensitivity, specificity, auc = compute_metrics(y_true_list, y_pred_list, y_prob_list)

#     print(f"[TRAIN] Acc: {accu_num.item()/sample_num:.4f} | Kappa: {train_kappa:.4f} | "
#           f"Sens: {sensitivity:.4f} | Spec: {specificity:.4f} | AUC: {auc:.4f}")

#     return (
#         accu_loss.item() / (step + 1),
#         accu_num.item() / sample_num,
#         train_kappa,
#         sensitivity,
#         specificity,
#         auc,
#         y_true_list,
#         y_pred_list
#     )


# # ===============================
# # VALIDACAO / TESTE
# # ===============================

# @torch.no_grad()
# def evaluate(model, data_loader, device, epoch, save_conf_matrix=False, conf_matrix_path=None):
#     """
#     Avalia modelo e retorna métricas detalhadas.
#     Pode salvar a matriz de confusão.
#     """
#     model.eval()
#     loss_function = nn.CrossEntropyLoss()

#     accu_num = torch.zeros(1).to(device)
#     accu_loss = torch.zeros(1).to(device)
#     sample_num = 0

#     y_true_list = []
#     y_pred_list = []
#     y_prob_list = []

#     data_loader = tqdm(data_loader, file=sys.stdout)

#     for step, data in enumerate(data_loader):
#         images, labels = data
#         images = images.to(device)
#         labels = labels.to(device)
#         sample_num += images.shape[0]

#         pred = model(images)
#         prob = F.softmax(pred, dim=1).cpu().numpy()
#         pred_classes = torch.max(pred, dim=1)[1]

#         accu_num += torch.eq(pred_classes, labels).sum()
#         loss = loss_function(pred, labels)
#         accu_loss += loss

#         y_pred_list.extend(pred_classes.cpu().numpy())
#         y_true_list.extend(labels.cpu().numpy())
#         y_prob_list.extend(prob)

#         data_loader.desc = "[eval epoch {}] loss: {:.4f}, acc: {:.4f}".format(
#             epoch,
#             accu_loss.item() / (step + 1),
#             accu_num.item() / sample_num
#         )

#     kappa = cohen_kappa_score(y_true_list, y_pred_list)
#     sensitivity, specificity, auc = compute_metrics(y_true_list, y_pred_list, y_prob_list)

#     print(f"[EVAL] Acc: {accu_num.item()/sample_num:.4f} | Kappa: {kappa:.4f} | "
#           f"Sens: {sensitivity:.4f} | Spec: {specificity:.4f} | AUC: {auc:.4f}")

#     # Matriz de confusão
#     if save_conf_matrix and conf_matrix_path is not None:
#         cm = confusion_matrix(y_true_list, y_pred_list)
#         print("\nConfusion Matrix:")
#         print(cm)

#         plt.figure()
#         sn.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#         plt.title("Confusion Matrix")
#         plt.xlabel("Predicted")
#         plt.ylabel("True")
#         plt.savefig(conf_matrix_path)
#         plt.close()
#         print(f"Confusion matrix saved at {conf_matrix_path}")

#     return (
#         accu_loss.item() / (step + 1),
#         accu_num.item() / sample_num,
#         kappa,
#         sensitivity,
#         specificity,
#         auc,
#         y_true_list,
#         y_pred_list
#     )


# # ===============================
# # LR SCHEDULER
# # ===============================

# def create_lr_scheduler(
#         optimizer,
#         num_step: int,
#         epochs: int,
#         warmup=True,
#         warmup_epochs=1,
#         warmup_factor=1e-3,
#         end_factor=1e-6):

#     assert num_step > 0 and epochs > 0
#     if not warmup:
#         warmup_epochs = 0

#     def f(x):
#         if warmup and x <= (warmup_epochs * num_step):
#             alpha = float(x) / (warmup_epochs * num_step)
#             return warmup_factor * (1 - alpha) + alpha
#         else:
#             current_step = (x - warmup_epochs * num_step)
#             cosine_steps = (epochs - warmup_epochs) * num_step
#             return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


# # ===============================
# # PARAMETER GROUPS
# # ===============================

# def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
#     """
#     Cria grupos de parâmetros com e sem decay
#     """
#     parameter_group_vars = {
#         "decay": {"params": [], "weight_decay": weight_decay},
#         "no_decay": {"params": [], "weight_decay": 0.}
#     }
#     parameter_group_names = {
#         "decay": {"params": [], "weight_decay": weight_decay},
#         "no_decay": {"params": [], "weight_decay": 0.}
#     }

#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue
#         group_name = "no_decay" if len(param.shape) == 1 or name.endswith(".bias") else "decay"
#         parameter_group_vars[group_name]["params"].append(param)
#         parameter_group_names[group_name]["params"].append(name)

#     print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
#     return list(parameter_group_vars.values())

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score
)

@torch.no_grad()
def evaluate_slice_level(model, data_loader, device, num_classes=3):


    model.eval()

    y_true = []
    y_pred = []
    y_score = []


    for imgs, labels in tqdm(data_loader, desc="Valid/Test", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true.append(labels.cpu())
        y_pred.append(preds.cpu())
        y_score.append(probs.cpu().numpy())

  
    if len(y_true) == 0:
        raise ValueError("DataLoader vazio — nenhuma amostra encontrada.")

  
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    y_score = np.concatenate(y_score)

   
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

  
    try:
        classes_present = np.unique(y_true)

        if len(classes_present) <= 1:
            auc = 0.5

        elif len(classes_present) < num_classes:
            auc = roc_auc_score(
                y_true,
                y_score[:, classes_present],
                multi_class="ovr",
                average="macro"
            )

        else:
            auc = roc_auc_score(
                y_true,
                y_score,
                multi_class="ovr",
                average="macro"
            )

    except ValueError:
        auc = 0.5

    
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes))
    )


    sens_list = []
    spec_list = []

    total = np.sum(cm)

    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total - (tp + fn + fp)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        sens_list.append(sens)
        spec_list.append(spec)

  
    return {
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "kappa": kappa,
        "sens": float(np.mean(sens_list)),
        "spec": float(np.mean(spec_list)),
        "sens_per_class": sens_list,
        "spec_per_class": spec_list,
        "cm": cm,
        "y_true": y_true,
        "y_pred": y_pred
    }
