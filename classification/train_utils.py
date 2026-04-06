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
def evaluate_slice_level(model, data_loader, device, num_classes=2):

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

    # ======================
    # DEBUG (opcional)
    # ======================
    # print("Classes presentes:", np.unique(y_true))
    # print("Shape y_score:", y_score.shape)

    # ======================
    # MÉTRICAS
    # ======================
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    # ======================
    # ✅ AUC CORRETA (BINÁRIA)
    # ======================
    try:
        if len(np.unique(y_true)) <= 1:
            auc = 0.5
        else:
            # probabilidade da classe positiva (classe 1)
            auc = roc_auc_score(y_true, y_score[:, 1])
    except ValueError:
        auc = 0.5

    # ======================
    # MATRIZ DE CONFUSÃO
    # ======================
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes))
    )

    # ======================
    # SENS / SPEC
    # ======================
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