
# import torch.nn as nn
# from torchvision import transforms, datasets
# import json
# import argparse
# import os
# import torch.optim as optim
# import torch
# from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sn
# from collections import Counter
# import matplotlib
# matplotlib.use('Agg')

# from Res2Net import Res2Net
# from train_utils import get_params_groups, create_lr_scheduler, train_one_epoch, evaluate
# import gc
# gc.collect()
# torch.cuda.empty_cache()

# from sklearn.metrics import confusion_matrix, classification_report

# def main(args):
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     tb_writer = SummaryWriter(log_dir="classfication_result/tensorboard_logs")

#     # -------------------------
#     # Transforms
#     # -------------------------
#     data_transform = {
#         "train": transforms.Compose([
#             transforms.Resize((64, 64)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ]),
#         "val": transforms.Compose([
#             transforms.Resize((64, 64)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ]),
#         "test": transforms.Compose([
#             transforms.Resize((64, 64)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])
#     }


#     # -------------------------
#     # Diretórios
#     # -------------------------
#     train_dir = os.path.join("dataset/train")
#     val_dir = os.path.join("dataset/val")
#     test_dir = os.path.join("dataset/test")

#     for d in [train_dir, val_dir, test_dir]:
#         if not os.path.isdir(d):
#             raise FileNotFoundError(f"Diretório não encontrado: {d}")

#     # -------------------------
#     # Datasets
#     # -------------------------
#     train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform["train"])
#     val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transform["val"])
#     test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transform["test"])

#     batch_size = args.batch_size
#     num_workers = 0
#     print(f'Using {num_workers} dataloader workers every process')

#     print("Train class distribution:", Counter([s[1] for s in train_dataset.samples]))
#     print("Val class distribution:", Counter([s[1] for s in val_dataset.samples]))
#     print("Test class distribution:", Counter([s[1] for s in test_dataset.samples]))

#     # -------------------------
#     # DataLoaders
#     # -------------------------
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
#     )
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
#     )
#     test_loader = torch.utils.data.DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
#     )

#     # -------------------------
#     # Modelo
#     # -------------------------
#     model = Res2Net(layers=[3, 4, 6, 3], num_classes=args.num_classes, width=16, scales=4, groups=1)
#     model.to(device)

#     pg = get_params_groups(model, weight_decay=args.wd)
#     optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
#     lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=10)

#     train_labels = [s[1] for s in train_dataset.samples]
#     class_counts = Counter(train_labels)
#     weights = [1.0 / class_counts[i] for i in range(args.num_classes)]
#     weights = torch.tensor(weights, dtype=torch.float).to(device)
#     criterion = nn.CrossEntropyLoss(weight=weights)

#     best_acc = 0.
#     best_train_acc = 0.
#     best_kappa = 0.

#     history = []

#     # -------------------------
#     # Loop de treinamento
#     # -------------------------
#     for epoch in range(args.epochs):

#         train_loss, train_acc, train_kappa1, train_truee, train_predd = train_one_epoch(
#             model=model,
#             optimizer=optimizer,
#             data_loader=train_loader,
#             device=device,
#             epoch=epoch,
#             lr_scheduler=lr_scheduler,
#             criterion=criterion
#         )
#         print(f"train_kappa= {train_kappa1:.4f}")

#         if train_acc > best_train_acc:
#             conf_matrix = confusion_matrix(train_truee, train_predd)
#             df = pd.DataFrame({'True label': train_truee, 'Predict label': train_predd})
#             confmtpd = pd.crosstab(df['True label'], df['Predict label'], dropna=False)
#             print(f"Confusion matrix with pandas:\n{confmtpd}")
#             plt.figure()
#             sn.heatmap(confmtpd, annot=True, cmap='Greens', fmt='d')
#             plt.tight_layout()
#             plt.savefig(f'classfication_result/train_confusion_epoch{epoch}.png')
#             plt.close()
#             best_train_acc = train_acc

#         val_loss, val_acc, val_kappa1, val_truee, val_predd = evaluate(
#             model=model,
#             data_loader=val_loader,
#             device=device,
#             epoch=epoch,
#             criterion=criterion
#         )
#         print(f"val_kappa= {val_kappa1:.4f}")

#         tb_writer.add_scalar("train_loss", train_loss, epoch)
#         tb_writer.add_scalar("train_acc", train_acc, epoch)
#         tb_writer.add_scalar("val_loss", val_loss, epoch)
#         tb_writer.add_scalar("val_acc", val_acc, epoch)
#         tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
#         tb_writer.add_scalar("train_kappa", train_kappa1, epoch)
#         tb_writer.add_scalar("val_kappa", val_kappa1, epoch)

#         history.append({
#             "epoch": epoch,
#             "train_loss": train_loss,
#             "val_loss": val_loss,
#             "train_acc": train_acc,
#             "val_acc": val_acc,
#             "train_kappa": train_kappa1,
#             "val_kappa": val_kappa1
#         })

#         checkpoint_path = f"classfication_result/epoch_{epoch:03d}_kappa{val_kappa1:.3f}.pth"
#         torch.save(model.state_dict(), checkpoint_path)
#         print(f"Checkpoint salvo: {checkpoint_path}")

#         if best_kappa < val_kappa1:
#             save_path = "classfication_result/cancer_res2net_best.pth"
#             torch.save(model.state_dict(), save_path)
#             print(f"✅ Novo melhor modelo salvo em {save_path}")

#             val_df = pd.DataFrame({'True label': val_truee, 'Predict label': val_predd})
#             val_confmtpd = pd.crosstab(val_df['True label'], val_df['Predict label'], dropna=False)
#             print(f"Confusion matrix with pandas:\n{val_confmtpd}")
#             plt.figure()
#             sn.heatmap(val_confmtpd, annot=True, cmap='Greens', fmt='d')
#             plt.tight_layout()
#             plt.savefig('classfication_result/val_confusion_matrix.png')
#             plt.close()
#             best_kappa = val_kappa1

#         print(f"best_train_acc = {best_train_acc:.4f}")
#         print(f"best_val_acc = {val_acc:.4f}")
#         print(f"best_val_kappa = {best_kappa:.4f}")

#         pd.DataFrame(history).to_csv("classfication_result/training_log.csv", index=False)

#     # -------------------------
#     # AVALIAÇÃO FINAL (TESTE)
#     # -------------------------
#     print("\n===== FINAL TEST EVALUATION =====")
#     model.load_state_dict(torch.load("classfication_result/cancer_res2net_best.pth", map_location=device))
#     model.eval()

#     test_loss, test_acc, test_kappa, test_true, test_pred = evaluate(
#         model=model,
#         data_loader=test_loader,
#         device=device,
#         epoch=args.epochs,
#         criterion=criterion
#     )

#     print(f"Test Accuracy: {test_acc:.4f}, Test Kappa: {test_kappa:.4f}")
#     print(classification_report(test_true, test_pred, target_names=test_dataset.classes))

#     test_df = pd.DataFrame({'True label': test_true, 'Predict label': test_pred})
#     test_confmtpd = pd.crosstab(test_df['True label'], test_df['Predict label'], dropna=False)
#     plt.figure()
#     sn.heatmap(test_confmtpd, annot=True, cmap='Blues', fmt='d')
#     plt.tight_layout()
#     plt.savefig('classfication_result/test_confusion_matrix.png')
#     plt.close()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num_classes', type=int, default=3)
#     parser.add_argument('--epochs', type=int, default=200)
#     parser.add_argument('--batch-size', type=int, default=4)
#     parser.add_argument('--lr', type=float, default=2e-4)
#     parser.add_argument('--wd', type=float, default=0.05)
#     parser.add_argument('--data-path', type=str, default="dataset")
#     parser.add_argument('--weights', type=str, default='', help='initial weights path')
#     parser.add_argument('--freeze-layers', type=bool, default=False)
#     parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

#     opt = parser.parse_args()
#     main(opt)
import os
import gc
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms, datasets

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, recall_score
from sklearn.preprocessing import label_binarize
from ConvNext import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge

# from resnet50 import resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d
from train_utils import get_params_groups, create_lr_scheduler, train_one_epoch, evaluate

gc.collect()
torch.cuda.empty_cache()

def compute_metrics(model, data_loader, device, num_classes):
    """
    Executa avaliação completa e retorna:
    - y_true, y_pred, y_score
    - métricas completas incluindo:
        • sensibilidade por classe
        • especificidade por classe
        • sensibilidade total (macro)
        • especificidade total (macro)
        • AUC por classe
        • AUC macro
    """
    model.eval()
    y_true = []
    y_pred = []
    y_score = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            probs = softmax(outputs)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_score.extend(probs.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    # -------------------------
    # MATRIZ DE CONFUSÃO
    # -------------------------
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # -------------------------
    # SENS / SPEC POR CLASSE
    # -------------------------
    sens = []
    spec = []

    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        sens.append(sensitivity)
        spec.append(specificity)

    # -------------------------
    # ⭐ SENSIBILIDADE TOTAL (MACRO)
    # -------------------------
    macro_sensitivity = np.mean(sens)
    macro_specificity = np.mean(spec)

    # -------------------------
    # AUC (one-vs-rest)
    # -------------------------
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

        aucs = []
        for i in range(num_classes):
            if np.unique(y_true_bin[:, i]).size == 1:
                aucs.append(np.nan)
            else:
                aucs.append(roc_auc_score(y_true_bin[:, i], y_score[:, i]))

        aucs_arr = np.array(aucs, dtype=np.float64)
        macro_auc = np.nanmean(aucs_arr)

    except Exception:
        aucs = [np.nan] * num_classes
        macro_auc = np.nan

    # -------------------------
    # RETORNO FINAL
    # -------------------------
    metrics = {
        "confusion_matrix": cm,
        "sensitivity_per_class": sens,
        "specificity_per_class": spec,
        "macro_sensitivity": macro_sensitivity,      # ⭐ AQUI ESTÁ SUA SENS TOTAL
        "macro_specificity": macro_specificity,
        "auc_per_class": aucs,
        "macro_auc": macro_auc,
    }

    return y_true.tolist(), y_pred.tolist(), y_score, metrics
def compute_metrics(model, data_loader, device, num_classes):
    """
    Executa avaliação completa e retorna:
    - y_true, y_pred, y_score
    - métricas completas incluindo:
        • sensibilidade por classe
        • especificidade por classe
        • sensibilidade total (macro)
        • especificidade total (macro)
        • AUC por classe
        • AUC macro
    """
    model.eval()
    y_true = []
    y_pred = []
    y_score = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            probs = softmax(outputs)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_score.extend(probs.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    # -------------------------
    # MATRIZ DE CONFUSÃO
    # -------------------------
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # -------------------------
    # SENS / SPEC POR CLASSE
    # -------------------------
    sens = []
    spec = []

    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        sens.append(sensitivity)
        spec.append(specificity)

    # -------------------------
    # ⭐ SENSIBILIDADE TOTAL (MACRO)
    # -------------------------
    macro_sensitivity = np.mean(sens)
    macro_specificity = np.mean(spec)

    # -------------------------
    # AUC (one-vs-rest)
    # -------------------------
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

        aucs = []
        for i in range(num_classes):
            if np.unique(y_true_bin[:, i]).size == 1:
                aucs.append(np.nan)
            else:
                aucs.append(roc_auc_score(y_true_bin[:, i], y_score[:, i]))

        aucs_arr = np.array(aucs, dtype=np.float64)
        macro_auc = np.nanmean(aucs_arr)

    except Exception:
        aucs = [np.nan] * num_classes
        macro_auc = np.nan

    # -------------------------
    # RETORNO FINAL
    # -------------------------
    metrics = {
        "confusion_matrix": cm,
        "sensitivity_per_class": sens,
        "specificity_per_class": spec,
        "macro_sensitivity": macro_sensitivity,      # ⭐ AQUI ESTÁ SUA SENS TOTAL
        "macro_specificity": macro_specificity,
        "auc_per_class": aucs,
        "macro_auc": macro_auc,
    }

    return y_true.tolist(), y_pred.tolist(), y_score, metrics

def make_train_sampler(train_dataset, num_classes):
    """
    Cria um WeightedRandomSampler que irá oversample até igualar à maior classe.
    Retorna sampler e num_samples por epoch (num_classes * max_count).
    """
    # Conta por classe
    labels = [s[1] for s in train_dataset.samples]
    class_counts = Counter(labels)
    max_count = max(class_counts.values())

    # peso por classe inverso à frequência -> classes menores ganham mais prob
    class_weights = {cls: max_count / count for cls, count in class_counts.items()}

    # sample weight para cada amostra
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    num_samples = max_count * num_classes  # cada epoch terá esse número de amostras (equilíbrio)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)
    return sampler, num_samples, class_counts

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(':')[1] if 'cuda' in args.device else ''
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tb_writer = SummaryWriter(log_dir="classfication_result/tensorboard_logs")

    # -------------------------
    # Transforms (adicionando augmentações no TRAIN)
    # -------------------------
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "test": transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    # -------------------------
    # Diretórios
    # -------------------------
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    test_dir = os.path.join(args.data_path, "test")

    for d in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Diretório não encontrado: {d}")

    # -------------------------
    # Datasets
    # -------------------------
    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transform["val"])
    test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transform["test"])

    batch_size = args.batch_size
    num_workers = args.num_workers
    print(f'Using {num_workers} dataloader workers every process')

    # sampler para balancear por epoch
    sampler, num_samples_per_epoch, class_counts = make_train_sampler(train_dataset, args.num_classes)
    print("Class distribution (original):", class_counts)
    print(f"Sampler num_samples per epoch: {num_samples_per_epoch} (num_classes * max_count)")

    # -------------------------
    # DataLoaders (usando sampler para train)
    # -------------------------
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # -------------------------
    # Modelo
    # -------------------------
    
    model = convnext_tiny(num_classes=args.num_classes).to(device)
    # model = resnet34(num_classes=args.num_classes, include_top=True).to(device)
    model.to(device)

    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, int(np.ceil(num_samples_per_epoch / batch_size)), args.epochs, warmup=True, warmup_epochs=10)

    # -------------------------
    # Pesos da loss: agora proporcional a max_count/class_count (maior peso para minoritárias)
    # -------------------------
    max_count = max(class_counts.values())
    weights = [max_count / class_counts[i] if class_counts[i] > 0 else 0.0 for i in range(args.num_classes)]
    weights = torch.tensor(weights, dtype=torch.float).to(device)
    print("Loss weights per class:", weights.cpu().numpy())
    criterion = nn.CrossEntropyLoss(weight=weights) 

    best_acc = 0.
    best_train_acc = 0.
    best_kappa = 0.

    history = []

    # -------------------------
    # Loop de treinamento
    # -------------------------
    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")
        train_loss, train_acc, train_kappa1, train_truee, train_predd = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
            criterion=criterion
        )
        print(f"train_kappa= {train_kappa1:.4f}")

        # CALCULAR métricas completas usando compute_metrics (probabilidades) no train (pode consumir mais tempo)
        tr_true, tr_pred, tr_score, tr_metrics = compute_metrics(model, train_loader, device, args.num_classes)
        print("Train sensitivity per class:", tr_metrics["sensitivity_per_class"])
        print("Train specificity per class:", tr_metrics["specificity_per_class"])
        print("Train AUC per class:", tr_metrics["auc_per_class"], "macro_auc:", tr_metrics["macro_auc"])

        # salvar confusion matrix do train quando melhora
        if train_acc > best_train_acc:
            cm = tr_metrics["confusion_matrix"]
            df = pd.DataFrame(cm, index=train_dataset.classes, columns=train_dataset.classes)
            plt.figure()
            sn.heatmap(df, annot=True, fmt='d', cmap='Greens')
            plt.tight_layout()
            plt.savefig(f'classfication_result/train_confusion_epoch{epoch}.png')
            plt.close()
            best_train_acc = train_acc

        val_loss, val_acc, val_kappa1, val_truee, val_predd = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch,
            criterion=criterion
        )
        print(f"val_kappa= {val_kappa1:.4f}")

        # métricas completas no val
        v_true, v_pred, v_score, v_metrics = compute_metrics(model, val_loader, device, args.num_classes)
        print("Val sensitivity per class:", v_metrics["sensitivity_per_class"])
        print("Val specificity per class:", v_metrics["specificity_per_class"])
        print("Val AUC per class:", v_metrics["auc_per_class"], "macro_auc:", v_metrics["macro_auc"])

        # log no tensorboard
        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_acc", train_acc, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("val_acc", val_acc, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar("train_kappa", train_kappa1, epoch)
        tb_writer.add_scalar("val_kappa", val_kappa1, epoch)
        # log sens/spec/auc (macro)
        tb_writer.add_scalar("train_macro_sensitivity", np.mean(tr_metrics["sensitivity_per_class"]), epoch)
        tb_writer.add_scalar("train_macro_specificity", np.mean(tr_metrics["specificity_per_class"]), epoch)
        if not np.isnan(tr_metrics["macro_auc"]):
            tb_writer.add_scalar("train_macro_auc", tr_metrics["macro_auc"], epoch)
        tb_writer.add_scalar("val_macro_sensitivity", np.mean(v_metrics["sensitivity_per_class"]), epoch)
        tb_writer.add_scalar("val_macro_specificity", np.mean(v_metrics["specificity_per_class"]), epoch)
        if not np.isnan(v_metrics["macro_auc"]):
            tb_writer.add_scalar("val_macro_auc", v_metrics["macro_auc"], epoch)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_kappa": train_kappa1,
            "val_kappa": val_kappa1,
            "train_macro_sensitivity": np.mean(tr_metrics["sensitivity_per_class"]),
            "train_macro_specificity": np.mean(tr_metrics["specificity_per_class"]),
            "train_macro_auc": tr_metrics["macro_auc"],
            "val_macro_sensitivity": np.mean(v_metrics["sensitivity_per_class"]),
            "val_macro_specificity": np.mean(v_metrics["specificity_per_class"]),
            "val_macro_auc": v_metrics["macro_auc"],
        })

        checkpoint_path = f"classfication_result/epoch_{epoch:03d}_kappa{val_kappa1:.3f}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint salvo: {checkpoint_path}")

        if best_kappa < val_kappa1:
            save_path = "classfication_result/cancer_Res2Net_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✅ Novo melhor modelo salvo em {save_path}")

            # salvar conf matrix da validação
            cm = v_metrics["confusion_matrix"]
            df = pd.DataFrame(cm, index=val_dataset.classes, columns=val_dataset.classes)
            plt.figure()
            sn.heatmap(df, annot=True, fmt='d', cmap='Greens')
            plt.tight_layout()
            plt.savefig('classfication_result/val_confusion_matrix.png')
            plt.close()
            best_kappa = val_kappa1

        print(f"best_train_acc = {best_train_acc:.4f}")
        print(f"val_acc = {val_acc:.4f}")
        print(f"best_val_kappa = {best_kappa:.4f}")

        pd.DataFrame(history).to_csv("classfication_result/training_log.csv", index=False)

    # -------------------------
    # AVALIAÇÃO FINAL (TESTE)
    # -------------------------
    print("\n===== FINAL TEST EVALUATION =====")
    model.load_state_dict(torch.load("classfication_result/cancer_Res2Net_best.pth", map_location=device))
    model.to(device)
    model.eval()

    # avaliar com evaluate (se quiser manter) e também com compute_metrics para AUC/sens/spec
    test_loss, test_acc, test_kappa, test_true, test_pred = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        epoch=args.epochs,
        criterion=criterion
    )
    print(f"Test Accuracy: {test_acc:.4f}, Test Kappa: {test_kappa:.4f}")
    print(classification_report(test_true, test_pred, target_names=test_dataset.classes))

    t_true, t_pred, t_score, t_metrics = compute_metrics(model, test_loader, device, args.num_classes)
    print("Test sensitivity per class:", t_metrics["sensitivity_per_class"])
    print("Test specificity per class:", t_metrics["specificity_per_class"])
    print("Test AUC per class:", t_metrics["auc_per_class"], "macro_auc:", t_metrics["macro_auc"])
    print("Sensibilidade total:", t_metrics["macro_sensitivity"])
    print("Specificity por total:", t_metrics["macro_specificity"])

    # salvar matriz de confusão do teste
    cm = t_metrics["confusion_matrix"]
    df = pd.DataFrame(cm, index=test_dataset.classes, columns=test_dataset.classes)
    plt.figure()
    sn.heatmap(df, annot=True, fmt='d', cmap='Blues')
    plt.tight_layout()
    plt.savefig('classfication_result/test_confusion_matrix.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--wd', type=float, default=0.05)
    parser.add_argument('--data-path', type=str, default="dataset")
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. cpu or cuda:0)')
    opt = parser.parse_args()
    main(opt)