import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import NoduleDataset
from EfficientNet import efficientnet_b0
from train_utils import evaluate_slice_level


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none",
            weight=self.weight,
        )
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


def parse_args():
    # project_root = Path(__file__).resolve().parents[2]
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Compara EfficientNet-B0 em multiplas resolucoes sem alterar o script original."
    )
    parser.add_argument("--root-dir", type=str, default=str(project_root / "dataset"))
    parser.add_argument("--output-dir", type=str, default=str(project_root / "classification_CPCNP" / "runs" / "efficientnet_compare"))
    parser.add_argument("--image-sizes", type=str, default="225,160,128")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--lr-decay-patience", type=int, default=5)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def build_loaders(root_dir: str, image_size: int, batch_size: int, num_workers: int):
    train_transform = make_transforms(image_size)
    eval_transform = make_transforms(image_size)

    train_set = NoduleDataset(root_dir, "train", train_transform)
    val_set = NoduleDataset(root_dir, "val", eval_transform)
    test_set = NoduleDataset(root_dir, "test", eval_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_set, val_set, test_set, train_loader, val_loader, test_loader


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def save_curves(history: dict, run_dir: Path):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.title("Loss por Epoca")
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs_range, history["val_auc"], label="Val AUC")
    plt.plot(epochs_range, history["val_f1"], label="Val F1")
    plt.plot(epochs_range, history["val_score"], label="Val Score")
    plt.title("Metricas de Validacao")
    plt.xlabel("Epocas")
    plt.ylabel("Score")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "metrics.png")
    plt.close()

    plt.figure()
    plt.plot(epochs_range, history["lr"], label="LR")
    plt.title("Learning Rate")
    plt.xlabel("Epocas")
    plt.ylabel("LR")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "lr.png")
    plt.close()


def write_history_csv(history: dict, run_dir: Path):
    csv_path = run_dir / "history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_auc", "val_f1", "val_acc", "val_score", "lr"])
        for idx in range(len(history["train_loss"])):
            writer.writerow([
                idx + 1,
                history["train_loss"][idx],
                history["val_auc"][idx],
                history["val_f1"][idx],
                history["val_acc"][idx],
                history["val_score"][idx],
                history["lr"][idx],
            ])


def save_metrics_txt(run_dir: Path, params: dict, train_metrics: dict, val_metrics: dict, test_metrics: dict, best_epoch: int):
    text = []
    text.append("===== PARAMETROS =====")
    for key, value in params.items():
        text.append(f"{key}: {value}")

    def add_block(name: str, metrics: dict):
        text.append("")
        text.append(f"===== {name} =====")
        text.append(f"AUC: {metrics['auc']:.4f}")
        text.append(f"ACC: {metrics['acc']:.4f}")
        text.append(f"F1: {metrics['f1']:.4f}")
        text.append(f"Kappa: {metrics['kappa']:.4f}")
        text.append(f"Sensibilidade: {metrics['sens']:.4f}")
        text.append(f"Especificidade: {metrics['spec']:.4f}")
        text.append(f"Matriz de Confusao:\n{metrics['cm']}")

    text.append(f"best_epoch: {best_epoch}")
    add_block("TRAIN_FINAL", train_metrics)
    add_block("VAL_FINAL", val_metrics)
    add_block("TEST_FINAL", test_metrics)

    (run_dir / "metrics.txt").write_text("\n".join(text) + "\n", encoding="utf-8")


def train_one_resolution(args, device, image_size: int, summary_rows: list[dict]):
    run_dir = Path(args.output_dir) / f"efficientnet_b0_{image_size}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 90}")
    print(f"Treinando EfficientNet-B0 | input={image_size}x{image_size}")
    print(f"Saida: {run_dir}")
    print(f"{'=' * 90}")

    train_set, val_set, test_set, train_loader, val_loader, test_loader = build_loaders(
        args.root_dir,
        image_size,
        args.batch_size,
        args.num_workers,
    )

    num_classes = len(train_set.class_to_idx)
    print("Distribuicao TRAIN:", Counter(train_set.labels))
    print("Distribuicao VAL:  ", Counter(val_set.labels))
    print("Distribuicao TEST: ", Counter(test_set.labels))

    model = efficientnet_b0(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience,
        min_lr=args.min_lr,
    )
    criterion = FocalLoss(gamma=2.0)

    best_score = float("-inf")
    best_epoch = 0
    best_model_path = run_dir / "best_model.pth"
    log_path = run_dir / "train.log"

    history = {
        "train_loss": [],
        "val_auc": [],
        "val_f1": [],
        "val_acc": [],
        "val_score": [],
        "lr": [],
    }

    epochs_without_improvement = 0

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("===== INICIO DO TREINAMENTO =====\n")
        log_file.write(f"image_size: {image_size}\n")
        log_file.write(f"epochs: {args.epochs}\n")
        log_file.write(f"batch_size: {args.batch_size}\n")
        log_file.write(f"lr: {args.lr}\n")
        log_file.write(f"weight_decay: {args.weight_decay}\n")
        log_file.write(f"early_stop_patience: {args.early_stop_patience}\n")
        log_file.write(f"lr_decay_patience: {args.lr_decay_patience}\n")
        log_file.write(f"lr_decay_factor: {args.lr_decay_factor}\n\n")

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0

            pbar = tqdm(
                train_loader,
                desc=f"[{image_size}] Epoch {epoch + 1}/{args.epochs}",
                ncols=120,
            )

            for imgs, labels in pbar:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{get_current_lr(optimizer):.2e}")

            epoch_loss = running_loss / max(1, len(train_loader))
            val_metrics = evaluate_slice_level(model, val_loader, device, num_classes=num_classes)
            val_score = 0.7 * val_metrics["auc"] + 0.3 * val_metrics["f1"]
            scheduler.step(val_score)

            history["train_loss"].append(epoch_loss)
            history["val_auc"].append(val_metrics["auc"])
            history["val_f1"].append(val_metrics["f1"])
            history["val_acc"].append(val_metrics["acc"])
            history["val_score"].append(val_score)
            history["lr"].append(get_current_lr(optimizer))

            epoch_message = (
                f"Epoch {epoch + 1:03d} | "
                f"Loss {epoch_loss:.4f} | "
                f"AUC {val_metrics['auc']:.4f} | "
                f"F1 {val_metrics['f1']:.4f} | "
                f"ACC {val_metrics['acc']:.4f} | "
                f"Score {val_score:.4f} | "
                f"LR {get_current_lr(optimizer):.6f}"
            )
            print(epoch_message)
            log_file.write(epoch_message + "\n")

            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
                best_message = f"Novo melhor modelo salvo em epoch {best_epoch} | score={best_score:.4f}"
                print(best_message)
                log_file.write(best_message + "\n")
            else:
                epochs_without_improvement += 1
                stop_message = (
                    f"Sem melhora por {epochs_without_improvement} epoca(s) "
                    f"| paciencia={args.early_stop_patience}"
                )
                print(stop_message)
                log_file.write(stop_message + "\n")

            if epochs_without_improvement >= args.early_stop_patience:
                early_stop_message = (
                    f"Early stopping acionado na epoca {epoch + 1} "
                    f"(melhor epoca: {best_epoch}, melhor score: {best_score:.4f})"
                )
                print(early_stop_message)
                log_file.write(early_stop_message + "\n")
                break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    train_metrics = evaluate_slice_level(model, train_loader, device, num_classes=num_classes)
    val_metrics = evaluate_slice_level(model, val_loader, device, num_classes=num_classes)
    test_metrics = evaluate_slice_level(model, test_loader, device, num_classes=num_classes)

    params = {
        "model": "efficientnet_b0",
        "image_size": image_size,
        "epochs_configured": args.epochs,
        "epochs_executed": len(history["train_loss"]),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "early_stop_patience": args.early_stop_patience,
        "lr_decay_patience": args.lr_decay_patience,
        "lr_decay_factor": args.lr_decay_factor,
        "best_score": f"{best_score:.4f}",
    }
    save_metrics_txt(run_dir, params, train_metrics, val_metrics, test_metrics, best_epoch)
    write_history_csv(history, run_dir)
    save_curves(history, run_dir)

    summary_rows.append({
        "image_size": image_size,
        "best_epoch": best_epoch,
        "epochs_executed": len(history["train_loss"]),
        "best_val_score": f"{best_score:.6f}",
        "val_auc": f"{val_metrics['auc']:.6f}",
        "val_f1": f"{val_metrics['f1']:.6f}",
        "val_acc": f"{val_metrics['acc']:.6f}",
        "test_auc": f"{test_metrics['auc']:.6f}",
        "test_f1": f"{test_metrics['f1']:.6f}",
        "test_acc": f"{test_metrics['acc']:.6f}",
        "test_kappa": f"{test_metrics['kappa']:.6f}",
        "test_sens": f"{test_metrics['sens']:.6f}",
        "test_spec": f"{test_metrics['spec']:.6f}",
        "run_dir": str(run_dir),
    })


def write_summary(summary_rows: list[dict], output_dir: Path):
    summary_path = output_dir / "comparison_summary.csv"
    if not summary_rows:
        return

    fieldnames = list(summary_rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    txt_lines = ["===== RESUMO FINAL ====="]
    for row in sorted(summary_rows, key=lambda item: float(item["test_auc"]), reverse=True):
        txt_lines.append(
            " | ".join([
                f"img={row['image_size']}",
                f"best_epoch={row['best_epoch']}",
                f"val_score={row['best_val_score']}",
                f"test_auc={row['test_auc']}",
                f"test_f1={row['test_f1']}",
                f"test_acc={row['test_acc']}",
                f"run_dir={row['run_dir']}",
            ])
        )
    (output_dir / "comparison_summary.txt").write_text("\n".join(txt_lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Treinando em: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    image_sizes = [int(size.strip()) for size in args.image_sizes.split(",") if size.strip()]
    summary_rows = []

    for image_size in image_sizes:
        train_one_resolution(args, device, image_size, summary_rows)

    write_summary(summary_rows, output_dir)

    print("\nComparacao concluida.")
    print(f"Resumo CSV: {output_dir / 'comparison_summary.csv'}")
    print(f"Resumo TXT: {output_dir / 'comparison_summary.txt'}")


if __name__ == "__main__":
    main()
