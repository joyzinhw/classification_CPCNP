import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import itertools
import numpy as np

from dataset import NoduleDataset
from utils_grid import evaluate_slice_level
from EfficientNet import efficientnet_b0

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=1.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            logits,
            targets,
            reduction='none',
            weight=self.weight
        )
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

def train_once(args, config):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    val_transform = train_transform

    train_set = NoduleDataset(args.root_dir, "train", train_transform)
    val_set   = NoduleDataset(args.root_dir, "val", val_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = efficientnet_b0(num_classes=2).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    criterion = FocalLoss(gamma=config["gamma"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    best_score = 0
    patience = 7
    counter = 0

    for epoch in range(args.epochs):

        model.train()

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_m = evaluate_slice_level(model, val_loader, device)
        score = 0.7 * val_m["auc"] + 0.3 * val_m["f1"]

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Score {score:.4f} | "
            f"Best {best_score:.4f} | "
            f"Patience {counter}/{patience}"
        )

        # Early stopping
        if score > best_score:
            best_score = score
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f" Early stopping na época {epoch+1}")
            break

    return best_score

def main(args):

    grid = {
        "lr": [3e-4, 5e-4],
        "weight_decay": [1e-4, 1e-3],
        "batch_size": [16, 32],
        "gamma": [1.0, 2.0],
    }

    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"\n🔎 Total de combinações: {len(combinations)}")

    log = open("grid_search_results.txt", "w")

    best_global_score = 0
    best_config = None

    for i, config in enumerate(combinations):

        print(f"\n🚀 Testando config {i+1}/{len(combinations)}")
        print(config)

     
        score = train_once(args, config)

        print(f"  Score final: {score:.4f}")

        log.write(f"{config} | score={score:.4f}\n")

        if score > best_global_score:
            best_global_score = score
            best_config = config

    log.write("\n===== MELHOR CONFIG =====\n")
    log.write(f"{best_config} | score={best_global_score:.4f}\n")

    log.close()

    print("\n MELHOR CONFIG:")
    print(best_config)
    print(f"Score: {best_global_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()
    main(args)