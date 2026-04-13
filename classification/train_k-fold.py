import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import Counter

from dataset import NoduleDataset
from train_utils import evaluate_slice_level
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
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()



def log_print(log_file, text):
    print(text)
    if log_file and not log_file.closed:
        log_file.write(text + "\n")
        log_file.flush()



def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Treinando em: {device}")

    transform = transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    model = efficientnet_b0(num_classes=2).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3 
    )

    criterion = FocalLoss(gamma=1.0)

    with open("log.txt", "w") as log_file:

        log_print(log_file, "===== INICIO TREINAMENTO =====")


        for epoch in range(args.epochs):

            fold = epoch % 5  

            fold_dir = os.path.join(args.root_dir, f"fold_{fold}")

            log_print(log_file, "\n" + "="*60)
            log_print(log_file, f"🚀 Epoch {epoch+1} usando fold_{fold}")
            log_print(log_file, "="*60)

            train_set = NoduleDataset(fold_dir, "train", transform)
            val_set   = NoduleDataset(fold_dir, "val", transform)

            log_print(log_file, f"TRAIN: {Counter(train_set.labels)}")
            log_print(log_file, f"VAL:   {Counter(val_set.labels)}")

            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            val_loader = DataLoader(
                val_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )


            model.train()
            running_loss = 0.0

            for imgs, labels in tqdm(train_loader):

                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            log_print(log_file, f"Loss: {running_loss / len(train_loader):.4f}")

            model.eval()
            metrics = evaluate_slice_level(model, val_loader, device)

            log_print(
                log_file,
                f"VAL | AUC: {metrics['auc']:.4f} | "
                f"ACC: {metrics['acc']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )


        log_print(log_file, "\n" + "="*60)
        log_print(log_file, "🧪 TESTE FINAL")
        log_print(log_file, "="*60)

        test_set = NoduleDataset(args.root_dir, "test", transform)

        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        test_metrics = evaluate_slice_level(model, test_loader, device)

        log_print(log_file, f"AUC: {test_metrics['auc']:.4f}")
        log_print(log_file, f"ACC: {test_metrics['acc']:.4f}")
        log_print(log_file, f"F1:  {test_metrics['f1']:.4f}")
        log_print(log_file, f"Kappa: {test_metrics['kappa']:.4f}")
        log_print(log_file, f"Sens: {test_metrics['sens']:.4f}")
        log_print(log_file, f"Spec: {test_metrics['spec']:.4f}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="dataset_kfold")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()
    main(args)

