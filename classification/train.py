import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

from dataset import NoduleDataset
from train_utils import evaluate_slice_level
# from ConvNext import convnext_tiny, convneaxt_small, convnext_base
from Res2Net import Res2Net
from ResNext import resNeXt50_32x4d_SE
from EfficientNet import efficientnet_b0
from resnet50 import resnet50



class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
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


def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Treinando em: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


    train_transform = transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


    train_set = NoduleDataset(args.root_dir, "train", train_transform)
    val_set   = NoduleDataset(args.root_dir, "val", val_test_transform)
    test_set  = NoduleDataset(args.root_dir, "test", val_test_transform)

    print("📊 Distribuição TRAIN:", Counter(train_set.labels))
    print("📊 Distribuição VAL:  ", Counter(val_set.labels))
    print("📊 Distribuição TEST: ", Counter(test_set.labels))


   
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # for imgs, labels in train_loader:
    #     print("Shape:", imgs.shape)
    #     print("Min:", imgs.min().item(), "Max:", imgs.max().item())
    #     break
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )



#     model = Res2Net(
#     layers=[3, 4, 6, 3],
#     num_classes=2,
#     width=16,
#     scales=4,
#     groups=1
# ).to(device)
    model = efficientnet_b0(num_classes=2).to(device)
    # model = convnext_tiny(num_classes=2).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3
    )


   #pesos para ativar para testar

    # class_counts = torch.tensor([2460, 1204
    # ], dtype=torch.float32)

    # weights = class_counts.sum() / class_counts   # inverso forte
    # weights = weights / weights.mean()            # normaliza

    # criterion = torch.nn.CrossEntropyLoss(
    #     weight=weights.to(device),
    #     label_smoothing=0.05
    # )

    criterion = FocalLoss(gamma=2.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    best_score = 0.0
    train_losses = []
    val_aucs = []
    val_f1s = []
    lrs = []

    log_file = open("treinamento_B0.txt", "w")
    log_file.write("===== INICIO DO TREINAMENTO =====\n")
    log_file.write(f"Epochs: {args.epochs}\n")
    log_file.write(f"Batch size: {args.batch_size}\n")
    log_file.write(f"Learning rate inicial: {args.lr}\n\n")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{args.epochs}]",
            ncols=120
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
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)

        val_m = evaluate_slice_level(model, val_loader, device)

        train_losses.append(epoch_loss)
        val_aucs.append(val_m["auc"])
        val_f1s.append(val_m["f1"])

        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)

        score = 0.7 * val_m["auc"] + 0.3 * val_m["f1"]

        scheduler.step()

        print(
            f"Epoch {epoch+1:03d} | "
            f"Loss {epoch_loss:.4f} | "
            f"AUC {val_m['auc']:.4f} | "
            f"F1 {val_m['f1']:.4f} | "
            f"ACC {val_m['acc']:.4f} | "
            f"LR {current_lr:.6f}"
        )

        log_file.write(
            f"Epoch {epoch+1:03d} | "
            f"Loss {epoch_loss:.4f} | "
            f"AUC {val_m['auc']:.4f} | "
            f"F1 {val_m['f1']:.4f} | "
            f"ACC {val_m['acc']:.4f} | "
            f"LR {current_lr:.6f}\n"
        )

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "best_model.pth")
            print(f" Novo melhor modelo | Score={best_score:.4f}")


    train_m = evaluate_slice_level(model, train_loader, device)

    log_file.write("\n===== RESULTADOS TREINO =====\n")
    log_file.write(f"AUC: {train_m['auc']:.4f}\n")
    log_file.write(f"Acurácia: {train_m['acc']:.4f}\n")
    log_file.write(f"F1-score: {train_m['f1']:.4f}\n")

    print("\n TESTE FINAL (SLICE-LEVEL)")
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    test_m = evaluate_slice_level(model, test_loader, device)

    print("\n RESULTADOS FINAIS")
    print(f"AUC:            {test_m['auc']:.4f}")
    print(f"Acurácia:       {test_m['acc']:.4f}")
    print(f"F1-score:       {test_m['f1']:.4f}")
    print(f"Kappa:          {test_m['kappa']:.4f}")
    print(f"Sensibilidade:  {test_m['sens']:.4f}")
    print(f"Especificidade: {test_m['spec']:.4f}")
    print("Matriz de Confusão:")
    print(test_m["cm"])

    log_file.write("\n===== RESULTADOS FINAIS =====\n")
    log_file.write(f"AUC: {test_m['auc']:.4f}\n")
    log_file.write(f"Acurácia: {test_m['acc']:.4f}\n")
    log_file.write(f"F1-score: {test_m['f1']:.4f}\n")
    log_file.write(f"Kappa: {test_m['kappa']:.4f}\n")
    log_file.write(f"Sensibilidade: {test_m['sens']:.4f}\n")
    log_file.write(f"Especificidade: {test_m['spec']:.4f}\n")
    log_file.write(f"Matriz de Confusão:\n{test_m['cm']}\n")

    log_file.close()

    

    epochs_range = range(1, len(train_losses) + 1)

   
    plt.figure()
    plt.plot(epochs_range, train_losses)
    plt.title("Loss por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("loss.png")

  
    plt.figure()
    plt.plot(epochs_range, val_aucs, label="AUC")
    plt.plot(epochs_range, val_f1s, label="F1")
    plt.title("Métricas de Validação")
    plt.xlabel("Épocas")
    plt.ylabel("Score")
    plt.legend()
    plt.grid()
    plt.savefig("metrics.png")

  
    plt.figure()
    plt.plot(epochs_range, lrs)
    plt.title("Learning Rate")
    plt.xlabel("Épocas")
    plt.ylabel("LR")
    plt.grid()
    plt.savefig("lr.png")

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    main(args)

