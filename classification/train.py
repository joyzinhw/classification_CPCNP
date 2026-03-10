import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import Counter

# from Dual import Res2Net_LungNodule
from simple_cnns import LungNoduleNet
from dataset import NoduleDataset
from train_utils import evaluate_slice_level
# from EfficientNet import efficientnet_nodule64



def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Treinando em: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),

    ])


    train_set = NoduleDataset(args.root_dir, "train", transform)
    val_set   = NoduleDataset(args.root_dir, "val", transform)
    test_set  = NoduleDataset(args.root_dir, "test", transform)

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


    model = LungNoduleNet(num_classes=3).to(device)
    # model = efficientnet_nodule64(num_classes=3).to(device)
    # model = NoduleCNN64_CBAM(num_classes=3).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    criterion = torch.nn.CrossEntropyLoss()


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5
    )


    best_score = 0.0

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

        score = 0.7 * val_m["auc"] + 0.3 * val_m["f1"]

        scheduler.step(score)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Loss {epoch_loss:.4f} | "
            f"AUC {val_m['auc']:.4f} | "
            f"F1 {val_m['f1']:.4f} | "
            f"ACC {val_m['acc']:.4f}"
        )

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "best_model.pth")
            print(f" Novo melhor modelo | Score={best_score:.4f}")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    main(args)
