import os
import random
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image

from augmentations import NoduleAugmentation  


class NoduleDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.transform = transform
        self.split = split

        self.data = []  

        split_dir = os.path.join(root_dir, split)
        classes = sorted(os.listdir(split_dir))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)

            for subdir in os.listdir(cls_dir): 
                subdir_path = os.path.join(cls_dir, subdir)

                if not os.path.isdir(subdir_path):
                    continue

                for img_name in os.listdir(subdir_path):
                    if img_name.lower().endswith((".png", ".jpg", ".bmp")):
                        self.data.append({
                            "path": os.path.join(subdir_path, img_name),
                            "label": self.class_to_idx[cls],
                            "augmented": False 
                        })

       
        original_labels = [item["label"] for item in self.data]
        original_count = Counter(original_labels)

       
        if split == "train":
            self.data = self.augment_minority(self.data)

            new_labels = [item["label"] for item in self.data]
            new_count = Counter(new_labels)

            print("\n📊 AUGMENTATION REPORT (TRAIN)")
            print(f"Classe 2 antes: {original_count[2]}")
            print(f"Classe 2 depois: {new_count[2]}")

            aumento = new_count[2] - original_count[2]
            fator = new_count[2] / original_count[2]

            print(f"Aumento absoluto: +{aumento}")
            print(f"Fator de aumento: {fator:.2f}x")

            print("\n📊 DISTRIBUIÇÃO COMPLETA (ANTES → DEPOIS)")
            for cls in sorted(original_count.keys()):
                before = original_count[cls]
                after = new_count[cls]
                print(f"Classe {cls}: {before} → {after} (x{after/before:.2f})")

       
        self.labels = [item["label"] for item in self.data]

        self.augment = NoduleAugmentation()

    
    def augment_minority(self, data):
        augmented = []

        for item in data:
            augmented.append(item)

           
            if item["label"] == 2:
                for _ in range(2): 
                    new_item = item.copy()
                    new_item["augmented"] = True
                    augmented.append(new_item)

        return augmented

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img = Image.open(item["path"]).convert("L")

       
        if self.split == "train" and item["augmented"]:
            img = self.augment(img)

        if self.transform:
            img = self.transform(img)

        label = item["label"]
        return img, label