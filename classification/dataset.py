import os
from torch.utils.data import Dataset
from PIL import Image

class NoduleDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.transform = transform
        self.img_paths = []
        self.labels = []

        split_dir = os.path.join(root_dir, split)
        classes = sorted(os.listdir(split_dir))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)

            # percorre subpastas (ex: 214975)
            for subdir in os.listdir(cls_dir):
                subdir_path = os.path.join(cls_dir, subdir)

                if not os.path.isdir(subdir_path):
                    continue

                for img_name in os.listdir(subdir_path):
                    if img_name.lower().endswith((".png", ".jpg", ".bmp")):
                        self.img_paths.append(
                            os.path.join(subdir_path, img_name)
                        )
                        self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

