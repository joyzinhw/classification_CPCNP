import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

dataset_path = "dataset/train"  # 🔥 usar TRAIN

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),  # padrão
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(dataset_path, transform=transform)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

# 🔥 acumuladores corretos
sum_ = 0.
sum_sq = 0.
num_pixels = 0

for images, _ in loader:
    # images: [B, C, H, W]
    images = images.float()

    sum_ += images.sum()
    sum_sq += (images ** 2).sum()
    num_pixels += images.numel()

# 🔥 média global
mean = sum_ / num_pixels

# 🔥 variância global
std = torch.sqrt((sum_sq / num_pixels) - (mean ** 2))

print(f"Mean: {mean.item():.6f}")
print(f"Std:  {std.item():.6f}")