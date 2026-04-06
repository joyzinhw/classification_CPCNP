import os
from collections import Counter


root_dir = "dataset/train"

counter = Counter()

for cls in os.listdir(root_dir):
    cls_path = os.path.join(root_dir, cls)

    if not os.path.isdir(cls_path):
        continue

    total = 0

    for subdir, _, files in os.walk(cls_path):
        total += len([f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])

    counter[cls] = total  # 👈 agora conta por classe

print("📊 Distribuição final:", counter)