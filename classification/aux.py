import os
import random
import uuid
from PIL import Image
from tqdm import tqdm

from augmentations import NoduleAugmentation


input_dir = "dataset/train/Carcinoma Escamoso (NSCLC)"
target_total = 2439

aug = NoduleAugmentation(p=1.0)


image_data = []

for subdir, _, files in os.walk(input_dir):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            if "_aug_" not in f: 
                full_path = os.path.join(subdir, f)
                image_data.append((full_path, subdir))

print(f"📊 Imagens originais: {len(image_data)}")

if len(image_data) == 0:
    print("❌ Nenhuma imagem original encontrada!")
    exit()


total_images = 0
for _, _, files in os.walk(input_dir):
    total_images += len([f for f in files if f.lower().endswith((".png",".jpg",".jpeg",".bmp"))])

to_generate = target_total - total_images

print(f"📊 Total atual: {total_images}")
print(f"🧪 Gerar: {to_generate}")

if to_generate <= 0:
    print("✔ Já está balanceado")
    exit()

count = 0
pbar = tqdm(total=to_generate)

while count < to_generate:
    img_path, save_dir = random.choice(image_data)

    img = Image.open(img_path).convert("L")
    aug_img = aug(img)

  
    unique_id = uuid.uuid4().hex
    base_name = os.path.basename(img_path).split('.')[0]
    new_name = f"{base_name}_aug_{unique_id}.png"

    save_path = os.path.join(save_dir, new_name)

  
    if os.path.exists(save_path):
        continue

    aug_img.save(save_path)

    count += 1
    pbar.update(1)

pbar.close()

print("✅ Augmentation finalizado!")