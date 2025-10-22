import os
import cv2
import numpy as np
import shutil
import random
from scipy.ndimage import rotate
import imageio.v2 as imageio

base_dir = 'dataset/Image'
train_dir = 'dataset/train'
test_dir = 'dataset/test'
classes = ['0', '1', '2']
train_ratio = 0.8
valid_exts = ('.bmp', '.jpg', '.jpeg', '.png')

def remove_salt_pepper_noise(img):
    """Remove ruído sal e pimenta com filtro mediano 3x3"""
    return cv2.medianBlur(img, 3)

def high_boost_filter(img, boost_factor=1.5):
    """Aplica realce high-boost mantendo a imagem colorida"""
    # Aplica Gaussian Blur com sigma = 1
    blurred = cv2.GaussianBlur(img, (5, 5), 1)
    # Calcula máscara (opcional)
    mask = cv2.subtract(img, blurred)
    # Aplica high-boost
    highboost = cv2.addWeighted(img, 1.0 + boost_factor, blurred, -boost_factor, 0)
    # Normaliza para 0-255
    highboost = cv2.normalize(highboost, None, 0, 255, cv2.NORM_MINMAX)
    return highboost.astype(np.uint8)

def preprocess_image(input_path):
    """Aplica pré-processamento e retorna imagem processada"""
    img = cv2.imread(input_path)
    if img is None:
        print(f"⚠️ Erro ao ler imagem: {input_path}")
        return None
    # Remove ruído usando mediana 3x3
    denoised = remove_salt_pepper_noise(img)
    # Aplica high-boost colorido
    enhanced = high_boost_filter(denoised, boost_factor=1.5)
    return enhanced

def augment_image(img):
    """Gera imagens aumentadas (rotações e flip)"""
    aug_images = []
    for angle in [90, 180, 270]:
        aug_images.append(rotate(img, angle, axes=(0, 1), reshape=False))
    flipped = np.ascontiguousarray(img[::-1, ::-1])
    aug_images.append(flipped)
    return aug_images

def balance_classes(target_dir):
    """Equilibra classes dentro de um diretório (train ou test)"""
    counts = {}
    for cls in classes:
        path = os.path.join(target_dir, cls)
        imgs = [f for f in os.listdir(path) if f.lower().endswith(valid_exts)]
        counts[cls] = len(imgs)

    max_count = max(counts.values()) if counts else 0
    print(f"Tamanhos em {target_dir}: {counts} | Classe alvo: {max_count}")

    for cls in classes:
        class_path = os.path.join(target_dir, cls)
        imgs = [f for f in os.listdir(class_path) if f.lower().endswith(valid_exts)]
        if len(imgs) < max_count:
            needed = max_count - len(imgs)
            print(f"🔄 Gerando {needed} imagens extras para classe {cls} em {target_dir}...")
            i = 0
            while needed > 0:
                for fname in imgs:
                    img = imageio.imread(os.path.join(class_path, fname))
                    aug_list = augment_image(img)
                    for aug in aug_list:
                        out_name = f"aug_{i}_{fname}"
                        out_path = os.path.join(class_path, out_name)
                        imageio.imsave(out_path, aug.astype(np.uint8))
                        needed -= 1
                        i += 1
                        if needed <= 0:
                            break
                    if needed <= 0:
                        break
    print(f"✅ Balanceamento concluído para {target_dir}!\n")

def split_and_preprocess():
    """Divide dataset e salva imagens pré-processadas apenas em train/test"""

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # Divide e processa
    for cls in classes:
        class_path = os.path.join(base_dir, cls)
        images = [f for f in os.listdir(class_path)
                  if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith(valid_exts)]
        random.shuffle(images)
        split_point = int(len(images) * train_ratio)
        train_imgs = images[:split_point]
        test_imgs = images[split_point:]

        for img_name in train_imgs:
            src = os.path.join(class_path, img_name)
            dst = os.path.join(train_dir, cls, img_name)
            processed = preprocess_image(src)
            if processed is not None:
                cv2.imwrite(dst, processed)

        for img_name in test_imgs:
            src = os.path.join(class_path, img_name)
            dst = os.path.join(test_dir, cls, img_name)
            processed = preprocess_image(src)
            if processed is not None:
                cv2.imwrite(dst, processed)

        print(f"Classe {cls}: {len(train_imgs)} treino, {len(test_imgs)} teste")

    print("\n✅ Divisão e pré-processamento concluídos com sucesso!\n")


if __name__ == '__main__':
    random.seed(42)

    print("📊 Dividindo e pré-processando dataset...")
    split_and_preprocess()

    print("⚖️ Balanceando classes em train...")
    balance_classes(train_dir)

    print("⚖️ Balanceando classes em test...")
    balance_classes(test_dir)

    print("\n✅ Pipeline completo executado com sucesso! 🚀")
