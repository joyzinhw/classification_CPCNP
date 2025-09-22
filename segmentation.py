import os
import cv2
import numpy as np

# Pastas de entrada e saída
input_dir = "dataset"
output_dir = "dataset_seg"
os.makedirs(output_dir, exist_ok=True)

min_area = 1000  # área mínima para manter o componente

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if not file.lower().endswith(".bmp"):
            continue
        
        caminho_img = os.path.join(root, file)
        img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Não consegui abrir {caminho_img}")
            continue

        # Segmentação binária
        _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Componentes conectados
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        mask_clean = np.zeros_like(mask)

        altura, largura = mask.shape

        for i, stat in enumerate(stats):
            if i == 0:  # fundo
                continue
            x, y, w, h, area = stat
            toca_borda_inferior = (y + h >= altura - 2)
            toca_lateral = (x <= 1) and (x + w >= largura - 2)
            if toca_borda_inferior or toca_lateral:
                continue
            if area >= min_area:
                mask_clean[labels == i] = 255

        # Caminho relativo para manter a hierarquia
        rel_path = os.path.relpath(root, input_dir)
        save_dir = os.path.join(output_dir, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, os.path.splitext(file)[0] + "_seg.bmp")
        cv2.imwrite(save_path, mask_clean)

        print(f"✅ Processada e salva: {save_path}")
