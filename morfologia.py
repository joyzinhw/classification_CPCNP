
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Caminho da imagem
caminho_img = "dataset/0/0006_33.bmp"

# Carrega a imagem em escala de cinza
img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Não encontrei a imagem em: {caminho_img}")

# Binariza
_, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# --- Análise de componentes conectados ---
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
mask_clean = np.zeros_like(mask)

altura, largura = mask.shape
min_area = 1000

for i, stat in enumerate(stats):
    if i == 0:  # fundo
        continue
    x, y, w, h, area = stat

    # Critérios para REMOVER
    toca_borda_inferior = (y + h >= altura - 2)
    toca_lateral = (x <= 1) and (x + w >= largura - 2)   # atravessa esquerda e direita

    if toca_borda_inferior or toca_lateral:
        continue

    # Caso contrário, mantém
    if area >= min_area:
        mask_clean[labels == i] = 255

# --- Mostrar resultado ---
plt.figure(figsize=(12,4))
plt.subplot(1,2,1); plt.imshow(mask, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(mask_clean, cmap="gray"); plt.title("Sem Linhas Inferiores/Laterais"); plt.axis("off")
plt.tight_layout()
plt.show()

# Salvar
cv2.imwrite("dataset/0/0006_33_sem_linhas_final.png", mask_clean)
print("[✓] Máscara salva em: dataset/0/0006_33_sem_linhas_final.png")


