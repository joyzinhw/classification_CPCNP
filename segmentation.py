import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import morphological_chan_vese
from skimage import img_as_float

# -------------------------
# Função: Alargamento de contraste
# -------------------------
def contrast_stretching(img, r1=50, r2=200, w1=0, w2=255):
    img_stretch = np.zeros_like(img, dtype=np.float32)
    alpha = w1 / r1
    beta = (w2 - w1) / (r2 - r1)
    gamma = (255 - w2) / (255 - r2)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = img[i, j]
            if r < r1:
                img_stretch[i, j] = alpha * r
            elif r < r2:
                img_stretch[i, j] = beta * (r - r1) + w1
            else:
                img_stretch[i, j] = gamma * (r - r2) + w2
    return np.clip(img_stretch, 0, 255).astype(np.uint8)

# -------------------------
# Função: Segmenta uma única imagem
# -------------------------
def segmentar_pulmao(path_img_bmp, usar_filtro=True, salvar_em="segmentadas"):
    nome_base = os.path.basename(path_img_bmp).split('.')[0]

    # Cria diretório de saída
    os.makedirs(salvar_em, exist_ok=True)

    # 1. Leitura da imagem BMP em escala de cinza
    img = cv2.imread(path_img_bmp, cv2.IMREAD_GRAYSCALE)

    # 2. Alargamento de contraste
    img_eq = contrast_stretching(img)

    # 3. Aplicar filtro de mediana (opcional)
    if usar_filtro:
        img_eq = cv2.medianBlur(img_eq, 3)

    # 4. Segmentação com contorno ativo (Chan-Vese)
    mask = morphological_chan_vese(img_as_float(img_eq), 35)

    # 5. Converter máscara em uint8 de forma segura
    mask_uint8 = np.clip(mask.astype(np.float32), 0, 1)
    mask_uint8 = (mask_uint8 * 255).astype(np.uint8)

    # 6. Salvar máscara
    path_saida = os.path.join(salvar_em, f"{nome_base}_mascara.png")
    cv2.imwrite(path_saida, mask_uint8)

    print(f"[✓] Segmentado e salvo: {path_saida}")

    # (Opcional) Exibir resultado
    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1); plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(img_eq, cmap='gray'); plt.title("Contraste + Filtro"); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(mask_uint8, cmap='gray'); plt.title("Máscara"); plt.axis('off')
    plt.tight_layout()
    plt.show()


def segmentar(img):
    # Converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplica uma segmentação binária (threshold)
    _, segmentada = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    return segmentada


# -------------------------
# Função: Segmenta todas as imagens BMP em uma pasta
# -------------------------
def segmentar_todas_as_imagens():
    pasta_base = "dataset"              # pasta com as imagens originais
    pasta_saida = "dataset_segmentado"  # pasta onde vamos salvar

    # garante que a pasta de saída exista
    os.makedirs(pasta_saida, exist_ok=True)

    for pasta in os.listdir(pasta_base):
        caminho_pasta = os.path.join(pasta_base, pasta)

        if os.path.isdir(caminho_pasta):
            # cria a subpasta correspondente em dataset_segmentado
            pasta_destino = os.path.join(pasta_saida, pasta)
            os.makedirs(pasta_destino, exist_ok=True)

            for nome_arquivo in os.listdir(caminho_pasta):
                caminho_arquivo = os.path.join(caminho_pasta, nome_arquivo)

                # carrega a imagem
                img = cv2.imread(caminho_arquivo)

                # === aqui entra sua função de segmentação ===
                img_segmentada = segmentar(img)

                # salva na pasta correspondente
                caminho_saida = os.path.join(pasta_destino, nome_arquivo)
                cv2.imwrite(caminho_saida, img_segmentada)

                print("Imagem segmentada salva em:", caminho_saida)

# -------------------------
# Execução principal
# -------------------------
if __name__ == "__main__":
    segmentar_todas_as_imagens()
