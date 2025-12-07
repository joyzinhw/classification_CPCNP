import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.segmentation import morphological_chan_vese
from skimage import img_as_float
from skimage.morphology import remove_small_objects, remove_small_holes


def clahe_equalization(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def preprocessar_imagem(img):
    # 1. CLAHE
    img_eq = clahe_equalization(img)
    # 2. Denoising
    img_filtrada = cv2.fastNlMeansDenoising(img_eq, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # 3. Fechamento e abertura morfológica
    kernel_fechamento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_fechada = cv2.morphologyEx(img_filtrada, cv2.MORPH_CLOSE, kernel_fechamento)
    kernel_abertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_aberta = cv2.morphologyEx(img_fechada, cv2.MORPH_OPEN, kernel_abertura)
    return img_aberta


def posprocessar_mascara_robusta(mask, min_area=5000, max_hole_size=1000):
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_bin = mask_uint8 > 0
    mask_clean = remove_small_objects(mask_bin, min_size=min_area)
    mask_clean = remove_small_holes(mask_clean, area_threshold=max_hole_size)
    mask_final = (mask_clean.astype(np.uint8)) * 255

    # Fechamento e abertura adicionais
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask_final = cv2.GaussianBlur(mask_final, (5,5), 0)

    # Mantém apenas a maior estrutura
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_final, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # ignora o fundo
        mask_maior = np.zeros_like(mask_final)
        mask_maior[labels == largest_label] = 255
        mask_final = mask_maior

    # Convex hull para suavizar bordas
    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_suave = np.zeros_like(mask_final)
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(mask_suave, [hull], 0, 255, -1)

    return mask_suave

# -------------------------
# Função: Segmentação principal com fallback adaptativo
# -------------------------
def segmentar_pulmao_melhorado(path_img):
    img_original = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    img_preprocessada = preprocessar_imagem(img_original)

    # adaptativa com Otsu
    _, otsu_mask = cv2.threshold(img_preprocessada, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Chan-Vese c/ inicialização
    mask_chan_vese = morphological_chan_vese(
        img_as_float(img_preprocessada),
        num_iter=60,
        smoothing=3,
        init_level_set=otsu_mask
    )

    # Verificação de falha
    area = np.sum(mask_chan_vese)
    total_pixels = img_preprocessada.size
    if area < 0.05 * total_pixels or area > 0.95 * total_pixels:
        print("⚠️ Chan-Vese falhou, aplicando fallback com Otsu.")
        _, mask_chan_vese = cv2.threshold(img_preprocessada, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask_final = posprocessar_mascara_robusta(mask_chan_vese)
    pulmao_segmentado = cv2.bitwise_and(img_original, img_original, mask=mask_final)

    return mask_final, pulmao_segmentado, img_original, img_preprocessada

def processar_pasta_unica(pasta_base="dataset/Image/1",
                           pasta_saida="dataset_segmentado/Image/1",
                           mostrar=True):
    if not os.path.exists(pasta_base):
        print(f"[!] A pasta {pasta_base} não existe!")
        return
    
    pasta_mascara = os.path.join(pasta_saida, "mascaras")
    pasta_pulmao = os.path.join(pasta_saida, "pulmoes")
    os.makedirs(pasta_mascara, exist_ok=True)
    os.makedirs(pasta_pulmao, exist_ok=True)
    
    arquivos = [f for f in os.listdir(pasta_base) if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))]
    if not arquivos:
        print("[!] Nenhuma imagem encontrada na pasta.")
        return
    
    registros = []
    
    for nome_arquivo in arquivos:
        caminho_arquivo = os.path.join(pasta_base, nome_arquivo)
        print(f"[→] Processando: {caminho_arquivo}")
        
        try:
            mascara, pulmao_segmentado, img_original, img_preprocessada = segmentar_pulmao_melhorado(caminho_arquivo)
            
            nome_sem_ext = os.path.splitext(nome_arquivo)[0]
            path_mascara = os.path.join(pasta_mascara, f"{nome_sem_ext}_mascara.png")
            path_pulmao = os.path.join(pasta_pulmao, f"{nome_sem_ext}_pulmao.png")
            
            cv2.imwrite(path_mascara, mascara)
            cv2.imwrite(path_pulmao, pulmao_segmentado)
            
            registros.append({
                "imagem_original": caminho_arquivo,
                "mascara": path_mascara,
                "pulmao_segmentado": path_pulmao
            })
            
            print(f"[✓] Salvo: {path_mascara} e {path_pulmao}")
            
            if mostrar:
                plt.figure(figsize=(12,4))
                plt.subplot(1,4,1); plt.imshow(img_original, cmap='gray'); plt.title("Original"); plt.axis('off')
                plt.subplot(1,4,2); plt.imshow(img_preprocessada, cmap='gray'); plt.title("Pré-processada"); plt.axis('off')
                plt.subplot(1,4,3); plt.imshow(mascara, cmap='gray'); plt.title("Máscara"); plt.axis('off')
                plt.subplot(1,4,4); plt.imshow(pulmao_segmentado, cmap='gray'); plt.title("Pulmão Segmentado"); plt.axis('off')
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            print(f"[!] Erro ao processar {nome_arquivo}: {e}")

if __name__ == "__main__":
    processar_pasta_unica(
        pasta_base="dataset/Image/1",
        pasta_saida="dataset_segmentado/Image/1",
        mostrar=True
    )
