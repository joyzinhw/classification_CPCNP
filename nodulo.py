import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Diretórios
img_root = "dataset/fatias"
anno_dir = "dataset/label"
output_root = "Nodules"

os.makedirs(output_root, exist_ok=True)
count_per_folder = {}

xml_sem_imagem = []
img_sem_nodulos = []

# Extensões possíveis
exts = [".bmp", ".png", ".jpg", ".jpeg"]

# Percorre todos os arquivos XML
for xml_file in os.listdir(anno_dir):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(anno_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_base = root.find("filename").text.strip()  # sem extensão

    img_path = None
    current_folder = None

    # Procura em todas as subpastas 1..6
    for subdir in os.listdir(img_root):
        folder_path = os.path.join(img_root, subdir)
        if not os.path.isdir(folder_path):
            continue

        # Procurar por cada extensão possível
        for ext in exts:
            possible = os.path.join(folder_path, filename_base + ext)
            if os.path.exists(possible):
                img_path = possible
                current_folder = subdir
                output_dir = os.path.join(output_root, subdir)
                os.makedirs(output_dir, exist_ok=True)
                break

        if img_path is not None:
            break

    if img_path is None:
        print("XML sem imagem correspondente:", xml_file)
        xml_sem_imagem.append(xml_file)
        continue

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Falha ao abrir imagem:", img_path)
        xml_sem_imagem.append(xml_file)
        continue

    objetos = root.findall("object")
    if len(objetos) == 0:
        print("XML sem nodulos:", xml_file)
        img_sem_nodulos.append(filename_base)
        continue

    nodulos_extraidos = 0

    for i, obj in enumerate(objetos):
        bbox = obj.find("bndbox")

        try:
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
        except:
            print("Erro de coordenadas no XML:", xml_file)
            img_sem_nodulos.append(filename_base)
            continue

        roi = img[ymin:ymax, xmin:xmax]

        if roi.size == 0:
            print("ROI vazia:", filename_base, "nodulo", i)
            img_sem_nodulos.append(filename_base)
            continue

        # --- Segmentação automática ---
        _, mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove ruído pequeno
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Fecha buracos
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Mantém maior contorno
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            precise_mask = np.zeros_like(roi)
            cv2.drawContours(precise_mask, [largest], -1, 255, thickness=-1)
        else:
            precise_mask = np.zeros_like(roi)

        # Aplica máscara
        masked = cv2.bitwise_and(roi, roi, mask=precise_mask)

        # Normaliza para 0-255
        if masked.max() > 0:
            masked = ((masked - masked.min()) / (masked.max() - masked.min()) * 255).astype(np.uint8)

        # Redimensiona para 64x64
        patch64 = cv2.resize(masked, (64, 64), interpolation=cv2.INTER_AREA)

        # Salva nódulo com _seg no final, sem numeração
        save_name = f"{filename_base}_seg.bmp"
        save_path = os.path.join(output_root, current_folder, save_name)
        cv2.imwrite(save_path, patch64)

        nodulos_extraidos += 1
        count_per_folder[current_folder] = count_per_folder.get(current_folder, 0) + 1

    if nodulos_extraidos == 0:
        img_sem_nodulos.append(filename_base)

# 🟦 RESULTADOS FINAIS
print("\nTotais por pasta:")
for folder, count in sorted(count_per_folder.items()):
    print(folder + ":", count)

print("\nXML sem imagem correspondente:")
for f in xml_sem_imagem:
    print(f)

print("\nImagens sem nodulos extraídos:")
for f in set(img_sem_nodulos):
    print(f)
