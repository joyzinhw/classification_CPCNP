import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

img_dir = "BMP_2D/Image"
anno_dir = "BMP_2D/Annotations"
output_dir = "Nodules_2D"
class_dir = os.path.join(output_dir, "nodule")
os.makedirs(class_dir, exist_ok=True)

for xml_file in os.listdir(anno_dir):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(anno_dir, xml_file))
    root = tree.getroot()

    filename = root.find("filename").text
    img_path = os.path.join(img_dir, filename)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Imagem não encontrada: {img_path}")
        continue

    for i, obj in enumerate(root.findall("object")):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        roi = img[ymin:ymax, xmin:xmax]
        _, mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        precise_mask = np.zeros_like(roi)
        cv2.drawContours(precise_mask, contours, -1, 255, thickness=-1)

        masked_img = np.zeros_like(img)
        masked_img[ymin:ymax, xmin:xmax] = cv2.bitwise_and(roi, roi, mask=precise_mask)

        save_name = f"{os.path.splitext(filename)[0]}_{i}.bmp"
        save_path = os.path.join(class_dir, save_name)

        cv2.imwrite(save_path, masked_img)

print("✅ Nódulos extraídos com contorno preciso e tamanho original. Salvos em:", output_dir)
