import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

img_dir = "BMP_2D/Image"
anno_dir = "BMP_2D/Annotations"
output_dir = "Comparison_2D"

os.makedirs(output_dir, exist_ok=True)

for xml_file in os.listdir(anno_dir):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(anno_dir, xml_file))
    root = tree.getroot()

    filename = root.find("filename").text
    img_path = os.path.join(img_dir, filename)

    img_orig = cv2.imread(img_path)
    if img_orig is None:
        print(f"⚠️ Imagem não encontrada: {img_path}")
        continue

    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

    for i, obj in enumerate(root.findall("object")):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        roi = img_gray[ymin:ymax, xmin:xmax]

        _, mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        precise_mask = np.zeros_like(roi)
        cv2.drawContours(precise_mask, contours, -1, 255, thickness=-1)

        original = img_orig.copy()
        highlighted = img_orig.copy()
        highlighted[ymin:ymax, xmin:xmax][precise_mask == 255] = [0, 0, 255]

        masked_img = np.zeros_like(img_gray)
        masked_img[ymin:ymax, xmin:xmax] = cv2.bitwise_and(roi, roi, mask=precise_mask)
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR)

        comparison = cv2.hconcat([original, highlighted, masked_img])

        os.makedirs(output_dir, exist_ok=True)

        save_name = f"{os.path.splitext(filename)[0]}_{i}_comparison.bmp"
        save_path = os.path.join(output_dir, save_name)

        cv2.imwrite(save_path, comparison)

print("✅ Comparações salvas em:", output_dir)
