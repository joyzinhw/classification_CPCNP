import os
import numpy as np
import nibabel as nib
import pandas as pd
import cv2
from tqdm import tqdm
from collections import defaultdict
from PIL import Image

BASE = "Dados_Macro"
OUT  = "NODULES"

MACROS_VALIDOS = [
    "Adenocarcinoma (NSCLC)",
    "Carcinoma Escamoso (NSCLC)",
    "Outros NSCLC"
]

os.makedirs(OUT, exist_ok=True)

label_df = pd.read_excel("dataset_info_excel/Label.xlsx")
label_df.columns = label_df.columns.str.strip()
label_df = label_df[label_df["labels_type"] == 1]

def normalize_ct(img):
    img = np.clip(img, -1000, 400)
    img = (img + 1000) / 1400
    return (img * 255).astype(np.uint8)


def unsharp_mask(img):
    blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)

RESAMPLING_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def preserve_useful_lesion_area(
    img: Image.Image,
    target_size: int = 160,
    threshold: int = 10,
    margin_ratio: float = 0.20,
):
    if img.mode != "L":
        img = img.convert("L")

    array = np.array(img)
    coords = np.argwhere(array >= threshold)

    if coords.size == 0:
        return img.resize((target_size, target_size), RESAMPLING_BILINEAR)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1
    margin = max(1, int(np.ceil(max(bbox_h, bbox_w) * margin_ratio)))
    side = max(bbox_h, bbox_w) + 2 * margin

    center_y = (y_min + y_max) / 2.0
    center_x = (x_min + x_max) / 2.0

    top = int(np.floor(center_y - side / 2))
    left = int(np.floor(center_x - side / 2))
    bottom = top + side
    right = left + side

    crop = np.zeros((side, side), dtype=array.dtype)

    src_top = max(top, 0)
    src_left = max(left, 0)
    src_bottom = min(bottom, array.shape[0])
    src_right = min(right, array.shape[1])

    dst_top = src_top - top
    dst_left = src_left - left
    dst_bottom = dst_top + (src_bottom - src_top)
    dst_right = dst_left + (src_right - src_left)

    crop[dst_top:dst_bottom, dst_left:dst_right] = array[src_top:src_bottom, src_left:src_right]

    crop_img = Image.fromarray(crop, mode="L")

    if side > target_size:
        return crop_img.resize((target_size, target_size), RESAMPLING_BILINEAR)

    if side < target_size:
        canvas = Image.new("L", (target_size, target_size), color=0)
        offset = ((target_size - side) // 2, (target_size - side) // 2)
        canvas.paste(crop_img, offset)
        return canvas

    return crop_img


resumo = defaultdict(lambda: {"pacientes": set(), "fatias": 0})
corrupted_files = []

for macro in MACROS_VALIDOS:

    macro_dir = os.path.join(BASE, macro)
    if not os.path.isdir(macro_dir):
        continue

    ids = [p for p in os.listdir(macro_dir) if p.isdigit()]

    for pid in tqdm(ids, desc=f"Extraindo {macro}"):

        pid_int = int(pid)
        folder = os.path.join(macro_dir, pid)

        ct_path    = os.path.join(folder, f"{pid}_CT.nii.gz")
        tumor_path = os.path.join(folder, f"{pid}_tumor.nii.gz")

        if not os.path.exists(ct_path) or not os.path.exists(tumor_path):
            continue

        try:
            ct   = nib.load(ct_path).get_fdata()
            mask = nib.load(tumor_path).get_fdata().astype(np.int32)
        except Exception:
            corrupted_files.append(pid)
            continue

        if ct.shape != mask.shape or ct.ndim != 3:
            continue

        labels_pid = label_df[label_df["ID"] == pid_int]["Mark_labels"].tolist()
        if len(labels_pid) == 0:
            continue

        out_dir = os.path.join(OUT, macro, pid)
        os.makedirs(out_dir, exist_ok=True)

        resumo[(macro, "GERAL")]["pacientes"].add(pid)

        for tumor_idx, lab in enumerate(labels_pid, start=1):

            slices_z = np.where(np.any(mask == lab, axis=(0,1)))[0]
            if len(slices_z) == 0:
                continue

            for z in slices_z:

                slice_ct   = ct[:, :, z]
                slice_mask = mask[:, :, z]

                nodule_mask = (slice_mask == lab)

                if not np.any(nodule_mask):
                    continue

                norm = normalize_ct(slice_ct)

                coords = np.column_stack(np.where(nodule_mask))
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)

                crop_img  = norm[y_min:y_max, x_min:x_max]
                crop_mask = nodule_mask[y_min:y_max, x_min:x_max]

                masked = np.zeros_like(crop_img, dtype=np.uint8)
                masked[crop_mask] = crop_img[crop_mask]


                pil_img = Image.fromarray(masked)

                final = preserve_useful_lesion_area(
                    pil_img,
                    target_size=160,
                    threshold=10,
                    margin_ratio=0.20
                )

                final = np.array(final)
                final = unsharp_mask(final)

                fname = f"{pid}_{tumor_idx}_{z}.png"
                cv2.imwrite(os.path.join(out_dir, fname), final)

                resumo[(macro, "GERAL")]["fatias"] += 1



print("-" * 85)
print(f"{'Categoria Macro':<30} | {'Pacientes':<10} | {'Fatias 2D':<10}")
print("-" * 85)

total_p = 0
total_f = 0

for (macro, _), v in sorted(resumo.items()):
    pac = len(v["pacientes"])
    fat = v["fatias"]

    total_p += pac
    total_f += fat

    print(f"{macro[:30]:<30} | {pac:<10} | {fat:<10}")

print("-" * 85)

if corrupted_files:
    print("Arquivos corrompidos:", corrupted_files)