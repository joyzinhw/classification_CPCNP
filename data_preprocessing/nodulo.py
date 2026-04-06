import os
import numpy as np
import nibabel as nib
import pandas as pd
import cv2
from tqdm import tqdm
from collections import defaultdict

BASE = "Dados_Macro"
OUT  = "NODULES"

MACROS_VALIDOS = [
    "Adenocarcinoma (NSCLC)",
    "Carcinoma Escamoso (NSCLC)",
    "Outros NSCLC"
]

os.makedirs(OUT, exist_ok=True)

label_df = pd.read_excel("Label.xlsx")
label_df.columns = label_df.columns.str.strip()
label_df = label_df[label_df["labels_type"] == 1]


def normalize_ct(img):
    img = np.clip(img, -1000, 400)
    img = (img + 1000) / 1400
    return (img * 255).astype(np.uint8)


def unsharp_mask(img):
    blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def pad_to_patch(img, patch_size):

    h, w = img.shape

    canvas = np.zeros((patch_size, patch_size), dtype=img.dtype)

    y_offset = (patch_size - h) // 2
    x_offset = (patch_size - w) // 2

    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img

    return canvas


def find_max_nodule_size():

    max_size = 0

    max_info = {
        "pid": None,
        "macro": None,
        "label": None,
        "slice": None,
        "width": 0,
        "height": 0
    }

    for macro in MACROS_VALIDOS:

        macro_dir = os.path.join(BASE, macro)
        if not os.path.isdir(macro_dir):
            continue

        ids = [p for p in os.listdir(macro_dir) if p.isdigit()]

        for pid in tqdm(ids, desc=f"Scanning {macro}"):

            pid_int = int(pid)
            folder = os.path.join(macro_dir, pid)

            tumor_path = os.path.join(folder, f"{pid}_tumor.nii.gz")

            if not os.path.exists(tumor_path):
                continue

            try:
                mask = nib.load(tumor_path).get_fdata().astype(np.int32)
            except:
                continue

            labels_pid = label_df[label_df["ID"] == pid_int]["Mark_labels"].tolist()

            for lab in labels_pid:

                slices_z = np.where(np.any(mask == lab, axis=(0,1)))[0]

                for z in slices_z:

                    slice_mask = mask[:,:,z]
                    nodule_mask = (slice_mask == lab)

                    if not np.any(nodule_mask):
                        continue

                    coords = np.column_stack(np.where(nodule_mask))

                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)

                    h = y_max - y_min
                    w = x_max - x_min

                    size = max(h, w)

                    if size > max_size:

                        max_size = size

                        max_info = {
                            "pid": pid,
                            "macro": macro,
                            "label": lab,
                            "slice": z,
                            "width": w,
                            "height": h
                        }

    return max_size, max_info


print("\nCalculando maior nódulo do dataset...\n")

PATCH_SIZE, max_info = find_max_nodule_size()

print("\nMaior nódulo encontrado:")
print("Paciente ID:", max_info["pid"])
print("Categoria:", max_info["macro"])
print("Label do nódulo:", max_info["label"])
print("Slice:", max_info["slice"])
print("Largura:", max_info["width"])
print("Altura:", max_info["height"])

print("\nTodos os patches terão tamanho:", PATCH_SIZE, "x", PATCH_SIZE)


resumo = defaultdict(lambda: {"pacientes": set(), "fatias": 0})
corrupted_files = []

for macro in MACROS_VALIDOS:

    macro_dir = os.path.join(BASE, macro)
    if not os.path.isdir(macro_dir):
        continue

    ids = [p for p in os.listdir(macro_dir) if p.isdigit()]

    for pid in tqdm(ids, desc=f"Extraindo {macro}", leave=False):

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

                slice_ct   = ct[:,:,z]
                slice_mask = mask[:,:,z]

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

                final = pad_to_patch(masked, PATCH_SIZE)

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

print("\nMaior nódulo do dataset:")
print("Paciente:", max_info["pid"])
print("Categoria:", max_info["macro"])
print("Label:", max_info["label"])
print("Slice:", max_info["slice"])
print("Largura:", max_info["width"])
print("Altura:", max_info["height"])

print("\nTamanho final de todos os patches:", PATCH_SIZE, "x", PATCH_SIZE)

if corrupted_files:
    print("Arquivos corrompidos:", corrupted_files)