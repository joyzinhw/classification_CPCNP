import os
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

BASE = "Dados_Macro"
OUT  = "NODULES_3D"

MACROS_VALIDOS = [
    "Adenocarcinoma (NSCLC)",
    "Carcinoma Escamoso (NSCLC)",
]

IMG_SIZE = 64
DEPTH    = 64
MARGIN   = 32

os.makedirs(OUT, exist_ok=True)

label_df = pd.read_excel("Label.xlsx")
label_df.columns = label_df.columns.str.strip()
label_df = label_df[label_df["labels_type"] == 1]

def pad_or_crop_depth(vol, target_depth):
    d = vol.shape[0]
    if d == target_depth:
        return vol
    if d < target_depth:
        pad_before = (target_depth - d) // 2
        pad_after  = target_depth - d - pad_before
        return np.pad(vol, ((pad_before, pad_after), (0,0), (0,0)), constant_values=0)
    start = (d - target_depth) // 2
    return vol[start:start+target_depth]

resumo = defaultdict(lambda: {"pacientes": set(), "nodulos": 0})
corrupted_files = []

for macro in MACROS_VALIDOS:

    macro_dir = os.path.join(BASE, macro)
    if not os.path.isdir(macro_dir):
        continue

    ids = [p for p in os.listdir(macro_dir) if p.isdigit()]

    for pid in tqdm(ids, desc=f"{macro}", leave=False):

        pid_int = int(pid)
        folder = os.path.join(macro_dir, pid)

        ct_path    = os.path.join(folder, f"{pid}_CT.nii.gz")
        tumor_path = os.path.join(folder, f"{pid}_tumor.nii.gz")

        if not os.path.exists(ct_path) or not os.path.exists(tumor_path):
            continue

        try:
            ct   = nib.load(ct_path).get_fdata()
            mask = nib.load(tumor_path).get_fdata()
        except Exception:
            corrupted_files.append(pid)
            continue

        if ct.shape != mask.shape or ct.ndim != 3:
            continue

        labels_pid = label_df[label_df["ID"] == pid_int]["Mark_labels"].tolist()

        if len(labels_pid) == 0:
            continue

        out_dir = os.path.join(OUT, macro)
        os.makedirs(out_dir, exist_ok=True)

        resumo[(macro, "GERAL")]["pacientes"].add(pid)

        for tumor_idx, lab in enumerate(labels_pid, start=1):

            slices_z = np.where(np.any(mask == lab, axis=(0,1)))[0]
            if len(slices_z) == 0:
                continue

            zmin, zmax = slices_z.min(), slices_z.max() + 1

            volume_ct   = ct[:, :, zmin:zmax]
            volume_mask = mask[:, :, zmin:zmax] == lab

            volume_ct = np.clip(volume_ct, -1000, 400)

            ys, xs, zs = np.where(volume_mask)
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()

            ymin = max(0, ymin - MARGIN)
            xmin = max(0, xmin - MARGIN)
            ymax = min(volume_ct.shape[0], ymax + MARGIN)
            xmax = min(volume_ct.shape[1], xmax + MARGIN)

            volume_ct = volume_ct[ymin:ymax, xmin:xmax, :]
            volume_ct = np.transpose(volume_ct, (2, 0, 1))

            volume_ct = pad_or_crop_depth(volume_ct, DEPTH)

            volume_ct = np.array([
                np.pad(
                    s,
                    (
                        (0, max(0, IMG_SIZE - s.shape[0])),
                        (0, max(0, IMG_SIZE - s.shape[1]))
                    ),
                    constant_values=0
                )[:IMG_SIZE, :IMG_SIZE]
                for s in volume_ct
            ])

            volume_ct = (volume_ct + 1000) / 1400
            volume_ct = volume_ct.astype(np.float32)

            fname = f"{pid}_tumor{tumor_idx}.npy"
            np.save(os.path.join(out_dir, fname), volume_ct)

            resumo[(macro, "GERAL")]["nodulos"] += 1

print("-" * 90)
print(f"{'Categoria Macro':<30} | {'Pacientes':<10} | {'Nódulos 3D':<10}")
print("-" * 90)

for (macro, _), v in sorted(resumo.items()):
    print(f"{macro[:30]:<30} | {len(v['pacientes']):<10} | {v['nodulos']:<10}")

print("-" * 90)

if corrupted_files:
    print("Arquivos corrompidos:", corrupted_files)