# import os
# import shutil
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
# from collections import Counter
# from PIL import Image

# print("="*60)
# print("PREPARANDO DATASET COM K-FOLD PURO (k=5)")
# print("="*60)

# NODULES_DIR = "NODULES"
# OUTPUT_DATASET_DIR = "dataset_kfold"
# K = 5

# classes = {
#     'Adenocarcinoma (NSCLC)': 0,
#     'Carcinoma Escamoso (NSCLC)': 1,
# }

# # ==============================
# # 1️⃣ LEITURA DOS DADOS
# # ==============================

# dados = []

# print("\n📁 Varredura das pastas...")

# for classe_nome, classe_id in classes.items():
#     classe_path = os.path.join(NODULES_DIR, classe_nome)

#     if not os.path.exists(classe_path):
#         print(f"⚠️ Pasta não encontrada: {classe_path}")
#         continue

#     for root, _, files in os.walk(classe_path):
#         for file in files:
#             if file.endswith(".png"):
#                 paciente_id = file.split('_')[0]

#                 caminho_relativo = os.path.relpath(
#                     os.path.join(root, file), NODULES_DIR
#                 )

#                 dados.append({
#                     "patient_id": paciente_id,
#                     "path": caminho_relativo,
#                     "label": classe_id,
#                     "class_name": classe_nome
#                 })

# df = pd.DataFrame(dados)

# print(f"\n📊 Total de nódulos: {len(df)}")
# print(f"📊 Total de pacientes: {df['patient_id'].nunique()}")

# # ==============================
# # 2️⃣ K-FOLD (PACIENTE LEVEL)
# # ==============================

# pacientes = df["patient_id"].unique()
# paciente_label = df.groupby("patient_id")["label"].first().to_dict()

# labels = [paciente_label[p] for p in pacientes]

# kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

# # ==============================
# # 3️⃣ GERAR FOLDS
# # ==============================

# for fold, (train_idx, val_idx) in enumerate(kf.split(pacientes, labels)):

#     print(f"\n🔁 FOLD {fold}")

#     fold_dir = os.path.join(OUTPUT_DATASET_DIR, f"fold_{fold}")

#     pac_train = [pacientes[i] for i in train_idx]
#     pac_val   = [pacientes[i] for i in val_idx]

#     print(f"   🧠 Train pacientes: {len(pac_train)}")
#     print(f"   🔎 Val pacientes:   {len(pac_val)}")

#     # Criar estrutura
#     for split in ["train", "val"]:
#         for classe in classes.keys():
#             os.makedirs(
#                 os.path.join(fold_dir, split, classe),
#                 exist_ok=True
#             )

#     # Filtrar dados
#     df_train = df[df["patient_id"].isin(pac_train)]
#     df_val   = df[df["patient_id"].isin(pac_val)]

#     # Distribuição
#     print("   📊 Train:", Counter(df_train["label"]))
#     print("   📊 Test:  ", Counter(df_val["label"]))

#     # Função copiar
#     def copiar(df_split, split):
#         for _, row in df_split.iterrows():
#             origem = os.path.join(NODULES_DIR, row["path"])

#             destino = os.path.join(
#                 fold_dir,
#                 split,
#                 row["class_name"],
#                 row["patient_id"]
#             )

#             os.makedirs(destino, exist_ok=True)

#             shutil.copy2(
#                 origem,
#                 os.path.join(destino, os.path.basename(origem))
#             )

#     copiar(df_train, "train")
#     copiar(df_val, "val")

# print("\n✅ Dataset K-Fold (puro) criado com sucesso!")
# intersecao = set(pac_train) & set(pac_val)
# print("🚨 Interseção:", len(intersecao))
# # ==============================
# # TESTE VISUAL
# # ==============================

# for root, _, files in os.walk(OUTPUT_DATASET_DIR):
#     for file in files:
#         if file.endswith(".png"):
#             img = Image.open(os.path.join(root, file))
#             print(f"\n📏 Imagem exemplo: {img.size}")
#             exit()



