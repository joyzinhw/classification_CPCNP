import os
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import Counter
from PIL import Image

print("="*60)
print("DATASET CORRETO: TESTE FIXO + K-FOLD NO TREINO")
print("="*60)

NODULES_DIR = "NODULES"
OUTPUT_DATASET_DIR = "dataset_kfold"
K = 5
TEST_SPLIT = 0.15

classes = {
    'Adenocarcinoma (NSCLC)': 0,
    'Carcinoma Escamoso (NSCLC)': 1,
}


dados = []

print("\n📁 Varredura das pastas...")

for classe_nome, classe_id in classes.items():
    classe_path = os.path.join(NODULES_DIR, classe_nome)

    if not os.path.exists(classe_path):
        print(f"⚠️ Pasta não encontrada: {classe_path}")
        continue

    for root, _, files in os.walk(classe_path):
        for file in files:
            if file.endswith(".png"):
                paciente_id = file.split('_')[0]

                caminho_relativo = os.path.relpath(
                    os.path.join(root, file), NODULES_DIR
                )

                dados.append({
                    "patient_id": paciente_id,
                    "path": caminho_relativo,
                    "label": classe_id,
                    "class_name": classe_nome
                })

df = pd.DataFrame(dados)

print(f"\n📊 Total de nódulos: {len(df)}")
print(f"📊 Total de pacientes: {df['patient_id'].nunique()}")


pacientes = df["patient_id"].unique()
paciente_label = df.groupby("patient_id")["label"].first().to_dict()
labels = [paciente_label[p] for p in pacientes]

pac_train, pac_test, y_train, y_test = train_test_split(
    pacientes,
    labels,
    test_size=TEST_SPLIT,
    stratify=labels,
    random_state=42
)

print(f"\n🧪 TESTE FIXO:")
print(f"   Pacientes: {len(pac_test)}")

df_test = df[df["patient_id"].isin(pac_test)]
print("   📊 Test:", Counter(df_test["label"]))

kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(pac_train, y_train)):

    print(f"\n🔁 FOLD {fold}")

    fold_dir = os.path.join(OUTPUT_DATASET_DIR, f"fold_{fold}")

    pac_train_fold = [pac_train[i] for i in train_idx]
    pac_val_fold   = [pac_train[i] for i in val_idx]

    print(f"   🧠 Train pacientes: {len(pac_train_fold)}")
    print(f"   🔎 Val pacientes:   {len(pac_val_fold)}")

    # Criar estrutura
    for split in ["train", "val"]:
        for classe in classes.keys():
            os.makedirs(
                os.path.join(fold_dir, split, classe),
                exist_ok=True
            )


    df_train_fold = df[df["patient_id"].isin(pac_train_fold)]
    df_val_fold   = df[df["patient_id"].isin(pac_val_fold)]

    print("   📊 Train:", Counter(df_train_fold["label"]))
    print("   📊 Val:  ", Counter(df_val_fold["label"]))


    def copiar(df_split, split):
        for _, row in df_split.iterrows():
            origem = os.path.join(NODULES_DIR, row["path"])

            destino = os.path.join(
                fold_dir,
                split,
                row["class_name"],
                row["patient_id"]
            )

            os.makedirs(destino, exist_ok=True)

            shutil.copy2(
                origem,
                os.path.join(destino, os.path.basename(origem))
            )

    copiar(df_train_fold, "train")
    copiar(df_val_fold, "val")


    intersecao = set(pac_train_fold) & set(pac_val_fold)
    if len(intersecao) > 0:
        print("🚨 ERRO: Vazamento entre train e val!")


print("\n📦 Salvando TESTE final...")

test_dir = os.path.join(OUTPUT_DATASET_DIR, "test")

for classe in classes.keys():
    os.makedirs(os.path.join(test_dir, classe), exist_ok=True)

def copiar_teste(df_split):
    for _, row in df_split.iterrows():
        origem = os.path.join(NODULES_DIR, row["path"])

        destino = os.path.join(
            test_dir,
            row["class_name"],
            row["patient_id"]
        )

        os.makedirs(destino, exist_ok=True)

        shutil.copy2(
            origem,
            os.path.join(destino, os.path.basename(origem))
        )

copiar_teste(df_test)


for root, _, files in os.walk(OUTPUT_DATASET_DIR):
    for file in files:
        if file.endswith(".png"):
            img = Image.open(os.path.join(root, file))
            print(f"\n📏 Imagem exemplo: {img.size}")
            exit()