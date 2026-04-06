import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

print("="*60)
print("PREPARANDO DATASET 2D EM PASTAS (TRAIN / VAL / TEST)")
print("="*60)


NODULES_DIR = "NODULES"
OUTPUT_DATASET_DIR = "dataset"

classes = {
    'Adenocarcinoma (NSCLC)': 0,
    'Carcinoma Escamoso (NSCLC)': 1,
     'Outros NSCLC': 2
}

dados = []
pacientes_por_classe = set()


print("\n📁 Varredura das pastas...")

for classe_nome, classe_id in classes.items():
    classe_path = os.path.join(NODULES_DIR, classe_nome)

    if not os.path.exists(classe_path):
        print(f"⚠️ Pasta não encontrada: {classe_path}")
        continue

    total_classe = 0

    for root, _, files in os.walk(classe_path):
        for file in files:
            if file.endswith(".png"):
                paciente_id = file.split('_')[0]
                caminho_relativo = os.path.relpath(
                    os.path.join(root, file), NODULES_DIR
                )

                pacientes_por_classe.add((classe_nome, paciente_id))
                total_classe += 1

                dados.append({
                    "patient_id": paciente_id,
                    "nodule_path": caminho_relativo,
                    "histology": classe_nome,
                    "Malignant_lbl": classe_id
                })

    print(f"📁 {classe_nome}: {total_classe} nódulos")

df = pd.DataFrame(dados)

print(f"\n📊 Total de nódulos: {len(df)}")
print(f"📊 Total de pacientes: {df['patient_id'].nunique()}")


pacientes_unicos = df["patient_id"].unique()
paciente_to_label = df.groupby("patient_id")["Malignant_lbl"].first().to_dict()

pacientes_train, pacientes_temp = train_test_split(
    pacientes_unicos,
    test_size=0.30,
    random_state=42,
    stratify=[paciente_to_label[p] for p in pacientes_unicos]
)

pacientes_val, pacientes_test = train_test_split(
    pacientes_temp,
    test_size=0.50,
    random_state=42,
    stratify=[paciente_to_label[p] for p in pacientes_temp]
)

split_map = {}
for p in pacientes_train:
    split_map[p] = "train"
for p in pacientes_val:
    split_map[p] = "val"
for p in pacientes_test:
    split_map[p] = "test"

df["split"] = df["patient_id"].map(split_map)


print("\n📂 Criando estrutura de pastas...")

for split in ["train", "val", "test"]:
    for classe in classes.keys():
        os.makedirs(
            os.path.join(OUTPUT_DATASET_DIR, split, classe),
            exist_ok=True
        )


print("\n🖼️ Copiando imagens...")

for _, row in df.iterrows():
    split = row["split"]
    classe = row["histology"]
    paciente = row["patient_id"]

    origem = os.path.join(NODULES_DIR, row["nodule_path"])

    destino_paciente = os.path.join(
        OUTPUT_DATASET_DIR,
        split,
        classe,
        paciente
    )

    os.makedirs(destino_paciente, exist_ok=True)

    destino = os.path.join(destino_paciente, os.path.basename(origem))
    shutil.copy2(origem, destino)

print("\n✅ Dataset salvo com sucesso em pastas físicas!")


for root, _, files in os.walk(OUTPUT_DATASET_DIR):
    for file in files:
        if file.endswith(".png"):
            img = Image.open(os.path.join(root, file))
            print(f"\n📏 Imagem exemplo carregada: {img.size}")
            exit()

