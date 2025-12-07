import os

# Ajuste os caminhos
base_dataset = "/home/joyzinhw/classification_CPCNP/datasett"
annotations_dir = "/home/joyzinhw/classification_CPCNP/BMP_2D/Annotations"

# Ler todas as anotações
annotations = {f.split(".")[0] for f in os.listdir(annotations_dir) if f.endswith(".xml")}

print("\n=========== RELATÓRIO COMPLETO ===========\n")

all_images = set()

for folder in ["0", "1", "2"]:
    folder_path = os.path.join(base_dataset, folder)

    if not os.path.exists(folder_path):
        print(f"[AVISO] Pasta não encontrada: {folder_path}")
        continue

    # Listar BMPs da pasta
    images = {f.split(".")[0] for f in os.listdir(folder_path) if f.endswith(".bmp")}
    all_images |= images  # acumula todas as imagens

    missing_xml = images - annotations

    print(f"\n📂 Pasta {folder}: {len(images)} imagens")
    print(f"❌ Imagens SEM XML ({len(missing_xml)}):")

    for name in sorted(missing_xml):
        print("   ", name)

# Agora procurar XMLs que não possuem imagem em NENHUMA pasta
extra_xml = annotations - all_images

print("\n===========================================")
print(f"\n❌ XMLs SEM imagem correspondente ({len(extra_xml)}):")
for name in sorted(extra_xml):
    print("   ", name)

print("\n✔ Verificação concluída.")
