import os
import pandas as pd

# ==============================
# CONFIG
# ==============================
caminho_macro = './Dados_Macro'
arquivo_patient = 'dataset_info_excel/Patient.xlsx'
arquivo_label = 'dataset_info_excel/Label.xlsx'
saida = 'Resumo_Macro_REAL_T1.xlsx'

# ==============================
# FUNÇÃO DE LIMPEZA DE ID
# ==============================
def limpar_id(x):
    return str(x).replace(".0", "").strip()

# ==============================
# CARREGAR DADOS
# ==============================
patient_df = pd.read_excel(arquivo_patient)
label_df = pd.read_excel(arquivo_label)

# ==============================
# FILTRAR APENAS T1
# ==============================
label_t1 = label_df[label_df["labels_type"] == 1].copy()

# ==============================
# LIMPAR IDs
# ==============================
patient_df["ID"] = patient_df["ID"].apply(limpar_id)
label_t1["ID"] = label_t1["ID"].apply(limpar_id)

# ==============================
# CONTAR NÓDULOS T1 POR PACIENTE
# ==============================
nodulos_por_id = label_t1.groupby("ID").size().to_dict()

# ==============================
# MAPA ID → SUBTIPO
# ==============================
id_para_tipo = dict(zip(patient_df["ID"], patient_df["de_type"]))

# ==============================
# COLETAR DADOS DAS PASTAS
# ==============================
dados = []
ids_pastas = set()
ids_sem_tipo = []
ids_nao_encontrados = []

for macro in os.listdir(caminho_macro):

    caminho_macro_dir = os.path.join(caminho_macro, macro)

    if not os.path.isdir(caminho_macro_dir):
        continue

    for paciente_id in os.listdir(caminho_macro_dir):

        paciente_id = limpar_id(paciente_id)
        ids_pastas.add(paciente_id)

        caminho_paciente = os.path.join(caminho_macro_dir, paciente_id)

        if not os.path.isdir(caminho_paciente):
            continue

        tipo = id_para_tipo.get(paciente_id)

        # 🔥 TRATAMENTO ROBUSTO DE FALTANTES
        if pd.isna(tipo) or str(tipo).strip().lower() in ["", "nan", "none"]:
            tipo = "Nao Informado"
            ids_sem_tipo.append(paciente_id)

        # 🔥 SE NÃO EXISTE NO EXCEL
        if paciente_id not in id_para_tipo:
            tipo = "Nao Informado"
            ids_nao_encontrados.append(paciente_id)

        # 🔥 NÚMERO REAL DE NÓDULOS T1
        num_nodulos = nodulos_por_id.get(paciente_id, 0)

        dados.append({
            "Macro": macro,
            "Subtipo": tipo,
            "ID": paciente_id,
            "Nodulos_T1": num_nodulos
        })

# ==============================
# DATAFRAME
# ==============================
df = pd.DataFrame(dados)

# ==============================
# DEBUG COMPLETO
# ==============================
print("\n🔎 DEBUG DE CONSISTÊNCIA")
print(f"Pacientes nas pastas: {len(ids_pastas)}")
print(f"Pacientes únicos coletados: {df['ID'].nunique()}")
print(f"Pacientes no Patient.xlsx: {patient_df['ID'].nunique()}")

print(f"\n⚠️ Pacientes sem subtipo: {len(ids_sem_tipo)}")
print(f"⚠️ Pacientes não encontrados no Excel: {len(ids_nao_encontrados)}")

# Mostrar alguns exemplos
if ids_sem_tipo:
    print("\nExemplo sem subtipo:", ids_sem_tipo[:5])

if ids_nao_encontrados:
    print("\nExemplo não encontrados:", ids_nao_encontrados[:5])

# ==============================
# AGRUPAMENTO
# ==============================
resumo = df.groupby(['Macro', 'Subtipo']).agg(
    Pacientes=('ID', 'nunique'),
    Nodulos_T1=('Nodulos_T1', 'sum')
).reset_index()

# ==============================
# TOTAL POR MACRO
# ==============================
totais_macro = resumo.groupby('Macro').agg(
    Pacientes=('Pacientes', 'sum'),
    Nodulos_T1=('Nodulos_T1', 'sum')
).reset_index()

totais_macro['Subtipo'] = 'TOTAL_MACRO'

# ==============================
# TOTAL GERAL
# ==============================
total_geral = pd.DataFrame([{
    'Macro': 'TOTAL_GERAL',
    'Subtipo': '',
    'Pacientes': resumo['Pacientes'].sum(),
    'Nodulos_T1': resumo['Nodulos_T1'].sum()
}])

# ==============================
# CONCATENAR
# ==============================
df_final = pd.concat([resumo, totais_macro, total_geral], ignore_index=True)

# ==============================
# ORGANIZAR
# ==============================
df_final = df_final.sort_values(by=['Macro', 'Subtipo'])

# ==============================
# SALVAR
# ==============================
df_final.to_excel(saida, index=False)

# ==============================
# PRINT FINAL
# ==============================
print("\n✅ Arquivo gerado com sucesso:", saida)
print(df_final)

# ==============================
# VALIDAÇÃO FINAL (CRÍTICA)
# ==============================
print("\n🔎 VALIDAÇÃO FINAL")
print(f"Pacientes únicos (pastas): {df['ID'].nunique()}")
print(f"Nódulos T1 total: {df['Nodulos_T1'].sum()}")


