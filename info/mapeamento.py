import pandas as pd
import os
import shutil

arquivo_label = 'Label.xlsx'
arquivo_patient = 'Patient.xlsx'
caminho_origem_atual = './Dados_Organizados'
caminho_destino_macro = './Dados_Macro'


mapeamento = {
    # --- ADENOCARCINOMA ---
    'Acinar_cell_carcinoma': 'Adenocarcinoma (NSCLC)',
    'Adenocarcinoma_NOS': 'Adenocarcinoma (NSCLC)',
    'Adenocarcinoma_with_mixed_subtypes': 'Adenocarcinoma (NSCLC)',
    'Adenocarcinoma_with_squamous_metaplasia': 'Adenocarcinoma (NSCLC)',
    'Bronchiolo_alveolar_carcinoma_non_mucinous': 'Adenocarcinoma (NSCLC)',
    'Invasive_mucinous_adenocarcinoma': 'Adenocarcinoma (NSCLC)',
    'Lepidic_adenocarcinoma': 'Adenocarcinoma (NSCLC)',
    'Mixed_cell_adenocarcinoma': 'Adenocarcinoma (NSCLC)',
    'Mixed_invasive_mucinous_and_non_mucinous_adenocarcinoma': 'Adenocarcinoma (NSCLC)',
    'Mucin_producing_adenocarcinoma': 'Adenocarcinoma (NSCLC)',
    'Mucinous_adenocarcinoma': 'Adenocarcinoma (NSCLC)',
    'Papillary_adenocarcinoma_NOS': 'Adenocarcinoma (NSCLC)',

    'Squamous_cell_carcinoma_NOS': 'Carcinoma Escamoso (NSCLC)',
    'Sq._cell_carcinoma_keratinizing_NOS': 'Carcinoma Escamoso (NSCLC)',
    'Sq._cell_carcinoma_lg._cell_non_ker': 'Carcinoma Escamoso (NSCLC)',
    'Basaloid_squamous_cell_carcinoma': 'Carcinoma Escamoso (NSCLC)',

    # --- OUTROS NSCLC ---
    'Adenosquamous_carcinoma': 'Outros NSCLC',
    'Large_cell_carcinoma_NOS': 'Outros NSCLC',
    'Signet_ring_cell_carcinoma': 'Outros NSCLC',
    'Non_small_cell_carcinoma': 'Outros NSCLC',

    # --- SCLC ---
    'Small_cell_carcinoma_NOS': 'Pequenas Celulas (SCLC)',
    'Combined_small_cell_carcinoma': 'Pequenas Celulas (SCLC)',
    'Oat_cell_carcinoma': 'Pequenas Celulas (SCLC)',

    # --- NEUROENDOCRINO ---
    'Large_cell_neuroendocrine_carcinoma': 'Neuroendocrino',
    'Carcinoid_tumor_malignant': 'Neuroendocrino',
    'Neuroendocrine_carcinoma': 'Neuroendocrino',

    # --- INCERTOS ---
    'Neoplasm_malignant': 'Nao Informado',
    'Carcinoma_in_situ_NOS': 'Nao Informado',

   
    'Nao_Informado': 'Nao Informado'
}

# ==============================
# FUNÇÕES AUXILIARES (CORRIGIDAS)
# ==============================
def normalizar(tipo):
    if pd.isna(tipo):
        return 'Nao_Informado'

    tipo = str(tipo).strip()

    # 🔥 tratamento robusto
    if tipo.lower() in ["", "nan", "none"]:
        return 'Nao_Informado'

    return (tipo
            .replace(", ", "_")
            .replace(",", "")
            .replace(" ", "_")
            .replace("/", "-")
            .replace("-", "_"))

def mapear(tipo):
    tipo_norm = normalizar(tipo)

    # 🔥 fallback seguro
    if tipo_norm not in mapeamento:
        print(f"[WARN] Tipo não mapeado: {tipo_norm}")
        return 'Nao Informado'

    return mapeamento[tipo_norm]

# ==============================
# CARREGAR DADOS
# ==============================
df_l = pd.read_excel(arquivo_label)
df_p = pd.read_excel(arquivo_patient)

df_l.columns = df_l.columns.str.strip()
df_p.columns = df_p.columns.str.strip()

# ==============================
# LIMPAR IDs
# ==============================
df_p['ID'] = df_p['ID'].astype(str).str.replace(".0", "").str.strip()
df_l['ID'] = df_l['ID'].astype(str).str.replace(".0", "").str.strip()

# ==============================
# MAPEAR MACRO (COM SEGURANÇA)
# ==============================
df_p['Macro'] = df_p['de_type'].apply(mapear)

# ==============================
# DEBUG: TIPOS NÃO MAPEADOS
# ==============================
print("\n🔎 Verificando tipos não mapeados...")

tipos_unicos = df_p['de_type'].unique()
nao_mapeados = []

for t in tipos_unicos:
    t_norm = normalizar(t)
    if t_norm not in mapeamento:
        nao_mapeados.append(t_norm)

if nao_mapeados:
    print("\n⚠️ Tipos NÃO mapeados encontrados:")
    for t in nao_mapeados:
        print(f" - {t}")
else:
    print("✅ Todos os tipos estão mapeados!")

# ==============================
# ORGANIZAR PASTAS
# ==============================
os.makedirs(caminho_destino_macro, exist_ok=True)

print("\n📁 Movendo arquivos para categorias Macro...")

for pasta_subtipo in os.listdir(caminho_origem_atual):

    subtipo_path = os.path.join(caminho_origem_atual, pasta_subtipo)

    if not os.path.isdir(subtipo_path):
        continue

    categoria_macro = mapear(pasta_subtipo)

    nova_pasta_macro = os.path.join(caminho_destino_macro, categoria_macro)
    os.makedirs(nova_pasta_macro, exist_ok=True)

    for paciente_id in os.listdir(subtipo_path):

        paciente_id = str(paciente_id).replace(".0", "").strip()

        origem = os.path.join(subtipo_path, paciente_id)
        destino = os.path.join(nova_pasta_macro, paciente_id)

        if not os.path.exists(destino):
            shutil.move(origem, destino)
        else:
            print(f"[INFO] Paciente {paciente_id} já existe, pulando...")

# ==============================
# ESTATÍSTICAS
# ==============================
df_merge = pd.merge(df_l, df_p[['ID', 'Macro']], on='ID', how='left')

# 🔥 garante que faltantes virem Nao Informado
df_merge['Macro'] = df_merge['Macro'].fillna('Nao Informado')

resumo_nodulos = df_merge.groupby('Macro').size()
resumo_pacientes = df_merge.groupby('Macro')['ID'].nunique()

# ==============================
# PRINT
# ==============================
print("\n" + "="*70)
print(f"{'Categoria Macro':<30} | {'Pacientes':<12} | {'Nódulos':<12}")
print("-"*70)

for cat in resumo_nodulos.index:
    print(f"{cat:<30} | {resumo_pacientes[cat]:<12} | {resumo_nodulos[cat]:<12}")

print("-"*70)
print(f"{'TOTAL':<30} | {resumo_pacientes.sum():<12} | {resumo_nodulos.sum():<12}")
print("="*70)

# ==============================
# VALIDAÇÃO FINAL
# ==============================
print("\n🔎 VALIDAÇÃO FINAL")

print(f"Pacientes original: {df_p['ID'].nunique()}")
print(f"Pacientes macro:    {resumo_pacientes.sum()}")

print(f"Nódulos original:   {len(df_l)}")
print(f"Nódulos macro:      {resumo_nodulos.sum()}")

assert resumo_pacientes.sum() == df_p['ID'].nunique(), "❌ ERRO: Pacientes não batem!"
assert resumo_nodulos.sum() == len(df_l), "❌ ERRO: Nódulos não batem!"

print("✅ Tudo consistente! Nenhum dado perdido.")