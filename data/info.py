import pandas as pd

# ==============================
# 1. CARREGAR OS ARQUIVOS
# ==============================

patient_df = pd.read_excel("Patient.xlsx")
label_df = pd.read_excel("Label.xlsx")

# ==============================
# 2. FILTRAR APENAS labels_type == 1
# ==============================

label_t1 = label_df[label_df["labels_type"] == 1].copy()

# ==============================
# 3. LIMPAR ESPAÇOS
# ==============================

patient_df["ID"] = patient_df["ID"].astype(str).str.strip()
label_t1["ID"] = label_t1["ID"].astype(str).str.strip()

# ==============================
# 4. MERGE
# ==============================

merged = label_t1.merge(
    patient_df[["ID", "de_type"]],
    on="ID",
    how="left"
)

# ==============================
# 5. NORMALIZAR SUBTIPO
# ==============================

merged["Subtipo"] = (
    merged["de_type"]
    .str.replace(",", "", regex=False)
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
)

# ==============================
# 6. MAPEAMENTO
# ==============================

mapeamento = {
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
    'Signet_ring_cell_carcinoma': 'Adenocarcinoma (NSCLC)',
    'Squamous_cell_carcinoma_NOS': 'Carcinoma Escamoso (NSCLC)',
    'Sq._cell_carcinoma_keratinizing_NOS': 'Carcinoma Escamoso (NSCLC)',
    'Sq._cell_carcinoma_lg._cell_non_ker': 'Carcinoma Escamoso (NSCLC)',
    'Basaloid_squamous_cell_carcinoma': 'Carcinoma Escamoso (NSCLC)',
    'Adenosquamous_carcinoma': 'Carcinoma Escamoso (NSCLC)',
    'Small_cell_carcinoma_NOS': 'Pequenas Celulas (SCLC)',
    'Combined_small_cell_carcinoma': 'Pequenas Celulas (SCLC)',
    'Oat_cell_carcinoma': 'Pequenas Celulas (SCLC)',
    'Non_small_cell_carcinoma': 'Outros NSCLC',
    'Large_cell_carcinoma_NOS': 'Outros NSCLC',
    'Neoplasm_malignant': 'Outros NSCLC',
    'Carcinoma_in_situ_NOS': 'Outros NSCLC',
    'Large_cell_neuroendocrine_carcinoma': 'Neuroendocrino',
    'Carcinoid_tumor_malignant': 'Neuroendocrino',
    'Neuroendocrine_carcinoma': 'Neuroendocrino',
    'Nao_Informado': 'Nao Informado'
}

merged["Macro"] = merged["Subtipo"].map(mapeamento).fillna("Nao Informado")

# ==============================
# 7. AGRUPAMENTO PRINCIPAL
# ==============================

resultado = (
    merged
    .groupby(["Macro", "Subtipo"])
    .agg(
        Pacientes=("ID", "nunique"),
        Nodulos_T1=("ID", "count")
    )
    .reset_index()
    .sort_values(["Macro", "Subtipo"])
)

# ==============================
# 8. TOTAL POR MACRO (opcional mas recomendado)
# ==============================

total_macro = (
    resultado
    .groupby("Macro")
    .agg(
        Pacientes=("Pacientes", "sum"),
        Nodulos_T1=("Nodulos_T1", "sum")
    )
    .reset_index()
)

total_macro["Subtipo"] = "TOTAL_MACRO"

# ==============================
# 9. TOTAL GERAL
# ==============================

total_geral = pd.DataFrame({
    "Macro": ["TOTAL_GERAL"],
    "Subtipo": [""],
    "Pacientes": [merged["ID"].nunique()],
    "Nodulos_T1": [len(merged)]
})

# ==============================
# 10. CONCATENAR TUDO
# ==============================

resultado_final = pd.concat(
    [resultado, total_macro, total_geral],
    ignore_index=True
)

# ==============================
# 11. SALVAR
# ==============================

resultado_final.to_excel("Resumo_Macro_Subtipo_T1.xlsx", index=False)

print("Arquivo gerado com sucesso: Resumo_Macro_Subtipo_T1.xlsx")
print(resultado_final)
