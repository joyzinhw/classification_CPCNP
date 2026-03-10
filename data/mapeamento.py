import pandas as pd
import os
import shutil

# --- CONFIGURAÇÕES ---
arquivo_label = 'Label.xlsx'
arquivo_patient = 'Patient.xlsx'
caminho_origem_atual = './Dados_Organizados' 
caminho_destino_macro = './Dados_Macro'

# --- MAPEAMENTO ---
mapeamento = {
    'Acinar_cell_carcinoma': 'Adenocarcinoma (NSCLC)',
    'Adenocarcinoma_NOS': 'Adenocarcinoma (NSCLC)',
    'Adenocarcinoma_with_mixed_subtypes': 'Adenocarcinoma (NSCLC)',
    'Adenocarcinoma_with_squamous_metaplasia': 'Adenocarcinoma (NSCLC)',
    'Bronchiolo-alveolar_carcinoma_non-mucinous': 'Adenocarcinoma (NSCLC)',
    'Invasive_mucinous_adenocarcinoma': 'Adenocarcinoma (NSCLC)',
    'Lepidic_adenocarcinoma': 'Adenocarcinoma (NSCLC)',
    'Mixed_cell_adenocarcinoma': 'Adenocarcinoma (NSCLC)',
    'Mixed_invasive_mucinous_and_non-mucinous_adenocarcinoma': 'Adenocarcinoma (NSCLC)',
    'Mucin-producing_adenocarcinoma': 'Adenocarcinoma (NSCLC)',
    'Mucinous_adenocarcinoma': 'Adenocarcinoma (NSCLC)',
    'Papillary_adenocarcinoma_NOS': 'Adenocarcinoma (NSCLC)',
    'Signet_ring_cell_carcinoma': 'Adenocarcinoma (NSCLC)',
    'Squamous_cell_carcinoma_NOS': 'Carcinoma Escamoso (NSCLC)',
    'Sq._cell_carcinoma_keratinizing_NOS': 'Carcinoma Escamoso (NSCLC)',
    'Sq._cell_carcinoma_lg._cell_non-ker': 'Carcinoma Escamoso (NSCLC)',
    'Basaloid_squamous_cell_carcinoma': 'Carcinoma Escamoso (NSCLC)',
    'Adenosquamous_carcinoma': 'Carcinoma Escamoso (NSCLC)',
    'Small_cell_carcinoma_NOS': 'Pequenas Celulas (SCLC)',
    'Combined_small_cell_carcinoma': 'Pequenas Celulas (SCLC)',
    'Oat_cell_carcinoma': 'Pequenas Celulas (SCLC)',
    'Non-small_cell_carcinoma': 'Outros NSCLC',
    'Large_cell_carcinoma_NOS': 'Outros NSCLC',
    'Neoplasm_malignant': 'Outros NSCLC',
    'Carcinoma_in_situ_NOS': 'Outros NSCLC',
    'Large_cell_neuroendocrine_carcinoma': 'Neuroendocrino',
    'Carcinoid_tumor_malignant': 'Neuroendocrino',
    'Neuroendocrine_carcinoma': 'Neuroendocrino',
    'Nao_Informado': 'Nao Informado'
}

# --- CARREGAR DADOS ---
df_l = pd.read_excel(arquivo_label)
df_p = pd.read_excel(arquivo_patient)
df_l.columns = df_l.columns.str.strip()
df_p.columns = df_p.columns.str.strip()

# --- PADRONIZAR E MAPEAR MACRO ---
df_p['Macro'] = (df_p['de_type']
                 .str.replace(", ", "_")
                 .str.replace(" ", "_")
                 .str.replace("/", "-")
                 .map(mapeamento))

# --- CRIAR PASTAS DESTINO ---
os.makedirs(caminho_destino_macro, exist_ok=True)

print("Iniciando transferência para pastas Macro...")
for pasta_subtipo in os.listdir(caminho_origem_atual):
    subtipo_path = os.path.join(caminho_origem_atual, pasta_subtipo)
    
    if os.path.isdir(subtipo_path):
        categoria_macro = mapeamento.get(pasta_subtipo, "Outros")
        nova_pasta_macro = os.path.join(caminho_destino_macro, categoria_macro)
        os.makedirs(nova_pasta_macro, exist_ok=True)
        
        for paciente_id in os.listdir(subtipo_path):
            origem = os.path.join(subtipo_path, paciente_id)
            destino = os.path.join(nova_pasta_macro, paciente_id)
            
            if not os.path.exists(destino):
                shutil.move(origem, destino)
            else:
                print(f"Paciente {paciente_id} já existe em {nova_pasta_macro}, pulando...")

# --- ESTATÍSTICAS ---
df_merge = pd.merge(df_l, df_p[['ID', 'Macro']], on='ID')
resumo_nodulos = df_merge.groupby('Macro').size()
resumo_pacientes = df_merge.groupby('Macro')['ID'].nunique()

print("\n" + "="*70)
print(f"{'Categoria Macro':<30} | {'Pacientes':<12} | {'Nódulos T1':<12}")
print("-"*70)
for cat in resumo_nodulos.index:
    print(f"{cat:<30} | {resumo_pacientes[cat]:<12} | {resumo_nodulos[cat]:<12}")
print("-"*70)
print(f"{'TOTAL':<30} | {resumo_pacientes.sum():<12} | {resumo_nodulos.sum():<12}")
print("="*70)
