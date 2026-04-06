import pandas as pd
import os
import shutil

arquivo_label = 'Label.xlsx'
arquivo_patient = 'Patient.xlsx'
caminho_origem = './data'
caminho_destino = './Dados_Organizados'

try:
    df_labels = pd.read_excel(arquivo_label)
    df_patient = pd.read_excel(arquivo_patient)
    df_labels.columns = df_labels.columns.str.strip()
    df_patient.columns = df_patient.columns.str.strip()
    df = pd.merge(df_labels, df_patient[['ID', 'de_type']], on='ID', how='left')
except Exception as e:
    print(f"Erro ao carregar ficheiros: {e}")
    exit()

ids_com_malignidade = df[df['labels_type'] == 1]['ID'].unique()

print(f"Total de pacientes com nódulo Tipo 1: {len(ids_com_malignidade)}")
print("A iniciar a transferência física dos dados...")
print("-" * 50)

for paciente_id in ids_com_malignidade:
    dados_paciente = df[df['ID'] == paciente_id].iloc[0]
    
    tipo_raw = str(dados_paciente['de_type']) if pd.notna(dados_paciente['de_type']) else "Nao_Informado"
    tipo_cancer = tipo_raw.replace(", ", "_").replace(" ", "_").replace("/", "-")
    
    pasta_tipo = os.path.join(caminho_destino, tipo_cancer)
    if not os.path.exists(pasta_tipo):
        os.makedirs(pasta_tipo, exist_ok=True)
    
    p_id_str = str(paciente_id)
    origem = os.path.join(caminho_origem, p_id_str)
    destino = os.path.join(pasta_tipo, p_id_str)
    
    if os.path.exists(origem):
        try:
            shutil.move(origem, destino)
            print(f"Transferido: ID {p_id_str} -> {tipo_cancer}")
        except Exception as e:
            print(f"Erro ao mover {p_id_str}: {e}")

print("\n" + "="*50)
print("TRANSFERÊNCIA CONCLUÍDA")
print("="*50)
df_final = df[df['ID'].isin(ids_com_malignidade)].drop_duplicates('ID')
contagem = df_final['de_type'].value_counts(dropna=False)

for tipo, qtd in contagem.items():
    print(f"Pasta: {str(tipo):<40} | Qtd: {qtd}")

print("-" * 50)
print(f"TOTAL DE PACIENTES TRANSFERIDOS: {len(ids_com_malignidade)}")
print("Nota: As pastas foram movidas de 'data' para 'Dados_Organizados'.")
