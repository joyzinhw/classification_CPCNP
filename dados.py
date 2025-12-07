# # # import pandas as pd

# # # # Nome do arquivo de entrada (ajuste se necessário)
# # # arquivo_entrada = 'entrada.csv'

# # # # Ler o CSV original (sem cabeçalho)
# # # df = pd.read_csv(arquivo_entrada, header=None)

# # # # Função para extrair o número do paciente
# # # def extrair_paciente(valor):
# # #     if isinstance(valor, str):
# # #         return valor.split('_')[0]
# # #     return None

# # # # Aplicar a função para cada valor do DataFrame
# # # pacientes_df = df.applymap(extrair_paciente)

# # # # Remover valores duplicados em cada coluna e resetar o índice
# # # colunas_processadas = []
# # # for col in pacientes_df.columns:
# # #     unicos = pacientes_df[col].dropna().unique().tolist()
# # #     colunas_processadas.append(unicos)

# # # # Padronizar o tamanho (caso colunas tenham quantidades diferentes)
# # # max_len = max(len(c) for c in colunas_processadas)
# # # for i in range(len(colunas_processadas)):
# # #     colunas_processadas[i] += [None] * (max_len - len(colunas_processadas[i]))

# # # # Criar novo DataFrame com os pacientes únicos
# # # df_final = pd.DataFrame({
# # #     '0': colunas_processadas[0],
# # #     '1': colunas_processadas[1],
# # #     '2': colunas_processadas[2]
# # # })

# # # # Salvar em um novo CSV
# # # df_final.to_csv('dados.csv', index=False)

# # # print("✅ Arquivo 'dados.csv' criado com sucesso!")
# # import pandas as pd

# # # Nome do arquivo original
# # arquivo = 'entrada.csv'

# # # Ler o CSV sem cabeçalho e forçar leitura como texto
# # df = pd.read_csv(arquivo, header=None, dtype=str)

# # # Função para extrair apenas o número do paciente
# # def extrair_paciente(valor):
# #     if isinstance(valor, str) and "_" in valor:
# #         return valor.split("_")[0].zfill(4)
# #     return None

# # # Extrair IDs dos pacientes
# # pacientes = df.applymap(extrair_paciente)

# # # Criar conjuntos de IDs únicos por coluna
# # colunas = pacientes.columns
# # conjuntos = {col: set(pacientes[col].dropna()) for col in colunas}

# # # Comparar colunas entre si e achar interseções
# # repetidos = []
# # for i in range(len(colunas)):
# #     for j in range(i + 1, len(colunas)):
# #         intersec = conjuntos[colunas[i]].intersection(conjuntos[colunas[j]])
# #         if intersec:
# #             repetidos.append({
# #                 "colunas": f"{colunas[i]} x {colunas[j]}",
# #                 "pacientes": ", ".join(sorted(intersec))
# #             })

# # # Mostrar resultado
# # if repetidos:
# #     print("📋 Pacientes repetidos entre colunas:")
# #     for r in repetidos:
# #         print(f"{r['colunas']}: {r['pacientes']}")
# #     # Salvar em CSV
# #     pd.DataFrame(repetidos).to_csv("pacientes_repetidos.csv", index=False)
# #     print("\n💾 Arquivo 'pacientes_repetidos.csv' criado com os resultados.")
# # else:
# #     print("✅ Nenhum paciente repetido entre colunas.")

# import pandas as pd

# # Nome do arquivo de entrada
# arquivo_entrada = 'entrada.csv'

# # Ler o CSV original (sem cabeçalho)
# df = pd.read_csv(arquivo_entrada, header=None, dtype=str)

# # Função para extrair o número do paciente
# def extrair_paciente(valor):
#     if isinstance(valor, str) and "_" in valor:
#         return valor.split('_')[0].zfill(4)
#     return None

# # Aplicar a função para cada valor do DataFrame
# pacientes_df = df.applymap(extrair_paciente)

# # Montar lista de (paciente, tipo)
# linhas = []
# for col in pacientes_df.columns:
#     tipo = str(col)  # número da coluna original
#     unicos = pacientes_df[col].dropna().unique()
#     for paciente in sorted(unicos):
#         linhas.append({"paciente": paciente, "tipo": tipo})

# # Criar DataFrame final ordenado por paciente
# df_final = pd.DataFrame(linhas).sort_values(by="paciente").reset_index(drop=True)

# # Salvar em CSV
# df_final.to_csv("dados.csv", index=False, encoding="utf-8")

# print("✅ Arquivo 'dados.csv' criado com sucesso!")
# print(df_final.head(10))
#!/usr/bin/env python3
import pandas as pd
import os
import shutil
import re

# Caminhos base (ajuste conforme sua estrutura)
base_dir = os.path.abspath(".")  # diretório atual
csv_path = os.path.join(base_dir, "nodulos.csv")
img_dir = os.path.join(base_dir, "dataset", "fatia")
output_base = os.path.join(base_dir, "BMP_2D", "Agrupadas")

# Verificar se pasta de origem existe
if not os.path.exists(img_dir):
    raise FileNotFoundError(f"Pasta de imagens não encontrada: {img_dir}")

df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
print("Detected columns:", df.columns.tolist())

# Tentar detectar colunas automaticamente
cols_lc = [c.lower() for c in df.columns]
tipo_col = None
img_col = None

for name in ['tipo', 'type', 'class']:
    if name in cols_lc:
        tipo_col = df.columns[cols_lc.index(name)]
        break

for name in ['imagens', 'imagem', 'images', 'image']:
    if name in cols_lc:
        img_col = df.columns[cols_lc.index(name)]
        break

if tipo_col is None or img_col is None:
    if df.shape[1] >= 2:
        tipo_col = df.columns[1] if tipo_col is None else tipo_col
        img_col = df.columns[-1] if img_col is None else img_col
    else:
        raise ValueError("Não foi possível identificar as colunas 'tipo' e 'imagens'.")

print(f"Usando coluna de tipo: {tipo_col}")
print(f"Usando coluna de imagens: {img_col}")

bmp_pattern = re.compile(r'\d{4}_\d+\.(?:bmp|BMP)$')
def extrair_imagens(celula):
    if not isinstance(celula, str) or celula.strip() == "":
        return []
    parts = [p.strip() for p in re.split(r'[,\n;]+', celula) if p.strip()]
    valid = [p for p in parts if bmp_pattern.search(p)]
    return valid

os.makedirs(output_base, exist_ok=True)

for _, row in df.iterrows():
    tipo = str(row.get(tipo_col, "")).strip()
    imagens = extrair_imagens(row.get(img_col, ""))

    if not tipo or not imagens:
        continue

    try:
        tipo = str(int(float(tipo)))  # normalizar (1.0 → 1)
    except:
        pass

    pasta_dest = os.path.join(output_base, tipo)
    os.makedirs(pasta_dest, exist_ok=True)

    for img in imagens:
        origem = os.path.join(img_dir, img)
        destino = os.path.join(pasta_dest, img)

        if not os.path.exists(origem):
            print(f"⚠️ Arquivo não encontrado: {origem}")
            continue

        shutil.copy2(origem, destino)
        print(f"✅ Copiado: {img} → {pasta_dest}/")

print("\n✅ Finalizado! Imagens agrupadas em:", output_base)
