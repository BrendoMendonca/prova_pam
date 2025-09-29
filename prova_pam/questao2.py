"""
Prova - Introdução a Inteligência Artificial - UFPB
Brendo de Almeida Mendonça - 20190172521
QUESTÃO 2:

Pré-processamento dos dados
Análise com o Teste do Cotovelo para encontrar o 'k' ideal para o K-Means
Execução do K-Means e do Agrupamento Hierárquico com 2 linkages
Avaliação e comparação dos resultados com métricas de clusterização e visualização com PCA
"""

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

#módulos do Scikit-learn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA


#PRÉ-PROCESSAMENTO E CARREGAMENTO DOS DADOS

def carregar_e_processar_dados(caminho_csv, caminho_json_folder):
    """
    Carrega os dados do arquivo CSV e processa cada arquivo JSON correspondente
    para extrair features estatísticas.
    """
    df_manifest = pd.read_csv(caminho_csv)
    df_manifest.dropna(subset=['file_name'], inplace=True)
    df_manifest.reset_index(drop=True, inplace=True)

    lista_features = []
    lista_rotulos = []

    print("Iniciando pré-processamento dos arquivos JSON...")
    for idx, row in tqdm(df_manifest.iterrows(), total=df_manifest.shape[0]):
        nome_arquivo_json = row['file_name'].strip()
        caminho_completo_json = os.path.join(caminho_json_folder, nome_arquivo_json)

        if not os.path.exists(caminho_completo_json):
            continue

        try:
            with open(caminho_completo_json, 'r', encoding='utf-8') as f:
                dados_json = json.load(f)

            todos_keypoints = []
            for frame in dados_json['frames']:
                #extrai as coordenadas x e y de cada keypoint
                frame_keypoints = [kp.get(coord, 0) for kp in frame['keypoints'] for coord in ('x', 'y')]
                '''#garante que cada frame tenha o mesmo número de features (33 keypoints*2 coords= 66)
                se um frame estiver incompleto, ele será ignorado '''
                if len(frame_keypoints) == 66:
                    todos_keypoints.append(frame_keypoints)

            #se, após filtrar os frames, não sobrar nenhum dado válido, pula o arquivo
            if not todos_keypoints:
                continue

            np_keypoints = np.array(todos_keypoints)
            
            #cálculos estatísticos-features
            media = np.mean(np_keypoints, axis=0)
            desvio_padrao = np.std(np_keypoints, axis=0)
            minimo = np.min(np_keypoints, axis=0)
            maximo = np.max(np_keypoints, axis=0)
            
            vetor_features = np.concatenate([media, desvio_padrao, minimo, maximo])
            
            lista_features.append(vetor_features)
            lista_rotulos.append(row['sinal'])

        except Exception as e:#tratamento caso tenha algo errado com o json
            print(f"Erro ao processar o arquivo {nome_arquivo_json}: {e}")

    return np.array(lista_features), np.array(lista_rotulos)


#recebendo caminho dos arquivos
base_path = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(base_path, 'sinais.csv')
json_folder_path = os.path.join(base_path, 'sinais')

#chamada da função para carregar e processar os dados
X, y_true_labels = carregar_e_processar_dados(csv_file_path, json_folder_path)

#padronização dos dados: Essencial para algoritmos baseados em distância como K-Means.
print("\nPadronizando as features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Pré-processamento concluído. {X.shape[0]} amostras processadas.")

# K-MEANS E TESTE DO COTOVELO (ELBOW METHOD)

print("\nExecutando o Teste do Cotovelo para encontrar o 'k' ideal...")
inertia_values = []
k_range = range(2, 21) #teste de clusters

for k in tqdm(k_range):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

#plotagem do gráfico do teste do cotovelo
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_values, 'bo-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia (Within-cluster sum of squares)')
plt.title('Teste do Cotovelo (Elbow Method) para K-Means')
plt.xticks(k_range)
plt.grid(True)
plt.show()

#ANÁLISE DO COTOVELO E EXECUÇÃO FINAL DO K-MEANS
"""om base no gráfico, o "cotovelo" (ponto de inflexão) parece estar entre 4 e 8
escolhendo 7 para k"""
k_otimo = 7
print(f"\nValor de 'k' escolhido a partir do gráfico: {k_otimo}")

kmeans_final = KMeans(n_clusters=k_otimo, init='k-means++', random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_scaled)


#AGRUPAMENTO HIERÁRQUICO
#O Agrupamento Hierárquico é executado com o mesmo k_otimo e dois métodos de linkage

"""Método de Linkage 1: 'ward'
Minimiza a variância dentro dos clusters que estão sendo mesclados"""

print("\nExecutando Agrupamento Hierárquico com linkage='ward'...")
hierarchical_ward = AgglomerativeClustering(n_clusters=k_otimo, linkage='ward')
hierarchical_ward_labels = hierarchical_ward.fit_predict(X_scaled)

""" Método de Linkage 2: 'complete'
 Utiliza a distância máxima entre as observações de dois clusters"""
print("Executando Agrupamento Hierárquico com linkage='complete'...")
hierarchical_complete = AgglomerativeClustering(n_clusters=k_otimo, linkage='complete')
hierarchical_complete_labels = hierarchical_complete.fit_predict(X_scaled)


#AVALIAÇÃO E COMPARAÇÃO

# Como é um problema não supervisionado, foram utilizadas métricas que não dependem dos rótulos verdadeiros

resultados_clustering = {
    "K-Means": kmeans_labels,
    "Hierárquico (Ward)": hierarchical_ward_labels,
    "Hierárquico (Complete)": hierarchical_complete_labels
}

metricas = []
for nome, labels in resultados_clustering.items():
    sil_score = silhouette_score(X_scaled, labels)
    calinski_score = calinski_harabasz_score(X_scaled, labels)
    davies_score = davies_bouldin_score(X_scaled, labels)
    metricas.append({
        "Algoritmo": nome,
        "Silhouette Score (↑)": sil_score,
        "Calinski-Harabasz (↑)": calinski_score,
        "Davies-Bouldin (↓)": davies_score
    })
    
resultados_df = pd.DataFrame(metricas)
print("\n--- Comparação das Métricas de Clusterização ---")
print(resultados_df.to_string(index=False))
print("(↑) Quanto maior, melhor. (↓) Quanto menor, melhor.")

"""Visualização dos Clusters com PCA ---
foi feita uma dimensionalidade para 2D para poder visualizar os clusters"""
print("\nReduzindo dimensionalidade com PCA para visualização...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

#plotagem dos resultados
fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)
fig.suptitle(f'Visualização dos Clusters (k={k_otimo}) após Redução com PCA', fontsize=16)

for i, (nome, labels) in enumerate(resultados_clustering.items()):
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=50, alpha=0.7, ax=axes[i], legend='full')
    axes[i].set_title(nome)
    axes[i].set_xlabel('Componente Principal 1')
    axes[i].set_ylabel('Componente Principal 2')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

"""Análise Cruzada com os Rótulos Verdadeiros
apesar do treinamento não ser supervisionado, foram utilizados rótulos verdadeiros
para ver como os clusters se alinham com as classes reais"""
print("\n--- Análise Cruzada: Rótulos Verdadeiros vs. Clusters do K-Means ---")
df_analise = pd.DataFrame({'Rótulo Verdadeiro': y_true_labels, 'Cluster K-Means': kmeans_labels})
tabela_cruzada = pd.crosstab(df_analise['Rótulo Verdadeiro'], df_analise['Cluster K-Means'])
print(tabela_cruzada)