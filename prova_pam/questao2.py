# -*- coding: utf-8 -*-
"""
PROVA DE PARADIGMAS DE APRENDIZAGEM DE MÁQUINA
Brendo de Almeida Mendonça
Questão 2

Este script é focado em Aprendizagem Não Supervisionada:
Carrega os dados de sinais em LIBRAS a partir de arquivos JSON,
normaliza os pontos-chave para garantir invariância à escala e posição, e extrai
features estatísticas para representar cada sinal como um vetor numérico

Utiliza o Teste do Cotovelo para determinar um
número ideal de clusters k para o conjunto de dados.

Aplica o algoritmo K-Means com o valor de k encontrado.
Aplica o algoritmo de Agrupamento Hierárquico com o mesmo k, testando
dois métodos de linkage - ward e complete para comparação

Avalia a qualidade dos clusters formados usando métricas próprias para
aprendizagem não supervisionada-Silhouette, Calinski-Harabasz, Davies-Bouldin
Visualiza os clusters em 2D usando a técnica de redução de dimensionalidade PCA
Realiza uma análise qualitativa cruzando os clusters encontrados com os
ótulos verdadeiros para interpretar o significado dos grupos
"""

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm #usada para criar barras de progresso visuais


import matplotlib.pyplot as plt
import seaborn as sns

#módulos do Scikit-learn para pré-processamento, modelagem e avaliação
from sklearn.preprocessing import StandardScaler, LabelEncoder #padronização de features e codificação de rótulos
from sklearn.cluster import KMeans, AgglomerativeClustering #lgoritmos de clusterização
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score #métricas de avaliação de clusters
from sklearn.decomposition import PCA #para redução de dimensionalidade e visualização

#PRÉ-PROCESSAMENTO E CARREGAMENTO DOS DADOS

def carregar_e_processar_dados(caminho_csv, caminho_json_folder):
    """
    Função principal para carregar e processar todos os dados.
    A função lê o arquivo CSV principal, itera sobre cada arquivo JSON,
    realiza a normalização dos pontos-chave, extrai features estatísticas
    e retorna os dados prontos para o treinamento

    Argumentos:
        caminho_csv (str):caminho para o arquivo 'sinais.csv'
        caminho_json_folder (str): caminho para a pasta contendo os arquivos json

    Retorna:
        np.array: Matriz de features (X)
        np.array: Vetor de rótulos (y)
    """
    #lê o arquivo CSV que serve como um índice para os arquivos JSON
    df_manifest = pd.read_csv(caminho_csv)
    #remove linhas que possam ter nomes de arquivos ausentes
    df_manifest.dropna(subset=['file_name'], inplace=True)
    df_manifest.reset_index(drop=True, inplace=True)

    #listas para armazenar os dados processados
    lista_features = []
    lista_rotulos = []

    print("Iniciando pré-processamento dos arquivos JSON...")
     #a barra de progresso (tqdm) dá um feedback visual do processo
    for idx, row in tqdm(df_manifest.iterrows(), total=df_manifest.shape[0]):
        nome_arquivo_json = row['file_name'].strip()
        caminho_completo_json = os.path.join(caminho_json_folder, nome_arquivo_json)

        #se o arquivo JSON não existir, pula para o próximo
        if not os.path.exists(caminho_completo_json):
            continue

        try:
            with open(caminho_completo_json, 'r', encoding='utf-8') as f:
                dados_json = json.load(f)

            todos_keypoints_normalizados = []
            """itera sobre cada frame do vídeo
            para cada frame, normalizamos os pontos antes de extrair features
            """
            for frame in dados_json['frames']:
                keypoints_dict = {kp['id']: kp for kp in frame['keypoints']}
                
                """o objetivo é fazer com que o modelo de foque apenas no gesto e ignore a posição e a escala do interprete no vídeo
                para isso, foram usados os ombros nos pontos 11 e 12 como referência"""
                if 11 in keypoints_dict and 12 in keypoints_dict:
                    ombro_direito_x, ombro_direito_y = keypoints_dict[11]['x'], keypoints_dict[11]['y']
                    ombro_esquerdo_x, ombro_esquerdo_y = keypoints_dict[12]['x'], keypoints_dict[12]['y']

                    #calcula um ponto central do tronco-meio dos ombros
                    centro_tronco_x = (ombro_direito_x + ombro_esquerdo_x) / 2
                    centro_tronco_y = (ombro_direito_y + ombro_esquerdo_y) / 2

                     #calcula uma escala-distância entre os ombros para normalizar o tamanho do esqueleto
                    escala_tronco = np.sqrt((ombro_direito_x - ombro_esquerdo_x)**2 + (ombro_direito_y - ombro_esquerdo_y)**2)
                    if escala_tronco == 0: continue#evita divisão por zero

                    frame_features = []
                    #itera sobre todos os 33 pontos-chave possíveis
                    for i in range(33):
                        if i in keypoints_dict:
                            kp = keypoints_dict[i]
                            #normaliza cada ponto: subtrai o centro e divide pela escala
                            x_norm = (kp.get('x', 0) - centro_tronco_x) / escala_tronco
                            y_norm = (kp.get('y', 0) - centro_tronco_y) / escala_tronco
                            frame_features.extend([x_norm, y_norm])
                        else:
                            #se um ponto-chave não foi detectado no frame, preenchemos com zeros
                            frame_features.extend([0.0, 0.0])

                    #garante que o frame processado tem o tamanho esperado (66 coordenadas)
                    if len(frame_features) == 66:
                        todos_keypoints_normalizados.append(frame_features)
            #se nenhum frame válido foi encontrado no arquivo, pula para o próximo
            if not todos_keypoints_normalizados:
                continue

            np_keypoints = np.array(todos_keypoints_normalizados)
            
            #ENGENHARIA DE FEATURES
            """transforma múltiplos frames em um único vetor de features de tamanho fixo
            calculando estatísticas para cada coordenada ao longo do tempo"""
            media = np.mean(np_keypoints, axis=0)
            desvio_padrao = np.std(np_keypoints, axis=0)
            minimo = np.min(np_keypoints, axis=0)
            maximo = np.max(np_keypoints, axis=0)
            
            #concatena todas as estatísticas em um único vetor de features para esta amostra
            vetor_features = np.concatenate([media, desvio_padrao, minimo, maximo])
            
            #adiciona os dados processados às listas
            lista_features.append(vetor_features)
            lista_rotulos.append(row['sinal'])

        except Exception as e:
            print(f"Erro ao processar o arquivo {nome_arquivo_json}: {e}")

    #retorna os dados como arrays NumPy, prontos para o scikit-learn
    return np.array(lista_features), np.array(lista_rotulos)

#define os caminhos de forma dinâmica para que o script funcione em qualquer pasta
base_path = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(base_path, 'sinais.csv')
json_folder_path = os.path.join(base_path, 'sinais')

#chama a função para obter as features X, rótulos y e grupos interpretes
X, y_true_labels = carregar_e_processar_dados(csv_file_path, json_folder_path)

"""padroniza as features para que tenham média 0 e desvio padrão 1.
importante para algoritmo K-Means que é baseado em distância"""
print("\nPadronizando as features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Pré-processamento concluído. {X.shape[0]} amostras processadas.")


#K-MEANS E TESTE DO COTOVELO (ELBOW METHOD)

"""O objetivo desta é encontrar um número k de clusters que faça sentido
para os dados, antes de rodar os algoritmos finais"""

print("\nExecutando o Teste do Cotovelo para encontrar o 'k' ideal...")
inertia_values = []
#define o intervalo de valores de k que será testado-de 2 a 20 clusters
k_range = range(2, 21)

#loop para treinar o K-Means com cada valor de k
for k in tqdm(k_range):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
    kmeans.fit(X_scaled)

    ''' inertia_ é a soma das distâncias ao quadrado de cada ponto ao centro de seu cluster
    quanto menor a inércia, mais compactos são os clusters'''
    inertia_values.append(kmeans.inertia_)

#plotando o gráfico para visualização do cotovelo
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_values, 'bo-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia (Soma das distâncias quadradas intra-cluster)')
plt.title('Teste do Cotovelo (Elbow Method) para K-Means')
plt.xticks(k_range)
plt.grid(True)
plt.show()

#o k_otimo é escolhido com base na análise visual do gráfico anterior do cotovelo
k_otimo = 7 
print(f"\nValor de 'k' escolhido a partir do gráfico: {k_otimo}")

#executa o K-Means final com o número de clusters decidido
kmeans_final = KMeans(n_clusters=k_otimo, init='k-means++', random_state=42, n_init='auto')
kmeans_labels = kmeans_final.fit_predict(X_scaled)


#AGRUPAMENTO HIERÁRQUICO

"""execução do algoritmo hierárquico com o mesmo k para uma melhor precisão,
mas variando o método de linkage conforme solicitado"""

""" Método de Linkage 1 ward:
Tende a criar clusters de tamanhos mais uniformes, minimizando a variância"""
print("\nExecutando Agrupamento Hierárquico com linkage='ward'...")
hierarchical_ward = AgglomerativeClustering(n_clusters=k_otimo, linkage='ward')
hierarchical_ward_labels = hierarchical_ward.fit_predict(X_scaled)

""" Método de Linkage 2 complete:
Mede a distância entre clusters pela distância máxima entre seus pontos"""
print("Executando Agrupamento Hierárquico com linkage='complete'...")
hierarchical_complete = AgglomerativeClustering(n_clusters=k_otimo, linkage='complete')
hierarchical_complete_labels = hierarchical_complete.fit_predict(X_scaled)


#AVALIAÇÃO E COMPARAÇÃO DOS RESULTADOS


#agrupa os resultados de cada algoritmo para facilitar a avaliação
resultados_clustering = {
    "K-Means": kmeans_labels,
    "Hierárquico (Ward)": hierarchical_ward_labels,
    "Hierárquico (Complete)": hierarchical_complete_labels
}

#AVALIAÇÃO QUANTITATIVA
#calcula as métricas que avaliam a qualidade dos clusters sem usar os rótulos verdadeiros
metricas = []
for nome, labels in resultados_clustering.items():
    #Silhouette Score para medir quão bem separados os clusters estão
    sil_score = silhouette_score(X_scaled, labels)
    #Calinski-Harabasz para medir a razão entre dispersão inter-cluster e intra-cluster
    calinski_score = calinski_harabasz_score(X_scaled, labels)
    #Davies-Bouldin para medir a similaridade média de um cluster com seu vizinho mais próximo
    davies_score = davies_bouldin_score(X_scaled, labels)
    metricas.append({
        "Algoritmo": nome,
        "Silhouette Score (↑)": sil_score,
        "Calinski-Harabasz (↑)": calinski_score,
        "Davies-Bouldin (↓)": davies_score
    })
    
#exibe uma tabela comparativa com os resultados
resultados_df = pd.DataFrame(metricas)
print("\n--- Comparação das Métricas de Clusterização ---")
print(resultados_df.to_string(index=False))


""" AVALIAÇÃO VISUAL COM PCA
reduz as 264 features para apenas para plotar em um gráfico 2D"""
print("\nReduzindo dimensionalidade com PCA para visualização...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

#cria um gráfico de dispersão para cada algoritmo, colorindo os pontos por cluster
fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)
fig.suptitle(f'Visualização dos Clusters (k={k_otimo}) após Redução com PCA', fontsize=16)

for i, (nome, labels) in enumerate(resultados_clustering.items()):
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=50, alpha=0.7, ax=axes[i], legend='full')
    axes[i].set_title(nome)
    axes[i].set_xlabel('Componente Principal 1')
    if i == 0: axes[i].set_ylabel('Componente Principal 2')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#ANALISE QUALITATIVA-CRUZAMENTO COM RÓTULOS REAIS
"""embora o treinamento não use os rótulos, será usado para interpretar
o que cada cluster significa. A tabela mostra quantos sinais de cada tipo
foram parar em cada cluster. O ideal é que cada cluster contenha amostras de um único sinal"""
print("\n--- Análise Cruzada: Rótulos Verdadeiros vs. Clusters do K-Means (Melhor Modelo) ---")
df_analise = pd.DataFrame({'Rótulo Verdadeiro': y_true_labels, 'Cluster K-Means': kmeans_labels})
tabela_cruzada = pd.crosstab(df_analise['Rótulo Verdadeiro'], df_analise['Cluster K-Means'])
print(tabela_cruzada)
