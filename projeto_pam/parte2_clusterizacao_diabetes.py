# -*- coding: utf-8 -*-
"""
Análise do Dataset de Indicadores de Saúde para Diabetes (BRFSS2015) - Parte 2 (Ajustada)

Este script implementa uma abordagem aprimorada para a clusterização:
1.  Carrega e prepara os dados do dataset.
2.  **MELHORIA:** Utiliza um modelo Random Forest para realizar a Seleção de Features,
    identificando as características mais importantes para a análise.
3.  Utiliza o Teste do Cotovelo e a Análise da Silhueta para encontrar o 'k' ideal,
    agora com base nas features mais relevantes.
4.  Aplica os algoritmos K-Means e Agrupamento Hierárquico nos dados "limpos".
5.  Avalia e compara os resultados finais.
"""

# --------------------------------------------------------------------------
# SEÇÃO 1: IMPORTAÇÃO DAS BIBLIOTECAS
# --------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Módulos do Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier # Importado para a seleção de features

# --------------------------------------------------------------------------
# SEÇÃO 2: CARREGAMENTO E PREPARAÇÃO DOS DADOS
# --------------------------------------------------------------------------

try:
    df_full = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
    print("Dataset completo carregado com sucesso!")
except FileNotFoundError:
    print("Erro: Arquivo 'diabetes_binary_health_indicators_BRFSS2015.csv' não encontrado.")
    exit()

# Para evitar MemoryError no Agrupamento Hierárquico e agilizar a análise.
SAMPLE_SIZE = 20000
df = df_full.sample(n=SAMPLE_SIZE, random_state=42)
print(f"\nTrabalhando com uma amostra de {SAMPLE_SIZE} pontos.")

target_column = 'Diabetes_binary'
X = df.drop(target_column, axis=1)
y = df[target_column] # Usaremos 'y' para a seleção de features

# --------------------------------------------------------------------------
# SEÇÃO 3: MELHORIA - SELEÇÃO DE FEATURES COM RANDOM FOREST
# --------------------------------------------------------------------------
print("\n--- Encontrando as Features Mais Importantes com Random Forest ---")
# Treina um modelo rápido de Random Forest para extrair a importância das features
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Cria uma série do Pandas para visualizar a importância de cada feature
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("Importância das Features (das mais para as menos importantes):")
print(feature_importances)

# Vamos selecionar as 10 features mais importantes para a clusterização
N_TOP_FEATURES = 10
top_features = feature_importances.head(N_TOP_FEATURES).index
print(f"\nSelecionando as {N_TOP_FEATURES} features mais importantes: {list(top_features)}")

# Cria um novo DataFrame 'X' contendo apenas essas features
X_selecionado = X[top_features]

# Padroniza as features selecionadas
print("\nPadronizando as features selecionadas...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selecionado)

# --------------------------------------------------------------------------
# SEÇÃO 4: ANÁLISE PARA ENCONTRAR O 'k' IDEAL
# --------------------------------------------------------------------------
print("\nExecutando a análise da Silhueta para encontrar o melhor 'k'...")
silhouette_scores = []
k_range = range(2, 11)

for k in tqdm(k_range):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette_avg)

# Plotando o gráfico da Silhueta
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, 'bo-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score Médio')
plt.title('Análise da Silhueta para Escolha de k (com Features Selecionadas)')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Encontra o melhor k com base na maior pontuação de silhueta
k_otimo = k_range[np.argmax(silhouette_scores)]
print(f"\nO melhor 'k' de acordo com a análise da Silhueta é: {k_otimo}")

# --------------------------------------------------------------------------
# SEÇÃO 5: EXECUÇÃO DOS ALGORITMOS FINAIS
# --------------------------------------------------------------------------
# K-Means Final
kmeans_final = KMeans(n_clusters=k_otimo, init='k-means++', random_state=42, n_init='auto')
kmeans_labels = kmeans_final.fit_predict(X_scaled)

# Agrupamento Hierárquico
print("\nExecutando Agrupamento Hierárquico...")
hierarchical_ward = AgglomerativeClustering(n_clusters=k_otimo, linkage='ward')
hierarchical_ward_labels = hierarchical_ward.fit_predict(X_scaled)

hierarchical_complete = AgglomerativeClustering(n_clusters=k_otimo, linkage='complete')
hierarchical_complete_labels = hierarchical_complete.fit_predict(X_scaled)

# --------------------------------------------------------------------------
# SEÇÃO 6: AVALIAÇÃO E COMPARAÇÃO DOS RESULTADOS
# --------------------------------------------------------------------------
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
print("\n--- Comparação das Métricas de Clusterização (com Features Selecionadas) ---")
print(resultados_df.to_string(index=False))
print("(↑) Quanto maior, melhor. (↓) Quanto menor, melhor.")

# --- Visualização dos Clusters com PCA ---
print("\nReduzindo dimensionalidade com PCA para visualização...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)
fig.suptitle(f'Visualização dos Clusters (k={k_otimo}) após Seleção de Features e PCA', fontsize=16)
for i, (nome, labels) in enumerate(resultados_clustering.items()):
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=10, alpha=0.5, ax=axes[i], legend='full')
    axes[i].set_title(nome)
    axes[i].set_xlabel('Componente Principal 1')
    if i == 0: axes[i].set_ylabel('Componente Principal 2')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Análise Cruzada com os Rótulos Verdadeiros ---
print("\n--- Análise Cruzada: Rótulos Verdadeiros vs. Clusters do K-Means ---")
df_analise = pd.DataFrame({'Rótulo Verdadeiro': y.map({0: 'Não Diabético', 1: 'Diabético'}), 'Cluster K-Means': kmeans_labels})
tabela_cruzada = pd.crosstab(df_analise['Cluster K-Means'], df_analise['Rótulo Verdadeiro'])
print(tabela_cruzada)