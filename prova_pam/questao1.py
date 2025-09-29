"""
Prova - Introdução a Inteligência Artificial - UFPB
Brendo de Almeida Mendonça - 20190172521
QUESTÃO 1:

Análise e Classificação de Sinais em LIBRAS a partir de Keypoints
Pré-processamento dos dados, treinamento de três modelos
de Machine Learning (Random Forest, K-NN e MLP) e avaliação de
desempenhos utilizando F1-Score e Matriz de Confusão.
"""

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

#módulos do scikit-learn para pré-processamento e modelagem
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# PRÉ-PROCESSAMENTO E CARREGAMENTO DOS DADOS
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
X, y = carregar_e_processar_dados(csv_file_path, json_folder_path)

print(f"\nPré-processamento concluído.")
print(f"Formato da matriz de features (X): {X.shape}")
print(f"Formato do vetor de rótulos (y): {y.shape}")

#codifica os rótulos dos nomes dos sinais para números inteiros
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#TREINAMENTO E VALIDAÇÃO DOS MODELOS

N_SPLITS = 10 #definição o número de folds para a validação cruzada
RANDOM_STATE = 42 #definição um estado aleatório fixo para garantir a reprodutibilidade dos resultados

#configuração da validação cruzada estratificada para garantir que a proporção de cada classe seja mantida em cada fold
cv_stratified = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

#dicionário com os modelos a serem testados
"""Para cada modelo, um Pipeline é criado. O Pipeline primeiro padroniza os dados(StandardScaler)
e depois aplica o classificador. Evitando o vazamento de dados,
pois a padronização é aprendida apenas com os dados de treino dos folds"""
modelos = {
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "K-NN": KNeighborsClassifier(n_jobs=-1),
    "MLP": MLPClassifier(random_state=RANDOM_STATE, max_iter=1500, hidden_layer_sizes=(128, 64))
}

pipelines = {}
resultados_f1 = {}

print("\nIniciando treinamento e validação cruzada dos modelos...")
for nome, modelo in modelos.items():
    # Cria um pipeline para cada modelo
    pipelines[nome] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', modelo)
    ])
    
    #executa a validação cruzada e calcula o F1-Score para cada fold
    #foi usado 'f1_weighted' que calcula o F1 para cada classe e pondera pela sua frequência
    scores = cross_val_score(pipelines[nome], X, y_encoded, cv=cv_stratified, scoring='f1_weighted', n_jobs=-1)
    resultados_f1[nome] = scores
    
    print(f"  - {nome}: F1-Score Médio = {np.mean(scores):.4f} (Desvio Padrão = {np.std(scores):.4f})")

#AVALIAÇÃO DOS RESULTADOS

print("\nGerando previsões e matrizes de confusão...")
"""Para gerar uma única matriz de confusão representativa, foi usado cross_val_predict
que retorna as previsões para cada amostra quando ela estava no conjunto de teste"""

previsoes = {}
for nome, pipeline in pipelines.items():
    previsoes[nome] = cross_val_predict(pipeline, X, y_encoded, cv=cv_stratified, n_jobs=-1)

#plota a matriz de confusão para cada modelo
fig, axes = plt.subplots(1, len(modelos), figsize=(20, 6))
fig.suptitle('Matrizes de Confusão para cada Modelo (Validação Cruzada)', fontsize=16)

for i, (nome, y_pred) in enumerate(previsoes.items()):
    cm = confusion_matrix(y_encoded, y_pred)
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=axes[i], xticklabels=False, yticklabels=False)
    axes[i].set_title(nome)
    axes[i].set_ylabel('Rótulo Verdadeiro')
    axes[i].set_xlabel('Rótulo Previsto')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#exibe o relatório de classificação detalhado para o melhor modelo
print("\n--- Relatório de Classificação Detalhado ---\n")
for nome, y_pred in previsoes.items():
    print(f"Modelo: {nome}")
    print(classification_report(y_encoded, y_pred, target_names=label_encoder.classes_))
    print("-" * 60)