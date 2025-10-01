"""
PROVA DE PARADIGMAS DE APRENDIZAGEM DE MÁQUINA
Brendo de Almeida Mendonça
Questão 1

Carrega os dados de sinais em LIBRAS a partir de arquivos JSON,
normaliza os pontos-chave para garantir invariância à escala e posição, e extrai
features estatísticas para representar cada sinal como um vetor numérico
Análise e Classificação de Sinais em LIBRAS a partir de Keypoints

Este script realiza o pré-processamento dos dados, treina três modelos
de Machine Learning (Random Forest, K-NN e MLP) e avalia seus
desempenhos utilizando F1-Score e Matriz de Confusão
"""


import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm #usada para criar barras de progresso visuais
import matplotlib.pyplot as plt
import seaborn as sns

#módulos do Scikit-learn para pré-processamento, modelagem e avaliação
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix, classification_report

#módulos de ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# PRÉ-PROCESSAMENTO E CARREGAMENTO DOS DADOS
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
        np.array: Vetor de grupos/intérpretes (groups)
    """
    #lê o arquivo CSV que serve como um índice para os arquivos JSON
    df_manifest = pd.read_csv(caminho_csv)
    #remove linhas que possam ter nomes de arquivos ausentes
    df_manifest.dropna(subset=['file_name'], inplace=True)
    df_manifest.reset_index(drop=True, inplace=True)

    #listas para armazenar os dados processados
    lista_features = []
    lista_rotulos = []
    lista_interpretes = []

    print("Iniciando pré-processamento com NORMALIZAÇÃO...")
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
                #cria um dicionário para acesso rápido aos keypoints pelo ID
                keypoints_dict = {kp['id']: kp for kp in frame['keypoints']}
                
                #LÓGICA DE NORMALIZAÇÃO
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
                    if escala_tronco == 0: continue #evita divisão por zero

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
            lista_interpretes.append(row['interprete'])

        except Exception as e:
            print(f"Erro ao processar o arquivo {nome_arquivo_json}: {e}")

    #retorna os dados como arrays NumPy, prontos para o scikit-learn
    return np.array(lista_features), np.array(lista_rotulos), np.array(lista_interpretes)

#define os caminhos de forma dinâmica para que o script funcione em qualquer pasta
base_path = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(base_path, 'sinais.csv')
json_folder_path = os.path.join(base_path, 'sinais')

#chama a função para obter as features X, rótulos y e grupos interpretes
X, y, groups = carregar_e_processar_dados(csv_file_path, json_folder_path)

#converte os rótulos de texto-ex: "Adição" para números ex: 0...
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# AJUSTE DE HIPERPARÂMETROS E COMPARAÇÃO DE MODELOS

#configuração da Validação Cruzada
N_SPLITS = 10
RANDOM_STATE = 42

"""Definção da estratégia de validação cruzada baseada em grupos de intérpretes
garantidno que os dados de um mesmo intérprete não apareçam no treino e no teste
# ao mesmo tempo, evitando o vazamento de dados e garantindo uma melhor avaliação"""
cv_group = GroupKFold(n_splits=N_SPLITS)

"""modelo random forest com GridSearchCV
cria um "pipeline" que encadeia o pré-processamento-StandardScaler com o modelo
garantidno que a padronização dos dados ocorra corretamente dentro de cada fold"""
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
])

"""#define a grade de hiperparâmetros a serem testados para o Random Forest
'rf__' indica que o parâmetro se aplica ao passo 'rf' do pipeline"""
param_grid = {
    'rf__n_estimators': [200, 300], #número de árvores
    'rf__max_depth': [20, 30],      #profundidade máxima das árvores
    'rf__min_samples_leaf': [1, 2]  #mínimo de amostras por folha da árvore
}

"""configura o GridSearchCV, o motor de busca de hiperparâmetros
ele testará todas as combinações do param_grid usando a validação cruzada por grupo ('cv=cv_group')"""
grid_search = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid,
    scoring='f1_weighted',
    cv=cv_group,
    verbose=2,#mostra o progresso do treinamento
    n_jobs=-1 #utiliza todos os núcleos do processador para acelerar
)

print("\nIniciando o ajuste de hiperparâmetros para o Random Forest")
"""executa a busca
o '.fit()' precisa dos grupos para fazer a divisão correta"""
grid_search.fit(X, y_encoded, groups=groups)

print("\nAjuste de hiperparâmetros concluído.")
print(f"Melhores parâmetros para RF: {grid_search.best_params_}")
print(f"Melhor F1-Score - RF: {grid_search.best_score_:.4f}")

#definição dos modelos para a comparação final
modelos_finais = {
    #pega o melhor modelo já encontrado e treinado pelo GridSearchCV
    "Random Forest": grid_search.best_estimator_, 
    
    #define os outros dois modelos com seus parâmetros padrão, dentro de um pipeline
    "K-NN": Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_jobs=-1))
    ]),
    "MLP": Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(random_state=RANDOM_STATE, max_iter=500, hidden_layer_sizes=(100, 50)))
    ])
}

#dicionários para armazenar os resultados da avaliação final
resultados_f1 = {}
previsoes = {}

print("\nIniciando a avaliação final comparativa dos modelos...")
for nome, modelo in modelos_finais.items():
    #o score do RF já foi calculado, então foi reutilizado
    if nome == "Random Forest":
        scores = [grid_search.best_score_]
    else:
        #para os outros modelos, é executada a validação cruzada para obter o F1-Score
        scores = cross_val_score(modelo, X, y_encoded, cv=cv_group, groups=groups, scoring='f1_weighted', n_jobs=-1)
    
    resultados_f1[nome] = scores
    
    #gera as previsões para cada modelo para construir a matriz de confusão
    previsoes[nome] = cross_val_predict(modelo, X, y_encoded, cv=cv_group, groups=groups, n_jobs=-1)
    
    print(f"  - {nome}: F1-Score Médio = {np.mean(scores):.4f}")


#APRESENTAÇÃO DOS RESULTADOS FINAIS

#Matrizes de Confusão
#plota as matrizes de confusão lado a lado para uma comparação visual
fig, axes = plt.subplots(1, len(modelos_finais), figsize=(24, 7))
fig.suptitle('Matrizes de Confusão Comparativas (Validação por Grupo)', fontsize=16)

for i, (nome, y_pred) in enumerate(previsoes.items()):
    cm = confusion_matrix(y_encoded, y_pred)
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=axes[i], xticklabels=False, yticklabels=False)
    axes[i].set_title(nome)
    axes[i].set_ylabel('Rótulo Verdadeiro')
    axes[i].set_xlabel('Rótulo Previsto')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#Relatórios de Classificação
#exibe um relatório detalhado com precisão, recall e f1-score para cada classe, para cada modelo
print("\n--- Relatórios de Classificação Detalhados ---\n")
for nome, y_pred in previsoes.items():
    print(f"Modelo: {nome}")
    # 'zero_division=0' evita avisos caso uma classe não tenha previsões corretas
    print(classification_report(y_encoded, y_pred, target_names=label_encoder.classes_, zero_division=0))
    print("-" * 60)