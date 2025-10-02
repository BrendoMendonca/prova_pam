# -*- coding: utf-8 -*-
"""
Análise do Dataset de Indicadores de Saúde para Diabetes (BRFSS2015)

Este script realiza a "Parte 1" da atividade, adaptada para um novo dataset:
1.  Carrega e prepara os dados tabulares do arquivo CSV.
2.  Otimiza o modelo Random Forest usando GridSearchCV.
3.  Compara o modelo otimizado com K-NN e MLP usando StratifiedKFold.
4.  Apresenta os resultados completos, incluindo F1-Score, Matrizes de Confusão
    e Relatórios de Classificação.
"""

# --------------------------------------------------------------------------
# SEÇÃO 1: IMPORTAÇÃO DAS BIBLIOTECAS
# --------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Módulos do Scikit-learn
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Algoritmos de Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# --------------------------------------------------------------------------
# SEÇÃO 2: CARREGAMENTO E PREPARAÇÃO DOS DADOS
# --------------------------------------------------------------------------

# Carrega o dataset a partir do arquivo CSV
try:
    df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
    print("Dataset carregado com sucesso!")
except FileNotFoundError:
    print("Erro: Arquivo 'diabetes_binary_health_indicators_BRFSS2015.csv' não encontrado.")
    print("Por favor, certifique-se de que o arquivo CSV está na mesma pasta que o script.")
    exit()

# --- CORREÇÃO: TRATAMENTO DE VALORES AUSENTES (NaN) ---
# Foi identificado um valor ausente na coluna 'Income'. Vamos preenchê-lo com a mediana.
if df.isnull().sum().any():
    print("\nTratando valores ausentes...")
    # Calcula a mediana da coluna 'Income'
    income_median = df['Income'].median()
    # Preenche os valores NaN com a mediana
    df['Income'].fillna(income_median, inplace=True)
    print("Valores ausentes foram preenchidos com a mediana.")

# Análise Exploratória Rápida
print("\n--- Informações do Dataset ---")
df.info()

# O alvo da nossa previsão é a coluna 'Diabetes_binary'
target_column = 'Diabetes_binary'

# Verifica o balanceamento das classes
print("\n--- Balanceamento da Classe Alvo ---")
print(df[target_column].value_counts())
print("O dataset é desbalanceado, o que reforça a importância de usar F1-Score e StratifiedKFold.")

# Separa as features (X) do alvo (y)
X = df.drop(target_column, axis=1)
y = df[target_column]

# --------------------------------------------------------------------------
# SEÇÃO 3: AJUSTE DE HIPERPARÂMETROS E COMPARAÇÃO DE MODELOS
# --------------------------------------------------------------------------

# Configuração da Validação Cruzada
N_SPLITS = 10
RANDOM_STATE = 42
cv_stratified = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Otimização do Random Forest com GridSearchCV
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
])
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 20],
    'rf__min_samples_leaf': [2, 4]
}
grid_search = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid,
    scoring='f1_weighted',
    cv=cv_stratified,
    verbose=2,
    n_jobs=-1
)

print("\nIniciando o ajuste de hiperparâmetros para o Random Forest (pode demorar)...")
grid_search.fit(X, y)

print("\nAjuste de hiperparâmetros concluído.")
print(f"Melhores parâmetros para RF: {grid_search.best_params_}")
print(f"Melhor F1-Score (RF Otimizado): {grid_search.best_score_:.4f}")

# Definição dos Modelos para a Comparação Final
modelos_finais = {
    "Random Forest Otimizado": grid_search.best_estimator_,
    "K-NN": Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_jobs=-1))
    ]),
    "MLP": Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(random_state=RANDOM_STATE, max_iter=500))
    ])
}

resultados_f1 = {}
previsoes = {}

print("\nIniciando a avaliação final comparativa dos modelos...")
for nome, modelo in modelos_finais.items():
    if nome == "Random Forest Otimizado":
        scores = [grid_search.best_score_]
    else:
        scores = cross_val_score(modelo, X, y, cv=cv_stratified, scoring='f1_weighted', n_jobs=-1)
    
    resultados_f1[nome] = scores
    previsoes[nome] = cross_val_predict(modelo, X, y, cv=cv_stratified, n_jobs=-1)
    
    print(f"  - {nome}: F1-Score Médio = {np.mean(scores):.4f} (Desvio Padrão = {np.std(scores):.4f})")

# --------------------------------------------------------------------------
# SEÇÃO 4: APRESENTAÇÃO DOS RESULTADOS FINAIS
# --------------------------------------------------------------------------

# Matrizes de Confusão
fig, axes = plt.subplots(1, len(modelos_finais), figsize=(24, 7))
fig.suptitle('Matrizes de Confusão Comparativas', fontsize=16)
class_names = ['Não Diabético', 'Diabético']

for i, (nome, y_pred) in enumerate(previsoes.items()):
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                xticklabels=class_names, yticklabels=class_names)
    axes[i].set_title(nome)
    axes[i].set_ylabel('Rótulo Verdadeiro')
    axes[i].set_xlabel('Rótulo Previsto')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Relatórios de Classificação
print("\n--- Relatórios de Classificação Detalhados ---\n")
for nome, y_pred in previsoes.items():
    print(f"Modelo: {nome}")
    print(classification_report(y, y_pred, target_names=class_names, zero_division=0))
    print("-" * 60)