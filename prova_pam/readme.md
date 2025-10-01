# Prova
## Questão 1

### Pré-processamento e Engenharia de Features

A base de dados consiste em 2501 arquivos JSON, cada um contendo uma série temporal de 33 pontos-chave do corpo. Para tornar esses dados utilizáveis pelos modelos, foram aplicadas as seguintes técnicas:

 **Normalização de Posição e Escala:** Para que o modelo foque apenas no gesto e não no tamanho ou posição do intérprete no vídeo, os dados de cada frame foram normalizados. Foram utilizados os ombros (pontos 11 e 12) como uma referência estável para centralizar e ajustar a escala de todos os outros pontos do corpo.
 **Extração de Features Estatísticas:** A série temporal de movimentos de cada sinal foi resumida em um vetor de 264 características. Para isso, foi calculada a **média, desvio padrão, mínimo e máximo** para as coordenadas `x` e `y` de cada um dos 33 pontos-chave normalizados.

### Estratégia de Validação Corrigida

Foi identificado que uma validação cruzada simples (`StratifiedKFold`) levaria a resultados artificialmente altos (~99%) devido a um problema de **vazamento de dados (data leakage)**. O modelo aprendia o estilo de cada um dos 10 intérpretes, em vez de generalizar os sinais.

Para corrigir isso, a estratégia de validação adotada foi o **`GroupKFold`** com **10 folds (`n_splits=10`)**. Essa abordagem garante que todos os vídeos de um mesmo intérprete permaneçam juntos, seja no conjunto de treino ou no de teste, em cada uma das 10 rodadas da validação. Isso força o modelo a aprender a reconhecer os sinais independentemente de quem os executa, fornecendo uma avaliação de desempenho melhor.

### Modelos Avaliados e Otimização

Foram avaliados três algoritmos de classificação:
* **Random Forest**
* **K-NN (K-Nearest Neighbors)**
* **MLP (Multi-Layer Perceptron)**

Após avaliação, o **Random Forest** se destacou. Por isso, ele foi otimizado com a ferramenta **`GridSearchCV`** para encontrar a melhor combinação de hiperparâmetros.

## Resultados

Resultados obtidos após a implementação da metodologia correta.
A busca com `GridSearchCV` testou 8 combinações de parâmetros. Devido a falta de tempo e limitação computacional, não foi possível realizar testes com maiores combinações de parâmentros. A melhor configuração encontrada foi:

<img width="621" height="101" alt="image" src="https://github.com/user-attachments/assets/c2956748-5d2b-40f2-9e31-c3db33e34f64" />




### Comparação Final de Desempenho

Com os melhores parâmetros para o Random Forest, foi realizada uma comparação final entre ele e os outros dois modelos. O F1-Score médio  obtido na validação cruzada por grupo foi:

<img width="430" height="85" alt="image" src="https://github.com/user-attachments/assets/c5db9c79-8b6f-4516-a8bb-7c3d162ff0fd" />


A análise dos resultados mostra que o **Random Forest Otimizado foi o modelo com o desempenho superior**, alcançando um F1-Score de aproximadamente **41.6%**. O MLP ficou em segundo lugar, enquanto o K-NN demonstrou maior dificuldade na tarefa, com um score consideravelmente mais baixo e instável (alto desvio padrão).

## Análise Visual (Matrizes de Confusão)

A imagem abaixo exibe as matrizes de confusão para os três modelos avaliados. A diagonal principal representa os acertos. Pontos fora da diagonal representam os erros (confusão entre sinais).

<img width="1366" height="655" alt="MATRIZ DE CONFUSÃO" src="https://github.com/user-attachments/assets/4ee7540f-723c-4b02-9d2c-82a28277f384" />


A análise visual confirma os resultados numéricos:
* **Random Forest Otimizado:** Apresenta a diagonal mais forte e definida, indicando um número maior de acertos e menos confusão entre as classes em comparação com os outros.
* **MLP:** Mostra uma diagonal visível, mas com mais "ruído" fora dela, confirmando seu desempenho intermediário.
* **K-NN:** Possui a matriz mais espalhada e a diagonal mais fraca, o que reflete visualmente seu baixo F1-Score.

## Conclusão

A aplicação de uma estratégia de validação robusta (`GroupKFold`) e uma engenharia de features aprimorada (normalização) foram cruciais para obter uma avaliação realista do problema. Após a otimização, o modelo **Random Forest se consolidou como a melhor abordagem**, superando a meta de 40% de F1-Score e demonstrando a maior capacidade de generalização para reconhecer sinais em LIBRAS de intérpretes não vistos durante o treinamento.

# Questão 2

### Pré-processamento e Engenharia de Features

Os dados brutos, contidos em 2501 arquivos JSON, foram processados para criar uma representação numérica para cada sinal.

 **Normalização dos Dados:** Para cada vídeo, as coordenadas dos pontos-chave foram normalizadas. Este processo envolveu a centralização dos pontos pela média e o ajuste de escala pela norma de Frobenius, tornando as features resultantes invariantes à posição e ao tamanho do intérprete no vídeo.
 **Extração de Features:** Para cada sinal, foi gerado um vetor de características fixas a partir da série temporal de movimentos. Foram calculadas 6 medidas estatísticas (`média`, `desvio padrão`, `mínimo`, `máximo`, `mediana` e `variância`) para cada coordenada `x` e `y` dos 33 pontos-chave.

### Algoritmos e Parâmetros

Foram utilizados dois tipos de algoritmos de clusterização:

1.  **K-Means:** Um método baseado em centróides. O número ideal de clusters `k` foi determinado através do Teste do Cotovelo
2.  **Agrupamento Hierárquico:** Um método que constrói uma hierarquia de clusters. Para uma comparação direta com o K-Means, ele foi configurado para gerar o mesmo número `k` de clusters. Foram testados dois métodos de `linkage` diferentes:
    * **`ward`**: Minimiza a variância dentro dos clusters que são agrupados.
    * **`complete`**: Usa a distância máxima entre os pontos de dois clusters para decidir o agrupamento.

### Medidas de Avaliação

Como a clusterização é uma tarefa não supervisionada, foram adotadas três métricas internas para avaliar a qualidade dos agrupamentos sem utilizar os rótulos verdadeiros dos sinais:

* **Silhouette Score:** Mede quão bem definidos e separados os clusters estão (quanto maior, melhor).
* **Calinski-Harabasz Score:** Avalia a densidade e separação dos clusters (quanto maior, melhor).
* **Davies-Bouldin Score:** Mede a sobreposição média entre os clusters (quanto menor, melhor).

### Determinação do Número de Clusters (k)

O Teste do Cotovelo foi executado para valores de `k` de 2 a 20. O gráfico gerado está abaixo:

<img width="1000" height="600" alt="TESTE DO COTOVELO" src="https://github.com/user-attachments/assets/ab57b74b-0f6b-4e82-a98a-2736226c7926" />


A análise do gráfico mostra uma inflexão em torno de **`k=7`**. A partir deste ponto, o ganho na redução da inércia diminui. Portanto, o valor **k=7** foi escolhido como o número de clusters para a análise.

### Comparação Quantitativa dos Algoritmos

Com `k=7`, os três modelos de clusterização foram executados e avaliados. A tabela abaixo resume o desempenho de cada um:

<img width="705" height="103" alt="image" src="https://github.com/user-attachments/assets/ddef1fb9-8f92-444d-a50e-e1ba6de4c27a" />


A análise dos resultados quantitativos aponta para uma conclusão clara: o **algoritmo K-Means foi superior em todas as três métricas**. Ele produziu clusters com melhor definição (maior Silhouette Score), maior densidade e separação (maior Calinski-Harabasz Score) e menor sobreposição entre si (menor Davies-Bouldin Score).

### Análise Visual dos Clusters (PCA)

Para visualizar os agrupamentos, a dimensionalidade dos dados foi reduzida usando PCA. Os gráficos abaixo mostram os clusters encontrados por cada algoritmo.

<img width="1366" height="655" alt="CLUSTERS" src="https://github.com/user-attachments/assets/ea2619da-6bb1-400a-8674-e9acb802a2cf" />


A visualização dos clusters confirma os resultados numéricos. O gráfico do **K-Means** (à esquerda) exibe agrupamentos de cores com uma separação visual um pouco mais clara em comparação com os dois métodos Hierárquicos, que apresentam maior sobreposição entre os pontos de diferentes clusters.

## Conclusão

O objetivo da atividade foi cumprido ao se aplicar e comparar diferentes técnicas de clusterização. A metodologia do Teste do Cotovelo permitiu uma escolha fundamentada para o número de clusters (`k=7`).

A comparação final, baseada em três métricas de avaliação distintas e na análise visual, demonstrou que o **K-Means foi o algoritmo mais eficaz** para encontrar uma estrutura de agrupamento coesa e significativa na base de dados de sinais em LIBRAS, considerando as features extraídas.
