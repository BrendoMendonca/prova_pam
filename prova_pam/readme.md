# Prova
## Questão 1

### 1.1. Pré-processamento e Engenharia de Features

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

## 2. Resultados

A seguir, são apresentados os resultados obtidos após a implementação da metodologia correta.

### 2.1. Otimização do Random Forest

A busca com `GridSearchCV` testou 8 combinações de parâmetros. A melhor configuração encontrada foi:



### 2.2. Comparação Final de Desempenho

Com os melhores parâmetros para o Random Forest, foi realizada uma comparação final entre ele e os outros dois modelos (com seus parâmetros padrão). O F1-Score médio (ponderado) obtido na validação cruzada por grupo foi:



A análise dos resultados mostra que o **Random Forest Otimizado foi o modelo com o desempenho superior**, alcançando um F1-Score de aproximadamente **41.6%**. O MLP ficou em segundo lugar, enquanto o K-NN demonstrou maior dificuldade na tarefa, com um score consideravelmente mais baixo e instável (alto desvio padrão).

## 3. Análise Visual (Matrizes de Confusão)

A imagem abaixo exibe as matrizes de confusão para os três modelos avaliados. A diagonal principal representa os acertos. Pontos fora da diagonal representam os erros (confusão entre sinais).

![Matrizes de Confusão](MATRIZ DE CONFUSÃO.png)

A análise visual confirma os resultados numéricos:
* **Random Forest Otimizado:** Apresenta a diagonal mais forte e definida, indicando um número maior de acertos e menos confusão entre as classes em comparação com os outros.
* **MLP:** Mostra uma diagonal visível, mas com mais "ruído" fora dela, confirmando seu desempenho intermediário.
* **K-NN:** Possui a matriz mais espalhada e a diagonal mais fraca, o que reflete visualmente seu baixo F1-Score.

## 4. Conclusão

A aplicação de uma estratégia de validação robusta (`GroupKFold`) e uma engenharia de features aprimorada (normalização) foram cruciais para obter uma avaliação realista do problema. Após a otimização, o modelo **Random Forest se consolidou como a melhor abordagem**, superando a meta de 40% de F1-Score e demonstrando a maior capacidade de generalização para reconhecer sinais em LIBRAS de intérpretes não vistos durante o treinamento.