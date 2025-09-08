# Classificação de gêneros musicais


<p align="center">
  <img src="https://github.com/biabernardi/my-ML-project/blob/main/vitrolinha.png?raw=true" alt="Capinha fofa do projeto" width="350">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange" />
  <img src="https://img.shields.io/badge/status-completo-green" />
</p>

### **Introdução**

Este projeto tem como objetivo aplicar técnicas de Machine Learning para classificar músicas em dois gêneros: blues ou metal. Para isso, utilizei as etapas de pré-processamento (normalização e PCA), análise de dados e, por fim, o treinamento de modelos de classificação (Logistic Regression e Decision Tree).

### **Dados e características**

Para iniciar, utilizei os dados obtidos do dataset GTZAN. Também utilizei o arquivo `features_30_sec.csv`, que já contém as características extraídas das músicas, como os coeficientes `MFCC`, `tempo` e `chroma_stft`. Logo veremos que foquei apenas nas características numéricas para o funcionamento da ML.

Para focar na classificação binária, o dataset foi filtrado para incluir apenas as músicas dos gêneros blues e metal —  inicialmente, tínhamos 10 gêneros! As colunas de características foram definidas como features (X) e a coluna de gênero como o alvo/target (Y) do modelo.

### **Análise exploratória**

Antes do treinamento, apliquei a normalização com `StandardScaler` e a redução de dimensionalidade com PCA para visualizar a correlação entre as features e a separação dos gêneros. O resultado foi um heatmap para a correlação e um gráfico PCA, que confirmou a distinção entre os dois gêneros. 
<p align="center">
  <img src="https://github.com/biabernardi/my-ML-project/blob/main/Figure_1.png?raw=true" alt="image alt">
</p>

Também fiz um gráfico de dispersão para confirmar a distinção entre metal (roxo) e blues (azul). 
<p align="center">
  <img src="https://github.com/biabernardi/my-ML-project/blob/main/Figure_2%20(new).png?raw=true" alt="image alt">
</p>


### Treino e teste

Avançando, dividi o dataset em 70% para treino e 30% para teste. O conjunto de treino foi usado para que o modelo aprendesse a generalizar, enquanto o conjunto de teste, que continha dados que o modelo nunca havia visto, foi usado para avaliar o desempenho real. Posteriormente, veremos que a acurácia da Regressão Logística foi de 93%. Para garantir a reprodutibilidade dos resultados, apliquei um estado randômico (`random_state`) no processo de embaralhamento dos dados.

### Treino e previsões

Para classificar os gêneros escolhidos, utilizei e comparei o desempenho de dois algoritmos de Machine Learning: Regressão Logística e Árvore de Decisão.

A Regressão Logística é um modelo linear que encontra a melhor forma de separar os dados em duas categorias. A função `fit()`  é usada para treinar o modelo, ajustando seus parâmetros internos para que ele aprendesse a relação entre as características das músicas e seus gêneros. Também usamos a função `predict()` para que o modelo, já treinado, pudesse prever se as músicas no conjunto de dados de teste, que ele não havia visto antes, são blues ou metal.

Já Árvore de Decisão é um modelo não linear que funciona como um fluxograma ou uma série de perguntas de "sim ou não" para tomar uma decisão. Esse “fluxograma” baseado nos dados de treino é dado pela função `fit()` . Aqui, a função `predict()` usou essa estrutura de perguntas para prever o gênero das músicas de teste.

### Pós treinamento

Após o treinamento e a avaliação, a Regressão Logística demonstrou um desempenho superior, alcançando uma acurácia de 93% nos testes. O modelo classificou corretamente 28 músicas de blues e 28 de metal, com apenas 3 erros em blues e 1 em metal. 

A Árvore de Decisão também obteve um bom resultado, com uma acurácia de 83%, mas foi menos precisa que a Regressão Logística. O modelo cometeu um total de 10 erros (5 em blues e 5 em metais) de classificação no conjunto de teste.

Vemos que aqui, a Regressão Logística foi mais eficaz em generalizar os padrões do dataset para classificar os novos dados de teste.

### Considerações

Esse foi meu primeiro projeto de Machine Learning, então pode estar um pouco rudimentar ou saturado, mas estou muito satisfeita de poder aprender sobre PCA, normalização e modelos de ML. Além disso, foi minha primeira vez criando um ambiente virtual e instalando (depois de muitos erros de modulação) as bibliotecas. 

<img src="https://github.com/biabernardi/ML-first-project/blob/main/imagem_2025-09-04_012918657.png?raw=true" width="200" /> 

Estou muito animada para futuros projetos!
