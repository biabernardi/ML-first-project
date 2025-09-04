import pandas as pd

import numpy as np

import librosa as lib

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix

#------------------------------------------------------------------

df = pd.read_csv("features_30_sec.csv") #download do kaggle
df.rename(columns={'label': 'genre'}, inplace=True) #renomear label (GTZAN dataset) para genre

print(df.head())
print(df.info())
#print(df['genre'].value_counts()) #verificar quantidade de cada gênero

df_filtered = df[(df['genre'] == 'blues') | (df['genre'] == 'metal')] #filtrar blues e metal

df_numerical = df.select_dtypes(include=np.number) #criar dataframe apenas com colunas numéricas
plt.figure(figsize=(12, 8)) # verificar correlação
sns.heatmap(df_numerical.corr(), annot=True, fmt=".2f") #uso do novo dataframe no heatmap
plt.show()

X = df_filtered.drop(["genre", "filename", "length"], axis=1)  #tira a coluna genre do dataframe e coloca em X (features)
Y = df_filtered["genre"] #seleciona apenas genre do dataframe e coloca em y (target)

#executando até aqui, obtemos o heatmap de correlação dos features 

scaler = StandardScaler() #scaler para padronização (calcula o desvio padrão e o médio)
X_scaled = scaler.fit_transform(X) 
#fit(X) calcula a média e desvio padrão de cada feature
#transform(X) subtrai a média e divide pelo desvio padrão

#PCA - Análise de Compontentes Principais
pca = PCA(n_components=2) #devolve 2 features numéricas
X_pca = pca.fit_transform(X_scaled)
#pca.fit(X_scaled)
#X_pca = pca.transform(X_scaled)

plt.scatter(X_pca[:,0], X_pca[:,1], c=Y.map({"blues":"#14149C", "metal":"#E87BDA"}))
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Visualização")
plt.show()
 #plt.scatter(): gráfico de dispersão
 #X_pca[:,0] array do numpy, todas as linhas, coluna 0
 
 #análise para determinar o gênero de cada eixo
print("Contribuição das features para o PCA 1:")
print(pca.components_[0])

print("\nContribuição das features para o PCA 2:")
print(pca.components_[1])

# Dividir dados em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)
#20% dos dados para teste
print(f"Tamanho do conjunto de treino: {X_train.shape[0]} músicas") #número de músicas separadas para treino (160)
print(f"Tamanho do conjunto de teste: {Y_test.shape[0]} músicas") #número de músicas separadas para treino (60)

#!!!!!!!!!!!!!!!!!!!!! treinando os modelos de ML e fazendo previsões !!!!!!!!!!!!!!!!!!!!!

#logistic regression
lr = LogisticRegression(class_weight='balanced') #separa os dados em duas categorias #lr aprende pelos dados de treino
lr.fit(X_train, Y_train) #ajusta os parâmetros internos para encontrar a melhor forma de mapear as características para os gêneros.
Y_pred_lr = lr.predict(X_test) #previsão #usa o que aprendeu para prever o gênero de cada música e armazena na variável y_pred_lr

#árvore de decisão # fluxograma ou uma série de perguntas "sim ou não" para chegar a uma decisão
dt = DecisionTreeClassifier() 
dt.fit(X_train, Y_train)
Y_pred_dt = dt.predict(X_test) #usa o que aprendeu para prever o gênero de cada música e armazena na variável y_pred_dt

#!!!!!!!!!!!!!!!!!!!!! avaliando os modelos !!!!!!!!!!!!!!!!!!!!!
#será que Y_pred_lr e Y_pred_dt estão corretos? vamos comparar com Y_test
print("Regressão Logística")
print(confusion_matrix( Y_test, Y_pred_lr))
print(classification_report(Y_test, Y_pred_lr))

print("Árvore de Decisão")
print(confusion_matrix(Y_test, Y_pred_dt)) #quantas músicas de cada gênero foram classificadas corretamente e quantas foram incorretamente
print(classification_report(Y_test, Y_pred_dt)) #precisão, recall e f1-score

