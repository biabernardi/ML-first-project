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

#-------------------------------------------------------------------

df = pd.read_csv("features_30_sec.csv") #download do kaggle
df.rename(columns={'label': 'genre'}, inplace=True) #renomear label (GTZAN dataset) para genre

print(df.head())
print(df.info())


df_numerical = df.select_dtypes(include=np.number) #criar dataframe apenas com colunas numéricas
plt.figure(figsize=(12, 8)) # verificar correlação
sns.heatmap(df_numerical.corr(), annot=True, fmt=".2f") #uso do novo dataframe no heatmap
plt.show()

X = df.drop("genre", axis=1)  #tira a coluna genre do dataframe e coloca em X (features)
y = df["genre"] #seleciona apenas genre do dataframe e coloca em y (target)

#executando até aqui, obtemos o heatmap de correlação dos features 

scaler = StandardScaler() #scaler para padronização (calcula o desvio padrão e o médio)
X_scaled = scaler.fit_transform(X) 
#fit(X) calcula a média e desvio padrão de cada feature
#transform(X) subtrai a média e divide pelo desvio padrão

