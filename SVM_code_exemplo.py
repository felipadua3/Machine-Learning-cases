# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:13:12 2022

@author: felip
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dados = pd.read_csv('https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/loan_data.csv')

print(dados)
print(dados.dtypes)

colunas = dados.dtypes.reset_index()  #Transforma a info do dtypes em Dataframe
print(colunas)


colunas[colunas[0] == 'object']['index']
categ_cols = colunas[colunas[0] == 'object']['index'].to_list()

#transformando variáveis qualitativas

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in categ_cols:
    le.fit(dados[i])
    dados[str(i) + '_encoded'] = le.transform(dados[i])
    
dados = dados.drop('purpose' , axis = 1)

# Implementando o modelo

"""
x = variáveis de entrada
y = valores de previsao

"""

x = dados.drop('not.fully.paid' , axis = 1)
y = dados['not.fully.paid']

"""
Qubrando treino e teste
"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# Aplicando o SVM

from sklearn.svm import SVC

svm = SVC(kernel = 'linear')
svm.fit(x_train, y_train)

#Fazendo a previsao

y_pred = svm.predict(x_test)

# Avaliando o Resultado

from sklearn.metrics import accuracy_score

print('Acurácia do modelo foi: '+ str(accuracy_score(y_test, y_pred)))



