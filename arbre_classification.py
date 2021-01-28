# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:45:51 2021

@author: theo
"""


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1)Loading & Prerocessing Data 
# a) Chargement des données
names_col=["Class","age","menopause","tumor-size","inv-nodes","node-caps","deg-malig","breast","breast-quad","irradiat"]
df = pd.read_csv("breast-cancer_data.csv",names =names_col)
# b) Drop les lignes où les colonnes  "node-caps" ou "breast-quad" sont égales  à "?" 

index_names1 = df.loc[ df["node-caps"] == "?"].index 
index_names2 = df.loc[ df["breast-quad"] == "?"].index 

df.drop(index_names1, inplace = True) 
df.drop(index_names2, inplace = True) 


df["Class"].loc[df["Class"] == "no-recurrence-events"] = 0
df["Class"].loc[df["Class"] == "recurrence-events"] = 1


# 2) Processing Data 
# a) on crée des dummies pour toutes les features qui sont du type "categorical
to_dummies = names_col[1:6]+names_col[7:]
dummies = pd.get_dummies(df[to_dummies])
classe = df["Class"]

test = df.drop(columns = to_dummies)
df2 = pd.concat([test, dummies], axis=1)
# b) Séparation des variables explicatives et de la vérité terrain
features = df2.drop(columns = ["Class"])
ground_truth = df2["Class"]
# c) Normalisation 

transformer = StandardScaler().fit(features) 
features = transformer.transform(features)

# d) création du dataset de training et de test
x_train, x_test, y_train, y_test = train_test_split(features, ground_truth.astype('int'))

# 3) Ccréation du modèle
# a) le modèle
clf = DecisionTreeClassifier(random_state=0, class_weight = 'balanced').fit(x_train, y_train)
# b) résultat
print("l'entraînement s'est fait sur ",y_train.count() )
print("le test de fin s'est fait sur ",y_test.count() )
print("la précision moyenne du modèle de Décision Tree est de" ,clf.score(x_test,y_test))