import numpy as np
import pandas as pd
from functii import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

set_antrenare_testare = pd.read_csv("in/train.csv", index_col=0)
set_aplicare = pd.read_csv("in/test.csv", index_col=0)

set_aplicare.drop(columns=['satisfaction'], inplace=True)

satisfaction_mapping = {'neutral or dissatisfied': 0, 'satisfied': 1}

# Apply the mapping to the 'satisfaction' column
set_antrenare_testare['satisfaction'] = set_antrenare_testare['satisfaction'].map(satisfaction_mapping)
nan_replace(set_antrenare_testare)
nan_replace(set_aplicare)
variabile_categoriale = ["Gender", "Customer Type" , "Type of Travel","Class"]
codificare(set_antrenare_testare, variabile_categoriale)
codificare(set_aplicare, variabile_categoriale)
#print(set_antrenare_testare)
#print(set_aplicare)

variabile_observate = list(set_antrenare_testare)
predictori = variabile_observate[:-1]
tinta = variabile_observate[-1]

x_antrenare, x_testare, y_antrenare, y_testare = (
    train_test_split(set_antrenare_testare[predictori], set_antrenare_testare[tinta], test_size=0.3))
# print(x_antrenare)
# print(x_testare)
# print(y_antrenare)
# print(y_testare)

tabel_predictii_test = pd.DataFrame(
    {
        tinta: y_testare
    }, index=x_testare.index
)

# Calcul IV WOE
iv_summary, woe_table = calc_woe_iv(set_antrenare_testare, tinta)
iv_summary.to_csv("out/IV_Summary.csv", index=0)
woe_table.to_csv("out/WoE_Table.csv", index=0)

model_optim = None
index_kappa = 0
# Naive Bayes
model_b = GaussianNB()
kappa = analiza_model(model_b, x_antrenare, y_antrenare, x_testare, y_testare, tabel_predictii_test)
if kappa > index_kappa:
    index_kappa = kappa
    model_optim = model_b
# Decision Tree
model_dt = DecisionTreeClassifier()
kappa = analiza_model(model_dt, x_antrenare, y_antrenare, x_testare, y_testare, tabel_predictii_test, "DT")
if kappa > index_kappa:
    index_kappa = kappa
    model_optim = model_dt

# Random Forest
model_rf = RandomForestClassifier()
kappa = analiza_model(model_rf, x_antrenare, y_antrenare, x_testare, y_testare, tabel_predictii_test, "RF")
if kappa > index_kappa:
    index_kappa = kappa
    model_optim = model_rf

# Ridge / LR
if len(model_b.classes_) == 2:
    model_rl = LogisticRegression(max_iter=2000)
    kappa = analiza_model(model_rl, x_antrenare, y_antrenare, x_testare, y_testare, tabel_predictii_test, "RL")
    if kappa > index_kappa:
        index_kappa = kappa
        model_optim = model_rl
else:
    model_ridge = RidgeClassifier(max_iter=2000)
    kappa = analiza_model(model_ridge, x_antrenare, y_antrenare, x_testare, y_testare, tabel_predictii_test, "Ridge")
    if kappa > index_kappa:
        index_kappa = kappa
        model_optim = model_ridge

# KNN
model_knn = KNeighborsClassifier()
kappa = analiza_model(model_knn, x_antrenare, y_antrenare, x_testare, y_testare, tabel_predictii_test, "kNN")
if kappa > index_kappa:
    index_kappa = kappa
    model_optim = model_knn

tabel_predictii_test.to_csv("out/Predictii_test.csv")

print(model_optim)

# Aplicare
set_aplicare["Predictie"] = model_optim.predict(set_aplicare[predictori])
set_aplicare.to_csv("out/Predictie.csv")

# SVM
model_svm = SVC(probability=True)
kappa = analiza_model(model_svm, x_antrenare, y_antrenare, x_testare, y_testare, tabel_predictii_test, "SVM")
if kappa > index_kappa:
    index_kappa = kappa
    model_optim = model_svm

print(model_optim)

