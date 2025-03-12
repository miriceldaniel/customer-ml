import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC

from grafice import *


def nan_replace(t):
    assert isinstance(t, pd.DataFrame)
    for v in t.columns:
        if t[v].isna().any():
            if is_numeric_dtype(t[v]):
                t[v].fillna(t[v].mean(), inplace=True)
            else:
                t[v].fillna(t[v].mode()[0], inplace=True)


def codificare(t, variabile_categoriale):
    for v in variabile_categoriale:
        t[v] = pd.Categorical(t[v]).codes + 1


def analiza_model(model, x, y, x_, y_, predictii_test, nume_model="Bayes"):
    model.fit(x, y)
    predictie = model.predict(x_)
    predictii_test[nume_model + "_Predictie"] = predictie
    proba = model.predict_proba(x_)
    # print(predictie)
    # print(proba)
    cm = confusion_matrix(y_, predictie)
    # print(cm)
    clase = model.classes_
    t_cm = pd.DataFrame(cm, clase, clase)
    t_cm["Acuratete"] = np.diag(cm) * 100 / np.sum(cm, axis=1)
    t_cm.to_csv("out/" + nume_model + "_cm.csv")
    plot_matrice_confuzie(y_, predictie, nume_model)
    plot_grafice_evaluare(y_, proba, nume_model)
    acuratete_globala = np.sum(np.diag(cm)) * 100 / len(y_)
    acuratete_medie = t_cm["Acuratete"].mean()
    kappa = cohen_kappa_score(y_, predictie)
    s_acuratete = pd.Series([acuratete_globala, acuratete_medie, kappa],
                            ["Acuratete Globala", "Acuratete Medie", "Index Kappa"], name="Acuratete")
    s_acuratete.to_csv("out/" + nume_model + "_acuratete.csv")
    show()
    return kappa

def calc_woe_iv(df, target, bins=10, show_woe=False):
    # Empty Dataframe
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()

    # Extract Column Names
    cols = df.columns

    # Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (df[ivars].dtype.kind in 'bifc') and (len(np.unique(df[ivars])) > 10):
            binned_x = pd.qcut(df[ivars], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': df[target]})
        else:
            d0 = pd.DataFrame({'x': df[ivars], 'y': df[target]})
        d0 = d0.astype({"x": str})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Non-Events'] / d['% of Events'])
        d['IV'] = d['WoE'] * (d['% of Non-Events'] - d['% of Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(), 6)))
        temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)

        # Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF
