# -*- coding: utf-8 -*-
"""
The data contains values, this script edits them and then saves them, also with the analysis, the melbony area has higher 
total power and there are some issues with current 3 actually, it contains very minimal values, also model is being
decided
"""
import matplotlib
import seaborn as sb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dtale
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
import joblib


data = pd.read_excel("C:/Users/user/Documents/electric_data.xlsx")
data = data.drop("Unnamed: 0", axis = 1)
data = data.drop("Total Genset Time", axis = 1)
data = data.replace(0, np.nan)
data = data.dropna()
voltage = data.iloc[:,5:8]
v2  = voltage.copy()
model_prep = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2))])
voltage = model_prep.fit_transform(voltage)
model2 = IsolationForest(contamination = 0.10)
model = OneClassSVM(nu = 0.1)
labels = model.fit_predict(voltage)
labels2 = model2.fit_predict(v2.values)
joblib.dump(model, "OneClassSVM.pkl")
joblib.dump(model2, "Isolationforest.pkl")
v2['isolation'] = labels2
v2['oneclassSVM'] = labels

