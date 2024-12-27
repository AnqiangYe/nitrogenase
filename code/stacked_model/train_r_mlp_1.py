# -*- coding: utf-8 -*-
from multiprocessing import Pool
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, StratifiedShuffleSplit, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import random
from sklearn import svm
import os
import sys
from scipy import stats
import threading
import time
from random import randint, sample
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve, auc, balanced_accuracy_score, classification_report
from sklearn.svm import SVC as SVC_gpu
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

random.seed(0)
df=pd.read_csv('../Data/dataset.csv', encoding='ISO-8859-1')
# Normalize the features extracted from the ProtT5 model
scaler = MinMaxScaler()
df.iloc[:, 1:1025] = scaler.fit_transform(df.iloc[:, 1:1025])

XX = df.iloc[:,1:1025]
YY = df["label_regression"]

train_index = pd.read_csv('../Data/train_index.csv')
indices = train_index['train_index']

XX = XX.iloc[indices]
YY = YY.iloc[indices]


print(XX)
print(YY)
XX = pd.DataFrame(XX)

XX = XX.astype(np.float32)
YY = YY.astype(np.float32)

#Create a file in advance to store the results
output = open(f'../result.csv', 'a')
output.write('train_r2,val_r2,train_spear,val_spear,mae_train,mae_val,mse_train,mse_val,rmse_train,rmse_val\n')

result = []


r2 = 0


train_r2 = []
val_r2 = []
train_spear=[]
val_spear=[]
mae_train = []
mae_val = []
mse_train = []
mse_val = []
rmse_train = []
rmse_val = []

if len(XX) == 0 or len(YY) == 0:
    print("Error: Empty dataset. Please check your data preprocessing.")
else:
    Kf1 = KFold(n_splits=5, random_state=0, shuffle=True)

    for XX0_index, val_index in Kf1.split(XX, YY):
        XX0 = XX.iloc[XX0_index]
        YY0 = YY.iloc[XX0_index]
        val_x = XX.iloc[val_index]
        val_y = YY.iloc[val_index]

        train_x = XX0
        train_y = YY0
        base_model_svr = MLPRegressor(hidden_layer_sizes=(100, 200, 200, 50), activation="relu", shuffle=False,
                                              solver="adam", alpha=0.001, random_state=100)

        clf = base_model_svr
        clf.fit(train_x, train_y)

        y_pred = clf.predict(train_x)
        r2_val = r2_score(train_y, y_pred)
        spearman_train, _ = spearmanr(train_y, y_pred)
        train_mae = mean_absolute_error(train_y, y_pred)
        train_mse = mean_squared_error(train_y, y_pred)
        train_rmse = np.sqrt(train_mse)
        print("Train R^2 score:", r2_val)
        y_pred2 = clf.predict(val_x)
        r2_val2 = r2_score(val_y, y_pred2)
        spearman_val, _ = spearmanr(val_y, y_pred2)
        val_mae = mean_absolute_error(val_y, y_pred2)
        val_mse = mean_squared_error(val_y, y_pred2)
        val_rmse = np.sqrt(val_mse)
        print("Val R^2 score:", r2_val2)
        train_r2.append(r2_val)
        val_r2.append(r2_val2)
        train_spear.append(spearman_train)
        val_spear.append(spearman_val)
        mae_train.append(train_mae)
        mae_val.append(val_mae)
        mse_train.append(train_mse)
        mse_val.append(val_mse)
        rmse_train.append(train_rmse)
        rmse_val.append(val_rmse)

        if r2_val2 > r2:
            r2 = r2_val2
            best_model = clf

    temp = [np.mean(train_r2), np.mean(val_r2), np.mean(train_spear), np.mean(val_spear),
            np.mean(mae_train), np.mean(mae_val), np.mean(mse_train), np.mean(mse_val), np.mean(rmse_train),
            np.mean(rmse_val)]
    output = open(f'../result.csv', 'a')
    output.write(
        f'{np.mean(train_r2)},{np.mean(val_r2)},{np.mean(train_spear)},{np.mean(val_spear)},{np.mean(mae_train)},{np.mean(mae_val)},{np.mean(mse_train)},{np.mean(mse_val)},{np.mean(rmse_train)},{np.mean(rmse_val)}\n')
    result.append(temp)
    output.close()
    output.close()
XX = df.iloc[:, 1:1025]
YY = df["label_regression"]
XX = pd.DataFrame(XX)
XX = XX.astype(np.float32)
YY = YY.astype(np.float32)
clf2 = best_model
#Obtain the first layer prediction data of the model
y_pred = clf2.predict(XX)
df2 = pd.DataFrame({"Actual": YY,"Predicted": y_pred})
df2.to_csv("../pre3.csv", index=False)
