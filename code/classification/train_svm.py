from multiprocessing import Pool
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV,StratifiedShuffleSplit
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
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score,roc_curve,auc,balanced_accuracy_score,classification_report
from sklearn.svm import SVC as SVC_gpu
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

random.seed(0)
df=pd.read_csv('../Data/dataset.csv', encoding='ISO-8859-1')
#Normalize the features extracted from the ProtT5 model
scaler = MinMaxScaler()
df.iloc[:, 1:1025] = scaler.fit_transform(df.iloc[:, 1:1025])

XX = df.iloc[:,1:1939]
YY = df["label_classification"]


train_index = pd.read_csv('../Data/train_index.csv')
indices = train_index['train_index']

XX = XX.iloc[indices]
YY = YY.iloc[indices]

print(XX)
print(YY)
XX = pd.DataFrame(XX)

XX = XX.astype(np.float32)
YY = YY.astype(np.int32)

counter = Counter(YY)
print(counter)


#Create a file in advance to store the results
output = open(f'../result.csv', 'a')
output.write('k,c,g,train_ACC,val_ACC,train_bACC,val_bACC,train_AUC,val_AUC,train_Pre,val_Pre,train_Rec,val_Rec,train_f1,val_f1\n')
k = 0
result = []
C = [0.1, 1, 10,100]
gamma = [0.001, 0.01, 0.1, 1]
aucmax=0
for c in C:
    for g in gamma:
        k += 1

        # AUC_m = []
        Sco_1 = []
        Sco_2 = []
        Sco_22 = []
        AUC_1 = []
        ACC_1 = []
        f1_1 = []
        recall_1 = []
        Pre_1 = []
        AUC_2 = []
        ACC_2 = []
        f1_2 = []
        recall_2 = []
        Pre_2 = []
        AUC_22 = []
        ACC_22 = []
        f1_22 = []
        recall_22 = []
        Pre_22 = []

        Kf1 = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        if len(XX) == 0 or len(YY) == 0:
            print("Error: Empty dataset. Please check your data preprocessing.")
        else:
            for XX0_index, val_index in Kf1.split(XX, YY):
                XX0 = XX.iloc[XX0_index]
                YY0 = YY.iloc[XX0_index]
                val_x = XX.iloc[val_index]
                val_y = YY.iloc[val_index]
                counter = Counter(val_y)
                print("val:",counter)

                counter = Counter(YY0)
                print(counter)
                print(counter)

                train_x = XX0
                train_y = YY0


                clf = SVC_gpu(kernel='rbf', class_weight='balanced',C=c,gamma=g,probability=True)


                clf.fit(train_x, train_y)


                s1 = clf.score(train_x, train_y)

                result_1 = clf.decision_function(train_x)#[:, 1]

                result_3 = clf.predict(train_x)


                s3 = balanced_accuracy_score(train_y, result_3)


                fpr1, tpr1, threshold1 = roc_curve(train_y, result_1)
                s5 = auc(fpr1, tpr1)



                s7 = precision_score(train_y, result_3)
                s9 = recall_score(train_y, result_3)
                s11 = f1_score(train_y, result_3)

                Sco_1.append(s1)
                ACC_1.append(s3)
                AUC_1.append(s5)
                Pre_1.append(s7)
                recall_1.append(s9)
                f1_1.append(s11)

                s22 = clf.score(val_x, val_y)
                result_22 = clf.predict_proba(val_x)[:, 1]
                result_44 = clf.predict(val_x)
                s44 = balanced_accuracy_score(val_y, result_44)
                fpr22, tpr22, threshold22 = roc_curve(val_y, result_22)
                s66 = auc(fpr22, tpr22)
                s88 = precision_score(val_y, result_44)
                s100 = recall_score(val_y, result_44)
                s122 = f1_score(val_y, result_44)
                Sco_22.append(s22)
                ACC_22.append(s44)
                AUC_22.append(s66)
                Pre_22.append(s88)
                recall_22.append(s100)
                f1_22.append(s122)
                print(s1, s3, s5, s7, s9,s11, s66, s88)
                if s66 > aucmax:
                    aucmax = s66
                    best_model=clf


        temp = [k,c,g,np.mean(Sco_1),np.mean(Sco_22),np.mean(ACC_1),np.mean(ACC_22),
                 np.mean(AUC_1),np.mean(AUC_22),np.mean(Pre_1),np.mean(Pre_22),
                 np.mean(recall_1),np.mean(recall_22),np.mean(f1_1),np.mean(f1_22)]
        print(k, c, g, np.mean(AUC_22), np.mean(Pre_22))
        output = open(f'../result.csv', 'a')
        output.write(f'{k},{c},{g},{np.mean(Sco_1)},{np.mean(Sco_22)},{np.mean(ACC_1)},{np.mean(ACC_22)},'
                 f'{np.mean(AUC_1)},{np.mean(AUC_22)},{np.mean(Pre_1)},{np.mean(Pre_22)},'
                 f'{np.mean(recall_1)},{np.mean(recall_22)},{np.mean(f1_1)},{np.mean(f1_22)}\n')
        result.append(temp)
        output.close()
#Validate the performance on the test set
XX = df.iloc[:, 1:1939]
YY = df["label_classification"]
test_index = pd.read_csv('../Data/test_index.csv')
indices_test = test_index['test_index']
XX = XX.iloc[indices_test]
YY = YY.iloc[indices_test]
XX = pd.DataFrame(XX)
XX = XX.astype(np.float32)
YY = YY.astype(np.int32)
clf2=best_model
s22 = clf2.score(XX, YY)
result_22 = clf2.predict_proba(XX)[:, 1]
result_44 = clf2.predict(XX)
s44 = balanced_accuracy_score(YY, result_44)
fpr22, tpr22, threshold22 = roc_curve(YY, result_22)
s66 = auc(fpr22, tpr22)
s88 = precision_score(YY, result_44)
s100 = recall_score(YY, result_44)
s122 = f1_score(YY, result_44)
print("AUC:",s66)
print("precision:",s88)
print("recall:",s100)
print("f1:",s122)
