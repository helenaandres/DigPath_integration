import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import os

import argparse

#from functions_analysis import *
from misc.dataset import Dataset, DatasetWhole, DatasetWhole_clasif, Dataset_alternative_bin
from misc.helpers import normalizeRNA,save_embedding

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelBinarizer
from misc.helpers import normalizeRNA,save_embedding



def classify(data, labels, clasif_type, cross_val = False):
    
    if cross_val == False:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, 
                                                    test_size=0.2, random_state=42, stratify=labels)
        if clasif_type == 'RF':
             clf = make_pipeline(StandardScaler(),RandomForestClassifier(max_depth=10, 
                                                                         random_state=42, class_weight = 'balanced'))
        elif clasif_type == 'SVM': 
             clf = make_pipeline(StandardScaler(),SVC(gamma='auto', class_weight = 'balanced', probability=True))
        elif clasif_type == 'LogReg': 
             clf = make_pipeline(StandardScaler(),LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000))
        elif clasif_type == 'NB':
             clf = make_pipeline(StandardScaler(),GaussianNB())

        print(X_train.shape)
        print(y_train.shape)
        print(clasif_type)
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_train)
        conf_mat = confusion_matrix(y_train, y_pred)
        y_pred_test = clf.predict(X_test)
        conf_mat_test = confusion_matrix(y_test, y_pred_test)
        print(conf_mat_test)

     
        if clf.predict_proba(X_test).shape[1] >2:
         
            roc_auc_test = roc_auc_score(y_test,
                                                  clf.predict_proba(X_test), multi_class = 'ovo')
            
            roc_auc_train = roc_auc_score(y_train,
                                                   clf.predict_proba(X_train), multi_class = 'ovo')               

            
        else:
             
            roc_auc_test = roc_auc_score(y_test,
                                                  clf.predict(X_test))
            
            roc_auc_train = roc_auc_score(y_train,
                                                   clf.predict(X_train))             


        train_score = np.mean(train_score)
        test_score = np.mean(test_score)
        roc_auc_train = np.mean(roc_auc_train)
        roc_auc_test = np.mean(roc_auc_test)
    
    elif cross_val == True:
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        train_score = []
        test_score = []
        roc_auc_test = []
        roc_auc_train = []
        i=0
        for train_index, test_index in skf.split(data, labels):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            if clasif_type == 'RF':
                 clf = make_pipeline(StandardScaler(),RandomForestClassifier(max_depth=2, random_state=0, class_weight = 'balanced'))
            elif clasif_type == 'SVM': 
                 clf = make_pipeline(StandardScaler(),SVC(gamma='auto', class_weight = 'balanced', probability=True))
            elif clasif_type == 'LogReg': 
                 clf = make_pipeline(StandardScaler(),LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000))
            elif clasif_type == 'NB':
                 clf = make_pipeline(StandardScaler(),GaussianNB())                    
            clf.fit(X_train, y_train)
            train_score.append(clf.score(X_train, y_train))
            test_score.append(clf.score(X_test, y_test))
            y_pred = clf.predict(data)
            conf_mat = confusion_matrix(labels, y_pred)
            y_pred_test = clf.predict(X_test)
            conf_mat_test = confusion_matrix(y_test, y_pred_test)
            if clf.predict_proba(X_test).shape[1] >2:
                #print(y_test.shape)
               # print(clf.predict_proba(X_test).shape)
                #print(y_train.shape)
                #print(clf.predict_proba(X_train).shape)
                roc_auc_test.append(roc_auc_score(y_test, clf.predict_proba(X_test), multi_class = 'ovo'))
                roc_auc_train.append(roc_auc_score(y_train, clf.predict_proba(X_train), multi_class = 'ovo'))
            else:
                roc_auc_test.append(roc_auc_score(y_test, clf.predict(X_test)))
                roc_auc_train.append(roc_auc_score(y_train, clf.predict(X_train)))
            #print(conf_mat_test)
            #ax = sns.heatmap(conf_mat_test, annot=True, cmap="YlGnBu")
            #plt.show()
            
            i+=1        
        train_score = np.mean(train_score)
        test_score = np.mean(test_score)
        roc_auc_train = np.mean(roc_auc_train)
        roc_auc_test = np.mean(roc_auc_test)
        
    return train_score, test_score, roc_auc_train, roc_auc_test

