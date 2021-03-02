import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import os

import argparse

#from misc.dataset import Dataset, DatasetWhole, DatasetWhole_clasif, Dataset_alternative_bin
from misc.dataset import Dataset, DatasetWhole
from misc.helpers import normalizeRNA,save_embedding
from misc.classification import classify

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


dataset = DatasetWhole('W')

targets = ['icnp', 'pam50np', 'ernp', 'drnp','hist_typnp','total_scorenp', 'tubule_scorenp', 'lymphonp', 'nuc_pleonp', 'overall_gradenp']
modalities = ['img_rhonp', 'img_snp', 'img_vnp','rnanp', 'cnanp', 'clin']
#modalities = ['rnanp', 'cnanp', 'clin']



mod_dict = {'rnanp':'GE', 'cnanp':'CNA', 'clin':'Clinical', 'img_rhonp':'Img_dens',
                   'img_snp':'Img_sim_s', 'img_vnp':'Img_sim_v'}
targets_dict = {'icnp':'iC10', 'pam50np':'PAM', 'ernp':'ER', 'drnp':'DR', 'hist_typnp':'Histological_Type',
                'total_scorenp':'total_score', 'tubule_scorenp':'tubule_score', 'lymphonp':'lymph_infilt', 
                'nuc_pleonp':'nuc_pleomorphism', 'overall_gradenp':'overall_grade'}

results = []
for target in targets:
    print('#################################')
    print(targets_dict[target])
    print('#################################')
    for modal in modalities: 
        print(mod_dict[modal])
        print('#############')
        if modal == 'rnanp':
            X_train = normalizeRNA(dataset.train[modal])
        else: 
            X_train = dataset.train[modal]

        y_train = dataset.train[target]
        
        for classif in ['RF', 'SVM', 'LogReg', 'NB']:
            print(classif)
            acc_tr, acc_test, roc_tr, roc_test = classify(X_train, y_train, 
                                                              clasif_type = classif,
                                                              cross_val = True)
            results.append([target, mod_dict[modal], classif, roc_tr, roc_test])


results = np.array(results)
print(results)        
colnames = ['target','input','classifier','train_roc_auc','test_roc_auc']
results_df = pd.DataFrame(data=results, columns=colnames)    
results_df.to_csv('./results/results_clasif_OneMod_k50_v1.csv')




