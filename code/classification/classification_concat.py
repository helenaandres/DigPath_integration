import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import os

import argparse

from misc.dataset import Dataset, DatasetWhole, DatasetWhole_clasif, Dataset_alternative_bin
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
modalities = ['rnanp', 'cnanp', 'clin', 'img_rhonp', 'img_snp', 'img_vnp']

modalities_combos = ['GE+CNA','GE+Clin','GE+rho','GE+S','GE+V','CNA+rho','CNA+S','CNA+V',
                               'Clin+rho','Clin+S','Clin+V','CNA+Clin','GE+CNA+Clin','GE+CNA+rho','GE+CNA+S','GE+CNA+V',
                              'GE+Clin+rho','GE+Clin+S','GE+Clin+V','CNA+Clin+rho','CNA+Clin+S','CNA+Clin+V',
                              'GE+CNA+Clin+rho','GE+CNA+Clin+S','GE+CNA+Clin+V']


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
    for input_type in modalities_combos: 
        print(input_type)
        print('#############')
        
        y_train = dataset.train[target]
        
        if input_type == 'GE+CNA':
            X_train = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['cnanp']), axis=1)

        elif input_type == 'GE+Clin':
            X_train = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['clin']), axis=1)

        elif input_type == 'GE+rho':
            X_train = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['img_rhonp']), axis=1)

        elif input_type == 'GE+S':
            X_train = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['img_snp']), axis=1)

        elif input_type == 'GE+V':
            X_train = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['img_vnp']), axis=1)

        elif input_type == 'CNA+Clin':
            X_train = np.concatenate((dataset.train['cnanp'],dataset.train['clin']), axis=1)

        elif input_type == 'CNA+rho':
            X_train = np.concatenate((dataset.train['cnanp'],dataset.train['img_rhonp']), axis=1)

        elif input_type == 'CNA+S':
            X_train = np.concatenate((dataset.train['cnanp'],dataset.train['img_snp']), axis=1)

        elif input_type == 'CNA+V':
            X_train = np.concatenate((dataset.train['cnanp'],dataset.train['img_vnp']), axis=1)

        elif input_type == 'Clin+rho':
            X_train = np.concatenate((dataset.train['clin'],dataset.train['img_rhonp']), axis=1)

        elif input_type == 'Clin+S':
            X_train = np.concatenate((dataset.train['clin'],dataset.train['img_snp']), axis=1)

        elif input_type == 'Clin+V':
            X_train = np.concatenate((dataset.train['clin'],dataset.train['img_vnp']), axis=1)

        elif input_type == 'GE+CNA+Clin':
            X_train = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['cnanp'],dataset.train['clin']), axis=1)

        elif input_type == 'GE+CNA+rho':
            X_train = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['cnanp'],dataset.train['img_rhonp']), axis=1)

        elif input_type == 'GE+CNA+S':
            X_train = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['cnanp'],dataset.train['img_snp']), axis=1)

        elif input_type == 'GE+CNA+V':
            X_train = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['cnanp'],dataset.train['img_vnp']), axis=1)

        elif input_type == 'CNA+Clin+rho':
            X_train = np.concatenate((dataset.train['cnanp'],dataset.train['clin'],dataset.train['img_rhonp']), axis=1)

        elif input_type == 'CNA+Clin+S':
            X_train = np.concatenate((dataset.train['cnanp'],dataset.train['clin'],dataset.train['img_snp']), axis=1)

        elif input_type == 'CNA+Clin+V':
            X_train = np.concatenate((dataset.train['cnanp'],dataset.train['clin'],dataset.train['img_vnp']), axis=1)

        elif input_type == 'GE+Clin+rho':
            X_train = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['clin'],dataset.train['img_rhonp']), axis=1)

        elif input_type == 'GE+Clin+S':
            X_train = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['clin'],dataset.train['img_snp']), axis=1)

        elif input_type == 'GE+Clin+V':
            X_train = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['clin'],dataset.train['img_vnp']), axis=1)

        elif input_type == 'GE+CNA+Clin+rho':
            X_train = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['cnanp'],dataset.train['clin'],dataset.train['img_rhonp']), axis=1)
               
        elif input_type == 'GE+CNA+Clin+S':
            X_train = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['cnanp'],dataset.train['clin'],dataset.train['img_snp']), axis=1)
              
        elif input_type == 'GE+CNA+Clin+V':
            X_train = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['cnanp'],dataset.train['clin'],dataset.train['img_vnp']), axis=1)
         
               
        for classif in ['RF', 'SVM', 'LogReg', 'NB']:
            print(classif)
            acc_tr, acc_test, roc_tr, roc_test = classify(X_train, y_train, 
                                                              clasif_type = classif,
                                                              cross_val = True)
            results.append([target, input_type, classif, roc_tr, roc_test])


results = np.array(results)
print(results)        
colnames = ['target','input','classifier','train_roc_auc','test_roc_auc']
results_df = pd.DataFrame(data=results, columns=colnames)    
results_df.to_csv('./analysis_concat/results_clasif_Concat_k50_v1.csv')











                              
                              