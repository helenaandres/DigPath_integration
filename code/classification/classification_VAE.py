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





targets_ = ['IC', 'PAM', 'ER', 'DR','total_score', 'tubule.formation', 'lymph_infiltrate', 'nuc_pleomorphism', 'overall_grade']

folds = ['1','2','3','4','5']

modalities_combos = ['Clin+mRNA','Clin+CNA', 'CNA+mRNA','img_rho+mRNA', 'img_s+mRNA', 'img_v+mRNA','img_rho+CNA',
              'img_s+CNA','img_v+CNA','img_rho+Clin','img_s+Clin','img_v+Clin','Clin+mRNA+CNA','Clin+mRNA+img_rho',
              'Clin+mRNA+img_s','Clin+mRNA+img_v','Clin+CNA+img_rho','Clin+CNA+img_s','Clin+CNA+img_v',  'mRNA+CNA+img_rho',
              'mRNA+CNA+img_s','mRNA+CNA+img_v','mRNA+CNA+Clin+img_rho','mRNA+CNA+Clin+img_s', 'mRNA+CNA+Clin+img_v'  ]
targets_dict = {'IC':'iC10', 'PAM':'PAM', 'ER':'ER', 'DR':'DR',
                'total_score':'total_score', 'tubule.formation':'tubule_score', 'lymph_infiltrate':'lymph_infilt', 
                'nuc_pleomorphism':'nuc_pleomorphism', 'overall_grade':'overall_grade'}



results=[]
for input_type in modalities_combos: 
        print('#################################')
        print(input_type)
        print('#################################')
        
        for target in targets_:
            print('###############')
            print(targets_dict[target])
            for fold in folds:
                if input_type in ['Clin+mRNA', 'Clin+CNA', 'CNA+mRNA','img_rho+mRNA', 'img_s+mRNA', 'img_v+mRNA','img_rho+CNA',
              'img_s+CNA','img_v+CNA','img_rho+Clin','img_s+Clin','img_v+Clin']:
                    file = './results/HVAE_'+input_type+'_integration/hvae_LS_64_DS_128_mmd_beta_25/'+target+fold+'.npz'
                else:    
                    file = './results/HVAE_'+input_type+'_integration/hvae_LS_48_DS_144_mmd_beta_25/'+target+fold+'.npz'
                embed=np.load(file)
                emb_train = embed['emb_train']
                emb_test = embed['emb_test']
                emb_comb = np.concatenate((emb_train, emb_test))
                dataset = Dataset(target,format(fold))
                train_labels=dataset.train["overall_gradenp"]
                test_labels=dataset.test["overall_gradenp"]  
                comb_labels=np.concatenate((train_labels, test_labels))
                for classif in ['RF', 'SVM', 'LogReg', 'NB']:
                    print('###############')
                    print(classif)                
                    acc_tr, acc_test, roc_tr, roc_test = classify(emb_comb, comb_labels,
                                                              clasif_type = 'RF',
                                                              cross_val = True)
                    results.append([target, input_type, classif, fold, roc_tr, roc_test])
            
results = np.array(results)
print(results)        
colnames = ['target','input','classifier','fold','train_roc_auc','test_roc_auc']
results_df = pd.DataFrame(data=results, columns=colnames)    
results_df.to_csv('./analysis_concat/results_clasif_VAE_k50_test.csv')











                              
                              