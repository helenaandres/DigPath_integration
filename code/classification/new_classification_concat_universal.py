import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import sys, os

import argparse

from misc.dataset import Dataset, DatasetWhole, DatasetWhole_T
#from misc.dataset_k100 import Dataset, DatasetWhole
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


dataset_public = sys.argv[1]
print(str(dataset_public))

if str(dataset_public) == 'METABRIC':
    dataset = DatasetWhole('W_new_M')
    
    targets = ['icnp', 'pam50np', 'ernp', 'drnp','hist_typnp','total_scorenp', 'tubule_scorenp', 'lymphonp', 'nuc_pleonp', 'overall_gradenp']
    #modalities = ['img_rhonp', 'img_snp', 'img_vnp','rnanp', 'cnanp', 'clin']
    modalities = ['GE+CNA','GE+Clin','CNA+Clin', 
                    'GE+Rho', 'GE+S', 'GE+S_f', 
                    'CNA+Rho', 'CNA+S', 'CNA+S_f', 'Clin+Rho', 'Clin+S', 'Clin+S_f',
                    'GE+CNA+Rho', 'GE+CNA+S','GE+CNA+S_f','GE+Clin+Rho', 'GE+Clin+S','GE+Clin+S_f',
                    'CNA+Clin+Rho', 'CNA+Clin+S','CNA+Clin+S_f','GE+CNA+Clin', 'Rho+S+S_f',
                    'GE+CNA+Clin+Rho', 'GE+CNA+Clin+S', 'GE+CNA+Clin+S_f',
                    ]    
    mod_dict = {'rnanp':'GE', 'cnanp':'CNA', 'clin':'Clinical', 'img_rhonp':'Img_dens',
                   'img_snp':'Img_sim_s', 'img_vnp':'Img_sim_v'}
    targets_dict = {'icnp':'iC10', 'pam50np':'PAM', 'ernp':'ER', 'drnp':'DR', 'hist_typnp':'Histological_Type',
                'total_scorenp':'total_score', 'tubule_scorenp':'tubule_score', 'lymphonp':'lymph_infilt', 
                'nuc_pleonp':'nuc_pleomorphism', 'overall_gradenp':'overall_grade'}

elif str(dataset_public) == 'TCGA':
    dataset = DatasetWhole_T('W_new_T')
    #dataset = DatasetWhole('W_new_T')
    
    modalities = ['GE+CNA','GE+Clin','CNA+Clin', 
                    'GE+Rho', 'GE+S', 'GE+S_f', 
                    'CNA+Rho', 'CNA+S', 'CNA+S_f', 'Clin+Rho', 'Clin+S', 'Clin+S_f',
                    'GE+CNA+Rho', 'GE+CNA+S','GE+CNA+S_f','GE+Clin+Rho', 'GE+Clin+S','GE+Clin+S_f',
                    'CNA+Clin+Rho', 'CNA+Clin+S','CNA+Clin+S_f','GE+CNA+Clin', 'Rho+S+S_f',
                    'GE+CNA+Clin+Rho', 'GE+CNA+Clin+S', 'GE+CNA+Clin+S_f',
                    ]
    
    #modalities = ['CNA+Rho']
    targets = ['ER Status By IHCnp', 
       'Neoplasm Histologic Type Namenp', 'IHC-HER2np',
       'Overall Survival Statusnp', 'IntClustnp']
    
    
    mod_dict = {'rnanp':'GE', 'cnanp':'CNA', 'clin':'Clinical', 'img_rhonp':'Img_dens',
                   'img_snp':'Img_sim_s', 'img_vnp':'Img_sim_v'}
    targets_dict = {'ER Status By IHCnp':'ER Status By IHC', 
       'Neoplasm Histologic Type Namenp':'Neoplasm Histologic Type Name', 'IHC-HER2np':'IHC-HER2',
       'Overall Survival Statusnp':'Overall Survival Status', 'IntClustnp':'IntClust'}




results = []
for target in targets:
    print('#################################')
    #print(targets_dict[target])
    print(target)
    print('#################################')
    
    #Y_data = data[target]
    Y_data_ = dataset.train[target]
    #print(Y_data.astype(str).unique())

    i=0
    for modal in modalities: 
        print(modal)
        print('#############')        
        if modal == 'GE+CNA':
            #X_data_ = data[np.concatenate([col_GE,col_CNA])]
            X_data_ = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['cnanp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)
        elif modal == 'GE+Clin':
            X_data_ = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['clin']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)
        elif modal == 'GE+Rho':
            X_data_ = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['img_rhonp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)
        elif modal == 'GE+S':
            X_data_ = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['img_snp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)
        elif modal == 'GE+S_f':
            X_data_ = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['img_vnp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)
        elif modal == 'CNA+Rho':
            X_data_ = np.concatenate((dataset.train['cnanp'],dataset.train['img_rhonp']), axis=1)
            #print(dataset.train['img_rhonp'])
            #X_data_ = np.concatenate((normalizeRNA(dataset.train['cnanp']),np.array(dataset.train['img_rhonp'][0])), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            print(dataset.train['cnanp'])
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            print(X_data_clean.shape)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)
        elif modal == 'CNA+S':
            X_data_ = np.concatenate((dataset.train['cnanp'],dataset.train['img_snp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)
        elif modal == 'CNA+S_f':
            X_data_ = np.concatenate((dataset.train['cnanp'],dataset.train['img_vnp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)
        elif modal == 'Clin+Rho':
            X_data_ = np.concatenate((dataset.train['clin'],dataset.train['img_rhonp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)
        elif modal == 'Clin+S':
            X_data_ = np.concatenate((dataset.train['clin'],dataset.train['img_snp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)
        elif modal == 'Clin+S_f':
            X_data_ = np.concatenate((dataset.train['clin'],dataset.train['img_vnp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)
        elif modal == 'GE+CNA+Clin':
            X_data_ = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['cnanp'],
                                     dataset.train['clin']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index) 
        elif modal == 'Rho+S+S_f':
            X_data_ = np.concatenate((dataset.train['img_rhonp'],dataset.train['img_snp'],
                                     dataset.train['img_vnp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)               
        elif modal == 'GE+CNA+Rho':
            X_data_ = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['cnanp'],
                                     dataset.train['img_rhonp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)       
        elif modal == 'GE+CNA+S':
            X_data_ = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['cnanp'],
                                     dataset.train['img_snp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)            
        elif modal == 'GE+CNA+S_f':
            X_data_ = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['cnanp'],
                                     dataset.train['img_vnp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)   
        elif modal == 'GE+Clin+Rho':
            X_data_ = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['clin'],
                                     dataset.train['img_rhonp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)      
        elif modal == 'GE+Clin+S':
            X_data_ = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['clin'],
                                     dataset.train['img_snp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)          
        elif modal == 'GE+Clin+S_f':
            X_data_ = np.concatenate((normalizeRNA(dataset.train['rnanp']),dataset.train['clin'],
                                     dataset.train['img_vnp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)  
        elif modal == 'CNA+Clin+Rho':
            X_data_ = np.concatenate((dataset.train['cnanp'],dataset.train['clin'],
                                     dataset.train['img_rhonp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)       
        elif modal == 'CNA+Clin+S':
            X_data_ = np.concatenate((dataset.train['cnanp'],dataset.train['clin'],
                                     dataset.train['img_snp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)           
        elif modal == 'CNA+Clin+S_f':
            X_data_ = np.concatenate((dataset.train['cnanp'],dataset.train['clin'],
                                     dataset.train['img_vnp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)            
        elif modal == 'GE+CNA+Clin+Rho':
            X_data_ = np.concatenate((normalizeRNA(dataset.train['rnanp']),
                                      dataset.train['cnanp'],dataset.train['clin'],
                                     dataset.train['img_rhonp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)           
        elif modal == 'GE+CNA+Clin+S':
            X_data_ = np.concatenate((normalizeRNA(dataset.train['rnanp']),
                                      dataset.train['cnanp'],dataset.train['clin'],
                                     dataset.train['img_snp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)
        elif modal == 'GE+CNA+Clin+S_f':
            X_data_ = np.concatenate((normalizeRNA(dataset.train['rnanp']),
                                      dataset.train['cnanp'],dataset.train['clin'],
                                     dataset.train['img_vnp']), axis=1)
            X_data_ = pd.DataFrame(X_data_)
            is_NaN = X_data_.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_data_[row_has_NaN]
            print('Rows with NaN : ',rows_with_NaN.shape[0])
            X_data_clean = X_data_.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_,rows_with_NaN.index)             
            

            
        for classif in ['RF', 'SVM', 'LogReg', 'NB']:
            print(classif)
  
            acc_tr, acc_test, roc_tr, roc_test, acc_tr_std, acc_test_std, roc_tr_std, roc_test_std = classify(X_data_clean.values, Y_data,  clasif_type = classif,cross_val = True)

            results.append([targets_dict[target], modal, classif, roc_tr, roc_tr_std, roc_test, roc_test_std])
            print([targets_dict[target], modal, classif, roc_test, roc_test_std])
        i+=1
        

results = np.array(results)
print(results)        
colnames = ['target','input','classifier','train_roc_auc_mean','train_roc_auc_std','test_roc_auc_mean', 'test_roc_auc_std']
results_df = pd.DataFrame(data=results, columns=colnames)    
results_df.to_csv('./results_universal/new_results_TCGA_NA_integ_concat_v0.csv', index=False)
