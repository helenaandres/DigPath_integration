import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import sys, os

import argparse

#from misc.dataset import Dataset, DatasetWhole, DatasetWhole_feat, DatasetWhole_clasif, Dataset_alternative_bin
from misc.dataset import Dataset, DatasetWhole, DatasetWhole_feat, DatasetWhole_feat_T
#from misc.dataset import Dataset, DatasetWhole
from misc.helpers import normalizeRNA,save_embedding
from misc.classification import classify, feature_analysis_train

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
    dataset = DatasetWhole_feat('W_new_M')
    

    img_rho_feat=dataset.train[1] 
    img_s_feat=dataset.train[2] 
    img_v_feat=dataset.train[3]
    clin_feat=dataset.train[4]
    rna_feat=dataset.train[5]
    cna_feat=dataset.train[6]
    print(clin_feat)


    targets = ['icnp', 'pam50np', 'ernp', 'drnp','hist_typnp','total_scorenp', 'tubule_scorenp', 'lymphonp', 'nuc_pleonp', 'overall_gradenp']
    modalities = ['rnanp', 'cnanp', 'clin', 'img_rhonp', 'img_snp', 'img_vnp']
    mod_dict = {'rnanp':'GE', 'cnanp':'CNA', 'clin':'Clinical', 'img_rhonp':'Img_dens',
                   'img_snp':'Img_sim_s', 'img_vnp':'Img_sim_v'}
    targets_dict = {'icnp':'iC10', 'pam50np':'PAM', 'ernp':'ER', 'drnp':'DR', 'hist_typnp':'Histological_Type',
                'total_scorenp':'total_score', 'tubule_scorenp':'tubule_score', 'lymphonp':'lymph_infilt', 
                'nuc_pleonp':'nuc_pleomorphism', 'overall_gradenp':'overall_grade'}
    for target in ['overall_gradenp','lymphonp','total_scorenp','pam50np', 'ernp', 'drnp','icnp','hist_typnp','tubule_scorenp','nuc_pleonp']:
#for target in ['nuc_pleonp']:

        print('#################################')
        print('TARGET: ', targets_dict[target])
        print('#################################')
        print(targets_dict[target])
        print('#################################')
        modal1='rnanp'
        modal2='clin'
        modal3='img_vnp'
        modal3='img_rhonp'    
        print(mod_dict[modal1])
        print(mod_dict[modal2])
        print(mod_dict[modal3])
        print('#############')

        input_type = 'mRNA+Clin+V'
    
        if target == 'drnp':
            input_type = 'mRNA+Rho'
            clasif_type = 'SVM'
        elif target == 'ernp':
            input_type = 'mRNA'
            clasif_type = 'SVM'
        elif target == 'pam50np':
            input_type = 'mRNA'
            clasif_type = 'LogReg'
        elif target == 'overall_gradenp':
            input_type = 'mRNA+Clin+Rho'
            clasif_type = 'SVM'
        elif target == 'total_scorenp':
            input_type = 'mRNA'
            clasif_type = 'SVM'
        elif target == 'lymphonp':
            input_type = 'mRNA+V'
            clasif_type = 'SVM'
        elif target == 'icnp':
            input_type = 'mRNA'
            clasif_type = 'LogReg' 
        elif target == 'hist_typnp':
            input_type = 'mRNA+Clin+V'
            clasif_type = 'SVM'         
        elif target == 'tubule_scorenp':
            input_type = 'mRNA+CNA+S'
            clasif_type = 'RF'       
        elif target == 'nuc_pleonp':
            input_type = 'mRNA+CNA+Clin+Rho'
            clasif_type = 'RF'
        
        
        if input_type == 'rnanp':
            X_train_ = normalizeRNA(dataset.train[0][modal1])
            X_train_feat_ = rna_feat
        elif input_type == 'Clin': 
            X_train_ = dataset.train[0][modal2]
            X_train_feat_ = clin_feat

        elif input_type == 'mRNA+V':
                    X_train_ = np.concatenate((normalizeRNA(dataset.train[0]['rnanp']),dataset.train[0]['img_vnp']), axis=1)
                    X_train_feat_ = np.concatenate((rna_feat,img_v_feat))
        elif input_type == 'mRNA+Rho':
                   X_train_ = np.concatenate((normalizeRNA(dataset.train[0]['rnanp']),dataset.train[0]['img_rhonp']), axis=1)
                   X_train_feat_ = np.concatenate((rna_feat,img_rho_feat))
        elif input_type == 'mRNA+Clin+V':
                X_train_ = np.concatenate((normalizeRNA(dataset.train[0]['rnanp']),dataset.train[0]['clin'],dataset.train[0]['img_vnp']), axis=1)
                X_train_feat_ = np.concatenate((rna_feat,clin_feat,img_v_feat))
        elif input_type == 'mRNA+Clin+Rho':
                X_train_ = np.concatenate((normalizeRNA(dataset.train[0]['rnanp']),dataset.train[0]['clin'],
                                          dataset.train[0]['img_rhonp']), axis=1)
                X_train_feat_ = np.concatenate((rna_feat,clin_feat,img_rho_feat))
        elif input_type == 'mRNA+Clin+S':
                X_train_ = np.concatenate((normalizeRNA(dataset.train[0]['rnanp']),dataset.train[0]['clin'],
                                          dataset.train[0]['img_snp']), axis=1)
                X_train_feat_ = np.concatenate((rna_feat,clin_feat,img_s_feat))                
        elif input_type == 'mRNA+CNA+Clin+Rho':
                X_train_ = np.concatenate((normalizeRNA(dataset.train[0]['rnanp']),normalizeRNA(dataset.train[0]['cnanp']),dataset.train[0]['clin'],
                                          dataset.train[0]['img_rhonp']), axis=1)
                X_train_feat_ = np.concatenate((rna_feat,cna_feat,clin_feat,img_rho_feat))  
                
        results = []

        y_train_ = dataset.train[0][target]
        impurity = False
        data = X_train_
        labels = y_train_

        roc_auc_train, roc_auc_test, vals_  = feature_analysis_train(data, labels, X_train_feat_, clasif_type, impurity = False)


        print('ROC_AUC test', roc_auc_test)
    
        feature_importance = pd.DataFrame(list(zip(X_train_feat_, np.abs(np.mean(np.array(vals_), axis=0)))), columns=['col_name','feature_importance_vals_mean'])
        for i in np.arange(np.array(vals_).shape[0]):           
            feature_importance['feature_importance_vals_fold'+str(i)]=np.abs(np.array(vals_)[i,:])
        feature_importance.sort_values(by=['feature_importance_vals_mean'], ascending=False,inplace=True)

        feature_importance.to_csv('./results_feature_importance_universal/METABRIC_CT_'+target+'feature_importance_'+clasif_type+'_v0.csv', index=False)

        print('plot mean Permutation measures done!')
                    
                
elif str(dataset_public) == 'TCGA':
    dataset = DatasetWhole_feat_T('W_new_T')
    X_data_GE = normalizeRNA(dataset.train[0]['rnanp'])
    X_data_CNA = dataset.train[0]['cnanp']
    X_data_Clin = dataset.train[0]['clin']
    X_data_Rho = dataset.train[0]['img_rhonp']
    X_data_S = dataset.train[0]['img_snp']
    X_data_Sf = dataset.train[0]['img_vnp']   
    
    img_rho_feat=dataset.train[1] 
    img_rho_feat = np.array(img_rho_feat).reshape(-1,)    
    img_s_feat=dataset.train[2] 
    img_s_feat = np.array(img_s_feat).reshape(-1,)
    img_v_feat=dataset.train[3]
    img_v_feat = np.array(img_v_feat).reshape(-1,)
    clin_feat=dataset.train[4]
    rna_feat=dataset.train[5]
    cna_feat=dataset.train[6]
    print(clin_feat)    
    targets = ['ER Status By IHC', 
       'Neoplasm Histologic Type Name', 'IHC-HER2',
       'Overall Survival Status', 'IntClust']
    modalities = ['rnanp', 'cnanp', 'clin', 'img_rhonp', 'img_snp', 'img_vnp']
    mod_dict = {'rnanp':'GE', 'cnanp':'CNA', 'clin':'Clinical', 'img_rhonp':'Img_dens',
                   'img_snp':'Img_sim_s', 'img_vnp':'Img_sim_v'}

    targets_dict = {'IntClust':'iC10', 'ER Status By IHC':'ER', 'Neoplasm Histologic Type Name':'Histological_Type',
                'Overall Survival Status':'OS', 'IHC-HER2':'IHC'}
    
    for target in ['ER Status By IHC','Neoplasm Histologic Type Name', 'IHC-HER2','Overall Survival Status', 'IntClust']:

        print('#################################')
        print('TARGET: ', targets_dict[target])
        print('#################################')
        print(targets_dict[target])
        print('#################################')
        modal1='rnanp'
        modal2='clin'
        modal3='img_vnp'
        modal3='img_rhonp'    
        print(mod_dict[modal1])
        print(mod_dict[modal2])
        print(mod_dict[modal3])
        print('#############')

    
        if target == 'ER Status By IHC':
            input_type = 'mRNA+Clin+V'
            clasif_type = 'SVM'
        elif target == 'Neoplasm Histologic Type Name':
            input_type = 'mRNA+Clin+V'
            clasif_type = 'SVM'
        elif target == 'IHC-HER2':
            input_type = 'mRNA+CNA+Clin+V'
            clasif_type = 'RF'
        elif target == 'Overall Survival Status':
            input_type = 'V'
            clasif_type = 'RF'
        elif target == 'IntClust':
            input_type = 'mRNA'
            clasif_type = 'LogReg'
                                        

        y_train_ = dataset.train[0][target]
        #print(y_train_)
        #print(y_train_.astype(str).unique())
        Y_data = np.array(y_train_)
    
        if target == 'ER Status By IHC':
            target_dict = {'Positive':0, 'Negative':1, 'nan':2, 'Indeterminate':2}
            Y_data_cat = np.array([target_dict[x] for x in Y_data.astype(str)])
    
        elif target == 'ER Status IHC Percent Positive':  
            target_dict = {'<10%':0, '10-19%':1, '20-29%' :2, '30-39%':3, '40-49%':4, 
                       '50-59%':5, '60-69%':6, '70-79%':7, '80-89%':8, '90-99%':9, 'nan':10 }
            Y_data_cat = np.array([target_dict[x] for x in Y_data.astype(str)])
        
        elif target == 'Neoplasm Histologic Type Name':  
            target_dict = {'Infiltrating Lobular Carcinoma':1, 'Other, specify':0,
         'Infiltrating Ductal Carcinoma':2, 'Mixed Histology (please specify)':0,
         'Mucinous Carcinoma':0, 'Metaplastic Carcinoma':0, 'Infiltrating Carcinoma NOS':2,
         'Medullary Carcinoma':0, 'nan':0}        
            Y_data_cat = np.array([target_dict[x] for x in Y_data.astype(str)])
        
        elif target == 'IHC-HER2':  
            target_dict = {'Equivocal':0, 'Negative':1, 'Positive':2, 'nan':3, 'Indeterminate':3   }
            Y_data_cat = np.array([target_dict[x] for x in Y_data.astype(str)])
        
        elif target == 'Overall Survival Status': 
            target_dict = {'0:LIVING':0, '1:DECEASED':1  }
            Y_data_cat = np.array([target_dict[x] for x in Y_data.astype(str)])
        
        elif target == 'IntClust':  
            target_dict = {'7.0':6, '3.0':2, '6.0':5, '4.0':3, '8.0':7, '2.0':1, '10.0':9, '5.0':4, '1.0':0, '9.0':8,  }
            Y_data_cat = np.array([target_dict[x] for x in Y_data.astype(str)])
                   
            
            
            
        if input_type == 'mRNA':
            X_train_ = X_data_GE
            X_train_feat_ = rna_feat
            X_train_df = pd.DataFrame(X_train_)
            is_NaN = X_train_df.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_train_df[row_has_NaN]
            print(rows_with_NaN.shape)
            X_train_clean = X_train_df.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_cat,rows_with_NaN.index)   
    
        elif input_type == 'Clin': 
            X_train_ = X_data_Clin
            X_train_feat_ = clin_feat
            X_train_df = pd.DataFrame(X_train_)
            is_NaN = X_train_df.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_train_df[row_has_NaN]
            print(rows_with_NaN.shape)
            X_train_clean = X_train_df.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_cat,rows_with_NaN.index)   
        
        elif input_type == 'V': 
            X_train_ = X_data_Sf
            X_train_feat_ = img_v_feat
            X_train_df = pd.DataFrame(X_train_)
            is_NaN = X_train_df.isnull()
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = X_train_df[row_has_NaN]
            print(rows_with_NaN.shape)
            X_train_clean = X_train_df.drop(rows_with_NaN.index)
            Y_data = np.delete(Y_data_cat,rows_with_NaN.index)           

        elif input_type == 'mRNA+V':
                X_train_ = np.concatenate((X_data_GE,
                                          X_data_Sf), axis=1)

                X_train_feat_ = np.concatenate((rna_feat,img_v_feat))
                X_train_df = pd.DataFrame(X_train_)
                is_NaN = X_train_df.isnull()
                row_has_NaN = is_NaN.any(axis=1)
                rows_with_NaN = X_train_df[row_has_NaN]
                print(rows_with_NaN.shape)
                X_train_clean = X_train_df.drop(rows_with_NaN.index)
                Y_data = np.delete(Y_data_cat,rows_with_NaN.index)   
    
        elif input_type == 'CNA+V':
                X_train_ = np.concatenate((X_data_CNA,
                                          X_data_Sf), axis=1)

                X_train_feat_ = np.concatenate((cna_feat,img_v_feat))
                X_train_df = pd.DataFrame(X_train_)
                is_NaN = X_train_df.isnull()
                row_has_NaN = is_NaN.any(axis=1)
                rows_with_NaN = X_train_df[row_has_NaN]
                print(rows_with_NaN.shape)
                X_train_clean = X_train_df.drop(rows_with_NaN.index)
                Y_data = np.delete(Y_data_cat,rows_with_NaN.index)  
    
        elif input_type == 'mRNA+Rho':

                X_data_ = data_o[np.append(col_GE,np.array(col_imgs[1]))]
                X_train_feat_ = np.append(rna_feat,img_rho_feat)
                X_train_df = pd.DataFrame(X_train_)
                is_NaN = X_train_df.isnull()
                row_has_NaN = is_NaN.any(axis=1)
                rows_with_NaN = X_train_df[row_has_NaN]
                print(rows_with_NaN.shape)
                X_train_clean = X_train_df.drop(rows_with_NaN.index)
                Y_data = np.delete(Y_data_cat,rows_with_NaN.index)   
    
        elif input_type == 'Clin+Rho':
                X_train_ = data_o[np.append(col_Clin,col_imgs[1])]
                X_train_feat_ = np.append(clin_feat,img_rho_feat)
                X_train_df = pd.DataFrame(X_train_)
                is_NaN = X_train_df.isnull()
                row_has_NaN = is_NaN.any(axis=1)
                rows_with_NaN = X_train_df[row_has_NaN]
                print(rows_with_NaN.shape)
                X_train_clean = X_train_df.drop(rows_with_NaN.index)
                Y_data = np.delete(Y_data_cat,rows_with_NaN.index)    
    
        elif input_type == 'mRNA+Clin+V':
                X_train_ = np.concatenate((X_data_GE,X_data_Clin,
                                          X_data_Sf), axis=1)
                
                print(rna_feat.shape)
                print(clin_feat.shape)
                print(np.array(img_v_feat).reshape(-1,).shape)

                X_train_feat_ = np.concatenate((rna_feat,clin_feat,img_v_feat))
                X_train_df = pd.DataFrame(X_train_)
                is_NaN = X_train_df.isnull()
                row_has_NaN = is_NaN.any(axis=1)
                rows_with_NaN = X_train_df[row_has_NaN]
                print(rows_with_NaN.shape)
                X_train_clean = X_train_df.drop(rows_with_NaN.index)
                Y_data = np.delete(Y_data_cat,rows_with_NaN.index)
        elif input_type == 'mRNA+CNA+Clin+V':
                X_train_ = np.concatenate((X_data_GE,X_data_CNA,X_data_Clin,
                                          X_data_Sf), axis=1)

                X_train_feat_ = np.concatenate((rna_feat,cna_feat,clin_feat,img_v_feat))
                X_train_df = pd.DataFrame(X_train_)
                is_NaN = X_train_df.isnull()
                row_has_NaN = is_NaN.any(axis=1)
                rows_with_NaN = X_train_df[row_has_NaN]
                print(rows_with_NaN.shape)
                X_train_clean = X_train_df.drop(rows_with_NaN.index)
                Y_data = np.delete(Y_data_cat,rows_with_NaN.index)   
    
        results = []

        y_train_ = np.array(dataset.train[0][target])
        impurity = False
    #data = X_train_
    #labels = y_train_
    #print(np.array(y_train_).shape)
        roc_auc_train, roc_auc_test, vals_  = feature_analysis_train(X_train_clean.values, Y_data, X_train_feat_, clasif_type, impurity = False)

    
        print('ROC_AUC test', roc_auc_test)
    
        feature_importance = pd.DataFrame(list(zip(X_train_feat_, np.abs(np.mean(np.array(vals_), axis=0)))), columns=['col_name','feature_importance_vals_mean'])
        for i in np.arange(np.array(vals_).shape[0]):           
            feature_importance['feature_importance_vals_fold'+str(i)]=np.abs(np.array(vals_)[i,:])
        feature_importance.sort_values(by=['feature_importance_vals_mean'], ascending=False,inplace=True)

        feature_importance.to_csv('./results_feature_importance_universal/TCGA_'+target+'feature_importance_'+clasif_type+'_v0.csv', index=False)

        print('plot mean Permutation measures done!')
                   
    
    


         