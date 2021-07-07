import numpy as np
import pandas as pd
import os


def to_categorical_TCGA(data, dtype=None):
    val_to_cat = {}
    cat = []
    index = 0
    #for val in data:
        
    if dtype == 'ER Status By IHC':
        target_dict = {'Positive':0, 'Negative':1, 'nan':2, 'Indeterminate':2}
        Y_data_cat = np.array([target_dict[str(x)] for x in data]) 
        #print(Y_data_cat)

    if dtype == 'ER Status IHC Percent Positive':  
        target_dict = {'<10%':0, '10-19%':1, '20-29%' :2, '30-39%':3, '40-49%':4, 
                       '50-59%':5, '60-69%':6, '70-79%':7, '80-89%':8, '90-99%':9, 'nan':10 }
        Y_data_cat = np.array([target_dict[str(x)] for x in data])
        #print(Y_data_cat)
        
    elif dtype == 'Neoplasm Histologic Type Name':  
        target_dict = {'Infiltrating Lobular Carcinoma':1, 'Other, specify':0,
         'Infiltrating Ductal Carcinoma':2, 'Mixed Histology (please specify)':0,
         'Mucinous Carcinoma':0, 'Metaplastic Carcinoma':0, 'Infiltrating Carcinoma NOS':2,
         'Medullary Carcinoma':0, 'nan':0}        
        Y_data_cat = np.array([target_dict[str(x)] for x in data])
        #print(Y_data_cat)
        
    elif dtype == 'IHC-HER2':  
        target_dict = {'Equivocal':0, 'Negative':1, 'Positive':2, 'nan':3, 'Indeterminate':3   }
        Y_data_cat = np.array([target_dict[str(x)] for x in data])
        #print(Y_data_cat)
        
    elif dtype == 'Overall Survival Status': 
        target_dict = {'0:LIVING':0, '1:DECEASED':1  }
        Y_data_cat = np.array([target_dict[str(x)] for x in data])
        #print(Y_data_cat)
        
    elif dtype == 'IntClust':  
        target_dict = {'7.0':6, '3.0':2, '6.0':5, '4.0':3, '8.0':7, '2.0':1, '10.0':9, '5.0':4, '1.0':0, '9.0':8,  }
        Y_data_cat = np.array([target_dict[str(x)] for x in data])
        #print(Y_data_cat)
        

    else:
        print('Target not recognised')
        #cat.append(val_to_cat[val])
            #print(val_to_cat)
    return Y_data_cat

def to_categorical_2(data, dtype=None):
    val_to_cat = {}
    cat = []
    index = 1
    for val in data:
        if dtype == 'ic':
            if val not in ['1', '2', '3', '4ER+', '4ER-', '5', '6', '7', '8', '9', '10']:
                val = '1'
            if val in ['4ER+','4ER-']:
                val='4'
        if str(val) == 'nan':
            #print(val)
            cat.append(0)
        elif val not in val_to_cat:
            #print(val)
            val_to_cat[val] = index
            cat.append(index)
            index += 1
        else:
            cat.append(val_to_cat[val])
    return np.array(cat)

def get_data(data):

    d = {}
    #clin_fold = data[["METABRIC_ID"]]
    clin_fold = data[["ID"]]

    rna = data[[col for col in data if col.startswith('GE')]]
    cna = data[[col for col in data if col.startswith('CNA')]]
    img_rho = data[[col for col in data if col.startswith('Rho_')]]
    img_s = data[[col for col in data if col.startswith('S_')]]
    img_v = data[[col for col in data if col.startswith('V_')]]
    
    
    
    d['ic'] = list(data['iC10'].values)
    d['pam50'] = list(data['Pam50Subtype'].values)
    d['er'] = list(data['ER_Expr'].values)
    d['pr'] = list(data['PR_Expr'].values)
    d['her2'] = list(data['Her2_Expr'].values)
    d['drnp'] = list(data['DR'].values)

    d['rnanp'] = normalizeRNA(rna.astype(np.float32).values)
    d['cnanp'] = ((cna.astype(np.float32).values + 2.0) / 4.0)
    d['img_rhonp'] = img_rho.astype(np.float32).values
    d['img_snp'] = img_s.astype(np.float32).values
    d['img_vnp'] = img_v.astype(np.float32).values
    d['icnp'] = to_categorical(d['ic'], dtype='ic')
    d['pam50np'] = to_categorical(d['pam50'])
    d['ernp'] = to_categorical(d['er'])
    d['prnp'] = to_categorical(d['pr'])
    d['her2np'] = to_categorical(d['her2'])
    d['drnp'] = to_categorical(d['drnp'])
    
   

    ## Oscar's targets
    
    d['total_scorenp'] = to_categorical(list(data['total_score'].values), dtype='total_score')
    d['tubule_scorenp'] = to_categorical(list(data['tubule.formation_score'].values), dtype='tubule.formation_score')
    d['lymphonp'] = to_categorical(list(data['lymphocytic_infiltrate'].values), dtype='lymphocytic_infiltrate')
    d['nuc_pleonp'] = to_categorical(list(data['nuc_pleomorphism_score'].values), dtype='nuc_pleomorphism_score')
    d['overall_gradenp'] = to_categorical(list(data['overall_grade'].values), dtype='overall_grade')
    d['hist_typnp'] = to_categorical(list(data['Histological_Type'].values), dtype='Histological_Type')
    
    """
    preprocessing for clinical data to match current pipeline
    """
    ## Clinical Data Quick Descriptions 
    # clin["Age_At_Diagnosis"]           # Truly numeric
    # clin["Breast_Tumour_Laterality"]   # Categorical "L, R" (3 unique)
    # clin["NPI"]                        # Truly numeric
    # clin["Inferred_Menopausal_State"]  # Categorical "Pre, Post" (3 unique)
    # clin["Lymph_Nodes_Positive"]       # Ordinal ints 0-24
    # clin["Grade"]                      # Ordinal string (come on) 1-3 + "?"
    # clin["Size"]                       # Truly Numeric
    # clin["Histological_Type"]          # Categorical strings (9 unique)
    # clin["Cellularity"]                # Categorical strings (4 unique)                              
    # clin["Breast_Surgery"]             # Categorical strings (3 Unique)
    # clin["CT"]                         # Categorical strings (9 unique)
    # clin["HT"]                         # Categorical strings (9 Unique)
    # clin["RT"]                         # Categorical strings (9 Unique)

    ## Clinical Data Transformations
    # On the basis of the above we will keep some as numeric and others into one-hot encodings 
    # (I am not comfortable binning the continuous numeric columns without some basis for their bins)
    # Or since we dont have that much anyway just one hot everything and use BCE Loss to train

    # We have to get the entire dataset, transform them into one-hots, bins
    #complete_data = r"../data/original/MBdata_33CLINwMiss_1KfGE_1KfCNA.csv"
    complete_data = r"/home/ICM_CG/Projects/METABRIC/DigPath_integration/data/MBdata_33CLINwMiss_1KfGE_1KfCNA_2.csv"
    # complete_data = pd.read_csv(complete_data).set_index("METABRIC_ID")
    complete_data =  pd.read_csv(complete_data, index_col=None, header=0)

    # Either we keep numerics as 
    clin_numeric = complete_data[["METABRIC_ID","Age_At_Diagnosis", "NPI", "Size"]]

    # Numerical binned to arbitrary ranges then one-hot dummies
    metabric_id = complete_data[["METABRIC_ID"]]
    aad = pd.get_dummies(pd.cut(complete_data["NPI"],10, labels=[1,2,3,4,5,6,7,8,9,10]),prefix="aad", dummy_na = True)  
    npi = pd.get_dummies(pd.cut(complete_data["NPI"],6, labels=[1,2,3,4,5,6]),prefix="npi", dummy_na = True)
    size = pd.get_dummies(complete_data["Size"], prefix = "size", dummy_na = True)


    # Categorical and ordinals to one-hot dummies
    btl = pd.get_dummies(complete_data["Breast_Tumour_Laterality"], prefix = "btl", dummy_na = True)
    ims = pd.get_dummies(complete_data["Inferred_Menopausal_State"], prefix = "ims", dummy_na = True)
    lnp = pd.get_dummies(complete_data["Lymph_Nodes_Positive"], prefix = "lnp", dummy_na = True)
    grade = pd.get_dummies(complete_data["Grade"], prefix = "grade", dummy_na = True)
    hist = pd.get_dummies(complete_data["Histological_Type"], prefix = "hist", dummy_na = True)
    cellularity = pd.get_dummies(complete_data["Cellularity"], prefix = "cellularity", dummy_na = True)
    ct = pd.get_dummies(complete_data["CT"], prefix = "ct", dummy_na = True)
    ht = pd.get_dummies(complete_data["HT"], prefix = "ht", dummy_na = True)
    rt = pd.get_dummies(complete_data["RT"], prefix = "rt", dummy_na = True)

    #clin_transformed = pd.concat([clin_numeric, btl, ims, lnp, grade, size, hist, cellularity, ct, ht, rt ], axis = 1) # 222 columns
    clin_transformed = pd.concat([clin_numeric, btl, ims, lnp, grade, size, cellularity, ct, ht, rt ], axis = 1) # 222 columns
    #clin_transformed = pd.concat([metabric_id, aad, npi, size, btl, ims, lnp, grade, size, hist, cellularity, ct, ht, rt ], axis = 1) # 2278 columns non binned, 350 columns if binned
    clin_transformed = pd.concat([metabric_id, aad, npi, size, btl, ims, lnp, grade, size, cellularity, ct, ht, rt ], axis = 1) # 2278 columns non binned, 350 columns if binned
    
    # Now create the fold data by selecting from the complete transformed clinical data
    # print(list(clin_fold.flatten()))
    fold_ids = [x.item() for x in list(clin_fold.values)]
    clin_transformed = clin_transformed.loc[clin_transformed['METABRIC_ID'].isin(fold_ids)]
    del clin_transformed['METABRIC_ID']

    d['clin'] = clin_transformed.astype(np.float32).values
    return d

def normalizeRNA(*args):
    if len(args) > 1: 
        normalizeData=np.concatenate((args[0],args[1]),axis=0)
        normalizeData=(normalizeData-normalizeData.min(axis=0))/(normalizeData.max(axis=0)-normalizeData.min(0))
        return normalizeData[:args[0].shape[0]], normalizeData[args[0].shape[0]:]
    else:
        return (args[0]-args[0].min(axis=0))/(args[0].max(axis=0)-args[0].min(0))
    

def save_embedding(savedir,savefile, *args):
    save_path = os.path.join(savedir, savefile)
    if len(args)>1:
        np.savez(save_path, emb_train=args[0],emb_test=args[1])
    else:
        np.savez(save_path, emb_train=args[0])
    
    

def get_data_feat(data):

    d = {}
    clin_fold = data[["ID"]]

    rna = data[[col for col in data if col.startswith('GE')]]
    cna = data[[col for col in data if col.startswith('CNA')]]
    img_rho = data[[col for col in data if col.startswith('Rho_')]]
    img_rho_feat = [col for col in data if col.startswith('Rho_')]
    img_s = data[[col for col in data if col.startswith('S_')]]
    img_s_feat = [col for col in data if col.startswith('S_')]
    img_v = data[[col for col in data if col.startswith('V_')]]
    img_v_feat = [col for col in data if col.startswith('V_')]
       
    d['ic'] = list(data['iC10'].values)
    d['pam50'] = list(data['Pam50Subtype'].values)
    d['er'] = list(data['ER_Expr'].values)
    d['pr'] = list(data['PR_Expr'].values)
    d['her2'] = list(data['Her2_Expr'].values)
    d['drnp'] = list(data['DR'].values)

    d['rnanp'] = normalizeRNA(rna.astype(np.float32).values)
    d['cnanp'] = ((cna.astype(np.float32).values + 2.0) / 4.0)
    d['img_rhonp'] = img_rho.astype(np.float32).values
    d['img_snp'] = img_s.astype(np.float32).values
    d['img_vnp'] = img_v.astype(np.float32).values
    d['icnp'] = to_categorical(d['ic'], dtype='ic')
    d['pam50np'] = to_categorical(d['pam50'])
    d['ernp'] = to_categorical(d['er'])
    d['prnp'] = to_categorical(d['pr'])
    d['her2np'] = to_categorical(d['her2'])
    d['drnp'] = to_categorical(d['drnp'])
    
   

    ## Oscar's targets
    
    d['total_scorenp'] = to_categorical(list(data['total_score'].values), dtype='total_score')
    d['tubule_scorenp'] = to_categorical(list(data['tubule.formation_score'].values), dtype='tubule.formation_score')
    d['lymphonp'] = to_categorical(list(data['lymphocytic_infiltrate'].values), dtype='lymphocytic_infiltrate')
    d['nuc_pleonp'] = to_categorical(list(data['nuc_pleomorphism_score'].values), dtype='nuc_pleomorphism_score')
    d['overall_gradenp'] = to_categorical(list(data['overall_grade'].values), dtype='overall_grade')
    d['hist_typnp'] = to_categorical(list(data['Histological_Type'].values), dtype='Histological_Type')
    
    """
    preprocessing for clinical data to match current pipeline
    """
    ## Clinical Data Quick Descriptions 
    # clin["Age_At_Diagnosis"]           # Truly numeric
    # clin["Breast_Tumour_Laterality"]   # Categorical "L, R" (3 unique)
    # clin["NPI"]                        # Truly numeric
    # clin["Inferred_Menopausal_State"]  # Categorical "Pre, Post" (3 unique)
    # clin["Lymph_Nodes_Positive"]       # Ordinal ints 0-24
    # clin["Grade"]                      # Ordinal string (come on) 1-3 + "?"
    # clin["Size"]                       # Truly Numeric
    # clin["Histological_Type"]          # Categorical strings (9 unique)
    # clin["Cellularity"]                # Categorical strings (4 unique)                              
    # clin["Breast_Surgery"]             # Categorical strings (3 Unique)
    # clin["CT"]                         # Categorical strings (9 unique)
    # clin["HT"]                         # Categorical strings (9 Unique)
    # clin["RT"]                         # Categorical strings (9 Unique)

    ## Clinical Data Transformations
    # On the basis of the above we will keep some as numeric and others into one-hot encodings 
    # (I am not comfortable binning the continuous numeric columns without some basis for their bins)
    # Or since we dont have that much anyway just one hot everything and use BCE Loss to train

    # We have to get the entire dataset, transform them into one-hots, bins
    complete_data = r"/home/ICM_CG/Projects/METABRIC/DigPath_integration/data/MBdata_33CLINwMiss_1KfGE_1KfCNA_2.csv"
    # complete_data = pd.read_csv(complete_data).set_index("METABRIC_ID")
    complete_data =  pd.read_csv(complete_data, index_col=None, header=0)

    # Either we keep numerics as 
    clin_numeric = complete_data[["METABRIC_ID","Age_At_Diagnosis", "NPI", "Size"]]

    # Numerical binned to arbitrary ranges then one-hot dummies
    metabric_id = complete_data[["METABRIC_ID"]]
    aad = pd.get_dummies(pd.cut(complete_data["NPI"],10, labels=[1,2,3,4,5,6,7,8,9,10]),prefix="aad", dummy_na = True)  
    npi = pd.get_dummies(pd.cut(complete_data["NPI"],6, labels=[1,2,3,4,5,6]),prefix="npi", dummy_na = True)
    size = pd.get_dummies(complete_data["Size"], prefix = "size", dummy_na = True)


    # Categorical and ordinals to one-hot dummies
    btl = pd.get_dummies(complete_data["Breast_Tumour_Laterality"], prefix = "btl", dummy_na = True)
    ims = pd.get_dummies(complete_data["Inferred_Menopausal_State"], prefix = "ims", dummy_na = True)
    lnp = pd.get_dummies(complete_data["Lymph_Nodes_Positive"], prefix = "lnp", dummy_na = True)
    grade = pd.get_dummies(complete_data["Grade"], prefix = "grade", dummy_na = True)
    hist = pd.get_dummies(complete_data["Histological_Type"], prefix = "hist", dummy_na = True)
    cellularity = pd.get_dummies(complete_data["Cellularity"], prefix = "cellularity", dummy_na = True)
    ct = pd.get_dummies(complete_data["CT"], prefix = "ct", dummy_na = True)
    ht = pd.get_dummies(complete_data["HT"], prefix = "ht", dummy_na = True)
    rt = pd.get_dummies(complete_data["RT"], prefix = "rt", dummy_na = True)

    #clin_transformed = pd.concat([clin_numeric, btl, ims, lnp, grade, size, hist, cellularity, ct, ht, rt ], axis = 1) # 222 columns
    clin_transformed = pd.concat([clin_numeric, btl, ims, lnp, grade, size, cellularity, ct, ht, rt ], axis = 1) # 222 columns
    #clin_transformed = pd.concat([metabric_id, aad, npi, size, btl, ims, lnp, grade, size, hist, cellularity, ct, ht, rt ], axis = 1) # 2278 columns non binned, 350 columns if binned
    clin_transformed = pd.concat([metabric_id, aad, npi, size, btl, ims, lnp, grade, size, cellularity, ct, ht, rt ], axis = 1) # 2278 columns non binned, 350 columns if binned
    
    # Now create the fold data by selecting from the complete transformed clinical data
    # print(list(clin_fold.flatten()))
    fold_ids = [x.item() for x in list(clin_fold.values)]
    clin_transformed = clin_transformed.loc[clin_transformed['METABRIC_ID'].isin(fold_ids)]
    del clin_transformed['METABRIC_ID']

    #print(clin_transformed.columns)
    #print(complete_data.columns[:100])
    d['clin'] = clin_transformed.astype(np.float32).values
    
    #print(rna.columns)
    return d, img_rho_feat, img_s_feat, img_v_feat, clin_transformed.columns, rna.columns, cna.columns

def get_data_T(data, cell_types=False):

    d = {}
    
    
    clin_fold = data[["ID"]]

    rna = data[[col for col in data if col.startswith('GE')]]
    rna = rna.drop(columns=['GE_ID'])
    cna = data[[col for col in data if col.startswith('CNA')]]
    cna = cna.drop(columns=['CNA_ID'])
    clin = data[[col for col in data if col.startswith('Clin')]]
    clin = clin.drop(columns=['Clin_Patient ID'])

    print(clin)
    if cell_types == True: 
        
        columns = data.columns
        col_imgs=columns[13:304] 
        col_GE=columns[304:1304]
        col_CNA=columns[1305:2304]
        col_Clin=columns[2307:]
        print('Columns clinical', col_Clin)
        print(col_imgs[1])
    
        img_rho = data[col_imgs[1]]
        img_rho= img_rho.values.reshape(-1, 1)
        img_s = data[col_imgs[2:4]]
        img_v = data[col_imgs[4:304]]
    if cell_types == False: 
        
        columns = data.columns
        col_imgs=columns[13:111] 
        col_GE=columns[111:1111]
        col_CNA=columns[1112:2112]
        col_Clin=columns[2115:]
        print('Columns clinical', col_Clin)
        print(col_imgs[1])
    
        img_rho = data[col_imgs[1]]
        img_rho= img_rho.values.reshape(-1, 1)
        img_s = data[col_imgs[2:4]]
        img_v = data[col_imgs[4:111]]
        #clin = data[col_Clin]
    #rna = data[col_GE]
    #cna = data[col_CNA]
    #clin = data[col_Clin]

    
    
    #clin_fold = data[["METABRIC_ID"]]
    clin_fold = data[["ID"]]

    #rna = data[[col for col in data if col.startswith('GE')]]
    #cna = data[[col for col in data if col.startswith('CNA')]]
    #img_rho = data[[col for col in data if col.startswith('Rho_')]]
    #img_s = data[[col for col in data if col.startswith('S_')]]
    #img_v = data[[col for col in data if col.startswith('V_')]]
    


    d['rnanp'] = normalizeRNA(rna.astype(np.float32).values)
    d['cnanp'] = ((cna.astype(np.float32).values + 2.0) / 4.0)
    #d['img_rhonp'] = img_rho.astype(np.float32).values
    d['img_rhonp'] = img_rho.astype(np.float32)
    d['img_snp'] = img_s.astype(np.float32).values
    d['img_vnp'] = img_v.astype(np.float32).values
    
    
    d['ER Status By IHC'] = list(data['ER Status By IHC'].values)
    d['Neoplasm Histologic Type Name'] = list(data['Neoplasm Histologic Type Name'].values)
    d['IHC-HER2'] = list(data['IHC-HER2'].values)
    d['Overall Survival Status'] = list(data['Overall Survival Status'].values)
    d['IntClust'] = list(data['IntClust'].values)

    
    d['ER Status By IHCnp'] = to_categorical_TCGA(d['ER Status By IHC'], dtype='ER Status By IHC')
    d['Neoplasm Histologic Type Namenp'] = to_categorical_TCGA(d['Neoplasm Histologic Type Name'], dtype = 'Neoplasm Histologic Type Name')
    d['IHC-HER2np'] = to_categorical_TCGA(d['IHC-HER2'], dtype ='IHC-HER2')
    d['Overall Survival Statusnp'] = to_categorical_TCGA(d['Overall Survival Status'], dtype='Overall Survival Status')
    d['IntClustnp'] = to_categorical_TCGA(d['IntClust'], dtype = 'IntClust')
    

    d['clin'] = clin.astype(np.float32).values
    return d
 
def get_data_feat_T(data, cell_types=False):

    d = {}
    
    
    clin_fold = data[["ID"]]

    rna = data[[col for col in data if col.startswith('GE')]]
    rna = rna.drop(columns=['GE_ID'])
    cna = data[[col for col in data if col.startswith('CNA')]]
    cna = cna.drop(columns=['CNA_ID'])
    clin = data[[col for col in data if col.startswith('Clin')]]
    clin = clin.drop(columns=['Clin_Patient ID'])

    print(clin)
    if cell_types == True: 
        
        columns = data.columns
        col_imgs=columns[13:304] 
        col_GE=columns[304:1304]
        col_CNA=columns[1305:2304]
        col_Clin=columns[2307:]
        print('Columns clinical', col_Clin)
        print(col_imgs[1])
    
        img_rho = data[col_imgs[1]]
        img_rho= img_rho.values.reshape(-1, 1)
        img_s = data[col_imgs[2:4]]
        img_v = data[col_imgs[4:304]]
        img_rho_feat = [col_imgs[1]]
        img_s_feat = [col_imgs[2:4]]
        img_v_feat = [col_imgs[4:304]]
        
        
    if cell_types == False: 
        
        columns = data.columns
        col_imgs=columns[13:111] 
        col_GE=columns[111:1111]
        col_CNA=columns[1112:2112]
        col_Clin=columns[2115:]
        print('Columns clinical', col_Clin)
        print(col_imgs[1])
    
        img_rho = data[col_imgs[1]]
        img_rho= img_rho.values.reshape(-1, 1)
        img_s = data[col_imgs[2:4]]
        img_v = data[col_imgs[4:111]]
        img_rho_feat = [col_imgs[1]]
        img_s_feat = [col_imgs[2:4]]
        img_v_feat = [col_imgs[4:111]]
        #clin = data[col_Clin]
    #rna = data[col_GE]
    #cna = data[col_CNA]
    #clin = data[col_Clin]

    
    
    #clin_fold = data[["METABRIC_ID"]]
    clin_fold = data[["ID"]]

    #rna = data[[col for col in data if col.startswith('GE')]]
    #cna = data[[col for col in data if col.startswith('CNA')]]
    #img_rho = data[[col for col in data if col.startswith('Rho_')]]
    #img_s = data[[col for col in data if col.startswith('S_')]]
    #img_v = data[[col for col in data if col.startswith('V_')]]
    


    d['rnanp'] = normalizeRNA(rna.astype(np.float32).values)
    d['cnanp'] = ((cna.astype(np.float32).values + 2.0) / 4.0)
    #d['img_rhonp'] = img_rho.astype(np.float32).values
    d['img_rhonp'] = img_rho.astype(np.float32)
    d['img_snp'] = img_s.astype(np.float32).values
    d['img_vnp'] = img_v.astype(np.float32).values
    
    
    d['ER Status By IHC'] = list(data['ER Status By IHC'].values)
    d['Neoplasm Histologic Type Name'] = list(data['Neoplasm Histologic Type Name'].values)
    d['IHC-HER2'] = list(data['IHC-HER2'].values)
    d['Overall Survival Status'] = list(data['Overall Survival Status'].values)
    d['IntClust'] = list(data['IntClust'].values)

    
    d['ER Status By IHCnp'] = to_categorical_TCGA(d['ER Status By IHC'], dtype='ER Status By IHC')
    d['Neoplasm Histologic Type Namenp'] = to_categorical_TCGA(d['Neoplasm Histologic Type Name'], dtype = 'Neoplasm Histologic Type Name')
    d['IHC-HER2np'] = to_categorical_TCGA(d['IHC-HER2'], dtype ='IHC-HER2')
    d['Overall Survival Statusnp'] = to_categorical_TCGA(d['Overall Survival Status'], dtype='Overall Survival Status')
    d['IntClustnp'] = to_categorical_TCGA(d['IntClust'], dtype = 'IntClust')
    

    d['clin'] = clin.astype(np.float32).values
    return d, img_rho_feat, img_s_feat, img_v_feat, clin.columns, rna.columns, cna.columns