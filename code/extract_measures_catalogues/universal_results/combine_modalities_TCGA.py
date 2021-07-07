import warnings
warnings.filterwarnings('ignore')
from collections import Counter

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import argparse
from scipy.stats import f_oneway



cell_type = False

if cell_type == True:
    
    name_output_file = './TCGA_combined_data_v0.csv'
    Imgs_TCGA = pd.read_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/code/extract_measures_catalogues/universal_results/TCGA_simil_measures_k10_v0.txt', delimiter = "\t", header=None)
    Imgs_TCGA_header = pd.read_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/code/extract_measures_catalogues/universal_results/header_simil_measures_k10__TCGA.txt', delimiter = " ", header=None)

    Imgs_TCGA_ = [np.array(i.split()) for i in Imgs_TCGA[0]]
    Imgs_TCGA__=[i for i in Imgs_TCGA_]
    head = Imgs_TCGA_header.values[0]
    Imgs_TCGA_df = pd.DataFrame(data=np.array(Imgs_TCGA__), columns=head)

    print('Digital pathology images data shape = ', np.array(Imgs_TCGA__).shape)
    print('Header shape = ',Imgs_TCGA_header.values.shape)     

    names_cat = pd.read_csv('/home/ICM_CG/Projects/METABRIC/TCGA/tcga-ffpe/025um/TCGA_catalogue_list_025.txt', header=None)

    Img_TCGA_IDs_ = [Imgs_TCGA_df['ID'][i][0:12] for i in np.arange(Imgs_TCGA_df['ID'].shape[0])]
    Img_TCGA_IDs_full_ = [Imgs_TCGA_df['ID'][i][0] for i in np.arange(Imgs_TCGA_df['ID'].shape[0])]
    Img_TCGA_IDs_ = np.array(Img_TCGA_IDs_)
    Imgs_TCGA_df['ID'] = Img_TCGA_IDs_
    Imgs_TCGA_df = Imgs_TCGA_df.replace({'nan': None})
    Imgs_TCGA_df.to_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/code/extract_measures_catalogues/universal_results/clean_measures_k10_TCGA_v0.csv', index=False)
    Img_TCGA = pd.read_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/code/extract_measures_catalogues/universal_results/clean_measures_k10_TCGA_v0.csv')    
    
    
elif cell_type == False:
    name_output_file = './TCGA_combined_data_NA_v0.csv'
    #Imgs_TCGA = pd.read_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/code/extract_measures_catalogues/universal_results/TCGA_simil_measures_k10_NA_v0.txt', delimiter = "\t", header=None)
    Imgs_TCGA = pd.read_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/code/extract_measures_catalogues/universal_results/TCGA_simil_measures_k10_NA_v0.txt', delimiter = " ", header=None)
    


    #Imgs_TCGA_ = [np.array(i.split()) for i in Imgs_TCGA]
    #print(Imgs_TCGA_)
    #Imgs_TCGA__=[i for i in Imgs_TCGA_]

    head = ['ID', 'Rho_AC', 'S_mean_AC', 'S_std_AC', 'X_global_AC_mean',
  'Y_global_AC_mean', 'area_cnt_AC_mean', 'area_minCircle_AC_mean',
  'area_ellipse_AC_mean', 'perimeter_AC_mean', 'eqDiamater_AC_mean',
  'extent_AC_mean', 'ell_angle_AC_mean', 'ell_majorAxis_AC_mean',
  'ell_minorAxis_AC_mean', 'ell_e_AC_mean', 'circularity_AC_mean',
  'roundness_AC_mean', 'compactness_AC_mean', 'AR_AC_mean', 'ell_AR_AC_mean',
  'solidity_AC_mean', 'flux_Ref_AC_mean', 'flux_B_AC_mean', 'flux_G_AC_mean',
  'flux_R_AC_mean', 'flux_Y_AC_mean', 'flux_Cr_AC_mean', 'flux_Cb_AC_mean',
  'flux_L_AC_mean', 'flux_a_AC_mean', 'flux_b_AC_mean',
  'fluxStd_Ref_AC_mean', 'fluxStd_B_AC_mean', 'fluxStd_G_AC_mean',
  'fluxStd_R_AC_mean', 'fluxStd_Y_AC_mean', 'fluxStd_Cr_AC_mean',
  'fluxStd_Cb_AC_mean', 'fluxStd_L_AC_mean', 'fluxStd_a_AC_mean',
  'fluxStd_b_AC_mean', 'Hu_01_AC_mean', 'Hu_02_AC_mean', 'Hu_03_AC_mean',
  'Hu_04_AC_mean', 'Hu_05_AC_mean', 'Hu_06_AC_mean', 'Hu_07_AC_mean',
  's2n_AC_mean', 'CNN_cell_type_conf_AC_mean', 'X_global_AC_std',
  'Y_global_AC_std', 'area_cnt_AC_std', 'area_minCircle_AC_std',
  'area_ellipse_AC_std', 'perimeter_AC_std', 'eqDiamater_AC_std',
  'extent_AC_std', 'ell_angle_AC_std', 'ell_majorAxis_AC_std',
  'ell_minorAxis_AC_std', 'ell_e_AC_std', 'circularity_AC_std',
  'roundness_AC_std', 'compactness_AC_std', 'AR_AC_std', 'ell_AR_AC_std',
  'solidity_AC_std', 'flux_Ref_AC_std', 'flux_B_AC_std', 'flux_G_AC_std',
  'flux_R_AC_std', 'flux_Y_AC_std', 'flux_Cr_AC_std', 'flux_Cb_AC_std',
  'flux_L_AC_std', 'flux_a_AC_std', 'flux_b_AC_std', 'fluxStd_Ref_AC_std',
  'fluxStd_B_AC_std', 'fluxStd_G_AC_std', 'fluxStd_R_AC_std',
  'fluxStd_Y_AC_std', 'fluxStd_Cr_AC_std', 'fluxStd_Cb_AC_std',
  'fluxStd_L_AC_std', 'fluxStd_a_AC_std', 'fluxStd_b_AC_std', 'Hu_01_AC_std',
  'Hu_02_AC_std', 'Hu_03_AC_std', 'Hu_04_AC_std', 'Hu_05_AC_std',
  'Hu_06_AC_std', 'Hu_07_AC_std', 's2n_AC_std', 'CNN_cell_type_conf_AC_std']
    Imgs_TCGA_df = pd.DataFrame(data=Imgs_TCGA.values, columns=head)

    print(Imgs_TCGA_df)
    
    print('Digital pathology images data shape = ', Imgs_TCGA.shape)
    print('Header shape = ',len(head)) 
    
    
    

    names_cat = pd.read_csv('/home/ICM_CG/Projects/METABRIC/TCGA/tcga-ffpe/025um/TCGA_catalogue_list_025.txt', header=None)

    Img_TCGA_IDs_ = [Imgs_TCGA_df['ID'][i][0:12] for i in np.arange(Imgs_TCGA_df['ID'].shape[0])]
    Img_TCGA_IDs_full_ = [Imgs_TCGA_df['ID'][i][0] for i in np.arange(Imgs_TCGA_df['ID'].shape[0])]
    Img_TCGA_IDs_ = np.array(Img_TCGA_IDs_)
    Imgs_TCGA_df['ID'] = Img_TCGA_IDs_
    Imgs_TCGA_df = Imgs_TCGA_df.replace({'nan': None})
    Imgs_TCGA_df.to_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/code/extract_measures_catalogues/universal_results/clean_measures_k10_TCGA_NA_v0.csv', index=False)
    Img_TCGA = pd.read_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/code/extract_measures_catalogues/universal_results/clean_measures_k10_TCGA_NA_v0.csv')

Clin_TCGA = pd.read_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/data/TCGA/brca_tcga_clinical_data.tsv', sep='\t')


GE_TCGA = pd.read_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/data/TCGA/data_RNA_Seq_v2_expression_median.txt', sep='\t')
CNA_TCGA = pd.read_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/data/TCGA/data_CNA.txt', sep='\t')
names_TCGA = pd.read_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/data/TCGA/GE_TCGA_names_breast.txt', sep='\t')

print('GE data size = ',GE_TCGA.shape)
print('CNA data size = ',CNA_TCGA.shape)
print('Clinical data columns = ',Clin_TCGA.columns)
#print(CNA_TCGA['Hugo_Symbol'])
#print(GE_TCGA['Hugo_Symbol'])
#print(np.array(list(set(list(CNA_TCGA['Hugo_Symbol'])).intersection(list(GE_TCGA['Hugo_Symbol'])))).shape)

Clin_features = ['Patient ID','Diagnosis Age','Disease Free (Months)','Prior Cancer Diagnosis Occurence',
                 'Positive Finding Lymph Node Hematoxylin and Eosin Staining Microscopy Count',
                 'Lymph Node(s) Examined Number','Menopause Status', 'Micromet detection by ihc', 
                 'New Neoplasm Event Post Initial Therapy Indicator', 'Disease Surgical Margin Status', 
                 'Race Category', 'Sex']
Clin_TCGA_subset = Clin_TCGA[Clin_features]
print('Clinical data subset = ',Clin_TCGA_subset)


#print(Clin_TCGA_subset['Sex'].astype(str).unique()) 

Clin_TCGA_subset_conv = []
for feat in Clin_features:
    if feat == 'Patient ID':
        Clin_TCGA_subset_d = Clin_TCGA_subset[feat]
        Clin_TCGA_subset_conv.append(np.array([x for x in Clin_TCGA_subset_d.astype(str)]))
    if feat == 'Diagnosis Age':
        Clin_TCGA_subset_d = Clin_TCGA_subset[feat]
        Clin_dict = {'55.0':2, '50.0':2, '62.0':3, '52.0':2, '42.0':1, '63.0':3, '70.0':4, '59.0':2, '56.0':2, '54.0':2,
 '61.0':3, '39.0':1, '77.0':4, '67.0':3, '40.0':1, '45.0':1, '66.0':3, '36.0':1, '48.0':1, '47.0':1,
 '34.0':1, '53.0':2, '60.0':3, '37.0':1, '85.0':4, '73.0':4, '71.0':4, '41.0':1, '46.0':1, '76.0':4,
 '64.0':3, '58.0':2, '79.0':4, '80.0':4, '82.0':4, '51.0':2, '74.0':4, '49.0':1, '44.0':1, '90.0':4,
 '35.0':1, '57.0':2, '78.0':4, '72.0':4, '65.0':3, '84.0':4, '68.0':3, '69.0':3, '75.0':4, '81.0':4,
 '89.0':4, '43.0':1, '83.0':4, '87.0':4, '88.0':4, '29.0':1, 'nan':0, '32.0':1, '31.0':1, '38.0':1,
 '27.0':1, '26.0':1, '30.0':1, '86.0':4, '28.0':1, '33.0':1,}
        Clin_TCGA_subset_conv.append(np.array([Clin_dict[x] for x in Clin_TCGA_subset_d.astype(str)]))
    if feat == 'Disease Free (Months)':
        Clin_TCGA_subset_d = Clin_TCGA_subset[feat]
        Clin_TCGA_subset_conv.append(np.array([x for x in Clin_TCGA_subset_d.astype(float)]))
    if feat == 'Prior Cancer Diagnosis Occurence':
        Clin_TCGA_subset_d = Clin_TCGA_subset[feat]
        Clin_dict = {'No':0, 'Yes':1, 'nan':2}
        Clin_TCGA_subset_conv.append(np.array([Clin_dict[x] for x in Clin_TCGA_subset_d.astype(str)]))
    if feat == 'Positive Finding Lymph Node Hematoxylin and Eosin Staining Microscopy Count':
        Clin_TCGA_subset_d = Clin_TCGA_subset[feat]
        Clin_TCGA_subset_conv.append(np.array([x for x in Clin_TCGA_subset_d.astype(float)]))
    if feat == 'Lymph Node(s) Examined Number' :
        Clin_TCGA_subset_d = Clin_TCGA_subset[feat]
        Clin_TCGA_subset_conv.append(np.array([x for x in Clin_TCGA_subset_d.astype(float)]))
    if feat == 'Menopause Status'  :
        Clin_TCGA_subset_d = Clin_TCGA_subset[feat]
        Clin_dict = {'Pre (<6 months since LMP AND no prior bilateral ovariectomy AND not on estrogen replacement)':0,
 'Post (prior bilateral ovariectomy OR >12 mo since LMP with no prior hysterectomy)':2,
 'nan':3, 'Indeterminate (neither Pre or Postmenopausal)':3,
 'Peri (6-12 months since last menstrual period)':1}
        Clin_TCGA_subset_conv.append(np.array([Clin_dict[x] for x in Clin_TCGA_subset_d.astype(str)]))
    if feat == 'Micromet detection by ihc'   :
        Clin_TCGA_subset_d = Clin_TCGA_subset[feat]
        Clin_dict = {'YES':1, 'nan':2, 'NO':0}
        Clin_TCGA_subset_conv.append(np.array([Clin_dict[x] for x in Clin_TCGA_subset_d.astype(str)]))
    if feat == 'New Neoplasm Event Post Initial Therapy Indicator'  :
        Clin_TCGA_subset_d = Clin_TCGA_subset[feat]
        Clin_dict = {'NO':0, 'nan':2, 'YES':1}
        Clin_TCGA_subset_conv.append(np.array([Clin_dict[x] for x in Clin_TCGA_subset_d.astype(str)]))
    if feat == 'Disease Surgical Margin Status' :
        Clin_TCGA_subset_d = Clin_TCGA_subset[feat]
        Clin_dict = {'Negative':0, 'Close':2, 'Positive':1, 'nan':3}
        Clin_TCGA_subset_conv.append(np.array([Clin_dict[x] for x in Clin_TCGA_subset_d.astype(str)]))
    if feat == 'Race Category'  :
        Clin_TCGA_subset_d = Clin_TCGA_subset[feat]
        Clin_dict = {'WHITE':0, 'BLACK OR AFRICAN AMERICAN':1, 'ASIAN':2, 'nan':4,
 'AMERICAN INDIAN OR ALASKA NATIVE':3}
        Clin_TCGA_subset_conv.append(np.array([Clin_dict[x] for x in Clin_TCGA_subset_d.astype(str)]))
    if feat == 'Sex'  :
        Clin_TCGA_subset_d = Clin_TCGA_subset[feat]
        Clin_dict = {'Female':0, 'Male':1, 'nan':2}
        Clin_TCGA_subset_conv.append(np.array([Clin_dict[x] for x in Clin_TCGA_subset_d.astype(str)]))
        
#print(np.array(Clin_TCGA_subset_conv).shape)         


Clin_TCGA_subset_conv_df = pd.DataFrame(np.array(Clin_TCGA_subset_conv).T, columns=Clin_features)
#print(Clin_TCGA_subset_conv_df)
Clin_TCGA_subset_conv_df.to_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/data/TCGA/new_TCGA_clin_data_test_v0.csv')


Clin_TCGA_subset_conv_df_unique = Clin_TCGA_subset_conv_df.drop_duplicates(subset=['Patient ID'], keep='first')
print(Clin_TCGA_subset_conv_df_unique.shape)


#GENETIC DATA
int_CNA_RNA = np.array(list(set(list(CNA_TCGA['Hugo_Symbol'])).intersection(list(GE_TCGA['Hugo_Symbol']))))
GE_TCGA_new = GE_TCGA.loc[GE_TCGA['Hugo_Symbol'].isin(int_CNA_RNA)]
CNA_TCGA_new = CNA_TCGA.loc[CNA_TCGA['Hugo_Symbol'].isin(int_CNA_RNA)]
GE_TCGA_new = GE_TCGA_new.drop_duplicates(subset=['Hugo_Symbol'], keep='first')

GE_TCGA_new_values = GE_TCGA_new.T.values[2:,:]
CNA_TCGA_new_values = CNA_TCGA_new.T.values[2:,:]
GE_TCGA_new_values_n = GE_TCGA_new_values/np.sum(GE_TCGA_new_values, axis=0).shape
CNA_TCGA_new_values_n = CNA_TCGA_new_values/np.sum(CNA_TCGA_new_values, axis=0).shape


F, p = f_oneway(GE_TCGA_new_values_n,CNA_TCGA_new_values_n)
p_idx = np.argsort(p)[:1000]
GE_TCGA_new_v = GE_TCGA_new.iloc[p_idx].T.values[2:,:]
GE_TCGA_new_v = GE_TCGA_new_v/np.sum(GE_TCGA_new_v, axis=0)
GE_TCGA_pd = pd.DataFrame(GE_TCGA_new_v, columns = GE_TCGA_new.iloc[p_idx].T.values[0,:])
print('New GE data size = ', GE_TCGA_pd.shape)

CNA_TCGA_new_v = CNA_TCGA_new.iloc[p_idx].T.values[2:,:]
CNA_TCGA_pd = pd.DataFrame(CNA_TCGA_new_v, columns = CNA_TCGA_new.iloc[p_idx].T.values[0,:])
print('New GE data size = ', CNA_TCGA_pd.shape)

GE_TCGA_pd.to_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/data/TCGA/new_GE_ANOVA_data_RNA_Seq_v2_expression_median.csv_v0', index=False)
CNA_TCGA_pd.to_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/data/TCGA/new_CNA_ANOVA_data_CNA_v0.csv', index=False)

IDs_patients_GE = GE_TCGA_new.columns[2:]
IDs_patients_CNA = CNA_TCGA_new.columns[2:]


#IMAGES DATA
Img_TCGA_IDs = Img_TCGA['ID'].values
a = np.intersect1d(Clin_TCGA['Patient ID'].values, Img_TCGA_IDs)
rep = [item for item, count in Counter(Img_TCGA_IDs).items() if count > 1]
rep_a = [item for item, count in Counter(a).items() if count > 1]
print(a.shape)

Clin_TCGA_ = Clin_TCGA.loc[Clin_TCGA['Patient ID'].isin(a)]
clin_targets = Clin_TCGA.columns[5:]

#ADD IC10
IC10_TCGA = pd.read_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/data/TCGA/TCGAIntClust1100.txt', sep='\t')
IC10_TCGA_=np.array([i[0:12] for i in IC10_TCGA['ID']])
IC10_TCGA['ID']=IC10_TCGA_
a = np.intersect1d(Clin_TCGA['Patient ID'].values, IC10_TCGA_)
rep_a = [item for item, count in Counter(a).items() if count > 1]
rep_a_ = [item for item, count in Counter(Clin_TCGA['Patient ID']).items() if count > 1]
rep_a_ = [item for item, count in Counter(IC10_TCGA['ID']).items() if count > 1]
mask_duplicated = Clin_TCGA['Patient ID'].isin(rep_a_)
Clin_TCGA_merged = pd.merge(Clin_TCGA,IC10_TCGA, left_on='Patient ID', right_on='ID', how='outer')
print('Clinical + Image data size = ',Clin_TCGA_merged.columns)

interesting_variables = ['Patient ID','ID','Disease Free (Months)', 'Disease Free Status', 'ER Status By IHC', 'ER Status IHC Percent Positive', 'Neoplasm Histologic Type Name', 'IHC-HER2', 'Overall Survival (Months)', 'Overall Survival Status', 'IntClust']
Clin_TCGA_merged = Clin_TCGA_merged.drop_duplicates(subset=['ID'], keep='first')
Img_TCGA_unique = Img_TCGA.drop_duplicates(subset=['ID'], keep='first')
Clin_TCGA_merged_interesting = Clin_TCGA_merged[interesting_variables]
print('Clinical + Image data subset size = ',Clin_TCGA_merged_interesting.shape)

IDs_patients_GE_new = np.array([i[:-3] for i in IDs_patients_GE])
GE_TCGA_pd['ID']=IDs_patients_GE_new
GE_TCGA_pd_unique = GE_TCGA_pd.drop_duplicates(subset=['ID'], keep='first')

IDs_patients_CNA_new = np.array([i[:-3] for i in IDs_patients_CNA])
CNA_TCGA_pd['ID']=IDs_patients_CNA_new
CNA_TCGA_pd_unique = CNA_TCGA_pd.drop_duplicates(subset=['ID'], keep='first')

#add sufixes to columns
columns_img = ['Img_'+x for x in Img_TCGA_unique.columns]
Img_TCGA_unique_c = pd.DataFrame(Img_TCGA_unique.values, columns = columns_img)
columns_GE = ['GE_'+x for x in GE_TCGA_pd_unique.columns]
GE_TCGA_pd_unique_c = pd.DataFrame(GE_TCGA_pd_unique.values, columns = columns_GE)
columns_CNA = ['CNA_'+x for x in CNA_TCGA_pd_unique.columns]
CNA_TCGA_pd_unique_c = pd.DataFrame(CNA_TCGA_pd_unique.values, columns = columns_CNA)
columns_Clin = ['Clin_'+x for x in Clin_TCGA_subset_conv_df_unique.columns]
Clin_TCGA_subset_conv_df_unique_c = pd.DataFrame(Clin_TCGA_subset_conv_df_unique.values, columns = columns_Clin)

a = np.intersect1d(Clin_TCGA_merged_interesting['Patient ID'].values, Img_TCGA_unique_c['Img_ID'].values)
a = np.intersect1d(a , GE_TCGA_pd_unique_c['GE_ID'].values)
mask_Clin = Clin_TCGA_merged_interesting['ID'].isin(a)
mask_Img = Img_TCGA_unique_c['Img_ID'].isin(a)
mask_GE = GE_TCGA_pd_unique_c['GE_ID'].isin(a)
Clin_Imgs_TCGA_merged = pd.merge(Clin_TCGA_merged_interesting.loc[mask_Clin],Img_TCGA_unique_c.loc[mask_Img], left_on='Patient ID', right_on='Img_ID', how='outer')
Clin_Imgs_GE_TCGA_merged = pd.merge(Clin_Imgs_TCGA_merged,GE_TCGA_pd_unique_c.loc[mask_GE], left_on='Patient ID', right_on='GE_ID', how='outer')
#print(Clin_Imgs_GE_TCGA_merged.shape)

a = np.intersect1d(a , CNA_TCGA_pd_unique_c['CNA_ID'].values)
mask_CNA = CNA_TCGA_pd_unique_c['CNA_ID'].isin(a)
Clin_Imgs_GE_CNA_TCGA_merged = pd.merge(Clin_Imgs_GE_TCGA_merged,CNA_TCGA_pd_unique_c.loc[mask_CNA], left_on='Patient ID', right_on='CNA_ID', how='outer')

a = np.intersect1d(a , Clin_TCGA_subset_conv_df_unique_c['Clin_Patient ID'].values)
mask_Clin_data = Clin_TCGA_subset_conv_df_unique_c['Clin_Patient ID'].isin(a)
Clin_Imgs_GE_CNA_Clin_TCGA_merged = pd.merge(Clin_Imgs_GE_CNA_TCGA_merged,Clin_TCGA_subset_conv_df_unique_c.loc[mask_Clin_data], left_on='Patient ID', right_on='Clin_Patient ID', how='outer')


print('Combined data size = ',Clin_Imgs_GE_CNA_Clin_TCGA_merged.columns[40:])
Clin_Imgs_GE_CNA_Clin_TCGA_merged.to_csv(name_output_file)