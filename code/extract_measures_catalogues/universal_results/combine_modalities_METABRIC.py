import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import os

import argparse
from astropy.table import Table, vstack



cell_type = False


if cell_type == True:
    Imgs_TCGA = pd.read_csv('./METABRIC_simil_measures_k10_v0.txt', delimiter = "\t", header=None)
    Imgs_TCGA_header = pd.read_csv('./header_simil_measures_k10__.txt', delimiter = " ", header=None)
    head = Imgs_TCGA_header.values[0]
    print('Digital pathology images data shape = ', Imgs_TCGA.shape)
    print('Header shape = ',Imgs_TCGA_header.values.shape)     
elif cell_type == False:
    Imgs_TCGA = pd.read_csv('./METABRIC_simil_measures_k10_NA_v0.txt', delimiter = "\t", header=None)
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
    print('Digital pathology images data shape = ', Imgs_TCGA.shape)
    print('Header shape = ',len(head))   

#Imgs_TCGA_header = pd.read_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/code/extract_measures_catalogues/universal_results/header_simil_measures_k10__.txt', delimiter = " ", header=None)

#Imgs_TCGA = pd.read_csv('./simil_measures_k10.txt', delimiter = "\t", header=None)


Imgs_TCGA_ = [np.array(i.split()) for i in Imgs_TCGA[0]]
Imgs_TCGA__=[i for i in Imgs_TCGA_]
#print(np.array(Imgs_TCGA__).shape)
#


#print(len(head))

Imgs_TCGA_df = pd.DataFrame(data=np.array(Imgs_TCGA__), columns=head)
print(Imgs_TCGA_df)

Imgs_TCGA_df.to_csv('./clean_measures_k10_METABRIC_v0.csv', index=False)


### MERGE CLINICAL DATA
data_imgs = pd.read_csv('./clean_measures_k10_METABRIC_v0.csv')

#data_imgs.columns=header_full[0]
#print(data_imgs.values.shape)
#print(data_imgs.columns)

#print(data_imgs)
data_imgs['ID'].unique().shape
data_imgs_unique = data_imgs.drop_duplicates(subset=['ID'], keep='first')
print('Digital pathology unique patients data shape = ',data_imgs_unique.shape)


data_medical= pd.read_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/data/MBdata_33CLINwMiss_1KfGE_1KfCNA_2.csv')
data_medical_oscar= pd.read_csv('/home/ICM_CG/Projects/METABRIC/DigPath_integration/data/FullPath_METABRIC.txt', sep='\t', low_memory=False)
print(data_medical_oscar)
ids = [i for i in data_imgs_unique['ID']]

print('Clinical data shape = ',len(ids))
    
data_medical = data_medical.rename(columns={"METABRIC_ID": "ID"})
data_oscar = data_medical_oscar.rename(columns={"METABRIC.ID": "ID"})
data_combo = pd.merge(data_imgs_unique, data_medical, on='ID')
#print(data_combo.shape)
data_combo = pd.merge(data_combo, data_oscar, on='ID')
print('Full combined data shape = ',data_combo.shape)

data_combo.to_csv('METABRIC_combined_data_NA_v0.csv', index=False)


