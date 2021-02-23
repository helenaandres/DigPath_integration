from misc.helpers import get_data
import os
import glob
import pandas as pd
import numpy as np

type_to_data = {
    
    'ER': "/local/scratch/ha376ICM/Imaging_data/digital_pathology_integration/data/modalities/stratified/5-fold_ERstratified",
    
    'IC': "/local/scratch/ha376ICM/Imaging_data/digital_pathology_integration/data/modalities/stratified/5-fold_iC10stratified",
    
    'PAM': "/local/scratch/ha376ICM/Imaging_data/digital_pathology_integration/data/modalities/stratified/5-fold_pam50stratified",
    
    'DR': "/local/scratch/ha376ICM/Imaging_data/digital_pathology_integration/data/modalities/stratified/5-fold_DRstratified",
    
    'Histological_Type': "/local/scratch/ha376ICM/Imaging_data/digital_pathology_integration/data/modalities/stratified/5-fold_Histological_Typestratified",
    
    'total_score': "/local/scratch/ha376ICM/Imaging_data/digital_pathology_integration/data/modalities/stratified/5-fold_total_scorestratified",
    
    'tubule.formation': "/local/scratch/ha376ICM/Imaging_data/digital_pathology_integration/data/modalities/stratified/data/5-fold_tubule.formation_scorestratified",
    
    'lymph_infiltrate': "/local/scratch/ha376ICM/Imaging_data/digital_pathology_integration/data/modalities/stratified/data/5-fold_lymphocytic_infiltratestratified",
    
    'nuc_pleomorphism': "/local/scratch/ha376ICM/Imaging_data/digital_pathology_integration/data/modalities/stratified/data/5-fold_nuc_pleomorphism_scorestratified",    
    
    'overall_grade': "/local/scratch/ha376ICM/Imaging_data/digital_pathology_integration/data/modalities/stratified/data/5-fold_overall_gradestratified",
    
    'W': "/local/scratch/ha376ICM/Imaging_data/digital_pathology_integration/data/modalities/images/",    
}


class Dataset:
    def __init__(self, dtype, fold):
        self.type = dtype
        self.fold = fold
        self.train, self.test = self._get_data(dtype, fold)

    def _get_data(self, dtype, fold):
        foldpath = os.path.join(type_to_data[dtype], "fold" + fold)      
        dev_file = glob.glob(foldpath + "/*k50_test.csv")
        train_file = glob.glob(foldpath + "/*k50_train.csv")

        for file_ in dev_file:
            dev = pd.read_csv(file_, index_col=None, header=0)
        for file_ in train_file:
            train = pd.read_csv(file_, index_col=None, header=0)
        return get_data(train), get_data(dev)

class DatasetWhole:
    def __init__(self, dtype):
        self.type = dtype
        self.train = self._get_data(dtype)

    def _get_data(self, dtype):
        foldpath = os.path.join(type_to_data[dtype])
        train_file = glob.glob(foldpath + "/*k50.csv")
        

        for file_ in train_file:
            train = pd.read_csv(file_, index_col=None, header=0)
        return get_data(train)
    
