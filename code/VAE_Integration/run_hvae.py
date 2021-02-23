import numpy as np
import argparse
import os

from models.hvae import HVAE
from misc.dataset import Dataset, DatasetWhole
from misc.helpers import normalizeRNA,save_embedding


configs = {
    128: {
 
        'ds': 128,  # Intermediate dense layer size
        'act': 'elu',
        'epochs': 150,
        'bs': 64,  # Batch size
        'dropout':0.2
        #'beta': 1.0,  # Beta VAE parameter
    },
    256: {
 
        'ds': 256,  # Intermediate dense layer size
        'act': 'elu',
        'epochs': 150,
        'bs': 64,  # Batch size
        'dropout':0.2
    },
    512: {

        'ds': 512,  # Intermediate dense layer size
        'act': 'elu',
        'epochs': 150,
        'bs': 64,  # Batch size
        'dropout':0.2
    },

    144: {

        'ds': 144,  # Intermediate dense layer size
        'act': 'elu',
        'epochs': 150,
        'bs': 64,  # Batch size
        'dropout':0.2
    },

    240: {

        'ds': 240,  # Intermediate dense layer size
        'act': 'elu',
        'epochs': 150,
        'bs': 64,  # Batch size
        'dropout':0.2
    },
    360: {

        'ds': 360,  # Intermediate dense layer size
        'act': 'elu',
        'epochs': 150,
        'bs': 64,  # Batch size
        'dropout':0.2
    },    
}

parser = argparse.ArgumentParser()
parser.add_argument('--integration', help='Type of integration Clin+mRNA, CNA+mRNA or Clin+CNA', type=str, required=True, default='Clin+mRNA')
parser.add_argument('--ds', help='The intermediate dense layers size', type=int, required=True)
parser.add_argument('--save_model', help='Saves the weights of the model', action='store_true')
parser.add_argument('--fold', help='The fold to train on, if 0 will train on the whole data set', type=str, default='0')
parser.add_argument('--dtype', help='The type of data (Pam50, Pam50C, IC, ER)', type=str, default='ER')
parser.add_argument('--beta', help='beta size', type=int, required=True)
parser.add_argument('--distance', help='regularization', type=str, default='kl')
parser.add_argument('--ls', help='latent dimension size', type=int, required=True)
parser.add_argument('--writedir', help='/PATH/TO/OUTPUT - Default is current dir', type=str, default='')
parser.add_argument('--modalities', help='number of modalities', type=int, required=True)



if __name__ == "__main__":
    args = parser.parse_args()
    config = configs[args.ds]
    for key, val in config.items():
        setattr(args, key, val)
        
        



if (args.fold == '0'): # whole data set
    
    print('TRAINING on the complete data')
    
    dataset = DatasetWhole('W')
    
        
    if (args.integration == 'Clin+mRNA'): #integrate Clin+mRNA
        s1_train = dataset.train['clin'] 
        s2_train = normalizeRNA(dataset.train['rnanp'])
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'catVAE')
        vae_s2 = HVAE(args, 'numVAE')
        
        
    elif (args.integration == 'Clin+CNA'): #integrate Clin+CNA
        s1_train = dataset.train['clin']
        s2_train = dataset.train['cnanp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'catVAE')
        vae_s2 = HVAE(args, 'catVAE')

    elif (args.integration == 'mRNA+CNA'): #integrate Clin+CNA
        s1_train = normalizeRNA(dataset.train['rnanp'])
        s2_train = dataset.train['cnanp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE')
        vae_s2 = HVAE(args, 'catVAE')
        
    elif (args.integration == 'img_rho+mRNA'): #integrate Clin+CNA
        s1_train = dataset.train['img_rhonp']
        s2_train = normalizeRNA(dataset.train['rnanp'])
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'numVAE', input_s = '2')   
    
    elif (args.integration == 'img_s+mRNA'): #integrate Clin+CNA
        s1_train = dataset.train['img_snp']
        s2_train = normalizeRNA(dataset.train['rnanp'])
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'numVAE', input_s = '2')       
    
    elif (args.integration == 'img_v+mRNA'): #integrate Clin+CNA
        s1_train = dataset.train['img_vnp']
        s2_train = normalizeRNA(dataset.train['rnanp'])
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'numVAE', input_s = '2')  
    
    elif (args.integration == 'img_rho+CNA'): #integrate Clin+CNA
        s1_train = dataset.train['img_rhonp']
        s2_train = dataset.train['cnanp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'catVAE', input_s = '2')   
    
    elif (args.integration == 'img_s+CNA'): #integrate Clin+CNA
        s1_train = dataset.train['img_snp']
        s2_train = dataset.train['cnanp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'catVAE', input_s = '2')       
    
    elif (args.integration == 'img_v+CNA'): #integrate Clin+CNA
        s1_train = dataset.train['img_vnp']
        s2_train = dataset.train['cnanp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'catVAE', input_s = '2')          
    
    elif (args.integration == 'img_rho+Clin'): #integrate Clin+CNA
        s1_train = dataset.train['img_rhonp']
        s2_train = dataset.train['clin']
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'catVAE', input_s = '2')   
    
    elif (args.integration == 'img_s+Clin'): #integrate Clin+CNA
        s1_train = dataset.train['img_snp']
        s2_train = dataset.train['clin']
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'catVAE', input_s = '2')       
    
    elif (args.integration == 'img_v+Clin'): #integrate Clin+CNA
        s1_train = dataset.train['img_vnp']
        s2_train = dataset.train['clin']
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'catVAE', input_s = '2')         
    
    else:
        s1_train = dataset.train['cnanp'] 
        s2_train = normalizeRNA(dataset.train['rnanp'])    
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'catVAE')
        vae_s2 = HVAE(args, 'numVAE')

    
    vae_s1.build_model(input_s = '1')
    vae_s2.build_model(input_s = '2')
    hvae = HVAE(args, 'H')
    hvae.build_model(input_s = '4')


    print('TRAINING low-level VAE 1')
    vae_s1.train(s1_train, s1_train)
    s1_emb_train = vae_s1.predict(s1_train)

    print('TRAINING low-level VAE 2')
    vae_s2.train(s2_train, s2_train)
    s2_emb_train = vae_s2.predict(s2_train)
    
    print('TRAINING low-level VAE 2')
    vae_s3.train(s3_train, s3_train)
    s3_emb_train = vae_s3.predict(s3_train)
    
    print('TRAINING HVAE')
    hvae_train = np.concatenate((s1_emb_train, s2_emb_train), axis=-1)
    hvae.train(hvae_train, hvae_train)
    emb_train = hvae.predict(hvae_train)

    if (args.writedir == ''):
        emb_save_dir = 'results/HVAE_'+format(args.integration)+'_integration/hvae_LS_'+format(args.ls)+'_DS_'+format(args.ds)+'_'+format(args.distance)+'_beta_'+format(args.beta)
    else:
        emb_save_dir = args.writedir+'/HVAE_'+format(args.integration)+'_integration/hvae_LS_'+format(args.ls)+'_DS_'+format(args.ds)+'_'+format(args.distance)+'_beta_'+format(args.beta)
    if not os.path.exists(emb_save_dir):
        os.makedirs(emb_save_dir)
    emb_save_file = args.dtype +'.npz'
    emb_save_file1 = 's1_'+args.dtype +'.npz'
    emb_save_file2 = 's2_'+args.dtype +'.npz'
    emb_save_file3 = 's3_'+args.dtype +'.npz'
    save_embedding(emb_save_dir,emb_save_file,emb_train)
    save_embedding(emb_save_dir,emb_save_file1,s1_emb_train)
    save_embedding(emb_save_dir,emb_save_file2,s2_emb_train)
    save_embedding(emb_save_dir,emb_save_file3,s3_emb_train)
    
else:

    print('TRAINING on the fold '+ format(args.fold))
    
    dataset = Dataset(args.dtype, args.fold)

    if (args.integration == 'Clin+mRNA+CNA'): #integrate Clin+mRNA+CNA
        s1_train = dataset.train['clin'] 
        s1_test = dataset.test['clin'] 
        s2_train = normalizeRNA(dataset.train['rnanp'])
        s2_test = normalizeRNA(dataset.test['rnanp'])
        s3_train = dataset.train['cnanp'] 
        s3_test = dataset.test['cnanp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        args.s3_input_size= s3_train.shape[1]
        vae_s1 = HVAE(args, 'catVAE')
        vae_s2 = HVAE(args, 'numVAE')
        vae_s3 = HVAE(args, 'catVAE')

    elif (args.integration == 'Clin+mRNA+img_rho'): 
        s1_train = dataset.train['clin'] 
        s1_test = dataset.test['clin'] 
        s2_train = normalizeRNA(dataset.train['rnanp'])
        s2_test = normalizeRNA(dataset.test['rnanp'])
        s3_train = dataset.train['img_rhonp']
        s3_test = dataset.test['img_rhonp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        args.s3_input_size= s3_train.shape[1]
        vae_s1 = HVAE(args, 'catVAE')
        vae_s2 = HVAE(args, 'numVAE')
        vae_s3 = HVAE(args, 'numVAE')        

    elif (args.integration == 'Clin+mRNA+img_s'): 
        s1_train = dataset.train['clin'] 
        s1_test = dataset.test['clin'] 
        s2_train = normalizeRNA(dataset.train['rnanp'])
        s2_test = normalizeRNA(dataset.test['rnanp'])
        s3_train = dataset.train['img_snp']
        s3_test = dataset.test['img_snp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        args.s3_input_size= s3_train.shape[1]
        vae_s1 = HVAE(args, 'catVAE')
        vae_s2 = HVAE(args, 'numVAE')
        vae_s3 = HVAE(args, 'numVAE')   
        
    elif (args.integration == 'Clin+mRNA+img_v'): 
        s1_train = dataset.train['clin'] 
        s1_test = dataset.test['clin'] 
        s2_train = normalizeRNA(dataset.train['rnanp'])
        s2_test = normalizeRNA(dataset.test['rnanp'])
        s3_train = dataset.train['img_vnp']
        s3_test = dataset.test['img_vnp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        args.s3_input_size= s3_train.shape[1]
        vae_s1 = HVAE(args, 'catVAE')
        vae_s2 = HVAE(args, 'numVAE')
        vae_s3 = HVAE(args, 'numVAE')   
    
    elif (args.integration == 'Clin+CNA+img_rho'): 
        s1_train = dataset.train['clin'] 
        s1_test = dataset.test['clin'] 
        s2_train = dataset.train['cnanp']
        s2_test = dataset.test['cnanp']
        s3_train = dataset.train['img_rhonp']
        s3_test = dataset.test['img_rhonp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        args.s3_input_size= s3_train.shape[1]
        vae_s1 = HVAE(args, 'catVAE')
        vae_s2 = HVAE(args, 'catVAE')
        vae_s3 = HVAE(args, 'numVAE')        

    elif (args.integration == 'Clin+CNA+img_s'): 
        s1_train = dataset.train['clin'] 
        s1_test = dataset.test['clin'] 
        s2_train = dataset.train['cnanp']
        s2_test = dataset.test['cnanp']
        s3_train = dataset.train['img_snp']
        s3_test = dataset.test['img_snp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        args.s3_input_size= s3_train.shape[1]
        vae_s1 = HVAE(args, 'catVAE')
        vae_s2 = HVAE(args, 'catVAE')
        vae_s3 = HVAE(args, 'numVAE')   
        
    elif (args.integration == 'Clin+CNA+img_v'): 
        s1_train = dataset.train['clin'] 
        s1_test = dataset.test['clin'] 
        s2_train = dataset.train['cnanp']
        s2_test = dataset.test['cnanp']
        s3_train = dataset.train['img_vnp']
        s3_test = dataset.test['img_vnp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        args.s3_input_size= s3_train.shape[1]
        vae_s1 = HVAE(args, 'catVAE')
        vae_s2 = HVAE(args, 'catVAE')
        vae_s3 = HVAE(args, 'numVAE')   

    elif (args.integration == 'mRNA+CNA+img_rho'): 
        s1_train = normalizeRNA(dataset.train['rnanp'])
        s1_test = normalizeRNA(dataset.test['rnanp'])
        s2_train = dataset.train['cnanp']
        s2_test = dataset.test['cnanp']
        s3_train = dataset.train['img_rhonp']
        s3_test = dataset.test['img_rhonp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        args.s3_input_size= s3_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE')
        vae_s2 = HVAE(args, 'catVAE')
        vae_s3 = HVAE(args, 'numVAE')        

    elif (args.integration == 'mRNA+CNA+img_s'): 
        s1_train = normalizeRNA(dataset.train['rnanp'])
        s1_test = normalizeRNA(dataset.test['rnanp'])
        s2_train = dataset.train['cnanp']
        s2_test = dataset.test['cnanp']
        s3_train = dataset.train['img_snp']
        s3_test = dataset.test['img_snp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        args.s3_input_size= s3_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE')
        vae_s2 = HVAE(args, 'catVAE')
        vae_s3 = HVAE(args, 'numVAE')   
        
    elif (args.integration == 'mRNA+CNA+img_v'): 
        s1_train = normalizeRNA(dataset.train['rnanp'])
        s1_test = normalizeRNA(dataset.test['rnanp']) 
        s2_train = dataset.train['cnanp']
        s2_test = dataset.test['cnanp']
        s3_train = dataset.train['img_vnp']
        s3_test = dataset.test['img_vnp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        args.s3_input_size= s3_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE')
        vae_s2 = HVAE(args, 'catVAE')
        vae_s3 = HVAE(args, 'numVAE')   
        
    elif (args.integration == 'mRNA+CNA+Clin+img_rho'): 
        s1_train = normalizeRNA(dataset.train['rnanp'])
        s1_test = normalizeRNA(dataset.test['rnanp'])
        s2_train = dataset.train['cnanp']
        s2_test = dataset.test['cnanp']
        s3_train = dataset.train['clin']
        s3_test = dataset.test['clin']
        s4_train = dataset.train['img_rhonp']
        s4_test = dataset.test['img_rhonp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        args.s3_input_size= s3_train.shape[1]
        args.s4_input_size= s4_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE')
        vae_s2 = HVAE(args, 'catVAE')
        vae_s3 = HVAE(args, 'catVAE')   
        vae_s4 = HVAE(args, 'numVAE')
        
    elif (args.integration == 'mRNA+CNA+Clin+img_s'): 
        s1_train = normalizeRNA(dataset.train['rnanp'])
        s1_test = normalizeRNA(dataset.test['rnanp'])
        s2_train = dataset.train['cnanp']
        s2_test = dataset.test['cnanp']
        s3_train = dataset.train['clin']
        s3_test = dataset.test['clin']
        s4_train = dataset.train['img_snp']
        s4_test = dataset.test['img_snp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        args.s3_input_size= s3_train.shape[1]
        args.s4_input_size= s4_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE')
        vae_s2 = HVAE(args, 'catVAE')
        vae_s3 = HVAE(args, 'catVAE')   
        vae_s4 = HVAE(args, 'numVAE')   
        
    elif (args.integration == 'mRNA+CNA+Clin+img_v'): 
        s1_train = normalizeRNA(dataset.train['rnanp'])
        s1_test = normalizeRNA(dataset.test['rnanp']) 
        s2_train = dataset.train['cnanp']
        s2_test = dataset.test['cnanp']
        s3_train = dataset.train['clin']
        s3_test = dataset.test['clin']
        s4_train = dataset.train['img_vnp']
        s4_test = dataset.test['img_vnp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        args.s3_input_size= s3_train.shape[1]
        args.s4_input_size= s4_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE')
        vae_s2 = HVAE(args, 'catVAE')
        vae_s3 = HVAE(args, 'catVAE')   
        vae_s4 = HVAE(args, 'numVAE')   
        
    elif (args.integration == 'Clin+CNA'): 
        s1_train = dataset.train['clin']
        s1_test = dataset.test['clin'] 
        s2_train = dataset.train['cnanp'] 
        s2_test = dataset.test['cnanp'] 
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'catVAE')
        vae_s2 = HVAE(args, 'catVAE')
    
    elif (args.integration == 'img_rho+mRNA'): 
        s1_train = dataset.train['img_rhonp']
        s1_test = dataset.test['img_rhonp'] 
        s2_train = normalizeRNA(dataset.train['rnanp'])
        s2_test = normalizeRNA(dataset.test['rnanp'])
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'numVAE', input_s = '2')   
    
    elif (args.integration == 'img_s+mRNA'): 
        s1_train = dataset.train['img_snp']
        s1_test = dataset.test['img_snp'] 
        s2_train = normalizeRNA(dataset.train['rnanp'])
        s2_test = normalizeRNA(dataset.test['rnanp'])
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'numVAE', input_s = '2')       
    
    elif (args.integration == 'img_v+mRNA'): 
        s1_train = dataset.train['img_vnp']
        s1_test = dataset.test['img_vnp'] 
        s2_train = normalizeRNA(dataset.train['rnanp'])
        s2_test = normalizeRNA(dataset.test['rnanp'])
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'numVAE', input_s = '2')  

    elif (args.integration == 'img_rho+CNA'): 
        s1_train = dataset.train['img_rhonp']
        s1_test = dataset.test['img_rhonp'] 
        s2_train = dataset.train['cnanp']
        s2_test = dataset.test['cnanp']
        print(s2_train)
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'catVAE', input_s = '2')   
    
    elif (args.integration == 'img_s+CNA'): 
        s1_train = dataset.train['img_snp']
        s1_test = dataset.test['img_snp']
        s2_train = dataset.train['cnanp'] 
        s2_test = dataset.test['cnanp']
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'catVAE', input_s = '2')       
    
    elif (args.integration == 'img_v+CNA'): 
        s1_train = dataset.train['img_vnp']
        s1_test = dataset.test['img_vnp']
        s2_train = dataset.train['cnanp'] 
        s2_test = dataset.test['cnanp']
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'catVAE', input_s = '2')          
    
    elif (args.integration == 'img_rho+Clin'):
        s1_train = dataset.train['img_rhonp']
        s1_test = dataset.test['img_rhonp']
        s2_train = dataset.train['clin']
        s2_test = dataset.test['clin']
        print(s2_train)
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'catVAE', input_s = '2')   
    
    elif (args.integration == 'img_s+Clin'): 
        s1_train = dataset.train['img_snp']
        s1_test = dataset.test['img_snp']
        s2_train = dataset.train['clin']
        s2_test = dataset.test['clin']
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'catVAE', input_s = '2')       
    
    elif (args.integration == 'img_v+Clin'): 
        s1_train = dataset.train['img_vnp']
        s1_test = dataset.test['img_vnp']
        s2_train = dataset.train['clin']
        s2_test = dataset.test['clin']
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'numVAE', input_s = '1')
        vae_s2 = HVAE(args, 'catVAE', input_s = '2')         
        
    else:#integrate CNA+mRNA
        s1_train = dataset.train['cnanp'] 
        s1_test = dataset.test['cnanp'] 
        s2_train = normalizeRNA(dataset.train['rnanp'])
        s2_test = normalizeRNA(dataset.test['rnanp'])
        args.s1_input_size= s1_train.shape[1]
        args.s2_input_size= s2_train.shape[1]
        vae_s1 = HVAE(args, 'catVAE', input_s = '1')
        vae_s2 = HVAE(args, 'numVAE', input_s = '2')


    vae_s1.build_model(input_s = '1')
    vae_s2.build_model(input_s = '2')
    if args.modalities >= 3 :
        vae_s3.build_model(input_s = '3')
    if args.modalities == 4 :
        vae_s4.build_model(input_s = '4')
    hvae = HVAE(args, 'H')
    hvae.build_model(input_s = '5')


    print('TRAINING low-level VAE 1')
    vae_s1.train(s1_train, s1_train)
    s1_emb_train = vae_s1.predict(s1_train)
    s1_emb_test = vae_s1.predict(s1_test)
    print(s1_emb_test)

    print('TRAINING low-level VAE 2')
    vae_s2.train(s2_train, s2_train)
    s2_emb_train = vae_s2.predict(s2_train)
    s2_emb_test = vae_s2.predict(s2_test)
    print(s2_emb_test)

    if args.modalities >= 3 :
        print('TRAINING low-level VAE 3')
        vae_s3.train(s3_train, s3_train)
        s3_emb_train = vae_s3.predict(s3_train)
        s3_emb_test = vae_s3.predict(s3_test)
        print(s3_emb_test)

    if args.modalities == 4:
        print('TRAINING low-level VAE 4')
        vae_s4.train(s4_train, s4_train)
        s4_emb_train = vae_s4.predict(s4_train)
        s4_emb_test = vae_s4.predict(s4_test)
        print(s4_emb_test)
    
    print('TRAINING HVAE')
    if args.modalities == 4:
        hvae_train = np.concatenate((s1_emb_train, s2_emb_train, s3_emb_train, s4_emb_train), axis=-1)
        hvae_test = np.concatenate((s1_emb_test, s2_emb_test, s3_emb_test, s4_emb_test), axis=-1)
    elif args.modalities == 3:
        hvae_train = np.concatenate((s1_emb_train, s2_emb_train, s3_emb_train), axis=-1)
        hvae_test = np.concatenate((s1_emb_test, s2_emb_test, s3_emb_test), axis=-1)
    else:
        hvae_train = np.concatenate((s1_emb_train, s2_emb_train), axis=-1)
        hvae_test = np.concatenate((s1_emb_test, s2_emb_test), axis=-1)
    
    hvae.train(hvae_train, hvae_train)
    emb_train = hvae.predict(hvae_train)    
    emb_test = hvae.predict(hvae_test)

    if (args.writedir == ''):
        emb_save_dir = 'results/HVAE_'+format(args.integration)+'_integration/hvae_LS_'+format(args.ls)+'_DS_'+format(args.ds)+'_'+format(args.distance)+'_beta_'+format(args.beta)
    else:
        emb_save_dir = args.writedir+'/HVAE_'+format(args.integration)+'_integration/hvae_LS_'+format(args.ls)+'_DS_'+format(args.ds)+'_'+format(args.distance)+'_beta_'+format(args.beta)
    if not os.path.exists(emb_save_dir):
        os.makedirs(emb_save_dir)
    emb_save_file = args.dtype +args.fold+'.npz'
    emb_save_file1 = 's1_'+args.dtype +args.fold+'.npz'
    emb_save_file2 = 's2_'+args.dtype +args.fold+'.npz'
    emb_save_file3 = 's3_'+args.dtype +args.fold+'.npz'
    emb_save_file4 = 's4_'+args.dtype +args.fold+'.npz'
    save_embedding(emb_save_dir,emb_save_file,emb_train, emb_test)
    save_embedding(emb_save_dir,emb_save_file1,s1_emb_train, s1_emb_test)
    save_embedding(emb_save_dir,emb_save_file2,s2_emb_train, s2_emb_test)
    if args.modalities == 3:
        save_embedding(emb_save_dir,emb_save_file3,s3_emb_train, s3_emb_test)
    if args.modalities == 4:
        save_embedding(emb_save_dir,emb_save_file4,s4_emb_train, s4_emb_test)       