from functions_extract_measures_catalogues import *


file = sys.argv[1]
Y = sys.argv[2]
K = sys.argv[3]

print(file)
print(Y)

X_t=Table.read('/home/ICM_CG/Projects/METABRIC/level2_catalogues/'+str(os.path.splitext(file)[0])+'.fits', format='fits')

#X_t=Table.read('/local/scratch/ha376/ICM/Imaging_data/digital_pathology_integration/data/catalogues/'+str(os.path.splitext(file)[0])+'.fits', format='fits')

print('./'+str(os.path.splitext(file)[0])+'.fits')
print('^good')

filename = "./results/simil_measures_k"+str(K)+".txt"
T_d, T_d_ind_mean, T_d_ind_std  = add_density(X_t, bandwidth = int(K))

rho_cells_medians, S_mean_cells_medians, S_std_cells_medians, Sf_mean_cells_medians, Sf_std_cells_medians = calculate_medians(T_d,T_d_ind_mean, T_d_ind_std, cell_type = True, k =int(K))
print('saving median...')


        
header = np.column_stack(('ID', 'Rho_CC', 'Rho_L', 'Rho_NC','S_mean_CC', 'S_mean_L', 'S_mean_NC',
                          'S_std_CC', 'S_std_L', 'S_std_NC'))

h_1_mean = [i+'_CC_mean' for i in Sf_mean_cells_medians[1].keys().values]
h_2_mean = [i+'_L_mean' for i in Sf_mean_cells_medians[2].keys().values]
h_3_mean = [i+'_NC_mean' for i in Sf_mean_cells_medians[3].keys().values]
h_1_std = [i+'_CC_std' for i in Sf_mean_cells_medians[1].keys().values]
h_2_std = [i+'_L_std' for i in Sf_mean_cells_medians[2].keys().values]
h_3_std = [i+'_NC_std' for i in Sf_mean_cells_medians[3].keys().values]

header_full = np.column_stack((header, np.array(h_1_mean).reshape(1,46), np.array(h_1_std).reshape(1,46),
                               np.array(h_2_mean).reshape(1,46), np.array(h_2_std).reshape(1,46),
                               np.array(h_3_mean).reshape(1,46), np.array(h_3_std).reshape(1,46),))

A = np.concatenate((np.array([Y]), np.array(rho_cells_medians)[1:],np.array(S_mean_cells_medians)[1:],
                    np.array(S_std_cells_medians)[1:], Sf_mean_cells_medians[1].values, 
                    Sf_std_cells_medians[1].values, Sf_mean_cells_medians[2].values, 
                    Sf_std_cells_medians[2].values,Sf_mean_cells_medians[3].values, 
                    Sf_std_cells_medians[3].values))
A = np.column_stack((A))

print(A.shape)
print(header_full.shape)
    
fmt = '%s'
f=open(filename,'ab')
np.savetxt(f, A ,fmt = fmt, delimiter =' ')
f.close()