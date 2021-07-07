from new_functions_extract_measures_catalogues_universal import *


file = sys.argv[1]
Y = sys.argv[2]
K = sys.argv[3]
dataset_public = sys.argv[4]

print(file)
print(Y)


if str(dataset_public) == 'METABRIC':
    directory = '/home/ICM_CG/Projects/METABRIC/new_data_Ali_May21/metabric/'
    filename = "./universal_results/METABRIC_simil_measures_k"+str(K)+"_v1.txt"
elif str(dataset_public) == 'TCGA':
    directory = '/home/ICM_CG/Projects/METABRIC/new_data_Ali_May21/tcga_ffpe/025_um/'
    filename = "./universal_results/TCGA_simil_measures_k"+str(K)+"_NA_v0.txt"

    
X_t=Table.read(directory+str(os.path.splitext(file)[0])+'.fits', format='fits')

cell_type = False
print('CELL TYPES = ', cell_type)
print('###############')
print('./'+str(os.path.splitext(file)[0])+'.fits')
print('^good')

T_d, T_d_ind_mean, T_d_ind_std  = add_density_universal(X_t, bandwidth = int(K), cell_types=cell_type)

rho_cells_medians, S_mean_cells_medians, S_std_cells_medians, Sf_mean_cells_medians, Sf_std_cells_medians = calculate_medians(T_d,T_d_ind_mean, T_d_ind_std, cell_types = cell_type, k =int(K))
print('saving median...')




if cell_type == True:
    print("#### With CELL TYPES ####")
    header = np.column_stack(('ID', 'Rho_Cancer', 'Rho_Lymph', 'Rho_Stroma','S_mean_Cancer', 'S_mean_Lymph', 'S_mean_Stroma',
                          'S_std_Cancer', 'S_std_Lymph', 'S_std_Stroma'))

    h_1_mean = [i+'_Cancer_mean' for i in Sf_mean_cells_medians[1].keys().values]
    h_2_mean = [i+'_Lymph_mean' for i in Sf_mean_cells_medians[2].keys().values]
    h_3_mean = [i+'_Stroma_mean' for i in Sf_mean_cells_medians[3].keys().values]
    h_1_std = [i+'_Cancer_std' for i in Sf_mean_cells_medians[1].keys().values]
    h_2_std = [i+'_Lymph_std' for i in Sf_mean_cells_medians[2].keys().values]
    h_3_std = [i+'_Stroma_std' for i in Sf_mean_cells_medians[3].keys().values]

    header_full = np.column_stack((header, np.array(h_1_mean).reshape(1,47), np.array(h_1_std).reshape(1,47),
                               np.array(h_2_mean).reshape(1,47), np.array(h_2_std).reshape(1,47),
                               np.array(h_3_mean).reshape(1,47), np.array(h_3_std).reshape(1,47),))
    A = np.concatenate((np.array([Y]), np.array(rho_cells_medians)[1:],np.array(S_mean_cells_medians)[1:],
                    np.array(S_std_cells_medians)[1:], Sf_mean_cells_medians[1].values, 
                    Sf_std_cells_medians[1].values, Sf_mean_cells_medians[2].values, 
                    Sf_std_cells_medians[2].values,Sf_mean_cells_medians[3].values, 
                    Sf_std_cells_medians[3].values))
else:
    print("#### NO CELL TYPES ####")   
    header = np.column_stack(('ID', 'Rho_AC', 'S_mean_AC', 'S_std_AC'))

    h_1_mean = [i+'_AC_mean' for i in Sf_mean_cells_medians[0].keys().values]
    h_1_std = [i+'_AC_std' for i in Sf_mean_cells_medians[0].keys().values]


    header_full = np.column_stack((header, np.array(h_1_mean).reshape(1,np.array(h_1_mean).shape[0]), np.array(h_1_std).reshape(1,np.array(h_1_std).shape[0])))    

    print(header_full)
    A = np.concatenate((np.array([Y]), np.array(rho_cells_medians),np.array(S_mean_cells_medians),
                    np.array(S_std_cells_medians), Sf_mean_cells_medians[0].values, 
                    Sf_std_cells_medians[0].values))
A = np.column_stack((A))
print(A.shape)
print(header_full.shape)
    
fmt = '%s'
f=open(filename,'ab')
np.savetxt(f, A ,fmt = fmt, delimiter =' ')
f.close()