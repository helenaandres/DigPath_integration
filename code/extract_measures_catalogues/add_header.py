from functions_extract_measures_catalogues import *



file = sys.argv[1]
Y = sys.argv[2]
K = sys.argv[3]

print(file)
print(Y)


#def main(argv=None):
#X_t=Table.read('/home/ICM_CG/Projects/METABRIC/level2_catalogues/'+str(os.path.splitext(file)[0])+'.fits', format='fits')
X_t=Table.read('/home/ICM_CG/Projects/METABRIC/level2_catalogues/'+str(os.path.splitext(file)[0])+'.fits', format='fits')

filename = "./simil_measures_k"+str(K)+".txt"

print('./'+str(os.path.splitext(file)[0])+'.fits')
print('^good')

T_d, T_d_ind_mean, T_d_ind_std  = add_density(X_t, bandwidth = int(K))

rho_cells_medians, S_mean_cells_medians, S_std_cells_medians, Sf_mean_cells_medians, Sf_std_cells_medians = calculate_medians(T_d,T_d_ind_mean, T_d_ind_std, cell_type = True, k =int(K))
print('saving median...')