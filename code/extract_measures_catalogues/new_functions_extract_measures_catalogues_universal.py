import numpy as np
import pandas as pd

from sklearn.neighbors.kde import KernelDensity
from astroML.density_estimation import KNeighborsDensity
from sklearn.neighbors import KernelDensity

from astropy.table import Table, vstack
import sys, os

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import special
from sklearn.neighbors import BallTree
from scipy.spatial.distance import cdist
import csv


def n_volume(r, n):
    """compute the n-volume of a sphere of radius r in n dimensions"""
    return np.pi ** (0.5 * n) / special.gamma(0.5 * n + 1) * (r ** n)

def similarity(ind, X_f, matrix=False, indiv=False):
    if matrix == True: #output similarity matrix
        d_m=[cdist(np.array(X_f[ind[i,:],:]),np.array(X_f[ind[i,:],:])) for i in ind]
    else: 
        d_m=0
        min_Xf = np.array([np.amin(np.array(X_f[:,i])) for i in np.arange(X_f.shape[1])]) 
        X_f = (X_f - min_Xf)
        max_Xf = np.array([np.amax(np.array(X_f[:,i])) for i in np.arange(X_f.shape[1])]) 
        X_f = X_f / max_Xf    
        
    if indiv == True:

        d=[np.sqrt((np.array(X_f[ind[i,:]]) - np.array([X_f[ind[i,0]]]))**2) for i in ind[:,0]]
        D = [np.mean(np.array(d),axis=1),np.std(np.array(d),axis=1)]

    elif indiv == False: #compute similarity for all features
        
        d=[cdist(np.array([X_f[ind[i,0],:]]),np.array(X_f[ind[i,:],:])) for i in ind[:,0]]
        #print(X_f)
        D = [np.mean(np.array(d),axis=2),np.std(np.array(d),axis=2)]
        
    return D, d_m
    
class KNeighborsDensity_modified(object):
    """K-neighbors density estimation

    Parameters
    ----------
    method : string
        method to use.  Must be one of ['simple'|'bayesian'] (see below)
    n_neighbors : int
        number of neighbors to use

    Notes
    -----
    The two methods are as follows:

    - simple:
        The density at a point x is estimated by n(x) ~ k / r_k^n
    - bayesian:
        The density at a point x is estimated by n(x) ~ sum_{i=1}^k[1 / r_i^n].

    See Also
    --------
    KDE : kernel density estimation
    """
    def __init__(self, method='bayesian', n_neighbors=10):
        if method not in ['simple', 'bayesian']:
            raise ValueError("method = %s not recognized" % method)
        self.n_neighbors = n_neighbors
        self.method = method

    def fit(self, X):
        """Train the K-neighbors density estimator
        Parameters
        ----------
        X : array_like
            array of points to use to train the KDE.  Shape is
            (n_points, n_dim)
        """
        self.X_ = np.atleast_2d(X)
        if self.X_.ndim != 2:
            raise ValueError('X must be two-dimensional')

        self.bt_ = BallTree(self.X_)
        return self

    def eval(self, X, X_f):
        """Evaluate the kernel density estimation

        Parameters
        ----------
        X : array_like
            array of points at which to evaluate the KDE.  Shape is
            (n_points, n_dim), where n_dim matches the dimension of
            the training points.

        Returns
        -------
        dens : ndarray
            array of shape (n_points,) giving the density at each point.
            The density will be normalized for metric='gaussian' or
            metric='tophat', and will be unnormalized otherwise.
        """
        self.X_f = X_f
        X = np.atleast_2d(X)
        if X.ndim != 2:
            raise ValueError('X must be two-dimensional')

        if X.shape[1] != self.X_.shape[1]:
            raise ValueError('dimensions of X do not match training dimension')

        dist, ind = self.bt_.query(X, self.n_neighbors, return_distance=True)
        

        #measure similarity with neighbors 
        D, d_m=similarity(ind, self.X_f)
        
        #print(D)
        
        k = float(self.n_neighbors)
        ndim = X.shape[1]       

        if self.method == 'simple':
            dens = k / n_volume(dist[:, -1], ndim)

        elif self.method == 'bayesian':
            # XXX this may be wrong in more than 1 dimension!
            dens =  (k * (k + 1) * 0.5 / n_volume(1, ndim)
                    / (dist ** ndim).sum(1))            
        else:
            raise ValueError("Unrecognized method '%s'" % self.method)
        return dens, D
            
    def eval_indiv_feat(self, X, X_f):
        """Evaluate the kernel density estimation

        Parameters
        ----------
        X : array_like
            array of points at which to evaluate the KDE.  Shape is
            (n_points, n_dim), where n_dim matches the dimension of
            the training points.

        Returns
        -------
        dens : ndarray
            array of shape (n_points,) giving the density at each point.
            The density will be normalized for metric='gaussian' or
            metric='tophat', and will be unnormalized otherwise.
        """
        self.X_f = X_f
        X = np.atleast_2d(X)
        if X.ndim != 2:
            raise ValueError('X must be two-dimensional')

        if X.shape[1] != self.X_.shape[1]:
            raise ValueError('dimensions of X do not match training dimension')

        dist, ind = self.bt_.query(X, self.n_neighbors, return_distance=True)

        
        D_j, d_m = similarity(ind, self.X_f, indiv=True)
        return D_j
    

def extract_XY_global_pairs(T_in):
    '''
    This function reads a table 'T' of length 'n' and returns an
    array of its X_global,Y_global coordinates in the form:
    
            [x1, y1]
            [x2, y2]
            [x3, y3]
            ...
            ...
            [xn, yn]
    '''
    return np.vstack( (T_in['X_global'] , T_in['Y_global']) ).T


def add_density_universal(T, bandwidth = 50, cell_types = True, cell_classif = 'CNN'):
    
    T = T[ T['overlap'] != True ]
    T_mean = T[ T['overlap'] != True ]
    T_std = T[ T['overlap'] != True ]
   
    names_ = [name for name in T.colnames if np.array(T[name]).dtype == '>f4']
    names_ = [name for name in names_ if np.array(T[name]).mean() == np.array(T[name]).mean() ]
    T_mean_2 = pd.DataFrame(np.array(T), columns = names_ )
    T_std_2 = pd.DataFrame(np.array(T), columns = names_ )

    bandwidth_list=[bandwidth]
    #
    if cell_classif =='CNN':
        oc_cellType_objType_colname='CNN_cell_type'
    elif cell_classif =='SVM':
        oc_cellType_objType_colname='cell_type_SVM'
    else:
        print('Cell classifier not correct')
    kernel_list = ['knn']
    unique_prefix_keyword_density = 'density_cell_type_'
    

    for bandwidth in bandwidth_list:
        for kernel in kernel_list:
            density_feature_name = unique_prefix_keyword_density + kernel + '_' + str( bandwidth)  # create density column name
            new_mean_simil_all_feat = 'mean_all' + unique_prefix_keyword_density + kernel + '_' + str( bandwidth)  
            new_std_simil_all_feat = 'std_all' + unique_prefix_keyword_density + kernel + '_' + str( bandwidth)  

            T[density_feature_name] =   0.0      # add new 'density' column, i.e. 'density_feature_name'
            T[new_mean_simil_all_feat] =   0.0 
            T[new_std_simil_all_feat] =   0.0 
            print('shape mean', T_mean.as_array().shape)
            print('shape T',T.as_array().shape)
            
            
            if cell_types == False:
                T_objType = []
                T_objType_index=[]
                key_list=[]
                i=0            
                T_objType.append(T)  
                T_objType_index.append(np.arange( T_mean.as_array().shape[0]))
            
                names = [name for name in T_objType[i].colnames if np.array(T_objType[i][name]).dtype == '>f4']
                names = [name for name in names if np.array(T[name]).mean() == np.array(T[name]).mean() ]
                X_=T_objType[i][names].to_pandas().values[:,:2]
                X_f=T_objType[i][names].to_pandas().values[:,2:]            
                if len(T_objType[i]) > bandwidth  :
                    XY_global = extract_XY_global_pairs(T_objType[i])
                    if kernel == 'knn':
                            knd = KNeighborsDensity_modified("bayesian", bandwidth ) 
                            # bayesian method (also can be used with 'simple' method)
                            knd.fit(X_)
                            log_density, dist = knd.eval(X_, X_f)
                            dist_indiv_feat = knd.eval_indiv_feat(X_, X_f)

                    elif kernel == 'tophat' or kernel == 'gaussian':
                            kde = KDE(kernel , h=bandwidth)
                            kde.fit(XY_global)
                            log_density = kde.eval(XY_global)
                                # # # based on scikit-learn 
                                # # else:
                                # #     # initialize a kernel object
                                # #     kde_c = KernelDensity(bandwidth= bandwidth,  kernel=kernel).fit(XY_global)
                                # #     
                                # #     # estimate density in logarithmic scale
                                # #     log_density = kde_c.score_samples(XY_global)    
                                # populate the density column
                             
                            #Density and similarities all features
                    T[density_feature_name][T_objType_index[i]] = log_density
                    T[new_mean_simil_all_feat][T_objType_index[i]] = np.array(dist[0]).reshape(dist[0].shape[0])
                    T[new_std_simil_all_feat][T_objType_index[i]] =  np.array(dist[1]).reshape(dist[0].shape[0])

                    #Density and similarities individual features                            
                    rows = T_objType_index[i]
                    cols = names[2:]
                    cols = np.arange(2,len(names))
                    T_mean_2.iloc[rows,cols] = np.array(dist_indiv_feat)[0,:,:]
                    T_std_2.iloc[rows,cols] = np.array(dist_indiv_feat)[1,:,:]                           
                    i+=1                        
                        
            elif cell_types == True:          
                if oc_cellType_objType_colname in T.colnames:
                    T_objType = []
                    T_objType_index=[]
                    key_list=[]
                    i=0
                    print('num cell types',np.unique(T[oc_cellType_objType_colname]).shape[0])

                    if cell_classif =='CNN':
                        cell_types = ['lymphocyte', 'other', 'stroma', 'tumor']
                    elif cell_classif =='SVM':
                        cell_types = [0,1,2,3]
                        cell_types_raw = np.unique(T[oc_cellType_objType_colname]).astype(str)                
                    else:
                        print('Cell types not identified')
                    
                    
                    for key in cell_types:
                        if key in np.unique(T[oc_cellType_objType_colname].astype(str)):
                        
                            T_objType.append(T[ T[oc_cellType_objType_colname] == key ])  
                            T_objType_index.append(np.where( T[oc_cellType_objType_colname] == key )[0])
                            key_list.append(key)
                            names = [name for name in T_objType[i].colnames if np.array(T_objType[i][name]).dtype == '>f4']
                            names = [name for name in names if np.array(T[name]).mean() == np.array(T[name]).mean() ]

                            X_=T_objType[i][names].to_pandas().values[:,:2]
                            X_f=T_objType[i][names].to_pandas().values[:,2:]
                            if len(T_objType[i]) > bandwidth   and     key != 0  :
                        #if len(T_objType[i]) > bandwidth   and     key != 'other'  :

                        #  <----- condition.1 ------>        <--cond.2 -->
                        # cond.1: put a lower-limit on the number of cells of a given type. 
                        # Currently it is set to the 'bandwidth'
                        # cond.2: if key == 0, it is mark and therefore no need to estimate the density
                        #  -------------------------------------------------------------
                        # |  E s t i m a t e   K D E   D e n s i t y  ( in Log scale )  |
                        #  -------------------------------------------------------------
                        # Create an array consisitng of [X_global,Y_global] pairs only
                                print(key)
                                XY_global = extract_XY_global_pairs(T_objType[i])

                        # based on astroML                        
                                if kernel == 'knn':
                                    knd = KNeighborsDensity_modified("bayesian", bandwidth ) 
                                # bayesian method (also can be used with 'simple' method)
                                    knd.fit(X_)
                                    log_density, dist = knd.eval(X_, X_f)
                                    dist_indiv_feat = knd.eval_indiv_feat(X_, X_f)
                         # based on astroML    
                                elif kernel == 'tophat' or kernel == 'gaussian':
                                    kde = KDE(kernel , h=bandwidth)
                                    kde.fit(XY_global)
                                    log_density = kde.eval(XY_global)
                                # # # based on scikit-learn 
                                # # else:
                                # #     # initialize a kernel object
                                # #     kde_c = KernelDensity(bandwidth= bandwidth,  kernel=kernel).fit(XY_global)
                                # #     
                                # #     # estimate density in logarithmic scale
                                # #     log_density = kde_c.score_samples(XY_global)    
                                # populate the density column
                             
                                #Density and similarities all features
                                T[density_feature_name][T_objType_index[i]] = log_density
                                T[new_mean_simil_all_feat][T_objType_index[i]] = np.array(dist[0]).reshape(dist[0].shape[0])
                                T[new_std_simil_all_feat][T_objType_index[i]] =  np.array(dist[1]).reshape(dist[0].shape[0])

                                #Density and similarities individual features                            
                                rows = T_objType_index[i]
                                cols = names[2:]
                                cols = np.arange(2,len(names))
                                T_mean_2.iloc[rows,cols] = np.array(dist_indiv_feat)[0,:,:]
                                T_std_2.iloc[rows,cols] = np.array(dist_indiv_feat)[1,:,:]                           
                                i+=1
                        
                            else:
                                print('Saving null values for marks/cell type ',key)

                                T[density_feature_name][T_objType_index[i]] = 0
                                T[new_mean_simil_all_feat][T_objType_index[i]] = 0
                                T[new_std_simil_all_feat][T_objType_index[i]] = 0
                            
                                rows = T_objType_index[i]
                                cols = names[2:]  
                                cols = np.arange(2,len(names))
                            
                                T_mean_2.iloc[rows,cols] = np.zeros_like(T_mean_2.iloc[rows,cols] )
                                T_std_2.iloc[rows,cols] = np.zeros_like(T_std_2.iloc[rows,cols] )                             

                                i+=1
                        else:
                            print('There are no cells of type',key)
                            T_objType.append(T[ T[oc_cellType_objType_colname] == key ])  
                            T_objType_index.append(np.where( T[oc_cellType_objType_colname] == key )[0])
                            key_list.append(key)
                            names = [name for name in T_objType[i].colnames if np.array(T_objType[i][name]).dtype == '>f4']

                            T[density_feature_name][T_objType_index[i]] = 0
                            T[new_mean_simil_all_feat][T_objType_index[i]] = 0
                            T[new_std_simil_all_feat][T_objType_index[i]] = 0

                            T_mean[names[2:]][T_objType_index[i]] = np.zeros_like(np.array(T_objType[i][names]))
                            T_std[names[2:]][T_objType_index[i]] = np.zeros_like(np.array(T_objType[i][names]))                     
                        
                            rows = T_objType_index[i]
                            cols = names[2:]    
                            cols = np.arange(2,len(names))
                            T_mean_2.iloc[rows,cols] = np.zeros_like(np.array(T_objType[i][names[2:]]))
                            T_std_2.iloc[rows,cols] = np.zeros_like(np.array(T_objType[i][names[2:]]))                        
                            i+=1
                  
            else:
                print (" ---------------------------------------------")
                print ("|  Input table has no column called '{}'  |".format(oc_cellType_objType_colname))
                print (" ---------------------------------------------")
                
    return T, T_mean_2, T_std_2


def calculate_medians(T_d,T_d_ind_mean, T_d_ind_std, cell_types = True, k=50, cell_classif = 'CNN'):
    
    names = [name for name in T_d.colnames if len(T_d[name].shape) <= 1]
    df = T_d[names].to_pandas()   
    names_v = [name for name in T_d.colnames if len(T_d[name].shape) <= 1]
    names_v2 = [name for name in names_v if np.array(T_d[name]).dtype == '>f4']
    names_v2 = [name for name in names_v2 if np.array(T_d[name]).mean() == np.array(T_d[name]).mean()]
    names_v2 = names_v2[2:]
    features = T_d.colnames[:-2] 
    
    df_mean = T_d_ind_mean[names_v2]
    df_std = T_d_ind_std[names_v2]  

    
    if cell_types == True:
        if cell_classif == 'CNN':
            cell_types = [b'lymphocyte', b'other', b'stroma', b'tumor']        
        elif cell_classif == 'SVM':
            cell_types = [0,1,2,3]
            cell_types_raw = np.unique(T[oc_cellType_objType_colname]).astype(str)
        rho_cells_medians=[]
        S_mean_cells_medians=[]
        S_std_cells_medians=[]

        
        print('shape d_mean', df_mean.shape)
        print('shape df', df.shape)
        
        if df_mean.empty==True:
            print('EMPTY!')
            Sf_mean_cells_medians=[np.zeros((46,1)) for i in cell_types]
            Sf_std_cells_medians=[np.zeros((46,1)) for i in cell_types]
        
        else:
            Sf_mean_cells_medians=[]
            Sf_std_cells_medians=[]            
            for key in cell_types:
                #if key in np.unique(T_d['cell_type_SVM']):        
                if key in np.unique(T_d['CNN_cell_type']):        
                    #Ali's density measure
                    #rho_cells_medians.append(df.loc[df['cell_type_SVM'].eq(key),'density_cell_type_knn_'+str(k)].median())
                    rho_cells_medians.append(df.loc[df['CNN_cell_type'].eq(key),'density_cell_type_knn_'+str(k)].median())
                    #New similarity measure
                   # S_mean_cells_medians.append(df.loc[df['cell_type_SVM'].eq(key),'mean_alldensity_cell_type_knn_'+str(k)].median() )
                   # S_std_cells_medians.append(df.loc[df['cell_type_SVM'].eq(key),'std_alldensity_cell_type_knn_'+str(k)].median())
        
#                    Sf_mean_cells_medians.append(df_mean.loc[df['cell_type_SVM'].eq(key)].median())
#                    Sf_std_cells_medians.append(df_std.loc[df['cell_type_SVM'].eq(key)].median())
                    S_mean_cells_medians.append(df.loc[df['CNN_cell_type'].eq(key),'mean_alldensity_cell_type_knn_'+str(k)].median() )
                    S_std_cells_medians.append(df.loc[df['CNN_cell_type'].eq(key),'std_alldensity_cell_type_knn_'+str(k)].median())
        
                    Sf_mean_cells_medians.append(df_mean.loc[df['CNN_cell_type'].eq(key)].median())
                    Sf_std_cells_medians.append(df_std.loc[df['CNN_cell_type'].eq(key)].median())
                
                else:
                    print('there are no cells of type ',key)
                    rho_cells_medians.append(0)
                    S_mean_cells_medians.append(0)
                    S_std_cells_medians.append(0)
                    #Sf_mean_cells_medians.append(np.zeros_like(df_mean.loc[df['cell_type_SVM']].median())) 
                    #Sf_std_cells_medians.append(np.zeros_like(df_mean.loc[df['cell_type_SVM']].median()))
                    Sf_mean_cells_medians.append(np.zeros_like(df_mean.loc[df['CNN_cell_type']].median())) 
                    Sf_std_cells_medians.append(np.zeros_like(df_mean.loc[df['CNN_cell_type']].median()))                    
                    
    else:
        rho_cells_medians=[]
        S_mean_cells_medians=[]
        S_std_cells_medians=[]

        
        print('shape d_mean', df_mean.shape)
        print('shape df', df.shape)
        print(df['density_cell_type_knn_'+str(k)])
        if df_mean.empty==True:
            print('EMPTY!')
            Sf_mean_cells_medians=[np.zeros((46,1)) for i in cell_types]
            Sf_std_cells_medians=[np.zeros((46,1)) for i in cell_types]
        else:

            Sf_mean_cells_medians=[]
            Sf_std_cells_medians=[]            

            rho_cells_medians.append(df['density_cell_type_knn_'+str(k)].median())

            S_mean_cells_medians.append(df['mean_alldensity_cell_type_knn_'+str(k)].median() )
            S_std_cells_medians.append(df['std_alldensity_cell_type_knn_'+str(k)].median())
        
            Sf_mean_cells_medians.append(df_mean.median())
            Sf_std_cells_medians.append(df_std.median())            
            
    return rho_cells_medians, S_mean_cells_medians, S_std_cells_medians, Sf_mean_cells_medians, Sf_std_cells_medians




def writer(header, data, filename, option):
        with open (filename, "w", newline = "") as csvfile:
            if option == "write":

                catalogs = csv.writer(csvfile, delimiter=",")
                catalogs.writerow(header)
                for x in data:
                    catalogs.writerow(x)
            elif option == "update":
                writer = csv.DictWriter(csvfile, fieldnames = header, delimiter=",")
                writer.writeheader()
                writer.writerows(data)
            else:
                print("Option is not known")


def updater(filename):
    with open(filename, newline= "") as file:
        readData = [row for row in csv.DictReader(file)]


    readHeader = readData[0].keys()
    writer(readHeader, readData, filename, "update")



