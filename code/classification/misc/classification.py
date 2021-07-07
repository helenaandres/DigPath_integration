
import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import os

import argparse

from misc.dataset import Dataset, DatasetWhole
from misc.helpers import normalizeRNA,save_embedding

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
from misc.helpers import normalizeRNA,save_embedding

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.inspection import permutation_importance

def classify(data, labels, clasif_type, cross_val = False):
    
    if cross_val == False:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, 
                                                    test_size=0.2, random_state=42, stratify=labels)
        if clasif_type == 'RF':
             clf = make_pipeline(StandardScaler(),RandomForestClassifier(max_depth=10, 
                                                                         random_state=42, class_weight = 'balanced'))
        elif clasif_type == 'SVM': 
             clf = make_pipeline(StandardScaler(),SVC(gamma='auto', class_weight = 'balanced', probability=True))
        elif clasif_type == 'LogReg': 
             clf = make_pipeline(StandardScaler(),LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000))
        elif clasif_type == 'NB':
             clf = make_pipeline(StandardScaler(),GaussianNB())

        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_train)
        conf_mat = confusion_matrix(y_train, y_pred)
        y_pred_test = clf.predict(X_test)
        conf_mat_test = confusion_matrix(y_test, y_pred_test)
     
        if clf.predict_proba(X_test).shape[1] >2:         
            roc_auc_test = roc_auc_score(y_test,clf.predict_proba(X_test), multi_class = 'ovo')           
            roc_auc_train = roc_auc_score(y_train,clf.predict_proba(X_train), multi_class = 'ovo')               
            
        else:             
            roc_auc_test = roc_auc_score(y_test,clf.predict(X_test))            
            roc_auc_train = roc_auc_score(y_train,clf.predict(X_train))             

        train_score = np.mean(train_score)
        test_score = np.mean(test_score)
        roc_auc_train = np.mean(roc_auc_train)
        roc_auc_test = np.mean(roc_auc_test)
    
    elif cross_val == True:
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        train_score = []
        test_score = []
        roc_auc_test = []
        roc_auc_train = []
        i=0
        for train_index, test_index in skf.split(data, labels):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            if clasif_type == 'RF':
                 clf = make_pipeline(StandardScaler(),RandomForestClassifier(max_depth=2, random_state=0, class_weight = 'balanced'))
            elif clasif_type == 'SVM': 
                 clf = make_pipeline(StandardScaler(),SVC(gamma='auto', class_weight = 'balanced', probability=True))
            elif clasif_type == 'LogReg': 
                 clf = make_pipeline(StandardScaler(),LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000))
            elif clasif_type == 'NB':
                 clf = make_pipeline(StandardScaler(),GaussianNB())                    
            clf.fit(X_train, y_train)
            train_score.append(clf.score(X_train, y_train))
            test_score.append(clf.score(X_test, y_test))
            y_pred = clf.predict(data)
            conf_mat = confusion_matrix(labels, y_pred)
            y_pred_test = clf.predict(X_test)
            conf_mat_test = confusion_matrix(y_test, y_pred_test)
            if clf.predict_proba(X_test).shape[1] >2:
                roc_auc_test.append(roc_auc_score(y_test, clf.predict_proba(X_test), multi_class = 'ovo'))
                roc_auc_train.append(roc_auc_score(y_train, clf.predict_proba(X_train), multi_class = 'ovo'))
            else:
                roc_auc_test.append(roc_auc_score(y_test, clf.predict(X_test)))
                roc_auc_train.append(roc_auc_score(y_train, clf.predict(X_train)))

            
            i+=1 
        train_score_m = np.mean(train_score)
        test_score_m = np.mean(test_score)
        roc_auc_train_m = np.mean(roc_auc_train)
        roc_auc_test_m = np.mean(roc_auc_test)
        
        train_score_std = np.std(train_score)
        test_score_std = np.std(test_score)
        roc_auc_train_std = np.std(roc_auc_train)
        roc_auc_test_std = np.std(roc_auc_test)        
    return train_score_m, test_score_m, roc_auc_train_m, roc_auc_test_m, train_score_std, test_score_std, roc_auc_train_std, roc_auc_test_std



def plot_learning_curve(clasifiers, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):    
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    
    
    for clasifier in clasifiers:
        if clasifier == 'RF':
            estimator = RandomForestClassifier(max_depth=10, random_state=42, class_weight = 'balanced')
            cl = 'r'
        elif clasifier == 'SVM':
            estimator = SVC(gamma='auto', class_weight = 'balanced', probability=True)
            cl = 'g'
        elif clasifier == 'LogReg':
            estimator = LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000)
            cl = 'y'
        elif clasifier == 'NB':
            estimator = GaussianNB()
            cl = 'm'
        train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, 
                       return_times=True)
        print(train_scores.shape)
        print(test_scores)
    
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
        print(test_scores_mean)
        print(test_scores_std)

    # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color=cl)
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color=cl)
        axes[0].plot(train_sizes, train_scores_mean, 'o--', color=cl,
                 label=clasifier+" training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color=cl,
                 label=clasifier+" cross-validation score")
        
        axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


def classify_plots(X,y,clasifiers,name_file,modality):

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))


    title = r"Learning Curves ("+modality+")"

    cv = 5

    plot_learning_curve(clasifiers, title, X, y, axes=None, ylim=(0.5, 1.01),
                    cv=cv, n_jobs=4)
    

    #plt.show()
    plt.savefig("./plots_training_curve/"+name_file)
    plt.close()
    return plt


def feature_analysis_train(data, labels, X_train_feat_, clasif_type_, impurity = False):
    
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    train_score = []
    test_score = []
    roc_auc_test = []
    roc_auc_train = []

    feat_imp = []
    feat_imp_values = []

    vals_ = []

    i=0
    for train_index, test_index in skf.split(data, labels):
        X_train, X_test = data[train_index], data[test_index]
        #print(labels)
        y_train, y_test = labels[train_index], labels[test_index]   
        X_train_df = pd.DataFrame(data=X_train, columns=X_train_feat_)
        X_test_df = pd.DataFrame(data=X_test, columns=X_train_feat_)

        if clasif_type_ == 'RF':
            clf = RandomForestClassifier(max_depth=2, random_state=0, class_weight = 'balanced')
            clf.fit(X_train, y_train)
            if impurity == True:
                tree_feature_importances = clf.feature_importances_
                sorted_idx = tree_feature_importances.argsort()
                y_ticks = np.arange(0, len(X_test_df.columns[-50:]))
                fig, ax = plt.subplots()
                ax.barh(y_ticks, tree_feature_importances[sorted_idx[-50:]])
                ax.set_yticklabels(X_test_df.columns[sorted_idx])
                ax.set_yticks(y_ticks)
                ax.set_title("Random Forest Feature Importances (MDI)")
                fig.tight_layout()
                plt.savefig('./new_results_feature_importance_METABRIC_v0/FeatImpurity_'+str(i)+'_'+clasif_type+'.pdf',
                            bbox_inches='tight',dpi=100)
                plt.close()
                print('plot Impurity measures done!')

            result = permutation_importance(clf, X_test, y_test, n_repeats=10,
                                    random_state=42, n_jobs=2)
            sorted_idx = result.importances_mean.argsort()
            vals_.append(result.importances_mean[sorted_idx])
            vals = result.importances_mean[sorted_idx]
            feature_importance = pd.DataFrame(list(zip(X_test_df.columns, vals)), columns=['col_name','feature_importance_vals'])
            feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)          

  
            print('plot Permutation measures done!')
        elif clasif_type_ == 'SVM': 
            clf = SVC(gamma='auto', class_weight = 'balanced', probability=True, kernel='linear')
            clf.fit(X_train, y_train)
            if impurity == True:
                tree_feature_importances = clf.feature_importances_
                sorted_idx = tree_feature_importances.argsort()
                y_ticks = np.arange(0, len(X_test_df.columns[-50:]))
                fig, ax = plt.subplots()
                ax.barh(y_ticks, tree_feature_importances[sorted_idx[-50:]])
                ax.set_yticklabels(X_test_df.columns[sorted_idx])
                ax.set_yticks(y_ticks)
                ax.set_title("Random Forest Feature Importances (MDI)")
                fig.tight_layout()
                plt.savefig('./new_results_feature_importance_METABRIC_v0/FeatImpurity_'+str(i)+'_'+clasif_type+'.pdf',
                                bbox_inches='tight',dpi=100)
                plt.close()
                print('plot Impurity measures done!')

            result = permutation_importance(clf, X_test, y_test, n_repeats=10,
                                    random_state=42, n_jobs=2)
            sorted_idx = result.importances_mean.argsort()
            vals_.append(result.importances_mean[sorted_idx])
            vals = result.importances_mean[sorted_idx]
            feature_importance = pd.DataFrame(list(zip(X_test_df.columns, vals)), columns=['col_name','feature_importance_vals'])
            feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)          

   
            print('plot Permutation measures done!')  

        elif clasif_type_ == 'LogReg': 

            clf = LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000)
            clf.fit(X_train, y_train)
            if impurity == True:
                tree_feature_importances = clf.feature_importances_
                sorted_idx = tree_feature_importances.argsort()
                y_ticks = np.arange(0, len(X_test_df.columns[-50:]))
                fig, ax = plt.subplots()
                ax.barh(y_ticks, tree_feature_importances[sorted_idx[-50:]])
                ax.set_yticklabels(X_test_df.columns[sorted_idx])
                ax.set_yticks(y_ticks)
                ax.set_title("Random Forest Feature Importances (MDI)")
                fig.tight_layout()
                plt.savefig('./new_results_feature_importance_METABRIC_v0/FeatImpurity_'+str(i)+'_'+clasif_type+'.pdf',
                                bbox_inches='tight',dpi=100)
                plt.close()
                print('plot Impurity measures done!')

            result = permutation_importance(clf, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
            sorted_idx = result.importances_mean.argsort()
            vals_.append(result.importances_mean[sorted_idx])
            vals = result.importances_mean[sorted_idx]
            feature_importance = pd.DataFrame(list(zip(X_test_df.columns, vals)), columns=['col_name','feature_importance_vals'])
            feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)          

            print('plot Permutation measures done!')            
        elif clasif_type_ == 'NB':
            clf = GaussianNB() 
            clf.fit(X_train, y_train)
            if impurity == True:
                tree_feature_importances = clf.feature_importances_
                sorted_idx = tree_feature_importances.argsort()
                y_ticks = np.arange(0, len(X_test_df.columns[-50:]))
                fig, ax = plt.subplots()
                ax.barh(y_ticks, tree_feature_importances[sorted_idx[-50:]])
                ax.set_yticklabels(X_test_df.columns[sorted_idx])
                ax.set_yticks(y_ticks)
                ax.set_title("Random Forest Feature Importances (MDI)")
                fig.tight_layout()
                plt.savefig('./new_results_feature_importance_METABRIC_v0/FeatImpurity_'+str(i)+'_'+clasif_type+'.pdf',
                                bbox_inches='tight',dpi=100)
                plt.close()
                print('plot Impurity measures done!')

            result = permutation_importance(clf, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
            sorted_idx = result.importances_mean.argsort()
            vals_.append(result.importances_mean[sorted_idx])
            vals = result.importances_mean[sorted_idx]
            feature_importance = pd.DataFrame(list(zip(X_test_df.columns, vals)), columns=['col_name','feature_importance_vals'])
            feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)          


            print('plot Permutation measures done!')  

        train_score.append(clf.score(X_train, y_train))
        test_score.append(clf.score(X_test, y_test))
        y_pred = clf.predict(X_train)
        conf_mat = confusion_matrix(y_train, y_pred)
        y_pred_test = clf.predict(X_test)
        conf_mat_test = confusion_matrix(y_test, y_pred_test)

        if clf.predict_proba(X_test).shape[1] >2:
            roc_auc_test.append(roc_auc_score(y_test, clf.predict_proba(X_test), multi_class = 'ovo'))
            roc_auc_train.append(roc_auc_score(y_train, clf.predict_proba(X_train), multi_class = 'ovo'))

        else:
            roc_auc_test.append(roc_auc_score(y_test, clf.predict(X_test)))
            roc_auc_train.append(roc_auc_score(y_train, clf.predict(X_train)))

        i+=1
    return roc_auc_train, roc_auc_test, vals_            
            
   