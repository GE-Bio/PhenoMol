#!/usr/bin/env python
# coding: utf-8


import os
from os import path
import datetime
import warnings
# warnings.filterwarnings('ignore')
import itertools

import numpy as np
import pandas as pd
import networkx as nx

from oi3_graph import OIgraph

import phenoMol_v1
from phenoMol_v1 import generateFinalPrizes
from phenoMol_v1 import Interactome
from phenoMol_v1 import init_OIruns
from phenoMol_v1 import hyperparameter_search
from phenoMol_v1 import perform_oiRuns
from phenoMol_v1 import _generate_oiRuns
from phenoMol_v1 import analyze_oi_runs
from phenoMol_v1 import generate_oiPrizes

import random

import math

import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import auc, precision_recall_fscore_support, roc_curve, PrecisionRecallDisplay, RocCurveDisplay, precision_recall_curve

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict


from scipy import optimize
from scipy import stats

import matplotlib.pyplot as plt

import subprocess # to call R-scripts

import shutil # copy files

import multiprocessing

def getMaxNumProcessors():
    max_num_processors = multiprocessing.cpu_count()
    # try: max_num_processors = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    # except KeyError: max_num_processors = multiprocessing.cpu_count()
    return max_num_processors

def impute_data_knn(X, n_neighbors=5, missing_values=np.nan):
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors, missing_values=missing_values)
    X_i = imputer.fit_transform(X)
    return X_i

def getUniqueKeepOrder(x):
    return list(dict.fromkeys(x)) # removes duplicates and keeps ordering

def choosek(x_cols,k):
    import itertools
    return list(itertools.combinations(x_cols,k))

def getDirName(fold_num, foldDirPrefix = 'fold'):
    if fold_num < 10:
        foldDirName = foldDirPrefix + '_00' + str(fold_num)
    elif fold_num < 100:
        foldDirName = foldDirPrefix + '_0' + str(fold_num)
    else:
        foldDirName = foldDirPrefix + '_' + str(fold_num)
    return foldDirName
    
def compute_ACFT_Score(df_ACFT_tables, ACFT_feature, rawValue):

    # if ACFT_feature not in ['ACFT_Maximum_Deadlift', 'ACFT_Standing_Power_Throw', 'ACFT_Hand_Release_Pushups', 'ACFT_Sprint_Drag_Carry', 'ACFT_Leg_Tuck_OR_Plank', 'ACFT_2_Mile_Run']:
    #     return rawValue
    
    # conversion of measurment value in seconds to minutes used in df_ACFT_tables
    if ACFT_feature == 'ACFT_Sprint_Drag_Carry':
        rawValue = rawValue / 60.0
    
    dt = df_ACFT_tables.copy()
    if ACFT_feature in ['ACFT_Sprint_Drag_Carry','ACFT_2_Mile_Run']:
        dt = dt.loc[dt[ACFT_feature] >= rawValue].copy()
    else:
        dt = dt.loc[dt[ACFT_feature] <= rawValue].copy()
    
    if len(dt) == 0:
        return 0
    else:
        return max(dt['Points'].values)


######################################################################

BASE_PA_MODEL_CSV = 'base_PA_models.csv'
BASE_PNM_EA_MODEL_CSV = 'base_PNM_EA_models.csv'


#################################################################################################################################################################
#####
#################################################################################################################################################################

def get_default_run_parameters(max_num_processors = 15):

    num_available_processors = getMaxNumProcessors()
    print('num_available_processors:', num_available_processors)
    
    max_num_processors = min(max_num_processors, num_available_processors)

    
    print('max_num_processors:', max_num_processors)
    print('')

    # for oi3, this is the number of pcsf runs to compute robustness (edge noise)
    NUM_SAMPLING_REPS = 100

    _W = 1
    _B = 5
    _G = 0
    _R = 1

    default_run_parameters = {

        # the following need to be updated with defined values
        'base_dir': '',
        'r_path': '',
        'R_LIBRARY_DIR': '',
        'RCODE_SOURCE_DIR': '',
        'interactome_db_dir': '', 
        'interactomeName': '', 
        'analysis_dir': '', 
        'analysis_results_dir': '', 
        'models_dir': '', 

        'max_num_cpus': max_num_processors,

        'parallelize_folds': True,

        'Compute_ACFT_Score': False,  # set to True when running case of six latent ensemble models, one latent model for each ACFT event
        
        'randomState': 123, 
        'permutationState': -1, # set to 0 to generate actual model during permutation testing and an integer value >0 for a specific permutation of the data
        'misclassifyOutcomeFraction': 0,

        'ASSEMBLE_PRIZE_DATA': True,
        'GENERATE_RAW_PRIZES': True,
        'GENERATE_FINAL_PRIZES': True,
        'SETUP_OI_RUNS': True,
        'CREATE_OI_GRAPH_INSTANCE': True,
        'GENERATE_OI_RUNS': True,
        'PROCESS_OI_RUNS': True,
        'DELETE_PCSF_RUN_POST_PROCESSING': True,
        'DELETE_PCSF_RUN_MEASURES_POST_PROCESSING': True,
        'BUILD_BASE_PNM_EA_MODELS': True,
        'COMPUTE_OUTCOME_BASE_EA_MODEL': True,    # set False for  specfic case when using ACFT scoring system to compute ACFT_Total_Points to save computation time
        'GENERATE_BASE_PNM_EA_MODEL_TEST_RESULTS': False,  #Not needed if ensemble models are built which will include single base models in the output

        'protein_protein_metabolite_interactome_fp': 'interactome_ppmi_string12_stitch5.txt',
        'metabolite_nodeInfo_fp': 'metabolite_node_Info_v5.txt', 
        'protein_nodeInfo_fp': 'protein_node_Info_v5.txt', 
        'transcript_nodeInfo_fp': 'transcript_node_Info_v5.txt', 
        'gene_nodeInfo_fp': 'gene_node_Info_v5.txt',
        'mature_mir_nodeInfo_fp': 'mature_mir_node_Info_v5.txt',
        'mature_mir_transcript_nodeInfo_fp': 'mature_mir_transcript_node_Info_v5.txt',
        'mature_mir_gene_nodeInfo_fp': 'mature_mir_gene_node_Info_v5.txt',       
        'protein_transcript_gene_links_fp': 'protein_transcript_gene_links_ensembl_v110.txt',
        'transcript_gene_links_fp': 'transcript_gene_links_ensembl_v110.txt',
        'interactome_transcript_pp_fp': 'interactome_transcript_pp_ensembl_v110.txt',
        'interactome_gene_pp_fp': 'interactome_gene_pp_ensembl_v110.txt',
        'interactome_gene_transcript_fp': 'interactome_gene_transcript_ensembl_v110.txt',
        'interactome_transcript_mature_mir_fp': 'interactome_transcript_mature_mir_ensembl_v110.txt',
        'interactome_gene_transcript_mature_mir_fp': 'interactome_gene_transcript_mature_mir_ensembl_v110.txt',
        'interactome_mature_mir_transcript_fp': 'interactome_mature_mir_transcript_ensembl_v110.txt',
        'interactome_mature_mir_protein_fp': 'interactome_mature_mir_protein_ensembl_v110.txt',
    
        'use_RNA': True,
        'use_DNA': True,
        'use_MicroRNA': False,
        'reduce_MicroRNA_PNM_Weighting': True,  # when generating PNM clusters from the PN, this flag when set to true will effectively zero the edge weight between a mature miRNA to its target mRNA
    
        # min_num_correlation_data_points sets the required minimum number of data points to compute a correlation
        'min_num_correlation_data_points' : 10,
    
    
        'oi_run_phenotype_critical_p_value' : 0.05,
        
        'prize_critical_p_value' : 0.2,


        'median_normalize_prizeNames' : True,  # set True when comparing across phenotypes using the same omics and False when comparing within a phenotype across omics
        'median_prize_value' : 0.2,
        

    
        'save_hyperparam_results': False,
        'fn_suffix_hyperparam_results': 'hyperparam_result',
        'save_hyperparam_summary': True,
        'fn_suffix_hyperparam_summary': 'hyperparam_summary',
        
        'weight_phenotype_prizes': False,
        'weight_phenotype_column':'weight_phenotype', # column that needs to be in the file referenced by prize_name_info_fp

        'removePrizeNodeNames' : [],

        'graph' : None,
        'source_interactome' : None,
        'targetPipelines' : [],   # empty list means run them all
        'targetNodeSpecies' : [],   # empty list means run them all
        'num_reps': NUM_SAMPLING_REPS,
        'num_tuning_sampling_reps': 10,
        'tuningPhenotypes' : [],  # hyperparameter search across phenotypes
        'tune_R' : True,
        'Ws' : [_W],           # hyperparameter search of w
        'Bs' : [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 6, 12, 24, 48],  # hyperparameter tuning of b
        'Gs' : [_G],           # hyperparameter search of g
        'Rs' : [_R],           # hyperparameter search of r
        'targetPhenotypes' : None,

        'tuneOIruns' : True,
        'targetNumTerminalNodes' : 128,
        
    
        # w parameter scales dummy node edge weights - controlling the number of trees
        # b parameter scales prizes - controlling the tradeoff between more terminals and using less reliable edges
        # g parameter scales the cost of node degree (reduce hubs) by 10**g 
        # r parameter scales the prizes of RNA/DNA species over Protein/Metabolite species
        # edge_noise parameter is the gausian edge noise
        
        'w':_W,           # scales dummy node edge weights
        'b':_B,         # scales prizes
        'g':_G,           # scales the cost of node degree (hub penalty)
        'r':_R,           # weight RNA/DNA species over Protein/Metabolite species
    
        'noise': 0.1, 
        'edge_noise':0.1, 
        'dummy_mode':'terminals', 
        'exclude_terminals': False, 
        'seed':123, 
        'skip_checks':False,
        'noisy_edges_reps':NUM_SAMPLING_REPS,
        'random_terminals_reps':min(100,NUM_SAMPLING_REPS),
        'min_robustness':0.2,
        'min_component_size':5,
        'max_graph_size':2000,   # laregest size of a robust graph
        'maxNumNodesToComputeAvgPath':2000,  # requires ~ 60 seconds to compute at 4000
        'maxNumNodesToComputeBetweenness':2000,
        'verbose_oi':False,
        
        ############################

        'SEX_COLUMN_NAME' : 'Sex',  # data frame column name for sex
        'SEX_VALUES' : [],   # if empty then do not filter for sex, otherwise enter values of which Sex to use
        'USE_SEX_MEDIAN_NORMALIZED_DATA': False,
    
        'Y_METRIC': 'outcome',
        'BINARY_OUTCOME': False,
        'INCLUDE_PNM_MINUS_99': False,
        'INCLUDE_PNM0': True,
        'MIN_NUM_MEASURES_PER_FEATURE': 3,
        'PA_MAX_N_FEATURES': 20,
        'EA_MAX_N_FEATURES': 20,
        'SPLS_RANDOM_SEED': 123,
        'TUNE_SPLS_MEASURE': 'MSE',
        'TUNE_SPLS_NUM_FOLDS': 3,
        'TUNE_SPLS_NUM_REPEATS': 50,
        'TUNE_SPLS_MIN_FEATURE_STABILITY': 0.8,
        'PLSDA_DISTANCE_METHOD': 'max.dist',    # "mahalanobis.dist"

        'NUM_OI_FOLDS': 5,

        'build_PAs': False, 
    
        'imputeByClass': True,
        'imputeTestDataWithTrainingSet': True,
        'knn_num_neighbors': 5,
        'num_cv_folds': 3,
        'numFoldGenerations': 33,

        'maxNumBaseModels': 2,  # maximum number of base PNM EA models (base_PNM_EA_models) to combine into an ensemble PNM EA model (ensemble_PNM_EA_model)
        'maxNumNCKruns': 2000,  #  basically this is to prevent a forever loop. (e.g. 20 PNMs nck(n=20, k=2) is 190 and 1140(k=3), 4845(k=4), 15504(k=5))

        'TOP_N_LATENT_METRIC_ENSEMBLE_MODELS': 1,
         # 2^6 = 64
         # 3^6 = 729
        'MAX_NUM_LATENT_BASE_MODELS': 100,  # for ACFT model (Step_2) note that when this is smaller than maxNumBaseModels it will fiter out candidate models
        'TOP_N_MODELS' : 3650,   # 3650 = 5 folds x  (729 + 1) NCK ACFT Ensemble Models per fold plus one in case PNM-99 case is run

        'USE_SINGLE_MODEL_RANKING_METRIC': True,  # when set to True, the metric within column MODEL_RANKING_METRIC is used, 
                                                  # and when set to False, two rankings are used, mean_cv_test_mse and mean_cv_test_adj_R2
        'MODEL_RANKING_METRIC': 'mean_cv_test_adj_R2',   # alternatives are 'mean_cv_test_R2'  'mean_cv_test_mse',
        'MODEL_RANKING_METRIC_ascending': False,   # True for 'mean_cv_test_mse' and False for 'mean_cv_test_R2' and 'mean_cv_test_adj_R2'

        'create_All_fold':False,  # The All fold uses all of the data

        'create_CV_test_folds':True,

    }
    return default_run_parameters

#################################################################################################################################################################
#####
#################################################################################################################################################################

def save_run_parameters(runParm):
    (pd.DataFrame.from_dict(data=runParm, orient='index').to_csv(path.join(runParm['analysis_dir'],'run_parameters.csv'), header=False))

#################################################################################################################################################################
#####
#################################################################################################################################################################

def _nck_TrainTestEnsembleModel(params):

    import datetime
    
    job_id = params['job_id']  
    proc_id = params['proc_id']
    # Y_METRIC = params['Y_METRIC']
    df_train_fold_imputed = params['df_train_fold_imputed']   
    df_test_fold_imputed = params['df_test_fold_imputed']   
    df_All_Base_Models = params['df_All_Base_Models']
    base_model_name_cn = params['base_model_name_cn']
    y_metric_list = params['sub_y_metric_list']
    nck_list = params['sub_nck_list']
    randomState = params['randomState']
    num_cv_folds = params['num_cv_folds']
    numFoldGenerations = params['numFoldGenerations']
    
    df = df_train_fold_imputed
    all_cols = df_train_fold_imputed.columns.to_list()
    dfnp = df.to_numpy()

    numObservations = len(df_train_fold_imputed)

    Y_METRICS = getUniqueKeepOrder(y_metric_list)
    
    # initialize data cuts
    data_cuts = {}
    for Y_METRIC in Y_METRICS:
    
        res = np.unique(df[Y_METRIC])
        if len(res)==2:
            ybin = np.array(df[Y_METRIC], dtype=int)
        else:
            medianValue = df[Y_METRIC].median()
            meanValue = df[Y_METRIC].mean()
            v = df[Y_METRIC].tolist()
            if meanValue <= medianValue:
                ybin = list(map(lambda x: 1 if x >= medianValue else 0, v))
            else:
                ybin = list(map(lambda x: 0 if x <= medianValue else 1, v))
            ybin = np.array(ybin, dtype=int)
                
        for si in range(numFoldGenerations):
            rsv = (randomState + si)
            
            skf = StratifiedKFold(n_splits=num_cv_folds,shuffle=True, random_state = rsv)
            skf.get_n_splits(dfnp, ybin)
    
            cvi = 0
            for train_index, test_index in skf.split(dfnp, ybin):
                data_cuts[(Y_METRIC + '_train_index_' + str(si) + '_' + str(cvi))] = train_index
                data_cuts[(Y_METRIC + '_test_index_' + str(si) + '_' +  str(cvi))] = test_index
                cvi += 1
    
    # initialize metrics
    ensembleModelMetrics = {}
    for nck_i in range(len(nck_list)):
        cv_train_mse = []
        cv_test_mse = []
        
        cv_train_R2 = []
        cv_test_R2 = []
        
        cv_train_adj_R2 = []
        cv_test_adj_R2 = []
    
        ensembleModelMetrics[str(nck_i) + 'cv_train_mse'] = cv_train_mse
        ensembleModelMetrics[str(nck_i) + 'cv_test_mse'] = cv_test_mse
    
        ensembleModelMetrics[str(nck_i) + 'cv_train_R2'] = cv_train_R2
        ensembleModelMetrics[str(nck_i) + 'cv_test_R2'] = cv_test_R2
    
        ensembleModelMetrics[str(nck_i) + 'cv_train_adj_R2'] = cv_train_adj_R2
        ensembleModelMetrics[str(nck_i) + 'cv_test_adj_R2'] = cv_test_adj_R2
    
                
    # generate cv metrics on the fold training data
    
    for si in range(numFoldGenerations):
    
        for cvi in range(num_cv_folds):
            
            # first generate all base models for this specific cut of the data
            
            baseModelLookup = {}
            baseModelIndx = 0
            y_pred_train_base_models = []
            y_pred_test_base_models = []
            num_features_base_models = []
            y_train_model = []
            y_test_model = []
    
            Y_METRICS = getUniqueKeepOrder(y_metric_list)
            
            for Y_METRIC in Y_METRICS:
    
                train_index = data_cuts[(Y_METRIC + '_train_index_' + str(si) + '_' + str(cvi))]
                test_index = data_cuts[(Y_METRIC + '_test_index_' + str(si) + '_' +  str(cvi))]
        
                fold_Train = pd.DataFrame(dfnp[train_index],columns=all_cols)
                fold_Test = pd.DataFrame(dfnp[test_index],columns=all_cols)
            
                y_train = fold_Train[Y_METRIC].astype(float)
                y_test = fold_Test[Y_METRIC].astype(float)
            
                df_base_models = df_All_Base_Models.loc[df_All_Base_Models['Y_METRIC'] == Y_METRIC].copy()
            
                for base_model_name in df_base_models[base_model_name_cn].tolist():
                    
                    baseModel = df_base_models.loc[df_base_models[base_model_name_cn] == base_model_name].copy()
    
                    baseModelLookup[base_model_name] = baseModelIndx
                    
                    
                    # get the set of features used by the ensemble models
                    ls1 = ','.join(baseModel['Features'])
                    ls1 = ls1.split(",")
                    ls1 = getUniqueKeepOrder(ls1)
                    baseModel_feature_cols = sorted(ls1)
                
                    num_features_base_models += [len(baseModel_feature_cols)]
                    
            
                    X_train = fold_Train[baseModel_feature_cols].to_numpy()
                    X_test = fold_Test[baseModel_feature_cols].to_numpy()
                    
                    n_components = 1
                    pls_model = PLSRegression(n_components=n_components)
                    pls_model.fit(X_train, y_train)
                    y_pred_train = pls_model.predict(X_train)
                    y_pred_test = pls_model.predict(X_test)
    
                    y_pred_train_base_models += [y_pred_train]
                    y_pred_test_base_models += [y_pred_test]
    
                    y_train_model += [y_train]
                    y_test_model += [y_test]
                    
                    baseModelIndx += 1
                    

    
            # Now generate ensemble models for this cut of the data
    
            for nck_i in range(len(nck_list)):
                Y_METRIC = y_metric_list[nck_i]
                ensembleBaseModelList = list(nck_list[nck_i])
    
                cv_train_mse = ensembleModelMetrics[str(nck_i) + 'cv_train_mse']
                cv_test_mse = ensembleModelMetrics[str(nck_i) + 'cv_test_mse']
            
                cv_train_R2 = ensembleModelMetrics[str(nck_i) + 'cv_train_R2']
                cv_test_R2 = ensembleModelMetrics[str(nck_i) + 'cv_test_R2'] 
            
                cv_train_adj_R2 = ensembleModelMetrics[str(nck_i) + 'cv_train_adj_R2']
                cv_test_adj_R2 = ensembleModelMetrics[str(nck_i) + 'cv_test_adj_R2']
    
     
                
                ensembleModels = df_All_Base_Models.loc[df_All_Base_Models[base_model_name_cn].isin(ensembleBaseModelList)].copy()
                # get the set of features used by the ensemble models
                ls1 = ','.join(ensembleModels['Features'])
                ls1 = ls1.split(",")
                ls1 = getUniqueKeepOrder(ls1)
                ensemble_feature_cols = sorted(ls1)
    
                # Note that the y_train and y_test set are the same for each base model
                baseModelIndx = baseModelLookup[ensembleBaseModelList[0]]
                y_train = y_train_model[baseModelIndx]
                y_test = y_test_model[baseModelIndx]
                
                # generate results for ensemble model
                trainEnsemPredy = []
                testEnsemPredy = []
                for base_model_name in ensembleBaseModelList:
                    baseModelIndx = baseModelLookup[base_model_name]
    
                    y_pred_train = y_pred_train_base_models[baseModelIndx]
                    y_pred_test = y_pred_test_base_models[baseModelIndx]
                    trainEnsemPredy += [y_pred_train]
                    testEnsemPredy += [y_pred_test]
                trainEnsemPredy = np.mean(trainEnsemPredy, axis = 0)
                testEnsemPredy = np.mean(testEnsemPredy, axis = 0)
    
    
                train_R2 = r2_score(y_train, trainEnsemPredy)
                test_R2 = r2_score(y_test, testEnsemPredy) 
                
                cv_train_R2 += [train_R2]
                cv_test_R2 += [test_R2]
    
            
                numPredictors = len(ensemble_feature_cols)
                if (numObservations - numPredictors - 1) > 0:
                    adjR2Factor = (numObservations - 1) / (numObservations - numPredictors - 1)
                else:
                    adjR2Factor = numObservations # should not reach this point but just in case
                        
                train_adj_R2 = 1 - adjR2Factor * (1 - train_R2)
                test_adj_R2 = 1 - adjR2Factor * (1 - test_R2)
                
                cv_train_adj_R2 += [train_adj_R2]
                cv_test_adj_R2 += [test_adj_R2]
        
                train_mse = mean_squared_error(y_train, trainEnsemPredy)
                test_mse = mean_squared_error(y_test, testEnsemPredy)  
                
                cv_train_mse += [train_mse]
                cv_test_mse += [test_mse]  
    
                ensembleModelMetrics[str(nck_i) + 'cv_train_mse'] = cv_train_mse
                ensembleModelMetrics[str(nck_i) + 'cv_test_mse'] = cv_test_mse
            
                ensembleModelMetrics[str(nck_i) + 'cv_train_R2'] = cv_train_R2
                ensembleModelMetrics[str(nck_i) + 'cv_test_R2'] = cv_test_R2
            
                ensembleModelMetrics[str(nck_i) + 'cv_train_adj_R2'] = cv_train_adj_R2
                ensembleModelMetrics[str(nck_i) + 'cv_test_adj_R2'] = cv_test_adj_R2
    
    
    
    ensembleYmetrics = []
    ensembleBaseModels = []
    ensembleNumBaseModels = []
    ensembleNumUniqueFeatures = []
    ensembleUniqueFeatures = []
    ensemble_mean_train_mse = []
    ensemble_mean_test_mse = []
    ensemble_stDev_train_mse = []
    ensemble_stDev_test_mse = []
    ensemble_N_test_mse = []
    
    ensemble_mean_train_R2 = []
    ensemble_mean_test_R2 = []
    ensemble_stDev_train_R2 = []
    ensemble_stDev_test_R2 = []
    
    ensemble_mean_train_adj_R2 = []
    ensemble_mean_test_adj_R2 = []
    ensemble_stDev_train_adj_R2 = []
    ensemble_stDev_test_adj_R2 = []
    
    ensemble_fold_train_mse = []
    ensemble_fold_test_mse = []
    
    ensemble_fold_train_R2 = []
    ensemble_fold_test_R2 = []
    
    ensemble_fold_train_adj_R2 = []
    ensemble_fold_test_adj_R2 = []
    
    for nck_i in range(len(nck_list)):
        Y_METRIC = y_metric_list[nck_i]
        ensembleBaseModelList = list(nck_list[nck_i])

        cv_train_mse = ensembleModelMetrics[str(nck_i) + 'cv_train_mse']
        cv_test_mse = ensembleModelMetrics[str(nck_i) + 'cv_test_mse']
    
        cv_train_R2 = ensembleModelMetrics[str(nck_i) + 'cv_train_R2']
        cv_test_R2 = ensembleModelMetrics[str(nck_i) + 'cv_test_R2'] 
    
        cv_train_adj_R2 = ensembleModelMetrics[str(nck_i) + 'cv_train_adj_R2']
        cv_test_adj_R2 = ensembleModelMetrics[str(nck_i) + 'cv_test_adj_R2']
    
     
                
        ensembleModels = df_All_Base_Models.loc[df_All_Base_Models[base_model_name_cn].isin(ensembleBaseModelList)].copy()
        # get the set of features used by the ensemble models
        ls1 = ','.join(ensembleModels['Features'])
        ls1 = ls1.split(",")
        ls1 = getUniqueKeepOrder(ls1)
        ensemble_feature_cols = sorted(ls1)

    
        # compute average and standard deviation of each of the cv data cuts=
        ensembleYmetrics += [Y_METRIC]
        ensembleBaseModels += [ensembleBaseModelList]
        ensembleNumBaseModels += [len(ensembleBaseModelList)]
        ensembleNumUniqueFeatures += [len(ensemble_feature_cols)]
        ensembleUniqueFeatures += [ensemble_feature_cols]
        
        ensemble_mean_train_mse += [np.mean(cv_train_mse)]
        ensemble_mean_test_mse += [np.mean(cv_test_mse)]
        ensemble_stDev_train_mse += [np.std(cv_train_mse, ddof=1)]
        ensemble_stDev_test_mse += [np.std(cv_test_mse, ddof=1)]
        ensemble_N_test_mse += [len(cv_test_mse)]
    
        ensemble_mean_train_R2 += [np.mean(cv_train_R2)]
        ensemble_mean_test_R2 += [np.mean(cv_test_R2)]
        ensemble_stDev_train_R2 += [np.std(cv_train_R2, ddof=1)]
        ensemble_stDev_test_R2 += [np.std(cv_test_R2, ddof=1)]
    
        ensemble_mean_train_adj_R2 += [np.mean(cv_train_adj_R2)]
        ensemble_mean_test_adj_R2 += [np.mean(cv_test_adj_R2)]
        ensemble_stDev_train_adj_R2 += [np.std(cv_train_adj_R2, ddof=1)]
        ensemble_stDev_test_adj_R2 += [np.std(cv_test_adj_R2, ddof=1)]
    
    
        # now train the ensemble model with the folds training data and test it using the folds test data
    
        fold_Train = df_train_fold_imputed.copy()
        fold_Test = df_test_fold_imputed.copy()
    
        y_train = fold_Train[Y_METRIC].astype(float)
        y_test = fold_Test[Y_METRIC].astype(float)

        trainEnsemPredy = []
        testEnsemPredy = []
        for ind in ensembleModels.index:
            EA_model_features = getUniqueKeepOrder((ensembleModels['Features'][ind].split(",")))
    
            X_train = fold_Train[EA_model_features].to_numpy()
            X_test = fold_Test[EA_model_features].to_numpy()
            
            n_components = 1
            pls_model = PLSRegression(n_components=n_components)
            pls_model.fit(X_train, y_train)
            y_pred_train = pls_model.predict(X_train)
            y_pred_test = pls_model.predict(X_test)
            
            trainEnsemPredy += [y_pred_train]
            testEnsemPredy += [y_pred_test]
        
        trainEnsemPredy = np.mean(trainEnsemPredy, axis = 0)
        testEnsemPredy = np.mean(testEnsemPredy, axis = 0)
    
        train_R2 = r2_score(y_train, trainEnsemPredy)
        test_R2 = r2_score(y_test, testEnsemPredy) 
        
        ensemble_fold_train_R2 += [train_R2]
        ensemble_fold_test_R2 += [test_R2]

        numPredictors = len(ensemble_feature_cols)
        if (numObservations - numPredictors - 1) > 0:
            adjR2Factor = (numObservations - 1) / (numObservations - numPredictors - 1)
        else:
            adjR2Factor = numObservations # should not reach this point but just in case
                
        train_adj_R2 = 1 - adjR2Factor * (1 - train_R2)
        test_adj_R2 = 1 - adjR2Factor * (1 - test_R2)
        
        ensemble_fold_train_adj_R2 += [train_adj_R2]
        ensemble_fold_test_adj_R2 += [test_adj_R2]

        
        train_mse = mean_squared_error(y_train, trainEnsemPredy)
        test_mse = mean_squared_error(y_test, testEnsemPredy) 
        
        ensemble_fold_train_mse += [train_mse]
        ensemble_fold_test_mse += [test_mse]
    
    
    df_ensemble_models = pd.DataFrame(list(zip(ensembleYmetrics, ensembleBaseModels, ensembleNumBaseModels, ensembleNumUniqueFeatures, ensembleUniqueFeatures, ensemble_mean_train_mse, ensemble_stDev_train_mse, ensemble_mean_test_mse, ensemble_stDev_test_mse, ensemble_mean_train_R2,ensemble_stDev_train_R2, ensemble_mean_test_R2, ensemble_stDev_test_R2, ensemble_mean_train_adj_R2,ensemble_stDev_train_adj_R2, ensemble_mean_test_adj_R2, ensemble_stDev_test_adj_R2, ensemble_N_test_mse, ensemble_fold_train_mse, ensemble_fold_test_mse, ensemble_fold_train_R2, ensemble_fold_test_R2, ensemble_fold_train_adj_R2, ensemble_fold_test_adj_R2)), columns = ['Y_METRIC','ensemble', 'NumBaseModels', 'NumUniqueFeatures', 'Features', 'mean_cv_train_mse', 'stDev_cv_train_mse', 'mean_cv_test_mse', 'stDev_cv_test_mse',  'mean_cv_train_R2', 'stDev_cv_train_R2', 'mean_cv_test_R2', 'stDev_cv_test_R2', 'mean_cv_train_adj_R2', 'stDev_cv_train_adj_R2', 'mean_cv_test_adj_R2', 'stDev_cv_test_adj_R2', 'N_cv_tests', 'fold_train_mse', 'fold_test_mse', 'fold_train_R2', 'fold_test_R2', 'fold_train_adj_R2', 'fold_test_adj_R2'])

    return (df_ensemble_models)

#################################################################################################################################################################
#####
#################################################################################################################################################################


def _evaluate_ACFT_EnsembleModels(params):

    
    job_id = params['job_id']  
    proc_id = params['proc_id']
    Y_METRIC = params['Y_METRIC']
    df_ACFT_tables = params['df_ACFT_tables']   
    df_base_models = params['df_base_models']   
    base_model_name_cn = params['base_model_name_cn']  
    df_ensemble_models = params['df_ensemble_models']   
    df_model_data_raw = params['df_model_data_raw']
    df_model_feature_info = params['df_model_feature_info']
    all_cols = params['all_cols']
    info_cols = params['info_cols']
    num_cv_folds = params['num_cv_folds']
    knn_num_neighbors = params['knn_num_neighbors']
    numFoldGenerations = params['numFoldGenerations']
    randomState = params['randomState']
    models_dir = params['models_dir']
    index_list = params['sub_index_list']

    start_time = datetime.datetime.now()
    
    print('Started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ACFT ensemble EA model cv performance.')
    

    # Remove subjects that do not have a measured outcome
    df_model_data_raw = df_model_data_raw[df_model_data_raw[Y_METRIC].notna()]

    numObservations = len(df_model_data_raw)

    ACFT_METRIC_LIST = ['ACFT_Maximum_Deadlift', 'ACFT_Standing_Power_Throw', 'ACFT_Hand_Release_Pushups', 'ACFT_Sprint_Drag_Carry', 'ACFT_Leg_Tuck_OR_Plank', 'ACFT_2_Mile_Run']
    
    tcns = info_cols.copy() 
    tcns += ACFT_METRIC_LIST
    tcns = getUniqueKeepOrder(tcns)
    # make sure local copy of info_cols contains all of the latent metrics
    info_cols = tcns


    
    ensemble_models_train_mse_mean = []
    ensemble_models_train_mse_stdev = []
    ensemble_models_test_mse_mean = []
    ensemble_models_test_mse_stdev = []
    ensemble_models_train_R2_mean = []
    ensemble_models_train_R2_stdev = []
    ensemble_models_test_R2_mean = []
    ensemble_models_test_R2_stdev = []

    ensemble_models_train_adj_R2_mean = []
    ensemble_models_train_adj_R2_stdev = []
    ensemble_models_test_adj_R2_mean = []
    ensemble_models_test_adj_R2_stdev = []

    ensemble_models_N = []
    
    ensemble_all_train_R2 = []
    ensemble_all_train_adj_R2 = []
    ensemble_all_train_mse = []
    
    for ind in index_list:

        np.random.seed(randomState)
        
        ensembleParm = {}
        
        ensembleModel = df_ensemble_models['ensemble'][ind]
        ls1 = ensembleModel.replace('[','').split("],")
        for mi in range(len(ACFT_METRIC_LIST)):
            ensemble_base_model_list = getUniqueKeepOrder(ls1[mi].replace(' ','').replace('[','').replace(']','').replace("'",'').split(","))
            ensembleParm = { **ensembleParm, str(mi) + '_ensemble_base_model_list': ensemble_base_model_list}

        # get the set of features used by the ensemble models
        ls1 = df_ensemble_models['Features'][ind].replace(' ','').replace('[','').replace(']','').replace("'",'')
        ls1 = ls1.split(",")
        ls1 = getUniqueKeepOrder(ls1)
        feature_cols = sorted(ls1)

        ensemble_feature_cols = feature_cols.copy()
        numPredictors = len(ensemble_feature_cols)
        
        ensembleModel_cols = info_cols + feature_cols
    
        
        df = df_model_data_raw[ensembleModel_cols].copy()
        
        # Remove subjects that do not have a measured outcome
        df = df[df[Y_METRIC].notna()]
        
        # Remove subjects that are missing more than 40% of their feature measures
        critNumMissingFeatures = 0.4 * len(feature_cols)
        df = df.drop(df[df[feature_cols].isna().sum(axis=1) > critNumMissingFeatures].index)
    
    
        df = df.reset_index(drop = True)
        dfnp = df.to_numpy()
        
        X = df[feature_cols].to_numpy()
        X = X.astype(float)
    
        res = np.unique(df[Y_METRIC])
        if len(res)==2:
            ybin = np.array(df[Y_METRIC], dtype=int)
        else:
            medianValue = df[Y_METRIC].median()
            meanValue = df[Y_METRIC].mean()
            v = df[Y_METRIC].tolist()
            if meanValue <= medianValue:
                ybin = list(map(lambda x: 1 if x >= medianValue else 0, v))
            else:
                ybin = list(map(lambda x: 0 if x <= medianValue else 1, v))
            ybin = np.array(ybin, dtype=int)
        

        cv_train_mse = []
        cv_test_mse = []
        cv_train_R2 = []
        cv_test_R2 = []
        cv_train_adj_R2 = []
        cv_test_adj_R2 = []
        
        for si in range(numFoldGenerations):

            rsv = (randomState + si)
            
            skf = StratifiedKFold(n_splits=num_cv_folds,shuffle=True, random_state = rsv)
            skf.get_n_splits(X, ybin)
            
            for train_index, test_index in skf.split(X, ybin):
                
                df_train_fold = pd.DataFrame(dfnp[train_index],columns=ensembleModel_cols)
                # inpute the training data by itself and excluding the test data
                dfimpute = pd.DataFrame(impute_data_knn(df_train_fold[feature_cols], n_neighbors=knn_num_neighbors), columns=feature_cols)
                df_train_fold_info = df_train_fold[info_cols].copy()
                df_train_fold_info = df_train_fold_info.reset_index(drop = True)
                dfimpute = dfimpute.reset_index(drop = True)
                df_train_fold_imputed = pd.concat([df_train_fold_info,dfimpute],axis=1)
            
                df_test_fold = pd.DataFrame(dfnp[test_index],columns=ensembleModel_cols)
                # inpute the test data
                # append the imputed training data with the raw test data
                df_test_with_train_fold = pd.concat([df_test_fold,df_train_fold_imputed[ensembleModel_cols].copy()],axis=0)
                dfimpute = pd.DataFrame(impute_data_knn(df_test_with_train_fold[feature_cols], n_neighbors=knn_num_neighbors), columns=feature_cols)
                # now only keep the top rows that are the test subjects
                dfimpute = dfimpute.iloc[:len(test_index)]
                df_test_fold_info = df_test_fold[info_cols].copy()
                df_test_fold_info = df_test_fold_info.reset_index(drop = True)
                dfimpute = dfimpute.reset_index(drop = True)
                df_test_fold_imputed = pd.concat([df_test_fold_info,dfimpute],axis=1)
            
                fold_Train = df_train_fold_imputed.copy()
                fold_Test = df_test_fold_imputed.copy()
    
                # y_train = fold_Train[Y_METRIC].astype(float)
                # y_test = fold_Test[Y_METRIC].astype(float)

                y_pred_train_scores_base_models = []
                y_pred_test_scores_base_models = []
                
                for mi in range(len(ACFT_METRIC_LIST)):

                    ACFT_LATENT_METRIC = ACFT_METRIC_LIST[mi]
            
                    ensemble_base_model_list = ensembleParm[str(mi) + '_ensemble_base_model_list']
                    ensembleModels = df_base_models.loc[df_base_models[base_model_name_cn].isin(ensemble_base_model_list)].copy()


                    y_pred_train_base_models = []
                    y_pred_test_base_models = []
        
                    y_train = fold_Train[ACFT_LATENT_METRIC].astype(float)
                    y_test = fold_Test[ACFT_LATENT_METRIC].astype(float)
                    
                    for ind in ensembleModels.index:
                        EA_model_features = getUniqueKeepOrder((ensembleModels['Features'][ind].split(",")))

                        X_train = fold_Train[EA_model_features].to_numpy()
                        X_test = fold_Test[EA_model_features].to_numpy()
                        
                        n_components = 1
                        pls_model = PLSRegression(n_components=n_components)
                        pls_model.fit(X_train, y_train)
                        y_pred_train = pls_model.predict(X_train)
                        y_pred_test = pls_model.predict(X_test)
                        
                        y_pred_train_base_models += [y_pred_train]
                        y_pred_test_base_models += [y_pred_test]
        
                    y_pred_train = np.mean(y_pred_train_base_models, axis = 0)
                    y_pred_test = np.mean(y_pred_test_base_models, axis = 0)

          
                    
                    y_pred_train_scores = [compute_ACFT_Score(df_ACFT_tables, ACFT_LATENT_METRIC, v) for v in y_pred_train]
                    y_pred_test_scores = [compute_ACFT_Score(df_ACFT_tables, ACFT_LATENT_METRIC, v) for v in y_pred_test]
                    
                    y_pred_train_scores_base_models += [y_pred_train_scores]
                    y_pred_test_scores_base_models += [y_pred_test_scores]
   
                    # ensembleParm[str(mi) + '_y_pred_train'] = y_pred_train
                    # ensembleParm[str(mi) + '_y_pred_test'] = y_pred_test
                    
                    # ensembleParm[str(mi) + '_y_pred_train_scores'] = y_pred_train_scores
                    # ensembleParm[str(mi) + '_y_pred_test_scores'] = y_pred_test_scores
                    
                ### For ACFT we sum the scores of the latent base models and NOT the mean!!!!
                y_pred_train_ensemble_model = np.sum(y_pred_train_scores_base_models, axis = 0)
                y_pred_test_ensemble_model = np.sum(y_pred_test_scores_base_models, axis = 0)
    
                y_train = fold_Train[Y_METRIC].astype(float)
                y_test = fold_Test[Y_METRIC].astype(float)
    
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    
                train_R2 = r2_score(y_train, y_pred_train_ensemble_model)
                test_R2 = r2_score(y_test, y_pred_test_ensemble_model) 
                
                cv_train_R2 += [train_R2]
                cv_test_R2 += [test_R2]

                numPredictors = len(ensemble_feature_cols)
                if (numObservations - numPredictors - 1) > 0:
                    adjR2Factor = (numObservations - 1) / (numObservations - numPredictors - 1)
                else:
                    adjR2Factor = numObservations # should not reach this point but just in case
                        
                train_adj_R2 = 1 - adjR2Factor * (1 - train_R2)
                test_adj_R2 = 1 - adjR2Factor * (1 - test_R2)
                
                cv_train_adj_R2 += [train_adj_R2]
                cv_test_adj_R2 += [test_adj_R2]

                
                train_mse = mean_squared_error(y_train, y_pred_train_ensemble_model)
                test_mse = mean_squared_error(y_test, y_pred_test_ensemble_model)  
                
                cv_train_mse += [train_mse]
                cv_test_mse += [test_mse]      
                
    
        
        ensemble_models_train_mse_mean += [np.mean(cv_train_mse)]
        ensemble_models_train_mse_stdev += [np.std(cv_train_mse, ddof=1)]
        ensemble_models_test_mse_mean += [np.mean(cv_test_mse)]
        ensemble_models_test_mse_stdev += [np.std(cv_test_mse, ddof=1)]
        ensemble_models_train_R2_mean += [np.mean(cv_train_R2)]
        ensemble_models_train_R2_stdev += [np.std(cv_train_R2, ddof=1)]
        ensemble_models_test_R2_mean += [np.mean(cv_test_R2)]
        ensemble_models_test_R2_stdev += [np.std(cv_test_R2, ddof=1)]

        ensemble_models_train_adj_R2_mean += [np.mean(cv_train_adj_R2)]
        ensemble_models_train_adj_R2_stdev += [np.std(cv_train_adj_R2, ddof=1)]
        ensemble_models_test_adj_R2_mean += [np.mean(cv_test_adj_R2)]
        ensemble_models_test_adj_R2_stdev += [np.std(cv_test_adj_R2, ddof=1)]
        
        ensemble_models_N += [len(cv_test_R2)]
    

    
    
        ####################################################
        # now train the ensemble model using all of the data
        ####################################################
        
        df = df_model_data_raw[ensembleModel_cols].copy()
    
        # Remove subjects that do not have a measured outcome
        df = df[df[Y_METRIC].notna()]
        
        # Remove subjects that are missing more than 40% of their feature measures
        critNumMissingFeatures = 0.4 * len(feature_cols)
        df = df.drop(df[df[feature_cols].isna().sum(axis=1) > critNumMissingFeatures].index)
    
        df = df.reset_index(drop = True)
        dfnp = df.to_numpy()
    
        df_train_fold = df_model_data_raw[ensembleModel_cols]
        
        dfimpute = pd.DataFrame(impute_data_knn(df_train_fold[feature_cols], n_neighbors=knn_num_neighbors), columns=feature_cols)
        df_train_fold_info = df_train_fold[info_cols].copy()
        df_train_fold_info = df_train_fold_info.reset_index(drop = True)
        dfimpute = dfimpute.reset_index(drop = True)
        df_train_fold_imputed = pd.concat([df_train_fold_info,dfimpute],axis=1)
    
        fold_Train = df_train_fold_imputed.copy()
    
        # y_train = fold_Train[Y_METRIC].astype(float)
    
        y_pred_train_scores_base_models = []
        # y_pred_test_scores_base_models = []
        
        for mi in range(len(ACFT_METRIC_LIST)):

            ACFT_LATENT_METRIC = ACFT_METRIC_LIST[mi]
    
            ensemble_base_model_list = ensembleParm[str(mi) + '_ensemble_base_model_list']
            ensembleModels = df_base_models.loc[df_base_models[base_model_name_cn].isin(ensemble_base_model_list)].copy()


            y_pred_train_base_models = []
            # y_pred_test_base_models = []

            y_train = fold_Train[ACFT_LATENT_METRIC].astype(float)
            # y_test = fold_Test[ACFT_LATENT_METRIC].astype(float)
            
            for ind in ensembleModels.index:
                EA_model_features = getUniqueKeepOrder((ensembleModels['Features'][ind].split(",")))
    
                X_train = fold_Train[EA_model_features].to_numpy()
                # X_test = fold_Test[EA_model_features].to_numpy()
                
                n_components = 1
                pls_model = PLSRegression(n_components=n_components)
                pls_model.fit(X_train, y_train)
                y_pred_train = pls_model.predict(X_train)
                # y_pred_test = pls_model.predict(X_test)
                
                y_pred_train_base_models += [y_pred_train]
                # y_pred_test_base_models += [y_pred_test]

            y_pred_train = np.mean(y_pred_train_base_models, axis = 0)
            # y_pred_test = np.mean(y_pred_test_base_models, axis = 0)

            y_pred_train_scores = [compute_ACFT_Score(df_ACFT_tables, ACFT_LATENT_METRIC, v) for v in y_pred_train]
            # y_pred_test_scores = [compute_ACFT_Score(df_ACFT_tables, ACFT_LATENT_METRIC, v) for v in y_pred_test]
            
            y_pred_train_scores_base_models += [y_pred_train_scores]
            # y_pred_test_scores_base_models += [y_pred_test_scores]

            # ensembleParm[str(mi) + '_y_pred_train'] = y_pred_train
            # ensembleParm[str(mi) + '_y_pred_test'] = y_pred_test
            
            # ensembleParm[str(mi) + '_y_pred_train_scores'] = y_pred_train_scores
            # ensembleParm[str(mi) + '_y_pred_test_scores'] = y_pred_test_scores
            
        ### For ACFT we sum the scores of the latent base models and NOT the mean!!!!
        y_pred_train_ensemble_model = np.sum(y_pred_train_scores_base_models, axis = 0)
        # y_pred_test_ensemble_model = np.sum(y_pred_test_scores_base_models, axis = 0)

        y_train = fold_Train[Y_METRIC].astype(float)
        
        train_R2 = r2_score(y_train, y_pred_train_ensemble_model)
        ensemble_all_train_R2 += [train_R2]

        numPredictors = len(ensemble_feature_cols)
        if (numObservations - numPredictors - 1) > 0:
            adjR2Factor = (numObservations - 1) / (numObservations - numPredictors - 1)
        else:
            adjR2Factor = numObservations # should not reach this point but just in case
                
        train_adj_R2 = 1 - adjR2Factor * (1 - train_R2)
        
        ensemble_all_train_adj_R2 += [train_adj_R2]

        train_mse = mean_squared_error(y_train, y_pred_train_ensemble_model)
        ensemble_all_train_mse += [train_mse]
    

    df_ensemble_models = df_ensemble_models.iloc[index_list].copy()
        
    df_ensemble_models['all_cv_train_mse_mean'] = ensemble_models_train_mse_mean
    df_ensemble_models['all_cv_train_mse_stdev'] = ensemble_models_train_mse_stdev
    df_ensemble_models['all_cv_test_mse_mean'] = ensemble_models_test_mse_mean
    df_ensemble_models['all_cv_test_mse_stdev'] = ensemble_models_test_mse_stdev
    df_ensemble_models['all_cv_train_R2_mean'] = ensemble_models_train_R2_mean
    df_ensemble_models['all_cv_train_R2_stdev'] = ensemble_models_train_R2_stdev
    df_ensemble_models['all_cv_test_R2_mean'] = ensemble_models_test_R2_mean
    df_ensemble_models['all_cv_test_R2_stdev'] = ensemble_models_test_R2_stdev
    
    df_ensemble_models['all_cv_train_adj_R2_mean'] = ensemble_models_train_adj_R2_mean
    df_ensemble_models['all_cv_train_adj_R2_stdev'] = ensemble_models_train_adj_R2_stdev
    df_ensemble_models['all_cv_test_adj_R2_mean'] = ensemble_models_test_adj_R2_mean
    df_ensemble_models['all_cv_test_adj_R2_stdev'] = ensemble_models_test_adj_R2_stdev

    df_ensemble_models['all_cv_N'] = ensemble_models_N
    df_ensemble_models['ratio_all_cv_test_over_train'] = df_ensemble_models['all_cv_test_mse_mean'] / df_ensemble_models['all_cv_train_mse_mean']
    
    df_ensemble_models['all_train_mse'] = ensemble_all_train_mse
    df_ensemble_models['all_train_R2'] = ensemble_all_train_R2
    df_ensemble_models['all_train_adj_R2'] = ensemble_all_train_adj_R2
    
    return df_ensemble_models

#################################################################################################################################################################
#####
#################################################################################################################################################################

def evaluate_ACFT_EnsembleModels(params):


    job_id = '1'

    index_list = params['index_list']
    
    num_runs = len(index_list)
    num_tasks = int(params['max_num_cpus'])
    runs_per_task = num_runs // num_tasks + 1 if num_runs % num_tasks else num_runs // num_tasks

    if 'minNumRunsPerTask' in params:
        minNumRunsPerTask = params['minNumRunsPerTask']
    else:
        minNumRunsPerTask = 1
        
    if num_tasks > 1 and runs_per_task < minNumRunsPerTask:
        num_tasks = round(num_runs / minNumRunsPerTask)
        if num_tasks < 1:
            num_tasks = 1
        elif num_tasks > int(params['max_num_cpus']):
            num_tasks = int(params['max_num_cpus'])
        runs_per_task = num_runs // num_tasks + 1 if num_runs % num_tasks else num_runs // num_tasks


    params = { **params, 'job_id': job_id}

    n_request_cpus = num_tasks
        
    if num_tasks == 1:
        proc_id = 0
        sub_index_list = index_list[(proc_id*runs_per_task):min(num_runs, (proc_id+1)*runs_per_task)]
        
        params = { **params, 'proc_id': proc_id}
        params = { **params, 'sub_index_list': sub_index_list}   

        # print("num runs to process in series:", num_runs,"cpu processors requested:", n_request_cpus)
        
        # print('Started processing _evaluate_ACFT_EnsembleModels',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    
        df_ensemble_models = _evaluate_ACFT_EnsembleModels(params)

        # print('Completed processing of _evaluate_ACFT_EnsembleModels',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    else:
      
        param_sets = []
        for proc_id in range(num_tasks):
          
          sub_index_list = index_list[(proc_id*runs_per_task):min(num_runs, (proc_id+1)*runs_per_task)]
          if (len(sub_index_list) > 0):  
              # make a copy of the parameter set and then overwrite it to customize
              params_copy = params.copy()
              params_copy = { **params_copy, 'proc_id': proc_id}
              params_copy = { **params_copy, 'sub_index_list': sub_index_list}   
    
              param_sets.append(params_copy)

        
        n_request_cpus = len(param_sets)
        
        # print('Started multiprocessing of _evaluate_ACFT_EnsembleModels',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        print('# tasks/processors: ' + str(num_runs) + '/' + str(n_request_cpus))

        pool = multiprocessing.Pool(n_request_cpus)
        results = pool.map(_evaluate_ACFT_EnsembleModels, param_sets)

        # print('Completed multiprocessing of _evaluate_ACFT_EnsembleModels',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        df_ensemble_models = pd.concat(results)
        df_ensemble_models = df_ensemble_models.reset_index(drop = True)

    return df_ensemble_models

#################################################################################################################################################################
#####
#################################################################################################################################################################

def _evaluate_EnsembleModels(params):

    
    import datetime
    
    job_id = params['job_id']  
    proc_id = params['proc_id']
    Y_METRIC = params['Y_METRIC']

    if 'Y_BIN_METRIC' in params.keys():
        Y_BIN_METRIC = params['Y_BIN_METRIC']
    else:
        Y_BIN_METRIC = params['Y_METRIC']

        
    if 'Compute_ACFT_Score' in params.keys():
        apply_ACFT_scoring = params['Compute_ACFT_Score']
    else:
        apply_ACFT_scoring = False
    if 'df_ACFT_tables' in params.keys():
        df_ACFT_tables = params['df_ACFT_tables']  
    
    df_base_models = params['df_base_models']   
    base_model_name_cn = params['base_model_name_cn']  
    df_ensemble_models = params['df_ensemble_models']   
    df_model_data_raw = params['df_model_data_raw']
    df_model_feature_info = params['df_model_feature_info']
    all_cols = params['all_cols']
    info_cols = params['info_cols']
    num_cv_folds = params['num_cv_folds']
    knn_num_neighbors = params['knn_num_neighbors']
    numFoldGenerations = params['numFoldGenerations']
    randomState = params['randomState']
    models_dir = params['models_dir']
    index_list = params['sub_index_list']

    start_time = datetime.datetime.now()
    
    print('Started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ensemble EA model cv performance.')


    # make sure local version of info_cols contains the Y_METRIC
    info_cols += [Y_METRIC]
    info_cols += [Y_BIN_METRIC]
    info_cols = getUniqueKeepOrder((info_cols))

    # Remove subjects that do not have a measured outcome
    df_model_data_raw = df_model_data_raw[df_model_data_raw[Y_METRIC].notna()].copy()
    
    numObservations = len(df_model_data_raw)


    ensemble_models_train_mse_mean = []
    ensemble_models_train_mse_stdev = []
    ensemble_models_test_mse_mean = []
    ensemble_models_test_mse_stdev = []
    ensemble_models_train_R2_mean = []
    ensemble_models_train_R2_stdev = []
    ensemble_models_test_R2_mean = []
    ensemble_models_test_R2_stdev = []

    ensemble_models_train_adj_R2_mean = []
    ensemble_models_train_adj_R2_stdev = []
    ensemble_models_test_adj_R2_mean = []
    ensemble_models_test_adj_R2_stdev = []

    ensemble_models_N = []
    
    ensemble_all_train_R2 = []
    ensemble_all_train_adj_R2 = []
    ensemble_all_train_mse = []
    
    for ind in index_list:

        # when evaluating each model, start from the same random state
        
        np.random.seed(randomState)
        
        ensembleParm = {}
        
        ensembleModel = df_ensemble_models['ensemble'][ind]
        ensemble_base_model_list = getUniqueKeepOrder(ensembleModel.replace(' ','').replace('[','').replace(']','').replace("'",'').split(","))

        ensembleModels = df_base_models.loc[df_base_models[base_model_name_cn].isin(ensemble_base_model_list)].copy()
        
        # get the set of features used by the ensemble models
        
        ls1 = df_ensemble_models['Features'][ind].replace(' ','').replace('[','').replace(']','').replace("'",'')
        # ls1 = ','.join(ensembleModels['Features'])
        ls1 = ls1.split(",")
        ls1 = getUniqueKeepOrder(ls1)
        feature_cols = sorted(ls1)

        ensemble_feature_cols = feature_cols.copy()
        numPredictors = len(ensemble_feature_cols)
        
        
        ensembleModel_cols = info_cols + feature_cols
        
        df = df_model_data_raw[ensembleModel_cols].copy()
        
        # Remove subjects that do not have a measured bin metric
        df = df[df[Y_BIN_METRIC].notna()]
        
        # Remove subjects that are missing more than 40% of their feature measures
        critNumMissingFeatures = 0.4 * len(feature_cols)
        df = df.drop(df[df[feature_cols].isna().sum(axis=1) > critNumMissingFeatures].index)
    
    
    
        df = df.reset_index(drop = True)
        dfnp = df.to_numpy()
        
        X = df[feature_cols].to_numpy()
        X = X.astype(float)
    
        res = np.unique(df[Y_BIN_METRIC])
        if len(res)==2:
            ybin = np.array(df[Y_BIN_METRIC], dtype=int)
        else:
            medianValue = df[Y_BIN_METRIC].median()
            meanValue = df[Y_BIN_METRIC].mean()
            v = df[Y_BIN_METRIC].tolist()
            if meanValue <= medianValue:
                ybin = list(map(lambda x: 1 if x >= medianValue else 0, v))
            else:
                ybin = list(map(lambda x: 0 if x <= medianValue else 1, v))
            ybin = np.array(ybin, dtype=int)
        
    
    
            
        cv_train_mse = []
        cv_test_mse = []
        cv_train_R2 = []
        cv_test_R2 = []
        cv_train_adj_R2 = []
        cv_test_adj_R2 = []
        
        for si in range(numFoldGenerations):
            rsv = (randomState + si)
            
            skf = StratifiedKFold(n_splits=num_cv_folds,shuffle=True, random_state = rsv)
            skf.get_n_splits(X, ybin)
            
            for train_index, test_index in skf.split(X, ybin):
                
                df_train_fold = pd.DataFrame(dfnp[train_index],columns=ensembleModel_cols)
                # inpute the training data by itself and excluding the test data
                dfimpute = pd.DataFrame(impute_data_knn(df_train_fold[feature_cols], n_neighbors=knn_num_neighbors), columns=feature_cols)
                df_train_fold_info = df_train_fold[info_cols].copy()
                df_train_fold_info = df_train_fold_info.reset_index(drop = True)
                dfimpute = dfimpute.reset_index(drop = True)
                df_train_fold_imputed = pd.concat([df_train_fold_info,dfimpute],axis=1)
            
                
                df_test_fold = pd.DataFrame(dfnp[test_index],columns=ensembleModel_cols)
                # inpute the test data
                # append the imputed training data with the raw test data
                df_test_with_train_fold = pd.concat([df_test_fold,df_train_fold_imputed[ensembleModel_cols].copy()],axis=0)
                dfimpute = pd.DataFrame(impute_data_knn(df_test_with_train_fold[feature_cols], n_neighbors=knn_num_neighbors), columns=feature_cols)
                # now only keep the top rows that are the test subjects
                dfimpute = dfimpute.iloc[:len(test_index)]
                df_test_fold_info = df_test_fold[info_cols].copy()
                df_test_fold_info = df_test_fold_info.reset_index(drop = True)
                dfimpute = dfimpute.reset_index(drop = True)
                df_test_fold_imputed = pd.concat([df_test_fold_info,dfimpute],axis=1)
            
                
            
                fold_Train = df_train_fold_imputed.copy()
                fold_Test = df_test_fold_imputed.copy()
    
         
                
                y_train = fold_Train[Y_METRIC].astype(float)
                y_test = fold_Test[Y_METRIC].astype(float)
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    
    
    
                ensembleModels = df_base_models.loc[df_base_models[base_model_name_cn].isin(ensemble_base_model_list)].copy()
    
                y_pred_train_base_models = []
                y_pred_test_base_models = []
    
                y_train = fold_Train[Y_METRIC].astype(float)
                y_test = fold_Test[Y_METRIC].astype(float)
                
                for ind in ensembleModels.index:
                    EA_model_features = getUniqueKeepOrder((ensembleModels['Features'][ind].split(",")))
        
                    X_train = fold_Train[EA_model_features].to_numpy()
                    X_test = fold_Test[EA_model_features].to_numpy()
                    
                    n_components = 1
                    pls_model = PLSRegression(n_components=n_components)
                    pls_model.fit(X_train, y_train)
                    y_pred_train = pls_model.predict(X_train)
                    y_pred_test = pls_model.predict(X_test)
                    
                    y_pred_train_base_models += [y_pred_train]
                    y_pred_test_base_models += [y_pred_test]
    
                y_pred_train_ensemble_model = np.mean(y_pred_train_base_models, axis = 0)
                y_pred_test_ensemble_model = np.mean(y_pred_test_base_models, axis = 0)
    
    
                y_train = fold_Train[Y_METRIC].astype(float)
                y_test = fold_Test[Y_METRIC].astype(float)
    
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

                if apply_ACFT_scoring:
                    if Y_METRIC in ['ACFT_Maximum_Deadlift', 'ACFT_Standing_Power_Throw', 'ACFT_Hand_Release_Pushups', 'ACFT_Sprint_Drag_Carry', 'ACFT_Leg_Tuck_OR_Plank', 'ACFT_2_Mile_Run']:
                        y_train = [compute_ACFT_Score(df_ACFT_tables, Y_METRIC, v) for v in y_train]
                        y_test = [compute_ACFT_Score(df_ACFT_tables, Y_METRIC, v) for v in y_test]
                        y_pred_train_ensemble_model = [compute_ACFT_Score(df_ACFT_tables, Y_METRIC, v) for v in y_pred_train_ensemble_model]
                        y_pred_test_ensemble_model = [compute_ACFT_Score(df_ACFT_tables, Y_METRIC, v) for v in y_pred_test_ensemble_model]

        
                train_R2 = r2_score(y_train, y_pred_train_ensemble_model)
                test_R2 = r2_score(y_test, y_pred_test_ensemble_model) 
                
                cv_train_R2 += [train_R2]
                cv_test_R2 += [test_R2]

                numPredictors = len(ensemble_feature_cols)
                if (numObservations - numPredictors - 1) > 0:
                    adjR2Factor = (numObservations - 1) / (numObservations - numPredictors - 1)
                else:
                    adjR2Factor = numObservations # should not reach this point but just in case
                        
                train_adj_R2 = 1 - adjR2Factor * (1 - train_R2)
                test_adj_R2 = 1 - adjR2Factor * (1 - test_R2)
                
                cv_train_adj_R2 += [train_adj_R2]
                cv_test_adj_R2 += [test_adj_R2]

                
                train_mse = mean_squared_error(y_train, y_pred_train_ensemble_model)
                test_mse = mean_squared_error(y_test, y_pred_test_ensemble_model)  
                
                cv_train_mse += [train_mse]
                cv_test_mse += [test_mse]      
                
    
        
        ensemble_models_train_mse_mean += [np.mean(cv_train_mse)]
        ensemble_models_train_mse_stdev += [np.std(cv_train_mse, ddof=1)]
        ensemble_models_test_mse_mean += [np.mean(cv_test_mse)]
        ensemble_models_test_mse_stdev += [np.std(cv_test_mse, ddof=1)]
        
        ensemble_models_train_R2_mean += [np.mean(cv_train_R2)]
        ensemble_models_train_R2_stdev += [np.std(cv_train_R2, ddof=1)]
        ensemble_models_test_R2_mean += [np.mean(cv_test_R2)]
        ensemble_models_test_R2_stdev += [np.std(cv_test_R2, ddof=1)]

        ensemble_models_train_adj_R2_mean += [np.mean(cv_train_adj_R2)]
        ensemble_models_train_adj_R2_stdev += [np.std(cv_train_adj_R2, ddof=1)]
        ensemble_models_test_adj_R2_mean += [np.mean(cv_test_adj_R2)]
        ensemble_models_test_adj_R2_stdev += [np.std(cv_test_adj_R2, ddof=1)]

        ensemble_models_N += [len(cv_test_R2)]
    

    
    
        ####################################################
        # now train the ensemble model using all of the data
        ####################################################
        
        df = df_model_data_raw[ensembleModel_cols].copy()
    
        # Remove subjects that do not have a measured outcome
        df = df[df[Y_METRIC].notna()]
        
        # Remove subjects that are missing more than 40% of their feature measures
        critNumMissingFeatures = 0.4 * len(feature_cols)
        df = df.drop(df[df[feature_cols].isna().sum(axis=1) > critNumMissingFeatures].index)
    
        df = df.reset_index(drop = True)
        dfnp = df.to_numpy()
    
        df_train_fold = df_model_data_raw[ensembleModel_cols]
        
        dfimpute = pd.DataFrame(impute_data_knn(df_train_fold[feature_cols], n_neighbors=knn_num_neighbors), columns=feature_cols)
        df_train_fold_info = df_train_fold[info_cols].copy()
        df_train_fold_info = df_train_fold_info.reset_index(drop = True)
        dfimpute = dfimpute.reset_index(drop = True)
        df_train_fold_imputed = pd.concat([df_train_fold_info,dfimpute],axis=1)
    
        fold_Train = df_train_fold_imputed.copy()
    
        y_train = fold_Train[Y_METRIC].astype(float)
    
        y_pred_train_base_models = []
    
        for ind in ensembleModels.index:
            EA_model_features = getUniqueKeepOrder((ensembleModels['Features'][ind].split(",")))
    
            X_train = fold_Train[EA_model_features].to_numpy()
    
            n_components = 1
            pls_model = PLSRegression(n_components=n_components)
            pls_model.fit(X_train, y_train)
            y_pred_train = pls_model.predict(X_train)
    
            y_pred_train_base_models += [y_pred_train]
    
        y_pred_train_ensemble_model = np.mean(y_pred_train_base_models, axis = 0)
        
        train_R2 = r2_score(y_train, y_pred_train_ensemble_model)
        ensemble_all_train_R2 += [train_R2]

        numPredictors = len(ensemble_feature_cols)
        if (numObservations - numPredictors - 1) > 0:
            adjR2Factor = (numObservations - 1) / (numObservations - numPredictors - 1)
        else:
            adjR2Factor = numObservations # should not reach this point but just in case
                
        train_adj_R2 = 1 - adjR2Factor * (1 - train_R2)
        ensemble_all_train_adj_R2 += [train_adj_R2]

        train_mse = mean_squared_error(y_train, y_pred_train_ensemble_model)
        ensemble_all_train_mse += [train_mse]
    

    df_ensemble_models = df_ensemble_models.iloc[index_list].copy()
        
    df_ensemble_models['all_cv_train_mse_mean'] = ensemble_models_train_mse_mean
    df_ensemble_models['all_cv_train_mse_stdev'] = ensemble_models_train_mse_stdev
    df_ensemble_models['all_cv_test_mse_mean'] = ensemble_models_test_mse_mean
    df_ensemble_models['all_cv_test_mse_stdev'] = ensemble_models_test_mse_stdev
    df_ensemble_models['all_cv_train_R2_mean'] = ensemble_models_train_R2_mean
    df_ensemble_models['all_cv_train_R2_stdev'] = ensemble_models_train_R2_stdev
    df_ensemble_models['all_cv_test_R2_mean'] = ensemble_models_test_R2_mean
    df_ensemble_models['all_cv_test_R2_stdev'] = ensemble_models_test_R2_stdev

    df_ensemble_models['all_cv_train_adj_R2_mean'] = ensemble_models_train_adj_R2_mean
    df_ensemble_models['all_cv_train_adj_R2_stdev'] = ensemble_models_train_adj_R2_stdev
    df_ensemble_models['all_cv_test_adj_R2_mean'] = ensemble_models_test_adj_R2_mean
    df_ensemble_models['all_cv_test_adj_R2_stdev'] = ensemble_models_test_adj_R2_stdev
    
    df_ensemble_models['all_cv_N'] = ensemble_models_N
    df_ensemble_models['ratio_all_cv_test_over_train'] = df_ensemble_models['all_cv_test_mse_mean'] / df_ensemble_models['all_cv_train_mse_mean']
    
    df_ensemble_models['all_train_mse'] = ensemble_all_train_mse
    df_ensemble_models['all_train_R2'] = ensemble_all_train_R2
    df_ensemble_models['all_train_adj_R2'] = ensemble_all_train_adj_R2
    
    return df_ensemble_models

#################################################################################################################################################################
#####
#################################################################################################################################################################

def evaluate_EnsembleModels(params):



    job_id = '1'

    index_list = params['index_list']
    
    num_runs = len(index_list)
    num_tasks = int(params['max_num_cpus'])
    runs_per_task = num_runs // num_tasks + 1 if num_runs % num_tasks else num_runs // num_tasks

    if 'minNumRunsPerTask' in params:
        minNumRunsPerTask = params['minNumRunsPerTask']
    else:
        minNumRunsPerTask = 1
        
    if num_tasks > 1 and runs_per_task < minNumRunsPerTask:
        num_tasks = round(num_runs / minNumRunsPerTask)
        if num_tasks < 1:
            num_tasks = 1
        elif num_tasks > int(params['max_num_cpus']):
            num_tasks = int(params['max_num_cpus'])
        runs_per_task = num_runs // num_tasks + 1 if num_runs % num_tasks else num_runs // num_tasks


    params = { **params, 'job_id': job_id}

    n_request_cpus = num_tasks
        
    if num_tasks == 1:
        proc_id = 0
        sub_index_list = index_list[(proc_id*runs_per_task):min(num_runs, (proc_id+1)*runs_per_task)]
        
        params = { **params, 'proc_id': proc_id}
        params = { **params, 'sub_index_list': sub_index_list}   

        # print("num runs to process in series:", num_runs,"cpu processors requested:", n_request_cpus)
        
        # print('Started processing_evaluate_EnsembleModels',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
          
        df_ensemble_models =_evaluate_EnsembleModels(params)

        # print('Completed processing of_evaluate_EnsembleModels',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    else:
      
        param_sets = []
        for proc_id in range(num_tasks):
          
          sub_index_list = index_list[(proc_id*runs_per_task):min(num_runs, (proc_id+1)*runs_per_task)]
          if (len(sub_index_list) > 0):
              
              # make a copy of the parameter set and then overwrite it to customize
              params_copy = params.copy()
              params_copy = { **params_copy, 'proc_id': proc_id}
              params_copy = { **params_copy, 'sub_index_list': sub_index_list}   
    
              param_sets.append(params_copy)
        
        n_request_cpus = len(param_sets)
        
        # print('Started multiprocessing of_evaluate_EnsembleModels',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        print('# tasks/processors: ' + str(num_runs) + '/' + str(n_request_cpus))
        
        pool = multiprocessing.Pool(n_request_cpus)
        results = pool.map(_evaluate_EnsembleModels, param_sets)

        # print('Completed multiprocessing of_evaluate_EnsembleModels',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        df_ensemble_models = pd.concat(results)
        df_ensemble_models = df_ensemble_models.reset_index(drop = True)

    return df_ensemble_models



#################################################################################################################################################################
#####
#################################################################################################################################################################

# stratified random sub sampling without replacement
def strat_rss_type2(model, foldNames, x_trains, y_trains, x_tests, y_tests, k, nck_list,
                       num_repeat=1, repeat_start=0,numPerTrain=10, numPerTest=10, ssTrain=True,ssTest=True,quantiles = False):
    import pandas
    import numpy as np
    import scipy
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold
    import warnings
    

    warnings.filterwarnings('ignore')

    if quantiles:
      summary_columns = ['Fold', 'Features', 'maxPairWise.abs.spearman.r', 'numTrainSamples', 'numTestSamples','N', 
                        'mean.Accuracy', 'mean.Precision', 'mean.Recall', 'mean.AUC',
                        'min.Accuracy', 'min.Precision', 'min.Recall', 'min.AUC',
                        'Q25.Accuracy', 'Q25.Precision', 'Q25.Recall', 'Q25.AUC',
                        'Q10.Accuracy', 'Q10.Precision', 'Q10.Recall', 'Q10.AUC',
                        'max.Accuracy', 'max.Precision', 'max.Recall', 'max.AUC',
                        'std.Accuracy', 'std.Precision', 'std.Recall', 'std.AUC']
      
    else:
      summary_columns = ['Fold', 'Features', 'maxPairWise.abs.spearman.r', 'numTrainSamples', 'numTestSamples','N', 
                        'mean.Accuracy', 'mean.Precision', 'mean.Recall', 'mean.AUC',
                        'min.Accuracy', 'min.Precision', 'min.Recall', 'min.AUC',
                        'max.Accuracy', 'max.Precision', 'max.Recall', 'max.AUC',
                        'std.Accuracy', 'std.Precision', 'std.Recall', 'std.AUC']

                      
    df_results = pandas.DataFrame(columns=summary_columns)
    for i in range(len(nck_list)):
        Xtup = nck_list[i]
        features = ', '.join(Xtup) 
        for foldName, x_train, y_train, x_test, y_test in zip(foldNames, x_trains, y_trains, x_tests, y_tests):
            x_train_data = x_train.loc[:, Xtup]
            x_test_data = x_test.loc[:, Xtup]  
            
            maxspearmanr = 0
            nf = len(list(Xtup))
            for i in range(nf-1):
              for j in range(i+1,nf):
                spearmanr = abs(scipy.stats.spearmanr(x_train_data.values[:, i], x_train_data.values[:, j])[0])
                if maxspearmanr < spearmanr:
                  maxspearmanr = spearmanr

                
 
            if ssTrain:
                numSplits_ssTrain = max(int(len(x_train_data) / numPerTrain), 2)
            else:
                numSplits_ssTrain = 1
              
            if ssTest:
                numSplits_ssTest = max(int(len(x_test_data) / numPerTest), 2)
            else:
                numSplits_ssTest = 1
              
            columns = ['Accuracy', 'Precision', 'Recall', 'AUC']
            df_result = pandas.DataFrame(index=range(num_repeat), columns=columns)
            nTrain = 0
            nTest = 0
            repeat = 0
            while repeat < num_repeat:
              
                samp_ssTrains = []
                if numSplits_ssTrain > 1:
                    fssTrainVer = -2

                    skf_ssTrain = StratifiedKFold(n_splits=numSplits_ssTrain,shuffle=True, random_state = repeat + repeat_start)
                    
                    X_ssTrain = x_train_data.to_numpy()
                    y_ssTrain = np.array(y_train, dtype=int)
                    
                    for (otheri, index_ssTrain) in skf_ssTrain.split(X_ssTrain, y_ssTrain):
                        samp_ssTrains = samp_ssTrains + [index_ssTrain]
                
                else:     
                    fssTrainVer = -1
                    samp_ssTrains = samp_ssTrains + [x_train_data.index]
                    
                samp_ssTests = []    
                if numSplits_ssTest > 1:    
                    fssTestVer = -2

                    skf_ssTest = StratifiedKFold(n_splits=numSplits_ssTest,shuffle=True, random_state = repeat + repeat_start)
                    
                    X_ssTest = x_test_data.to_numpy()
                    y_ssTest = np.array(y_test, dtype=int)
                    
                    for (otheri, index_ssTest) in skf_ssTest.split(X_ssTest, y_ssTest):
                        samp_ssTests = samp_ssTests + [index_ssTest]
                
                else:
                    fssTestVer = -1
                    samp_ssTests = samp_ssTests + [x_test_data.index]

                    
                for index_ssTrain in samp_ssTrains:
                  for index_ssTest in samp_ssTests:
                    
                    x_train_ss = x_train_data.loc[index_ssTrain]
                    y_train_ss = y_train.loc[index_ssTrain]
                        
                    x_test_ss = x_test_data.loc[index_ssTest]
                    y_test_ss = y_test.loc[index_ssTest]
                

                    if len(np.unique(y_train_ss.values)) < 2:
                        continue

                    if len(np.unique(y_test_ss.values)) < 2:
                        continue
    
      
 
                    model.fit(x_train_ss.values, y_train_ss.values)
                    predicted = model.predict(x_test_ss.values)
    
                    auc = metrics.roc_auc_score(y_test_ss, predicted)
                    prec = metrics.precision_score(y_test_ss, predicted, pos_label=True, average='binary')
                    recall = metrics.recall_score(y_test_ss, predicted, pos_label=True, average='binary')
                    accuracy = metrics.accuracy_score(y_test_ss, predicted)
    
                    df_result.loc[repeat, 'Accuracy'] = accuracy
                    df_result.loc[repeat, 'Precision'] = prec
                    df_result.loc[repeat, 'Recall'] = recall
                    df_result.loc[repeat, 'AUC'] = auc
                    
                    nTrain = nTrain + len(x_train_ss)
                    nTest = nTest + len(x_test_ss)
                    
                    repeat += 1
    

            avgNumTrainSamples = nTrain / len(df_result)
            avgNumTestSamples = nTest / len(df_result)
            
            if quantiles:
                row = [foldName, features, round(maxspearmanr, 3), avgNumTrainSamples, avgNumTestSamples, len(df_result)] + list(df_result.mean()) + list(df_result.min()) \
                + [np.quantile(df_result["Accuracy"].values, .25), np.quantile(df_result["Precision"].values, .25), np.quantile(df_result["Recall"].values, .25), np.quantile(df_result["AUC"].values, .25)] \
                + [np.quantile(df_result["Accuracy"].values, .10), np.quantile(df_result["Precision"].values, .10), np.quantile(df_result["Recall"].values, .10), np.quantile(df_result["AUC"].values, .10)] \
                + list(df_result.max()) + list(df_result.std())
            else:
                row = [foldName, features, round(maxspearmanr, 3), avgNumTrainSamples, avgNumTestSamples, len(df_result)] + list(df_result.mean()) + list(df_result.min()) + list(df_result.max()) + list(df_result.std()) 

                                
            df_results.loc[len(df_results.index)] = row
                
    return df_results

#################################################################################################################################################################
#####
#################################################################################################################################################################

def _nck_strat_rss(params):
  
    import pandas as pd
    from sklearn.naive_bayes import GaussianNB
    import os
    import argparse
    from glob import glob
    import time
      

    job_id = params['job_id']  
    proc_id = params['proc_id']
    output_dir = params['output_dir']
    fps = params['fps']
    train_xs = params['train_xs']
    train_ys = params['train_ys']
    val_xs = params['val_xs']   
    val_ys = params['val_ys']   
    k_features = params['k_features']   
    sub_nck_list = params['sub_nck_list']
    num_repeat = params['num_repeat']   
    repeat_start = params['repeat_start']   

             
    # proc_id = 0
    # sub_nck_list = nck_list[(proc_id*runs_per_task):min(num_runs, (proc_id+1)*runs_per_task)]
    
    start_time = time.perf_counter()
    
    model = GaussianNB()
    
    df_results = strat_rss_type2(model, fps, train_xs, train_ys, val_xs, val_ys, k_features, sub_nck_list,
                                              num_repeat=num_repeat, repeat_start=repeat_start,ssTrain=True,ssTest=True, quantiles = False)
    
    output_file = f"job_{job_id}_{proc_id:05d}.csv"
    df_results.to_csv(os.path.join(output_dir, output_file), index=False)
                                         
    time_elapse = time.perf_counter() - start_time
    time_per_model = time_elapse/len(sub_nck_list) if len(sub_nck_list) > 0 else time_elapse
    
    print(f"completed: {job_id}_{proc_id:05d} | "
          f"per model: {time_per_model:.3f} (seconds) | "
          f"total time {time_elapse:.2f} (seconds) over {len(sub_nck_list)} models"
          )

    return len(sub_nck_list)

#################################################################################################################################################################
#####
#################################################################################################################################################################

def nck_strat_rss(params):
  
    import pandas as pd
    from sklearn.naive_bayes import GaussianNB
    import os
    import argparse
    from glob import glob
    import time

    feature_file = os.path.join(params['analysis_results_dir'], params['input_feature_filename'])
    # args = get_args_parser().parse_args()
    feature_df = pd.read_csv(feature_file)
    data_root = params['data_root']
    output_root = params['output_root']
    k_features = params['k_features']
    exp_dir, exp_name = os.path.split(os.path.splitext(feature_file)[0])
    rel_dir = os.path.relpath(exp_dir, data_root)
    array_job = False
    if os.environ.get('SLURM_ARRAY_JOB_ID') is not None:
        array_job = True
    if array_job:
        job_id = os.environ.get('SLURM_ARRAY_JOB_ID')
    else:
        job_id = os.environ.get('SLURM_JOB_ID')
    output_dir = os.path.join(output_root, rel_dir, exp_name, f"NCK_{k_features}", f"job_{job_id}")
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)
    
    fps = sorted(glob(os.path.join(exp_dir, 'fold_*')))
    # get the dir name
    fps = [os.path.basename(fp) for fp in fps]
    train_dfs = [pd.read_csv(os.path.join(exp_dir, fp, f"{fp}_train_features_imputed.csv")) for fp in fps]
    val_dfs = [pd.read_csv(os.path.join(exp_dir, fp, f"{fp}_test_features_imputed.csv")) for fp in fps]
    
    col_set_list = [set(df.columns) for df in train_dfs] + [set(df.columns) for df in val_dfs]
    common_cols = set.intersection(*col_set_list)
    sel_feature_df = feature_df.loc[feature_df['featureName'].isin(common_cols)]
    feature_list = sel_feature_df['featureName'].to_list()
    
    train_xs = [df[feature_list] for df in train_dfs]
    train_ys = [df[params['outcomeCN']] for df in train_dfs]
    
    val_xs = [df[feature_list] for df in val_dfs]
    val_ys = [df[params['outcomeCN']] for df in val_dfs]
    
    if params['nck_list_filename'] is None:
        fold_features = [item for item in feature_list if item.startswith('fold_')] 
        foldNames = fold_features
        for i in range(len(foldNames)):
            foldNames[i] = '_'.join(foldNames[i].split('_',2)[:2])
        foldNames = list(set(foldNames))
        foldNames.sort()
        if len(foldNames) > 0 and params['apply_fold_constraints']:
            nonfold_features = [item for item in feature_list if not item.startswith('fold_')] 
            nck_list = []
            for foldName in foldNames:
                target_fold_features = [item for item in feature_list if item.startswith(foldName)]
                sfl = nonfold_features + target_fold_features
                nck_list = nck_list + choosek(sfl, k_features)
        else:
            nck_list = choosek(feature_list, k_features)
            
    else:
        nck_list_file = os.path.join(params['analysis_results_dir'], params['nck_list_filename'])
        nck_list_df = pd.read_csv(nck_list_file)
        nck_list = nck_list_df.to_records(index=False).tolist()




            
            
    if params['no_slurm']:

        num_runs = len(nck_list)
        num_tasks = int(params['max_num_cpus'])
        runs_per_task = num_runs // num_tasks + 1 if num_runs % num_tasks else num_runs // num_tasks
    
        minNumRunsPerTask = 10
        if num_tasks > 1 and runs_per_task < minNumRunsPerTask:
            num_tasks = round(num_runs / minNumRunsPerTask)
            if num_tasks < 1:
                num_tasks = 1
            elif num_tasks > int(params['max_num_cpus']):
                num_tasks = int(params['max_num_cpus'])
            runs_per_task = num_runs // num_tasks + 1 if num_runs % num_tasks else num_runs // num_tasks
    
        print('')
        print('feature_list length',len(feature_list))
        print('NCK k_features:',k_features)
        print('num_tasks',num_tasks)
        print('num_runs',num_runs)
        print('runs_per_task',runs_per_task)
        
        sec_per_run = 0.662
        print('estimated time per task (minutes)',round((runs_per_task*sec_per_run)/60, 1))
        print('estimated time per task  (hours)',round((runs_per_task*sec_per_run)/60/60, 1))
        if params['max_time_limit_in_minutes'] < (runs_per_task*sec_per_run)/60:
            print('')
            print('Not processing NCK runs due to estimated conputational time above max_time_limit_in_minutes!')
            print('')
            return
    
        sel_feature_df.to_csv(os.path.join(output_dir, 'feature_file.csv'), index=False)
        # run_config = dict(num_tasks=num_tasks)
        # run_config.update(vars(args))
        # if not params['no_slurm']:
        #   import yaml
        #   with open(os.path.join(output_dir, 'config_info.yml'), 'w') as f:
        #       yaml.safe_dump(run_config, f)
        

      
        params = { **params, 'job_id': job_id}
        params = { **params, 'output_dir': output_dir}
        params = { **params, 'fps': fps}
        params = { **params, 'train_xs': train_xs}
        params = { **params, 'train_ys': train_ys}
        params = { **params, 'val_xs': val_xs}
        params = { **params, 'val_ys': val_ys}


        n_request_cpus = num_tasks
            
        if num_tasks == 1:
            proc_id = 0
            sub_nck_list = nck_list[(proc_id*runs_per_task):min(num_runs, (proc_id+1)*runs_per_task)]
            
            params = { **params, 'proc_id': proc_id}
            params = { **params, 'sub_nck_list': sub_nck_list}   

            print("num NCK runs to process:", num_runs,"cpu processors requested:", n_request_cpus)
            
            print('Started processing _nck_strat_rss',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        
            _nck_strat_rss(params)

            print('Completed processing of _nck_strat_rss',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
 
        else:
          
            param_sets = []
            for proc_id in range(num_tasks):
              
              sub_nck_list = nck_list[(proc_id*runs_per_task):min(num_runs, (proc_id+1)*runs_per_task)]
                
              # make a copy of the parameter set and then overwrite it to customize
              params_copy = params.copy()
              params_copy = { **params_copy, 'proc_id': proc_id}
              params_copy = { **params_copy, 'sub_nck_list': sub_nck_list}   

              param_sets.append(params_copy)
            

            print('Started multiprocessing of _nck_strat_rss',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            print('# tasks/processors: ' + str(num_runs) + '/' + str(n_request_cpus))

            pool = multiprocessing.Pool(n_request_cpus)
            results = pool.map(_nck_strat_rss, param_sets)

            print('Completed multiprocessing of _nck_strat_rss',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    

    
    else:  # params['no_slurm'] == False
        
        if array_job:
            if int(os.environ.get('SLURM_PROCID')) > 0:
                raise EnvironmentError(f"Expect 1 task with array job, got SLURM_PROCID={os.environ.get('SLURM_PROCID')}")
            num_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT'))
            proc_id = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
        else:
            num_tasks = int(os.environ.get('SLURM_NTASKS'))
            proc_id = int(os.environ.get('SLURM_PROCID'))
    
        if proc_id == 0:
            sel_feature_df.to_csv(os.path.join(output_dir, 'feature_file.csv'), index=False)
            # run_config = dict(num_tasks=num_tasks)
            # run_config.update(vars(args))
            # if not params['no_slurm']:
            #   import yaml
            #   with open(os.path.join(output_dir, 'config_info.yml'), 'w') as f:
            #       yaml.safe_dump(run_config, f)
    

        num_runs = len(nck_list)
        runs_per_task = num_runs // num_tasks + 1 if num_runs % num_tasks else num_runs // num_tasks
        sub_nck_list = nck_list[(proc_id*runs_per_task):min(num_runs, (proc_id+1)*runs_per_task)]
        
        start_time = time.perf_counter()
        
        model = GaussianNB()
        
        df_results = strat_rss_type2(model, fps, train_xs, train_ys, val_xs, val_ys, k_features, sub_nck_list,
                                                  num_repeat=params['num_repeat'], repeat_start=params['repeat_start'],ssTrain=True,ssTest=True, quantiles = False)
        
        output_file = f"job_{job_id}_{proc_id:05d}.csv"
        df_results.to_csv(os.path.join(output_dir, output_file), index=False)
                                             
        time_elapse = time.perf_counter() - start_time
        time_per_model = time_elapse/len(sub_nck_list) if len(sub_nck_list) > 0 else time_elapse
        
        print(f"completed: {job_id}_{proc_id:05d} | "
              f"per model: {time_per_model:.3f} (seconds) | "
              f"total time {time_elapse:.2f} (seconds) over {len(sub_nck_list)} models"
              )
    
#################################################################################################################################################################
#####
#################################################################################################################################################################

def nck_assemble_results(params):

    import pandas as pd
    import os
    import argparse
    from glob import glob
    from os.path import normpath, basename


    args_result_dir = path.join(params['analysis_results_dir'], params['feature_output_name'])

    print('')
    print('Assembling NCK Results...')
    print('')
    
    
    output_file_list = []
    counter = 0
    # args = get_args_parser().parse_args()
    
    rfpp = normpath(args_result_dir).split(os.path.sep)
    resultsFN = 'NCK_' + rfpp[len(rfpp)-1] + '.zip'
        
    NCK_fps = sorted(glob(os.path.join(args_result_dir, 'NCK_*')))
    if len(NCK_fps) > 0:
      for NCK_fp in NCK_fps:
        if os.path.isdir(NCK_fp): 
          nck_name = basename(normpath(NCK_fp))
  
          job_fps = sorted(glob(os.path.join(NCK_fp, 'job_*')),reverse=True)
          if len(job_fps) > 0:
            job_fp = job_fps[0]
            job_name = basename(normpath(job_fp))
            print(nck_name,' ', job_name)
            fps = sorted(glob(os.path.join(job_fp, 'job_*.csv')))
            if len(fps) > 0:
              dfs = [pd.read_csv(f) for f in fps]
              df = pd.concat(dfs)
              counter += 1
    
              output_fn = os.path.join(args_result_dir, nck_name + '_' + job_name + '_results.csv')
              df.to_csv(output_fn, index=False)   
              output_file_list = output_file_list + [basename(normpath(output_fn))]
            else:
              print('**** NO RESULTS FOR : ' + nck_name)

    if len(output_file_list) > 0:
      import zipfile

      with zipfile.ZipFile(os.path.join(args_result_dir, resultsFN), 'w') as zipMe:        
          for fn in output_file_list:
              zipMe.write(os.path.join(args_result_dir,fn), arcname=fn, compress_type=zipfile.ZIP_DEFLATED)
        
        
      print('')
      print('saved zipped results to:')  
      print('')
      print(os.path.join(args_result_dir,resultsFN))
      print('')               


      reply = None
        
      import sys
      if sys.version_info[0] < 3:
        while reply not in ('y', 'n'):
            reply = raw_input('Do you want to delete source files? (Enter y/n)').lower()
      else:
        while reply not in ('y', 'n'):
            reply = input('Do you want to delete source files? (Enter y/n)').lower()
            
      if reply in ('y'):
          import shutil
  
          for NCK_fp in NCK_fps:
            if os.path.isdir(NCK_fp):
              shutil.rmtree(NCK_fp)

          
    else:
      print('')
      print('Problem? No results were found.')  
      print('')
      
#################################################################################################################################################################
#####
#################################################################################################################################################################

def applySexMedianNormalization(mDat, ignore_columns = [], uids_sex = {}):
    sex_values =  list(uids_sex.keys())
    if len(sex_values) > 1:
        cns = mDat.columns
        cns = [cn for cn in cns if cn not in ignore_columns]
        for cn in cns:
            mv = mDat[cn].median()
            for sex_value in sex_values:
                indices = mDat['UID'].isin(uids_sex[str(sex_value)])
                smv = mDat.loc[indices, cn].median()
                if smv == 0:
                    smv = 1e-8;
                nv = mv / smv
                v = list(mDat.loc[indices, cn].values)
                if mDat[cn].dtypes == 'float64':
                    v = [nv * x for x in v]
                    mDat.loc[indices, cn] = v
                elif mDat[cn].dtypes == 'int64':     
                    v = [int(round(nv * x)) for x in v]
                    mDat.loc[indices, cn] = v  
    return mDat.copy()
                
#################################################################################################################################################################
#####
#################################################################################################################################################################

def setup_analysis_run(runParm):

    stepStartTime = datetime.datetime.now()
    

    base_dir = runParm['base_dir']
    max_num_cpus = runParm['max_num_cpus']

    randomState = runParm['randomState']



    misclassifyOutcomeFraction = runParm['misclassifyOutcomeFraction']

    interactome_db_dir = runParm['interactome_db_dir']
    interactomeName = runParm['interactomeName']

    info_cols = runParm['info_cols']
    analysis_dir = runParm['analysis_dir']
    analysis_results_dir = runParm['analysis_results_dir']

    OUTCOME_METRIC = runParm['OUTCOME_METRIC']
        


    # check that data source exists
    if not os.path.exists(runParm['db_cohort_data_measures_dir']):
        print('Error : db_cohort_data_measures_dir does not exist!')
        print(runParm['db_cohort_data_measures_dir'])
        raise Exception('Error : db_cohort_data_measures_dir does not exist! : ' + runParm['db_cohort_data_measures_dir'])
        
    
    if not os.path.exists(runParm['analysis_dir']):
        os.mkdir(runParm['analysis_dir'])
        print('Created analysis_dir:')
        print(runParm['analysis_dir'])
    
    
    if not os.path.exists(runParm['input_measures_dir']):
        os.mkdir(runParm['input_measures_dir'])
        print('Created input_measures_dir:')
        print(runParm['input_measures_dir'])

    

    
    uids = set()
    uids_sex = {}

    all_features_raw = pd.DataFrame()
    phenotype_features_info = pd.DataFrame()
    molecular_features_info = pd.DataFrame()
    
    for mDatFN in runParm['phenotypeMeasures']:
        
        print('loading phenotype measures:', mDatFN)
        # load data from directory that contains the raw hDat and mDat data files for the entire cohort
        mDat = pd.read_csv(path.join(runParm['db_cohort_data_measures_dir'], mDatFN))
    
        hDatFN = mDatFN.replace(runParm['phenotype_mDat_prefix'], runParm['hDat_prefix'])
        
        # if there is an hDat_ file available, load it
        if os.path.isfile(path.join(runParm['db_cohort_data_measures_dir'], hDatFN)):
            hDat = pd.read_csv(path.join(runParm['db_cohort_data_measures_dir'], hDatFN))
    
            # hDat.loc[hDat.columnName == OUTCOME_METRIC, 'Data_Flavor'] = "OUTCOME"
            
            hDat.to_csv(path.join(runParm['input_measures_dir'], hDatFN),index=False)
    
            if len(phenotype_features_info) == 0:
                phenotype_features_info = hDat
            else:
                phenotype_features_info = pd.concat([phenotype_features_info, hDat], axis=0)
    
        if runParm['SEX_COLUMN_NAME'] in mDat.columns:
            # filter subjects for sex
            if len(runParm['SEX_VALUES']) != 0:
                mDat = mDat[mDat[runParm['SEX_COLUMN_NAME']].isin(runParm['SEX_VALUES'])].copy()                         
            for sex_value in getUniqueKeepOrder(mDat[runParm['SEX_COLUMN_NAME']].tolist()):
                uids_sex[str(sex_value)] = mDat.loc[mDat[runParm['SEX_COLUMN_NAME']] == sex_value, 'UID'].tolist()

        uids.update(mDat['UID'].tolist())

        if runParm['USE_SEX_MEDIAN_NORMALIZED_DATA']:
            ignore_columns = [cn for cn in runParm['info_cols'] if cn != runParm['OUTCOME_METRIC']]
            mDat = applySexMedianNormalization(mDat, ignore_columns, uids_sex)

        if 'outcome' not in mDat.columns:
            mDat['outcome'] = mDat[OUTCOME_METRIC]

        if misclassifyOutcomeFraction > 0 and 'ground_truth' not in mDat.columns:
            mDat['ground_truth'] = mDat[OUTCOME_METRIC]
    
        # This is to permutate the phenotype measures with the subjects (UIDs)
        if runParm['permutationState'] > 0:
            uid_list = mDat['UID'].tolist()
            random.seed(int(runParm['permutationState']))
            random.shuffle(uid_list)
            mDat['UID'] = uid_list
            # mDat['UID'] = mDat['UID'].sample(frac=1, replace=False, random_state=int(runParm['permutationState'])).reset_index(drop=True)
                
        # save data to directory that contains the hDat and mDat data files for the specific analysis
        mDat.to_csv(path.join(runParm['input_measures_dir'], mDatFN),index=False)
    
    
        if len(all_features_raw) == 0:
            all_features_raw = mDat
        else:
            all_features_raw = pd.merge(all_features_raw, mDat, on='UID', how='outer')
    
    for mDatFN in runParm['omicMeasures']:
    
        print('loading omic measures:', mDatFN)
        # load data from directory that contains the raw hDat and mDat data files for the entire cohort
        mDat = pd.read_csv(path.join(runParm['db_cohort_data_measures_dir'], mDatFN))
    
        hDatFN = mDatFN.replace(runParm['omic_mDat_prefix'], runParm['hDat_prefix'])
        
        # if there is an hDat_ file available, load it
        if os.path.isfile(path.join(runParm['db_cohort_data_measures_dir'], hDatFN)):
            hDat = pd.read_csv(path.join(runParm['db_cohort_data_measures_dir'], hDatFN))
    
            hDat = hDat[hDat['nodeName'].notna()].copy()   
    
            mDat = mDat[['UID']+list(hDat['measurement_id'])].copy()
    
            hDat.to_csv(path.join(runParm['input_measures_dir'], hDatFN),index=False)
    
            if len(molecular_features_info) == 0:
                molecular_features_info = hDat
            else:
                molecular_features_info = pd.concat([molecular_features_info, hDat], axis=0)
                
        # filter data and only keep rows that with the set : uids
        mDat = mDat[mDat[runParm['UID']].isin(uids)].copy()

        if runParm['USE_SEX_MEDIAN_NORMALIZED_DATA']:
            ignore_columns = [cn for cn in runParm['info_cols'] if cn != runParm['OUTCOME_METRIC']]
            mDat = applySexMedianNormalization(mDat, ignore_columns, uids_sex)

        
        mDat_median = mDat.groupby([runParm['UID']])[list(hDat['measurement_id'])].median().reset_index()
        
        # save data to directory that contains the hDat and mDat data files for the specific analysis
        mDat_median.to_csv(path.join(runParm['input_measures_dir'], mDatFN),index=False)
    
        if len(all_features_raw) == 0:
             all_features_raw = mDat_median.copy
        else:
             all_features_raw = pd.merge(all_features_raw, mDat_median, on='UID', how='outer').copy()
    

    if not os.path.exists(runParm['analysis_results_dir']):
        os.mkdir(runParm['analysis_results_dir'])
        print('Created analysis_results_dir:')
        print(runParm['analysis_results_dir'])
        
    
    
    molecular_features_info.insert(loc=0, column='featureName', value=molecular_features_info['measurement_id'].tolist())
    
    phenotype_features_info.rename(columns={'columnName': 'featureName'}, inplace = True) 
    phenotype_features_info = phenotype_features_info[['featureName', 'FeatureDescription', 'Data_Flavor']]
    
    header_columun_info = phenotype_features_info[phenotype_features_info['featureName'].isin(info_cols)].copy()
    header_columun_info.to_csv(path.join(analysis_results_dir, 'header_columun_info.csv'),index=False)
    
    if runParm['Only_Outcome_No_Phenotypes']:
        all_features_raw = all_features_raw[info_cols + molecular_features_info['measurement_id'].tolist()].copy()
    elif len(runParm['phenotype_feature_list']) > 0:
        all_features_raw = all_features_raw[info_cols + runParm['phenotype_feature_list'] +  molecular_features_info['measurement_id'].tolist()].copy()
        phenotype_features_info = phenotype_features_info[phenotype_features_info['featureName'].isin(runParm['phenotype_feature_list'])].copy()
        if len(runParm['remove_phenotype_feature_list']) > 0:
            phenotype_features_info = phenotype_features_info[~phenotype_features_info['featureName'].isin(runParm['remove_phenotype_feature_list'])].copy()
        phenotype_features_info.to_csv(path.join(analysis_results_dir, 'phenotype_features_info.csv'),index=False)
    else:
        cns = all_features_raw.columns.to_list()
        cns = [cn for cn in cns if cn not in runParm['remove_phenotype_feature_list']]
        all_features_raw = all_features_raw[cns].copy()
        phenotype_features_info = phenotype_features_info[~phenotype_features_info['featureName'].isin(info_cols)].copy()
        if len(runParm['remove_phenotype_feature_list']) > 0:
            phenotype_features_info = phenotype_features_info[~phenotype_features_info['featureName'].isin(runParm['remove_phenotype_feature_list'])].copy()
        phenotype_features_info.to_csv(path.join(analysis_results_dir, 'phenotype_features_info.csv'),index=False)
    

    molecular_features_info.to_csv(path.join(analysis_results_dir, 'molecular_features_info.csv'),index=False)
    all_features_raw.to_csv(path.join(analysis_results_dir, 'all_features_raw.csv'),index=False)
    
    return runParm

                        
#################################################################################################################################################################
#####
#################################################################################################################################################################

def setup_prepost_analysis_run(runParm, DRAW_COLUMN_NAME = 'BLOOD_DRAW', preDrawValue = 'BD__0min', postDrawValue = 'BD_30min'):
    
# BLOOD_DRAW
# BD__0min
# BD__2min
# BD_15min
# BD_30min
# BD_60min

    
    stepStartTime = datetime.datetime.now()
    

    base_dir = runParm['base_dir']
    max_num_cpus = runParm['max_num_cpus']

    randomState = runParm['randomState']



    misclassifyOutcomeFraction = runParm['misclassifyOutcomeFraction']

    interactome_db_dir = runParm['interactome_db_dir']
    interactomeName = runParm['interactomeName']

    info_cols = runParm['info_cols']
    analysis_dir = runParm['analysis_dir']
    analysis_results_dir = runParm['analysis_results_dir']

    OUTCOME_METRIC = runParm['OUTCOME_METRIC']
        


    # check that data source exists
    if not os.path.exists(runParm['db_cohort_data_measures_dir']):
        print('Error : db_cohort_data_measures_dir does not exist!')
        print(runParm['db_cohort_data_measures_dir'])
        raise Exception('Error : db_cohort_data_measures_dir does not exist! : ' + runParm['db_cohort_data_measures_dir'])
        
    
    if not os.path.exists(runParm['analysis_dir']):
        os.mkdir(runParm['analysis_dir'])
        print('Created analysis_dir:')
        print(runParm['analysis_dir'])
    
    
    if not os.path.exists(runParm['input_measures_dir']):
        os.mkdir(runParm['input_measures_dir'])
        print('Created input_measures_dir:')
        print(runParm['input_measures_dir'])

    

    
    uids = set()
    uids_sex = {}

    all_features_raw = pd.DataFrame()
    phenotype_features_info = pd.DataFrame()
    molecular_features_info = pd.DataFrame()
    
    for mDatFN in runParm['phenotypeMeasures']:
        
        print('loading phenotype measures:', mDatFN)
        # load data from directory that contains the raw hDat and mDat data files for the entire cohort
        mDat = pd.read_csv(path.join(runParm['db_cohort_data_measures_dir'], mDatFN))

        mDat_pre = mDat.copy()
        mDat_pre['UID'] = mDat_pre['UID'] + '_' + preDrawValue
        mDat_pre[runParm['OUTCOME_METRIC']] = 0
        mDat_post = mDat.copy()
        mDat_post['UID'] = mDat_post['UID'] + '_' + postDrawValue
        mDat_post[runParm['OUTCOME_METRIC']] = 1
        mDat = pd.concat([mDat_pre, mDat_post], axis=0)
        mDat = mDat.reset_index(drop = True)
        
        hDatFN = mDatFN.replace(runParm['phenotype_mDat_prefix'], runParm['hDat_prefix'])
        
        # if there is an hDat_ file available, load it
        if os.path.isfile(path.join(runParm['db_cohort_data_measures_dir'], hDatFN)):
            hDat = pd.read_csv(path.join(runParm['db_cohort_data_measures_dir'], hDatFN))
    
            # hDat.loc[hDat.columnName == OUTCOME_METRIC, 'Data_Flavor'] = "OUTCOME"
            
            hDat.to_csv(path.join(runParm['input_measures_dir'], hDatFN),index=False)
    
            if len(phenotype_features_info) == 0:
                phenotype_features_info = hDat
            else:
                phenotype_features_info = pd.concat([phenotype_features_info, hDat], axis=0)
    
        if runParm['SEX_COLUMN_NAME'] in mDat.columns:
            # filter subjects for sex
            if len(runParm['SEX_VALUES']) != 0:
                mDat = mDat[mDat[runParm['SEX_COLUMN_NAME']].isin(runParm['SEX_VALUES'])].copy()                         
            for sex_value in getUniqueKeepOrder(mDat[runParm['SEX_COLUMN_NAME']].tolist()):
                uids_sex[str(sex_value)] = mDat.loc[mDat[runParm['SEX_COLUMN_NAME']] == sex_value, 'UID'].tolist()

        uids.update(mDat['UID'].tolist())

        if runParm['USE_SEX_MEDIAN_NORMALIZED_DATA']:
            ignore_columns = [cn for cn in runParm['info_cols'] if cn != runParm['OUTCOME_METRIC']]
            mDat = applySexMedianNormalization(mDat, ignore_columns, uids_sex)

        if 'outcome' not in mDat.columns:
            mDat['outcome'] = mDat[OUTCOME_METRIC]

        if misclassifyOutcomeFraction > 0 and 'ground_truth' not in mDat.columns:
            mDat['ground_truth'] = mDat[OUTCOME_METRIC]
    
        # This is to permutate the phenotype measures with the subjects (UIDs)
        if runParm['permutationState'] > 0:
            uid_list = mDat['UID'].tolist()
            random.seed(int(runParm['permutationState']))
            random.shuffle(uid_list)
            mDat['UID'] = uid_list
            # mDat['UID'] = mDat['UID'].sample(frac=1, replace=False, random_state=int(runParm['permutationState'])).reset_index(drop=True)
                
        # save data to directory that contains the hDat and mDat data files for the specific analysis
        mDat.to_csv(path.join(runParm['input_measures_dir'], mDatFN),index=False)
    
    
        if len(all_features_raw) == 0:
            all_features_raw = mDat
        else:
            all_features_raw = pd.merge(all_features_raw, mDat, on='UID', how='outer')
    
    for mDatFN in runParm['omicMeasures']:
    
        print('loading omic measures:', mDatFN)
        # load data from directory that contains the raw hDat and mDat data files for the entire cohort
        mDat = pd.read_csv(path.join(runParm['db_cohort_data_measures_dir'], mDatFN))

        mDat_pre = mDat.loc[mDat[DRAW_COLUMN_NAME]  == preDrawValue].copy()
        mDat_pre['UID'] = mDat_pre['UID'] + '_' + preDrawValue
        mDat_post = mDat.loc[mDat[DRAW_COLUMN_NAME]  == postDrawValue].copy()
        mDat_post['UID'] = mDat_post['UID'] + '_' + postDrawValue
        mDat = pd.concat([mDat_pre, mDat_post], axis=0)
        mDat = mDat.reset_index(drop = True)

        
        hDatFN = mDatFN.replace(runParm['omic_mDat_prefix'], runParm['hDat_prefix'])
        
        # if there is an hDat_ file available, load it
        if os.path.isfile(path.join(runParm['db_cohort_data_measures_dir'], hDatFN)):
            hDat = pd.read_csv(path.join(runParm['db_cohort_data_measures_dir'], hDatFN))
    
            hDat = hDat[hDat['nodeName'].notna()].copy()   
    
            mDat = mDat[['UID']+list(hDat['measurement_id'])].copy()
    
            hDat.to_csv(path.join(runParm['input_measures_dir'], hDatFN),index=False)
    
            if len(molecular_features_info) == 0:
                molecular_features_info = hDat
            else:
                molecular_features_info = pd.concat([molecular_features_info, hDat], axis=0)
                
        # filter data and only keep rows that with the set : uids
        mDat = mDat[mDat[runParm['UID']].isin(uids)].copy()

        if runParm['USE_SEX_MEDIAN_NORMALIZED_DATA']:
            ignore_columns = [cn for cn in runParm['info_cols'] if cn != runParm['OUTCOME_METRIC']]
            mDat = applySexMedianNormalization(mDat, ignore_columns, uids_sex)


        mDat_median = mDat.groupby([runParm['UID']])[list(hDat['measurement_id'])].median().reset_index()
        
        # save data to directory that contains the hDat and mDat data files for the specific analysis
        mDat_median.to_csv(path.join(runParm['input_measures_dir'], mDatFN),index=False)
    
        if len(all_features_raw) == 0:
             all_features_raw = mDat_median.copy
        else:
             all_features_raw = pd.merge(all_features_raw, mDat_median, on='UID', how='outer').copy()
    

    if not os.path.exists(runParm['analysis_results_dir']):
        os.mkdir(runParm['analysis_results_dir'])
        print('Created analysis_results_dir:')
        print(runParm['analysis_results_dir'])
        
    
    
    molecular_features_info.insert(loc=0, column='featureName', value=molecular_features_info['measurement_id'].tolist())
    
    phenotype_features_info.rename(columns={'columnName': 'featureName'}, inplace = True) 
    phenotype_features_info = phenotype_features_info[['featureName', 'FeatureDescription', 'Data_Flavor']]
    
    header_columun_info = phenotype_features_info[phenotype_features_info['featureName'].isin(info_cols)].copy()
    header_columun_info.to_csv(path.join(analysis_results_dir, 'header_columun_info.csv'),index=False)
    
    if runParm['Only_Outcome_No_Phenotypes']:
        all_features_raw = all_features_raw[info_cols + molecular_features_info['measurement_id'].tolist()].copy()
    elif len(runParm['phenotype_feature_list']) > 0:
        all_features_raw = all_features_raw[info_cols + runParm['phenotype_feature_list'] +  molecular_features_info['measurement_id'].tolist()].copy()
        phenotype_features_info = phenotype_features_info[phenotype_features_info['featureName'].isin(runParm['phenotype_feature_list'])].copy()
        if len(runParm['remove_phenotype_feature_list']) > 0:
            phenotype_features_info = phenotype_features_info[~phenotype_features_info['featureName'].isin(runParm['remove_phenotype_feature_list'])].copy()
        phenotype_features_info.to_csv(path.join(analysis_results_dir, 'phenotype_features_info.csv'),index=False)
    else:
        cns = all_features_raw.columns.to_list()
        cns = [cn for cn in cns if cn not in runParm['remove_phenotype_feature_list']]
        all_features_raw = all_features_raw[cns].copy()
        phenotype_features_info = phenotype_features_info[~phenotype_features_info['featureName'].isin(info_cols)].copy()
        if len(runParm['remove_phenotype_feature_list']) > 0:
            phenotype_features_info = phenotype_features_info[~phenotype_features_info['featureName'].isin(runParm['remove_phenotype_feature_list'])].copy()
        phenotype_features_info.to_csv(path.join(analysis_results_dir, 'phenotype_features_info.csv'),index=False)
    

    molecular_features_info.to_csv(path.join(analysis_results_dir, 'molecular_features_info.csv'),index=False)
    all_features_raw.to_csv(path.join(analysis_results_dir, 'all_features_raw.csv'),index=False)
    
    return runParm

#################################################################################################################################################################
#####
#################################################################################################################################################################

def _fold_assemble_training_data(oi3fold):
    
    findx = oi3fold['findx']
    train_index = oi3fold['train_index']
    test_index = oi3fold['test_index']
    dfnp = oi3fold['dfnp']
    df_NoOutcome = oi3fold['df_NoOutcome']
    feature_cols = oi3fold['feature_cols']
    phenotype_feature_cols = oi3fold['phenotype_feature_cols']
    molecular_feature_cols = oi3fold['molecular_feature_cols']
    metabolite_feature_cols = oi3fold['metabolite_feature_cols']
    protein_feature_cols = oi3fold['protein_feature_cols']
    rna_feature_cols = oi3fold['rna_feature_cols']
    dna_feature_cols = oi3fold['dna_feature_cols']
    source_interactome = oi3fold['source_interactome']

    all_cols =  oi3fold['all_cols']
    info_cols = oi3fold['info_cols']


    randomState = oi3fold['randomState']
    
    misclassifyOutcomeFraction = oi3fold['misclassifyOutcomeFraction']



    np.random.seed(randomState + findx)

    
    FN_CV_FOLD_TRAIN_IMPUTED = path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], (oi3fold['FOLD_DIR_NAME'] + '_train_features_imputed.csv'))

    FN_CV_FOLD_TRAIN_PA_FEATURES = path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], (oi3fold['FOLD_DIR_NAME'] + '_train_PA_features.csv'))


    os.makedirs(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME']), exist_ok=True)

    df_train_fold = pd.DataFrame(dfnp[train_index],columns=all_cols)

    # This is to introduce random misclassification into the training data for the fold
    if misclassifyOutcomeFraction > 0:
        np.random.seed(randomState + findx)
        change = df_train_fold.sample(frac=misclassifyOutcomeFraction, replace=False).index
        v = df_train_fold.loc[change, oi3fold['OUTCOME_METRIC']]
        v = abs(v - 1)
        df_train_fold.loc[change, oi3fold['OUTCOME_METRIC']] = v
        df_train_fold.loc[change, 'outcome'] = v
        np.random.seed(randomState + findx)


    if oi3fold['imputeByClass']:
        df_train_fold_info = df_train_fold[info_cols].copy()
        df_train_fold_info = df_train_fold_info.reset_index(drop = True)

        dfList = [df_train_fold_info]
        
        if len(phenotype_feature_cols) > 0:
            dfimpute = pd.DataFrame(impute_data_knn(df_train_fold[phenotype_feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=phenotype_feature_cols)
            dfimpute = dfimpute.reset_index(drop = True)
            dfList = dfList + [dfimpute]

        if len(metabolite_feature_cols) > 0:
            dfimpute = pd.DataFrame(impute_data_knn(df_train_fold[metabolite_feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=metabolite_feature_cols)
            dfimpute = dfimpute.reset_index(drop = True)
            dfList = dfList + [dfimpute]

        if len(protein_feature_cols) > 0:
            dfimpute = pd.DataFrame(impute_data_knn(df_train_fold[protein_feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=protein_feature_cols)
            dfimpute = dfimpute.reset_index(drop = True)
            dfList = dfList + [dfimpute]

        if len(rna_feature_cols) > 0:
            dfimpute = pd.DataFrame(impute_data_knn(df_train_fold[rna_feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=rna_feature_cols)
            dfimpute = dfimpute.reset_index(drop = True)
            dfList = dfList + [dfimpute]

        if len(dna_feature_cols) > 0:
            dfimpute = pd.DataFrame(impute_data_knn(df_train_fold[dna_feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=dna_feature_cols)
            dfimpute = dfimpute.reset_index(drop = True)
            dfList = dfList + [dfimpute]

        df_train_fold_imputed = pd.concat(dfList,axis=1)

    else:
        dfimpute = pd.DataFrame(impute_data_knn(df_train_fold[feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=feature_cols)
        df_train_fold_info = df_train_fold[info_cols].copy()
        df_train_fold_info = df_train_fold_info.reset_index(drop = True)
        dfimpute = dfimpute.reset_index(drop = True)
        df_train_fold_imputed = pd.concat([df_train_fold_info,dfimpute],axis=1)



    # save imputed training data to fold directory for R script
    df_train_fold_imputed.to_csv(FN_CV_FOLD_TRAIN_IMPUTED, index=False, sep=',')

    # -------------------------------------------------------------------------------------
    # Perform feature selection (i.e training) of the phenotype PA and molecular MA EA models
    # Do this within this fold and only use the training data set

    pa_phenotype_features = []

    if oi3fold['build_PAs'] or oi3fold['build_PN_With_PAs'] or oi3fold['build_PN_With_Latent_Metrics'] or oi3fold['build_PN_With_Phenotypes']:

        fn = oi3fold['FOLD_DIR_NAME'] + '_PA_Phenotype_Ranking.csv'
        
        df_PA_Phenotype_Ranking = pd.read_csv(path.join(oi3fold['analysis_results_dir'], 'phenotype_features_info.csv'))
        df_PA_Phenotype_Ranking['phenotype'] = df_PA_Phenotype_Ranking['featureName']
        pa_phenotype_features = [cn for cn in phenotype_feature_cols if cn not in oi3fold['remove_phenotype_PA_feature_list']]
        df_PA_Phenotype_Ranking = df_PA_Phenotype_Ranking[df_PA_Phenotype_Ranking['phenotype'].isin(pa_phenotype_features)]
            
        spearman_cor = []
        spearman_pvalue = []
        for phenotype in df_PA_Phenotype_Ranking['phenotype'].tolist():
            res = stats.spearmanr(df_train_fold_imputed[phenotype], df_train_fold_imputed[oi3fold['OUTCOME_METRIC']])
            spearman_cor += [res.statistic]
            spearman_pvalue += [res.pvalue]
        df_PA_Phenotype_Ranking['spearman.cor'] = spearman_cor
        df_PA_Phenotype_Ranking['spearman.pValue'] = spearman_pvalue
        fn = oi3fold['FOLD_DIR_NAME'] + '_PA_Phenotype_Ranking.csv'
        df_PA_Phenotype_Ranking.to_csv(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], fn), index=False, sep=',')

    
    if oi3fold['build_PAs'] or oi3fold['build_PN_With_PAs']:
                
        # save imputed training data for phenotypes to compute PAs to fold directory for R script
        pa_phenotype_features = [cn for cn in phenotype_feature_cols if cn not in oi3fold['remove_phenotype_PA_feature_list']]
        # make sure to include the LATENT_METRIC_LIST
        cns = getUniqueKeepOrder(info_cols + [oi3fold['OUTCOME_METRIC']] + oi3fold['LATENT_METRIC_LIST'] + pa_phenotype_features)
        df_train_fold_PA_features = df_train_fold_imputed[cns].copy()
        df_train_fold_PA_features.to_csv(FN_CV_FOLD_TRAIN_PA_FEATURES, index=False, sep=',')
    
        print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + oi3fold['FOLD_DIR_NAME']  + '  ' + 'Build_Fold_Base_PA_Models')


        
        if oi3fold['COMPUTE_OUTCOME_BASE_EA_MODEL']:
            Y_METRICS = [oi3fold['OUTCOME_METRIC']] + oi3fold['LATENT_METRIC_LIST']
        else:
            Y_METRICS = oi3fold['LATENT_METRIC_LIST']
            if len(oi3fold['LATENT_METRIC_LIST']) == 0:
                print('LATENT_METRIC_LIST = [] and COMPUTE_OUTCOME_BASE_EA_MODEL = False. Either add y metrics to list or change boolean to True.')
                # break

        
        # R script to build Molecular Expression Axis (EA) Models
        script_path = oi3fold['RCODE_SOURCE_DIR'] + '/' + 'PhenoMol_sPLS_PAs_v1.R'
        args = [oi3fold['R_LIBRARY_DIR'], oi3fold['RCODE_SOURCE_DIR'], oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME']]
        args +=  ['RESULTS_FILENAME=' + BASE_PA_MODEL_CSV]
        Y_METRICS_STRING = ','.join(Y_METRICS)
        args += ['Y_METRICS=' + Y_METRICS_STRING]
        # args +=  ['Y_METRIC=' + oi3fold['OUTCOME_METRIC']]
        args += ['BINARY_OUTCOME=' + str(oi3fold['BINARY_OUTCOME']).upper()]
        args += ['MIN_NUM_MEASURES_PER_FEATURE=' + str(oi3fold['MIN_NUM_MEASURES_PER_FEATURE'])]
        args += ['PA_MAX_N_FEATURES=' + str(oi3fold['PA_MAX_N_FEATURES'])]
        args += ['TUNE_SPLS_MEASURE=' + str(oi3fold['TUNE_SPLS_MEASURE'])]
        args += ['TUNE_SPLS_NUM_FOLDS=' + str(oi3fold['TUNE_SPLS_NUM_FOLDS'])]
        args += ['TUNE_SPLS_NUM_REPEATS=' + str(oi3fold['TUNE_SPLS_NUM_REPEATS'])]
        args += ['TUNE_SPLS_MIN_FEATURE_STABILITY=' + str(oi3fold['TUNE_SPLS_MIN_FEATURE_STABILITY'])]
        args += ['SPLS_RANDOM_SEED=' + str(oi3fold['SPLS_RANDOM_SEED'])]
        r_path = oi3fold['r_path']
        cmd = [r_path, script_path]  + args
        result = subprocess.check_output(cmd, universal_newlines=True)
        
        print(result)

        print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + oi3fold['FOLD_DIR_NAME']  + '  ' + 'Build_Fold_Base_PA_Models')



    
    # Create oi_train_measures folder which will be used by oi3
    fold_measures_dir = path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], 'oi_train_measures')
    os.makedirs(fold_measures_dir, exist_ok=True)

    # make sure to include the LATENT_METRIC_LIST
    # cns = getUniqueKeepOrder(info_cols + [oi3fold['OUTCOME_METRIC']] + oi3fold['LATENT_METRIC_LIST'] + pa_phenotype_features)
    cns = getUniqueKeepOrder(['UID', oi3fold['OUTCOME_METRIC']] + oi3fold['LATENT_METRIC_LIST'] + pa_phenotype_features)
    mDat_phenotype = df_train_fold_imputed[cns]

    select_PAs = []
    
    if oi3fold['Only_Outcome_No_Phenotypes']:
        hDatRec = {'featureName': oi3fold['OUTCOME_METRIC'], 'FeatureDescription': oi3fold['OUTCOME_METRIC'], 'Data_Flavor': 'Outcome'}
        # hDat_phenotype = hDat_phenotype._append(hDatRec, ignore_index = True)
        hDat_phenotype = pd.DataFrame(hDatRec, index = [0])
    else:
        hDat_phenotype = pd.read_csv(path.join(oi3fold['analysis_results_dir'], 'phenotype_features_info.csv'))

        # make sure to include the LATENT_METRIC_LIST
        cns = getUniqueKeepOrder([oi3fold['OUTCOME_METRIC']] + oi3fold['LATENT_METRIC_LIST'] + pa_phenotype_features)
        hDat_phenotype = hDat_phenotype[hDat_phenotype['featureName'].isin(cns)]

        cns_missing = [cn for cn in cns if cn not in hDat_phenotype['featureName'].to_list()]
        # add records if they don't exist
        for cn in cns_missing:
            hDatRec = {'featureName': cn, 'FeatureDescription': cn, 'Data_Flavor': 'Outcome'}
            hDat_phenotype = hDat_phenotype._append(hDatRec, ignore_index = True)
        
        
        # get PA Summary Table
        fn = oi3fold['FOLD_DIR_NAME'] + '_PA_Summary_Table.csv'
        if os.path.exists(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], fn)):
            pa_summary_table = pd.read_csv(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], fn))
            pa_summary_table = pa_summary_table[pa_summary_table['spearman.pValue'] <= oi3fold['oi_run_phenotype_critical_p_value']].copy()
            select_PAs = pa_summary_table['PA'].tolist()
    
        # get PA measures for training subjects
        fn = oi3fold['FOLD_DIR_NAME'] + '_PA_Measures.csv'
        if os.path.exists(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], fn)):
            pa_measures = pd.read_csv(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], fn))
            # cns = pa_measures.columns.to_list()
            # for cn in cns:
            #     pa_measures.rename(columns={cn: cn.replace('PA','Phenotypic_Axis_')}, inplace=True)
            mDat_phenotype = pd.merge(mDat_phenotype, pa_measures, on='UID', how='left')
            pa_cols = [x for x in pa_measures.columns.to_list() if x not in ['UID']]
            for pa in pa_cols:
                hDatRec = {'featureName': pa, 'FeatureDescription': pa.replace('_',' '), 'Data_Flavor': 'Phenotypic_Axis'}
                hDat_phenotype = hDat_phenotype._append(hDatRec, ignore_index = True)
    
        
    mDat_phenotype.to_csv(path.join(fold_measures_dir,'mDat_phenotypes.csv'), index=False, sep=',')
    
    
    hDat_phenotype['columnName'] = hDat_phenotype['featureName']
    hDat_phenotype.to_csv(path.join(fold_measures_dir,'hDat_phenotypes.csv'), index=False, sep=',')
    

    
    mDat_molecular = df_train_fold_imputed[['UID'] + molecular_feature_cols]
    mDat_molecular.to_csv(path.join(fold_measures_dir,'mDat_molecular.csv'), index=False, sep=',')

    hDat_molecular = pd.read_csv(path.join(oi3fold['analysis_results_dir'], 'molecular_features_info.csv'))
    hDat_molecular = hDat_molecular[hDat_molecular['featureName'].isin(molecular_feature_cols)]
    hDat_molecular.to_csv(path.join(fold_measures_dir,'hDat_molecular.csv'), index=False, sep=',')


    oi3fold = { **oi3fold, 'df_train_fold_imputed': df_train_fold_imputed}

        

    if oi3fold['ASSEMBLE_PRIZE_DATA']:
        
        
        print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + oi3fold['FOLD_DIR_NAME'] + '  ' + 'ASSEMBLE_PRIZE_DATA')
            
        met_nodes = set()
        pro_nodes = set()
        rna_nodes = set()
        dna_nodes = set()
        hsa_nodes = set()
        
        d1 = source_interactome.interactome[source_interactome.interactome['nodeName1'].str.startswith('ENSP', na=False)]
        d2 = source_interactome.interactome[source_interactome.interactome['nodeName2'].str.startswith('ENSP', na=False)]
        pro_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())
        
        d1 = source_interactome.interactome[source_interactome.interactome['nodeName1'].str.startswith('CID', na=False)]
        d2 = source_interactome.interactome[source_interactome.interactome['nodeName2'].str.startswith('CID', na=False)]
        met_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())
        
        pro_pro_edges = source_interactome.interactome[source_interactome.interactome['nodeName1'].str.startswith('ENSP', na=False) & source_interactome.interactome['nodeName2'].str.startswith('ENSP', na=False)].copy()
        
        met_pro_edges = source_interactome.interactome[source_interactome.interactome['nodeName1'].str.startswith('CID', na=False) | source_interactome.interactome['nodeName2'].str.startswith('CID', na=False)].copy()
        
        if oi3fold['use_RNA']:
          
          rna2pro_edges = source_interactome.interactome_transcript_pp
          rna2pro_edges = rna2pro_edges.drop_duplicates()
          
          d1 = rna2pro_edges[rna2pro_edges['nodeName1'].str.startswith('ENST', na=False)]
          d2 = rna2pro_edges[rna2pro_edges['nodeName2'].str.startswith('ENST', na=False)]
          rna_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())
        
          d1 = rna2pro_edges[rna2pro_edges['nodeName1'].str.startswith('ENSP', na=False)]
          d2 = rna2pro_edges[rna2pro_edges['nodeName2'].str.startswith('ENSP', na=False)]
          pro_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())
          
        if oi3fold['use_DNA']:
          
          dna2rna_edges = source_interactome.interactome_gene_transcript
          dna2rna_edges = dna2rna_edges.drop_duplicates()
          
          d1 = dna2rna_edges[dna2rna_edges['nodeName1'].str.startswith('ENSG', na=False)]
          d2 = dna2rna_edges[dna2rna_edges['nodeName2'].str.startswith('ENSG', na=False)]
          dna_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())
        
          d1 = dna2rna_edges[dna2rna_edges['nodeName1'].str.startswith('ENST', na=False)]
          d2 = dna2rna_edges[dna2rna_edges['nodeName2'].str.startswith('ENST', na=False)]
          rna_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())
        
        if oi3fold['use_MicroRNA']:
        
          dna2rna_of_hsa_edges = source_interactome.interactome_gene_transcript_mature_mir
          dna2rna_of_hsa_edges = dna2rna_of_hsa_edges.drop_duplicates()
        
          d1 = dna2rna_of_hsa_edges[dna2rna_of_hsa_edges['nodeName1'].str.startswith('ENSG', na=False)]
          d2 = dna2rna_of_hsa_edges[dna2rna_of_hsa_edges['nodeName2'].str.startswith('ENSG', na=False)]
          dna_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())
        
          d1 = dna2rna_of_hsa_edges[dna2rna_of_hsa_edges['nodeName1'].str.startswith('ENST', na=False)]
          d2 = dna2rna_of_hsa_edges[dna2rna_of_hsa_edges['nodeName2'].str.startswith('ENST', na=False)]
          rna_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())
        
        
          rna2hsa_edges = source_interactome.interactome_transcript_mature_mir
          rna2hsa_edges = rna2hsa_edges.drop_duplicates()
          
          d1 = rna2hsa_edges[rna2hsa_edges['nodeName1'].str.startswith('ENST', na=False)]
          d2 = rna2hsa_edges[rna2hsa_edges['nodeName2'].str.startswith('ENST', na=False)]
          rna_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())
        
          # using ~ operator
          d1 = rna2hsa_edges[~rna2hsa_edges['nodeName1'].str.startswith('ENST', na=False)]
          d2 = rna2hsa_edges[~rna2hsa_edges['nodeName2'].str.startswith('ENST', na=False)]
          hsa_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())
        
        
          # mature miRNA binding to mRNA transcripts
          hsa2rna_edges = source_interactome.interactome_mature_mir_transcript
          hsa2rna_edges = hsa2rna_edges.drop_duplicates()
        
          d1 = hsa2rna_edges[hsa2rna_edges['nodeName1'].str.startswith('ENST', na=False)]
          d2 = hsa2rna_edges[hsa2rna_edges['nodeName2'].str.startswith('ENST', na=False)]
          rna_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())
        
          # using ~ operator
          d1 = hsa2rna_edges[~hsa2rna_edges['nodeName1'].str.startswith('ENST', na=False)]
          d2 = hsa2rna_edges[~hsa2rna_edges['nodeName2'].str.startswith('ENST', na=False)]
          hsa_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())
        
        
        print('Number of flat chemical nodes (CID) : ' , len(met_nodes))
        print('Number of protein nodes (ENSP) : ' , len(pro_nodes))
        print('Number of RNA transcript nodes (ENST) : ' , len(rna_nodes))
        print('Number of DNA gene nodes (ENSG) : ' , len(dna_nodes))
        print('Number of mature microRNA nodes (hsa) : ' , len(hsa_nodes))
        
        valid_nodeNames = met_nodes | pro_nodes | rna_nodes | dna_nodes | hsa_nodes
        
        
        print('number of valid nodes:',len(valid_nodeNames))

    
        # if not os.path.exists(oi3fold['db_data_dir']):
        #     print('Error : db_data_dir does not exist!')
        #     print(oi3fold['db_data_dir'])
        #     raise Exception('Error : db_data_dir does not exist! : ' + oi3fold['db_data_dir'])
        
        if not os.path.exists(oi3fold['input_measures_dir']):
            print('Error : input_measures_dir does not exist!')
            print(oi3fold['input_measures_dir'])
            raise Exception('Error : input_measures_dir does not exist! : ' + oi3fold['input_measures_dir'])
            
        if not os.path.exists(oi3fold['oi_prize_dir']):
            # print('Creating directory oi_prize_dir: ', oi3fold['oi_prize_dir'])
            os.makedirs(oi3fold['oi_prize_dir'])
            
        
        inputMeasures = {}
        
        mDat_phenotypes = pd.DataFrame()
        hDat_phenotypes = pd.DataFrame()
        
        for mDatFN in oi3fold['phenotypeMeasures']:
    
            
            print('loading phenotype measures:', mDatFN)
            mDat = pd.read_csv(path.join(oi3fold['input_measures_dir'], mDatFN))
        
            hDatFN = mDatFN.replace(oi3fold['phenotype_mDat_prefix'], oi3fold['hDat_prefix'])
            
            # if there is an hDat_ file available, load it
            if os.path.isfile(path.join(oi3fold['input_measures_dir'], hDatFN)):
                hDat = pd.read_csv(path.join(oi3fold['input_measures_dir'], hDatFN))
        
                inputMeasures = { **inputMeasures, hDatFN: hDat}
    
                if len(hDat_phenotypes) == 0:
                    hDat_phenotypes = hDat.copy() 
                else:
                    hDat_phenotypes = pd.concat([hDat_phenotypes, hDat.copy()])
        
            # make sure to convert measurement_ids to numeric
            cns = mDat.columns
            measurement_ids = [cn for cn in cns if cn not in oi3fold['UID']]
            # print(len(measurement_ids))
            mDat[measurement_ids] = mDat[measurement_ids].apply(pd.to_numeric, errors='coerce')

            # filter fold subjects for sex
            if oi3fold['SEX_COLUMN_NAME'] in mDat.columns:
                if len(oi3fold['SEX_VALUES']) != 0:
                    mDat = mDat[mDat[oi3fold['SEX_COLUMN_NAME']].isin(oi3fold['SEX_VALUES'])].copy()
    
            if len(mDat_phenotypes) == 0:
                mDat_phenotypes = mDat[[oi3fold['UID']] + measurement_ids].copy()
            else:
                mDat_phenotypes = mDat_phenotypes.merge(mDat[[oi3fold['UID']] + measurement_ids].copy(), how='left', on=oi3fold['UID'])
            
            inputMeasures = { **inputMeasures, 'mDat_phenotypes': mDat_phenotypes}    

        # Important to keep this set to False in order to compute the correlations of molecular features that do not show up on interactome
        filter_out_nonvalid_nodeNames = False
        
        for mDatFN in oi3fold['omicMeasures']:
        
            print('loading omic measures:', mDatFN)
            mDat = pd.read_csv(path.join(oi3fold['input_measures_dir'], mDatFN))
        
            hDatFN = mDatFN.replace(oi3fold['omic_mDat_prefix'], oi3fold['hDat_prefix'])
            
            # if there is an hDat_ file available, load it
            if os.path.isfile(path.join(oi3fold['input_measures_dir'], hDatFN)):
                hDat = pd.read_csv(path.join(oi3fold['input_measures_dir'], hDatFN))
                nMeasures = len(hDat)
                # filter out measurement ids that do not have a corresponding nodeName
                hDat = hDat.loc[hDat['nodeName'].notnull()].copy()
                hDat = hDat.loc[hDat['nodeName'].notna()].copy()
                
                # Code to deal with comma delimited values within 'nodeName' multiple nodeName(s) per measurement_id record
                val = [False]*len(hDat)
                for i in range(len(hDat)):
                    nodeNames = hDat['nodeName'][i].split(',')
                    val[i] = any(x in valid_nodeNames for x in nodeNames)
                hDat['valid_nodeName'] = val  # put this column back into the dataframe
                if filter_out_nonvalid_nodeNames:
                    hDat = hDat.loc[hDat['valid_nodeName'] == True].copy()
                measurement_ids = getUniqueKeepOrder((hDat['measurement_id']))
                # print('raw', nMeasures, 'filtered',len(measurement_ids))
                mDat = mDat[[oi3fold['UID']] + measurement_ids].copy()
                inputMeasures = { **inputMeasures, hDatFN: hDat}
        
            else:
                cns = mDat.columns
                measurement_ids = [cn for cn in cns if cn not in [oi3fold['UID']]]
                # if there is no hDat file for the omic measures, then assume the column name is the nodeName
                if filter_out_nonvalid_nodeNames:
                    measurement_ids = [cn for cn in measurement_ids if cn in valid_nodeNames]
                # print(len(measurement_ids))
                mDat = mDat[[oi3fold['UID']] + measurement_ids].copy()
                
            # make sure to convert measurement_ids to numeric
            mDat[measurement_ids] = mDat[measurement_ids].apply(pd.to_numeric, errors='coerce')
            
            inputMeasures = { **inputMeasures, mDatFN: mDat}
    
    
        hDat_phenotypes.rename(columns = {'columnName':'prizeName'}, inplace = True) 
        hDat_phenotypes.to_csv(oi3fold['prize_name_info_fp'],index=False)
    
        # prizeNames = list(hDat_phenotypes['prizeName'])

        ##################################################################################################################
        prizeNames = []
        if oi3fold['build_PN_With_Outcome']:
            prizeNames = prizeNames + [oi3fold['OUTCOME_METRIC']]
        if oi3fold['build_PN_With_Latent_Metrics']:
            prizeNames = prizeNames + oi3fold['LATENT_METRIC_LIST']
        if oi3fold['build_PN_With_PAs']:
            prizeNames = prizeNames + select_PAs
        if oi3fold['build_PN_With_Phenotypes']:
            fn = oi3fold['FOLD_DIR_NAME'] + '_PA_Phenotype_Ranking.csv'
            if os.path.exists(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], fn)):
                phenotype_Ranking = pd.read_csv(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], fn))
                phenotype_Ranking = phenotype_Ranking[phenotype_Ranking['spearman.pValue'] <= oi3fold['oi_run_phenotype_critical_p_value']].copy()
                phenotypeNames = list(phenotype_Ranking['phenotype'])
            else:
                prizeNameInfo = pd.read_csv(oi3fold['prize_name_info_fp'])
                phenotypeNames = list(prizeNameInfo['prizeName']) 
            prizeNames = prizeNames + phenotypeNames
        prizeNames = getUniqueKeepOrder((prizeNames))
        ##################################################################################################################

        print('number of sets of prizes (one for each outcome, PA, or targeted phenotype) that need to be generated:',len(prizeNames))
    
        oi3fold = { **oi3fold, 'inputMeasures': inputMeasures}  # from prior cells    
        oi3fold = { **oi3fold, 'prizeNames': prizeNames}        # from prior cells    

        print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + oi3fold['FOLD_DIR_NAME'] + '  ' + 'ASSEMBLE_PRIZE_DATA')
        print('')


    return oi3fold

#################################################################################################################################################################
#####
#################################################################################################################################################################

def _fold_generate_final_prizes_setup_oi_graph(oi3fold): 
    
    if oi3fold['GENERATE_FINAL_PRIZES']:

        print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + oi3fold['FOLD_DIR_NAME'] + '  ' + 'GENERATE_FINAL_PRIZES')

        generateFinalPrizes(oi3fold)
    
        print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + oi3fold['FOLD_DIR_NAME'] + '  ' + 'GENERATE_FINAL_PRIZES')
        print('')

    
    if oi3fold['SETUP_OI_RUNS']:
        
        print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + oi3fold['FOLD_DIR_NAME'] + '  ' + 'SETUP_OI_RUNS')
        
        if not os.path.exists(oi3fold['pcsf_run_dir']):
            # print('Creating directory pcsf_run_dir: ', oi3fold['pcsf_run_dir'])
            os.makedirs(oi3fold['pcsf_run_dir'])
        
        # copy prizeDataFinal and prizeNameInfo files from oi_prize_dir to pcsf_run_dir
        
        # Note it is important to overwrite files in case of an update.  Issues can occur when using  # if not os.path.isfile(oi3fold['prizeDataFinal_fp']):
        # print('Copied prizeDataFinal file from oi_prize_dir to: ', oi3fold['prizeDataFinal_fp'])
        shutil.copy(oi3fold['prize_data_final_fp'], oi3fold['prizeDataFinal_fp']) 
        
        # print('Copied prizeNameInfo file from oi_prize_dir to: ', oi3fold['prizeNameInfo_fp'])
        shutil.copy(oi3fold['prize_name_info_fp'], oi3fold['prizeNameInfo_fp'])
    
        init_OIruns(oi3fold)
    
        print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + oi3fold['FOLD_DIR_NAME'] + '  ' + 'SETUP_OI_RUNS')
        print('')

    if oi3fold['CREATE_OI_GRAPH_INSTANCE']:

        #create oi graph instance and load interactome
        
        print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + oi3fold['FOLD_DIR_NAME'] + '  ' + 'CREATE_OI_GRAPH_INSTANCE')

        graph = OIgraph()
    
        # Using the copy of the file that was created within the fold directory
        graph.set_interactome(oi3fold['oirun_interactome_fp'], oi3fold)
        
        # add to parameters
        oi3fold = { **oi3fold, 'graph': graph}
        
        print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + oi3fold['FOLD_DIR_NAME'] + '  ' + 'CREATE_OI_GRAPH_INSTANCE')
        print('')
    
    if oi3fold['GENERATE_OI_RUNS']:

        # Only generating OI runs that will be used to build the PN
        targetPhenotypes = []
        if oi3fold['build_PN_With_Outcome']:
            targetPhenotypes = targetPhenotypes + [oi3fold['OUTCOME_METRIC']]
        if oi3fold['build_PN_With_Latent_Metrics']:
            targetPhenotypes = targetPhenotypes + oi3fold['LATENT_METRIC_LIST']
        if oi3fold['build_PN_With_PAs']:
            targetPhenotypes = targetPhenotypes + select_PAs
        if oi3fold['build_PN_With_Phenotypes']:
            fn = oi3fold['FOLD_DIR_NAME'] + '_PA_Phenotype_Ranking.csv'
            if os.path.exists(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], fn)):
                phenotype_Ranking = pd.read_csv(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], fn))
                phenotype_Ranking = phenotype_Ranking[phenotype_Ranking['spearman.pValue'] <= oi3fold['oi_run_phenotype_critical_p_value']].copy()
                phenotypeNames = list(phenotype_Ranking['phenotype'])
            else:
                prizeNameInfo = pd.read_csv(oi3fold['prize_name_info_fp'])
                phenotypeNames = list(prizeNameInfo['prizeName'])
            targetPhenotypes = targetPhenotypes + phenotypeNames
        targetPhenotypes = getUniqueKeepOrder((targetPhenotypes))
        oi3fold = { **oi3fold, 'targetPhenotypes': targetPhenotypes}
    

    return oi3fold

#################################################################################################################################################################
#####
#################################################################################################################################################################
        
def _fold_process_oi_runs(oi3fold): 
    
    if oi3fold['PROCESS_OI_RUNS']:


        print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + oi3fold['FOLD_DIR_NAME'] + '  ' + 'PROCESS_OI_RUNS')
        
        if not os.path.exists(oi3fold['pcsf_run_dir']):
            print('Error : pcsf_run_dir does not exist!')
            print(oi3fold['pcsf_run_dir'])
            raise Exception('Error : pcsf_run_dir does not exist! : ' + oi3fold['pcsf_run_dir'])
        
        if not os.path.exists(oi3fold['oi_network_dir']):
            # print('Creating directory oi_network_dir: ', oi3fold['oi_network_dir'])
            os.makedirs(oi3fold['oi_network_dir'])
    
        shutil.copy(oi3fold['prizeDataFinal_fp'], path.join(oi3fold['oi_network_dir'], os.path.basename(oi3fold['prizeDataFinal_fp'])))
        shutil.copy(oi3fold['prizeNameInfo_fp'], path.join(oi3fold['oi_network_dir'], os.path.basename(oi3fold['prizeNameInfo_fp'])))
        shutil.copy(oi3fold['prizes_all_fp'], path.join(oi3fold['oi_network_dir'], os.path.basename(oi3fold['prizes_all_fp'])))
        shutil.copy(oi3fold['oirun_interactome_fp'], path.join(oi3fold['oi_network_dir'], os.path.basename(oi3fold['oirun_interactome_fp'])))

        # move hyperparam summary results if they exist
        if oi3fold['save_hyperparam_summary']:
            for tuningPhenotype in oi3fold['tuningPhenotypes']:
                fn = tuningPhenotype.replace(' ','_').replace('.','_') + '_' + oi3fold['fn_suffix_hyperparam_summary'] + '.csv'
                if os.path.exists(path.join(oi3fold['pcsf_run_dir'], fn)):
                    shutil.copy(path.join(oi3fold['pcsf_run_dir'], fn), path.join(oi3fold['oi_network_dir'], fn))

 
        # assemble PN and cluster to generate PNMs
        numNodesPN = analyze_oi_runs(oi3fold)

        if numNodesPN == 0:
            print('No PN and PNMs were generated!')
            

        ####### remove the pcsf_run_dir directory purely to save disk space ########################
        if oi3fold['DELETE_PCSF_RUN_POST_PROCESSING']:
            if os.path.isdir(oi3fold['pcsf_run_dir']):
                shutil.rmtree(oi3fold['pcsf_run_dir'])

        ####### remove the OI run input_measures_dir directory purely to save disk space ########################
        if oi3fold['DELETE_PCSF_RUN_MEASURES_POST_PROCESSING']:
            if os.path.isdir(oi3fold['input_measures_dir']):
                shutil.rmtree(oi3fold['input_measures_dir'])
        
 
        print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + oi3fold['FOLD_DIR_NAME'] + '  ' + 'PROCESS_OI_RUNS')
        print('')

    return oi3fold

    
#################################################################################################################################################################
#####
#################################################################################################################################################################

def _fold_impute_test_data(oi3fold):

    
    findx = oi3fold['findx']
    train_index = oi3fold['train_index']
    test_index = oi3fold['test_index']
    dfnp = oi3fold['dfnp']
    df_NoOutcome = oi3fold['df_NoOutcome']
    feature_cols = oi3fold['feature_cols']
    phenotype_feature_cols = oi3fold['phenotype_feature_cols']
    molecular_feature_cols = oi3fold['molecular_feature_cols']
    metabolite_feature_cols = oi3fold['metabolite_feature_cols']
    protein_feature_cols = oi3fold['protein_feature_cols']
    rna_feature_cols = oi3fold['rna_feature_cols']
    dna_feature_cols = oi3fold['dna_feature_cols']
    # source_interactome = oi3fold['source_interactome']

    all_cols =  oi3fold['all_cols']
    info_cols = oi3fold['info_cols']


    randomState = oi3fold['randomState']

    df_train_fold_imputed = oi3fold['df_train_fold_imputed']
    

    # get PA measures for training subjects
    fn = oi3fold['FOLD_DIR_NAME'] + '_PA_Measures.csv'
    if os.path.exists(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], fn)):
        pa_measures = pd.read_csv(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], fn))
    
        pa_measures_cols = pa_measures.columns.to_list()
        pa_cols = [x for x in pa_measures_cols if x not in ['UID']]
    
        df_train_fold_imputed = pd.merge(df_train_fold_imputed, pa_measures, on='UID', how='left')

    # get EA measures for training subjects
    fn = oi3fold['FOLD_DIR_NAME'] + '_EA_Measures.csv'
    if os.path.exists(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], fn)):
        ea_measures = pd.read_csv(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], fn))
    
        ea_measures_cols = ea_measures.columns.to_list()
        ea_cols = [x for x in ea_measures_cols if x not in ['UID']]
        
        df_train_fold_imputed = pd.merge(df_train_fold_imputed, ea_measures, on='UID', how='left')
    

 
    # save training data that now includes PAs and MAs and EAs out to fold directory for R script
    FN_CV_FOLD_TRAIN_IMPUTED = path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], (oi3fold['FOLD_DIR_NAME'] + '_train_features_imputed.csv'))
    df_train_fold_imputed.to_csv(FN_CV_FOLD_TRAIN_IMPUTED, index=False, sep=',')


    if (len(test_index) > 0) or (len(df_NoOutcome) > 0 and oi3fold['include_hold_out_test_subjects']):

        if oi3fold['imputeTestDataWithTrainingSet']:


            if oi3fold['imputeByClass']:

                # get the data for the test subjects
                if len(test_index) > 0:
                    df_test_fold = pd.DataFrame(dfnp[test_index],columns=all_cols)
                else:
                    df_test_fold = pd.DataFrame(df_NoOutcome,columns=all_cols)
                    
                df_test_fold_info = df_test_fold[info_cols].copy()
                df_test_fold_info = df_test_fold_info.reset_index(drop = True)

                dfList = [df_test_fold_info]

                # append the imputed training data
                df_test_with_train_fold = pd.concat([df_test_fold,df_train_fold_imputed[all_cols].copy()],axis=0)
                
                if len(phenotype_feature_cols) > 0:
                    dfimpute = pd.DataFrame(impute_data_knn(df_test_with_train_fold[phenotype_feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=phenotype_feature_cols)
                    # now only keep the top rows that are the test subjects
                    dfimpute = dfimpute.iloc[:len(test_index)]
                    dfimpute = dfimpute.reset_index(drop = True)
                    dfList = dfList + [dfimpute]
        
                if len(metabolite_feature_cols) > 0:
                    dfimpute = pd.DataFrame(impute_data_knn(df_test_with_train_fold[metabolite_feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=metabolite_feature_cols)
                    # now only keep the top rows that are the test subjects
                    dfimpute = dfimpute.iloc[:len(test_index)]
                    dfimpute = dfimpute.reset_index(drop = True)
                    dfList = dfList + [dfimpute]
        
                if len(protein_feature_cols) > 0:
                    dfimpute = pd.DataFrame(impute_data_knn(df_test_with_train_fold[protein_feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=protein_feature_cols)
                    # now only keep the top rows that are the test subjects
                    dfimpute = dfimpute.iloc[:len(test_index)]
                    dfimpute = dfimpute.reset_index(drop = True)
                    dfList = dfList + [dfimpute]
        
                if len(rna_feature_cols) > 0:
                    dfimpute = pd.DataFrame(impute_data_knn(df_test_with_train_fold[rna_feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=rna_feature_cols)
                    # now only keep the top rows that are the test subjects
                    dfimpute = dfimpute.iloc[:len(test_index)]
                    dfimpute = dfimpute.reset_index(drop = True)
                    dfList = dfList + [dfimpute]
        
                if len(dna_feature_cols) > 0:
                    dfimpute = pd.DataFrame(impute_data_knn(df_test_with_train_fold[dna_feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=dna_feature_cols)
                    # now only keep the top rows that are the test subjects
                    dfimpute = dfimpute.iloc[:len(test_index)]
                    dfimpute = dfimpute.reset_index(drop = True)
                    dfList = dfList + [dfimpute]
        
                df_test_fold_imputed = pd.concat(dfList,axis=1)
        
            else:  # oi3fold['imputeByClass'] == False

                # get the data for the test subjects
                if len(test_index) > 0:
                    df_test_fold = pd.DataFrame(dfnp[test_index],columns=all_cols)
                else:
                    df_test_fold = pd.DataFrame(df_NoOutcome,columns=all_cols)
                # append the imputed training data
                df_test_with_train_fold = pd.concat([df_test_fold,df_train_fold_imputed[all_cols].copy()],axis=0)
                dfimpute = pd.DataFrame(impute_data_knn(df_test_with_train_fold[feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=feature_cols)
                # now only keep the top rows that are the test subjects
                dfimpute = dfimpute.iloc[:len(test_index)]
                df_test_fold_info = df_test_fold[info_cols].copy()
                df_test_fold_info = df_test_fold_info.reset_index(drop = True)
                dfimpute = dfimpute.reset_index(drop = True)
                df_test_fold_imputed = pd.concat([df_test_fold_info,dfimpute],axis=1)
       
        else:  # oi3fold['imputeTestDataWithTrainingSet'] == False

            if oi3fold['imputeByClass']:

                if len(test_index) > 0:
                    df_test_fold = pd.DataFrame(dfnp[test_index],columns=all_cols)
                else:
                    df_test_fold = pd.DataFrame(df_NoOutcome,columns=all_cols)
                    
                df_test_fold_info = df_test_fold[info_cols].copy()
                df_test_fold_info = df_test_fold_info.reset_index(drop = True)
        
                dfList = [df_test_fold_info]
                
                if len(phenotype_feature_cols) > 0:
                    dfimpute = pd.DataFrame(impute_data_knn(df_test_fold[phenotype_feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=phenotype_feature_cols)
                    dfimpute = dfimpute.reset_index(drop = True)
                    dfList = dfList + [dfimpute]
        
                if len(metabolite_feature_cols) > 0:
                    dfimpute = pd.DataFrame(impute_data_knn(df_test_fold[metabolite_feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=metabolite_feature_cols)
                    dfimpute = dfimpute.reset_index(drop = True)
                    dfList = dfList + [dfimpute]
        
                if len(protein_feature_cols) > 0:
                    dfimpute = pd.DataFrame(impute_data_knn(df_test_fold[protein_feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=protein_feature_cols)
                    dfimpute = dfimpute.reset_index(drop = True)
                    dfList = dfList + [dfimpute]
        
                if len(rna_feature_cols) > 0:
                    dfimpute = pd.DataFrame(impute_data_knn(df_test_fold[rna_feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=rna_feature_cols)
                    dfimpute = dfimpute.reset_index(drop = True)
                    dfList = dfList + [dfimpute]
        
                if len(dna_feature_cols) > 0:
                    dfimpute = pd.DataFrame(impute_data_knn(df_test_fold[dna_feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=dna_feature_cols)
                    dfimpute = dfimpute.reset_index(drop = True)
                    dfList = dfList + [dfimpute]
        
                df_test_fold_imputed = pd.concat(dfList,axis=1)
        
            else:   # oi3fold['imputeByClass'] == False

                # get the data for the test subjects
                if len(test_index) > 0:
                    df_test_fold = pd.DataFrame(dfnp[test_index],columns=all_cols)
                else:
                    df_test_fold = pd.DataFrame(df_NoOutcome,columns=all_cols)
                dfimpute = pd.DataFrame(impute_data_knn(df_test_fold[feature_cols], n_neighbors=oi3fold['knn_num_neighbors']), columns=feature_cols)
                df_test_fold_info = df_test_fold[info_cols].copy()
                df_test_fold_info = df_test_fold_info.reset_index(drop = True)
                dfimpute = dfimpute.reset_index(drop = True)
                df_test_fold_imputed = pd.concat([df_test_fold_info,dfimpute],axis=1)


        # save imputed test data to fold directory for R script
        FN_CV_FOLD_TEST_IMPUTED = path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], (oi3fold['FOLD_DIR_NAME'] + '_test_features_imputed.csv'))
        df_test_fold_imputed.to_csv(FN_CV_FOLD_TEST_IMPUTED, index=False, sep=',')

        if oi3fold['build_PAs'] or oi3fold['build_PN_With_PAs']:
            FN_CV_FOLD_TEST_PA_FEATURES = path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], (oi3fold['FOLD_DIR_NAME'] + '_test_PA_features.csv'))

            # save imputed testing data for phenotypes to compute PAs to fold directory for R script
            pa_phenotype_features = [cn for cn in phenotype_feature_cols if cn not in oi3fold['remove_phenotype_PA_feature_list']]
            # make sure to include the LATENT_METRIC_LIST
            cns = getUniqueKeepOrder(info_cols + [oi3fold['OUTCOME_METRIC']] + oi3fold['LATENT_METRIC_LIST'] + pa_phenotype_features)
            df_test_fold_PA_features = df_test_fold_imputed[cns].copy()
            df_test_fold_PA_features.to_csv(FN_CV_FOLD_TEST_PA_FEATURES, index=False, sep=',')

        # set using the last data frame reference address. for example df = df.copy()  
        oi3fold = { **oi3fold, 'df_train_fold_imputed': df_train_fold_imputed}
        oi3fold = { **oi3fold, 'df_test_fold_imputed': df_test_fold_imputed}
        
    return oi3fold

#################################################################################################################################################################
#####
#################################################################################################################################################################

def _fold_generate_base_PNM_EA_models(oi3fold):

    base_model_name_cn = 'EA_Model_Name'

    info_cols = oi3fold['info_cols']
    randomState = oi3fold['randomState']
    df_train_fold_imputed = oi3fold['df_train_fold_imputed']
    df_test_fold_imputed = oi3fold['df_test_fold_imputed']

    if oi3fold['BUILD_BASE_PNM_EA_MODELS']:
        
        print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + oi3fold['FOLD_DIR_NAME']  + '  ' + 'BUILD_BASE_PNM_EA_MODELS')

        import pyreadr    # for writing to .RData files.
        # Save df_train_fold_imputed to .RData file to allow rapid import to R source : PhenoMol_sPLS_PNMs_v1.R
        pyreadr.write_rdata(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], (oi3fold['FOLD_DIR_NAME'] + '_train_features_imputed.RData')), df_train_fold_imputed, df_name='df_train_fold_imputed') 
                                

        if oi3fold['COMPUTE_OUTCOME_BASE_EA_MODEL']:
            Y_METRICS = [oi3fold['OUTCOME_METRIC']] + oi3fold['LATENT_METRIC_LIST']
        else:
            Y_METRICS = oi3fold['LATENT_METRIC_LIST']
            if len(oi3fold['LATENT_METRIC_LIST']) == 0:
                print('LATENT_METRIC_LIST = [] and COMPUTE_OUTCOME_BASE_EA_MODEL = False. Either add y metrics to list or change boolean to True.')
                # break

            
        # R script to build Molecular Expression Axis (EA) Models
        script_path = oi3fold['RCODE_SOURCE_DIR'] + '/' + 'PhenoMol_sPLS_PNMs_v1.R'
        args = [oi3fold['R_LIBRARY_DIR'], oi3fold['RCODE_SOURCE_DIR'], oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], oi3fold['prize_data_raw_fn']]
        args += ['RESULTS_FILENAME=' + BASE_PNM_EA_MODEL_CSV]
        Y_METRICS_STRING = ','.join(Y_METRICS)
        args += ['Y_METRICS=' + Y_METRICS_STRING]
        # args +=  ['Y_METRIC=' + oi3fold['OUTCOME_METRIC']]
        args += ['BINARY_OUTCOME=' + str(oi3fold['BINARY_OUTCOME']).upper()]
        args += ['MIN_NUM_MEASURES_PER_FEATURE=' + str(oi3fold['MIN_NUM_MEASURES_PER_FEATURE'])]
        args += ['EA_MAX_N_FEATURES=' + str(oi3fold['EA_MAX_N_FEATURES'])]
        args += ['TUNE_SPLS_MEASURE=' + str(oi3fold['TUNE_SPLS_MEASURE'])]
        args += ['TUNE_SPLS_NUM_FOLDS=' + str(oi3fold['TUNE_SPLS_NUM_FOLDS'])]
        args += ['TUNE_SPLS_NUM_REPEATS=' + str(oi3fold['TUNE_SPLS_NUM_REPEATS'])]
        args += ['TUNE_SPLS_MIN_FEATURE_STABILITY=' + str(oi3fold['TUNE_SPLS_MIN_FEATURE_STABILITY'])]
        args += ['SPLS_RANDOM_SEED=' + str(oi3fold['SPLS_RANDOM_SEED'])]
        args += ['INCLUDE_PNM_MINUS_99=' + str(oi3fold['INCLUDE_PNM_MINUS_99']).upper()]
        args += ['INCLUDE_PNM0=' + str(oi3fold['INCLUDE_PNM0']).upper()]
        r_path = oi3fold['r_path']
        cmd = [r_path, script_path]  + args
        result = subprocess.check_output(cmd, universal_newlines=True)
        
        # print(result)

        # Not needed if ensemble models are built which will include single base models in the output
        if oi3fold['GENERATE_BASE_PNM_EA_MODEL_TEST_RESULTS']:
            
            df_Base_PNM_EA_models = pd.read_csv(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], BASE_PNM_EA_MODEL_CSV))
    
            base_EA_models = pd.concat([df_train_fold_imputed[info_cols].copy(),df_test_fold_imputed[info_cols].copy()],axis=0)
            
            cv_train_R2 = []
            cv_train_mse = []
            pls_test_R2 = []
            pls_test_mse = []
            for ind in df_Base_PNM_EA_models.index:
                EA_model_features = getUniqueKeepOrder((df_Base_PNM_EA_models['Features'][ind].split(",")))
                Y_METRIC = df_Base_PNM_EA_models['Y_METRIC'][ind]

                
                X_train = df_train_fold_imputed[EA_model_features].to_numpy()
                y_train = df_train_fold_imputed[Y_METRIC].astype(float)
            
                n_components = 1
                pls_model = PLSRegression(n_components=n_components)
                Y_cv = cross_val_predict(pls_model, X_train, y_train, cv=oi3fold['num_cv_folds'])
                R2_cv = r2_score(y_train, Y_cv)
                mse_cv = mean_squared_error(y_train, Y_cv)
                
                cv_train_R2 += [R2_cv]
                cv_train_mse += [mse_cv]
    
                pls_model.fit(X_train, y_train)
                y_pred_train = pls_model.predict(X_train)
        
                X_test = df_test_fold_imputed[EA_model_features].to_numpy()
                y_test = df_test_fold_imputed[Y_METRIC].astype(float)
                
                y_pred_test = pls_model.predict(X_test)
    
                test_score = r2_score(y_test, y_pred_test)
                test_mse = mean_squared_error(y_test, y_pred_test)
                
                pls_test_R2 += [test_score]
                pls_test_mse += [test_mse]
                
                EA_Model_Name = df_Base_PNM_EA_models[base_model_name_cn][ind]
                
                base_EA_models[EA_Model_Name] = list(y_pred_train) + list(y_pred_test) 
    
                # print(df_Base_PNM_EA_models[base_model_name_cn][ind],'N',df_Base_PNM_EA_models['TotalNumFeatures'][ind],'sPLS_N_keep',df_Base_PNM_EA_models['NumFeaturesUsed'][ind], f"train_cv_mse: {mse_cv:.3f}" , f"test_mse: {test_mse:.3f}")

            df_Base_PNM_EA_models['cv_train_R2'] = cv_train_R2
            df_Base_PNM_EA_models['cv_train_mse'] = cv_train_mse   
            
            df_Base_PNM_EA_models['test_R2'] = pls_test_R2
            df_Base_PNM_EA_models['test_mse'] = pls_test_mse    
            
            df_Base_PNM_EA_models.to_csv(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], BASE_PNM_EA_MODEL_CSV), index=False, sep=',')
    
            base_EA_models.to_csv(path.join(oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], 'base_EA_measures.csv'), index=False, sep=',')
        
        print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + oi3fold['FOLD_DIR_NAME']  + '  ' + 'BUILD_BASE_PNM_EA_MODELS')
        
    return oi3fold


    
#################################################################################################################################################################
#####
#################################################################################################################################################################

def build_ensemble_models(runParm, modelType = 'PNM_EA', base_model_name_cn = 'EA_Model_Name', base_models_fn = BASE_PNM_EA_MODEL_CSV, ensemble_models_fn = 'ensemble_PNM_EA_models.csv', train_suffix_fn = '_train_features_imputed.csv', test_suffix_fn = '_test_features_imputed.csv'):

    stepStartTime = datetime.datetime.now()

    randomState = runParm['randomState']
    
    # Build Ensemble Models for each Y Metric (outcome + latent metrics)
    
    print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Build Ensemble ' + modelType + ' Models')

    param_sets = []
    for findx in range(runParm['NUM_OI_FOLDS']):
    
        # np.random.seed(randomState + findx)
    
        fold_num = findx + 1
        
        FOLD_DIR_NAME = getDirName(fold_num)

    
        FN_CV_FOLD_TRAIN_IMPUTED = path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, (FOLD_DIR_NAME + train_suffix_fn))
        df_train_fold_imputed = pd.read_csv(FN_CV_FOLD_TRAIN_IMPUTED)
        
        FN_CV_FOLD_TEST_IMPUTED = path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, (FOLD_DIR_NAME + test_suffix_fn))
        df_test_fold_imputed = pd.read_csv(FN_CV_FOLD_TEST_IMPUTED)
    
    
        df_All_Base_Models = pd.read_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, base_models_fn))

        if len(df_All_Base_Models) == 0:
            print('There are no base models!')
            print('File is empty!')
            print(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, base_models_fn))
            print('Model R2 = 0 for permutation testing because no model generated.')
            return 0
            
                        
        Y_METRICS = [runParm['OUTCOME_METRIC']] + runParm['LATENT_METRIC_LIST']

        comp_nck_list = []
        comp_y_metric_list = []
        
        for Y_METRIC in Y_METRICS:
    
            df_base_models = df_All_Base_Models.loc[df_All_Base_Models['Y_METRIC'] == Y_METRIC].copy()
        
            # nck to generate a list of ensemble models
            nck_list = []
        
            sfl = df_base_models[base_model_name_cn].tolist()
            pnm99s = [x for x in sfl if x.endswith('_PNM-99')]
            # now remove any pnm99s from the list sfl
            sfl = [x for x in sfl if not x.endswith('_PNM-99')]
            max_k_features = min(runParm['maxNumBaseModels'], len(sfl)) 
            for ki in range(0,max_k_features):
                k_features = ki + 1
                nck = choosek(sfl, k_features)
                if len(nck) > runParm['maxNumNCKruns']:
                    break
                nck_list += nck
            # now add any pnm99s as a single feature ensemble to the begining of nck_list
            if len(pnm99s) > 0:
                nck_list = choosek(pnm99s, 1) + nck_list

            y_metric_list = []
            for i in range(len(nck_list)):
                y_metric_list += [Y_METRIC]

            comp_nck_list += nck_list
            comp_y_metric_list += y_metric_list

        params = {
            # 'max_num_cpus': runParm['max_num_cpus'],
            'job_id': '1',
            'proc_id': findx,
            'max_num_cpus': 1,
            'randomState': randomState,
            'num_cv_folds': runParm['num_cv_folds'],
            'numFoldGenerations': runParm['numFoldGenerations'],
            'df_train_fold_imputed': df_train_fold_imputed,
            'df_test_fold_imputed': df_test_fold_imputed, 
            'df_All_Base_Models': df_All_Base_Models,
            'base_model_name_cn': base_model_name_cn,
            # 'y_metric_list':comp_y_metric_list,
            # 'nck_list': comp_nck_list 
            'sub_y_metric_list':comp_y_metric_list,
            'sub_nck_list': comp_nck_list 
        }
        
        param_sets += [params]


        
    if runParm['parallelize_folds']:

        num_tasks = len(param_sets)

        n_request_cpus = num_tasks
        
        if n_request_cpus < 1:
            n_request_cpus = 1
        elif n_request_cpus > int(runParm['max_num_cpus']):
            n_request_cpus = int(runParm['max_num_cpus'])


        print('')
        print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' # tasks/processors: ' + str(num_tasks) + '/' + str(n_request_cpus) + '  ' + 'Generate:', (len(nck_list) * len(Y_METRICS)), ' Ensemble Models')

        pool = multiprocessing.Pool(n_request_cpus)
        results = pool.map(_nck_TrainTestEnsembleModel, param_sets)
        
        # can ignore results

        print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Build Ensemble ' + modelType + ' Models')


        for findx in range(runParm['NUM_OI_FOLDS']):

            fold_num = findx + 1
        
            FOLD_DIR_NAME = getDirName(fold_num)

            df_ensemble_models = results[findx]
            
            if len(df_ensemble_models) > 0:
                df_ensemble_models.to_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, ensemble_models_fn),index=False)
            else:
                print('No ensemble models were generated!')

        
    else:

        for findx in range(len(fold_FOLD_DIR_NAME)):

            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + FOLD_DIR_NAME + ' '  +  'Generate:', (len(nck_list) * len(Y_METRICS)), ' Ensemble Models')
            params = param_sets[findx]
            df_ensemble_models = _nck_TrainTestEnsembleModel(params)

            if len(df_ensemble_models) > 0:
                df_ensemble_models.to_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, ensemble_models_fn),index=False)
            else:
                print('No ensemble models were generated!')


    print('')
    print('Completed Build Ensemble ' + modelType + ' Models')
    print('  End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Start:' + stepStartTime.strftime('%Y-%m-%d %H:%M:%S'))
    print('')




#################################################################################################################################################################
#####
################################################################################################################################################################# 
    
def run_analysis_step_1(runParm):

    stepStartTime = datetime.datetime.now()

    base_dir = runParm['base_dir']
    max_num_cpus = runParm['max_num_cpus']

    randomState = runParm['randomState']
    

    misclassifyOutcomeFraction = runParm['misclassifyOutcomeFraction']

    interactome_db_dir = runParm['interactome_db_dir']
    interactomeName = runParm['interactomeName']

    info_cols = runParm['info_cols']
    analysis_dir = runParm['analysis_dir']
    analysis_results_dir = runParm['analysis_results_dir']

    OUTCOME_METRIC = runParm['OUTCOME_METRIC']
        


    

    # load in interactome
    
    source_interactome = Interactome(runParm)
    source_interactome.load_interactome_edges()
    source_interactome.load_interactome_links()
    source_interactome.load_interactome_node_info()
    
    # add to parameters
    runParm = { **runParm, 'source_interactome': source_interactome}
    
    

    
    
    # Load in subjects and raw feature data from file generated by R that includes phenotypes and molecular measures that may include missing values
    df = pd.read_csv(path.join(runParm['analysis_results_dir'], 'all_features_raw.csv'))
    
    all_cols = df.columns.to_list()
    feature_cols = [x for x in all_cols if x not in info_cols]
    
    print('label: ' + OUTCOME_METRIC)
    print('number of participants (with or without a label): ' + str(len(df.index)))
    print('number of participants with a label: ' +  str(len(df[df['outcome'].notna()])))
    
    
    df_NoOutcome = df[df['outcome'].isna()].copy()
    df_NoOutcome = df_NoOutcome.reset_index(drop = True)
    
    df = df[df['outcome'].notna()]
    df = df.reset_index(drop = True)
    
    all_cols = df.columns.to_list()
    feature_cols = [x for x in all_cols if x not in info_cols]
    
    # Need to remove features that have greater than 40% missing values
    
    critNumNAs = 0.6 * df.shape[0]
    remove_cns = []
    for cn in feature_cols:
        if df[cn].isna().sum() > critNumNAs:
            remove_cns = remove_cns + [cn]
    df = df.drop(remove_cns,axis=1)  
    df = df.reset_index(drop = True)
    
    all_cols = df.columns.to_list()
    feature_cols = [x for x in all_cols if x not in info_cols]

    # Need to remove features that have no variation
    
    # from sklearn.feature_selection import VarianceThreshold
    # sel = VarianceThreshold(threshold=0.05)
    # X_selection = sel.fit_transform(X)
    
    remove_cns = []
    for cn in feature_cols:
        d = df[cn]
        d = d[d.notna()]
        if d.std() < 1e-8:
            remove_cns = remove_cns + [cn]
            
    df = df.drop(remove_cns,axis=1)  
    df = df.reset_index(drop = True)
    
    all_cols = df.columns.to_list()
    feature_cols = [x for x in all_cols if x not in info_cols]
    
    if runParm['Only_Outcome_No_Phenotypes']:
        phenotype_feature_cols = []
    else:
        phenotype_features_info = pd.read_csv(path.join(runParm['analysis_results_dir'], 'phenotype_features_info.csv'))
        phenotype_feature_cols = list(phenotype_features_info['featureName'])
        phenotype_feature_cols = [x for x in phenotype_feature_cols if x in feature_cols]
    
    molecular_features_info = pd.read_csv(path.join(runParm['analysis_results_dir'], 'molecular_features_info.csv'))
    molecular_feature_cols = list(molecular_features_info['featureName'])
    molecular_feature_cols = [x for x in molecular_feature_cols if x in feature_cols]
    
    metabolite_feature_cols = list(molecular_features_info[molecular_features_info['nodeSpecies']=='Metabolite']['featureName'].values)
    metabolite_feature_cols = [x for x in metabolite_feature_cols if x in feature_cols]
    
    protein_feature_cols = list(molecular_features_info[molecular_features_info['nodeSpecies']=='Protein']['featureName'].values)
    protein_feature_cols = [x for x in protein_feature_cols if x in feature_cols]

    rna_feature_cols = list(molecular_features_info[molecular_features_info['nodeSpecies']=='RNA']['featureName'].values)
    rna_feature_cols = [x for x in rna_feature_cols if x in feature_cols]

    dna_feature_cols = list(molecular_features_info[molecular_features_info['nodeSpecies']=='DNA']['featureName'].values)
    dna_feature_cols = [x for x in dna_feature_cols if x in feature_cols]

    # info_cols
    # feature_cols
    print()
    print('Number of info columns:', len(info_cols))
    print()
    print('Number of phenotype features:', len(phenotype_feature_cols))
    print()
    print('number of metabolite feature cols:',len(metabolite_feature_cols))
    print('number of protein feature cols:',len(protein_feature_cols))
    print('number of RNA feature cols:',len(rna_feature_cols))
    print('number of DNA feature cols:',len(dna_feature_cols))
    print()
    print('Number of molecular features:',len(molecular_feature_cols))
    print()
    print('Number of features:', len(feature_cols))
    
    
    # In[6]:
    
    
    np.random.seed(randomState)
    
    
    
    
    
    fold_FOLD_DIR_NAME = []
    fold_train_index = []
    fold_test_index = []
    
    
    dfnp = df.to_numpy()
    
    
    if runParm['create_All_fold']:
        # append the All which is using all of the data for training.
        # If there are subjects with no outcome, (see df_NoOutcome) their predicted values will be generated if include_hold_out_test_subjects = True
        
        FOLD_DIR_NAME = 'All'
        print(FOLD_DIR_NAME)
        train_index = list(df.index)
        test_index = []
        
        fold_FOLD_DIR_NAME = fold_FOLD_DIR_NAME + [FOLD_DIR_NAME]
        fold_train_index = fold_train_index + [train_index]
        fold_test_index = fold_test_index + [test_index]
        
        print('-------------------------------------------------------------------')
        print(FOLD_DIR_NAME)
        print('TRAIN N =', len(train_index),train_index)
        print('TEST N =', len(test_index), test_index)  
        print('HOLD-OUT TEST N =', len(df_NoOutcome))  
    
    
    if runParm['create_CV_test_folds']:

        uids = df['UID'].to_numpy()

        # Special case where the test train splits are given in a data frame
        if 'df_test_train_splits' in runParm and isinstance(runParm['df_test_train_splits'], pd.DataFrame) and not runParm['df_test_train_splits'].empty:

            df_test_train_splits = runParm['df_test_train_splits']
            df_test_train_splits = df_test_train_splits[df_test_train_splits['uid'].isin(uids)].copy()
            
            fold_num = 1
            for findx in range(runParm['NUM_OI_FOLDS']):
        
                FOLD_DIR_NAME = getDirName(fold_num)
        
                train_uids = df_test_train_splits.loc[df_test_train_splits[FOLD_DIR_NAME] == 0, 'uid'].tolist()
                test_uids = df_test_train_splits.loc[df_test_train_splits[FOLD_DIR_NAME] == 1, 'uid'].tolist()
        
                train_index = [index for index,value in enumerate(uids) if value in train_uids]
                test_index = [index for index,value in enumerate(uids) if value in test_uids]
                
                print('-------------------------------------------------------------------')
                print(FOLD_DIR_NAME)
                print('TRAIN N =', len(train_index),train_index)
                print('TEST N =', len(test_index), test_index)   
        
                df_test_train_splits[FOLD_DIR_NAME] = 0
                df_test_train_splits.loc[df_test_train_splits['uid'].isin(uids[test_index]), FOLD_DIR_NAME] = 1
                
                fold_FOLD_DIR_NAME = fold_FOLD_DIR_NAME + [FOLD_DIR_NAME]
                fold_train_index = fold_train_index + [train_index]
                fold_test_index = fold_test_index + [test_index]
                fold_num = fold_num + 1

            # save out for recording purposes    
            df_test_train_splits.to_csv(path.join(analysis_results_dir, 'cv_test_train_splits.csv'),index=False)
            
        else:
            # Standard case where the test train splits are generated using the given randomState
            
            df_test_train_splits = pd.DataFrame(uids, columns=['uid']) 
    
            X = df[feature_cols].to_numpy()
            X = X.astype(float)
            
            res = np.unique(df['outcome'])
            if len(res)==2:
                ybin = np.array(df['outcome'], dtype=int)
            else:
                medianValue = df['outcome'].median()
                meanValue = df['outcome'].mean()
                v = df['outcome'].tolist()
                if meanValue <= medianValue:
                    ybin = list(map(lambda x: 1 if x >= medianValue else 0, v))
                else:
                    ybin = list(map(lambda x: 0 if x <= medianValue else 1, v))
                ybin = np.array(ybin, dtype=int)
            
            # skf = StratifiedKFold(n_splits=num_Testing_Splits) # note that for default shuffle=False which could be a bad thing
            skf = StratifiedKFold(n_splits=runParm['NUM_OI_FOLDS'],shuffle=True, random_state = randomState)
            skf.get_n_splits(X, ybin)
    
            fold_num = 1
            for train_index, test_index in skf.split(X, ybin):
                FOLD_DIR_NAME = getDirName(fold_num)
                
                print('-------------------------------------------------------------------')
                print(FOLD_DIR_NAME)
                print('TRAIN N =', len(train_index),train_index)
                print('TEST N =', len(test_index), test_index)   
    
                df_test_train_splits[FOLD_DIR_NAME] = 0
                df_test_train_splits.loc[df_test_train_splits['uid'].isin(uids[test_index]), FOLD_DIR_NAME] = 1
        
                fold_FOLD_DIR_NAME = fold_FOLD_DIR_NAME + [FOLD_DIR_NAME]
                fold_train_index = fold_train_index + [train_index]
                fold_test_index = fold_test_index + [test_index]
                fold_num = fold_num + 1
    
            # save out for recording purposes    
            df_test_train_splits.to_csv(path.join(analysis_results_dir, 'cv_test_train_splits.csv'),index=False)
        

    folds_oi3fold_parmsets = []
    for findx in range(len(fold_FOLD_DIR_NAME)):
    
        FOLD_DIR_NAME = fold_FOLD_DIR_NAME[findx]
        train_index = fold_train_index[findx]
        test_index = fold_test_index[findx]
    
        measures_dir = path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, 'oi_train_measures')
        oi_prize_dir = path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, 'oi_prizes')
        pcsf_run_dir = path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, 'oi_runs')
        oi_network_dir = path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, 'oi_network')

        prize_data_raw_fn = runParm['prize_data_raw_fn']
        prize_name_info_fn = runParm['prize_name_info_fn']
        prize_data_final_fn = runParm['prize_data_final_fn']

            
        oi3fold = runParm.copy()
        
        oi3fold = { **oi3fold,   
            'FOLD_DIR_NAME': FOLD_DIR_NAME, 
            'findx': findx,
            'train_index': train_index,
            'test_index': test_index,
            'dfnp': dfnp,    
            'df_NoOutcome': df_NoOutcome,
            'all_cols': all_cols,    
            'feature_cols': feature_cols,    
            'phenotype_feature_cols': phenotype_feature_cols,
            'molecular_feature_cols': molecular_feature_cols,
            'metabolite_feature_cols': metabolite_feature_cols,
            'protein_feature_cols': protein_feature_cols,
            'rna_feature_cols': rna_feature_cols,
            'dna_feature_cols': dna_feature_cols,
            'source_interactome': source_interactome,
    
            'oi_prize_dir': oi_prize_dir, 
            'pcsf_run_dir': pcsf_run_dir, 
            'oi_network_dir' : oi_network_dir,
            'prize_data_raw_fp': path.join(oi_prize_dir, prize_data_raw_fn),
            'prize_name_info_fp': path.join(oi_prize_dir, prize_name_info_fn),
            'prize_data_final_fp': path.join(oi_prize_dir, prize_data_final_fn),
            
            'prizeDataFinal_fp': path.join(pcsf_run_dir, prize_data_final_fn),
            'prizeNameInfo_fp': path.join(pcsf_run_dir, prize_name_info_fn),
            'oirun_interactome_fp': path.join(pcsf_run_dir, interactomeName + '_interactome.zip'),
            'prizes_all_fp': path.join(pcsf_run_dir, interactomeName + '_prizes_all.zip'),
    
            'input_measures_dir': measures_dir,
                   
            'phenotypeMeasures': ['mDat_phenotypes.csv'],
            'phenotype_mDat_prefix': 'mDat_',
            'omicMeasures': ['mDat_molecular.csv'], 
            'omic_mDat_prefix': 'mDat_'}

        folds_oi3fold_parmsets += [oi3fold]

    maxFoldNumProcessors = int(runParm['max_num_cpus'])
    
    ########################################################################
    ###### _fold_assemble_training_data
    ########################################################################   
    
    # ASSEMBLE_PRIZE_DATA
    # note that is routine also sets up data frames and generates PA models if required

    if runParm['parallelize_folds']:

        num_tasks = runParm['NUM_OI_FOLDS']

        n_request_cpus = num_tasks
        
        if n_request_cpus < 1:
            n_request_cpus = 1
        elif n_request_cpus > int(runParm['max_num_cpus']):
            n_request_cpus = int(runParm['max_num_cpus'])


        for i in range(len(folds_oi3fold_parmsets)):
            oi3fold = folds_oi3fold_parmsets[i]     
            oi3fold = { **oi3fold, 'max_num_cpus': 1 } 
            folds_oi3fold_parmsets[i] = oi3fold

        infostr = 'Impute fold training data'
        if oi3fold['build_PAs'] or oi3fold['build_PN_With_PAs']:
            infostr += ' and build PA models'

        print('')
        print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' # tasks/processors: ' + str(num_tasks) + '/' + str(n_request_cpus) + '  ' + infostr)

        pool = multiprocessing.Pool(n_request_cpus)
        results = pool.map(_fold_assemble_training_data, folds_oi3fold_parmsets)
        
        folds_oi3fold_parmsets = results

        print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + infostr)
        
    else:
        
        for findx in range(len(fold_FOLD_DIR_NAME)):

            oi3fold = folds_oi3fold_parmsets[findx]
            oi3fold = { **oi3fold, 'max_num_cpus': maxFoldNumProcessors } 
            
            oi3fold = _fold_assemble_training_data(oi3fold)

            folds_oi3fold_parmsets[findx] = oi3fold


    ########################################################################
    ###### generate_oiPrizes 
    ########################################################################   

    if oi3fold['GENERATE_RAW_PRIZES']:
        
        if runParm['parallelize_folds']:
            # generate parameter set for all folds and for each individual prizeName
            param_sets = []
            for findx in range(len(fold_FOLD_DIR_NAME)):
                oi3fold = folds_oi3fold_parmsets[findx]
                results_fn_suffix = os.path.basename(oi3fold['prize_data_raw_fp'])
                i = 0
                for prizeName in oi3fold['prizeNames']:
                    params_copy = oi3fold.copy()
                    params_copy = { **params_copy, 'prizeNames': [prizeName] } 
                    params_copy = { **params_copy, 'prize_data_raw_fp': path.join(oi3fold['oi_prize_dir'], ('bin_' +  str(i+1) + '_' + results_fn_suffix))}
                    params_copy = { **params_copy, 'max_num_cpus': 1 } 
                    param_sets.append(params_copy)
                    i += 1
                    
            num_tasks = len(param_sets)
            n_request_cpus = num_tasks
            
            if n_request_cpus < 1:
                n_request_cpus = 1
            elif n_request_cpus > int(runParm['max_num_cpus']):
                n_request_cpus = int(runParm['max_num_cpus'])
    
            print('')
            print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' # tasks/processors: ' + str(num_tasks) + '/' + str(n_request_cpus) + '  ' + 'GENERATE_RAW_PRIZES')
    
            pool = multiprocessing.Pool(n_request_cpus)
            results = pool.map(generate_oiPrizes, param_sets)
            
            # can ignore results
    
            for findx in range(len(fold_FOLD_DIR_NAME)):
                oi3fold = folds_oi3fold_parmsets[findx]
                results_fn_suffix = os.path.basename(oi3fold['prize_data_raw_fp'])
                i = 0
                dfList = []
                fpList = []
                for prizeName in oi3fold['prizeNames']:
                    results_fp = path.join(oi3fold['oi_prize_dir'], ('bin_' +  str(i+1) + '_' + results_fn_suffix))
                    df = pd.read_csv(results_fp)
                    dfList = dfList + [df]
                    fpList = fpList + [results_fp]
                    i += 1
                    
                prizeData = pd.concat(dfList)
                prizeData.to_csv(oi3fold['prize_data_raw_fp'],index=False)
    
                # remove individual results files
                for results_fp in fpList:
                    if os.path.isfile(results_fp):
                        os.remove(results_fp)

            print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'GENERATE_RAW_PRIZES')
    
        else:
    
            for findx in range(len(fold_FOLD_DIR_NAME)):
    
                oi3fold = folds_oi3fold_parmsets[findx]
                oi3fold = { **oi3fold, 'max_num_cpus': maxFoldNumProcessors } 
                
                print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + oi3fold['FOLD_DIR_NAME'] + '  ' + 'GENERATE_RAW_PRIZES')
          
                generate_oiPrizes(oi3fold)
            
                print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' +  oi3fold['FOLD_DIR_NAME'] + '  ' + 'GENERATE_RAW_PRIZES')
                print('')
                    
    ########################################################################
    ###### _fold_generate_final_prizes_setup_oi_graph
    ########################################################################   

    if oi3fold['GENERATE_FINAL_PRIZES'] or oi3fold['SETUP_OI_RUNS'] or oi3fold['CREATE_OI_GRAPH_INSTANCE']:
    
        if runParm['parallelize_folds']:
    
            num_tasks = runParm['NUM_OI_FOLDS']

            n_request_cpus = num_tasks
            
            if n_request_cpus < 1:
                n_request_cpus = 1
            elif n_request_cpus > int(runParm['max_num_cpus']):
                n_request_cpus = int(runParm['max_num_cpus'])

            for i in range(len(folds_oi3fold_parmsets)):
                oi3fold = folds_oi3fold_parmsets[i]     
                oi3fold = { **oi3fold, 'max_num_cpus': 1 } 
                folds_oi3fold_parmsets[i] = oi3fold
    
            print('')
            print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' # tasks/processors: ' + str(num_tasks) + '/' + str(n_request_cpus) + '  ' + 'SETUP_OI_RUNS')
    
            pool = multiprocessing.Pool(n_request_cpus)
            results = pool.map(_fold_generate_final_prizes_setup_oi_graph, folds_oi3fold_parmsets)
            
            folds_oi3fold_parmsets = results

            print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'SETUP_OI_RUNS')

            
        else:
    
            for findx in range(len(fold_FOLD_DIR_NAME)):
            
                oi3fold = folds_oi3fold_parmsets[findx]
                oi3fold = { **oi3fold, 'max_num_cpus': maxFoldNumProcessors } 
                
                oi3fold = _fold_generate_final_prizes_setup_oi_graph(oi3fold)
    
                folds_oi3fold_parmsets[findx] = oi3fold

   
    ########################################################################
    ###### perform_oiRuns  
    ########################################################################   
    
    if oi3fold['GENERATE_OI_RUNS']:
        
        if runParm['parallelize_folds']:
    
            # generate parameter set for all folds and for each individual targetPhenotype 
            param_sets = []
            for findx in range(len(fold_FOLD_DIR_NAME)):
                oi3fold = folds_oi3fold_parmsets[findx]
                for targetPhenotype in oi3fold['targetPhenotypes']:
                    params_copy = oi3fold.copy()
                    params_copy = { **params_copy, 'targetPhenotypes': [targetPhenotype] } 
                    # params_copy = { **params_copy, 'max_num_cpus': 1 } 
    
                    graph = oi3fold['graph']
                    graph_i = OIgraph()
                    graph_i.interactome_dataframe = graph.interactome_dataframe 
                    graph_i.interactome_graph = graph.interactome_graph 
                    graph_i.nodes = graph.nodes 
                    graph_i.edges = graph.edges 
                    graph_i.edge_costs = graph.edge_costs 
                    graph_i.node_degrees = graph.node_degrees 
                    graph_i._reset_hyperparameters(params=oi3fold)
                    
                    params_copy = { **params_copy, 'graph': graph_i}
    
                    param_sets.append(params_copy)
    
            num_tasks = len(param_sets)
            n_request_cpus = num_tasks
            
            if n_request_cpus < 1:
                n_request_cpus = 1
            elif n_request_cpus > int(runParm['max_num_cpus']):
                n_request_cpus = int(runParm['max_num_cpus'])
    
            print('')
            print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' # tasks/processors: ' + str(num_tasks) + '/' + str(n_request_cpus) + '  ' + 'GENERATE_OI_RUNS')
    
            pool = multiprocessing.Pool(n_request_cpus)
            results = pool.map(perform_oiRuns, param_sets)
            
            # does not return any results

            print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'GENERATE_OI_RUNS')
        
        else:
    
            for findx in range(len(fold_FOLD_DIR_NAME)):
            
                oi3fold = folds_oi3fold_parmsets[findx]
                oi3fold = { **oi3fold, 'max_num_cpus': maxFoldNumProcessors } 
                
                print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' +  oi3fold['FOLD_DIR_NAME'] + '  ' + 'GENERATE_OI_RUNS')
        
                print('Number of prizes names going to run: ' + str(len(oi3fold['targetPhenotypes'])))
                print(oi3fold['targetPhenotypes'])
                perform_oiRuns(oi3fold)

                print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' +  oi3fold['FOLD_DIR_NAME'] + '  ' + 'GENERATE_OI_RUNS')
                print('')
                
    ########################################################################
    ###### _fold_process_oi_runs
    ########################################################################   

    if oi3fold['PROCESS_OI_RUNS']:
                            
        if runParm['parallelize_folds']:
    
            num_tasks = runParm['NUM_OI_FOLDS']

            n_request_cpus = num_tasks
            
            if n_request_cpus < 1:
                n_request_cpus = 1
            elif n_request_cpus > int(runParm['max_num_cpus']):
                n_request_cpus = int(runParm['max_num_cpus'])
    

            for i in range(len(folds_oi3fold_parmsets)):
                oi3fold = folds_oi3fold_parmsets[i]     
                oi3fold = { **oi3fold, 'max_num_cpus': 1 } 
                folds_oi3fold_parmsets[i] = oi3fold
    
            print('')
            print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' # tasks/processors: ' + str(num_tasks) + '/' + str(n_request_cpus) + '  ' + 'PROCESS_OI_RUNS')
    
            pool = multiprocessing.Pool(n_request_cpus)
            results = pool.map(_fold_process_oi_runs, folds_oi3fold_parmsets)
            
            # can ignore results

            print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'PROCESS_OI_RUNS')

            
        else:
    
            for findx in range(len(fold_FOLD_DIR_NAME)):
    
                oi3fold = folds_oi3fold_parmsets[findx]
                oi3fold = { **oi3fold, 'max_num_cpus': maxFoldNumProcessors } 
                
                _fold_process_oi_runs(oi3fold)
    
                # no results

              
    ########################################################################
     ###### _fold_impute_test_data
    ########################################################################   

    if runParm['parallelize_folds']:

        num_tasks = runParm['NUM_OI_FOLDS']

        n_request_cpus = num_tasks
        
        if n_request_cpus < 1:
            n_request_cpus = 1
        elif n_request_cpus > int(runParm['max_num_cpus']):
            n_request_cpus = int(runParm['max_num_cpus'])

        for i in range(len(folds_oi3fold_parmsets)):
            oi3fold = folds_oi3fold_parmsets[i]     
            oi3fold = { **oi3fold, 'max_num_cpus': 1 } 
            folds_oi3fold_parmsets[i] = oi3fold

        print('')
        print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' # tasks/processors: ' + str(num_tasks) + '/' + str(n_request_cpus) + '  ' + 'Impute fold test data')

        pool = multiprocessing.Pool(n_request_cpus)
        results = pool.map(_fold_impute_test_data, folds_oi3fold_parmsets)

        folds_oi3fold_parmsets = results

        print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Impute fold test data')


    else:
        
        for findx in range(len(fold_FOLD_DIR_NAME)):
    
            oi3fold = folds_oi3fold_parmsets[findx]
            oi3fold = { **oi3fold, 'max_num_cpus': maxFoldNumProcessors } 

            oi3fold = _fold_impute_test_data(oi3fold)

            folds_oi3fold_parmsets[findx] = oi3fold
            
    
    ########################################################################
     ###### _fold_generate_base_PNM_EA_models
    ########################################################################   

    if oi3fold['BUILD_BASE_PNM_EA_MODELS']:
        
        if runParm['parallelize_folds']:
    
            num_tasks = runParm['NUM_OI_FOLDS']
    
            n_request_cpus = num_tasks
            
            if n_request_cpus < 1:
                n_request_cpus = 1
            elif n_request_cpus > int(runParm['max_num_cpus']):
                n_request_cpus = int(runParm['max_num_cpus'])
    
            for i in range(len(folds_oi3fold_parmsets)):
                oi3fold = folds_oi3fold_parmsets[i]     
                oi3fold = { **oi3fold, 'max_num_cpus': 1 } 
                folds_oi3fold_parmsets[i] = oi3fold
    
            print('')
            print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' # tasks/processors: ' + str(num_tasks) + '/' + str(n_request_cpus) + '  ' + 'BUILD_BASE_PNM_EA_MODELS')
    
            pool = multiprocessing.Pool(n_request_cpus)
            results = pool.map(_fold_generate_base_PNM_EA_models, folds_oi3fold_parmsets)
    
            folds_oi3fold_parmsets = results
    
            print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'BUILD_BASE_PNM_EA_MODELS')
    
    
        else:
            
            for findx in range(len(fold_FOLD_DIR_NAME)):
        
                oi3fold = folds_oi3fold_parmsets[findx]
                oi3fold = { **oi3fold, 'max_num_cpus': maxFoldNumProcessors } 
    
                oi3fold = _fold_generate_base_PNM_EA_models(oi3fold)
    
                folds_oi3fold_parmsets[findx] = oi3fold
                



    if oi3fold['build_PAs'] or oi3fold['build_PN_With_PAs']:
        build_ensemble_models(runParm, modelType = 'PA', base_model_name_cn = 'PA_Model_Name', base_models_fn = BASE_PA_MODEL_CSV, ensemble_models_fn = 'ensemble_PA_models.csv', train_suffix_fn = '_train_PA_features.csv', test_suffix_fn = '_test_PA_features.csv')

    if oi3fold['BUILD_BASE_PNM_EA_MODELS']:
        build_ensemble_models(runParm, modelType = 'PNM_EA', base_model_name_cn = 'EA_Model_Name', base_models_fn = BASE_PNM_EA_MODEL_CSV, ensemble_models_fn = 'ensemble_PNM_EA_models.csv', train_suffix_fn = '_train_features_imputed.csv', test_suffix_fn = '_test_features_imputed.csv')

            
    print('')
    print('Completed Step1 Build PN PNMs and base PNM EA models')
    print('  End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Start:' + stepStartTime.strftime('%Y-%m-%d %H:%M:%S'))
    print('')


#################################################################################################################################################################
#####  # Generate list of ensemble of latent ensemble models to evaluate by NCK top N latent models. For ACFT latent model Top N = 3,  N^6  where 3^6 = 729 per 
#################################################################################################################################################################

def run_analysis_step_2(runParm, modelType = 'PNM_EA', ensemble_models_fn = 'ensemble_PNM_EA_models.csv', nck_ensemble_models_fn = 'nck_ensemble_EA_models.csv'):

        
    stepStartTime = datetime.datetime.now()
    
    randomState = runParm['randomState']
    OUTCOME_METRIC = runParm['OUTCOME_METRIC']

    if runParm['Compute_ACFT_Score']:
        infoStr = 'ACFT '
    else:
        df_ACFT_tables = pd.DataFrame()
        infoStr = ''
        
    # 2^6 = 64
    # 3^6 = 729
    # 4^6 = 4,096
    # 5^6 = 15,625
    TOP_N_LATENT_METRIC_ENSEMBLE_MODELS = runParm['TOP_N_LATENT_METRIC_ENSEMBLE_MODELS']
    MAX_NUM_LATENT_BASE_MODELS = runParm['MAX_NUM_LATENT_BASE_MODELS']

    USE_SINGLE_MODEL_RANKING_METRIC = runParm['USE_SINGLE_MODEL_RANKING_METRIC']
    MODEL_RANKING_METRIC = runParm['MODEL_RANKING_METRIC']
    MODEL_RANKING_METRIC_ascending = runParm['MODEL_RANKING_METRIC_ascending']



    print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + 'Generate nck list of ' + infoStr + 'ensemble ' + modelType + ' models')

    for findx in range(runParm['NUM_OI_FOLDS']):
    
        fold_num = findx + 1
        
        FOLD_DIR_NAME = getDirName(fold_num)

        # assemble the top results from each latent metric, from each ensemble latent metric EA models file.

        df_All_Ensemble_Models = pd.read_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, ensemble_models_fn))

        Y_METRICS = runParm['LATENT_METRIC_LIST'].copy()
        if len(Y_METRICS) == 0:   
            Y_METRICS = [runParm['OUTCOME_METRIC']]

        ensemble_latent_metric_models = []
        pnm99_latent_metric_models = []
        for Y_METRIC in Y_METRICS:
            
            # print('Generate Ensemble Models:',Y_METRIC)
    
            df_ensemble_models = df_All_Ensemble_Models.loc[df_All_Ensemble_Models['Y_METRIC'] == Y_METRIC].copy()
            
            df_ensemble_models = df_ensemble_models.loc[df_ensemble_models['NumBaseModels'] <= MAX_NUM_LATENT_BASE_MODELS].copy()
            
            # filter and only keep the PNM-99 models
            df_pnm99_models = df_ensemble_models.loc[df_ensemble_models['ensemble'].str.contains('_PNM-99', regex=False)].copy()
            if len(df_pnm99_models) > 0:
                
                if USE_SINGLE_MODEL_RANKING_METRIC:
                    
                    df_pnm99_models.sort_values(MODEL_RANKING_METRIC, ascending=MODEL_RANKING_METRIC_ascending, inplace=True)
                    df_pnm99_models = df_pnm99_models.reset_index(drop = True)
                    df_pnm99_models = df_pnm99_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()
                    
                else:

                    df_pnm99_models.sort_values('mean_cv_test_mse', ascending=True, inplace=True)
                    df_pnm99_models = df_pnm99_models.reset_index(drop = True)
                    df_1 = df_pnm99_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()
                    
                    df_pnm99_models.sort_values('mean_cv_test_adj_R2', ascending=False, inplace=True)
                    df_pnm99_models = df_pnm99_models.reset_index(drop = True)
                    df_2 = df_pnm99_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()

                    df_pnm99_models = pd.concat([df_1, df_2])
                    df_pnm99_models = df_pnm99_models.drop_duplicates()
                    df_pnm99_models = df_pnm99_models.reset_index(drop = True)
                
                cns = df_pnm99_models.columns.to_list()
                df_pnm99_models['fold'] = FOLD_DIR_NAME 
                df_pnm99_models = df_pnm99_models[['fold'] + cns].copy()
                pnm99_latent_metric_models += [df_pnm99_models]    
                
            
            # removed reference to any models with PNM-99 in them
            df_ensemble_models = df_ensemble_models.loc[~df_ensemble_models['ensemble'].str.contains('_PNM-99', regex=False)].copy()

            # force consistency and seek higher  PNM1, PNM2, or PNM3 vs PNM0
            df_ensemble_models.sort_values('ensemble', ascending=False, inplace=True)
            df_ensemble_models = df_ensemble_models.reset_index(drop = True)
            # remove redundant cases PNM0 and PNM1 could have the same feature set
            df_ensemble_models.drop_duplicates(subset='Features', keep='first', inplace=True)

            if USE_SINGLE_MODEL_RANKING_METRIC:
                                    
                df_ensemble_models.sort_values(MODEL_RANKING_METRIC, ascending=MODEL_RANKING_METRIC_ascending, inplace=True)
                df_ensemble_models = df_ensemble_models.reset_index(drop = True)
                df_ensemble_models = df_ensemble_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()

            else:
                                    
                df_ensemble_models.sort_values('mean_cv_test_mse', ascending=True, inplace=True)
                df_ensemble_models = df_ensemble_models.reset_index(drop = True)
                df_1 = df_ensemble_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()
                
                df_ensemble_models.sort_values('mean_cv_test_adj_R2', ascending=False, inplace=True)
                df_ensemble_models = df_ensemble_models.reset_index(drop = True)
                df_2 = df_ensemble_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()

                df_ensemble_models = pd.concat([df_1, df_2])
                df_ensemble_models = df_ensemble_models.drop_duplicates()
                df_ensemble_models = df_ensemble_models.reset_index(drop = True)

                
            cns = df_ensemble_models.columns.to_list()
            df_ensemble_models['fold'] = FOLD_DIR_NAME 
            df_ensemble_models = df_ensemble_models[['fold'] + cns].copy()
            ensemble_latent_metric_models += [df_ensemble_models]

    
        df_ensemble_latent_metric_models = pd.concat(ensemble_latent_metric_models)
        df_ensemble_latent_metric_models = df_ensemble_latent_metric_models.reset_index(drop = True)


        df_select = df_ensemble_latent_metric_models
        if runParm['INCLUDE_PNM_MINUS_99']:
            if len(pnm99_latent_metric_models) > 0:
                df_pnm99_latent_metric_models = pd.concat(pnm99_latent_metric_models)
                df_pnm99_latent_metric_models = df_pnm99_latent_metric_models.reset_index(drop = True)
                df_select = pd.concat([df_pnm99_latent_metric_models, df_ensemble_latent_metric_models])
                df_select = df_select.reset_index(drop = True)
            else:
                df_pnm99_latent_metric_models = pd.DataFrame()

        # for record keeping
        df_select.to_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, modelType + '_top_latent_metric_ensemble_models.csv'),index=False)

        
        # df_base_models = pd.read_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, BASE_PNM_EA_MODEL_CSV))
        
        ################################

        def chooseLatentModels(s_list, s_num_base_list, s_feature_list, s_R2_list, df_select_latent_metric_models, latent_metrics, mi, li):
            if li == 0:
                y_metric = latent_metrics[li]
                n_list = []
                n_num_base_list = []
                n_feature_list = []
                n_R2_list = []
                latent_models = df_select_latent_metric_models.loc[df_select_latent_metric_models['Y_METRIC'] == y_metric]
                for ind in latent_models.index:
                    ensembleModel = latent_models['ensemble'][ind]
                    ls1 = ensembleModel.replace(' ','').replace('[','').replace(']','').replace("'",'')
                    ls1 = ls1.split(",")
                    ls1 = getUniqueKeepOrder(ls1)
                    ensemble_base_model_list = ls1

                    numBaseModels = latent_models['NumBaseModels'][ind]
                    
                    ensembleModel = latent_models['Features'][ind]
                    ls1 = ensembleModel.replace(' ','').replace('[','').replace(']','').replace("'",'')
                    ls1 = ls1.split(",")
                    ls1 = getUniqueKeepOrder(ls1)
                    feature_cols = sorted(ls1)

                    meanCVtestR2 = latent_models[MODEL_RANKING_METRIC][ind]
                    
                    n_list += [ensemble_base_model_list]
                    n_num_base_list += [numBaseModels]
                    n_feature_list += [feature_cols]
                    n_R2_list += [meanCVtestR2]
                    
                if li + 1 < mi:
                    results = chooseLatentModels(n_list, n_num_base_list, n_feature_list, n_R2_list, df_select_latent_metric_models, latent_metrics, mi, li+1)
                    n_list = results[0]
                    n_num_base_list = results[1]
                    n_feature_list = results[2]
                    n_R2_list = results[3]
                    
                return [n_list, n_num_base_list, n_feature_list, n_R2_list]
            elif li < mi:
                y_metric = latent_metrics[li]
                n_list = []
                n_num_base_list = []
                n_feature_list = []
                n_R2_list = []
                latent_models = df_select_latent_metric_models.loc[df_select_latent_metric_models['Y_METRIC'] == y_metric]
                for ind in latent_models.index:
                    ensembleModel = latent_models['ensemble'][ind]
                    ls1 = ensembleModel.replace(' ','').replace('[','').replace(']','').replace("'",'')
                    ls1 = ls1.split(",")
                    ls1 = getUniqueKeepOrder(ls1)
                    ensemble_base_model_list = ls1

                    numBaseModels = latent_models['NumBaseModels'][ind]

                    ensembleModel = latent_models['Features'][ind]
                    ls1 = ensembleModel.replace(' ','').replace('[','').replace(']','').replace("'",'')
                    ls1 = ls1.split(",")
                    ls1 = getUniqueKeepOrder(ls1)
                    feature_cols = sorted(ls1)
                    
                    meanCVtestR2 = latent_models[MODEL_RANKING_METRIC][ind]
                    
                    for si in range(len(s_list)):
                        comboitem = [s_list[si]]
                        comboitem += [ensemble_base_model_list]
                        n_list += [comboitem]

                    for si in range(len(s_num_base_list)):
                        comboitem = [s_num_base_list[si]]
                        comboitem += [numBaseModels]
                        comboitem = sum(comboitem)
                        n_num_base_list += [comboitem]

                    for si in range(len(s_feature_list)):
                        comboitem = [s_feature_list[si]]
                        comboitem += [feature_cols]
                        # combine prior features with this new latent ensemble, make unique and sort it
                        comboitemStr = str(comboitem)
                        ls1 = comboitemStr.replace(' ','').replace('[','').replace(']','').replace("'",'')
                        ls1 = ls1.split(",")
                        ls1 = getUniqueKeepOrder(ls1)
                        comboitem = sorted(ls1)
                        n_feature_list += [comboitem]

                    for si in range(len(s_R2_list)):
                        comboitem = [s_R2_list[si]]
                        comboitem += [meanCVtestR2]
                        n_R2_list += [comboitem]
                        
                if li + 1 < mi:
                    results = chooseLatentModels(n_list, n_num_base_list, n_feature_list, n_R2_list, df_select_latent_metric_models, latent_metrics, mi, li+1)
                    n_list = results[0]
                    n_num_base_list = results[1]
                    n_feature_list = results[2]
                    n_R2_list = results[3]
                    
                return [n_list, n_num_base_list, n_feature_list, n_R2_list]
                
            return [s_list, s_num_base_list, s_feature_list, s_R2_list]

        #########################
        
        def getNCKofLatentModels(df_select_latent_models, latent_metrics):
        
            n_list = []
            n_num_base_list = []
            n_feature_list = []
            n_R2_list = []
            results = chooseLatentModels(n_list, n_num_base_list, n_feature_list, n_R2_list, df_select_latent_models, latent_metrics, len(latent_metrics), 0)
            n_list = results[0]
            n_num_base_list = results[1]
            n_feature_list = results[2]
            n_R2_list = results[3]
    
            s_list = []
            for i in range(len(n_list)):
                s_list += [str(n_list[i])]
    
            s_num_base_list = []
            for i in range(len(n_num_base_list)):
                s_num_base_list += [str(n_num_base_list[i])]
                
            s_feature_list = []
            s_NumUniqueFeatures = []
            for i in range(len(n_feature_list)):
                comboitemStr = str(n_feature_list[i])
                ls1 = comboitemStr.replace(' ','').replace('[','').replace(']','').replace("'",'')
                ls1 = ls1.split(",")
                ls1 = getUniqueKeepOrder(ls1)
                comboitem = sorted(ls1)
    
                s_feature_list += [comboitem]
                s_NumUniqueFeatures += [len(ls1)]
    
            s_R2_list = []
            for i in range(len(n_R2_list)):
                comboitemStr = str(n_R2_list[i])
                ls1 = comboitemStr.replace(' ','').replace('[','').replace(']','').replace("'",'')
                ls1 = ls1.split(",")
                ls1 = [float(x) for x in ls1]
                s_R2_list += [str(ls1)]
                # ls1 = [float(x) for x in ls1]
                # s_R2_list += [str(np.mean(ls1))]
    
                
            df_ensemble_models = pd.DataFrame(list(zip(s_list, s_num_base_list, s_NumUniqueFeatures, s_feature_list, s_R2_list)), columns = ['ensemble', 'NumBaseModels', 'NumUniqueFeatures', 'Features', 'metric_list'])

            return df_ensemble_models

        
        # Y_METRICS = getUniqueKeepOrder(df_ensemble_latent_metric_models['Y_METRIC'])   
        Y_METRICS = runParm['LATENT_METRIC_LIST'].copy()
        if len(Y_METRICS) == 0:   
            Y_METRICS = [runParm['OUTCOME_METRIC']]

        df_ensemble_models = getNCKofLatentModels(df_ensemble_latent_metric_models, Y_METRICS)
        if runParm['INCLUDE_PNM_MINUS_99']:
            if len(df_pnm99_latent_metric_models) > 0:
                df_pnm99 = getNCKofLatentModels(df_pnm99_latent_metric_models, Y_METRICS)
                df_ensemble_models = pd.concat([df_pnm99, df_ensemble_models])


        df_ensemble_models.to_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, nck_ensemble_models_fn),index=False)
    

    print('')
    print('Completed Step2 Generate select list of ' + infoStr + 'ensemble ' + modelType + ' models')
    print('  End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Start:' + stepStartTime.strftime('%Y-%m-%d %H:%M:%S'))
    print('')


#################################################################################################################################################################
#####
#################################################################################################################################################################

def run_analysis_step_3(runParm, modelType = 'PNM_EA', base_model_name_cn = 'EA_Model_Name', base_models_fn = BASE_PNM_EA_MODEL_CSV, ensemble_models_fn = 'ensemble_PNM_EA_models.csv' , nck_ensemble_models_fn = 'nck_ensemble_EA_models.csv', features_info_fn = 'molecular_features_info.csv', all_features_fn = 'all_features_raw.csv', ensemble_model_performance_fn = 'ensemble_EA_models_cv_performance.csv'):

    

    stepStartTime = datetime.datetime.now()

    randomState = runParm['randomState']

    TOP_N_MODELS = runParm['TOP_N_MODELS']
    
    # Starting STEP 3 BY CREATING MODEL DIRECTORY AND GATHERING RESULTS FROM ALL THE FOLDS

    if not os.path.exists(runParm['models_dir']):
        os.mkdir(runParm['models_dir'])
        print('Created models directory:')
        print(runParm['models_dir'])


    
    #####################################################################################################
    ##### SPECIAL CASE TO COMPUTE ACFT TOTAL POINTS FROM SIX ACFT EVENTS ENSEMBLE MODELS  ###############
    #####################################################################################################
    if runParm['Compute_ACFT_Score']:
        df_ACFT_tables = pd.read_csv(path.join(runParm['base_dir'], 'data', 'ACFT_tables_FY20_Standards.csv'))
        infoStr = 'ACFT '
    else:
        df_ACFT_tables = pd.DataFrame()
        infoStr = ''
        
    # gather nck ACFT Ensemble EA Models and base EA models generated within each fold using only the training data for that fold
    
    list_base_EA_models = []
    list_ensemble_EA_models = []
    for findx in range(runParm['NUM_OI_FOLDS']):
        fold_num = findx + 1
        FOLD_DIR_NAME = getDirName(fold_num)
    
        df_base_models = pd.read_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, base_models_fn))
        list_base_EA_models += [df_base_models]
        
        df_ensemble_models = pd.read_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, nck_ensemble_models_fn))
        list_ensemble_EA_models += [df_ensemble_models]
    
    df_base_models = pd.concat(list_base_EA_models)
    df_base_models = df_base_models.reset_index(drop = True)
    df_ensemble_models = pd.concat(list_ensemble_EA_models)
    df_ensemble_models = df_ensemble_models.reset_index(drop = True)
    
    if 'mean_cv_test_mse' in df_ensemble_models.columns:
        df_ensemble_models['ratio_cv_test_over_train'] = df_ensemble_models['mean_cv_test_mse'] / df_ensemble_models['mean_cv_train_mse']
        df_ensemble_models.sort_values('ratio_cv_test_over_train', ascending=True, inplace=True)
        df_ensemble_models = df_ensemble_models.reset_index(drop = True)
    
    df_base_models.to_csv(path.join(runParm['models_dir'], base_models_fn),index=False)
    df_ensemble_models.to_csv(path.join(runParm['models_dir'], ensemble_models_fn),index=False)
    
    
    df_ensemble_models = pd.read_csv(path.join(runParm['models_dir'], ensemble_models_fn))
    
    df_selected_ensemble_EA_models = df_ensemble_models
    

    if 'mean_cv_test_mse' in df_selected_ensemble_EA_models.columns:
        df_selected_ensemble_EA_models.sort_values('mean_cv_test_mse', ascending=True, inplace=True)
        df_selected_ensemble_EA_models.sort_values('ratio_cv_test_over_train', ascending=True, inplace=True)
        df_selected_ensemble_EA_models = df_selected_ensemble_EA_models.reset_index(drop = True)
    
    if len(df_selected_ensemble_EA_models) > TOP_N_MODELS:
        df_selected_ensemble_EA_models = df_selected_ensemble_EA_models.iloc[0:TOP_N_MODELS].copy()
    
    df_selected_ensemble_EA_models.to_csv(path.join(runParm['models_dir'], 'selected_' + ensemble_models_fn),index=False)
    
    print('number of ensemble models across all folds:',len(df_ensemble_models))
    print('number of selected ensemble models:',len(df_selected_ensemble_EA_models))
    
    # get the set of features used by the ensemble models
    ls1 = ','.join(df_ensemble_models['Features'].str.replace(' ','').str.replace('[','').str.replace(']','').str.replace("'",''))
    ls1 = ls1.split(",")
    ls1 = getUniqueKeepOrder(ls1)
    feature_cols = sorted(ls1)
    
    print('number of unique features in all ensemble models:',len(feature_cols))
    
    # get the set of features used by the ensemble models
    ls1 = ','.join(df_selected_ensemble_EA_models['Features'].str.replace(' ','').str.replace('[','').str.replace(']','').str.replace("'",''))
    ls1 = ls1.split(",")
    ls1 = getUniqueKeepOrder(ls1)
    feature_cols = sorted(ls1)
    
    print('number of unique features in selected ensemble models:',len(feature_cols))
    
    
    # This creates the two data files for the model sensitivity analysis

    df_all_features_raw = pd.read_csv(path.join(runParm['analysis_results_dir'], all_features_fn))
    df_molecular_features_info = pd.read_csv(path.join(runParm['analysis_results_dir'], features_info_fn))
    
    # get a copy of the info columns, then append the latent metrics, and the features
    tcns = runParm['info_cols'].copy() 
    if runParm['Compute_ACFT_Score']:
        tcns += ['ACFT_Total_Points','ACFT_Maximum_Deadlift', 'ACFT_Standing_Power_Throw', 'ACFT_Hand_Release_Pushups', 'ACFT_Sprint_Drag_Carry', 'ACFT_Leg_Tuck_OR_Plank', 'ACFT_2_Mile_Run']
    tcns += feature_cols
    tcns = getUniqueKeepOrder(tcns) 
    df_model_data_raw = df_all_features_raw[tcns].copy()
    
    df_model_feature_info = df_molecular_features_info.loc[df_molecular_features_info['featureName'].isin(feature_cols)].copy()
    
    
    df_model_data_raw.to_csv(path.join(runParm['models_dir'], modelType + '_model_data_raw.csv'),index=False)
    df_model_feature_info.to_csv(path.join(runParm['models_dir'], modelType + '_model_feature_info.csv'),index=False)
    

    
    print('Started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' model cv performance.')


    # load in the ensemble models and their base model information
    df_base_models = pd.read_csv(path.join(runParm['models_dir'], base_models_fn))
    df_ensemble_models = pd.read_csv(path.join(runParm['models_dir'], 'selected_' + ensemble_models_fn))
    
    
    # load in all raw data used by ensemble models
    df_model_data_raw = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_data_raw.csv'))
    df_model_feature_info = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_feature_info.csv'))
    
    
    all_cols = df_model_data_raw.columns.to_list()
    
    
    index_list = df_ensemble_models.index.to_list()
    
    num_cv_folds = runParm['num_cv_folds']

    
    params = {'max_num_cpus': runParm['max_num_cpus'],
              'num_cv_folds': num_cv_folds,
              'knn_num_neighbors': runParm['knn_num_neighbors'],
              'numFoldGenerations': runParm['numFoldGenerations'],
              'Y_METRIC': runParm['OUTCOME_METRIC'],
              'Compute_ACFT_Score': runParm['Compute_ACFT_Score'],
              'df_ACFT_tables': df_ACFT_tables.copy(),
              'df_base_models': df_base_models,
              'base_model_name_cn': base_model_name_cn,
              'df_ensemble_models': df_ensemble_models,
              'df_model_data_raw': df_model_data_raw,
              'df_model_feature_info': df_model_feature_info,
              'all_cols': all_cols,
              'info_cols': runParm['info_cols'],
              'randomState': randomState,
              'models_dir':runParm['models_dir'],
              'index_list': index_list}

    if runParm['Compute_ACFT_Score']:
        df_ensemble_models = evaluate_ACFT_EnsembleModels(params)
    else:
        df_ensemble_models = evaluate_EnsembleModels(params)
    
    df_ensemble_models.to_csv(path.join(runParm['models_dir'], ensemble_model_performance_fn),index=False)
    
    print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' model cv performance.')

    print('')
    print('Completed Step3 Evaluate ' + infoStr + 'models across folds and evaluate using all data')
    print('  End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Start:' + stepStartTime.strftime('%Y-%m-%d %H:%M:%S'))
    print('')



#################################################################################################################################################################
#####
#################################################################################################################################################################

def run_analysis_step_3_select(runParm, Number_Select_Models = 10, RANKING_METRIC = 'all_cv_test_R2_mean', RANKING_METRIC_ascending = False, modelType = 'PNM_EA', base_model_name_cn = 'EA_Model_Name', base_models_fn = BASE_PNM_EA_MODEL_CSV, candidate_ensemble_models_fn = 'ensemble_EA_models_cv_performance.csv', ensemble_model_performance_fn = 'ensemble_EA_models_cv_performance_select.csv'):

    

    stepStartTime = datetime.datetime.now()

    randomState = runParm['randomState']

    TOP_N_MODELS = runParm['TOP_N_MODELS']
    
    
    #####################################################################################################
    ##### SPECIAL CASE TO COMPUTE ACFT TOTAL POINTS FROM SIX ACFT EVENTS ENSEMBLE MODELS  ###############
    #####################################################################################################
    if runParm['Compute_ACFT_Score']:
        df_ACFT_tables = pd.read_csv(path.join(runParm['base_dir'], 'data', 'ACFT_tables_FY20_Standards.csv'))
        infoStr = 'ACFT '
    else:
        df_ACFT_tables = pd.DataFrame()
        infoStr = ''
        
    # gather nck ACFT Ensemble EA Models and base EA models generated within each fold using only the training data for that fold
    
    print('Started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' model cv performance.')


    # load in the ensemble models and their base model information
    df_base_models = pd.read_csv(path.join(runParm['models_dir'], base_models_fn))
    df_ensemble_models = pd.read_csv(path.join(runParm['models_dir'], candidate_ensemble_models_fn))

    df_ensemble_models = df_ensemble_models.sort_values(RANKING_METRIC, ascending=RANKING_METRIC_ascending)
    df_ensemble_models = df_ensemble_models.reset_index(drop=True)
    df_ensemble_models = df_ensemble_models.iloc[0:min([Number_Select_Models,len(df_ensemble_models)])].copy()

    
    # load in all raw data used by ensemble models
    df_model_data_raw = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_data_raw.csv'))
    df_model_feature_info = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_feature_info.csv'))
    
    
    all_cols = df_model_data_raw.columns.to_list()
    
    
    index_list = df_ensemble_models.index.to_list()
    
    num_cv_folds = runParm['num_cv_folds']

    
    params = {'max_num_cpus': runParm['max_num_cpus'],
              'num_cv_folds': num_cv_folds,
              'knn_num_neighbors': runParm['knn_num_neighbors'],
              'numFoldGenerations': runParm['numFoldGenerations'],
              'Y_METRIC': runParm['OUTCOME_METRIC'],
              'Compute_ACFT_Score': runParm['Compute_ACFT_Score'],
              'df_ACFT_tables': df_ACFT_tables.copy(),
              'df_base_models': df_base_models,
              'base_model_name_cn': base_model_name_cn,
              'df_ensemble_models': df_ensemble_models,
              'df_model_data_raw': df_model_data_raw,
              'df_model_feature_info': df_model_feature_info,
              'all_cols': all_cols,
              'info_cols': runParm['info_cols'],
              'randomState': randomState,
              'models_dir':runParm['models_dir'],
              'index_list': index_list}

    if runParm['Compute_ACFT_Score']:
        df_ensemble_models = evaluate_ACFT_EnsembleModels(params)
    else:
        df_ensemble_models = evaluate_EnsembleModels(params)
    
    df_ensemble_models.to_csv(path.join(runParm['models_dir'], ensemble_model_performance_fn),index=False)
    
    print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' model cv performance.')

    print('')
    print('Completed Step3 Evaluate select ' + infoStr + 'models across folds and evaluate using all data')
    print('  End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Start:' + stepStartTime.strftime('%Y-%m-%d %H:%M:%S'))
    print('')



#################################################################################################################################################################
#####
#################################################################################################################################################################


def run_analysis_step_2A(runParm, modelType = 'PNM_EA', ensemble_models_fn = 'ensemble_PNM_EA_models.csv', nck_ensemble_models_fn = 'nck_ensemble_EA_models.csv'):

        
    stepStartTime = datetime.datetime.now()
    
    randomState = runParm['randomState']
    OUTCOME_METRIC = runParm['OUTCOME_METRIC']

    if runParm['Compute_ACFT_Score']:
        infoStr = 'ACFT '
    else:
        df_ACFT_tables = pd.DataFrame()
        infoStr = ''
        
    # 2^6 = 64
    # 3^6 = 729
    # 4^6 = 4,096
    # 5^6 = 15,625
    TOP_N_LATENT_METRIC_ENSEMBLE_MODELS = runParm['TOP_N_LATENT_METRIC_ENSEMBLE_MODELS']
    MAX_NUM_LATENT_BASE_MODELS = runParm['MAX_NUM_LATENT_BASE_MODELS']

    USE_SINGLE_MODEL_RANKING_METRIC = runParm['USE_SINGLE_MODEL_RANKING_METRIC']
    MODEL_RANKING_METRIC = runParm['MODEL_RANKING_METRIC']
    MODEL_RANKING_METRIC_ascending = runParm['MODEL_RANKING_METRIC_ascending']



    print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + 'Generate nck list of ' + infoStr + 'ensemble ' + modelType + ' models')

    for findx in range(runParm['NUM_OI_FOLDS']):
    
        fold_num = findx + 1
        
        FOLD_DIR_NAME = getDirName(fold_num)

        # assemble the top results from each latent metric, from each ensemble latent metric EA models file.

        df_All_Ensemble_Models = pd.read_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, ensemble_models_fn))

        Y_METRICS = runParm['LATENT_METRIC_LIST'].copy()
        if len(Y_METRICS) == 0:   
            Y_METRICS = [runParm['OUTCOME_METRIC']]

        ensemble_latent_metric_models = []
        pnm99_latent_metric_models = []
        for Y_METRIC in Y_METRICS:
            
            # print('Generate Ensemble Models:',Y_METRIC)
    
            df_ensemble_models = df_All_Ensemble_Models.loc[df_All_Ensemble_Models['Y_METRIC'] == Y_METRIC].copy()
            
            df_ensemble_models = df_ensemble_models.loc[df_ensemble_models['NumBaseModels'] <= MAX_NUM_LATENT_BASE_MODELS].copy()
            
            # filter and only keep the PNM-99 models
            df_pnm99_models = df_ensemble_models.loc[df_ensemble_models['ensemble'].str.contains('_PNM-99', regex=False)].copy()
            if len(df_pnm99_models) > 0:
                
                if USE_SINGLE_MODEL_RANKING_METRIC:
                    
                    df_pnm99_models.sort_values(MODEL_RANKING_METRIC, ascending=MODEL_RANKING_METRIC_ascending, inplace=True)
                    df_pnm99_models = df_pnm99_models.reset_index(drop = True)
                    df_pnm99_models = df_pnm99_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()
                    
                else:

                    df_pnm99_models.sort_values('mean_cv_test_mse', ascending=True, inplace=True)
                    df_pnm99_models = df_pnm99_models.reset_index(drop = True)
                    df_1 = df_pnm99_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()
                    
                    df_pnm99_models.sort_values('mean_cv_test_adj_R2', ascending=False, inplace=True)
                    df_pnm99_models = df_pnm99_models.reset_index(drop = True)
                    df_2 = df_pnm99_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()

                    df_pnm99_models = pd.concat([df_1, df_2])
                    df_pnm99_models = df_pnm99_models.drop_duplicates()
                    df_pnm99_models = df_pnm99_models.reset_index(drop = True)
                
                cns = df_pnm99_models.columns.to_list()
                df_pnm99_models['fold'] = FOLD_DIR_NAME 
                df_pnm99_models = df_pnm99_models[['fold'] + cns].copy()
                pnm99_latent_metric_models += [df_pnm99_models]    
                
            
            # removed reference to any models with PNM-99 in them
            df_ensemble_models = df_ensemble_models.loc[~df_ensemble_models['ensemble'].str.contains('_PNM-99', regex=False)].copy()

            # force consistency and seek higher  PNM1, PNM2, or PNM3 vs PNM0
            df_ensemble_models.sort_values('ensemble', ascending=False, inplace=True)
            df_ensemble_models = df_ensemble_models.reset_index(drop = True)
            # remove redundant cases PNM0 and PNM1 could have the same feature set
            df_ensemble_models.drop_duplicates(subset='Features', keep='first', inplace=True)

            if USE_SINGLE_MODEL_RANKING_METRIC:
                                    
                df_ensemble_models.sort_values(MODEL_RANKING_METRIC, ascending=MODEL_RANKING_METRIC_ascending, inplace=True)
                df_ensemble_models = df_ensemble_models.reset_index(drop = True)
                df_ensemble_models = df_ensemble_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()

            else:
                                    
                df_ensemble_models.sort_values('mean_cv_test_mse', ascending=True, inplace=True)
                df_ensemble_models = df_ensemble_models.reset_index(drop = True)
                df_1 = df_ensemble_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()
                
                df_ensemble_models.sort_values('mean_cv_test_adj_R2', ascending=False, inplace=True)
                df_ensemble_models = df_ensemble_models.reset_index(drop = True)
                df_2 = df_ensemble_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()

                df_ensemble_models = pd.concat([df_1, df_2])
                df_ensemble_models = df_ensemble_models.drop_duplicates()
                df_ensemble_models = df_ensemble_models.reset_index(drop = True)

                
            cns = df_ensemble_models.columns.to_list()
            df_ensemble_models['fold'] = FOLD_DIR_NAME 
            df_ensemble_models = df_ensemble_models[['fold'] + cns].copy()
            ensemble_latent_metric_models += [df_ensemble_models]

    
        df_ensemble_latent_metric_models = pd.concat(ensemble_latent_metric_models)
        df_ensemble_latent_metric_models = df_ensemble_latent_metric_models.reset_index(drop = True)


        df_select = df_ensemble_latent_metric_models
        if runParm['INCLUDE_PNM_MINUS_99']:
            if len(pnm99_latent_metric_models) > 0:
                df_pnm99_latent_metric_models = pd.concat(pnm99_latent_metric_models)
                df_pnm99_latent_metric_models = df_pnm99_latent_metric_models.reset_index(drop = True)
                df_select = pd.concat([df_pnm99_latent_metric_models, df_ensemble_latent_metric_models])
                df_select = df_select.reset_index(drop = True)
            else:
                df_pnm99_latent_metric_models = pd.DataFrame()

        # for record keeping
        df_select.to_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, modelType + '_top_latent_metric_ensemble_models.csv'),index=False)

        
        # df_base_models = pd.read_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, BASE_PNM_EA_MODEL_CSV))
        
        ################################

        def chooseLatentModels(s_list, s_num_base_list, s_feature_list, s_R2_list, df_select_latent_metric_models, latent_metrics, mi, li):
            if li == 0:
                y_metric = latent_metrics[li]
                n_list = []
                n_num_base_list = []
                n_feature_list = []
                n_R2_list = []
                latent_models = df_select_latent_metric_models.loc[df_select_latent_metric_models['Y_METRIC'] == y_metric]
                for ind in latent_models.index:
                    ensembleModel = latent_models['ensemble'][ind]
                    ls1 = ensembleModel.replace(' ','').replace('[','').replace(']','').replace("'",'')
                    ls1 = ls1.split(",")
                    ls1 = getUniqueKeepOrder(ls1)
                    ensemble_base_model_list = ls1

                    numBaseModels = latent_models['NumBaseModels'][ind]
                    
                    ensembleModel = latent_models['Features'][ind]
                    ls1 = ensembleModel.replace(' ','').replace('[','').replace(']','').replace("'",'')
                    ls1 = ls1.split(",")
                    ls1 = getUniqueKeepOrder(ls1)
                    feature_cols = sorted(ls1)

                    meanCVtestR2 = latent_models[MODEL_RANKING_METRIC][ind]
                    
                    n_list += [ensemble_base_model_list]
                    n_num_base_list += [numBaseModels]
                    n_feature_list += [feature_cols]
                    n_R2_list += [meanCVtestR2]
                    
                if li + 1 < mi:
                    results = chooseLatentModels(n_list, n_num_base_list, n_feature_list, n_R2_list, df_select_latent_metric_models, latent_metrics, mi, li+1)
                    n_list = results[0]
                    n_num_base_list = results[1]
                    n_feature_list = results[2]
                    n_R2_list = results[3]
                    
                return [n_list, n_num_base_list, n_feature_list, n_R2_list]
            elif li < mi:
                y_metric = latent_metrics[li]
                n_list = []
                n_num_base_list = []
                n_feature_list = []
                n_R2_list = []
                latent_models = df_select_latent_metric_models.loc[df_select_latent_metric_models['Y_METRIC'] == y_metric]
                for ind in latent_models.index:
                    ensembleModel = latent_models['ensemble'][ind]
                    ls1 = ensembleModel.replace(' ','').replace('[','').replace(']','').replace("'",'')
                    ls1 = ls1.split(",")
                    ls1 = getUniqueKeepOrder(ls1)
                    ensemble_base_model_list = ls1

                    numBaseModels = latent_models['NumBaseModels'][ind]

                    ensembleModel = latent_models['Features'][ind]
                    ls1 = ensembleModel.replace(' ','').replace('[','').replace(']','').replace("'",'')
                    ls1 = ls1.split(",")
                    ls1 = getUniqueKeepOrder(ls1)
                    feature_cols = sorted(ls1)
                    
                    meanCVtestR2 = latent_models[MODEL_RANKING_METRIC][ind]
                    
                    for si in range(len(s_list)):
                        comboitem = [s_list[si]]
                        comboitem += [ensemble_base_model_list]
                        n_list += [comboitem]

                    for si in range(len(s_num_base_list)):
                        comboitem = [s_num_base_list[si]]
                        comboitem += [numBaseModels]
                        comboitem = sum(comboitem)
                        n_num_base_list += [comboitem]

                    for si in range(len(s_feature_list)):
                        comboitem = [s_feature_list[si]]
                        comboitem += [feature_cols]
                        # combine prior features with this new latent ensemble, make unique and sort it
                        comboitemStr = str(comboitem)
                        ls1 = comboitemStr.replace(' ','').replace('[','').replace(']','').replace("'",'')
                        ls1 = ls1.split(",")
                        ls1 = getUniqueKeepOrder(ls1)
                        comboitem = sorted(ls1)
                        n_feature_list += [comboitem]

                    for si in range(len(s_R2_list)):
                        comboitem = [s_R2_list[si]]
                        comboitem += [meanCVtestR2]
                        n_R2_list += [comboitem]
                        
                if li + 1 < mi:
                    results = chooseLatentModels(n_list, n_num_base_list, n_feature_list, n_R2_list, df_select_latent_metric_models, latent_metrics, mi, li+1)
                    n_list = results[0]
                    n_num_base_list = results[1]
                    n_feature_list = results[2]
                    n_R2_list = results[3]
                    
                return [n_list, n_num_base_list, n_feature_list, n_R2_list]
                
            return [s_list, s_num_base_list, s_feature_list, s_R2_list]

        #########################
        
        def getNCKofLatentModels(df_select_latent_models, latent_metrics):
        
            n_list = []
            n_num_base_list = []
            n_feature_list = []
            n_R2_list = []

            # version 2
            for latent_metric in latent_metrics:
                results = chooseLatentModels(n_list, n_num_base_list, n_feature_list, n_R2_list, df_select_latent_models, [latent_metric], 1, 0)
                n_list += results[0]
                n_num_base_list += results[1]
                n_feature_list += results[2]
                n_R2_list += results[3]

            # # version 1
            # results = chooseLatentModels(n_list, n_num_base_list, n_feature_list, n_R2_list, df_select_latent_models, latent_metrics, len(latent_metrics), 0)
            # n_list = results[0]
            # n_num_base_list = results[1]
            # n_feature_list = results[2]
            # n_R2_list = results[3]
    
            s_list = []
            for i in range(len(n_list)):
                s_list += [str(n_list[i])]
    
            s_num_base_list = []
            for i in range(len(n_num_base_list)):
                s_num_base_list += [str(n_num_base_list[i])]
                
            s_feature_list = []
            s_NumUniqueFeatures = []
            for i in range(len(n_feature_list)):
                comboitemStr = str(n_feature_list[i])
                ls1 = comboitemStr.replace(' ','').replace('[','').replace(']','').replace("'",'')
                ls1 = ls1.split(",")
                ls1 = getUniqueKeepOrder(ls1)
                comboitem = sorted(ls1)
    
                s_feature_list += [comboitem]
                s_NumUniqueFeatures += [len(ls1)]
    
            s_R2_list = []
            for i in range(len(n_R2_list)):
                comboitemStr = str(n_R2_list[i])
                ls1 = comboitemStr.replace(' ','').replace('[','').replace(']','').replace("'",'')
                ls1 = ls1.split(",")
                ls1 = [float(x) for x in ls1]
                s_R2_list += [str(ls1)]
                # ls1 = [float(x) for x in ls1]
                # s_R2_list += [str(np.mean(ls1))]
    
                
            df_ensemble_models = pd.DataFrame(list(zip(s_list, s_num_base_list, s_NumUniqueFeatures, s_feature_list, s_R2_list)), columns = ['ensemble', 'NumBaseModels', 'NumUniqueFeatures', 'Features', 'metric_list'])

            return df_ensemble_models

        ################################################

        # Y_METRICS = getUniqueKeepOrder(df_ensemble_latent_metric_models['Y_METRIC'])   
        Y_METRICS = runParm['LATENT_METRIC_LIST'].copy()
        if len(Y_METRICS) == 0:   
            Y_METRICS = [runParm['OUTCOME_METRIC']]

        df_ensemble_models = getNCKofLatentModels(df_ensemble_latent_metric_models, Y_METRICS)
        if runParm['INCLUDE_PNM_MINUS_99']:
            if len(df_pnm99_latent_metric_models) > 0:
                df_pnm99 = getNCKofLatentModels(df_pnm99_latent_metric_models, Y_METRICS)
                df_ensemble_models = pd.concat([df_pnm99, df_ensemble_models])
                df_ensemble_models = df_ensemble_models.reset_index(drop = True)


        df_ensemble_models.to_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, nck_ensemble_models_fn),index=False)
    

    print('')
    print('Completed Step2A Generate select list of ' + infoStr + 'ensemble ' + modelType + ' models')
    print('  End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Start:' + stepStartTime.strftime('%Y-%m-%d %H:%M:%S'))
    print('')



#################################################################################################################################################################
#####
#################################################################################################################################################################

def run_analysis_step_3A(runParm, apply_ACFT_scoring_to_events = False, modelType = 'PNM_EA', base_model_name_cn = 'EA_Model_Name', base_models_fn = BASE_PNM_EA_MODEL_CSV, ensemble_models_fn = 'ensemble_PNM_EA_models.csv' , nck_ensemble_models_fn = 'nck_ensemble_EA_models.csv', features_info_fn = 'molecular_features_info.csv', all_features_fn = 'all_features_raw.csv', ensemble_model_performance_fn = 'ensemble_EA_models_cv_performance.csv'):


    stepStartTime = datetime.datetime.now()

    randomState = runParm['randomState']

    TOP_N_MODELS = runParm['TOP_N_MODELS']
    
    # Starting STEP 3 BY CREATING MODEL DIRECTORY AND GATHERING RESULTS FROM ALL THE FOLDS

    if not os.path.exists(runParm['models_dir']):
        os.mkdir(runParm['models_dir'])
        print('Created models directory:')
        print(runParm['models_dir'])


    
    #####################################################################################################
    ##### SPECIAL CASE TO COMPUTE ACFT TOTAL POINTS FROM SIX ACFT EVENTS ENSEMBLE MODELS  ###############
    #####################################################################################################
    if runParm['Compute_ACFT_Score']:
        df_ACFT_tables = pd.read_csv(path.join(runParm['base_dir'], 'data', 'ACFT_tables_FY20_Standards.csv'))
        infoStr = 'ACFT '
    else:
        df_ACFT_tables = pd.DataFrame()
        infoStr = ''
        
    # gather nck ACFT Ensemble EA Models and base EA models generated within each fold using only the training data for that fold
    
    list_base_EA_models = []
    list_ensemble_EA_models = []
    for findx in range(runParm['NUM_OI_FOLDS']):
        fold_num = findx + 1
        FOLD_DIR_NAME = getDirName(fold_num)
    
        df_base_models = pd.read_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, base_models_fn))
        list_base_EA_models += [df_base_models]
        
        df_ensemble_models = pd.read_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, nck_ensemble_models_fn))
        list_ensemble_EA_models += [df_ensemble_models]
    
    df_base_models = pd.concat(list_base_EA_models)
    df_base_models = df_base_models.reset_index(drop = True)

    df_ensemble_models = pd.concat(list_ensemble_EA_models)
    df_ensemble_models = df_ensemble_models.reset_index(drop = True)
    
    if 'mean_cv_test_mse' in df_ensemble_models.columns:
        df_ensemble_models['ratio_cv_test_over_train'] = df_ensemble_models['mean_cv_test_mse'] / df_ensemble_models['mean_cv_train_mse']
        df_ensemble_models.sort_values('ratio_cv_test_over_train', ascending=True, inplace=True)
        df_ensemble_models = df_ensemble_models.reset_index(drop = True)
    
    df_base_models.to_csv(path.join(runParm['models_dir'], base_models_fn),index=False)
    df_ensemble_models.to_csv(path.join(runParm['models_dir'], ensemble_models_fn),index=False)
    
    
    df_ensemble_models = pd.read_csv(path.join(runParm['models_dir'], ensemble_models_fn))
    
    df_selected_ensemble_EA_models = df_ensemble_models
    

    if 'mean_cv_test_mse' in df_selected_ensemble_EA_models.columns:
        df_selected_ensemble_EA_models.sort_values('mean_cv_test_mse', ascending=True, inplace=True)
        df_selected_ensemble_EA_models.sort_values('ratio_cv_test_over_train', ascending=True, inplace=True)
        df_selected_ensemble_EA_models = df_selected_ensemble_EA_models.reset_index(drop = True)
    
    if len(df_selected_ensemble_EA_models) > TOP_N_MODELS:
        df_selected_ensemble_EA_models = df_selected_ensemble_EA_models.iloc[0:TOP_N_MODELS].copy()
    
    df_selected_ensemble_EA_models.to_csv(path.join(runParm['models_dir'], 'selected_' + ensemble_models_fn),index=False)
    
    print('number of ensemble models across all folds:',len(df_ensemble_models))
    print('number of selected ensemble models:',len(df_selected_ensemble_EA_models))
    
    # get the set of features used by the ensemble models
    ls1 = ','.join(df_ensemble_models['Features'].str.replace(' ','').str.replace('[','').str.replace(']','').str.replace("'",''))
    ls1 = ls1.split(",")
    ls1 = getUniqueKeepOrder(ls1)
    feature_cols = sorted(ls1)
    
    print('number of unique features in all ensemble models:',len(feature_cols))
    
    # get the set of features used by the ensemble models
    ls1 = ','.join(df_selected_ensemble_EA_models['Features'].str.replace(' ','').str.replace('[','').str.replace(']','').str.replace("'",''))
    ls1 = ls1.split(",")
    ls1 = getUniqueKeepOrder(ls1)
    feature_cols = sorted(ls1)
    
    print('number of unique features in selected ensemble models:',len(feature_cols))
    
    
    # This creates the two data files for the model sensitivity analysis

    df_all_features_raw = pd.read_csv(path.join(runParm['analysis_results_dir'], all_features_fn))
    df_molecular_features_info = pd.read_csv(path.join(runParm['analysis_results_dir'], features_info_fn))
    
    # get a copy of the info columns, then append the latent metrics, and the features
    tcns = runParm['info_cols'].copy() 
    if runParm['Compute_ACFT_Score']:
        tcns += ['ACFT_Total_Points','ACFT_Maximum_Deadlift', 'ACFT_Standing_Power_Throw', 'ACFT_Hand_Release_Pushups', 'ACFT_Sprint_Drag_Carry', 'ACFT_Leg_Tuck_OR_Plank', 'ACFT_2_Mile_Run']
    tcns += feature_cols
    tcns = getUniqueKeepOrder(tcns) 
    df_model_data_raw = df_all_features_raw[tcns].copy()
    
    df_model_feature_info = df_molecular_features_info.loc[df_molecular_features_info['featureName'].isin(feature_cols)].copy()
    
    
    df_model_data_raw.to_csv(path.join(runParm['models_dir'], modelType + '_model_data_raw.csv'),index=False)
    df_model_feature_info.to_csv(path.join(runParm['models_dir'], modelType + '_model_feature_info.csv'),index=False)
    

    
    print('Started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' model cv performance.')


    # load in the ensemble models and their base model information
    df_base_models = pd.read_csv(path.join(runParm['models_dir'], base_models_fn))
    df_ensemble_models = pd.read_csv(path.join(runParm['models_dir'], 'selected_' + ensemble_models_fn))
    
    
    # load in all raw data used by ensemble models
    df_model_data_raw = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_data_raw.csv'))
    df_model_feature_info = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_feature_info.csv'))
    
    
    all_cols = df_model_data_raw.columns.to_list()
    

    
    num_cv_folds = runParm['num_cv_folds']

    # FIND ME

    Y_METRICS = runParm['LATENT_METRIC_LIST'].copy()
    if len(Y_METRICS) == 0:   
        Y_METRICS = [runParm['OUTCOME_METRIC']]

    df_list = []
    for Y_METRIC in Y_METRICS:
        
        print('Generate Ensemble Models:',Y_METRIC)

        df_latent_ensemble_models = df_ensemble_models.loc[df_ensemble_models['ensemble'].str.contains(Y_METRIC, regex=False)].copy()   # VERSION2
        df_latent_ensemble_models = df_latent_ensemble_models.reset_index(drop = True)
        
        # print('length df_latent_ensemble_models:',len(df_latent_ensemble_models)
        # print('length df_latent_ensemble_models:',len(df_latent_ensemble_models)

                  
        params = {'max_num_cpus': runParm['max_num_cpus'],
              'num_cv_folds': num_cv_folds,
              'knn_num_neighbors': runParm['knn_num_neighbors'],
              'numFoldGenerations': runParm['numFoldGenerations'],
              'Y_METRIC': Y_METRIC,
              'Compute_ACFT_Score': apply_ACFT_scoring_to_events, #runParm['Compute_ACFT_Score'],
              'df_ACFT_tables': df_ACFT_tables.copy(),
              'df_base_models': df_base_models,
              'base_model_name_cn': base_model_name_cn,
              'df_ensemble_models': df_latent_ensemble_models,
              'df_model_data_raw': df_model_data_raw,
              'df_model_feature_info': df_model_feature_info,
              'all_cols': all_cols,
              'info_cols': runParm['info_cols'],
              'randomState': randomState,
              'models_dir':runParm['models_dir'],
              'index_list': df_latent_ensemble_models.index.to_list()}

        df = evaluate_EnsembleModels(params)
        cns = df.columns.to_list()
        df['Y_METRIC'] = Y_METRIC
        df = df[['Y_METRIC'] + cns]
        df_list += [df]
        
    df_ensemble_models = pd.concat(df_list,axis=0)
    df_ensemble_models = df_ensemble_models.reset_index(drop = True)
    
    
    df_ensemble_models.to_csv(path.join(runParm['models_dir'], ensemble_model_performance_fn),index=False)

    
    print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' model cv performance.')

    print('')
    print('Completed Step3A Evaluate ' + infoStr + 'models across folds and evaluate using all data')
    print('  End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Start:' + stepStartTime.strftime('%Y-%m-%d %H:%M:%S'))
    print('')


#################################################################################################################################################################
#####
#################################################################################################################################################################

def run_analysis_step_3B(runParm, modelType = 'PNM_EA', ensemble_models_fn = 'ensemble_PNM_EA_models.csv', nck_ensemble_models_fn = 'nck_ensemble_EA_models.csv'):

        
    stepStartTime = datetime.datetime.now()
    
    randomState = runParm['randomState']
    OUTCOME_METRIC = runParm['OUTCOME_METRIC']

    if runParm['Compute_ACFT_Score']:
        infoStr = 'ACFT '
    else:
        df_ACFT_tables = pd.DataFrame()
        infoStr = ''
        
    # 2^6 = 64
    # 3^6 = 729
    # 4^6 = 4,096
    # 5^6 = 15,625
    TOP_N_LATENT_METRIC_ENSEMBLE_MODELS = runParm['TOP_N_LATENT_METRIC_ENSEMBLE_MODELS']
    MAX_NUM_LATENT_BASE_MODELS = runParm['MAX_NUM_LATENT_BASE_MODELS']

    USE_SINGLE_MODEL_RANKING_METRIC = runParm['USE_SINGLE_MODEL_RANKING_METRIC']
    MODEL_RANKING_METRIC = runParm['MODEL_RANKING_METRIC']
    MODEL_RANKING_METRIC_ascending = runParm['MODEL_RANKING_METRIC_ascending']



    
    df_All_Ensemble_Models = pd.read_csv(path.join(runParm['models_dir'], ensemble_models_fn))  # VERSION2

    print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + 'Generate nck list of ' + infoStr + 'ensemble ' + modelType + ' models')

    df_fold_select_list = []
    df_fold_ensemble_models_list = []
    
    for findx in range(runParm['NUM_OI_FOLDS']):
    
        fold_num = findx + 1
        
        FOLD_DIR_NAME = getDirName(fold_num)

        # assemble the top results from each latent metric, from each ensemble latent metric EA models file.

        df_Fold_Ensemble_Models = df_All_Ensemble_Models[df_All_Ensemble_Models['ensemble'].str.contains(FOLD_DIR_NAME, regex=False)].copy()  # VERSION2
        df_Fold_Ensemble_Models['fold'] = FOLD_DIR_NAME

        Y_METRICS = runParm['LATENT_METRIC_LIST'].copy()
        if len(Y_METRICS) == 0:   
            Y_METRICS = [runParm['OUTCOME_METRIC']]

        ensemble_latent_metric_models = []
        pnm99_latent_metric_models = []
        for Y_METRIC in Y_METRICS:
            
            # print('Generate Ensemble Models:',Y_METRIC)
    
            df_ensemble_models = df_Fold_Ensemble_Models.loc[df_Fold_Ensemble_Models['ensemble'].str.contains(Y_METRIC, regex=False)].copy()   # VERSION2
            df_ensemble_models['Y_METRIC'] = Y_METRIC
            
            df_ensemble_models = df_ensemble_models.loc[df_ensemble_models['NumBaseModels'] <= MAX_NUM_LATENT_BASE_MODELS].copy()
            
            # filter and only keep the PNM-99 models
            df_pnm99_models = df_ensemble_models.loc[df_ensemble_models['ensemble'].str.contains('_PNM-99', regex=False)].copy()
            if len(df_pnm99_models) > 0:
                
                if USE_SINGLE_MODEL_RANKING_METRIC:
                    
                    df_pnm99_models.sort_values(MODEL_RANKING_METRIC, ascending=MODEL_RANKING_METRIC_ascending, inplace=True)
                    df_pnm99_models = df_pnm99_models.reset_index(drop = True)
                    df_pnm99_models = df_pnm99_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()
                    
                else:

                    df_pnm99_models.sort_values('all_cv_test_mse_mean', ascending=True, inplace=True)
                    df_pnm99_models = df_pnm99_models.reset_index(drop = True)
                    df_1 = df_pnm99_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()
                    
                    df_pnm99_models.sort_values('all_cv_test_adj_R2_mean', ascending=False, inplace=True)
                    df_pnm99_models = df_pnm99_models.reset_index(drop = True)
                    df_2 = df_pnm99_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()

                    df_pnm99_models = pd.concat([df_1, df_2])
                    df_pnm99_models = df_pnm99_models.drop_duplicates()
                    df_pnm99_models = df_pnm99_models.reset_index(drop = True)
                
                cns = df_pnm99_models.columns.to_list()
                df_pnm99_models['fold'] = FOLD_DIR_NAME 
                df_pnm99_models = df_pnm99_models[['fold'] + cns].copy()
                pnm99_latent_metric_models += [df_pnm99_models]    
                
            
            # removed reference to any models with PNM-99 in them
            df_ensemble_models = df_ensemble_models.loc[~df_ensemble_models['ensemble'].str.contains('_PNM-99', regex=False)].copy()

            # force consistency and seek higher  PNM1, PNM2, or PNM3 vs PNM0
            df_ensemble_models.sort_values('ensemble', ascending=False, inplace=True)
            df_ensemble_models = df_ensemble_models.reset_index(drop = True)
            # remove redundant cases PNM0 and PNM1 could have the same feature set
            df_ensemble_models.drop_duplicates(subset='Features', keep='first', inplace=True)

            if USE_SINGLE_MODEL_RANKING_METRIC:
                                    
                df_ensemble_models.sort_values(MODEL_RANKING_METRIC, ascending=MODEL_RANKING_METRIC_ascending, inplace=True)
                df_ensemble_models = df_ensemble_models.reset_index(drop = True)
                df_ensemble_models = df_ensemble_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()

            else:
                                    
                df_ensemble_models.sort_values('all_cv_test_mse_mean', ascending=True, inplace=True)
                df_ensemble_models = df_ensemble_models.reset_index(drop = True)
                df_1 = df_ensemble_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()
                
                df_ensemble_models.sort_values('all_cv_test_adj_R2_mean', ascending=False, inplace=True)
                df_ensemble_models = df_ensemble_models.reset_index(drop = True)
                df_2 = df_ensemble_models.iloc[0:TOP_N_LATENT_METRIC_ENSEMBLE_MODELS].copy()

                df_ensemble_models = pd.concat([df_1, df_2])
                df_ensemble_models = df_ensemble_models.drop_duplicates()
                df_ensemble_models = df_ensemble_models.reset_index(drop = True)

                
            cns = df_ensemble_models.columns.to_list()
            df_ensemble_models['fold'] = FOLD_DIR_NAME 
            df_ensemble_models = df_ensemble_models[['fold'] + cns].copy()
            ensemble_latent_metric_models += [df_ensemble_models]

    
        df_ensemble_latent_metric_models = pd.concat(ensemble_latent_metric_models)
        df_ensemble_latent_metric_models = df_ensemble_latent_metric_models.reset_index(drop = True)


        df_select = df_ensemble_latent_metric_models
        if runParm['INCLUDE_PNM_MINUS_99']:
            if len(pnm99_latent_metric_models) > 0:
                df_pnm99_latent_metric_models = pd.concat(pnm99_latent_metric_models)
                df_pnm99_latent_metric_models = df_pnm99_latent_metric_models.reset_index(drop = True)
                df_select = pd.concat([df_pnm99_latent_metric_models, df_ensemble_latent_metric_models])
                df_select = df_select.reset_index(drop = True)
            else:
                df_pnm99_latent_metric_models = pd.DataFrame()

        # for record keeping
        # df_select.to_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, modelType + '_top_latent_metric_ensemble_models.csv'),index=False)  # VERSION2
        df_fold_select_list += [df_select]   # VERSION2
        
        # df_base_models = pd.read_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, BASE_PNM_EA_MODEL_CSV))
        
        ################################

        def chooseLatentModels(s_list, s_num_base_list, s_feature_list, s_R2_list, df_select_latent_metric_models, latent_metrics, mi, li):
            if li == 0:
                y_metric = latent_metrics[li]
                n_list = []
                n_num_base_list = []
                n_feature_list = []
                n_R2_list = []
                latent_models = df_select_latent_metric_models.loc[df_select_latent_metric_models['Y_METRIC'] == y_metric]
                for ind in latent_models.index:
                    ensembleModel = latent_models['ensemble'][ind]
                    ls1 = ensembleModel.replace(' ','').replace('[','').replace(']','').replace("'",'')
                    ls1 = ls1.split(",")
                    ls1 = getUniqueKeepOrder(ls1)
                    ensemble_base_model_list = ls1

                    numBaseModels = latent_models['NumBaseModels'][ind]
                    
                    ensembleModel = latent_models['Features'][ind]
                    ls1 = ensembleModel.replace(' ','').replace('[','').replace(']','').replace("'",'')
                    ls1 = ls1.split(",")
                    ls1 = getUniqueKeepOrder(ls1)
                    feature_cols = sorted(ls1)

                    meanCVtestR2 = latent_models[MODEL_RANKING_METRIC][ind]
                    
                    n_list += [ensemble_base_model_list]
                    n_num_base_list += [numBaseModels]
                    n_feature_list += [feature_cols]
                    n_R2_list += [meanCVtestR2]
                    
                if li + 1 < mi:
                    results = chooseLatentModels(n_list, n_num_base_list, n_feature_list, n_R2_list, df_select_latent_metric_models, latent_metrics, mi, li+1)
                    n_list = results[0]
                    n_num_base_list = results[1]
                    n_feature_list = results[2]
                    n_R2_list = results[3]
                    
                return [n_list, n_num_base_list, n_feature_list, n_R2_list]
            elif li < mi:
                y_metric = latent_metrics[li]
                n_list = []
                n_num_base_list = []
                n_feature_list = []
                n_R2_list = []
                latent_models = df_select_latent_metric_models.loc[df_select_latent_metric_models['Y_METRIC'] == y_metric]
                for ind in latent_models.index:
                    ensembleModel = latent_models['ensemble'][ind]
                    ls1 = ensembleModel.replace(' ','').replace('[','').replace(']','').replace("'",'')
                    ls1 = ls1.split(",")
                    ls1 = getUniqueKeepOrder(ls1)
                    ensemble_base_model_list = ls1

                    numBaseModels = latent_models['NumBaseModels'][ind]

                    ensembleModel = latent_models['Features'][ind]
                    ls1 = ensembleModel.replace(' ','').replace('[','').replace(']','').replace("'",'')
                    ls1 = ls1.split(",")
                    ls1 = getUniqueKeepOrder(ls1)
                    feature_cols = sorted(ls1)
                    
                    meanCVtestR2 = latent_models[MODEL_RANKING_METRIC][ind]
                    
                    for si in range(len(s_list)):
                        comboitem = [s_list[si]]
                        comboitem += [ensemble_base_model_list]
                        n_list += [comboitem]

                    for si in range(len(s_num_base_list)):
                        comboitem = [s_num_base_list[si]]
                        comboitem += [numBaseModels]
                        comboitem = sum(comboitem)
                        n_num_base_list += [comboitem]

                    for si in range(len(s_feature_list)):
                        comboitem = [s_feature_list[si]]
                        comboitem += [feature_cols]
                        # combine prior features with this new latent ensemble, make unique and sort it
                        comboitemStr = str(comboitem)
                        ls1 = comboitemStr.replace(' ','').replace('[','').replace(']','').replace("'",'')
                        ls1 = ls1.split(",")
                        ls1 = getUniqueKeepOrder(ls1)
                        comboitem = sorted(ls1)
                        n_feature_list += [comboitem]

                    for si in range(len(s_R2_list)):
                        comboitem = [s_R2_list[si]]
                        comboitem += [meanCVtestR2]
                        n_R2_list += [comboitem]
                        
                if li + 1 < mi:
                    results = chooseLatentModels(n_list, n_num_base_list, n_feature_list, n_R2_list, df_select_latent_metric_models, latent_metrics, mi, li+1)
                    n_list = results[0]
                    n_num_base_list = results[1]
                    n_feature_list = results[2]
                    n_R2_list = results[3]
                    
                return [n_list, n_num_base_list, n_feature_list, n_R2_list]
                
            return [s_list, s_num_base_list, s_feature_list, s_R2_list]

        #########################
        
        def getNCKofLatentModels(df_select_latent_models, latent_metrics):
        
            n_list = []
            n_num_base_list = []
            n_feature_list = []
            n_R2_list = []

            results = chooseLatentModels(n_list, n_num_base_list, n_feature_list, n_R2_list, df_select_latent_models, latent_metrics, len(latent_metrics), 0)
            n_list = results[0]
            n_num_base_list = results[1]
            n_feature_list = results[2]
            n_R2_list = results[3]
    
            s_list = []
            for i in range(len(n_list)):
                s_list += [str(n_list[i])]
    
            s_num_base_list = []
            for i in range(len(n_num_base_list)):
                s_num_base_list += [str(n_num_base_list[i])]
                
            s_feature_list = []
            s_NumUniqueFeatures = []
            for i in range(len(n_feature_list)):
                comboitemStr = str(n_feature_list[i])
                ls1 = comboitemStr.replace(' ','').replace('[','').replace(']','').replace("'",'')
                ls1 = ls1.split(",")
                ls1 = getUniqueKeepOrder(ls1)
                comboitem = sorted(ls1)
    
                s_feature_list += [comboitem]
                s_NumUniqueFeatures += [len(ls1)]
    
            s_R2_list = []
            for i in range(len(n_R2_list)):
                comboitemStr = str(n_R2_list[i])
                ls1 = comboitemStr.replace(' ','').replace('[','').replace(']','').replace("'",'')
                ls1 = ls1.split(",")
                ls1 = [float(x) for x in ls1]
                s_R2_list += [str(ls1)]
                # ls1 = [float(x) for x in ls1]
                # s_R2_list += [str(np.mean(ls1))]
    
                
            df_ensemble_models = pd.DataFrame(list(zip(s_list, s_num_base_list, s_NumUniqueFeatures, s_feature_list, s_R2_list)), columns = ['ensemble', 'NumBaseModels', 'NumUniqueFeatures', 'Features', 'metric_list'])

            return df_ensemble_models

        ################################################

        # Y_METRICS = getUniqueKeepOrder(df_ensemble_latent_metric_models['Y_METRIC'])   
        Y_METRICS = runParm['LATENT_METRIC_LIST'].copy()
        if len(Y_METRICS) == 0:   
            Y_METRICS = [runParm['OUTCOME_METRIC']]

        df_ensemble_models = getNCKofLatentModels(df_ensemble_latent_metric_models, Y_METRICS)
        if runParm['INCLUDE_PNM_MINUS_99']:
            if len(df_pnm99_latent_metric_models) > 0:
                df_pnm99 = getNCKofLatentModels(df_pnm99_latent_metric_models, Y_METRICS)
                df_ensemble_models = pd.concat([df_pnm99, df_ensemble_models])
                df_ensemble_models = df_ensemble_models.reset_index(drop = True)

        # df_ensemble_models.to_csv(path.join(runParm['analysis_results_dir'], FOLD_DIR_NAME, nck_ensemble_models_fn),index=False)   # VERSION2
        df_fold_ensemble_models_list += [df_ensemble_models]   # VERSION2



    df_select = pd.concat(df_fold_select_list,axis=0)   # VERSION2
    df_select = df_select.reset_index(drop = True)
    df_ensemble_models = pd.concat(df_fold_ensemble_models_list,axis=0)   # VERSION2
    df_ensemble_models = df_ensemble_models.reset_index(drop = True)
    df_select.to_csv(path.join(runParm['models_dir'], modelType + '_top_latent_metric_ensemble_models.csv'),index=False)  # VERSION2
    df_ensemble_models.to_csv(path.join(runParm['models_dir'], nck_ensemble_models_fn),index=False)   # VERSION2
    
    print('')
    print('Completed Step3B Generate select list of ' + infoStr + 'ensemble ' + modelType + ' models')
    print('  End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Start:' + stepStartTime.strftime('%Y-%m-%d %H:%M:%S'))
    print('')


#################################################################################################################################################################
#####
#################################################################################################################################################################

def run_analysis_step_3C(runParm, modelType = 'PNM_EA', base_model_name_cn = 'EA_Model_Name', base_models_fn = BASE_PNM_EA_MODEL_CSV,  ensemble_models_fn = 'ensemble_PNM_EA_models.csv', ensemble_model_performance_fn = 'ensemble_EA_models_cv_performance_ACFT_Score.csv'):

    

    stepStartTime = datetime.datetime.now()

    randomState = runParm['randomState']

    TOP_N_MODELS = runParm['TOP_N_MODELS']
    

    
    #####################################################################################################
    ##### SPECIAL CASE TO COMPUTE ACFT TOTAL POINTS FROM SIX ACFT EVENTS ENSEMBLE MODELS  ###############
    #####################################################################################################
    if runParm['Compute_ACFT_Score']:
        df_ACFT_tables = pd.read_csv(path.join(runParm['base_dir'], 'data', 'ACFT_tables_FY20_Standards.csv'))
        infoStr = 'ACFT '
    else:
        df_ACFT_tables = pd.DataFrame()
        infoStr = ''
        

    
    print('Started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' model cv performance.')


    # load in the ensemble models and their base model information
    df_base_models = pd.read_csv(path.join(runParm['models_dir'], base_models_fn))
    df_ensemble_models = pd.read_csv(path.join(runParm['models_dir'], ensemble_models_fn))
    
    
    # load in all raw data used by ensemble models
    df_model_data_raw = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_data_raw.csv'))
    df_model_feature_info = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_feature_info.csv'))
    
    
    all_cols = df_model_data_raw.columns.to_list()
    
    
    index_list = df_ensemble_models.index.to_list()
    
    num_cv_folds = runParm['num_cv_folds']

    
    params = {'max_num_cpus': runParm['max_num_cpus'],
              'num_cv_folds': num_cv_folds,
              'knn_num_neighbors': runParm['knn_num_neighbors'],
              'numFoldGenerations': runParm['numFoldGenerations'],
              'Y_METRIC': runParm['OUTCOME_METRIC'],
              'Compute_ACFT_Score': runParm['Compute_ACFT_Score'],
              'df_ACFT_tables': df_ACFT_tables.copy(),
              'df_base_models': df_base_models,
              'base_model_name_cn': base_model_name_cn,
              'df_ensemble_models': df_ensemble_models,
              'df_model_data_raw': df_model_data_raw,
              'df_model_feature_info': df_model_feature_info,
              'all_cols': all_cols,
              'info_cols': runParm['info_cols'],
              'randomState': randomState,
              'models_dir':runParm['models_dir'],
              'index_list': index_list}

     
    if runParm['Compute_ACFT_Score']:
        df_ensemble_models = evaluate_ACFT_EnsembleModels(params)
    else:
        df_ensemble_models = evaluate_EnsembleModels(params)

    df_ensemble_models.to_csv(path.join(runParm['models_dir'], ensemble_model_performance_fn),index=False)
    
    print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' model cv performance.')

    print('')
    print('Completed Step3C Evaluate ' + infoStr + 'models across folds and evaluate using all data')
    print('  End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Start:' + stepStartTime.strftime('%Y-%m-%d %H:%M:%S'))
    print('')
    
#################################################################################################################################################################
#####
#################################################################################################################################################################


def run_analysis_step_3_plot(runParm, plotData = True, verbose = True, Model_Rank = 1, RANKING_METRIC = 'all_cv_test_R2_mean', RANKING_METRIC_ascending = False, Model_Rank_Name = None, modelType = 'PNM_EA', plotPNM99 = False,  base_model_name_cn = 'EA_Model_Name',  base_models_fn = BASE_PNM_EA_MODEL_CSV, ensemble_model_performance_fn = 'ACFT_ensemble_EA_models_cv_performance.csv'):

    stepStartTime = datetime.datetime.now()

    randomState = runParm['randomState']

    num_cv_folds = runParm['num_cv_folds']

    if Model_Rank_Name == None:
        Model_Rank_Name = str(Model_Rank)

    # load in the ensemble models and their base model information
    df_base_models = pd.read_csv(path.join(runParm['models_dir'], base_models_fn))
    df_ensemble_models = pd.read_csv(path.join(runParm['models_dir'], ensemble_model_performance_fn))
    
    # load in all raw data used by ensemble models
    df_model_data_raw = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_data_raw.csv'))
    df_model_feature_info = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_feature_info.csv'))


    model_perf = df_ensemble_models.copy()
    
    if modelType != 'PA':
        if plotPNM99:
            model_perf = model_perf.loc[model_perf['ensemble'].str.contains('_PNM-99', regex=False)].copy()
        else:
            model_perf = model_perf.loc[~model_perf['ensemble'].str.contains('_PNM-99', regex=False)].copy()

    if RANKING_METRIC != None:
        model_perf = model_perf.sort_values(RANKING_METRIC, ascending=RANKING_METRIC_ascending)
        model_perf = model_perf.reset_index(drop=True)
    
    ind = model_perf.index[Model_Rank-1]
    
    #############################################################################

    # Remove subjects that do not have a measured outcome
    df_model_data_raw = df_model_data_raw[df_model_data_raw[runParm['OUTCOME_METRIC']].notna()]
    
    ensembleParm = {}
    
    ensembleName = model_perf.iloc[(Model_Rank-1):Model_Rank]['ensemble'].values[0]
    ls1 = ensembleName.replace('[','').split("],")

    numEnsembleOfEnsembleModels = len(ls1)
    for ei in range(numEnsembleOfEnsembleModels):
        ensembleParm = { **ensembleParm, str(ei) + '_ensemble_base_model_list': getUniqueKeepOrder((ls1[ei].replace(' ','').replace('[','').replace(']','').replace("'",'').split(",")))}

    ensemble_base_model_list = ensembleParm[str(0) + '_ensemble_base_model_list']
    
    ensembleModels = df_base_models.loc[df_base_models[base_model_name_cn].isin(ensemble_base_model_list)].copy()
    
    # get the set of features used by the ensemble models
    
    ls1 = df_ensemble_models['Features'][ind].replace(' ','').replace('[','').replace(']','').replace("'",'')
    # ls1 = ','.join(ensembleModels['Features'])
    ls1 = ls1.split(",")
    ls1 = getUniqueKeepOrder(ls1)
    feature_cols = sorted(ls1)
    
    
    # make sure local version of info_cols contains the OUTCOME_METRIC
    information_cols = getUniqueKeepOrder(([runParm['OUTCOME_METRIC']] + runParm['LATENT_METRIC_LIST'] + runParm['info_cols']))
    
    df_train_fold = df_model_data_raw
    
    # This appears to be an unnecssary block of code for this cell
    dfimpute = pd.DataFrame(impute_data_knn(df_train_fold[feature_cols], n_neighbors=runParm['knn_num_neighbors']), columns=feature_cols)
    df_train_fold_info = df_train_fold[information_cols].copy()
    df_train_fold_info = df_train_fold_info.reset_index(drop = True)
    dfimpute = dfimpute.reset_index(drop = True)
    df_train_fold_imputed = pd.concat([df_train_fold_info,dfimpute],axis=1)
    
    
    # y_train = df_train_fold_imputed[runParm['OUTCOME_METRIC']].astype(float)
    

    ############################################################################
    
    ensembleName = model_perf.iloc[(Model_Rank-1):Model_Rank]['ensemble'].values[0]
    ls1 = ensembleName.replace(' ','').replace('[','').replace(']','').replace("'",'')
    ls1 = ls1.split(",")
    ls1 = getUniqueKeepOrder(ls1)
    ensemble_base_model_list = ls1
    
    model_perf = model_perf.iloc[(Model_Rank-1):Model_Rank]
    
    ensembleModels = df_base_models.loc[df_base_models[base_model_name_cn].isin(ensemble_base_model_list)].copy()
    
    
    ls1 = ','.join(ensembleModels['Features'])
    ls1 = ls1.split(",")
    ls1 = getUniqueKeepOrder(ls1)
    feature_cols = ls1
    
    
    df_train_fold = df_model_data_raw
    
    dfimpute = pd.DataFrame(impute_data_knn(df_train_fold[feature_cols], n_neighbors=runParm['knn_num_neighbors']), columns=feature_cols)
    df_train_fold_info = df_train_fold[information_cols].copy()
    df_train_fold_info = df_train_fold_info.reset_index(drop = True)
    dfimpute = dfimpute.reset_index(drop = True)
    df_train_fold_imputed = pd.concat([df_train_fold_info,dfimpute],axis=1)
    
    fold_Train = df_train_fold_imputed
    
    #######################################################################

    Y_METRICS = runParm['LATENT_METRIC_LIST'].copy()
    if len(Y_METRICS) == 0:   
        Y_METRICS = [runParm['OUTCOME_METRIC']]

    df_model_details = []
        
    for ei in range(numEnsembleOfEnsembleModels):
        
        ensemble_base_model_list = ensembleParm[str(ei) + '_ensemble_base_model_list']
        ensembleModels = df_base_models.loc[df_base_models[base_model_name_cn].isin(ensemble_base_model_list)].copy()
        
        y_pred_train_base_models = []
        
        y_train = fold_Train[Y_METRICS[ei]].astype(float)
        
        for ind in ensembleModels.index:
            model_features = getUniqueKeepOrder((ensembleModels['Features'][ind].split(",")))
        
            X_train = fold_Train[model_features].to_numpy()
        
            n_components = 1
            pls_model = PLSRegression(n_components=n_components)
            pls_model.fit(X_train, y_train)
            y_pred_train = pls_model.predict(X_train)



            dfinfo = df_model_feature_info.loc[df_model_feature_info['featureName'].isin(model_features)].copy()
            if modelType != 'PA':  
                dfinfo['featureLabel'] = dfinfo['nodeTitle']
                indices = dfinfo.index[dfinfo['nodeSpecies']=='Protein'].tolist()
                dfinfo.loc[indices, 'featureLabel'] = dfinfo.loc[indices, 'nodeLabel'] # for proteins prefer to use nodeLabel over nodeTitle
                dfinfo = dfinfo[['featureName','featureLabel','nodeSpecies']]


            
            # center scale x
            x_scaled = X_train.copy()
            x_scaled = x_scaled.astype('float') 
            x_mean = x_scaled.mean(axis=0)
            x_scaled -= x_mean
            x_std = x_scaled.std(axis=0, ddof=1)  # note that ddof = 1
            x_std[x_std == 0.0] = 1.0
            x_scaled /= x_std
            
            df = pd.DataFrame()
            df['featureName'] = model_features
            df['base_EA_model'] = ensembleModels[base_model_name_cn][ind]
            df['Y_METRIC'] = Y_METRICS[ei]
            df = df[['base_EA_model', 'Y_METRIC','featureName']]
            df = pd.merge(df, dfinfo, on='featureName', how='left')
            df['x_scale_mean'] = pd.Series(x_mean)
            df['x_scale_std'] = pd.Series(x_std)
            df['intercept_'] = pls_model.intercept_[0]
            df['coef_'] = pls_model.coef_[0]
            v = pls_model.x_rotations_[:, 0]
            df['varContrib'] = 100*v*v

            df.sort_values('varContrib', ascending=False, inplace=True)
            df = df.reset_index(drop = True)
            df_model_details += [df]
            
            y_pred_train_base_models += [y_pred_train]
        
        y_pred_train_ensemble_model = np.mean(y_pred_train_base_models, axis = 0)
        
        ensembleParm[str(ei) + '_y_pred_train_ensemble_model'] = y_pred_train_ensemble_model
        
    df_model_details = pd.concat(df_model_details)
        
    #####################################
    
    if runParm['Compute_ACFT_Score']:
        
        #####################################################################################################
        ##### SPECIAL CASE TO COMPUTE ACFT TOTAL POINTS FROM SIX ACFT EVENTS ENSEMBLE MODELS  ###############
        #####################################################################################################

        df_ACFT_tables = pd.read_csv(path.join(runParm['base_dir'], 'data', 'ACFT_tables_FY20_Standards.csv'))

    
        for ei in range(numEnsembleOfEnsembleModels):
            ensembleParm[str(ei) + '_y_pred_train_scores'] =[compute_ACFT_Score(df_ACFT_tables, Y_METRICS[ei],v) for v in ensembleParm[str(ei) + '_y_pred_train_ensemble_model']]

        y_pred_train_base_models = []
        
        for ei in range(numEnsembleOfEnsembleModels):
            y_pred_train_base_models += [ensembleParm[str(ei) + '_y_pred_train_scores']]
    
        ### XXXXXXXXXXXXXXX  SUMMATION NOT MEAN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        y_pred_train_ensemble_model = np.sum(y_pred_train_base_models, axis = 0)
    
    
    y_train = fold_Train[runParm['OUTCOME_METRIC']].astype(float)
    

    ################################
    
    train_R2 = r2_score(y_train, y_pred_train_ensemble_model)
    
    train_mse = mean_squared_error(y_train, y_pred_train_ensemble_model)
    
    
    y_pred_train = y_pred_train_ensemble_model
    
    
    
    all_train_r_squared =  train_R2
    all_train__mse = train_mse

    OUTCOME_LABEL = runParm['OUTCOME_METRIC'].replace('_', ' ') 
    

    
    if verbose:
        print('Outcome:', OUTCOME_LABEL)
        print(f"All Train Number of Subjects: {len(df_train_fold):.1f}")
        print()
        print(f"All Train R2: {all_train_r_squared:.3f}")
        print(f"All Train MSE: {all_train__mse:.3f}")
        print(f"All Train RMSE: {all_train__mse**0.5:.3f}")
        print()
        print(f"Number of Samplings: {model_perf['all_cv_N'].values[0]:.1f}")
        print()
        print(f"{num_cv_folds:0}-fold CV Train R2 (mean ± stdev): {model_perf['all_cv_train_R2_mean'].values[0]:.3f} ± {model_perf['all_cv_train_R2_stdev'].values[0]:.3f}")
        print(f"{num_cv_folds:0}-fold CV Test  R2 (mean ± stdev): {model_perf['all_cv_test_R2_mean'].values[0]:.3f} ± {model_perf['all_cv_test_R2_stdev'].values[0]:.3f}")
        print()
        print(f"{num_cv_folds:0}-fold CV Train MSE (mean ± stdev): {model_perf['all_cv_train_mse_mean'].values[0]:.3f} ± {model_perf['all_cv_train_mse_stdev'].values[0]:.3f}")
        print(f"{num_cv_folds:0}-fold CV Test  MSE (mean ± stdev): {model_perf['all_cv_test_mse_mean'].values[0]:.3f} ± {model_perf['all_cv_test_mse_stdev'].values[0]:.3f}")
        print()
    
    
        print('Total number of unique features:',len(feature_cols))
        print()
    
        for ei in range(numEnsembleOfEnsembleModels):
     
            print('Number of Ensemble Models:',len(ensembleParm[str(ei) + '_ensemble_base_model_list']))
            for base_model_name in ensembleParm[str(ei) + '_ensemble_base_model_list']:
                ensembleModels = df_base_models.loc[df_base_models[base_model_name_cn].isin([base_model_name])].copy()
                print()
                print(base_model_name)
    
                base_model_features = ensembleModels[ensembleModels[base_model_name_cn] == base_model_name]['Features'].values[0].split(",")
    
    
                dfinfo = df_model_feature_info.loc[df_model_feature_info['featureName'].isin(base_model_features)].copy()
                if modelType != 'PA':  
                    dfinfo['display_feature_label'] = dfinfo['nodeTitle']
                    indices = dfinfo.index[dfinfo['nodeSpecies']=='Protein'].tolist()
                    dfinfo.loc[indices, 'display_feature_label'] = dfinfo.loc[indices, 'nodeLabel'] # for proteins prefer to use nodeLabel over nodeTitle
                    print(dfinfo['display_feature_label'].to_list())
                    print()
                else:
                    print(dfinfo['featureName'].to_list())
                    print() 

            
    MAIN_PLOT_LABEL = OUTCOME_LABEL

    df_model_y_values = pd.DataFrame()
    df_model_y_values[runParm['OUTCOME_METRIC'] + '_Actual'] = y_train
    df_model_y_values[runParm['OUTCOME_METRIC'] + '_Predicted'] = y_pred_train

    if plotData:
        plt.scatter(y_train, y_pred_train, c='blue', label='Actual vs Predicted')
        plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], '--', c='red', label='Perfect Prediction')
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.title(MAIN_PLOT_LABEL + ' : PLS regression using All data to train model')
        plt.legend()
        plt.show()


    if numEnsembleOfEnsembleModels > 1:
        for ei in range(numEnsembleOfEnsembleModels):
        
            y_pred_train_ensemble_model = ensembleParm[str(ei) + '_y_pred_train_ensemble_model']
            y_train = fold_Train[Y_METRICS[ei]].astype(float)
            MAIN_PLOT_LABEL = Y_METRICS[ei]

            train_R2 = r2_score(y_train, y_pred_train_ensemble_model)
            
            train_mse = mean_squared_error(y_train, y_pred_train_ensemble_model)
            
            y_pred_train = y_pred_train_ensemble_model

            all_train_r_squared =  train_R2
            all_train__mse = train_mse

            df_model_y_values[Y_METRICS[ei] + '_Actual'] = y_train
            df_model_y_values[Y_METRICS[ei] + '_Predicted'] = y_pred_train

            if plotData:
                plt.scatter(y_train, y_pred_train, c='blue', label='Actual vs Predicted')
                plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], '--', c='red', label='Perfect Prediction')
                plt.xlabel('Actual Value')
                plt.ylabel('Predicted Value')
                plt.title(MAIN_PLOT_LABEL + ' : PLS regression using All data to train model')
                plt.legend()
                plt.show()

    
    df_model_y_values = pd.concat([fold_Train['UID'], df_model_y_values], axis=1)

    if plotPNM99:
        model_details_fn = modelType + '_PNM99_model_' + Model_Rank_Name + '_details.csv'
        model_y_values_fn = modelType + '_PNM99_model_' + Model_Rank_Name + '_y_values.csv'
    else:
        model_details_fn = modelType + '_model_' + Model_Rank_Name + '_details.csv'
        model_y_values_fn = modelType + '_model_' + Model_Rank_Name + '_y_values.csv'

    df_model_details.to_csv(path.join(runParm['models_dir'], model_details_fn),index=False)
    df_model_y_values.to_csv(path.join(runParm['models_dir'], model_y_values_fn),index=False)


#################################################################################################################################################################
#####
#################################################################################################################################################################

def _run_sPLS_Correlation_Analysis(oi3fold):

    print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  + '  ' + 'Perform sPLS Correlation Analysis')
        
    if oi3fold['COMPUTE_OUTCOME_BASE_EA_MODEL']:
        Y_METRICS = [oi3fold['OUTCOME_METRIC']] + oi3fold['LATENT_METRIC_LIST']
    else:
        Y_METRICS = oi3fold['LATENT_METRIC_LIST']
        if len(oi3fold['LATENT_METRIC_LIST']) == 0:
            print('LATENT_METRIC_LIST = [] and COMPUTE_OUTCOME_BASE_EA_MODEL = False. Either add y metrics to list or change boolean to True.')
            # break

    # R script to build Molecular Expression Axis (EA) Models
    script_path = oi3fold['RCODE_SOURCE_DIR'] + '/' + 'PhenoMol_sPLS_Correlation_Analysis_v1.R'
    args = [oi3fold['R_LIBRARY_DIR'], oi3fold['RCODE_SOURCE_DIR'], oi3fold['analysis_results_dir'], oi3fold['FOLD_DIR_NAME'], oi3fold['ANALYSIS_RUN_SEED']]
    args += ['RESULTS_FILENAME=' + BASE_PNM_EA_MODEL_CSV]
    Y_METRICS_STRING = ','.join(Y_METRICS)
    args += ['Y_METRIC=' + Y_METRICS_STRING]
    # args +=  ['Y_METRIC=' + oi3fold['OUTCOME_METRIC']]
    args += ['BINARY_OUTCOME=' + str(oi3fold['BINARY_OUTCOME']).upper()]
    args += ['MIN_NUM_MEASURES_PER_FEATURE=' + str(oi3fold['MIN_NUM_MEASURES_PER_FEATURE'])]
    args += ['TUNE_SPLS_NUM_FOLDS=' + str(oi3fold['TUNE_SPLS_NUM_FOLDS'])]
    args += ['TUNE_SPLS_NUM_REPEATS=' + str(oi3fold['TUNE_SPLS_NUM_REPEATS'])]
    args += ['SPLS_RANDOM_SEED=' + str(oi3fold['SPLS_RANDOM_SEED'])]
    r_path = oi3fold['r_path']
    cmd = [r_path, script_path]  + args
    result = subprocess.check_output(cmd, universal_newlines=True)
    
    # print(result)

    print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  + '  ' + 'Perform sPLS Correlation Analysis')
    
    return oi3fold


#################################################################################################################################################################
#####
#################################################################################################################################################################

def run_sPLS_Correlation_Analysis(runParm, ANALYSIS_RUN_SEEDS = [123,321,456,654,789,987], ANALYSIS_FOLD_DIR_NAMES = ['fold_001','fold_002','fold_003','fold_004','fold_005'], COMPUTE_OUTCOME_BASE_EA_MODEL = False):
    
    stepStartTime = datetime.datetime.now()
    
    
    folds_oi3fold_parmsets = []

    for ANALYSIS_RUN_SEED in ANALYSIS_RUN_SEEDS:
        for FOLD_DIR_NAME in ANALYSIS_FOLD_DIR_NAMES:
            oi3fold = runParm.copy()
            oi3fold = { **oi3fold,   
                'ANALYSIS_RUN_SEED': str(ANALYSIS_RUN_SEED), 
                'FOLD_DIR_NAME': FOLD_DIR_NAME,
                'COMPUTE_OUTCOME_BASE_EA_MODEL': COMPUTE_OUTCOME_BASE_EA_MODEL}
            folds_oi3fold_parmsets += [oi3fold]
            
    
    maxFoldNumProcessors = int(runParm['max_num_cpus'])
    
    if runParm['parallelize_folds']:
    
        num_tasks = len(folds_oi3fold_parmsets)
    
        n_request_cpus = num_tasks
        
        if n_request_cpus < 1:
            n_request_cpus = 1
        elif n_request_cpus > int(runParm['max_num_cpus']):
            n_request_cpus = int(runParm['max_num_cpus'])
    
        for i in range(len(folds_oi3fold_parmsets)):
            oi3fold = folds_oi3fold_parmsets[i]     
            oi3fold = { **oi3fold, 'max_num_cpus': 1 } 
            folds_oi3fold_parmsets[i] = oi3fold
    
        print('')
        print('Start     ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' # tasks/processors: ' + str(num_tasks) + '/' + str(n_request_cpus) + '  ' + 'Perform sPLS Correlation Analysis')
    
        pool = multiprocessing.Pool(n_request_cpus)
        results = pool.map(_run_sPLS_Correlation_Analysis, folds_oi3fold_parmsets)
    
        folds_oi3fold_parmsets = results
    
        print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Perform sPLS Correlation Analysis')
    
    
    else:
        
        for findx in range(len(folds_oi3fold_parmsets)):
    
            oi3fold = folds_oi3fold_parmsets[findx]
            oi3fold = { **oi3fold, 'max_num_cpus': maxFoldNumProcessors } 
    
            oi3fold = _run_sPLS_Correlation_Analysis(oi3fold)
    
            folds_oi3fold_parmsets[findx] = oi3fold
            
    
    print('')
    print('Completed Step Perform sPLS Correlation Analysis')
    print('  End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Start:' + stepStartTime.strftime('%Y-%m-%d %H:%M:%S'))
    print('')



#################################################################################################################################################################
#####
#################################################################################################################################################################


def evaluate_sPLS_correlation_models(runParm, ANALYSIS_RUN_SEEDS = [123,321,456,654,789,987], ANALYSIS_FOLD_DIR_NAMES = ['fold_001','fold_002','fold_003','fold_004','fold_005'], x_block_analysis_results_dir = None, y_block_analysis_results_dir = None, base_model_name_cn = 'Model_Name', features_info_fn = 'molecular_features_info.csv', all_features_fn = 'all_features_raw.csv', suffix_base_models_fn = '_base_models.csv', suffix_ensemble_models_fn = '_ensemble_models.csv', suffix_ensemble_model_performance_fn = '_ensemble_models_cv_performance.csv'):



    stepStartTime = datetime.datetime.now()

    randomState = runParm['randomState']

    TOP_N_MODELS = runParm['TOP_N_MODELS']
    
    # Starting BY CREATING MODEL DIRECTORY AND GATHERING RESULTS FROM ALL THE FOLDS

    if not os.path.exists(runParm['models_dir']):
        os.mkdir(runParm['models_dir'])
        print('Created models directory:')
        print(runParm['models_dir'])


    
    #####################################################################################################
    ##### SPECIAL CASE TO COMPUTE ACFT TOTAL POINTS FROM SIX ACFT EVENTS ENSEMBLE MODELS  ###############
    #####################################################################################################
    if runParm['Compute_ACFT_Score']:
        df_ACFT_tables = pd.read_csv(path.join(runParm['base_dir'], 'data', 'ACFT_tables_FY20_Standards.csv'))
        infoStr = 'ACFT '
        Y_METRIC_LIST = runParm['LATENT_METRIC_LIST']
    else:
        df_ACFT_tables = pd.DataFrame()
        infoStr = ''
        Y_METRIC_LIST = [runParm['OUTCOME_METRIC']]
        
    df_spls_correlation_base_models_list = []

    col_list_ensemble = []	
    col_list_NumBaseModels = []		
    col_list_x_block_NumUniqueFeatures = []	
    col_list_x_block_Features = []	
    col_list_y_block_NumUniqueFeatures = []	
    col_list_y_block_Features = []	

    for analysis_run_seed in ANALYSIS_RUN_SEEDS:
        for FOLD_DIR_NAME in ANALYSIS_FOLD_DIR_NAMES:
    
            df_spls_correlation_base_models = pd.read_csv(path.join(runParm['analysis_results_dir'], ('spls_correlation_' + str(analysis_run_seed) + '_' + FOLD_DIR_NAME  + '_results.csv')))

            df_spls_correlation_base_models[base_model_name_cn] = df_spls_correlation_base_models[['analysis_run_seed', 'fold', 'Y_METRIC']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

            df_spls_correlation_base_models_list += [df_spls_correlation_base_models]

            base_model_list = []
            ensemble_model_x_block_features = []
            ensemble_model_y_block_features = []
            
            for Y_METRIC in Y_METRIC_LIST:

                dft = df_spls_correlation_base_models.loc[df_spls_correlation_base_models['Y_METRIC'] == Y_METRIC].copy()

                base_model_list += [dft[base_model_name_cn].to_list()]
                
                ls1 = dft['x_block_Features'].loc[dft.index[0]]
                ls1 = ls1.replace(' ','').replace('[','').replace(']','').replace("'",'')
                ls1 = ls1.split(",")
                ensemble_model_x_block_features = getUniqueKeepOrder(ensemble_model_x_block_features + ls1)
    
                ls1 = dft['y_block_Features'].loc[dft.index[0]]
                ls1 = ls1.replace(' ','').replace('[','').replace(']','').replace("'",'')
                ls1 = ls1.split(",")
                ensemble_model_y_block_features = getUniqueKeepOrder(ensemble_model_y_block_features + ls1)

            col_list_ensemble += [base_model_list]	
            col_list_NumBaseModels += [len(Y_METRIC_LIST)]		
            col_list_x_block_NumUniqueFeatures += [len(ensemble_model_x_block_features)]	
            col_list_x_block_Features += [ensemble_model_x_block_features]	
            col_list_y_block_NumUniqueFeatures += [len(ensemble_model_y_block_features)]	
            col_list_y_block_Features += [ensemble_model_y_block_features]	
            
            # ensemble	NumBaseModels	NumUniqueFeatures	Features	metric_list

    
    df_spls_correlation_base_models = pd.concat(df_spls_correlation_base_models_list)

    df_spls_correlation_ensemble_models = pd.DataFrame(list(zip(col_list_ensemble, col_list_NumBaseModels, col_list_x_block_NumUniqueFeatures, col_list_x_block_Features, col_list_y_block_NumUniqueFeatures, col_list_y_block_Features)), columns = ['ensemble','NumBaseModels', 'x_block_NumUniqueFeatures', 'x_block_Features', 'y_block_NumUniqueFeatures', 'y_block_Features'])


    # get the set of features used by the ensemble models
    ls1 = ','.join(df_spls_correlation_base_models['x_block_Features'].str.replace(' ','').str.replace('[','').str.replace(']','').str.replace("'",''))
    ls1 = ls1.split(",")
    ls1 = getUniqueKeepOrder(ls1)
    all_x_block_features = sorted(ls1)
    
    # print('number of unique x block features in selected ensemble models:',len(all_x_block_features))

    # get the set of features used by the ensemble models
    ls1 = ','.join(df_spls_correlation_base_models['y_block_Features'].str.replace(' ','').str.replace('[','').str.replace(']','').str.replace("'",''))
    ls1 = ls1.split(",")
    ls1 = getUniqueKeepOrder(ls1)
    all_y_block_features = sorted(ls1)
    
    # print('number of unique y block features in selected ensemble models:',len(all_y_block_features))



    # This creates the df_base_models data file for the X and Y Block

    df_base_models = df_spls_correlation_base_models.copy()
    base_models_fn = 'x_block' + suffix_base_models_fn
    df_base_models['NumFeaturesUsed'] = df_base_models['x_block_NumFeatures']
    df_base_models['Features'] = df_base_models['x_block_Features']
    df_base_models.to_csv(path.join(runParm['models_dir'], base_models_fn),index=False)

    df_base_models = df_spls_correlation_base_models.copy()
    base_models_fn = 'y_block' + suffix_base_models_fn
    df_base_models['NumFeaturesUsed'] = df_base_models['y_block_NumFeatures']
    df_base_models['Features'] = df_base_models['y_block_Features']
    df_base_models.to_csv(path.join(runParm['models_dir'], base_models_fn),index=False)

    # This creates the df_ensemble_models data file for the X and Y Block

    df_ensemble_models = df_spls_correlation_ensemble_models.copy()
    ensemble_models_fn = 'x_block' + suffix_ensemble_models_fn
    df_ensemble_models['NumUniqueFeatures'] = df_ensemble_models['x_block_NumUniqueFeatures']
    df_ensemble_models['Features'] = df_ensemble_models['x_block_Features']
    df_ensemble_models.to_csv(path.join(runParm['models_dir'], ensemble_models_fn),index=False)

    df_ensemble_models = df_spls_correlation_ensemble_models.copy()
    ensemble_models_fn = 'y_block' + suffix_ensemble_models_fn
    df_ensemble_models['NumUniqueFeatures'] = df_ensemble_models['y_block_NumUniqueFeatures']
    df_ensemble_models['Features'] = df_ensemble_models['y_block_Features']
    df_ensemble_models.to_csv(path.join(runParm['models_dir'], ensemble_models_fn),index=False)


    # This creates the two data files for the model sensitivity analysis for the X Block

    analysis_results_dir = x_block_analysis_results_dir
    feature_cols = all_x_block_features
    modelType = 'x_block'

    df_all_features_raw = pd.read_csv(path.join(analysis_results_dir, all_features_fn))
    df_molecular_features_info = pd.read_csv(path.join(analysis_results_dir, features_info_fn))
    
    # get a copy of the info columns, then append the latent metrics, and the features
    tcns = runParm['info_cols'].copy() 
    if runParm['Compute_ACFT_Score']:
        tcns += ['ACFT_Total_Points','ACFT_Maximum_Deadlift', 'ACFT_Standing_Power_Throw', 'ACFT_Hand_Release_Pushups', 'ACFT_Sprint_Drag_Carry', 'ACFT_Leg_Tuck_OR_Plank', 'ACFT_2_Mile_Run']
    tcns += feature_cols
    tcns = getUniqueKeepOrder(tcns) 
    df_model_data_raw = df_all_features_raw[tcns].copy()
    
    df_model_feature_info = df_molecular_features_info.loc[df_molecular_features_info['featureName'].isin(feature_cols)].copy()

    df_model_data_raw.to_csv(path.join(runParm['models_dir'], modelType + '_model_data_raw.csv'),index=False)
    df_model_feature_info.to_csv(path.join(runParm['models_dir'], modelType + '_model_feature_info.csv'),index=False)


    # This creates the two data files for the model sensitivity analysis for the Y Block

    analysis_results_dir = y_block_analysis_results_dir
    feature_cols = all_y_block_features
    modelType = 'y_block'

    df_all_features_raw = pd.read_csv(path.join(analysis_results_dir, all_features_fn))
    df_molecular_features_info = pd.read_csv(path.join(analysis_results_dir, features_info_fn))
    
    # get a copy of the info columns, then append the latent metrics, and the features
    tcns = runParm['info_cols'].copy() 
    if runParm['Compute_ACFT_Score']:
        tcns += ['ACFT_Total_Points','ACFT_Maximum_Deadlift', 'ACFT_Standing_Power_Throw', 'ACFT_Hand_Release_Pushups', 'ACFT_Sprint_Drag_Carry', 'ACFT_Leg_Tuck_OR_Plank', 'ACFT_2_Mile_Run']
    tcns += feature_cols
    tcns = getUniqueKeepOrder(tcns) 
    df_model_data_raw = df_all_features_raw[tcns].copy()
    
    df_model_feature_info = df_molecular_features_info.loc[df_molecular_features_info['featureName'].isin(feature_cols)].copy()

    df_model_data_raw.to_csv(path.join(runParm['models_dir'], modelType + '_model_data_raw.csv'),index=False)
    df_model_feature_info.to_csv(path.join(runParm['models_dir'], modelType + '_model_feature_info.csv'),index=False)


    # Evaluate X Block models

    modelType = 'x_block'
    base_models_fn = modelType + suffix_base_models_fn
    ensemble_models_fn = modelType + suffix_ensemble_models_fn
    ensemble_model_performance_fn = modelType + suffix_ensemble_model_performance_fn


    print('Started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' model cv performance.')


    # load in the ensemble models and their base model information
    df_base_models = pd.read_csv(path.join(runParm['models_dir'], base_models_fn))
    df_ensemble_models = pd.read_csv(path.join(runParm['models_dir'], ensemble_models_fn))
    
    
    # load in all raw data used by ensemble models
    df_model_data_raw = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_data_raw.csv'))
    df_model_feature_info = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_feature_info.csv'))
    
    
    all_cols = df_model_data_raw.columns.to_list()
    
    
    index_list = df_ensemble_models.index.to_list()
    
    num_cv_folds = runParm['num_cv_folds']

    
    params = {'max_num_cpus': runParm['max_num_cpus'],
              'num_cv_folds': num_cv_folds,
              'knn_num_neighbors': runParm['knn_num_neighbors'],
              'numFoldGenerations': runParm['numFoldGenerations'],
              'Y_METRIC': runParm['OUTCOME_METRIC'],
              'Compute_ACFT_Score': runParm['Compute_ACFT_Score'],
              'df_ACFT_tables': df_ACFT_tables.copy(),
              'df_base_models': df_base_models,
              'base_model_name_cn': base_model_name_cn,
              'df_ensemble_models': df_ensemble_models,
              'df_model_data_raw': df_model_data_raw,
              'df_model_feature_info': df_model_feature_info,
              'all_cols': all_cols,
              'info_cols': runParm['info_cols'],
              'randomState': randomState,
              'models_dir':runParm['models_dir'],
              'index_list': index_list}

    if runParm['Compute_ACFT_Score']:
        df_ensemble_models = evaluate_ACFT_EnsembleModels(params)
    else:
        df_ensemble_models = evaluate_EnsembleModels(params)
    
    df_ensemble_models.to_csv(path.join(runParm['models_dir'], ensemble_model_performance_fn),index=False)
    
    print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' model cv performance.')



    # Evaluate Y Block models

    modelType = 'y_block'
    base_models_fn = modelType + suffix_base_models_fn
    ensemble_models_fn = modelType + suffix_ensemble_models_fn
    ensemble_model_performance_fn = modelType + suffix_ensemble_model_performance_fn

    print('Started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' model cv performance.')


    # load in the ensemble models and their base model information
    df_base_models = pd.read_csv(path.join(runParm['models_dir'], base_models_fn))
    df_ensemble_models = pd.read_csv(path.join(runParm['models_dir'], ensemble_models_fn))
    
    
    # load in all raw data used by ensemble models
    df_model_data_raw = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_data_raw.csv'))
    df_model_feature_info = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_feature_info.csv'))
    
    
    all_cols = df_model_data_raw.columns.to_list()
    
    
    index_list = df_ensemble_models.index.to_list()
    
    num_cv_folds = runParm['num_cv_folds']

    
    params = {'max_num_cpus': runParm['max_num_cpus'],
              'num_cv_folds': num_cv_folds,
              'knn_num_neighbors': runParm['knn_num_neighbors'],
              'numFoldGenerations': runParm['numFoldGenerations'],
              'Y_METRIC': runParm['OUTCOME_METRIC'],
              'Compute_ACFT_Score': runParm['Compute_ACFT_Score'],
              'df_ACFT_tables': df_ACFT_tables.copy(),
              'df_base_models': df_base_models,
              'base_model_name_cn': base_model_name_cn,
              'df_ensemble_models': df_ensemble_models,
              'df_model_data_raw': df_model_data_raw,
              'df_model_feature_info': df_model_feature_info,
              'all_cols': all_cols,
              'info_cols': runParm['info_cols'],
              'randomState': randomState,
              'models_dir':runParm['models_dir'],
              'index_list': index_list}

    if runParm['Compute_ACFT_Score']:
        df_ensemble_models = evaluate_ACFT_EnsembleModels(params)
    else:
        df_ensemble_models = evaluate_EnsembleModels(params)
    
    df_ensemble_models.to_csv(path.join(runParm['models_dir'], ensemble_model_performance_fn),index=False)
    
    print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' model cv performance.')


    
    print('')
    print('Completed Evaluate sPLS-Correlation X-Block and Y-Block ' + infoStr + 'models across folds and evaluate using all data')
    print('  End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Start:' + stepStartTime.strftime('%Y-%m-%d %H:%M:%S'))
    print('')

    
    
#################################################################################################################################################################
#####
#################################################################################################################################################################


def evaluate_ACFT_event_models(runParm, base_model_name_cn = 'Model_Name', features_info_fn = 'molecular_features_info.csv', all_features_fn = 'all_features_raw.csv', modelType = 'PNM_EA', base_models_fn = 'base_PNM_EA_models.csv', suffix_ensemble_models_fn = '_ensemble_models.csv', suffix_ensemble_model_performance_fn = '_ensemble_models_cv_performance.csv'):

    #####################################################################################################
    ##### SPECIAL CASE TO COMPUTE ACFT TOTAL POINTS FROM SIX ACFT EVENTS ENSEMBLE MODELS  ###############
    #####################################################################################################
    if runParm['Compute_ACFT_Score']:
        df_ACFT_tables = pd.read_csv(path.join(runParm['base_dir'], 'data', 'ACFT_tables_FY20_Standards.csv'))
        infoStr = 'ACFT '
    else:
        df_ACFT_tables = pd.DataFrame()
        infoStr = ''

        
    stepStartTime = datetime.datetime.now()

    randomState = runParm['randomState']
    

    ACFT_METRIC_LIST = ['ACFT_Maximum_Deadlift', 'ACFT_Standing_Power_Throw', 'ACFT_Hand_Release_Pushups', 'ACFT_Sprint_Drag_Carry', 'ACFT_Leg_Tuck_OR_Plank', 'ACFT_2_Mile_Run']

    ensemble_models_fn = 'ensemble_PNM_EA_models.csv'

    # Evaluate PNM_EA models
    # modelType = 'PNM_EA'
    # base_models_fn = modelType + suffix_base_models_fn
    # ensemble_models_fn = modelType + suffix_ensemble_models_fn
    # ensemble_model_performance_fn = modelType + suffix_ensemble_model_performance_fn
    ensemble_model_performance_fn = modelType  + '_ACFT_Events' + suffix_ensemble_model_performance_fn
    
    # print('Started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' model cv performance.')

    
    # load in the ensemble models and their base model information
    df_base_models = pd.read_csv(path.join(runParm['models_dir'], base_models_fn))
    df_base_models['Model_Name'] = df_base_models['EA_Model_Name']
    # df_ensemble_models = pd.read_csv(path.join(runParm['models_dir'], ensemble_models_fn))

    df_selected_ensemble_EA_models = pd.read_csv(path.join(runParm['models_dir'], 'selected_' + ensemble_models_fn))

    
    # get the set of features used by the ensemble models
    ls1 = ','.join(df_base_models['Features'].str.replace(' ','').str.replace('[','').str.replace(']','').str.replace("'",''))
    ls1 = ls1.split(",")
    ls1 = getUniqueKeepOrder(ls1)
    feature_cols = sorted(ls1)
    
    print('number of unique features in all base models:',len(feature_cols))


    # get the set of features used by the ensemble models
    ls1 = ','.join(df_selected_ensemble_EA_models['Features'].str.replace(' ','').str.replace('[','').str.replace(']','').str.replace("'",''))
    ls1 = ls1.split(",")
    ls1 = getUniqueKeepOrder(ls1)
    feature_cols = sorted(ls1)
    
    print('number of unique features in selected ensemble models:',len(feature_cols))


    
    # This creates the two data files for the model sensitivity analysis that includes the ACFT Metrics in addition to the raw molecular data

    df_all_features_raw = pd.read_csv(path.join(runParm['analysis_results_dir'], all_features_fn))
    df_molecular_features_info = pd.read_csv(path.join(runParm['analysis_results_dir'], features_info_fn))
    
    # get a copy of the info columns, then append the latent metrics, and the features
    tcns = runParm['info_cols'].copy() 
    #if runParm['Compute_ACFT_Score']:
    tcns += ['ACFT_Total_Points','ACFT_Maximum_Deadlift', 'ACFT_Standing_Power_Throw', 'ACFT_Hand_Release_Pushups', 'ACFT_Sprint_Drag_Carry', 'ACFT_Leg_Tuck_OR_Plank', 'ACFT_2_Mile_Run']
    tcns += feature_cols
    tcns = getUniqueKeepOrder(tcns) 
    df_model_data_raw = df_all_features_raw[tcns].copy()
    
    df_model_feature_info = df_molecular_features_info.loc[df_molecular_features_info['featureName'].isin(feature_cols)].copy()
    
    
    df_model_data_raw.to_csv(path.join(runParm['models_dir'], 'Updated_' + modelType + '_model_data_raw.csv'),index=False)
    df_model_feature_info.to_csv(path.join(runParm['models_dir'], 'Updated_' + modelType + '_model_feature_info.csv'),index=False)


    # load in all raw data used by all base models
    df_model_data_raw = pd.read_csv(path.join(runParm['models_dir'], 'Updated_' + modelType + '_model_data_raw.csv'))
    df_model_feature_info = pd.read_csv(path.join(runParm['models_dir'], 'Updated_' + modelType + '_model_feature_info.csv'))

    model_name_list = []
    Y_METRIC_list = []
    features_list = []
    df_selected_ensemble_EA_models = df_selected_ensemble_EA_models.loc[~df_selected_ensemble_EA_models['ensemble'].str.contains('_PNM-99', regex=False)].copy()
    for ind in df_selected_ensemble_EA_models.index:
        ensembleModel = df_selected_ensemble_EA_models['ensemble'][ind]
        features =  df_selected_ensemble_EA_models['Features'][ind]
        ls1 = ensembleModel.replace('[','').split("],")
        for mi in range(len(ACFT_METRIC_LIST)):
            ensemble_base_model_list = getUniqueKeepOrder(ls1[mi].replace(' ','').replace('[','').replace(']','').replace("'",'').split(","))
            Y_METRIC_list += [ACFT_METRIC_LIST[mi]]
            model_name_list += [str(ensemble_base_model_list)]
            features_list += [features]
    df_ACFT_ensemble_models = pd.DataFrame(list(zip(Y_METRIC_list, model_name_list, features_list)), columns = ['Y_METRIC', 'ensemble', 'Features'])

    df_ACFT_ensemble_models.to_csv(path.join(runParm['models_dir'], 'Updated_' + modelType + '_ACFT_ensemble_models.csv'),index=False)
    df_ACFT_ensemble_models = pd.read_csv(path.join(runParm['models_dir'], 'Updated_' + modelType + '_ACFT_ensemble_models.csv'))
    
    
    all_cols = df_model_data_raw.columns.to_list()

    list_df = []
    for Y_METRIC in ACFT_METRIC_LIST:

        print('Started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' ' + Y_METRIC + ' model cv performance.')

        # ensemble_model_performance_fn = modelType  + '_' + Y_METRIC + suffix_ensemble_model_performance_fn

        # ACFT sPLS-C models are ensemble of the six events, each of which are individual base models, and not sub ensemble models
        df_ensemble_models = df_ACFT_ensemble_models[df_ACFT_ensemble_models['Y_METRIC'] == Y_METRIC].copy()
        # df_ensemble_models['ensemble'] =  df_ensemble_models['Model_Name'].to_list()
        # df_ensemble_models['ensemble'] =  '[' + df_ensemble_models['ensemble'] + ']'
        df_ensemble_models = df_ensemble_models.reset_index(drop = True)


    
        index_list = df_ensemble_models.index.to_list()
        
        num_cv_folds = runParm['num_cv_folds']
    
        
        params = {'max_num_cpus': runParm['max_num_cpus'],
                  'num_cv_folds': num_cv_folds,
                  'knn_num_neighbors': runParm['knn_num_neighbors'],
                  'numFoldGenerations': runParm['numFoldGenerations'],
                  'Y_METRIC': Y_METRIC,
                  'Y_BIN_METRIC': runParm['OUTCOME_METRIC'], 
                  'Compute_ACFT_Score': runParm['Compute_ACFT_Score'],
                  'df_ACFT_tables': df_ACFT_tables.copy(),
                  'df_base_models': df_base_models,
                  'base_model_name_cn': base_model_name_cn,
                  'df_ensemble_models': df_ensemble_models,
                  'df_model_data_raw': df_model_data_raw,
                  'df_model_feature_info': df_model_feature_info,
                  'all_cols': all_cols,
                  'info_cols': runParm['info_cols'],
                  'randomState': randomState,
                  'models_dir':runParm['models_dir'],
                  'index_list': index_list}
    
        # if runParm['Compute_ACFT_Score']:
        #     df_ensemble_models = evaluate_ACFT_EnsembleModels(params)
        # else:
        df_ensemble_models = evaluate_EnsembleModels(params)

        list_df += [df_ensemble_models.copy()]
        # df_ensemble_models.to_csv(path.join(runParm['models_dir'], ensemble_model_performance_fn),index=False)
        
        print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' ' + Y_METRIC + ' model cv performance.')
    
    df_ensemble_models = pd.concat(list_df)
    ensemble_model_performance_fn = modelType  + '_ACFT_Events' + suffix_ensemble_model_performance_fn
    df_ensemble_models.to_csv(path.join(runParm['models_dir'], ensemble_model_performance_fn),index=False)


    
    print('')
    print('Completed Evaluate ACFT Event ' + infoStr + 'models across folds and evaluate using all data')
    print('  End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Start:' + stepStartTime.strftime('%Y-%m-%d %H:%M:%S'))
    print('')

#################################################################################################################################################################
#####
#################################################################################################################################################################


def evaluate_ACFT_event_sPLS_correlation_models(runParm, base_model_name_cn = 'Model_Name', features_info_fn = 'molecular_features_info.csv', all_features_fn = 'all_features_raw.csv', suffix_base_models_fn = '_base_models.csv', suffix_ensemble_models_fn = '_ensemble_models.csv', suffix_ensemble_model_performance_fn = '_ensemble_models_cv_performance.csv'):

    #####################################################################################################
    ##### SPECIAL CASE TO COMPUTE ACFT TOTAL POINTS FROM SIX ACFT EVENTS ENSEMBLE MODELS  ###############
    #####################################################################################################
    if runParm['Compute_ACFT_Score']:
        df_ACFT_tables = pd.read_csv(path.join(runParm['base_dir'], 'data', 'ACFT_tables_FY20_Standards.csv'))
        infoStr = 'ACFT '
    else:
        df_ACFT_tables = pd.DataFrame()
        infoStr = ''

        
    stepStartTime = datetime.datetime.now()

    randomState = runParm['randomState']
    

    ACFT_METRIC_LIST = ['ACFT_Maximum_Deadlift', 'ACFT_Standing_Power_Throw', 'ACFT_Hand_Release_Pushups', 'ACFT_Sprint_Drag_Carry', 'ACFT_Leg_Tuck_OR_Plank', 'ACFT_2_Mile_Run']


    # Evaluate X Block models

    modelType = 'x_block'
    base_models_fn = modelType + suffix_base_models_fn
    # ensemble_models_fn = modelType + suffix_ensemble_models_fn
    # ensemble_model_performance_fn = modelType + suffix_ensemble_model_performance_fn
    ensemble_model_performance_fn = modelType  + '_ACFT_Events' + suffix_ensemble_model_performance_fn
    
    # print('Started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' model cv performance.')

    
    # load in the ensemble models and their base model information
    df_base_models = pd.read_csv(path.join(runParm['models_dir'], base_models_fn))
    # df_ensemble_models = pd.read_csv(path.join(runParm['models_dir'], ensemble_models_fn))
    
    # load in all raw data used by ensemble models
    df_model_data_raw = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_data_raw.csv'))
    df_model_feature_info = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_feature_info.csv'))
    
    
    all_cols = df_model_data_raw.columns.to_list()

    list_df = []
    for Y_METRIC in ACFT_METRIC_LIST:

        print('Started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' ' + Y_METRIC + ' model cv performance.')

        # ensemble_model_performance_fn = modelType  + '_' + Y_METRIC + suffix_ensemble_model_performance_fn

        # ACFT sPLS-C models are ensemble of the six events, each of which are individual base models, and not sub ensemble models
        df_ensemble_models = df_base_models[df_base_models['Y_METRIC'] == Y_METRIC].copy()
        df_ensemble_models['ensemble'] =  df_ensemble_models['Model_Name'].to_list()
        df_ensemble_models['ensemble'] =  '[' + df_ensemble_models['ensemble'] + ']'
        df_ensemble_models = df_ensemble_models.reset_index(drop = True)

    
        index_list = df_ensemble_models.index.to_list()
        
        num_cv_folds = runParm['num_cv_folds']
    
        
        params = {'max_num_cpus': runParm['max_num_cpus'],
                  'num_cv_folds': num_cv_folds,
                  'knn_num_neighbors': runParm['knn_num_neighbors'],
                  'numFoldGenerations': runParm['numFoldGenerations'],
                  'Y_METRIC': Y_METRIC,
                  'Y_BIN_METRIC': runParm['OUTCOME_METRIC'], 
                  'Compute_ACFT_Score': runParm['Compute_ACFT_Score'],
                  'df_ACFT_tables': df_ACFT_tables.copy(),
                  'df_base_models': df_base_models,
                  'base_model_name_cn': base_model_name_cn,
                  'df_ensemble_models': df_ensemble_models,
                  'df_model_data_raw': df_model_data_raw,
                  'df_model_feature_info': df_model_feature_info,
                  'all_cols': all_cols,
                  'info_cols': runParm['info_cols'],
                  'randomState': randomState,
                  'models_dir':runParm['models_dir'],
                  'index_list': index_list}
    
        # if runParm['Compute_ACFT_Score']:
        #     df_ensemble_models = evaluate_ACFT_EnsembleModels(params)
        # else:
        df_ensemble_models = evaluate_EnsembleModels(params)

        list_df += [df_ensemble_models.copy()]
        # df_ensemble_models.to_csv(path.join(runParm['models_dir'], ensemble_model_performance_fn),index=False)
        
        print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' ' + Y_METRIC + ' model cv performance.')
    
    df_ensemble_models = pd.concat(list_df)
    ensemble_model_performance_fn = modelType  + '_ACFT_Events' + suffix_ensemble_model_performance_fn
    df_ensemble_models.to_csv(path.join(runParm['models_dir'], ensemble_model_performance_fn),index=False)



    # Evaluate Y Block models
    

    modelType = 'y_block'
    base_models_fn = modelType + suffix_base_models_fn
    # ensemble_models_fn = modelType + suffix_ensemble_models_fn
    # ensemble_model_performance_fn = modelType + suffix_ensemble_model_performance_fn

    # print('Started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' model cv performance.')


    # load in the ensemble models and their base model information
    df_base_models = pd.read_csv(path.join(runParm['models_dir'], base_models_fn))
    # df_ensemble_models = pd.read_csv(path.join(runParm['models_dir'], ensemble_models_fn))
    
    # load in all raw data used by ensemble models
    df_model_data_raw = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_data_raw.csv'))
    df_model_feature_info = pd.read_csv(path.join(runParm['models_dir'], modelType + '_model_feature_info.csv'))
    
    
    all_cols = df_model_data_raw.columns.to_list()

    list_df = []
    for Y_METRIC in ACFT_METRIC_LIST:

        print('Started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' ' + Y_METRIC + ' model cv performance.')

        # ensemble_model_performance_fn = modelType  + '_' + Y_METRIC + suffix_ensemble_model_performance_fn

        # ACFT sPLS-C models are ensemble of the six events, each of which are individual base models, and not sub ensemble models
        df_ensemble_models = df_base_models[df_base_models['Y_METRIC'] == Y_METRIC].copy()
        df_ensemble_models['ensemble'] =  df_ensemble_models['Model_Name'].to_list()
        df_ensemble_models['ensemble'] =  '[' + df_ensemble_models['ensemble'] + ']'
        df_ensemble_models = df_ensemble_models.reset_index(drop = True)

    
        index_list = df_ensemble_models.index.to_list()
        
        num_cv_folds = runParm['num_cv_folds']
    
        
        params = {'max_num_cpus': runParm['max_num_cpus'],
                  'num_cv_folds': num_cv_folds,
                  'knn_num_neighbors': runParm['knn_num_neighbors'],
                  'numFoldGenerations': runParm['numFoldGenerations'],
                  'Y_METRIC': Y_METRIC,
                  'Y_BIN_METRIC': runParm['OUTCOME_METRIC'], 
                  'Compute_ACFT_Score': runParm['Compute_ACFT_Score'],
                  'df_ACFT_tables': df_ACFT_tables.copy(),
                  'df_base_models': df_base_models,
                  'base_model_name_cn': base_model_name_cn,
                  'df_ensemble_models': df_ensemble_models,
                  'df_model_data_raw': df_model_data_raw,
                  'df_model_feature_info': df_model_feature_info,
                  'all_cols': all_cols,
                  'info_cols': runParm['info_cols'],
                  'randomState': randomState,
                  'models_dir':runParm['models_dir'],
                  'index_list': index_list}
    
        # if runParm['Compute_ACFT_Score']:
        #     df_ensemble_models = evaluate_ACFT_EnsembleModels(params)
        # else:
        df_ensemble_models = evaluate_EnsembleModels(params)

        list_df += [df_ensemble_models.copy()]
        # df_ensemble_models.to_csv(path.join(runParm['models_dir'], ensemble_model_performance_fn),index=False)
        
        print('Completed ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + 'Generate ' + infoStr + 'ensemble ' + modelType + ' ' + Y_METRIC + ' model cv performance.')
    
    df_ensemble_models = pd.concat(list_df)
    ensemble_model_performance_fn = modelType  + '_ACFT_Events' + suffix_ensemble_model_performance_fn
    df_ensemble_models.to_csv(path.join(runParm['models_dir'], ensemble_model_performance_fn),index=False)
    
    print('')
    print('Completed Evaluate ACFT Event sPLS-Correlation X-Block and Y-Block ' + infoStr + 'models across folds and evaluate using all data')
    print('  End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Start:' + stepStartTime.strftime('%Y-%m-%d %H:%M:%S'))
    print('')

#################################################################################################################################################################
#####
#################################################################################################################################################################


    

# In[ END ]:

