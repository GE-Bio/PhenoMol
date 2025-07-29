#!/usr/bin/env python3

import os
from os import path
import datetime
import warnings
# warnings.filterwarnings('ignore')

from phenoMol_ML_v1 import getUniqueKeepOrder
from phenoMol_ML_v1 import get_default_run_parameters
from phenoMol_ML_v1 import setup_analysis_run
from phenoMol_ML_v1 import save_run_parameters
from phenoMol_ML_v1 import run_analysis_step_1
from phenoMol_ML_v1 import run_analysis_step_2
from phenoMol_ML_v1 import run_analysis_step_3
from phenoMol_ML_v1 import run_analysis_step_3_plot

def set_run_parameters(runParm):

    ################################# CORRECT THIS ##############################################
    base_dir = os.path.dirname(os.getcwd())
    #############################################################################################
    
    # path to R
    r_path = 'C:/Program Files/R/R-4.2.3/bin/x64/Rscript'
    
    # path for R to find user installed packages
    R_LIBRARY_DIR = os.path.expanduser('~').replace('\\','/') + '/AppData/Local/R/win-library/4.2'
    
    # path to R scripts to be run
    RCODE_SOURCE_DIR = base_dir + '/Rcode'
    
        
    # the interactome directory is within the same parent directory as the python code directory that is set as the current working directory
    interactomeName = 'ppmi_70_percent_confidence_v5'   # 70% Confidence Interactome, 
    interactome_db_dir = path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),'interactome', interactomeName)


    max_num_cpus = runParm['max_num_cpus']

    randomState = runParm['randomState']
    
    permutationState = runParm['permutationState']

    oi_run_phenotype_critical_p_value  = runParm['oi_run_phenotype_critical_p_value']

    misclassifyOutcomeFraction = runParm['misclassifyOutcomeFraction']

    build_PN_With_Outcome = True
    build_PN_With_Latent_Metrics = False
    build_PN_With_PAs = False 
    build_PN_With_Phenotypes = False
    build_PAs = True
    
    RUN_MODEL_FLAVOR = "VO2"
    OUTCOME_METRIC = 'Peak_VO2_mL_kg_min' 
    LATENT_METRIC_LIST = []
     
    info_cols = ['UID', 'outcome', OUTCOME_METRIC,'IR_IS_classification', 'Sex','Gender','Race_Ethnicity','Smoker','Diabetes','Insulin_resistant','Recovery_BP_comment','ASCVD_risk_score','Adjusted_ASCVD_risk_score','SSPG','Peak_VO2_mL_kg_min','VO2_Rest_mL_kg_min']
    # info_cols += LATENT_METRIC_LIST

    # List phenotypes otherwise leave blank to use all available ones
    # phenotype_feature_list = []
    phenotype_feature_list = LATENT_METRIC_LIST

    # The following phenotype features not to be used from the available ones
    # remove_phenotype_feature_list = []
    # # The following are problem phenotype features for sPLS-R because they are basically binary (True/False)
    remove_phenotype_feature_list = ['Hypertensive_GT_140_90','Hypertensive_GT_130_80','Presence_of_Carotid_Plaque','Right_Carotid_Plaque','Left_carotid_plaque','Abdominal_Aorta_Plaque','Right_Femoral_Plaque','Left_Femoral_Plaque','RER_GT_1pt05','Ventricular_Ectopics_Ex','Ventricular_Ecoptics_Rec','Arrythmia']

    # The following will be removed for sPLS-R for the PAs
    # remove_phenotype_PA_feature_list =  []
    # remove_phenotype_PA_feature_list =  LATENT_METRIC_LIST
    remove_phenotype_PA_feature_list = remove_phenotype_feature_list

    
    omicsTypes = 'mptp'   
    omicMeasures = ['mDat_lipidomics.csv','mDat_metabolomics.csv','mDat_proteomics.csv','mDat_transcriptomics.csv','mDat_HMP_targeted_proteins.csv']
    
    use_RNA = True
    use_DNA = True
    use_MicroRNA = False
    NUM_OI_FOLDS = 3
    RUN_COHORT = 'CAE'
    RUN_COHORT = 'CAE'
    RUN_MODEL_GROUP = 'MFn'
    SEX_COLUMN_NAME = 'Sex'
    SEX_VALUES = ['M','F']
    USE_SEX_MEDIAN_NORMALIZED_DATA = True

    db_data_dir = path.join(base_dir,'db_CAE')
    
    RUN_VERSION = str(NUM_OI_FOLDS) + '_' + omicsTypes
    RUN_VERSION += '_' 
    if build_PN_With_Outcome:
        RUN_VERSION += 'o'  
    if build_PN_With_PAs:
        RUN_VERSION += 'pa'  
    if build_PN_With_Latent_Metrics or build_PN_With_Phenotypes:
        RUN_VERSION += 'p'  
    if build_PAs and not build_PN_With_PAs:
        RUN_VERSION += '_pa'  
    # if build_PN_With_PAs or build_PN_With_Phenotypes:
    #     RUN_VERSION += '{:02.0f}'.format(100*oi_run_phenotype_critical_p_value).replace('.','_')
    if misclassifyOutcomeFraction > 0:
        RUN_VERSION += '_rand{:02.0f}'.format(100*misclassifyOutcomeFraction).replace('.','_')
    RUN_VERSION += '_' + str(randomState)

    # RUN_VERSION += '_IVaV4code'   ####################

    if permutationState < 0:
        analysis_dir = path.join(base_dir,('analysis_' + RUN_COHORT + '_' + RUN_MODEL_GROUP + '_' + RUN_MODEL_FLAVOR + '_' + RUN_VERSION))
    else:
        analysis_dir = path.join(base_dir,('permRun_' + str(permutationState) + '_' + RUN_COHORT + '_' + RUN_MODEL_GROUP + '_' + RUN_MODEL_FLAVOR + '_' + RUN_VERSION))

    analysis_results_dir = path.join(analysis_dir, 'results')

    models_dir = path.join(analysis_dir, 'models')

    phenotypeMeasures = ['mDat_phenotypes.csv']
    phenotype_mDat_prefix = 'mDat_'
    omic_mDat_prefix = 'mDat_'
 
    tuningPhenotypes = [OUTCOME_METRIC]

    # directory that contains the raw hDat and mDat data files for the entire cohort
    db_cohort_data_measures_dir = path.join(db_data_dir, 'measures')
    
    # directory that contains the hDat and mDat data files for the specific analysis
    input_measures_dir = path.join(analysis_dir, 'measures')
    
    # directory for the output phenotype prize set files
    oi_prize_dir = path.join(analysis_dir,'oi_prizes')
    
    pcsf_run_dir = path.join(analysis_dir, 'oi_runs')
    
    oi_network_dir = path.join(analysis_dir, 'oi_network')
    
    if OUTCOME_METRIC not in info_cols:
        info_cols = info_cols + [OUTCOME_METRIC]
    
    # This is to introduce random misclassification into the training data for the fold
    if misclassifyOutcomeFraction > 0 and 'ground_truth' not in info_cols:
        info_cols = info_cols + ['ground_truth']
            

    info_cols = getUniqueKeepOrder(info_cols) 

    prize_data_raw_fn = 'oi3_PrizeDataRaw.zip'  
    prize_data_final_fn = 'oi3_PrizeDataFinal.zip'
    prize_name_info_fn = 'oi3_PrizeNameInfo.csv'


    if build_PAs or build_PN_With_Latent_Metrics or build_PN_With_PAs or build_PN_With_Phenotypes:
        Only_Outcome_No_Phenotypes = False
    else:
        Only_Outcome_No_Phenotypes = True
        
    runParm = { **runParm, 

        'base_dir': base_dir,
        'r_path': r_path,
        'R_LIBRARY_DIR': R_LIBRARY_DIR,
        'RCODE_SOURCE_DIR': RCODE_SOURCE_DIR,
        'interactome_db_dir': interactome_db_dir, 
        'interactomeName': interactomeName, 
        
        'analysis_dir': analysis_dir,
        'analysis_results_dir': analysis_results_dir,
        'models_dir': models_dir,
        
        'pcsf_run_dir': pcsf_run_dir,
        'oi_network_dir': oi_network_dir,

        'prize_data_raw_fn': prize_data_raw_fn,
        'prize_data_final_fn': prize_data_final_fn,
        'prize_name_info_fn': prize_name_info_fn,
               
        'prize_data_raw_fp': path.join(oi_prize_dir, prize_data_raw_fn),
        'prize_name_info_fp': path.join(oi_prize_dir, prize_name_info_fn),
        'prize_data_final_fp': path.join(oi_prize_dir, prize_data_final_fn),

        'prizeDataFinal_fp': path.join(pcsf_run_dir, prize_data_final_fn),
        'prizeNameInfo_fp': path.join(pcsf_run_dir, prize_name_info_fn),
               
        'oirun_interactome_fp': path.join(pcsf_run_dir, interactomeName + '_interactome.zip'),
        'prizes_all_fp': path.join(pcsf_run_dir, interactomeName + '_prizes_all.zip'),
              
        'db_data_dir': db_data_dir,
        'db_cohort_data_measures_dir':db_cohort_data_measures_dir,
    
        'input_measures_dir': input_measures_dir, 
        'oi_prize_dir': oi_prize_dir, 

               
        'NUM_OI_FOLDS': NUM_OI_FOLDS,
               
        'OUTCOME_METRIC': OUTCOME_METRIC,
        'BINARY_OUTCOME': False,
               
        'LATENT_METRIC_LIST': LATENT_METRIC_LIST,
        
        'phenotypeMeasures' : phenotypeMeasures,
        'omicMeasures' : omicMeasures,
               
        'use_RNA': use_RNA,
        'use_DNA': use_DNA,
        'use_MicroRNA': use_MicroRNA,

        'tuningPhenotypes' : tuningPhenotypes,  # hyperparameter search across phenotypes
    
        # file name prefix to measurements (mDat files), and the meta information (hDat files)
        'phenotype_mDat_prefix' : phenotype_mDat_prefix,
        'omic_mDat_prefix' : omic_mDat_prefix,
        'hDat_prefix' : 'hDat_',
        'info_cols' : info_cols,
        'UID' : 'UID',
        'SEX_COLUMN_NAME' : SEX_COLUMN_NAME, 
        'SEX_VALUES' : SEX_VALUES, 
        'USE_SEX_MEDIAN_NORMALIZED_DATA' : USE_SEX_MEDIAN_NORMALIZED_DATA,

        'build_PN_With_Outcome': build_PN_With_Outcome,
        'build_PN_With_Latent_Metrics': build_PN_With_Latent_Metrics,
        'build_PN_With_PAs': build_PN_With_PAs,
        'build_PN_With_Phenotypes': build_PN_With_Phenotypes,
        'build_PAs': build_PAs, 
        'Only_Outcome_No_Phenotypes': Only_Outcome_No_Phenotypes,

        'phenotype_feature_list': phenotype_feature_list,
        'remove_phenotype_feature_list': remove_phenotype_feature_list,
        'remove_phenotype_PA_feature_list': remove_phenotype_PA_feature_list,

               
        'include_hold_out_test_subjects': False}

    return runParm


def main():
    
    analysisStart = datetime.datetime.now()
    
    # ~ 15 minutes for one analysis run : 12th Gen Intel(R) Core(TM) i7-12700H, 2300 Mhz, 14 Core(s), 20 Logical Processor(s)
    
    max_num_processors = 6
    randomState = 654
    
    runParm = get_default_run_parameters(max_num_processors = max_num_processors).copy()
    runParm = { **runParm,         
        'min_num_correlation_data_points' : 5,  # default is 10
        # 'INCLUDE_PNM_MINUS_99': True,
        'randomState': randomState}  

    runParm = set_run_parameters(runParm)
    
    print(analysisStart.strftime('%Y-%m-%d %H:%M:%S') + ' ' + 'Started Analysis Run: ' + os.path.basename(runParm['analysis_dir']))
    
    setup_analysis_run(runParm)
    save_run_parameters(runParm)
    run_analysis_step_1(runParm)

    if runParm['build_PAs'] or runParm['build_PN_With_PAs']:
        run_analysis_step_2(runParm, modelType = 'PA', ensemble_models_fn = 'ensemble_PA_models.csv', nck_ensemble_models_fn = 'nck_ensemble_PA_models.csv')
        run_analysis_step_3(runParm, modelType = 'PA', base_model_name_cn = 'PA_Model_Name', base_models_fn = 'base_PA_models.csv', ensemble_models_fn = 'ensemble_PA_models.csv' , nck_ensemble_models_fn = 'nck_ensemble_PA_models.csv', features_info_fn = 'phenotype_features_info.csv', all_features_fn = 'all_features_raw.csv', ensemble_model_performance_fn = 'ensemble_PA_models_cv_performance.csv')
        # run_analysis_step_3_plot(runParm, Model_Rank = 1, RANKING_METRIC = 'all_cv_test_R2_mean', RANKING_METRIC_ascending = False, modelType = 'PA',  base_model_name_cn = 'PA_Model_Name',  base_models_fn = 'base_PA_models.csv', ensemble_model_performance_fn = 'ensemble_PA_models_cv_performance.csv')

    run_analysis_step_2(runParm, modelType = 'PNM_EA', ensemble_models_fn = 'ensemble_PNM_EA_models.csv', nck_ensemble_models_fn = 'nck_ensemble_EA_models.csv')
    run_analysis_step_3(runParm, modelType = 'PNM_EA', base_model_name_cn = 'EA_Model_Name', base_models_fn = 'base_PNM_EA_models.csv', ensemble_models_fn = 'ensemble_PNM_EA_models.csv' , nck_ensemble_models_fn = 'nck_ensemble_EA_models.csv', features_info_fn = 'molecular_features_info.csv', all_features_fn = 'all_features_raw.csv', ensemble_model_performance_fn = 'ensemble_EA_models_cv_performance.csv')
    
    run_analysis_step_3_plot(runParm, Model_Rank = 1, RANKING_METRIC = 'all_cv_test_R2_mean', RANKING_METRIC_ascending = False, modelType = 'PNM_EA',  base_model_name_cn = 'EA_Model_Name',  base_models_fn = 'base_PNM_EA_models.csv', ensemble_model_performance_fn = 'ensemble_EA_models_cv_performance.csv')

    if runParm['INCLUDE_PNM_MINUS_99']:
        run_analysis_step_3_plot(runParm, plotData = False, plotPNM99 = True, Model_Rank = 1, RANKING_METRIC = 'all_cv_test_R2_mean', RANKING_METRIC_ascending = False, modelType = 'PNM_EA',  base_model_name_cn = 'EA_Model_Name',  base_models_fn = 'base_PNM_EA_models.csv', ensemble_model_performance_fn = 'ensemble_EA_models_cv_performance.csv')

    print('')
    print('Completed Analysis Run: ' + os.path.basename(runParm['analysis_dir']))
    print('  End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Start:' + analysisStart.strftime('%Y-%m-%d %H:%M:%S'))
    print('')


if __name__ == '__main__':
    main()




