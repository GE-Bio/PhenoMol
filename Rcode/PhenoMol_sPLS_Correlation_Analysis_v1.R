if (FALSE) {
  while(dev.next()>1) dev.off()
  options(stringsAsFactors=FALSE)
  rm(list=objects())
}

options(stringsAsFactors=FALSE)

library("stringr")

library(mixOmics)  

MIN_NUM_MEASURES_PER_FEATURE = 3
SPLS_RANDOM_SEED = 123

TUNE_SPLS_VALIDATION  = 'Mfold'
TUNE_SPLS_NUM_FOLDS = 3
TUNE_SPLS_NUM_REPEATS = 50

NUM_COMP_TO_USE = 1
SHOW_PLOTS = FALSE

IF_ERROR_TRY_REVERSING_XY_BLOCK = TRUE

CARP_VERBOSE = FALSE

args=(commandArgs(trailingOnly = TRUE))
if(length(args) > 0){
  RUN_FROM_COMMAND_LINE = TRUE
} else {
  print("No command arguments supplied.")
  RUN_FROM_COMMAND_LINE = FALSE
}

CARP_PARAMETERS_UPDATES = list()

if(RUN_FROM_COMMAND_LINE == FALSE){
  
  # BASE_DIR = file.path(dirname(getwd()), "GEHC_IVandV_4")
  BASE_DIR = getwd()
  
  R_LIBRARY_DIR = ""
  RCODE_SOURCE_DIR = file.path(BASE_DIR, "Rcode")
  
  analysis_results_dir =  file.path(BASE_DIR, "analysis_sPLSC_WP_M_PhysACFT_Saliva_Blood_byPNM_smptr_mptd", "results") 
  # analysis_results_dir =  file.path(BASE_DIR, "analysis_WP_M_PhysACFT_Phenotype_Saliva_5_CV_Fold_Generation", "results") 
  FOLD_DIR_NAME = 'fold_002'
  ANALYSIS_RUN_SEED = 123

  # analysis_results_dir =  file.path(BASE_DIR, "analysis_sPLSC_WP_M_PhysACFT_Saliva_Blood_smptr_mptd_Permutate", "results") 
  # FOLD_DIR_NAME = 'fold_002_3'
  
  # ['C:/Program Files/R/R-4.2.3/bin/x64/Rscript', 
  #   'C:\\Users\\223109047\\Documents\\Project_DARPA_MBA\\GEHC_PhenoMol_v1/Rcode/PhenoMol_sPLS_Correlation_Analysis_v1.R', 
  #   'C:/Users/223109047.HCAD/AppData/Local/R/win-library/4.2',
  #   'C:\\Users\\223109047\\Documents\\Project_DARPA_MBA\\GEHC_PhenoMol_v1/Rcode',
  #   'C:\\Users\\223109047\\Documents\\Project_DARPA_MBA\\GEHC_PhenoMol_v1\\analysis_sPLSC_WP_M_PhysACFT_Saliva_Blood_smptr_mptd_Permutate\\results', 
  #   'fold_002_3', 
  #   '987', 
  #   'RESULTS_FILENAME=base_PNM_EA_models.csv',
  #   'Y_METRIC=ACFT_Maximum_Deadlift,ACFT_Standing_Power_Throw,ACFT_Hand_Release_Pushups,ACFT_Sprint_Drag_Carry,ACFT_Leg_Tuck_OR_Plank,ACFT_2_Mile_Run', 
  #   'BINARY_OUTCOME=FALSE', 'MIN_NUM_MEASURES_PER_FEATURE=3', 'TUNE_SPLS_NUM_FOLDS=3', 'TUNE_SPLS_NUM_REPEATS=50', 'SPLS_RANDOM_SEED=123']
  
  
  
  Y_METRIC='ACFT_Total_Points'
  
  MIN_NUM_MEASURES_PER_FEATURE=3

  # Note ACFT_Total_Points not included
  # Y_METRICS = "ACFT_Total_Points,ACFT_Maximum_Deadlift,ACFT_Standing_Power_Throw,ACFT_Hand_Release_Pushups,ACFT_Sprint_Drag_Carry,ACFT_Leg_Tuck_OR_Plank,ACFT_2_Mile_Run"
  Y_METRICS = "ACFT_Maximum_Deadlift,ACFT_Standing_Power_Throw,ACFT_Hand_Release_Pushups,ACFT_Sprint_Drag_Carry,ACFT_Leg_Tuck_OR_Plank,ACFT_2_Mile_Run"
  
} else {
  
  
  R_LIBRARY_DIR = args[[1]]
  RCODE_SOURCE_DIR = args[[2]]
  analysis_results_dir = args[[3]]
  if (length(args) > 3) {
    FOLD_DIR_NAME = args[[4]]
  } else {
    cat("Missing FOLD_DIR_NAME argument")
    quit("no")
  }
  if (length(args) > 4) {
    ANALYSIS_RUN_SEED = args[[5]]
  } else {
    cat("Missing ANALYSIS_RUN_SEED argument")
    quit("no")
  }
  

  
  
  EXPECTED_ARGS = 5
  
  if (CARP_VERBOSE) {
    cat("\n")
    for(i in 1:min(length(args),EXPECTED_ARGS)){
      cat(args[[i]],"\n")
    }
    cat("\n")  
  }
  for(i in 1:length(args)){
    if (i > EXPECTED_ARGS) {
      parmNameValuePair = as.character(unlist(strsplit(as.character(args[[i]]), "\\=")))
      objectName = parmNameValuePair[1]
      objectValue = parmNameValuePair[2]
      CARP_PARAMETERS_UPDATES[[objectName]] = objectValue
      if (CARP_VERBOSE) {
        cat(objectName, " --> " ,objectValue,"\n")
      }
    }
  }
  if (CARP_VERBOSE) {
    cat("\n")
    cat("\n")
  }
  
  
  if(.Platform$OS.type == "windows") {
    
    .libPaths(c(R_LIBRARY_DIR, .libPaths()))
    
  }
  
}




source(file.path(RCODE_SOURCE_DIR,"PhenoMol_ParmFunc_v1.R"))


if (length(CARP_PARAMETERS_UPDATES) > 0) {
  cat("Parameters being updated:","\n")
  applyCARPparameters(CARP_PARAMETERS_UPDATES)
}


if("Y_METRIC" %in% objects()) {
  if (str_count(Y_METRIC,"\\,") > 0) {
    Y_METRIC = trimws(unlist(strsplit(Y_METRIC,",")))
  }
} else {
  # should never reach this point
  stopCode("Y_METRIC parameter does not exist!")
}
if("Y_METRICS" %in% objects()) {
  if (str_count(Y_METRICS,"\\,") > 0) {
    Y_METRICS = trimws(unlist(strsplit(Y_METRICS,",")))
  }
} else {
  Y_METRICS = Y_METRIC
}


# for(ANALYSIS_RUN_SEED in c(123,321,456,654,789,987)) {
#   for(FOLD_DIR_NAME in c('fold_001','fold_002','fold_003','fold_004','fold_005')) {
  


######################################################################################################################
############### Load Data ############################################################################################
######################################################################################################################

cat(ANALYSIS_RUN_SEED, FOLD_DIR_NAME, "loading ..." , "\n")

df_x_block_latent_metric_ensemble_models = load_csv_file(file.path(analysis_results_dir,paste0("x_block_", ANALYSIS_RUN_SEED, "_" ,FOLD_DIR_NAME, "_splsc_feature_sets.csv")))
df_y_block_latent_metric_ensemble_models = load_csv_file(file.path(analysis_results_dir,paste0("y_block_", ANALYSIS_RUN_SEED, "_" ,FOLD_DIR_NAME, "_splsc_feature_sets.csv")))

df_x_block_train_fold_imputed = load_csv_file(file.path(analysis_results_dir,paste0("x_block_", ANALYSIS_RUN_SEED, "_" ,FOLD_DIR_NAME, "_train_features_imputed.csv")))
df_y_block_train_fold_imputed = load_csv_file(file.path(analysis_results_dir,paste0("y_block_", ANALYSIS_RUN_SEED, "_" ,FOLD_DIR_NAME, "_train_features_imputed.csv")))

# df_x_block_test_fold_imputed = load_csv_file(file.path(analysis_results_dir,paste0("x_block_", ANALYSIS_RUN_SEED, "_" ,FOLD_DIR_NAME, "_test_features_imputed.csv")))
# df_y_block_test_fold_imputed = load_csv_file(file.path(analysis_results_dir,paste0("y_block_", ANALYSIS_RUN_SEED, "_" ,FOLD_DIR_NAME, "_test_features_imputed.csv")))


######################################################################################################################
############### Begin of Analysis ####################################################################################
######################################################################################################################

res = data.frame(analysis_run_seed = character(),
                 fold = character(),
                 Y_METRIC = character(),
                 stringsAsFactors = FALSE)

pes = data.frame()



# make sure all are numeric
for (Y_METRIC in  Y_METRICS) {
  df_x_block_train_fold_imputed[,Y_METRIC] = as.numeric(df_x_block_train_fold_imputed[,Y_METRIC])
  df_y_block_train_fold_imputed[,Y_METRIC] = as.numeric(df_y_block_train_fold_imputed[,Y_METRIC])
}


for (Y_METRIC in  Y_METRICS) {
  
  cat(Y_METRIC,"\n")
  
  x_block_featues = df_x_block_latent_metric_ensemble_models[df_x_block_latent_metric_ensemble_models[,"Y_METRIC"] == Y_METRIC, "Features"]
  x_block_featues = gsub("\\[","",gsub("\\]","",gsub("\\'","",gsub(" ","",x_block_featues))))
  x_block_featues = trimws(unlist(strsplit(x_block_featues,",")))
  x_block_featues = unique(x_block_featues)

  x_block_data = df_x_block_train_fold_imputed[,c("UID", x_block_featues)]
  x_block_measurement_ids = paste0("xBlock.",x_block_featues)
  colnames(x_block_data) = c("UID", x_block_measurement_ids)
  
  y_block_featues = df_y_block_latent_metric_ensemble_models[df_y_block_latent_metric_ensemble_models[,"Y_METRIC"] == Y_METRIC, "Features"]
  y_block_featues = gsub("\\[","",gsub("\\]","",gsub("\\'","",gsub(" ","",y_block_featues))))
  y_block_featues = trimws(unlist(strsplit(y_block_featues,",")))
  y_block_featues = unique(y_block_featues)
  
  y_block_data = df_y_block_train_fold_imputed[,c("UID", y_block_featues)]
  y_block_measurement_ids = paste0("yBlock.",y_block_featues)
  colnames(y_block_data) = c("UID", y_block_measurement_ids)
  
  training_data = merge(x_block_data,y_block_data,by="UID")
  

    
   
  UPPER_LIMIT_MAX_NUM_COMP = floor(nrow(training_data) / MIN_NUM_MEASURES_PER_FEATURE)
  
  # note that canonical the ncomp < ncol(Y) so we subtract 1 from the number of Y features
  MAX_NUM_COMP = min(c(10, UPPER_LIMIT_MAX_NUM_COMP, length(x_block_measurement_ids), length(y_block_measurement_ids)-1))
  
  X_Block = training_data[,x_block_measurement_ids, drop = FALSE]
  Y_Block = training_data[,y_block_measurement_ids, drop = FALSE]
  
  row.names(X_Block) = training_data[,"UID"]
  row.names(Y_Block) = training_data[,"UID"]
  
  
  set.seed(SPLS_RANDOM_SEED) # for reproducibility
  
  
  sPLS_SUCCESS = FALSE
  Reverse_XY_BLOCK = FALSE
  PLS_SUCCESS = FALSE
  
  result = tryCatch({
    
    spls.correlation <- spls(X = X_Block, Y = Y_Block, ncomp = MAX_NUM_COMP, mode = "canonical")
    
    perf.spls.correlation <- perf(spls.correlation, validation = TUNE_SPLS_VALIDATION , folds = TUNE_SPLS_NUM_FOLDS, nrepeat = TUNE_SPLS_NUM_REPEATS) 
    
    Q2.total_1_Comp = perf.spls.correlation$measures$Q2.total$summary[1,"mean"]
    
    if (SHOW_PLOTS) {
      print(plot(perf.spls.correlation, criterion = 'Q2.total'))
    }
    
  
    list.keepX <- c(1:min(c(UPPER_LIMIT_MAX_NUM_COMP, (length(x_block_measurement_ids))))) 
    list.keepY <- c(1:min(c(UPPER_LIMIT_MAX_NUM_COMP, (length(y_block_measurement_ids))))) 
    
    
    # cat(format(Sys.time(), "%H:%M:%S"),"Start tune.spls","\n")
    
    tune.spls.correlation <- tune.spls(X = X_Block, Y = Y_Block, ncomp = NUM_COMP_TO_USE,
                                       test.keepX = list.keepX,
                                       test.keepY = list.keepY,
                                       folds = TUNE_SPLS_NUM_FOLDS, nrepeat = TUNE_SPLS_NUM_REPEATS,
                                       mode = 'canonical', measure = 'cor') 
    
    # cat(format(Sys.time(), "%H:%M:%S"),"End tune.spls","\n")
    
    if (SHOW_PLOTS) {
      print(plot(tune.spls.correlation))    
    }
    
    optimal.keepX = tune.spls.correlation$choice.keepX 
    optimal.keepY = tune.spls.correlation$choice.keepY
    
    
    final.spls.correlation <- spls(X = X_Block, Y = Y_Block, ncomp = NUM_COMP_TO_USE, 
                                   keepX = optimal.keepX,
                                   keepY = optimal.keepY,
                                   mode = "canonical")
    
    sPLS_SUCCESS = TRUE
    
  }, error = function(e) {
    
    sPLS_SUCCESS = FALSE
    
  })

  
  if (sPLS_SUCCESS == FALSE & IF_ERROR_TRY_REVERSING_XY_BLOCK == TRUE) {
    # Try Reverse_XY_BLOCK
    
    result = tryCatch({
      
      # Reverse_XY_BLOCK
      spls.correlation <- spls(X = Y_Block, Y = X_Block, ncomp = MAX_NUM_COMP, mode = "canonical")
      
      perf.spls.correlation <- perf(spls.correlation, validation = TUNE_SPLS_VALIDATION , folds = TUNE_SPLS_NUM_FOLDS, nrepeat = TUNE_SPLS_NUM_REPEATS) 
      
      Q2.total_1_Comp = perf.spls.correlation$measures$Q2.total$summary[1,"mean"]
      
      if (SHOW_PLOTS) {
        print(plot(perf.spls.correlation, criterion = 'Q2.total'))
      }
      
      
      # Reverse_XY_BLOCK
      list.keepY <- c(1:min(c(UPPER_LIMIT_MAX_NUM_COMP, (length(x_block_measurement_ids))))) 
      list.keepX <- c(1:min(c(UPPER_LIMIT_MAX_NUM_COMP, (length(y_block_measurement_ids))))) 
      
      
      # cat(format(Sys.time(), "%H:%M:%S"),"Start tune.spls","\n")
      
      # Reverse_XY_BLOCK
      tune.spls.correlation <- tune.spls(X = Y_Block, Y = X_Block, ncomp = NUM_COMP_TO_USE,
                                         test.keepX = list.keepX,
                                         test.keepY = list.keepY,
                                         folds = TUNE_SPLS_NUM_FOLDS, nrepeat = TUNE_SPLS_NUM_REPEATS,
                                         mode = 'canonical', measure = 'cor') 
      
      # cat(format(Sys.time(), "%H:%M:%S"),"End tune.spls","\n")
      
      if (SHOW_PLOTS) {
        print(plot(tune.spls.correlation))    
      }
      
      optimal.keepX = tune.spls.correlation$choice.keepX 
      optimal.keepY = tune.spls.correlation$choice.keepY
      
      # Reverse_XY_BLOCK
      final.spls.correlation <- spls(X = Y_Block, Y = X_Block, ncomp = NUM_COMP_TO_USE, 
                                     keepX = optimal.keepX,
                                     keepY = optimal.keepY,
                                     mode = "canonical")
      
      sPLS_SUCCESS = TRUE
      Reverse_XY_BLOCK = TRUE
      
    }, error = function(e) {
      
      sPLS_SUCCESS = FALSE
      
    })
    
    
  }
  
  
  
  if (sPLS_SUCCESS == FALSE) {
    
    result = tryCatch({
      
      spls.correlation <- pls(X = X_Block, Y = Y_Block, ncomp = MAX_NUM_COMP, mode = "canonical")
      
      perf.spls.correlation <- perf(spls.correlation, validation = TUNE_SPLS_VALIDATION , folds = TUNE_SPLS_NUM_FOLDS, nrepeat = TUNE_SPLS_NUM_REPEATS) 
      
      Q2.total_1_Comp = perf.spls.correlation$measures$Q2.total$summary[1,"mean"]
      
      if (SHOW_PLOTS) {
        print(plot(perf.spls.correlation, criterion = 'Q2.total'))
      }
      
      
      optimal.keepX = length(x_block_measurement_ids)
      optimal.keepY = length(y_block_measurement_ids)
      
      
      final.spls.correlation <- pls(X = X_Block, Y = Y_Block, ncomp = NUM_COMP_TO_USE, mode = "canonical")
      
      PLS_SUCCESS = TRUE
      
    }, error = function(e) {
      
      PLS_SUCCESS = FALSE
      
    })
    
  }
  
  
  
  
  if (sPLS_SUCCESS == FALSE & IF_ERROR_TRY_REVERSING_XY_BLOCK == TRUE) {
    
    result = tryCatch({
      
      # Reverse_XY_BLOCK
      spls.correlation <- pls(X = Y_Block, Y = X_Block, ncomp = MAX_NUM_COMP, mode = "canonical")
      
      perf.spls.correlation <- perf(spls.correlation, validation = TUNE_SPLS_VALIDATION , folds = TUNE_SPLS_NUM_FOLDS, nrepeat = TUNE_SPLS_NUM_REPEATS) 
      
      Q2.total_1_Comp = perf.spls.correlation$measures$Q2.total$summary[1,"mean"]
      
      if (SHOW_PLOTS) {
        print(plot(perf.spls.correlation, criterion = 'Q2.total'))
      }
      
      # Reverse_XY_BLOCK
      optimal.keepY = length(x_block_measurement_ids)
      optimal.keepX = length(y_block_measurement_ids)
      
      # Reverse_XY_BLOCK
      final.spls.correlation <- pls(X = Y_Block, Y = X_Block, ncomp = NUM_COMP_TO_USE, mode = "canonical")
      
      PLS_SUCCESS = TRUE
      Reverse_XY_BLOCK = TRUE
      
    }, error = function(e) {
      
      PLS_SUCCESS = FALSE
      
    })
    
  }
  
  
  
    
  if (sPLS_SUCCESS == FALSE & PLS_SUCCESS == FALSE) {
    
      Q2.total_1_Comp = NA
      Pearson_Cor_Block_X_vs_Y_comp1 = NA
      Spearman_Cor_Block_X_vs_Y_comp1 = NA
      optimal.keepX = c(NA)
      optimal.keepY = c(NA)
  
      optimal_keepX = optimal.keepX
      optimal_keepY = optimal.keepY
      
      x_block_finalFeatures = x_block_measurement_ids
      y_block_finalFeatures = y_block_measurement_ids
    
  } else {

    if (Reverse_XY_BLOCK == TRUE) {
      
      final_spls_correlation_variates_Y = final.spls.correlation$variates$X
      final_spls_correlation_variates_X = final.spls.correlation$variates$Y
      final_spls_correlation_loadings_Y = final.spls.correlation$loadings$X
      final_spls_correlation_loadings_X = final.spls.correlation$loadings$Y
      optimal_keepY = optimal.keepX
      optimal_keepX = optimal.keepY
      
    } else {
      
      final_spls_correlation_variates_X = final.spls.correlation$variates$X
      final_spls_correlation_variates_Y = final.spls.correlation$variates$Y
      final_spls_correlation_loadings_X = final.spls.correlation$loadings$X
      final_spls_correlation_loadings_Y = final.spls.correlation$loadings$Y
      optimal_keepX = optimal.keepX
      optimal_keepY = optimal.keepY
    }

    
    
    
  
    if (SHOW_PLOTS) {
      for(nc in c(1:NUM_COMP_TO_USE)) {
        print(plotLoadings(final.spls.correlation, comp = nc))
      }
    }
    
    pDat = data.frame(final_spls_correlation_variates_X)
    colnames(pDat) = paste0("X_",colnames(pDat))
    pDat[,"UID"] = row.names(pDat)
    row.names(pDat) = NULL
    pDat.x = pDat
    
    pDat = data.frame(final_spls_correlation_variates_Y)
    colnames(pDat) = paste0("Y_",colnames(pDat))
    pDat[,"UID"] = row.names(pDat)
    row.names(pDat) = NULL
    pDat.y = pDat
    
    pDat = merge(pDat.x,pDat.y,by="UID")
    
    # record correlation between the X and Y block variants
    Pearson_Cor_Block_X_vs_Y_comp1 = cor(pDat[,"X_comp1"],pDat[,"Y_comp1"], method = "pearson")
    Spearman_Cor_Block_X_vs_Y_comp1 = cor(pDat[,"X_comp1"],pDat[,"Y_comp1"], method = "spearman")
    
    if (SHOW_PLOTS) {
      # fit = lm(pDat[,"Y_comp1"]~pDat[,"X_comp1"])
      plot(pDat[,"X_comp1"],pDat[,"Y_comp1"],xlab="X-Block Component 1 Variants",ylab="Y-Block Component 1 Variants",yaxt='n',main=Y_METRIC)
      axis(2,las=1)
      # abline(fit, col = 'red',lty = 2)
    }
    
    pDat = merge(df_x_block_train_fold_imputed[,c("UID",Y_METRIC)],pDat,by="UID")


    if (SHOW_PLOTS) {
      fit = lm(pDat[,Y_METRIC]~pDat[,"X_comp1"])
      plot(pDat[,"X_comp1"],pDat[, Y_METRIC],xlab="X-Block Component 1 Variants",ylab=Y_METRIC,yaxt='n', main = paste0("Adjusted R Squared: ", format(summary(fit)$adj.r.squared,digits = 3)))
      axis(2,las=1)
      abline(fit, col = 'red',lty = 2)
    }
    
    if (SHOW_PLOTS) {
      fit = lm(pDat[,Y_METRIC]~pDat[,"Y_comp1"])
      plot(pDat[,"Y_comp1"],pDat[, Y_METRIC],xlab="Y-Block Component 1 Variants",ylab=Y_METRIC,yaxt='n', main = paste0("Adjusted R Squared: ", format(summary(fit)$adj.r.squared,digits = 3)))
      axis(2,las=1)
      abline(fit, col = 'red',lty = 2)
  
    }
    
    
    cns = colnames(pDat)
    colnames(pDat) = c(cns[1],cns[2], paste0(cns[3],"_of_", Y_METRIC), paste0(cns[4],"_of_", Y_METRIC))
    
    if (nrow(pes) == 0) {
      pes = pDat
    } else {
      pes = merge(pes,pDat,by="UID")
    }
    
    
    
    df = as.data.frame(final_spls_correlation_loadings_X)
    for(ci in 1:ncol(df)) {
      df[,ci] = as.numeric( df[,ci])
    }
    df.SumAbsRows = apply(df, 1, function(x) {
      return(sum(abs(as.numeric(x))))
    })
    df = df[df.SumAbsRows > 0, , drop = FALSE]
    df[,"x_block_measurement_ids"] = row.names(df)
    row.names(df)  = NULL
    df[,"measurement_id"] = apply(df,1,function(x) {
      return(substr(x["x_block_measurement_ids"],3,nchar(x["x_block_measurement_ids"])))
    })
    pcns = c("measurement_id", "x_block_measurement_ids")
    cns = colnames(df)
    df = df[,c(pcns, cns[!cns %in% pcns])]
    df = df[order(abs(df[,"comp1"]),decreasing = TRUE),]
    row.names(df) = NULL
    df_X_Loadings = df
    
    
    
    df = as.data.frame(final_spls_correlation_loadings_Y)
    for(ci in 1:ncol(df)) {
      df[,ci] = as.numeric( df[,ci])
    }
    df.SumAbsRows = apply(df, 1, function(x) {
      return(sum(abs(as.numeric(x))))
    })
    df = df[df.SumAbsRows > 0, , drop = FALSE]
    df[,"y_block_measurement_ids"] = row.names(df)
    row.names(df)  = NULL
    df[,"measurement_id"] = apply(df,1,function(x) {
      return(substr(x["y_block_measurement_ids"],3,nchar(x["y_block_measurement_ids"])))
    })
    pcns = c("measurement_id", "y_block_measurement_ids")
    cns = colnames(df)
    df = df[,c(pcns, cns[!cns %in% pcns])]
    df = df[order(abs(df[,"comp1"]),decreasing = TRUE),]
    row.names(df) = NULL
    df_Y_Loadings = df
  
    x_block_finalFeatures = df_X_Loadings[!is.na(df_X_Loadings[ ,"comp1"]) & abs(df_X_Loadings[ ,"comp1"]) > 0,"x_block_measurement_ids"]
    y_block_finalFeatures = df_Y_Loadings[!is.na(df_Y_Loadings[ ,"comp1"]) & abs(df_Y_Loadings[ ,"comp1"]) > 0,"y_block_measurement_ids"]
    
    
  }
      

  
  
  ##############################################################
  # Perform sPLS-Regression on Y_METRIC vs x-block features and y-block features 
  

  

  x_block_data = df_x_block_train_fold_imputed[,c("UID", Y_METRIC, gsub("xBlock.","",x_block_finalFeatures))]
  colnames(x_block_data) = c("UID", Y_METRIC, x_block_finalFeatures)
  
  y_block_data = df_y_block_train_fold_imputed[,c("UID", gsub("yBlock.","",y_block_finalFeatures))]
  colnames(y_block_data) = c("UID", y_block_finalFeatures)
  
  latent_training_data = merge(x_block_data,y_block_data,by="UID")
  

  
  y_train <- latent_training_data[,Y_METRIC]
  names(y_train) = latent_training_data[,"UID"]
  
  #### Retrain X block features to Y_METRIC ####################
  

  X_final = latent_training_data[, x_block_finalFeatures, drop= FALSE]
  row.names(X_final) = latent_training_data[, "UID"]
  if (ncol(X_final) == 1) {
    
    trainData = data.frame(y=(y_train), x= X_final[, 1])
    plsModelFinal = lm(y ~ x, data = trainData)
    
    train_predict_plsModelFinal <- data.frame(predict = predict(plsModelFinal, newdata = data.frame(x=X_final[, 1])))
    row.names(train_predict_plsModelFinal) = row.names(X_final)
    
  } else {
    
    
    plsModelFinal <- mixOmics::pls(X_final, y_train, ncomp = 1, mode = "regression") 
    
    
    train_predict_plsModelFinal <- predict(plsModelFinal, X_final)
  }
  
  SSres = sum((as.numeric(y_train) - as.numeric(train_predict_plsModelFinal$predict))^2)
  SStot = sum((as.numeric(y_train) - mean(as.numeric(y_train)))^2)
  x_block_train_R2 = 1 - SSres / SStot
  
  x_block_train_mse = SSres / length(y_train)
  
  
  #### Retrain Y block features to Y_METRIC ####################
  
  

  X_final = latent_training_data[, y_block_finalFeatures, drop= FALSE]
  if (ncol(X_final) == 1) {
    
    trainData = data.frame(y=(y_train), x= X_final[, 1])
    plsModelFinal = lm(y ~ x, data = trainData)
    
    train_predict_plsModelFinal <- data.frame(predict = predict(plsModelFinal, newdata = data.frame(x=X_final[, 1])))
    row.names(train_predict_plsModelFinal) = row.names(X_final)
    
  } else {
    
    
    plsModelFinal <- mixOmics::pls(X_final, y_train, ncomp = 1, mode = "regression") 
    
    
    train_predict_plsModelFinal <- predict(plsModelFinal, X_final)
  }
  
  SSres = sum((as.numeric(y_train) - as.numeric(train_predict_plsModelFinal$predict))^2)
  SStot = sum((as.numeric(y_train) - mean(as.numeric(y_train)))^2)
  y_block_train_R2 = 1 - SSres / SStot
  
  y_block_train_mse = SSres / length(y_train)
  

  
  
  ri = nrow(res) + 1
  res[ri,"analysis_run_seed"] = ANALYSIS_RUN_SEED
  res[ri,"fold"] = FOLD_DIR_NAME
  res[ri,"Y_METRIC"] = Y_METRIC

  res[ri,"sPLS_SUCCESS"] = sPLS_SUCCESS
  res[ri,"PLS_SUCCESS"] = PLS_SUCCESS
  res[ri,"Reverse_XY_BLOCK"] = Reverse_XY_BLOCK
  
  res[ri,"Q2.total_1_Comp"] = Q2.total_1_Comp  
  res[ri,"keepX"] = optimal_keepX[1]
  res[ri,"keepY"] = optimal_keepY[1]

  

  
  # res[ri,"X_Block_Comp1.adj.r.squared"] = summary(fit)$adj.r.squared
  # res[ri,"Y_Block_Comp1.adj.r.squared"] = summary(fit)$adj.r.squared
  
  res[ri,"pearson_Cor_Block_X_vs_Y_comp1"] = Pearson_Cor_Block_X_vs_Y_comp1
  res[ri,"spearman_Cor_Block_X_vs_Y_comp1"] = Spearman_Cor_Block_X_vs_Y_comp1
  
  
  
  res[ri,"x_block_NumFeatures"] = length(x_block_finalFeatures)
  res[ri,"x_block_Features"] = paste0(gsub("xBlock.","",x_block_finalFeatures),collapse = ",")
  res[ri,"x_block_train_R2"] = x_block_train_R2
  res[ri,"x_block_train_mse"] = x_block_train_mse
  
  
  res[ri,"y_block_NumFeatures"] = length(y_block_finalFeatures)
  res[ri,"y_block_Features"] = paste0(gsub("yBlock.","",y_block_finalFeatures),collapse = ",")
  res[ri,"y_block_train_R2"] = y_block_train_R2
  res[ri,"y_block_train_mse"] = y_block_train_mse
  
  

}


# for each latent metric, perform sPLS-correlation analysis
# for each latent metric, Perform sPLS-Regression on Y_METRIC vs x-block features and y-block features 

write.csv(res, file.path(analysis_results_dir, paste0("spls_correlation_", ANALYSIS_RUN_SEED, "_", FOLD_DIR_NAME, "_results.csv")),row.names = FALSE)

write.csv(pes, file.path(analysis_results_dir, paste0("block_comp1_variants_", ANALYSIS_RUN_SEED, "_", FOLD_DIR_NAME, ".csv")),row.names = FALSE)

#   }
# }

####################################################################################


if(RUN_FROM_COMMAND_LINE == TRUE){
  quit("no")
}




