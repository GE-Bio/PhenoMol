if (FALSE) {
  while(dev.next()>1) dev.off()
  options(stringsAsFactors=FALSE)
  rm(list=objects())
}

options(stringsAsFactors=FALSE)

library("stringr")

library(mixOmics)  

RESULTS_FILENAME = "spls_PA_model_results.csv"
INPUT_TRAIN_PA_FEATURES_SUFFIX_FILENAME = "_train_PA_features.csv"
INPUT_PA_PHENOTYPE_RANKING_SUFFIX_FILENAME = "_PA_Phenotype_Ranking.csv"
SPLS_RANDOM_SEED = 123
PA_MAX_N_FEATURES = 100
PA_NUM_COMPONENTS = 1
MIN_NUM_MEASURES_PER_FEATURE = 3
TUNE_SPLS_MEASURE  = 'MSE'
TUNE_SPLS_VALIDATION  = 'Mfold'
TUNE_SPLS_NUM_FOLDS = 3
TUNE_SPLS_NUM_REPEATS = 50
TUNE_SPLS_MIN_FEATURE_STABILITY = 0.8
Y_METRIC = "outcome"
BINARY_OUTCOME = FALSE

PLSDA_DISTANCE_METHOD = "max.dist"    # "mahalanobis.dist"

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
  analysis_results_dir =  file.path(BASE_DIR, "analysis_WP_M_PhysACFT_Blood_5_mptd_op_456", "results") 
  FOLD_DIR_NAME = 'fold_001'
  
  analysis_results_dir =  file.path(BASE_DIR, "analysis_WP_F_PhysACFT_Blood_5_mptd_op_pa_456", "results") 
  FOLD_DIR_NAME = 'fold_003'
  
  RESULTS_FILENAME = "test_spls_PA_model_results.csv"
  
  Y_METRIC='ACFT_Total_Points'
  BINARY_OUTCOME=FALSE
  MIN_NUM_MEASURES_PER_FEATURE=3
  PA_MAX_N_FEATURES=20

  

  # Y_METRIC = c(Y_METRIC,'ACFT_Points_Maximum_Deadlift', 'ACFT_Points_Standing_Power_Throw', 'ACFT_Points_Hand_Release_Pushups', 'ACFT_Points_Sprint_Drag_Carry', 'ACFT_Points_Leg_Tuck_OR_Plank', 'ACFT_Points_2_Mile_Run')
  # Y_METRICS = paste0(c(Y_METRIC,'ACFT_Points_Maximum_Deadlift'),collapse = ", ")
  Y_METRICS = paste0(c(Y_METRIC),collapse = ", ")
  
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
  
  EXPECTED_ARGS = 4
  
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


######################################################################################################################
############### Load Data ############################################################################################
######################################################################################################################



cat(FOLD_DIR_NAME, "loading ..." , "\n")
df_train_features_imputed = load_csv_file(file.path(analysis_results_dir,FOLD_DIR_NAME ,paste0(FOLD_DIR_NAME, INPUT_TRAIN_PA_FEATURES_SUFFIX_FILENAME)))


df_PA_Phenotype_Ranking = load_csv_file(file.path(analysis_results_dir,FOLD_DIR_NAME ,paste0(FOLD_DIR_NAME, INPUT_PA_PHENOTYPE_RANKING_SUFFIX_FILENAME)))
all_phenotype_names = df_PA_Phenotype_Ranking[, "phenotype"]
all_phenotype_names = all_phenotype_names[all_phenotype_names %in% colnames(df_train_features_imputed)]


######################################################################################################################
############### Begin of Analysis ####################################################################################
######################################################################################################################

analysis_dir_name = basename(dirname(analysis_results_dir))

res = data.frame(analysis_dir = character(),
                 fold = character(),
                 stringsAsFactors = FALSE)



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

# remove any latent metrics from phenotype list
all_phenotype_names = all_phenotype_names[!all_phenotype_names %in% Y_METRICS]

for (Y_METRIC in  Y_METRICS) {
  

  
  featureRanking = data.frame(Feature=all_phenotype_names,stringsAsFactors = TRUE)
  featureRanking[,"Feature"] = as.character(featureRanking[,"Feature"])
  
  tDat = cbind(data.frame(UID = df_train_features_imputed[,"UID"], y = df_train_features_imputed[,Y_METRIC]),df_train_features_imputed[,all_phenotype_names, drop=FALSE])
  
  corRes = apply(featureRanking, 1, function(x) {
                                  featureName = x["Feature"]
                                  cDat = na.omit(tDat[,c("y",featureName)])
                                  numUniqueValues = length(unique(cDat[,featureName]))
                                  if (nrow(cDat) > 2) {
                                    if (sd(cDat[,1]) > 0 & sd(cDat[,2]) > 0) {
                                      corTest = cor.test(cDat[,1],cDat[,2],method="spearman")
                                      return (c(as.numeric(corTest$estimate),as.numeric(corTest$p.value), numUniqueValues))
                                    } else {
                                      return (c(NA,1,numUniqueValues))
                                    }
                                  } else {
                                    return (c(NA,1,numUniqueValues))
                                  }
                                })
  featureRanking[ ,"abs.cor.spearman"] = abs(as.numeric(corRes[1,]))
  featureRanking[ ,"spearman.pValue"] = as.numeric(corRes[2,])
  featureRanking[ ,"numUniqueValues"] = as.numeric(corRes[3,])
  featureRanking = featureRanking[order(featureRanking[,"Feature"]), ]
  featureRanking = featureRanking[order(featureRanking[,"abs.cor.spearman"],decreasing = TRUE), ]
  featureRanking = featureRanking[order(featureRanking[,"spearman.pValue"]), ]
  row.names(featureRanking) = NULL
  
  # print(featureRanking[featureRanking[,"spearman.pValue"] == 1, c("Feature", "spearman.pValue", "numUniqueValues")])
  
  featureRanking = featureRanking[!is.na(featureRanking[,"abs.cor.spearman"]), ]
  
  # featureRanking = featureRanking[featureRanking[,"spearman.pValue"] < 1, ] # FIND ME

  # featureRanking = featureRanking[featureRanking[,"numUniqueValues"] > 1, ] # FIND ME
  
  featureNames = featureRanking[,"Feature"]
  
  
  if (length(featureNames) == 0) {
    next()
  }
  
  
  subjects_train <- df_train_features_imputed[,"UID"]
  y_train <- df_train_features_imputed[,Y_METRIC]
  names(y_train) = subjects_train
  x_train <- df_train_features_imputed[,featureNames, drop=FALSE]
  row.names(x_train) = subjects_train
  
  BINARY_Y_METRIC = BINARY_OUTCOME
  if (BINARY_Y_METRIC) {
    # making sure y metric is binary (two classes)
    if (length(unique(na.omit(y_train))) > 2) {
      BINARY_Y_METRIC = FALSE
    }
  }
  
  #######################################################################################
  #######################################################################################

  NUM_FEATURES = nrow(featureRanking)
  # NUM_FEATURES = length(featureNames)

  
  x_train_matrix = as.matrix(x_train[,featureNames, drop=FALSE])
  row.names(x_train_matrix) = row.names(x_train)
  

  
  NUM_COMP_AFTER_TUNING = 1
  
  totalNumFeatures = nrow(featureRanking)
  
  nfeatures_ = nrow(featureRanking)
  
  selFeatures = featureRanking[1:nfeatures_,"Feature"]
  
  # Cannot be greater than PA_MAX_N_FEATURES
  nfeatures_ = min(nfeatures_, PA_MAX_N_FEATURES)
  # Cannot be greater than the number of outcome measures divided by minimum measures per outcome
  nfeatures_ = min(nfeatures_, floor(length(y_train) / MIN_NUM_MEASURES_PER_FEATURE)) 
  if (nfeatures_ < 1) {
    nfeatures_ = 1
  }
  

  
  if (BINARY_Y_METRIC) {
    ##############################################################################
    #### BINARY_Y_METRIC == TRUE  perform sPLS-Discriminant Analysis
    ##############################################################################
    
    if (nfeatures_ == 1) {
      
      finalFeatures = selFeatures[1]
      
    } else {
      
      # perform spls-Regression
      
      ncomp_  = min(PA_NUM_COMPONENTS, ncol(x_train_matrix))
      
      list.keepX <- c(1:nfeatures_)
      
      set.seed(SPLS_RANDOM_SEED)  # For reproducibility 
      
      tuneModel <- mixOmics::tune.splsda(x_train_matrix, as.factor(y_train), ncomp = ncomp_, validation = TUNE_SPLS_VALIDATION,
                                         folds = TUNE_SPLS_NUM_FOLDS, dist = PLSDA_DISTANCE_METHOD,
                                         test.keepX = list.keepX, nrepeat = TUNE_SPLS_NUM_REPEATS)
      
      select.keepX <- tuneModel$choice.keepX[1:ncomp_]
      
      
      optimizeModel <- mixOmics::splsda(x_train_matrix, as.factor(y_train), ncomp = ncomp_, keepX = select.keepX)
      
      perfModel <- perf(optimizeModel, folds = TUNE_SPLS_NUM_FOLDS, validation = TUNE_SPLS_VALIDATION,
                        dist = PLSDA_DISTANCE_METHOD, progressBar = FALSE, nrepeat = TUNE_SPLS_NUM_REPEATS)
      
      select.name <- selectVar(optimizeModel, comp = 1)$name
      stability <- perfModel$features$stable$comp1[select.name]
      
      # Be very stringent
      stableFeatures = names(stability[stability >= TUNE_SPLS_MIN_FEATURE_STABILITY])
      if (length(stableFeatures) < 1) {
        #for case where there are no features, then keep those with the maximum stability
        stableFeatures = names(stability[stability >= max(stability)])
      }
      
      finalFeatures = stableFeatures
      
    }
    
    
    
    X_final = x_train_matrix[, finalFeatures, drop= FALSE]
    ncomp_  = min(PA_NUM_COMPONENTS,ncol(X_final))
    plsModelFinal <- mixOmics::plsda(X_final, as.factor(y_train), ncomp = ncomp_) 
    
    train_predict_plsModelFinal <- predict(plsModelFinal, x_train_matrix[, finalFeatures, drop= FALSE])
    
    
    train_accuracy = 1 - sum(abs(y_train - as.numeric(train_predict_plsModelFinal$MajorityVote[[PLSDA_DISTANCE_METHOD]])))/length(y_train)
    
    PA_Model_Name = paste0(FOLD_DIR_NAME,"_", Y_METRIC , "_PA1")
    
    
    
    # This is an important step of ordering before writing out to file
    finalFeatures = finalFeatures[order(finalFeatures)]
    
    ri = nrow(res) + 1
    res[ri,"analysis_dir"] = analysis_dir_name
    res[ri,"fold"] = FOLD_DIR_NAME
    res[ri,"Y_METRIC"] = Y_METRIC
    res[ri,"TotalNumFeatures"] = NUM_FEATURES
    res[ri,"PA_Model_Name"]  = PA_Model_Name
    res[ri,"NumFeaturesUsed"] = length(finalFeatures)
    res[ri,"Features"] = paste0(finalFeatures,collapse = ",")
    res[ri,"train_accuracy"] = train_accuracy
    
    
    cat(FOLD_DIR_NAME, Y_METRIC, "N=", totalNumFeatures, "sPLS_N_keep=", length(finalFeatures), PA_Model_Name, "train_accuracy=", round(train_accuracy,digits = 3), "\n")
    
    
  } else {  
    ##############################################################################
    #### BINARY_Y_METRIC == FALSE  perform sPLS-Regression
    ##############################################################################
    
    if (nfeatures_ == 1) {
      
      finalFeatures = selFeatures[1]
      
    } else {
      
      # perform spls-Regression
      
      ncomp_  = min(PA_NUM_COMPONENTS, ncol(x_train_matrix))
      
      list.keepX <- c(1:nfeatures_)
      set.seed(SPLS_RANDOM_SEED)  # For reproducibility 
      tuneModel <- mixOmics::tune.spls(x_train_matrix, y_train, ncomp = ncomp_, validation = TUNE_SPLS_VALIDATION,
                                       folds = TUNE_SPLS_NUM_FOLDS, measure = TUNE_SPLS_MEASURE,
                                       test.keepX = list.keepX, nrepeat = TUNE_SPLS_NUM_REPEATS)
      
      # plot(tuneModel)
      
      
      select.keepX <- tuneModel$choice.keepX[1:ncomp_]
      
      splsModel <- mixOmics::spls(x_train_matrix, y_train, ncomp = ncomp_, keepX = select.keepX, mode = "regression") 
      

      #result = tryCatch({
      splsModelPerf <- mixOmics::perf(splsModel, validation = TUNE_SPLS_VALIDATION, folds =TUNE_SPLS_NUM_FOLDS, nrepeat = TUNE_SPLS_NUM_REPEATS, progressBar = FALSE)
      # }, error = function(e) {
      #   print("mixOmics::perf Failed for validation = Mfold, Now attempting validation = loo")
      #   splsModelPerf <<- mixOmics::perf(splsModel, validation = "loo", folds =TUNE_SPLS_NUM_FOLDS, nrepeat = TUNE_SPLS_NUM_REPEATS, progressBar = FALSE)
      # })
      
      

      splsModelPerf$measures$MSEP$summary
      
      stability <- splsModelPerf$features$stability.X$comp1
      
      stableFeatures = names(stability[stability >= TUNE_SPLS_MIN_FEATURE_STABILITY])
      if (length(stableFeatures) < 1) {
        #for case where there are no features, then keep those with the maximum stability
        stableFeatures = names(stability[stability >= TUNE_SPLS_MIN_FEATURE_STABILITY * max(stability)])
      }
      
      finalFeatures = stableFeatures
      
    }
    
    ############################################################################################## 
    
    if (length(finalFeatures) == 1) {
      
      trainData = data.frame(y=(y_train), x= x_train_matrix[, finalFeatures])
      plsModelFinal = lm(y ~ x, data = trainData)
      
      train_predict_plsModelFinal <- data.frame(predict = predict(plsModelFinal, newdata = data.frame(x=x_train_matrix[, finalFeatures])))
      row.names(train_predict_plsModelFinal) = row.names(x_train_matrix)
      
      
    } else {
      
      X_final = x_train_matrix[, finalFeatures, drop= FALSE]
      ncomp_  = min(PA_NUM_COMPONENTS,ncol(X_final))
      plsModelFinal <- mixOmics::pls(X_final, y_train, ncomp = ncomp_, mode = "regression") 
      
      # plsModelFinal$loadings  # loading vectors: see 
      # plsModelFinal$variates  # variates: index value for each subject
      # plsModelFinal$names      # variable names    96 variables 
      
      train_predict_plsModelFinal <- predict(plsModelFinal, x_train_matrix[, finalFeatures, drop= FALSE])
      
    }
    
    
    ##############################################################################################
    
    SSres = sum((as.numeric(y_train) - as.numeric(train_predict_plsModelFinal$predict))^2)
    SStot = sum((as.numeric(y_train) - mean(as.numeric(y_train)))^2)
    train_R2 = 1 - SSres / SStot
    
    train_mse = SSres / length(y_train)
    
    PA_Model_Name = paste0(FOLD_DIR_NAME,"_", Y_METRIC , "_PA1")
    
    
    
    # This is an important step of ordering before writing out to file
    finalFeatures = finalFeatures[order(finalFeatures)]
    
    ri = nrow(res) + 1
    res[ri,"analysis_dir"] = analysis_dir_name
    res[ri,"fold"] = FOLD_DIR_NAME
    res[ri,"Y_METRIC"] = Y_METRIC
    res[ri,"TotalNumFeatures"] = NUM_FEATURES
    res[ri,"PA_Model_Name"]  = PA_Model_Name
    res[ri,"NumFeaturesUsed"] = length(finalFeatures)
    res[ri,"Features"] = paste0(finalFeatures,collapse = ",")
    res[ri,"train_R2"] = train_R2
    res[ri,"train_mse"] = train_mse
    
    
    cat(FOLD_DIR_NAME, Y_METRIC, "N=", totalNumFeatures, "sPLS_N_keep=", length(finalFeatures), PA_Model_Name, "train_R2=", round(train_R2,digits = 3), "\n")
    
  } # end of sPLS-DA or sPLS-R
  
  
} # end of Y_METRIC


write.csv(res, file.path(analysis_results_dir, FOLD_DIR_NAME, RESULTS_FILENAME),row.names = FALSE)


####################################################################################


if(RUN_FROM_COMMAND_LINE == TRUE){
  quit("no")
}




