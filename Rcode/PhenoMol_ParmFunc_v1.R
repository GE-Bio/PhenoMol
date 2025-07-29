

load_csv_file = function(file, sep = ",", HARD_QUIT = TRUE) {
  file = as.character(file)
  df = data.frame()
  if (file.exists(file)) {
    df = read.csv(file, sep = sep, check.names = FALSE)
  } else if (HARD_QUIT == TRUE) {
    cat("File does not exist:", file, "\n")
    if(RUN_FROM_COMMAND_LINE == TRUE){
      quit("no")
    } else {
      stop("File does not exist!")
    }
  } else {
    cat("File does not exist:","\n")
    cat(file,"\n")
  }
  return(df)
}

stopCode = function(s) {
  cat(s,"\n")
  if(RUN_FROM_COMMAND_LINE == TRUE){
    quit("no")
  } else {
    stop(s)
  }
}



logical_carp_parameters = c("RUN_BORUTA",
                            "INCLUDE_PNM0",
                            "INCLUDE_PNM_MINUS_99",
                            "APPLY_TARGET_PA_COMPONENTS",
                            "APPLY_TARGET_EA_COMPONENTS",
                            "APPLY_TARGET_PA_MA_COMPONENTS",
                            "CARP_VERBOSE",
                            "BINARY_OUTCOME",
                            "DISPLAY_ELBOW_PLOT",
                            "RUN_COMPUTE_PA_PRIZES",
                            "RUN_COMPUTE_PA",
                            "RUN_COMPUTE_MA",
                            "RUN_COMPUTE_EA",
                            "SAVE_OUTPUT",
                            "DISPLAY_PCA_PLOTS",
                            "SUMMARY_PLOT_ALL_UNKNOWNS",
                            "COMPUTE_FEATURE_RATIOS",
                            "DISPLAY_PLOT_PREDICTED_LABELS",
                            "DISPLAY_PLOT_DECISION_BOUNDARY",
                            "DISPLAY_PLOT_LOADINGS",
                            "RUN_FROM_COMMAND_LINE")



numerical_carp_parameters = c("SPLS_RANDOM_SEED",
                              "BORUTA_MAX_RUNS",
                              "CRITICAL_FEATURE_CV",
                              "CRITICAL_FEATURE_FreqNAorZero",
                              "CRITICAL_FEATURE_PAIRWISE_COR",
                              "CRITICAL_FEATURE_VIF",
                              "CRITICAL_P_VALUE_OUTCOME_METRIC",
                              "CRITICAL_P_VALUE_TARGET_PA",
                              "CRITICAL_RANK_FEATURE_COR",
                              "CRITICAL_RATIO_COR",
                              "EA_MAX_N_FEATURES",
                              "EA_SELECT_TABLE_FILTER_COR",
                              "EA_SELECT_TABLE_FILTER_PVALUE",
                              "MA_MAX_N_FEATURES",
                              "MA_SELECT_TABLE_FILTER_COR",
                              "MA_SELECT_TABLE_FILTER_PVALUE",
                              "MAX_NUM_TOP_PA",
                              "MIN_NUM_BORUTA_FEATURES",
                              "MIN_NUM_MEASURES_PER_COMP",
                              "MIN_NUM_MEASURES_PER_FEATURE",
                              "MIN_NUM_TOP_PA",
                              "MIN_NUM_TREES",
                              "MINIMUM_EA_PA_CORRELATION",
                              "MINIMUM_MA_PA_CORRELATION",
                              "MINIMUM_PA_OUTCOME_P_VALUE",
                              "NUM_TREES_PER_FEATURE",
                              "OPTION_MAX_NUM_PCA_COMP",
                              "PA_MAX_N_FEATURES",
                              "PA_NUM_COMPONENTS",
                              "EA_NUM_COMPONENTS",
                              "MA_NUM_COMPONENTS",
                              "TUNE_SPLS_NUM_REPEATS",
                              "TUNE_SPLS_NUM_FOLDS",
                              "TUNE_SPLS_MIN_FEATURE_STABILITY",
                              "ROBUST_TRAINING_PROB")

applyCARPparameters = function(CARP_PARAMETERS, verbose = FALSE) {
  
  # Do not apply this specific parameter if found in the list
  CARP_PARAMETERS[["RUN_FROM_COMMAND_LINE"]] = NULL
  
  if (length(CARP_PARAMETERS) > 0) {

    if (verbose) {
      cat("Setting Run Parameters.................","\n")
    }
    for(objectName in names(CARP_PARAMETERS)) {
      
      if (objectName %in% logical_carp_parameters) {
        assign(objectName,  as.logical(CARP_PARAMETERS[[objectName]]), envir = .GlobalEnv)
      } else if (objectName %in% numerical_carp_parameters) {
        assign(objectName,  as.numeric(CARP_PARAMETERS[[objectName]]), envir = .GlobalEnv)
      } else {
        assign(objectName, CARP_PARAMETERS[[objectName]], envir = .GlobalEnv)
      }
      
      if (verbose) {
        cat(objectName, " --> " , CARP_PARAMETERS[[objectName]],"\n")
      }
    }

    
  } else {
    if (verbose) {
      cat("No Run Parameters.................","\n")
    }
  }

}




