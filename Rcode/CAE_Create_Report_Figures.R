while(dev.next()>1) dev.off()
options(stringsAsFactors=FALSE)
rm(list=objects())

pathToRcode = "Rcode"

baseDir = getwd()
pathToAnalysisDir = baseDir

# if (!requireNamespace("BiocManager", quietly=TRUE))
#   install.packages("BiocManager")
#
# BiocManager::install("ComplexHeatmap")

library(gplots)
library(stringr)
library(circlize)
library(ComplexHeatmap)
library(gridExtra)
library(grid)
library(ggplot2)
library(lattice)
library(png)
library(ggplotify) # required for the function as.grob
library(shadowtext)
library(igraph)
library(visNetwork)
# library(mixOmics)  
library(seriation)
library(hypeR)
library(shape)

SCREEN = "SCREEN"
TIFF = "TIFF"
EPS = "EPS"

PLOT_EPS = FALSE

COLOR_RED = rgb(red=238, green=31, blue=37, maxColorValue = 256)
COLOR_BLUE= rgb(red=57, green=82, blue=164, maxColorValue = 256)
COLOR_BOX_PLOT_BLUE= rgb(red=57*1.5, green=82*1.5, blue=164*1.5, maxColorValue = 256)
COLOR_ORANGE= rgb(red=246, green=147, blue=147, maxColorValue = 256)

COLOR_PNM1= rgb(red=204, green=32, blue=39, maxColorValue = 256)
COLOR_PNM2= rgb(red=76, green=194, blue=197, maxColorValue = 256)
COLOR_PNM3= rgb(red=66, green=182, blue=72, maxColorValue = 256)
COLOR_PNM4= rgb(red=166, green=68, blue=154, maxColorValue = 256)
COLOR_PNM5= rgb(red=208, green=219, blue=70, maxColorValue = 256)
COLOR_PNM6= rgb(red=52, green=70, blue=157, maxColorValue = 256)
COLOR_PNM7= rgb(red=234, green=138, blue=126, maxColorValue = 256)

###############################################################################################

pathToFigures = "analysis_CAE_Report"

source(file.path(pathToRcode, "PhenoMol_Analysis_Report.R"))

geneSets <- load_geneSets()


###############################################################################################
###############################################################################################



if (!dir.exists(pathToFigures)) {
  dir.create(pathToFigures)
  print("Created directory:")
  print(pathToFigures)
}


###### Sex Normalized Data
FIGURE_PLOT_FOLDS = c("fold_001","fold_002","fold_003")
FIGURE_PLOT_ANALYSIS_TEMPLATE = "analysis_CAE_MFn_VO2_3_mptp_o_pa_"
FIGURE_PLOT_ANALYSIS_LIST = paste0(FIGURE_PLOT_ANALYSIS_TEMPLATE, c("654"))
# FIGURE_PLOT_ANALYSIS_LIST = paste0(FIGURE_PLOT_ANALYSIS_TEMPLATE, c("123",  "321",  "456", "654", "789", "987"))
# PLOT_CONTROL_PNM99 = TRUE
OUTCOME_METRIC = "Peak_VO2_mL_kg_min"
OUTCOME_METRIC_PLOT_LABEL = "Peak VO2 (mL/kg/min)"

TARGET_PLOT_TOP = 20

all_model_perf = data.frame()
for (i in 1:length(FIGURE_PLOT_ANALYSIS_LIST)) {
  analysis_run = FIGURE_PLOT_ANALYSIS_LIST[i]
  # load in the ensemble models and their base model information
  analysis_dir =  file.path(pathToAnalysisDir, analysis_run)
  models_dir = file.path(analysis_dir, "models")
  model_perf = read.csv(file.path(models_dir, 'ensemble_EA_models_cv_performance.csv'))
  model_perf = model_perf[order(model_perf[,"all_cv_test_R2_mean"], decreasing = TRUE), ]
  row.names(model_perf)  = NULL
  model_perf[,"analysis_run"] = analysis_run
  model_perf[,"run_seed"] = gsub(FIGURE_PLOT_ANALYSIS_TEMPLATE,"",analysis_run)
  if (nrow(all_model_perf) == 0) {
    all_model_perf = model_perf
  } else {
    all_model_perf = rbind(all_model_perf, model_perf)
  }
}
all_model_perf = all_model_perf[order(all_model_perf[,"all_cv_test_adj_R2_mean"], decreasing = TRUE), ]
row.names(all_model_perf)  = NULL
all_PNM_model_perf = all_model_perf[grep("PNM-99",all_model_perf[,"ensemble"], invert = TRUE), ]
all_PNM99_model_perf = all_model_perf[grep("PNM-99",all_model_perf[,"ensemble"]), ]

# Generate figures for the top analysis run / models
for(rankNum in c(1)) {
  
  analysis_run_rankings = unique(all_PNM_model_perf[ ,"analysis_run"])
  i = which(FIGURE_PLOT_ANALYSIS_LIST == analysis_run_rankings[rankNum])
  
  FIGURE_PLOT_ANALYSIS = FIGURE_PLOT_ANALYSIS_LIST[i]
  for(FIGURE_PLOT_ANALYSIS in FIGURE_PLOT_ANALYSIS_LIST) {
    
    FIGURE_PLOT_ANALYSIS_SEED = gsub(FIGURE_PLOT_ANALYSIS_TEMPLATE,"",FIGURE_PLOT_ANALYSIS)
    
    PNM_EA_model_details = read.csv(file.path(pathToAnalysisDir, FIGURE_PLOT_ANALYSIS, "models", "PNM_EA_model_1_details.csv"))
    for(fold in FIGURE_PLOT_FOLDS) {
      if (nrow(PNM_EA_model_details[startsWith(PNM_EA_model_details[,"base_EA_model"],fold), ]) > 0) {
        FIGURE_PLOT_TARGET_FOLD = fold
        break()
      }
    }
    dataList = LOAD_TOP_PNM_EA_MODEL_DATA(analysis_run = FIGURE_PLOT_ANALYSIS, fold = FIGURE_PLOT_TARGET_FOLD) 
    df_PNM_EA_model_details = dataList[["PNM_EA_model_details"]]
    df_model_info = df_PNM_EA_model_details[,c("measurement_id", "base_EA_model",  "PNM", "nodeLabel", "nodeTitle", "nodeSpecies", "intercept_", "coef_", "varContrib")]
    write.csv(df_model_info, file = file.path(pathToFigures, paste0(FIGURE_PLOT_ANALYSIS, ".csv")), row.names = FALSE)
    
  
    createFigure(FIGURE_PLOT_ANALYSIS, PLOT_TO = TIFF)
    if (PLOT_EPS) {
      createFigure(FIGURE_PLOT_ANALYSIS, PLOT_TO = EPS)
    }
  }

}





