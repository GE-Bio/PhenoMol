FIGURE_PLOT_FOLDS = c("fold_001","fold_002","fold_003","fold_004","fold_005")
# FIGURE_PLOT_ANALYSIS_TEMPLATE = "analysis_CAE_MFn_VO2_3_mptp_o_pa_"
# FIGURE_PLOT_ANALYSIS_LIST = paste0(FIGURE_PLOT_ANALYSIS_TEMPLATE, c("123",  "321",  "456", "654", "789", "987"))
# OUTCOME_METRIC = "Peak_VO2_mL_kg_min"
# OUTCOME_METRIC_PLOT_LABEL = "Peak VO2 (mL/kg/min)"
# TARGET_PLOT_TOP = 20
TARGET_PLOT_TOP = NULL
# FIGURE_PLOT_ANALYSIS
# FIGURE_PLOT_ANALYSIS_SEED
# FIGURE_PLOT_TARGET_FOLD
PLOT_CONTROL_PNM99 = FALSE

ENSEMBLE_EA_MODEL_CV_PERFORMANCE_FN = "ensemble_EA_models_cv_performance.csv"

createFigure = function(figure_Name, PLOT_TO = SCREEN) {
  
  cat("Creating Figure : ", figure_Name,"\n")
  
  
  fig_width <<- 17   # inches
  fig_height <<-13   # inches
  
  hmap_size_inches <<- 2.5
  
  if (PLOT_TO == EPS) {
    outputFN = file.path(pathToFigures, paste0(figure_Name, ".eps"))
    setEPS()
    postscript(outputFN,  height = fig_height, width = fig_width)
  } else if (PLOT_TO == TIFF) {
    outputFN = file.path(pathToFigures, paste0(figure_Name, ".tif"))
    tiff(outputFN, height = fig_height, width = fig_width, units = 'in', compression="lzw",  type = "windows", res = 300)
  } else if (PLOT_TO == SCREEN) {
    
  }
  

  
  

  n = layout(mat = rbind(c(1, 2, 3, 10), c(4, 5, 6, 11 ), c(7, 8, 9, 12)), widths =c(0.9, 0.9, 1.2, 1), heights =  c(1,1,1.25), respect = FALSE)
  # layout.show(n)
  
  dataList = LOAD_PN_PNM_Data(analysis_run = FIGURE_PLOT_ANALYSIS, PLOT_TARGET_FOLD = FIGURE_PLOT_TARGET_FOLD, PNM_Colors = c(COLOR_PNM1,COLOR_PNM2,COLOR_PNM3,COLOR_PNM4,COLOR_PNM5, COLOR_PNM6, COLOR_PNM7))
  nodeInfo = dataList[["nodeInfo"]]
  subnet = dataList[["subnet"]]
  
  
  xylim = 1.4
  offsetDeta = 0.04
  
  par(mar=c(0,0,0,0)) # par(mar = c(bottom, left, top, right))
  plot(x=NA, xlim=c(-xylim,xylim), ylim=c(-xylim,xylim), frame = FALSE, xaxt = 'n', yaxt = 'n')
  offset = (length(FIGURE_PLOT_FOLDS) - 1) * offsetDeta
  rect(-1.1+offset,-1.1+offset,1.1+offset,1.1+offset,col='white', xpd = NA)
  if (length(FIGURE_PLOT_FOLDS) > 1) {
    offset = (length(FIGURE_PLOT_FOLDS) - 2) * offsetDeta
    rect(-1.1+offset,-1.1+offset,1.1+offset,1.1+offset,col='white', xpd = NA)
  }
  if (length(FIGURE_PLOT_FOLDS) > 2) {
    offset = (length(FIGURE_PLOT_FOLDS) - 3) * offsetDeta
    rect(-1.1+offset,-1.1+offset,1.1+offset,1.1+offset,col='white', xpd = NA)
  }
  if (length(FIGURE_PLOT_FOLDS) > 3) {
    offset = (length(FIGURE_PLOT_FOLDS) - 4) * offsetDeta
    rect(-1.1+offset,-1.1+offset,1.1+offset,1.1+offset,col='white', xpd = NA)
  }
  if (length(FIGURE_PLOT_FOLDS) > 4) {
    offset = (length(FIGURE_PLOT_FOLDS) - 5) * offsetDeta
    rect(-1.1+offset,-1.1+offset,1.1+offset,1.1+offset,col='white', xpd = NA)
  }
  
  
  pnNodeInfo = nodeInfo
  rci = which(colnames(pnNodeInfo) %in% c("x", "y" ))
  if (length(rci) > 0) {
    pnNodeInfo = pnNodeInfo[,-rci]
  }
  pnNodeInfo[,"nodeColor"] = "black"
  # set.seed(563)
  set.seed(1)
  par(mar=c(2,2,2,2)) # par(mar = c(bottom, left, top, right))
  PLOT_FIGURE_PNM_NETWORK_VIEW(subnet, ADD_TO_PLOT = TRUE, pnNodeInfo, PLOT_TO = PLOT_TO, cex.sym = 2, edge.color = "grey30")  # , margin = c(0, 0, 0, 0)
  
  mx = 1
  my = 1
  arr.length = 0.3
  arr.width = 0.2
  text.cex = 1.5
  lwd = 2
  
  xy0xy1 = c(1.1, 0.7, 1.8+ 4*offsetDeta, 0.7)
  Arrows(xy0xy1[1]/mx, 1-xy0xy1[2]/my, xy0xy1[3]/mx, 1-xy0xy1[4]/my, code = 2, arr.type="triangle", arr.length = arr.length, arr.width = arr.width, lwd = lwd, xpd = NA)
  text("Louvain\nClustering", x= 1.66, y = 0.6, adj = 0.5, cex = 1.2*text.cex, xpd = NA)
  
  foldLabel = gsub("_00", " ", FIGURE_PLOT_TARGET_FOLD)
  text(foldLabel, x= -0.75, y = 0.95, adj = 0.5, cex = 2.2, xpd = NA)
  
  
  legGroups = c("small molecule","protein","transcript")
  legColors = c("black", "black", "black")
  legSymbols = c(21,24,22)
  legend(x = -1.1, y = -0.55,  ncol = 1, horiz = FALSE, y.intersp = 1,  cex = 1.2, pt.cex = 1.5, bty="n", legend = legGroups, pch = legSymbols, pt.bg  = legColors, col = legColors, xpd = NA)
  
  
  
  xylim = 1.4
  offsetDeta = 0.04
  plotOffset = 4*offsetDeta
  
  par(mar=c(0,0,0,0)) # par(mar = c(bottom, left, top, right))
  plot(x=NA, xlim=c(-xylim,xylim)-plotOffset, ylim=c(-xylim,xylim), frame = FALSE, xaxt = 'n', yaxt = 'n')
  offset = (length(FIGURE_PLOT_FOLDS) - 1) * offsetDeta
  rect(-1.1+offset,-1.1+offset,1.1+offset,1.1+offset,col='white', xpd = NA)
  if (length(FIGURE_PLOT_FOLDS) > 1) {
    offset = (length(FIGURE_PLOT_FOLDS) - 2) * offsetDeta
    rect(-1.1+offset,-1.1+offset,1.1+offset,1.1+offset,col='white', xpd = NA)
  }
  if (length(FIGURE_PLOT_FOLDS) > 2) {
    offset = (length(FIGURE_PLOT_FOLDS) - 3) * offsetDeta
    rect(-1.1+offset,-1.1+offset,1.1+offset,1.1+offset,col='white', xpd = NA)
  }
  if (length(FIGURE_PLOT_FOLDS) > 3) {
    offset = (length(FIGURE_PLOT_FOLDS) - 4) * offsetDeta
    rect(-1.1+offset,-1.1+offset,1.1+offset,1.1+offset,col='white', xpd = NA)
  }
  if (length(FIGURE_PLOT_FOLDS) > 4) {
    offset = (length(FIGURE_PLOT_FOLDS) - 5) * offsetDeta
    rect(-1.1+offset,-1.1+offset,1.1+offset,1.1+offset,col='white', xpd = NA)
  }
  

  PLOT_FIGURE_PNM_NETWORK_VIEW(subnet, nodeInfo, ADD_TO_PLOT = TRUE, PLOT_TO = PLOT_TO, cex.sym = 2, margin = c(0.1, 0.1, 0.1, 0.1), edge.color = "grey30")
  
  legend(x = -1.1, y = -0.53, bty = "n" ,c(paste0("PNM",c(1,2,3,4,5,6, 7), " ")), ncol=2, cex = 1.2, pch=15, pt.cex = 1.5,col=c(COLOR_PNM1, COLOR_PNM2, COLOR_PNM3, COLOR_PNM4, COLOR_PNM5, COLOR_PNM6,COLOR_PNM7), xpd = NA)
  
  
  
  
  
  ###########################################################
  
  
  
  
  par(mar=c(6,0,9,1)) # par(mar = c(bottom, left, top, right))
  plot.new()
  # rect(0,0,1,1)
  
  mx = 350
  my = 100
  
  arr.length = 0.3
  arr.width = 0.2
  text.cex = 1.5
  arrowBaseOffset = 2
  arrowHeadOffset = 9
  lwd = 2
  
  xy0xy1 = c(0+arrowBaseOffset, 15, 80-arrowHeadOffset, 15)
  Arrows(xy0xy1[1]/mx, 1-xy0xy1[2]/my, xy0xy1[3]/mx, 1-xy0xy1[4]/my, code = 2, arr.type="triangle", arr.length = arr.length, arr.width = arr.width, lwd = lwd)
  text("sPLSR", x= (xy0xy1[1]+xy0xy1[3])/2/mx, y = 1-xy0xy1[2]/2/my + 0.03, adj = 0.5, cex = 1.2*text.cex)
  
  xy0xy1 = c(200+arrowBaseOffset, 15, 230-arrowHeadOffset, 15)
  Arrows(xy0xy1[1]/mx, 1-xy0xy1[2]/my, xy0xy1[3]/mx, 1-xy0xy1[4]/my, code = 2, arr.type="triangle", arr.length = arr.length, arr.width = arr.width, lwd = lwd)
  
  xy0xy1 = c(200+arrowBaseOffset, 85, 230-arrowHeadOffset, 85)
  Arrows(xy0xy1[1]/mx, 1-xy0xy1[2]/my, xy0xy1[3]/mx, 1-xy0xy1[4]/my, code = 2, arr.type="triangle", arr.length = arr.length, arr.width = arr.width, lwd = lwd)
  
  xy0xy1 = c(140, 30+arrowBaseOffset, 140, 70+3-arrowHeadOffset)
  Arrows(xy0xy1[1]/mx, 1-xy0xy1[2]/my, xy0xy1[3]/mx, 1-xy0xy1[4]/my, code = 2, arr.type="triangle", arr.length = arr.length, arr.width = arr.width, lwd = lwd)
  text("N Choose K", x= (xy0xy1[1]+5)/mx, y = 1-(xy0xy1[2]+xy0xy1[4])/2/my, adj = 0, cex = 1.2*text.cex)
  
  xy0xy1 = c(80, 0, 200, 30)
  rect(xy0xy1[1]/mx, 1-xy0xy1[2]/my,xy0xy1[3]/mx,1-xy0xy1[4]/my, col = "darkgrey")
  text("Train PNM\nBase Models", x= (xy0xy1[1]+xy0xy1[3])/2/mx, y = 1-(xy0xy1[2]+xy0xy1[4])/2/my, adj = 0.5, col="white", cex = text.cex)
  
  xy0xy1 = c(80, 70, 200, 100)
  rect(xy0xy1[1]/mx, 1-xy0xy1[2]/my,xy0xy1[3]/mx,1-xy0xy1[4]/my, col = "darkgrey")
  text("Train PNM\nEnsemble Models", x= (xy0xy1[1]+xy0xy1[3])/2/mx, y = 1-(xy0xy1[2]+xy0xy1[4])/2/my, adj = 0.5, col="white", cex = text.cex)
  
  xy0xy1 = c(230, 0, 350, 30)
  rect(xy0xy1[1]/mx, 1-xy0xy1[2]/my,xy0xy1[3]/mx,1-xy0xy1[4]/my, col = "darkgrey")
  text("Test PNM\nBase Models", x= (xy0xy1[1]+xy0xy1[3])/2/mx, y = 1-(xy0xy1[2]+xy0xy1[4])/2/my, adj = 0.5, col="white", cex = text.cex)
  
  xy0xy1 = c(230, 70, 350, 100)
  rect(xy0xy1[1]/mx, 1-xy0xy1[2]/my,xy0xy1[3]/mx,1-xy0xy1[4]/my, col = "darkgrey")
  text("Test PNM\nEnsemble Models", x= (xy0xy1[1]+xy0xy1[3])/2/mx, y = 1-(xy0xy1[2]+xy0xy1[4])/2/my, adj = 0.5, col="white", cex = text.cex)
  
  ###########################################################
  
  
  
  
  ###########################################################
  
  
  
  # plot.new()
  # rect(0,0,1,1)
  
  
  
  grobList = get_Figure_Grob_List()
  
  grid.arrange(newpage = FALSE, arrangeGrob(grobs= grobList,  clip = "off", layout_matrix = rbind(c(NA, NA, NA, NA, NA), c(1, 2, 2, 3, 6), c(4, 4, 5, 5, NA)), widths =c(0.9, 0.6, 0.3, 1.2, 1.2), heights =  c(1,1,1.25), respect = FALSE))
  

  grobList = gList()
  gcount = 1
  grobList[[gcount]] = as.grob(function() {
    grid.text("A.", x = unit(0.25, "inches"), y = unit(fig_height - 0.25, "inches"), gp=gpar(col="black", fontface = 'plain', fontsize=22))
    grid.text("D.", x = unit(0.25, "inches"), y = unit(fig_height - 4.25, "inches"), gp=gpar(col="black", fontface = 'plain', fontsize=22))
    grid.text("B.", x = unit(3.75, "inches"), y = unit(fig_height - 0.25, "inches"), gp=gpar(col="black", fontface = 'plain', fontsize=22))
    grid.text("E.", x = unit(3.75, "inches"), y = unit(fig_height - 4.25, "inches"), gp=gpar(col="black", fontface = 'plain', fontsize=22))
    grid.text("C.", x = unit(7.75, "inches"), y = unit(fig_height - 0.25, "inches"), gp=gpar(col="black", fontface = 'plain', fontsize=22))
    grid.text("F.", x = unit(7.75, "inches"), y = unit(fig_height - 4.25, "inches"), gp=gpar(col="black", fontface = 'plain', fontsize=22))
    grid.text("G.", x = unit(12.25, "inches"), y = unit(fig_height - 4.25, "inches"), gp=gpar(col="black", fontface = 'plain', fontsize=22))
    grid.text("H.", x = unit(0.25, "inches"), y = unit(fig_height - 7.75, "inches"), gp=gpar(col="black", fontface = 'plain', fontsize=22))
    grid.text("I.", x = unit(7.75, "inches"), y = unit(fig_height - 7.75, "inches"), gp=gpar(col="black", fontface = 'plain', fontsize=22))
    
  })
  
  grid.arrange(newpage = FALSE, arrangeGrob(grobs= grobList,  clip = "off", layout_matrix = rbind(c(1)), widths =c(1), heights =  c(1), respect = FALSE))
  
  
  
  if (PLOT_TO %in% c(EPS, TIFF)) {
    dev.off()
  }
}


###############################################################################################
###############################################################################################

get_Figure_Grob_List = function() {
  
  grobList = gList()
  gcount = 0
  
  MAR_MULT = 3
  
  
  
  #############################################################################
  
  
  dataList = LOAD_DATA_PNM_ADJACENCY_MATRIX(FIGURE_PLOT_ANALYSIS_LIST, FIGURE_PLOT_FOLDS)
  pnNodeInfo_list = dataList[["pnNodeInfo_list"]]
  PNM_NAMES = dataList[["PNM_NAMES"]]
  adjMat = dataList[["adjMat"]]
  
  gcount = gcount + 1
  grobList[[gcount]] = as.grob(function()  {
    
    par(mar=MAR_MULT*c(1.5,1,1.5,1)) # par(mar = c(bottom, left, top, right))
    
    
    x <- list()
    for (fold in FIGURE_PLOT_FOLDS) {
      fold_Name = fold
      if (length(FIGURE_PLOT_FOLDS) < 10) {
        fold_Name = gsub("_00","_",fold_Name)
      } else if (length(FIGURE_PLOT_FOLDS) < 100) {
        fold_Name = gsub("_0","_",fold_Name)
      }
      x[[fold_Name]] = pnNodeInfo_list[[paste0(fold, "_", FIGURE_PLOT_ANALYSIS_SEED)]][,"nodeName"]
    }

    # x <- list(
    #   fold_1 = pnNodeInfo_list[[paste0("fold_001_", FIGURE_PLOT_ANALYSIS_SEED)]][,"nodeName"],
    #   fold_2 = pnNodeInfo_list[[paste0("fold_002_", FIGURE_PLOT_ANALYSIS_SEED)]][,"nodeName"],
    #   fold_3 = pnNodeInfo_list[[paste0("fold_003_", FIGURE_PLOT_ANALYSIS_SEED)]][,"nodeName"]
    # )
    
    names(x) = gsub("_"," ",names(x))
    

    v.table = venn(x, small =1)
    
  })
  
  
  vm = max(adjMat)
  my_palette <- colorRampPalette(c("white", "red"))(n = 100)
  myBreaks=seq(0, vm, length.out=length(my_palette))
  col_fun = colorRamp2(myBreaks, my_palette, space = "RGB")
  hmap = Heatmap(adjMat, col = col_fun,
                 width = unit(hmap_size_inches, "inch"),
                 height = unit(hmap_size_inches, "inch"),
                 show_row_names = FALSE, show_column_names = FALSE,
                 clustering_method_rows = "ward.D2",
                 clustering_method_columns = "ward.D2",
                 heatmap_legend_param = list(title = 'PNM\nfraction\noverlap'))
  

  
  gcount = gcount + 1
  grobList[[gcount]] = grid::grid.grabExpr(print(hmap))
  
  
  df_accuracy = LOAD_DATA_MODEL_TEST_ERROR(FIGURE_PLOT_ANALYSIS_LIST, FIGURE_PLOT_FOLDS)

  gcount = gcount + 1
  grobList[[gcount]] = as.grob(function()  {
    
    
    HORIZONTAL_LINE_MEDIAN_PN = FALSE
    
    
    
    
    test_metric = "fold_test_rmse"
    
    df_accuracy_99 = df_accuracy[endsWith(df_accuracy[,"ensemble"], "PNM-99']") & df_accuracy[,"NumBaseModels"] == 1, ]
    df_accuracy_PN = df_accuracy[endsWith(df_accuracy[,"ensemble"], "PNM0']") & df_accuracy[,"NumBaseModels"] == 1, ]
    df_accuracy_PNMs = df_accuracy[!endsWith(df_accuracy[,"ensemble"], "PNM-99']") & !endsWith(df_accuracy[,"ensemble"], "PNM0']") & df_accuracy[,"NumBaseModels"] == 1, ]

    
    df_accuracy_ensemble_PN_and_PNMs = df_accuracy[df_accuracy[, "NumBaseModels"] > 1 , ]
    df_accuracy_ensemble_PN_and_PNMs = df_accuracy_ensemble_PN_and_PNMs[grep("PNM-99",df_accuracy_ensemble_PN_and_PNMs[,"ensemble"], invert = TRUE), ]
    # df_accuracy_ensemble_PNMs = df_accuracy_ensemble_PN_and_PNMs[grep("PNM0",df_accuracy_ensemble_PN_and_PNMs[,"ensemble"], invert = TRUE), ]
    df_accuracy_ensemble_PNMs = df_accuracy_ensemble_PN_and_PNMs
    
    
    if (is.null(TARGET_PLOT_TOP)) {

      t <- as.vector(summary(df_accuracy[ ,test_metric]))
      iqr.range <- t[5]-t[2]
      upper_outliers <- t[5]+iqr.range*1.5
      plotTop = 1.3 * upper_outliers
        
    } else {
      plotTop = TARGET_PLOT_TOP
    }

    plotBottom = 0
    ylim = c(plotBottom, plotTop)
    
    par(mar=MAR_MULT*c(2.5,3,2,0.5))
    

    if (PLOT_CONTROL_PNM99 == TRUE & nrow(df_accuracy_99) > 0) {
      SHOW_ALL_COMPARISONS = FALSE
      boxplot(df_accuracy_99[ ,test_metric],df_accuracy_PN[ ,test_metric], df_accuracy_PNMs[ ,test_metric],df_accuracy_ensemble_PNMs[ ,test_metric], ylim = ylim, ylab = "", yaxt = 'n', outpch = NA, frame = FALSE)
      axis(2, las = 1)
      mtext("Model Test Error (RMSE)", side = 2, line = 2.5)
      xNames = c("Control\nModels","PN\nModels","PNM\nBase\nModels","PNM\nEnsemble\nModels")
      if (HORIZONTAL_LINE_MEDIAN_PN) {
        abline(h=median(df_accuracy_PN[ ,test_metric]),col=COLOR_ORANGE,lty=2, lwd = 2, xpd = FALSE)
      } else {
        abline(h=median(df_accuracy_99[ ,test_metric]),col=COLOR_ORANGE,lty=2, lwd = 2, xpd = FALSE)
      }
    } else {
      SHOW_ALL_COMPARISONS = TRUE
      boxplot(df_accuracy[ ,test_metric],df_accuracy_PN[ ,test_metric], df_accuracy_PNMs[ ,test_metric],df_accuracy_ensemble_PNMs[ ,test_metric], ylim = ylim, ylab = "", yaxt = 'n', outpch = NA, frame = FALSE)
      axis(2, las = 1)
      mtext("Model Test Error (RMSE)", side = 2, line = 2.5)
      xNames = c("All\nModels","PN\nModels","PNM\nBase\nModels","PNM\nEnsemble\nModels")
      if (HORIZONTAL_LINE_MEDIAN_PN) {
        abline(h=median(df_accuracy_PN[ ,test_metric]),col=COLOR_ORANGE,lty=2, lwd = 2, xpd = FALSE)
      } else {
        abline(h=median(df_accuracy[ ,test_metric]),col=COLOR_ORANGE,lty=2, lwd = 2, xpd = FALSE)
      }
    }

    yLine = c(2.0, 2.0, 2.5, 2.5)
    for(gi in c(1:length(xNames))) {
      mtext(xNames[gi],side=1,at=gi,line=yLine[gi],cex=1)
    }
    
    # stripchart(df_accuracy_99[ ,test_metric], at =1 ,vertical = TRUE, method = "jitter", jitter = 0.1, pch = 1, col = 'black',  add = TRUE, cex=1, xpd = NA)
    # stripchart(df_accuracy_PN[ ,test_metric], at =2 ,vertical = TRUE, method = "jitter", jitter = 0.1, pch = 1, col = 'black',  add = TRUE, cex=1, xpd = NA)
    # stripchart(df_accuracy_PNMs[ ,test_metric], at =3 ,vertical = TRUE, method = "jitter", jitter = 0.1, pch = 1, col = 'black',  add = TRUE, cex=1, xpd = NA)
    # stripchart(df_accuracy_ensemble_PNMs[ ,test_metric], at =4 ,vertical = TRUE, method = "jitter", jitter = 0.1, pch = 1, col = 'black',  add = TRUE, cex=1, xpd = NA)
    
    

    
    
    
    
    pValueLabelAdj = -0
    
    
    if (PLOT_CONTROL_PNM99 == TRUE & nrow(df_accuracy_99) > 2) {
      g1 = 1
      g2 = 2
      

      
      # THERE IS A ONE TO ONE CORRESPONDANCE BETWEEN df_accuracy_99 AND df_accuracy_PN
      pDat = merge(df_accuracy_99[,c("analysis_run","fold",test_metric)],df_accuracy_PN[,c("analysis_run","fold",test_metric)],by=c("analysis_run","fold"))
      pValue = wilcox.test(pDat[ ,paste0(test_metric,".x")],
                           pDat[ ,paste0(test_metric,".y")],paired = TRUE)$p.value
      
      # pValue = wilcox.test(df_accuracy_99[ ,test_metric]
      #                      , df_accuracy_PN[ ,test_metric])$p.value
      
      maxDat = 0.7 * plotTop
      yMarking = plotTop - 0.3 * (plotTop - plotBottom)
      bsp = 1 / 20
      bhp = (plotTop - plotBottom) / 50
      bh = maxDat + 3 * bhp
      mylwd <<- 1
      draw_StatsBar(g1+bsp,g2-bsp,bh, bhp, format(pValue,digits=2), adjText=pValueLabelAdj*bhp, cexText = 1)
    }
    
    if (PLOT_CONTROL_PNM99 == TRUE & nrow(df_accuracy_99) > 2) {
      g1 = 1
      g2 = 3
      
      # THERE IS A ONE TO MANY CORRESPONDANCE BETWEEN df_accuracy_99 AND df_accuracy_PNMs
      # AGGREGATE THE DATA FOR EACH UNIQUE FOLD BY TAKING THE MEDIAN VALUE
      aggDat = aggregate(df_accuracy_PNMs[,test_metric],by=list(df_accuracy_PNMs[,"analysis_run"],df_accuracy_PNMs[,"fold"]),median)
      colnames(aggDat) = c("analysis_run","fold",test_metric)
      pDat = merge(df_accuracy_99[,c("analysis_run","fold",test_metric)],aggDat[,c("analysis_run","fold",test_metric)],by=c("analysis_run","fold"))
      
      # pDat = merge(df_accuracy_99[,c("analysis_run","fold",test_metric)],df_accuracy_PNMs[,c("analysis_run","fold",test_metric)],by=c("analysis_run","fold"))
      pValue = wilcox.test(pDat[ ,paste0(test_metric,".x")],
                           pDat[ ,paste0(test_metric,".y")],paired = TRUE)$p.value
      
      # pValue = wilcox.test(df_accuracy_99[ ,test_metric]
      #                      , df_accuracy_PNMs[ ,test_metric])$p.value
      
      maxDat = 0.8 * plotTop
      yMarking = plotTop - 0.3 * (plotTop - plotBottom)
      bsp = 1 / 20
      bhp = (plotTop - plotBottom) / 50
      bh = maxDat + 3 * bhp
      mylwd <<- 1
      draw_StatsBar(g1+bsp,g2-bsp,bh, bhp, format(pValue,digits=2), adjText=pValueLabelAdj*bhp, cexText = 1)
    }
    
    
    if (PLOT_CONTROL_PNM99 == TRUE & nrow(df_accuracy_99) > 2) {
      g1 = 1
      g2 = 4
      
      # THERE IS A ONE TO MANY CORRESPONDANCE BETWEEN df_accuracy_99 AND df_accuracy_ensemble_PNMs
      # AGGREGATE THE DATA FOR EACH UNIQUE FOLD BY TAKING THE MEDIAN VALUE
      aggDat = aggregate(df_accuracy_ensemble_PNMs[,test_metric],by=list(df_accuracy_ensemble_PNMs[,"analysis_run"],df_accuracy_ensemble_PNMs[,"fold"]),median)
      colnames(aggDat) = c("analysis_run","fold",test_metric)
      pDat = merge(df_accuracy_99[,c("analysis_run","fold",test_metric)],aggDat[,c("analysis_run","fold",test_metric)],by=c("analysis_run","fold"))
      
      # pDat = merge(df_accuracy_99[,c("analysis_run","fold",test_metric)],df_accuracy_ensemble_PNMs[,c("analysis_run","fold",test_metric)],by=c("analysis_run","fold"))
      pValue = wilcox.test(pDat[ ,paste0(test_metric,".x")],
                           pDat[ ,paste0(test_metric,".y")],paired = TRUE)$p.value
      
      # pValue = wilcox.test(df_accuracy_99[ ,test_metric]
      #                      , df_accuracy_ensemble_PNMs[ ,test_metric])$p.value
      
      
      maxDat = 0.9 * plotTop
      yMarking = plotTop - 0.3 * (plotTop - plotBottom)
      bsp = 1 / 20
      bhp = (plotTop - plotBottom) / 50
      bh = maxDat + 3 * bhp
      mylwd <<- 1
      draw_StatsBar(g1+bsp,g2-bsp,bh, bhp, format(pValue,digits=2), adjText=pValueLabelAdj*bhp, cexText = 1)
    }
    
    
    
    if (SHOW_ALL_COMPARISONS & nrow(df_accuracy_PNMs) > 2) {
      g1 = 2
      g2 = 3
      
      
      
      # THERE IS A ONE TO MANY CORRESPONDANCE BETWEEN df_accuracy_PN AND df_accuracy_PNMs
      # AGGREGATE THE DATA FOR EACH UNIQUE FOLD BY TAKING THE MEDIAN VALUE
      aggDat = aggregate(df_accuracy_PNMs[,test_metric],by=list(df_accuracy_PNMs[,"analysis_run"],df_accuracy_PNMs[,"fold"]),median)
      colnames(aggDat) = c("analysis_run","fold",test_metric)
      pDat = merge(df_accuracy_PN[,c("analysis_run","fold",test_metric)],aggDat[,c("analysis_run","fold",test_metric)],by=c("analysis_run","fold"))
      
      pValue = wilcox.test(pDat[ ,paste0(test_metric,".x")],
                           pDat[ ,paste0(test_metric,".y")],paired = TRUE)$p.value

      if (pValue < 0.2) {
        maxDat = 0.8 * plotTop
        yMarking = plotTop - 0.3 * (plotTop - plotBottom)
        bsp = 1 / 20
        bhp = (plotTop - plotBottom) / 50
        bh = maxDat + 3 * bhp
        mylwd <<- 1
        draw_StatsBar(g1+bsp,g2-bsp,bh, bhp, format(pValue,digits=2), adjText=pValueLabelAdj*bhp, cexText = 1)
      }
    }
    
    
    if (SHOW_ALL_COMPARISONS & nrow(df_accuracy_ensemble_PNMs) > 2) {
      g1 = 2
      g2 = 4
      
      # THERE IS A ONE TO MANY CORRESPONDANCE BETWEEN df_accuracy_PN AND df_accuracy_ensemble_PNMs
      # AGGREGATE THE DATA FOR EACH UNIQUE FOLD BY TAKING THE MEDIAN VALUE
      aggDat = aggregate(df_accuracy_ensemble_PNMs[,test_metric],by=list(df_accuracy_ensemble_PNMs[,"analysis_run"],df_accuracy_ensemble_PNMs[,"fold"]),median)
      colnames(aggDat) = c("analysis_run","fold",test_metric)
      pDat = merge(df_accuracy_PN[,c("analysis_run","fold",test_metric)],aggDat[,c("analysis_run","fold",test_metric)],by=c("analysis_run","fold"))
      
      pValue = wilcox.test(pDat[ ,paste0(test_metric,".x")],
                           pDat[ ,paste0(test_metric,".y")],paired = TRUE)$p.value
      
      if (pValue < 0.2) {
        maxDat = 0.9 * plotTop
        yMarking = plotTop - 0.3 * (plotTop - plotBottom)
        bsp = 1 / 20
        bhp = (plotTop - plotBottom) / 50
        bh = maxDat + 3 * bhp
        mylwd <<- 1
        draw_StatsBar(g1+bsp,g2-bsp,bh, bhp, format(pValue,digits=2), adjText=pValueLabelAdj*bhp, cexText = 1)
      }

    }
    
    
    
  })
  
  
  #######################################################################################
  
  
  
  
  dataList = LOAD_DATA_PN_PNM_ANNOTATIONS(FIGURE_PLOT_ANALYSIS_LIST, FIGURE_PLOT_FOLDS)
  PN_Annotations = dataList[["PN_Annotations"]]
  PNM_Annotations = dataList[["PNM_Annotations"]]
  PN_FOLD_ANALYSIS_NAMES = dataList[["PN_FOLD_ANALYSIS_NAMES"]]
  PNM_FOLD_ANALYSIS_NAME = dataList[["PNM_FOLD_ANALYSIS_NAME"]]
  PNMs = dataList[["PNMs"]]
  
  
 
  
  
  
  CRITICAL_ENRICHMENT_ANALYSIS_PVALUE = 0.05
  
  gcount = gcount + 1
  grobList[[gcount]] = as.grob(function()  {
    
    par(mar=MAR_MULT*c(2, 7, 0.5, 0.5)) # par(mar = c(bottom, left, top, right))
    
    # mtext("Plot C", side = 3, line = 1, cex = 1)
    
    TOP_N = 20
    
    pDat = PN_Annotations
    # pDat = pDat[order(pDat[,"Avg_pval"]), ]
    pDat = pDat[match(PNM_Annotations[,"label"],pDat[,"label"]), ]
    row.names(pDat)  = NULL
    pDat = pDat[1:TOP_N, ]
    
    pDat[,"plotLabel"] = pDat[,"label"]
    pDat[,"plotLabel"] = gsub("KEGG_"," ",pDat[,"plotLabel"])
    pDat[,"plotLabel"] = gsub("_"," ",pDat[,"plotLabel"])
    pDat[,"plotLabel"] = tolower(pDat[,"plotLabel"])
    
    
    tcns = paste0(PN_FOLD_ANALYSIS_NAMES,"_pval")
    
    gDat = list()
    for(gi in rev(c(1:nrow(pDat)))) {
      Y_LABEL = pDat[gi,"plotLabel"]
      gDat[[Y_LABEL]] = -log10(as.numeric(pDat[gi, tcns]))
    }
    
    
    boxplot(gDat,  ylab = "", xlab = "", names = rep("",times = length(gDat)), horizontal = TRUE, outpch = NA, frame = FALSE)
    
    for(gi in rev(c(1:nrow(pDat)))) {
      Y_LABEL = pDat[gi,"plotLabel"]
      stripchart(gDat[[Y_LABEL]], at = length(gDat) - (gi - 1) ,vertical = FALSE, method = "jitter", jitter = 0.1, pch = 4, col = 'blue',  add = TRUE, cex=0.7, xpd = NA)
      mtext(Y_LABEL, side = 2, line = 1, at = length(gDat) - (gi - 1), las = 1)
    }
    mtext("p-Value (-log10)", side = 1, line = 2.5)
    
    
    
  })
  
  
  
  
  
  gcount = gcount + 1
  grobList[[gcount]] = as.grob(function()  {
    
    par(mar=MAR_MULT*c(2, 7, 0.5, 0.5)) # par(mar = c(bottom, left, top, right))
    
    
    
    
    
    TOP_N = 20
    pDat = PNM_Annotations
    pDat = pDat[order(pDat[,paste0(PNM_FOLD_ANALYSIS_NAME,"_PNM0_pval")]),]
    row.names(pDat)  = NULL
    pDat = pDat[1:TOP_N, ]
    
    pDat[,"plotLabel"] = pDat[,"label"]
    pDat[,"plotLabel"] = gsub("KEGG_"," ",pDat[,"plotLabel"])
    pDat[,"plotLabel"] = gsub("_"," ",pDat[,"plotLabel"])
    pDat[,"plotLabel"] = tolower(pDat[,"plotLabel"])
    
    
    
    legend_cols = c("black",COLOR_PNM1, COLOR_PNM2, COLOR_PNM3, COLOR_PNM4, COLOR_PNM5, COLOR_PNM6, "green","cyan")
    legend_pch = c(0,1,2,3,4,5,6,7,8)
    
    # tcns = colnames(pDat)
    # tcns = tcns[endsWith(tcns,"pval")]
    # pDat[,"min_pval"] = apply(pDat,1,function(x) {
    #    return(min(as.numeric(x[tcns])))
    # })
    # pDat = pDat[order(pDat[,"min_pval"]), ]
    # row.names(pDat)  = NULL
    # pDat = pDat[1:TOP_N, ]
    
    pDat = pDat[order(pDat[,paste0(PNM_FOLD_ANALYSIS_NAME,"_PNM0_","pval")]), ]
    row.names(pDat)  = NULL
    pDat = pDat[1:TOP_N, ]
    y = (nrow(pDat)-as.numeric(row.names(pDat)))
    x = -log10(pDat[,paste0(PNM_FOLD_ANALYSIS_NAME,"_PNM0_","pval")])
    
    
    maxx = max(x,na.rm = TRUE)
    for(pnm in c(0,PNMs)) {
      maxx = max(c(maxx,-log10(pDat[,paste0(PNM_FOLD_ANALYSIS_NAME,"_PNM",pnm,"_","pval")])),na.rm = TRUE)
    }
    maxx = ceiling(maxx)
    
    maxx = 30
    
    ############################################################
    # HACK TO SCALE DATA FOR complement and coagulation cascades
    j = which(x > 20)
    x[j] = 25 + 5 * (x[j] - 40) / (60 - 40)
    ############################################################
    
    ii = 1
    plot(x, y, xlim=c(0,maxx), ylab="", xlab = "", yaxt = 'n', xaxt = 'n', col = legend_cols[ii],pch=legend_pch[ii], frame = FALSE)
    axis(1, at = c(0,5,10,15,20))
    axis(1, at = c(25,30), labels = c(40,60),xpd = NA)
    axis(2,at=y,pDat[,"plotLabel"],las=1)
    mtext("p-Value (-log10)", side = 1, line = 2.5)
    
    for(pnm in c(PNMs)) {
      x_pnm = -log10(pDat[,paste0(PNM_FOLD_ANALYSIS_NAME,"_PNM",pnm,"_","pval")])
      
      ############################################################
      # HACK TO SCALE DATA FOR complement and coagulation cascades
      j = which(x_pnm > 20)
      x_pnm[j] = 25 + 5 * (x_pnm[j] - 40) / (60 - 40)
      ############################################################
      
      ii = ii + 1
      xy = data.frame(x = x_pnm, y = y)
      xy = xy[xy[,"x"] > -log10(CRITICAL_ENRICHMENT_ANALYSIS_PVALUE), ]
      points(xy[,"x"], xy[,"y"], col = legend_cols[ii], pch=legend_pch[ii] ,cex = 1.2)
    }
    
    # if (length(folds) > 1) {
    #   legend("bottomright",inset = c(0.02,0.02),title = gsub("_"," ", PNM_FOLD_ANALYSIS_NAME), c("PN",paste0("PNM",PNMs)),col=legend_cols,pch=legend_pch)
    #   
    # } else {
    legend("bottomright",inset = c(0.02,0.02), y.intersp = 1.3,  cex = 1, pt.cex = 1.5,  bty="n", c("PN",paste0("PNM",PNMs, " ")),col=legend_cols,pch=legend_pch)
    #}
    
    
    
  })
  
  
  if (PLOT_CONTROL_PNM99) {
    dataList = LOAD_TOP_PNM_EA_MODEL_DATA(analysis_run = FIGURE_PLOT_ANALYSIS, fold = FIGURE_PLOT_TARGET_FOLD, LOAD_Control_PNM99 = TRUE)
    df_PNM99_Control_Model_y_values = dataList[["PNM_EA_model_y_values"]]
    y_ctrl_pred_train = df_PNM99_Control_Model_y_values[, paste0(OUTCOME_METRIC,"_Predicted")]
  }

  dataList = LOAD_TOP_PNM_EA_MODEL_DATA(analysis_run = FIGURE_PLOT_ANALYSIS, fold = FIGURE_PLOT_TARGET_FOLD)
  df_Model_y_values = dataList[["PNM_EA_model_y_values"]]
  y_pred_train = df_Model_y_values[, paste0(OUTCOME_METRIC,"_Predicted")]
  y_train = df_Model_y_values[, paste0(OUTCOME_METRIC,"_Actual")]
  

  gcount = gcount + 1
  grobList[[gcount]] = as.grob(function()  {
    
    
    ylab = paste0("Predicted ", OUTCOME_METRIC_PLOT_LABEL) 
    xlab = paste0("Measured ", OUTCOME_METRIC_PLOT_LABEL) 
    par(mar=MAR_MULT*c(2,3,1,1)) # par(mar = c(bottom, left, top, right))
    

    xylim = c(min(c(y_train,y_pred_train)), max(c(y_train,y_pred_train)))
    plot(c(), c(), col=COLOR_BLUE,cex.main = 0.8, cex = 0.7, yaxt = 'n', xlim = xylim, ylim = xylim, xlab="", ylab = "", frame = FALSE)
    
    axis(2,las=1)
    abline(a=0, b=1,lty=2,col='black')
    mtext(ylab, side= 2, line = 3)
    mtext(xlab, side= 1, line = 2.5)
  
    points(y_train, y_pred_train,col=COLOR_BLUE,cex = 1)
    
    if (PLOT_CONTROL_PNM99) {
      points(y_train, y_ctrl_pred_train,col=COLOR_RED, pch=4, cex = 1)
      legend("topleft",c(" Control Model", " PNM Ensemble Model"), inset = c(0.05, 0), xpd=NA, col = c(COLOR_RED, COLOR_BLUE), y.intersp = 1.6, cex = 1.1, pch = c(4, 1), bty = 'n', horiz = FALSE)
    }
    
  })
  
  
  return(grobList)
  
}




###############################################################################################
###############################################################################################


draw_StatsBar = function(x0,x1,y0,yl=2, statMarking="*", adjText = 0, cexText = 1) {
  #x0 = xcp - 0.4
  #x1 = xcp + 0.4
  #y0 = plotTopAdj
  y1 = y0
  y2 = y0-yl
  segments(x0, y0, x1, y1, col = par("fg"),lwd = mylwd, lty = par("lty"), xpd = TRUE)
  segments(x0, y0, x0, y2, col = par("fg"),lwd = mylwd,  lty = par("lty"), xpd = TRUE)
  segments(x1, y0, x1, y2, col = par("fg"),lwd = mylwd,  lty = par("lty"), xpd = TRUE)
  
  if (statMarking == "ns" & cexText == 2.5) {
    cexText = 1.8
  }
  
  text(x=(x0+x1)/2,y=y0-adjText,pos=3,cex=cexText,statMarking,xpd=NA)
}

###############################################################################################
###############################################################################################

# margin = c(bottom, left, top, right)

PLOT_FIGURE_PNM_NETWORK_VIEW = function(subnet, nodeInfo, PLOT_TO = "PNG", ADD_TO_PLOT = FALSE, cex.sym = 1, margin = c(0.1, 0.1, 0.1, 0.1), edge.color = "grey", HIDE_LABELS = TRUE, vertex.label.degree = pi/2,  vldMult = 0.15) {  
  
  
  
  
  MyDiamond <- function(coords, v=NULL, params) {
    vertex.color <- params("vertex", "color")
    if (length(vertex.color) != 1 && !is.null(v)) {
      vertex.color <- vertex.color[v]
    }
    vertex.frame.color <- params("vertex", "frame.color")
    if (length(vertex.frame.color) != 1 && !is.null(v)) {
      vertex.frame.color <- vertex.frame.color[v]
    }
    vertex.frame.width <- params("vertex", "frame.width")
    if (length(vertex.frame.width) != 1 && !is.null(v)) {
      vertex.frame.width <- vertex.frame.width[v]
    }
    vertex.size <- 1/200 * params("vertex", "size")
    if (length(vertex.size) != 1 && !is.null(v)) {
      vertex.size <- vertex.size[v]
    }
    
    symbols(x=coords[,1], y=coords[,2], bg=vertex.color, fg=vertex.frame.color, lwd = vertex.frame.width,
            stars=cbind(vertex.size, vertex.size, vertex.size, vertex.size),
            add=TRUE, inches=FALSE)
  }
  add_shape("diamond", clip=shapes("circle")$clip,
            plot=MyDiamond, parameters=list(vertex.frame.color="white",
                                            vertex.frame.width=1))
  
  #################################################################
  
  
  myUpTriangle <- function(coords, v=NULL, params) {
    vertex.color <- params("vertex", "color")
    if (length(vertex.color) != 1 && !is.null(v)) {
      vertex.color <- vertex.color[v]
    }
    vertex.frame.color <- params("vertex", "frame.color")
    if (length(vertex.frame.color) != 1 && !is.null(v)) {
      vertex.frame.color <- vertex.frame.color[v]
    }
    vertex.frame.width <- params("vertex", "frame.width")
    if (length(vertex.frame.width) != 1 && !is.null(v)) {
      vertex.frame.width <- vertex.frame.width[v]
    }
    vertex.size  <- 1/200 * params("vertex", "size")
    if (length(vertex.size) != 1 && !is.null(v)) {
      vertex.size <- vertex.size[v]
    }
    norays <- params("vertex", "norays")
    if (length(norays) != 1 && !is.null(v)) {
      norays <- norays[v]
    }
    
    symbols(x=coords[,1], y=coords[,2]-0.5*1.732051 *vertex.size, bg=vertex.color, fg=vertex.frame.color, lwd = vertex.frame.width,
            stars=cbind(vertex.size, 1.732051 *vertex.size, vertex.size, NA),
            add=TRUE, inches=FALSE)
  }
  # no clipping, edges will be below the vertices anyway
  add_shape("up_triangle", clip=shape_noclip,
            plot=myUpTriangle, parameters=list(vertex.frame.color="white", 
                                               vertex.frame.width=1, 
                                               vertex.norays=3))
  
  #################################################################
  
  
  myDownTriangle <- function(coords, v=NULL, params) {
    vertex.color <- params("vertex", "color")
    if (length(vertex.color) != 1 && !is.null(v)) {
      vertex.color <- vertex.color[v]
    }
    vertex.frame.color <- params("vertex", "frame.color")
    if (length(vertex.frame.color) != 1 && !is.null(v)) {
      vertex.frame.color <- vertex.frame.color[v]
    }
    vertex.frame.width <- params("vertex", "frame.width")
    if (length(vertex.frame.width) != 1 && !is.null(v)) {
      vertex.frame.width <- vertex.frame.width[v]
    }
    vertex.size  <- 1/200 * params("vertex", "size")
    if (length(vertex.size) != 1 && !is.null(v)) {
      vertex.size <- vertex.size[v]
    }
    norays <- params("vertex", "norays")
    if (length(norays) != 1 && !is.null(v)) {
      norays <- norays[v]
    }
    
    symbols(x=coords[,1], y=coords[,2]+0.5*1.732051 *vertex.size, bg=vertex.color, fg=vertex.frame.color, lwd = vertex.frame.width,
            stars=cbind(vertex.size, NA, vertex.size, 1.732051 *vertex.size),
            add=TRUE, inches=FALSE)
  }
  # no clipping, edges will be below the vertices anyway
  add_shape("down_triangle", clip=shape_noclip,
            plot=myDownTriangle, parameters=list(vertex.frame.color="white", 
                                                 vertex.frame.width=1, 
                                                 vertex.norays=3))
  
  #################################################################
  
  
  edge_width=5
  
  ############ IMPORTANT #########################################
  #First Sort nodeInfo to match subnet ordering
  nodeInfo = nodeInfo[match(V(subnet)$name,nodeInfo[,"nodeName"]), ]
  
  
  prize = abs(as.numeric(nodeInfo[,"nodePrize"]))
  min1 = 20
  max1 = 40
  r1 = max1 - min1
  min2 =  min(prize,na.rm = TRUE)
  max2 = max(prize,na.rm = TRUE)
  r2 = max2 - min2
  nodeInfo[,"adjusted_prize"] = r1*(prize - min2)/r2 + min1
  # for nodes without a prize use the min Node size
  nodeInfo[is.na(nodeInfo[,"adjusted_prize"]),"adjusted_prize"] = min1
  
  
  
  # Calculate the adjusted edge weights
  weight = E(subnet)$weight
  min1 = 1
  max1 = edge_width
  r1 = max1 - min1
  min2 =  min(weight)
  max2 = max(weight)
  r2 = max2 - min2
  adjusted_weight = r1*(weight - min2)/r2 + min1
  
  # List of nodes in the subnet
  nodes = data.frame(1:length(V(subnet)), V(subnet)$name)
  names(nodes) = c("id", "name")
  
  V(subnet)$label = nodeInfo[,"nodeLabel"]
  
  # Differentiate the type of nodes
  V(subnet)$group = paste0(nodeInfo[,"nodeShape"])  # V(subnet)$type
  
  
  
  nodeInfo[,"shape"] = "triangle"
  nodeInfo[startsWith(nodeInfo[,"nodeShape"],"Terminal"),"shape"] = "circle"
  nodeInfo[grep("DNA",nodeInfo[,"nodeShape"]),"shape"] = "diamond"
  nodeInfo[grep("RNA",nodeInfo[,"nodeShape"]),"shape"] = "square"
  nodeInfo[grep("Protein",nodeInfo[,"nodeShape"]),"shape"] =   "up_triangle"
  nodeInfo[grep("Metabolite",nodeInfo[,"nodeShape"]),"shape"] = "circle"
  
  
  nodeInfo[,"size"] = 3 * cex.sym
  nodeInfo[startsWith(nodeInfo[,"nodeShape"],"Terminal"),"size"] = 3 * cex.sym
  nodeInfo[grep("DNA",nodeInfo[,"nodeShape"]),"size"] = 3 * cex.sym
  nodeInfo[grep("RNA",nodeInfo[,"nodeShape"]),"size"] = 2.5 * cex.sym
  nodeInfo[grep("Protein",nodeInfo[,"nodeShape"]),"size"] =   3 * cex.sym
  nodeInfo[grep("Metabolite",nodeInfo[,"nodeShape"]),"size"] = 3 * cex.sym
  
  # nodeInfo[,"color"] = "white"
  # nodeInfo[grep("\\+",nodeInfo[,"nodeShape"]),"color"] = "red"
  # nodeInfo[grep("\\-",nodeInfo[,"nodeShape"]),"color"] = "blue"
  
  
  V(subnet)$shape = nodeInfo[,"shape"]
  V(subnet)$color = nodeInfo[,"nodeColor"]
  # vertex.frame.color="black"
  V(subnet)$frame.color = nodeInfo[,"nodeColor"]
  
  # Attach the node attributes
  V(subnet)$size = nodeInfo[,"size"] # 0.15 * nodeInfo[,"adjusted_prize"] #adjusted_prize
  
  # nodeInfo[,"displayLabel"] = ""  #
  # ii = which(nodeInfo[,"nodeLabel"] %in% c("Urea", "GPX3 : P22352"))
  # nodeInfo[ii,"displayLabel"] = nodeInfo[ii,"nodeLabel"]
  
  V(subnet)$title = nodeInfo[,"nodeTitle"] # nodes$name
  V(subnet)$label = nodeInfo[,"nodeLabel"] # nodes$name
  
  if (HIDE_LABELS) {
    V(subnet)$label = ""
  }
  
  
  V(subnet)$label.color <- "black"
  V(subnet)$label.cex <- 0.8
  V(subnet)$label.family <- "sans"
  # V(subnet)$font.size = 12
  
  # List of edges in the subnet
  edges = data.frame(ends(subnet,es = E(subnet)), adjusted_weight)
  names(edges) = c("from", "to", "width")
  edges$from = match(edges$from, nodes$name)
  edges$to = match(edges$to, nodes$name)
  
  
  if ("x" %in% colnames(nodeInfo)) {
    V(subnet)$x = nodeInfo[,"x"]
    V(subnet)$y = nodeInfo[,"y"]
  }
  # ,width=960,height=960 are used for browser HTML
  
  
  
  if (PLOT_TO == "EPS")  {
    V(subnet)$label.family <- "arial"
  } else {
    V(subnet)$label.family <- "sans"
  }
  
  vld = vldMult * V(subnet)$size 
  
  
  
  
  plot(subnet, add = ADD_TO_PLOT,
       vertex.label.dist = vld, # vector of distance
       vertex.label.font = 1,            
       vertex.label.degree = vertex.label.degree,
       # vertex.frame.color = 'black', # using PNM colors
       vertex.frame.width = 1,
       edge.color= edge.color,
       edge.width = 0.1, #0.5* adjusted_weight,
       margin = margin,
       # asp=0.55 #aspect ratio
       asp=1 #aspect ratio
       
  )
  
  
}


###############################################################################################
###############################################################################################

# margin = c(bottom, left, top, right)

PLOT_FIGURE_PN_NETWORK_VIEW = function(subnet, nodeInfo, PLOT_TO = "PNG", cex.sym = 1, margin = c(0.1, 0.1, 0.1, 0.1), edge.color = "grey",  vldMult = 0.25) {  
  
  
  
  
  MyDiamond <- function(coords, v=NULL, params) {
    vertex.color <- params("vertex", "color")
    if (length(vertex.color) != 1 && !is.null(v)) {
      vertex.color <- vertex.color[v]
    }
    vertex.frame.color <- params("vertex", "frame.color")
    if (length(vertex.frame.color) != 1 && !is.null(v)) {
      vertex.frame.color <- vertex.frame.color[v]
    }
    vertex.frame.width <- params("vertex", "frame.width")
    if (length(vertex.frame.width) != 1 && !is.null(v)) {
      vertex.frame.width <- vertex.frame.width[v]
    }
    vertex.size <- 1/200 * params("vertex", "size")
    if (length(vertex.size) != 1 && !is.null(v)) {
      vertex.size <- vertex.size[v]
    }
    
    symbols(x=coords[,1], y=coords[,2], bg=vertex.color, fg=vertex.frame.color, lwd = vertex.frame.width,
            stars=cbind(vertex.size, vertex.size, vertex.size, vertex.size),
            add=TRUE, inches=FALSE)
  }
  add_shape("diamond", clip=shapes("circle")$clip,
            plot=MyDiamond, parameters=list(vertex.frame.color="white",
                                            vertex.frame.width=1))
  
  #################################################################
  
  
  myUpTriangle <- function(coords, v=NULL, params) {
    vertex.color <- params("vertex", "color")
    if (length(vertex.color) != 1 && !is.null(v)) {
      vertex.color <- vertex.color[v]
    }
    vertex.frame.color <- params("vertex", "frame.color")
    if (length(vertex.frame.color) != 1 && !is.null(v)) {
      vertex.frame.color <- vertex.frame.color[v]
    }
    vertex.frame.width <- params("vertex", "frame.width")
    if (length(vertex.frame.width) != 1 && !is.null(v)) {
      vertex.frame.width <- vertex.frame.width[v]
    }
    vertex.size  <- 1/200 * params("vertex", "size")
    if (length(vertex.size) != 1 && !is.null(v)) {
      vertex.size <- vertex.size[v]
    }
    norays <- params("vertex", "norays")
    if (length(norays) != 1 && !is.null(v)) {
      norays <- norays[v]
    }
    
    symbols(x=coords[,1], y=coords[,2]-0.5*1.732051 *vertex.size, bg=vertex.color, fg=vertex.frame.color, lwd = vertex.frame.width,
            stars=cbind(vertex.size, 1.732051 *vertex.size, vertex.size, NA),
            add=TRUE, inches=FALSE)
  }
  # no clipping, edges will be below the vertices anyway
  add_shape("up_triangle", clip=shape_noclip,
            plot=myUpTriangle, parameters=list(vertex.frame.color="white", 
                                               vertex.frame.width=1, 
                                               vertex.norays=3))
  
  #################################################################
  
  
  myDownTriangle <- function(coords, v=NULL, params) {
    vertex.color <- params("vertex", "color")
    if (length(vertex.color) != 1 && !is.null(v)) {
      vertex.color <- vertex.color[v]
    }
    vertex.frame.color <- params("vertex", "frame.color")
    if (length(vertex.frame.color) != 1 && !is.null(v)) {
      vertex.frame.color <- vertex.frame.color[v]
    }
    vertex.frame.width <- params("vertex", "frame.width")
    if (length(vertex.frame.width) != 1 && !is.null(v)) {
      vertex.frame.width <- vertex.frame.width[v]
    }
    vertex.size  <- 1/200 * params("vertex", "size")
    if (length(vertex.size) != 1 && !is.null(v)) {
      vertex.size <- vertex.size[v]
    }
    norays <- params("vertex", "norays")
    if (length(norays) != 1 && !is.null(v)) {
      norays <- norays[v]
    }
    
    symbols(x=coords[,1], y=coords[,2]+0.5*1.732051 *vertex.size, bg=vertex.color, fg=vertex.frame.color, lwd = vertex.frame.width,
            stars=cbind(vertex.size, NA, vertex.size, 1.732051 *vertex.size),
            add=TRUE, inches=FALSE)
  }
  # no clipping, edges will be below the vertices anyway
  add_shape("down_triangle", clip=shape_noclip,
            plot=myDownTriangle, parameters=list(vertex.frame.color="white", 
                                                 vertex.frame.width=1, 
                                                 vertex.norays=3))
  
  #################################################################
  HIDE_LABELS = TRUE
  
  edge_width=5
  
  ############ IMPORTANT #########################################
  #First Sort nodeInfo to match subnet ordering
  nodeInfo = nodeInfo[match(V(subnet)$name,nodeInfo[,"nodeName"]), ]
  
  
  prize = abs(as.numeric(nodeInfo[,"nodePrize"]))
  min1 = 20
  max1 = 40
  r1 = max1 - min1
  min2 =  min(prize,na.rm = TRUE)
  max2 = max(prize,na.rm = TRUE)
  r2 = max2 - min2
  nodeInfo[,"adjusted_prize"] = r1*(prize - min2)/r2 + min1
  # for nodes without a prize use the min Node size
  nodeInfo[is.na(nodeInfo[,"adjusted_prize"]),"adjusted_prize"] = min1
  
  
  
  # Calculate the adjusted edge weights
  weight = E(subnet)$weight
  min1 = 1
  max1 = edge_width
  r1 = max1 - min1
  min2 =  min(weight)
  max2 = max(weight)
  r2 = max2 - min2
  adjusted_weight = r1*(weight - min2)/r2 + min1
  
  # List of nodes in the subnet
  nodes = data.frame(1:length(V(subnet)), V(subnet)$name)
  names(nodes) = c("id", "name")
  
  V(subnet)$label = nodeInfo[,"nodeLabel"]
  
  # Differentiate the type of nodes
  V(subnet)$group = paste0(nodeInfo[,"nodeShape"])  # V(subnet)$type
  
  
  
  nodeInfo[,"shape"] = "triangle"
  nodeInfo[startsWith(nodeInfo[,"nodeShape"],"Terminal"),"shape"] = "circle"
  nodeInfo[grep("DNA",nodeInfo[,"nodeShape"]),"shape"] = "diamond"
  nodeInfo[grep("RNA",nodeInfo[,"nodeShape"]),"shape"] = "square"
  nodeInfo[grep("Protein",nodeInfo[,"nodeShape"]),"shape"] =   "up_triangle"
  nodeInfo[grep("Metabolite",nodeInfo[,"nodeShape"]),"shape"] = "circle"
  
  
  nodeInfo[,"size"] = 3 * cex.sym
  nodeInfo[startsWith(nodeInfo[,"nodeShape"],"Terminal"),"size"] = 3 * cex.sym
  nodeInfo[grep("DNA",nodeInfo[,"nodeShape"]),"size"] = 3 * cex.sym
  nodeInfo[grep("RNA",nodeInfo[,"nodeShape"]),"size"] = 2.5 * cex.sym
  nodeInfo[grep("Protein",nodeInfo[,"nodeShape"]),"size"] =   3 * cex.sym
  nodeInfo[grep("Metabolite",nodeInfo[,"nodeShape"]),"size"] = 3 * cex.sym
  
  # nodeInfo[,"color"] = "white"
  # nodeInfo[grep("\\+",nodeInfo[,"nodeShape"]),"color"] = "red"
  # nodeInfo[grep("\\-",nodeInfo[,"nodeShape"]),"color"] = "blue"
  
  
  V(subnet)$shape = nodeInfo[,"shape"]
  V(subnet)$color = nodeInfo[,"nodeColor"]
  # vertex.frame.color="black"
  V(subnet)$frame.color = nodeInfo[,"nodeColor"]
  
  # Attach the node attributes
  V(subnet)$size = nodeInfo[,"size"] # 0.15 * nodeInfo[,"adjusted_prize"] #adjusted_prize
  
  # nodeInfo[,"displayLabel"] = ""  #
  # ii = which(nodeInfo[,"nodeLabel"] %in% c("Urea", "GPX3 : P22352"))
  # nodeInfo[ii,"displayLabel"] = nodeInfo[ii,"nodeLabel"]
  
  V(subnet)$title = nodeInfo[,"nodeTitle"] # nodes$name
  V(subnet)$label = nodeInfo[,"nodeLabel"] # nodes$name
  
  if (HIDE_LABELS) {
    V(subnet)$label = ""
  }
  
  
  V(subnet)$label.color <- "black"
  V(subnet)$label.cex <- 0.8
  V(subnet)$label.family <- "sans"
  # V(subnet)$font.size = 12
  
  # List of edges in the subnet
  edges = data.frame(ends(subnet,es = E(subnet)), adjusted_weight)
  names(edges) = c("from", "to", "width")
  edges$from = match(edges$from, nodes$name)
  edges$to = match(edges$to, nodes$name)
  
  
  if ("x" %in% colnames(nodeInfo)) {
    V(subnet)$x = nodeInfo[,"x"]
    V(subnet)$y = nodeInfo[,"y"]
  }
  # ,width=960,height=960 are used for browser HTML
  
  
  
  if (PLOT_TO == "EPS")  {
    V(subnet)$label.family <- "arial"
  } else {
    V(subnet)$label.family <- "sans"
  }
  
  vld = vldMult * V(subnet)$size 
  
  
  plot(subnet,
       vertex.label.dist = vld, # vector of distance
       vertex.label.font = 1,            
       vertex.label.degree = pi/2,
       # vertex.frame.color = 'black', # using PNM colors
       vertex.frame.width = 1,
       edge.color= edge.color,
       edge.width = 0.1, #0.5* adjusted_weight,
       margin = margin,
       # asp=0.55 #aspect ratio
       asp=1 #aspect ratio
  )
  
}



###############################################################################################
###############################################################################################


LOAD_DATA_MODEL_TEST_ERROR = function(analysis_runs, folds) {
  

  ENSEMBLE_FN = "ensemble_PNM_EA_models.csv" 
  
  df_accuracy = data.frame()
  for(analysis_run in analysis_runs) {
    analysis_dir =  file.path(pathToAnalysisDir, analysis_run)
    for(CV_FOLD_DIR_NAME in folds) {
      
      df = read.csv(file.path(analysis_dir, "results", CV_FOLD_DIR_NAME, ENSEMBLE_FN))
      cns = colnames(df)
      
      df[,"analysis_run"] = analysis_run
      df[,"fold"] = CV_FOLD_DIR_NAME
      df = df[,c("analysis_run","fold",cns)]
      df_base = df[df[,"NumBaseModels"] == 1, ]
      df_base_PNM99 = df_base[grep("PNM-99",df_base[,"ensemble"]), ]
      df_base_PN = df_base[grep("PNM0",df_base[,"ensemble"]), ]
      df_base_PNM = df_base
      df_base_PNM = df_base_PNM[grep("PNM-99",df_base_PNM[,"ensemble"], invert = TRUE), ]
      df_base_PNM = df_base_PNM[grep("PNM0",df_base_PNM[,"ensemble"], invert = TRUE), ]
      rm(df_base)
      df_all_base_PNM = df_base_PNM
      
      all_base_PNM_ensemble = gsub("\\[","", gsub("\\]","", df_base_PNM[,"ensemble"]))
      
      keep_base_PNM_ensemble = gsub("\\[","", gsub("\\]","", df_base_PNM[,"ensemble"]))
      remove_base_PNM_ensemble = all_base_PNM_ensemble[!all_base_PNM_ensemble %in% keep_base_PNM_ensemble]
      
      df_ensemble = df[df[,"NumBaseModels"] > 1, ]
      
      df_ensemble = df_ensemble[grep("PNM-99",df_ensemble[,"ensemble"], invert = TRUE), ]
      if (length(remove_base_PNM_ensemble) > 0) {
        for (rbPNM in remove_base_PNM_ensemble) {
          df_ensemble = df_ensemble[grep(rbPNM,df_ensemble[,"ensemble"], invert = TRUE), ]
        }
      }
      
      df = rbind(df_base_PNM99, df_base_PN, df_all_base_PNM, df_ensemble)
      
      if(nrow(df_accuracy) == 0) {
        df_accuracy = df
      } else {
        df_accuracy = rbind(df_accuracy, df)
      }
    }
  }
  
  df_Simple = df_accuracy[grep("PNM-99",df_accuracy[,"ensemble"]), ]
  df2 = df_accuracy[grep("PNM-99",df_accuracy[,"ensemble"], invert = TRUE), ]
  df_PNM0 = df2[grep("PNM0'",df2[,"ensemble"]), ]
  df_PNMs = df2[grep("PNM0'",df2[,"ensemble"], invert = TRUE), ]
  df_accuracy = rbind(df_PNM0,df_PNMs,df_Simple)
  # Order the list of features
  df_accuracy[, "Features"] = apply(df_accuracy, 1, function(x) {
    v = x["Features"]
    v = gsub("\\[","", gsub("\\]","", gsub("\\'","", gsub(" ","", v))))
    v = str_split_1(v, ",")
    v = v[order(v)]
    v = paste0("'",v,"'")
    return(paste0("[",paste(v,collapse = ", "),"]"))
  })
  
  
  df_accuracy[,"fold_train_rmse"] = df_accuracy[,"fold_train_mse"] ^ 0.5
  df_accuracy[,"fold_test_rmse"] = df_accuracy[,"fold_test_mse"] ^ 0.5
  
  
  return(df_accuracy)
}





###############################################################################################
###############################################################################################


LOAD_DATA_PNM_ADJACENCY_MATRIX = function(analysis_runs, folds) {
  
  

  
  
  pnNodeInfo_list = list()
  pnSubNetwork_list = list()
  pnmNetwork_enrichment_summary = list()
  fold_EA_Loadings = list()
  
  FOLD_ANALYSIS_NAMES = c()
  PNM_NAMES = c()
  PNM_NUMBERS = c()
  PN_NAMES = c()
  for(analysis_run in analysis_runs) {
    analysis_dir =  file.path(pathToAnalysisDir, analysis_run)
    
    for(CV_FOLD_DIR_NAME in folds) {
      FOLD_ANALYSIS_NAME = paste0(CV_FOLD_DIR_NAME,"_",analysis_run)
      FOLD_ANALYSIS_NAME = gsub(FIGURE_PLOT_ANALYSIS_TEMPLATE,"",FOLD_ANALYSIS_NAME)
      
      pnNodeInfo_list[[FOLD_ANALYSIS_NAME]] = read.csv(file.path(analysis_dir, "results", CV_FOLD_DIR_NAME, "oi_network", "pn_nodes.csv"))
      pnSubNetwork_list[[FOLD_ANALYSIS_NAME]] = read.csv(file.path(analysis_dir, "results", CV_FOLD_DIR_NAME, "oi_network", "pn_edges.csv"))
      
      nodeInfo = pnNodeInfo_list[[FOLD_ANALYSIS_NAME]]
      
      cat(analysis_run, CV_FOLD_DIR_NAME, nrow(nodeInfo[nodeInfo[,"nodeType"] == "Terminal", ]),nrow(nodeInfo[nodeInfo[,"nodeType"] == "Steiner", ]),nrow(nodeInfo),"\n")
      
      FOLD_ANALYSIS_NAMES = c(FOLD_ANALYSIS_NAMES, FOLD_ANALYSIS_NAME)
      
      PNMs = unique(nodeInfo[,"PNM"])
      PNMs = PNMs[order(PNMs)]
      PNM_NAMES = c(PNM_NAMES,paste0(FOLD_ANALYSIS_NAME,"_PNM",PNMs))
      
      PNM_NUMBERS = c(PNM_NUMBERS, PNMs)
      
      
      PN_NAMES = c(PN_NAMES,paste0(FOLD_ANALYSIS_NAME))
      
    }
  }
  
  
  
  
  adjMat = matrix(nrow = length(PNM_NAMES), ncol= length(PNM_NAMES))
  colnames(adjMat)  = PNM_NAMES
  row.names(adjMat)  = PNM_NAMES
  
  xi = 0
  
  for(x_FOLD_ANALYSIS_NAME in FOLD_ANALYSIS_NAMES) {
    
    x_nodeInfo = pnNodeInfo_list[[x_FOLD_ANALYSIS_NAME]]
    x_PNMs = unique(x_nodeInfo[,"PNM"])
    x_PNMs = x_PNMs[order(x_PNMs)]
    for(x_PNM in x_PNMs) {
      xi = xi + 1
      x_nodeInfo = pnNodeInfo_list[[x_FOLD_ANALYSIS_NAME]]
      x_nodeInfo = x_nodeInfo[x_nodeInfo[,"PNM"] == x_PNM, ]
      
      yi = 0
      for(y_FOLD_ANALYSIS_NAME in FOLD_ANALYSIS_NAMES) {
        
        y_nodeInfo = pnNodeInfo_list[[y_FOLD_ANALYSIS_NAME]]
        y_PNMs = unique(y_nodeInfo[,"PNM"])
        y_PNMs = y_PNMs[order(y_PNMs)]
        for(y_PNM in y_PNMs) {
          yi = yi + 1
          y_nodeInfo = pnNodeInfo_list[[y_FOLD_ANALYSIS_NAME]]
          y_nodeInfo = y_nodeInfo[y_nodeInfo[,"PNM"] == y_PNM, ]
          
          navg = (length(x_nodeInfo[,"nodeName"]) + length(y_nodeInfo[,"nodeName"])) / 2
          
          noverlap = length(intersect(x_nodeInfo[,"nodeName"],y_nodeInfo[,"nodeName"])) 
          fractionOverlap = noverlap / navg
          
          adjMat[xi,yi] = fractionOverlap
          
        }
      }
      
    }
  }
  
  dataList = list()
  dataList[["pnNodeInfo_list"]] = pnNodeInfo_list
  dataList[["PNM_NAMES"]] = PNM_NAMES
  dataList[["adjMat"]] = adjMat
  
  
  
  
  return(dataList)
}


###############################################################################################
###############################################################################################


load_entrez = function() {
  
  load( file.path(pathToAnalysisDir, "Interactome", "ensembl_GRCh38_110","ensembl_GRCh38_110.RData"),verbose = TRUE)
  
  colnames(df.entrez)[which(colnames(df.entrez) == "xref")] = "entrez_id"
  df.entrez = df.entrez[,c("entrez_id", "gene_stable_id",  "transcript_stable_id", "protein_stable_id")]
  df.entrez[df.entrez[,"gene_stable_id"] == "-", "gene_stable_id"] = NA
  # df.entrez[df.entrez[,"transcript_stable_id"] == "-", "transcript_stable_id"] = NA
  # df.entrez[df.entrez[,"protein_stable_id"] == "-", "protein_stable_id"] = NA
  # df.entrez <<- unique(df.entrez[,c("entrez_id", "gene_stable_id")])
  
  ri = nrow(df.entrez) + 1
  df.entrez[ri,"entrez_id"] = "7012"
  df.entrez[ri,"gene_stable_id"] = "ENSG00000270141"
  
  df.entrez = merge(df.entrez,df.gene[,c("gene_id","gene_name")],by.x="gene_stable_id",by.y="gene_id",all.x=TRUE)
  
  df.entrez = unique(df.entrez[,c("entrez_id", "gene_stable_id", "gene_name")])
  
  d = df.uniprot[,c("gene_stable_id","xref")]
  d = unique(d)
  colnames(d) = c("gene_stable_id","uniprot_id")
  
  df.entrez = merge(df.entrez,d,by="gene_stable_id",all.x=TRUE)
  
  df.entrez <<- df.entrez
}

###############################################################################################
###############################################################################################

load_geneSets = function() {
  
  load_entrez()
  
  GENE_SET_FILENAME = "KEGG_LEGACY_WITH_HMDB"
  
  # GENE_SET_FILENAME = "c5.go.v2024.1.Hs.entrez.gmt"
  GENESET_DIR = file.path(baseDir, "Interactome", "MSigDB")
  
  cat("Loading and processing ", GENE_SET_FILENAME, "...","\n")
  if (GENE_SET_FILENAME == "KEGG_LEGACY_WITH_HMDB") {
    load(file = file.path(pathToAnalysisDir, "Interactome",  "HMDB_v5", "hmdb_kegg_pathway_genesets.RData"), verbose = TRUE)
    geneSets = kegg_genesets
    
    CANCER_PATHWAYS = names(geneSets)
    CANCER_PATHWAYS = CANCER_PATHWAYS[grep("CANCER",CANCER_PATHWAYS)]
    
    DISEASE_PATHWAYS = names(geneSets)
    DISEASE_PATHWAYS = DISEASE_PATHWAYS[grep("DISEASE",DISEASE_PATHWAYS)]
    
    LEUKEMIA_PATHWAYS = names(geneSets)
    LEUKEMIA_PATHWAYS = LEUKEMIA_PATHWAYS[grep("LEUKEMIA",LEUKEMIA_PATHWAYS)]
    
    INFECTION_PATHWAYS = names(geneSets)
    INFECTION_PATHWAYS = INFECTION_PATHWAYS[grep("INFECTION",INFECTION_PATHWAYS)]
    
    ri = which(names(geneSets) %in% c("SYSTEMIC_LUPUS_ERYTHEMATOSUS",
                                      "RENAL_CELL_CARCINOMA",
                                      "AMYOTROPHIC_LATERAL_SCLEROSIS_ALS",
                                      LEUKEMIA_PATHWAYS,
                                      "GLIOMA",
                                      "MELANOMA",
                                      INFECTION_PATHWAYS,
                                      DISEASE_PATHWAYS,
                                      CANCER_PATHWAYS))
    
    
    geneSets = geneSets[-ri]
    
  } else if (GENE_SET_FILENAME %in% c("c5.go.v2024.1.Hs.entrez.gmt")) {
    
    geneSetDat = read.delim(file=file.path(GENESET_DIR, GENE_SET_FILENAME), header = FALSE,  col.names = paste0("V",seq_len(2000)), sep = "\t", blank.lines.skip = TRUE, na.strings = c("NA","NaN"))
    
    if (GENE_SET_FILENAME == "c5.go.v2024.1.Hs.entrez.gmt") {
      geneSetDat = geneSetDat[startsWith(geneSetDat[,"V1"],"GOBP_"), ] 
      geneSetDat[,"V1"] = gsub("GOBP_","",geneSetDat[,"V1"] )
    }
    
    # Convert geneSetDat dataframe to geneSetDat list and convert entrez id to Ensembl ID
    totalCount = nrow(geneSetDat)
    count = 0
    pb <- txtProgressBar(min=0,max=totalCount,initial=0,style=3) 
    
    geneSets = list()
    for(ri in 1:nrow(geneSetDat)) {
      count = count + 1
      setTxtProgressBar(pb, (count-1))
      
      v = na.omit(as.character(t(geneSetDat[ri,-c(1,2)])))
      v = df.entrez[df.entrez[,"entrez_id"] %in% v, "gene_stable_id"]
      v = unique(v)
      v = v[order(v)]
      geneSets[[geneSetDat[ri,"V1"]]] = v
    }
    
    setTxtProgressBar(pb, totalCount) 
    close(pb) 
    
  } else {
    stop(paste0("Unknown GENE_SET_FILENAME : ",GENE_SET_FILENAME))
  }
  
  
  # FILTER GENE SETS THAT HAVE MORE THAN 500 members
  ri = which(lengths(geneSets) > 500)
  geneSets = geneSets[-ri]
  
  return (geneSets)
  
}

###############################################################################################
###############################################################################################

LOAD_DATA_PN_PNM_ANNOTATIONS = function(analysis_runs, folds) {
  
  

  
  PNM_TARGET_ANALYSIS = FIGURE_PLOT_ANALYSIS
  PNM_TARGET_FOLD = FIGURE_PLOT_TARGET_FOLD
  PNM_FOLD_ANALYSIS_NAME = paste0(PNM_TARGET_FOLD,"_",PNM_TARGET_ANALYSIS)
  
  analysis_run = c(FIGURE_PLOT_ANALYSIS)
  fold = FIGURE_PLOT_TARGET_FOLD
  
  pnNodeInfo_list = list()
  pnSubNetwork_list = list()
  pnmNetwork_enrichment_summary = list()
  fold_EA_Loadings = list()
  
  for(analysis_run in analysis_runs) {
    analysis_dir =  file.path(pathToAnalysisDir, analysis_run)
    
    for(CV_FOLD_DIR_NAME in folds) {
      FOLD_ANALYSIS_NAME = paste0(CV_FOLD_DIR_NAME,"_",analysis_run)
      
      pnNodeInfo_list[[FOLD_ANALYSIS_NAME]] = read.csv(file.path(analysis_dir, "results", CV_FOLD_DIR_NAME, "oi_network", "pn_nodes.csv"))
      pnSubNetwork_list[[FOLD_ANALYSIS_NAME]] = read.csv(file.path(analysis_dir, "results", CV_FOLD_DIR_NAME, "oi_network", "pn_edges.csv"))
      
      nodeInfo = pnNodeInfo_list[[FOLD_ANALYSIS_NAME]]
      
      cat(analysis_run, CV_FOLD_DIR_NAME, nrow(nodeInfo[nodeInfo[,"nodeType"] == "Terminal", ]),nrow(nodeInfo[nodeInfo[,"nodeType"] == "Steiner", ]),nrow(nodeInfo),"\n")
    }
  }
  
  
  ########################################################################################
  
  
  PN_FOLD_ANALYSIS_NAMES = c()
  
  res = data.frame()
  for(analysis_run in analysis_runs) {
    analysis_dir =  file.path(pathToAnalysisDir, analysis_run)
    
    for(CV_FOLD_DIR_NAME in folds) {
      FOLD_ANALYSIS_NAME = paste0(CV_FOLD_DIR_NAME,"_",analysis_run)
      PN_FOLD_ANALYSIS_NAMES = c(PN_FOLD_ANALYSIS_NAMES, FOLD_ANALYSIS_NAME)
      
      nodeInfo = pnNodeInfo_list[[FOLD_ANALYSIS_NAME]]
      
      # nodeInfo = nodeInfo[nodeInfo[,"PNM"] == 1, ]
      
      signature_ENSG = nodeInfo[,"Alt3RefId"]
      signature_ENSG = signature_ENSG[startsWith(signature_ENSG,"ENSG")]
      
      signature_HMDB = nodeInfo[,"Alt1RefId"]
      signature_HMDB = signature_HMDB[startsWith(signature_HMDB,"HMDB")]
      
      signature = c(signature_ENSG, signature_HMDB)
      
      hyp_obj <- hypeR(signature, geneSets)
      
      # Plot dots plot
      # hyp_dots(hyp_obj,top = 20) # , top = 30)
      
      # Plot enrichment map
      # hyp_emap(hyp_obj) #, similarity_cutoff = 0.15)
      
      if (!dir.exists(file.path(analysis_dir,"report"))) {
        dir.create(file.path(analysis_dir,"report"))
      }
      
      
      # Save to excel
      hyp_to_excel(hyp_obj, file=file.path(analysis_dir,"report",paste0(CV_FOLD_DIR_NAME,"_","hypeR.xlsx")))
      
      df = data.frame(hyp_obj$data)
      
      df = df[,c("label",
                 "pval",
                 "fdr",
                 "geneset",
                 "background",
                 "signature",
                 "overlap",
                 "hits")]
      colnames(df) = c("label",
                       paste0(FOLD_ANALYSIS_NAME,"_","pval"),
                       paste0(FOLD_ANALYSIS_NAME,"_","fdr"),
                       "geneset",
                       "background",
                       paste0(FOLD_ANALYSIS_NAME,"_","signature"),
                       paste0(FOLD_ANALYSIS_NAME,"_","overlap"),
                       paste0(FOLD_ANALYSIS_NAME,"_","hits"))
      
      if (nrow(res) == 0) {
        res = df
      } else {
        res = merge(res,df,by=c("label","geneset","background"),all.x=TRUE,all.y=TRUE)
      }
      
    }
  }
  
  
  res[,"Avg_pval"] = apply(res,1, function(x) {
    tcns = paste0(PN_FOLD_ANALYSIS_NAMES,"_pval")
    v = as.numeric(x[tcns])
    return(10^(sum(log10(v)) / length(v)))
  })
  
  res = res[order(res[,"Avg_pval"]), ]
  row.names(res)  = NULL
  PN_Annotations = res
  
  ###################################################################
  
  
  # Now run annotation on the PNMs for a target fold analysis
  
  
  res = data.frame()
  
  nodeInfo = pnNodeInfo_list[[PNM_FOLD_ANALYSIS_NAME]]
  
  PNMs = unique(nodeInfo[,"PNM"])
  PNMs = PNMs[order(PNMs)]
  
  for(pnm in c(0,PNMs)) {
    nodeInfo = pnNodeInfo_list[[PNM_FOLD_ANALYSIS_NAME]]
    
    if (pnm > 0) {
      nodeInfo = nodeInfo[nodeInfo[,"PNM"] == pnm, ]
    }
    
    
    signature_ENSG = nodeInfo[,"Alt3RefId"]
    signature_ENSG = signature_ENSG[startsWith(signature_ENSG,"ENSG")]
    
    signature_HMDB = nodeInfo[,"Alt1RefId"]
    signature_HMDB = signature_HMDB[startsWith(signature_HMDB,"HMDB")]
    
    signature = c(signature_ENSG, signature_HMDB)
    
    hyp_obj <- hypeR(signature, geneSets)
    
    # Plot dots plot
    # hyp_dots(hyp_obj,top = 20) # , top = 30)
    
    # Plot enrichment map
    # hyp_emap(hyp_obj) #, similarity_cutoff = 0.15)
    
    # if (!dir.exists(file.path(analysis_dir,"report"))) {
    #   dir.create(file.path(analysis_dir,"report"))
    # }
    
    
    # # Save to excel
    # hyp_to_excel(hyp_obj, file=file.path(analysis_dir,"report",paste0(CV_FOLD_DIR_NAME,"_","hypeR.xlsx")))
    
    df = data.frame(hyp_obj$data)
    
    df = df[,c("label",
               "pval",
               "fdr",
               "geneset",
               "background",
               "signature",
               "overlap",
               "hits")]
    colnames(df) = c("label",
                     paste0(PNM_FOLD_ANALYSIS_NAME,"_PNM",pnm, "_","pval"),
                     paste0(PNM_FOLD_ANALYSIS_NAME,"_PNM",pnm,"_","fdr"),
                     "geneset",
                     "background",
                     paste0(PNM_FOLD_ANALYSIS_NAME,"_PNM",pnm,"_","signature"),
                     paste0(PNM_FOLD_ANALYSIS_NAME,"_PNM",pnm,"_","overlap"),
                     paste0(PNM_FOLD_ANALYSIS_NAME,"_PNM",pnm,"_","hits"))
    
    
    if (nrow(res) == 0) {
      res = df
    } else {
      res = merge(res,df,by=c("label","geneset","background"),all.x=TRUE,all.y=TRUE)
    }
  }
  
  res = res[order(res[,paste0(PNM_FOLD_ANALYSIS_NAME,"_PNM0_pval")]), ]
  row.names(res)  = NULL
  PNM_Annotations = res
  
  
  
  dataList = list()
  dataList[["PN_Annotations"]] = PN_Annotations
  dataList[["PNM_Annotations"]] = PNM_Annotations
  dataList[["PN_FOLD_ANALYSIS_NAMES"]] = PN_FOLD_ANALYSIS_NAMES
  dataList[["PNM_FOLD_ANALYSIS_NAME"]] = PNM_FOLD_ANALYSIS_NAME
  dataList[["PNMs"]] = PNMs
  
  
  return(dataList)
  
}


###############################################################################################
###############################################################################################


LOAD_TOP_PNM_EA_MODEL_DATA = function(analysis_run, fold, LOAD_Control_PNM99 = FALSE) {
  
  analysis_dir =  file.path(pathToAnalysisDir, analysis_run)
  
  models_dir = file.path(analysis_dir, "models")
  
  
  # load in the ensemble models and their base model information
  df_base_EA_models = read.csv(file.path(models_dir, 'base_PNM_EA_models.csv'))
  df_ensemble_EA_models = read.csv(file.path(models_dir, ENSEMBLE_EA_MODEL_CV_PERFORMANCE_FN))
  if (LOAD_Control_PNM99 == FALSE) {
    df_ensemble_EA_models = df_ensemble_EA_models[grep("PNM-99",df_ensemble_EA_models[,"ensemble"], invert = TRUE), ]
    #  Model_Rank = 1
    PNM_EA_model_details = read.csv(file.path(models_dir, "PNM_EA_model_1_details.csv"))
    PNM_EA_model_y_values = read.csv(file.path(models_dir, "PNM_EA_model_1_y_values.csv"))
  } else {
    df_ensemble_EA_models = df_ensemble_EA_models[grep("PNM-99",df_ensemble_EA_models[,"ensemble"]), ]
    #  Model_Rank = 1
    PNM_EA_model_details = read.csv(file.path(models_dir, "PNM_EA_PNM99_model_1_details.csv"))
    PNM_EA_model_y_values = read.csv(file.path(models_dir, "PNM_EA_PNM99_model_1_y_values.csv"))
  }
  
  df_model_feature_info = read.csv(file.path(models_dir, 'PNM_EA_model_feature_info.csv'))
  

  
  "ONLY NEEDED TO FIGURE OUT PNM OF A FEATURE"
  pnNodeInfo = read.csv(file.path(analysis_dir, "results", fold, "oi_network", "pn_nodes.csv"))
  # pnSubNetwork = read.csv(file.path(analysis_dir, "results", fold, "oi_network", "pn_edges.csv"))
  
  PNM_EA_model_details[,"measurement_id"] = PNM_EA_model_details[,"featureName"]
  PNM_EA_model_details[,"order"] = c(1:nrow(PNM_EA_model_details))
  PNM_EA_model_details = merge(PNM_EA_model_details,unique(pnNodeInfo[, c("measurement_id","nodeLabel","nodeTitle","PNM")]),by="measurement_id",all.x=TRUE)
  PNM_EA_model_details = PNM_EA_model_details[order(PNM_EA_model_details[ ,"order"]), ]
  row.names(PNM_EA_model_details) = NULL
  
  dataList = list()
  dataList[["PNM_EA_model_details"]] = PNM_EA_model_details
  dataList[["PNM_EA_model_y_values"]] = PNM_EA_model_y_values
  
  
  return (dataList)
  
}


###############################################################################################
###############################################################################################


LOAD_PN_PNM_Data = function(analysis_run, PLOT_TARGET_FOLD, PNM_Colors = NULL) {
  
  analysis_dir =  file.path(pathToAnalysisDir, analysis_run)
  
  
  
  
  CV_FOLD_DIR_NAME = PLOT_TARGET_FOLD
  pnmNetwork.nodeInfo = read.csv(file.path(analysis_dir, "results", CV_FOLD_DIR_NAME, "oi_network", "pn_nodes.csv"))
  pnmNetwork.subNetwork = read.csv(file.path(analysis_dir, "results", CV_FOLD_DIR_NAME, "oi_network", "pn_edges.csv"))
  pnmNetwork.numClusters = length(unique(pnmNetwork.nodeInfo[,"PNM"]))
  
  molecular_features_info = read.csv(file.path(analysis_dir, "results", "molecular_features_info.csv"))
  
  
  
  PNMs = unique(pnmNetwork.nodeInfo[,"PNM"])
  PNMs = PNMs[order(PNMs)]
  numClusters = length(PNMs)
  fitClusterColorLegendOrder = rainbow(numClusters) ##[fit$cluster]
  names(fitClusterColorLegendOrder) = PNMs
  if (length(PNM_Colors) > 0) {
    pnmNetwork.fitClusterColors = fitClusterColorLegendOrder
    for(i in c(1:min(length(PNM_Colors), length(pnmNetwork.fitClusterColors)))) {
      pnmNetwork.fitClusterColors[i] = PNM_Colors[i]
    }
  } else {
    set.seed(1) # randomly assign a color to a given PNM. Note that PNMs are sorted by how many nodes they contain
    pnmNetwork.fitClusterColors = sample(fitClusterColorLegendOrder)
  }
  
  
  subNetwork = pnmNetwork.subNetwork
  nodeInfo = pnmNetwork.nodeInfo
  nodeInfo[,"cluster"] = nodeInfo[,"PNM"]
  
  
  
  s1 = pnmNetwork.subNetwork
  n1 = pnmNetwork.nodeInfo
  s1 = merge(s1,n1[,c("nodeName","PNM")],by.x="nodeName1",by.y="nodeName",all.x=TRUE)
  s1 = merge(s1,n1[,c("nodeName","PNM")],by.x="nodeName2",by.y="nodeName",all.x=TRUE)
  d1 = s1[,c("PNM.x", "PNM.y")]
  d2 = s1[,c("PNM.y", "PNM.x")]
  colnames(d2) = c("PNM.x", "PNM.y")
  d = rbind(d1,d2)
  table(is.na(d[,"PNM.x"]))
  table(is.na(d[,"PNM.y"]))
  nt = data.frame(table(d[,c("PNM.x", "PNM.y")]), stringsAsFactors = FALSE)
  x = matrix(data=0,nrow=pnmNetwork.numClusters,ncol=pnmNetwork.numClusters)
  for(ri in 1:nrow(nt)) {
    x[nt[ri,"PNM.x"],nt[ri,"PNM.y"]] = nt[ri,"Freq"]
    x[nt[ri,"PNM.x"],nt[ri,"PNM.y"]] = nt[ri,"Freq"]
  }
  
  for(ci in 1:ncol(x)) {
    x[x[,ci]<1,ci] = -1
    ii = which(x[,ci]>0)
    x[ii,ci] = log2(x[ii,ci])
  }
  

  pm = x
  cns =  c(1:ncol(pm)) 
  rns = c(1:nrow(pm))
  cns = paste0("PNM-",c(1:ncol(pm))," ") #,pnmNetwork.topGeneSet[,"Term"])
  rns = paste0("PNM-",c(1:nrow(pm))," ") #,pnmNetwork.topGeneSet[,"Term"])
  
  seriate.method = "MDS"
  
  # dist_row <- dist(x)
  # o_row <- seriate(dist_row, method = seriate.method, control = NULL)[[1]]
  dist_col <- dist(t(x))
  o_col <- seriate(dist_col, method = seriate.method, control = NULL)[[1]]
  
  
  ######################################################################
  
  
  n = nrow(nodeInfo)
  nt = table(nodeInfo[,"PNM"])
  mnx = floor(max(nt)^0.5 + 2)
  mnx2 = floor(length(nt)^0.5 + 2)
  pnmOrder = c(1:pnmNetwork.numClusters)
  pnmOrder = pnmOrder[o_col]
  
  
  REVERSE_PLOT_ORDER = FALSE
  
  mv1 = mean(pnmOrder[1:(length(pnmOrder)/2)])
  mv2 = mean(pnmOrder[(length(pnmOrder)/2):(length(pnmOrder))])
  if (mv1 > mv2) {
    REVERSE_PLOT_ORDER = TRUE
  }
  
  if (REVERSE_PLOT_ORDER) {
    pnmOrder = rev(pnmOrder)
  }
  
  
  nodeInfo = nodeInfo[order(nodeInfo[,"degree"],decreasing = TRUE), ]
  nodeInfoUpdate = data.frame()
  
  ny2 = 1
  nx2 = 1
  xyMax = mnx * mnx2
  for(pnm in pnmOrder) {
    pnmNodeInfo = nodeInfo[!is.na(nodeInfo[,"PNM"]) & nodeInfo[,"PNM"] == pnm, ]
    
    nx = 1
    ny = 1
    for(ri in 1:nrow(pnmNodeInfo)) {
      pnmNodeInfo[ri,"x"] =  nx
      pnmNodeInfo[ri,"y"] = ny + (nx %% 2) / 2
      ny = ny + 1
      if (ny > mnx) {
        nx = nx + 1
        ny = 1
      }
    }
    
    pnmNodeInfo[ ,"x"] = pnmNodeInfo[ ,"x"] + nx2 * mnx
    pnmNodeInfo[ ,"y"] = pnmNodeInfo[ ,"y"] + ny2 * mnx
    
    ny2 = ny2 + 1
    if (ny2 > mnx2) {
      nx2 = nx2 + 1
      ny2 = 1
    }
    
    if (nrow(nodeInfoUpdate) == 0) {
      nodeInfoUpdate = pnmNodeInfo
    } else {
      nodeInfoUpdate = rbind(nodeInfoUpdate,pnmNodeInfo)
    }
    
    
  }
  
  print("pnmOrder")
  print(pnmOrder)
  
  nodeInfoUpdate[ ,"x"] =  xyMax - nodeInfoUpdate[ ,"x"]
  nodeInfoUpdate[ ,"y"] =  xyMax -  nodeInfoUpdate[ ,"y"]
  nodeInfoUpdate[ ,"x"] =  2 * nodeInfoUpdate[ ,"x"]
  nodeInfoUpdate[ ,"y"] =  2 * nodeInfoUpdate[ ,"y"]
  nodeInfoUpdate[,"size"] = 5
  nodeInfo = nodeInfoUpdate
  
  oirun_nodeNames = unique(c(subNetwork[, "nodeName1"], subNetwork[,"nodeName2"]))
  subnet = graph_from_data_frame(subNetwork[, c("nodeName1","nodeName2")], vertices = oirun_nodeNames,   directed = F)
  E(subnet)$weight = 1 / as.numeric(subNetwork[, "cost"])
  subnet = simplify(subnet)
  
  ########################################################################
  ###### Plot whole Phenotype_Network coloring nodes by their cluster
  ########################################################################
  nodeInfo[,"nodeColor"] = "lightgrey"
  for(clusterID in 1:pnmNetwork.numClusters) {
    ii = which(nodeInfo[,"nodeName"] %in% unique(pnmNetwork.nodeInfo[!is.na(pnmNetwork.nodeInfo[,"PNM"]) & pnmNetwork.nodeInfo[,"PNM"] == clusterID,"nodeName"]))
    if (length(ii) > 0) {
      nodeInfo[ii,"nodeColor"] = pnmNetwork.fitClusterColors[clusterID]
      
    }
  }
  
  
  dataList = list()
  dataList[["nodeInfo"]] = nodeInfo
  dataList[["subnet"]] = subnet
  
  return(dataList)
}



###############################################################################################
###############################################################################################

