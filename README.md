# PhenoMol
Copyright (c) 2025 GE HealthCare

The PhenoMol_v1 repository provides Python and R-scripts to integrate phenotypic data with multi-omic data via graph theory constrained by prior biological knowledge of molecular interactions and reactions. The graph theory approach is based on the work of the [Fraenkel Lab at MIT](https://github.com/fraenkel-lab) specifically omics integrator (OI). PhenoMol has been applied to predict elite physical performance in a healthy cohort with deep behavioral, cognitive, and molecular characterization that has been described in a manuscript that is currently under review and will be cited here once published. This repository contains the code used to generate the results of that analysis. This material is based upon work supported by the United States Air Force and Defense Advanced Research Projects Agency (DARPA) under Contract No. FA8650-19-C7945.

## Getting Started

Download the content of the PhenoMol_v1 repository to your workstation.

PhenoMol currently requires [python](https://www.python.org) version 3.10 and [R](https://www.r-project.org) version 4.2 to be installed.

The PhenoMol python files provided are dependent upon the following python packages.

#### Python Packages: 
1. [numpy](https://pypi.org/project/numpy) version 1.26.4
2. [pandas](https://pypi.org/project/pandas) version 2.1.3
3. [networkx](https://pypi.org/project/networkx) version 3.2.1
4. [seaborn](https://pypi.org/project/seaborn) version 0.13.2
5. [scipy](https://pypi.org/project/scipy) version 1.11.4
6. [scikit-learn](https://pypi.org/project/scikit-learn) version 1.3.2
7. [matplotlib](https://pypi.org/project/matplotlib) version 3.8.2
8. [pcst-fast](https://pypi.org/project/pcst-fast) version 1.0.10
9. [python-louvain](https://pypi.org/project/python-louvain) version 0.16
10. [fsspec](https://pypi.org/project/fsspec) version 2024.9.0
11. [pyreadr](https://pypi.org/project/pyreadr) version 0.5.2
12. [jupyterlab](https://pypi.org/project/jupyterlab)

The specific versions of the required python packages are listed in the 'requirements.txt' file within the PhenoMol_v1 repository directory. The packages can be installed into the active python environment by running the following command.

>\> pip install -r requirements.txt

The R-script files provided are dependent upon several R libraries. The R packages that must be installed prior to running PhenoMol include:

#### R Packages to run PhenoMol analysis: 
1. [mixOmics](https://www.bioconductor.org/packages/release/bioc/html/mixOmics.html)  
2. [stringr](https://cran.r-project.org/web/packages/stringr)

#### Example running PhenoMol


To demonstrate the applicability of the PhenoMol network-based approach to multi-omic studies in health and disease, we have setup an example analysis that uses the publicly available dataset published by [Contrepois et al. (2020)](https://pubmed.ncbi.nlm.nih.gov/32470399). The unidentifiable cohort data was [downloaded](http://hmp2-data.stanford.edu/index.php) under the sub-study Exercise and was formatted so it could be used by PhenoMol. The formatted data files can be found in the "db_CAE" subdirectory of the PhenoMol_v1 repository. Note that acronym CAE stands for Choreography of Acute Exercise and is the identifier used for this specific dataset. The process of setting up and formatting a dataset can be found in the PhenoMol reference manual.

Peak VO2 was selected as the outcome metric for the example analysis. Peak VO2 was measured in thirty-six (21-Male, 15-Female) subjects along with metabolomics, proteomics, and transcriptomics pre- and post-acute exercise. The example analysis performs stratified 3-fold cross validation in which each fold will have 24 subjects to be used to train models and 12 subjects to test models. The PhenoMol analysis run will require approximately 12 minutes to complete on an Intel® Core™ i7-12700H Processor running at 2300 Mhz. Upon completion of the analysis, a plot should appear presenting the predicted vs actual Peak VO2 for the highest ranked model by R2. The model that was used to generate the plot will be printed as text output. Upon completion of the analysis run, all output will be saved as comma-separated value (csv) files within an analysis output subdirectory within the PhenoMol_v1 repository directory.

#### Running the example PhenoMol analysis from the command line.

Open a command line console and set the current path to the "python" subdirectory within the PhenoMol_v1 repository directory.

>\> cd {path to PhenoMol_v1 repository} \PhenoMol_v1\python

Execute the PhenoMol python file " CAE_MFn_VO2_3_mptp_o_pa_654.py"

>\> python CAE_MFn_VO2_3_mptp_o_pa_654.py

The python file contains the analysis parameter settings and methods used to conduct the analysis. The code will also create and save all output to a directory called “analysis_CAE_MFn_VO2_3_mptp_o_pa_654” within the PhenoMol_v1 repository directory. Upon completion of the analysis, a window should appear presenting a plot of predicted vs actual peak VO2. The performance of the highest ranked model by R2 will be summarized in the console window as

> Outcome: Peak VO2 mL kg min
> All Train Number of Subjects: 36.0
> 
> All Train R2: 0.655
> All Train MSE: 15.441
> All Train RMSE: 3.930
> 
> Number of Samplings: 99.0
> 
> 3-fold CV Train R2 (mean ± stdev): 0.660 ± 0.054
> 3-fold CV Test  R2 (mean ± stdev): 0.578 ± 0.149
> 
> 3-fold CV Train MSE (mean ± stdev): 15.013 ± 2.318
> 3-fold CV Test  MSE (mean ± stdev): 17.641 ± 5.001


#### Running the example PhenoMol analysis from a Jupyter notebook.

Open a command line console and set the current path to the "python" subdirectory within the PhenoMol_v1 repository directory.

>\> cd {path to PhenoMol_v1 repository} \PhenoMol_v1\python

Next start JupyterLab by running the command

>\> jupyter lab

Within JupyterLab, open the notebook “CAE_MFn_VO2_3_mptp_o_pa_654.ipynb” and then select “Run all cells” from the Run menu. Upon completion of the analysis, a plot of predicted vs actual Peak VO2 along with a printout of the model performance will be presented in the notebook’s output cell for the highest ranked model.

#### PhenoMol analysis output files

Upon completion of the example analysis run, all output will be saved as comma-separated value (csv) files within an analysis output subdirectory called “analysis_CAE_MFn_VO2_3_mptp_o_pa_654” within the PhenoMol_v1 repository directory. A file called “run_parameters.csv” that is generated by the save_run_parameters method will be found in the analysis output subdirectory. This file lists all the parameter settings for the analysis run. The output subdirectory will also contain three subdirectories: measures, results, and models.  

The “measures” subdirectory contains the inputted filtered and processed phenotypic and omics data for each subject to be used in the analysis. The file names within the “measures” subdirectory start with either a “mDat_” or “hDat_” prefix. Those files with a “mDat_” prefix contain the measures in a table format with a row for each subject. The subject identifier is listed in the first column called “UID” and the remaining columns represent a unique molecular or phenotypic feature. The files with a “hDat_” prefix are tables where each row contains the header information for a single feature. These mDat and hDat files are generated from the respective database source after going through a filtering and processing step. The given database source for this example was "db_CAE" and is defined by the analysis run parameter “db_data_dir” that is listed in the “run_parameters.csv” file. The input data was processed by normalizing the measures by sex (see run parameter USE_SEX_MEDIAN_NORMALIZED_DATA). After normalizing for gender, the median of all blood draw measures was computed for each molecular feature. The filtering and processing steps are defined and implemented within the python “setup_analysis_run” method.

The “results” subdirectory contains an output directory for each independent omics integrator (oi) fold. Each contains the imputed training and test data, the oi prizes, principal network (PN), and principal network modules (PNMs). There are also the output files for the oi fold’s base and ensemble models for the PNM expression axes (EAs) as well as the phenotypic axis (PAs).

The “models” subdirectory contains a set of files, one set for the PNM EAs and another set for the PAs. Opening the "ensemble_EA_models_cv_performance.csv" file will present the top selected model from each of the three folds. The model and the test R2 under three-fold cross-validation using all the data for the three oi folds is listed as.

Ensemble  | all_cv_test_R2_mean
-------------------------- | -----------------------------------
['fold_001_Peak_VO2_mL_kg_min_EA1_PNM0', 'fold_001_Peak_VO2_mL_kg_min_EA1_PNM2'] | 0.395343
['fold_002_Peak_VO2_mL_kg_min_EA1_PNM3'] | 0.577564
['fold_003_Peak_VO2_mL_kg_min_EA1_PNM1', 'fold_003_Peak_VO2_mL_kg_min_EA1_PNM2'] | 0.505678

The model from oi fold 2 (i.e. 'fold_002’) is the top ranked model and what is being presented at the end of the analysis.  Note that this specific ensemble model ['fold_002_Peak_VO2_mL_kg_min_EA1_PNM3'] can be found listed in the /results/fold_002/ensemble_PNM_EA_models.csv file along with all the other models for oi fold 2. If the contents of this file are sorted by the mean_cv_test_adj_R2 column, the model ['fold_002_Peak_VO2_mL_kg_min_EA1_PNM3'] will be ranked as number one with an adjusted R2 of 0.5390.

The details of the top ranked model can be found in the “models” subdirectory within the PNM_EA_model_1_details.csv file.

featureLabel | nodeSpecies | coef | variable_percent_contribution
-------------|------------- | --------|--------
L-Glutamic acid : HMDB0000148 | Metabolite | -2.08 | 32.4
Arachidonic acid : HMDB0001043 | Metabolite | -1.85 | 25.8
FN1 : P02751 | Protein | -1.72 | 22.2
LEP : P41159 | Protein | -1.62 | 19.6

Notice that Peak VO2 is inversely related to L-Glutamic acid since the coefficient (-2.08) is negative. Lower levels of L-Glutamic acid are observed in individuals with higher Peak VO2. [Contrepois et al. (2020)](https://pubmed.ncbi.nlm.nih.gov/32470399) suggested that glutamic acid was catabolized by skeletal muscle cells to produce energy. They also observed a strong negative association of Leptin (LEP : P41159) with peak VO2.  Furthermore, they observed the Triglycerides (TAG) like Arachidonic acid to also be associated with lower peak VO2 noting that they are biomarkers of poor metabolic health.
