#!/usr/bin/env python3


import os
from os import path

import datetime

import math

import numpy as np
import pandas as pd
import networkx as nx

import community    # pip install python-louvain
from sklearn.cluster import SpectralClustering

from oi3_graph import OIgraph
from oi3_graph import get_robust_subgraph_Robustness
from oi3_graph import get_robust_subgraph_from_randomizations
from oi3_graph import get_networkx_graph_as_dataframe_of_nodes
from oi3_graph import get_networkx_graph_as_dataframe_of_edges
from oi3_graph import summarize_grid_search
from oi3_graph import filter_graph_by_component_size
from oi3_graph import louvain_clustering
from oi3_graph import betweenness


import scipy   # to compute spearman correlation

from scipy import optimize

import multiprocessing

##########################################################################################
##########################################################################################
##########################################################################################

class Options(object):
    def __init__(self, options):
        self.__dict__.update(options)
    def __repr__(self):
        return dict(self.__dict__)
      
      
# class name Change
class Interactome:
  """
  A Interactome object is a representation of a graph, with methods to work with it.
  """
  ###########################################################################
  #######          Initialization            #######
  ###########################################################################

  def __init__(self, params=dict()):

    defaults = {'interactome_db_dir': 'interactome', 
                'interactome_name': 'ppmi_70_percent_confidence_v5', 
                
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
                'use_MicroRNA': False}
                
                

    # Overwrite the defaults with any user-specified parameters.
    self.params = Options({**defaults, **params})
    
    
  def load_interactome_edges(self):

    fn = path.join(self.params.interactome_db_dir,self.params.protein_protein_metabolite_interactome_fp) 
    self.interactome = pd.read_csv(fn, sep='\t')
    print('loaded protein <-> protein <-> metabolite edges:',len(self.interactome))
    
    fn = path.join(self.params.interactome_db_dir,self.params.interactome_transcript_pp_fp) 
    self.interactome_transcript_pp = pd.read_csv(fn, sep='\t')
    print('loaded transcript --> protein edges:',len(self.interactome_transcript_pp))
    
    fn = path.join(self.params.interactome_db_dir,self.params.interactome_gene_transcript_fp) 
    self.interactome_gene_transcript = pd.read_csv(fn, sep='\t')
    print('loaded gene --> transcript edges:',len(self.interactome_gene_transcript))
    
    if self.params.use_MicroRNA:
      fn = path.join(self.params.interactome_db_dir,self.params.interactome_transcript_mature_mir_fp) 
      self.interactome_transcript_mature_mir = pd.read_csv(fn, sep='\t')
      print('loaded transcript --> mature miRNA edges:',len(self.interactome_transcript_mature_mir))

      fn = path.join(self.params.interactome_db_dir,self.params.interactome_gene_transcript_mature_mir_fp) 
      self.interactome_gene_transcript_mature_mir = pd.read_csv(fn, sep='\t')
      print('loaded gene --> transcripts of mature miRNA edges:',len(self.interactome_gene_transcript_mature_mir))
      
      fn = path.join(self.params.interactome_db_dir,self.params.interactome_mature_mir_transcript_fp) 
      self.interactome_mature_mir_transcript = pd.read_csv(fn, sep='\t')
      print('loaded mature miRNA --> transcript (miRNA binding to mRNA) edges:',len(self.interactome_mature_mir_transcript))
      
      # fn = path.join(self.params.interactome_db_dir,self.params.interactome_mature_mir_protein_fp) 
      # self.interactome_mature_mir_protein = pd.read_csv(fn, sep='\t')
      # print('loaded mature miRNA - protein edges:',len(self.interactome_mature_mir_protein))


    
    
  def load_interactome_links(self):
    
    fn = path.join(self.params.interactome_db_dir,self.params.protein_transcript_gene_links_fp) 
    self.protein_transcript_gene_links = pd.read_csv(fn, sep='\t')
    print('loaded protein-transcript-gene links:',len(self.protein_transcript_gene_links))
        
    fn = path.join(self.params.interactome_db_dir,self.params.transcript_gene_links_fp) 
    self.transcript_gene_links = pd.read_csv(fn, sep='\t')
    print('loaded transcript-gene links:',len(self.transcript_gene_links))
    
    
  def load_interactome_node_info(self):
    
    dfcns = ['nodeName', 'nodeLabel', 'nodeTitle', 'Alt1RefId', 'Alt2RefId', 'Alt3RefId', 'nodeSpecies']
    
    fn = path.join(self.params.interactome_db_dir,self.params.metabolite_nodeInfo_fp) 
    df = pd.read_csv(fn, sep='\t')
    df = df[dfcns]
    df.drop_duplicates(inplace = True)
    print('loaded metabolite (biochemical) nodeInfo:',len(df))
    self.nodeInfo_metabolite = df
    
    fn = path.join(self.params.interactome_db_dir,self.params.protein_nodeInfo_fp) 
    df = pd.read_csv(fn, sep='\t')
    df = df[dfcns]
    df.drop_duplicates(inplace = True)
    print('loaded protein nodeInfo:',len(df))
    self.nodeInfo_protein = df
    
    fn = path.join(self.params.interactome_db_dir,self.params.transcript_nodeInfo_fp) 
    df = pd.read_csv(fn, sep='\t')
    df = df[dfcns]
    df.drop_duplicates(inplace = True)
    print('loaded transcript nodeInfo:',len(df))
    self.nodeInfo_transcript = df
    
    fn = path.join(self.params.interactome_db_dir,self.params.gene_nodeInfo_fp) 
    df = pd.read_csv(fn, sep='\t')
    df = df[dfcns]
    df.drop_duplicates(inplace = True)
    print('loaded gene nodeInfo:',len(df))
    self.nodeInfo_gene = df
    
    if self.params.use_MicroRNA:
          
      fn = path.join(self.params.interactome_db_dir,self.params.mature_mir_nodeInfo_fp) 
      df = pd.read_csv(fn, sep='\t')
      df = df[dfcns]
      df.drop_duplicates(inplace = True)
      print('loaded mature mir nodeInfo:',len(df))
      self.mature_mir_nodeInfo = df
          
      fn = path.join(self.params.interactome_db_dir,self.params.mature_mir_transcript_nodeInfo_fp) 
      df = pd.read_csv(fn, sep='\t')
      df = df[dfcns]
      df.drop_duplicates(inplace = True)
      print('loaded mature mir transcript nodeInfo:',len(df))
      self.mature_mir_transcript_nodeInfo = df
      
      fn = path.join(self.params.interactome_db_dir,self.params.mature_mir_gene_nodeInfo_fp) 
      df = pd.read_csv(fn, sep='\t')
      df = df[dfcns]
      df.drop_duplicates(inplace = True)
      print('loaded mature mir gene nodeInfo:',len(df))
      self.mature_mir_gene_nodeInfo = df

    
    self.nodeInfo = self.nodeInfo_metabolite.copy()
    self.nodeInfo = pd.concat([self.nodeInfo, self.nodeInfo_protein])
    self.nodeInfo = pd.concat([self.nodeInfo, self.nodeInfo_transcript])
    self.nodeInfo = pd.concat([self.nodeInfo, self.nodeInfo_gene])
    if self.params.use_MicroRNA:
      self.nodeInfo = pd.concat([self.nodeInfo, self.mature_mir_nodeInfo])    
      self.nodeInfo = pd.concat([self.nodeInfo, self.mature_mir_transcript_nodeInfo])    
      self.nodeInfo = pd.concat([self.nodeInfo, self.mature_mir_gene_nodeInfo])    
    self.nodeInfo.drop_duplicates(subset=['nodeName'], inplace = True)
    self.nodeInfo.reset_index(drop=True, inplace = True)

    print('nodeInfo:',len(self.nodeInfo))
    
##########################################################################################
##########################################################################################
##########################################################################################


##########################################################################################
##########################################################################################
##########################################################################################
  
def _generate_oiPrizes(params):

    inputMeasures = params['inputMeasures']
    
    mDat_phenotypes = inputMeasures['mDat_phenotypes']
    
    hDat_phenotypes = inputMeasures['hDat_phenotypes.csv']
    
    prizeData = pd.DataFrame()
    
    for phenotype in params['prizeNames']:
    
        print(phenotype)
    
        if len(mDat_phenotypes[phenotype].dropna()) < params['min_num_correlation_data_points']:
            continue
        
        if np.std(mDat_phenotypes[phenotype], ddof=1) < 1e-8:
            continue
    
        prizeType = 'phenotype'
        prizeType = list(hDat_phenotypes.loc[hDat_phenotypes['columnName'] == phenotype,'Data_Flavor'])[0]
    
        for mDatFN in params['omicMeasures']:
            mDat = inputMeasures[mDatFN]
            cns = mDat.columns
            measurement_ids = [cn for cn in cns if cn not in [params['UID']]]
            cDat = mDat_phenotypes.merge(mDat, how='left', on=params['UID'])
    
            hDatFN = mDatFN.replace(params['omic_mDat_prefix'], params['hDat_prefix'])
            hDat = inputMeasures[hDatFN]
            
            rho_values = []
            p_value_values = []
            n_values = []
            df1 = pd.DataFrame(measurement_ids, columns=['measurement_id'])
            for measurement_id in measurement_ids:
                pDat = cDat[[phenotype, measurement_id]].dropna()
                if len(pDat) < params['min_num_correlation_data_points']:
                    rho_values.append(np.nan)
                    p_value_values.append(1)
                    n_values.append(0)
                    continue
        
                if np.std(pDat[phenotype], ddof=1) < 1e-8:
                    rho_values.append(np.nan)
                    p_value_values.append(1)
                    n_values.append(0)
                    continue
                    
                if np.std(pDat[measurement_id], ddof=1) < 1e-8:
                    rho_values.append(np.nan)
                    p_value_values.append(1)
                    n_values.append(0)
                    continue
    
                n = len(pDat.index)
                n_values.append(n)
                if n >= params['min_num_correlation_data_points']:
                    rho, p_value = scipy.stats.spearmanr(pDat[phenotype], pDat[measurement_id])
                    rho_values.append(rho)
                    p_value_values.append(p_value)
                    
                else:
                    rho_values.append(np.nan)
                    p_value_values.append(1)
    
                    
            df1['prizeName'] = phenotype
            df1['prizeType'] = prizeType
            df1['Spearman_Cor'] = rho_values
            df1['pvalue'] = p_value_values
            df1['N'] = n_values
    
            df1 = df1[df1['N'] > 0]
            
            df1 = df1.merge(hDat, how='left', on = "measurement_id")
                        
        
            if len(prizeData.index) == 0:
                prizeData = df1.copy()
            else:
                prizeData = pd.concat([prizeData, df1])
    
    prizeData['raw_prize'] = abs(prizeData['Spearman_Cor'])
    
    prizeData.to_csv(params['prize_data_raw_fp'],index=False)
    
    return(len(params['prizeNames']))

##########################################################################################
##########################################################################################

def generate_oiPrizes(params):
  
    
  
  max_num_cpus = params['max_num_cpus']
  prizeNames = params['prizeNames']
    
  results_fn_suffix = os.path.basename(params['prize_data_raw_fp'])

  
  n_request_cpus = min(max_num_cpus,len(prizeNames))
  
  
  if n_request_cpus == 1:
      _generate_oiPrizes(params)

  elif n_request_cpus > 1:

      
      param_sets = []
      
      bins = eq_div(len(prizeNames), n_request_cpus)
      i0 = 0
      for i in range(len(bins)):
          i1 = i0 + bins[i]-1
          bin_phenotypes = prizeNames[i0:(i0 + bins[i])]
          i0 += bins[i]
  
          # make a copy of the parameter set and then overwrite the prizeNames
          params_copy = params.copy()
          params_copy = { **params_copy, 'prize_data_raw_fp': path.join(params['oi_prize_dir'], ('bin_' +  str(i+1) + '_' + results_fn_suffix))}
          params_copy = { **params_copy, 'prizeNames': bin_phenotypes}
          param_sets.append(params_copy)
      

      print('Started multiprocessing of _generate_oiPrizes',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

      print("num prizeNames to process:", len(prizeNames),"cpu processors requested:", n_request_cpus)
      
      pool = multiprocessing.Pool(n_request_cpus)
      results = pool.map(_generate_oiPrizes, param_sets)
      
      print('Completed multiprocessing of _generate_oiPrizes',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

      
      print(results)

      dfList = []
      for i in range(len(bins)):
          results_fp = path.join(params['oi_prize_dir'], ('bin_' +  str(i+1) + '_' + results_fn_suffix))
          # print('reading results file:',results_fp)
          df = pd.read_csv(results_fp)
          dfList = dfList + [df]
      prizeData = pd.concat(dfList)
  
  
      # print('writing combined results file:',params['prize_data_raw_fp'])
      
      prizeData.to_csv(params['prize_data_raw_fp'],index=False)
  
  
  
  # The following will save the combined results file to a zip file, and then delete all of the individual and combined results .csv files 
  # # https://note.nkmk.me/en/python-zipfile/
  # import zipfile
  # # import shutil
  # 
  # zipFN = params['prize_data_raw_fp']
  # zipFN = zipFN.replace('.csv','.zip')
  # 
  # print('writing combined results file to zip file to:',zipFN)
  # with zipfile.ZipFile(zipFN, 'w',
  #                      compression=zipfile.ZIP_DEFLATED,
  #                      compresslevel=9) as zf:
  #     zf.write(params['prize_data_raw_fp'], arcname=os.path.basename(params['prize_data_raw_fp']))
  # 
  
  
  if n_request_cpus > 1:
      # remove individual results files
      for i in range(len(bins)):
          myfile = path.join(params['oi_prize_dir'], ('bin_' +  str(i+1) + '_' + results_fn_suffix))
          if os.path.isfile(myfile):
              os.remove(myfile)
  
  
  
      
  # print('Completed generate_oiPrizes')  
  # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
  
##########################################################################################
##########################################################################################
##########################################################################################

def generateFinalPrizes(params, MAX_NUM_PRIZES = np.Infinity, balance_method = 0, near_method = 0):

  # print('Started generateFinalPrizes')
  # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

  # balance_method =  1 :  will balance the number of RNA or DNA measurements based on the maximum of either the number of metabolite or protein measurements.
  # near_method = 1 :  will filter RNA/DNA keeping only those Near one of the existing metabolite or protein measurements
  # can use the MAX_NUM_PRIZES to limit the number of RNA or DNA measurements used.
  # In all cases the DNA and RNA are sorted by their p-value


  source_interactome = params['source_interactome']

  allNodeInfo = source_interactome.nodeInfo

  prizeNameInfo = pd.read_csv(params['prize_name_info_fp'])
  # can do something ....
  # Will use prizeNameInfo below to weight prizes
  prizeNameInfo.to_csv(params['prize_name_info_fp'],index=False)
  

  prizeData = pd.read_csv(params['prize_data_raw_fp'])
  print('Number of Raw Prizes',len(prizeData))
  print('Number of nodeNames',len(set(prizeData['nodeName'])))
  print('Number of prizeNames',len(set(prizeData['prizeName']))) 

  ##########################################
  
  
  prizeNames = set(prizeData['prizeName'])
  
  # Remove rows that will not appear on the network (ANION GAP, Red blood cell counts, etc.)
  prizeData = prizeData[prizeData['nodeSpecies'].isin(['Metabolite','Protein','RNA','DNA'])]
  
  # Remove rows that do not have a p-value for the raw prize
  prizeData = prizeData.dropna(subset=['pvalue'])
  
  # Remove rows that do not have a nodeName or are not part of the interactome 
  prizeData = prizeData.dropna(subset=['nodeName'])
  
  # only keep prizes in which the nodeName for the measurement appears on the interactome
  # prizeData = prizeData[prizeData['nodeName'].isin(set(allNodeInfo['nodeName']))]
  # code to deal with comma delimited values within 'nodeName' multiple nodeName(s) per measurement_id record
  # expand (explode) nodeName (e.g. explode 1 record with nodeName = 'ENST00000230419,ENST00000481273'  to 2 reconds one for each ENST )
  df = prizeData[['nodeName','measurement_id']].copy()
  df['nodeNames'] = df['nodeName']
  df['nodeName'] = df['nodeName'].str.split(',')
  df = df.explode(['nodeName']).reset_index(drop=True)
  df = df[df['nodeName'].isin(set(allNodeInfo['nodeName']))]
  # only keep prizes that have a measurement that appears on the interactome
  prizeData = prizeData[prizeData['measurement_id'].isin(set(df['measurement_id']))]



  prizeData['norm_prize'] = prizeData['raw_prize']
  
  if params['median_normalize_prizeNames']:
      print('Applying median normalization to prizes across phenotypes and/or treatments')
      print('Normalized to median prize value : ',params['median_prize_value'])
      # prize scaling factor which transforms median of correlation data
      mpv = params['median_prize_value']
      for prizeName in prizeNames:
          maxAbsRawPrizeValue =  max(abs(prizeData.loc[(prizeData['prizeName'] == prizeName), 'raw_prize']))
          sigPrizeValues = prizeData.loc[(prizeData['prizeName'] == prizeName) & (prizeData['pvalue'] <= params['prize_critical_p_value']), 'raw_prize']
          if len(sigPrizeValues) > 0:
              medianAbsRawSigPrizeValues = np.median(abs(sigPrizeValues))
              prizeData.loc[(prizeData['prizeName'] == prizeName), 'norm_prize'] = prizeData.loc[(prizeData['prizeName'] == prizeName), 'norm_prize'] * mpv / medianAbsRawSigPrizeValues
          elif maxAbsRawPrizeValue > mpv:
              prizeData.loc[(prizeData['prizeName'] == prizeName), 'norm_prize'] = prizeData.loc[(prizeData['prizeName'] == prizeName), 'norm_prize'] * mpv / maxAbsRawPrizeValue
      
      
  
  # Remove prizes that do not meet a mimimum p-Value
  prizeData = prizeData.loc[(prizeData['pvalue']  <= params['prize_critical_p_value'])]
  
  # Balance number of RNA and DNA measurements with number of Protein and Metabolite mesurements
  
  # It is important to first sort all the prize data by p-Value
  prizeData = prizeData.sort_values(by=['pvalue'], ascending=True)
  
  
  Metabolites = set(prizeData.loc[prizeData['nodeSpecies'] == 'Metabolite', 'nodeName'])
  Proteins = set(prizeData.loc[prizeData['nodeSpecies'] == 'Protein', 'nodeName'])
  RNA = set(prizeData.loc[prizeData['nodeSpecies'] == 'RNA', 'nodeName'])
  DNA = set(prizeData.loc[prizeData['nodeSpecies'] == 'DNA', 'nodeName'])

  
  interactome = source_interactome.interactome
  
  protein_transcript_gene_links = source_interactome.protein_transcript_gene_links
  
  # Get list that is the union of measured proteins and the proteins that link to the measured metabolites
  
  linkedMetaboliteProteins = {}
  if len(Metabolites) > 0:
      linkedMetaboliteProteins = set(list(interactome.loc[interactome['nodeName1'].isin(Metabolites), 'nodeName2']) + list(interactome.loc[interactome['nodeName2'].isin(Metabolites), 'nodeName1']))
  nearProteins = set(list(Proteins) + list(linkedMetaboliteProteins))

  
  near_ptg_links = protein_transcript_gene_links[protein_transcript_gene_links['protein_id'].isin(nearProteins)]
  
  if near_method == 1:
      print('Near protein ids',len(set(near_ptg_links['protein_id'])))
      print('Near transcript ids',len(set(near_ptg_links['transcript_id'])))
      print('Near gene ids',len(set(near_ptg_links['gene_id'])))

  MAX_NUM_PRIZES = min([MAX_NUM_PRIZES, max([len(RNA), len(DNA)])])
  
  balancedNum = np.Infinity
  if balance_method == 1:
      balancedNum = max([100, len(Proteins), len(Metabolites)])
      print('Using balance method for the maximum number of RNA/DNA')
    
  if near_method == 1:
      print('Applying Near method to order RNA/DNA')

      
  MAX_NUM_PRIZES = min([MAX_NUM_PRIZES, balancedNum, max([len(RNA), len(DNA)])])
  
  print('Num Proteins:', len(Proteins))
  print('Num Metabolites:', len(Metabolites))
  print('Num RNA:', len(RNA))
  print('Num DNA:', len(DNA))
  print('Max Number of RNA/DNA prizes : ',MAX_NUM_PRIZES)
  
  finalPrizeDat = prizeData[~prizeData['nodeSpecies'].isin(['RNA', 'DNA'])].copy()
  
  RNA_PrizeDat = prizeData[prizeData['nodeSpecies'] == 'RNA'] 
  if len(RNA_PrizeDat.index) > 0:
      if near_method == 1:
        RNA_Near = RNA_PrizeDat[RNA_PrizeDat['nodeName'].isin(set(near_ptg_links['transcript_id']))].copy()
        RNA_Far = RNA_PrizeDat[~RNA_PrizeDat['nodeName'].isin(set(near_ptg_links['transcript_id']))].copy()
        RNA_PrizeDat = pd.concat([RNA_Near,RNA_Far])
      for prizeName in prizeNames:
          finalPrizeDat = pd.concat([finalPrizeDat,RNA_PrizeDat[RNA_PrizeDat['prizeName'] == prizeName].head(MAX_NUM_PRIZES).copy()])
  
  DNA_PrizeDat = prizeData[prizeData['nodeSpecies'] == 'DNA'] 
  if len(DNA_PrizeDat.index) > 0:
      if near_method == 1:
        DNA_Near = DNA_PrizeDat[DNA_PrizeDat['nodeName'].isin(set(near_ptg_links['gene_id']))].copy()
        DNA_Far = DNA_PrizeDat[~DNA_PrizeDat['nodeName'].isin(set(near_ptg_links['gene_id']))].copy()
        DNA_PrizeDat = pd.concat([DNA_Near,DNA_Far])
      for prizeName in prizeNames:
          finalPrizeDat = pd.concat([finalPrizeDat,DNA_PrizeDat[DNA_PrizeDat['prizeName'] == prizeName].head(MAX_NUM_PRIZES).copy()])
  
  cns = finalPrizeDat.columns
  pcns = ['nodeName','norm_prize','measurement_id']
  cns = pcns + [cn for cn in cns if cn not in pcns]
  finalPrizeDat = finalPrizeDat[cns]
  
  
  
  finalPrizeDat = finalPrizeDat.sort_values(by=['pvalue'], ascending=True)
  finalPrizeDat.reset_index(drop=True, inplace=True)
  

  #### weight_phenotype
  if params['weight_phenotype_prizes']:
     for prizeName in set(prizeNameInfo['prizeName']):
       weight_phenotype = float(prizeNameInfo.loc[prizeNameInfo['prizeName'] == prizeName ,params['weight_phenotype_column']].values[0])
       finalPrizeDat['norm_prize'] = np.where(finalPrizeDat['prizeName'] == prizeName, finalPrizeDat['norm_prize'] * weight_phenotype,  finalPrizeDat['norm_prize'])
       ## finalPrizeDat.loc[(prizeData['prizeName'] == prizeName), 'norm_prize'] *= weight_phenotype
       # values = list(finalPrizeDat.loc[(prizeData['prizeName'] == prizeName), 'norm_prize'])
       # values = weight_phenotype * values
       # finalPrizeDat.loc[(prizeData['prizeName'] == prizeName), 'norm_prize'] = values
       

  finalPrizeDat.to_csv(params['prize_data_final_fp'],index=False)
  
  # print('Completed generateFinalPrizes')
  # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

##########################################################################################
##########################################################################################
##########################################################################################
  
  
def init_OIruns(params, onlyKeepMaxFullyConnectedGraph = False, verbose = False):
  
  # print('Started init_OIruns')
  # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
  
  source_interactome = params['source_interactome']
  prizeDataFinal_fp = params['prizeDataFinal_fp']
  oirun_interactome_fp = params['oirun_interactome_fp']
  prizes_all_fp = params['prizes_all_fp']

  interactome = source_interactome.interactome
  
  # met = metabolite
  # pro = protein
  # rna = transcript
  # dna = gene
  # mir = miRNA micro-RNA
  # hsa = mature miRNA
 
  met_nodes = set()
  pro_nodes = set()
  rna_nodes = set()
  dna_nodes = set()
  hsa_nodes = set()
  
  d1 = interactome[interactome['nodeName1'].str.startswith('ENSP', na=False)]
  d2 = interactome[interactome['nodeName2'].str.startswith('ENSP', na=False)]
  pro_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())

  d1 = interactome[interactome['nodeName1'].str.startswith('CID', na=False)]
  d2 = interactome[interactome['nodeName2'].str.startswith('CID', na=False)]
  met_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())

  pro_pro_edges = interactome[interactome['nodeName1'].str.startswith('ENSP', na=False) & interactome['nodeName2'].str.startswith('ENSP', na=False)].copy()

  met_pro_edges = interactome[interactome['nodeName1'].str.startswith('CID', na=False) | interactome['nodeName2'].str.startswith('CID', na=False)].copy()

  if params['use_RNA']:
      
      rna2pro_edges = source_interactome.interactome_transcript_pp
      rna2pro_edges = rna2pro_edges.drop_duplicates()
      
      d1 = rna2pro_edges[rna2pro_edges['nodeName1'].str.startswith('ENST', na=False)]
      d2 = rna2pro_edges[rna2pro_edges['nodeName2'].str.startswith('ENST', na=False)]
      rna_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())

      d1 = rna2pro_edges[rna2pro_edges['nodeName1'].str.startswith('ENSP', na=False)]
      d2 = rna2pro_edges[rna2pro_edges['nodeName2'].str.startswith('ENSP', na=False)]
      pro_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())
      
  if params['use_DNA']:
      
      dna2rna_edges = source_interactome.interactome_gene_transcript
      dna2rna_edges = dna2rna_edges.drop_duplicates()
      
      d1 = dna2rna_edges[dna2rna_edges['nodeName1'].str.startswith('ENSG', na=False)]
      d2 = dna2rna_edges[dna2rna_edges['nodeName2'].str.startswith('ENSG', na=False)]
      dna_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())

      d1 = dna2rna_edges[dna2rna_edges['nodeName1'].str.startswith('ENST', na=False)]
      d2 = dna2rna_edges[dna2rna_edges['nodeName2'].str.startswith('ENST', na=False)]
      rna_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())
      
  if params['use_MicroRNA']:

      # DNA gene that encodes the transcript of the mature miRNA
      dna2rna_of_hsa_edges = source_interactome.interactome_gene_transcript_mature_mir
      dna2rna_of_hsa_edges = dna2rna_of_hsa_edges.drop_duplicates()

      d1 = dna2rna_of_hsa_edges[dna2rna_of_hsa_edges['nodeName1'].str.startswith('ENSG', na=False)]
      d2 = dna2rna_of_hsa_edges[dna2rna_of_hsa_edges['nodeName2'].str.startswith('ENSG', na=False)]
      dna_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())

      d1 = dna2rna_of_hsa_edges[dna2rna_of_hsa_edges['nodeName1'].str.startswith('ENST', na=False)]
      d2 = dna2rna_of_hsa_edges[dna2rna_of_hsa_edges['nodeName2'].str.startswith('ENST', na=False)]
      rna_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())


      # transcript that is the source of of the mature miRNA
      rna2hsa_edges = source_interactome.interactome_transcript_mature_mir
      rna2hsa_edges = rna2hsa_edges.drop_duplicates()
      
      d1 = rna2hsa_edges[rna2hsa_edges['nodeName1'].str.startswith('ENST', na=False)]
      d2 = rna2hsa_edges[rna2hsa_edges['nodeName2'].str.startswith('ENST', na=False)]
      rna_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())

      # using ~ operator for ENST and looking for nodeName to start with 'hsa-miR-' or 'hsa-let-'
      d1 = rna2hsa_edges[~rna2hsa_edges['nodeName1'].str.startswith('ENST', na=False)]
      d2 = rna2hsa_edges[~rna2hsa_edges['nodeName2'].str.startswith('ENST', na=False)]
      hsa_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())


      # mature miRNA binding to mRNA transcripts
      hsa2rna_edges = source_interactome.interactome_mature_mir_transcript
      hsa2rna_edges = hsa2rna_edges.drop_duplicates()

      d1 = hsa2rna_edges[hsa2rna_edges['nodeName1'].str.startswith('ENST', na=False)]
      d2 = hsa2rna_edges[hsa2rna_edges['nodeName2'].str.startswith('ENST', na=False)]
      rna_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())

      # using ~ operator for ENST and looking for nodeName to start with 'hsa-miR-' or 'hsa-let-'
      d1 = hsa2rna_edges[~hsa2rna_edges['nodeName1'].str.startswith('ENST', na=False)]
      d2 = hsa2rna_edges[~hsa2rna_edges['nodeName2'].str.startswith('ENST', na=False)]
      hsa_nodes.update(d1['nodeName1'].tolist() + d2['nodeName2'].tolist())


  if verbose:
      print('Number of flat chemical nodes (CID) : ' , len(met_nodes))
      print('Number of protein nodes (ENSP) : ' , len(pro_nodes))
      print('Number of RNA transcript nodes (ENST) : ' , len(rna_nodes))
      print('Number of DNA gene nodes (ENSG) : ' , len(dna_nodes))
      print('Number of mature microRNA nodes (hsa) : ' , len(hsa_nodes))
  
      print('Number pp interactions: ' + str(len(pro_pro_edges.index)))
      print('Number pm interactions: ' + str(len(met_pro_edges.index)))
  
  
  ##############################################################################
  
  # load in Final Prize Data
  prizeData = pd.read_csv(prizeDataFinal_fp)
  
  # expand prize data file
  # valid_nodeNames = list(met_nodes + pro_nodes + rna_nodes + dna_nodes + hsa_nodes)
  

  prizeData['nodeNames'] = prizeData['nodeName']
  prizeData['nodeName'] = prizeData['nodeName'].str.split(',')
  prizeData = prizeData.explode(['nodeName']).reset_index(drop=True)
  
  # df1 = prizeData.loc[prizeData["nodeName"].str.count(",") > 0].copy() 
  # prizeData = prizeData.loc[prizeData["nodeName"].str.count(",") == 0].copy()

  # cns = []
  # # for cn in ['nodeName','nodeLabel', 'nodeTitle',  'Alt1RefId',  'Alt2RefId',  'Alt3RefId',  'nodeSpecies']:
  # # ToDO add ability to do across other columns that have the same number of commas.
  # for cn in ['nodeName']:
  #   if max(df1[cn].str.count(",")) > 0:
  #       cns = cns + [cn]
  #       df1[cn] = df1[cn].str.split(',')
  # expandedRows = df1.explode(cns).reset_index(drop=True)
  # prizeData = pd.concat([prizeData, expandedRows], ignore_index = True)
    
  # for i in range(len(df1)):
  #     row = df1.iloc[i]
  #     nodeNames = row['nodeName'].split(',')
  #     replication_factor = len(nodeNames)
  #     # val[i] = any(x in valid_nodeNames for x in nodeNames)
  #     replicated_rows = np.tile(row, (replication_factor, 1))
  #     expandedRows = pd.DataFrame(replicated_rows, columns=df1.columns)
  #     expandedRows['nodeName'] = nodeNames
  #     # expandedRows = expandedRows.assign(nodeName=nodeNames)
  #     for cn in ['nodeLabel', 'nodeTitle',  'Alt1RefId',  'Alt2RefId',  'Alt3RefId',  'nodeSpecies']:
  #         if row[cn].count(",") > 0:
  #             v = row[cn].split(',')
  #             expandedRows[cn] = v
  #     prizeData = pd.concat([prizeData, expandedRows], ignore_index = True)
  # prizeData = prizeData.copy()
    
    
  prizeData['ppmi_node_id'] = 'X'
  prizeData.loc[prizeData['nodeName'].isin(met_nodes),'ppmi_node_id'] = prizeData.loc[prizeData['nodeName'].isin(met_nodes),'nodeName']
  prizeData.loc[prizeData['nodeName'].isin(pro_nodes),'ppmi_node_id'] = prizeData.loc[prizeData['nodeName'].isin(pro_nodes),'nodeName']
  prizeData.loc[prizeData['nodeName'].isin(rna_nodes),'ppmi_node_id'] = prizeData.loc[prizeData['nodeName'].isin(rna_nodes),'nodeName']
  prizeData.loc[prizeData['nodeName'].isin(dna_nodes),'ppmi_node_id'] = prizeData.loc[prizeData['nodeName'].isin(dna_nodes),'nodeName']
  prizeData.loc[prizeData['nodeName'].isin(hsa_nodes),'ppmi_node_id'] = prizeData.loc[prizeData['nodeName'].isin(hsa_nodes),'nodeName']
  

  
  cns = prizeData.columns.tolist()
  cns.remove('ppmi_node_id')
  cns.insert(0,'ppmi_node_id')
  prizeData = prizeData[cns]
  
  # The following is for legacy code/runs
  cns = prizeData.columns.tolist()
  if 'NodeLabel' in cns:
      prizeData.rename(columns = {'NodeLabel':'nodeLabel'}, inplace = True) 
  if 'NodeTitle' in cns:
      prizeData.rename(columns = {'NodeTitle':'nodeTitle'}, inplace = True) 
  if 'NodeSpecies' in cns:
      prizeData.rename(columns = {'NodeSpecies':'nodeSpecies'}, inplace = True) 
  
  if verbose:
      print('Number of prizeData rows : ' , len(prizeData.index))
  prizeData = prizeData[prizeData['ppmi_node_id'] != 'X'].copy()
  if verbose:
      print('Number of prizeData rows linked to PPMI : ' , len(prizeData.index))
  
  # d1 = prizeData[prizeData['ppmi_node_id'].str.startswith('CID', na=False)]
  # metNodesOnPPMI = d1['ppmi_node_id'].unique().tolist()
  # metNodesOnPPMI = set(metNodesOnPPMI)
  
  # d1 = prizeData[prizeData['ppmi_node_id'].str.startswith('ENSP', na=False)]
  # proNodesOnPPMI = d1['ppmi_node_id'].unique().tolist()
  # proNodesOnPPMI = set(proNodesOnPPMI)
  
  d1 = prizeData[prizeData['ppmi_node_id'].str.startswith('ENST', na=False)]
  rnaNodesOnPPMI = d1['ppmi_node_id'].unique().tolist()
  rnaNodesOnPPMI = set(rnaNodesOnPPMI)
  
  d1 = prizeData[prizeData['ppmi_node_id'].str.startswith('ENSG', na=False)]
  dnaNodesOnPPMI = d1['ppmi_node_id'].unique().tolist()
  dnaNodesOnPPMI = set(dnaNodesOnPPMI)
  
  d1 = prizeData[prizeData['ppmi_node_id'].isin(hsa_nodes)]
  hsaNodesOnPPMI = d1['ppmi_node_id'].unique().tolist()
  hsaNodesOnPPMI = set(hsaNodesOnPPMI)
  
  
  


  prizeData.to_csv(prizes_all_fp,index=False)

        
  
  # prizeData.head()
  
  
  ##############################################################################
  
  if verbose:
      print('Build Interactome')
      print('Add protein-protein actions : ' , len(pro_pro_edges.index))
      print('Add chemical-protein actions : ' , len(met_pro_edges.index))
      
  interactome2 = pd.concat([pro_pro_edges, met_pro_edges])
  
  if verbose:
      print(len(interactome2.index))
  
  if (params['use_RNA']):
  
      rna2pro_edges = rna2pro_edges[rna2pro_edges['nodeName1'].isin(rnaNodesOnPPMI) | rna2pro_edges['nodeName2'].isin(rnaNodesOnPPMI)].copy()
      if verbose:
          print('Add RNA transcript actions : ' , len(rna2pro_edges.index))
      interactome2 = pd.concat([interactome2.copy(), rna2pro_edges])
      if verbose:
          print(len(interactome2.index))
  
      
  if (params['use_DNA']):
  
      dna2rna_edges = dna2rna_edges[dna2rna_edges['nodeName1'].isin(dnaNodesOnPPMI) | dna2rna_edges['nodeName2'].isin(dnaNodesOnPPMI)].copy()
      if verbose:
          print('Add DNA gene actions : ' , len(dna2rna_edges.index))
      interactome2 = pd.concat([interactome2.copy(), dna2rna_edges])
      if verbose:
          print(len(interactome2.index))
      
  if (params['use_MicroRNA']):
      
      # only add edges if there is a prize for the upstream DNA node
      dna2rna_of_hsa_edges = dna2rna_of_hsa_edges[dna2rna_of_hsa_edges['nodeName1'].isin(dnaNodesOnPPMI) | dna2rna_of_hsa_edges['nodeName2'].isin(dnaNodesOnPPMI)].copy()
      if verbose:
          print('Add DNA to RNA of mature miRNA actions : ' , len(dna2rna_of_hsa_edges.index))
      interactome2 = pd.concat([interactome2.copy(), dna2rna_of_hsa_edges])
      if verbose:
          print(len(interactome2.index))
      
      # only add edges if there is a prize for the upstream RNA node
      # rna2hsa_edges = rna2hsa_edges[rna2hsa_edges['nodeName1'].isin(rnaNodesOnPPMI) | rna2hsa_edges['nodeName2'].isin(rnaNodesOnPPMI)].copy()
      if verbose:
          print('Add RNA to mature miRNA actions : ' , len(rna2hsa_edges.index))
      interactome2 = pd.concat([interactome2.copy(), rna2hsa_edges])
      if verbose:
          print(len(interactome2.index))
      
      
      # add edges if there is a prize for the downstream RNA node
      # hsa2rna_edges = hsa2rna_edges[hsa2rna_edges['nodeName1'].isin(rnaNodesOnPPMI) | hsa2rna_edges['nodeName2'].isin(rnaNodesOnPPMI)].copy()
      if verbose:
          print('Add mature miRNA ot target mRNA actions : ' , len(hsa2rna_edges.index))
      interactome2 = pd.concat([interactome2.copy(), hsa2rna_edges])
      if verbose:
          print(len(interactome2.index))


  G = nx.Graph()
  G.add_weighted_edges_from(interactome2.to_records(index=False))
  print('numNodes', len(G), 'numEdges', len(G.edges), 'in Interactome')

    
  if onlyKeepMaxFullyConnectedGraph:
    # Retain largest connected component
    G = G.subgraph(max(nx.connected_components(G),key=len))
    print('numNodes', len(G), 'numEdges', len(G.edges), 'in Interactome sub-graph of max connected components')
  else:
    G = filter_graph_by_component_size(G, params["min_component_size"])
    print('numNodes', len(G), 'numEdges', len(G.edges), 'in Interactome sub-graph after filtering out small components of size < ',params["min_component_size"])


  connected = pd.DataFrame([{'nodeName1':u,'nodeName2':v,'cost':w} for u,v,w in G.edges(data='weight')])
  
  print('numEdges', len(connected.index), 'in fully connected Interactome')
  
  connected.head()
  
  
  connected.to_csv(oirun_interactome_fp, index=False)

    
  # print('interactome file:',oirun_interactome_fp)

  # print('Completed init_OIruns')
  # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

##########################################################################################
##########################################################################################
##########################################################################################
# , graph, source_interactome, targetPhenotypes = [], targetPipelines = [], pcsf_run_dir = '', prizes_all_fp = '',  num_reps = 100, Ws = [1], Bs = [16], Gs = [5], Rs = [0.2], noisy_edges_reps = 100, random_terminals_reps = 100, edge_noise = 0.1, min_robustness = 0.1, seed=123, max_num_cpus = 1

def hyperparameter_search(params):

  graph = params["graph"]
  source_interactome = params["source_interactome"]
  tuningPhenotypes = params["tuningPhenotypes"]
  targetPipelines = params["targetPipelines"]
  pcsf_run_dir = params["pcsf_run_dir"]
  prizes_all_fp  = params["prizes_all_fp"]
  num_reps = params["num_reps"]
  Ws = params["Ws"]
  Bs = params["Bs"]
  Gs = params["Gs"]
  Rs = params["Rs"]
  noisy_edges_reps = params["noisy_edges_reps"]
  random_terminals_reps = params["random_terminals_reps"]
  edge_noise = params["edge_noise"]
  verbose_oi = params["verbose_oi"]
  
  maxNumNodesToComputeAvgPath = params["maxNumNodesToComputeAvgPath"]
    
  seed = params["seed"]
  np.random.seed(seed=params["seed"])
  
  max_num_cpus = params["max_num_cpus"]
  
  gres = None
  numOIruns = 0
  # targetPhenotype = tuningPhenotypes[0]
  for tuningPhenotype in tuningPhenotypes:
      
      numOIruns = numOIruns + 1
  
      print('')
      print(numOIruns, ' of ', len(tuningPhenotypes))
      print(tuningPhenotype)
      print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
      
      phenotypeName = tuningPhenotype.replace(' ','_').replace('.','_')
      # prize_fp = path.join(pcsf_run_dir, (phenotypeName + '_' + 'NodeMaxPrizes.zip')) # save as zip file
      prize_fp = path.join(pcsf_run_dir, (phenotypeName + '_' + 'NodeMaxPrizes.csv'))
      
      prizeData = pd.read_csv(prizes_all_fp)
      prizeData.rename(columns = {'ppmi_node_id':'name'}, inplace = True) 
  
      rpd = prizeData.copy()
      rpd['prize'] = abs(rpd['norm_prize'])
      rpd.sort_values(by=['prize'], ascending=False, inplace = True)
      rpd.drop_duplicates(subset=['Alt1RefId'], inplace = True)
      rpd = set(rpd['Alt1RefId'])
      allNodeInfo = source_interactome.nodeInfo_metabolite.copy()
      allNodeInfo = pd.concat([allNodeInfo[allNodeInfo['Alt1RefId'].isin(rpd)], allNodeInfo[~allNodeInfo['Alt1RefId'].isin(rpd)]])  
      allNodeInfo = pd.concat([allNodeInfo, source_interactome.nodeInfo_protein])
      allNodeInfo = pd.concat([allNodeInfo, source_interactome.nodeInfo_transcript])
      allNodeInfo = pd.concat([allNodeInfo, source_interactome.nodeInfo_gene])
      if params['use_MicroRNA']:
        allNodeInfo = pd.concat([allNodeInfo, source_interactome.mature_mir_nodeInfo])    
        allNodeInfo = pd.concat([allNodeInfo, source_interactome.mature_mir_transcript_nodeInfo])    
        allNodeInfo = pd.concat([allNodeInfo, source_interactome.mature_mir_gene_nodeInfo])    
      allNodeInfo.drop_duplicates(subset=['nodeName'], inplace = True)
      allNodeInfo.reset_index(drop=True, inplace = True)
    
    
      if (len(targetPipelines) > 0):
          prizeData = prizeData.loc[prizeData['Pipeline'].isin(targetPipelines)].copy()
      
      prizeData = prizeData.loc[prizeData['prizeName'] == tuningPhenotype].copy()
      
      prizeData['prize'] = abs(prizeData['norm_prize'])
      prizeData['prizeSigned'] = prizeData['norm_prize']
      
      # filtering by p-value 
      prizeData = prizeData[prizeData['pvalue'] <= params['prize_critical_p_value']]
  
      # Sort prizes by decending order and only keep highest one
      prizeData.sort_values(by=['prize'], ascending=False, inplace = True)
      prizeData.drop_duplicates(subset=['name'], inplace = True)
      prizeData.reset_index(drop=True, inplace = True)
  
      if len(prizeData.index) < 2:
          print('Skip due to less than 2 nodes with prizes found.')
          print('')
          continue
          
      print('Total number node prizes:  ' + str(len(prizeData.index)))
      print('Median prize:', prizeData['prize'].median())
  
  
      prizeData['species'] = ''
      prizeData.loc[prizeData['nodeSpecies'] == 'RNA', 'species'] = 'R'
      prizeData.loc[prizeData['nodeSpecies'] == 'DNA', 'species'] = 'D'
      prizeData.loc[prizeData['nodeSpecies'] == 'Metabolite', 'species'] = 'M'
      prizeData.loc[prizeData['nodeSpecies'] == 'Protein', 'species'] = 'P'
            
      cns = prizeData.columns.tolist()
      pcns = ['name', 'prize', 'species']
      cns = pcns + [cn for cn in cns if cn not in pcns ]
      prizeData = prizeData[cns]
      
      
      if len(params["removePrizeNodeNames"]) > 0:
          prizeData = prizeData[~prizeData['name'].isin(params["removePrizeNodeNames"])]
      
      prizeData.set_index('name', inplace = True) 
      prizeData.to_csv(prize_fp,index=True) # Note it is important to include index

  
      results = graph.grid_randomization(prize_fp, Ws=Ws, Bs=Bs, Gs=Gs, Rs=Rs, noisy_edges_reps=noisy_edges_reps, random_terminals_reps=random_terminals_reps, edge_noise=edge_noise, seed=seed,verbose_oi=verbose_oi, max_num_cpus=max_num_cpus)
  
      # # note that augmented_forest may contain edges that were not in the pcsf solutions.
      robust_min_robustness = {}
      data = []
      for paramstring, forests in results.items(): 
          if forests['augmented_forest'].number_of_nodes() < params["min_component_size"]:
              forests['robust'] = nx.empty_graph(0)
              continue
        
          # meet both criteria a robustness > min_robustness and a robustness > than the robustness at the given max_graph_size
          robustness_at_max_graph_size = get_robust_subgraph_Robustness(forests['augmented_forest'], max_size=params["max_graph_size"])
          min_robustness = max(robustness_at_max_graph_size, params["min_robustness"])
          
          # add to dictionary to be recalled and used later in method
          robust_min_robustness = {**robust_min_robustness, (paramstring + '_N_' + str(num_reps)): min_robustness}
          
          forests['robust'] = get_robust_subgraph_from_randomizations(forests['augmented_forest'], min_robustness = min_robustness, max_size=np.Infinity, min_component_size=params["min_component_size"])

          # if ((forests['robust'].number_of_nodes() > 0) and (forests['robust'].number_of_nodes() <= maxNumNodesToComputeBetweenness)):
          #     betweenness(forests['robust'])
              
          # louvain_clustering(forests['robust'])
          
          avgShortestPathlength = np.nan
          if forests['robust'].number_of_nodes() >= params["min_component_size"]:
              G = forests['robust'].subgraph(max(nx.connected_components(forests['robust']),key=len))
              # computation of average_shortest_path_length of a graph of 4000 nodes requires ~ 1 minute of computational time, 6700 nodes requires ~ 6.3 minutes
              if G.number_of_nodes() <= maxNumNodesToComputeAvgPath:
                  avgShortestPathlength = nx.average_shortest_path_length(G)

              
              
          vs = str(paramstring).split('_')   # for example ['W', '1.00', 'B', '16.00', 'G', '5.00', 'R', '0.25'] 
          w = vs[1]
          b = vs[3]
          g = vs[5]
          r = vs[7]
          
          data.append([tuningPhenotype, (str(paramstring) + '_N_' + str(num_reps)), w, b, g, r, num_reps, avgShortestPathlength])
          
      graph_metrics = pd.DataFrame(data, columns=['Phenotype','Hyperparameters','W','B','G','R','num_randomizations', 'average_shortest_path_length'])
      
      
      
      
      # fn = phenotypeName + '_hyperparam_graph_metrics.csv'
      # graph_metrics.to_csv(path.join(pcsf_run_dir, fn), index=False)
          
      robustness_df = summarize_grid_search(results, mode = 'robustness', graphName = 'robust')
      robustness_df = robustness_df[robustness_df.sum(axis=1) > 0]
      if len(robustness_df.index) > 0:
          specificity_df = summarize_grid_search(results, mode = 'specificity', graphName = 'robust')
          node_attributes_df = graph.node_attributes
          node_attributes_df = node_attributes_df[['prize','terminal','degree']]
          node_attributes_df = allNodeInfo.set_index('nodeName').merge(node_attributes_df, how='right', left_index=True, right_index=True)
          node_attributes_df = node_attributes_df.merge(robustness_df.add_suffix('_N_' + str(num_reps)).add_prefix('robustness_'), how='right', left_index=True, right_index=True)
          node_attributes_df = node_attributes_df.merge(specificity_df.add_suffix('_N_' + str(num_reps)).add_prefix('specificity_'), how='left', left_index=True, right_index=True)
  
          

          if params["save_hyperparam_results"]:
              fn = phenotypeName + '_' + params["fn_suffix_hyperparam_results"] + '.csv'
              node_attributes_df.to_csv(path.join(pcsf_run_dir, fn), index=True) # Note it is important to include index

          nodeSpecies = list(set(node_attributes_df['nodeSpecies']))
          nodeSpecies = ['Protein','Metabolite'] +  [cn for cn in nodeSpecies if cn not in ['Protein','Metabolite']]
  
          parmSets = node_attributes_df.columns
          parmSets = [parmSet.replace('robustness_','') for parmSet in parmSets if parmSet.startswith('robustness_')]
          parmSets = sorted(parmSets)
          
          colValues = []
          colValues.append('Min Robustness')
          colValues.append('Num Nodes')
          colValues.append('Num Terminal')
          colValues.append('Num Steiner')
          colValues.append('mean Terminal degree')
          colValues.append('mean Steiner degree')
          colValues.append('mean Terminal robustness')
          colValues.append('mean Steiner robustness')
          colValues.append('mean Terminal specificity')
          colValues.append('mean Steiner specificity')
          
          for Species in nodeSpecies:
              Species = str(Species)
              colValues.append('Num ' + Species + ' Terminal')
              colValues.append('Num ' + Species + ' Steiner')
              colValues.append('mean ' + Species + ' Terminal degree')
              colValues.append('mean ' + Species + ' Steiner degree')
              colValues.append('mean ' + Species + ' Terminal robustness')
              colValues.append('mean ' + Species + ' Steiner robustness')
              colValues.append('mean ' + Species + ' Terminal specificity')
              colValues.append('mean ' + Species + ' Steiner specificity')
          
          res = pd.DataFrame(colValues, columns=['Attribute'])
          
          for parmSet in parmSets:
              colValues = []
          
              df = node_attributes_df[['terminal','nodeSpecies','degree',('robustness_' + parmSet),('specificity_' + parmSet)]]
              df = df[df['robustness_' + parmSet] > 0]

              # recall min_robustness for this parmSet 
              min_robustness = robust_min_robustness[parmSet]
              df = df[df['robustness_' + parmSet] >= min_robustness]
              
              colValues.append(min_robustness)
              colValues.append(len(df.index))
              colValues.append(len(df[df['terminal'] == True].index))
              colValues.append(len(df[df['terminal'] == False].index))
              colValues.append(df.loc[(df['terminal'] == True), 'degree'].mean())
              colValues.append(df.loc[(df['terminal'] == False), 'degree'].mean())
              colValues.append(df.loc[(df['terminal'] == True), 'robustness_' + parmSet].mean())
              colValues.append(df.loc[(df['terminal'] == False), 'robustness_' + parmSet].mean())
              colValues.append(df.loc[(df['terminal'] == True), 'specificity_' + parmSet].mean())
              colValues.append(df.loc[(df['terminal'] == False), 'specificity_' + parmSet].mean())
              
              for Species in nodeSpecies:
                  colValues.append(len(df[((df['terminal'] == True) & (df['nodeSpecies'] == Species))].index))
                  colValues.append(len(df[((df['terminal'] == False) & (df['nodeSpecies'] == Species))].index))
                  colValues.append(df.loc[((df['terminal'] == True) & (df['nodeSpecies'] == Species)), 'degree'].mean())
                  colValues.append(df.loc[((df['terminal'] == False) & (df['nodeSpecies'] == Species)), 'degree'].mean())
                  colValues.append(df.loc[((df['terminal'] == True) & (df['nodeSpecies'] == Species)), 'robustness_' + parmSet].mean())
                  colValues.append(df.loc[((df['terminal'] == False) & (df['nodeSpecies'] == Species)), 'robustness_' + parmSet].mean())
                  colValues.append(df.loc[((df['terminal'] == True) & (df['nodeSpecies'] == Species)), 'specificity_' + parmSet].mean())
                  colValues.append(df.loc[((df['terminal'] == False) & (df['nodeSpecies'] == Species)), 'specificity_' + parmSet].mean())
          
              res[parmSet] = list(colValues)
              res = res.copy()
          
          res.set_index('Attribute',inplace=True)
          res = res.transpose()
  
          res.insert(loc = 0, column = 'Delta_Num_Nodes', value = 0)
          res['Delta_Num_Nodes'] = (res['Num Terminal'] - res['Num Steiner']) * 2 / (res['Num Terminal'] + res['Num Steiner'])
          # res.sort_values(by=['Delta_Num_Nodes'], ascending=True, inplace = True)
  
          
  
          res.insert(loc = 0, column = 'Delta_RNA_Nodes', value = 0)
          if 'RNA' in nodeSpecies:
              res['Delta_RNA_Nodes'] = (res['Num Protein Terminal'] + res['Num Metabolite Terminal'] - res['Num RNA Terminal']) / (res['Num Protein Terminal'] + res['Num Metabolite Terminal'] + res['Num RNA Terminal'])
          
          res.insert(loc = 0, column = 'Delta_DNA_Nodes', value = 0)
          if 'DNA' in nodeSpecies:
              res['Delta_DNA_Nodes'] = (res['Num Protein Terminal'] + res['Num Metabolite Terminal'] - res['Num DNA Terminal']) / (res['Num Protein Terminal'] + res['Num Metabolite Terminal'] + res['Num DNA Terminal'])
          
          res.insert(loc = 0, column = 'Delta_Node_Degree', value = 0)
          # must have at least measurements for 10% of the protein nodes in order to use them for balancing of degree 
          ftn = len(res['mean Protein Terminal degree']) / (len(res['mean Protein Terminal degree']) + len(res['mean Protein Steiner degree']))
          if ((ftn > 0.1) &  (np.mean(res['mean Protein Terminal degree']) > 0)):
              res['Delta_Node_Degree'] = (res['mean Protein Terminal degree'] - res['mean Protein Steiner degree']) * 2 / (res['mean Protein Terminal degree'] + res['mean Protein Steiner degree'])
          else:
              res['Delta_Node_Degree'] = (res['mean Terminal degree'] - res['mean Steiner degree']) * 2 / (res['mean Terminal degree'] + res['mean Steiner degree'])
          # res.sort_values(by=['Delta_Node_Degree'], ascending=True, inplace = True)
  
          res.insert(loc = 0, column = 'Merit', value = 0)
          
          res['Merit'] = 4 * abs(1 - res['mean Terminal robustness']) + 2 * abs(1 - res['mean Steiner robustness']) + abs(res['mean Terminal specificity']) + abs(res['mean Steiner specificity']) + abs(res['Delta_Num_Nodes']) + abs(res['Delta_Node_Degree']) + abs(res['Delta_RNA_Nodes']) + abs(res['Delta_DNA_Nodes'])
          
          res.sort_values(by=['Merit'], ascending=True, inplace = True)
  
          
          res.insert(loc = 0, column = 'Hyperparameters', value = res.index)
          
          res.insert(loc = 0, column = 'Phenotype', value = tuningPhenotype)
  

          res = graph_metrics.merge(res, how='left', on=['Phenotype','Hyperparameters'])
          
          
          res["noisy_edges_reps"] = params["noisy_edges_reps"]
          res["random_terminals_reps"] = params["random_terminals_reps"]
          res["edge_noise"] = params["edge_noise"]
          res["max_graph_size"] = params["max_graph_size"]
          res["min_robustness"] = params["min_robustness"]
          res["min_component_size"] = params["min_component_size"]
          res["seed"] = params["seed"]
          res = res.copy()
          
          if params["save_hyperparam_summary"]:
              fn = phenotypeName + '_' + params["fn_suffix_hyperparam_summary"] + '.csv'
              res.to_csv(path.join(pcsf_run_dir, fn), index=False)
              
          if gres == None:
              gres = res
          else:
              gres = pd.concat([gres, res])
          
  print('Completed hyperparameter search')
  print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
  print('')
  
  return gres
  
##########################################################################################
##########################################################################################
##########################################################################################

def eq_div(N, i):
    return [] if i <= 0 else [N // i + 1] * (N % i) + [N // i] * (i - N % i)
        
def perform_oiRuns(oiRun_params):

      
  graph = oiRun_params['graph']
  targetPhenotypes = oiRun_params['targetPhenotypes']
  max_num_cpus = oiRun_params['max_num_cpus']
  tuneOIruns = oiRun_params['tuneOIruns']
  verbose = oiRun_params['verbose_oi']
    


  n_request_cpus = min(max_num_cpus,len(targetPhenotypes))
      
  if n_request_cpus == 1:

      if tuneOIruns:
          _tune_and_generate_oiRuns(oiRun_params, verbose)
      else:
          _generate_oiRuns(oiRun_params, verbose)
          
  elif n_request_cpus > 1:
  
      param_sets = []
      
      bins = eq_div(len(targetPhenotypes), n_request_cpus)
      i0 = 0
      for i in range(len(bins)):
          i1 = i0 + bins[i]-1
          bin_phenotypes = targetPhenotypes[i0:(i0 + bins[i])]
          i0 += bins[i]

          graph_i = OIgraph()
          graph_i.interactome_dataframe = graph.interactome_dataframe 
          graph_i.interactome_graph = graph.interactome_graph 
          graph_i.nodes = graph.nodes 
          graph_i.edges = graph.edges 
          graph_i.edge_costs = graph.edge_costs 
          graph_i.node_degrees = graph.node_degrees 
          graph_i._reset_hyperparameters(params=oiRun_params)
          
          # make a copy of the parameter set and then overwrite the targetPhenotypes
          params_copy = oiRun_params.copy()
          params_copy = { **params_copy, 'graph': graph_i}
          params_copy = { **params_copy, 'targetPhenotypes': bin_phenotypes}
          param_sets.append(params_copy)


      print("num OIruns for multiprocessing:", len(targetPhenotypes),"cpu processors requested:", n_request_cpus)
      pool = multiprocessing.Pool(n_request_cpus)
      if tuneOIruns:
          results = pool.map(_tune_and_generate_oiRuns, param_sets)
      else:
          results = pool.map(_generate_oiRuns, param_sets)

      # print(results)
  

########################################################################

def tuneOI_Rootfun(x, xn, params):


    params = { **params, xn: x}
    
    results = _generate_oiRuns(params, version = 1)
    
    nodes = results['nodes']
    if len(nodes) > 0:
        numTerminalNodes = len(nodes[nodes['nodeType'] == 'Terminal'])
        numSteinerNodes = len(nodes[nodes['nodeType'] == 'Steiner'])
        numNodes = numTerminalNodes + numSteinerNodes
    else:
        numTerminalNodes = 0
        numSteinerNodes = 0
        numNodes = 0
    
    delta = numTerminalNodes - params['targetNumTerminalNodes']

    
    # print('(' + xn + '):', x, 'numNodes',  numNodes, 'numSteinerNodes', numSteinerNodes, 'numTerminalNodes:', numTerminalNodes, 'targetNumTerminalNodes', params['targetNumTerminalNodes'])

    return delta


########################################################################

def tuneOI_Minimizefun(x, xn, params):

    params = { **params, xn: x}

    print('===================================================')
    print('(' + xn + '):', x)

    RUN_PARAM = "W_{:04.2f}_B_{:04.2f}_G_{:04.2f}_R_{:04.2f}".format(params['w'], params['b'], params['g'], params['r'])

    results = _generate_oiRuns(params, version = 1)
    graph_metrics = results['graph_metrics']
    if len(graph_metrics) > 0:
        average_shortest_path_length = list(graph_metrics['average_shortest_path_length'])[0]
    else:
        average_shortest_path_length = 0

    print('average_shortest_path_length:', average_shortest_path_length, RUN_PARAM)

    return -average_shortest_path_length
  
##################################################################

def _tune_and_generate_oiRuns(params, verbose = False):

    targetPhenotypes = params["targetPhenotypes"]

    num_reps = params['num_reps']
    noisy_edges_reps = params['noisy_edges_reps']
    random_terminals_reps = params['random_terminals_reps']
    num_tuning_sampling_reps = params['num_tuning_sampling_reps']
    tune_R = params["tune_R"]
    
    _w = params['w']
    _b = params['b']
    _g = params['g']
    _r = params['r']
        
    _Ws = params['Ws']
    _Bs = params['Bs']
    _Gs = params['Bs']
    _Rs = params['Rs']
    
    numOIruns = 0
    
    for targetPhenotype in targetPhenotypes:
    
        numOIruns = numOIruns + 1
          

        xn = 'b' 
        xTolerance = 0.01  # Note that absolute(a - b) <= (xtol + rtol * absolute(b))
 
        
        lowerBound = min(_Bs)
        uppperBound = max(_Bs)
        
        params = { **params, 'phenotype': targetPhenotype, 'xn': xn, 'xTolerance': xTolerance}

        params = { **params, 'targetPhenotypes': [targetPhenotype]}

        if params['noisy_edges_reps'] != num_tuning_sampling_reps:
            # to save computational time, the number of noisy_edges_reps and random_terminals_reps can be reduced by a factor of 10 by setting the parameter num_tuning_sampling_reps
            # However the number of targeted terminal nodes achieved in tunining may not be achieved in the final evaluation.
            # In some low frequency cases, the final evaluation may fail to return a graph becuase of this offset between tuning and final solution.
            params = { **params, 'num_reps': num_tuning_sampling_reps, 'noisy_edges_reps': num_tuning_sampling_reps, 'random_terminals_reps' : num_tuning_sampling_reps}

        
        if tune_R:

            params = { **params, 'targetNodeSpecies': ['Metabolite','Protein']}
            params = { **params, 'w': _w, 'b': _b, 'g': _g, 'r': _r}

            try:
              root = optimize.brentq(tuneOI_Rootfun, lowerBound, uppperBound, args=(xn, params), xtol=xTolerance, rtol=1e-8, maxiter=25, full_output=False, disp=True)
            except:
              # typical failure is due not having an upper bound. The upper bound b value does not generate more nodes than the target number.
              root = uppperBound
              
            b_metabolite_protein = root


            params = { **params, 'targetNodeSpecies': ['RNA','DNA']}
            params = { **params, 'w': _w, 'b': _b, 'g': _g, 'r': _r}

            try:
              root = optimize.brentq(tuneOI_Rootfun, lowerBound, uppperBound, args=(xn, params), xtol=xTolerance, rtol=1e-8, maxiter=25, full_output=False, disp=True)
            except:
              # typical failure is due not having an upper bound. The upper bound b value does not generate more nodes than the target number.
              root = uppperBound
              
            b_rna_dna = root

            w = _w
            b = _b
            g = _g
            r = round((b_rna_dna / b_metabolite_protein), 2) 
 
        else:
          
            w = _w
            b = _b
            g = _g
            r = _r
      
      
        params = { **params, 'targetNodeSpecies': []}
        params = { **params, 'w': w, 'b': b, 'g': g, 'r': r}

        try:
          root = optimize.brentq(tuneOI_Rootfun, lowerBound, uppperBound, args=(xn, params), xtol=xTolerance, rtol=1e-8, maxiter=25, full_output=False, disp=True)
        except:
          # typical failure is due not having an upper bound. The upper bound b value does not generate more nodes than the target number.
          root = uppperBound
          
        b = round((root + xTolerance), 2)

        # print('Tuned b:', root, b)

    
        params = { **params, 'w': w, 'b': b, 'g': g, 'r' : r}
    
        params = { **params, 'num_reps': num_reps, 'noisy_edges_reps': noisy_edges_reps, 'random_terminals_reps' : random_terminals_reps}
    
        results = _generate_oiRuns(params)  # note that this will save out a result

        
        nodes = results['nodes']
        if (len(nodes) == 0) and (num_reps != num_tuning_sampling_reps):
    
            # Although rare, there are occurances when tuning with fewer sampling reps to  produces a tuned 'b' value that generates
            # the targeted number of terminal nodes but when using the full number of reps zero terminal nodes will be generated.
    
            # delete the previous _generate_oiRuns generated result files
            
            pcsf_run_dir = params["pcsf_run_dir"]
            fnPrefix = results['fnPrefix']
            file_path = path.join(pcsf_run_dir, fnPrefix + '_edges.csv')
            if os.path.exists(file_path):
                os.remove(file_path)
            file_path = path.join(pcsf_run_dir, fnPrefix + '_nodes.csv')
            if os.path.exists(file_path):
                os.remove(file_path)
            
            bMult = 1.05
        
            numTerminalNodes = 0
            trial_Bs = [b]
            trial_NumTerminalNodes = [numTerminalNodes]
            if verbose:
                print(b, numTerminalNodes)
            while (numTerminalNodes < params['targetNumTerminalNodes']) and (bMult * b <= max(_Bs)):
                b = bMult * b
                b = round(b, 2)
                params = { **params, 'w': w, 'b': b, 'g': g, 'r' : r}
                results = _generate_oiRuns(params, version = 1)
                nodes = results['nodes']
                numTerminalNodes = len(nodes)
                trial_Bs += [b]
                trial_NumTerminalNodes += [numTerminalNodes]
                if verbose:
                    print(b, numTerminalNodes)
        
            numTrials = len(trial_Bs)
            if trial_NumTerminalNodes[numTrials-1] <= params['targetNumTerminalNodes']:
                b = trial_Bs[numTrials-1]
                if verbose:
                    print('Could not bound number of terminals')
            else:
            
                delta = numTerminalNodes - params['targetNumTerminalNodes']
                lower_bound_b = trial_Bs[numTrials-2]
                upper_bound_b = trial_Bs[numTrials-1]
                lower_bound_NumTerminalNodes = trial_NumTerminalNodes[numTrials-2]
                upper_bound_NumTerminalNodes = trial_NumTerminalNodes[numTrials-1]
                while (abs(delta) > 0) and (upper_bound_b - lower_bound_b > 0.011):
                    b = (upper_bound_b + lower_bound_b) / 2
                    b = round(b, 2)
            
                    params = { **params, 'w': w, 'b': b, 'g': g, 'r' : r}
                    results = _generate_oiRuns(params, version = 1)
                    nodes = results['nodes']
                    numTerminalNodes = len(nodes)
            
                    trial_Bs += [b]
                    trial_NumTerminalNodes += [numTerminalNodes]
                    if verbose:
                        print(b, numTerminalNodes)
                    
                    delta = numTerminalNodes - params['targetNumTerminalNodes']
            
                    if delta < 0:
                        lower_bound_b = b
                        lower_bound_NumTerminalNodes = numTerminalNodes
                    else:
                        upper_bound_b = b
                        upper_bound_NumTerminalNodes = numTerminalNodes
        
                
                if (lower_bound_NumTerminalNodes > 0) and (abs(lower_bound_NumTerminalNodes - params['targetNumTerminalNodes']) < abs(upper_bound_NumTerminalNodes - params['targetNumTerminalNodes'])):
                    b = lower_bound_b
                else:
                    b = upper_bound_b

            
            params = { **params, 'w': w, 'b': b, 'g': g, 'r' : r}
        
            results = _generate_oiRuns(params)

      
    return numOIruns


##################################################################
  
  
  
def _generate_oiRuns(params, version = 0, verbose = False):
  
  graph = params["graph"]
  source_interactome = params["source_interactome"]
  targetPhenotypes = params["targetPhenotypes"]
  targetPipelines = params["targetPipelines"]
  targetNodeSpecies = params["targetNodeSpecies"]
  pcsf_run_dir = params["pcsf_run_dir"]
  prizes_all_fp  = params["prizes_all_fp"]
  num_reps = params["num_reps"]
  Ws = [params["w"]]
  Bs = [params["b"]]
  Gs = [params["g"]]
  Rs = [params["r"]]
  noisy_edges_reps = params["noisy_edges_reps"]
  random_terminals_reps = params["random_terminals_reps"]
  edge_noise  = params["edge_noise"]
  
  maxNumNodesToComputeBetweenness = params["maxNumNodesToComputeBetweenness"]
  seed = params["seed"]
  np.random.seed(seed=params["seed"])

  max_num_cpus = params["max_num_cpus"]
  
  maxNumNodesToComputeAvgPath = params["maxNumNodesToComputeAvgPath"]

  fnPrefix = ''
  nodes = pd.DataFrame()
  edges = pd.DataFrame()
  graph_metrics = pd.DataFrame()
    
  
  numOIruns = 0
  # targetPhenotype = targetPhenotypes[0]
  for targetPhenotype in targetPhenotypes :
      
      numOIruns = numOIruns + 1
  
      if verbose:
          print('')
          print(numOIruns, ' of ', len(targetPhenotypes))
          print(targetPhenotype)
          print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
      
      phenotypeName = targetPhenotype.replace(' ','_').replace('.','_')
      # prize_fp = path.join(pcsf_run_dir, (phenotypeName + '_' + 'NodeMaxPrizes.zip')) # save as zip file
      prize_fp = path.join(pcsf_run_dir, (phenotypeName + '_' + 'NodeMaxPrizes.csv'))
      
      
      prizeData = pd.read_csv(prizes_all_fp)
      prizeData.rename(columns = {'ppmi_node_id':'name'}, inplace = True) 

      rpd = prizeData.copy()
      rpd['prize'] = abs(rpd['norm_prize'])
      rpd.sort_values(by=['prize'], ascending=False, inplace = True)
      rpd.drop_duplicates(subset=['Alt1RefId'], inplace = True)
      rpd = set(rpd['Alt1RefId'])
      allNodeInfo = source_interactome.nodeInfo_metabolite.copy()
      allNodeInfo = pd.concat([allNodeInfo[allNodeInfo['Alt1RefId'].isin(rpd)], allNodeInfo[~allNodeInfo['Alt1RefId'].isin(rpd)]])  
      allNodeInfo = pd.concat([allNodeInfo, source_interactome.nodeInfo_protein])
      allNodeInfo = pd.concat([allNodeInfo, source_interactome.nodeInfo_transcript])
      allNodeInfo = pd.concat([allNodeInfo, source_interactome.nodeInfo_gene])
      if params['use_MicroRNA']:
        allNodeInfo = pd.concat([allNodeInfo, source_interactome.mature_mir_nodeInfo])    
        allNodeInfo = pd.concat([allNodeInfo, source_interactome.mature_mir_transcript_nodeInfo])    
        allNodeInfo = pd.concat([allNodeInfo, source_interactome.mature_mir_gene_nodeInfo])    
      allNodeInfo.drop_duplicates(subset=['nodeName'], inplace = True)
      allNodeInfo.reset_index(drop=True, inplace = True)

      if (len(targetPipelines) > 0):
          prizeData = prizeData.loc[prizeData['Pipeline'].isin(targetPipelines)].copy()
      
      if (len(targetNodeSpecies) > 0):
          prizeData = prizeData.loc[prizeData['nodeSpecies'].isin(targetNodeSpecies)].copy()
          
      prizeData = prizeData.loc[prizeData['prizeName'] == targetPhenotype].copy()
      
      
      
      prizeData['prize'] = abs(prizeData['norm_prize'])
      prizeData['prizeSigned'] = prizeData['norm_prize']
      
      # filtering by p-value 
      prizeData = prizeData[prizeData['pvalue'] <= params['prize_critical_p_value']]
  
      # Sort prizes by decending order and only keep highest one
      prizeData.sort_values(by=['prize'], ascending=False, inplace = True)
      prizeData.drop_duplicates(subset=['name'], inplace = True)
      prizeData.reset_index(drop=True, inplace = True)
  
      if len(prizeData.index) < 2:
          if verbose:
              print('Skip due to less than 2 nodes with prizes found.')
              print('')
          continue
          
      if verbose:
          print('Total number node prizes:  ' + str(len(prizeData.index)))
          print('Median prize:', prizeData['prize'].median())
  
  
      # cns = prizeData.columns.tolist()
      # cns.remove('prize')
      # cns.insert(0,'prize')
      # cns.remove('name')
      # cns.insert(0,'name')
      # prizeData = prizeData[cns]
      
      prizeData['species'] = ''
      prizeData.loc[prizeData['nodeSpecies'] == 'RNA', 'species'] = 'R'
      prizeData.loc[prizeData['nodeSpecies'] == 'DNA', 'species'] = 'D'
      prizeData.loc[prizeData['nodeSpecies'] == 'Metabolite', 'species'] = 'M'
      prizeData.loc[prizeData['nodeSpecies'] == 'Protein', 'species'] = 'P'
      
      cns = prizeData.columns.tolist()
      pcns = ['name', 'prize', 'species']
      cns = pcns + [cn for cn in cns if cn not in pcns ]
      prizeData = prizeData[cns]
        
      if len(params["removePrizeNodeNames"]) > 0:
          prizeData = prizeData[~prizeData['name'].isin(params["removePrizeNodeNames"])]
      
      prizeData.set_index('name', inplace = True) 
      prizeData.to_csv(prize_fp,index=True) # Note it is important to include index

      results = graph.grid_randomization(prize_fp, Ws=Ws, Bs=Bs, Gs=Gs, Rs=Rs, noisy_edges_reps=noisy_edges_reps, random_terminals_reps=random_terminals_reps, edge_noise=edge_noise, seed=seed, max_num_cpus=max_num_cpus)
  
      # remove the prize file for this OI run
      # os.remove(prize_fp)
      
      
      # # note that augmented_forest may contain edges that were not in the pcsf solutions.
      data = []
      for paramstring, forests in results.items(): 
          if forests['augmented_forest'].number_of_nodes() < params["min_component_size"]:
              forests['robust'] = nx.empty_graph(0)
              continue
            
          # meet both criteria a robustness > min_robustness and a robustness > than the robustness at the given max_graph_size
          robustness_at_max_graph_size = get_robust_subgraph_Robustness(forests['augmented_forest'], max_size=params["max_graph_size"])
          min_robustness = max(robustness_at_max_graph_size, params["min_robustness"])
          
          forests['robust'] = get_robust_subgraph_from_randomizations(forests['augmented_forest'], min_robustness = min_robustness, max_size=np.Infinity, min_component_size=params["min_component_size"])

          if ((forests['robust'].number_of_nodes() > 0) and (forests['robust'].number_of_nodes() <= maxNumNodesToComputeBetweenness)):
              betweenness(forests['robust'])
              
          louvain_clustering(forests['robust'])
              
          avgShortestPathlength = np.nan
          if forests['robust'].number_of_nodes() >= params["min_component_size"]:
              G = forests['robust'].subgraph(max(nx.connected_components(forests['robust']),key=len))
              # computation of average_shortest_path_length of a graph of 4000 nodes requires ~ 1 minute of computational time, 6700 nodes requires ~ 6.3 minutes
              if G.number_of_nodes() <= maxNumNodesToComputeAvgPath:
                  avgShortestPathlength = nx.average_shortest_path_length(G)

              
              
          vs = str(paramstring).split('_')   # for example ['W', '1.00', 'B', '16.00', 'G', '5.00', 'R', '0.25'] 
          w = vs[1]
          b = vs[3]
          g = vs[5]
          r = vs[7]
          
          data.append([targetPhenotype, (str(paramstring) + '_N_' + str(num_reps)), w, b, g, r, num_reps, avgShortestPathlength])
 
      graph_metrics = pd.DataFrame(data, columns=['Phenotype','Hyperparameters','W','B','G','R','num_randomizations', 'average_shortest_path_length'])

      for parameter_set in list(results):
          G = results[parameter_set]['robust']  # 'robust' 'forest' or  'augmented_forest'
          parameter_set = parameter_set + '_N_' + str(num_reps) 
          if verbose:
              print(parameter_set)
          if G.number_of_nodes() < params["min_component_size"]:
              if verbose:
                  print(parameter_set, ' generated an empty graph!')
              if version == 0:
                  continue
              else:
                  return {'nodes':pd.DataFrame(), 'edges':pd.DataFrame(), 'graph_metrics':pd.DataFrame(), 'fnPrefix':fnPrefix}
              
  
          nodes = get_networkx_graph_as_dataframe_of_nodes(G)
      
          edges = get_networkx_graph_as_dataframe_of_edges(G)


          acns = edges.columns
          
          if not all(x in acns for x in ['nodeName1','nodeName2','cost','in_solution']):
              return {'nodes':pd.DataFrame(), 'edges':pd.DataFrame(), 'graph_metrics':pd.DataFrame(), 'fnPrefix':fnPrefix}
          
          acns = [cn for cn in acns if cn not in ['nodeName1','nodeName2','cost','in_solution']]

          # # Following fills in the missing edge cost values, fills in the NAs for the in_solution, and reorders the nodeName1 < nodeName2
          edges.rename(columns = {'nodeName1':'id1'}, inplace = True) 
          edges.rename(columns = {'nodeName2':'id2'}, inplace = True) 
          df12 = graph.interactome_dataframe.merge(edges, left_on=['nodeName1','nodeName2'], right_on = ['id1','id2'], how='inner')
          df21 = graph.interactome_dataframe.merge(edges, left_on=['nodeName1','nodeName2'], right_on = ['id2','id1'], how='inner')
          df12 = df12[['nodeName1',	'nodeName2','cost_y','in_solution']+acns].copy()
          df21 = df21[['nodeName1',	'nodeName2','cost_y','in_solution']+acns].copy()
          df = pd.concat([df12, df21])
          df = df.reset_index(drop=True)
          df.rename(columns = {'cost_y':'cost'}, inplace = True) 
          df['in_solution'].fillna(False, inplace=True)
          for i in range(len(df)):
              if df.loc[i, 'nodeName1'] > df.loc[i, 'nodeName2']:
                  temp = df.loc[i, 'nodeName1']
                  df.loc[i, 'nodeName1'] = df.loc[i, 'nodeName2']
                  df.loc[i, 'nodeName2'] = temp
      
          df.sort_values(['nodeName1', 'nodeName2'], ascending=[True, True], inplace=True)
          df = df.reset_index(drop=True)
          edges = df
      
      
      
          # get the unique nodeNames from the edges that are in the subnetwork solution
          nodeSet = set(edges['nodeName1'].unique().tolist() + edges['nodeName2'].unique().tolist())
          
          # Filter the nodes and keep only those in the subnetwork solution
          nodes['nodeName'] = list(nodes.index)
          first_column = nodes.pop('nodeName') 
          nodes.insert(0, 'nodeName', first_column) 
          nodes = nodes[nodes['nodeName'].isin(nodeSet)].copy()
          
          # This should never occur if the the oi code is working properly
          nodeSetCheck = set(nodes['nodeName'].unique().tolist())
          if nodeSet != nodeSetCheck:
              if verbose:
                  print('WARNING Not all of the nodes used in the subnetwork solution were found in the nodes dataframe!')
          
          nodes['nodeType'] = 'Steiner'
          nodes.loc[nodes['terminal'] == True, 'nodeType'] = 'Terminal'
          
          nodes.loc[(nodes['nodeName'].str.startswith('ENSG', na=False)),'type']='DNA' 
          nodes.loc[(nodes['nodeName'].str.startswith('ENST', na=False)),'type']='RNA'
          nodes.loc[(nodes['nodeName'].str.startswith('ENSP', na=False)),'type']='Protein' 
          nodes.loc[(nodes['nodeName'].str.startswith('CID', na=False)),'type']='Metabolite'
          nodes.loc[(nodes['nodeName'].str.startswith('hsa-miR-', na=False)),'type']='RNA'
          nodes.loc[(nodes['nodeName'].str.startswith('hsa-let-', na=False)),'type']='RNA'
          
          
          # get node Info based on nodeName, 
          # Important : more than one metabolite or protein can map to the same nodeName
          nodeInfo = pd.DataFrame(list(nodeSet), columns=['nodeName'])
          nodeInfo = pd.merge(nodeInfo, allNodeInfo, on='nodeName', how='left')
          nodeInfo.drop_duplicates(inplace = True)
          nodeInfo = nodeInfo[~nodeInfo['nodeName'].isnull()]
          nodeInfo.reset_index(drop=True, inplace = True)
      
          # For the Steiner nodes, fill in the missing info using the nodeInfo
          for index, row in nodes.iterrows():
              if row['terminal'] == False:   # do not want to overwrite the prize terminal nodes
                  selInfo = nodeInfo.loc[nodeInfo['nodeName'] == row['nodeName']]
                  # What should be done if there are multiple mappings????????????????????????????????????????
                  for index2, row2 in selInfo.iterrows():
                      nodes.at[index, 'nodeSpecies'] = row['type']
                      nodes.at[index, 'nodeLabel'] = row2['nodeLabel']
                      nodes.at[index, 'nodeTitle'] = row2['nodeTitle']
                      nodes.at[index, 'Alt1RefId'] = row2['Alt1RefId']
                      nodes.at[index, 'Alt2RefId'] = row2['Alt2RefId']
                      nodes.at[index, 'Alt3RefId'] = row2['Alt3RefId']
                      break  # Currently keep the first mapping????????????????????????????????????????
      
                      
          cns = nodes.columns.tolist()
          #JFG pcns = ['nodeName','prize','nodeType','terminal','type','degree','betweenness','louvain_clusters','robustness','specificity']
          pcns = ['nodeName','prize','nodeType','terminal','type','degree','robustness','specificity']
          cns = pcns + [cn for cn in cns if cn not in pcns ]
          nodes = nodes[cns]
          nodes.sort_values(by=['nodeName'], ascending=[True], inplace=True)

          PCSF_RUN = phenotypeName
          
          # PCSF_RUN_TEMPLATE = "W_{:04.2f}_B_{:04.2f}_G_{:04.2f}_R_{:04.2f}_N_{}".format(oi3fold['w'], oi3fold['b'], oi3fold['g'], oi3fold['r'], oi3fold['num_reps'])
          # PCSF_RUN_TEMPLATE = PCSF_RUN_TEMPLATE.replace('.','_')
          
          PCSF_RUN_TEMPLATE = parameter_set.replace(".","_")
          fnPrefix = PCSF_RUN + '_' + PCSF_RUN_TEMPLATE

          if verbose:
              print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
          if version == 0:
              edges.to_csv(path.join(pcsf_run_dir, fnPrefix + '_edges.csv'),index=False)
              nodes.to_csv(path.join(pcsf_run_dir, fnPrefix + '_nodes.csv'),index=False)
              # node_attributes_df.to_csv(path.join(pcsf_run_dir, fnPrefix + '_nodes_raw.csv'),index=True) # Note it is important to include index
          else:
              return {'nodes':nodes, 'edges':edges, 'graph_metrics':graph_metrics, 'fnPrefix':fnPrefix}

          # del nodes
          # del edges
          # del nodeInfo
          
  if version == 0:   
      if verbose:
          print('Done')
      # return nodes, edges, graph_metrics and fnPrefix from last successful generated targetPhenotype....  for targetPhenotype in targetPhenotypes :
      return {'nodes':nodes, 'edges':edges, 'graph_metrics':graph_metrics, 'fnPrefix':fnPrefix}
  else:
      return {'nodes':pd.DataFrame(), 'edges':pd.DataFrame(), 'graph_metrics':pd.DataFrame(), 'fnPrefix':fnPrefix}

##########################################################################################
##########################################################################################
##########################################################################################

def analyze_oi_runs(params):

  pcsf_run_dir = params['pcsf_run_dir']
  oi_network_dir = params['oi_network_dir']
  min_robustness = params['min_robustness']
  min_component_size = params['min_component_size']
  oirun_interactome_fp = params['oirun_interactome_fp']
  source_interactome = params['source_interactome']

  fns = os.listdir(pcsf_run_dir)
  # fns = list(filter(lambda x: x.endswith('_nodes.csv'), fns))
  fns = [x for x in fns if x.endswith('_nodes.csv')]
  fns = [x.replace(('_nodes.csv') , '') for x in fns]
  PCSF_RUNS = []
  PCSF_RUN_TEMPLATES = []
  for v in fns:
      v = v.split('_W_', 1)
      PCSF_RUNS = PCSF_RUNS + [str(v[0])]
      PCSF_RUN_TEMPLATES = PCSF_RUN_TEMPLATES + ['W_' + str(v[1])]
    
  oiRuns = {}
  for i in range(len(PCSF_RUNS)):
  #for PCSF_RUN in PCSF_RUNS:
      PCSF_RUN = str(PCSF_RUNS[i])
      PCSF_RUN_TEMPLATE = str(PCSF_RUN_TEMPLATES[i])
      
      fnPrefix = PCSF_RUN + '_' + PCSF_RUN_TEMPLATE
  
      fn = path.join(pcsf_run_dir,(fnPrefix + '_edges.csv'))
      subNetwork = pd.read_csv(fn)
      #subNetwork = pd.read_csv(fn, sep='\t', engine='python', quoting=3)
      
      if len(subNetwork.index) == 0:
          print('ignoring oi run with no edges: ' + PCSF_RUN)
          continue
  
      # print(PCSF_RUN)
      
      fn = path.join(pcsf_run_dir,(fnPrefix + '_nodes.csv'))
      nodeInfo = pd.read_csv(fn)
      # nodeInfo = pd.read_csv(fn, sep='\t', engine='python', quoting=3)
  
      nodeInfo.rename(columns = {'prize':'nodePrize'}, inplace = True) 
      
      rcns = ['Unnamed: 0.1','Unnamed: 0']
      
      cns = subNetwork.columns
      cns = [cn for cn in cns if cn not in rcns]
      subNetwork = subNetwork[cns]
  
      cns = nodeInfo.columns
      cns = [cn for cn in cns if cn not in rcns]
      nodeInfo = nodeInfo[cns]
      
      nodeSet = set(list(subNetwork['nodeName1']) + list(subNetwork['nodeName2']))
  
      cns = nodeInfo['nodeName']
      cns = [cn for cn in cns if cn not in nodeSet]
      if len(cns) > 0:
          print(cns)
          print('problem: nodeInfo contains nodes with no edges in subNetwork')
          exit()
  
      cns = nodeSet
      cns = [cn for cn in cns if cn not in set(list(nodeInfo['nodeName']))]
      if len(cns) > 0:
          print(cns)
          print('problem: subNetwork contains nodes not found in nodeInfo')
          exit()
  
      pcns = ['nodeName','nodeLabel','nodeTitle','Alt1RefId','Alt2RefId','Alt3RefId','nodeSpecies','NodeDataType','measurement_id','Pipeline']
      cns = nodeInfo.columns
      cns = pcns + [cn for cn in cns if cn not in pcns]
      nodeInfo = nodeInfo[cns]
  
      nodeInfo['nodeShape'] = nodeInfo['nodeType'] + ' ' + nodeInfo['nodeSpecies']
  
      PCSF_DIRECTION_METRIC = 'Spearman_Cor'
  
      nodeInfo.fillna({PCSF_DIRECTION_METRIC:0}, inplace=True)
      
      nodeInfo.loc[nodeInfo[PCSF_DIRECTION_METRIC] > 0,'nodeShape'] = nodeInfo.loc[nodeInfo[PCSF_DIRECTION_METRIC] > 0,'nodeShape'] + ' +'
      nodeInfo.loc[nodeInfo[PCSF_DIRECTION_METRIC] < 0,'nodeShape'] = nodeInfo.loc[nodeInfo[PCSF_DIRECTION_METRIC] < 0,'nodeShape'] + ' -'
  
      nodeInfo['prizeName'] = PCSF_RUN
      nodeInfo['PCSF_RUN_TEMPLATE'] = PCSF_RUN_TEMPLATE
      
      # use to perform louvain clustering on individual oirun
      
      
      # fn = path.join(pcsf_run_dir,(fnPrefix + '_nodes_raw.csv'))
      # nodes_raw = pd.read_csv(fn)
      
      
      # oiRuns = {**oiRuns, PCSF_RUN: {'nodeInfo': nodeInfo, 'subNetwork': subNetwork, 'nodes_raw': nodes_raw}}
      oiRuns = {**oiRuns, PCSF_RUN: {'nodeInfo': nodeInfo, 'subNetwork': subNetwork}}
  


  print('number of assembled OI runs: ', len(oiRuns))

  if len(oiRuns) == 0:
      print('No robust OI subnetworks were generated!')
      # create empty data frames and save them out
      pn_nodeInfo = pd.DataFrame(columns=['nodeName', 'nodeLabel', 'nodeTitle', 'Alt1RefId', 'Alt2RefId', 'Alt3RefId', 'nodeSpecies', 'NodeDataType', 'measurement_id', 'Pipeline', 'nodePrize', 'pvalue', 'PNM'])
      pn_subNetwork = pd.DataFrame(columns=['nodeName1', 'nodeName2', 'cost', 'weight'])
      pn_nodeInfo.to_csv(path.join(oi_network_dir,'pn_nodes.csv'),index=False)
      pn_subNetwork.to_csv(path.join(oi_network_dir, 'pn_edges.csv'),index=False)
      return 0

  # Note that this interactome is what was used to generate oi runs and thus 
  # includes any added transcript-protein and gene-transcript actions
  interactome = pd.read_csv(oirun_interactome_fp)
  # interactome = pd.read_csv(oirun_interactome_fp,sep='\t',engine='python',quoting=3)
  
  
  
  # protein_transcript_gene_links = pd.read_csv(SOURCE_FILE_protein_transcript_gene_links,sep='\t',engine='python',quoting=3)
  # protein_transcript_gene_links = source_interactome.protein_transcript_gene_links
  
  # This one includes non-coding RNA
  # transcript_gene_Links = pd.read_csv(SOURCE_FILE_transcript_gene_Links,sep='\t',engine='python',quoting=3)
  # transcript_gene_Links = source_interactome.transcript_gene_Links
  

  Union_OIruns_nodeInfo = pd.DataFrame()
  nodeRobustnessValues = []
  
  
  
  allRobustNodeInfo = pd.DataFrame()
  for PCSF_RUN in list(oiRuns):
      nodeInfo = oiRuns[PCSF_RUN]['nodeInfo'] 
      # subNetwork = oiRuns[PCSF_RUN]['subNetwork']
  
      robustNodeInfo = nodeInfo[nodeInfo['robustness'] >= min_robustness].copy()
      if len(robustNodeInfo) > 0:
          if len(allRobustNodeInfo) == 0:
              allRobustNodeInfo = robustNodeInfo
          else:
              allRobustNodeInfo = pd.concat([allRobustNodeInfo, robustNodeInfo])
  
  
  # INFORMATION distinct measurment_id each highly correlated with different phenotypes could be linked to the same nodeName within the union of phenotype networks
  
  # sort to the top the terminal nodes with the lowest p-value measurement_id
  allRobustNodeInfo = allRobustNodeInfo.sort_values(by=['nodeType','pvalue'], ascending=[False,True])  #  Terminal above Steiner
  robustNodeInfo = allRobustNodeInfo.drop_duplicates(subset=['nodeName'])
  
  robustNodes = robustNodeInfo['nodeName']
  subNetwork = interactome[(interactome['nodeName1'].isin(robustNodes) & interactome['nodeName2'].isin(robustNodes))].copy()
  
  
  # DiGraph.add_weighted_edges_from(ebunch, weight='weight', **attr
  subNetwork['weight'] = 1.5 - subNetwork['cost']
  subNetwork.loc[subNetwork['weight'] < 0,'weight'] = 0
  
#########################################################################################

  # nxgraph = nx.Graph()
  # nxgraph.add_weighted_edges_from(subNetwork.to_records(index=False),weight='weight')
  acns = [cn for cn in subNetwork.columns if cn not in ['nodeName1','nodeName2']]
  nxgraph = nx.from_pandas_edgelist(subNetwork, 'nodeName1', 'nodeName2', edge_attr=acns)

  print('numNodes', len(nxgraph), 'numEdges', len(nxgraph.edges), 'in union')
  nNodes = len(nxgraph)
  nEdges = len(nxgraph.edges)
  
  import math
  
  crit_min_size = max(min_component_size,math.floor(float(len(nxgraph)) * 0.02))
  
  nxgraph = filter_graph_by_component_size(nxgraph, min_size=crit_min_size)
  #### Only retain largest connected component removing any isolated nodes
  # nxgraph = nxgraph.subgraph(max(nx.connected_components(nxgraph),key=len))

  if (nNodes != len(nxgraph)) or (nEdges != len(nxgraph.edges)):
      print('numNodes', len(nxgraph), 'numEdges', len(nxgraph.edges), 'after filtering out small components')
  
  subNetwork = pd.DataFrame([{'nodeName1':u,'nodeName2':v,'cost':w} for u,v,w in nxgraph.edges(data='cost')])
  
  # DiGraph.add_weighted_edges_from(ebunch, weight='weight', **attr
  subNetwork['weight'] = 1.5 - subNetwork['cost']
  subNetwork.loc[subNetwork['weight'] < 0,'weight'] = 0
#######################################################################################
  
  if (params['use_MicroRNA']):
  # special code use

      subModNetwork = subNetwork.copy()

      # remove edges
      # REMOVE BEFORE CLUSTERING the edges of mature miRNA binding to mRNA transcripts
      hsa2rna_edges = source_interactome.interactome_mature_mir_transcript.copy()
      hsa2rna_edges = hsa2rna_edges.drop_duplicates()
      hsa2rna_edges['edgeName'] = hsa2rna_edges['nodeName1'] + '_' + hsa2rna_edges['nodeName2']

      subModNetwork['edgeName'] = subModNetwork['nodeName1'] + '_' + subModNetwork['nodeName2']
      print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
      print('Number of Edges:',len(subModNetwork))

      if params['reduce_MicroRNA_PNM_Weighting']:
          # reduce weight of edge for mature microRNA to its targeted Transcript
          subModNetwork.loc[subModNetwork['edgeName'].isin(hsa2rna_edges['edgeName']),'weight'] = 0.000001
      
      # subModNetwork = subModNetwork[~subModNetwork['edgeName'].isin(hsa2rna_edges['edgeName'])].copy()

      subModNetwork = subModNetwork.drop(columns=['edgeName'])
      print('Number of Edges:',len(subModNetwork))
      print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

      acns = [cn for cn in subModNetwork.columns if cn not in ['nodeName1','nodeName2']]
      nxgraph = nx.from_pandas_edgelist(subModNetwork, 'nodeName1', 'nodeName2', edge_attr=acns)
      
  
  
  
  
  # Now remove any nodes that are not part of the PN with maximum connected_components
  numNodes = len(robustNodeInfo)
  robustNodes = set(list(subNetwork['nodeName1']) + list(subNetwork['nodeName2']))
  allRobustNodeInfo = allRobustNodeInfo[allRobustNodeInfo['nodeName'].isin(robustNodes)]
  robustNodeInfo = robustNodeInfo[robustNodeInfo['nodeName'].isin(robustNodes)]
  
  if len(robustNodeInfo) < numNodes:
      print('Number of isolated nodes removed', numNodes - len(robustNodeInfo))


  
  pn_subNetwork = subNetwork.copy()
  pn_nodeInfo = robustNodeInfo.copy()
  pn_allNodeNameInfo = allRobustNodeInfo.copy()
  
  print('Generated PN')
  
  # PN edges are in subNetwork
  # PN information is in pn_nodeInfo and pn_allNodeNameInfo
  # note that the robusness and other attributes is prize specific - see columns  'prizeName' and 'PCSF_RUN_TEMPLATE' within the nodeInfo
  

  # now perform clustering to identify PNMs
  
  import math
  
  def shannonIndex(x):
      v = list(x.value_counts(dropna=True))
      s = sum(x.astype('int'))
      if s == 0:
          return 0
      else:
          p = [i / s for i in v]
          d = [i * math.log(i) for i in p]
          return(-sum(d))
  
  def heterogenityIndex(x):
      v = list(x.value_counts(dropna=True))
      s = sum(x.astype('int'))
      if s == 0:
          return 0
      else:
          p = [i / s for i in v]
          d = [i * math.log(i) for i in p]
          return(-sum(d) / math.log(len(d)))
  
  
  # np.random.seed(1)
  # nx.set_node_attributes(nxgraph, {node: {'louvain_clusters':str(cluster)} for node,cluster in community.best_partition(nxgraph, weight='weight').items()})
  # nodes = pd.DataFrame.from_dict(dict(nxgraph.nodes(data=True)), orient='index')
  # nodes.head()
  
  NUM_CLUSTERING_CYCLES = 100
  min_cluster_size = 100000
  numClusterIDs = []
  heterogenity = []
  mcolnames = []
  m = pd.DataFrame()
  for i in range(NUM_CLUSTERING_CYCLES):
      np.random.seed(i+1)
      nx.set_node_attributes(nxgraph, {node: {'louvain_clusters':str(cluster)} for node,cluster in community.best_partition(nxgraph, weight='weight').items()})
      nodes = pd.DataFrame.from_dict(dict(nxgraph.nodes(data=True)), orient='index')
      clusters = set(nodes['louvain_clusters'])
      if min_cluster_size > len(clusters):
          min_i = i
          min_cluster_size = len(clusters)
      cn = 'c' + str(i+1)
      mcolnames.append(cn)
      nodes.rename(columns = {'louvain_clusters':cn}, inplace = True) 
      if len(m) == 0:
          m = nodes
      else:
          m = pd.merge(m, nodes, left_index=True, right_index=True)
      m = m.copy()
      numClusterIDs.append(len(set(m[cn])))
      heterogenity.append(heterogenityIndex(m[cn]))
      
      if i % 10 == 0:
          print('Optimizing clusters: ', i, ' of ', NUM_CLUSTERING_CYCLES, ' Current size: ',  min_cluster_size)
  
  m['nodeName'] = m.index
  
  d = pd.DataFrame(mcolnames, columns=['ci'])
  d['numClusterIDs'] = numClusterIDs
  d['heterogenity'] = heterogenity
  d = d.sort_values(by=['numClusterIDs','heterogenity'], ascending=[True,False])
  pci = [list(d['ci'])[0]]
  oci = [cn for cn in list(d['ci']) if cn not in pci]
  
  
  mm = m[['nodeName'] + pci + oci]
  xm = mm
  # # map oci clusters to pci cluster
  for ti in oci:
      tab = mm.groupby(pci + [ti]).size()
      nt = pd.DataFrame(tab)
      nt = nt.reset_index() 
      nt.rename(columns = {0:'count'}, inplace = True) 
      nt = nt.sort_values(by=['count'], ascending=[False])
      nt = nt[nt['count'] != 0]
      nt = nt.drop_duplicates(subset=[ti])
      nt.rename(columns = {pci[0]:(ti + '.new')}, inplace = True) 
      xm = xm.merge(nt[[ti,(ti + '.new')]], how='left', on = ti)
      xm.rename(columns = {ti:(ti + '.old')}, inplace = True) 
      
  ocin = [(cn + '.new') for cn in oci]
  xm = xm[['nodeName'] + pci + ocin]
  cns = xm.columns
  cns = [cn.replace('.new','') for cn in cns]
  xm.columns = cns
  
  
  print('Computing robust louvain clusters')
  
  # now for each nodeName look at its frequency of membership across all of the clustering runs
  
  for ri in range(len(xm)):
      nt = pd.DataFrame(list(xm.loc[ri][pci + oci]), columns=['cluster'])
      nt = nt.value_counts(dropna=True)
      nt = pd.DataFrame(nt)
      nt = nt.reset_index() 
      nt.rename(columns = {0:'count'}, inplace = True) #  bug fix
      nt = nt.sort_values(by=['count'], ascending=[False])
      
      xm.at[ri,'PNM']  = int(nt['cluster'][0])
      xm.at[ri,'PNM_Assignment_Robustness']  = float(nt['count'][0]) / sum(nt['count'].astype('int'))
      xm.at[ri ,'PNM_NumberOf_Assignments'] = len(nt)
  
  # The Prinicipal Network Module ID starts with 1 (not zero).
  xm= xm.astype({'PNM': int})
  xm['PNM'] += 1 
  
  xm= xm.astype({'PNM_NumberOf_Assignments': int})
  
  # re-order numbering of PNMs by size of the PNM (i.e how many nodes belong to the PNM)
  tab = xm.groupby(['PNM']).size()
  nt = pd.DataFrame(tab)
  nt = nt.reset_index() 
  nt.rename(columns = {0:'count'}, inplace = True) 
  nt = nt.sort_values(by=['count'], ascending=[False])
  nt['newPNM'] = range(len(nt))
  nt['newPNM'] += 1
  xm = xm.merge(nt, how='left', on = 'PNM')
  xm['PNM'] = xm['newPNM']
  
  
  pnms = xm[['nodeName','PNM','PNM_Assignment_Robustness','PNM_NumberOf_Assignments']]
  print('Number of robust louvain clusters:',len(set(pnms['PNM'])))
  
  tab = pnms.groupby(['PNM']).size()
  nt = pd.DataFrame(tab)
  nt = nt.reset_index() 
  nt.rename(columns = {0:'nodeCount'}, inplace = True) 
  nt = nt.sort_values(by=['nodeCount'], ascending=[False])
  nt.head(100)
  
  

  pn_allNodeNameInfo = pn_allNodeNameInfo.merge(pnms, how='left', on = 'nodeName')
  pn_nodeInfo = pn_nodeInfo.merge(pnms, how='left', on = 'nodeName')

  # remove this since it provides no value at this point onward
  pn_allNodeNameInfo = pn_allNodeNameInfo.drop('louvain_clusters', axis=1)
  pn_nodeInfo = pn_nodeInfo.drop('louvain_clusters', axis=1)

  # Update all of the indivudal runs assigning the PNM, note some nodes may end up with an PNM of NaN
  oiRunsUpdated = {}
  
  for PCSF_RUN in list(oiRuns):
      # nodes_raw = oiRuns[PCSF_RUN]['nodes_raw'] 
      nodeInfo = oiRuns[PCSF_RUN]['nodeInfo'] 
      subNetwork = oiRuns[PCSF_RUN]['subNetwork']
      nodeInfo = nodeInfo.merge(pnms[['nodeName','PNM']], how='left', on = 'nodeName')
      # oiRunsUpdated = {**oiRunsUpdated, PCSF_RUN: {'nodeInfo': nodeInfo, 'subNetwork': subNetwork, 'nodes_raw': nodes_raw}}
      oiRunsUpdated = {**oiRunsUpdated, PCSF_RUN: {'nodeInfo': nodeInfo, 'subNetwork': subNetwork}}
  
  oiRuns = oiRunsUpdated
  
  # This file can have more rows than pn_nodeInfo because of multiple measurement_id (s) for the same nodeName
  pn_allNodeNameInfo.to_csv(path.join(oi_network_dir,'pn_allNodeNameInfo.csv'),index=False)
  pn_nodeInfo.to_csv(path.join(oi_network_dir,'pn_nodes.csv'),index=False)
  pn_subNetwork.to_csv(path.join(oi_network_dir, 'pn_edges.csv'),index=False)


    
  output_file_list = []
  for PCSF_RUN in list(oiRuns):
      nodeInfo = oiRuns[PCSF_RUN]['nodeInfo'] 
      subNetwork = oiRuns[PCSF_RUN]['subNetwork']
      # nodes_raw = oiRuns[PCSF_RUN]['nodes_raw']  
      
      # fn = path.join(oi_network_dir,(PCSF_RUN + '_nodes_raw.csv'))
      # output_file_list = output_file_list + [fn]
      # nodes_raw.to_csv(fn,index=False)
      
      fn = path.join(oi_network_dir,(PCSF_RUN + '_nodes.csv'))
      output_file_list = output_file_list + [fn]
      nodeInfo.to_csv(fn,index=False)
  
      fn = path.join(oi_network_dir,(PCSF_RUN + '_edges.csv'))
      output_file_list = output_file_list + [fn]
      subNetwork.to_csv(fn,index=False)
  
  
  if len(output_file_list) > 0:
      import zipfile
      # (PCSF_RUN_TEMPLATE + '_' +'oiRuns.zip')
      with zipfile.ZipFile(os.path.join(oi_network_dir, 'oiRuns.zip'), 'w') as zipMe:        
        for fn in output_file_list:
            zipMe.write(fn, arcname=os.path.basename(os.path.normpath(fn)), compress_type=zipfile.ZIP_DEFLATED)
            
      for fn in output_file_list:
          os.remove(fn)
  
  
  fns = os.listdir(pcsf_run_dir)
  # fns1 = list(filter(lambda x: x.endswith(('_hyperparam_results.csv')), fns))
  # fns2 = list(filter(lambda x: x.endswith(('_hyperparam_summary.csv')), fns))
  fns1 = [x for x in fns if x.endswith('_hyperparam_results.csv')]
  fns2 = [x for x in fns if x.endswith('_hyperparam_summary.csv')]

  if ((len(fns1) > 0) | (len(fns2) > 0)):
    import zipfile
    with zipfile.ZipFile(os.path.join(oi_network_dir, 'hyperparameters.zip'), 'w') as zipMe:        
      for fn in fns1:
          zipMe.write(os.path.join(pcsf_run_dir, fn), arcname=os.path.basename(os.path.normpath(fn)), compress_type=zipfile.ZIP_DEFLATED)
      for fn in fns2:
          zipMe.write(os.path.join(pcsf_run_dir, fn), arcname=os.path.basename(os.path.normpath(fn)), compress_type=zipfile.ZIP_DEFLATED)
    # for fn in output_file_list:
    #     os.remove(fn)
  

  print('Generated PNMs')
  return len(pn_nodeInfo)




