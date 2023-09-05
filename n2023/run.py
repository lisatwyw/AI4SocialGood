import json, os
import pandas as pd

import os, sys, pickle, json
import pandas as pd
import polars as pol
import numpy as np
from pathlib import Path

#from IPython.display import clear_output;  clear_output()

import textwrap

import nltk
#nltk.download('punkt'); 
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

import multiprocessing as mp
import re

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio; pio.renderers.default = 'colab' # enable renderer
'''
['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
  'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
  'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
  'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
  'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']
'''
from tqdm import tqdm

from PIL import Image
from functools import wraps

#from transformers import AutoTokenizer, FeatureExtractionPipeline, pipeline

import random, math
import torch

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


folder = '/kaggle/input/neiss-2023/'

def get_data():
    with Path( folder + "variable_mapping.json").open("r") as f:
        mapping = json.load(f, parse_int=True)

    for c in mapping.keys():
        mapping[c] = {int(k): v for k, v in mapping[c].items()}

    df = pd.read_csv(folder+"primary_data.csv", parse_dates=['treatment_date'], 
                   dtype={"body_part_2": "Int64", "diagnosis_2": "Int64", 'other_diagnosis_2': 'string'} )

    df2 = pd.read_csv(folder+"supplementary_data.csv",  parse_dates=['treatment_date'], 
                    dtype={"body_part_2": "Int64", "diagnosis_2": "Int64", 'other_diagnosis_2': 'string' } )

    org_columns = df2.columns

    df2['month'] = df2.treatment_date.dt.month
    df2['year'] = df2.treatment_date.dt.year

    df2['severity'] = df2['disposition'].replace(
        {'1 - TREATED/EXAMINED AND RELEASED': 3,
         '2 - TREATED AND TRANSFERRED': 4,
         '4 - TREATED AND ADMITTED/HOSPITALIZED': 5,  # question, more or less severe?
         '5 - HELD FOR OBSERVATION': 2,
         '6 - LEFT WITHOUT BEING SEEN': 1
        })

    df2['age_cate']= pd.cut(
    df2.age,
    bins=[0,65,75,85,95,150],
    labels=["1: 65 or under", "2: 65-74", "3: 74-85", "4: 85-94", "5: 95+"],
    )
    # drop the 3 cases of unknown sex and then map codes to English words
    df2 = df2[df2.sex != 0]

    decoded_df2 = df2.copy()
    for col in mapping.keys():
        if col != 'disposition':
            decoded_df2[col] = decoded_df2[col].map(mapping[col])    
    
    decoded_df = df.copy()
    for col in mapping.keys():
        if col != 'disposition':
            decoded_df[col] = decoded_df[col].map(mapping[col])        

  
    tst_case_nums = np.setdiff1d( decoded_df2.cpsc_case_number, df.cpsc_case_number )
    trn_case_nums = np.setdiff1d( decoded_df2.cpsc_case_number, tst_case_nums )
    
    return decoded_df, decoded_df2, org_columns, trn_case_nums, tst_case_nums

if ( 'org_columns' in globals())==False:
    decoded_df, decoded_df2, org_columns, trn_case_nums, tst_case_nums = get_data()


def get_time2hosp(r):    
    t2hosp = -1      
    r = r.upper()           
    if (r.find('SEVERAL HOURS AGO')>=0) | (r.find('COUPLE OF HOURS AGO')>=0) | (r.find('MULTIPLE HOURS AGO')>=0) | (r.find('COUPLE HOURS AGO')>=0) | (r.find('FEW HOURS AGO')>=0) | (r.find('SEV. HOURS AGO')>=0):
        t2hosp=6/24
    elif (r.find('SEVERAL DAYS AGO')>=0) | (r.find('COUPLE OF DAYS AGO')>=0) | (r.find('MULTIPLE DAYS AGO')>=0) | (r.find('COUPLE DAYS AGO')>=0) | (r.find('FEW DAYS AGO')>=0) | (r.find('SEV. DAYS AGO')>=0):
        t2hosp=3
    elif r.find('2 WEEKS AGO')>=0:
        t2hosp=14
    elif r.find('A WEEK AGO')>=0:
        t2hosp=7        
    elif r.find('MONTH AGO')>=0:
        t2hosp=30        
    elif r.find('TODAY')>=0:
        t2hosp=1/2                
    elif r.find('YESTERDAY')>=0:        
        t2hosp=1
    elif r.find('PREVIOUS DAY')>=0:        
        t2hosp=1
    elif r.find('PREV DAY')>=0:        
        t2hosp=1    
    elif r.find('YESTERDAY MORNING')>=0:
        t2hosp=1
    elif r.find('MORNING')>=0:
        t2hosp=6/24
    elif r.find('THIS AFTERNOON')>=0:
        t2hosp=6/24
    elif r.find('LAST NIGHT')>=0:
        t2hosp=18/24
    else:    
        s0 = r.find('HOURS AGO')  
        if s0==-1:
            s0 = r.find('HOUR AGO')                
        if s0==-1: # not hour
            s1 = r.find('DAYS AGO')

        if s0>=0: # hours =============
            nh = r[s0-2:s0]
            try:
                o=np.float64(nh)/24
                t2hosp=o
            except:
                try:
                    nh = r[s0-1:s0]
                    o=np.float64(nh)/24
                    t2hosp=o
                except:
                    if r.find('HOUR AGO') >0:
                        t2hosp=1/24
                    else:
                        t2hosp=6/24                            
        elif s1>=0: # days
            st= r[s1-2:s1].replace(' ','')
            try:
                t2hosp=np.int8(st)
            except:
                try:
                    st= r[s1-1:s1]
                    t2hosp=np.int8(st)
                except:                
                    t2hosp=1 # days  
    return t2hosp 

decoded_df2['time2hosp']=0
with mp.Pool(mp.cpu_count()) as pool:
    decoded_df2['time2hosp'] = pool.map(get_time2hosp, decoded_df2['narrative_cleaned'] )
    
# size of survival data
print( (decoded_df2['time2hosp']>0 ).sum()/decoded_df2.shape[0] )    

print( 'If models were to be developed, we may split into trn set of', len(trn_case_nums), 'samples and tst set of',
    len(tst_case_nums), 'samples. \n\tOverlapped indices?', np.intersect1d( tst_case_nums, trn_case_nums ), '\n\n' )


# consumer product safety commission

surv_pols = {}

def meta_data():
    cohort_inds = np.where( decoded_df2.time2hosp > 0 )[0]  
    sub = decoded_df2.iloc[cohort_inds,:]
    sub.set_index('cpsc_case_number', inplace=True)
  
    k  = 'narrative'
    kk = 'mentioned_recurrent_falls' 
  
    i=np.intersect1d( sub.cpsc_case_number, trn_case_nums ); 
    j=np.intersect1d( sub.cpsc_case_number, tst_case_nums );
    print( len(i)+len(j), sub.shape  )
    
    sub1 = sub.loc[ i ]
    sub2 = sub.loc[ j ]

    # Recurrent falls
    t='trn'
    surv_pols[t] = pol.DataFrame(sub1).with_columns(pol.when(
      pol.col(k).str.contains('PREV F') | 
      pol.col(k).str.contains('RECURRENT F') | 
      pol.col("narrative").str.contains(r'FALLEN * TIMES')).then(1).otherwise(0).alias( kk ))  
    t='tst'
    surv_pols[t] = pol.DataFrame(sub2).with_columns(pol.when(
      pol.col(k).str.contains('PREV F') | 
      pol.col(k).str.contains('RECURRENT F') | 
      pol.col("narrative").str.contains(r'FALLEN * TIMES')).then(1).otherwise(0).alias( kk ))

    return cohort_inds


if ( 'cohort_inds' in globals())==False:
    cohort_inds= meta_data()

# https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/
def get_embeddings(sentences, pretrained="paraphrase-multilingual-mpnet-base-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(pretrained)
    embeddings = model.encode(sentences)
    return embeddings

sentences,meta,embeddings={},{},{}

if os.path.isfile( f"../input/embeddings_{t}.pkl"):
    for t in ['trn','tst']:
        with open(  f"embeddings_{t}.pkl", 'rb') as handle:
            meta = pickle.load(handle)    
        embeddings[t]= meta["embeddings"]
        sentences[t] = meta["sentences"]    
else:
    for t in ['tst','trn']:
        sentences[t] = list( surv_pols[t].to_pandas()["narrative_cleaned"])  
        embeddings[t]= get_embeddings(sentences[t])   
        # Save
        meta[t] = {
            "sentences": sentences,
            "embeddings": embeddings
        }
        with open( f"embeddings_{t}.pkl", 'wb') as handle:
            pickle.dump(meta[t], handle)        



