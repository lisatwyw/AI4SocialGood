import json, os
import pandas as pd

import os, sys, pickle, json
import pandas as pd
import polars as pol
import numpy as np
from pathlib import Path

#from IPython.display import clear_output;  clear_output()

import textwrap

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
''';
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

# =================================== defaults ===================================

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
os.environ['TOKENIZERS_PARALLELISM']= "false"



# =================================== define functions ===================================

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

    '''
    1. White: A person having origins in any of the Europe, Middle East, or North Africa.
    
    2. Black/African American: A person having origins in any of the black racial groups of Africa.
    
    4. Asian: A person having origins in any of the original peoples of the Far East, Southeast Asia, or the Indian subcontinent
    
    5. American Indian/Alaska Native: A person having origins in any of the original peoples of North and South America (including Central America), and who maintains tribal affiliation or community attachment.
    
    6. Native Hawaiian/Pacific Islander: A person having origins in any of the original peoples of Hawaii, Guam, Samoa, or other Pacific Islands.
    
    7. White Hispanic 1 Race=1    
    8. Black Hispanic 1 Race=2
    
    
    3. ED record indicates more than one race (e.g., multiracial, biracial)    
    '''
    
    df2['race_recoded'] =df2['race']      
    q=np.where( (df2['hispanic'] == 1 ) & (df2['race'] == 1) )[0]    
    df2['race_recoded'][q] = 7
    q=np.where( (df2['hispanic'] == 1 ) & (df2['race'] == 2) )[0]
    df2['race_recoded'][q] = 8 
    
    
    
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



def rid_typos( r ):
    r=r.replace('TWO','2').replace('TWPO','2').replace('TW D','2 D')
    r=r.replace('FIOUR DAYS AGO', 'FOUR DAYS AGO').replace('6X DAYS','6 DAYS').replace('4 FDAYS','4 DAYS').replace('FOOUR DAYS','4 DAYS')
    r=r.replace('SIX','6').replace('SEVEN','7').replace('THREE','3').replace('ONE','1').replace('FOUR','4').replace('FIVE','5').replace(')','')
    r=r.replace('TEN','10').replace('NINE','9').replace('24 HOURS AGO','1 DAY AGO').replace('EIGHT','8').replace('E DAYS AGO','8').replace('A DAY','1 DAY')
    r=r.replace('TWELVE','12').replace('HRS', 'HOURS').replace('HR AGO', 'HOUR AGO').replace(' F DAYS AGO', ' FEW DAYS AGO')
    r=r.replace('SEVERALDAYS', 'SEVERAL DAYS').replace('2 AND A HALF','3').replace('2 AND HALF','3').replace('5 DADAYS','5 DAYS')
    r=r.replace('2HOURS','2 HOURS').replace('1HOUR','1 HOUR').replace('AN HOUR','1 HOUR').replace('FERW HOUR','FEW HOUR')
    r=r.replace('2,DAYS', '2 DAYS').replace('SEV DAYS', 'SEVERAL DAYS')
    r=r.replace('COUPLEOF','COUPLE OF').replace('A DAY','1 DAY').replace('HALF HOUR','1 HOUR') # round to 1 hour
    r=r.replace('   ',' ').replace('  ',' ').replace('DAYSA GO', 'DAYS AGO')
    r=r.replace('LAST MIGHT AND', 'LAST NIGHT AND').replace('AT NH','AT NURSING HOME').replace('BAKC', 'BACK')
    r=r.replace('DXZ','DX').replace('10NIS', 'TENNIS').replace('N/S INJURY', 'NOT SIGNIFICANT INJURY').replace('*','')
    return r

def strip_basic_info( r ):
    for a in ['YOF', 'YO FEMALE', 'Y/O FEMALE', 'YO F', 'YF', 'Y O F', 'YOM', ' YWF',
              'YO MALE', 'YO M', 'Y O M', 'YM', 'Y/O WM',
              'Y/O MALE' , 'Y/O M', 'OLD FE', 'OLD MALE ', 'FEMALE', 'MALE']:
        try:
            r = r.split(a[:10])[1]
            #r = r[:2].replace(' ','').replace(', ', '').replace(',', '').replace('-', '').replace('#', '').replace('.', '') + r[2:]
            break
        except:
            pass
    parts=r.split('DX')
    try:
        dx = parts[1]
    except:
        dx = '' # assumed not narrated 
    return parts[0], dx



def clean_narrative(text0):

    abbr_terms = {
      "&": "and",
      "***": "",      
      ">>": "DX",
      "@": "at",
      "abd": "abdomen",
      "af": "accidental fall",
      "afib": "atrial fibrillation",
      "aki": "acute kidney injury",
      "am": "morning",
      "a.m.": "morning",
      "ams": "altered mental status",    
      "bac": "blood alcohol content",
      "bal": "blood alcohol level,",
      "biba": "brought in by ambulance",
      "c/o": "complains of",
      "chi": "closed-head injury",    # "clsd": "closed", 
      "cpk": "creatine phosphokinase", 
      "cva": "cerebral vascular accident",
      "dx": "clinical diagnosis",    #"ecf": "extended-care facility", # "er": "emergency room",
      "etoh": "ethyl alcohol", #"eval": "evaluation", "fd": "fall detected",
      "fx": "fracture",
      "fxs": "fractures",    # "glf": "ground level fall", "h/o": "history of", "htn": "hypertension",
      "hx": "history of",
      "inj": "injury",   # "inr": "international normalized ratio",
      "intox": "intoxication",    
      "loc": "loss of consciousness",
      "lt": "left",
      "mech": "mechanical",
      "mult": "multiple",
      "n.h.": "nursing home", #    "nh": "nursing home",
      "p/w": "presents with",
      "pm": "afternoon",    # "pt": "patient", # overlaps with physical therapist assistant 
      "p.m.": "afternoon",
      'prev': "previous",
      "pta": "prior to arrival",
      "pts": "patient's", #    "px": "physical examination", # not "procedure",
      "r": "right", "l": "left",
      "r/o": "rules out",
      "rt": "right",    #"s'd&f": "slipped and fell", "s/p": "after", "t'd&f": "tripped and fell", "tr": "trauma",
      "rt.": "right",
      "lt.": "left",
      "sah": "subarachnoid hemorrhage", "sdh": "acute subdural hematoma","sts": "sit-to-stand", "uti": "urinary tract infection",
      "w/": "with", "w":"with",   
      "w/o": "without",
      "wks": "weeks" 
    }  
    
    # rid of typos
    text = rid_typos(text0)
    text, dx = strip_basic_info( text )
        
    # lowercase everything
    text = text.lower()  
    
    # Ack: https://www.drivendata.org/competitions/217/cdc-fall-narratives/community-code/50/  
    #
    # map abbrevations back to English words 
    for term, replacement in abbr_terms.items():
        if term == "@" or term == ">>" or term == "&" or term == "***":
            pattern = fr"({re.escape(term)})"
            text = re.sub(pattern, f" {replacement} ", text) # force spaces around replacement            
        else:
            pattern = fr"(?<!-)\b({re.escape(term)})\b(?!-)"
            text = re.sub(pattern, replacement, text )
    
    sentences = sent_tokenizer.tokenize(text)
    sentences = [s.capitalize() for s in sentences]
    return " ".join(sentences), dx

def get_cleaned_narratives():
    with mp.Pool(mp.cpu_count()) as pool:
        res =  pool.map(clean_narrative, decoded_df2['narrative'] )     

    A = [ r[0] for r in res ]
    B = [ r[1] for r in res ]

    decoded_df2['narrative_cleaned']= A
    decoded_df2['narrated_dx'] = B    
    
    

def _get_time2hosp(r):  # in hours   
    t2hosp = -1      
    r = r.upper()           
    if (r.find('SEVERAL HOURS AGO')>=0) | (r.find('COUPLE OF HOURS AGO')>=0) | (r.find('MULTIPLE HOURS AGO')>=0) | (r.find('COUPLE HOURS AGO')>=0) | (r.find('FEW HOURS AGO')>=0) | (r.find('SEV. HOURS AGO')>=0):
        t2hosp=6
    elif (r.find('SEVERAL DAYS AGO')>=0) | (r.find('COUPLE OF DAYS AGO')>=0) | (r.find('MULTIPLE DAYS AGO')>=0) | (r.find('COUPLE DAYS AGO')>=0) | (r.find('FEW DAYS AGO')>=0) | (r.find('SEV. DAYS AGO')>=0):
        t2hosp=3*24
    elif r.find('2 WEEKS AGO')>=0:
        t2hosp=14*24
    elif r.find('A WEEK AGO')>=0:
        t2hosp=7*24        
    elif r.find('MONTH AGO')>=0:
        t2hosp=30*24        
    elif r.find('TODAY')>=0:
        t2hosp=12                
    elif r.find('YESTERDAY')>=0:        
        t2hosp=24
    elif r.find('PREVIOUS DAY')>=0:        
        t2hosp=24
    elif r.find('PREV DAY')>=0:        
        t2hosp=24    
    elif r.find('YESTERDAY MORNING')>=0:
        t2hosp=24
    elif r.find('MORNING')>=0:
        t2hosp=6
    elif r.find('THIS AFTERNOON')>=0:
        t2hosp=6
    elif r.find('LAST NIGHT')>=0:
        t2hosp=18
    else:    
        s0 = r.find('HOURS AGO')  
        if s0==-1:
            s0 = r.find('HOUR AGO')                
        if s0==-1: # not hour
            s1 = r.find('DAYS AGO')

        if s0>=0: # hours =============
            nh = r[s0-2:s0]
            try:
                o=np.float64(nh)
                t2hosp=o
            except:
                try:
                    nh = r[s0-1:s0]
                    o=np.float64(nh)/24
                    t2hosp=o
                except:
                    if r.find('HOUR AGO') >0:
                        t2hosp=1
                    else:
                        t2hosp=6                            
        elif s1>=0: # days
            st= r[s1-2:s1].replace(' ','')
            try:
                t2hosp=np.int8(st)
            except:
                try:
                    st= r[s1-1:s1]
                    t2hosp=np.int8(st)
                except:                
                    t2hosp=24 # in day  
    return t2hosp 


def get_time2hosp():
    decoded_df2['time2hosp']=0
    with mp.Pool(mp.cpu_count()) as pool:
        decoded_df2['time2hosp'] = pool.map( _get_time2hosp, decoded_df2['narrative_cleaned'] )

    # size of survival data
    print( (decoded_df2['time2hosp']>0 ).sum()/decoded_df2.shape[0] )    

    print( 'If models were to be developed, we may split into trn set of', len(trn_case_nums), 'samples and tst set of',
        len(tst_case_nums), 'samples. \n\tOverlapped indices?', np.intersect1d( tst_case_nums, trn_case_nums ), '\n\n' )

    
def get_meta_data():
    cohort_inds = np.where( decoded_df2.time2hosp > 0 )[0]  

    sub = decoded_df2.iloc[cohort_inds,:]    
    
    time_labels={0: '4 hrs', 1: '10 hrs', 2: '20 hrs', 3: '25 hrs', 4: '49 hrs', 5: '73 hrs', 6: '168 hrs', 7: '337 hrs', 8: '500 hrs' }
    
    sub['time2hosp_binned']= pd.cut(
    sub.time2hosp,
    bins=[0,4,10,20,25,49,73,7*24+1,14*24+1,10000], 
    labels=time_labels )  

    sub['time2hosp_binned'] = pd.Categorical(sub.time2hosp_binned)
    sub['time2hosp_binned'] = sub.time2hosp_binned.cat.codes

    i=np.intersect1d( sub.cpsc_case_number, trn_case_nums ); 
    j=np.intersect1d( sub.cpsc_case_number, tst_case_nums );

    sub.set_index('cpsc_case_number', inplace=True)  
    k  = 'narrative'
    kk = 'mentioned_recurrent_falls' 
  
    print( len(i)+len(j), sub.shape  )
    subs={}
    subs['trn'] = sub.loc[ i ]
    subs['tst'] = sub.loc[ j ]

    # Recurrent falls
    for t in ['tst','trn']:
        surv_pols[t] = pol.DataFrame(subs[t]).with_columns(pol.when(
          pol.col(k).str.contains('PREV F') | 
          pol.col(k).str.contains('RECURRENT F') | 
          pol.col(k).str.contains(r'FALLEN * TIMES')).then(1).otherwise(0).alias( kk ))  
    
    return sub, i, j, cohort_inds

# https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/
def get_embeddings(sentences, pretrained="paraphrase-multilingual-mpnet-base-v2"):

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(pretrained)
    embeddings = model.encode(sentences)
    return embeddings







# =================================== main ===================================

# get preprocess dataframes and data splits
if ( 'org_columns' in globals())==False:

    surv_pols,indices,sentences,meta,embeddings={},{},{},{},{}    
    folder = '/kaggle/input/neiss-2023/'    
    
    decoded_df, decoded_df2, org_columns, trn_case_nums, tst_case_nums = get_data()
    
    import nltk
    nltk.download('punkt'); 
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')    

    get_cleaned_narratives()            
    get_time2hosp()    
    sub, ii, jj, cohort_inds= get_meta_data()    
 
    indices['trn']=ii
    indices['tst']=jj
    print(decoded_df2['narrated_dx'].shape)   
    
# embeddings 
t='tst'
if 1:#os.path.isfile( f"../input/embeddings_{t}.pkl"):
    for t in ['trn','tst']:
        with open(  f"../input/neiss/embeddings_{t}.pkl", 'rb') as handle:
            meta = pickle.load(handle)    
        embeddings[t]= meta["embeddings"]
        sentences[t] = meta["sentences"]    
else:
    for t in ['tst','trn']:
        sentences[t] = list( surv_pols[t].to_pandas()["narrative_cleaned"])  
        embeddings[t]= get_embeddings(sentences[t])           
        
        # Save
        meta = {
            'indices': indices[t],
            "sentences": sentences[t],
            "embeddings": embeddings[t]
        }
        with open( f"embeddings_{t}.pkl", 'wb') as handle:
            pickle.dump(meta, handle)        
