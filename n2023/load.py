
import multiprocessing as mp
import re

import os, sys, pickle, json
from pathlib import Path

import pandas as pd
import polars as pol
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

'''
['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
  'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
  'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
  'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
  'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']
''';

from tqdm import tqdm
import textwrap

#from transformers import AutoTokenizer, FeatureExtractionPipeline, pipeline

import random, math
import torch
'''
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
import tensorflow as tf; tf.random.set_seed(SEED)

'''

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

try:
    os.environ['KAGGLE_KERNEL_RUN_TYPE'] == 'Interactive'
    IS_INTERACTIVE = 1
    folder = '../input/neiss-2023/'
except:
    try:
        if 'ipykernel' in os.environ['MPLBACKEND']:
            IS_INTERACTIVE = 2
            folder = ''
    except:
        IS_INTERACTIVE = 0

mode = 'publish' if IS_INTERACTIVE else 'demo'
print( 'Running in interative mode?', IS_INTERACTIVE, '(0=No | 1=Kaggle | 2=Colab)\n\nRunning mode:', mode, folder )

try:
     Path( folder + "variable_mapping.json").open("r")
except:
    if IS_INTERACTIVE==2:
        !unzip /content/drive/MyDrive/datasets/neiss-2023.zip # unzip if not found
        !unzip /content/drive/MyDrive/datasets/neiss-2013-2022-surv-embeddings.zip # unzip if not found
    
#from IPython.display import clear_output;  clear_output()
#from transformers import AutoTokenizer, FeatureExtractionPipeline, pipeline

# =================================== defaults ===================================

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
os.environ['TOKENIZERS_PARALLELISM']= "false"

SEED=101

np.random.seed(SEED)

# =================================== NLP packages ===================================

#https://github.com/JohnSnowLabs/spark-nlp
if 0:
    !pip install "/kaggle/input/pyspellchecker/pyspellchecker-0.7.2-py3-none-any.whl"
    !pip install sparknlp pyspark==3.3.1
    
    from sparknlp.base import *
    from sparknlp.annotator import *
    from sparknlp.pretrained import PretrainedPipeline

    import sparknlp
    spark = sparknlp.start()


# =================================== define functions ===================================
def get_data( folder='../input/neiss-2023/' ):
    with Path( folder + "variable_mapping.json").open("r") as f:
        mapping = json.load(f, parse_int=True)

    for c in mapping.keys():
        mapping[c] = {int(k): v for k, v in mapping[c].items()}

    df = pd.read_csv(folder+"primary_data.csv", parse_dates=['treatment_date'],
                   dtype={"body_part_2": "Int64", "diagnosis_2": "Int64", 'other_diagnosis_2': 'string'} )

    df2 = pd.read_csv(folder+"supplementary_data.csv",  parse_dates=['treatment_date'],
                    dtype={"body_part_2": "Int64", "diagnosis_2": "Int64", 'other_diagnosis_2': 'string' } )

    org_columns = df2.columns 
    df['source']=1     
    df2['source']=2     
        
    merged_df = pd.concat( (df, df2)) #, left_on='cpsc_case_number', right_on='cpsc_case_number', how='outer' )    
    merged_df.drop_duplicates( inplace=True )
    merged_df.reset_index( inplace=True )

    primary_cid = merged_df.iloc[ np.where( merged_df.source==1)[0],: ].cpsc_case_number    
    supp_cid    = merged_df.iloc[ np.where( merged_df.source==2)[0],: ].cpsc_case_number    
    
    trn_case_nums = np.intersect1d( primary_cid, supp_cid )
    tst_case_nums = np.setdiff1d( supp_cid, primary_cid )

    print(  'Trn:Tst ratio:', len(trn_case_nums)/ len(tst_case_nums),  )
        
    def add_cols( df2 ):
        
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
        
        df2['race_recoded'] = 0
        df2['race_recoded'] = df2['race'].copy()
        q=np.where( (df2['hispanic'] == 1 ) & (df2['race'] == 1) )[0]
        df2.loc[q, 'race_recoded'] = 7
        q=np.where( (df2['hispanic'] == 1 ) & (df2['race'] == 2) )[0]
        df2.loc[q, 'race_recoded'] = 8
        
        df2['severity'] = df2['disposition'].replace(
            {'1 - TREATED/EXAMINED AND RELEASED': 3,
             '2 - TREATED AND TRANSFERRED': 4,
             '4 - TREATED AND ADMITTED/HOSPITALIZED': 5,  # question, more or less severe?
             '5 - HELD FOR OBSERVATION': 2,
             '6 - LEFT WITHOUT BEING SEEN': 1
            }).copy()

        df2['age_cate']= 0
        df2['age_cate']= pd.cut(
        df2.age,
        bins=[0,65,75,85,95,150],
        labels=["1: 65 or under", "2: 65-74", "3: 74-85", "4: 85-94", "5: 95+"],
        )

        df2['age_cate'] = pd.Categorical(df2.age_cate).copy() 
        df2['age_cate_binned'] = df2.age_cate.cat.codes

        # drop the 3 cases of unknown sex and then map codes to English words
        df2 = df2[df2.sex != 0]
        return df2
    
    # add variables
    merged_df = add_cols( merged_df )    
        
    for col in mapping.keys():        
        if col != 'disposition':
            merged_df[col] = merged_df[col].map(mapping[col])
         
    return merged_df, org_columns, trn_case_nums, tst_case_nums, mapping

def rid_typos( r ):
    r=r.replace(' TWO ','2').replace('TWPO','2').replace('TW D','2 D')
    r=r.replace('FIOUR DAYS AGO', 'FOUR DAYS AGO').replace('6X DAYS','6 DAYS').replace('4 FDAYS','4 DAYS').replace('FOOUR DAYS','4 DAYS')
    r=r.replace(' SIX ',' 6 ').replace('SEVEN','7').replace('THREE','3').replace(' ONE ',' 1 ').replace(' FOUR ',' 4 ').replace(' FIVE ',' 5 ').replace(')','')
    r=r.replace(' TEN ',' 10 ').replace(' NINE ',' 9 ').replace('24 HOURS AGO','1 DAY AGO').replace('EIGHT','8').replace('E DAYS AGO','8').replace('A DAY','1 DAY')
    r=r.replace('ELEVEN','11').replace('TWELVE','12').replace('HRS', 'HOURS').replace('HR AGO', 'HOUR AGO').replace(' F DAYS AGO', ' FEW DAYS AGO')
    r=r.replace('SEVERALDAYS', 'SEVERAL DAYS').replace('2 AND A HALF','2.5').replace('2 AND HALF','2.5').replace('5 DADAYS','5 DAYS')
    r=r.replace('2HOURS','2 HOURS').replace('1HOUR','1 HOUR').replace('AN HOUR','1 HOUR').replace('FERW HOUR','FEW HOUR')
    r=r.replace('2,DAYS', '2 DAYS').replace('SEV DAYS', 'SEVERAL DAYS')
    r=r.replace('COUPLEOF','COUPLE OF').replace('A DAY','1 DAY').replace('HALF HOUR','1 HOUR') # round to 1 hour
    r=r.replace('   ',' ').replace('  ',' ').replace('DAYSA GO', 'DAYS AGO').replace('witho ','with')
    r=r.replace('LAST MIGHT AND', 'LAST NIGHT AND').replace('AT NH','AT NURSING HOME').replace(' BAKC ', ' BACK ')
    r=r.replace('DXZ','DX').replace('10NIS', 'TENNIS').replace('N/S INJURY', 'NOT SIGNIFICANT INJURY')
    r=r.replace('***','*').replace('**','*').replace('>>>','>').replace('>>','>').replace('...','.').replace('..','.')
    r=r.replace('&','and').replace("@", "at").replace('+', ' ').replace('--','. ')
    r=r.replace('VERTABRA','VERTEBRA').replace('+LOCDX','LOC DX').replace(' ONTPO ', ' ONTO ').replace(' STAT ES ', ' STATES ').replace('BALANACE', 'BALANCE')
    r=r.replace('WJILE','WHILE')
    return r

abbr_terms = {      
      "abd": "abdomen", # "af": "accidental fall",
      "afib": "atrial fibrillation",
      "aki": "acute kidney injury",
      "am": "morning", #"a.m.": "morning",
      "ams": "ALTERED mental status",
      "bac": "BLOOD ALCOHOL CONTENT",
      "bal": "BLOOD ALCOHOL LEVEL",
      "biba": "brought in by ambulance",
      "c/o": "complains of",
      "chi": "CLOSED head injury",    
      "clsd hd": "CLOSED head",
      "clsd": "CLOSED",
      "cpk": "creatine phosphokinase",
      "cva": "cerebral vascular accident", # stroke
      "dx": "diagnosis",   
      "ecf": "EXTENDED CARE FACILITY", 
      "elf": "EXTENDED CARE FACILITY", 
      "er": "emergency room",
      "ed": "emergency room",
      "etoh": "ETHYL ALCOHOL", 
      "eval": "evaluate", # "fd": "fall detected",
      "fxs": "fractures",  # "glf": "ground level fall", 
      "fx": "fracture",
      "h/o": "history of", # "htn": "hypertension",
      "hx": "history of",
      "inj": "injury",  
      "inr": "INR",  # "international normalized ratio": special type of measurement; keep abbv to retain clinical meaning ==> capitalize
      "intox": "intoxication",
      "lac": "lacerations",
      "loc": "loss of consciousness", # capitalize so that "left" will be ignored by lemmatizer, rather than convert to "leave" (present tense of "left")
      "mech": "mechanical",
      "mult": "multiple",
      "n.h.": "NURSING HOME",  
      "nh": "NURSING HOME",
      "p/w": "presents with",
      "pm": "afternoon",    
      "pt": "patient", # "p.m.": "afternoon",
      'prev': "previous",
      "pta": "prior to arrival",
      "pts": "patient's", # "px": "physical examination", # not "procedure", "r": "right", "l": "left",
      "r/o": "rules out",
      "rt": "right",   
      "s'dandf": "slipped and fell",         
      "s'd&f": "slipped and fell", 
      "t'd&f": "tripped and fell", "t'dandf": "tripped and fell",
      "tr": "trauma",
      "s/p": "after", 
      "rt.": "right",
      "lt.": "LEFT",
      "lt": "LEFT",
      "sah": "subarachnoid hemorrhage", 
      "sdh": "acute subdural hematoma",
      "sts": "sit-to-stand", 
      "uti": "UTI", # "urinary tract infection",
      "unwit'd": "unwitnessed",
      "w/": "with", 
      "w": "with",
      "w/o": "without",
      "wks": "weeks"
    }

def strip_basic_info( r ):
    for a in ['YOF', 'YO FEMALE', 'Y/O FEMALE', 'YO F', 'YF', 'Y O F', 'YOM', ' YWF',
              'YO MALE', 'YO M', 'Y O M', 'YM', 'Y/O WM',
              'Y/O MALE' , 'Y/O M', 'OLD FE', 'OLD MALE ', 'FEMALE', 'MALE']:
        try:
            r = r.split(a[:10])[1]
            break
        except:
            pass
    if r.find('DX')>-1:
        parts=r.split('DX')
    elif r.find('>')>-1:
        parts=r.split('>')
    else:
        parts=r.split('*')
    try:
        dx = parts[1]        
    except:
        dx = '' # assumed not narrated        
    return parts[0], dx

def lemmatizer(narrative:str) -> str:
    doc = nlp(narrative)    
    p = doc._.blob.polarity 
    s = doc._.blob.subjectivity    
    return " ".join([token.lemma_ for token in doc ]), p, s 

def fix_missing_sp( text ):
    def try_find(u): # ------- helper, exhaustive search
        m=-1    
        for i in range(len(u),1,-1):       
            a = u[:i].split()            
            b = spellchecker.unknown( a )                        
            if (len(b)==0):      
                m=i                
                break
        if m>-1:
            u=f'{u[:m]} {u[m:]}'                        
            return u
        else:
            return u
    text  = text.replace('.',' ').replace(',', ' ')        
    wlist = text.split()
    un=spellchecker.unknown( wlist )                 
    for u in un:        
        text=text.replace(u, try_find(u))             
    return text
    
'''    
def mask_typos( text ):
    wlist = text.split()
    un=spellchecker.unknown( wlist ) 
    if len(un)>0:
        fd.write( ' '.join(un)+'\n' )        
        t=[text.replace(u, '<mask>' ) for u in un]    
        text=t[0]
    return text
''';

def _clean_narrative(text0):    # Step 1) rid of typos
    text = rid_typos( text0 )   
    
    # Step 2) strip off demo graphic info + diagnosis 
    text, dx = strip_basic_info( text )    
    dx = dx.replace('*',' ').replace('LBP','low back pain')
    
    # Step 3) 
    text += ' ' # add char space to end so that step 5 will not ignore words immediately followed by period

    # Step 4) lowercase and add char spaces so not interprettted as spelling errors 
    text = text.lower().replace(',',' ').replace('.',' .') 
        
    # Ack: https://www.drivendata.org/competitions/217/cdc-fall-narratives/community-code/50/
    #
    # Changes need: special words are capitalized "CLOSED" "LEFT" to prevent lemmatization        
    #
    # Step 5) map abbrevations back to English words; 
    for term, replacement in abbr_terms.items():        
        text = text.replace( f' {term} ', f' {replacement} ' )        
    '''
    lemmatization: 
    - nursing home stays same
    - turned -> turn
    - sitting on chair, sit on chair
    - reaching -> reach 
    - striking -> strike  
    ''';
    # Step 6) lemmatization
    text, pol, subj = lemmatizer(text)    
    
    # Step 7)  try to fix missing char space
    text = fix_missing_sp( text.lower() )
    
    # sentences = sent_tokenizer.tokenize(text) # no effect besides compute load    
    return text, dx, pol, subj

def get_cleaned_narratives():
    with mp.Pool(mp.cpu_count()) as pool:
        res =  pool.map( _clean_narrative, decoded_df2['narrative'] )
    A = [ r[0] for r in res ]
    B = [ r[1] for r in res ]
    P = [ r[2] for r in res ]
    S = [ r[3] for r in res ]
    decoded_df2['narrative_cleaned']= A
    decoded_df2['narrative_dx'] = B
    decoded_df2['narrative_pol']= P
    decoded_df2['narrative_sub'] = S

def _get_time2hosp(r):  # in hours
    t2hosp = np.nan
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
    elif (r.find('P.M')>=0) | (r.find('A.M.')>=0):
        t2hosp=6
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

def _get_time2hosp(r):  # in hours
    r.text.split('FELL')
    
if ( 'org_columns' in globals())==False:
    surv_pols,indices,sentences,meta,embeddings={},{},{},{},{}    
    decoded_df2, org_columns, trn_case_nums, tst_case_nums, mapping = get_data( folder )
    
if 0:   
    !rm unknown_words.txt
    fd = open('unknown_words.txt','a')

    from spellchecker import SpellChecker; 
    spellchecker = SpellChecker()

    from spellchecker import SpellChecker; 
    spellchecker = SpellChecker()

    import spacy
    nlp = spacy.load( 'en_core_web_lg' ) 

    from spacytextblob.spacytextblob import SpacyTextBlob
    nlp.add_pipe('spacytextblob')

    !pip install spacytextblob
    # just a demo
    m = decoded_df2.iloc[3::70000,:].copy()
    with mp.Pool(mp.cpu_count()) as pool:
        sample_narr_cleaned = pool.map( _clean_narrative, m['narrative'] )      
    for a,b in zip(m['narrative'], sample_narr_cleaned):    
        print(a, '\n', b[0], '\n\tDX:', b[1], '\n\tpolarity:', b[2], '\n\tsubjectivity:', b[3])                  




if 0:
    get_cleaned_narratives()                
    get_time2hosp()    
    
    decoded_df2.drop('index',axis=1)
    decoded_df2.to_csv('processed_merged.csv')
else:
    decoded_df2= pd.read_csv('../input/neiss-2/processed_merged.csv', )    
    
    v=20; decoded_df2.iloc[:,v]=decoded_df2.iloc[:,v].fillna('No/Unk')   
    v=15; decoded_df2.iloc[:,v]=decoded_df2.iloc[:,v].fillna('87 - NOT STATED/UNK')
    decoded_df2.iloc[:,v].unique() 


 
def get_time2hosp( decoded_df2 ):
    decoded_df2['time2hosp']=0
    with mp.Pool(mp.cpu_count()) as pool:
        decoded_df2['time2hosp'] = pool.map( _get_time2hosp, decoded_df2[['narrative_cleaned']] )

    # size of survival data
    print( (decoded_df2['time2hosp']>0 ).sum()/decoded_df2.shape[0] )

    print( 'If models were to be developed, we may split into trn set of', len(trn_case_nums), 'samples and tst set of',
        len(tst_case_nums), 'samples. \n\tOverlapped indices?', np.intersect1d( tst_case_nums, trn_case_nums ), '\n\n' )

    return decoded_df2


if 0:
    decoded_df2 = get_time2hosp( decoded_df2 )
    decoded_df2.time2hosp.describe()



P = decoded_df2.loc[ (decoded_df2.severity==5) & (decoded_df2.time2hosp>0) ]
px.histogram(P, x='severity', y='sex', color='age_cate' )

def get_meta_data( decoded_df2, useAll=True ):
    case_ids = {}
    
    if useAll:
        cohort_inds = np.arange( decoded_df2.shape[0] )
    else:
        cohort_inds = np.where( decoded_df2.time2hosp > 0 )[0]
    sub = decoded_df2.iloc[cohort_inds,:].copy()

    time_labels={0: '4 hrs', 1: '10 hrs', 2: '20 hrs', 3: '25 hrs', 4: '49 hrs', 5: '73 hrs', 6: '168 hrs', 7: '337 hrs', 8: '500 hrs' }

    sub['time2hosp_binned']=0 
    sub['time2hosp_binned']=pd.cut(
    sub.time2hosp,
    bins=[0,4,10,20,25,49,73,7*24+1,14*24+1,10000],
    labels=time_labels )
    
    sub['time2hosp_binned'] = pd.Categorical(sub.time2hosp_binned)
    sub['time2hosp_binned'] = sub.time2hosp_binned.cat.codes

    i=np.intersect1d( sub.cpsc_case_number, trn_case_nums );
    j=np.intersect1d( sub.cpsc_case_number, tst_case_nums );

    sub['cpsc_id']=sub['cpsc_case_number'].copy()
    #sub.set_index('cpsc_case_number', inplace=True)
    
    k  = 'narrative'
    
    # Recurrent falls
    kk = 'mentioned_recurrent_falls'    
    sub = pol.DataFrame(sub).with_columns(pol.when(
      pol.col(k).str.contains('PREV F') |
      pol.col(k).str.contains('RECURRENT F') |
      pol.col(k).str.contains(r'FALLEN * TIMES')).then(1).otherwise(0).alias( kk ))
    print( kk, sub[kk].sum())
        
    kk = 'mentioned_alcohol'
    sub = sub.with_columns(pol.when(
      pol.col(k).str.contains('INTOXICATED') |
      pol.col(k).str.contains('ALCOHOL') |
      pol.col(k).str.contains('BAC OF') |
      pol.col(k).str.contains('BAL OF') |
      pol.col(k).str.contains('\+ETOH') |
      pol.col(k).str.contains('ETOH ')).then(1).otherwise(0).alias( kk ))        
    print( kk, sub[kk].sum())
        
    kk = 'involved_pet'    
    sub = sub.with_columns(pol.when(
      pol.col(k).str.contains('CHASING CAT') |
      pol.col(k).str.contains('CHASING DOG') |
      pol.col(k).str.contains('CHASING GRAND') |
      pol.col(k).str.contains('CHASING BABY') |
      pol.col(k).str.contains(' PET ')         
    ).then(1).otherwise(0).alias( kk ))        
    print( kk, sub[kk].sum())        
                
    #print( len(i)+len(j), sub.shape  )    
    #subs={}
    sub=sub.to_pandas()
    #sub['cpsc_case_number']=sub['cpsc_id'].copy()
    sub = sub.set_index('cpsc_case_number')

    #subs['trn'] = pol.DataFrame(sub.loc[i])
    #subs['tst'] = pol.DataFrame(sub.loc[j])
    
    case_ids['trn']=i
    case_ids['tst']=j

    return sub, case_ids, cohort_inds

sub, case_ids, cohort_inds = get_meta_data( decoded_df2, useAll=1 )

print('sub', sub['narrative_dx'].shape) 


def get_X_E():
    surv_pols, E={},{}        
    for t in ['trn' ,'tst']:
        p = sub.loc[ case_ids[t] ].drop_duplicates( 'cpsc_id')
        surv_pols[t] = pol.DataFrame(p)
        tt= p['index'].values         
        E[ t ] = embeddings[ tt,: ]        
    return surv_pols, E
  
surv_pols, E = get_X_E()

t='trn' 
for i,r in enumerate(surv_pols[t].filter( pol.col('time2hosp')>21 ).sample(10).iter_rows()):
    print(i, r[2], '\n\n', r[30], '\n\tBody:', r[13], '\n\tNarrativeDx:',r[-9], '\n\tPolarity:', r[-8],'\n\tSubjectivity:', r[-7], '\n\tt2e:', r[-6] )

print( 'Variables:\n\n surv_pols[t] polar, E[t], sub, case_ids, cohort_inds' ) 
  
