from time import time
import tensorflow_hub as hub
import pickle
import pandas as pd
import polars as pol
import numpy as np
import json 
from pathlib import Path

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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

    print( 'Trn:Tst ratio:', len(trn_case_nums)/ len(tst_case_nums),  )
        
    def add_cols( df2 ):
        
        df2['month'] = df2.treatment_date.dt.month
        df2['year'] = df2.treatment_date.dt.year

        '''
        0. Not stated
        1. White: A person having origins in any of the Europe, Middle East, or North Africa.
        2. Black/African American: A person having origins in any of the black racial groups of Africa.
        3. ED record indicates more than one race (e.g., multiracial, biracial)
        4. Asian: A person having origins in any of the original peoples of the Far East, Southeast Asia, or the Indian subcontinent
        5. American Indian/Alaska Native: A person having origins in any of the original peoples of North and South America (including Central America), and who maintains tribal affiliation or community attachment.
        6. Native Hawaiian/Pacific Islander: A person having origins in any of the original peoples of Hawaii, Guam, Samoa, or other Pacific Islands.
        7. White Hispanic 1 Race=1
        8. Black Hispanic 1 Race=2        
        '''

        k = 'race_white'
        df2[k] = 0 # non-white 
        q=np.where( df2['race'] == 0 )[0]
        df2.loc[ q, k] = -1 # not stated 
        q=np.where( df2['race'] == 1 )[0]
        df2.loc[ q, k] = 1   

        k = 'race_4'
        df2[k] = 0 # non-white 
        q=np.where( df2['race'] == 0 )[0]
        df2.loc[ q, k] = -1
        q=np.where( df2['race'] == 4 )[0]
        df2.loc[ q, k] = -2
        q=np.where( df2['race'] == 1 )[0]
        df2.loc[ q, k] = 1   
        
        
        df2['race_recoded'] = 0
        df2['race_recoded'] = df2['race'].copy()
        q=np.where( (df2['hispanic'] == 1 ) & (df2['race'] == 1) )[0]
        df2.loc[q, 'race_recoded'] = 7
        q=np.where( (df2['hispanic'] == 1 ) & (df2['race'] == 2) )[0]
        df2.loc[q, 'race_recoded'] = 8
        
        df2['severity'] = df2['disposition'].copy() 
        df2['severity'].replace( {9: np.nan, 6: 1, 5:2,  1:3,  2:4 ,  4:5,  8: 6 }, inplace=True)
        

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

def load_decoded():    
    decoded_df2=pd.read_csv('../input/n-raw-extract-4/decoded_df2_unique.csv')
    #pol.DataFrame(merged_df)[:, 5].value_counts().tail()

    #decoded_df2=decoded_df2.drop_duplicates('cpsc_case_number',)
    #pol.DataFrame(decoded_df2).filter( pol.col('cpsc_case_number') == 200430360)

    decoded_df2.sex = (decoded_df2.sex == 'MALE').astype(int)
    for k in [ 'alcohol','fire_involvement','drug', ]:
        decoded_df2[k] = ( decoded_df2[k] == 'Yes').astype(int)    
    dic = {}
    for k in [ 'location','product_1','product_2','product_3','body_part','body_part_2' ]:
        dic[k] = {k: { i:l for l,i in enumerate( decoded_df2[k].unique() ) } }
        decoded_df2.replace( dic[k], inplace=True )  
    return decoded_df2

if ('decoded_df2' in globals())==False:
    _, org_columns, trn_case_nums, tst_case_nums, mapping = get_data()
    #pol.DataFrame(merged_df).filter( pol.col('cpsc_case_number') == 200430360)
    decoded_df2 = load_decoded()
    
att =['location','product_1','product_2','product_3','fire_involvement','body_part','drug','alcohol', 'sex', 'age_cate_binned','race_recoded','year','month']
