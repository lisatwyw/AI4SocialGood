install_packages(['pip install sksurv'], INTERACTIVE )

from time import time
import tensorflow_hub as hub
import pickle
import pandas as pd
import polars as pol
import numpy as np
import json 

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from pathlib import Path

import scipy
from sksurv.metrics import *
labels = ['3h','6h','9h','12h','15h','18h', '24h','2d','3d','1w','2w','1mo']

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


def get_surv( df ):
    from sksurv.util import Surv
    ev = ( df['severity'] >= 3 ) .values
    #ev = ( df['outcome']  ) .values
    time = ( df['time2hosp']  ) .values
    return ev, time, Surv.from_arrays(ev, time)

def get_cate_outcome( S ):
    scaler = StandardScaler()
    scaler.fit( S['trn'].to_pandas()[att] )
    S2={}
    for t in ['trn','val','tst']:                
        S2[t] = pd.DataFrame( scaler.transform( S[t][att] ), columns=att  )        
        S2[t]['year']= (S[t]['year']-2013)/9
        S2[t]['month']=S[t]['month']/12 
        S2[t]['time2hosp']=S[t]['time2hosp'] 
        S2[t]['outcome']=S[t]['severity'] > O
    return S2, scaler 



def split_ds( df2 ):
    df2 = df2.filter((pol.col('narrative').str.contains(' LAST ')|
                     pol.col('narrative').str.contains('MORNING')|
                     pol.col('narrative').str.contains('A.M.',literal=True, strict=True )|
                     pol.col('narrative').str.contains('P.M.',literal=True, strict=True )|
                     pol.col('narrative').str.contains('AFTERNOON')|
                     pol.col('narrative').str.contains(' DAY')|
                     pol.col('narrative').str.contains('TODAY') |
                     pol.col('narrative').str.contains('EARLIER TONIGHT') |
                     pol.col('narrative').str.contains('AROUND') |
                     pol.col('narrative').str.contains('YESTERDAY') |
                     pol.col('narrative').str.contains(' AGO') )&
                   ~(
                    pol.col('narrative').str.contains("LAST 2 STEPS", literal=True) |
                    pol.col('narrative').str.contains("LAST FEW STEP", literal=True) |
                    pol.col('narrative').str.contains("LAST STAIR", literal=True) |
                    pol.col('narrative').str.contains("LAST SEV STEP", literal=True) |
                    pol.col('narrative').str.contains("LAST STEP", literal=True) |
                    pol.col('narrative').str.contains("YR AGO", literal=True) |
                    pol.col('narrative').str.contains("YRS AGO", literal=True) |
                    pol.col('narrative').str.contains("YEAR AGO", literal=True) |
                    pol.col('narrative').str.contains("YEARS AGO", literal=True)
                   ))
    k  = 'narrative'
    kk = 'time2hosp_1'
    df = df2.with_columns( pol.col("treatment_date").str.to_datetime().dt.weekday().alias("weekday") )
    
    df = df.with_columns(pol.when(
      pol.col(k).str.contains('SEVERAL HOUR') |
      pol.col(k).str.contains('SEV HOUR', literal=True, strict=True) |
      pol.col(k).str.contains('SEV. HR', literal=True, strict=True) |
      pol.col(k).str.contains('SEV. HOUR', literal=True, strict=True)
      ).then(5).otherwise(np.nan).alias( 'a' ))
    df = df.with_columns(pol.when(
      pol.col(k).str.contains('SEVERAL DAY') | pol.col('narrative_cleaned').str.contains('several day') |
      pol.col(k).str.contains('SEV DAY') |
      pol.col(k).str.contains('SEVRAL DAY')|
      pol.col(k).str.contains('SEV. D')
      ).then(5*24).otherwise(np.nan).alias('d' ))
    df = df.with_columns(pol.when(
      pol.col(k).str.contains('SEVERAL WK') |
      pol.col(k).str.contains('SEV. WEEKS') |
      pol.col(k).str.contains('SEVERAL WEEK')|
      pol.col(k).str.contains('SEV. WK') |
      pol.col(k).str.contains('SEV WK')
      ).then(5*7*24).otherwise(np.nan).alias('y' ))

    df = df.with_columns(pol.when(
        pol.col(k).str.contains('COUPLE HOUR') | pol.col(k).str.contains('COUPLE OF HOUR')
      ).then(3).otherwise(np.nan).alias( 'c' ))
    df = df.with_columns(pol.when(
        pol.col(k).str.contains('COUPLE DAY') | pol.col(k).str.contains('COUPLE OF DAY') | pol.col(k).str.contains('3 NITES AGO')
      ).then(3*24).otherwise(np.nan).alias( 'f'))

    df = df.with_columns(pol.when(
        pol.col(k).str.contains('FEW HOUR')
      ).then(4).otherwise(np.nan).alias( 'b' ))
    df = df.with_columns(pol.when(
        pol.col(k).str.contains('FEW DAY') | pol.col(k).str.contains('FEW NIGHT') |  pol.col(k).str.contains('FEW NITE')
      ).then(4*24).otherwise(np.nan).alias( 'e' ))
    df = df.with_columns(pol.when(
        pol.col(k).str.contains('FEW HOUR')
      ).then(4).otherwise(np.nan).alias( 'ee' ))
    df = df.with_columns(pol.when(
        pol.col(k).str.contains('FEW WEEK')
      ).then(4*7*24).otherwise(np.nan).alias( 'eee' ))
    df = df.with_columns(pol.when(
        pol.col(k).str.contains('OTHER DAY')
      ).then(9*24).otherwise(np.nan).alias( 'eeee' ))

    df = df.with_columns(pol.when(
      pol.col(k).str.contains('LAST MTH')  | pol.col(k).str.contains('LAST MONTH')
      ).then(30*24).otherwise(np.nan).alias( 'g' ))
    
    df = df.with_columns(pol.when(
      pol.col(k).str.contains('PREVIOUS DAY') | pol.col(k).str.contains('PREV DAY') |
      pol.col(k).str.contains('YESTERDAY') | pol.col(k).str.contains('LAST AM')    | pol.col(k).str.contains('LAST PM') |
      pol.col(k).str.contains('LAST EVEN') | pol.col(k).str.contains('DAY BEFORE') |
      pol.col(k).str.contains('LAST NOC')  | pol.col(k).str.contains('LAST NIGHT') | pol.col(k).str.contains('LAST NITE') | pol.col(k).str.contains('LAST NGHT')
      ).then(28).otherwise(np.nan).alias( 'g' ))

    df = df.with_columns(pol.when(
      pol.col(k).str.contains('P.M.', literal=True, strict=True ) |
      pol.col(k).str.contains('THIS AFTERNOON') | pol.col(k).str.contains('THIS EVEN') | pol.col(k).str.contains('TONIGHT')
      ).then(9).otherwise(np.nan).alias( 'h' ))

    df = df.with_columns(pol.when(
        pol.col(k).str.contains('A.M.', literal=True, strict=True )|
      pol.col(k).str.contains('MORNING', literal=True, strict=True )
      ).then(12).otherwise(np.nan).alias( 'i' ))

    df = df.with_columns(pol.when(
        pol.col(k).str.contains('THIS MORNING') |pol.col(k).str.contains('THIS AM') | pol.col(k).str.contains('TDY')|
        pol.col(k).str.contains('TODAY')
      ).then(12).otherwise(np.nan).alias( 'j' ))

    df = df.with_columns(pol.when(
        pol.col(k).str.contains('LAST EVE') | pol.col(k).str.contains('LAST NGIHT') |  pol.col(k).str.contains('AROUND MIDNIGHT') |
        pol.col(k).str.contains('LAST WK') | pol.col(k).str.contains('LAST MIGHT') | pol.col(k).str.contains('LAST WEEK') | pol.col(k).str.contains('WK AGO')
      ).then(10.5*7).otherwise(np.nan).alias( 'k' ))

    df = df.with_columns(pol.when(
        pol.col('narrative_cleaned').str.contains('1 day',literal=True, strict=True)
      ).then(24).otherwise(np.nan).alias( 'x'))

    df = df.with_columns(pol.when(
        pol.col('narrative_cleaned').str.contains('1+ day',literal=True, strict=True)
      ).then(36).otherwise(np.nan).alias( 'xx'))

    df = df.with_columns(pol.when(
        pol.col('narrative_cleaned').str.contains('an hour') | pol.col('narrative_cleaned').str.contains('a hour')|
        pol.col('narrative_cleaned').str.contains('1 hour')
      ).then(1).otherwise(np.nan).alias( 'am'))

    df = df.with_columns(pol.when(
        pol.col(k).str.contains('LAST MOND') | pol.col(k).str.contains('MONDAY')
      ).then( pol.col('weekday')+7 ).otherwise(np.nan).alias( '1' ))
    df = df.with_columns(pol.when(
        pol.col(k).str.contains('LAST TUE') | pol.col(k).str.contains('TUESDAY')
      ).then(pol.col('weekday')+6).otherwise(np.nan).alias( '2' ))
    df = df.with_columns(pol.when(
        pol.col(k).str.contains('LAST WED') | pol.col(k).str.contains('WEDNESDAY')
      ).then(pol.col('weekday')+5).otherwise(np.nan).alias( '3' ))
    df = df.with_columns(pol.when(
        pol.col(k).str.contains('LAST THU') | pol.col(k).str.contains('THURSDAY')
      ).then(pol.col('weekday')+4).otherwise(np.nan).alias( '4' ))
    df = df.with_columns(pol.when(
        pol.col(k).str.contains('LAST FRI') | pol.col(k).str.contains('FRIDAY')
      ).then(pol.col('weekday')+3).otherwise(np.nan).alias( '5' ))
    df = df.with_columns(pol.when(
        pol.col(k).str.contains('LAST SAT') | pol.col(k).str.contains('SATURDAY')
      ).then(pol.col('weekday')+2).otherwise(np.nan).alias( '6' ))
    df = df.with_columns(pol.when(
        pol.col(k).str.contains('LAST SUN') | pol.col(k).str.contains('SUNDAY')
      ).then(pol.col('weekday')+1).otherwise(np.nan).alias( '7' ))
    df = df.with_columns(pol.when(
        pol.col('narrative_cleaned').str.contains('1 mon')  | pol.col('narrative_cleaned').str.contains('a mon')
      ).then(24*30).otherwise(np.nan).alias( 'ba'))
    df = df.with_columns(pol.when(
        pol.col('narrative_cleaned').str.contains('2 month')
      ).then(24*60).otherwise(np.nan).alias( 'bb'))
    df = df.with_columns(pol.when(
        pol.col('narrative_cleaned').str.contains('3 month')
      ).then(24*90).otherwise(np.nan).alias( 'bc'))
    df = df.with_columns(pol.when(
        pol.col('narrative_cleaned').str.contains('4 month')
      ).then(24*120).otherwise(np.nan).alias( 'bd'))

    df = df.with_columns(pol.when(
        pol.col('narrative_cleaned').str.contains('1 week')  | pol.col('narrative_cleaned').str.contains('a week')
      ).then(24*7).otherwise(np.nan).alias( 'be'))
    df = df.with_columns(pol.when(
        pol.col('narrative_cleaned').str.contains('2 week')
      ).then(24*14).otherwise(np.nan).alias( 'bf'))
    df = df.with_columns(pol.when(
        pol.col('narrative_cleaned').str.contains('3 week')
      ).then(24*21).otherwise(np.nan).alias( 'bg'))
    df = df.with_columns(pol.when(
        pol.col('narrative_cleaned').str.contains('4 week')
      ).then(24*28).otherwise(np.nan).alias( 'bh'))
    df = df.with_columns(pol.when(
        pol.col('narrative_cleaned').str.contains('5 week')
      ).then(24*35).otherwise(np.nan).alias( 'bi'))

    df = df.with_columns(pol.when(
      pol.col('narrative').str.contains('SEVERAL MONTH') |
      pol.col('narrative').str.contains('SEV MON') |
      pol.col('narrative').str.contains('5 MOS AGO')
      ).then(24*7*4*5).otherwise(np.nan).alias( 'bj'))

    k='narrative'
    for n in range(30):
        df = df.with_columns(pol.when(
          pol.col(k).str.contains(f'{n}D AGO') |
          pol.col(k).str.contains(f'{n} DAY') |  pol.col('narrative_cleaned').str.contains(f'{n} day') |
          pol.col(k).str.contains(f'{n}DAY')
      ).then(n*24).otherwise(np.nan).alias( f'd{n}' ))

    k='narrative'
    for n in range(24):
        df = df.with_columns(pol.when(
        pol.col(k).str.contains(f'AROUND {n}:') |pol.col(k).str.contains(f'AROUND {n}:') |
        pol.col(k).str.contains(f'AROUND {n} AM') | pol.col(k).str.contains(f'AROUND {n} PM')
      ).then(24).otherwise(np.nan).alias( f'time{n}' ))

    k='narrative_cleaned'
    for n in range(6): # ================ Month
        df = df.with_columns(pol.when(
        pol.col(k).str.contains(f'{n} mth')
      ).then(n*30*24).otherwise(np.nan).alias( f'n{n}' ))

    for n in range(10): # ================ weeks
        df = df.with_columns(pol.when(
          pol.col(k).str.contains(f'{n} wk',literal=True, strict=True ) | pol.col(k).str.contains(f'{n}wk',literal=True, strict=True ) |
          pol.col(k).str.contains(f'{n} week',literal=True, strict=True ) | pol.col(k).str.contains(f'{n}week',literal=True, strict=True ) |
          pol.col('narrative').str.contains(f'{n}WEEK',literal=True, strict=True ) |
          pol.col('narrative').str.contains(f'{n}WKS AGO',literal=True, strict=True ) |
          pol.col(k).str.contains(f'{n}weeks ago' ) |
          pol.col('narrative').str.contains(f'{n} WEEK',literal=True, strict=True )
      ).then(n*7*24).otherwise(np.nan).alias( f'n{n}' ))

    for n in range(30): # ================ DAY
        df = df.with_columns(pol.when(
      pol.col(k).str.contains(f'{n}night') |
      pol.col(k).str.contains(f'{n} night') |pol.col(k).str.contains(f'{n} d ago') |
      pol.col(k).str.contains(f'{n} day') | pol.col(k).str.contains(f'{n} dy ago') |
      pol.col(k).str.contains(f'{n}day')
      ).then(n*24).otherwise(np.nan).alias( f'n{n}' ))

    for n in range(50): # ================ HOUR
        df = df.with_columns(pol.when(
          pol.col(k).str.contains('hour ago') |
          pol.col(k).str.contains(f'{n} hour') |
          pol.col(k).str.contains(f'{n}hour') |
          pol.col('narrative').str.contains(f'{n} HOURS AGO') |
          pol.col(k).str.contains(f'{n}hrs ago') |
          pol.col(k).str.contains(f'{n}h ago')
      ).then(n).otherwise(np.nan).alias( f'h{n}' ))

    for n in range(90): # ================ minutes
        df = df.with_columns(pol.when(
        pol.col(k).str.contains(f'{n} minute') | pol.col(k).str.contains(f'{n} min ago')
      ).then(n/60).otherwise(np.nan).alias( f'm{n}' ))

    for n in range(90): # ================ minutes
        df = df.with_columns(pol.when(
        pol.col('narrative').str.contains(f'{n} MIN')
      ).then(n/60).otherwise(np.nan).alias( f'M{n}' ))

    rr=-119-90-140
    print( 'Sample size:', df.shape, df[:,rr:].head() )
    time2hosp=np.nanmax( df[:,rr:].to_numpy(),1 )
    df = df.with_columns(pol.lit(time2hosp).alias('time2hosp'))
    p  = df.filter( pol.col( 'time2hosp') .is_nan() )

    print( '\n\n',p.shape , 'remaining')
    for r in p.sample(5).iter_rows():
        print( r[3], )
        print( r[31], '\n' )

    p2 = df.filter( pol.col( 'time2hosp') >0 )
    trn_df = p2.filter( pol.col('cpsc_case_number').is_in( trn_case_nums ) )
    tst_df = p2.filter( pol.col('cpsc_case_number').is_in( tst_case_nums ) )
    print( trn_df.shape[0],'dev samples', tst_df.shape[0], 'test samples' )

    surv_pols = {}
    surv_pols['trn'] = trn_df[0::2,:]
    surv_pols['val'] = trn_df[1::2,:]
    surv_pols['tst'] = tst_df
        
    surv_dfs = {}
    for t in ['trn','val', 'tst']:
        surv_dfs[t] = surv_pols[t].to_pandas()
    
    surv_inter={}
    for t in ['trn','val','tst']:
        surv_inter[t]={}
        surv_inter[t]=pd.DataFrame( {'label_lower_bound': surv_dfs[t]['time2hosp'] ,
                                     'label_upper_bound': surv_dfs[t]['time2hosp'] } )
        q=np.where(  surv_dfs[t]['severity'] <= O )[0] # unseen + observed
        surv_inter[t].iloc[q, 1] = np.inf 
        
    return surv_pols, surv_dfs, surv_inter


# ----------------------------------------
# Load precomputed variables
# ----------------------------------------
if ('decoded_df2' in globals())==False:
    _, org_columns, trn_case_nums, tst_case_nums, mapping = get_data()
    #pol.DataFrame(merged_df).filter( pol.col('cpsc_case_number') == 200430360)

    decoded_df2=pd.read_csv('/kaggle/input/neiss-sentence-transform-embeddings/decoded_df2__l1.csv')

    #pol.DataFrame(merged_df)[:, 5].value_counts().tail()

    decoded_df2=decoded_df2.drop_duplicates('cpsc_case_number',)
    pol.DataFrame(decoded_df2).filter( pol.col('cpsc_case_number') == 200430360)

    decoded_df2.sex = (decoded_df2.sex == 'MALE').astype(int)
    for k in [ 'alcohol','fire_involvement','drug', ]:
        decoded_df2[k] = ( decoded_df2[k] == 'Yes').astype(int)    
    dic = {}
    for k in [ 'location','product_1','product_2','product_3','body_part','body_part_2' ]:
        dic[k] = {k: { i:l for l,i in enumerate( decoded_df2[k].unique() ) } }
        decoded_df2.replace( dic[k], inplace=True )
    
    O=2 # thresold on severity
    att =['location','product_1','product_2','product_3','fire_involvement','body_part','drug','alcohol', 'sex', 'age_cate_binned','race_recoded','year','month']
        
    surv_pols, _, surv_inter = split_ds( pol.DataFrame( decoded_df2 ) )
    
    time2event, surv_str, event_indicator= {},{},{}
    t='trn'; event_indicator[t], time2event[t], surv_str[t] = get_surv( surv_pols[t].to_pandas() )
    t='val'; event_indicator[t], time2event[t], surv_str[t] = get_surv( surv_pols[t].to_pandas() )
    t='tst'; event_indicator[t], time2event[t], surv_str[t] = get_surv( surv_pols[t].to_pandas() )
    
    surv_dfs_norm, scaler = get_cate_outcome( surv_pols )
