install_packages(['pip install sksurv optuna'], INTERACTIVE )

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

import optuna
import xgboost as xgb
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


# ====================================
# XGB/ Optuna search
# ====================================
def run_xgb_optuna( emb, X, surv_inter ):        
    ds = {}
    base_params = {'verbosity': 0, 
                  'objective': 'survival:aft',
                  'eval_metric': 'aft-nloglik',
                  'tree_method': 'hist'}  # Hyperparameters common to all trials
    samp_choices = ['uniform']

    for t in [ 'trn','val','tst']:
        ds[t] = xgb.DMatrix( X[t] )
        # see details https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html            
        ds[t].set_float_info('label_lower_bound', surv_inter[t]['label_lower_bound'] )
        ds[t].set_float_info('label_upper_bound', surv_inter[t]['label_upper_bound'] )
            
    print( ds['trn'].num_row(),surv_inter['trn'].shape, ds['val'].num_row(),surv_inter['val'].shape, ds['tst'].num_row(), surv_inter['tst'].shape )
    
    if gpus:
        base_params.update( {'tree_method': 'gpu_hist', 'device':'cuda', } )
        samp_choices = ['gradient_based','uniform']

    def tuner(trial):
        params = {'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0),
                  'aft_loss_distribution': trial.suggest_categorical('aft_loss_distribution',
                                                                      ['normal', 'logistic', 'extreme']),
                  'aft_loss_distribution_scale': trial.suggest_float('aft_loss_distribution_scale', 0.1, 10.0),
                  'max_depth': trial.suggest_int('max_depth', 3, 10),
                  'booster': trial.suggest_categorical('booster',['gbtree','dart',]),
                  'scale_pos_weight': trial.suggest_float('scale_pos_weight', pos_ratio*0.1, 10*pos_ratio ),  # L1 reg on weights
                  'alpha': trial.suggest_float('alpha', 1e-8, 10 ),  # L1 reg on weights
                  'lambda': trial.suggest_float('lambda', 1e-8, 10 ),  # L2 reg on weights
                  'eta': trial.suggest_float('eta', 0, 1.0),  # step size
                  'sampling_method': trial.suggest_categorical('sampling_method', samp_choices ),
                  'subsample': trial.suggest_float('subsample', 0.01, 1 ),
                  'gamma': trial.suggest_float('gamma', 1e-8, 10)  # larger, more conservative; min loss reduction required to make leaf
        }
        params.update(base_params)
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'valid-aft-nloglik')
        bst = xgb.train(params, ds['trn'], num_boost_round=10000,
                        evals=[(ds['trn'], 'train'), (ds['val'], 'valid')],  # <---- data matrices
                        early_stopping_rounds=50, verbose_eval=False, callbacks=[pruning_callback])
        if bst.best_iteration >= 25:
            return bst.best_score
        else:
            return np.inf  # Reject models with < 25 trees

    # Run hyperparameter search
    study = optuna.create_study(direction='minimize')
    study.optimize( tuner, n_trials=500 )
    print('Completed hyperparameter tuning with best aft-nloglik = {}.'.format(study.best_trial.value))
    params = {}
    params.update(base_params)
    params.update(study.best_trial.params)

    print('Re-running the best trial... params = {}'.format(params))
    bst = xgb.train(params, ds['trn'], num_boost_round=10000, verbose_eval=False,
                    evals=[(ds['trn'], 'train'), (ds['val'], 'valid')],
                    early_stopping_rounds=50)
    
    # Save trained model
    bst.save_model(f'aft_best_model_{emb}.json')
    
    # Explore hyperparameter search space 
    #plt.figure()
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
    
    for t in ['trn','val','tst']:         
        res[t]= pd.DataFrame({'Label (lower bound)': surv_inter[t]['label_lower_bound'],
                   'Label (upper bound)': surv_inter[t]['label_upper_bound'],
                   'Predicted label': bst.predict(ds[t]) } )

        sp=scipy.stats.spearmanr( res[t].iloc[:,-2], res[t].iloc[:,-1]  )

        c=concordance_index_censored( event_time = time2event[t], event_indicator = event_indicator[t] , estimate=1/res[t].iloc[:,-1] )

        print(t.upper(), f'| R2:{sp[0]:.3f}; p:{sp[1]:.4} | C:{c[0]*100:.2f} ', end='|' )
        for d,h in enumerate([3,6,9,12,15,18,24,49,73, 7*24+1, 7*24*2+1, 7*24*4+1 ]):
            bs = brier_score( surv_str['trn'], surv_str[t], estimate=1/res[t].iloc[:,-1], times=[h] )
            print( end=f'{labels[d]}:{bs[1][0]:.3f} | ' )

    return res

t='trn'
from sksurv.kernels import clinical_kernel
from sksurv.svm import FastKernelSurvivalSVM
#kernel_matrix = clinical_kernel( surv_dfs_norm[t][att] )

kssvm = FastKernelSurvivalSVM(optimizer="rbtree", kernel="precomputed", random_state=0)
if 0:
    kgcv = GridSearchCV(kssvm, param_grid, scoring=score_survival_model, n_jobs=1, refit=False, cv=cv)
    kgcv = kgcv.fit(kernel_matrix, surv_str[t])

    round(kgcv.best_score_, 3), kgcv.best_params_
if 0:
    coxnet_pred = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, fit_baseline_model=True))
    coxnet_pred.set_params(**gcv.best_params_)
    coxnet_pred.fit(surv_dfs_norm[t][att], surv_str[t])



# ====================================
# TabNet
# ====================================

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor 
from pytorch_tabnet.augmentations import ClassificationSMOTE
from sklearn.metrics import confusion_matrix

from sklearn.metrics import log_loss
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import roc_auc_score, log_loss

class LogitsLogLoss(Metric):
    """
    LogLoss with sigmoid applied
    """

    def __init__(self):
        self._name = "logits_ll"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        """
        Compute LogLoss of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            LogLoss of predictions vs targets.
        """
        logits = 1 / (1 + np.exp(-y_pred))
        aux = (1-y_true)*np.log(1-logits+1e-15) + y_true*np.log(logits+1e-15)
        return np.mean(-aux)

    
def run_tabnet(emb, X, event_indicator ):
    for mid in range(3):
        D = 24
        if mid==0:
            aug=None
        elif mid==1:            
            aug = ClassificationSMOTE(p=0.2)    
        elif mid==2:       
            aug=None
            D = 48
        tabnet_params = dict(n_d=D, n_a=D, n_steps=1, gamma=1.3,
                             lambda_sparse=0.0, optimizer_fn=torch.optim.Adam,
                             optimizer_params=dict(lr=2e-3, weight_decay=1e-5),
                             mask_type='entmax',
                             scheduler_params=dict(mode="min",
                                                   patience=5,
                                                   min_lr=1e-5,
                                                   factor=0.9,),
                             scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                             verbose=10, 
                             )
        MAX_EP = 500
        clf = TabNetClassifier(**tabnet_params ) 
        clf.fit(
            X['trn'], event_indicator['trn'].astype(int),
            eval_set=[( X['val'], event_indicator['val'].astype(int)) ],
            max_epochs=MAX_EP, compute_importance=False, augmentations = aug, 
            num_workers = 4, batch_size=1024, virtual_batch_size=128, weights=.08,
        )   
        res= {}
        for t in ['trn','val','tst']:
            y_preds = clf.predict( X[t]) 
            tn,fp,fn,tp = confusion_matrix( y_preds,  event_indicator[t] ).ravel()
            sp=tn/(tn+fp)
            se=tp/(tp+fn)
            npv=tn/(tn+fn)
            ppv=tp/(tp+fp)    
            print(mid, t.upper(), f'SEN={se*100:.2f}, SPE={sp*100:.2f}, PPV={ppv*100:.2f}, NPV={npv*100:.2f}')
            res[t] = f'SEN={se*100:.2f}, SPE={sp*100:.2f}, PPV={ppv*100:.2f}, NPV={npv*100:.2f}'
        return res

    

for mid in ['xgb',]:
    for emb in EMB:        
    #for emb in [19,20,1,2,3,4,]:        
        X,res ={},{}    
        for t in [ 'trn','val','tst']:
            if t=='samp':
                ds[t] = xgb.DMatrix(Embeddings[emb, 'val'].iloc[::10,:].to_numpy() )
                ds[t].set_float_info('label_lower_bound', surv_inter['val']['label_lower_bound'][::10] )
                ds[t].set_float_info('label_upper_bound', surv_inter['val']['label_upper_bound'][::10] )
            else:            
                if emb>=19:
                    X[t] = np.hstack( (Embeddings[1,t],Embeddings[2,t],Embeddings[3,t], Embeddings[4,t] ) )                 
                else:
                    X[t] = Embeddings[emb,t].to_numpy()            
                if emb == 20:
                    X[t] = np.hstack( (X[t], surv_dfs_norm[t][att] ) )

        if mid == 'xgb':
            res = run_xgb_optuna( emb, X, surv_inter)            
        else:
            res = run_tabnet( emb, X, event_indicator )

