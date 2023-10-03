try:
    import sksurv    
except:
    install_packages(['pip install scikit-survival'], INTERACTIVE )
    import sksurv

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


def get_surv( df ):
    from sksurv.util import Surv
    ev = ( df['severity'] >= 3 ) .values
    #ev = ( df['outcome']  ) .values
    time = ( df['time2hosp']  ) .values
    return ev, time, Surv.from_arrays(ev, time)

if ( 'surv_pols' in globals()) ==False:
    surv_pols, _, surv_inter = split_ds( pol.DataFrame( decoded_df2 ) )
    time2event, surv_str, event_indicator= {},{},{}
    
    t='trn'; event_indicator[t], time2event[t], surv_str[t] = get_surv( surv_pols[t].to_pandas() )
    t='val'; event_indicator[t], time2event[t], surv_str[t] = get_surv( surv_pols[t].to_pandas() )
    t='tst'; event_indicator[t], time2event[t], surv_str[t] = get_surv( surv_pols[t].to_pandas() )
    
    surv_dfs_norm, scaler = get_cate_outcome( surv_pols )
