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
  
time2event, surv_str, event_indicator= {},{},{}

t='trn'; event_indicator[t], time2event[t], surv_str[t] = get_surv( surv_pols[t].to_pandas() )
t='val'; event_indicator[t], time2event[t], surv_str[t] = get_surv( surv_pols[t].to_pandas() )
t='tst'; event_indicator[t], time2event[t], surv_str[t] = get_surv( surv_pols[t].to_pandas() )

surv_dfs_norm, scaler = get_cate_outcome( surv_pols )
