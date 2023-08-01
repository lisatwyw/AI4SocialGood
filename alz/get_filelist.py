import matplotlib.pyplot as plt
import optuna
from glob import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import os, sys #, csv
import polars as pl
from tqdm import tqdm
import gc

from torchmetrics.classification import MultilabelF1Score, MultilabelAccuracy
 
import tensorflow_datasets as tfds
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve,auc,f1_score,recall_score,precision_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import ElasticNet  
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LassoCV
from sklearn.metrics import average_precision_score, precision_recall_curve
from scipy.stats import pearsonr


 
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)

def grab():
    #if ('trn_df' in globals() )==False:
    bigfile='../input/predict-alz/traindata.csv'
    trn_df = pl.read_csv(bigfile, has_header=True, n_rows=2 )
    display( trn_df.head())

    bigfile='../input/predict-alz/trainmap.csv'
    trn_map = pd.read_csv(bigfile, )

    trn_map['id'] = trn_map.sample_id 
    trn_map.set_index( 'id', inplace=True)
    
    m_inds = np.where(trn_map.gender == 'M')[0]
    f_inds = np.where(trn_map.gender == 'F' )[0]
    
    fig=px.histogram(trn_map, x='disease')
    fig.show()
    
    p_key=trn_map.disease.unique()[4]
    park_inds = np.where(trn_map.disease == p_key )[0]

    c_key=trn_map.disease.unique()[1]
    control_inds = np.where(trn_map.disease == c_key )[0]

    d_key=trn_map.disease.unique()[0]
    alz_inds = np.where(trn_map.disease == d_key )[0]
    
    import matplotlib.pyplot as plt
    plt.hist( trn_map.age[ np.intersect1d( control_inds, f_inds) ], label=c_key+' female')
    plt.hist( trn_map.age[ np.intersect1d( control_inds, m_inds) ], label=c_key+' male')
    plt.legend(); plt.show()

    plt.hist( trn_map.age[ np.intersect1d( park_inds, f_inds) ], label=p_key+' female')
    plt.hist( trn_map.age[ np.intersect1d( park_inds, m_inds) ], label=p_key+' male')
    plt.legend(); plt.show()
    
    plt.hist( trn_map.age[ np.intersect1d( alz_inds, f_inds) ], label=d_key+' female')
    plt.hist( trn_map.age[ np.intersect1d( alz_inds, m_inds) ], label=d_key+' male')
    plt.legend(); plt.show()
    
    fig = px.histogram( trn_map, 'disease')
    fig.show()
    
    Ds=list(sorted(trn_map.disease.unique()))
    Ds.remove('control')
    D={ d:(1+i) for i,d in enumerate( Ds) }
    D['control']=0

    trn_map.disease.replace( D, inplace=True )
    return trn_df.columns, trn_map, d_key, D

if ('train_map' in globals())==False:
    train_ids, trn_map, d_key, disease_codes = grab()
    print(trn_map.disease.unique(),  len(trn_map.age.unique() ), '# of unique age values' )
      
if ( 'Files' in globals())==False:   
    NFEATS = 485512
    def read_chunk(files, trn_map):
        print( len(files), files[0] )
        Ages,Ids = [],[]
        A = np.zeros( (len(files), NFEATS) )
        for i,g in enumerate( files ):
            f=os.path.basename( g) .split('.')[0]    
            Ids.append( f )
            Ages.append(trn_map.age[ np.where( trn_map.sample_id == f)[0]  ].values )
            A[i,:] = pl.read_parquet( g )[f]
        return np.array(Ages).squeeze(), np.array(Ids).squeeze(), A
    
    name = ['sjog']
    dirs = ['../input/meth-sjog/sjog/kaggle/working/train/sjog/*.*']
    name += ['stroke']
    dirs += ['../input/meth-sjog/stroke/kaggle/working/train/stroke/*.*']
    
    name += ['schi0']
    dirs += ['../input/predict-schi-bk0of2/train/schi/*.*']
    name += ['schi']
    dirs += ['../input/predict-schi-bk1of2/train/schi/*.*']

    for d in range(1,9,2):
        name += ['control%d' %d]
        dirs += ['../input/predict-control-bk%dof50/train/control/*.*'%d ]
        
    name += ['control9']
    dirs += [ '../input/meth-control-9of50/train/control/*.*' ] 
    
    for d in range(10,21):
        name += ['control%d' %d]
        dirs += ['../input/predict-control-bk%dof50/train/control/*.*'%d ]

    name +=['alz1']
    dirs +=['../input/predict-proc-alz/train/alzh/*.*']
    name +=['alz4']
    dirs +=['../input/predict-proc-alz672/train/alzh/*.*'] 
    name +=['alz2']
    dirs +=['../input/meth-ad-240-480/train/alzh/*.*'] 
    name +=['alz3']
    dirs +=['../input/meth-ad-780/train/alzh/*.*' ]
    
    name +=[ 'grav']
    dirs +=['../input/predict-grav/train/grav/*.*']
    
    name +=[ 'park']
    dirs +=['../input/ad-con-park-frst60/train/park/*.*']
    name +=[ 'park1']
    dirs +=['../input/predict-park/train/park/*.*'] 
    
    name +=['diab2']
    dirs +=['../input/predict-last1/train/diab2/*.*']
    
    name +=['rheu','hunt']
    dirs +=['../input/predict-%s/train/%s/*.*'%('rheu','rheu')]
    dirs +=['../input/predict-%s/train/%s/*.*'%('hunt','hunt')]

    Files=[]
    for dr in dirs:    
        files = glob( dr )
        Files+=files
    print( len(Files) , 'avail train samples')
    t_ids = [ os.path.basename(g).split('.')[0] for g in Files ]
    id2files = { os.path.basename(g).split('.')[0]:g for g in Files }
     
