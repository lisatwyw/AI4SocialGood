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

# https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3

trn,val,tst=0,1,2
SEED=101
np.random.seed(SEED)
class DataGenerator(tf.keras.utils.Sequence):    
    def __init__(self, df, 
                 batch_size=64,
                 input_size=(5120,1),
                 task = 'disease',
                 shuffle=True):
        
        self.ndims=len( input_size )
        self.seq_len = input_size[0] 
        self.input_size = input_size
        
        self.df = df.copy()
        self.t_ids = df.sample_id
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.n = len(self.t_ids)
        
        self.curr_segment=None
        if self.ndims==2:
            self.curr_segment=0
            
        self.outcome = (df['sample_type'] != 'control').values.astype(int)
        self.gender = df['gender'] .values
        self.disease = df['disease'].values

        self.task =task
        self.nclasses=2         
        self.__shuffle()
        
    def __shuffle(self):
        self.inds_by_class={}        
        for c in range(self.nclasses):
            s = np.where(self.outcome==c )[0]
            r = np.random.permutation( len(s))            
            self.inds_by_class[ c ] = s[r]
            print(c, f'has {len( self.inds_by_class[ c ] )} samples')

    def on_epoch_end(self):
        if self.ndims==2:
            self.curr_segment +=1
            self.n_segments = NFEATS//self.seq_len
        if self.shuffle:
            self.__shuffle()
        print(f'{self.curr_segment} of {self.n_segments} segment' )

    def __get_data(self, batches):
        
        # Generates data containing batch_size samples
        y_control= (self.df.loc[batches].sample_type.values != 'control').astype(int) # 0: control; 1: diseased
        y_gender = (self.df.loc[batches].gender.values == 'M').astype(int) #0: female; 1: male
        y_age = self.df.loc[batches].age.values
        if self.ndims==3:
            X_batch = np.zeros((self.batch_size,696,696 ),dtype=float)
        else:
            X_batch = np.zeros((self.batch_size, self.seq_len ),dtype=float)
            
        for i,f in enumerate(batches):
            g = id2files[f]
            x = pd.read_parquet( g ).fillna( 1 )[f].values
            #print(x.shape, X_batch.shape )
            if self.ndims==2:
                xx= x[self.curr_segment*self.seq_len:(self.curr_segment+1)*self.seq_len ]
                X_batch[i,:len(xx)] =xx
            else:
                X_batch[i,:] = np.reshape( x[:696*696], [696,696])
            
        r = np.random.permutation( len(batches) )            
        
        self.current_ids = batches[r].values.copy()
                    
        xx = X_batch[r,]
        if self.ndims==2:
            xx = np.expand_dims( xx, -1 )
            #print(xx.shape,end=', ')
     
        if 'age' in self.task:
            out = y_age[r]
        elif 'gender' in self.task:
            out = y_gender[r]
        elif 'disease' in self.task:
            out = y_control[r]
        return xx, out #{'regression':y_age[r], 'autoencoder':X_batch[r,] }
                
        #return X_batch[r,], {'classification':y_control, 'autoencoder':X_batch[r,] }
        #return X_batch[r,], {'control_class':y_control, 'gender_class': y_gender, 'age': y_age[r], 'autoencoder':X_batch[r,] }
    
    def __getitem__(self, index):
        half = self.batch_size//2
        print(self.nclasses, half )
        batches = np.empty(0)
        for c in range( self.nclasses ):
            nrounds = len(self.inds_by_class[ c ])//half
            index = index % ( nrounds )            
            s = self.inds_by_class[ c ][ index*half:(index+1)*half]
            batches = np.hstack((batches, s)) 
            
        batches = self.t_ids[batches.astype(int)] 
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size
    

