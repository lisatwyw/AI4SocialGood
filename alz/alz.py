
import numpy as np # linear algebra
np.random.seed(101)

import math
from scipy.stats import pearsonr

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import matplotlib.pyplot as plt
import optuna
from glob import glob

import os, sys, gc 
#, csv
IS_INTERACTIVE = os.environ['KAGGLE_KERNEL_RUN_TYPE'] == 'Interactive'

import polars as pl
from tqdm import tqdm

from torchmetrics.classification import MultilabelF1Score, MultilabelAccuracy
 
#import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))); tf.debugging.set_log_device_placement(False)

from tensorflow.keras.layers import Dense, Bidirectional, Flatten, LSTM, Conv1D, MaxPooling1D, RepeatVector
from tensorflow.keras.layers import TimeDistributed, Input, GRU, Dropout, Masking 
from tensorflow.keras.models import Model, load_model


import sklearn
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
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# leave for optuna 
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


    

# ================== Problem specific admin ==================
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
    '''
    plt.hist( trn_map.age[ np.intersect1d( control_inds, f_inds) ], label=c_key+' female')
    plt.hist( trn_map.age[ np.intersect1d( control_inds, m_inds) ], label=c_key+' male')
    plt.legend(); plt.show()

    plt.hist( trn_map.age[ np.intersect1d( park_inds, f_inds) ], label=p_key+' female')
    plt.hist( trn_map.age[ np.intersect1d( park_inds, m_inds) ], label=p_key+' male')
    plt.legend(); plt.show()
    
    plt.hist( trn_map.age[ np.intersect1d( alz_inds, f_inds) ], label=d_key+' female')
    plt.hist( trn_map.age[ np.intersect1d( alz_inds, m_inds) ], label=d_key+' male')
    plt.legend(); plt.show()
    '''    
    Ds=list(sorted(trn_map.disease.unique()))
    Ds.remove('control')
    D={ d:(1+i) for i,d in enumerate( Ds) }
    D['control']=0
    C2D = {(1+i):d for i,d in enumerate(Ds) }
    C2D[0]='control'
    trn_map.disease.replace( D, inplace=True )
    return trn_df.columns, trn_map, d_key, D, C2D

if ('train_map' in globals())==False:
    NFEATS = 485512
    train_ids, trn_map, d_key, disease2code, code2disease = grab()
    print(trn_map.disease.unique(),  len(trn_map.age.unique() ), '# of unique age values' )


def get_filelist():       
    def read_chunk(files, trn_map):
        # print( len(files), files[0] )
        Ages,Ids = [],[]
        A = np.zeros( (len(files), NFEATS) )
        for i,g in enumerate( files ):
            f=os.path.basename( g) .split('.')[0]    
            Ids.append( f )
            Ages.append(trn_map.age[ np.where( trn_map.sample_id == f)[0]  ].values )
            A[i,:] = pl.read_parquet( g )[f]
        return np.array(Ages).squeeze(), np.array(Ids).squeeze(), A
     
    name = ['control9']
    dirs = ['../input/meth-control-9of50/train/control/*.*' ] 
    
    for d in range(1,9):
        name += ['control%d' %d]
        dirs += ['../input/predict-control-bk%dof50/train/control/*.*'%d ]  
    
    for d in range(10,29):
        name += ['control%d' %d]
        dirs += ['../input/predict-control-bk%dof50/train/control/*.*'%d ]

    name += ['control14a']
    dirs += ['../input/predict-control-bk0of14/train/control/*.*' ] 

    name += ['control14d']
    dirs += ['../kaggle/input/meth-control-bk3of14/train/control/*.*']

    name +=['alz0']
    dirs +=['../input/ad-con-park-frst60/train/alz/*.*']
    name +=['alz1']
    dirs +=['../input/predict-proc-alz/train/alzh/*.*']
    name +=['alz4']
    dirs +=['../input/predict-proc-alz672/train/alzh/*.*'] 
    name +=['alz2']
    dirs +=['../input/meth-ad-240-480/train/alzh/*.*'] 
    name +=['alz3']
    dirs +=['../input/meth-ad-780/train/alzh/*.*' ]
    name +=['alz5']
    dirs +=['../input/predict-proc-alz-last/train/alzh/*.*' ]
    
    name +=['grav']
    dirs +=['../input/predict-grav/train/grav/*.*']
    
    name +=['park']
    dirs +=['../input/ad-con-park-frst60/train/park/*.*']
    name +=[ 'park1']
    dirs +=['../input/predict-park/train/park/*.*'] 
    
    name +=['diab2']
    dirs +=['../input/predict-last1/train/diab2/*.*']
    
    name +=['rheu','hunt']
    dirs +=['../input/predict-%s/train/%s/*.*'%('rheu','rheu')]
    dirs +=['../input/predict-%s/train/%s/*.*'%('hunt','hunt')]
 
    name += ['sjog']
    dirs += ['../input/meth-sjog/sjog/kaggle/working/train/sjog/*.*']
    name += ['stroke']
    dirs += ['../input/meth-sjog/stroke/kaggle/working/train/stroke/*.*']
    
    name += ['schi0']
    dirs += ['../input/predict-schi-bk0of2/train/schi/*.*']
    name += ['schi']
    dirs += ['../input/predict-schi-bk1of2/train/schi/*.*']
 
    Files=[]
    for dr in dirs:    
        files = glob( dr )
        Files+=files
    t_ids = [ os.path.basename(g).split('.')[0] for g in Files ]
    id2file = { os.path.basename(g).split('.')[0]:g for g in Files }   
 
    t_ids = np.unique(t_ids) # remove duplicates
 
    if len(t_ids)>0:
        fig =px.histogram(trn_map.loc[t_ids],'disease', title='Current sample of training set', )
        fig.show()
    print( len(t_ids), 'avail train samples', len(Files), len(id2file) )
    return Files, t_ids, id2file

# ================== evaluation code snippet provided by University 
def evaluate_ml(y_true, y_pred, sample_type):
    '''
    This function is used to evaluate the performance of the model. 

    Parameters:
    ------------
    y_true: true age
    y_pred: predicted age
    sample_type: sample type, 0 for control, 1 for case
    
    Return:
    ------------
    mae: mean absolute error.
    mae_control: mean absolute error of control samples.
    mae_case: mean absolute error of case samples.

    We use MAE to evaluate the performance.
    Please refer to evaluation section in the the official website for more details.
    '''
    mae_control = np.mean(
        np.abs(y_true[sample_type == 0] - y_pred[sample_type == 0]))

    case_true = y_true[sample_type == 1]
    case_pred = y_pred[sample_type == 1]
    above = np.where(case_pred >= case_true)
    below = np.where(case_pred < case_true)

    ae_above = np.sum(np.abs(case_true[above] - case_pred[above])) / 2
    ae_below = np.sum(np.abs(case_true[below] - case_pred[below]))
    mae_case = (ae_above + ae_below) / len(case_true)

    mae = np.mean([mae_control, mae_case])
    return mae, mae_control, mae_case

         
# ================== data generators ==================
class DataGenerator(tf.keras.utils.Sequence):    
    def __init__(self, df, 
                 hp, id2file,
                 shuffle=True):
        self.hp=hp
        self.NFEATS = hp['NFEATS']; input_size = hp['input_size']
        self.ndims=len( input_size )
        self.seq_len = input_size[0] 
        self.input_size = input_size
        
        self.df = df.copy()
        self.t_ids = df.sample_id
        
        self.batch_size = hp['BS']
        self.shuffle = shuffle
        
        self.n = len(self.t_ids)
        
        self.curr_segment=None
        if self.ndims==2:
            self.curr_segment=0
            
        self.outcome = (df['sample_type'] != 'control').values.astype(int)
        self.gender = df['gender'] .values
        self.disease = df['disease'].values
        self.age = df['age'] .values

        Y = {'age': self.age,  'gender':self.gender, 'disease': self.outcome } 
        
        self.task = hp['task']
        self.nclasses= 2 # each batch will have equal number of this grp 
        #len( np.unique( Y[self.task]) )     
                  
        self.__shuffle()
        self.shown=0
        self.id2file=id2file
        self.id_order=[]
                  
    def __shuffle(self):
        self.inds_by_class={}        
        for c in range(self.nclasses):
            s = np.where(self.outcome==c )[0]

            if self.shuffle:
                r = np.random.permutation( len(s))            
                self.inds_by_class[ c ] = s[r]
            else:
                self.inds_by_class[ c ] = s 
            print(c, f'has {len( self.inds_by_class[ c ] )} samples', end=' | ')

    def on_epoch_start(self):
        self.id_order=[]
     
    def on_epoch_end(self):
        if self.ndims==2:
            self.curr_segment +=1
            self.n_segments = self.NFEATS//self.seq_len
        if self.shuffle:
            self.__shuffle()
        print(f'{self.curr_segment} of {self.n_segments} segment', )

    def __get_data(self, batches):
        
        # Generates data containing batch_size samples
        y_control= (self.df.loc[batches].sample_type.values != 'control').astype(int) # 0: control; 1: diseased
        y_gender = (self.df.loc[batches].gender.values == 'M').astype(int) #0: female; 1: male
        y_age = self.df.loc[batches].age.values
        
        if self.shown==0:
            #self.shown=1
            print( 'Age samples:',y_age[::(len(y_age)//2)], end=' |' )
            print( f'class A:{np.sum(y_control==0)}, class B:{np.sum(y_control==1)}',end=' | ' )
            print( np.sum(y_gender==0), 'females',np.sum(y_gender==1), 'males',)
            
        y_gender = tf.keras.utils.to_categorical( y_gender )
        y_control = tf.keras.utils.to_categorical( y_control )
        if self.ndims==3:
            X_batch = np.zeros((self.batch_size, 696, 696), dtype=float)
        else:
            X_batch = np.zeros((self.batch_size, self.seq_len), dtype=float)
            
        for i,f in enumerate(batches):
            g = self.id2file[f]
            x = pd.read_parquet( g ).fillna( 1 )[f].values
            #print(x.shape, X_batch.shape )
            if self.ndims==2:
                xx= x[self.curr_segment*self.seq_len:(self.curr_segment+1)*self.seq_len ]
                X_batch[i,:len(xx)] =xx
            else:
                X_batch[i,:] = np.reshape( x[:696*696], [696,696])
        
        r = np.random.permutation( len(batches) )            
        
        self.current_ids = batches[r].values.copy(); self.id_order.append(  batches[r].values.copy() )
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
        else:
            out = {'regression':y_age[r], 'classify': y_control[r], 'classify2': y_gender[r] }
        return xx, out #
                
        #return X_batch[r,], {'classification':y_control, 'autoencoder':X_batch[r,] }
        #return X_batch[r,], {'control_class':y_control, 'gender_class': y_gender, 'age': y_age[r], 'autoencoder':X_batch[r,] }
    
    def __getitem__(self, index):
        half = self.batch_size//2
        # print(self.nclasses, half )
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



# ================== trainning  metrics ==================
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def get_f1macro(y_true, y_pred): #taken from old keras source code
    #print( y_true , y_pred  )
    f1_val = tf.keras.metrics.F1Score(average='macro',name='f1_score',)( y_true, y_pred )
    return f1_val



# ================== model defns ==================
def get_model(  hp ):
    DO=hp['DO'] 
    if 'conv1d' in hp['mid']:     
        inputs = Input(shape=( hp['SEQLEN'], 1))
        x = Conv1D( 64, 3, activation='gelu', padding='same')(inputs)
        x = MaxPooling1D(2, padding='same')(x)
        x = Dropout(DO)(x)
        x = Conv1D( 64, 5, activation='gelu', padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D( 64, 7, activation='gelu', padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Flatten()(x)
        x = Dense( 512, activation='gelu')(x)
        x = Dropout(DO/2)(x)
        x = Dense( 10, activation='gelu')(x)
        x = Dropout(DO/3)(x)

    elif 'sleep' in hp['mid']:
        DO=hp['DO'] 
        inputs = Input(shape=( hp['SEQLEN'], 1))
        x = Conv1D( 64, 3, activation='relu', padding='same')(inputs)
        x = Conv1D( 128, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D( 128, 5, activation='relu', padding='same')(x)
        x = Conv1D( 256, 5, activation='relu', padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Dropout(DO)(x)
        
        if hp['lstm']==1:
            x = LSTM( units=300, return_sequences=True)(x)
            x = LSTM( units=300, return_sequences=True)(x)        
            x = LSTM( units=300, return_sequences=False)(x)            
        else:
            x = Bidirectional(LSTM( units=300, return_sequences=True))(x)
            x = Bidirectional(LSTM( units=300, return_sequences=True))(x)        
            x = Bidirectional(LSTM( units=300, return_sequences=False))(x)
        x = Dropout(DO/2)(x)
        x = Flatten()(x)        

    opt=tf.keras.optimizers.Adam( lr=.01 )    

    if 'age' in hp['task']:
        out_name='regression'
        pred = Dense(1, activation='linear', name=out_name)(x)
        losses = {out_name: 'mse'}; metrics=['mse'] #= {out_name: ['mse' ]}
        mon = 'mse'
    else:
        out_name='classify'
        pred = Dense( hp['nc'], activation='softmax', name=out_name)(x)
        losses = {out_name: 'binary_crossentropy',}
        metrics= ['binary_crossentropy', get_f1macro]
        mon = 'f1_macro'
     
    model = Model(inputs=inputs, outputs=pred)  
    #model.compile(loss=losses, metrics=metrics, optimizer=opt )
    model.compile( loss=losses, metrics=metrics, optimizer=opt, run_eagerly=True, )
    return model, mon, losses, metrics

# ================== utils ==================
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


