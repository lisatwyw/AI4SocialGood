import torch
import pandas as pd # read from parquet format
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

SEED=101
np.random.seed(SEED)

class DataGenerator(tf.keras.utils.Sequence):    
    def __init__(self, df, 
                 ndims = 2,
                 batch_size=64,
                 input_size=(696, 696),
                 seq_len=5120,
                 shuffle=True):
        
        self.seq_len = seq_len 
        self.df = df.copy()
        self.t_ids = df.sample_id
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        
        self.n = len(self.t_ids)

        self.ndims=ndims
        self.curr_segment=None
        if self.ndims==2:
            self.curr_segment=0
            
        self.class2 = (df['sample_type'] != 'control').values.astype(int)
                   
        self.gender = df['gender'] .values
        self.disease = df['disease'].values

        self.nclasses=2         
        self.__shuffle()
        
    def __shuffle(self):
        self.inds_by_class={}        
        for c in range(self.nclasses):
            s = np.where(self.class2==c )[0]
            r = np.random.permutation( len(s))            
            self.inds_by_class[ c ] = s[r]
            print(c, f'has {len( self.inds_by_class[ c ] )} samples')

    def on_epoch_end(self):
        if self.ndims==2:
            self.curr_segment +=1
            self.n_segments =NFEATS//self.seq_len
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
            X_batch = np.zeros((self.batch_size,self.seq_len ),dtype=float)
            
        for i,f in enumerate(batches):
            g = id2files[f]
            x = pd.read_parquet( g ).fillna( 1 )[f].values            
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
        return xx, y_age[r]  
               
    def __getitem__(self, index):
        half = self.batch_size//2
        batches = np.empty(0)
        for c in range( self.nclasses):
            nrounds = len(self.inds_by_class[ c ])//half
            index = index % ( nrounds )            
            s = self.inds_by_class[ c ][ index*half:(index+1)*half]
            batches = np.hstack((batches, s)) 
            
        batches = self.t_ids[batches.astype(int)] 
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size

# sample data
d=[0,0,1,0,0,1,0,0,1]
dat = dict( gender=[0,1,1,0,1,1,0,1,1], sample_type=d, age=[11,34,55,11,34,55,11,34,55], disease=d, sample_id=np.arange(1,10)  )
df = pd.DataFrame( dat )
df.set_index( 'sample_id' )

# list of sample_ids that are accessible from directory
t_ids = np.arange(1,10)
trn,val,tst=0,1,2

# init
ND=2
BS=2
val_gen = DataGenerator( df= df.loc[t_ids[val::3]], batch_size=BS, ndims=ND )

# get first batch
xx,yy=val_gen.__getitem__(0) 
