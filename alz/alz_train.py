import torch
from torchvision import datasets, models, transforms, utils
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader 
from tensorflow.keras import backend as K
print( 'tf', tf.__version__ )


TK ='disease'
TK = 'isAD'
DEBUG=1
USE_OPTUNA = 0

EP=300; BS=64
if DEBUG:
    BS,EP=16,3
### Define F1 measures: F1 = 2 * (precision * recall) / (precision + recall)
def f1macro(y_true, y_pred):
    nv =len( np.unique( y_true ))
    nclasses =len( np.unique( y_true ))
    assert nv <= nclasses
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        return recall
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives+K.epsilon())
        return precision
    precision=np.zeros(nclasses)
    recall=np.zeros(nclasses)
    
    f=0
    for c in range(nclasses):
        y=(y_true== c)*1.
        yp=(y_pred ==c)*1.
        precision[c], recall[c] = precision_m(y, yp), recall_m(y, yp)
        f += 2*((precision[c]*recall[c])/(precision[c]+recall[c]+K.epsilon()))
    return f/nclasses

def load( id2file, flist, NFEATS, reduct=1 ):
    files = [id2file[file] for file in flist ]
    # print( files )
    A = np.zeros( (len(files), NFEATS//reduct, ) )
    for i,g in enumerate( files):
        f=os.path.basename( g) .split('.')[0]    
        #Ids.append( f )
        #Ages.append(trn_map.age[ np.where( trn_map.sample_id == f)[0]  ].values )
        A[i,:] = pl.read_parquet( g )[f][::reduct]
    # A[ np.isnan(A) ] = 1
    print('data loaded', A.shape ) 
    return A
#    A = load( id2file, flist, hp['NFEATS'] )

class Dataset( Dataset ):
    """
    Dataset
    """
    def __init__(self, df, id2file, hp ):                   
        self.id2file = id2file
        self.hp=hp
        self.t_ids = df.index 
        self.input_size = input_size = hp['input_size']
        
        self.n = df.shape[0]
        self.ndims=len( input_size )
        self.seq_len = input_size[0]         
        self.df = df.copy()

        self.outcome = (df['sample_type'] != 'control').values.astype(int)
        self.isAD = (df['disease'] == 1 ).values.astype(int)
        
        self.gender = df['gender'] .values
        self.disease = df['disease'].values
        self.age = df['age'] .values

        self.Y = {'age': self.age, 'gender':self.gender, 'iscase': self.outcome, 'isAD': self.isAD }         
        self.task = hp['task']
        
        self.batch_size = self.hp['BS']
        self.curr_segment = 0
        
        self.__order_seq_by_grp()
        print( '\n\nWarning: data read from parquet; NANs replaced with d (d is a value not in raw data)\n\n')
        
    def __order_seq_by_grp(self):
        self.inds_by_class={}        
        
        if self.task == 'age':
            y= self.outcome.copy()
        else:
            y= self.Y[self.task].copy()
        self.nc = len(np.unique( y ))
        print( self.nc, 'classes')
        
        for c in range( self.nc ):
            s = np.where( y==c )[0]
            if self.hp['shuffle']:
                r = np.random.permutation( len(s))            
                self.inds_by_class[ c ] = s[r]
            else:
                self.inds_by_class[ c ] = s 
            print(c, f'has {len( self.inds_by_class[ c ] )} samples', end=' | ')
            
    def __len__(self):
        return self.n // self.batch_size
    
    def on_epoch_end(self):
        if self.ndims==1:
            self.curr_segment +=1
            self.n_segments = self.hp['NFEATS']//self.seq_len
        self.__order_seq_by_grp()
        print(f'{self.curr_segment} of {self.n_segments} segment', )
        
    def __getitem__(self, index):
        half = self.batch_size//2
        batches = np.empty(0)
        for c in range( self.nc  ):
            #print( c, self.inds_by_class[c] )
            nrounds = len(self.inds_by_class[ c ])//half
            index = index % ( nrounds )            
            s = self.inds_by_class[ c ][ index*half:(index+1)*half]
            batches = np.hstack((batches, s))             
        batches = self.t_ids[batches.astype(int)] 
        X, y = self.__get_data(batches)        
        #print( np.sum(y ==0), '0s', np.sum(y ==1), '1s' )
        return X, y
    
    def __get_X(self, batches):
        if self.ndims==2:
            s = self.input_size[0]
            X_batch = np.zeros((self.batch_size, 696, 696), dtype=float)
        else:
            X_batch = np.zeros((self.batch_size, self.seq_len), dtype=float)         
        for i,f in enumerate(batches):
            g = self.id2file[f]
            x = pd.read_parquet( g ).fillna( 1 )[f].values
            #print(x.shape, X_batch.shape )
            if self.ndims==1:
                xx= x[self.curr_segment*self.seq_len:(self.curr_segment+1)*self.seq_len ]
                X_batch[i,:len(xx)] =xx
            else:
                X_batch[i,:] = np.reshape( x[:s*s], [s,s])
        return X_batch
        
    def __get_data(self, batches):                           
        y_control= (self.df.loc[batches].sample_type.values != 'control').astype(int) # 0: control; 1: diseased
        y_gender = (self.df.loc[batches].gender.values == 'M').astype(int) #0: female; 1: male
        y_age = self.df.loc[batches].age.values
        y_isAD = (self.df.loc[batches].disease.values == 1  ).astype(int) # non-AD; 1: AD
        
        xx = self.__get_X( batches )
         
        a= np.random.permutation( len(batches) ); b=np.arange( self.batch_size )
        r = a if (self.hp['shuffle']) else b        
        self.current_ids = batches[r].values.copy()
        
        if 'age' in self.task:
            out = y_age[r]
        elif 'gender' in self.task:
            out = y_gender[r] 
        elif 'iscase' in self.task:
            out = y_control[r] 
        elif 'isAD' in self.task:
            out = y_isAD[r] 
        else:
            out = {'regression':y_age[r], 'classify': y_control[r], 'classify3': y_isAD[r], 'classify2': y_gender[r] }

        print(batches[r], out, '<--' )
        
        xx = xx[r,]  # shuffle      
        
        return torch.tensor(xx, dtype=torch.float32), torch.tensor(out) 
        #torch.unsqueeze( xx,0)
        
ds={}
avail_trn=trn_map.loc[ t_ids].copy()
trn,val,tst=0,1,2

n, nvars = avail_trn.shape

if TK =='disease':    
    stratified_sample = np.empty(0)
    for c in range(10):
        m=20*2
        if c==0:
            m=300*2
        stratified_sample =np.hstack( (stratified_sample,np.array( avail_trn[avail_trn.disease== c ].index[:m].T.values) ))
    r=np.random.permutation(len(stratified_sample) )

    avail_ids = avail_trn.loc[stratified_sample].iloc[r, ].index.copy()
    TIDS=dict(trn=avail_ids[trn::3],val=avail_ids[val::3],tst=avail_ids[tst::3]) 

    hp=dict( input_size=(696,696), BS=BS, shuffle=True, task='iscase' )

    for tid in ['val','trn']: 
        ds[tid] = Dataset( avail_trn.loc[ TIDS[tid] ], id2file=id2file, hp=hp )    
        for n in range(1):
            xx,yy = ds[tid].__getitem__(n)
            print(tid, n, yy[::10])
            
elif TK =='isAD':
    print( n, 'samples', nvars, 'non-meth. variables')
    r=np.random.permutation( n )
    avail_ids = avail_trn.iloc[r, ].index.copy()
    TIDS=dict(trn=avail_ids[trn::3],val=avail_ids[val::3],tst=avail_ids[tst::3]) 

    hp=dict( input_size=(696,696), BS=BS, shuffle=True, task='isAD' )
    for tid in ['val','trn']: 
        ds[tid] = Dataset( avail_trn.loc[ TIDS[tid] ], id2file=id2file, hp=hp )    
        for n in range(1):
            xx,yy = ds[tid].__getitem__(n)
            ##print( yy )#  np.sum( yy.numpy()==0 ), np.sum( yy.numpy()==1 ) )
    
