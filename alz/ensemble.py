 

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier,  BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from catboost import Pool, CatBoostClassifier
import xgboost

from tabpfn import TabPFNClassifier
from sklearn.utils import class_weight

import time

class Ensemble(BaseEstimator):
    def __init__(self, impute_nan=False,reweigh_prob=False):
        
        
        self.impute_nan= impute_nan
        self.reweigh_prob = reweigh_prob
        #self.dpool_set_id = np.hstack((np.zeros(8), np.ones(8)))
                
        self.classifiers =  [BaggingClassifier(KNeighborsClassifier(n_neighbors=5), max_samples=0.9, max_features=0.7, n_estimators=MX) ]
        self.classifiers += [BaggingClassifier(KNeighborsClassifier(n_neighbors=3), max_samples=0.9, max_features=0.7, n_estimators=300) ]
        self.classifiers += [TabPFNClassifier(N_ensemble_configurations=3,)]
        self.classifiers += [TabPFNClassifier(N_ensemble_configurations=29,)]
        self.classifiers += [xgboost.XGBClassifier(n_estimators=300,max_depth=5, learning_rate=0.01, subsample=0.9, colsample_bytree=0.85)]                           
        self.classifiers += [xgboost.XGBClassifier(n_estimators=MX, max_depth=3, learning_rate=0.01, subsample=0.9, colsample_bytree=0.85)]                           
        self.classifiers += [CatBoostClassifier(verbose=False, n_estimators=300,  learning_rate=.01, auto_class_weights='Balanced', depth=5)]                           
        self.classifiers += [CatBoostClassifier(verbose=False, n_estimators=MX, learning_rate=.01, auto_class_weights='Balanced', depth=3)]                           
         
        if self.impute_nan:
            self.imp = SimpleImputer(missing_values=np.nan, strategy='median')
    
        self.names = []
        self.N=N=len(self.classifiers)
        
        for n in range( N ):            
            for hp in ['n_neighbors', 'n_estimators', 'N_ensemble_configurations']:
                try:
                    print('?', self.classifiers[n].get_params()[ 'estimator' ].get_params(), hp )
                except:
                    pass
                try:
                    info=self.classifiers[n].get_params()[ hp ]                    
                    break
                except:
                    try:
                        #ensembles[dpool].classifiers[n].get_params()['estimator'].get_params()['n_neighbors']
                        info=self.classifiers[n].get_params()[ 'estimator' ].get_params()[ hp ]            
                        break                
                    except:
                        info=''
            hp2 = hp.split('_')[1]
            na = str(self.classifiers[n].__class__).split(' ')[1][1:].split('.')[-1][:-2]
            self.names.append( na + f'-{hp2}{info}' )

    def fit(self, X, y):        
        cls, y = np.unique(y, return_inverse=True)
        self.classes_ = cls
        if self.impute_nan:
            X = self.imp.fit_transform(X)
            print( 'Imputed missing vals with medians' )
        
        for i,cl in enumerate(self.classifiers):
            start_time = time.time()
            na = self.names[i].lower()
            if 'xgb' in na:
                wts = class_weight.compute_sample_weight('balanced', y )
                print('wts:', np.unique(wts),'\n' )
                cl.fit(X,y, sample_weight = wts )            
            elif 'tabpfn' in na:
                cl.fit(X, y, overwrite_warning=True  )                
            else:
                cl.fit(X,y)                   
            fit_time=(time.time() - start_time)/60
            print(na, "fitted in %.2f minutes ---\n" % fit_time )
            
    def predict(self, X, ):        
        if self.impute_nan:
            X = self.imp.transform(X)

        ps = np.zeros( (self.N, X.shape[0] ) )
        for i,cl in enumerate(self.classifiers):
            start_time = time.time()
            na = self.names[i].lower()
            gu = cl.predict(X) 
            print(gu.shape )
            if 'cat' in na:
                ps[i,:] = gu[:,0]
            else:
                ps[i,:] = gu

            _time=(time.time() - start_time)/60
            print(na, "Infer in %.2f minutes ---\n" % _time )
                
        p = np.sum(ps>0,axis=0) # majority voting
        p1 = p/self.N        
        
        class_0_est_instances = 1-p1
        others_est_instances = p1
        
        # we reweight the probs, since the loss is also balanced like this
        # our models out of the box optimize CE
        # with these changes they optimize balanced CE
        if self.reweigh_prob:
            new_p = p * np.array([[1/(class_0_est_instances if i==0 else others_est_instances) for i in range(p.shape[1])]])
            p=new_p / np.sum(new_p,axis=1,keepdims=1)
        return p, ps
    
    def predict_proba(self, X, selected = None ):
        
        if self.impute_nan:
            X = self.imp.transform(X)
        '''    
        self.classifiers = np.array(self.classifiers)
        if selected is not None:
            ps = np.stack([cl.predict_proba(X) for cl in self.classifiers[selected]])
            print( 'perform prediction using only selected classifiers' )
        else:
        '''
        ps = np.stack([cl.predict_proba(X) for cl in self.classifiers])
        
        p = np.mean(ps,axis=0) # mean voting
        
        class_0_est_instances = p[:,0].sum()
        others_est_instances = p[:,1:].sum()
        # we reweight the probs, since the loss is also balanced like this
        # our models out of the box optimize CE
        # with these changes they optimize balanced CE
        if self.reweigh_prob:
            new_p = p * np.array([[1/(class_0_est_instances if i==0 else others_est_instances) for i in range(p.shape[1])]])
            p=new_p / np.sum(new_p,axis=1,keepdims=1)
        return p, ps
if 0:    
#if ( 'ensemble' in globals())==False:
    ensemble = Ensemble(); N,names=ensemble.N,ensemble.names 
    NPOOLS = 3
    ensembles={}
    for dpool in range( NPOOLS ):
        ensembles[dpool] = Ensemble()
        print('Training with pool',dpool, ensembles[dpool].names )
        ensembles[dpool].fit( X[dpool], Y[dpool] )
