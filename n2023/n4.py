install_packages(['pip install optuna'], INTERACTIVE )
import optuna
import xgboost as xgb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score, log_loss

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

    # Save trained model
    today = date.today()
    bst.save_model(f'aft_model_{emb}_{today}.json')
    
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

if 0:
    from pytorch_tabnet.metrics import Metric
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor 
    from pytorch_tabnet.augmentations import ClassificationSMOTE

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

    

pos_ratio = 1-np.isinf( surv_inter['trn']['label_upper_bound'] ).sum() / surv_inter['trn'].shape[0]
print( 'pos-neg-ratio:', pos_ratio,  )


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

