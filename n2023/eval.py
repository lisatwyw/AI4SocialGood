if data_id==1: # narratives only
  XX= X['trn']
  val = (X['val'],Y['val'])
  tst = (X['tst'],Y['tst'])

elif data_id==2: # non-narratives
  XX = X2[t]
  val = (X2['val'],Y['val'])
  tst = (X2['tst'],Y['tst'])
  tabnet_params.update({"grouped_features": [ list(np.arange(X2[t].shape[1] )) ]} )

elif data_id==3: # both

  nf1=X['trn'].shape[1]
  nf2=X2['trn'].shape[1]
  XX =   np.hstack( (X['trn'], X2['trn']))     
  trn = (np.hstack( (X['trn'], X2['trn'])),Y['trn'])
  val = (np.hstack( (X['val'], X2['val'])),Y['val'])
  tst = (np.hstack( (X['tst'], X2['tst'])),Y['tst'])

if tid<4:
  # https://stackoverflow.com/questions/65040696/spacy-aggressive-lemmatization-and-removing-unexpected-words
  # accuracy_metric
  
  #plt.hist( np.abs(y_preds - Y_val[:,0])  );
  
  from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
  from pytorch_tabnet.metrics import Metric
  from sklearn.metrics import roc_auc_score
  
  class Gini(Metric):
      def __init__(self):
          self._name = "gini"
          self._maximize = True
      def __call__(self, y_true, y_score):
          auc = roc_auc_score(y_true, y_score[:, 1])
          return max(2*auc - 1, 0.)
  
  os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
  def seed_everything(seed_value):
      random.seed(seed_value)
      np.random.seed(seed_value)
  
      os.environ['PYTHONHASHSEED'] = str(seed_value)
  
      if torch.cuda.is_available():
          torch.cuda.manual_seed(seed_value)
          torch.cuda.manual_seed_all(seed_value)
          torch.backends.cudnn.deterministic = True
          torch.backends.cudnn.benchmark = False
        
  seed_everything(SEED)
  SCHEDULER_MIN_LR = 1e-6
  SCHEDULER_FACTOR = 0.9
  DEVICE_NAME = "cuda"

  from pytorch_tabnet.augmentations import ClassificationSMOTE
  
  aug = ClassificationSMOTE(p=0.4)
  
  '''
  tabnet_params2 = {"cat_idxs":cat_idxs,
                   "cat_dims":cat_dims,
                   "cat_emb_dim":2,
                   "optimizer_fn":torch.optim.Adam,
  
                   "scheduler_params":{"step_size":50, # how to use learning rate scheduler
                                   "gamma":0.9},
                   "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                   "mask_type":'entmax', # "sparsemax"
                   "grouped_features" : grouped_features
                  }
  '''
  # mask_type= 'entmax', n_d=64, n_a=64,

  tabnet_params = dict(n_steps=2, gamma=1.1,lambda_sparse=0.1, mask_type='entmax',# "sparsemax"
                     verbose=10, device_name=DEVICE_NAME,  )
  if 0:
    tabnet_params["optimizer_params"]=dict(lr=0.001, weight_decay=1e-5, momentum=0.9)
    tabnet_params['optimizer_fn']=torch.optim.SGD
  else:
    tabnet_params['optimizer_fn']=torch.optim.AdamW
    tabnet_params["optimizer_params"]=dict(lr=1e-3)
  
  if 0:
    tabnet_params['scheduler_params']=dict(mode="min",
                                           patience=20,
                                           min_lr=SCHEDULER_MIN_LR,
                                           factor=SCHEDULER_FACTOR)
    tabnet_params['scheduler_fn']=torch.optim.lr_scheduler.ReduceLROnPlateau
  else:
    tabnet_params['scheduler_fn'] = torch.optim.lr_scheduler.StepLR
    tabnet_params['scheduler_params'] = dict(step_size=30,gamma=0.1)
  
  
  if tid==0: # eval_metric: ‘mse’, ‘mae’, ‘rmse’, ‘rmsle’
      model = TabNetRegressor(**tabnet_params)
  else:
    if tid==3: # both
      tabnet_params.update({#"n_shared":3, "n_independent":3,"cat_idxs": cat_idxs, 
                            "grouped_features": [ list(np.arange(0,nf1)), list(np.arange(nf1,nf1+nf2 )) ]} )
   
    model = TabNetClassifier(**tabnet_params)
    
    hist = model.fit(XX,Y['trn'],
      eval_set= [val],
      eval_metric = ['balanced_accuracy'], # balanced_accuracy, Gini
      max_epochs=1000,
      patience=20, batch_size=512, virtual_batch_size= 64 )
    try:
        plt.plot(model.history['val_0_balanced_accuracy'], label='val')
        plt.plot(model.history['loss'], label='loss')
        plt.legend()
        plt.figure()
        plt.plot(model.history['lr'])
    except:
        pass
 

assert XX.shape[0] == Y['trn'].shape[0] 
assert val[0].shape[0] == val[1].shape[0] 
if tid==4:
  from xgboost import XGBClassifier
  '''
  Use fewer trees. ... 1000 -> 100
  Use shallow trees. ...  # max_depth 8 -> 3       
  Use a lower learning rate. ... 0.1 --> 0.01
  Reduce the number of features. ...
  '''
  clf_xgb = XGBClassifier(max_depth=3,
    learning_rate=0.01,
    n_estimators= 1000,
    verbosity=0,
    silent=None,
    objective='binary:logistic',
    booster='gbtree',
    n_jobs=-1,
    nthread=None,
    gamma=2, # minimum loss reduction required to make a further split; gamma. Larger values avoid over-fitting
    eta =.01,
    min_child_weight=10, # the minimum sum of instance weight needed in a leaf, related to min no. instances needed in a node; larger values avoid over-fitting.
    max_delta_step=0,
    subsample=0.5,
    colsample_bytree=.5, # lower avoids overfit colsample_bytree
    colsample_bylevel=1,
    colsample_bynode=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=.2, # handle imbalance
    base_score=0.5,
    gpu_id=0,
    tree_method='gpu_hist',
    random_state=0,
    seed=None,
    early_stopping_rounds=50,)
  
  clf_xgb.fit( XX, Y['trn'],
        eval_set=[ val ],   
        verbose=10)

y_trn_preds =  clf_xgb.predict_proba(XX )
y_val_preds =  clf_xgb.predict_proba( val[0] )
y_tst_preds = clf_xgb.predict_proba( tst[0] )

if tid==4:
  y_trn_preds = np.argmax( y_trn_preds,1 )
  y_val_preds= np.argmax( y_val_preds,1 )
  y_tst_preds = np.argmax( y_tst_preds,1 )

if 1:
  from sklearn.metrics import *
  conf = confusion_matrix( y_trn_preds, Y['trn'] )
  tn, fp, fn, tp = conf.ravel()
  #print( t, tn, fp, fn, tp )
  print(conf )

  conf = confusion_matrix( y_val_preds, Y['val'] )
  tn, fp, fn, tp = conf.ravel()
  #print( t, tn, fp, fn, tp )
  print(conf )
  conf = confusion_matrix( y_tst_preds, Y['tst'] )
  tn, fp, fn, tp = conf.ravel()
  #print( t, tn, fp, fn, tp )
  print(conf )
  from sklearn.metrics import *  # with AdamW
  print( 'trn', roc_auc_score( y_trn_preds, Y['trn'], average='macro') )
  print( 'val', roc_auc_score( y_val_preds, Y['val'], average='macro') )
  print( 'tst', roc_auc_score( y_tst_preds, Y['tst'], average='macro') )
  
  try:
    if tid<4:
      fig=px.line( model.feature_importances_ )
      fig.show()
  except:
      pass

  
